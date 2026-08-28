# snake/scripts/canals/epi_ale_density_precompute.py
#
# Two descriptive views of how one head's two uncertainty components sit relative to each other,
# both at per-(spot,gene) resolution:
#
#   1. epistemic (y) vs aleatoric (x), one 2-D histogram per donor      -> density.parquet
#      Linear axes, data-driven range (see the grids section below).
#   2. the ratio epistemic/aleatoric,  one 1-D histogram per slide      -> ratio.parquet
#      log10 axis, fixed range. The ratio spans several decades, which is unreadable on a
#      linear axis, so this one view keeps the log. Being fixed, its grid is shared by every
#      job, so ratio distributions are comparable across heads and cohorts too, not only
#      across the three estimators of one panel.
#
# What question this serves. The canal-2 scripts ask whether an uncertainty ranks the error.
# This one asks something upstream and purely descriptive: for a given head, how big is the
# epistemic term next to the aleatoric one, and does that change from donor/slide to
# donor/slide and from estimator to estimator? The answer is a distribution, not a scalar, so
# the distribution itself is what gets precomputed and the plotting only draws it. Nothing is
# reduced to a mean: the mean of a variance is dominated by its tail and would hide exactly the
# spread the figures are about.
#
# One job per (dataset, method, head), all concurrent, CPU-only. A test_summaries.h5ad carries
# one value per (spot,gene) for each of a few layers; the histograms it reduces to are a few
# thousand rows.
#
# ── THE THREE ESTIMATORS IT READS ────────────────────────────────────────────────────────────
# All three write the identical summary layout, which is why one loader serves them:
#   MC-dropout     <run>/mc_dropout/test_summaries.h5ad     (epistemic/mc_dropout_posthoc.py)
#   Laplace        <run>/laplace/test_summaries.h5ad        (epistemic/laplace_posthoc.py)
#   deep ensemble  <model>/ensemble/test_summaries.h5ad     (epistemic/deep_ensemble_combine.py)
# X = pred = the mean over draws/members ; layers["epistemic"] = the disagreement variance ;
# plus one per-family aleatoric ingredient: "aleatoric_var" (Gaussian) | "mean_alpha_m2" (NB) |
# nothing at all (Poisson, whose aleatoric is the mean).
#
# ── THE TWO QUANTITIES ───────────────────────────────────────────────────────────────────────
#   epistemic = layers["epistemic"]                         used as-is, no computation
#   aleatoric = layers["aleatoric_var"]                     (Gaussian)
#             | X + layers["mean_alpha_m2"]                 (NB)
#             | X                                           (Poisson: var == mean)
# The aleatoric formula is picked by a runtime layer-presence check, exactly as
# canal2_uncertainty_precompute.py does in its --mc-dropout-dir / --laplace-dir modes, so it
# covers the whole count set without a per-model whitelist. Both terms are then divided by the
# run's depth_scale**2 for a count head, a per-run constant and the only thing allowed to touch
# an uncertainty, landing on the CP10k variance axis; a Gaussian head stays in log1p-CP10k
# variance. Those are different axes, which is why one figure panel is always one head: the
# magnitudes are comparable across donors/slides and across the three estimators, never across
# heads. The ratio is dimensionless and is the one quantity that stays comparable across heads,
# but it is still a ratio of two head-specific variances, so read it per head all the same.
#
# ── THE GRIDS ────────────────────────────────────────────────────────────────────────────────
# 2-D density: linear, data-driven, per job. Each axis is binned linearly from 0 to a high
# quantile (--range-quantile, default 0.999) of that axis pooled over the whole file, so all of
# a job's donors share one frame. Entries above the top edge fold into the last bin and are
# counted (n_over_x / n_over_y in summary.json), never dropped. At q=0.999 that is about 0.1% of
# the mass by construction, so a nonzero count is expected here.
#   The three estimators of one (dataset, head) therefore get three different 2-D grids. That is
#   deliberate: the plots overlay them as contours in data units, where each artifact carries
#   its own edges, and nothing is ever added bin-by-bin across jobs. A shared linear grid would
#   need a range known before any job runs, which is exactly what cannot be guessed on a linear
#   axis.
#   The pooled quantiles are read off a deterministic strided subsample (--subsample-max sets
#   its target size, stride nudged so it can never be a multiple of n_genes and silently sample
#   one gene). A quantile from a subsample that large is far finer than the bin it feeds.
#
# Ratio: log10, fixed (--ratio-lo/--ratio-hi/--ratio-step). No guessing is needed on a log axis:
# one wide fixed grid covers every head, so all artifacts share it and the distributions are
# comparable across heads and cohorts as well as across estimators. Values outside the range
# fold into the edge bin and are counted (n_below / n_above); a ratio of exactly 0, an
# identically-zero epistemic term, has no logarithm and is counted separately in n_zero, never
# silently binned.
#
# ── WHAT IS WRITTEN ──────────────────────────────────────────────────────────────────────────
#   density.parquet   donor, filt, ix, iy, count       sparse 2-D, only occupied bins, LINEAR
#   ratio.parquet     slide, filt, ib, count           sparse 1-D over log10(epi/ale), FIXED grid
#   summary.json      both grid specs + per-donor and per-slide counts, quantiles and shares
#
# Two filters, both written, because the choice is a reporting one:
#   "ypos"  observed y > 0, the restriction every canal-2 number uses, so these figures sit on
#           the same population as the rest of that section. This is the default view.
#   "all"   the whole panel, zeros included. A robustness view, and the honest denominator for
#           "how much of the panel is this".
#
# Usage (one run):
#   python -u snake/scripts/canals/epi_ale_density_precompute.py \
#       --summary-file <...>/mc_dropout/test_summaries.h5ad \
#       --dataset 10x_TuPro --model-type heteroscedastic --method mc_dropout
# The sweep is launched by launch_epi_ale_density_precompute.sh (one SLURM job per run).

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import scanpy as sc

T0 = time.time()
def _el():
    return time.time() - T0

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
from src.data.splitting import load_slide_to_donor, donor_labels   # noqa: E402

# The Gaussian heads. This set decides only the scale: no depth_scale applies to a log1p-CP10k
# variance. The aleatoric formula comes from the layer-presence check below, never from a list.
CONTINUOUS_MODELS = {"msemeanonly", "homoscedastic", "heteroscedastic", "spot_homoscedastic"}

ap = argparse.ArgumentParser()
ap.add_argument("--summary-file", required=True,
                help="path to a test_summaries.h5ad (mc_dropout/ | laplace/ | ensemble/)")
ap.add_argument("--dataset", required=True, help="10x_TuPro | TLS | LUNG_CANCER (for the donor map)")
ap.add_argument("--model-type", required=True)
ap.add_argument("--method", required=True, choices=["mc_dropout", "laplace", "deep_ensemble"],
                help="label only -- the three files have the same layout, so nothing branches on it")
ap.add_argument("--out-dir", default=None, help="default: <dir of --summary-file>/epi_ale_density")
ap.add_argument("--donor-map", default=os.path.join(PROJECT_ROOT, "snake", "config", "donor_slide_map.yaml"))
ap.add_argument("--bins", type=int, default=400, help="bins per axis of the LINEAR 2-D grid")
ap.add_argument("--range-quantile", type=float, default=0.999,
                help="upper edge of each linear 2-D axis, as a quantile of that axis pooled over the file")
# The fixed log10 grid of the ratio, wide enough for every head. Anything outside folds into the
# edge bin and is counted, so a silent truncation is impossible.
ap.add_argument("--ratio-lo", type=float, default=-10.0, help="log10 of the smallest ratio binned")
ap.add_argument("--ratio-hi", type=float, default=3.0, help="log10 of the largest ratio binned")
ap.add_argument("--ratio-step", type=float, default=0.02, help="log10 bin width of the ratio grid")
ap.add_argument("--subsample-max", type=int, default=20_000_000,
                help="cap on the strided subsample the pooled quantiles are read off")
args = ap.parse_args()

summary_file = os.path.abspath(args.summary_file)
out_dir = args.out_dir or os.path.join(os.path.dirname(summary_file), "epi_ale_density")
os.makedirs(out_dir, exist_ok=True)
print(f"[{_el():6.1f}s] {args.method} | {args.dataset} | {args.model_type}\n"
      f"          in  {summary_file}\n          out {out_dir}", flush=True)

# ── Read ─────────────────────────────────────────────────────────────────────────────────────
B = sc.read_h5ad(summary_file)
if B.uns.get("count_convention") != "v2_no_offset":
    raise ValueError(f"{summary_file}: artifact with no count_convention marker. "
                     f"Re-run the post-hoc step.")
assert "epistemic" in B.layers, f"{summary_file}: no layers['epistemic']"
assert np.unique(B.obs["adata_row"].to_numpy()).size == B.n_obs, \
    f"{summary_file}: test spots overlap -- non-LOPO file?"

def _dense(a):
    return np.asarray(a.todense()) if hasattr(a, "todense") else np.asarray(a)

X   = _dense(B.X).astype(np.float32)
epi = _dense(B.layers["epistemic"]).astype(np.float32)
y   = _dense(B.layers["y"]).astype(np.float32)

is_continuous = args.model_type in CONTINUOUS_MODELS

# Aleatoric, by the runtime layer check (see header). The one case that must not fall through to
# the Poisson branch is a continuous head with no aleatoric_var: that is a mean-only model
# (msemeanonly), which has no aleatoric term and so no x axis. Taking X as its aleatoric would
# silently plot the predicted mean against the epistemic variance and label it a decomposition.
if is_continuous and "aleatoric_var" not in B.layers:
    raise SystemExit(f"{args.model_type}: continuous head with no layers['aleatoric_var'] -- "
                     f"a mean-only model has no aleatoric component, so epistemic-vs-aleatoric "
                     f"is undefined for it. Excluded from the sweep on purpose.")
if "aleatoric_var" in B.layers:
    ale = _dense(B.layers["aleatoric_var"]).astype(np.float32)
    ale_formula = "layers['aleatoric_var'] (Gaussian: mean sigma^2 over draws/members)"
elif "mean_alpha_m2" in B.layers:
    ale = X + _dense(B.layers["mean_alpha_m2"]).astype(np.float32)
    ale_formula = "X + layers['mean_alpha_m2'] (NB: mu + mean(alpha*m^2))"
else:
    ale = X.copy()
    ale_formula = "X (Poisson-family: var == mean, no extra layer)"

# depth_scale: nested under uns['deep_ensemble'] on a combine summary, top-level on the two
# post-hoc ones. Absent everywhere == None == depth-blind (*_raw) == divisor 1.0.
if "deep_ensemble" in B.uns:
    ds_val = B.uns["deep_ensemble"].get("depth_scale")
else:
    ds_val = B.uns.get("depth_scale")
div2 = 1.0 if (is_continuous or ds_val is None) else float(ds_val) ** 2
if div2 != 1.0:                     # skip the (N,G) pass when it is a no-op
    epi /= div2
    ale /= div2
var_axis = ("log1p-CP10k variance" if is_continuous else
            ("raw-count variance (depth-blind head)" if ds_val is None else "CP10k variance"))
print(f"[{_el():6.1f}s] aleatoric = {ale_formula}\n"
      f"          depth_scale={ds_val}  divisor(var)={div2}  axis={var_axis}", flush=True)

slides = np.asarray(B.obs["slide"]).astype(str)
slide_to_donor = load_slide_to_donor(args.donor_map, args.dataset)
donors = donor_labels(slides, slide_to_donor)
n_spots, n_genes = X.shape
del B, X
print(f"[{_el():6.1f}s] {n_spots} spots x {n_genes} genes = {n_spots * n_genes:,} entries, "
      f"{len(np.unique(donors))} donors, {len(np.unique(slides))} slides", flush=True)

# ── The linear grids, from a strided subsample of the pooled values ──────────────────────────
# A variance is >= 0 by definition, so anything negative or non-finite is a broken input and is
# excluded rather than silently binned; the per-donor and per-slide n_candidate against n_used in
# summary.json say how many were dropped. The ratio additionally needs ale > 0, since it divides.
epi_flat, ale_flat = epi.ravel(), ale.ravel()
n_all = epi_flat.size
stride = max(1, n_all // max(args.subsample_max, 1))
if stride % n_genes == 0:           # a stride that is a multiple of n_genes samples ONE gene
    stride += 1
sub_e, sub_a = epi_flat[::stride], ale_flat[::stride]
ok = np.isfinite(sub_e) & np.isfinite(sub_a) & (sub_e >= 0) & (sub_a >= 0)
sub_e, sub_a = sub_e[ok].astype(np.float64), sub_a[ok].astype(np.float64)
Y_HI = float(np.quantile(sub_e, args.range_quantile))
X_HI = float(np.quantile(sub_a, args.range_quantile))
# A degenerate axis (a constant-zero epistemic term would do it) has no width to bin; fall back
# to a unit range so the job still produces a readable, obviously-degenerate artifact.
Y_HI = Y_HI if Y_HI > 0 else 1.0
X_HI = X_HI if X_HI > 0 else 1.0
NB = int(args.bins)
# The ratio's own FIXED log10 grid, identical in every job (see THE GRIDS in the header).
R_LO, R_HI_LOG, R_STEP = args.ratio_lo, args.ratio_hi, args.ratio_step
NR = int(round((R_HI_LOG - R_LO) / R_STEP))
assert NR > 1, f"empty ratio grid: [{R_LO}, {R_HI_LOG}] step {R_STEP}"
print(f"[{_el():6.1f}s] 2-D grid ({args.range_quantile:g} quantile of {sub_e.size:,} sampled "
      f"entries, stride {stride}): aleatoric [0, {X_HI:.6g}]  epistemic [0, {Y_HI:.6g}]  x {NB} bins\n"
      f"          ratio grid (FIXED, log10): [{R_LO}, {R_HI_LOG}] step {R_STEP} -> {NR} bins",
      flush=True)
del sub_e, sub_a, ok, epi_flat, ale_flat

def bin_index(v, hi):
    """Linear bin from 0 to `hi`. Values above `hi` fold into the last bin (counted by the
    caller); the axis deliberately starts at 0, since these are variances."""
    return np.clip(np.floor(v / hi * NB).astype(np.int64), 0, NB - 1)

# ── 1. epistemic vs aleatoric, per DONOR ─────────────────────────────────────────────────────
dens_rows, donor_stats = [], []
for donor in sorted(np.unique(donors)):
    sel = np.where(donors == donor)[0]
    e_d, a_d, y_d = epi[sel].ravel(), ale[sel].ravel(), y[sel].ravel()
    valid = np.isfinite(e_d) & np.isfinite(a_d) & (e_d >= 0) & (a_d >= 0)
    for filt in ("ypos", "all"):
        keep = valid & (y_d > 0) if filt == "ypos" else valid
        n_used = int(keep.sum())
        if n_used == 0:
            print(f"  {donor:<12} {filt:<5} EMPTY (0 usable entries)", flush=True)
            continue
        e_k, a_k = e_d[keep].astype(np.float64), a_d[keep].astype(np.float64)
        ix, iy = bin_index(a_k, X_HI), bin_index(e_k, Y_HI)
        counts = np.bincount(iy * NB + ix, minlength=NB * NB)
        occ = np.nonzero(counts)[0]
        dens_rows.append(pd.DataFrame(dict(donor=donor, filt=filt,
                                           ix=(occ % NB).astype(np.int32),
                                           iy=(occ // NB).astype(np.int32),
                                           count=counts[occ].astype(np.int64))))
        q_e = np.quantile(e_k, [0.10, 0.25, 0.50, 0.75, 0.90])
        q_a = np.quantile(a_k, [0.10, 0.25, 0.50, 0.75, 0.90])
        a_pos = a_k > 0
        stat = dict(
            donor=donor, filt=filt,
            n_candidate=int(y_d.size if filt == "all" else (y_d > 0).sum()),
            n_used=n_used, n_spots=int(sel.size),
            n_over_x=int((a_k > X_HI).sum()), n_over_y=int((e_k > Y_HI).sum()),
            epi_q10=float(q_e[0]), epi_q25=float(q_e[1]), epi_med=float(q_e[2]),
            epi_q75=float(q_e[3]), epi_q90=float(q_e[4]),
            ale_q10=float(q_a[0]), ale_q25=float(q_a[1]), ale_med=float(q_a[2]),
            ale_q75=float(q_a[3]), ale_q90=float(q_a[4]),
            # The single number the 2-D figure is really about: how big is the epistemic term
            # next to the aleatoric one. Median of the per-entry RATIO, not a ratio of medians.
            ratio_med=float(np.median(e_k[a_pos] / a_k[a_pos])) if a_pos.any() else float("nan"),
            frac_epi_gt_ale=float(np.mean(e_k > a_k)),
            # Pearson on the NATURAL values, so it is the linear-scale association the figure
            # shows -- and therefore outlier-sensitive, unlike a rank correlation. Read it as a
            # description of the plotted cloud, not as a robust dependence measure.
            pearson_natural=float(np.corrcoef(a_k, e_k)[0, 1]) if n_used > 2 else float("nan"),
        )
        donor_stats.append(stat)
        print(f"  {donor:<12} {filt:<5} n={n_used:>12,}  med ale={q_a[2]:.6g}  "
              f"med epi={q_e[2]:.6g}  med ratio={stat['ratio_med']:.6g}  "
              f"frac(epi>ale)={stat['frac_epi_gt_ale']:.4f}  "
              f"over[x,y]=[{stat['n_over_x']},{stat['n_over_y']}]", flush=True)
        del e_k, a_k
    del e_d, a_d, y_d, valid

# ── 2. the ratio epistemic/aleatoric, per SLIDE ──────────────────────────────────────────────
# SLIDE, not donor, and on purpose: the slide is the metric-grouping unit everywhere else in
# this project (per-(slide,gene) correlations, nanmean over slides), so a per-slide ratio
# distribution is the one that lines up with how everything else is reported. On TLS donor ==
# slide, so the two groupings coincide there and nowhere else.
ratio_rows, slide_stats = [], []
for slide in sorted(np.unique(slides)):
    sel = np.where(slides == slide)[0]
    e_d, a_d, y_d = epi[sel].ravel(), ale[sel].ravel(), y[sel].ravel()
    valid = np.isfinite(e_d) & np.isfinite(a_d) & (e_d >= 0) & (a_d > 0)   # ale > 0: it divides
    for filt in ("ypos", "all"):
        keep = valid & (y_d > 0) if filt == "ypos" else valid
        n_used = int(keep.sum())
        if n_used == 0:
            print(f"  {slide:<14} {filt:<5} EMPTY (0 usable entries)", flush=True)
            continue
        r_k = (e_d[keep].astype(np.float64) / a_d[keep].astype(np.float64))
        # Quantiles stay in natural units; a ratio already reads as "1% of the aleatoric". Only
        # the histogram is log10, because the range spans decades.
        q_r = np.quantile(r_k, [0.10, 0.25, 0.50, 0.75, 0.90])
        n_zero = int((r_k <= 0).sum())          # epistemic identically 0 -> no logarithm
        lr = np.log10(r_k[r_k > 0])
        raw_ib = np.floor((lr - R_LO) / R_STEP).astype(np.int64)
        n_below, n_above = int((raw_ib < 0).sum()), int((raw_ib >= NR).sum())
        counts = np.bincount(np.clip(raw_ib, 0, NR - 1), minlength=NR)
        occ = np.nonzero(counts)[0]
        ratio_rows.append(pd.DataFrame(dict(slide=slide, filt=filt,
                                            ib=occ.astype(np.int32),
                                            count=counts[occ].astype(np.int64))))
        stat = dict(
            slide=slide, donor=str(donors[sel[0]]), filt=filt,
            n_candidate=int(y_d.size if filt == "all" else (y_d > 0).sum()),
            n_used=n_used, n_spots=int(sel.size),
            n_zero=n_zero, n_below=n_below, n_above=n_above,
            ratio_q10=float(q_r[0]), ratio_q25=float(q_r[1]), ratio_med=float(q_r[2]),
            ratio_q75=float(q_r[3]), ratio_q90=float(q_r[4]),
            ratio_mean=float(r_k.mean()), frac_gt_1=float(np.mean(r_k > 1.0)),
        )
        slide_stats.append(stat)
        print(f"  {slide:<14} {filt:<5} n={n_used:>12,}  med ratio={q_r[2]:.6g}  "
              f"[q10,q90]=[{q_r[0]:.4g}, {q_r[4]:.4g}]  frac(>1)={stat['frac_gt_1']:.4f}  "
              f"zero/below/above={n_zero}/{n_below}/{n_above}", flush=True)
        del r_k, lr
    del e_d, a_d, y_d, valid

if not dens_rows or not ratio_rows:
    raise SystemExit(f"{summary_file}: no usable entries -- nothing written")

print(f"[{_el():6.1f}s] writing outputs to {out_dir} ...", flush=True)
pd.concat(dens_rows, ignore_index=True).to_parquet(
    os.path.join(out_dir, "density.parquet"), index=False)
pd.concat(ratio_rows, ignore_index=True).to_parquet(
    os.path.join(out_dir, "ratio.parquet"), index=False)

summary = dict(
    method=args.method, dataset=args.dataset, model_type=args.model_type,
    summary_file=summary_file,
    aleatoric_formula=ale_formula, depth_scale=(None if ds_val is None else float(ds_val)),
    variance_divisor=div2, variance_axis=var_axis, is_continuous=bool(is_continuous),
    scale="2-D density: NATURAL (linear) axes. ratio histogram: LOG10 axis, fixed grid.",
    grid=dict(n_bins=NB, x_hi=X_HI, y_hi=Y_HI,
              range_quantile=args.range_quantile, subsample_stride=int(stride),
              ratio_lo=R_LO, ratio_hi=R_HI_LOG, ratio_step=R_STEP, ratio_n_bins=NR,
              note="2-D: linear bins from 0; bin i of an axis with top edge hi = "
                   "floor(v/hi*n_bins) clipped to [0, n_bins-1] (values above hi fold into the "
                   "last bin, counted in n_over*); center = (i + 0.5)*hi/n_bins. Each job has "
                   "its OWN 2-D grid -- overlay in DATA units, never bin-by-bin. "
                   "RATIO: bin i = floor((log10(r) - ratio_lo)/ratio_step) clipped to "
                   "[0, ratio_n_bins-1] (outside -> n_below/n_above; r == 0 -> n_zero, not "
                   "binned); center in log10 = ratio_lo + (i + 0.5)*ratio_step. This grid is "
                   "FIXED and identical in every job, so ratio histograms ARE stackable."),
    n_spots=int(n_spots), n_genes=int(n_genes), n_entries=int(n_spots) * int(n_genes),
    n_donors=int(len(np.unique(donors))), n_slides=int(len(np.unique(slides))),
    slide_to_donor={s: str(donors[np.where(slides == s)[0][0]]) for s in sorted(np.unique(slides))},
    per_donor=donor_stats, per_slide=slide_stats,
)
with open(os.path.join(out_dir, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"[{_el():6.1f}s] done. {summary['n_donors']} donors (2-D density), "
      f"{summary['n_slides']} slides (ratio) -> {out_dir}", flush=True)
