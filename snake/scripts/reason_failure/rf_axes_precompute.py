# snake/scripts/reason_failure/rf_axes_precompute.py
#
# Three diagnostics of the "reason for failure" analysis (D3, D5, D6), for ONE classic run dir.
# It writes parquets; the plotting happens downstream.
#
# All three ask how the predicted spread sigma relates to the predicted mean mu and to the
# realised error, and on which axis sigma can vary at all:
#
#   D3  Does the predicted uncertainty rank the genes the way the predicted mean does?
#       -> per (slide, gene) means of sigma and of mu, from which a Spearman over the genes of a
#          slide is taken downstream. The aggregated pendant of canal2's per-entry sigma-vs-mu
#          correlation (its r_um).
#   D5  Which axis carries the variation? A two-way (spot x gene) variance decomposition of
#       log(sigma) and of log|error|. An uncertainty that cannot vary inside a spot cannot order
#       errors that do.
#   D6  How large is the variation where it exists? -> p10 / p50 / p90 of sigma and of |error| per
#       (slide, gene) cell, from which the ratios are taken downstream.
#
# D5 also exists per DONOR in snake/scripts/bio/bio_axis_precompute.py (variance_shares.parquet),
# and that is the version this repository reports. This file recomputes it per SLIDE and on the DETECTED
# entries, as a robustness arm, for three reasons that only appear at that resolution:
#   1. per slide, because a spot belongs to one slide: on a multi-section donor (TuPro, NSCLC) the
#      slide effect is otherwise absorbed into the spot main effect and inflates share_spot;
#   2. on detected entries, because at y = 0 the error is the prediction (|0 - mu| = |mu|), so a
#      decomposition of log|error| over all entries is partly a decomposition of log(mu);
#   3. with two eps values and a rank version, because the share of log|error| depends on the
#      floor, and the rank transform removes the floor question entirely (any strictly increasing
#      transform gives the same ranks).
# The two are meant to be read together: agreement is the robustness statement.
#
# ══ THE SCALE RULE: NOTHING IS CONVERTED ═════════════════════════════════════════════════════
# Everything is taken on the model's own output scale, as canal2_uncertainty_precompute.py does,
# and for the same reason:
#     mu    = X                 log1p-CP10k for the Gaussians, the fitted count target otherwise
#     y     = layers["y"]       the same axis as mu, by construction
#     sigma = sqrt(variance)    pred_var for the Gaussians, count_pred_var(mu, alpha) otherwise
#     Delta = |y - mu|          hence on the same axis as sigma, which D6 needs (a ratio of two
#                               quantities on different axes would mean nothing)
# No expm1, no /depth_scale, no delta method. Every statistic here is a rank correlation (D3), a
# ratio (D6) or a variance share of a log (D5), and all three are invariant to a per-run constant,
# so applying one would buy nothing and only make a column name disagree with what it holds.
# Consequence: a Gaussian sigma is in log1p-CP10k units and a count sigma in its own count units,
# so the two families never share a sigma axis. summary.json records which scale a run is on.
#
# ══ DETECTED = y > 0 ON THE MODEL'S OWN TARGET ═══════════════════════════════════════════════
# The same population canal2 calls lin_pos, so these diagnostics and the canal2 columns are read
# on identical entries. For a *_cp head this excludes the entries the common-depth rounding sent
# to zero: they are zeros of that model's target, which is a property of the target rather than a
# filter choice, and it is the population the head was fitted on.
#
# ══ SCOPE ════════════════════════════════════════════════════════════════════════════════════
# TEST spots only (obs["split"] == "test"). Under LOPO each spot is the test spot of exactly one
# fold, so the folds' test sets are disjoint and their union is every spot once, asserted below.
# A run trained with a per-gene target transform is rejected: in standardized space "y > 0" means
# "above the gene mean", not "detected", so the detected/undetected split loses its meaning.
#
# Output (under <run-dir>/reason_failure/):
#   gene_slide.parquet     Gene axis. One row per (test_donor, slide, gene): n_spots, n_pos, and
#                          for both populations (_all = every entry, _pos = detected only) the
#                          means of sigma / mu / error / y and the p10, p50, p90 of sigma and of
#                          the error, all reduced across the spots of that gene.
#                          Serves D3 (means -> Spearman over genes) and D6 (percentiles -> ratios).
#   spot_slide.parquet     Spot axis, the exact dual: one row per (test_donor, slide, spot), the
#                          same columns reduced across the genes of that spot (n_genes, n_pos =
#                          detected genes). Serves D3 and D6 on that axis. Its `pos` columns are
#                          taken over a gene set that differs from spot to spot, so the `all`
#                          columns are the composition-free control to read beside them.
#   variance_shares_slide.parquet   one row per (test_donor, slide, quantity, population,
#                          transform): share_gene, share_spot, share_resid, share_sum, sd_total,
#                          sd_resid, eps, n_spots, n_genes, n_entries, n_sweeps.
#                          quantity   log_sigma | log_abs_err
#                          population all | pos
#                          transform  log (sigma only) | log_eps1 | log_eps10 (error only) | rank
#   summary.json           run identity, resolved scales, eps values, coverage counts.
#
# Cost. Each fold's test block is materialised whole (n_test x n_genes, four float32 arrays), then
# worked on one slide at a time. CPU only, nothing here calls count_expected_log1p.
#
# Usage (repo root, on the cluster; CPU only):
#   python -u snake/scripts/reason_failure/rf_axes_precompute.py \
#       --run-dir results/10x_TuPro/<run_id>/_10x_TuPro_hvg_top3000/r75/heteroscedastic/lambda_0.0
#   # whole sweep: bash snake/scripts/reason_failure/launch_rf_axes_precompute.sh

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
from scipy.stats import rankdata

_T0 = time.time()
def _el():
    return time.time() - _T0

p = argparse.ArgumentParser()
p.add_argument("--run-dir", required=True,
               help="A classic run leaf dir: holds metrics.json and predictions/preds_fold*.h5ad "
                    "(for the Gaussian heads that is the lambda_* sub-dir).")
p.add_argument("--out-dir", default=None, help="Default <run-dir>/reason_failure/.")
p.add_argument("--max-sweeps", type=int, default=40,
               help="Cap on the alternating-centring sweeps of the masked two-way fit. The "
                    "complete-matrix case needs exactly ONE and never enters the loop.")
p.add_argument("--tol", type=float, default=1e-8,
               help="Convergence tolerance on the gene effect between two sweeps. The effects are "
                    "logs (or ranks) of order 1 and the shares are read to four decimals, so 1e-8 "
                    "is already far past what any reported digit can see; it is the SWEEP COUNT "
                    "that costs, one sweep being two passes over the slide's whole matrix.")
args = p.parse_args()

run_dir = Path(args.run_dir)
out_dir = Path(args.out_dir) if args.out_dir else run_dir / "reason_failure"
out_dir.mkdir(parents=True, exist_ok=True)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(PROJECT_ROOT)
from src.evaluation.count_metrics import count_pred_var   # noqa: E402  (needs sys.path)

# ── Family taxonomy — the aleatoric formula differs by family, nothing else does ───────────────
# Same three sets as canal2_uncertainty_precompute.py / bio_axis_precompute.py, kept as a local
# copy because those are argparse scripts, not importable modules.
CONTINUOUS_MODELS = {"msemeanonly", "homoscedastic", "heteroscedastic", "spot_homoscedastic"}
POISSON_MODELS    = {"poisson_cp", "poisson_cp_nr", "poisson_raw"}
NB_MODELS         = {"negativebinomial_hetero_noanchor_cp",
                     "negativebinomial_hetero_noanchor_cp_nr",
                     "negativebinomial_hetero_noanchor_raw",
                     "negativebinomial_spotdisp_cp", "negativebinomial_genedisp_cp",
                     "negativebinomial_spotdisp_cp_nr", "negativebinomial_genedisp_cp_nr",
                     "STI_nb_cp", "STI_nb_raw"}
NO_UNCERTAINTY_MODELS = {"msemeanonly"}   # is_probabilistic=False: no pred_var layer exists


def model_family(model_type):
    if model_type in CONTINUOUS_MODELS:
        return "continuous"
    if model_type in POISSON_MODELS:
        return "poisson"
    if model_type in NB_MODELS:
        return "nb"
    raise ValueError(f"unknown model family for model_type={model_type!r} -- add it to a set above")


def _dense(a):
    return a.toarray() if sp.issparse(a) else np.asarray(a)


# ── Run identity, read from metrics.json (authoritative), never parsed from the path ──────────
with open(run_dir / "metrics.json") as fh:
    M = json.load(fh)
dataset = M["dataset"]
model_type = M["model_type"]
print(f"run_dir : {run_dir}", flush=True)
print(f"dataset : {dataset}   model_type: {model_type}", flush=True)

if model_type in NO_UNCERTAINTY_MODELS:
    # Not a failure: a launcher that sweeps every head must not see this as one. Exit 0 with a
    # summary that says why -- the same contract bio_axis_precompute.py uses.
    with open(out_dir / "summary.json", "w") as fh:
        json.dump(dict(dataset=dataset, model_type=model_type, empty=True,
                       reason=f"model_type={model_type!r} has no uncertainty head"), fh, indent=2)
    print(f"{model_type} has no uncertainty head -- nothing to export.", flush=True)
    sys.exit(0)

transform = M.get("target_transform", "none")
if transform not in (None, "none"):
    raise ValueError(f"{run_dir}: target_transform={transform!r}. In standardized space 'y > 0' "
                     "means 'above the gene mean', not 'detected', so the detected population of "
                     "D3/D5/D6 is undefined. Refused rather than silently mis-labeled.")

family = model_family(model_type)
if family == "continuous":
    sigma_scale = "log1p-CP10k"
else:
    if M.get("count_convention") != "v2_no_offset":
        raise ValueError(f"{run_dir}: count artifact with no count_convention marker "
                         f"(count_convention={M.get('count_convention')!r}). Such a file is rejected rather "
                         "than reinterpreted: mu and alpha would sit on two different scales.")
    _ds = M.get("depth_scale")
    sigma_scale = "raw counts" if (_ds is None and M.get("target_source") == "raw_counts") else "CP_T"
print(f"family  : {family}   sigma and error both on {sigma_scale} (NATIVE, nothing converted)",
      flush=True)

pred_files = sorted((run_dir / "predictions").glob("preds_fold*.h5ad"))
if not pred_files:
    raise FileNotFoundError(f"{run_dir}: no predictions/preds_fold*.h5ad")
print(f"folds   : {len(pred_files)}", flush=True)


# ── The two reusable pieces ───────────────────────────────────────────────────────────────────
def rank_quantiles(S, mask, qs, axis=0):
    """Nearest-rank quantiles along `axis`, over the masked entries only.

    axis=0 reduces down each COLUMN (one value per gene, across the spots) -- the GENE axis.
    axis=1 reduces along each ROW    (one value per spot, across the genes) -- the SPOT axis.
    Nearest-rank, i.e. an order statistic of the group, rather than an interpolated percentile:
    with the 60-entry floor applied downstream the difference is immaterial, and an order statistic
    is a value the data actually took. Masked-out entries are pushed to +inf and sorted to the end,
    so one sort serves every quantile; a group with nothing observed returns NaN.
    """
    A = np.sort(np.where(mask, S, np.inf), axis=axis)
    k = mask.sum(axis=axis)
    other = np.arange(k.size)
    out = {}
    for q in qs:
        idx = np.clip(np.rint(q * (k - 1)).astype(np.int64), 0, S.shape[axis] - 1)
        v = A[idx, other] if axis == 0 else A[other, idx]
        out[q] = np.where(k > 0, v, np.nan)
    return out


def two_way_shares(S, mask=None, max_sweeps=50, tol=1e-10):
    """Additive two-way decomposition of an (n_spots x n_genes) matrix.

    Fits S ~ g + b_i (spot main effect) + a_j (gene main effect) by alternating centring over the
    OBSERVED entries, then splits the observed sum of squares:
        SS_tot = sum_obs (S - g)^2,  SS_gene = sum_obs a_j^2,  SS_spot = sum_obs b_i^2,
        SS_res = sum_obs (S - g - a_j - b_i)^2 .
    On a COMPLETE matrix ONE sweep already lands on the exact orthogonal solution
    (a_j = colmean - g, b_i = rowmean - g), the closed form used in bio_axis_precompute.py, and the
    three shares sum to 1 exactly.
    On an INCOMPLETE one (detected entries only) the design is unbalanced, the two main effects are
    no longer orthogonal, and the shares need NOT sum to 1 -- share_sum is returned precisely so
    that departure is visible instead of hidden, and the split between gene and spot is then not
    unique. The third term is named RESIDUAL and not interaction: without repetitions, interaction
    and noise are not separately identifiable.
    """
    W = np.ones(S.shape, dtype=np.float64) if mask is None else mask.astype(np.float64)
    n_col, n_row, n_obs = W.sum(axis=0), W.sum(axis=1), W.sum()
    g = float((W * S).sum() / n_obs)
    a = np.zeros(S.shape[1], dtype=np.float64)
    b = np.zeros(S.shape[0], dtype=np.float64)
    sweeps = 0
    for _ in range(1 if mask is None else max_sweeps):
        a_prev = a
        a = np.divide((W * (S - g - b[:, None])).sum(axis=0), n_col,
                      out=np.zeros_like(a), where=n_col > 0)
        b = np.divide((W * (S - g - a[None, :])).sum(axis=1), n_row,
                      out=np.zeros_like(b), where=n_row > 0)
        sweeps += 1
        if np.abs(a - a_prev).max() < tol:
            break
    ss_tot = float((W * (S - g) ** 2).sum())
    ss_gene = float((W * a[None, :] ** 2).sum())
    ss_spot = float((W * b[:, None] ** 2).sum())
    ss_res = float((W * (S - g - a[None, :] - b[:, None]) ** 2).sum())
    return dict(share_gene=ss_gene / ss_tot, share_spot=ss_spot / ss_tot,
                share_resid=ss_res / ss_tot,
                share_sum=(ss_gene + ss_spot + ss_res) / ss_tot,
                sd_total=float(np.sqrt(ss_tot / n_obs)),
                sd_resid=float(np.sqrt(ss_res / n_obs)),
                n_entries=int(n_obs), n_sweeps=int(sweeps))


def ranked(S, mask):
    """Average ranks of the OBSERVED entries, in place, as a matrix on the same support.

    Average ranks and not ordinal ones: a gene-homoscedastic head has one sigma per gene, so
    log(sigma) carries huge blocks of exact ties, and ordinal ranks would split them and fabricate
    spot-to-spot variation out of a constant. With average ranks that head returns share_gene = 1,
    which is the built-in sanity check of this whole section.
    """
    out = np.zeros(S.shape, dtype=np.float64)
    v = S[mask] if mask is not None else S.ravel()
    r = rankdata(v)
    if mask is None:
        out = r.reshape(S.shape)
    else:
        out[mask] = r
    return out


QS = (0.10, 0.50, 0.90)
gene_rows, spot_rows, share_rows, seen_rows = [], [], [], []
n_spots_total = 0

for f in pred_files:
    B = sc.read_h5ad(f)
    B = B[B.obs["split"].to_numpy() == "test"].copy()
    donor = str(B.obs["test_donor"].iloc[0]) if "test_donor" in B.obs else f.stem
    genes = np.asarray(B.var_names).astype(str)
    n_spots, n_genes = B.shape
    n_spots_total += n_spots

    mu = _dense(B.X).astype(np.float32)                    # the model's own scale, untouched
    y = _dense(B.layers["y"]).astype(np.float32)           # the same axis, by construction
    if family == "continuous":
        sigma = np.sqrt(_dense(B.layers["pred_var"]).astype(np.float64)).astype(np.float32)
    elif family == "poisson":
        sigma = np.sqrt(count_pred_var(mu.astype(np.float64))).astype(np.float32)
    else:
        alpha = _dense(B.layers["dispersion"]).astype(np.float64)   # alpha at mu's OWN scale
        sigma = np.sqrt(count_pred_var(mu.astype(np.float64), dispersion=alpha)).astype(np.float32)
        del alpha
    err = np.abs(y - mu)
    det = y > 0                                            # canal2's lin_pos population

    slides = np.asarray(B.obs["slide"]).astype(str)
    adata_rows = np.asarray(B.obs["adata_row"]).astype(np.int64)
    seen_rows.append(adata_rows)
    del B

    for slide in pd.unique(slides):
        r = np.flatnonzero(slides == slide)
        Sg, Er, Mu, Yv = sigma[r], err[r], mu[r], y[r]
        Dt = det[r]
        n_r = r.size
        n_pos = Dt.sum(axis=0)

        # ---- GENE AXIS (D3 + D6): one row per (slide, gene), reduced across the SPOTS ----
        rec = {"dataset": dataset, "model_type": model_type, "test_donor": donor, "slide": slide,
               "gene": genes, "n_spots": n_r, "n_pos": n_pos.astype(np.int64)}
        full = np.ones_like(Dt)
        for tag, msk in (("all", full), ("pos", Dt)):
            k = msk.sum(axis=0)
            den = np.where(k > 0, k, 1.0)
            for name, A in (("sigma", Sg), ("mu", Mu), ("err", Er), ("y", Yv)):
                rec[f"mean_{name}_{tag}"] = np.where(
                    k > 0, (np.where(msk, A, 0.0)).sum(axis=0, dtype=np.float64) / den, np.nan)
            for name, A in (("sigma", Sg), ("err", Er)):
                q = rank_quantiles(A.astype(np.float64), msk, QS, axis=0)
                for qq in QS:
                    rec[f"{name}_p{int(qq * 100):02d}_{tag}"] = q[qq]
        gene_rows.append(pd.DataFrame(rec))

        # ---- SPOT AXIS (D3 + D6, the dual): one row per (slide, spot), across the GENES ----
        # Identical reductions, transposed. canal2 measures the sigma-vs-mu coupling on this axis
        # as well as on the gene axis, so the aggregated pendant of that number needs both.
        # One caveat, and it applies only to the `pos` columns: the detected gene set differs from
        # spot to spot, so a per-spot mean over detected genes is taken over a spot-dependent gene
        # set, and part of any across-spot contrast is that composition. The `all` columns are free
        # of it (every spot is reduced over the same gene panel) and are the control to read
        # beside it.
        srec = {"dataset": dataset, "model_type": model_type, "test_donor": donor, "slide": slide,
                "adata_row": adata_rows[r], "n_genes": n_genes,
                "n_pos": Dt.sum(axis=1).astype(np.int64)}
        for tag, msk in (("all", full), ("pos", Dt)):
            k = msk.sum(axis=1)
            den = np.where(k > 0, k, 1.0)
            for name, A in (("sigma", Sg), ("mu", Mu), ("err", Er), ("y", Yv)):
                srec[f"mean_{name}_{tag}"] = np.where(
                    k > 0, (np.where(msk, A, 0.0)).sum(axis=1, dtype=np.float64) / den, np.nan)
            for name, A in (("sigma", Sg), ("err", Er)):
                q = rank_quantiles(A.astype(np.float64), msk, QS, axis=1)
                for qq in QS:
                    srec[f"{name}_p{int(qq * 100):02d}_{tag}"] = q[qq]
        spot_rows.append(pd.DataFrame(srec))

        # ---- D5: two-way shares, per SLIDE, both populations, three transforms ----
        # sigma is strictly positive, so its log needs no floor; the error does (a near-exact
        # prediction, and on the `all` population every undetected entry). The floor is taken from
        # the slice's OWN positive errors, at two magnitudes, and the rank version removes the
        # question altogether -- any strictly increasing transform gives the same ranks.
        for pop, msk in (("all", None), ("pos", Dt)):
            keep = (Er > 0) if msk is None else (Dt & (Er > 0))
            med = float(np.median(Er[keep])) if keep.any() else 1.0
            # The matrices are built one at a time (a thunk per plan, not five ready arrays):
            # five (n_spots x n_genes) float64 matrices held at once would be five times the peak
            # for no reason, since two_way_shares consumes them one after the other.
            plans = [
                ("log_sigma", "log", 0.0,
                 lambda: np.log(Sg.astype(np.float64) + 1e-12)),
                ("log_abs_err", "log_eps1", 0.01 * med,
                 lambda: np.log(Er.astype(np.float64) + 0.01 * med)),
                ("log_abs_err", "log_eps10", 0.10 * med,
                 lambda: np.log(Er.astype(np.float64) + 0.10 * med)),
                ("log_sigma", "rank", np.nan,
                 lambda: ranked(Sg.astype(np.float64), msk)),
                ("log_abs_err", "rank", np.nan,
                 lambda: ranked(Er.astype(np.float64), msk)),
            ]
            for qname, tname, eps, build in plans:
                S = build()
                st = two_way_shares(S, mask=msk, max_sweeps=args.max_sweeps, tol=args.tol)
                share_rows.append(dict(dataset=dataset, model_type=model_type, test_donor=donor,
                                       slide=slide, quantity=qname, population=pop,
                                       transform=tname, eps=eps, n_spots=n_r, n_genes=n_genes,
                                       **st))
                del S
        del Sg, Er, Mu, Yv, Dt, full

    print(f"[{_el():6.1f}s] {f.name}: donor {donor}, {n_spots:,} test spots x {n_genes} genes, "
          f"{len(pd.unique(slides))} slide(s); detected {100 * det.mean():.1f}%", flush=True)
    del mu, y, sigma, err, det, slides

# Under LOPO each spot is the held-out test spot of exactly one fold, so the folds' test sets are
# disjoint and their union is every spot once. Same invariant canal2 and bio_axis assert.
all_rows = np.concatenate(seen_rows)
assert np.unique(all_rows).size == all_rows.size, \
    f"{run_dir}: test spots overlap across folds -- non-LOPO split?"

per_gene_slide = pd.concat(gene_rows, ignore_index=True)
per_spot_slide = pd.concat(spot_rows, ignore_index=True)
shares = pd.DataFrame(share_rows)
per_gene_slide.to_parquet(out_dir / "gene_slide.parquet", index=False)
shares.to_parquet(out_dir / "variance_shares_slide.parquet", index=False)
# LAST on purpose: this is the launcher's sentinel, so a job killed mid-write is re-submitted
# instead of being skipped as already done.
per_spot_slide.to_parquet(out_dir / "spot_slide.parquet", index=False)

summary = dict(
    run_dir=str(run_dir), dataset=dataset, model_type=model_type, family=family,
    sigma_scale=sigma_scale, error_scale=sigma_scale, mu_scale=sigma_scale,
    scale_note=("sigma, mu and the error are all on the model's OWN target scale; nothing is "
                "converted. Every statistic here is a rank, a ratio or a variance share of a log, "
                "so a per-run constant would change none of them."),
    detected_definition="y > 0 on the model's own target (canal2's lin_pos population)",
    n_folds=len(pred_files), n_spots=int(n_spots_total),
    n_slides=int(per_gene_slide["slide"].nunique()),
    n_genes=int(per_gene_slide["gene"].nunique()),
    n_rows_gene_slide=int(len(per_gene_slide)), n_rows_shares=int(len(shares)),
    n_rows_spot_slide=int(len(per_spot_slide)),
    spot_axis_note=("spot_slide.parquet is gene_slide.parquet transposed: the same means and "
                    "quantiles reduced across the GENES of a spot instead of across the spots of "
                    "a gene. Its `pos` columns are taken over a gene set that differs from spot to "
                    "spot, so part of any across-spot contrast there is gene composition; the "
                    "`all` columns reduce every spot over the same 3,000 genes and are the "
                    "control to read beside them."),
    quantiles=[float(q) for q in QS],
    max_sweeps=args.max_sweeps, tol=args.tol,
    max_sweeps_used=int(shares["n_sweeps"].max()),
    d5_note=("per SLIDE and on the DETECTED entries, the robustness arm of the per-DONOR "
             "decomposition in snake/scripts/bio/bio_axis_precompute.py. On the detected "
             "population the design is unbalanced, so share_sum can depart from 1 and the "
             "gene/spot split is not unique -- read share_sum before reading the shares."),
)
with open(out_dir / "summary.json", "w") as fh:
    json.dump(summary, fh, indent=2)

print(f"[{_el():6.1f}s] wrote {len(per_gene_slide):,} (donor, slide, gene) rows -> "
      f"{out_dir / 'gene_slide.parquet'}", flush=True)
print(f"[{_el():6.1f}s] wrote {len(per_spot_slide):,} (donor, slide, spot) rows -> "
      f"{out_dir / 'spot_slide.parquet'}", flush=True)
print(f"[{_el():6.1f}s] wrote {len(shares):,} share rows -> "
      f"{out_dir / 'variance_shares_slide.parquet'}", flush=True)
print()
print("DETECTION, the population every `pos` column runs on (it is NOT the whole panel):")
_tot = float(per_gene_slide["n_spots"].sum())
print(f"  entries detected            : {100 * per_gene_slide['n_pos'].sum() / _tot:.1f}% "
      f"of all (spot, gene) entries")
print(f"  (gene, slide) cells >= 60   : {int((per_gene_slide['n_pos'] >= 60).sum()):,} / "
      f"{len(per_gene_slide):,} "
      f"({100 * (per_gene_slide['n_pos'] >= 60).mean():.1f}%)")
print(f"  (spot, slide) cells >= 60   : {int((per_spot_slide['n_pos'] >= 60).sum()):,} / "
      f"{len(per_spot_slide):,} "
      f"({100 * (per_spot_slide['n_pos'] >= 60).mean():.1f}%)")
print()
print("MEAN OVER SLIDES of share_gene, by population and transform:")
print(shares.pivot_table(index=["population", "transform"], columns="quantity",
                         values="share_gene", aggfunc="mean").round(4).to_string())
print()
print(f"share_sum: min {shares['share_sum'].min():.4f}  max {shares['share_sum'].max():.4f}  "
      "(exactly 1 on the complete matrix; a departure marks the unbalanced detected design)")
