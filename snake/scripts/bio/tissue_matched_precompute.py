# snake/scripts/bio/tissue_matched_precompute.py
#
# Does the predicted uncertainty differ between tissue types at the same predicted mean?
#
# Why this exists next to tissue_sigma_precompute.py. That script holds the mean fixed by direct
# standardisation: the spots of a (gene, slide) are cut into equal-size bins of the predicted mean,
# the tissue means are taken inside each bin, and the bin means are averaged with equal weight.
# That equalises the distribution over bins. It does not equalise the mean inside a bin, so two
# tissue types whose predicted means differ within the same bin stay confounded, and five bins is
# coarse.
#
# What this does instead is matching. Inside each (gene, slide), every spot of a tissue type is
# paired with the spot of the reference tissue type whose predicted mean is closest, within a
# caliper. The difference in uncertainty is read inside the pair. If the two spots of a pair carry
# the same predicted mean, the mean cannot account for their difference in uncertainty, and there
# is no bin, so no residual inside a bin to argue about.
#
# Two columns make the method auditable and must be read before the contrast itself:
#   mean_abs_dmu  the mean absolute difference in predicted mean inside the pairs. Near zero by
#                 construction; if it is not, the caliper is too loose.
#   match_rate    the fraction of spots that found a partner. Unmatched spots are dropped, so a low
#                 rate means the comparison has silently moved to a sub-population.
#
# The caliper is a fraction of the standard deviation of the predicted mean over the pooled spots
# of that (gene, slide). Tighter means better-matched pairs and fewer of them; the trade-off is
# visible in the two columns above.
#
# Also computed, because the matrices are already in memory:
#   ancova     the textbook alternative. Inside each (gene, slide) the uncertainty is regressed on
#              the predicted mean, linearly, and the tissue means of the residual are reported. It
#              assumes a straight line where matching assumes nothing, so agreement between the two
#              is worth more than either alone.
#   examples   the spot-level (predicted mean, uncertainty, tissue type) of a handful of named
#              genes, for the scatter that makes the control visible: at a given predicted mean,
#              are the spots of one tissue type below the others?
#
# The error is carried through both, so "does the error follow" is answered on the same pairs and
# never on a different population.
#
# Scope: test spots only, so under LOPO every spot appears once, predicted by the model that never
# saw its donor. A cohort with no per-spot annotation is not a failure: empty outputs are written
# with the reason in summary.json and the script exits 0.
#
# Reads <run-dir>/metrics.json, <run-dir>/predictions/preds_fold*.h5ad and the merged preprocessed
# adata named by metrics.json["input_adata"].
#
# Writes (under <run-dir>/tissue_matched/):
#   matched.parquet   gene, slide, test_donor, label_col, tissue, ref_tissue, n_pairs, match_rate,
#                     mean_abs_dmu, mean_dsigma, median_dsigma, mean_derr, median_derr,
#                     mean_sigma_a, mean_sigma_b, mean_dlogsigma, n_pairs_log.
#                     The contrast is written both ways and neither replaces the other. The
#                     difference (mean_dsigma) is in the units of the uncertainty. The ratio is
#                     exp(mean_dlogsigma), a geometric mean, which is what compares across genes
#                     and cohorts whose sigma sit at different levels. Both levels are written so
#                     any other normalization needs no re-run.
#   ancova.parquet    gene, slide, test_donor, label_col, tissue, slope, resid_sigma, resid_err, n
#   examples.parquet  gene, slide, test_donor, tissue, mu, sigma, err   (named genes only)
#   summary.json      run identity, caliper, reference tissue, coverage, row counts.
#
# Usage (repo root, on the cluster; CPU only):
#   python -u snake/scripts/bio/tissue_matched_precompute.py \
#       --run-dir <results>/10x_TuPro/<run_id>/_10x_TuPro_hvg_top3000/r75/heteroscedastic/lambda_0.0
#   # whole sweep: bash snake/scripts/bio/launch_tissue_matched_precompute.sh

import argparse
import json
import os
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc

_T0 = time.time()
def _el():
    return time.time() - _T0

p = argparse.ArgumentParser()
p.add_argument("--run-dir", required=True, help="Classic run leaf dir (the lambda_* dir for Gaussians).")
p.add_argument("--out-dir", default=None, help="Default <run-dir>/tissue_matched/.")
p.add_argument("--data", default=None, help="Override metrics.json['input_adata'].")
p.add_argument("--caliper-sd", type=float, default=0.10,
               help="Caliper as a fraction of the SD of the predicted mean within a (gene, slide).")
p.add_argument("--min-pairs", type=int, default=10, help="Min pairs for a (gene, slide, tissue) cell.")
p.add_argument("--min-tissue", type=int, default=30, help="Min spots for a (slide, tissue) to be used.")
p.add_argument("--example-genes", default=None,
               help="Comma-separated gene names for the spot-level export. Default: the dataset panel.")
args = p.parse_args()

run_dir = Path(args.run_dir)
out_dir = Path(args.out_dir) if args.out_dir else run_dir / "tissue_matched"
out_dir.mkdir(parents=True, exist_ok=True)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(PROJECT_ROOT)
from src.evaluation.count_metrics import count_pred_var

CONTINUOUS_MODELS = {"msemeanonly", "homoscedastic", "heteroscedastic", "spot_homoscedastic"}
POISSON_MODELS    = {"poisson_cp", "poisson_cp_nr", "poisson_raw"}
NB_MODELS         = {"negativebinomial_hetero_noanchor_cp",
                     "negativebinomial_hetero_noanchor_cp_nr", "negativebinomial_hetero_noanchor_raw",
                     "negativebinomial_spotdisp_cp", "negativebinomial_genedisp_cp",
                     "negativebinomial_spotdisp_cp_nr", "negativebinomial_genedisp_cp_nr",
                     "STI_nb_cp", "STI_nb_raw"}

# The genes whose spot-level cloud is exported, per dataset. A name absent from the run's panel is
# recorded in summary.json under example_genes_missing, not fatal.
EXAMPLE_GENES = {
    "10x_TuPro":   ["TYR", "PMEL", "S100B", "MLPH", "SOX10", "COL1A1", "PTPRC"],
    "TLS":         ["LTB", "MS4A1", "CXCL13", "CCL19", "CD19", "CD79A", "SFTPC"],
    "LUNG_CANCER": ["SPP1", "TREM2", "APOE", "APOC1", "FABP5", "KRT19", "CD3D"],
}


def _dense(a):
    return a.toarray() if sp.issparse(a) else np.asarray(a)


with open(run_dir / "metrics.json") as fh:
    M = json.load(fh)
dataset, model_type = M["dataset"], M["model_type"]
family = ("continuous" if model_type in CONTINUOUS_MODELS else
          "poisson" if model_type in POISSON_MODELS else
          "nb" if model_type in NB_MODELS else None)
if family is None:
    raise ValueError(f"unknown family for model_type={model_type!r}")
print(f"run_dir : {run_dir}\ndataset : {dataset}   model_type: {model_type}   family: {family}", flush=True)


def write_empty(reason):
    print(f"{reason} -- writing empty outputs.", flush=True)
    for name, cols in (("matched", ["gene", "slide", "test_donor", "tissue", "ref_tissue"]),
                       ("ancova", ["gene", "slide", "test_donor", "tissue"]),
                       ("examples", ["gene", "slide", "test_donor", "tissue", "mu", "sigma", "err"])):
        pd.DataFrame(columns=cols).to_parquet(out_dir / f"{name}.parquet", index=False)
    with open(out_dir / "summary.json", "w") as fh:
        json.dump(dict(dataset=dataset, model_type=model_type, empty=True, reason=reason), fh, indent=2)
    sys.exit(0)


if model_type == "msemeanonly":
    write_empty("msemeanonly has no uncertainty head")

if family == "continuous":
    X_DIV, native = 1.0, "log1p-CP10k"
else:
    if M.get("count_convention") != "v2_no_offset":
        raise ValueError(f"{run_dir}: count artifact with no count_convention marker; re-run the precompute.")
    _ds = M.get("depth_scale")
    X_DIV = 1.0 if _ds is None else float(_ds)
    native = "raw counts" if (_ds is None and M.get("target_source") == "raw_counts") else "CP10k"
print(f"scale   : X_DIV={X_DIV}; mu, sigma and error all on {native}", flush=True)

# ── Tissue labels ─────────────────────────────────────────────────────────────────────────────
data_path = args.data or M["input_adata"]
A = ad.read_h5ad(data_path, backed="r")
obs_all = A.obs.copy()
A.file.close()
del A
LABEL_COLS = [c for c in ("ground_truth", "ground_truth_knn_spatial_15") if c in obs_all.columns]
if not LABEL_COLS:
    write_empty(f"{dataset} carries no per-spot tissue annotation")
LABELS = {c: obs_all[c].to_numpy().astype(str) for c in LABEL_COLS}
del obs_all
print(f"labels  : {LABEL_COLS}", flush=True)

DROP = {"PIGMENT", "UNASSIGNED", "NAN", "NONE", "NA", "N/A", "", "<NA>", "MIXED", "WHITESPACE"}
# The reference every other tissue type is matched against. The tumor is the natural one: it is the
# comparison the section is about, and it is the largest group on both annotated cohorts, so it is
# the one most likely to hold a partner at any predicted mean.
REF_CANDIDATES = ("Tumor", "TUM")


def nearest_partner(mu_a, mu_b, sg_b, er_b, caliper):
    """For every spot of A, the spot of B with the closest predicted mean, per gene column.

    mu_a is (nA, G), mu_b is (nB, G). Returns the matched mu, sigma and error of B shaped like A,
    the signed difference mu_a minus the matched mu_b, and a boolean mask of the pairs that fall
    inside the caliper.

    Vectorised across genes by shifting each column into its own disjoint interval, so one global
    searchsorted over the column-major flattening does all G columns at once. Column j occupies
    [off_j, off_j + span_j) and off is increasing, so the flattened sorted array is globally sorted
    and the search is exact.
    """
    nA, G = mu_a.shape
    nB = mu_b.shape[0]
    order = np.argsort(mu_b, axis=0, kind="stable")
    mb = np.take_along_axis(mu_b, order, axis=0)          # sorted per column
    sb = np.take_along_axis(sg_b, order, axis=0)
    eb = np.take_along_axis(er_b, order, axis=0)

    lo = np.minimum(mu_a.min(axis=0), mb[0])
    hi = np.maximum(mu_a.max(axis=0), mb[-1])
    span = (hi - lo) + 1.0                                 # +1 keeps the intervals strictly disjoint
    off = np.concatenate([[0.0], np.cumsum(span)[:-1]])

    flat_b = (mb - lo + off).ravel(order="F")              # globally sorted
    qa = (mu_a - lo + off).ravel(order="F")
    pos = np.searchsorted(flat_b, qa)                      # global insertion point

    col = np.repeat(np.arange(G), nA)                      # column of each query, column-major
    local = pos - col * nB
    lo_i = np.clip(local - 1, 0, nB - 1)
    hi_i = np.clip(local, 0, nB - 1)
    g_lo = col * nB + lo_i
    g_hi = col * nB + hi_i
    d_lo = np.abs(qa - flat_b[g_lo])
    d_hi = np.abs(qa - flat_b[g_hi])
    take = np.where(d_lo <= d_hi, g_lo, g_hi)

    fb_mu = mb.ravel(order="F")[take].reshape(G, nA).T     # column-major reshape, then transpose
    fb_sg = sb.ravel(order="F")[take].reshape(G, nA).T
    fb_er = eb.ravel(order="F")[take].reshape(G, nA).T
    dmu = mu_a - fb_mu
    inside = np.abs(dmu) <= caliper[None, :]
    return fb_mu, fb_sg, fb_er, dmu, inside


pred_files = sorted((run_dir / "predictions").glob("preds_fold*.h5ad"))
assert pred_files, f"no predictions in {run_dir / 'predictions'}"

match_rows, anc_rows, ex_rows, seen = [], [], [], []
want_genes = ([g.strip() for g in args.example_genes.split(",")] if args.example_genes
              else EXAMPLE_GENES.get(dataset, []))

for f in pred_files:
    P = sc.read_h5ad(f)
    P = P[P.obs["split"].to_numpy() == "test"].copy()
    donor = str(P.obs["test_donor"].iloc[0]) if "test_donor" in P.obs else f.stem
    genes = np.asarray(P.var_names).astype(str)
    adata_row = np.asarray(P.obs["adata_row"]).astype(np.int64)
    slides = np.asarray(P.obs["slide"]).astype(str)
    G = len(genes)

    X = _dense(P.X).astype(np.float32)
    Y = _dense(P.layers["y"]).astype(np.float32)
    if family == "continuous":
        mu, y = X, Y
        sigma = np.sqrt(_dense(P.layers["pred_var"]).astype(np.float32))
    else:
        mu, y = X / X_DIV, Y / X_DIV
        if family == "poisson":
            sigma = np.sqrt(count_pred_var(X.astype(np.float64)) / X_DIV ** 2).astype(np.float32)
        else:
            al = _dense(P.layers["dispersion"]).astype(np.float64)
            sigma = np.sqrt(count_pred_var(X.astype(np.float64), dispersion=al) / X_DIV ** 2).astype(np.float32)
    err = np.abs(mu - y)
    del X, Y, P

    ex_idx = [i for i, g in enumerate(genes) if g in want_genes]

    for sl in pd.unique(slides):
        m = slides == sl
        mu_s, sg_s, er_s = mu[m], sigma[m], err[m]
        rows_of_slide = np.flatnonzero(m)

        for lab_col in LABEL_COLS:
            tis = LABELS[lab_col][adata_row[m]]
            up = np.char.upper(tis.astype(str))
            ok = ~np.isin(up, list(DROP))
            names, counts = np.unique(tis[ok], return_counts=True)
            names = names[counts >= args.min_tissue]
            ref = next((r for r in REF_CANDIDATES if r in names), None)
            if ref is None or len(names) < 2:
                continue
            b = tis == ref
            mu_b, sg_b, er_b = mu_s[b], sg_s[b], er_s[b]
            if mu_b.shape[0] < args.min_pairs:
                continue
            # One caliper per gene, from the pooled spots of this (gene, slide).
            caliper = args.caliper_sd * mu_s[ok].std(axis=0)
            caliper = np.where(caliper > 0, caliper, np.inf)

            for t in names:
                if t == ref:
                    continue
                a = tis == t
                if a.sum() < args.min_pairs:
                    continue
                mu_a, sg_a, er_a = mu_s[a], sg_s[a], er_s[a]
                _, fb_sg, fb_er, dmu, inside = nearest_partner(mu_a, mu_b, sg_b, er_b, caliper)
                n_pairs = inside.sum(axis=0)
                keep = n_pairs >= args.min_pairs
                if not keep.any():
                    continue
                dsg = np.where(inside, sg_a - fb_sg, np.nan)
                der = np.where(inside, er_a - fb_er, np.nan)
                adm = np.where(inside, np.abs(dmu), np.nan)
                # Ratio form, next to the difference. The same absolute gap means different things
                # on a gene whose sigma is 0.1 and on one at 1.0, so the ratio is what compares
                # across genes, slides and cohorts.
                # It is aggregated as the mean of log(sigma_a / sigma_b), whose exponential is the
                # geometric mean ratio. That aggregation is symmetric: a doubling and a halving
                # cancel, which a mean of plain ratios does not, since that is dragged upward by
                # the values above one.
                # sigma is divided by the sigma of its matched partner, never by the mean, so no
                # mean is injected and no coupling is fabricated. Both members of a pair sit at the
                # same predicted mean by construction.
                # Both levels are written too, so any other normalization stays possible offline
                # without re-running this.
                sg_A = np.where(inside, sg_a, np.nan)
                sg_B = np.where(inside, fb_sg, np.nan)
                with np.errstate(divide="ignore", invalid="ignore"):
                    # a degenerate gene can give sigma exactly 0 (a count mean that underflows);
                    # those pairs drop out of the log mean and are counted in n_pairs_log
                    dls = np.where(inside & (sg_a > 0) & (fb_sg > 0),
                                   np.log(sg_a) - np.log(fb_sg), np.nan)
                with np.errstate(invalid="ignore"):
                    match_rows.append(pd.DataFrame(dict(
                        gene=genes[keep], slide=str(sl), test_donor=donor, label_col=lab_col,
                        tissue=t, ref_tissue=ref,
                        n_pairs=n_pairs[keep].astype(np.int32),
                        match_rate=(n_pairs[keep] / mu_a.shape[0]),
                        mean_abs_dmu=np.nanmean(adm, axis=0)[keep],
                        mean_dsigma=np.nanmean(dsg, axis=0)[keep],
                        median_dsigma=np.nanmedian(dsg, axis=0)[keep],
                        mean_sigma_a=np.nanmean(sg_A, axis=0)[keep],
                        mean_sigma_b=np.nanmean(sg_B, axis=0)[keep],
                        mean_dlogsigma=np.nanmean(dls, axis=0)[keep],
                        n_pairs_log=np.isfinite(dls).sum(axis=0)[keep].astype(np.int32),
                        mean_derr=np.nanmean(der, axis=0)[keep],
                        median_derr=np.nanmedian(der, axis=0)[keep])))
                del dsg, der, adm, sg_A, sg_B, dls, fb_sg, fb_er, dmu, inside

            # ---- the textbook alternative: regress out a LINEAR mean, then take tissue means ----
            # Slope per gene from the pooled annotated spots of this (gene, slide). Assumes a
            # straight line, which matching does not; agreement between the two is the point.
            mo, so, eo = mu_s[ok], sg_s[ok], er_s[ok]
            mc = mo - mo.mean(axis=0)
            var = (mc ** 2).sum(axis=0)
            var = np.where(var > 0, var, np.nan)
            b_sg = (mc * (so - so.mean(axis=0))).sum(axis=0) / var
            b_er = (mc * (eo - eo.mean(axis=0))).sum(axis=0) / var
            r_sg = so - (so.mean(axis=0) + mc * np.nan_to_num(b_sg))
            r_er = eo - (eo.mean(axis=0) + mc * np.nan_to_num(b_er))
            t_ok = tis[ok]
            for t in names:
                sel = t_ok == t
                if sel.sum() < args.min_pairs:
                    continue
                anc_rows.append(pd.DataFrame(dict(
                    gene=genes, slide=str(sl), test_donor=donor, label_col=lab_col, tissue=t,
                    slope=b_sg, resid_sigma=r_sg[sel].mean(axis=0), resid_err=r_er[sel].mean(axis=0),
                    n=int(sel.sum()))))
            del mo, so, eo, mc, r_sg, r_er

            # ---- spot-level cloud for the named genes, first label column only ----
            if lab_col == LABEL_COLS[0] and ex_idx:
                for gi in ex_idx:
                    ex_rows.append(pd.DataFrame(dict(
                        gene=genes[gi], slide=str(sl), test_donor=donor,
                        tissue=tis[ok], mu=mu_s[ok, gi], sigma=sg_s[ok, gi], err=er_s[ok, gi])))

    seen.append(adata_row)
    print(f"[{_el():6.1f}s] {f.name}: donor {donor}, {mu.shape[0]:,} test spots x {G} genes, "
          f"{len(pd.unique(slides))} slide(s)", flush=True)
    del mu, sigma, err

all_rows = np.concatenate(seen)
assert np.unique(all_rows).size == all_rows.size, f"{run_dir}: test spots overlap -- non-LOPO split?"

if not match_rows:
    write_empty("no (gene, slide, tissue) cell reached the minimum number of pairs")

MT = pd.concat(match_rows, ignore_index=True)
AN = pd.concat(anc_rows, ignore_index=True) if anc_rows else pd.DataFrame()
EX = pd.concat(ex_rows, ignore_index=True) if ex_rows else pd.DataFrame(
    columns=["gene", "slide", "test_donor", "tissue", "mu", "sigma", "err"])
MT.to_parquet(out_dir / "matched.parquet", index=False)
AN.to_parquet(out_dir / "ancova.parquet", index=False)
EX.to_parquet(out_dir / "examples.parquet", index=False)

missing = sorted(set(want_genes) - set(EX["gene"].unique())) if len(EX) else want_genes
with open(out_dir / "summary.json", "w") as fh:
    json.dump(dict(dataset=dataset, model_type=model_type, family=family, run_dir=str(run_dir),
                   input_adata=str(data_path), native_scale=native,
                   label_cols=LABEL_COLS, caliper_sd=args.caliper_sd,
                   min_pairs=args.min_pairs, min_tissue=args.min_tissue,
                   ref_tissue=sorted(MT["ref_tissue"].unique().tolist()),
                   tissues=sorted(MT["tissue"].unique().tolist()),
                   donors=sorted(MT["test_donor"].unique().tolist()),
                   n_rows_matched=int(len(MT)), n_rows_ancova=int(len(AN)),
                   example_genes_requested=want_genes, example_genes_missing=missing,
                   median_match_rate=float(MT["match_rate"].median()),
                   median_abs_dmu=float(MT["mean_abs_dmu"].median())), fh, indent=2)

print(f"[{_el():6.1f}s] wrote {out_dir}", flush=True)
print(f"  matched {MT.shape}   ancova {AN.shape}   examples {EX.shape}", flush=True)
print(f"  READ THESE FIRST: median match rate {MT['match_rate'].median():.3f}, "
      f"median |dmu| inside pairs {MT['mean_abs_dmu'].median():.6f}", flush=True)
print("  A near-zero |dmu| is what makes the contrast readable; a low match rate means the "
      "comparison moved to a sub-population.", flush=True)
