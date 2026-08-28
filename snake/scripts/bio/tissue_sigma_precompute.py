# snake/scripts/bio/tissue_sigma_precompute.py
#
# Does the predicted uncertainty vary with the tissue, and does it still vary once the predicted
# mean is held fixed? One classic run dir in, one small per-(gene, slide, tissue) table out.
#
# Why per gene and not averaged over genes. Most of the variance of log(sigma) is a per-gene
# constant, so any statistic that averages sigma over the panel is dominated by gene composition:
# two tissues would be compared mostly on which genes happen to be expressed in them. Staying
# inside one (gene, slide) removes that term by construction rather than by modeling it.
#
# How the predicted mean is controlled: direct standardisation, no functional form. Inside each
# (gene, slide) the spots are cut into B equal-size bins of the predicted mean, the tissue means are
# taken within each bin, and those bin-level means are averaged with equal weight. A tissue is only
# ever compared to another tissue at a comparable predicted mean, and nothing is assumed about the
# shape of sigma(mu). This is preferred over a loess or isotonic residual, which assumes the fit is
# right, and over sigma/mu, which assumes sigma is proportional to mu.
#   sigma_raw = the plain tissue mean, confounded, reported for reference
#   sigma_adj = the mu-standardised tissue mean, the readable one
# The gap between the two is the part of the tissue contrast that was only the expression level.
#
# The coefficient of variation cv = sigma / mu is exported per cell as a secondary view. For the NB
# heads it has an exact reading:
#       cv^2 = (mu + alpha*mu^2) / mu^2 = 1/mu + alpha
# so alpha is the part of the relative spread that does not depend on the mean, by construction of
# the parametrisation. That makes the NB dispersion the natural mean-free uncertainty, and it is
# exported alongside sigma with the identical stratification, so the two can be read side by side.
#
# Native scale, and how it differs from bio_axis_precompute.py. Here mu, sigma and the error all sit
# on the same axis, the one the uncertainty natively lives on: log1p-CP10k for the Gaussian heads (X
# and sqrt(pred_var) are both in log space), the model's own count target divided by depth_scale for
# the count heads. bio_axis pushes the Gaussian mean through expm1 to reach CP10k, which is right
# for reporting a mean but would make sigma/mu mix two scales. Do not compare a mu column across the
# two files.
#
# sigma is exported raw; only a per-run constant (depth_scale) ever touches it. The standardisation
# acts on which spots are averaged together, never on the value of sigma.
#
# Scope: test spots only, so under LOPO every spot appears exactly once, predicted by the model that
# never saw its donor. A cohort with no per-spot tissue annotation is not an early exit: the shares
# and the two spread tables need no label, so only per_gene_tissue.parquet comes out empty.
#
# Reads <run-dir>/metrics.json, <run-dir>/predictions/preds_fold*.h5ad and the merged preprocessed
# adata named by metrics.json["input_adata"].
#
# Writes (under <run-dir>/tissue_sigma/):
#   per_gene_tissue.parquet   gene, slide, test_donor, label_col, tissue, n, n_bins_used,
#                             min_bin_n, median_bin_n, sigma_raw/adj, mu_raw/adj, err_raw/adj,
#                             cv_raw/adj, and alpha_raw/adj for an NB head only
#   shares.parquet            two-way variance decomposition of log(sigma), log|err|, log(mu) and
#                             log(alpha) per donor, so the NB dispersion and sigma are read off the
#                             same decomposition.
#   spread_gene.parquet       gene, test_donor, quantity, sd_within_gene: the SD of that quantity
#                             across the fold's test spots.
#   spread_spot.parquet       adata_row, test_donor, quantity, sd_within_spot: the SD across genes.
#   summary.json              run identity, scales, label columns, bin settings, row counts.
#
# Usage (repo root, on the cluster; CPU only):
#   python -u snake/scripts/bio/tissue_sigma_precompute.py \
#       --run-dir <results>/10x_TuPro/<run_id>/_10x_TuPro_hvg_top3000/r75/heteroscedastic/lambda_0.0
#   # whole sweep: bash snake/scripts/bio/launch_tissue_sigma_precompute.sh

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
p.add_argument("--out-dir", default=None, help="Default <run-dir>/tissue_sigma/.")
p.add_argument("--data", default=None, help="Override metrics.json['input_adata'].")
p.add_argument("--n-bins", type=int, default=5, help="Equal-size predicted-mean bins per (gene, slide).")
p.add_argument("--min-cell", type=int, default=5, help="Min spots in a (gene, slide, tissue, bin) cell.")
p.add_argument("--min-tissue", type=int, default=30, help="Min spots for a (slide, tissue) to be kept.")
args = p.parse_args()

run_dir = Path(args.run_dir)
out_dir = Path(args.out_dir) if args.out_dir else run_dir / "tissue_sigma"
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
    """Exit 0 with the reason on disk: a launcher sweeping every run must not see this as failure."""
    print(f"{reason} -- writing an EMPTY per_gene_tissue.parquet.", flush=True)
    pd.DataFrame(columns=["gene", "slide", "test_donor", "label_col", "tissue"]).to_parquet(
        out_dir / "per_gene_tissue.parquet", index=False)
    pd.DataFrame(columns=["test_donor", "quantity"]).to_parquet(out_dir / "shares.parquet", index=False)
    pd.DataFrame(columns=["gene", "test_donor", "quantity", "sd_within_gene"]).to_parquet(
        out_dir / "spread_gene.parquet", index=False)
    pd.DataFrame(columns=["adata_row", "test_donor", "quantity", "sd_within_spot"]).to_parquet(
        out_dir / "spread_spot.parquet", index=False)
    with open(out_dir / "summary.json", "w") as fh:
        json.dump(dict(dataset=dataset, model_type=model_type, empty=True, reason=reason), fh, indent=2)
    sys.exit(0)


if model_type == "msemeanonly":
    write_empty("msemeanonly has no uncertainty head")

# ── The one constant, resolved once ───────────────────────────────────────────────────────────
if family == "continuous":
    X_DIV, native = 1.0, "log1p-CP10k"
else:
    if M.get("count_convention") != "v2_no_offset":
        raise ValueError(f"{run_dir}: count artifact with no count_convention marker; re-run the precompute.")
    _ds = M.get("depth_scale")
    X_DIV = 1.0 if _ds is None else float(_ds)
    native = "raw counts" if (_ds is None and M.get("target_source") == "raw_counts") else "CP10k"
print(f"scale   : X_DIV={X_DIV}; mu, sigma and error all on {native}", flush=True)

# ── Tissue labels, from the merged adata (obs only, read backed) ──────────────────────────────
data_path = args.data or M["input_adata"]
A = ad.read_h5ad(data_path, backed="r")
obs_all = A.obs.copy()
A.file.close()
del A
# Only the SPATIAL imputation is admissible: `expr` imputes from the model target and `emb` from
# the model input, so either would define the tissue group with a quantity derived from what is
# then averaged inside it.
LABEL_COLS = [c for c in ("ground_truth", "ground_truth_knn_spatial_15") if c in obs_all.columns]
# A cohort with no per-spot annotation is not an early exit: the variance shares and the
# within-gene / within-spot spreads need no tissue label. Only the per-(gene, slide, tissue) table
# ends up empty, which the tail handles.
print(f"labels  : {LABEL_COLS if LABEL_COLS else 'NONE (tissue table will be empty, shares and '
                                                 'spreads are still computed)'}", flush=True)
LABELS = {c: obs_all[c].to_numpy().astype(str) for c in LABEL_COLS}
del obs_all

# "No label" is not spelled one single way, and a sentinel slipping through would become a real
# tissue with its own mean. Pigment is dropped for size, not because it is not a tissue.
DROP = {"PIGMENT", "UNASSIGNED", "NAN", "NONE", "NA", "N/A", "", "<NA>", "MIXED", "WHITESPACE"}
B = args.n_bins

pred_files = sorted((run_dir / "predictions").glob("preds_fold*.h5ad"))
assert pred_files, f"no predictions in {run_dir / 'predictions'}"

rows, share_rows, seen = [], [], []
spread_gene_rows, spread_spot_rows = [], []
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
    # NATIVE axis: mu, sigma and the error share one scale, so a ratio between them is meaningful.
    if family == "continuous":
        mu, y = X, Y                                                   # both log1p-CP10k
        sigma = np.sqrt(_dense(P.layers["pred_var"]).astype(np.float32))
        alpha = None
    else:
        mu, y = X / X_DIV, Y / X_DIV
        if family == "poisson":
            sigma = np.sqrt(count_pred_var(X.astype(np.float64)) / X_DIV ** 2).astype(np.float32)
            alpha = None
        else:
            alpha = _dense(P.layers["dispersion"]).astype(np.float32)   # at X's own scale
            sigma = np.sqrt(count_pred_var(X.astype(np.float64), dispersion=alpha.astype(np.float64))
                            / X_DIV ** 2).astype(np.float32)
    err = np.abs(mu - y)
    del X, Y, P

    # ---- two-way variance shares, including log(alpha): the NB dispersion on the very axis that
    # made sigma look uninformative. cv^2 = 1/mu + alpha, so alpha is the mean-free part of the
    # relative spread; if it puts more of its variance in the interaction term than sigma does,
    # the dispersion carries per-(spot, gene) structure that sigma alone hides.
    n_spots = mu.shape[0]
    pos = err[err > 0]
    eps_err = float(0.01 * np.median(pos)) if pos.size else 1e-6
    # log_mu is the expression arm of the decomposition, on the same log axis as the other three so
    # the four shares are comparable. The Gaussian mean lives on log1p-CP10k and can be negative
    # there, so it is clipped at a tiny positive value and the clipped count is recorded. The count
    # means are strictly positive (exp link) and are never clipped.
    n_clipped = int((mu <= 0).sum())
    for qname, Q in (("log_sigma", np.log(sigma.astype(np.float64) + 1e-12)),
                     ("log_abs_err", np.log(err.astype(np.float64) + eps_err)),
                     ("log_mu", np.log(np.clip(mu.astype(np.float64), 1e-6, None))),
                     ("log_alpha", None if alpha is None else np.log(alpha.astype(np.float64) + 1e-12))):
        if Q is None:
            continue
        a_j, b_i, g = Q.mean(axis=0), Q.mean(axis=1), float(Q.mean())
        ss_tot = float(((Q - g) ** 2).sum())
        ss_gene = float(n_spots * ((a_j - g) ** 2).sum())
        ss_spot = float(G * ((b_i - g) ** 2).sum())
        resid = Q - a_j[None, :] - b_i[:, None] + g
        share_rows.append(dict(test_donor=donor, quantity=qname,
                               share_gene=ss_gene / ss_tot, share_spot=ss_spot / ss_tot,
                               share_interaction=(ss_tot - ss_gene - ss_spot) / ss_tot,
                               sd_interaction=float(resid.std()), n_spots=n_spots, n_genes=G,
                               n_clipped=(n_clipped if qname == "log_mu" else 0)))
        # The same fact as the shares, but as two distributions instead of two scalars, which is
        # what a reader can actually look at. WITHIN-GENE spread = how much the quantity moves from
        # spot to spot for a fixed gene; WITHIN-SPOT spread = how much it moves from gene to gene
        # for a fixed spot. The two are computed on different counts (n_spots vs n_genes), but a
        # standard deviation is not biased by its sample size, so the comparison is legitimate --
        # only the smoothness of a density drawn from them differs.
        spread_gene_rows.append(pd.DataFrame(dict(
            gene=genes, test_donor=donor, quantity=qname, sd_within_gene=Q.std(axis=0))))
        spread_spot_rows.append(pd.DataFrame(dict(
            adata_row=adata_row, test_donor=donor, quantity=qname, sd_within_spot=Q.std(axis=1))))
        del Q, a_j, b_i, resid

    # ---- per (gene, slide, tissue), with the predicted mean held fixed ----
    # Everything below is vectorised per slide. The bins are ranks of mu WITHIN each (gene, slide),
    # so each gene gets its own cut points and a gene expressed only in one tissue simply ends up
    # with that tissue in every bin (and is then dropped by the coverage rule at read time).
    for sl in pd.unique(slides):
        m = slides == sl
        n_s = int(m.sum())
        if n_s < B * args.min_cell:
            continue
        mu_s, sg_s, er_s = mu[m], sigma[m], err[m]
        al_s = alpha[m] if alpha is not None else None
        # equal-size bins of the predicted mean, per gene (column-wise ranks)
        order = np.argsort(mu_s, axis=0, kind="stable")
        ranks = np.empty(mu_s.shape, dtype=np.int32)
        np.put_along_axis(ranks, order, np.arange(n_s, dtype=np.int32)[:, None].repeat(G, axis=1), axis=0)
        bins = (ranks.astype(np.int64) * B) // n_s
        del order, ranks

        for lab_col in LABEL_COLS:
            tis = LABELS[lab_col][adata_row[m]]
            keep_spot = ~np.isin(np.char.upper(tis.astype(str)), list(DROP))
            if keep_spot.sum() < args.min_cell:
                continue
            names, codes = np.unique(tis[keep_spot], return_inverse=True)
            T = len(names)
            big = np.array([int((tis[keep_spot] == t).sum()) >= args.min_tissue for t in names])
            if not big.any():
                continue
            # flat cell index over (gene, tissue, bin); one bincount per accumulated quantity
            gi = np.arange(G, dtype=np.int64)[None, :]
            idx = (gi * (T * B) + codes[:, None].astype(np.int64) * B + bins[keep_spot]).ravel()
            size = G * T * B
            cnt = np.bincount(idx, minlength=size).astype(np.float64)
            acc = {k: np.bincount(idx, weights=v[keep_spot].ravel().astype(np.float64), minlength=size)
                   for k, v in (("sigma", sg_s), ("mu", mu_s), ("err", er_s))}
            if al_s is not None:
                acc["alpha"] = np.bincount(idx, weights=al_s[keep_spot].ravel().astype(np.float64),
                                           minlength=size)
            del idx

            cnt3 = cnt.reshape(G, T, B)
            usable = cnt3 >= args.min_cell          # a bin only counts if the tissue is present in it
            n_bins_used = usable.sum(axis=2)
            n_tot = cnt3.sum(axis=2)
            # How WELL FILLED the used bins are, not just how many there are. A cell standardised on
            # five bins of five spots is not the one standardised on five bins of two hundred, and
            # the bin count alone cannot tell them apart.
            _c = np.where(usable, cnt3, np.nan)
            with np.errstate(invalid="ignore"):
                min_bin_n = np.nan_to_num(np.nanmin(_c, axis=2), nan=0.0)
                med_bin_n = np.nan_to_num(np.nanmedian(_c, axis=2), nan=0.0)
            del _c
            out = {}
            for k, v in acc.items():
                v3 = v.reshape(G, T, B)
                with np.errstate(invalid="ignore", divide="ignore"):
                    cell = np.where(usable, v3 / np.where(cnt3 > 0, cnt3, 1), np.nan)
                    out[k + "_adj"] = np.nanmean(cell, axis=2)        # equal weight per mu bin
                    out[k + "_raw"] = np.where(n_tot > 0, v3.sum(axis=2) / np.where(n_tot > 0, n_tot, 1), np.nan)
            del acc, cnt3

            gg, tt = np.meshgrid(np.arange(G), np.arange(T), indexing="ij")
            sel = (n_bins_used >= 2) & big[None, :]   # need >= 2 bins for a standardised mean
            if not sel.any():
                continue
            d = dict(gene=genes[gg[sel]], slide=str(sl), test_donor=donor, label_col=lab_col,
                     tissue=names[tt[sel]], n=n_tot[sel].astype(np.int64),
                     n_bins_used=n_bins_used[sel].astype(np.int16),
                     min_bin_n=min_bin_n[sel].astype(np.int32),
                     median_bin_n=med_bin_n[sel].astype(np.int32))
            for k in ("sigma", "mu", "err") + (("alpha",) if al_s is not None else ()):
                d[k + "_raw"] = out[k + "_raw"][sel]
                d[k + "_adj"] = out[k + "_adj"][sel]
            # cv on the NATIVE axis: for the NB, cv^2 = 1/mu + alpha exactly.
            d["cv_raw"] = d["sigma_raw"] / np.maximum(d["mu_raw"], 1e-9)
            d["cv_adj"] = d["sigma_adj"] / np.maximum(d["mu_adj"], 1e-9)
            rows.append(pd.DataFrame(d))
            del out, d
        del mu_s, sg_s, er_s, bins

    seen.append(adata_row)
    print(f"[{_el():6.1f}s] {f.name}: donor {donor}, {n_spots:,} test spots x {G} genes, "
          f"{len(pd.unique(slides))} slide(s)", flush=True)
    del mu, sigma, err, alpha

all_rows = np.concatenate(seen)
assert np.unique(all_rows).size == all_rows.size, f"{run_dir}: test spots overlap -- non-LOPO split?"

TCOLS = ["gene", "slide", "test_donor", "label_col", "tissue", "n", "n_bins_used",
         "min_bin_n", "median_bin_n",
         "sigma_raw", "sigma_adj", "mu_raw", "mu_adj", "err_raw", "err_adj", "cv_raw", "cv_adj"]
T = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=TCOLS)
if not rows:
    print("no (gene, slide, tissue) cell: the cohort has no tissue annotation, or none survived "
          "the coverage rules. shares and spreads are still written.", flush=True)
S = pd.DataFrame(share_rows)
T.to_parquet(out_dir / "per_gene_tissue.parquet", index=False)
S.to_parquet(out_dir / "shares.parquet", index=False)
pd.concat(spread_gene_rows, ignore_index=True).to_parquet(out_dir / "spread_gene.parquet", index=False)
pd.concat(spread_spot_rows, ignore_index=True).to_parquet(out_dir / "spread_spot.parquet", index=False)

_g = S.groupby("quantity")[["share_gene", "share_interaction", "sd_interaction"]].mean()
with open(out_dir / "summary.json", "w") as fh:
    json.dump(dict(dataset=dataset, model_type=model_type, family=family, run_dir=str(run_dir),
                   input_adata=str(data_path), native_scale=native, depth_scale=M.get("depth_scale"),
                   label_cols=LABEL_COLS, n_bins=B, min_cell=args.min_cell,
                   min_tissue=args.min_tissue, n_rows=int(len(T)),
                   tissues={c: sorted(T[T["label_col"] == c]["tissue"].unique().tolist())
                            for c in LABEL_COLS},
                   donors=sorted(T["test_donor"].unique().tolist()),
                   has_alpha=bool("alpha_raw" in T.columns),
                   mean_shares=_g.round(4).to_dict(orient="index"),
                   no_delta_method=True), fh, indent=2)

print(f"[{_el():6.1f}s] wrote {out_dir}", flush=True)
print(f"  per_gene_tissue {T.shape}   shares {S.shape}", flush=True)
print(_g.round(4).to_string(), flush=True)
print("  read share_interaction: the fraction of the quantity's variance that depends on BOTH the "
      "spot and the gene. Compare log_sigma (and log_alpha, if present) against log_abs_err.", flush=True)
