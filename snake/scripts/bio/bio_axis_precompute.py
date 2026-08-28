# snake/scripts/bio/bio_axis_precompute.py
#
# Biological-insight axes export for ONE classic run dir. Three reductions of the same per-fold
# prediction matrices, produced in one pass:
#   per gene    which genes are predictable, and which the model is least sure about.
#   per spot    where in the tissue the model is uncertain, and whether sigma rises on spots that
#               mix several tissue compartments.
#   SS shares   how much of log(sigma) is a per-gene constant, and the same share for log|error|.
#               The gap between the two is the readable quantity, never the level of either: a high
#               gene share for sigma means nothing if the error is equally gene-level.
#
# sigma is exported raw: sqrt(pred_var) for the Gaussians, sqrt(mu + alpha*mu^2) for NB. The expm1
# Jacobian (1 + mu) is a function of the mean, so applying it would put mu into the sigma axis and
# manufacture the coupling this analysis measures. Only a per-run constant (depth_scale, for the
# count heads) is applied. Consequence: the Gaussian sigma is in log1p-CP10k units and the count
# sigma in CP10k units, so the two families never share a sigma axis. summary.json records which.
#
# sigma_z, the second per-spot index. A plain mean of sigma over the panel is dominated by the
# highly expressed genes, so two spots would be compared mostly on gene composition. sigma_z
# standardises each gene's log(sigma) across the fold's test spots, then averages over genes: a
# per-gene constant computed across spots, so it injects no per-spot mean. sigma_mean_raw is
# exported beside it because the two answer different questions and can disagree.
#   Standardising within the fold also removes the donor effect, since one LOPO fold is one
#   held-out donor. A gene whose sigma is constant within the fold (the gene-homoscedastic
#   Gaussian) has SD 0, so its z is set to 0 and it is counted in summary.json under
#   n_gene_columns_with_zero_sd. That head gives sigma_z == 0 everywhere, which is a built-in
#   sanity check.
#
# Scope: test spots only (obs["split"] == "test"). Under LOPO each spot is the test spot of exactly
# one fold, so the folds' test sets are disjoint and their union is every spot once, each predicted
# by the model that never saw its donor. Asserted after the fold loop.
#
# Cost: the SS shares and the per-gene table are panel-wide, so each fold's whole test block is
# materialised, (n_test x n_genes) x 3 layers, float32 with float64 reductions. CPU only: nothing
# here calls count_expected_log1p.
#
# Reads <run-dir>/metrics.json, <run-dir>/predictions/preds_fold*.h5ad, <run-dir>/atoms.parquet and
# the merged preprocessed adata named by metrics.json["input_adata"].
#
# Writes (under <run-dir>/bio_axis/):
#   per_gene.parquet     gene, test_donor, mean_sigma_raw, mean_log_sigma, mean_mu, mean_y,
#                        mean_abs_err, mean_log_abs_err, pct_zero, n_spots, + pearson (and spearman
#                        when atoms carries it) joined from the run's own atoms.parquet, so the
#                        accuracy number follows the headline convention instead of being
#                        recomputed here.
#   per_gene_tissue.parquet  gene, tissue, slide, test_donor, label_col, n_spots, mean_log_sigma,
#                        mean_sigma, mean_pred, mean_target, mean_log_err, mean_err, pct_zero. One
#                        axis only, the model's own, so any two columns can be contrasted directly;
#                        the column names differ from per_gene's on purpose. One row per gene per
#                        tissue per slide, so a slide-to-slide difference within one donor stays
#                        visible as the noise floor a claimed tissue effect has to beat. Empty but
#                        typed for a cohort with no annotation.
#   per_spot.parquet     obs_name, adata_row, slide, test_donor, sigma_mean_raw, sigma_z,
#                        err_mean_raw, err_z, mu_mean, mu_z, y_mean, frac_zero, + every joined obs
#                        column that exists (tissue labels, spatial x/y, the k-NN label
#                        homogeneity, depth, and whatever annotation extras the cohort carries).
#   variance_shares.parquet  test_donor, quantity ("log_sigma"|"log_abs_err"), share_gene,
#                        share_spot, share_interaction, sd_total, sd_interaction, n_spots, n_genes.
#   summary.json         run identity, resolved scales, the eps used, label columns found, row
#                        counts. Read this before trusting an empty or surprising parquet.
#
# Usage (repo root, on the cluster; CPU only):
#   python -u snake/scripts/bio/bio_axis_precompute.py \
#       --run-dir <results>/10x_TuPro/<run_id>/_10x_TuPro_hvg_top3000/r75/heteroscedastic/lambda_0.0
#   # whole sweep: bash snake/scripts/bio/launch_bio_axis_precompute.sh

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
from sklearn.neighbors import NearestNeighbors

_T0 = time.time()
def _el():
    return time.time() - _T0

p = argparse.ArgumentParser()
p.add_argument("--run-dir", required=True,
               help="A classic run leaf dir: holds metrics.json and predictions/preds_fold*.h5ad "
                    "(for the Gaussian heads that is the lambda_* sub-dir).")
p.add_argument("--out-dir", default=None, help="Default <run-dir>/bio_axis/.")
p.add_argument("--data", default=None,
               help="Override the merged preprocessed adata path (default metrics.json['input_adata']).")
p.add_argument("--n-neighbors", type=int, default=6,
               help="Neighbors used for the spatial label-homogeneity index (default 6, the Visium "
                    "hex ring). Self is excluded.")
args = p.parse_args()

run_dir = Path(args.run_dir)
out_dir = Path(args.out_dir) if args.out_dir else run_dir / "bio_axis"
out_dir.mkdir(parents=True, exist_ok=True)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(PROJECT_ROOT)
from src.evaluation.count_metrics import count_pred_var

# ── Family taxonomy — the aleatoric formula differs by family, nothing else does ───────────────
# Same three sets as canal2_uncertainty_precompute.py, copied here because that is an
# argparse script, not an importable module.
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
    # summary that says why.
    with open(out_dir / "summary.json", "w") as fh:
        json.dump(dict(dataset=dataset, model_type=model_type, empty=True,
                       reason=f"model_type={model_type!r} has no uncertainty head"), fh, indent=2)
    print(f"{model_type} has no uncertainty head -- nothing to export.", flush=True)
    sys.exit(0)

family = model_family(model_type)

# ── The count scale — one CONSTANT per run, never a Jacobian ──────────────────────────────────
# A count artifact fits its likelihood directly on its own target: X is mu on that scale and
# layers["dispersion"] is alpha at the same scale, marked by count_convention == "v2_no_offset".
# depth_scale divides the mean and its square divides the variance, which changes units without
# putting mu into sigma. An artifact without that marker is rejected, never guessed at.
if family == "continuous":
    X_DIV = 1.0
    mu_scale, sigma_scale = "CP10k", "log1p-CP10k"    # sigma = sqrt(pred_var), left in log1p space
    depth_blind = False
else:
    if M.get("count_convention") != "v2_no_offset":
        raise ValueError(f"{run_dir}: count artifact with no count_convention marker "
                         f"(count_convention={M.get('count_convention')!r}). Such a file is rejected rather than reinterpreted.")
    _ds = M.get("depth_scale")
    X_DIV = 1.0 if _ds is None else float(_ds)
    depth_blind = _ds is None and M.get("target_source") == "raw_counts"
    mu_scale = "raw counts" if depth_blind else "CP10k"
    sigma_scale = mu_scale
print(f"scale   : X_DIV={X_DIV}; mu on {mu_scale}, sigma on {sigma_scale} (RAW, no delta method)"
      + ("   [DEPTH-BLIND: comparable only to other *_raw runs]" if depth_blind else ""), flush=True)

# ── Per-spot metadata from the merged adata, joined later by adata_row ─────────────────────────
# Read BACKED: obs and obsm come off disk without ever materialising X, which is the whole
# (n_spots x ~17k) expression matrix and is not needed here.
data_path = args.data or M["input_adata"]
A = ad.read_h5ad(data_path, backed="r")
obs_all = A.obs.copy()
spatial_all = np.asarray(A.obsm["spatial"], dtype=np.float64) if "spatial" in A.obsm else None
A.file.close()
del A
print(f"[{_el():6.1f}s] merged adata obs: {obs_all.shape[0]:,} spots, "
      f"spatial={'yes' if spatial_all is not None else 'MISSING'}", flush=True)

# Tissue label columns. `ground_truth` is the expert annotation; the `_knn_` variants are the
# spatially-imputed extensions that exist only for cohorts with UNASSIGNED spots. Only the SPATIAL
# imputation is carried: `expr` imputes from the model TARGET and `emb` from the model INPUT, so
# either would define the tissue group with the very quantity whose per-group mean is then
# reported. LUNG_CANCER is expected to have no label column at all, which is a skip, not a failure.
LABEL_COLS = [c for c in ("ground_truth", "ground_truth_knn_spatial_15") if c in obs_all.columns]
# Everything else worth joining, taken only if present. label_conf and the per-tissue proportion
# columns (TuPro) are an annotation-confidence / spot-purity axis; blur_vars is a per-spot H&E
# focus measure; the size factor is the sequencing-depth covariate a low spot-share of sigma must
# be checked against.
EXTRA_CANDIDATES = ["label", "label_conf", "blur_vars", "foreground_ratios", "n_counts",
                    "size_factor_total_over_1e4", "size_factor_median_ratios", "sample_id",
                    "Whitespace", "Normal lymphoid tissue", "Pigment", "Stroma", "Tumor",
                    "Blood and necrosis", "Mixed"]
EXTRA_COLS = [c for c in EXTRA_CANDIDATES if c in obs_all.columns]
print(f"          label columns: {LABEL_COLS or 'NONE'}", flush=True)
print(f"          extra obs columns joined: {EXTRA_COLS}", flush=True)

meta = pd.DataFrame(index=obs_all.index)
for c in LABEL_COLS:
    meta[c] = obs_all[c].to_numpy().astype(str)
for c in EXTRA_COLS:
    meta[c] = obs_all[c].to_numpy()
if spatial_all is not None:
    meta["spatial_x"] = spatial_all[:, 0]
    meta["spatial_y"] = spatial_all[:, 1]

# ── Spatial label homogeneity: the mixture index the "purity" analysis rests on ────────────────
# For each spot, the fraction of its k nearest neighbors WITHIN ITS OWN SLIDE that carry the same
# tissue label. 1.0 = deep inside a homogeneous region; low = a boundary or an isolated spot, i.e.
# a spot whose ~10 pooled cells plausibly span two transcriptional programmes. Neighbors are taken
# per slide because two slides' coordinate systems are unrelated -- pooling them would pair spots
# from different tissues as neighbors.
# A spot whose own label is a no-label sentinel gets NaN (it has no label to agree with); a
# NEIGHBOR carrying a sentinel simply counts as "different", which is the conservative reading.
_NO_LABEL = {"UNASSIGNED", "NAN", "NONE", "NA", "N/A", "", "<NA>", "NAN "}


def label_homogeneity(labels, coords, slides, k):
    out = np.full(labels.shape[0], np.nan)
    for s in pd.unique(slides):
        idx = np.flatnonzero(slides == s)
        if idx.size <= k:
            continue                      # too few spots on this slide for a k-ring
        nn = NearestNeighbors(n_neighbors=k + 1).fit(coords[idx])
        _, nbr = nn.kneighbors(coords[idx])
        nbr = nbr[:, 1:]                  # column 0 is the spot itself
        lab = labels[idx]
        same = (lab[nbr] == lab[:, None]).mean(axis=1)
        same[np.isin(np.char.upper(lab.astype(str)), list(_NO_LABEL))] = np.nan
        out[idx] = same
    return out


if spatial_all is not None and LABEL_COLS and "sample_id" in obs_all.columns:
    slides_all = obs_all["sample_id"].to_numpy().astype(str)
    for c in LABEL_COLS:
        meta[f"homog_{c}"] = label_homogeneity(meta[c].to_numpy().astype(str),
                                               spatial_all, slides_all, args.n_neighbors)
        n_ok = int(np.isfinite(meta[f"homog_{c}"]).sum())
        print(f"[{_el():6.1f}s] homogeneity on {c!r}: {n_ok:,} spots scored "
              f"(mean {np.nanmean(meta[f'homog_{c}']):.3f})", flush=True)
else:
    print("          spatial homogeneity SKIPPED (needs obsm['spatial'], a label column and "
          "obs['sample_id'])", flush=True)

del obs_all, spatial_all

# Tissue label indexed by adata_row, for the per-(gene, tissue) reduction inside the fold loop.
# `meta` was built with one row per merged-adata spot in positional order, which is exactly what
# obs["adata_row"] indexes into (the same join the per-spot table does after the loop).
tissue_by_row = {c: meta[c].to_numpy().astype(str) for c in LABEL_COLS}
_NO_LABEL_LIST = list(_NO_LABEL)

# ── Per-fold extraction ───────────────────────────────────────────────────────────────────────
pred_files = sorted((run_dir / "predictions").glob("preds_fold*.h5ad"))
assert pred_files, f"no predictions in {run_dir / 'predictions'}"

gene_rows, gene_tissue_rows, spot_rows, share_rows, seen_rows = [], [], [], [], []
n_genes_zero_sd_total, eps_err_used, eps_err_native_used = 0, [], []


def z_mean(S):
    """Per-spot mean of the per-GENE standardised column, plus how many genes had no variation.

    Each gene is standardised across THIS fold's test spots, so a spot is scored on how unusual it
    is for each gene rather than on which genes happen to be highly expressed. That is a per-gene
    constant computed across spots: it never injects the per-spot mean (the no-delta-method rule).
    A gene with zero SD (the gene-homoscedastic Gaussian learns one log-variance per gene) carries
    no spot information and is set to 0 rather than dividing by zero.
    """
    sd = S.std(axis=0)
    zero = sd <= 0
    Z = (S - S.mean(axis=0)) / np.where(zero, 1.0, sd)
    Z[:, zero] = 0.0
    return Z.mean(axis=1), int(zero.sum())

for f in pred_files:
    B = sc.read_h5ad(f)
    B = B[B.obs["split"].to_numpy() == "test"].copy()
    donor = str(B.obs["test_donor"].iloc[0]) if "test_donor" in B.obs else f.stem
    genes = np.asarray(B.var_names).astype(str)
    n_spots, n_genes = B.shape

    X = _dense(B.X).astype(np.float32)
    Y = _dense(B.layers["y"]).astype(np.float32)
    if family == "continuous":
        mu = np.expm1(X)                                              # the ONLY transform on the mean
        y = np.expm1(Y)                                               # truth through the SAME transform
        sigma = np.sqrt(_dense(B.layers["pred_var"]).astype(np.float32))   # log1p space, untouched
    else:
        mu = X / X_DIV
        y = Y / X_DIV                                                 # the SAME constant on both sides
        if family == "poisson":
            var = count_pred_var(X.astype(np.float64)) / X_DIV ** 2
        else:
            alpha = _dense(B.layers["dispersion"]).astype(np.float64)  # alpha at X's own scale
            var = count_pred_var(X.astype(np.float64), dispersion=alpha) / X_DIV ** 2
        sigma = np.sqrt(var).astype(np.float32)
        del var
    # Transforming the ERROR is legitimate where transforming sigma is not: truth and prediction go
    # through the SAME change of variable, so nothing is manufactured and their difference means
    # something on one axis. That axis is CP10k, which is what makes this column comparable ACROSS
    # families, and is exactly why a second error is needed right below.
    abs_err = np.abs(mu - y)
    # The error on the head's own likelihood scale. For a Gaussian the expm1 above moves the mean
    # and the truth to CP10k while sigma stays in log1p-CP10k (it must: the expm1 Jacobian is
    # (1 + mu), so applying it to sigma would inject the mean into sigma). Consequence: abs_err and
    # sigma do not share an axis, they are separated by roughly that same (1 + mu), which would
    # re-enter the analysis through the error instead of through sigma. So the residual the NLL
    # models, |X - Y| in log1p-CP10k, is carried alongside. For the count heads mu, y and sigma are
    # all divided by the same constant, so they already share an axis and this is the same array.
    abs_err_native = np.abs(X - Y) if family == "continuous" else abs_err
    del X, Y

    # Logs. sigma is strictly positive by construction; the error is not (a perfect prediction, and
    # more commonly y == 0 where mu is tiny), so it needs a floor. The floor is set from the fold's
    # own positive errors rather than hard-coded, and is recorded, because the SS share of
    # log|error| is mildly sensitive to it.
    pos = abs_err[abs_err > 0]
    eps_err = float(0.01 * np.median(pos)) if pos.size else 1e-6
    eps_err_used.append(eps_err)
    log_sigma = np.log(sigma.astype(np.float64) + 1e-12)
    log_err = np.log(abs_err.astype(np.float64) + eps_err)
    # log1p, NOT log, on the mean: a Gaussian head predicts in log1p-CP10k space, so mu =
    # expm1(pred) is NEGATIVE wherever the prediction is, and log(mu) would be undefined. log1p(mu)
    # recovers that head's own prediction exactly and is bounded for the count heads. Consequence
    # to remember when reading a contrast on the mean: it is a true log ratio only for mu >> 1.
    log_mu = np.log1p(mu.astype(np.float64))
    # The TRUTH on the same axis. Needed because a contrast on log_mu answers "does the MODEL raise
    # its prediction in this tissue", which is not the same question as "is the gene more expressed
    # in this tissue". Both are wanted: the first is the redundancy control, the second is the
    # biology. For a Gaussian this recovers the target exactly (log1p(expm1(Y)) == Y).
    log_y = np.log1p(y.astype(np.float64))
    # Its own floor: the native residual lives on a different axis, so the CP10k eps would be wrong
    # by orders of magnitude. Identical to log_err for the count heads (same array, same floor).
    if family == "continuous":
        pos_n = abs_err_native[abs_err_native > 0]
        eps_native = float(0.01 * np.median(pos_n)) if pos_n.size else 1e-6
        log_err_native = np.log(abs_err_native.astype(np.float64) + eps_native)
        del pos_n
    else:
        eps_native = eps_err
        log_err_native = log_err
    eps_err_native_used.append(eps_native)
    del pos

    # ---- per gene (one row per gene, for THIS donor) ----
    gene_rows.append(pd.DataFrame({
        "gene": genes,
        "test_donor": donor,
        "mean_sigma_raw": sigma.mean(axis=0, dtype=np.float64),
        "mean_log_sigma": log_sigma.mean(axis=0),
        "mean_mu": mu.mean(axis=0, dtype=np.float64),
        "mean_y": y.mean(axis=0, dtype=np.float64),
        "mean_abs_err": abs_err.mean(axis=0, dtype=np.float64),
        "mean_log_abs_err": log_err.mean(axis=0),
        "pct_zero": (y == 0).mean(axis=0, dtype=np.float64),
        "n_spots": n_spots,
    }))

    rows_f = np.asarray(B.obs["adata_row"]).astype(np.int64)
    slides_f = np.asarray(B.obs["slide"]).astype(str)

    # ---- per (gene, tissue, slide): the tissue axis ----
    # Same reductions as the per-gene block above, but over the spots of one tissue on one slide.
    # Genes, tissues, slides and donors all stay keys: the only averaging is across spots. What to
    # pool afterwards is a downstream decision, not this script's.
    #
    # A tissue contrast is read from the log columns, because a difference of mean-logs is a log
    # ratio: Delta = mean_log_sigma(gene, tissue) - mean_log_sigma(gene, all tissues of the same
    # slide) says sigma is exp(Delta) times higher for that gene in that tissue. Two reasons it has
    # to be the log. Genes span orders of magnitude, so a raw difference would re-rank them by
    # magnitude. And it puts the contrast on sigma and the contrast on the mean in the same units,
    # which is the point: Delta_sigma tracking Delta_mu means the variance head only follows
    # expression, Delta_sigma tracking Delta_err instead is a real signal. Differencing a gene
    # against itself also cancels the gene main effect, so no per-gene constant and no function of
    # the mean is ever multiplied into sigma. mean_mu / mean_y stay raw: log1p is monotone, so it
    # does not change how genes rank by expression.
    #
    # One axis, the model's own. This table carries no CP10k version of the mean, the truth or the
    # error. That detour exists upstream only to make the mean comparable across families, and it
    # costs an axis split: expm1 moves the mean to CP10k while sigma must stay where it is, so a
    # CP10k error and a log1p-CP10k sigma are separated by the (1 + mu) Jacobian and their ratio is
    # not a z-score. Here every column lives where sigma lives, so any contrast between them means
    # something without a correction:
    #   mean_pred / mean_target   log1p of the mean and of the truth. For a Gaussian
    #                             log1p(expm1(X)) == X exactly, so these are the network output and
    #                             the target as trained. For a count head, log1p of the count.
    #   mean_err / mean_log_err   |X - Y| for a Gaussian, the residual its NLL models; |mu - y| on
    #                             its own count scale otherwise.
    # Column names differ from per_gene's (mean_mu / mean_y / mean_abs_err) so the two files cannot
    # be read as if they were on the same axis.
    #
    # No-label sentinels (UNASSIGNED and friends) are dropped rather than made their own group: on
    # TLS that category is a deliberate expert abstention, not a tissue. The imputed label column,
    # when the cohort has one, is carried in the same table under its own label_col value, so the
    # genuine-only view and the imputed extension are compared rather than silently merged.
    # A tiny group is kept, not filtered: n_spots travels with the row, so a reader can see what a
    # threshold would drop instead of it being frozen here.
    for lab_col in LABEL_COLS:
        tis_f = tissue_by_row[lab_col][rows_f]
        labeled = ~np.isin(np.char.upper(tis_f), _NO_LABEL_LIST)
        for t in np.unique(tis_f[labeled]):
            for s in np.unique(slides_f):
                m = np.flatnonzero((tis_f == t) & (slides_f == s) & labeled)
                if m.size == 0:
                    continue
                y_m = y[m]
                gene_tissue_rows.append(pd.DataFrame({
                    "gene": genes,
                    "tissue": t,
                    "slide": s,
                    "test_donor": donor,
                    "label_col": lab_col,
                    "n_spots": m.size,
                    "mean_log_sigma": log_sigma[m].mean(axis=0),
                    "mean_sigma": sigma[m].mean(axis=0, dtype=np.float64),
                    "mean_pred": log_mu[m].mean(axis=0),
                    "mean_target": log_y[m].mean(axis=0),
                    "mean_log_err": log_err_native[m].mean(axis=0),
                    "mean_err": abs_err_native[m].mean(axis=0, dtype=np.float64),
                    "pct_zero": (y_m == 0).mean(axis=0, dtype=np.float64),
                }))
                del y_m

    # ---- two-way SS decomposition, on log(sigma) AND on log|error| ----
    # For a complete (spots x genes) matrix S with column means a_j, row means b_i and grand mean g:
    #   SS_gene = n_spots * sum_j (a_j - g)^2 ; SS_spot = n_genes * sum_i (b_i - g)^2 ;
    #   SS_int  = SS_total - SS_gene - SS_spot   (exact, the two main effects are orthogonal here).
    # The share of sigma alone is uninterpretable: the row that matters is the GAP against the same
    # share computed on the error.
    for qname, S in (("log_sigma", log_sigma), ("log_abs_err", log_err)):
        a = S.mean(axis=0)
        b = S.mean(axis=1)
        g = float(S.mean())
        ss_total = float(((S - g) ** 2).sum())
        ss_gene = float(n_spots * ((a - g) ** 2).sum())
        ss_spot = float(n_genes * ((b - g) ** 2).sum())
        ss_int = ss_total - ss_gene - ss_spot
        # The double-centerd residual, in ABSOLUTE units: a 5% share can still be a 1.4-fold
        # spot-to-spot swing within a gene, so the share alone under-reports the effect.
        resid = S - a[None, :] - b[:, None] + g
        share_rows.append(dict(test_donor=donor, quantity=qname,
                               share_gene=ss_gene / ss_total, share_spot=ss_spot / ss_total,
                               share_interaction=ss_int / ss_total,
                               sd_total=float(np.sqrt(ss_total / S.size)),
                               sd_interaction=float(resid.std()),
                               n_spots=n_spots, n_genes=n_genes))
        del a, b, resid

    # ---- per spot ----
    # Two indices per quantity. The raw one is the plain mean over genes. The z one is z_mean above.
    sigma_z, n_zero_sd = z_mean(log_sigma)
    n_genes_zero_sd_total += n_zero_sd          # counted on sigma only; it is the diagnostic one
    spot = pd.DataFrame({
        "obs_name": np.asarray(B.obs["obs_name"]).astype(str),
        "adata_row": rows_f,
        "slide": slides_f,
        "test_donor": donor,
        "sigma_mean_raw": sigma.mean(axis=1, dtype=np.float64),
        "sigma_z": sigma_z,
        "err_mean_raw": abs_err.mean(axis=1, dtype=np.float64),
        "err_z": z_mean(log_err)[0],
        "mu_mean": mu.mean(axis=1, dtype=np.float64),
        "mu_z": z_mean(log_mu)[0],
        "y_mean": y.mean(axis=1, dtype=np.float64),
        "frac_zero": (y == 0).mean(axis=1, dtype=np.float64),
    })
    spot_rows.append(spot)
    seen_rows.append(spot["adata_row"].to_numpy())
    print(f"[{_el():6.1f}s] {f.name}: donor {donor}, {n_spots:,} test spots x {n_genes} genes"
          f"  share_gene(sigma)={share_rows[-2]['share_gene']:.3f}"
          f"  share_gene(err)={share_rows[-1]['share_gene']:.3f}", flush=True)
    del B, mu, y, sigma, abs_err, abs_err_native, log_sigma, log_err, log_err_native, log_mu, log_y, spot

# Under LOPO each spot is the held-out test spot of exactly one fold, so the folds' test sets are
# disjoint and their union is every spot once. Same invariant canal2 asserts after its concat.
all_rows = np.concatenate(seen_rows)
assert np.unique(all_rows).size == all_rows.size, \
    f"{run_dir}: test spots overlap across folds -- non-LOPO split?"

per_gene = pd.concat(gene_rows, ignore_index=True)
per_spot = pd.concat(spot_rows, ignore_index=True)
shares = pd.DataFrame(share_rows)

# A cohort with no tissue annotation at all (LUNG_CANCER) still gets the file, empty but typed, so
# every dataset loads the same way and reads 0 rows instead of hitting a missing path.
GENE_TISSUE_COLS = ["gene", "tissue", "slide", "test_donor", "label_col", "n_spots",
                    "mean_log_sigma", "mean_sigma", "mean_pred", "mean_target",
                    "mean_log_err", "mean_err", "pct_zero"]
per_gene_tissue = (pd.concat(gene_tissue_rows, ignore_index=True) if gene_tissue_rows
                   else pd.DataFrame({c: pd.Series(dtype="float64") for c in GENE_TISSUE_COLS}))

# ── Per-gene accuracy, taken from the run's OWN atoms.parquet ─────────────────────────────────
# Not recomputed here on purpose: atoms holds the pipeline's per-(fold, slide, gene) correlation
# (g* for the count heads), so reducing IT reproduces the headline convention exactly -- per
# (donor, slide, gene), then mean over that donor's slides. Re-deriving a Pearson in this script
# would silently answer a slightly different question.
atoms_path = run_dir / "atoms.parquet"
if atoms_path.exists():
    at = pd.read_parquet(atoms_path)
    at = at[at["split"] == "test"]
    corr_cols = [c for c in ("pearson", "spearman") if c in at.columns]
    acc = (at.groupby(["test_donor", "slide", "gene"], observed=True)[corr_cols].mean()
             .groupby(["test_donor", "gene"], observed=True).mean().reset_index())
    acc["test_donor"] = acc["test_donor"].astype(str)
    per_gene["test_donor"] = per_gene["test_donor"].astype(str)
    per_gene = per_gene.merge(acc, on=["test_donor", "gene"], how="left")
    print(f"[{_el():6.1f}s] joined accuracy from atoms.parquet: columns {corr_cols}, "
          f"{int(per_gene[corr_cols[0]].notna().sum()):,}/{len(per_gene):,} rows matched", flush=True)
else:
    corr_cols = []
    print(f"WARNING: no atoms.parquet at {atoms_path} -- per_gene carries no accuracy column.", flush=True)

# ── Join the per-spot metadata by adata_row (positional row into the merged adata) ─────────────
meta_by_row = meta.reset_index(drop=True)
per_spot = per_spot.join(meta_by_row, on="adata_row", rsuffix="_meta")

per_gene.to_parquet(out_dir / "per_gene.parquet", index=False)
per_gene_tissue.to_parquet(out_dir / "per_gene_tissue.parquet", index=False)
per_spot.to_parquet(out_dir / "per_spot.parquet", index=False)
shares.to_parquet(out_dir / "variance_shares.parquet", index=False)

# The (tissue, slide) group sizes decide what is readable downstream, so they are printed here
# rather than left for the reader to discover: a gene's mean in a tissue present on one slide as a
# handful of spots is noise, and this is where you see it.
if len(per_gene_tissue):
    grp = (per_gene_tissue.drop_duplicates(["label_col", "tissue", "slide"])
                          .pivot_table(index=["label_col", "tissue"], columns="slide",
                                       values="n_spots", aggfunc="first"))
    print(f"[{_el():6.1f}s] spots per (tissue, slide) -- the group sizes behind every "
          f"per-(gene, tissue) mean:", flush=True)
    print(grp.fillna(0).astype(int).to_string(), flush=True)

_s = shares.groupby("quantity")["share_gene"].mean()
summary = dict(
    dataset=dataset, model_type=model_type, family=family, run_dir=str(run_dir),
    input_adata=str(data_path), depth_scale=M.get("depth_scale"),
    count_convention=M.get("count_convention"), mu_scale=mu_scale, sigma_scale=sigma_scale,
    depth_blind=bool(depth_blind), n_folds=len(pred_files),
    donors=sorted(per_spot["test_donor"].unique().tolist()),
    n_spots=int(len(per_spot)), n_genes=int(per_gene["gene"].nunique()),
    label_cols=LABEL_COLS, extra_obs_cols=EXTRA_COLS,
    n_gene_tissue_rows=int(len(per_gene_tissue)),
    tissues_found={c: sorted(per_gene_tissue.loc[per_gene_tissue["label_col"] == c,
                                                 "tissue"].unique().tolist())
                   for c in LABEL_COLS} if len(per_gene_tissue) else {},
    homogeneity_k=args.n_neighbors,
    homogeneity_cols=[c for c in per_spot.columns if c.startswith("homog_")],
    accuracy_cols=corr_cols,
    n_gene_columns_with_zero_sd=int(n_genes_zero_sd_total),
    eps_err_per_fold=eps_err_used,
    eps_err_native_per_fold=eps_err_native_used,
    per_gene_err_scale=mu_scale,
    per_gene_tissue_scale=("log1p-CP10k" if family == "continuous" else mu_scale),
    mean_share_gene_log_sigma=float(_s.get("log_sigma", np.nan)),
    mean_share_gene_log_abs_err=float(_s.get("log_abs_err", np.nan)),
    no_delta_method=True,
)
with open(out_dir / "summary.json", "w") as fh:
    json.dump(summary, fh, indent=2)

print(f"[{_el():6.1f}s] wrote {out_dir}", flush=True)
print(f"  per_gene {per_gene.shape}  per_gene_tissue {per_gene_tissue.shape}  "
      f"per_spot {per_spot.shape}  shares {shares.shape}", flush=True)
print(f"  mean share_gene: log_sigma {summary['mean_share_gene_log_sigma']:.3f} vs "
      f"log_abs_err {summary['mean_share_gene_log_abs_err']:.3f}  "
      f"(the GAP is the readable quantity, not either level)", flush=True)
