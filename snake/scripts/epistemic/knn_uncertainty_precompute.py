# snake/scripts/epistemic/knn_uncertainty_precompute.py
#
# Compute the kNN local-uncertainty signals (src/evaluation/knn_uncertainty.py) for one held-out
# LOPO donor of a baseline classic run. All the heavy work lives here: the reference index build,
# the neighbor queries, the gene-chunked (spot x gene x k) signal grid and the OOD embedding
# queries. Everything it writes is an artifact to be loaded and plotted elsewhere. CPU only:
# sklearn NearestNeighbors plus numpy/pandas/torch, no GPU. One SLURM job per (donor, stage); see
# launch_knn_uncertainty_precompute.sh.
#
# Reads the baseline run dir
# <project-root>/results/<dataset>/<run-id>/<gene-set>/<radius>/<model-type>[/<lambda-dir>]:
#   metrics.json                     run identity, gene panel, target_source, input_adata path
#   splits.json                      the LOPO fold whose test_donor is --donor
#   predictions/preds_fold<k>.h5ad   the query spot order and the model's mu / pred_var
# plus the preprocessed adata metrics.json names, and
# <project-root>/results/_analysis/gene_slide_nplus_regime.parquet, the per-(gene, slide)
# sample-size gate the canal 2 blocks apply. Canal 3 does not use it; it has its own MIN_SIDE gate.
#
# A DONOR is excluded from the reference index, not a slide. The CV split is donor-grouped and a
# donor can have several slides, so excluding one slide would leave a sister slide of the same
# donor in the index and its spots would be artificially close neighbors. The reference pool is
# splits.json's own inner_splits[0] train+val rows, read from the baseline run rather than
# recomputed, so sigma_knn and sigma_MLP are compared on exactly the same test spots.
#
# Vocabulary: sigma_knn is an empirical local dispersion and d_mean/d_k a density/coverage score.
# They are not "aleatoric" and "epistemic". The analogy to SNGP/DDU/Deep k-NN is loose and this is
# not a Bayesian decomposition.
#
# Scope.
#   - The baseline model_type must be a continuous (Gaussian) spec: the target_source == "X" assert
#     rejects a count baseline, whose X is a count-scale mean, not log1p(CP10k), so it does not
#     share a scale with the kNN signals. A saved pred_var layer is what supplies the sigma_MLP
#     comparator; without one those rows are simply not produced.
#   - Raw L2 distance only, no PCA-L2 or whitening variant. See build_reference_index.
#   - No z-scoring of neighbor values by (gene, slide): pooled raw log1p(CP10k). The
#     inter_intra_slide_variance diagnostic below is the check that says whether it is needed.
#
# Usage (repo root, on the cluster):
#   python -u snake/scripts/epistemic/knn_uncertainty_precompute.py --dataset 10x_TuPro --donor MACEGEJ --stage signals
#   python -u snake/scripts/epistemic/knn_uncertainty_precompute.py --dataset 10x_TuPro --donor MACEGEJ --stage ood_corruption
#   python -u snake/scripts/epistemic/knn_uncertainty_precompute.py --dataset 10x_TuPro --donor MACEGEJ --stage ood_crossdataset
#
# Writes under
# <out-root>/<dataset>/<run_id>__<gene_set>__<radius>__<model_type>[__<lambda_dir>]/donor=<donor>/
#   stage=signals, directly in that dir:
#     knn_signals.h5ad   the per-(spot, gene) signals at k=<k-headline>: X=mu_knn_all, layers
#                        sigma_knn/mu_knn_pos/pi_hat, obs adata_row/slide/d_mean/d_k/d_norm*/
#                        cluster_label*. The only non-aggregated artifact, and the one
#                        canal2_uncertainty_precompute.py's --knn-dir mode reads. See the block
#                        that writes it for the scale caveat: sigma_knn is a standard deviation on
#                        the log1p-CP10k axis, usable only as a monotone score.
#     spot_distance.parquet, canal1_atoms.parquet, canal2_gene_spearman.parquet,
#     canal2_sparsification.parquet, canal2_pos_neighbor_nan_rate.parquet,
#     canal2_gene_spearman_canonical.parquet, canal2_sparsification_canonical.parquet,
#     canal2_partial_spearman_canonical.parquet, canal3_pair_auroc_knn.parquet,
#     canal3_pair_auroc_mlp.parquet, canal3_pair_auroc_nb_bonus.parquet (only when the NB sibling
#     run exists), inter_intra_slide_variance.parquet, summary.json
#   stage=ood_corruption, under donor=<donor>/ood_corruption/:
#     blur_distance_by_k.parquet, blur_mechanistic.parquet, saltpepper_distance_by_k.parquet,
#     saltpepper_mechanistic.parquet, summary.json
#   stage=ood_crossdataset, under donor=<donor>/ood_crossdataset/:
#     crossdataset_distance.parquet, summary.json
# leiden_labels.npz also lands in donor=<donor>/ and is shared by the three stages; see the leiden
# block below.
#
# d_norm comes as one column per cluster scheme (d_norm_gt / d_norm_gt_spatial15 /
# d_norm_gt_emb10 / d_norm_leiden, each with a cluster_label_<scheme> and a cluster_size_<scheme>),
# because which clusters a cohort can define differs: a fully labeled cohort supports gt, a cohort
# with many UNASSIGNED spots only the imputed annotations, a cohort with no ground_truth column only
# the unsupervised leiden scheme. Plain d_norm / cluster_label / cluster_size alias that cohort's
# primary scheme. summary.json["cluster_schemes"] records the source, sizes and radii of each, and
# ["primary_cluster_scheme"] which one the aliases point at. The corruption and cross-dataset
# parquets carry the same d_norm columns; in the corruption stage the cluster assignment is fixed
# from the clean embedding across all severities.

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import torch
from sklearn.metrics import roc_auc_score

_T0 = time.time()
def _el():
    return time.time() - _T0

# ── Args ──────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--dataset", default="10x_TuPro", choices=["10x_TuPro", "TLS", "LUNG_CANCER"])
p.add_argument("--donor", required=True, help="Held-out LOPO donor id (must appear exactly once "
               "as a fold's test_donor in the baseline run's splits.json).")
p.add_argument("--stage", default="signals", choices=["signals", "ood_corruption", "ood_crossdataset"])
p.add_argument("--run-id", default="pr_classic3", help="run_id_classic SEGMENT inside the results tree "
               "(results/<dataset>/<run-id>/...). Also the first token of the output run_tag.")
p.add_argument("--gene-set", default=None, help="Default: _<dataset>_hvg_top3000.")
p.add_argument("--radius", default="r75")
p.add_argument("--model-type", default="heteroscedastic", help="Baseline model_type (continuous/"
               "Gaussian only -- see the SCOPE note at the top of this file).")
p.add_argument("--lambda-dir", default="lambda_0.0", help="Subdir under model_type ('none' to omit).")
p.add_argument("--data", default=None, help="Override the preprocessed adata path "
               "(default: metrics.json['input_adata']).")
p.add_argument("--project-root", default=None)
p.add_argument("--nplus-regime-parquet", default=None,
               help="Default: <project_root>/results/_analysis/gene_slide_nplus_regime.parquet")
p.add_argument("--out-root", default=None,
               help="Default: <project_root>/results/_analysis/knn_uncertainty")
p.add_argument("--k-grid", default="5,10,25,50,100")
p.add_argument("--k-headline", type=int, default=25)
p.add_argument("--tau", type=float, default=1.0)
p.add_argument("--kernel", default="exp", choices=["exp", "uniform"])
p.add_argument("--min-pos-neighbors", type=int, default=5)
p.add_argument("--gene-chunk-size", type=int, default=200,
               help="Gene columns per knn_signal_grid call -- bounds the (n_query, k_max, chunk) "
               "intermediate; see the module docstring's gene-chunking note.")
p.add_argument("--seed", type=int, default=0)
# ── Unsupervised cluster scheme for d_norm (see the CLUSTER_SCHEMES block) ────────────────────
p.add_argument("--no-leiden-clustering", dest="leiden_clustering", action="store_false",
               help="Skip the UNSUPERVISED d_norm scheme. On a cohort with no usable ground_truth "
                    "column (LUNG_CANCER) this leaves no scheme at all and the run fails on purpose.")
p.set_defaults(leiden_clustering=True)
p.add_argument("--leiden-resolution", type=float, default=1.0,
               help="Leiden granularity. Higher -> more, smaller communities. Leiden takes no k; "
                    "this is the parameter that replaces it. 1.0 = scanpy's default.")
p.add_argument("--leiden-neighbors", type=int, default=15,
               help="n_neighbors of the kNN graph Leiden partitions (scanpy default). Unrelated to "
                    "--k-grid, which is the neighborhood the SIGNALS are read from.")
p.add_argument("--nb-model-type", default="negativebinomial_hetero_noanchor_cp",
               help="Sibling run (same dataset/run-id/gene-set/radius) to try for the OPTIONAL "
               "canal-3 NB P(Y=0) bonus comparator. Skipped gracefully if absent. Must be a _cp "
               "or _raw variant, the only count model types the registry carries.")
p.add_argument("--n-corruption-spots", type=int, default=400,
               help="stage=ood_corruption: subsample of this donor's own test spots to query.")
p.add_argument("--emb-dataset", default=None,
               help="stage=ood_corruption ONLY: the patch_embeddings/<name>/ directory the CORRUPTED "
               "extractions live under. Defaults to --dataset, which is right for 10x_TuPro and TLS "
               "but NOT for the merged LUNG_CANCER: the blur/salt-and-pepper extractions were "
               "launched per sub-cohort (LUNG_CANCER_LUAD / LUNG_CANCER_LUSC) and no merged "
               "patch_embeddings/LUNG_CANCER dir exists. Never affects the signals stage.")
p.add_argument("--blur-factors", default="0.0,0.005,0.01,0.02,0.05")
p.add_argument("--sp-amounts", default="0.0,0.01,0.05,0.1,0.2")
p.add_argument("--radius-factor", type=float, default=0.75,
               help="Patch-embedding extraction radius factor (naming template only).")
p.add_argument("--swap-xy", default="True", help="Patch-embedding extraction swapxy flag "
               "(naming template only; matches the extraction dirs' literal 'True'/'False').")
p.add_argument("--cross-dataset", default="TLS")
p.add_argument("--cross-preproc-id", default="v5")
p.add_argument("--cross-radius", default="r75")
p.add_argument("--n-cross-spots", type=int, default=3000)
p.add_argument("--overwrite", action="store_true")
args = p.parse_args()

PROJECT_ROOT = args.project_root or os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(PROJECT_ROOT)

from src.data.load_and_preprocess import read_arrays
from src.evaluation.knn_uncertainty import (
    K_MAX, build_reference_index, query_neighbors, knn_signal_grid, spot_distance_signals,
    inter_intra_slide_variance, l2_normalize, cluster_centroids_and_radii, assign_nearest_centroid,
    leiden_cluster_labels,
)
from src.evaluation.metrics_slide_aware import per_slide_gene_corr, CORR_METRICS
from src.evaluation.benchmark_summary import _rc_curve as risk_coverage_curve, COVERAGE_GRID

EMB_OBSM_KEY = "X_emb"   # verified against src/data/load_and_preprocess.py (read_arrays)
K_GRID = tuple(sorted(int(k) for k in args.k_grid.split(",")))
K_HEADLINE = args.k_headline
assert K_HEADLINE in K_GRID, f"--k-headline {K_HEADLINE} must be one of --k-grid {K_GRID}"
assert max(K_GRID) <= K_MAX, f"--k-grid max ({max(K_GRID)}) exceeds K_MAX ({K_MAX})"
MIN_SIDE = 20   # canal-3 AUROC gate (same literal as canal3_zero_detection_precompute.py) --
                # DIFFERENT from the global gene_slide_nplus_regime.parquet regime thresholds (canal 2).
REGIME_TO_NBINS = {"terciles": 3, "quintiles": 5, "deciles": 10}
FRACS = COVERAGE_GRID   # reuse benchmark_summary's 0.01..1.0 grid for the sparsification curves

gene_set = args.gene_set or f"_{args.dataset}_hvg_top3000"
lambda_dir = None if args.lambda_dir.lower() in ("none", "null", "") else args.lambda_dir
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results")
run_dir = os.path.join(RESULTS_ROOT, args.dataset, args.run_id, gene_set, args.radius, args.model_type)
if lambda_dir:
    run_dir = os.path.join(run_dir, lambda_dir)
assert os.path.isdir(run_dir), f"not a directory: {run_dir}"

NPLUS_REGIME_PARQUET = args.nplus_regime_parquet or os.path.join(
    PROJECT_ROOT, "results", "_analysis", "gene_slide_nplus_regime.parquet")
assert os.path.isfile(NPLUS_REGIME_PARQUET), f"missing: {NPLUS_REGIME_PARQUET}"

OUT_ROOT = args.out_root or os.path.join(PROJECT_ROOT, "results", "_analysis", "knn_uncertainty")
run_tag = f"{args.run_id}__{gene_set}__{args.radius}__{args.model_type}" + (f"__{lambda_dir}" if lambda_dir else "")
donor_out = os.path.join(OUT_ROOT, args.dataset, run_tag, f"donor={args.donor}")
base_out = donor_out if args.stage == "signals" else os.path.join(donor_out, args.stage)
os.makedirs(base_out, exist_ok=True)

# Cache sentinel. A stage counts as done only when both its summary.json and its stage-specific
# artifact are on disk. summary.json alone is not enough for stage=signals: it is written last, so a
# donor dir can hold it and still be missing knn_signals.h5ad, the one artifact canal 2's --knn-dir
# mode reads. Skipping such a dir exits 0 and produces nothing, and the failure surfaces much later
# as canal 2's "no donor=*/knn_signals.h5ad" skip.
_done = os.path.isfile(os.path.join(base_out, "summary.json")) and (
    args.stage != "signals" or os.path.isfile(os.path.join(base_out, "knn_signals.h5ad")))
if not args.overwrite and _done:
    print(f"skip (already done, pass --overwrite to force): {base_out}", flush=True)
    sys.exit(0)

print(f"run_dir  : {run_dir}", flush=True)
print(f"dataset  : {args.dataset}  donor: {args.donor}  stage: {args.stage}", flush=True)
print(f"out_dir  : {base_out}", flush=True)
print(f"k_grid   : {K_GRID}  k_headline: {K_HEADLINE}  tau: {args.tau}  kernel: {args.kernel}", flush=True)

# ── Baseline run identity (metrics.json is authoritative, never re-derived) ──────────────────
with open(os.path.join(run_dir, "metrics.json")) as f:
    M = json.load(f)
model_type = M["model_type"]
# gene_names is absent from the stub metrics.json a compute_final_metrics=false run
# writes. Fall back to the sibling genes.txt that train.py writes at startup.
gene_names = M.get("gene_names")
if gene_names is None:
    with open(os.path.join(run_dir, "genes.txt")) as gf:
        gene_names = [ln.strip() for ln in gf if ln.strip()]
target_source = M["target_source"]
target_transform = M.get("target_transform", "none")
primary_metric = M.get("primary_corr_metric", "pearson")
assert target_source == "X", (
    f"{run_dir}: target_source={target_source!r} -- this first pass only supports a continuous "
    f"(Gaussian, target_source='X') baseline (see the SCOPE note at the top of this file)."
)
baseline_headline = None
if M.get("lopo_stats") and primary_metric in M["lopo_stats"]:
    baseline_headline = M["lopo_stats"][primary_metric]["mean"]

# ── splits.json — locate THIS donor's fold, build the reference-pool / query-pool row indices ──
with open(os.path.join(run_dir, "splits.json")) as f:
    SPLITS = json.load(f)
matches = [fo for fo in SPLITS["folds"] if str(args.donor) in [str(d) for d in fo["test_donor"]]]
assert len(matches) == 1, (
    f"donor {args.donor!r} found {len(matches)} times as a test_donor in {run_dir}/splits.json "
    f"(expected exactly 1 under LOPO); donors_ordered={SPLITS.get('donors_ordered')}"
)
fold_entry = matches[0]
fold_num = fold_entry["fold"]
inner = fold_entry["inner_splits"][0]
train_idx_only = np.asarray(inner["train_idx"], dtype=np.int64)
ref_idx = np.concatenate([train_idx_only, np.asarray(inner["val_idx"], dtype=np.int64)])
test_idx_declared = np.asarray(fold_entry["test_idx"], dtype=np.int64)
print(f"[{_el():6.1f}s] fold {fold_num}: ref pool {ref_idx.size} spots (train+val, donor "
      f"{args.donor!r} entirely excluded), declared test {test_idx_declared.size} spots", flush=True)

# ── Baseline predictions for this fold (authoritative source of the query row order + MLP mu/var) ──
pred_path = os.path.join(run_dir, "predictions", f"preds_fold{fold_num}.h5ad")
assert os.path.isfile(pred_path), f"missing predictions file: {pred_path}"
_p = sc.read_h5ad(pred_path, backed="r")
pred_test = _p[_p.obs["split"].to_numpy() == "test"].to_memory()
del _p
pred_donor = str(pd.unique(pred_test.obs["test_donor"].astype(str))[0])
assert pred_donor == str(args.donor), (
    f"{pred_path}: obs['test_donor']={pred_donor!r} != requested --donor {args.donor!r}"
)
assert list(pred_test.var_names) == list(gene_names), (
    f"{pred_path}: var_names order != metrics.json['gene_names'] -- unexpected for this run."
)
query_idx = pred_test.obs["adata_row"].to_numpy().astype(np.int64)
assert set(query_idx.tolist()) == set(test_idx_declared.tolist()), (
    "predictions test adata_row set != splits.json test_idx set -- inconsistent run artifacts."
)
query_slides = pred_test.obs["slide"].to_numpy().astype(str)
query_obs_names = pred_test.obs_names.to_numpy().astype(str)
n_query = query_idx.size

if target_transform != "none":
    assert "means_logcp10k" in pred_test.layers and "y_logcp10k" in pred_test.layers, (
        f"{pred_path}: target_transform={target_transform!r} but no *_logcp10k layers found."
    )
    mu_mlp = np.asarray(pred_test.layers["means_logcp10k"], dtype=np.float64)
    query_y = np.asarray(pred_test.layers["y_logcp10k"], dtype=np.float64)
else:
    mu_mlp = np.asarray(pred_test.X, dtype=np.float64)
    query_y = np.asarray(pred_test.layers["y"], dtype=np.float64)
sigma_mlp_available = ("pred_var" in pred_test.layers) and (target_transform == "none")
var_mlp = np.asarray(pred_test.layers["pred_var"], dtype=np.float64) if sigma_mlp_available else None
print(f"[{_el():6.1f}s] loaded predictions: {n_query} test spots x {len(gene_names)} genes "
      f"(sigma_mlp_available={sigma_mlp_available})", flush=True)

# ── Main dataset adata — embeddings (frozen H&E patch embedding) + log1p(CP10k) target matrix ──
data_path = args.data or M["input_adata"]
adata = ad.read_h5ad(data_path)
assert EMB_OBSM_KEY in adata.obsm, f"{data_path}: missing obsm['{EMB_OBSM_KEY}'] -- schema changed?"
obs_names_all = np.asarray(adata.obs_names).astype(str)
# Every annotation column that could define a supervised cluster scheme, captured before `del adata`
# and WITHOUT asserting any of them exists: LUNG_CANCER has no ground_truth at all and still gets an
# unsupervised d_norm, so a missing column is a scheme that is skipped, not a failure. Which ones
# were actually usable is decided (and printed) in the cluster-scheme block below.
obs_cluster_cols = {c: adata.obs[c].to_numpy().astype(str)
                    for c in ("ground_truth", "ground_truth_knn_spatial_15", "ground_truth_knn_emb_10")
                    if c in adata.obs.columns}
embeddings, y_all, sample_labels, available_genes = read_arrays(adata, target_source="X")
del adata
gene_idx = pd.Index(available_genes).get_indexer(gene_names)
assert (gene_idx >= 0).all(), f"{data_path}: {int((gene_idx < 0).sum())} gene(s) from metrics.json missing"
y_all = y_all[:, gene_idx]
assert np.array_equal(obs_names_all[query_idx], query_obs_names), (
    "adata.obs_names[query_idx] != predictions obs_name -- adata_row misaligned with this run."
)
print(f"[{_el():6.1f}s] adata loaded: {embeddings.shape[0]} spots x {y_all.shape[1]} genes "
      f"(emb_dim={embeddings.shape[1]})", flush=True)

ref_emb = embeddings[ref_idx]
ref_y = y_all[ref_idx]
query_emb = embeddings[query_idx]
n_genes = len(gene_names)

# ── d_mean type-rarity de-confound: one d_norm per cluster scheme ───────────────────────────────
# Raw d_mean conflates "far from its neighbors because this tissue type is rare" with "far because
# the embedding is genuinely unusual". Dividing by the assigned cluster's radius removes the first;
# see cluster_centroids_and_radii. Which clusters to use is a choice with no single right answer, so
# every applicable scheme is computed and all of them are reported side by side:
#
#   gt          the histological annotation itself. The interpretable one, and the only one whose
#               cluster_label means the same thing across folds. Needs a fully labeled reference
#               pool: 10x_TuPro qualifies, TLS has many UNASSIGNED spots (a deliberate expert
#               abstention) and LUNG_CANCER has no ground_truth column at all.
#   gt_*        the same idea on TLS, from an imputed annotation rather than by dropping the
#               UNASSIGNED spots. Dropping them is not an option when they are a large share of the
#               pool: the radii would then be fitted on a biased subset. Two variants, confounded
#               differently: spatial_15 imputes from (x,y) only, independent of both the model input
#               and the target; emb_10 imputes from the same embedding space this distance is
#               measured in, so its clusters are tighter by construction and its d_norm
#               correspondingly larger. Report which one you used.
#   leiden      unsupervised, on the embeddings. The only scheme LUNG_CANCER can use, and on the
#               labeled cohorts a control: if d_norm behaves the same supervised and unsupervised,
#               the de-confounding is a property of the local density rather than of one particular
#               annotation. Its cluster ids are per-fold and not comparable across folds; see
#               leiden_cluster_labels.
#
# A scheme whose column is absent is skipped with a printed reason, never an assert: the cohorts
# legitimately support different subsets.
CLUSTER_SCHEMES = {}          # scheme -> dict(centroids, radii, labels_order, counts, source)
_scheme_sources = [("gt", "ground_truth"),
                   ("gt_spatial15", "ground_truth_knn_spatial_15"),
                   ("gt_emb10", "ground_truth_knn_emb_10")]
# "No label" is not spelled one single way: obs_cluster_cols went through .astype(str), so a pandas
# NaN arrives as the literal "nan", an empty cell as "", and a hand-edited column could carry
# "Unassigned". Any of those slipping through the gate becomes a real cluster with its own centroid
# and radius, and that radius silently divides a reported quantity. Hence an upper-cased sentinel
# set rather than one literal comparison.
_NO_LABEL = ("UNASSIGNED", "NAN", "NONE", "NA", "N/A", "", "<NA>")
for _scheme, _col in _scheme_sources:
    if _col not in obs_cluster_cols:
        print(f"[{_el():6.1f}s] cluster scheme {_scheme!r}: skipped (no obs column {_col!r})", flush=True)
        continue
    _lab = obs_cluster_cols[_col][ref_idx]
    _bad = np.isin(np.char.upper(_lab.astype(str)), _NO_LABEL)   # astype: np.char needs a str dtype
    if _bad.any():
        _offending = ", ".join(f"{str(v)!r}x{int(c)}"
                               for v, c in zip(*np.unique(_lab[_bad], return_counts=True)))
        print(f"[{_el():6.1f}s] cluster scheme {_scheme!r}: skipped ({int(_bad.sum()):,}/{len(_lab):,} "
              f"reference spots carry no usable label in {_col!r} [{_offending}] -- an imputed column "
              f"is needed for this cohort, see the block comment)", flush=True)
        continue
    _c, _r, _o, _n = cluster_centroids_and_radii(ref_emb, _lab)
    CLUSTER_SCHEMES[_scheme] = dict(centroids=_c, radii=_r, labels_order=_o, counts=_n, source=_col)
    print(f"[{_el():6.1f}s] cluster scheme {_scheme!r} ({_col}): {len(_o)} clusters (radius, n): " +
          ", ".join(f"{l}={r:.4f}(n={int(n)})" for l, r, n in zip(_o, _r, _n)), flush=True)

# ── leiden: computed once per donor, cached on disk, shared by the three stages ──────────────────
# This block sits above the stage branch, so without the cache every stage re-runs sc.pp.neighbors +
# Leiden on the same reference pool. That is not merely wasteful, it is the most expensive step of
# the two OOD stages, which otherwise need no graph at all. It is also a correctness problem:
# pynndescent is approximate and its parallel graph build is not bit-reproducible across thread
# counts, so two stages could end up with different partitions, hence different radii, hence a
# d_norm_leiden in blur_distance_by_k.parquet on a different divisor from the one in
# knn_signals.h5ad. The cache pins the divisor: the first stage to run fixes the partition and the
# others reuse it, recomputing only if the cached reference pool or parameters differ.
# A leiden failure is a skip with a printed reason, like every other scheme, not an abort: it needs
# a clustering backend nothing else in this script does, and a cohort whose gt scheme is usable must
# not lose its whole sweep to it. A cohort where leiden is the only applicable scheme still fails
# loudly on the `assert CLUSTER_SCHEMES` below.
LEIDEN_CACHE = os.path.join(donor_out, "leiden_labels.npz")
if args.leiden_clustering:
    try:
        _lab, _cached = None, False
        if os.path.isfile(LEIDEN_CACHE):
            _z = np.load(LEIDEN_CACHE, allow_pickle=False)
            if (np.array_equal(_z["ref_idx"], ref_idx) and float(_z["resolution"]) == args.leiden_resolution
                    and int(_z["n_neighbors"]) == args.leiden_neighbors and int(_z["seed"]) == args.seed):
                _lab, _cached = _z["labels"].astype(str), True
            else:
                print(f"[{_el():6.1f}s] leiden cache {LEIDEN_CACHE} ignored (different reference pool "
                      f"or parameters) -- recomputing", flush=True)
        if _lab is None:
            _lab, _ = leiden_cluster_labels(ref_emb, resolution=args.leiden_resolution,
                                            n_neighbors=args.leiden_neighbors, seed=args.seed)
            _tmp = LEIDEN_CACHE + f".tmp{os.getpid()}"    # atomic: three stages can race here
            # Written through an open FILE OBJECT, not a path: np.savez APPENDS ".npz" to a path
            # that does not already end in it, so `np.savez(_tmp, ...)` would create
            # "<...>.npz.tmp<pid>.npz" and the os.replace below would then fail on a missing
            # "<...>.npz.tmp<pid>". Given a file object numpy writes exactly where it is told.
            with open(_tmp, "wb") as _fh:
                np.savez(_fh, ref_idx=ref_idx, labels=_lab, resolution=float(args.leiden_resolution),
                         n_neighbors=int(args.leiden_neighbors), seed=int(args.seed))
            os.replace(_tmp, LEIDEN_CACHE)
        _c, _r, _o, _n = cluster_centroids_and_radii(ref_emb, _lab)
        CLUSTER_SCHEMES["leiden"] = dict(centroids=_c, radii=_r, labels_order=_o, counts=_n,
                                         source=f"leiden(resolution={args.leiden_resolution})")
        print(f"[{_el():6.1f}s] cluster scheme 'leiden' (resolution={args.leiden_resolution}, "
              f"{'from cache' if _cached else 'computed'}): {len(_o)} clusters, radii "
              f"{_r.min():.4f}-{_r.max():.4f}, sizes {int(_n.min())}-{int(_n.max())}", flush=True)
    except Exception as e:
        print(f"[{_el():6.1f}s] cluster scheme 'leiden': skipped ({type(e).__name__}: {e})", flush=True)

assert CLUSTER_SCHEMES, (
    f"{data_path}: no cluster scheme is applicable, so no d_norm can be built at all. "
    f"Every scheme printed its own reason on stdout just above -- read those lines rather than "
    f"this one: a missing obs column is expected on some cohorts, whereas a 'leiden: skipped "
    f"(<Error>)' line is a real failure of the unsupervised fallback."
)

def assign_all_schemes(q_emb):
    """{scheme: nearest-centroid row index} for one query set. Separate from d_norm_columns because
    the assignment does not depend on k while d_norm does: call this once per query set, then
    d_norm_columns once per k."""
    return {name: assign_nearest_centroid(q_emb, C["centroids"]) for name, C in CLUSTER_SCHEMES.items()}


def d_norm_columns(d_mean_arr, assign):
    """One {d_norm_<scheme>: array} + {cluster_label_<scheme>: array} + {cluster_size_<scheme>: array}
    mapping per applicable scheme, from a precomputed assignment (assign_all_schemes).

    The three stages each need this on a different query set (the fold's test spots, the corruption
    subsample, the cross-dataset spots) while the centroids and radii, fitted on the reference pool
    alone, are shared. Assignment is nearest-centroid, so a query spot needs no label of its own;
    only the reference pool does."""
    out = {}
    for name, idx in assign.items():
        C = CLUSTER_SCHEMES[name]
        out[f"d_norm_{name}"] = d_mean_arr / C["radii"][idx]
        out[f"cluster_label_{name}"] = C["labels_order"][idx]
        out[f"cluster_size_{name}"] = C["counts"][idx].astype(np.int64)   # counts is float (np.add.at)
    return out

# The scheme whose `d_norm` / `cluster_label` / `cluster_size` are also emitted under the unsuffixed
# names, so a consumer can ask for d_norm without knowing the cohort. Preference order is most
# interpretable first; on a cohort with no annotation it lands on leiden, the only one available.
PRIMARY_SCHEME = next(s for s in ("gt", "gt_spatial15", "gt_emb10", "leiden") if s in CLUSTER_SCHEMES)
# Per-scheme provenance for summary.json: what defined the clusters, how many there are, and their
# radii -- the divisor behind every d_norm_<scheme>, so it must be readable offline without re-running.
CLUSTER_SCHEME_PROVENANCE = {
    name: dict(source=C["source"], n_clusters=int(len(C["labels_order"])),
               labels=[str(x) for x in C["labels_order"]],
               sizes=[int(x) for x in C["counts"]], radii=[float(x) for x in C["radii"]])
    for name, C in CLUSTER_SCHEMES.items()
}
print(f"[{_el():6.1f}s] cluster schemes available: {sorted(CLUSTER_SCHEMES)} | "
      f"primary (unsuffixed d_norm) = {PRIMARY_SCHEME!r}", flush=True)

# ── n+ regime lookup: the canal-2 sample-size gate, never an ad hoc threshold ──────────────────
regime_df = pd.read_parquet(NPLUS_REGIME_PARQUET)
regime_df = regime_df[regime_df["dataset"] == args.dataset]
regime_df = regime_df[regime_df["regime"].isin(REGIME_TO_NBINS)].copy()
regime_df["n_bins"] = regime_df["regime"].map(REGIME_TO_NBINS).astype(int)
regime_lookup = regime_df[["gene", "slide", "n_plus", "n_bins"]]
# Named here rather than surfacing much later as a bare "np.interp: array of sample points is empty"
# from risk_coverage_curve: every canal-2 block below inner-joins on (gene, slide), so an empty or
# non-overlapping lookup silently empties the frames instead of failing where the cause is.
assert not regime_lookup.empty, (
    f"{NPLUS_REGIME_PARQUET}: no terciles/quintiles/deciles rows for dataset={args.dataset!r} -- "
    f"regenerate it before running the kNN precompute on this cohort.")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE: signals — the 3 kNN signals over the full k-grid, canal 1/2/3, diagnostics
# ══════════════════════════════════════════════════════════════════════════════
if args.stage == "signals":
    index = build_reference_index(ref_emb)
    dist, idx = query_neighbors(index, query_emb, k_max=K_MAX)
    print(f"[{_el():6.1f}s] kNN index built ({ref_emb.shape[0]} ref spots) + queried "
          f"({n_query} query spots, k_max={K_MAX})", flush=True)

    d_by_k = spot_distance_signals(dist, k_grid=K_GRID)
    # The nearest-centroid assignment does not depend on k, so it is computed ONCE for this query set
    # and reused for every k; the unsuffixed aliases are then read out of the dict d_norm_columns
    # returns rather than recomputed.
    query_assign = assign_all_schemes(query_emb)
    dn_by_k = {k: d_norm_columns(d_by_k[k]["d_mean"], query_assign) for k in K_GRID}
    spot_distance_df = pd.concat([
        pd.DataFrame({"obs_name": query_obs_names, "adata_row": query_idx, "slide": query_slides,
                      "k": k, "d_mean": d_by_k[k]["d_mean"], "d_k": d_by_k[k]["d_k"],
                      # one d_norm_<scheme> triple per applicable scheme, plus the unsuffixed trio
                      # aliasing PRIMARY_SCHEME
                      **dn_by_k[k],
                      "d_norm": dn_by_k[k][f"d_norm_{PRIMARY_SCHEME}"],
                      "cluster_label": dn_by_k[k][f"cluster_label_{PRIMARY_SCHEME}"],
                      "cluster_size": dn_by_k[k][f"cluster_size_{PRIMARY_SCHEME}"]})
        for k in K_GRID
    ], ignore_index=True)

    # ── gene-chunked signal grid: accumulate the full (n_query, n_genes) arrays PER k ───────────
    SIG = {k: {name: np.empty((n_query, n_genes), dtype=np.float32)
              for name in ("pi_hat", "mu_knn_all", "mu_knn_pos", "sigma_knn_pos")}
          for k in K_GRID}
    NPOS = {k: np.empty((n_query, n_genes), dtype=np.int32) for k in K_GRID}
    for c0 in range(0, n_genes, args.gene_chunk_size):
        c1 = min(c0 + args.gene_chunk_size, n_genes)
        chunk = knn_signal_grid(idx, dist, ref_y[:, c0:c1], k_grid=K_GRID,
                                tau=args.tau, kernel=args.kernel,
                                min_pos_neighbors=args.min_pos_neighbors)
        for k in K_GRID:
            for name in ("pi_hat", "mu_knn_all", "mu_knn_pos", "sigma_knn_pos"):
                SIG[k][name][:, c0:c1] = chunk[k][name]
            NPOS[k][:, c0:c1] = chunk[k]["n_pos_neighbors"]
        print(f"[{_el():6.1f}s] signal grid genes {c0}:{c1}/{n_genes}", flush=True)

    # ── Per-(spot, gene) signal file: the one input canal 2's --knn-dir mode reads ───────────────
    # Every other artifact this stage writes is aggregated. canal2_uncertainty_precompute.py needs
    # the signals per (spot, gene) so it can score the kNN uncertainty against a trained model's
    # error with canal 2's own error definition, grouping axes, floor policy and donor aggregation,
    # rather than this script re-implementing those four things and producing numbers that do not
    # compare with the aleatoric column they sit beside.
    #
    # Written here, immediately after the signal grid, not at the end of the stage. Everything it
    # reads is already bound at this point (SIG from the loop just above, d_by_k / dn_by_k and the
    # query_* arrays from before it), while the canal 1/2/3 blocks that follow are the stage's RAM
    # peak: they melt the positives into a very long frame. An OOM kill or an exception down there
    # would otherwise destroy the one artifact this file exists to produce, and the donor would have
    # to be recomputed from the index build up.
    #
    # Only K_HEADLINE is written. The full grid would be len(K_GRID) times this file, for a
    # sensitivity-to-k question the aggregated parquets above already answer.
    #
    # Scale: y_all comes from read_arrays(target_source="X"), so mu_knn/sigma_knn are on the
    # log1p-CP10k axis, and sigma_knn is a standard deviation, not a variance like the layers the
    # other post-hoc writers emit. That asymmetry is harmless because every canal-2 consumer of
    # `uncertainty` is rank-based (partial Spearman, binned Spearman, sparsification sort) and
    # sigma -> sigma^2 is monotone on sigma > 0, but it does mean these values must never be
    # compared in magnitude with a model's pred_var. d_mean/d_norm are distances, not a dispersion
    # of any kind, which is the same argument one step further.
    knn_signals = ad.AnnData(
        X=SIG[K_HEADLINE]["mu_knn_all"],                    # point prediction, all-neighbors variant
        obs=pd.DataFrame({
            "obs_name": query_obs_names, "adata_row": query_idx, "slide": query_slides,
            "d_mean": d_by_k[K_HEADLINE]["d_mean"], "d_k": d_by_k[K_HEADLINE]["d_k"],
            # every applicable d_norm_<scheme> (canal 2 selects one via --knn-signal) + the
            # unsuffixed alias for PRIMARY_SCHEME. cluster_size_* is dropped here: it is a per-scheme
            # constant per cluster, already in spot_distance.parquet, and of no use to canal 2.
            **{c: v for c, v in dn_by_k[K_HEADLINE].items() if not c.startswith("cluster_size_")},
            "d_norm": dn_by_k[K_HEADLINE][f"d_norm_{PRIMARY_SCHEME}"],
            "cluster_label": dn_by_k[K_HEADLINE][f"cluster_label_{PRIMARY_SCHEME}"],
        }, index=pd.Index(query_obs_names, name=None)),
        var=pd.DataFrame(index=pd.Index(gene_names, name=None)),
    )
    knn_signals.layers["sigma_knn"] = SIG[K_HEADLINE]["sigma_knn_pos"]   # positives-only, the canal-2 signal
    knn_signals.layers["mu_knn_pos"] = SIG[K_HEADLINE]["mu_knn_pos"]
    knn_signals.layers["pi_hat"] = SIG[K_HEADLINE]["pi_hat"]
    knn_signals.uns.update(dict(
        k=int(K_HEADLINE), tau=float(args.tau), kernel=args.kernel,
        min_pos_neighbors=int(args.min_pos_neighbors), donor=str(args.donor), fold=int(fold_num),
        dataset=str(args.dataset), scale="log1p_cp10k", sigma_is_standard_deviation=True,
        # Which scheme the unsuffixed `d_norm` column aliases, recorded in the file canal 2 opens.
        # Without it a canal-2 table built with --knn-signal d_norm silently mixes three different
        # quantities in one column (d_norm_gt on a labeled cohort, d_norm_gt_spatial15 on an
        # imputed one, d_norm_leiden on an unannotated one), and the choice is made per fold, so two
        # donors of one cohort can disagree. canal 2 asserts it is constant across donors.
        primary_cluster_scheme=str(PRIMARY_SCHEME),
        cluster_schemes=sorted(CLUSTER_SCHEMES),
    ))
    # Compressed like the repo's other h5ad writers: four dense float32 (n_query x n_genes)
    # matrices are large uncompressed, and sigma_knn is NaN-heavy so it compresses well. lzf, not
    # gzip: decompression cost is what canal 2 pays.
    knn_signals.write_h5ad(os.path.join(base_out, "knn_signals.h5ad"), compression="lzf")
    print(f"[{_el():6.1f}s] wrote knn_signals.h5ad ({n_query} spots x {n_genes} genes @k={K_HEADLINE})",
          flush=True)

    # ── NaN-rate diagnostic for the positives-only variant, per (gene, k) ───────────────────────
    nan_rows = []
    for k in K_GRID:
        nan_rate_g = np.isnan(SIG[k]["sigma_knn_pos"]).mean(axis=0)
        mean_npos_g = NPOS[k].mean(axis=0)
        nan_rows.append(pd.DataFrame({"gene": gene_names, "k": k,
                                      "nan_rate": nan_rate_g, "mean_n_pos_neighbors": mean_npos_g}))
    pd.concat(nan_rows, ignore_index=True).to_parquet(
        os.path.join(base_out, "canal2_pos_neighbor_nan_rate.parquet"), index=False)

    # ── Canal 1 — kNN regression quality (all-neighbor mu_knn vs y), per (slide, gene), all k ──
    pearson_fn = CORR_METRICS["pearson"]
    canal1_rows = []
    for k in K_GRID:
        atoms, _, _ = per_slide_gene_corr(SIG[k]["mu_knn_all"], query_y, query_slides, pearson_fn, key="pearson")
        for a in atoms:
            canal1_rows.append({"slide": a["slide"], "gene": gene_names[a["gene"]], "k": k,
                                "pearson": a["pearson"], "n_spots": a["n_spots"]})
    canal1_df = pd.DataFrame(canal1_rows)
    canal1_df.to_parquet(os.path.join(base_out, "canal1_atoms.parquet"), index=False)
    canal1_headline_donor = float(canal1_df[canal1_df["k"] == K_HEADLINE]
                                  .groupby("gene")["pearson"].mean().mean())
    print(f"[{_el():6.1f}s] canal 1 done | this-donor kNN pearson @k={K_HEADLINE}: "
          f"{canal1_headline_donor:.4f} (baseline heteroscedastic headline: {baseline_headline})",
          flush=True)

    # ── Canal 3 — zero detection AUROC, per (slide, gene): 1-pi_hat (all k) vs -mu_MLP ──────────
    def auroc_matrix(is_zero, score):
        """Vectorized per-column AUROC via the rank-sum identity (Mann-Whitney U), the same
        implementation as canal3_zero_detection_precompute.py. is_zero/score: (n_spots, G) ->
        (auroc, n_zero, n_pos)."""
        n_zero = is_zero.sum(axis=0)
        n_pos = (~is_zero).sum(axis=0)
        order = np.argsort(score, axis=0)
        ranks = np.empty_like(order, dtype=np.float64)
        cols = np.arange(score.shape[1])
        ranks[order, cols] = np.arange(1, score.shape[0] + 1)[:, None]
        sum_ranks_zero = np.where(is_zero, ranks, 0.0).sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            auroc = (sum_ranks_zero - n_zero * (n_zero + 1) / 2.0) / (n_zero * n_pos)
        return auroc, n_zero, n_pos

    def pair_auroc(is_zero, score):
        parts = []
        for sl in pd.unique(query_slides):
            m = query_slides == sl
            auroc, n_zero, n_pos = auroc_matrix(is_zero[m], score[m])
            ok = (n_zero >= MIN_SIDE) & (n_pos >= MIN_SIDE)
            parts.append(pd.DataFrame({"slide": sl, "gene": np.asarray(gene_names)[ok],
                                       "n_zero": n_zero[ok].astype(int), "n_pos": n_pos[ok].astype(int),
                                       "auroc": auroc[ok]}))
        return pd.concat(parts, ignore_index=True)

    is_zero = (query_y == 0)
    canal3_knn_parts = []
    for k in K_GRID:
        df_k = pair_auroc(is_zero, 1.0 - SIG[k]["pi_hat"])
        df_k["k"] = k
        canal3_knn_parts.append(df_k)
    pd.concat(canal3_knn_parts, ignore_index=True).to_parquet(
        os.path.join(base_out, "canal3_pair_auroc_knn.parquet"), index=False)
    pair_auroc(is_zero, -mu_mlp).to_parquet(
        os.path.join(base_out, "canal3_pair_auroc_mlp.parquet"), index=False)
    print(f"[{_el():6.1f}s] canal 3 (knn + mlp) done", flush=True)

    # ── Canal 3 bonus (optional): NB P(Y=0) from a sibling run, same donor/fold, same test spots ──
    nb_bonus_available = False
    try:
        nb_run_dir = os.path.join(RESULTS_ROOT, args.dataset, args.run_id, gene_set, args.radius,
                                  args.nb_model_type)
        with open(os.path.join(nb_run_dir, "metrics.json")) as f:
            M_nb = json.load(f)
        assert list(M_nb["gene_names"]) == list(gene_names), "NB sibling run has a different gene panel"
        with open(os.path.join(nb_run_dir, "splits.json")) as f:
            splits_nb = json.load(f)
        fold_nb = [fo for fo in splits_nb["folds"]
                  if str(args.donor) in [str(d) for d in fo["test_donor"]]][0]["fold"]
        p_nb = sc.read_h5ad(os.path.join(nb_run_dir, "predictions", f"preds_fold{fold_nb}.h5ad"),
                            backed="r")
        pred_nb = p_nb[p_nb.obs["split"].to_numpy() == "test"].to_memory()
        del p_nb
        order_nb = pd.Index(pred_nb.obs["obs_name"]).get_indexer(query_obs_names)
        assert (order_nb >= 0).all(), (
            "NB sibling run's test spots do not cover this donor's query spots -- "
            "get_indexer returned -1 for at least one obs_name (would silently alias the last row)."
        )
        pred_nb = pred_nb[order_nb]
        assert np.array_equal(pred_nb.obs["obs_name"].to_numpy().astype(str), query_obs_names)
        mu_nb = np.asarray(pred_nb.X, dtype=np.float64)
        alpha_nb = np.asarray(pred_nb.layers["dispersion"], dtype=np.float64)
        # P(Y=0) needs mu on the scale the NB was fitted on: X is that mu and layers["dispersion"]
        # is alpha at the same scale, so both are used as they are. A file without the
        # count_convention marker cannot be read on that assumption and raises; the except below
        # reports it as a skip of this optional bonus, with the reason printed.
        if pred_nb.uns.get("count_convention") != "v2_no_offset":
            raise ValueError(
                f"{nb_run_dir}: artifact with no count_convention marker. "
                f"Re-run the precompute."
            )
        mu_c, a_t = torch.as_tensor(mu_nb), torch.as_tensor(alpha_nb)
        dist_nb = torch.distributions.NegativeBinomial(
            total_count=1.0 / a_t, logits=torch.log(mu_c) + torch.log(a_t))
        score_nb = dist_nb.log_prob(torch.zeros_like(mu_c)).exp().numpy()   # exact P(Y=0)
        pair_auroc(is_zero, score_nb).to_parquet(
            os.path.join(base_out, "canal3_pair_auroc_nb_bonus.parquet"), index=False)
        nb_bonus_available = True
        print(f"[{_el():6.1f}s] canal 3 NB bonus ({args.nb_model_type}) done", flush=True)
    # Narrow, not bare Exception: every way the sibling run can legitimately be ABSENT or stale is
    # listed here, so a genuine bug inside the block surfaces instead of being reported as
    # "unavailable" -- which is indistinguishable in the log from the artifact simply not existing.
    except (OSError, AssertionError, ValueError, KeyError, IndexError) as e:
        print(f"[{_el():6.1f}s] canal 3 NB bonus skipped ({args.nb_model_type} unavailable: "
              f"{type(e).__name__}: {e})", flush=True)

    # ── Canal 2 — Spearman(sigma_knn_pos, |Delta|) within mu-deciles, for both |Delta_knn| and
    # |Delta_MLP|, swept over k, plus the sigma_mlp comparator. Deliberately not on the log/price
    # scale that canal2_uncertainty_precompute.py uses: mu_knn/sigma_knn/mu_mlp already share the
    # log1p(CP10k) scale by construction (the no-z-score decision above), so |Delta| stays there too
    # and nothing is reconverted.
    def rankcorr_per_group(df, group_cols, col_x, col_y):
        """Spearman per group via the closed-form Pearson-on-ranks identity, one groupby().agg(),
        no per-group scipy call. Same method as canal2_uncertainty_precompute.py's
        rankcorr_per_group."""
        d = df.copy()
        d["rx"] = d.groupby(group_cols)[col_x].rank()
        d["ry"] = d.groupby(group_cols)[col_y].rank()
        d["rxy"] = d["rx"] * d["ry"]
        d["rx2"] = d["rx"] ** 2
        d["ry2"] = d["ry"] ** 2
        g = d.groupby(group_cols).agg(n=("rx", "size"), sx=("rx", "sum"), sy=("ry", "sum"),
                                      sxy=("rxy", "sum"), sxx=("rx2", "sum"), syy=("ry2", "sum")).reset_index()
        num = g["n"] * g["sxy"] - g["sx"] * g["sy"]
        den = np.sqrt((g["n"] * g["sxx"] - g["sx"] ** 2) * (g["n"] * g["syy"] - g["sy"] ** 2))
        with np.errstate(divide="ignore", invalid="ignore"):
            g["spearman"] = num / den
        return g[group_cols + ["n", "spearman"]]

    def spearman_by_mu_decile(gene_arr, slide_arr, mu_arr, x_arr, y_arr):
        """bin -> pair (n-weighted) -> gene (n_plus-weighted) cascade, same weighting as
        canal2_uncertainty_precompute.py's stratified_spearman. Drops NaN x (the <min_pos_neighbors
        gate) and (gene,slide) pairs outside the n+ regime's terciles/quintiles/deciles buckets."""
        df = pd.DataFrame({"gene": gene_arr, "slide": slide_arr, "mu": mu_arr, "x": x_arr, "y": y_arr})
        df = df[np.isfinite(df["x"])]
        df = df.merge(regime_lookup, on=["gene", "slide"], how="inner")
        if df.empty:
            return pd.DataFrame(columns=["gene", "spearman_gene", "n_pairs"])
        df["mu_rank_pct"] = df.groupby(["gene", "slide"])["mu"].rank(pct=True)
        df["bin"] = np.minimum((df["mu_rank_pct"] * df["n_bins"]).astype(int), df["n_bins"] - 1)
        bin_stats = rankcorr_per_group(df, ["gene", "slide", "bin"], "x", "y")
        bin_stats = bin_stats[bin_stats["spearman"].notna()].copy()
        bin_stats["w"] = bin_stats["spearman"] * bin_stats["n"]
        pair = bin_stats.groupby(["gene", "slide"]).agg(sum_w=("w", "sum"), sum_n=("n", "sum")).reset_index()
        pair["spearman_pair"] = pair["sum_w"] / pair["sum_n"]
        pair = pair.merge(df[["gene", "slide", "n_plus"]].drop_duplicates(), on=["gene", "slide"])
        pair["wg"] = pair["spearman_pair"] * pair["n_plus"]
        gene = pair.groupby("gene").agg(sum_wg=("wg", "sum"), sum_np=("n_plus", "sum"),
                                        n_pairs=("gene", "size")).reset_index()
        gene["spearman_gene"] = gene["sum_wg"] / gene["sum_np"]
        return gene[["gene", "spearman_gene", "n_pairs"]]

    positive = query_y > 0
    spot_i, gene_j = np.nonzero(positive)
    gene_arr = np.asarray(gene_names)[gene_j]
    slide_arr = query_slides[spot_i]
    mu_arr = mu_mlp[spot_i, gene_j]                  # shared mu-decile conditioning variable (MLP mean)
    y_true_arr = query_y[spot_i, gene_j]
    abs_err_mlp_arr = np.abs(mu_arr - y_true_arr)

    gene_rows = []
    for k in K_GRID:
        sigma_knn_arr = SIG[k]["sigma_knn_pos"][spot_i, gene_j]
        abs_err_knn_arr = np.abs(SIG[k]["mu_knn_pos"][spot_i, gene_j] - y_true_arr)
        for target_name, err_arr in (("mlp", abs_err_mlp_arr), ("knn", abs_err_knn_arr)):
            g = spearman_by_mu_decile(gene_arr, slide_arr, mu_arr, sigma_knn_arr, err_arr)
            g["k"], g["signal"], g["target"] = k, "sigma_knn", target_name
            gene_rows.append(g)
    if sigma_mlp_available:
        sigma_mlp_arr = np.sqrt(var_mlp)[spot_i, gene_j]
        g = spearman_by_mu_decile(gene_arr, slide_arr, mu_arr, sigma_mlp_arr, abs_err_mlp_arr)
        g["k"], g["signal"], g["target"] = -1, "sigma_mlp", "mlp"
        gene_rows.append(g)
    canal2_gene_spearman = pd.concat(gene_rows, ignore_index=True)
    canal2_gene_spearman = canal2_gene_spearman.rename(columns={"n_pairs": "n_pairs_gene"})
    canal2_gene_spearman.to_parquet(os.path.join(base_out, "canal2_gene_spearman.parquet"), index=False)
    print(f"[{_el():6.1f}s] canal 2 Spearman-vs-k done ({len(canal2_gene_spearman)} gene rows)", flush=True)

    # ── Canal 2 headline: sparsification / risk-coverage curves at k=K_HEADLINE, 5 sort keys ────
    sigma_knn_h = SIG[K_HEADLINE]["sigma_knn_pos"][spot_i, gene_j]
    d_mean_h = d_by_k[K_HEADLINE]["d_mean"][spot_i]      # per-spot, broadcast over gene_j
    sparse_df = pd.DataFrame({"gene": gene_arr, "slide": slide_arr, "sigma_knn": sigma_knn_h,
                              "d_mean": d_mean_h, "abs_err_mlp": abs_err_mlp_arr})
    if sigma_mlp_available:
        sparse_df["sigma_mlp"] = np.sqrt(var_mlp)[spot_i, gene_j]
    sparse_df = sparse_df.merge(regime_lookup[["gene", "slide"]].drop_duplicates(), on=["gene", "slide"])
    sparse_df = sparse_df[np.isfinite(sparse_df["sigma_knn"])]
    if sigma_mlp_available:
        sparse_df = sparse_df[np.isfinite(sparse_df["sigma_mlp"])]

    rng = np.random.default_rng(args.seed)
    err = sparse_df["abs_err_mlp"].to_numpy()
    curves = {
        "sigma_knn": risk_coverage_curve(err, sparse_df["sigma_knn"].to_numpy()),
        "d_mean":    risk_coverage_curve(err, sparse_df["d_mean"].to_numpy()),
        "random":    risk_coverage_curve(err, rng.random(len(sparse_df))),
        "oracle":    risk_coverage_curve(err, err),
    }
    if sigma_mlp_available:
        curves["sigma_mlp"] = risk_coverage_curve(err, sparse_df["sigma_mlp"].to_numpy())
    sparsification_df = pd.DataFrame({"coverage": FRACS,
                                      **{f"curve_{name}": v for name, v in curves.items()}})
    sparsification_df.to_parquet(os.path.join(base_out, "canal2_sparsification.parquet"), index=False)
    print(f"[{_el():6.1f}s] canal 2 sparsification done ({len(sparse_df)} (spot,gene) pairs "
          f"@k={K_HEADLINE})", flush=True)

    # ── Canal 2 error on the canonical scale, alongside the native log1p(CP10k) one above and not a
    # replacement for it. Matches canal2_uncertainty_precompute.py's build_long_df exactly: the CP10k
    # point estimate is expm1(mu), the predictive median for a continuous/Gaussian family, not the
    # mean exp(mu+var/2)-1, which would bake sigma into the very error metric used to judge sigma.
    # The error is |log(y_hat) - log(y)| in natural log, not log1p, on the same (spot, gene) pairs as
    # the native Delta above (query_y > 0).
    #
    # Substitution, not clamping. The prediction is replaced only where log(mu) does not exist, i.e.
    # mu <= 0; everywhere else the model's own mean is used however small it is. Clamping with
    # np.maximum(mu, floor) instead would also replace every 0 < mu < floor, rewriting the middle of
    # the distribution rather than trimming a tail and manufacturing ties that attenuate the rank
    # statistic being reported.
    # This block exists to match canal 2, so it has to move whenever canal 2 moves: putting the kNN
    # baseline beside the probabilistic heads under two different error definitions would compare
    # the definitions, not the signals.
    y_price_arr = np.expm1(y_true_arr)
    mu_mlp_price_arr = np.expm1(mu_arr)
    floor_price = float(np.nanpercentile(y_price_arr, 1))
    abs_err_mlp_canon_arr = np.abs(
        np.log(np.where(mu_mlp_price_arr > 0, mu_mlp_price_arr, floor_price)) - np.log(y_price_arr))

    canon_gene_rows = []
    abs_err_knn_canon_by_k = {}   # kept for the partial-correlation block below -- avoids recomputing
    for k in K_GRID:
        sigma_knn_arr = SIG[k]["sigma_knn_pos"][spot_i, gene_j]
        mu_knn_price_arr = np.expm1(SIG[k]["mu_knn_pos"][spot_i, gene_j])
        abs_err_knn_canon_arr = np.abs(          # same substitution rule as the MLP error above
            np.log(np.where(mu_knn_price_arr > 0, mu_knn_price_arr, floor_price)) - np.log(y_price_arr))
        abs_err_knn_canon_by_k[k] = abs_err_knn_canon_arr
        for target_name, err_arr in (("mlp", abs_err_mlp_canon_arr), ("knn", abs_err_knn_canon_arr)):
            g = spearman_by_mu_decile(gene_arr, slide_arr, mu_arr, sigma_knn_arr, err_arr)
            g["k"], g["signal"], g["target"] = k, "sigma_knn", target_name
            canon_gene_rows.append(g)
    if sigma_mlp_available:
        sigma_mlp_arr = np.sqrt(var_mlp)[spot_i, gene_j]
        g = spearman_by_mu_decile(gene_arr, slide_arr, mu_arr, sigma_mlp_arr, abs_err_mlp_canon_arr)
        g["k"], g["signal"], g["target"] = -1, "sigma_mlp", "mlp"
        canon_gene_rows.append(g)
    canal2_gene_spearman_canonical = pd.concat(canon_gene_rows, ignore_index=True)
    canal2_gene_spearman_canonical = canal2_gene_spearman_canonical.rename(columns={"n_pairs": "n_pairs_gene"})
    canal2_gene_spearman_canonical["floor_price_cp10k"] = floor_price
    canal2_gene_spearman_canonical.to_parquet(
        os.path.join(base_out, "canal2_gene_spearman_canonical.parquet"), index=False)
    print(f"[{_el():6.1f}s] canal 2 CANONICAL Spearman-vs-k done "
          f"({len(canal2_gene_spearman_canonical)} gene rows, floor={floor_price:.6f} cp10k)", flush=True)

    sparse_canon_df = pd.DataFrame({"gene": gene_arr, "slide": slide_arr, "sigma_knn": sigma_knn_h,
                                    "d_mean": d_mean_h, "abs_err_mlp": abs_err_mlp_canon_arr})
    if sigma_mlp_available:
        sparse_canon_df["sigma_mlp"] = np.sqrt(var_mlp)[spot_i, gene_j]
    sparse_canon_df = sparse_canon_df.merge(regime_lookup[["gene", "slide"]].drop_duplicates(), on=["gene", "slide"])
    sparse_canon_df = sparse_canon_df[np.isfinite(sparse_canon_df["sigma_knn"])]
    if sigma_mlp_available:
        sparse_canon_df = sparse_canon_df[np.isfinite(sparse_canon_df["sigma_mlp"])]

    rng_canon = np.random.default_rng(args.seed)
    err_canon = sparse_canon_df["abs_err_mlp"].to_numpy()
    curves_canon = {
        "sigma_knn": risk_coverage_curve(err_canon, sparse_canon_df["sigma_knn"].to_numpy()),
        "d_mean":    risk_coverage_curve(err_canon, sparse_canon_df["d_mean"].to_numpy()),
        "random":    risk_coverage_curve(err_canon, rng_canon.random(len(sparse_canon_df))),
        "oracle":    risk_coverage_curve(err_canon, err_canon),
    }
    if sigma_mlp_available:
        curves_canon["sigma_mlp"] = risk_coverage_curve(err_canon, sparse_canon_df["sigma_mlp"].to_numpy())
    sparsification_canon_df = pd.DataFrame({"coverage": FRACS,
                                            **{f"curve_{name}": v for name, v in curves_canon.items()}})
    sparsification_canon_df.to_parquet(
        os.path.join(base_out, "canal2_sparsification_canonical.parquet"), index=False)
    print(f"[{_el():6.1f}s] canal 2 CANONICAL sparsification done "
          f"({len(sparse_canon_df)} (spot,gene) pairs @k={K_HEADLINE})", flush=True)

    # ── Canal 2, partial Spearman given mu: the parametric complement to the mu-decile bins above.
    # The bin cascade controls for mu by stratifying; this controls for it analytically via the
    # classic partial rank correlation identity, on the same (gene, slide) pairs, with no binning.
    # Canonical scale only: the mu control is the last step of the relative-error diagnostic, so it
    # is run on the scale that diagnostic uses, not on the native log1p one.
    def partial_spearman_given_mu(gene_arr_, slide_arr_, mu_arr_, x_arr_, y_arr_):
        """rho(x, y | mu) = (rho_xy - rho_xmu*rho_ymu) / sqrt((1-rho_xmu^2)(1-rho_ymu^2)), each rho a
        Spearman rank correlation from rankcorr_per_group, so no scipy call per group. Same
        drop-NaN-x, n+ regime gate and pair-to-gene n_plus-weighted aggregation as
        spearman_by_mu_decile, for direct comparability; the only difference is no mu-bin split."""
        df = pd.DataFrame({"gene": gene_arr_, "slide": slide_arr_, "mu": mu_arr_, "x": x_arr_, "y": y_arr_})
        df = df[np.isfinite(df["x"])]
        df = df.merge(regime_lookup, on=["gene", "slide"], how="inner")
        if df.empty:
            return pd.DataFrame(columns=["gene", "spearman_partial_gene", "n_pairs"])
        rho_xy = rankcorr_per_group(df, ["gene", "slide"], "x", "y")[["gene", "slide", "spearman"]] \
            .rename(columns={"spearman": "rho_xy"})
        rho_xmu = rankcorr_per_group(df, ["gene", "slide"], "x", "mu")[["gene", "slide", "spearman"]] \
            .rename(columns={"spearman": "rho_xmu"})
        rho_ymu = rankcorr_per_group(df, ["gene", "slide"], "y", "mu")[["gene", "slide", "spearman"]] \
            .rename(columns={"spearman": "rho_ymu"})
        pair = rho_xy.merge(rho_xmu, on=["gene", "slide"]).merge(rho_ymu, on=["gene", "slide"])
        denom = np.sqrt((1 - pair["rho_xmu"] ** 2) * (1 - pair["rho_ymu"] ** 2))
        with np.errstate(divide="ignore", invalid="ignore"):
            pair["spearman_partial"] = (pair["rho_xy"] - pair["rho_xmu"] * pair["rho_ymu"]) / denom
        pair = pair[np.isfinite(pair["spearman_partial"])]
        pair = pair.merge(df[["gene", "slide", "n_plus"]].drop_duplicates(), on=["gene", "slide"])
        pair["wg"] = pair["spearman_partial"] * pair["n_plus"]
        gene = pair.groupby("gene").agg(sum_wg=("wg", "sum"), sum_np=("n_plus", "sum"),
                                        n_pairs=("gene", "size")).reset_index()
        gene["spearman_partial_gene"] = gene["sum_wg"] / gene["sum_np"]
        return gene[["gene", "spearman_partial_gene", "n_pairs"]]

    partial_gene_rows = []
    for k in K_GRID:
        sigma_knn_arr = SIG[k]["sigma_knn_pos"][spot_i, gene_j]
        for target_name, err_arr in (("mlp", abs_err_mlp_canon_arr), ("knn", abs_err_knn_canon_by_k[k])):
            g = partial_spearman_given_mu(gene_arr, slide_arr, mu_arr, sigma_knn_arr, err_arr)
            g["k"], g["signal"], g["target"] = k, "sigma_knn", target_name
            partial_gene_rows.append(g)
    if sigma_mlp_available:
        sigma_mlp_arr = np.sqrt(var_mlp)[spot_i, gene_j]
        g = partial_spearman_given_mu(gene_arr, slide_arr, mu_arr, sigma_mlp_arr, abs_err_mlp_canon_arr)
        g["k"], g["signal"], g["target"] = -1, "sigma_mlp", "mlp"
        partial_gene_rows.append(g)
    canal2_partial_spearman = pd.concat(partial_gene_rows, ignore_index=True)
    canal2_partial_spearman = canal2_partial_spearman.rename(columns={"n_pairs": "n_pairs_gene"})
    canal2_partial_spearman.to_parquet(
        os.path.join(base_out, "canal2_partial_spearman_canonical.parquet"), index=False)
    print(f"[{_el():6.1f}s] canal 2 CANONICAL partial-Spearman-given-mu done "
          f"({len(canal2_partial_spearman)} gene rows)", flush=True)

    # ── z-score diagnostic: inter- vs intra-slide variance, TRAIN spots of this fold ONLY ───────
    train_y = y_all[train_idx_only]
    train_slides = sample_labels[train_idx_only]
    diag_df = inter_intra_slide_variance(train_y, train_slides, gene_names=gene_names)
    diag_df.reset_index().to_parquet(os.path.join(base_out, "inter_intra_slide_variance.parquet"),
                                     index=False)

    spot_distance_df.to_parquet(os.path.join(base_out, "spot_distance.parquet"), index=False)

    summary = dict(
        dataset=args.dataset, run_id=args.run_id, gene_set=gene_set, radius=args.radius,
        model_type=model_type, lambda_dir=lambda_dir, donor=args.donor, fold=fold_num,
        n_ref_spots=int(ref_emb.shape[0]), n_query_spots=int(n_query), n_genes=int(n_genes),
        n_slides_donor=int(pd.unique(query_slides).size),
        k_grid=list(K_GRID), k_headline=K_HEADLINE, tau=args.tau, kernel=args.kernel,
        min_pos_neighbors=args.min_pos_neighbors, seed=args.seed,
        primary_corr_metric=primary_metric, baseline_headline_pearson=baseline_headline,
        canal1_this_donor_pearson_at_k_headline=canal1_headline_donor,
        sigma_mlp_available=sigma_mlp_available, nb_bonus_available=nb_bonus_available,
        target_transform=target_transform,
        # The same two keys both OOD stages write, and the only offline record of the RADII -- the
        # divisor behind every d_norm_<scheme> column of knn_signals.h5ad. signals is the stage
        # canal 2 consumes, so without them neither the divisor nor the meaning of the unsuffixed
        # `d_norm` alias could be recovered without re-running the whole donor.
        cluster_schemes=CLUSTER_SCHEME_PROVENANCE, primary_cluster_scheme=PRIMARY_SCHEME,
        # Floor policy marker for the canonical-scale canal 2 error, mirroring the key
        # canal2_uncertainty_precompute.py writes. Readers must gate on it: an artifact without the
        # key follows the clamp rule np.maximum(mu, floor) instead, and its errors are not
        # comparable with these outputs or with any canal 2 artifact.
        error_floor_policy="v2_substitute_where_undefined",
        canal2_floor_price_cp10k=floor_price,
        elapsed_seconds=_el(),
    )
    with open(os.path.join(base_out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[{_el():6.1f}s] DONE (stage=signals). {summary}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE: ood_corruption — image-degradation robustness (blur + salt-and-pepper), monotonicity
# of d_mean vs severity + the (norm-diff, centroid-distance, neighbor-Jaccard) mechanistic triptych.
# ══════════════════════════════════════════════════════════════════════════════
elif args.stage == "ood_corruption":
    index = build_reference_index(ref_emb)
    ref_centroid = l2_normalize(ref_emb).mean(axis=0)

    rng = np.random.default_rng(args.seed)
    n_sel = min(args.n_corruption_spots, n_query)
    sel = rng.choice(n_query, size=n_sel, replace=False)
    sel_slide = query_slides[sel]
    sel_obs_name = query_obs_names[sel]
    sel_barcode = np.array([n[:-(len(s) + 1)] if n.endswith(f"-{s}") else n
                            for n, s in zip(sel_obs_name, sel_slide)])
    print(f"[{_el():6.1f}s] sampled {n_sel}/{n_query} test spots for corruption OOD", flush=True)

    # Cluster assignment is fixed from the clean embedding (the main adata's query_emb, not a
    # re-loaded blur_dir(0.0) / SP_CLEAN_REF_DIR pass) and reused for every severity level of every
    # corruption type below. The 2x2 validation (type rare/common x clean/corrupted) needs "type"
    # held constant while severity varies; a corrupted embedding drifting into a different cluster
    # would conflate the two axes it is meant to separate. One nearest-centroid assignment per
    # scheme, all from the clean embedding. Precomputed here rather than through d_norm_columns()
    # inside the sweep, because that helper re-assigns from whatever embedding it is handed, which is
    # right everywhere else and wrong here.
    sel_cluster_idx = assign_all_schemes(query_emb[sel])

    # --emb-dataset defaults to --dataset. Set it when this cohort's corrupted extractions live under
    # a patch_embeddings/<name>/ dir that is not named after the cohort itself, otherwise the default
    # dies deep inside torch.load.
    EMB_BASE = os.path.join(PROJECT_ROOT, "patch_embeddings", args.emb_dataset or args.dataset)

    def blur_dir(bf):
        return os.path.join(EMB_BASE, f"embeddings_radius_factor_{args.radius_factor}_"
                                      f"swapxy_{args.swap_xy}_blurry_bf{bf}")

    # amount=0.0 has no dir under the saltpepper naming template, unlike blur_dir(0.0) which is a
    # real extracted dir. Both salt-and-pepper variants share one clean reference dir instead, and
    # that reference must be the dir the training embeddings came from: *_swapxy_<flag>_2 where it
    # exists, the plain *_swapxy_<flag> otherwise. The existence test reproduces the per-dataset
    # emb_subdir the preprocessing is configured with, without reading the config here.
    _SP_REF_2    = os.path.join(EMB_BASE, f"embeddings_radius_factor_{args.radius_factor}_"
                                          f"swapxy_{args.swap_xy}_2")
    _SP_REF_PLAIN = os.path.join(EMB_BASE, f"embeddings_radius_factor_{args.radius_factor}_"
                                           f"swapxy_{args.swap_xy}")
    SP_CLEAN_REF_DIR = _SP_REF_2 if os.path.isdir(_SP_REF_2) else _SP_REF_PLAIN

    # Both anchors checked HERE, before a single .pt is opened: a cohort whose corrupted extraction
    # was never run would otherwise surface as a bare FileNotFoundError from torch.load halfway
    # through the sweep. Name the dir instead.
    for _d, _what in ((blur_dir(0.0), "blur"), (SP_CLEAN_REF_DIR, "salt-and-pepper clean reference")):
        assert os.path.isdir(_d), (
            f"stage=ood_corruption: missing the {_what} embedding dir\n  {_d}\n"
            f"Corrupted embeddings are extracted per cohort by snake/scripts/embeddings_extraction/"
            f"launch_{{blur,saltpepper}}_extraction.sh (DATASET in 10x_TuPro | TLS | "
            f"LUNG_CANCER_LUAD | LUNG_CANCER_LUSC -- the two lung sub-cohorts both write into the "
            f"MERGED patch_embeddings/LUNG_CANCER/, so --emb-dataset LUNG_CANCER is what you want "
            f"there). Extract them first.")

    def load_severity_embeddings(emb_dir, slide_for_sel, barcode_for_sel):
        out = np.empty((len(barcode_for_sel), embeddings.shape[1]), dtype=np.float32)
        for sl in pd.unique(slide_for_sel):
            m = slide_for_sel == sl
            blob = torch.load(os.path.join(emb_dir, f"{sl}.pt"), map_location="cpu", weights_only=False)
            e = blob["embeddings"].float().numpy()
            names = np.asarray(blob["obs_names"]).astype(str)
            order = pd.Index(names).get_indexer(barcode_for_sel[m])
            assert (order >= 0).all(), f"barcode mismatch loading {emb_dir}/{sl}.pt"
            out[m] = e[order]
        return out

    def severity_sweep(dirs_and_labels):
        """dirs_and_labels: list of (embedding_dir, {label_col: value}). The first entry is the
        clean/zero-severity reference for embed_norm_diff and the neighbor Jaccard. For blur that is
        the real bf=0 extracted dir; for salt-and-pepper there is no amount=0 dir, so sp_dir(0.0, ...)
        resolves to the shared SP_CLEAN_REF_DIR. Both are a single real on-disk embedding set, just
        not always the same pipeline dir."""
        by_k_parts, mech_parts = [], []
        clean_emb, clean_idx_h = None, None
        for emb_dir, label in dirs_and_labels:
            emb = load_severity_embeddings(emb_dir, sel_slide, sel_barcode)
            dist_s, idx_s = query_neighbors(index, emb, k_max=max(K_GRID))
            d_s = spot_distance_signals(dist_s, k_grid=K_GRID)
            for k in K_GRID:
                by_k_parts.append(pd.DataFrame({
                    "obs_name": sel_obs_name, "slide": sel_slide, **label,
                    "k": k, "d_mean": d_s[k]["d_mean"], "d_k": d_s[k]["d_k"],
                    # per scheme, with the CLEAN-embedding assignment held fixed (see sel_cluster_idx)
                    **{f"d_norm_{n}": d_s[k]["d_mean"] / CLUSTER_SCHEMES[n]["radii"][ix]
                       for n, ix in sel_cluster_idx.items()},
                    **{f"cluster_label_{n}": CLUSTER_SCHEMES[n]["labels_order"][ix]
                       for n, ix in sel_cluster_idx.items()},
                    **{f"cluster_size_{n}": CLUSTER_SCHEMES[n]["counts"][ix]
                       for n, ix in sel_cluster_idx.items()},
                    "d_norm": d_s[k]["d_mean"] / CLUSTER_SCHEMES[PRIMARY_SCHEME]["radii"][sel_cluster_idx[PRIMARY_SCHEME]],
                    "cluster_label": CLUSTER_SCHEMES[PRIMARY_SCHEME]["labels_order"][sel_cluster_idx[PRIMARY_SCHEME]],
                    "cluster_size": CLUSTER_SCHEMES[PRIMARY_SCHEME]["counts"][sel_cluster_idx[PRIMARY_SCHEME]],
                }))
            if clean_emb is None:
                clean_emb, clean_idx_h = emb, idx_s[:, :K_HEADLINE]
            jacc = np.array([
                len(set(clean_idx_h[i].tolist()) & set(idx_s[i, :K_HEADLINE].tolist())) /
                len(set(clean_idx_h[i].tolist()) | set(idx_s[i, :K_HEADLINE].tolist()))
                for i in range(len(sel_obs_name))
            ])
            embed_norm_diff = np.linalg.norm(emb - clean_emb, axis=1)
            dist_to_centroid = np.linalg.norm(l2_normalize(emb) - ref_centroid, axis=1)
            mech_parts.append(pd.DataFrame({"obs_name": sel_obs_name, "slide": sel_slide, **label,
                                            "embed_norm_diff": embed_norm_diff,
                                            "dist_to_centroid": dist_to_centroid,
                                            "jaccard_headline": jacc}))
            print(f"[{_el():6.1f}s] severity {label} done", flush=True)
        return pd.concat(by_k_parts, ignore_index=True), pd.concat(mech_parts, ignore_index=True)

    blur_factors = [float(x) for x in args.blur_factors.split(",")]
    assert blur_factors[0] == 0.0, "--blur-factors must start at 0.0 (the clean/zero-severity anchor)"
    assert len(blur_factors) >= 2, "--blur-factors needs at least one non-zero severity besides the 0.0 anchor"

    blur_by_k, blur_mech = severity_sweep([(blur_dir(bf), {"blur_factor": bf}) for bf in blur_factors])
    blur_by_k.to_parquet(os.path.join(base_out, "blur_distance_by_k.parquet"), index=False)
    blur_mech.to_parquet(os.path.join(base_out, "blur_mechanistic.parquet"), index=False)

    sp_amounts = [float(x) for x in args.sp_amounts.split(",")]
    assert sp_amounts[0] == 0.0, "--sp-amounts must start at 0.0 (the clean/zero-severity anchor)"
    assert len(sp_amounts) >= 2, "--sp-amounts needs at least one non-zero severity besides the 0.0 anchor"
    SP_VARIANTS = {"monopixel": (1, 1), "dust_2-4": (2, 4)}

    def sp_dir(amount, smin, smax):
        if amount == 0.0:
            return SP_CLEAN_REF_DIR
        return os.path.join(EMB_BASE, f"embeddings_radius_factor_{args.radius_factor}_"
                                      f"swapxy_{args.swap_xy}_saltpepper_a{amount}_sp{smin}-{smax}")

    sp_by_k_parts, sp_mech_parts = [], []
    for vname, (smin, smax) in SP_VARIANTS.items():
        dirs_labels = [(sp_dir(a, smin, smax), {"variant": vname, "amount": a}) for a in sp_amounts]
        by_k_v, mech_v = severity_sweep(dirs_labels)
        sp_by_k_parts.append(by_k_v)
        sp_mech_parts.append(mech_v)
    pd.concat(sp_by_k_parts, ignore_index=True).to_parquet(
        os.path.join(base_out, "saltpepper_distance_by_k.parquet"), index=False)
    pd.concat(sp_mech_parts, ignore_index=True).to_parquet(
        os.path.join(base_out, "saltpepper_mechanistic.parquet"), index=False)

    blur_h = blur_by_k[blur_by_k["k"] == K_HEADLINE]
    blur_dmean_h = blur_h.groupby("blur_factor")["d_mean"].mean()
    blur_dnorm_h = blur_h.groupby("blur_factor")["d_norm"].mean()
    blur_monotone = bool(np.all(np.diff(blur_dmean_h.sort_index().to_numpy()) >= 0))
    blur_dnorm_monotone = bool(np.all(np.diff(blur_dnorm_h.sort_index().to_numpy()) >= 0))

    summary = dict(
        dataset=args.dataset, run_id=args.run_id, gene_set=gene_set, radius=args.radius,
        model_type=model_type, donor=args.donor, fold=fold_num, n_spots_sampled=n_sel,
        blur_factors=blur_factors, sp_amounts=sp_amounts, sp_variants=list(SP_VARIANTS.keys()),
        k_headline=K_HEADLINE, blur_dmean_by_severity=blur_dmean_h.to_dict(),
        blur_dmean_monotone_increasing=blur_monotone,
        blur_dnorm_by_severity=blur_dnorm_h.to_dict(),
        blur_dnorm_monotone_increasing=blur_dnorm_monotone,
        cluster_schemes=CLUSTER_SCHEME_PROVENANCE, primary_cluster_scheme=PRIMARY_SCHEME,
        elapsed_seconds=_el(),
    )
    with open(os.path.join(base_out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[{_el():6.1f}s] DONE (stage=ood_corruption). {summary}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE: ood_crossdataset — population-shift robustness: query a sample of a second dataset's spots
# against this fold's reference index, and compare their d_mean to the in-distribution one from this
# donor's own test spots.
# ══════════════════════════════════════════════════════════════════════════════
elif args.stage == "ood_crossdataset":
    # This stage measures population shift, so the two cohorts must differ. --cross-dataset defaults
    # to TLS and the launcher sweeps TLS too, so `--dataset TLS` on defaults would query TLS spots
    # against a TLS reference index, partly the held-out donor's own spots since sel_cross samples
    # the whole cross adata, and report an AUROC near 0.5 that looks like a legitimate negative
    # result. Nothing downstream can detect that, hence a hard stop.
    assert args.cross_dataset != args.dataset, (
        f"--cross-dataset {args.cross_dataset!r} == --dataset: querying a cohort against its own "
        f"reference index is not a population-shift control (AUROC ~0.5 by construction). Pass an "
        f"explicit --cross-dataset (the python default is TLS).")
    index = build_reference_index(ref_emb)

    cross_adata_path = os.path.join(PROJECT_ROOT, "results", args.cross_dataset, "_preprocessed",
                                    args.cross_preproc_id, args.cross_radius, "adata.h5ad")
    assert os.path.isfile(cross_adata_path), f"missing cross-dataset adata: {cross_adata_path}"
    cross_adata = ad.read_h5ad(cross_adata_path)
    assert EMB_OBSM_KEY in cross_adata.obsm, f"{cross_adata_path}: missing obsm['{EMB_OBSM_KEY}']"
    cross_emb_full = np.asarray(cross_adata.obsm[EMB_OBSM_KEY])
    cross_obs_names = np.asarray(cross_adata.obs_names).astype(str)
    del cross_adata

    rng = np.random.default_rng(args.seed)
    n_cross = min(args.n_cross_spots, cross_emb_full.shape[0])
    sel_cross = rng.choice(cross_emb_full.shape[0], size=n_cross, replace=False)
    cross_emb = cross_emb_full[sel_cross]
    print(f"[{_el():6.1f}s] sampled {n_cross}/{cross_emb_full.shape[0]} {args.cross_dataset} spots",
          flush=True)

    dist_in, _ = query_neighbors(index, query_emb, k_max=max(K_GRID))
    d_in = spot_distance_signals(dist_in, k_grid=K_GRID)
    dist_cross, _ = query_neighbors(index, cross_emb, k_max=max(K_GRID))
    d_cross = spot_distance_signals(dist_cross, k_grid=K_GRID)
    print(f"[{_el():6.1f}s] both queries done against the donor-{args.donor} reference index",
          flush=True)

    # Nearest-centroid assignment for both query sets, against this cohort's own clusters. cross_emb
    # never needs a real label: it is forced into that cluster structure exactly like a corrupted
    # embedding is. One assignment per query set, k-independent, reused for every k and by the AUROC
    # block below.
    in_assign, cross_assign = assign_all_schemes(query_emb), assign_all_schemes(cross_emb)
    dn_in_h, dn_cross_h = None, None
    rows = []
    for k in K_GRID:
        dn_in = d_norm_columns(d_in[k]["d_mean"], in_assign)
        dn_cross = d_norm_columns(d_cross[k]["d_mean"], cross_assign)
        if k == K_HEADLINE:
            dn_in_h, dn_cross_h = dn_in, dn_cross
        rows.append(pd.DataFrame({"source": "in_distribution", "id": query_obs_names,
                                  "slide_or_dataset": query_slides, "k": k,
                                  "d_mean": d_in[k]["d_mean"], "d_k": d_in[k]["d_k"],
                                  **{c: v for c, v in dn_in.items() if not c.startswith("cluster_size_")},
                                  "d_norm": dn_in[f"d_norm_{PRIMARY_SCHEME}"],
                                  "cluster_label": dn_in[f"cluster_label_{PRIMARY_SCHEME}"]}))
        rows.append(pd.DataFrame({"source": "cross_dataset", "id": cross_obs_names[sel_cross],
                                  "slide_or_dataset": args.cross_dataset, "k": k,
                                  "d_mean": d_cross[k]["d_mean"], "d_k": d_cross[k]["d_k"],
                                  **{c: v for c, v in dn_cross.items() if not c.startswith("cluster_size_")},
                                  "d_norm": dn_cross[f"d_norm_{PRIMARY_SCHEME}"],
                                  "cluster_label": dn_cross[f"cluster_label_{PRIMARY_SCHEME}"]}))
    pd.concat(rows, ignore_index=True).to_parquet(
        os.path.join(base_out, "crossdataset_distance.parquet"), index=False)

    y_true = np.concatenate([np.zeros(n_query), np.ones(n_cross)])   # 1 = cross-dataset (expected farther)
    score_dmean = np.concatenate([d_in[K_HEADLINE]["d_mean"], d_cross[K_HEADLINE]["d_mean"]])
    auroc_sep = float(roc_auc_score(y_true, score_dmean))
    # One separability AUROC per cluster scheme, plus the unsuffixed one for PRIMARY_SCHEME: which
    # scheme normalizes the distance is exactly the question this stage is a control for.
    auroc_sep_dnorm_by_scheme = {          # reuses the k=K_HEADLINE dicts kept by the loop above
        name: float(roc_auc_score(y_true, np.concatenate([dn_in_h[f"d_norm_{name}"],
                                                          dn_cross_h[f"d_norm_{name}"]])))
        for name in CLUSTER_SCHEMES
    }
    auroc_sep_dnorm = auroc_sep_dnorm_by_scheme[PRIMARY_SCHEME]

    summary = dict(
        dataset=args.dataset, run_id=args.run_id, gene_set=gene_set, radius=args.radius,
        model_type=model_type, donor=args.donor, fold=fold_num,
        cross_dataset=args.cross_dataset, cross_preproc_id=args.cross_preproc_id,
        cross_radius=args.cross_radius, n_in_distribution=int(n_query), n_cross_dataset=int(n_cross),
        k_headline=K_HEADLINE, auroc_separability_at_k_headline=auroc_sep,
        auroc_separability_dnorm_at_k_headline=auroc_sep_dnorm,   # == the PRIMARY_SCHEME entry below
        auroc_separability_dnorm_by_scheme=auroc_sep_dnorm_by_scheme,
        cluster_schemes=CLUSTER_SCHEME_PROVENANCE, primary_cluster_scheme=PRIMARY_SCHEME,
        elapsed_seconds=_el(),
    )
    with open(os.path.join(base_out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[{_el():6.1f}s] DONE (stage=ood_crossdataset). {summary}", flush=True)
