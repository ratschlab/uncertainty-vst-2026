# src/data/load_and_preprocess.py
#
# Consolidated spatial-transcriptomics preprocessing.
#
# Produces ONE unified, preprocessed AnnData per (radius), holding everything
# downstream needs:
#   .X                       CP10k + log1p expression (shared genes)
#   .layers["raw_counts"]    raw integer counts (shared genes)
#   .obs["size_factor_total_over_1e4"]   per-spot total/1e4 (CP10k convention):
#                            raw_counts/s == CP10k == exp(.X)-1.
#   .obs["size_factor_median_ratios"]    per-spot DESeq2 median-of-ratios (poscounts,
#                            per slide). Both size-factor columns are diagnostic.
#                            No model reads them: there is no offset and no per-spot
#                            exposure in the modeling path. They exist so offline
#                            analysis can inspect the library-size distribution.
#   .obs["sample_id"]        sample/slide of each spot
#   .obs[...]                everything from each sample's obs (incl. ground_truth)
#   .obs["ground_truth_knn_{space}_{k}"]  KNN-imputed labels for the UNASSIGNED
#                            spots (space ∈ spatial|expr|emb, k ∈ 5|10|15); added
#                            only for slides that have UNASSIGNED ground_truth.
#   .var_names               gene names
#   .var["highly_variable_top{N}"]  HVG flag per requested N
#   .obsm["X_emb"]           patch embeddings
#   .obsm["spatial"]         spot coordinates
#   .uns["hires_images"]     {sample: (H,W,3)} hires images (if any)
#
# HVG runs with batch_key on raw counts, which seurat_v3 requires. The selection
# is recorded in .var and the gene lists are written to small text files for reuse
# as named gene_sets. One adata holds every N: slicing to a gene_set happens at
# train time, so only radius changes the heavy output.
#
# Ordering matters. Zero-count spots are dropped and CP10k+log1p is applied per
# slide, before the concat. Each spot's normalization denominator is therefore its
# own full measured transcriptome, not the post-alignment shared subset. The single
# concat(join="inner") then aligns both .X and the raw_counts layer to the shared
# gene set in one consistent spot order. HVG runs on the aligned raw layer.

import os
# Pin BLAS threads to 1 in worker processes so N parallel loaders don't
# oversubscribe the cores. Must run before numpy/scanpy import.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import concurrent.futures
import numpy as np
import anndata as ad
import scanpy as sc


# ─────────────────────────────────────────────────────────────────────────────
# KNN ground_truth imputation (one slide, one feature space)
# ─────────────────────────────────────────────────────────────────────────────

def _impute_knn_all_k(features, labels, is_unlab, k_values):
    """Majority-vote imputed labels for the UNASSIGNED rows, one result per k.

    Each UNASSIGNED row takes the majority label of its k nearest labeled rows in
    `features` (euclidean). One NearestNeighbors fit at the largest k serves every
    k, by slicing the neighbor-index array.

    Returns {k: labels_filled}, a copy of `labels` in which only the UNASSIGNED
    entries are replaced.
    """
    from sklearn.neighbors import NearestNeighbors
    import pandas as pd

    labeled        = ~is_unlab
    n_labeled      = int(labeled.sum())
    labeled_labels = labels[labeled]
    k_max          = min(max(k_values), n_labeled)   # can't ask for more than we have

    nbrs = NearestNeighbors(n_neighbors=k_max, metric="euclidean")
    nbrs.fit(features[labeled])
    _, idx = nbrs.kneighbors(features[is_unlab])      # (n_unlab, k_max)

    out = {}
    for k in k_values:
        k_eff  = min(k, n_labeled)
        voted  = [pd.Series(labeled_labels[row[:k_eff]]).mode()[0] for row in idx]
        filled = labels.copy()
        filled[is_unlab] = voted
        out[k] = filled
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Size factors (diagnostic columns only)
# ─────────────────────────────────────────────────────────────────────────────

def _median_ratio_size_factors(X):
    """DESeq2-style median-of-ratios size factors (poscounts variant), per spot,
    for one slide's raw count matrix X (n_spots, n_genes).

    The per-gene reference is the geometric mean over the spots that express that
    gene. That is the poscounts variant, robust to the many zeros in spatial data. A
    spot's size factor is the median of count / gene-reference over the genes it
    expresses. The reference comes from this slide's own spots, which matches
    size_factor_total_over_1e4 and the slide-aware metric. A spot with no usable gene
    returns NaN, and the caller substitutes the total-count factor.

    Spatial data is sparse enough that this estimate can be unstable, or dominated by
    ubiquitously expressed genes. size_factor_total_over_1e4 is the safer default.
    """
    import warnings
    # float32 keeps the per-slide dense copy small (slides load in parallel); a
    # median-of-ratios is a robust statistic, so float32 precision is ample.
    X = np.asarray(X.toarray() if hasattr(X, "toarray") else X, dtype=np.float32)
    pos = X > 0
    counts_pos = pos.sum(axis=0)                          # spots expressing each gene
    logX = np.zeros_like(X)
    logX[pos] = np.log(X[pos])
    loggeo = logX.sum(axis=0) / np.maximum(counts_pos, 1)    # log geom-mean per gene
    logratio = np.where(pos, logX - loggeo[None, :], np.nan)
    with warnings.catch_warnings():                          # all-NaN rows -> NaN
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.exp(np.nanmedian(logratio, axis=1))        # per spot


# ─────────────────────────────────────────────────────────────────────────────
# Per-sample load: read h5ad, attach embeddings before any filtering
# ─────────────────────────────────────────────────────────────────────────────

def _load_one(args):
    sample, h5ad_path, emb_path, target_sum = args
    import torch  # local import keeps the module importable without torch

    # Load the slide and attach its embeddings, checking the two are row-aligned by barcode.
    import pandas as pd  # local, like the torch import above

    adata = sc.read_h5ad(h5ad_path)
    blob = torch.load(emb_path, map_location="cpu", weights_only=False)
    emb = blob["embeddings"].numpy()

    # Count check: necessary but NOT sufficient (equal length ≠ equal order).
    assert emb.shape[0] == adata.n_obs, (
        f"Embedding/h5ad row mismatch for '{sample}': emb has {emb.shape[0]} "
        f"rows, h5ad has {adata.n_obs}. Re-run patch embedding extraction."
    )

    # Order check: if the .pt carries barcodes, verify against obs_names and
    # realign if the order differs. Guards against a silent upstream reorder.
    if "obs_names" in blob:
        emb_names = np.asarray(blob["obs_names"]).astype(str)
        obs_names = adata.obs_names.to_numpy().astype(str)
        if not np.array_equal(emb_names, obs_names):
            order = pd.Index(emb_names).get_indexer(obs_names)
            assert (order >= 0).all(), (
                f"'{sample}': embedding barcodes don't match h5ad obs_names "
                f"({int((order < 0).sum())} spots unmatched). Re-run extraction."
            )
            emb = emb[order]          # realign to this slide's obs order
    else:
        print(f"  [warn] {sample}: .pt has no 'obs_names'; positional order "
              f"only. Re-extract to enable the barcode check.")

    adata.obsm["X_emb"] = emb
    adata.obs["sample_id"] = sample
    # __________________________

    # Ground-truth label distribution, captured before the zero-count drop, the
    # normalization and the KNN imputation. This is the per-slide
    # "__groundtruth types diagnostic___" line in the log.
    gt_counts = (
        adata.obs["ground_truth"].value_counts().to_dict()
        if "ground_truth" in adata.obs.columns else None
    )

    # Record this slide's full panel size (before any gene alignment) so the
    # diagnostic can report how many genes the inner-join later drops.
    panel = list(map(str, adata.var_names))

    # Drop zero-count spots using this slide's full panel. The embedding row goes
    # with the spot: obsm is row-aligned, so no spot/embedding shift can creep in.
    totals = np.asarray(adata.X.sum(axis=1)).ravel()
    keep = totals > 0
    n_zero = int((~keep).sum())
    if n_zero:
        adata = adata[keep].copy()

    # CP10k + log1p run per slide, on the full panel, before the inner-join concat.
    # Each spot's normalization denominator is therefore its true measured total,
    # not the truncated total of the post-alignment shared gene subset.
    adata.layers["raw_counts"] = adata.X.copy()      # seurat_v3 HVG needs these
    # ── Per-spot size factors (diagnostic columns) ──────────────────────────────
    # Computed before normalize_total, while .X still holds raw counts, per slide on
    # the full panel, and stored in obs so they survive the inner-join concat. No
    # model reads them: every count model fits its likelihood directly on its own
    # target (cp_counts, cp_counts_nr, raw_counts), with no offset and no per-spot
    # exposure. Offline analysis uses them to inspect the library-size distribution.
    #   size_factor_total_over_1e4 : total / 1e4 (CP10k convention), so
    #       raw_counts / s == CP10k == exp(.X)-1.
    #   size_factor_median_ratios  : DESeq2 median-of-ratios (poscounts), per slide.
    totals = np.asarray(adata.X.sum(axis=1)).ravel()
    adata.obs["size_factor_total_over_1e4"] = (totals / target_sum).astype(np.float32)
    sf_mr = _median_ratio_size_factors(adata.X)
    sf_mr = np.where(np.isfinite(sf_mr) & (sf_mr > 0), sf_mr,
                     adata.obs["size_factor_total_over_1e4"].to_numpy())   # fallback
    adata.obs["size_factor_median_ratios"] = sf_mr.astype(np.float32)
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)                               # .X is now log-CP10k

    # ── KNN ground_truth imputation (per slide, before the inner-join concat) ──
    # UNASSIGNED spots get a label by majority vote over their k nearest labeled
    # spots, computed independently in each available feature space:
    #   spatial : .obsm["spatial"]  (x,y), when present. Region labels are spatial
    #             domains.
    #   expr    : .X                log-CP10k expression, already normalized here
    #   emb     : .obsm["X_emb"]    H&E patch embedding
    # One obs column per (space, k): ground_truth_knn_{space}_{k}, added only for a
    # slide that has both UNASSIGNED and labeled spots. ground_truth itself is left
    # untouched, so its UNASSIGNED entries remain.
    K_VALUES = (5, 10, 15)
    knn_info = None
    if gt_counts is not None and "UNASSIGNED" in gt_counts:
        labels   = adata.obs["ground_truth"].to_numpy().astype(str)
        is_unlab = labels == "UNASSIGNED"
        if is_unlab.any() and (~is_unlab).any():     # need both classes present
            spaces = {
                "expr": np.asarray(adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X),
                "emb":  np.asarray(adata.obsm["X_emb"]),
            }
            if "spatial" in adata.obsm:
                spaces = {"spatial": np.asarray(adata.obsm["spatial"]), **spaces}
            after = {}
            for space_name, feats in spaces.items():
                for k, imputed in _impute_knn_all_k(feats, labels, is_unlab, K_VALUES).items():
                    col = f"ground_truth_knn_{space_name}_{k}"
                    adata.obs[col] = pd.Categorical(imputed)
                    after[col] = pd.Series(imputed).value_counts().to_dict()
            knn_info = {"n_unassigned": int(is_unlab.sum()),
                        "n_total": int(labels.size), "after": after}

    # stash hires image (per-sample, not per-spot) for later viz
    img = None
    if "spatial" in adata.uns:
        lib = next(iter(adata.uns["spatial"]))
        img = adata.uns["spatial"][lib].get("images", {}).get("hires")

    diag = {"panel": panel, "n_zero": n_zero,
            "gt_counts": gt_counts, "knn_info": knn_info}
    return sample, adata, img, diag


# ─────────────────────────────────────────────────────────────────────────────
# Build the unified preprocessed AnnData
# ─────────────────────────────────────────────────────────────────────────────

def _print_groundtruth_diagnostic(samples, diags):
    """Log the per-slide ground_truth label distribution (before any change),
    then a compact KNN-imputation summary for slides that had UNASSIGNED spots.

    diags : {sample: {"gt_counts": dict|None, "knn_info": dict|None, ...}}
    """
    print("__groundtruth types diagnostic___")
    for s in samples:
        gt = diags[s]["gt_counts"]
        print(f"  {s} Types : {gt if gt is not None else '<no ground_truth column>'}")

    if any(diags[s]["knn_info"] for s in samples):
        print("__groundtruth KNN imputation (UNASSIGNED → majority vote)___")
        for s in samples:
            info = diags[s]["knn_info"]
            if not info:
                continue
            print(f"  {s}: imputed {info['n_unassigned']}/{info['n_total']} UNASSIGNED spots")
            for col, dist in info["after"].items():
                print(f"      {col} : {dist}")
    print("──────────────────────────────────────────────────────────────────")


def _print_panel_diagnostic(panels, zeros):
    """Log gene-panel overlap across slides + zero-count spots dropped.

    panels : {sample: list[gene]}   full panel of each slide (pre-alignment)
    zeros  : {sample: int}          zero-count spots dropped from each slide
    """
    sets = {s: set(g) for s, g in panels.items()}
    inter = set.intersection(*sets.values())
    union = set.union(*sets.values())
    n_inter, n_union = len(inter), len(union)
    drop = n_union - n_inter

    print("── panel diagnostic ──────────────────────────────────────────────")
    for s in panels:
        print(f"  {s:<14} {len(panels[s]):>6} genes   {zeros[s]:>5} zero-spots dropped")
    print(f"  intersection (inner keeps): {n_inter}")
    print(f"  union (all genes seen)    : {n_union}")
    pct = 100 * drop / n_union if n_union else 0.0
    print(f"  inner-join drops          : {drop} genes ({pct:.2f}% of union)")
    if drop == 0:
        print("  → panels identical; inner-join is a no-op.")
    print("──────────────────────────────────────────────────────────────────")


def build_unified_adata(
    samples,
    h5ad_paths,        # {sample: path}
    emb_paths,         # {sample: path}
    n_top_list,        # list[int | "all"] — one HVG selection saved per value
    batch_key="sample_id",
    target_sum=1e4,
    hvg_span=0.3,
    num_workers=1,
):
    """Return (adata, hvg_lists) where hvg_lists = {N: [gene names]}."""
    # ── parallel per-sample load (incl. per-slide CP10k+log1p) ───────────────
    work = [(s, h5ad_paths[s], emb_paths[s], target_sum) for s in samples]
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as ex:
        results = list(ex.map(_load_one, work))

    by_sample = {s: a   for s, a, _, _ in results}
    images    = {s: img for s, _, img, _ in results if img is not None}
    diags     = {s: d   for s, _, _, d in results}
    panels    = {s: diags[s]["panel"]  for s in samples}
    zeros     = {s: diags[s]["n_zero"] for s in samples}
    adatas    = [by_sample[s] for s in samples]   # preserve sample order

    # ── diagnostics (printed to logs as a sanity check) ──────────────────────
    _print_groundtruth_diagnostic(samples, diags)
    _print_panel_diagnostic(panels, zeros)

    # ── single concat(inner): aligns .X (log-CP10k) and the raw_counts layer ──
    #    to the shared gene set, in one consistent spot order. Normalization has
    #    already happened per slide, so the inner-join only decides which gene
    #    columns survive. It changes neither the normalization denominators nor
    #    the size-factor columns.
    adata = ad.concat(
        adatas, join="inner", label=None,
        index_unique="-", keys=samples,
    )
    adata.obs["sample_id"] = adata.obs["sample_id"].astype("category")
    if images:
        adata.uns["hires_images"] = images

    # A dataset can mix slides that had UNASSIGNED spots with fully-labeled ones.
    # Concat then leaves the fully-labeled slides' ground_truth_knn_* cells NaN.
    # Those slides need no imputation, so their own ground_truth fills the gap and
    # the columns stay complete. This is a no-op when every slide was imputed, and
    # when none was.
    import pandas as pd
    knn_cols = [c for c in adata.obs.columns if c.startswith("ground_truth_knn_")]
    if knn_cols and "ground_truth" in adata.obs.columns:
        gt = adata.obs["ground_truth"].astype(str)
        for c in knn_cols:
            missing = adata.obs[c].isna()           # boolean Series, obs-indexed
            if missing.any():
                filled = adata.obs[c].astype(str)   # NaN -> 'nan', overwritten below
                filled[missing] = gt[missing]       # index-aligned fill
                adata.obs[c] = pd.Categorical(filled)

    # ── HVG on the aligned RAW counts, per requested N ───────────────────────
    hvg_lists = {}
    for n in n_top_list:
        col = f"highly_variable_top{n}"
        if n == "all":
            adata.var[col] = True
            hvg_lists["all"] = adata.var_names.tolist()
            print(f"HVG topall: {len(hvg_lists['all'])} genes (all inner-join genes)")
        else:
            sc.pp.highly_variable_genes(
                adata, flavor="seurat_v3", n_top_genes=int(n),
                batch_key=batch_key, span=hvg_span, layer="raw_counts",
            )
            adata.var[col] = adata.var["highly_variable"].to_numpy()
            hvg_lists[int(n)] = adata.var_names[adata.var[col]].tolist()
            print(f"HVG top{n}: {len(hvg_lists[int(n)])} genes (batch_key={batch_key})")
    # drop the transient single-selection columns scanpy leaves behind
    adata.var.drop(
        columns=[c for c in ("highly_variable", "highly_variable_rank",
                             "highly_variable_nbatches", "means", "variances",
                             "variances_norm") if c in adata.var.columns],
        inplace=True, errors="ignore",
    )

    print(f"Unified adata: {adata.n_obs} spots × {adata.n_vars} genes, "
          f"emb dim {adata.obsm['X_emb'].shape[1]}")
    return adata, hvg_lists


# ─────────────────────────────────────────────────────────────────────────────
# Reader helpers: train.py / train_gcm.py get their arrays from here
# ─────────────────────────────────────────────────────────────────────────────

def read_arrays(adata, target_source="X", depth_scale=None):
    """Reconstruct (embeddings, y_all, sample_labels, gene_names) from adata.

    Parameters
    ----------
    target_source : "X", "raw_counts", "cp_counts" or "cp_counts_nr"
        Which expression matrix becomes the training target:
          - "X"          : log-CP10k expression (adata.X), for the Gaussian and MSE
                          models.
          - "raw_counts" : raw integer counts (adata.layers["raw_counts"]), for the
                          depth-blind Poisson / Negative Binomial family. The
                          likelihood is fitted directly on these counts, with no size
                          factor and no offset, so mu and alpha both live on the
                          raw-count scale.
          - "cp_counts"  : counts renormalized to one common sequencing depth T and
                          rounded to integers. Requires depth_scale = T/1e4.
          - "cp_counts_nr" : the same common-depth target without the rounding, so it
                          is continuous. Requires depth_scale.
        The preprocessed adata stores X and raw_counts; the two cp targets are derived
        from adata.X here, so switching target_source needs no re-preprocessing.
    depth_scale : float or None
        The common depth as a multiple of 1e4 (T/1e4), for the cp targets only.
        adata.X is log1p(CP10k), so expm1 recovers CP10k and the target is
        round(CP10k * depth_scale) = round(CP_T), or the continuous CP_T for
        cp_counts_nr. Every spot then sits at the same depth T, so nothing the model
        fits depends on a per-spot size factor. That is what keeps the predictive
        computable from H&E alone.

    The other three returns are model-independent: every spec uses the same spot
    embeddings (kaiko midnight ViT, 3072-dim), the same slide labels, and the same
    gene panel.
    """
    if target_source == "X":
        src = adata.X
    elif target_source == "raw_counts":
        if "raw_counts" not in adata.layers:
            raise KeyError(
                "target_source='raw_counts' requested but adata has no "
                "layers['raw_counts']. The preprocessing rule should have "
                "saved this; re-run preprocess for this dataset/radius."
            )
        src = adata.layers["raw_counts"]
    elif target_source == "cp_counts":
        if depth_scale is None:
            raise ValueError("target_source='cp_counts' requires depth_scale (= T/1e4).")
        X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
        src = np.rint(np.expm1(X) * float(depth_scale))   # round(CP10k * T/1e4) = round(CP_T)
    elif target_source == "cp_counts_nr":
        # Continuous common-depth target: CP_T = CP10k * T/1e4, no rounding. A discrete
        # Poisson/NB predictive is still fitted to it, which the losses permit
        # (nb_nll_loss passes validate_args=False, and Poisson full=False is already
        # continuous). Calibration then uses the smeared CDF (calibration_pit smear=True).
        if depth_scale is None:
            raise ValueError("target_source='cp_counts_nr' requires depth_scale (= T/1e4).")
        X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
        src = np.expm1(X) * float(depth_scale)         # CP_T continuous (no rounding)
    else:
        raise ValueError(
            f"target_source must be 'X', 'raw_counts', 'cp_counts' or 'cp_counts_nr', "
            f"got {target_source!r}"
        )
    y_all = np.asarray(src.toarray() if hasattr(src, "toarray") else src)
    embeddings = np.asarray(adata.obsm["X_emb"])
    sample_labels = adata.obs["sample_id"].astype(str).to_numpy()
    gene_names = adata.var_names.tolist()
    return embeddings, y_all, sample_labels, gene_names


# Cache: the merged adata's target matrix is identical for every fold / split /
# ensemble-member of one run (and any other run sharing the same preproc_id) --
# read it once per (path, target_source, depth_scale) rather than once per caller.
_TARGET_CACHE = {}


def target_values_for_predictions(adata_path, target_source, gene_names, adata_row,
                                  depth_scale=None, target_loc=None, target_scale=None):
    """Ground truth for a saved predictions slice that carries no `layers["y"]`.

    GCM `preds_fold{k}.h5ad` files hold test spots only and no `y` layer
    (snake/scripts/train_gcm.py::fold_predictions_adata), so ground truth is recovered
    offline from the shared preprocessed adata that run trained on
    (metrics.json["input_adata"]). The join is the one used for any other per-spot
    value: positional via `adata_row`, by gene name via `gene_names`.

    Parameters
    ----------
    adata_path    : path to the merged preprocessed adata.h5ad
                    (metrics.json["input_adata"]).
    target_source : "X" | "raw_counts" | "cp_counts" | "cp_counts_nr"
                    (spec.target_source); dispatched by read_arrays.
    gene_names    : predictions var_names, in the order columns should come back.
                    May be a reordered subset of the merged adata's genes.
    adata_row     : positional rows into adata_path (predictions obs["adata_row"]),
                    valid only against that exact adata.h5ad.
    depth_scale   : only for target_source="cp_counts"/"cp_counts_nr" (T/1e4); see
                    read_arrays.
    target_loc, target_scale : per-gene arrays aligned to `gene_names`. Pass the
                    predictions' var["target_loc"/"target_scale"] when a per-gene
                    target transform is active, so the returned `y` lands on the same
                    z-space scale as the saved `X`/`pred_var`, as the classic
                    pipeline's own `layers["y"]` does.

    Returns an (len(adata_row), len(gene_names)) float32 array.
    """
    key = (str(adata_path), target_source, depth_scale)
    if key not in _TARGET_CACHE:
        adata = ad.read_h5ad(adata_path)
        _, y_all, _, avail_genes = read_arrays(adata, target_source=target_source,
                                               depth_scale=depth_scale)
        _TARGET_CACHE[key] = (y_all, {g: i for i, g in enumerate(avail_genes)})
    y_all, gene_pos = _TARGET_CACHE[key]
    col = np.asarray([gene_pos[g] for g in gene_names], dtype=np.int64)
    row = np.asarray(adata_row, dtype=np.int64)
    y = y_all[np.ix_(row, col)].astype(np.float32)
    if target_loc is not None:
        y = ((y - np.asarray(target_loc, dtype=np.float32)[None, :])
             / np.asarray(target_scale, dtype=np.float32)[None, :])
    return y



def main():
    import argparse, json, sys, yaml

    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True,
                   help="dataset id; key into config['datasets']")
    p.add_argument("--radius", required=True)
    p.add_argument("--n-top", required=True,
                   help="comma-separated list of n_top_genes values, e.g. 2000,5000")
    p.add_argument("--config", required=True)
    p.add_argument("--out-adata", required=True)
    p.add_argument("--out-hvg-dir", required=True)
    p.add_argument("--out-config", required=True)
    a = p.parse_args()

    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    sys.path.insert(0, cfg["project_root"])  # this project_root's src/ wins over an editable install

    ds = cfg["datasets"][a.dataset]            # per-dataset paths
    with open(ds["samples_file"]) as f:
        samples = [ln.strip() for ln in f if ln.strip()]

    parent = ds["parent_dir"]
    h5ad_subdir = ds.get("h5ad_subdir", "")    # "" if h5ads sit directly in parent
    # Embedding directory, named per dataset in the config because a dataset extracted
    # more than once must say which extraction to read. {radius} is filled in here.
    emb_subdir = ds.get("emb_subdir", "embeddings_radius_factor_{radius}_swapxy_True/")
    emb_dir = f"{ds['emb_base_dir']}{emb_subdir.format(radius=a.radius)}"
    h5ad_paths = {s: f"{parent}{h5ad_subdir}{s}.h5ad" for s in samples}
    emb_paths = {s: f"{emb_dir}{s}.pt" for s in samples}
    n_top_list = ["all" if x.strip().lower() == "all" else int(x) for x in a.n_top.split(",") if x.strip()]
    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    batch_key = cfg.get("hvg_batch_key", "sample_id")

    print(f"Dataset '{a.dataset}': {len(samples)} samples, radius={a.radius}, "
          f"n_top={n_top_list}, workers={num_workers}")

    adata, hvg_lists = build_unified_adata(
        samples, h5ad_paths, emb_paths, n_top_list,
        batch_key=batch_key, num_workers=num_workers,
    )

    os.makedirs(os.path.dirname(a.out_adata), exist_ok=True)
    os.makedirs(a.out_hvg_dir, exist_ok=True)
    adata.write_h5ad(a.out_adata)

    for n, genes in hvg_lists.items():
        with open(os.path.join(a.out_hvg_dir, f"hvg_top{n}.txt"), "w") as f:
            f.write("\n".join(genes) + "\n")

    with open(a.out_config, "w") as f:
        json.dump({
            "dataset": a.dataset,
            "radius": a.radius, "n_top_list": n_top_list,
            "batch_key": batch_key, "n_samples": len(samples),
            "n_spots": int(adata.n_obs), "n_genes": int(adata.n_vars),
            "hvg_counts": {str(k): len(v) for k, v in hvg_lists.items()},
        }, f, indent=2)
    print(f"Wrote {a.out_adata} and {len(hvg_lists)} HVG list(s)")


if __name__ == "__main__":
    main()
