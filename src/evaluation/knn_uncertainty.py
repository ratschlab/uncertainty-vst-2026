"""kNN local uncertainty signals: non-parametric, untrained, complementary to the MLP.

For a query spot, find its k nearest neighbors (by embedding) among spots the model never saw for
that donor, and read off their true expression as a non-parametric estimate of what a "typical"
spot near this one looks like. The embedding is the frozen H&E patch embedding, the MLP's input
and never its learned features, so these signals are computed independently of anything the MLP
learned.

Three signals:
  1. pi_hat            - weighted fraction of the k neighbors with y > 0 for a gene (presence).
  2. mu_knn / sigma_knn - weighted mean/std of neighbor values (log1p(CP10k)), in two variants:
                          "all neighbors" (zeros included; channel 1, regression quality against
                          the ground truth) and "positives only" (channel 2). Never mix zeros into
                          the positives-only variant: it would then measure presence rather than
                          dispersion, the very confound this channel exists to avoid.
  3. d_mean / d_k       - mean / k-th neighbor distance, one per spot, shared across all its genes.

Raw d_mean is elevated both by image corruption and by belonging to an under-represented tissue
type, and the OOD corruption and cross-dataset tests (same severity, same spots) cannot tell those
two causes apart on their own. `cluster_centroids_and_radii` + `assign_nearest_centroid` normalize
d_mean by the local scale of the query's nearest reference cluster (radius = how spread out that
cluster's own members already are), which isolates the corruption-specific part from type rarity.

Vocabulary discipline: never call these "epistemic" or "aleatoric", inside this module or out. They
are loose analogies to that literature (SNGP, DDU, Deep k-NN), not a proven Bayesian decomposition.
Say "empirical local dispersion" (sigma_knn) and "density/coverage score" (d_mean/d_k).

Pure numpy/pandas functions, no config, adata or file I/O here. The caller
(snake/scripts/epistemic/knn_uncertainty_precompute.py) owns all I/O: it reads the baseline run's
splits.json for the LOPO donor partition, builds the reference pool by excluding every slide of the
held-out donor (excluding only one of them leaves a sister slide of the same donor in the pool,
which shows up as an artificially close neighbor), and gene-chunks the calls into knn_signal_grid
to bound memory at the all-genes scale.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

K_MAX = 100
K_GRID = (5, 10, 25, 50, 100)
MIN_POS_NEIGHBORS = 5   # canal 2 gate: NaN sigma_knn_pos below this many positive neighbors


# --------------------------------------------------------------------------- embedding space
def l2_normalize(x, eps=1e-12):
    """Row-wise L2 normalization. Euclidean distance on unit-norm rows is a monotone function of
    cosine similarity (d^2 = 2 - 2*cos_sim), so an exact-Euclidean NearestNeighbors index on these rows
    IS a cosine-similarity index, without sklearn's separate (slower) cosine metric path."""
    x = np.asarray(x, dtype=np.float64)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norm, eps, None)


def build_reference_index(ref_embeddings, transform=None, algorithm="auto"):
    """Fit an exact NearestNeighbors index on the (L2-normalized) reference-pool embeddings.

    ref_embeddings : (n_ref, emb_dim) array, the frozen H&E patch embedding (adata.obsm["X_emb"]),
                     already restricted by the caller to spots outside the held-out donor's slides.
                     This function does no donor or slide logic; it only ever sees the array it is
                     handed. Every returned neighbor index is a row position into exactly this
                     array, so the caller must keep its own mapping back to global adata_row /
                     obs_name.
    transform      : optional callable applied to ref_embeddings before L2-normalizing, e.g. a
                     fitted PCA's .transform. None means raw L2.
    algorithm      : passed straight to sklearn's NearestNeighbors; the search is exact whatever
                     the value.

    Returns a fitted sklearn.neighbors.NearestNeighbors (metric="euclidean", exact search).
    """
    ref_embeddings = np.asarray(ref_embeddings, dtype=np.float64)
    if ref_embeddings.ndim != 2 or ref_embeddings.shape[0] == 0:
        raise ValueError(
            f"build_reference_index: expected a non-empty 2-D array, got shape {ref_embeddings.shape}."
        )
    if transform is not None:
        ref_embeddings = transform(ref_embeddings)
    index = NearestNeighbors(metric="euclidean", algorithm=algorithm)
    index.fit(l2_normalize(ref_embeddings))
    return index


def query_neighbors(index, query_embeddings, k_max=K_MAX, transform=None):
    """Query the k_max nearest reference-pool neighbors (+ distances) for a set of query spots.

    index            : a NearestNeighbors returned by build_reference_index.
    query_embeddings : (n_query, emb_dim) array, same embedding space as the index was built on.
    k_max            : neighbors to retrieve. K_MAX=100 by default so every k in K_GRID is a prefix
                       slice of this one query, never a fresh query per k.
    transform        : must match whatever was passed to build_reference_index.

    Returns (distances, indices), each (n_query, k_max), sorted ascending by distance (sklearn's
    kneighbors guarantee). `indices` are row positions into the array build_reference_index was
    called with, not global adata_row; map back via the caller's own reference-pool row-id array.
    """
    query_embeddings = np.asarray(query_embeddings, dtype=np.float64)
    if query_embeddings.ndim != 2 or query_embeddings.shape[0] == 0:
        raise ValueError(
            f"query_neighbors: expected a non-empty 2-D array, got shape {query_embeddings.shape}."
        )
    n_ref = index.n_samples_fit_
    if k_max > n_ref:
        raise ValueError(f"query_neighbors: k_max ({k_max}) exceeds the reference pool size ({n_ref}).")
    if transform is not None:
        query_embeddings = transform(query_embeddings)
    return index.kneighbors(l2_normalize(query_embeddings), n_neighbors=k_max)


# --------------------------------------------------------------------------- weighting
def neighbor_weights(distances, tau=1.0, kernel="exp"):
    """Per-neighbor weight from distance. distances: any-shape array (typically (n_query, k)).

    kernel="exp" (default): exp(-d_i^2 / tau^2). kernel="uniform": all ones (plain average over the k
    neighbors). tau is on the SAME scale as `distances`; since embeddings are L2-normalized, Euclidean
    distances lie in [0, 2] (d^2 = 2 - 2*cos_sim), so tau ~ 0.1-1.0 is the range to sweep. The
    default below is 1.0; the launcher sets it explicitly for a run that reports a value.

    Weights are NOT row-normalized (the caller's weighted mean/std divides by the weight sum itself),
    so this can be computed ONCE on the full k_max-wide distance array and sliced per k in K_GRID.
    """
    distances = np.asarray(distances, dtype=np.float64)
    if kernel == "uniform":
        return np.ones_like(distances)
    if kernel == "exp":
        if tau is None or tau <= 0:
            raise ValueError("neighbor_weights: kernel='exp' requires tau > 0.")
        return np.exp(-(distances ** 2) / (tau ** 2))
    raise ValueError(f"neighbor_weights: unknown kernel '{kernel}' (expected 'exp' or 'uniform').")


def _weighted_stats(values, weights, axis=1):
    """Weighted mean + population std of `values` along `axis`, weighted by `weights` (broadcast-
    compatible with `values`). NaN wherever the weight sum is <= 0 (no admitted neighbors — e.g. every
    neighbor is a structural zero in the positives-only variant, or a kernel-weight underflow)."""
    w_sum = weights.sum(axis=axis, keepdims=True)
    safe_w_sum = np.where(w_sum > 0, w_sum, np.nan)
    mean = (weights * values).sum(axis=axis, keepdims=True) / safe_w_sum
    var = (weights * (values - mean) ** 2).sum(axis=axis, keepdims=True) / safe_w_sum
    return np.squeeze(mean, axis=axis), np.sqrt(np.clip(np.squeeze(var, axis=axis), 0.0, None))


# --------------------------------------------------------------------------- the 3 signals (gene-chunked)
def knn_signal_grid(neighbor_idx, neighbor_dist, y_ref_chunk, k_grid=K_GRID, tau=1.0, kernel="exp",
                    min_pos_neighbors=MIN_POS_NEIGHBORS):
    """pi_hat / mu_knn / sigma_knn for the whole k-grid, for ONE gene chunk, vectorized (no python loop
    over spots or genes — only over the (short) k_grid).

    neighbor_idx, neighbor_dist : (n_query, k_max) arrays from query_neighbors (k_max >= max(k_grid)).
    y_ref_chunk       : (n_ref, n_genes_chunk) array of log1p(CP10k) values (adata.X) for the
                        reference pool, restricted to this gene chunk. n_ref must match the pool
                        neighbor_idx indexes into. The caller chunks the full gene panel and calls
                        this once per chunk; there is no chunking inside, so the chunk width is what
                        bounds the (n_query, k_max, n_genes_chunk) intermediate.
    k_grid            : k values to report, all <= neighbor_idx.shape[1]; each is a prefix slice of
                        the k_max-cached, distance-sorted neighbors, so no kNN query is repeated.
    tau, kernel       : see neighbor_weights.
    min_pos_neighbors : NaN-gate for the positives-only variant (mu_knn_pos/sigma_knn_pos), keyed on
                        the raw unweighted count of positive neighbors among the k. This is a local
                        per-(spot, gene, k) gate, distinct from the global per-(gene, slide)
                        detection-count regime used elsewhere in the project.

    Returns {k: {"pi_hat", "mu_knn_all", "sigma_knn_all", "mu_knn_pos", "sigma_knn_pos",
    "n_pos_neighbors"}} for each k in k_grid, every array shaped (n_query, n_genes_chunk). A high
    NaN rate in the positives-only variant is a property of the cohort's zero rate, not a bug;
    n_pos_neighbors is returned so the caller can report that rate by (gene, k).
    """
    neighbor_idx = np.asarray(neighbor_idx)
    neighbor_dist = np.asarray(neighbor_dist, dtype=np.float64)
    y_ref_chunk = np.asarray(y_ref_chunk, dtype=np.float64)
    if neighbor_idx.shape != neighbor_dist.shape:
        raise ValueError(
            f"knn_signal_grid: neighbor_idx {neighbor_idx.shape} and neighbor_dist "
            f"{neighbor_dist.shape} must have the same shape."
        )
    n_ref = y_ref_chunk.shape[0]
    if neighbor_idx.max() >= n_ref:
        raise ValueError(
            f"knn_signal_grid: neighbor_idx reaches row {int(neighbor_idx.max())} but y_ref_chunk only "
            f"has {n_ref} rows — pass the SAME reference pool (row-aligned) here."
        )
    k_max = neighbor_idx.shape[1]
    k_grid = sorted(set(int(k) for k in k_grid))
    if k_grid[-1] > k_max:
        raise ValueError(f"knn_signal_grid: k_grid max ({k_grid[-1]}) exceeds cached k_max ({k_max}).")

    w = neighbor_weights(neighbor_dist, tau=tau, kernel=kernel)      # (n_query, k_max) — computed ONCE
    y_gathered = y_ref_chunk[neighbor_idx]                           # (n_query, k_max, n_genes_chunk)
    present = y_gathered > 0

    out = {}
    for k in k_grid:
        w_k = w[:, :k, None]
        y_k = y_gathered[:, :k, :]
        pres_k = present[:, :k, :]

        pi_hat, _ = _weighted_stats(pres_k.astype(w_k.dtype), w_k)
        mu_knn_all, sigma_knn_all = _weighted_stats(y_k, w_k)

        n_pos_neighbors = pres_k.sum(axis=1)
        mu_knn_pos, sigma_knn_pos = _weighted_stats(y_k, w_k * pres_k)
        gate = n_pos_neighbors < min_pos_neighbors
        mu_knn_pos = np.where(gate, np.nan, mu_knn_pos)
        sigma_knn_pos = np.where(gate, np.nan, sigma_knn_pos)

        out[k] = {
            "pi_hat": pi_hat,
            "mu_knn_all": mu_knn_all, "sigma_knn_all": sigma_knn_all,
            "mu_knn_pos": mu_knn_pos, "sigma_knn_pos": sigma_knn_pos,
            "n_pos_neighbors": n_pos_neighbors,
        }
    return out


def spot_distance_signals(neighbor_dist, k_grid=K_GRID):
    """d_mean / d_k for the whole k-grid, ONE PER SPOT (gene-independent — shared across all genes of
    that spot, unlike knn_signal_grid). d_mean = mean distance to the first k neighbors; d_k = distance
    to the k-th (last) of those k neighbors.

    neighbor_dist : (n_query, k_max) array from query_neighbors, ascending-sorted per row.
    Returns {k: {"d_mean": (n_query,), "d_k": (n_query,)}}.
    """
    neighbor_dist = np.asarray(neighbor_dist, dtype=np.float64)
    k_max = neighbor_dist.shape[1]
    k_grid = sorted(set(int(k) for k in k_grid))
    if k_grid[-1] > k_max:
        raise ValueError(f"spot_distance_signals: k_grid max ({k_grid[-1]}) exceeds cached k_max ({k_max}).")
    return {k: {"d_mean": neighbor_dist[:, :k].mean(axis=1), "d_k": neighbor_dist[:, k - 1]}
            for k in k_grid}


# --------------------------------------------------------------------------- raw-vs-z-score diagnostic
def inter_intra_slide_variance(X_train, slide_labels_train, gene_names=None):
    """Per-gene inter-slide vs intra-slide variance, on train spots only. It answers whether pooling
    raw un-z-scored neighbor values across slides is defensible, or whether per-(gene, slide)
    z-scoring is needed before pooling. No kNN involved.

    X_train            : (n_train_spots, n_genes) log1p(CP10k) matrix (adata.X), TRAIN spots of the
                          reference pool only.
    slide_labels_train : (n_train_spots,) slide label per row of X_train (same order).
    gene_names          : optional (n_genes,) array for the returned index; else a plain integer range.

    Per gene: inter_var = population variance (ddof=0) of the per-slide MEANS (each slide one unit,
    UNWEIGHTED by its spot count — "does this gene's level itself shift between slides"); intra_var =
    spot-count-weighted (pooled) average of each slide's own within-slide population variance. ratio =
    inter_var / intra_var (NaN where intra_var <= 0). Small ratio for most genes => raw pooling is fine;
    large ratio for a sizeable fraction => flag for a future per-(gene, slide) z-scored pass.

    Returns a DataFrame indexed by gene with columns [inter_var, intra_var, ratio, n_slides].
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    slides = np.asarray(slide_labels_train)
    if X_train.shape[0] != slides.shape[0]:
        raise ValueError(
            f"inter_intra_slide_variance: X_train has {X_train.shape[0]} rows but "
            f"slide_labels_train has {slides.shape[0]}."
        )
    unique_slides, slide_idx = np.unique(slides, return_inverse=True)
    n_slides, n_genes = unique_slides.size, X_train.shape[1]

    sums = np.zeros((n_slides, n_genes))
    sums_sq = np.zeros((n_slides, n_genes))
    counts = np.zeros(n_slides)
    np.add.at(sums, slide_idx, X_train)
    np.add.at(sums_sq, slide_idx, X_train ** 2)
    np.add.at(counts, slide_idx, 1)

    slide_means = sums / counts[:, None]
    slide_var = np.clip(sums_sq / counts[:, None] - slide_means ** 2, 0.0, None)   # per-slide, ddof=0

    intra_var = (counts[:, None] * slide_var).sum(axis=0) / counts.sum()           # pooled within-slide
    inter_var = slide_means.var(axis=0, ddof=0) if n_slides >= 2 else np.full(n_genes, np.nan)
    ratio = np.where(intra_var > 0, inter_var / intra_var, np.nan)

    index = (pd.Index(gene_names, name="gene") if gene_names is not None
             else pd.RangeIndex(n_genes, name="gene_idx"))
    return pd.DataFrame(
        {"inter_var": inter_var, "intra_var": intra_var, "ratio": ratio, "n_slides": n_slides},
        index=index,
    )


# --------------------------------------------------------------------------- d_mean type-rarity de-confound
def leiden_cluster_labels(ref_embeddings, resolution=1.0, n_neighbors=15, seed=0):
    """Unsupervised cluster labels for the reference pool: Leiden on a kNN graph of the embeddings.

    The supervised counterpart is the histological annotation itself, handed straight to
    cluster_centroids_and_radii. Two reasons this one exists next to it:
      * LUNG_CANCER carries no ground_truth column at all, so the supervised d_norm cannot be built
        there; an unsupervised partition is the only way that cohort gets a normalized distance.
      * On the cohorts that do have labels, running both is a control: if d_norm behaves the same
        either way, the de-confounding is a property of the local density and not of the particular
        annotation used to define the groups.

    Leiden takes no k. Granularity is set by `resolution` instead (higher gives more, smaller
    communities); 1.0 is scanpy's default. Consequence to keep in mind: the number of clusters, and
    the meaning of a cluster id, differ from fold to fold, because each fold's reference pool
    excludes a different donor. That is harmless for d_norm, whose centroids and radii are per-fold
    anyway, but an unsupervised `cluster_label` must never be pooled or compared across folds the
    way a tissue name can be.

    Embeddings are L2-normalized first, so the graph's Euclidean metric is cosine, the same
    convention as build_reference_index and cluster_centroids_and_radii. Every distance in this
    module then lives on one geometry.

    Magnitude caveat: this partition is fitted on the very embedding space d_mean is measured in, so
    its clusters are the most compact of any scheme by construction, its radii the smallest, and
    d_norm_leiden systematically the largest. Only the within-scheme ranking or trend is comparable
    across cohorts, never the magnitude of d_norm_leiden against that of d_norm_gt.

    anndata and scanpy are imported inside the function: the rest of this module is
    numpy/pandas/sklearn only, and a caller that never asks for the unsupervised scheme should not
    have to install the heavier dependency.

    Returns (labels, n_clusters): labels is an (n_ref,) array of strings ("0", "1", ...), the shape
    cluster_centroids_and_radii already accepts for a tissue-name array.
    """
    import anndata as _ad
    import scanpy as _sc

    a = _ad.AnnData(X=l2_normalize(np.asarray(ref_embeddings, dtype=np.float64)))
    _sc.pp.neighbors(a, n_neighbors=n_neighbors, use_rep="X", random_state=seed)
    # flavor="igraph" + n_iterations=2 + directed=False is scanpy's recommended call and replaces
    # the deprecated legacy flavor. random_state makes the partition reproducible, which matters
    # because these labels feed a radius that divides a reported quantity. The backend needs the
    # python-igraph package (not leidenalg) and scanpy >= 1.10, so a missing package, or a scanpy
    # whose signature has no `directed`, falls back to the leidenalg backend with a printed reason
    # rather than aborting a job whose other cluster schemes are fine.
    try:
        _sc.tl.leiden(a, resolution=resolution, key_added="leiden", flavor="igraph",
                      n_iterations=2, directed=False, random_state=seed)
    except (TypeError, ValueError, ImportError) as e:
        print(f"leiden_cluster_labels: igraph flavor unavailable ({type(e).__name__}: {e}) -- "
              f"falling back to the legacy leidenalg backend.", flush=True)
        _sc.tl.leiden(a, resolution=resolution, key_added="leiden", random_state=seed)
    labels = a.obs["leiden"].to_numpy().astype(str)
    return labels, int(pd.unique(labels).size)


def cluster_centroids_and_radii(ref_embeddings, cluster_labels, min_cluster_size=30):
    """L2-normalized centroid + shrunk radius per cluster of the reference pool.

    Raw d_mean conflates two causes of "far from neighbors": an under-represented type (naturally
    sparse region, clean image) and image corruption (embedding pushed out of every region). A
    cluster's radius, how spread out its own members are around their centroid, is the local scale
    that separates them. Dividing a query's d_mean by its assigned cluster's radius keeps a clean
    member of a naturally sparse cluster low, while a corrupted embedding lands far from whatever
    cluster it is nearest to and still stands out.

    d_norm is a relative index, not a ratio calibrated to 1. Numerator and denominator are different
    kinds of distance: d_mean is nearest-neighbor spacing, a density statistic over the k closest
    reference points, while the radius is member spread around a centroid. Spacing is typically one
    to two orders of magnitude smaller than spread, so d_norm sits well below 1 and only its
    ranking, or its trend under corruption, is meaningful. Dividing by the radius removes the spread
    component of type rarity, not the count component: halving a cluster's membership at fixed
    spread raises spacing but leaves the radius unchanged, so the de-confounding is partial by
    construction.

    ref_embeddings : (n_ref, emb_dim) array, SAME rows/order as the pool build_reference_index is
                     called on (raw, not yet L2-normalized -- normalized internally here).
    cluster_labels : (n_ref,) array-like, one label per reference-pool row. Any hashable label
                     works: a ground_truth tissue-type string, or a Leiden cluster id.
    min_cluster_size : clusters with fewer members than this shrink their radius toward the pooled
                     (all-reference, membership-weighted) radius -- a raw radius from a handful of
                     points is too noisy to trust (same shrinkage idea as the n+ regime bucketing).

    Returns (centroids, radii, labels_order, counts): centroids (n_clusters, emb_dim) L2-normalized;
    radii (n_clusters,) already shrunk; labels_order (n_clusters,) the label each row corresponds to;
    counts (n_clusters,) raw membership count (before shrinkage) -- a size proxy for "how rare is this
    type", used by callers building the common/rare 2x2 validation split.
    """
    ref_embeddings = l2_normalize(np.asarray(ref_embeddings, dtype=np.float64))
    labels = np.asarray(cluster_labels)
    if ref_embeddings.shape[0] != labels.shape[0]:
        raise ValueError(
            f"cluster_centroids_and_radii: ref_embeddings has {ref_embeddings.shape[0]} rows but "
            f"cluster_labels has {labels.shape[0]}."
        )
    labels_order, label_idx = np.unique(labels, return_inverse=True)
    n_clusters = labels_order.size

    sums = np.zeros((n_clusters, ref_embeddings.shape[1]))
    counts = np.zeros(n_clusters)
    np.add.at(sums, label_idx, ref_embeddings)
    np.add.at(counts, label_idx, 1)
    centroids = l2_normalize(sums / counts[:, None])

    dist_to_own_centroid = np.linalg.norm(ref_embeddings - centroids[label_idx], axis=1)
    radius_sums = np.zeros(n_clusters)
    np.add.at(radius_sums, label_idx, dist_to_own_centroid)
    raw_radii = radius_sums / counts

    global_radius = float(dist_to_own_centroid.mean())   # pooled over ALL points, not a mean-of-clusters
    m = float(min_cluster_size)
    radii = (counts * raw_radii + m * global_radius) / (counts + m)   # shrink small clusters toward global

    return centroids, radii, labels_order, counts


def assign_nearest_centroid(query_embeddings, centroids):
    """Nearest-centroid cluster assignment (row index into `centroids`) for arbitrary query
    embeddings. L2-normalizes internally (matching cluster_centroids_and_radii / build_reference_index
    convention), so nearest-by-Euclidean-distance == highest-cosine-similarity -- computed as a single
    small (n_query, n_clusters) dot product, never an (n_query, n_clusters, emb_dim) intermediate.

    Pure embedding-space assignment: never needs the query's own label, so it works identically for
    in-distribution, corrupted, or cross-dataset queries -- exactly the point (a corrupted or foreign
    spot still gets assigned to SOME nearest cluster, whose radius is then the yardstick it's judged
    against).

    query_embeddings : (n_query, emb_dim) array, same embedding space as `centroids`.
    centroids        : (n_clusters, emb_dim) array, already L2-normalized (as returned by
                       cluster_centroids_and_radii -- NOT re-normalized here, so passing raw
                       embeddings by mistake would silently give wrong assignments).

    Returns (n_query,) int array, row index into `centroids`.
    """
    query_embeddings = l2_normalize(np.asarray(query_embeddings, dtype=np.float64))
    centroids = np.asarray(centroids, dtype=np.float64)
    if query_embeddings.shape[1] != centroids.shape[1]:
        raise ValueError(
            f"assign_nearest_centroid: query embeddings have dim {query_embeddings.shape[1]} but "
            f"centroids have dim {centroids.shape[1]}."
        )
    cos_sim = query_embeddings @ centroids.T   # (n_query, n_clusters); both L2-normalized rows
    return np.argmax(cos_sim, axis=1)
