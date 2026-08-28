# src/evaluation/metrics_slide_aware.py
#
# Model-agnostic evaluation metrics, aggregated per slide. The CPU functions take
# numpy arrays. The on-device twins take torch tensors and return the same scalars
# for the per-epoch / early-stop path, with no scipy and no host round-trip.
#
# Correlation and calibration are computed per slide, then averaged. Pooling all
# slides into one statistic lets between-slide mean shifts masquerade as predictive
# correlation. Each spot's slide label arrives in `groups`.
#
# Averaging order, one definition for both selection and reporting:
#   per-(slide, gene) atom -> nanmean over slides (per gene) -> nanmean over genes
# A NaN atom means the prediction or the target is near-constant on that slide.
# NaN atoms are dropped from the means, never counted as 0. The full atom table is
# returned, so the NaN pairs stay visible and an offline bootstrap can re-aggregate.
#
# Shape convention: everything is 2-D (n_spots, n_genes) internally. A single gene
# is n_genes == 1, so there is no separate 1-D code path.

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr, rankdata

# Native-CDF PIT calibration math, the single source of truth in numpy and torch,
# shared with the offline analysis. The family-specific CDF bounds (F_lo, F_hi) come
# from the caller: count_metrics_final passes native Poisson / NB bounds, and
# compute_metrics falls back to Gaussian bounds when none are given. This module only
# aggregates the coverage curve they imply, slide by slide.
from src.evaluation.calibration_pit import (
    CONF_LEVELS,
    gaussian_pit_bounds, pit_calibration_np,
    gaussian_pit_bounds_t, poisson_pit_bounds_t, nb_pit_bounds_t,
    coverage_curve_t, scalars_from_coverage_t,
)


# ─────────────────────────────────────────────────────────────────────────────
# Correlation registry
# ─────────────────────────────────────────────────────────────────────────────
# Add an entry here to make a new correlation metric available to the pipeline;
# callers select it with `corr_metrics=(...)`. Each function takes two 1-D numpy
# arrays and returns a float, NaN-safe. A new entry also needs a matching branch in
# corr_scalars_gpu, or that on-device twin raises KeyError for the name.

def _safe_pearson(a, b, eps=1e-12):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or b.size < 2:
        return float("nan")
    if np.std(a) < eps or np.std(b) < eps:
        return float("nan")
    r = pearsonr(a, b)[0]
    return float(r) if np.isfinite(r) else float("nan")


def _safe_spearman(a, b, eps=1e-12):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or b.size < 2:
        return float("nan")
    if np.std(a) < eps or np.std(b) < eps:
        return float("nan")
    r = spearmanr(a, b).correlation
    return float(r) if np.isfinite(r) else float("nan")


CORR_METRICS = {
    "pearson":  _safe_pearson,
    "spearman": _safe_spearman,
}


def _safe_partial_spearman(a, b, c, eps=1e-12):
    """Spearman partial correlation of a and b, controlling for c. Used as
    Spearman(pred_std, |resid|) with the predicted mean partialled out, so it asks
    whether uncertainty tracks error at a fixed predicted level. It is the companion
    of var_residual_corr, which correlates pred_var with resid² without controlling
    for the mean.

    Partial correlation on ranks: rank all three, then
        r_partial = (r_ab − r_ac·r_bc) / sqrt((1−r_ac²)(1−r_bc²))
    with each r a Pearson on the ranks. NaN if any input is near-constant, since the
    rank correlation is then undefined, or if the denominator collapses. Needs n >= 3,
    because controlling for one variable costs a degree of freedom."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float); c = np.asarray(c, dtype=float)
    if a.size < 3:
        return float("nan")
    if np.std(a) < eps or np.std(b) < eps or np.std(c) < eps:
        return float("nan")
    ra, rb, rc = rankdata(a), rankdata(b), rankdata(c)
    rab, rac, rbc = _safe_pearson(ra, rb), _safe_pearson(ra, rc), _safe_pearson(rb, rc)
    denom = np.sqrt((1.0 - rac ** 2) * (1.0 - rbc ** 2))
    if not np.isfinite(denom) or denom < eps:
        return float("nan")
    r = (rab - rac * rbc) / denom
    return float(r) if np.isfinite(r) else float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Small shape / averaging helpers
# ─────────────────────────────────────────────────────────────────────────────

def _nanmean_finite(values):
    """Mean over finite values only; NaN if none are finite."""
    finite = [v for v in np.asarray(values, dtype=float).ravel() if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def _as_2d(arr):
    """(n_spots,) -> (n_spots, 1); (n_spots, n_genes) unchanged."""
    arr = np.asarray(arr)
    return arr[:, None] if arr.ndim == 1 else arr


def _slide_order(groups):
    """Unique slide labels in first-appearance order (stable, reproducible)."""
    groups = np.asarray(groups)
    _, first_idx = np.unique(groups, return_index=True)
    return groups[np.sort(first_idx)]


# ─────────────────────────────────────────────────────────────────────────────
# Per-(slide, gene) correlation atoms — generic over the correlation metric
# ─────────────────────────────────────────────────────────────────────────────

def per_slide_gene_corr(means, y, groups, corr_fn, key):
    """Correlation atoms for every (slide, gene). Returns:
        (atoms, slides, n_genes)
    where each atom is a dict {slide, gene, <key>, n_spots} and <key> holds the
    correlation value. NaN atoms are KEPT (they are the NaN record); the
    aggregator filters them out via _nanmean_finite.

    Storing the value under <key> (e.g. "pearson", "spearman") rather than a
    generic name keeps the saved atoms.parquet column self-describing.
    """
    means, y = _as_2d(means), _as_2d(y)
    groups = np.asarray(groups)
    slides = _slide_order(groups)
    n_genes = means.shape[1]

    atoms = []
    for s in slides:
        m = groups == s
        ms, ys = means[m], y[m]
        n_spots = int(m.sum())
        for g in range(n_genes):
            atoms.append({
                "slide":   str(s),
                "gene":    g,
                key:       corr_fn(ms[:, g], ys[:, g]),
                "n_spots": n_spots,
            })
    return atoms, list(map(str, slides)), n_genes


def _aggregate_corr(atoms, n_genes, key):
    """atoms -> (scalar, per_gene list) using the module's averaging order:
    per-(slide,gene) -> nanmean over slides (per gene) -> nanmean over genes.

    One pass buckets each atom under its gene, then nanmeans each bucket, so the cost
    is O(n_atoms + n_genes). Scanning the whole atom list once per gene instead is
    quadratic in the gene count and does not scale to an all-genes run. Each bucket
    keeps its atoms in slide order."""
    buckets = [[] for _ in range(n_genes)]
    for a in atoms:
        buckets[a["gene"]].append(a[key])
    per_gene = [_nanmean_finite(vals) for vals in buckets]
    scalar = _nanmean_finite(per_gene)
    return scalar, per_gene


# ─────────────────────────────────────────────────────────────────────────────
# Per-(slide, gene) magnitude error (MAE / MSE), complement to the correlation.
# ─────────────────────────────────────────────────────────────────────────────
# Pearson is scale-free: a perfectly-shaped but mis-scaled prediction still scores 1.
# MAE and MSE add the magnitude axis. They ride the same (means, y) arrays the
# per-gene correlation consumes, so they sit on the axis the correlation is reported
# on: g* against log1p(y/depth_scale) for the count specs, the X target otherwise
# (log1p-CP10k, or z-space under the target transform). That is a different quantity
# from run_inference's per-epoch `mse` diagnostic, which stays on the model's own
# target scale. Aggregation is slide-aware, per-(slide,gene) -> nanmean over slides
# -> per gene, and the headline is the median over genes, because a mean lets
# high-expression genes dominate.
# Compare these numbers within one likelihood family only. A count model's log1p value
# arrives through the g* projection, so its absolute error carries an asymmetry the
# Gaussian does not have. Cross-family comparison uses the correlation and the PIT
# scalars.

def per_slide_gene_error(means, y, groups):
    """MAE + MSE atoms for every (slide, gene), on the given (means, y).
    Returns (atoms, n_genes); each atom is {slide, gene, mae, mse}. NaN where a
    (slide, gene) slice has no finite residual."""
    means, y = _as_2d(means), _as_2d(y)
    groups = np.asarray(groups)
    slides = _slide_order(groups)
    n_genes = means.shape[1]
    atoms = []
    for s in slides:
        m = groups == s
        diff = means[m] - y[m]
        for g in range(n_genes):
            d = diff[:, g]
            d = d[np.isfinite(d)]
            atoms.append({
                "slide": str(s), "gene": g,
                "mae": float(np.mean(np.abs(d))) if d.size else float("nan"),
                "mse": float(np.mean(d ** 2))    if d.size else float("nan"),
            })
    return atoms, n_genes


def _aggregate_error(atoms, n_genes):
    """atoms -> {mae_median, mse_median, mae_per_gene, mse_per_gene}.
    per-(slide,gene) -> nanmean over slides (per gene) -> nanMEDIAN over genes."""
    mae_buckets = [[] for _ in range(n_genes)]
    mse_buckets = [[] for _ in range(n_genes)]
    for a in atoms:
        mae_buckets[a["gene"]].append(a["mae"])
        mse_buckets[a["gene"]].append(a["mse"])
    mae_per_gene = [_nanmean_finite(v) for v in mae_buckets]
    mse_per_gene = [_nanmean_finite(v) for v in mse_buckets]
    _med = lambda arr: (float(np.nanmedian(arr))
                        if np.any(np.isfinite(arr)) else float("nan"))
    return {
        "mae_median":   _med(mae_per_gene),
        "mse_median":   _med(mse_per_gene),
        "mae_per_gene": mae_per_gene,
        "mse_per_gene": mse_per_gene,
    }


# ─────────────────────────────────────────────────────────────────────────────
# On-device (GPU) correlation, twin of the per-(slide,gene) corr above.
# ─────────────────────────────────────────────────────────────────────────────
# The per-epoch / early-stop path uses this, via run_inference(metrics_on_gpu=True),
# so the training hot loop gets its selection metric without moving predictions to the
# host or calling scipy. It returns only the aggregated scalars, one per metric. The
# per-(slide,gene) atoms written to atoms.parquet come from the CPU functions above.
#
# Same averaging order and NaN gate as the CPU path:
#   per-(slide,gene) atom -> nanmean over slides -> nanmean over genes,
#   atom = NaN when a slide has < 2 spots or pred/target is near-constant (std < eps).
# Operands stay float32, the GPU's fast dtype. Upcasting the (n_spots, n_genes) matrix
# to float64 exhausts GPU memory at the all-genes scale. The per-column sums accumulate
# in float64 instead (sum(dtype=float64) in _pearson_cols), a memory-bound reduction
# with a tiny (G,) output, so the result stays within ~1e-9 of the all-float64 scipy
# value. These scalars feed logging and early stopping only.

def _as_2d_t(t):
    """(n,) -> (n, 1); (n, G) unchanged."""
    return t if t.ndim == 2 else t.unsqueeze(1)


def _std_cols(v):
    """Population std of each column of v (n, G) along dim 0 -> (G,)."""
    vc = v - v.mean(dim=0, keepdim=True)
    return vc.pow(2).mean(dim=0).sqrt()


def _pearson_cols(x, y, eps):
    """Pearson r of each column of x and y (both (n, G)) along dim 0 -> (G,).
    Centered two-pass form, the algorithm scipy uses. NaN where a column of x or y is
    near-constant (population std < eps) or the result is non-finite.

    x and y stay in the caller's dtype, float32. Upcasting the (n, G) operand to
    float64 exhausts GPU memory at the all-genes scale. Only the column reductions
    accumulate in float64 (sum(dtype=torch.float64)), a memory-bound reduction with a
    tiny (G,) output, so r stays within ~1e-9 of the all-float64 value at no extra
    memory."""
    xc = x - x.mean(dim=0, keepdim=True)
    yc = y - y.mean(dim=0, keepdim=True)
    ssx = xc.pow(2).sum(dim=0, dtype=torch.float64)
    ssy = yc.pow(2).sum(dim=0, dtype=torch.float64)
    r = (xc * yc).sum(dim=0, dtype=torch.float64) / (ssx.sqrt() * ssy.sqrt())
    n = x.shape[0]
    bad = ((ssx / n).sqrt() < eps) | ((ssy / n).sqrt() < eps) | ~torch.isfinite(r)
    return torch.where(bad, torch.full_like(r, float("nan")), r)


def _rank_cols(v):
    """Ordinal ranks along dim 0, per column (double argsort). Ties get ordinal ranks,
    not scipy's average ranks. Predictions and targets are effectively tie-free
    continuous data, so only an exact tie makes the two differ."""
    return v.argsort(dim=0).argsort(dim=0).to(v.dtype)


def _nanmean_t(t, dim):
    """Mean over finite entries along `dim`; NaN where a slice has none finite.
    Matches the CPU _nanmean_finite (NaNs excluded, never counted as 0)."""
    finite = torch.isfinite(t)
    count = finite.sum(dim=dim)
    summ = torch.where(finite, t, torch.zeros_like(t)).sum(dim=dim)
    return torch.where(count > 0, summ / count, torch.full_like(summ, float("nan")))


def _slide_row_indices(group_ids, n_groups, device):
    """Per-slide row-index tensors, built from one host copy of `group_ids`.

    The per-slide loops below need each slide's rows. A boolean mask
    (`means[group_ids == s]`) has a data-dependent output size, so CUDA must
    synchronize to size it, once per slide per call. Copying group_ids to the host a
    single time and precomputing each slide's ascending row indices lets the loop
    gather with `index_select`, whose output size is the index length and is known on
    the host, so no sync happens. The rows come out in the same ascending order a
    boolean mask gives, so every downstream per-slide sum is bit-identical."""
    gids = group_ids.detach().to("cpu").numpy()
    return [torch.as_tensor(np.nonzero(gids == s)[0], dtype=torch.long, device=device)
            for s in range(n_groups)]


def corr_scalars_gpu(means, y, group_ids, n_groups,
                     corr_metrics=("pearson",), eps=1e-12):
    """Slide-aware correlation SCALARS, computed entirely on the data device.

    GPU twin of the corr part of compute_metrics, for the per-epoch / early-stop
    metric. Returns {name: float} — one aggregated scalar per metric, using the
    same averaging order (per-(slide,gene) -> nanmean over slides -> nanmean over
    genes) and NaN gate as the CPU path.

    Parameters
    ----------
    means, y   : (n_spots,) or (n_spots, n_genes) float tensors on the device.
    group_ids  : (n_spots,) long tensor — slide id in [0, n_groups) per spot.
    n_groups   : number of distinct slides.
    corr_metrics : same names as the CPU CORR_METRICS registry ('pearson',
                   'spearman'). Add a GPU branch here when you add a CPU metric.
    """
    means = _as_2d_t(means).float()
    y = _as_2d_t(y).float()
    n_genes = means.shape[1]

    # Per-(slide, gene) atoms per metric; NaN where an atom is undefined.
    atoms = {name: torch.full((n_groups, n_genes), float("nan"),
                              device=means.device, dtype=torch.float64)
             for name in corr_metrics}
    slide_rows = _slide_row_indices(group_ids, n_groups, means.device)
    for s in range(n_groups):
        rows = slide_rows[s]
        if rows.numel() < 2:
            continue                       # < 2 spots -> NaN atom (left as-is)
        ms, ys = means.index_select(0, rows), y.index_select(0, rows)
        for name in corr_metrics:
            if name == "pearson":
                atoms[name][s] = _pearson_cols(ms, ys, eps)
            elif name == "spearman":
                # scipy gates spearman on the RAW values' std, then ranks.
                gate = (_std_cols(ms) >= eps) & (_std_cols(ys) >= eps)
                r = _pearson_cols(_rank_cols(ms), _rank_cols(ys), eps)
                atoms[name][s] = torch.where(gate, r, torch.full_like(r, float("nan")))
            else:
                raise KeyError(
                    f"corr_scalars_gpu has no on-device implementation for "
                    f"'{name}'. Add one here (mirrors CORR_METRICS)."
                )

    # nanmean over slides (per gene), then nanmean over genes.
    return {name: float(_nanmean_t(_nanmean_t(atoms[name], dim=0), dim=0))
            for name in corr_metrics}


# ─────────────────────────────────────────────────────────────────────────────
# Per-slide calibration: native-CDF, one-sided PIT coverage, cross-family.
# ─────────────────────────────────────────────────────────────────────────────
# Each probabilistic model is scored with its own native CDF (Gaussian / Poisson / NB)
# via the analytic expected one-sided PIT coverage in calibration_pit, which holds the
# math and the discrete-jump handling. This module only aggregates it slide by slide.
# The caller passes the precomputed per-(spot,gene) bounds
# (F_lo, F_hi) = (P(Y<y), P(Y≤y)): compute_metrics fills in Gaussian ones for the
# X-target models, count_metrics_final passes native Poisson/NB ones. PIT is invariant
# under monotone reparameterization, so a count model on its count scale is directly
# comparable to a Gaussian on log1p-CP10k or z-score.
#
# Three scalars per (slide, gene), all reductions of the same coverage curve:
# ece_pit (magnitude |coverage−α|), pit_bias (= ½−E[PIT], a location statistic and not
# a confidence one), disp_signed (over/under-confidence, the sign(½−α)-weighted error).
# Plus the uncertainty-error pair: var_residual_corr (marginal
# Pearson(pred_var, resid²)) and var_residual_corr_partial (within-gene partial
# Spearman, see _safe_partial_spearman).
#
# _CONF re-exports the confidence grid from calibration_pit, so the CPU and the GPU
# paths score on the same levels.

_CONF = CONF_LEVELS


def per_slide_gene_calibration(means, y, pred_var, F_lo, F_hi, groups):
    """Per-(slide, gene) PIT calibration + uncertainty-error atoms.

    means / y / pred_var drive the uncertainty-error correlations, on whatever scale
    the caller uses: the X target for Gaussian models, the model's own fitted target
    scale for count models. F_lo / F_hi are the native-CDF PIT bounds, same shape, and
    drive the coverage curve and the three calibration scalars.
    """
    means, y, pred_var = _as_2d(means), _as_2d(y), _as_2d(pred_var)
    F_lo, F_hi = _as_2d(F_lo), _as_2d(F_hi)
    groups = np.asarray(groups)
    slides = _slide_order(groups)
    n_genes = means.shape[1]

    atoms = []
    for s in slides:
        m = groups == s
        ms, ys, pvs = means[m], y[m], pred_var[m]
        flo, fhi = F_lo[m], F_hi[m]
        std, resid = np.sqrt(pvs), ys - ms
        for g in range(n_genes):
            cov, sc = pit_calibration_np(flo[:, g], fhi[:, g], _CONF)
            atoms.append({
                "slide": str(s), "gene": g,
                "ece_pit":     sc["ece_pit"],
                "pit_bias":    sc["pit_bias"],
                "disp_signed": sc["disp_signed"],
                "var_residual_corr":         _safe_pearson(pvs[:, g], resid[:, g] ** 2),
                "var_residual_corr_partial": _safe_partial_spearman(
                    std[:, g], np.abs(resid[:, g]), ms[:, g]),
                "coverage": cov,
            })
    return atoms, list(map(str, slides)), n_genes


# Calibration scalar keys (every one is aggregated + emitted per-gene). Coverage
# curve is handled separately (it is a vector, not a scalar).
_CALIB_SCALAR_KEYS = (
    "ece_pit", "pit_bias", "disp_signed",
    "var_residual_corr", "var_residual_corr_partial",
)


def _aggregate_calibration(atoms, n_genes):
    """atoms -> {key, key_per_gene for each scalar} + avg coverage curve.

    One-pass bucketing (same as _aggregate_corr): each gene's bucket keeps the
    atoms in append (slide) order, so every nanmean / coverage average is the
    design's per-(slide,gene) -> nanmean over slides -> nanmean over genes."""
    buckets = [[] for _ in range(n_genes)]
    for a in atoms:
        buckets[a["gene"]].append(a)
    per_gene = {k: [] for k in _CALIB_SCALAR_KEYS}
    cov_per_gene = []
    for g_atoms in buckets:
        for k in _CALIB_SCALAR_KEYS:
            per_gene[k].append(_nanmean_finite([a[k] for a in g_atoms]))
        curves = [a["coverage"] for a in g_atoms if np.all(np.isfinite(a["coverage"]))]
        cov_per_gene.append(np.mean(curves, axis=0) if curves else np.full_like(_CONF, np.nan))
    avg_coverage = np.nanmean(np.stack(cov_per_gene), axis=0)

    out = {"empirical_coverage": avg_coverage, "confidence_levels": _CONF}
    for k in _CALIB_SCALAR_KEYS:
        out[k] = _nanmean_finite(per_gene[k])
        out[f"{k}_per_gene"] = per_gene[k]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# On-device (GPU) calibration, twin of the calibration above.
# ─────────────────────────────────────────────────────────────────────────────
# Supplies the per-epoch / early-stop path with the calibration scalars the training
# loop logs (ece_pit, pit_bias, disp_signed, var_residual_corr,
# var_residual_corr_partial), on the data device with no scipy and no host round-trip.
# Same native-CDF one-sided PIT coverage and averaging order as the CPU path. The
# per-(slide,gene) atoms and the per-gene / coverage arrays saved offline come from the
# CPU functions. Bounds are built per slice inside the loop, in float32, never as a
# full (N, G) float64 matrix, which exhausts GPU memory at the all-genes scale. The
# Gaussian bounds come from (means, std, y); count specs pass `count_pit_inputs`, the
# mean and dispersion on the fitted target scale, and the native Poisson/NB bounds are
# built per slice.

def calibration_scalars_gpu(means, y, pred_var, group_ids, n_groups, eps=1e-12,
                            is_count_spec=False, count_pit_inputs=None, smear=False):
    """Slide-aware calibration scalars on the data device (per-epoch / early-stop).

    Returns {ece_pit, pit_bias, disp_signed, var_residual_corr,
    var_residual_corr_partial}. The 3 PIT scalars are the native-CDF one-sided PIT
    coverage, built per slice in float32, never as a full (N, G) float64 matrix, which
    exhausts GPU memory at the all-genes scale:
      • Gaussian (is_count_spec=False): erf bounds from (means, std, y).
      • Count (is_count_spec=True) with count_pit_inputs=(mu_count, alpha_or_None,
        y_raw): native Poisson (alpha None) or NB (alpha given) bounds, taken as given
        on the model's own fitted target scale (mu_c = mu_count, no rescaling).
      • Count without count_pit_inputs (None): PIT is skipped and left NaN. It is
        computed at final inference instead, on CPU, native and exact, and that is the
        value that gets saved. Poisson always passes its inputs, since gammaincc is
        cheap; only NB is gated off on the train pass, where the NB cumsum is too
        heavy. See run_inference's `nb_pit`.
    var_residual_corr is a float32 Pearson(pred_var, resid²), both families, every
    pass. var_residual_corr_partial is final-inference only, because its argsort ranks
    are too heavy for the hot loop, so it is NaN here.

    means, y, pred_var : (n_spots,) or (n_spots, n_genes) tensors on the caller's
                         scale: the X target for Gaussian, the fitted target scale for
                         count. They drive var_residual_corr and the Gaussian PIT.
    count_pit_inputs   : (mu_count, alpha_or_None, y_raw) for native count PIT this
                         pass, or None to skip it. Gaussian ignores it. mu_count is the
                         predictive mean already on the scale the likelihood was fitted
                         on, and this function applies no factor to it. That keeps the
                         mean and the dispersion it is scored against on one scale.
    group_ids          : (n_spots,) long tensor, slide id in [0, n_groups).
    """
    means    = _as_2d_t(means).float()
    y        = _as_2d_t(y).float()
    pred_var = _as_2d_t(pred_var).float()
    n_genes  = means.shape[1]

    # PIT computed this pass? Gaussian: always. Count: only if inputs were supplied.
    do_pit = (not is_count_spec) or (count_pit_inputs is not None)
    mu_count = alpha = y_raw = None
    if is_count_spec and count_pit_inputs is not None:
        mu_count, alpha, y_raw = count_pit_inputs
        mu_count, y_raw = _as_2d_t(mu_count).float(), _as_2d_t(y_raw).float()
        if alpha is not None:
            alpha = _as_2d_t(alpha).float()
    conf = torch.as_tensor(_CONF, device=means.device, dtype=torch.float32)   # (C,)

    ece_atoms  = torch.full((n_groups, n_genes), float("nan"), device=means.device, dtype=torch.float64)
    bias_atoms = torch.full_like(ece_atoms, float("nan"))
    disp_atoms = torch.full_like(ece_atoms, float("nan"))
    vrc_atoms  = torch.full_like(ece_atoms, float("nan"))
    vrcp_atoms = torch.full_like(ece_atoms, float("nan"))   # partial: final-only
    slide_rows = _slide_row_indices(group_ids, n_groups, means.device)
    for s in range(n_groups):
        rows = slide_rows[s]
        ms, ys, pvs = (means.index_select(0, rows), y.index_select(0, rows),
                       pred_var.index_select(0, rows))
        if do_pit:
            # Native PIT bounds for THIS slide only (peak (n_s, G) float32);
            # coverage_curve_t loops the C levels internally (no (C, n_s, G) cube).
            if not is_count_spec:
                flo, fhi = gaussian_pit_bounds_t(ms, pvs.sqrt(), ys)        # erf
            else:
                # mu arrives on the fitted target scale, so nothing is rescaled here.
                mu_c = mu_count.index_select(0, rows)
                yr = y_raw.index_select(0, rows)
                if alpha is not None:
                    flo, fhi = nb_pit_bounds_t(mu_c, alpha.index_select(0, rows), yr, smear=smear)
                else:
                    flo, fhi = poisson_pit_bounds_t(mu_c, yr, smear=smear)
            cov = coverage_curve_t(flo.float(), fhi.float(), conf)          # (C, G) float32
            sc = scalars_from_coverage_t(cov, conf)
            ece_atoms[s], bias_atoms[s], disp_atoms[s] = sc["ece_pit"], sc["pit_bias"], sc["disp_signed"]
        # var_residual_corr stays float32. A homoscedastic model has constant variance,
        # and in float32 that column can slip past the near-constant gate and read ~0
        # here where the CPU path returns NaN. The saved value is the CPU one.
        # Partial Spearman is final-inference only.
        resid = ys - ms
        vrc_atoms[s] = _pearson_cols(pvs, resid ** 2, eps)     # (G,) float32

    agg = lambda t: float(_nanmean_t(_nanmean_t(t, dim=0), dim=0))
    return {
        "ece_pit":                   agg(ece_atoms),
        "pit_bias":                  agg(bias_atoms),
        "disp_signed":               agg(disp_atoms),
        "var_residual_corr":         agg(vrc_atoms),
        "var_residual_corr_partial": agg(vrcp_atoms),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(means, y, groups,
                    pred_var=None,
                    corr_metrics=("pearson",), compute_calibration=True,
                    pit_bounds=None):
    """Slide-aware metrics.

    Parameters
    ----------
    means, y     : (n_spots,) or (n_spots, n_genes)
    groups       : (n_spots,) slide label per spot — REQUIRED.
    pred_var     : same shape as means, or None (deterministic models).
                   The PREDICTIVE VARIANCE itself (already exp'd from log_var
                   if the model produces log_var). For non-Gaussian probabilistic
                   models, pred_var is whatever the model's variance is (mean
                   for Poisson, mean + mean^2/theta for NB, etc.).
    corr_metrics : iterable of correlation-metric names from CORR_METRICS. Pass ()
                   to skip the scipy correlation loop entirely, as the count
                   calibration-only call in count_metrics_final does.
    compute_calibration : when False, skip the calibration block even if pred_var is
                   given. The calibration keys stay None; `pred_var` is still
                   returned.
    pit_bounds   : (F_lo, F_hi) native-CDF PIT bounds (same shape as means), or
                   None. None -> Gaussian bounds from (means, sqrt(pred_var), y),
                   correct for the X-target models. Count callers pass native
                   Poisson/NB bounds (count_metrics_final).

    Returns (dict). For each name in corr_metrics, keys
        <name>, <name>_per_gene, <name>_per_slide_gene
    are populated. When pred_var is provided AND
    compute_calibration is True, calibration keys (ece_pit, pit_bias, disp_signed,
    var_residual_corr, var_residual_corr_partial, + _per_gene, empirical_coverage,
    confidence_levels) are populated too (native-CDF PIT; see calibration block).
    """
    # ── Correlation metrics ────────────────────────────────────────────────
    result = {
        # Magnitude-error slots, filled in below alongside the correlation.
        # They stay None when corr_metrics is empty.
        "mae_median":                 None,
        "mse_median":                 None,
        "mae_per_gene":               None,
        "mse_per_gene":               None,
        "error_per_slide_gene":       None,
        # Calibration slots — filled in below if pred_var is provided.
        "pred_var":                   pred_var,
        "ece_pit":                    None,
        "ece_pit_per_gene":           None,
        "pit_bias":                   None,
        "pit_bias_per_gene":          None,
        "disp_signed":                None,
        "disp_signed_per_gene":       None,
        "var_residual_corr":          None,
        "var_residual_corr_per_gene": None,
        "var_residual_corr_partial":          None,
        "var_residual_corr_partial_per_gene": None,
        "empirical_coverage":         None,
        "confidence_levels":          None,
        "calibration_per_slide_gene": None,
    }

    # n_genes is a property of the data, not of which corr metrics are computed, so
    # derive it up front. That lets calibration run when corr_metrics is empty, which
    # is the case for the count calibration-only call in count_metrics_final.
    n_genes = _as_2d(means).shape[1]
    result["n_genes"] = n_genes
    n_total_primary = None
    n_finite_primary = None
    for i, name in enumerate(corr_metrics):
        if name not in CORR_METRICS:
            raise KeyError(
                f"Unknown corr_metric '{name}'. Known: {sorted(CORR_METRICS)}"
            )
        atoms, slides, _ = per_slide_gene_corr(
            means, y, groups, CORR_METRICS[name], key=name,
        )
        scalar, per_gene = _aggregate_corr(atoms, n_genes, key=name)
        result[name]                       = scalar
        result[f"{name}_per_gene"]         = per_gene
        result[f"{name}_per_slide_gene"]   = atoms

        if i == 0:
            # The first metric in the tuple is the primary one. Its atom counts go out
            # under fixed pearson-named keys below, whatever that metric actually is.
            n_total_primary = len(atoms)
            n_finite_primary = sum(1 for a in atoms if np.isfinite(a[name]))
            result["slides"]   = slides

    # Field names train.py reads when it assembles metrics.json.
    result["n_total_pearson"]  = n_total_primary
    result["n_finite_pearson"] = n_finite_primary

    # ── Magnitude error (MAE / MSE) ─────────────────────────────────────────
    # Rides the same (means, y) as the correlation, so it lands on the same axis the
    # correlation is reported on: count specs call compute_metrics with g* against
    # log1p(y/depth_scale). Gating on corr_metrics keeps it off the count
    # calibration-only call, whose arrays are on the model's own linear target scale
    # and would put a different quantity under the same key.
    if corr_metrics:
        e_atoms, _ = per_slide_gene_error(means, y, groups)
        result["error_per_slide_gene"] = e_atoms
        result.update(_aggregate_error(e_atoms, n_genes))

    # ── Calibration ────────────────────────────────────────────────────────
    if pred_var is not None and compute_calibration:
        if pit_bounds is None:
            # Gaussian default: native PIT from the model's own (mean, std). Correct
            # for every X-target model; count callers pass native bounds instead.
            std = np.sqrt(_as_2d(pred_var))
            F_lo, F_hi = gaussian_pit_bounds(_as_2d(means), std, _as_2d(y))
        else:
            F_lo, F_hi = pit_bounds
        c_atoms, _, _ = per_slide_gene_calibration(means, y, pred_var, F_lo, F_hi, groups)
        result["calibration_per_slide_gene"] = c_atoms
        result.update(_aggregate_calibration(c_atoms, n_genes))

    return result


