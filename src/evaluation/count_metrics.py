# src/evaluation/count_metrics.py
#
# Count-model (Poisson / NB) metric code, shared by both pipelines so the g* headline,
# the option-A cross-check and the calibration have one implementation.
#
# SCALES. There is no size factor and no exposure offset here. Every count model fits its
# likelihood directly on its own target, so `means` is mu on that target scale and `alpha`
# is the dispersion fitted at the same scale. The module's only constant is `depth_scale`,
# one float per run, and it has exactly one role: the divisor applied before log1p when
# reporting the mean correlation.
#     *_cp / *_cp_nr   depth_scale = count_common_depth / 1e4, so the reported axis is
#                      log1p(CP10k) and is comparable to the Gaussian models.
#     *_raw            depth_scale = None, since raw counts are aligned to no depth, so
#                      the reported axis is log1p(raw counts) and compares only to other _raw.
# Everything else, the PIT, pred_var, var_residual_corr and the residuals, stays on the
# model's own target scale and uses no constant at all.
#
# Three entry points:
#   count_expected_log1p   g* = E[log1p(Y/depth_scale)], the Jensen-exact estimand
#   count_corr_inputs      the per-epoch option-A tensors: log1p(mu/depth_scale),
#                          log1p(y/depth_scale), and raw y
#   count_metrics_final    the final metric dict, holding both log1p estimators: g* under
#                          {metric} and the mean-only option A under {metric}_logmean

import numpy as np
import torch

from src.evaluation.metrics_slide_aware import compute_metrics
from src.evaluation.calibration_pit import poisson_pit_bounds, nb_pit_bounds


def count_expected_log1p(mu, depth_scale, alpha=None, device="cpu",
                         gene_chunk=512, k_cap=500000,
                         eps=1e-12, work_dtype=None, return_var=False):
    """
    B estimator as opposed to the mean-only estimator A. 
    g* = E[log1p(Y / depth_scale)] under the per-(spot,gene) predictive count
    distribution.

    The count correlation should compare the SAME quantity the Gaussian predicts —
    the MEAN of log1p(CP10k) — not log1p of the mean. `log1p(mu/depth_scale)` is the
    transform-OF-the-mean; the Gaussian's prediction is `E[log1p(CP10k)]`, the
    mean-OF-the-transform. log1p is concave, so the two differ by a (>= 0) Jensen
    gap. This evaluates the mean-of-the-transform directly from the fitted
    distribution:

        g*[i,j] = E[ log1p(Y / d) ] = sum_{k>=0} log1p(k / d) * P(Y = k | mu_ij, alpha_ij)

    with d = depth_scale, so corr(g*, log1p(y/d)) is the count analogue of the
    Gaussian's corr(E[log1p(CP10k)], log1p(CP10k)) — apples to apples for the cp
    models. mu and the divisor are separate arguments: mu stays on the model's own
    fitted target scale and depth_scale only moves the REPORTING axis.

    The predictive distribution is the SAME object the training loss used
    (losses.poisson_nll_loss / nb_nll_loss), so g* is consistent with what was
    optimized:
        alpha is None : Poisson(rate = mu)
        otherwise     : NegativeBinomial(total_count = 1/alpha,
                                         logits = log mu + log alpha)

    NUMERICS — mode-centerd, residual-adaptive windowed recurrence. The sum is
    evaluated without recomputing P(Y=k) from scratch at each k (a per-k
    log_prob / lgamma was the wall-clock cost). Consecutive probabilities have a
    closed ratio — Poisson P(k)/P(k-1) = mu/k; NB (k-1+r)/k * p with r=1/alpha and
    p = alpha*mu/(1+alpha*mu) — so each step is ONE multiply. We start at the mode
    (~round(mu)), where P is O(1/std) and so never underflows (starting at
    P(0)=e^{-mu} would underflow for large mu and the recurrence would then multiply
    zeros), evaluate that one anchor with a single log_prob, then march OUTWARD in
    both directions until the running term drops below `eps` (past the mode the PMF
    is monotone, so a negligible leading term bounds the whole remaining tail). The
    window is therefore centerd on the mass, so there is no chopped upper tail (the
    residual-mass warning is gone) and only as wide as the distribution needs —
    cheap for Poisson, automatically wider for an overdispersed (right-skewed) NB.
    g* is self-normalized by the captured mass to absorb any eps-level remainder.

    PRECISION. The marching tensors run in `work_dtype` (default: float32 on CUDA —
    the Ampere/Turing FP64 path is ~1/32 speed — float64 on CPU), while g* and the
    captured mass ACCUMULATE in float64 (the matrix-float32 / reduce-float64 pattern
    used by the metric twins). The ~1e-6 float32 effect on g* is far below the
    experiment's noise (the output feeds a correlation).

    MEMORY. Genes are processed in chunks of `gene_chunk` columns and k is a LOOP, so
    the largest tensor is (n_spots, gene_chunk) — the number of terms never becomes a
    stored axis, so capturing the full mass costs no extra RAM. Scales to all-genes.

    Parameters
    ----------
    mu     : (N,) or (N, G) predicted count means on the model's own fitted target
             scale (== the `means` saved to predictions).
    depth_scale : scalar divisor applied inside log1p (see the module header), or
             None for the depth-blind raw models (treated as 1.0). Broadcasts, so an
             array is accepted too.
    alpha  : (N,) / (N, G) NB dispersion, fitted at the SAME scale as mu, or None
             for Poisson.
    device : where to run the (vectorized) sum; pass the data device for speed.
    eps    : term-magnitude floor that stops each outward march (residual ~ eps).
    k_cap  : hard cap on the march length, a runaway guard rather than the normal cut,
             which is the eps floor above. The march is anchored at the mode, so the
             support covered is [0, mode + k_cap]. The cap only binds when the eps break
             never fires, which needs roughly alpha*mu >= 4e4: a highly expressed gene
             with a large free dispersion, most plausible on the *_raw family. The result
             is then a truncated conditional mean, and it is visible, since residual_mass
             exceeds 1e-4 and the callers print a warning.
    work_dtype : marching dtype; None -> float32 on CUDA else float64.
    return_var : if True, ALSO accumulate the 2nd moment E[log1p(Y/d)^2] in the same
                 windowed recurrence and return Var[log1p(Y/d)] = E[l^2] - E[l]^2 (the
                 count predictive variance on the reporting axis, i.e. log1p(CP10k)
                 for the cp families and log1p(raw counts) when depth_scale is None).
                 Intended as the per-draw ALEATORIC variance for a post-hoc epistemic
                 decomposition on the reporting axis. NO caller uses it today; it is kept
                 because the recurrence that produces it is the same one g* already runs,
                 so the variance is free here. Default False -> 2-tuple return, so the
                 live callers (count_metrics_final, assess_gstar_truncation) are unchanged.

    Returns
    -------
    gstar         : np.ndarray, same shape as `mu`.
    var_log1p     : np.ndarray (ONLY if return_var=True) — Var[log1p(Y/d)] under the
                    same predictive distribution.
    residual_mass : float, max over cells of |1 - sum_k P(Y=k)| (now ~eps by design).
    """
    if work_dtype is None:
        work_dtype = torch.float32 if "cuda" in str(device) else torch.float64
    mu_t = torch.as_tensor(np.asarray(mu), dtype=torch.float64, device=device)
    orig_shape = mu_t.shape
    if mu_t.ndim == 1:
        mu_t = mu_t.unsqueeze(1)
    N, G = mu_t.shape
    # depth_scale=None (the raw models) means "aligned to no depth" -> divisor 1.0.
    div = 1.0 if depth_scale is None else depth_scale
    div_t = torch.as_tensor(np.asarray(div, dtype=float).ravel(), dtype=torch.float64,
                            device=device).unsqueeze(1)              # (1, 1) or (N, 1)
    a_t = None
    if alpha is not None:
        a_t = torch.as_tensor(np.asarray(alpha), dtype=torch.float64, device=device)
        if a_t.ndim == 1:
            a_t = a_t.unsqueeze(1)
        if a_t.shape[1] == 1 and G > 1:
            # The gene loop below indexes alpha with an explicit column PERMUTATION, which
            # needs a real gene axis -- a (N, 1) broadcast column would be an out-of-range
            # index. expand() is a view, so this costs nothing. In production alpha is
            # already (N, G) (the saved dispersion layer); this only covers a per-spot
            # alpha passed in 1-D, which broadcasting alone would let through.
            a_t = a_t.expand(N, G)

    gstar = torch.empty((N, G), dtype=torch.float64, device=device)
    var2  = torch.empty((N, G), dtype=torch.float64, device=device) if return_var else None
    residual = 0.0

    # GENE ORDER, a cost fix rather than a correctness one. Both marches below break on a
    # chunk-wide maximum, so the single widest (spot, gene) cell sets the iteration count
    # every cell of that chunk pays. Panel order is uncorrelated with the predicted spread,
    # so most chunks end up paying close to the global maximum. Sorting by the predicted
    # variance mu*(1+alpha*mu), which is monotone in both drivers of the window width, lets
    # each chunk pay only for its own genes. Ascending, so the expensive chunk is last.
    #
    # This is exact: reordering only changes how long an already-converged cell keeps being
    # multiplied by a negligible ratio. Every cell still exits with P < eps, and g* is
    # self-normalized by the captured mass. The width is itself computed chunked, so it
    # never materialises a second (N, G) float64 array.
    width_g = torch.empty(G, dtype=torch.float64, device=device)
    for c0 in range(0, G, gene_chunk):
        c1 = min(c0 + gene_chunk, G)
        m_w = mu_t[:, c0:c1]
        width_g[c0:c1] = (m_w if a_t is None
                          else m_w * (1.0 + a_t[:, c0:c1] * m_w)).amax(0)
    order = torch.argsort(width_g)

    for c0 in range(0, G, gene_chunk):
        c1 = min(c0 + gene_chunk, G)
        cols = order[c0:c1]                                          # this chunk's columns
        mu_c = mu_t[:, cols]                                         # (N, Gc) f64
        poisson = a_t is None
        if poisson:
            dist = torch.distributions.Poisson(rate=mu_c)
        else:
            a_c = a_t[:, cols]
            r_c = 1.0 / a_c                                          # NB total_count
            p_c = (a_c * mu_c) / (1.0 + a_c * mu_c)                  # NB success prob
            dist = torch.distributions.NegativeBinomial(
                total_count=r_c, logits=torch.log(mu_c) + torch.log(a_c))

        # Anchor at the mode of the ACTUAL distribution; P there is O(1/std), so it never
        # underflows (which is the whole reason the anchor exists — starting at P(0) would
        # underflow for large mu). round(mu) is the POISSON mode, and using it for the NB
        # too would sit far out in the NB's right tail, since the NB mode here is mu*(1-alpha),
        # i.e. 0 whenever alpha >= 1. The down leg would then march TOWARD the mass instead of
        # away from it and its eps break could never fire: P(0) = (1+alpha*mu)^(-1/alpha) is
        # ~4e-6 at alpha=0.5, mu=1000, six orders above eps, so every NB chunk would run its
        # full max(round(mu)) bound. The Poisson has no such problem: there P(0) = exp(-mu)
        # drops under eps as soon as mu > 27.6, the break fires and the window collapses.
        k0 = torch.clamp(torch.round(mu_c) if poisson
                         else torch.floor(mu_c * (1.0 - a_c)), min=0.0)   # (N, Gc) f64
        P0 = dist.log_prob(k0).exp()                                 # one log_prob eval
        l0 = torch.log1p(k0 / div_t)
        g_c    = l0 * P0                                             # f64 accumulators
        g2_c   = (l0 * l0 * P0) if return_var else None             # E[l^2] (2nd moment)
        mass_c = P0.clone()

        # The marching operands are in work_dtype: matrix in float32, reductions in
        # float64. Keeping k, the step ratio and the log1p in float32 avoids the slow FP64
        # path on CUDA, while the accumulators stay float64, which is what keeps the sum
        # accurate even though the terms are not.
        #
        # Accepted precision cost: at large alpha*mu the NB ratio p = alpha*mu/(1+alpha*mu)
        # approaches 1, and float32 loses relative precision in (1-p), which biases the
        # tail decay over a long march. The width-ordered chunking above shortens the
        # march, so the two effects work against each other.
        mu_w  = mu_c.to(work_dtype)
        div_w = div_t.to(work_dtype)
        k0_w  = k0.to(work_dtype)
        r_w   = None if poisson else r_c.to(work_dtype)
        p_w   = None if poisson else p_c.to(work_dtype)

        # March DOWN toward k=0: P(k-1) = P(k) * (k / mu)   [Poisson]
        #                               = P(k) * k / ((k-1+r) p)   [NB]
        P = P0.to(work_dtype)
        k = k0_w.clone()
        for d in range(int(k0.max().item())):
            k_next = k - 1.0
            kc = k_next.clamp(min=0.0)                               # avoid k<0 in the ratio
            inv_ratio = (kc + 1.0) / mu_w if poisson else (kc + 1.0) / ((kc + r_w) * p_w)
            P = P * inv_ratio * (k_next >= 0.0).to(work_dtype)
            l = torch.log1p(kc / div_w)
            lP = (l * P).double()
            g_c    += lP
            if return_var:
                g2_c += l.double() * lP                              # l^2 * P
            mass_c += P.double()
            k = k_next
            if d % 16 == 0 and float(P.max()) < eps:
                break

        # March UP: P(k+1) = P(k) * (mu / (k+1))   [Poisson]
        #                  = P(k) * (k+r)/(k+1) p   [NB]   (here written at the new k)
        P = P0.to(work_dtype)
        k = k0_w.clone()
        for d in range(k_cap):
            k = k + 1.0
            ratio = mu_w / k if poisson else (k - 1.0 + r_w) / k * p_w
            P = P * ratio
            l = torch.log1p(k / div_w)
            lP = (l * P).double()
            g_c    += lP
            if return_var:
                g2_c += l.double() * lP
            mass_c += P.double()
            if d % 16 == 0 and float(P.max()) < eps:
                break

        # Scatter back through the permutation (cols), NOT c0:c1 — the loop walks genes in
        # width order, so the output column of a chunk is not its position in the loop.
        gstar[:, cols] = g_c / mass_c                               # self-normalize
        if return_var:
            mean_c = g_c / mass_c
            var2[:, cols] = g2_c / mass_c - mean_c * mean_c         # E[l^2] - E[l]^2
        residual = max(residual, float((1.0 - mass_c).abs().max()))

    g_out = gstar.cpu().numpy().reshape(orig_shape)
    if return_var:
        v_out = var2.clamp(min=0.0).cpu().numpy().reshape(orig_shape)
        return g_out, v_out, residual
    return g_out, residual


def count_corr_inputs(means_t, y_t, depth_scale):
    """Per-epoch (option A) count metric tensors, shared by both pipelines.

    `depth_scale` is the run's single reporting constant (float, or None for the
    depth-blind raw models). Both sides of the correlation are divided by it, so the
    cp models are read on the log1p(CP10k) axis and the raw models on the log1p(raw)
    axis. Do NOT drop the division on the means side: log1p(m/d) and log1p(m) are not
    affine-equivalent, so on 10x_TuPro (d = 2.5) that would silently move early
    stopping and the {metric}_logmean cross-check to a different axis and change the
    SELECTED EPOCH.

    Returns (corr_means, corr_y, y_for_diag):
        corr_means = log1p(means_t / d)   predicted, on the reporting axis
        corr_y     = log1p(y_t / d)       observed, on the reporting axis
        y_for_diag = y_t                  the target on ITS OWN scale (for the MSE
                                          diag + the variance-residual correlation,
                                          which must stay coherent with pred_var)
    """
    div = 1.0 if depth_scale is None else float(depth_scale)
    return torch.log1p(means_t / div), torch.log1p(y_t / div), y_t


def count_pred_var(means, dispersion=None):
    """Count-model predictive variance on the model's OWN fitted target scale, recomputed
    from the saved X=means and (NB) the "dispersion" layer. Mean and dispersion are
    always saved at the same scale, so this needs no constant:
        Poisson : var = mu
        NB      : var = mu + alpha*mu^2
    pred_var is NOT persisted for count models (it is exactly this deterministic function
    of X [+ dispersion]), so offline loaders recompute it here instead of reading a layer.
    numpy in/out; broadcasts over (N, G). dispersion=None -> Poisson."""
    m = np.asarray(means, dtype=float)
    if dispersion is None:
        return m
    return m + np.asarray(dispersion, dtype=float) * m * m


# Calibration keys copied out of the compute_metrics call (kept as a constant so
# both the classic and the GCM caller pull the SAME set). Mirrors the calibration
# slots compute_metrics fills (native-CDF PIT: ece_pit / pit_bias / disp_signed +
# the uncertainty-error pair, each with a _per_gene array).
_CALIBRATION_KEYS = (
    "ece_pit", "ece_pit_per_gene",
    "pit_bias", "pit_bias_per_gene",
    "disp_signed", "disp_signed_per_gene",
    "var_residual_corr", "var_residual_corr_per_gene",
    "var_residual_corr_partial", "var_residual_corr_partial_per_gene",
    "empirical_coverage", "confidence_levels", "calibration_per_slide_gene",
)


def count_metrics_final(means, y, depth_scale, groups, corr_metrics, *,
                        alpha=None, pred_var=None, device="cpu", smear=False,
                        gstar=None):
    """Final-inference count metrics, shared by both pipelines.

    The mean correlation is reported on the log1p(y / depth_scale) axis (log1p(CP10k)
    for the cp models, log1p(raw counts) for the raw models); the calibration and the
    variance-residual correlation stay on the model's own fitted target scale.

    Parameters
    ----------
    means    : (N,) or (N, G) predictive mean on the model's own target scale
               (== the saved prediction `X`).
    y        : same shape — the model's own target (cp counts, unrounded cp, or raw).
    depth_scale : the run's single reporting constant (float), or None for the
               depth-blind raw models. NOT an array, and NOT a per-spot quantity.
    groups   : (N,) per-spot slide label (slide-aware aggregation).
    corr_metrics : iterable of metric names.
    alpha    : (N,)/(N,G) NB dispersion at the same scale as means, or None (Poisson).
    pred_var : (N,G) predictive variance on that same target scale, or None.
    device   : data device (the g* truncated sum runs here; both metric passes are CPU).

    Returns
    -------
    dict to `result.update(...)`, carrying BOTH estimators of the same log1p quantity:
        {metric}, _per_gene, _per_slide_gene
            option B — from g* = E[log1p(Y/depth_scale)], the FULL fitted predictive
            (uses mu AND alpha). The Gaussian's estimand; drives the headline.
        {metric}_logmean, _logmean_per_gene, _logmean_per_slide_gene
            option A — from log1p(mu/depth_scale), MEAN-ONLY (ignores alpha), the same
            scale the per-epoch / early-stop metric uses. Above g* by the Jensen gap.
        plus the calibration keys when pred_var is given.
    Having BOTH per (slide, gene) is what lets an offline analysis measure whether
    routing the prediction through the full predictive helps or hurts, per gene and
    per slide, instead of only at the aggregate.
    """
    means = np.asarray(means)
    y = np.asarray(y)
    div = 1.0 if depth_scale is None else float(depth_scale)
    target_log = np.log1p(y / div)
    out = {}

    # Option B, the headline: correlate g* = E[log1p(Y/depth_scale)], the predictive mean
    # of the reported transform, against the observed log1p(y/depth_scale). depth_scale
    # enters only inside the log1p.
    #
    # `gstar` may be passed in precomputed, which skips the truncated sum. That is how a
    # deferred run splits the work: a GPU job computes g* once per prediction file, and the
    # CPU completion then only pays the slide-aware aggregation. Passing the same array
    # rather than recomputing it keeps the two paths byte-identical.
    if gstar is None:
        gstar, resid = count_expected_log1p(means, div, alpha=alpha, device=device)
        if resid > 1e-4:
            print(f"[count_metrics_final] WARNING: count g* residual tail mass {resid:.2e} "
                  f"(>1e-4) — the windowed recurrence hit its k_cap safety bound before "
                  f"the tail closed; lower eps / raise k_cap in count_expected_log1p.")
    else:
        gstar = np.asarray(gstar)
        if gstar.shape != np.shape(means):
            raise ValueError(f"count_metrics_final: precomputed gstar has shape {gstar.shape}, "
                             f"expected {np.shape(means)} (same as means). A shape mismatch here "
                             f"means the companion file's rows were not aligned to this slice.")
    out.update(compute_metrics(gstar, target_log, groups=groups, pred_var=None,
                               corr_metrics=corr_metrics, compute_calibration=False))

    # Option A, the cross-check: log1p(mu/depth_scale), the transform-of-mean estimator of
    # the same quantity, against the same target. The two differ only in how the prediction
    # is formed. Option B routes it through the full fitted predictive, so it uses mu and
    # alpha. Option A uses mu alone, so it sits above g* by the concave Jensen gap, which
    # grows with the predictive variance.
    #
    # Both are emitted per (slide, gene), not just as a scalar, so an offline analysis can
    # ask where bringing the dispersion in helps or hurts. The keys mirror the g* ones with
    # a _logmean suffix, so atoms.parquet gets its own column next to the g* one.
    #
    # This scalar comes from the CPU scipy pass while the per-epoch path uses the on-device
    # twin, so the two agree only to about 1e-9. {metric}_logmean is a cross-check and
    # never a selection criterion. compute_metrics also returns mae/mse for these arrays,
    # which are deliberately not copied out: the reported MAE and MSE stay the g* ones.
    a_res = compute_metrics(np.log1p(means / div), target_log, groups=groups,
                            pred_var=None, corr_metrics=corr_metrics,
                            compute_calibration=False)
    for name in corr_metrics:
        out[f"{name}_logmean"] = a_res[name]
        out[f"{name}_logmean_per_gene"] = a_res[f"{name}_per_gene"]
        for atom in a_res[f"{name}_per_slide_gene"]:
            atom[f"{name}_logmean"] = atom.pop(name)
        out[f"{name}_logmean_per_slide_gene"] = a_res[f"{name}_per_slide_gene"]

    # Calibration. Everything here stays on the model's OWN target scale, with no
    # constant applied: the PIT coverage uses the model's NATIVE count CDF (Poisson /
    # NB) at mu = means with the NB dispersion α fitted at that same scale against the
    # target y, so it is unbiased for the fitted distribution and, being PIT, directly
    # comparable to the Gaussian models (calibration_pit). The uncertainty-error corrs
    # (var_residual_corr[_partial]) likewise pair means / pred_var / y, all three on
    # that one scale.
    if pred_var is not None:
        mu_c = means
        # smear=True (cp_counts_nr continuous target) -> smeared CDF F~; no-op for the
        # integer-target count models (raw_counts / cp_counts).
        if alpha is not None:
            F_lo, F_hi = nb_pit_bounds(mu_c, np.asarray(alpha), y, smear=smear)
        else:
            F_lo, F_hi = poisson_pit_bounds(mu_c, y, smear=smear)
        cal = compute_metrics(means, y, groups=groups, pred_var=np.asarray(pred_var),
                              corr_metrics=(), compute_calibration=True,
                              pit_bounds=(F_lo, F_hi))
        for k in _CALIBRATION_KEYS:
            out[k] = cal[k]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Self-test (CPU; no cluster needed):  python -m src.evaluation.count_metrics
# Verifies count_expected_log1p against an independent scipy brute-force sum, the
# NB->Poisson limit, the Jensen direction, the adaptive truncation tail mass, and
# the depth_scale=1.0 / depth_scale=None identities.
# ─────────────────────────────────────────────────────────────────────────────

def _count_expected_log1p_selftest():
    from scipy.stats import poisson as sp_poisson, nbinom as sp_nbinom
    rng = np.random.default_rng(0)
    N, G = 7, 4
    mu = rng.uniform(0.2, 30.0, size=(N, G))
    # `s` here is just an arbitrary per-row DIVISOR, used to exercise the broadcasting
    # and the recurrence math against scipy. depth_scale is a scalar in production;
    # the scalar cases are covered by the identity block at the end.
    s  = rng.uniform(0.5, 3.0,  size=N)

    # Poisson g* vs an independent scipy brute-force sum (torch-free reference).
    gstar, resid = count_expected_log1p(mu, s, alpha=None, device="cpu")
    k = np.arange(2001)
    ref = np.empty((N, G))
    for i in range(N):
        for j in range(G):
            ref[i, j] = np.sum(np.log1p(k / s[i]) * sp_poisson.pmf(k, mu[i, j]))
    assert np.allclose(gstar, ref, atol=1e-8), \
        f"Poisson g* != scipy ref (max |Δ| {np.abs(gstar - ref).max():.2e})"
    assert resid < 1e-9, f"Poisson residual tail mass too large: {resid:.2e}"

    # return_var: Var[log1p(Y/s)] = E[l^2] - E[l]^2 vs the same scipy brute-force.
    # Same g* is returned (regression guard), plus the new variance channel.
    g_v, var_v, _ = count_expected_log1p(mu, s, alpha=None, device="cpu", return_var=True)
    ref_var = np.empty((N, G))
    for i in range(N):
        for j in range(G):
            l = np.log1p(k / s[i]); pk = sp_poisson.pmf(k, mu[i, j])
            ref_var[i, j] = np.sum(l ** 2 * pk) - np.sum(l * pk) ** 2
    assert np.allclose(g_v, gstar, atol=1e-12), "return_var changed g* (must be identical)"
    assert np.all(var_v >= 0.0), "Var[log1p] must be non-negative"
    assert np.allclose(var_v, ref_var, atol=1e-8), \
        f"Poisson Var[log1p] != scipy ref (max |Δ| {np.abs(var_v - ref_var).max():.2e})"
    # NB variance too (overdispersed).
    _, var_nb_v, _ = count_expected_log1p(mu, s, alpha=np.full((N, G), 0.5),
                                          device="cpu", return_var=True)
    ref_var_nb = np.empty((N, G))
    for i in range(N):
        for j in range(G):
            p = 1.0 / (1.0 + 0.5 * mu[i, j]); l = np.log1p(k / s[i])
            pk = sp_nbinom.pmf(k, 1.0 / 0.5, p)
            ref_var_nb[i, j] = np.sum(l ** 2 * pk) - np.sum(l * pk) ** 2
    assert np.allclose(var_nb_v, ref_var_nb, atol=1e-6), \
        f"NB Var[log1p] != scipy ref (max |Δ| {np.abs(var_nb_v - ref_var_nb).max():.2e})"

    # Jensen: g* = E[log1p(Y/s)] <= log1p(E[Y]/s) = log1p(mu/s), with a real gap.
    log_mean = np.log1p(mu / s[:, None])
    assert np.all(gstar <= log_mean + 1e-9), "g* must be <= log1p(mu/s) (Jensen)"
    assert (log_mean - gstar).max() > 1e-3, "Jensen gap should be non-trivial somewhere"

    # NB -> Poisson as alpha -> 0.
    g_nb0, _ = count_expected_log1p(mu, s, alpha=np.full((N, G), 1e-6), device="cpu")
    assert np.allclose(g_nb0, gstar, atol=1e-4), \
        f"NB(alpha->0) g* != Poisson g* (max |Δ| {np.abs(g_nb0 - gstar).max():.2e})"

    # NB with real overdispersion: still <= log1p(mu/s); tail still negligible.
    g_nb, resid_nb = count_expected_log1p(mu, s, alpha=np.full((N, G), 0.5), device="cpu")
    assert np.all(g_nb <= log_mean + 1e-9), "NB g* must be <= log1p(mu/s)"
    assert resid_nb < 1e-6, f"NB residual tail mass too large: {resid_nb:.2e}"

    # LARGE alpha: the NB mode is mu*(1-alpha), i.e. <= 0 here, so the anchor is k=0 and
    # the down leg must not run at all. This is the branch a Poisson-mode anchor
    # (round(mu)) mishandled -- it started far out in the right tail and marched toward
    # the mass -- so check the VALUES against scipy, not just a bound.
    a_big = np.full((N, G), 2.0)
    g_big, resid_big = count_expected_log1p(mu, s, alpha=a_big, device="cpu")
    ref_big = np.empty((N, G))
    for i in range(N):
        for j in range(G):
            p = 1.0 / (1.0 + 2.0 * mu[i, j])
            ref_big[i, j] = np.sum(np.log1p(k / s[i]) * sp_nbinom.pmf(k, 1.0 / 2.0, p))
    assert np.allclose(g_big, ref_big, atol=1e-6), \
        f"NB(alpha=2, mode at 0) g* != scipy ref (max |Δ| {np.abs(g_big - ref_big).max():.2e})"
    assert resid_big < 1e-6, f"NB(alpha=2) residual tail mass too large: {resid_big:.2e}"

    # PERMUTATION ROUND TRIP -- the guard for the width-ordered gene loop, which walks
    # genes in width order and scatters each chunk back through `cols`. A wrong inverse
    # mapping would transpose whole gene columns: every value still plausible, all attached
    # to the wrong gene.
    # It MUST be checked elementwise against an independent reference. Comparing two
    # gene_chunk values against EACH OTHER cannot catch it: `cols` is a contiguous slice of
    # ONE fixed permutation, so a scatter that wrongly used c0:c1 would apply the SAME
    # wrong permutation at every chunk size and all of them would agree with each other.
    # The fixture has mu DESCENDING across columns so the ascending width sort is forced to
    # reverse it -- a guaranteed non-identity permutation, independent of the RNG seed.
    mu_p = np.tile(np.array([25.0, 10.0, 3.0, 0.5])[:G], (N, 1))
    ref_p = np.empty((N, G))
    for i in range(N):
        for j in range(G):
            p = 1.0 / (1.0 + 0.5 * mu_p[i, j])
            ref_p[i, j] = np.sum(np.log1p(k / s[i]) * sp_nbinom.pmf(k, 1.0 / 0.5, p))
    for gc in (1, 2, 3, 512):            # 512 = the default: one chunk, full permutation
        g_p, _ = count_expected_log1p(mu_p, s, alpha=np.full((N, G), 0.5),
                                      device="cpu", gene_chunk=gc)
        assert np.allclose(g_p, ref_p, atol=1e-6), \
            (f"gene_chunk={gc}: g* != scipy elementwise -- the width-order permutation is "
             f"not inverted on write-back (max |Δ| {np.abs(g_p - ref_p).max():.2e})")

    # float32 MARCHING PATH -- what production actually runs on CUDA, and what this CPU
    # self-test would otherwise never touch (device="cpu" defaults work_dtype to float64).
    # The marching terms are float32 while g_c / mass_c accumulate in float64, so the
    # answer must track the float64 reference to FLOAT32 precision, not to float64
    # precision -- hence rtol=1e-5 rather than the atol=1e-12 used above. alpha=2 is the
    # stress case: p = alpha*mu/(1+alpha*mu) sits closest to 1 there, which is where the
    # small quantity (1-p) loses the most relative precision in float32.
    for a_case in (None, np.full((N, G), 0.5), np.full((N, G), 2.0)):
        g64, _ = count_expected_log1p(mu, s, alpha=a_case, device="cpu",
                                      work_dtype=torch.float64)
        g32, _ = count_expected_log1p(mu, s, alpha=a_case, device="cpu",
                                      work_dtype=torch.float32)
        tag = "Poisson" if a_case is None else f"NB(alpha={a_case[0, 0]})"
        assert np.allclose(g32, g64, rtol=1e-5, atol=1e-6), \
            (f"float32 marching drifted from the float64 reference, {tag}: "
             f"max |Δ| {np.abs(g32 - g64).max():.2e}")

    # 1-D (single-gene) shape is preserved.
    g1, _ = count_expected_log1p(mu[:, 0], s, alpha=None, device="cpu")
    assert g1.shape == (N,) and np.allclose(g1, gstar[:, 0], atol=1e-12), \
        "single-gene (1-D) g* shape/values wrong"

    # --- HIGH MEAN, the regime where a fixed window chops or underflows: exact vs scipy ---
    # Starting at P(0)=e^{-mu} would underflow here; the mode-start window must not.
    s_hi  = np.array([0.4, 1.0, 2.5])
    mu_hi = np.array([[2.0e3], [1.5e4], [2.5e4]])              # (3, 1): 3 spots, 1 gene
    g_hi, resid_hi = count_expected_log1p(mu_hi[:, 0], s_hi, alpha=None, device="cpu")
    kk = np.arange(60000)
    ref_hi = np.array([np.sum(np.log1p(kk / s_hi[i]) * sp_poisson.pmf(kk, mu_hi[i, 0]))
                       for i in range(3)])
    assert np.allclose(g_hi, ref_hi, atol=1e-4), \
        f"high-mu Poisson g* != scipy ref (max |Δ| {np.abs(g_hi - ref_hi).max():.2e})"
    assert resid_hi < 1e-6, f"high-mu Poisson residual too large: {resid_hi:.2e}"

    # --- HIGH MEAN x HIGH DISPERSION NB (broad right tail: the timeout case) ---
    # scipy nbinom(n=1/alpha, p=1/(1+alpha*mu)) matches torch NB(total_count=1/alpha,
    # probs=alpha*mu/(1+alpha*mu)); brute-force out far enough that the tail closes.
    mu_nb = np.array([[2.0e3], [5.0e3]])                       # (2, 1), 2 spots, 1 gene
    s_nb  = np.array([1.0, 0.7])
    al    = 0.5
    g_nbhi, resid_nbhi = count_expected_log1p(mu_nb[:, 0], s_nb,
                                              alpha=np.full(2, al), device="cpu")
    kk = np.arange(200000)
    ref_nb = np.empty(2)
    for i in range(2):
        p = 1.0 / (1.0 + al * mu_nb[i, 0])
        ref_nb[i] = np.sum(np.log1p(kk / s_nb[i]) * sp_nbinom.pmf(kk, 1.0 / al, p))
    assert np.allclose(g_nbhi, ref_nb, atol=1e-3), \
        f"high-mu/high-alpha NB g* != scipy ref (max |Δ| {np.abs(g_nbhi - ref_nb).max():.2e})"
    assert resid_nbhi < 1e-5, f"high-mu NB residual too large: {resid_nbhi:.2e}"

    # --- float32 marching path (the GPU production dtype) matches float64 closely ---
    g32, _ = count_expected_log1p(mu, s, alpha=np.full((N, G), 0.5),
                                  device="cpu", work_dtype=torch.float32)
    assert np.allclose(g32, g_nb, atol=1e-4), \
        f"float32 g* != float64 g* (max |Δ| {np.abs(g32 - g_nb).max():.2e})"

    # ...and at HIGH mu*alpha, which is the regime where float32 marching actually costs
    # something and the ONLY one that matters in production. Everything above runs
    # mu <= 30, i.e. a few dozen steps; here alpha*mu is ~2500, so p = alpha*mu/(1+alpha*mu)
    # is 0.9996 and the march is tens of thousands of steps, over which float32 error in
    # the ratio compounds. That is exactly the shape of a 10x_TuPro cp_counts run (mu on
    # the CP_T scale) on CUDA, where work_dtype defaults to float32. Reuses the float64
    # result computed just above as the reference.
    g_hi32, _ = count_expected_log1p(mu_nb[:, 0], s_nb, alpha=np.full(2, al),
                                     device="cpu", work_dtype=torch.float32)
    assert np.allclose(g_hi32, g_nbhi, rtol=1e-4), \
        (f"float32 marching drifted from float64 at high alpha*mu: "
         f"max |Δ| {np.abs(g_hi32 - g_nbhi).max():.2e}")

    # return_var on the float32 path: no caller uses return_var today, so this only pins
    # that the second-moment branch RUNS and stays sane under float32 (the accumulators
    # g2_c / mass_c are float64, so the E[l^2] - E[l]^2 cancellation itself is not float32).
    # Deliberately not asserted tightly against float64 -- an untested tolerance on a
    # code path with no consumer is a flaky test, not a safety net.
    _, var32, _ = count_expected_log1p(mu, s, alpha=np.full((N, G), 0.5), device="cpu",
                                       work_dtype=torch.float32, return_var=True)
    assert np.all(np.isfinite(var32)) and np.all(var32 >= 0.0), \
        "float32 return_var produced non-finite or negative Var[log1p]"

    # --- depth_scale identities: 1.0 and None are EXACT no-ops -------------------
    # This is what pins TLS / LUNG_CANCER (count_common_depth 10000 -> depth_scale
    # 1.0) and the depth-blind *_raw models (depth_scale None) as bit-identical to a
    # division-free path, at EVERY place a divisor was introduced. 10x_TuPro
    # (depth_scale 2.5) is checked to actually divide, so the division can never be
    # dropped silently.
    g_one,  _ = count_expected_log1p(mu, 1.0,  alpha=None, device="cpu")
    g_none, _ = count_expected_log1p(mu, None, alpha=None, device="cpu")
    assert np.array_equal(g_one, g_none), "count_expected_log1p: None must equal 1.0"
    ref_one = np.array([[np.sum(np.log1p(k) * sp_poisson.pmf(k, mu[i, j]))
                         for j in range(G)] for i in range(N)])
    assert np.allclose(g_one, ref_one, atol=1e-8), \
        "count_expected_log1p(depth_scale=1.0) != divisor-free scipy ref"

    # count_corr_inputs: 1.0 / None leave BOTH sides untouched, and y comes back raw.
    m_t = torch.as_tensor(mu)
    y_t = torch.as_tensor(np.round(mu * 2.0))
    for d in (1.0, None):
        cm, cy, ydiag = count_corr_inputs(m_t, y_t, d)
        assert torch.equal(cm, torch.log1p(m_t)), f"count_corr_inputs means side moved (d={d})"
        assert torch.equal(cy, torch.log1p(y_t)), f"count_corr_inputs y side moved (d={d})"
        assert torch.equal(ydiag, y_t), f"count_corr_inputs must return y on its own scale (d={d})"
    cm25, cy25, _ = count_corr_inputs(m_t, y_t, 2.5)
    assert torch.allclose(cm25, torch.log1p(m_t / 2.5)) and \
           torch.allclose(cy25, torch.log1p(y_t / 2.5)), \
        "depth_scale=2.5 must divide BOTH sides before log1p"

    # count_metrics_final: 1.0 and None agree on every scalar it reports.
    groups = np.array(["A"] * 3 + ["B"] * (N - 3))
    y_c = np.round(mu * 2.0)
    a_c = np.full((N, G), 0.5)
    pv  = count_pred_var(mu, a_c)
    f_one  = count_metrics_final(mu, y_c, 1.0,  groups, ("pearson",),
                                 alpha=a_c, pred_var=pv)
    f_none = count_metrics_final(mu, y_c, None, groups, ("pearson",),
                                 alpha=a_c, pred_var=pv)
    for key in ("pearson", "pearson_logmean", "ece_pit", "pit_bias",
                "disp_signed", "var_residual_corr"):
        assert np.isclose(f_one[key], f_none[key], equal_nan=True), \
            f"count_metrics_final: depth_scale None != 1.0 for {key}"

    # --- BOTH log1p estimators reach the atoms, under matching key names ----------
    # option B (g*) under {metric}; option A (log1p(mu/d), mean-only) under
    # {metric}_logmean, with the same per-gene / per-(slide,gene) fan-out, so the
    # atoms writers of both pipelines emit one column per estimator.
    f25 = count_metrics_final(mu, y_c, 2.5, groups, ("pearson", "spearman"),
                              alpha=a_c, pred_var=pv)
    for name in ("pearson", "spearman"):
        for suffix in ("", "_per_gene", "_per_slide_gene"):
            assert f"{name}_logmean{suffix}" in f25, \
                f"count_metrics_final: missing option-A key {name}_logmean{suffix}"
        b_atoms = f25[f"{name}_per_slide_gene"]
        a_atoms = f25[f"{name}_logmean_per_slide_gene"]
        assert len(a_atoms) == len(b_atoms), \
            (f"option-A per-(slide,gene) array has {len(a_atoms)} atoms vs {len(b_atoms)} "
             f"for g* ({name}) — the two estimators must share the same atom grid")
        assert len(f25[f"{name}_logmean_per_gene"]) == len(f25[f"{name}_per_gene"]), \
            f"option-A per-gene array length != g* per-gene length ({name})"
        for a, b in zip(a_atoms, b_atoms):
            assert (a["slide"], a["gene"]) == (b["slide"], b["gene"]), \
                f"option-A / g* atoms are not aligned on (slide, gene) for {name}"
            # The VALUE must sit under the _logmean key (this is what makes it a
            # separate atoms.parquet column instead of overwriting the g* one).
            assert f"{name}_logmean" in a and name not in a, \
                f"option-A atom must carry only the '{name}_logmean' value key"
    # Jensen again, now at the aggregate: option A correlates log1p(mu/d), which is
    # >= g* everywhere, so the two scalars are genuinely different numbers here.
    assert np.isfinite(f25["pearson"]) and np.isfinite(f25["pearson_logmean"]), \
        "both estimators must produce a finite pearson on this fixture"

    print("count_expected_log1p self-test: PASS")


if __name__ == "__main__":
    _count_expected_log1p_selftest()
