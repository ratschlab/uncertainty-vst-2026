# src/evaluation/calibration_pit.py
#
# Cross-family PIT calibration. This is the single source of truth for the
# calibration math: the CPU final path (metrics_slide_aware.per_slide_gene_calibration),
# the GPU per-epoch twin (metrics_slide_aware.calibration_scalars_gpu) and the
# offline notebooks all call the primitives here, so the logged scalars and the
# offline curves come from one implementation.
#
# Each model is scored with its own native CDF, through a one-sided PIT coverage:
#
#   coverage(α) = mean_i  P(PIT_i ≤ α)
#
# computed as the analytic expected coverage under the randomized PIT, so no random
# draw is taken and the value is deterministic. A continuous CDF has jump 0 and the
# formula collapses to the plain PIT coverage 1{F(y) ≤ α}, which is what lets one
# code path serve Gaussian, Poisson and NB. A two-sided central interval built from
# the Gaussian CDF would instead be biased for the count models, because a discrete
# CDF can never make the raw PIT uniform. PIT is invariant under monotone
# reparameterization, so a count model scored on its count scale is directly
# comparable to a Gaussian scored on log1p-CP10k or on z-scores.
#
# The three scalars are reductions of that one coverage curve. The signed coverage
# error telescopes to ½ − E[PIT], a pure location statistic blind to symmetric
# over/under-confidence, so it is split three ways:
#   ece_pit     = mean_α |coverage(α) − α|                 magnitude
#   pit_bias    = mean_α (coverage(α) − α) = ½ − E[PIT]    location: <0 ⟺ E[PIT]>½
#                                                          ⟺ predictions too low
#   disp_signed = mean_α (coverage(α) − α)·sign(½ − α)     >0 over-confident
#                                                          (∪-PIT), <0 under-conf.
#
# Native-CDF bounds per family, F_lo = P(Y < y) and F_hi = P(Y ≤ y):
#   Gaussian : F_lo = F_hi = Φ((y − μ)/σ)             (continuous, jump 0)
#   Poisson  : μ_c is the mean on the model's own target scale, y that same target
#   NB       : (total_count r = 1/α, μ_c); p = r/(r+μ_c) = 1/(1+α·μ_c)
#
# A numpy/scipy flavor serves the CPU final path and the offline work; a torch
# flavor serves the GPU per-epoch twin. The self-test pins the torch flavor to scipy.

import numpy as np
import torch
from scipy.special import ndtr               # standard-normal CDF (fast)
from scipy.stats import poisson as _sp_poisson, nbinom as _sp_nbinom


# Confidence grid for the one-sided PIT coverage. α is a point on the [0,1] PIT/CDF
# axis, so an evenly-spaced grid is the natural sampling of the reliability curve.
# The grid is symmetric around 0.5 so disp_signed's sign(0.5−α) weighting is
# balanced; the α=0.5 level contributes 0.
CONF_LEVELS = np.linspace(0.05, 0.95, 19)


# ─────────────────────────────────────────────────────────────────────────────
# Native-CDF bounds (F_lo, F_hi) — numpy / scipy (CPU final path + offline).
# Each returns (F_lo, F_hi) with the same shape as the inputs. Continuous models
# return F_lo == F_hi (jump 0). For y == 0 the lower bound P(Y < 0) = 0.
# ─────────────────────────────────────────────────────────────────────────────

def gaussian_pit_bounds(means, std, y):
    F = ndtr((np.asarray(y) - np.asarray(means)) / np.asarray(std))
    return F, F


def poisson_pit_bounds(mu_c, y, smear=False):
    mu_c = np.asarray(mu_c, dtype=float)
    y = np.asarray(y)
    F_hi = _sp_poisson.cdf(y, mu_c)
    F_lo = _sp_poisson.cdf(y - 1, mu_c)        # cdf(-1) = 0 at y == 0
    if smear:                                  # cp_counts_nr continuous target -> smeared CDF F~
        # Spread each integer atom P(Y=k) uniformly over [k-1/2, k+1/2]:
        #   F~(y) = P(Y<=k-1) + (y-(k-1/2))*P(Y=k),  k = round(y)
        # Positive non-integer y -> F_lo=F_hi=F~(y) (jump 0 -> continuous coverage branch);
        # y==0 and exact integer y keep the discrete bounds (nonint False), so this is
        # byte-identical to the discrete path on the integer grid.
        yf = y.astype(float); k = np.rint(yf)
        nonint = yf != k
        F_km1 = _sp_poisson.cdf(k - 1, mu_c); pmf_k = _sp_poisson.pmf(k, mu_c)
        Ft = F_km1 + (yf - (k - 0.5)) * pmf_k
        F_lo = np.where(nonint, Ft, F_lo); F_hi = np.where(nonint, Ft, F_hi)
    return F_lo, F_hi


def nb_pit_bounds(mu_c, alpha, y, smear=False):
    mu_c = np.asarray(mu_c, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    y = np.asarray(y)
    r = 1.0 / alpha                            # total_count
    p = 1.0 / (1.0 + alpha * mu_c)             # = r / (r + mu_c)
    F_hi = _sp_nbinom.cdf(y, r, p)
    F_lo = _sp_nbinom.cdf(y - 1, r, p)         # cdf(-1) = 0 at y == 0
    if smear:                                  # cp_counts_nr continuous target -> smeared CDF F~
        yf = y.astype(float); k = np.rint(yf)  # see poisson_pit_bounds for the F~ formula
        nonint = yf != k
        F_km1 = _sp_nbinom.cdf(k - 1, r, p); pmf_k = _sp_nbinom.pmf(k, r, p)
        Ft = F_km1 + (yf - (k - 0.5)) * pmf_k
        F_lo = np.where(nonint, Ft, F_lo); F_hi = np.where(nonint, Ft, F_hi)
    return F_lo, F_hi


# ─────────────────────────────────────────────────────────────────────────────
# Coverage curve + scalars — numpy. Operate on axis 0 (over spots), so a 1-D
# slice (n,) -> (C,) [the CPU per-(slide,gene) loop] and a 2-D block (n, G) ->
# (C, G) both work from one implementation.
# ─────────────────────────────────────────────────────────────────────────────

def coverage_curve_np(F_lo, F_hi, conf=CONF_LEVELS, jump_eps=1e-12):
    """coverage(α) = mean over spots of the analytic expected indicator c_i(α).

    c_i(α) = clip((α − F_lo)/jump, 0, 1)  where jump = F_hi − F_lo > 0
           = 1{ F_hi ≤ α }                where jump == 0 (continuous collapse)
    """
    F_lo = np.asarray(F_lo, dtype=float)
    F_hi = np.asarray(F_hi, dtype=float)
    jump = F_hi - F_lo
    has_jump = jump > jump_eps
    safe_jump = np.where(has_jump, jump, 1.0)          # avoid 0/0; masked out below
    conf = np.asarray(conf, dtype=float)

    cov = np.empty((conf.shape[0],) + F_hi.shape[1:], dtype=float)
    for j, a in enumerate(conf):
        ratio = np.clip((a - F_lo) / safe_jump, 0.0, 1.0)
        c = np.where(has_jump, ratio, (F_hi <= a).astype(float))
        cov[j] = c.mean(axis=0)
    return cov


def scalars_from_coverage_np(cov, conf=CONF_LEVELS):
    """(C,) or (C, G) coverage -> the 3 scalars (floats, or (G,) arrays)."""
    conf = np.asarray(conf, dtype=float)
    sign = np.sign(0.5 - conf)
    if cov.ndim == 1:
        d = cov - conf
        return {
            "ece_pit":     float(np.mean(np.abs(d))),
            "pit_bias":    float(np.mean(d)),
            "disp_signed": float(np.mean(d * sign)),
        }
    d = cov - conf[:, None]                            # (C, G)
    return {
        "ece_pit":     np.mean(np.abs(d), axis=0),
        "pit_bias":    np.mean(d, axis=0),
        "disp_signed": np.mean(d * sign[:, None], axis=0),
    }


def pit_calibration_np(F_lo, F_hi, conf=CONF_LEVELS):
    """Convenience: bounds -> (coverage curve, {ece_pit, pit_bias, disp_signed})."""
    cov = coverage_curve_np(F_lo, F_hi, conf)
    return cov, scalars_from_coverage_np(cov, conf)


# ─────────────────────────────────────────────────────────────────────────────
# Native-CDF bounds (F_lo, F_hi), torch flavor for the GPU per-epoch twin.
# Gaussian: erf. Poisson: regularized upper incomplete gamma, O(1) and exact. NB:
# regularized incomplete beta when torch.special exposes it, O(1); otherwise a
# capped running-cumulative PMF sum, O(K) in the largest observed count rather than
# O(y) per atom. The Poisson and NB versions work in float64, the precision the NB
# cumulative sum needs. The Gaussian one returns the input dtype: the GPU twin calls
# it on float32 slices, and forcing float64 there would allocate an (n_s, G)
# transient large enough to exhaust GPU memory at the all-genes scale.
# ─────────────────────────────────────────────────────────────────────────────

_SQRT2 = float(np.sqrt(2.0))


def gaussian_pit_bounds_t(means, std, y):
    F = 0.5 * (1.0 + torch.erf((y - means) / (std * _SQRT2)))
    return F, F


def poisson_pit_bounds_t(mu_c, y, smear=False):
    mu_c = mu_c.double()
    yf = y.double()
    # Poisson CDF F(k; μ) = Q(k+1, μ) = gammaincc(k+1, μ)  (regularized upper γ).
    F_hi = torch.special.gammaincc(yf + 1.0, mu_c)
    # F_lo = P(Y ≤ y−1) = gammaincc(y, μ) for y ≥ 1, else 0.
    F_lo = torch.where(yf >= 1.0,
                       torch.special.gammaincc(torch.clamp(yf, min=1.0), mu_c),
                       torch.zeros_like(mu_c))
    if smear:                                    # same piecewise-linear F~ as numpy poisson_pit_bounds
        k = torch.round(yf); nonint = yf != k    # pmf_k = F(k) - F(k-1) (matches scipy pmf(k))
        F_km1 = torch.where(k >= 1.0,
                            torch.special.gammaincc(torch.clamp(k, min=1.0), mu_c),
                            torch.zeros_like(mu_c))
        F_k = torch.special.gammaincc(k + 1.0, mu_c)
        Ft = F_km1 + (yf - (k - 0.5)) * (F_k - F_km1)
        F_lo = torch.where(nonint, Ft, F_lo); F_hi = torch.where(nonint, Ft, F_hi)
    return F_lo, F_hi


def _nb_cdf_cumsum_t(mu_c, alpha, y, k_cap=20000, gene_chunk=512):
    """NB (F_lo, F_hi) from a log-space PMF recurrence, the fallback for when
    torch.special.betainc is unavailable. The recurrence carries no lgamma:

        r = 1/α ;  q = αμ/(1+αμ) = torch NB `probs`  ->  log q = log(αμ) − log1p(αμ)
        log P(0)   = −r·log1p(αμ)
        log P(k+1) = log P(k) + log(k+r) − log(k+1) + log q

    Staying in log space is what keeps it safe. P(0)=(1+αμ)^(−1/α) underflows to zero
    once μ is large and α small, and a linear recurrence started from zero stays
    zero, returning an all-zero CDF with no error raised. Only the per-k `exp` into
    the cumulative sum touches linear space, where a genuinely-zero tail term is
    correct. Per spot: F_lo = P(Y<y) = cum at k=y−1, F_hi = P(Y≤y) = cum at k=y. A y
    above the k_cap bound returns ~1.

    Dtype: inputs and output float32, recurrence entirely in float64. That float64
    lives only in the gene-chunked working buffer (n_s, gene_chunk), never in a full
    (n_s, G) float64 matrix, which would exhaust GPU memory at the all-genes scale.
    Float64 covers the increments as well as the accumulator: a march of thousands of
    float32 increments drifts far enough to miss the tolerance the self-test checks
    against scipy. A 1-D input is one chunk."""
    F_lo = torch.zeros_like(mu_c, dtype=torch.float32)
    F_hi = torch.zeros_like(mu_c, dtype=torch.float32)
    G = mu_c.shape[1] if mu_c.ndim == 2 else 1
    starts = range(0, G, gene_chunk) if mu_c.ndim == 2 else [0]
    for c0 in starts:
        cs = slice(c0, min(c0 + gene_chunk, G))
        mc = (mu_c[:, cs] if mu_c.ndim == 2 else mu_c).double()
        ac = (alpha[:, cs] if alpha.ndim == 2 else alpha).double()
        yc = (y[:, cs] if y.ndim == 2 else y).double()
        r = 1.0 / ac
        am = ac * mc
        log_q = torch.log(am) - torch.log1p(am)                      # log q,  f64
        log_p = -r * torch.log1p(am)                                 # log P(0), f64
        cum = torch.zeros_like(log_p)
        flo = torch.zeros_like(log_p)
        fhi = torch.zeros_like(log_p)
        K = int(torch.clamp(yc.max(), max=k_cap).item()) if yc.numel() else 0
        for k in range(K + 1):
            at_k = (yc == k)
            if at_k.any():
                flo = torch.where(at_k, cum, flo)                    # cum = P(Y<k)
            cum = cum + log_p.exp()                                   # -> P(Y≤k)
            if at_k.any():
                fhi = torch.where(at_k, cum, fhi)
            log_p = log_p + torch.log(k + r) - float(np.log1p(k)) + log_q
        beyond = yc > K                                              # past cap -> ~1
        if beyond.any():
            flo = torch.where(beyond, torch.ones_like(flo), flo)
            fhi = torch.where(beyond, torch.ones_like(fhi), fhi)
        if mu_c.ndim == 2:
            F_lo[:, cs], F_hi[:, cs] = flo.float(), fhi.float()
        else:
            F_lo, F_hi = flo.float(), fhi.float()
    return F_lo, F_hi


def nb_pit_bounds_t(mu_c, alpha, y, smear=False):
    """NB native PIT bounds for the GPU per-epoch path. Closed-form regularized
    incomplete beta if torch.special exposes betainc (O(1)); else the underflow-safe
    log-space recurrence above (float32). smear=True (cp_counts_nr continuous target)
    applies the SAME piecewise-linear F~ as the numpy nb_pit_bounds."""
    if hasattr(torch.special, "betainc"):
        mu_c, alpha, yf = mu_c.double(), alpha.double(), y.double()
        r = 1.0 / alpha
        p = 1.0 / (1.0 + alpha * mu_c)                       # = r / (r + μ)
        # NB CDF F(y; r, p) = I_p(r, y+1)  (regularized incomplete beta).
        F_hi = torch.special.betainc(r, yf + 1.0, p)
        F_lo = torch.where(yf >= 1.0,
                           torch.special.betainc(r, torch.clamp(yf, min=1.0), p),
                           torch.zeros_like(mu_c))
        if smear:                                            # F~: spread atom over [k-1/2, k+1/2]
            k = torch.round(yf); nonint = yf != k
            F_km1 = torch.where(k >= 1.0,
                                torch.special.betainc(r, torch.clamp(k, min=1.0), p),
                                torch.zeros_like(mu_c))
            F_k = torch.special.betainc(r, k + 1.0, p)
            Ft = F_km1 + (yf - (k - 0.5)) * (F_k - F_km1)    # pmf_k = F_k - F_km1
            F_lo = torch.where(nonint, Ft, F_lo); F_hi = torch.where(nonint, Ft, F_hi)
        return F_lo.float(), F_hi.float()
    if smear:                          # cumsum fallback: integer-k bounds via k=round(y), then F~
        k = torch.round(y)
        F_km1, F_k = _nb_cdf_cumsum_t(mu_c, alpha, k)        # (P(Y<k), P(Y<=k)) at integer k
        Ft = F_km1 + (y - (k - 0.5)) * (F_k - F_km1)
        nonint = y != k
        return torch.where(nonint, Ft, F_km1), torch.where(nonint, Ft, F_k)
    return _nb_cdf_cumsum_t(mu_c, alpha, y)


# ─────────────────────────────────────────────────────────────────────────────
# Coverage curve + scalars — torch. (C, G) coverage from (n_s, G) bounds; loop
# over the C levels so the largest tensor stays (n_s, G) (mirrors the OOM-safe
# per-level loop in calibration_scalars_gpu — no (C, n_s, G) cube).
# ─────────────────────────────────────────────────────────────────────────────

def coverage_curve_t(F_lo, F_hi, conf, jump_eps=1e-12):
    """F_lo/F_hi: (n_s,) or (n_s, G) on device; conf: (C,) -> coverage (C, ...).
    Output dtype follows F_hi: the GPU per-epoch twin passes float32 (the
    (n_s, G) operand stays the GPU's fast dtype — float64 OOMs at the all-genes
    scale); the numpy/self-test path passes float64. conf should match F_hi's dtype."""
    jump = F_hi - F_lo
    has_jump = jump > jump_eps
    safe_jump = torch.where(has_jump, jump, torch.ones_like(jump))
    cov = torch.empty((conf.shape[0],) + F_hi.shape[1:],
                      device=F_hi.device, dtype=F_hi.dtype)
    for j in range(conf.shape[0]):
        a = conf[j]
        ratio = torch.clamp((a - F_lo) / safe_jump, 0.0, 1.0)
        c = torch.where(has_jump, ratio, (F_hi <= a).to(F_hi.dtype))
        cov[j] = c.mean(dim=0)
    return cov


def scalars_from_coverage_t(cov, conf):
    """(C, G) coverage -> {ece_pit, pit_bias, disp_signed} each (G,) float64."""
    sign = torch.sign(0.5 - conf).double()                  # (C,)
    d = cov - conf[:, None].double()                        # (C, G)
    return {
        "ece_pit":     d.abs().mean(dim=0),
        "pit_bias":    d.mean(dim=0),
        "disp_signed": (d * sign[:, None]).mean(dim=0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Self-test (CPU; no cluster needed):  python -m src.evaluation.calibration_pit
# A CALIBRATION sanity check, not a smoke test:
#   1. discrete-jump correctness: y ~ predictive  -> ece_pit ≈ 0, coverage ≈ diag
#   2. dispersion direction: narrowed -> disp_signed > 0; widened -> < 0
#   3. bias direction: mean shifted low -> pit_bias < 0; high -> > 0
#   4. torch twin matches scipy bounds; NB->Poisson limit; 1-D shape preserved.
# ─────────────────────────────────────────────────────────────────────────────

def _selftest():
    rng = np.random.default_rng(0)
    N = 200_000
    conf = CONF_LEVELS

    # ── 1. well-specified -> calibrated (ece_pit ≈ 0, coverage ≈ diagonal) ──
    # Gaussian
    mu = rng.normal(0, 1, N); sd = np.full(N, 0.7)
    y = rng.normal(mu, sd)
    cov, sc = pit_calibration_np(*gaussian_pit_bounds(mu, sd, y), conf)
    assert sc["ece_pit"] < 5e-3, f"Gaussian ece_pit not ~0: {sc['ece_pit']:.4f}"
    assert np.max(np.abs(cov - conf)) < 1e-2, "Gaussian coverage off-diagonal"

    # Poisson (discrete jump handling is the real test)
    mu_c = rng.uniform(0.5, 40.0, N)
    y = rng.poisson(mu_c)
    cov, sc = pit_calibration_np(*poisson_pit_bounds(mu_c, y), conf)
    assert sc["ece_pit"] < 5e-3, f"Poisson ece_pit not ~0: {sc['ece_pit']:.4f}"
    assert np.max(np.abs(cov - conf)) < 1e-2, "Poisson coverage off-diagonal"

    # NB
    mu_c = rng.uniform(0.5, 40.0, N); alpha = np.full(N, 0.5)
    r = 1.0 / alpha; p = 1.0 / (1.0 + alpha * mu_c)
    y = _sp_nbinom.rvs(r, p, random_state=rng)
    cov, sc = pit_calibration_np(*nb_pit_bounds(mu_c, alpha, y), conf)
    assert sc["ece_pit"] < 5e-3, f"NB ece_pit not ~0: {sc['ece_pit']:.4f}"
    assert np.max(np.abs(cov - conf)) < 1e-2, "NB coverage off-diagonal"

    # ── 2. dispersion direction (Gaussian, well-specified y from sd=0.7) ──
    mu = rng.normal(0, 1, N); sd_true = np.full(N, 0.7); y = rng.normal(mu, sd_true)
    _, sc_narrow = pit_calibration_np(*gaussian_pit_bounds(mu, sd_true * 0.5, y), conf)
    _, sc_wide   = pit_calibration_np(*gaussian_pit_bounds(mu, sd_true * 2.0, y), conf)
    assert sc_narrow["disp_signed"] > 0, "narrowed predictive must be over-confident"
    assert sc_wide["disp_signed"] < 0, "widened predictive must be under-confident"

    # ── 3. bias direction ──
    # predictions too LOW => observations land high in their predictive
    # => E[PIT] > 0.5 => pit_bias < 0.
    _, sc_low  = pit_calibration_np(*gaussian_pit_bounds(mu - 1.0, sd_true, y), conf)
    _, sc_high = pit_calibration_np(*gaussian_pit_bounds(mu + 1.0, sd_true, y), conf)
    assert sc_low["pit_bias"] < 0, "under-predicting the mean must give pit_bias < 0"
    assert sc_high["pit_bias"] > 0, "over-predicting the mean must give pit_bias > 0"

    # ── 4. torch twin matches scipy bounds (fresh INTEGER count samples — the
    # NB/Poisson bounds are only defined on integer y; do not reuse the float y
    # from the Gaussian sections, whose int cast truncates toward zero). ──
    # Poisson
    mu_cp = rng.uniform(0.5, 40.0, 5000); yp = rng.poisson(mu_cp)
    flo_np, fhi_np = poisson_pit_bounds(mu_cp, yp)
    flo_t, fhi_t = poisson_pit_bounds_t(torch.as_tensor(mu_cp),
                                        torch.as_tensor(yp.astype(np.int64)))
    assert np.allclose(fhi_np, fhi_t.numpy(), atol=1e-9), "Poisson F_hi torch≠scipy"
    assert np.allclose(flo_np, flo_t.numpy(), atol=1e-9), "Poisson F_lo torch≠scipy"
    # NB
    mu_cn = rng.uniform(0.5, 40.0, 5000); alpha_n = np.full(5000, 0.5)
    rn = 1.0 / alpha_n; pn = 1.0 / (1.0 + alpha_n * mu_cn)
    yn = _sp_nbinom.rvs(rn, pn, random_state=rng)
    flo_np, fhi_np = nb_pit_bounds(mu_cn, alpha_n, yn)
    flo_t, fhi_t = nb_pit_bounds_t(torch.as_tensor(mu_cn), torch.as_tensor(alpha_n),
                                   torch.as_tensor(yn.astype(np.int64)))
    fhi_t, flo_t = fhi_t.numpy(), flo_t.numpy()
    print(f"  [diag] torch.special.betainc available: {hasattr(torch.special, 'betainc')}")
    print(f"  [diag] NB F_hi max|Δ|={np.abs(fhi_np - fhi_t).max():.3e}  "
          f"F_lo max|Δ|={np.abs(flo_np - flo_t).max():.3e}")
    assert np.allclose(fhi_np, fhi_t, atol=1e-6), "NB F_hi torch≠scipy"
    assert np.allclose(flo_np, flo_t, atol=1e-6), "NB F_lo torch≠scipy"

    # ── 4b. Underflow gate: the case the log-space recurrence exists for.
    # At μ=1000, α=0.01, P(0)=(1+αμ)^(−1/α)=11^−100≈1e−104 underflows float32 to 0,
    # and a linear recurrence would start at 0 and stay 0, giving an all-zero CDF.
    # The gate asserts the NB CDF still matches scipy to ≤1e−5 at y spanning the mode.
    mu_u, al_u = 1000.0, 0.01
    p0_f32 = float(np.float32((1.0 + al_u * mu_u) ** (-1.0 / al_u)))
    assert p0_f32 == 0.0, f"underflow cell isn't underflowing in float32 (P0={p0_f32})"
    yu = np.arange(700, 1400)                                   # spots spanning the mode
    mu_arr = np.full(yu.shape, mu_u); al_arr = np.full(yu.shape, al_u)
    _, fhi_ref = nb_pit_bounds(mu_arr, al_arr, yu)              # scipy float64 reference
    _, fhi_u = nb_pit_bounds_t(torch.as_tensor(mu_arr), torch.as_tensor(al_arr),
                               torch.as_tensor(yu.astype(np.int64)))
    dmax = float(np.abs(fhi_ref - fhi_u.numpy()).max())
    print(f"  [diag] UNDERFLOW GATE μ={mu_u} α={al_u}: float32 P(0)={p0_f32} (underflowed); "
          f"log-space NB F_hi max|Δ| vs scipy = {dmax:.2e}")
    assert dmax < 1e-5, f"UNDERFLOW GATE FAILED: NB F_hi vs scipy max|Δ|={dmax:.2e} (>1e-5)"

    # ── 5. torch coverage curve + scalars match the numpy path (the GPU per-epoch
    # twin in calibration_scalars_gpu relies on these). Mixed continuous (jump=0)
    # and discrete (jump>0) bounds in one (n, G) block. ──
    G = 6
    flo_b = rng.uniform(0.0, 0.9, (4000, G))
    fhi_b = flo_b.copy()
    jump_cols = [1, 3, 5]                          # these columns get a discrete jump
    fhi_b[:, jump_cols] = np.minimum(flo_b[:, jump_cols] + rng.uniform(0.02, 0.1,
                                     (4000, len(jump_cols))), 1.0)
    cov_np = coverage_curve_np(flo_b, fhi_b, CONF_LEVELS)
    sc_np = scalars_from_coverage_np(cov_np, CONF_LEVELS)
    conf_t = torch.as_tensor(CONF_LEVELS)
    cov_t = coverage_curve_t(torch.as_tensor(flo_b), torch.as_tensor(fhi_b), conf_t)
    sc_t = scalars_from_coverage_t(cov_t, conf_t)
    assert np.allclose(cov_np, cov_t.numpy(), atol=1e-12), "coverage_curve torch≠numpy"
    for k in ("ece_pit", "pit_bias", "disp_signed"):
        assert np.allclose(sc_np[k], sc_t[k].numpy(), atol=1e-12), f"{k} torch≠numpy"

    # ── 6. F~ smeared CDF (cp_counts_nr continuous target) ──────────────────────
    # (a) at INTEGER y the smear collapses to the discrete bounds (byte-identical);
    # (b) at the half-integer knot y=k+0.5, F_lo == F_hi == F(k); (c) numpy == torch
    # on strictly non-integer y (the hard cross-path agreement constraint).
    mu_s = rng.uniform(0.5, 40.0, 5000); ks = rng.integers(0, 30, 5000)
    alp = np.full(5000, 0.5)
    for lo_hi, lo_hi_s in ((poisson_pit_bounds(mu_s, ks), poisson_pit_bounds(mu_s, ks, smear=True)),
                           (nb_pit_bounds(mu_s, alp, ks), nb_pit_bounds(mu_s, alp, ks, smear=True))):
        assert np.allclose(lo_hi[0], lo_hi_s[0]) and np.allclose(lo_hi[1], lo_hi_s[1]), \
            "smear != discrete bounds at integer y"
    yhalf = ks.astype(float) + 0.5
    flo_h, fhi_h = poisson_pit_bounds(mu_s, yhalf, smear=True)
    assert np.allclose(flo_h, fhi_h) and np.allclose(flo_h, _sp_poisson.cdf(ks, mu_s), atol=1e-9), \
        "F~(k+0.5) != F(k)"
    ynr = ks.astype(float) + rng.uniform(0.1, 0.9, 5000)      # strictly non-integer
    _, fhi_pnp = poisson_pit_bounds(mu_s, ynr, smear=True)
    _, fhi_pt = poisson_pit_bounds_t(torch.as_tensor(mu_s), torch.as_tensor(ynr), smear=True)
    assert np.allclose(fhi_pnp, fhi_pt.numpy(), atol=1e-9), "Poisson F~ torch≠numpy"
    _, fhi_nnp = nb_pit_bounds(mu_s, alp, ynr, smear=True)
    _, fhi_nt = nb_pit_bounds_t(torch.as_tensor(mu_s), torch.as_tensor(alp),
                                torch.as_tensor(ynr), smear=True)
    assert np.allclose(fhi_nnp, fhi_nt.numpy(), atol=1e-6), "NB F~ torch≠numpy"
    print("  [diag] F~ smear: integer-collapse OK, F~(k+0.5)=F(k) OK, numpy==torch OK")

    print("calibration_pit self-test: PASS")


if __name__ == "__main__":
    _selftest()
