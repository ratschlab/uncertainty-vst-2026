# src/training/losses.py
#
# Loss functions for gene expression prediction. A model spec in
# src/models/specs.py names one of these per model_type, and apply_spec_loss
# below is the one call site both pipelines use.
#
# ── Public API ────────────────────────────────────────────────────────────────
#
# Mean-only loss:
#   mse_loss(preds, targets)                     preds = a tensor of means
#
# Probabilistic losses:
#   gaussian_nll_loss(preds, targets, lam=0.0)   preds = (means, log_var)
#       lam is the beta-NLL exponent of Seitzer et al. 2022, applied as a detached
#       weight on the whole NLL, not as a weight on the variance term.
#       lam=0.0 is the plain Gaussian NLL, a proper likelihood.
#       Sweep lam across values in [0, 1] (e.g. 0.0, 0.5, 1.0) via config.yaml.
#   poisson_nll_loss(log_rate, targets)          log_rate = the raw log-rate tensor
#   nb_nll_loss(preds, targets)                  preds = (log_mu, log_alpha)
#
# ── Notes ─────────────────────────────────────────────────────────────────────
#
#   - Correlations are not losses. They are evaluation metrics and live in
#     src/evaluation/metrics_slide_aware.py.
#   - All losses expect preds and targets to be float32 tensors on the same
#     device.

import torch
import torch.nn.functional as F


# ── Mean-only losses ──────────────────────────────────────────────────────────

def mse_loss(preds, targets):
    """Standard mean squared error loss."""
    if targets.shape != preds.shape:
        targets = targets.view_as(preds)
    return F.mse_loss(preds, targets)


# ── Probabilistic losses ──────────────────────────────────────────────────────

def gaussian_nll_loss(preds, targets, lam=0.0, eps=1e-6, reduction="mean"):
    """
    Heteroscedastic / homoscedastic Gaussian negative log-likelihood loss,
    beta-NLL weighted (Seitzer et al. 2022, "On the Pitfalls of Heteroscedastic
    Uncertainty Estimation with Probabilistic Neural Networks").

    preds   : tuple (means, log_var), each of shape (batch,) or (batch, n_genes)
    targets : tensor of same shape as means
    lam     : the beta-NLL exponent beta. The detached var^lam multiplies the whole
              NLL, so it rescales that example's gradient; it is not a weight on
              the variance term.
              lam=0.0 -> plain Gaussian NLL, a proper likelihood.
              lam=1.0 -> the mean's effective gradient becomes (target-mean)
              instead of (target-mean)/var, which behaves like a variance-weighted
              MSE. Deliberately not a proper likelihood.
              Sweep across [0, 1] (e.g. 0.0, 0.5, 1.0) via config.yaml.

    reduction:
        'mean' → scalar (default, used by training loop)
        'none' → per-element tensor, same shape as targets (used by eval per-gene)

    Loss = stopgrad(var)^lam * [ (targets - means)^2 / (2*var) + log_var / 2 ],
    mean over batch. The stopgrad is required. Without it autograd differentiates
    through the var^lam factor too and adds a term on top of the intended
    reweighting; the self-test at the bottom of this file pins that failure mode.
    Detaching makes var^lam a fixed per-example weight, so d(loss)/d(theta) =
    stopgrad(var)^lam * d(NLL)/d(theta), a clean rescale of the ordinary NLL
    gradient and the only case the paper's derivation covers.
    """
    means, log_var = preds
    # Single-gene case: model heads squeeze to (batch,) but targets arrive as
    # (batch, 1) from SharedTensors. Without this reshape (targets - means)
    # broadcasts (B, 1) against (B,) into a (B, B) matrix, a silent bug that
    # averages B^2 cross terms and pulls every prediction toward the batch mean.
    # Every loss in this file applies the same guard.
    if targets.shape != means.shape:
        targets = targets.view_as(means)
    var = torch.exp(log_var) + eps   # prevents division by near-zero
    nll = ((targets - means) ** 2) / (2 * var) + log_var / 2
    if lam == 0.0:
        # Fast path: at lam=0 the weight is identically 1, so this is the plain NLL,
        # and it skips a redundant pow() and detach().
        loss = nll
    else:
        weight = var.detach() ** lam   # stopgrad -- value only, no gradient path
        loss = weight * nll
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'none':
        return loss
    else:
        raise ValueError(f"Unknown reduction: {reduction!r}")


def poisson_nll_loss(log_rate, targets, reduction="mean"):
    """
    Poisson negative log-likelihood with log-input parameterization.

    log_rate : tensor (batch,) or (batch, n_genes), the log-rate f(x), the raw
               output of PoissonMLP. Not exponentiated: passing the log-rate is
               torch's `log_input=True` convention and stays stable for counts
               spanning orders of magnitude. exp(f(x)) is mu on the target's own
               scale. There is no size factor and no offset, so the likelihood is
               fitted directly on whatever target the spec declares (cp_counts,
               cp_counts_nr, or raw_counts).
    targets  : same shape as log_rate. Counts on that same scale (float is fine).
    reduction: "mean" (default) or "none". Mirrors gaussian_nll_loss.

    Loss = mean over batch of: exp(log_rate) - targets * log_rate (Stirling term
    omitted via full=False; constant in log_rate, no gradient effect).
    """
    if targets.shape != log_rate.shape:
        targets = targets.view_as(log_rate)
    return F.poisson_nll_loss(
        log_rate, targets,
        log_input=True, full=False, reduction=reduction,
    )


def nb_nll_loss(preds, targets, reduction="mean"):
    """Negative Binomial negative log-likelihood (homoscedastic-dispersion variant).

    preds   : tuple whose first two elements are (log_mu, log_alpha), each of shape
              (batch,) or (batch, n_genes). Any further elements are ignored.
              exp(log_mu) is mu on the target's own scale: there is no size factor
              and no offset, the NB is fitted directly on whatever target the spec
              declares.
    targets : counts on that same scale, same shape as log_mu.
    reduction: "mean" (default) or "none".

    alpha = exp(log_alpha) is the dispersion of the NB fitted at that target's
    scale. It is a property of the fit, not a scale-free constant: an alpha fitted
    on CP_T at T=25000 is not comparable to one fitted at T=10000, and a
    raw-count-fitted alpha is comparable to neither. Mean and dispersion always
    travel together on one scale.

    Parameterization:
        var = mu + alpha*mu^2 ; total_count r = 1/alpha ; PyTorch logits =
        log(probs/(1-probs)) = log(alpha*mu) = log_mu + log_alpha. Passing logits
        (not probs) avoids the probs-vs-(1-probs) convention trap and is stable.
    """
    log_mu, log_alpha = preds[0], preds[1]
    if targets.shape != log_mu.shape:
        targets = targets.view_as(log_mu)
    alpha  = torch.exp(log_alpha)
    r      = 1.0 / alpha                    # total_count
    logits = log_mu + log_alpha             # = log(alpha*mu)
    # validate_args=False so the continuous cp_counts_nr target is accepted. For an
    # integer target this only disables a support check that already passes, so the
    # value is unchanged. For the continuous target, log_prob's lgamma normalizer is
    # the continuous NB form (edgeR/Smyth).
    dist = torch.distributions.NegativeBinomial(total_count=r, logits=logits, validate_args=False)
    nll = -dist.log_prob(targets)
    if reduction == "mean":
        return nll.mean()
    elif reduction == "none":
        return nll
    else:
        raise ValueError(f"Unknown reduction: {reduction!r}")



# -- Spec-driven loss application (shared by BOTH pipelines) -------------------

def apply_spec_loss(spec, raw, targets, *, lam=1.0):
    """Call spec.loss_fn with exactly the kwargs the spec needs.

    The single place that decides whether to pass `lam`, from spec.uses_lam. Both
    train loops and both inference paths go through here, so every loss is called
    identically in the classic and the GCM pipeline. No loss takes a size-factor
    offset or a per-spot weight: every count model fits its likelihood directly on
    its own target, so mu and alpha always live on that one target scale.
    `spec` is duck-typed, reading only .uses_lam and .loss_fn, so losses.py never
    imports specs.py, which would be a circular import.
    """
    kw = {}
    if spec.uses_lam:
        kw["lam"] = lam
    return spec.loss_fn(raw, targets, **kw)




# ─────────────────────────────────────────────────────────────────────────────
# Self-test. Checks that the count losses are fitted directly on their own target,
# with no size factor, offset or per-spot exposure: exp(raw) is mu on that scale,
# no loss accepts a log_s or weight kwarg, and apply_spec_loss adds nothing but
# `lam`.
#   Run: python -m src.training.losses
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import inspect

    torch.manual_seed(0)
    B, G = 64, 5
    f       = torch.randn(B, G)                       # log-rate / log-mean f(x)
    y       = torch.randint(0, 20, (B, G)).float()    # counts on the target's own scale
    log_a   = torch.randn(B, G)

    # (1) No loss accepts an offset or weight channel. This guard fires if a
    #     size-factor offset or a per-spot weight is reintroduced, whether in a
    #     loss signature or as a module-level helper.
    for fn in (poisson_nll_loss, nb_nll_loss, gaussian_nll_loss, mse_loss):
        params = set(inspect.signature(fn).parameters)
        assert "log_s" not in params, f"{fn.__name__} still takes log_s"
        assert "weight" not in params, f"{fn.__name__} still takes weight"
    assert "_add_offset" not in globals(), \
        "_add_offset is back: the size-factor offset must not exist any more"

    # (2) exp(raw) IS mu on the target's own scale — the Poisson NLL equals the
    #     textbook mu - y*log(mu) at mu = exp(f), with nothing rescaling f.
    mu = torch.exp(f)
    assert torch.allclose(poisson_nll_loss(f, y), (mu - y * f).mean(), atol=1e-5), \
        "Poisson NLL must be exp(f) - y*f, i.e. mu = exp(f) on the target's own scale"

    # (3) Same for the NB: mu = exp(log_mu), alpha = exp(log_alpha), both fitted at
    #     the target's scale (var = mu + alpha*mu^2, total_count 1/alpha).
    alpha = torch.exp(log_a)
    ref = -torch.distributions.NegativeBinomial(
        total_count=1.0 / alpha, logits=f + log_a, validate_args=False).log_prob(y).mean()
    assert torch.allclose(nb_nll_loss((f, log_a), y), ref, atol=1e-5), "NB NLL parameterization changed"

    # (4) Single-gene (1-D) path: n_genes=1 is a valid multi-gene case.
    f1, y1 = torch.randn(B), torch.randint(0, 20, (B,)).float()
    assert torch.isfinite(poisson_nll_loss(f1, y1)), "single-gene Poisson NLL not finite"

    # (5) apply_spec_loss passes ONLY lam, and only when the spec asks for it. Two
    #     tiny duck-typed stand-ins (it never imports specs).
    class _Spec:
        def __init__(self, uses_lam, loss_fn):
            self.uses_lam, self.loss_fn = uses_lam, loss_fn
    assert torch.allclose(apply_spec_loss(_Spec(False, nb_nll_loss), (f, log_a), y),
                          nb_nll_loss((f, log_a), y), atol=1e-6)
    assert torch.allclose(apply_spec_loss(_Spec(True, gaussian_nll_loss), (f, log_a), y, lam=0.3),
                          gaussian_nll_loss((f, log_a), y, lam=0.3), atol=1e-6)
    print("losses no-offset self-test: PASS")

    # ── beta-NLL self-test: lam is the Seitzer-et-al. exponent, stopgrad is load-bearing ──
    torch.manual_seed(1)
    Bg, Gg = 32, 4
    m_  = torch.randn(Bg, Gg, requires_grad=True)
    lv_ = torch.randn(Bg, Gg, requires_grad=True)
    yy  = torch.randn(Bg, Gg)

    # (a) lam=0.0 equals the plain Gaussian NLL exactly.
    v_ref   = torch.exp(lv_) + 1e-6
    nll_ref = (((yy - m_) ** 2) / (2 * v_ref) + lv_ / 2).mean()
    assert torch.allclose(gaussian_nll_loss((m_, lv_), yy, lam=0.0), nll_ref, atol=1e-7), \
        "lam=0.0 must equal the plain Gaussian NLL"

    # (b) the stopgrad is load-bearing: an UNDETACHED var**lam multiplier gives a
    #     DIFFERENT d(loss)/d(log_var) than the real (detached) loss, at lam != 0.
    #     (the mean gradient is unaffected either way, since var doesn't depend on m_.)
    loss_detached = gaussian_nll_loss((m_, lv_), yy, lam=0.7)
    g_m_detached, g_lv_detached = torch.autograd.grad(loss_detached, (m_, lv_))

    v_plain       = torch.exp(lv_) + 1e-6
    nll_plain     = ((yy - m_) ** 2) / (2 * v_plain) + lv_ / 2
    loss_nodetach = ((v_plain ** 0.7) * nll_plain).mean()   # same formula, NO detach
    g_m_nodetach, g_lv_nodetach = torch.autograd.grad(loss_nodetach, (m_, lv_))

    assert torch.allclose(g_m_detached, g_m_nodetach, atol=1e-6), \
        "mean gradient should match with/without detach (var doesn't depend on means)"
    assert not torch.allclose(g_lv_detached, g_lv_nodetach, atol=1e-6), \
        "without stopgrad, d(loss)/d(log_var) picks up an extra term from " \
        "differentiating through var**lam -- if this assert fires, the detach() " \
        "in gaussian_nll_loss is missing or a no-op"

    # (c) homoscedastic sanity check: if log_var is CONSTANT across the batch (the
    #     homoscedastic case), lam only rescales the log_var gradient's MAGNITUDE,
    #     never its sign -- i.e. no change to the argmin, only to step size.
    lv_shared_value = -0.3
    signs = []
    for lam_val in (0.0, 0.3, 0.7, 1.0):
        lv_i = torch.full((Bg, Gg), lv_shared_value, requires_grad=True)
        loss_i = gaussian_nll_loss((m_.detach(), lv_i), yy, lam=lam_val)
        (g_i,) = torch.autograd.grad(loss_i, lv_i)
        signs.append(torch.sign(g_i))
    assert all(torch.equal(signs[0], s) for s in signs[1:]), \
        "homoscedastic log_var gradient changed sign across lam values -- lam " \
        "should only rescale magnitude when var doesn't vary across the batch"

    print("beta-NLL self-test: PASS")