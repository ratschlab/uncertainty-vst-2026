# src/models/specs.py
#
# The single source of truth for everything that varies by model_type.
#
# Adding a model means one entry in SPECS below, plus a model class in mlp.py, a loss in
# losses.py, and a kind string in registry.py for the Snakemake routing. Everything else
# that varies by model is a ModelSpec field: the target source, whether the loss takes
# lambda, what the raw output means, what to write alongside the predictions.
#
# Training, inference and metrics read spec.* and never branch on the model_type string.

import inspect
from dataclasses import dataclass, field
from typing import Callable, List

import torch

from src.models.registry import MODEL_KINDS, kind_of, types_of_kind, COUNT_MODEL_TYPES  # re-export
from src.models.mlp import (
    MLPMeanOnly,
    HomoscedasticGaussianMLP,
    HeteroscedasticGaussianMLP,
    SpotVarianceGaussianMLP,
    PoissonMLP,
    NegativeBinomialMLP,
    NegativeBinomialHeteroFreeMLP,
    NegativeBinomialSpotDispMLP,
    NegativeBinomialHeteroSTImageMLP,
    # GCM per-gene readout descriptor + assemble() helpers (GCM-only; classic ignores)
    GcmHead,
    _gcm_assemble_mean,
    _gcm_assemble_vec_pair,
    _gcm_assemble_vec_genescalar,
    _gcm_assemble_vec_spotscalar,
)
from src.training.losses import (
    mse_loss,
    gaussian_nll_loss,
    poisson_nll_loss,
    nb_nll_loss,
)


# ─────────────────────────────────────────────────────────────────────────────
# ModelSpec
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelSpec:
    """All model-specific choices for the classic per-gene pipeline.

    The training loop, inference, and storage code all read these fields
    instead of branching on the model_type string.
    """

    name: str                       # "msemeanonly", "heteroscedastic", "poisson", ...

    # What is the model? --------------------------------------------------------
    model_class: type               # nn.Module subclass
    loss_fn: Callable               # (raw_output, targets, **kw) -> scalar tensor

    # How does it relate to the data and the loss? ------------------------------
    target_source: str
    # Which array read_arrays returns as the training target:
    #   "X"               log-CP10k (adata.X), the Gaussian axis
    #   "cp_counts"       round(CP_T), counts at the common depth T
    #   "cp_counts_nr"    CP_T, the same but not rounded, so continuous
    #   "raw_counts"      depth-blind integer counts
    # These four are all read_arrays accepts; any other value raises there.
    uses_lam: bool                  # does loss_fn take a `lam=` kwarg?
    is_probabilistic: bool          # does the model produce a predictive variance?
    loss_display_name: str          # pretty name for epoch logs

    # How do we read the model's raw output? ------------------------------------
    # extract_predictions takes the raw model output (whatever model(x) returned)
    # and returns a dict of tensors with these conventions:
    #   "means"   : (n_spots, n_genes) — always present
    #   "pred_var": (n_spots, n_genes) or None — present iff is_probabilistic
    #   plus optional extra keys named in `extra_prediction_columns`.
    extract_predictions: Callable

    # Behavioral knobs ----------------------------------------------------------
    # NB: early stopping uses corr_metrics[0] (the configured primary correlation
    # metric), resolved in train_loop_indexed.py — not a per-spec field.
    extra_prediction_columns: List[str] = field(default_factory=list)
    # extra layers to write to the predictions h5ad beyond the standard
    # (y, pred, pred_var). Each must be a key produced by extract_predictions.
    # Empty for the Gaussian and MSE heads; the NB heads add "dispersion".

    gcm_head: object = None
    # GCM only: the per-gene readout, declared as a GcmHead in mlp.py. The classic
    # pipeline ignores it, since its model_class fixes the head.
    # gcm_head=None means the model is not available in the GCM, and build raises.
    # This is the single source of truth for GCM support.


# ─────────────────────────────────────────────────────────────────────────────
# extract_predictions implementations.
# Each takes whatever model(x) returns and emits a uniform dict.
# Kept tiny and inline so the mapping raw-output -> dict is obvious.
# ─────────────────────────────────────────────────────────────────────────────

def _extract_mean_only(raw):
    # raw : (B,) or (B, n_genes) mean tensor
    return {"means": raw, "pred_var": None}


def _extract_gaussian(raw):
    # raw : (means, log_var) — both same shape
    means, log_var = raw
    return {"means": means, "pred_var": torch.exp(log_var)}


def _extract_poisson(raw):
    # raw : log_rate tensor (single head, log link). exp(log_rate) IS mu on the
    # target's own scale (no size factor, no offset). Poisson has mean ==
    # variance, both equal to exp(log_rate).
    rate = torch.exp(raw)
    return {"means": rate, "pred_var": rate}


def _extract_nb(raw):
    # raw : (log_mu, log_alpha[, anchor]) — uses only the first two. mu and alpha are
    # BOTH on the target's own scale (the scale the NB was fitted at). NB var =
    # mu + alpha*mu^2, so pred_var is on that same scale too.
    log_mu, log_alpha = raw[0], raw[1]
    mu, alpha = torch.exp(log_mu), torch.exp(log_alpha)
    return {"means": mu, "pred_var": mu + alpha * mu ** 2, "dispersion": alpha}

# ─────────────────────────────────────────────────────────────────────────────
# The registry. One entry per model_type.
# ─────────────────────────────────────────────────────────────────────────────

SPECS = {

    "msemeanonly": ModelSpec(
        name="msemeanonly",
        model_class=MLPMeanOnly,
        loss_fn=mse_loss,
        target_source="X",
        uses_lam=False,
        is_probabilistic=False,
        loss_display_name="MSE",
        extract_predictions=_extract_mean_only,
        gcm_head=GcmHead(1, assemble=_gcm_assemble_mean),
    ),

    "homoscedastic": ModelSpec(
        name="homoscedastic",
        model_class=HomoscedasticGaussianMLP,
        loss_fn=gaussian_nll_loss,   # lam = beta-NLL exponent; lam=0 = proper NLL (see losses.py)
        target_source="X",
        uses_lam=True,
        is_probabilistic=True,
        loss_display_name="NLL",
        extract_predictions=_extract_gaussian,
        gcm_head=GcmHead(1, n_gene_scalars=1, var_init_slots=(("gene_scalar", 0),),
                         assemble=_gcm_assemble_vec_genescalar),
    ),

    "heteroscedastic": ModelSpec(
        name="heteroscedastic",
        model_class=HeteroscedasticGaussianMLP,
        loss_fn=gaussian_nll_loss,   # lam = beta-NLL exponent; lam=0 = proper NLL (see losses.py)
        target_source="X",
        uses_lam=True,
        is_probabilistic=True,
        loss_display_name="NLL",
        extract_predictions=_extract_gaussian,
        gcm_head=GcmHead(2, var_init_slots=(("vector", 1),),
                         assemble=_gcm_assemble_vec_pair),
    ),

    # One predicted log_var per spot, broadcast over genes: the spot-axis analogue of
    # homoscedastic. Its gcm_head reads a per-spot scalar off the spot tower, so the
    # variance stays gene-independent and the model remains zero-shot in the GCM.
    "spot_homoscedastic": ModelSpec(
        name="spot_homoscedastic",
        model_class=SpotVarianceGaussianMLP,
        loss_fn=gaussian_nll_loss,   # lam = beta-NLL exponent; lam=0 = proper NLL (see losses.py)
        target_source="X",
        uses_lam=True,
        is_probabilistic=True,
        loss_display_name="NLL",
        extract_predictions=_extract_gaussian,
        gcm_head=GcmHead(1, n_spot_scalars=1, var_init_slots=(("spot_scalar", 0),),
                         assemble=_gcm_assemble_vec_spotscalar),
    ),

    # ── Common-depth count models — target_source "cp_counts" ────────────────────
    # The target is counts renormalized to one common depth T per dataset, then rounded:
    # round(CP_T). The likelihood is fitted directly on it, with no size factor and no
    # per-spot exposure, so mu and alpha both live on the CP_T scale and the predictive
    # carries no per-spot depth. It is therefore deployable from H&E alone.
    #
    # One constant is applied, and only to report the mean correlation: depth_scale =
    # T/1e4 divides the counts before log1p, which puts the reported axis back on
    # log1p(CP10k), comparable to the Gaussian models. Because alpha is fitted at one
    # specific T, alphas fitted at different T are not comparable.
    "poisson_cp": ModelSpec(
        name="poisson_cp",
        model_class=PoissonMLP,
        loss_fn=poisson_nll_loss,
        target_source="cp_counts",
        uses_lam=False,
        is_probabilistic=True,
        loss_display_name="PoissonNLL",
        extract_predictions=_extract_poisson,
        gcm_head=GcmHead(1, assemble=_gcm_assemble_mean),
    ),

    "negativebinomial_hetero_noanchor_cp": ModelSpec(
        name="negativebinomial_hetero_noanchor_cp",
        model_class=NegativeBinomialHeteroFreeMLP,
        loss_fn=nb_nll_loss,
        target_source="cp_counts",
        uses_lam=False,
        is_probabilistic=True,
        loss_display_name="NB-NLL",
        extract_predictions=_extract_nb,
        extra_prediction_columns=["dispersion"],   # per-(spot,gene) alpha
        gcm_head=GcmHead(2, disp_init_slots=(("vector", 1),),
                         assemble=_gcm_assemble_vec_pair),
    ),

    # ── Restricted-dispersion cp NB: the spot-axis and gene-axis analogues of the
    # one alpha per spot, or one per gene, on the same cp_counts target as the hetero
    # head above. The three together form a dispersion-grain ladder: per (spot, gene),
    # per spot, per gene.
    #
    # The analogy with the Gaussian heads is only partial, and that is the point.
    # Constraining alpha does not make the predictive variance constant, because
    # var = mu + alpha*mu^2 still varies through mu. So var_residual_corr is well
    # defined for both, unlike the gene-homoscedastic Gaussian, whose variance is
    # constant within a gene and whose var_residual_corr is all-NaN.
    "negativebinomial_spotdisp_cp": ModelSpec(
        name="negativebinomial_spotdisp_cp",
        model_class=NegativeBinomialSpotDispMLP,
        loss_fn=nb_nll_loss,
        target_source="cp_counts",
        uses_lam=False,
        is_probabilistic=True,
        loss_display_name="NB-NLL",
        extract_predictions=_extract_nb,
        extra_prediction_columns=["dispersion"],   # per-spot alpha (broadcast over genes)
        # GCM: log_mu = 1 readout vector; log_alpha = 1 per-SPOT scalar, started near
        # Poisson. Mirrors spot_homoscedastic's head with a dispersion init.
        gcm_head=GcmHead(1, n_spot_scalars=1, disp_init_slots=(("spot_scalar", 0),),
                         assemble=_gcm_assemble_vec_spotscalar),
    ),

    "negativebinomial_genedisp_cp": ModelSpec(
        name="negativebinomial_genedisp_cp",
        model_class=NegativeBinomialMLP,            # already one learnable log_alpha per gene
        loss_fn=nb_nll_loss,
        target_source="cp_counts",
        uses_lam=False,
        is_probabilistic=True,
        loss_display_name="NB-NLL",
        extract_predictions=_extract_nb,
        extra_prediction_columns=["dispersion"],   # per-gene alpha (broadcast over spots)
        # 1 readout vector plus 1 per-gene scalar, started near Poisson.
        gcm_head=GcmHead(1, n_gene_scalars=1, disp_init_slots=(("gene_scalar", 0),),
                         assemble=_gcm_assemble_vec_genescalar),
    ),

    # ── No-rounding (CONTINUOUS CP_T target) cp count models — target_source
    # "cp_counts_nr". Identical to their _cp twins except that the target keeps its
    # fractional part. A discrete Poisson/NB predictive is fitted to a continuous
    # target, and PIT calibration then uses the smeared CDF, which keeps P(Y=0) as a
    # real mass. Available in both pipelines.
    "poisson_cp_nr": ModelSpec(
        name="poisson_cp_nr",
        model_class=PoissonMLP,
        loss_fn=poisson_nll_loss,
        target_source="cp_counts_nr",
        uses_lam=False,
        is_probabilistic=True,
        loss_display_name="PoissonNLL",
        extract_predictions=_extract_poisson,
        gcm_head=GcmHead(1, assemble=_gcm_assemble_mean),   # available in GCM (mirror poisson_cp)
    ),

    "negativebinomial_hetero_noanchor_cp_nr": ModelSpec(
        name="negativebinomial_hetero_noanchor_cp_nr",
        model_class=NegativeBinomialHeteroFreeMLP,
        loss_fn=nb_nll_loss,
        target_source="cp_counts_nr",
        uses_lam=False,
        is_probabilistic=True,
        loss_display_name="NB-NLL",
        extract_predictions=_extract_nb,
        extra_prediction_columns=["dispersion"],   # per-(spot,gene) alpha
        gcm_head=GcmHead(2, disp_init_slots=(("vector", 1),),   # available in GCM (mirror _cp)
                         assemble=_gcm_assemble_vec_pair),
    ),

    # The _nr counterparts of the two restricted-dispersion heads, so the whole
    # dispersion-grain ladder can be run as matched cp / cp_nr pairs and the cost of
    # rounding can be attributed to a dispersion grain. Identical to the _cp twins in
    # every other respect.
    "negativebinomial_spotdisp_cp_nr": ModelSpec(
        name="negativebinomial_spotdisp_cp_nr",
        model_class=NegativeBinomialSpotDispMLP,
        loss_fn=nb_nll_loss,
        target_source="cp_counts_nr",
        uses_lam=False,
        is_probabilistic=True,
        loss_display_name="NB-NLL",
        extract_predictions=_extract_nb,
        extra_prediction_columns=["dispersion"],   # per-spot alpha (broadcast over genes)
        gcm_head=GcmHead(1, n_spot_scalars=1, disp_init_slots=(("spot_scalar", 0),),
                         assemble=_gcm_assemble_vec_spotscalar),
    ),

    "negativebinomial_genedisp_cp_nr": ModelSpec(
        name="negativebinomial_genedisp_cp_nr",
        model_class=NegativeBinomialMLP,            # already one learnable log_alpha per gene
        loss_fn=nb_nll_loss,
        target_source="cp_counts_nr",
        uses_lam=False,
        is_probabilistic=True,
        loss_display_name="NB-NLL",
        extract_predictions=_extract_nb,
        extra_prediction_columns=["dispersion"],   # per-gene alpha (broadcast over spots)
        gcm_head=GcmHead(1, n_gene_scalars=1, disp_init_slots=(("gene_scalar", 0),),
                         assemble=_gcm_assemble_vec_genescalar),
    ),

    # ── Depth-BLIND raw-count models — target_source "raw_counts" ────────────────
    # The third count family, mirroring the _cp entries exactly except for the
    # target: the likelihood is fitted directly on the raw integer counts, with no
    # size factor and no offset, so mu and alpha both live on the RAW scale and the
    # model is depth-blind (it must predict a spot's sequencing depth from H&E, which
    # it cannot). depth_scale is None for this family, so the reported correlation is
    # on log1p(raw counts): comparable to NEITHER the Gaussian models NOR the _cp
    # family, only to other _raw runs.
    "poisson_raw": ModelSpec(
        name="poisson_raw",
        model_class=PoissonMLP,
        loss_fn=poisson_nll_loss,
        target_source="raw_counts",
        uses_lam=False,
        is_probabilistic=True,
        loss_display_name="PoissonNLL",
        extract_predictions=_extract_poisson,
        gcm_head=GcmHead(1, assemble=_gcm_assemble_mean),
    ),

    "negativebinomial_hetero_noanchor_raw": ModelSpec(
        name="negativebinomial_hetero_noanchor_raw",
        model_class=NegativeBinomialHeteroFreeMLP,
        loss_fn=nb_nll_loss,
        target_source="raw_counts",
        uses_lam=False,
        is_probabilistic=True,
        loss_display_name="NB-NLL",
        extract_predictions=_extract_nb,
        extra_prediction_columns=["dispersion"],   # per-(spot,gene) alpha
        gcm_head=GcmHead(2, disp_init_slots=(("vector", 1),),
                         assemble=_gcm_assemble_vec_pair),
    ),

    # STimage benchmark: a bare Linear(in_dim, out_dim) head per (log_mu, log_alpha),
    # with no hidden layer, matching STimage's own per-gene Dense(2). Two variants share
    # that architecture and differ only in target_source:
    #   STI_nb_cp   cp_counts, the same target as negativebinomial_hetero_noanchor_cp, so
    #               the pair is a pure architecture comparison with the data held fixed.
    #   STI_nb_raw  raw_counts, depth-blind, reported on log1p(raw counts), so it compares
    #               only to other _raw runs.
    # Both are classic-only: gcm_head is None, because a bare per-gene readout is
    # undefined for a gene the model has never seen.
    "STI_nb_cp": ModelSpec(
        name="STI_nb_cp",
        model_class=NegativeBinomialHeteroSTImageMLP,
        loss_fn=nb_nll_loss,
        target_source="cp_counts",
        uses_lam=False,
        is_probabilistic=True,
        loss_display_name="NB-NLL",
        extract_predictions=_extract_nb,
        extra_prediction_columns=["dispersion"],
        gcm_head=None,
    ),

    "STI_nb_raw": ModelSpec(
        name="STI_nb_raw",
        model_class=NegativeBinomialHeteroSTImageMLP,
        loss_fn=nb_nll_loss,
        target_source="raw_counts",
        uses_lam=False,
        is_probabilistic=True,
        loss_display_name="NB-NLL",
        extract_predictions=_extract_nb,
        extra_prediction_columns=["dispersion"],
        gcm_head=None,
    ),
}


# Sanity check at import time: every model_type in registry.MODEL_KINDS has a
# corresponding ModelSpec entry, and vice versa. Catches typos in one but not
# the other.
_missing_specs = set(MODEL_KINDS) - set(SPECS)
_missing_kinds = set(SPECS) - set(MODEL_KINDS)
assert not _missing_specs, f"models in registry but not specs: {_missing_specs}"
assert not _missing_kinds, f"models in specs but not registry: {_missing_kinds}"


# The target_source values the scoring path treats as counts: g* for the correlation and
# the native Poisson/NB CDF for the PIT. The guard below and the router itself both read
# this tuple, so they cannot drift apart.
#
# Not the same thing as benchmark_summary.COUNT_TARGET_SOURCES, which holds the same
# values but asks a reading-side question of a saved run. Kept separate on purpose.
COUNT_ROUTED_TARGETS = ("raw_counts", "cp_counts", "cp_counts_nr")

# Loss and target must agree, checked at import so a bad pairing cannot be constructed.
#
# The scoring path routes on target_source alone, never on the loss. A count likelihood
# paired with a target outside COUNT_ROUTED_TARGETS would fall through to the generic
# path and be scored with Gaussian PIT bounds, reported under the same ece_pit key, with
# no error raised. The failure is silent, so it has to be caught at registration time.
for _name, _spec in SPECS.items():
    if (_spec.loss_fn in (poisson_nll_loss, nb_nll_loss)
            and _spec.target_source not in COUNT_ROUTED_TARGETS):
        raise ValueError(
            f"ModelSpec '{_name}' pairs a count likelihood "
            f"({_spec.loss_fn.__name__}) with target_source="
            f"{_spec.target_source!r}, which the scoring path does NOT route as a "
            f"count (routed: {COUNT_ROUTED_TARGETS}). Such a run would be silently "
            f"scored with Gaussian PIT bounds, and correlated on the wrong axis if "
            f"the target is a log scale. Either give it a routed count target or "
            f"give it a non-count loss."
        )

# registry.COUNT_MODEL_TYPES must list exactly the count-routed specs. This is the one
# place that can see both sets, so a count model added here but forgotten there fails at
# the first import.
_spec_count_types = {_n for _n, _s in SPECS.items() if _s.target_source in COUNT_ROUTED_TARGETS}
assert _spec_count_types == set(COUNT_MODEL_TYPES), (
    "registry.COUNT_MODEL_TYPES out of sync with specs count target_sources: "
    f"only in specs={sorted(_spec_count_types - set(COUNT_MODEL_TYPES))}, "
    f"only in registry={sorted(set(COUNT_MODEL_TYPES) - _spec_count_types)}"
)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_spec(model_type):
    """Return the ModelSpec for a model_type, or KeyError with a clear message."""
    if model_type not in SPECS:
        raise KeyError(
            f"Unknown model_type '{model_type}'. Known: {sorted(SPECS)}"
        )
    return SPECS[model_type]


def build_model(model_type, in_dim=3072, n_genes=1, device="cpu", arch=None,
                model_kwargs=None):
    """Instantiate a model from its spec.

    Parameters
    ----------
    arch : dict or None
        Optional architecture overrides (for hyperparameter search):
            {"hidden_dims": [...], "dropout": float|list, "activation": str}.
        When None, the model's built-in defaults are used, which reproduce the
        original architecture EXACTLY.
    model_kwargs : dict or None
        Optional model-SPECIFIC constructor kwargs (e.g. {"log_alpha_init": -2.0} for
        the NB models, {"log_mu_bias_init": log(depth_scale)} for the count models'
        mean head). Only the keys the
        target model's __init__ actually accepts are forwarded, so the same dict
        can be passed for every model_type (extra keys are ignored). None for
        models that take no extra kwargs.
    """
    spec = get_spec(model_type)
    kwargs = dict(in_dim=in_dim, out_dim=n_genes)
    if arch:
        # hidden_dims, dropout and activation are accepted by every model;
        # "normalization" only by the ones wired for it. Filter against the constructor
        # signature so a model that does not accept a key simply ignores it.
        accepted = (set(inspect.signature(spec.model_class.__init__).parameters)
                    if isinstance(spec.model_class, type) else None)
        for k in ("hidden_dims", "dropout", "activation", "normalization"):
            if k in arch and arch[k] is not None and (accepted is None or k in accepted):
                kwargs[k] = arch[k]
    # Forward model-specific kwargs, filtered to what this constructor accepts.
    # Skipped for stub model_classes (plain functions, not classes).
    if model_kwargs and isinstance(spec.model_class, type):
        accepted = set(inspect.signature(spec.model_class.__init__).parameters)
        kwargs.update({k: v for k, v in model_kwargs.items()
                       if v is not None and k in accepted})
    return spec.model_class(**kwargs).to(device)
