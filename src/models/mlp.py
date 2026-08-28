# src/models/mlp.py
#
# MLP architectures for gene expression prediction.
#
# ── File layout ───────────────────────────────────────────────────────────────
#   1. build_backbone: the shared block stack most models here are assembled from.
#   2. The classic-pipeline models: mean-only, three Gaussian variance shapes,
#      and the Poisson / Negative Binomial count heads. Each takes `in_dim`
#      (the spot-embedding width) and `out_dim` (the number of genes).
#   3. The GCM model: GeneHyperMLP and the GcmHead descriptor that declares a
#      model's per-gene readout.
#   4. Self-tests: python -m src.models.mlp

from dataclasses import dataclass
from typing import Callable, Tuple

import torch.nn as nn
import torch

# ─────────────────────────────────────────────────────────────────────────────
# Shared architecture helper.
#
# Builds [Linear -> LayerNorm -> activation -> Dropout] x len(hidden_dims).
# The defaults below are what a model uses when the caller passes no architecture
# override: hidden_dims=[512, 128], relu, dropout=[0.2, 0.1].
#
# `dropout` is one float applied to every hidden layer, or one value per layer.
# ─────────────────────────────────────────────────────────────────────────────

_ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU}

_DEFAULT_HIDDEN  = [512, 128]
_DEFAULT_DROPOUT = [0.2, 0.1]
_DEFAULT_ACT     = "relu"
# Normalization between Linear and activation. "none" drops the LayerNorm, which
# lets the architecture search treat normalization as an axis (see snake_search/).
_NORMALIZATIONS  = ("layernorm", "none")
_DEFAULT_NORM    = "layernorm"


def _norm_dropout(dropout, n_layers):
    if isinstance(dropout, (list, tuple)):
        assert len(dropout) == n_layers, (
            f"dropout list len {len(dropout)} != n hidden layers {n_layers}"
        )
        return list(dropout)
    return [float(dropout)] * n_layers


def build_backbone(in_dim, hidden_dims, dropout, activation, normalization=_DEFAULT_NORM):
    """Return (nn.Sequential backbone, last_hidden_dim).

    The backbone maps in_dim -> hidden_dims[-1] through repeated
    [Linear -> (LayerNorm) -> activation -> Dropout]. `normalization="none"`
    omits the LayerNorm.
    """
    if activation not in _ACTIVATIONS:
        raise ValueError(f"activation '{activation}' not in {list(_ACTIVATIONS)}")
    if normalization not in _NORMALIZATIONS:
        raise ValueError(f"normalization '{normalization}' not in {list(_NORMALIZATIONS)}")
    act_cls = _ACTIVATIONS[activation]
    drops = _norm_dropout(dropout, len(hidden_dims))

    layers, prev = [], in_dim
    for h, d in zip(hidden_dims, drops):
        block = [nn.Linear(prev, h)]
        if normalization == "layernorm":
            block.append(nn.LayerNorm(h))
        block += [act_cls(), nn.Dropout(d)]
        layers += block
        prev = h
    return nn.Sequential(*layers), prev


# Count models carry no size-factor offset and no per-spot exposure. Each fits its
# likelihood directly on its own target (cp_counts, cp_counts_nr or raw_counts), so
# the forwards below return log_mu with exp(log_mu) = mu on that target's own scale,
# and the fitted dispersion alpha lives on that same scale.


class MLPMeanOnly(nn.Module):
    """
    This model was optimized via random search over 60 configurations on the cluster on the gene SOX10,
    using 5-fold group CV with radius=0.75, and optimizing for mean val Pearson correlation.

    (lr, weight_decay, dropout, hidden_dims, scheduler, pca_dim, activation).
    
    Best config: lr=1e-4, weight_decay=1e-4, dropout=0.2, hidden_dims=[512,128],
    activation=ReLU, no PCA, scheduler=ReduceLROnPlateau.
    
    Mean val Pearson: 0.766, mean holdout Pearson: 0.795 (5-fold group CV, SOX10, radius=0.75).
    
    This mean-only version is the backbone used for subsequent Gaussian head extensions.
    """
    
    def __init__(self, in_dim=3072, out_dim=1, hidden_dims=None,
                 dropout=None, activation=_DEFAULT_ACT, normalization=_DEFAULT_NORM):
        super().__init__()
        hidden_dims = _DEFAULT_HIDDEN  if hidden_dims is None else hidden_dims
        dropout     = _DEFAULT_DROPOUT if dropout     is None else dropout
        # Only this model accepts `normalization`: it is the architecture search's
        # proxy model. The others take build_backbone's default LayerNorm. build_model
        # filters kwargs by signature, so wiring another model to the axis is one
        # kwarg plus one pass-through in that model's __init__.
        backbone, last = build_backbone(in_dim, hidden_dims, dropout, activation, normalization)
        self.net = nn.Sequential(backbone, nn.Linear(last, out_dim))
    def forward(self, x):
        return self.net(x).squeeze(-1)


class HeteroscedasticGaussianMLP(nn.Module):
    """
    Shared backbone → two heads predicting mean and log_var per spot.
    Variance is predicted per-spot (heteroscedastic).
    Extends MLPMeanOnly with a log_var head and NLL loss.
    """

    def __init__(self, in_dim=3072, out_dim=1, hidden_dims=None,
                 dropout=None, activation=_DEFAULT_ACT):
        super().__init__()
        hidden_dims = _DEFAULT_HIDDEN  if hidden_dims is None else hidden_dims
        dropout     = _DEFAULT_DROPOUT if dropout     is None else dropout
        self.backbone, last = build_backbone(in_dim, hidden_dims, dropout, activation)
        self.head_mean = nn.Sequential(
            nn.Linear(last, out_dim)
        )
        self.head_log_var = nn.Sequential(
            nn.Linear(last, out_dim)
        )
        # Zero bias plus tiny weights start log_var near 0, so every spot begins with
        # variance ~1. The NLL divides the squared error by the variance, so a wild
        # initial variance destabilizes the first epochs.
        nn.init.zeros_(self.head_log_var[-1].bias)
        nn.init.normal_(self.head_log_var[-1].weight, mean=0.0, std=0.01)

    def forward(self, x):
        shared  = self.backbone(x)
        mean    = self.head_mean(shared).squeeze(-1)
        log_var = self.head_log_var(shared).squeeze(-1)
        return mean, log_var


class HomoscedasticGaussianMLP(nn.Module):
    """
    Shared backbone → mean head + one learnable log_var parameter per gene.
    Variance is constant across all spots (homoscedastic).
    """
    def __init__(self, in_dim=3072, out_dim=1, hidden_dims=None,
                 dropout=None, activation=_DEFAULT_ACT):
        super().__init__()
        hidden_dims = _DEFAULT_HIDDEN  if hidden_dims is None else hidden_dims
        dropout     = _DEFAULT_DROPOUT if dropout     is None else dropout
        self.backbone, last = build_backbone(in_dim, hidden_dims, dropout, activation)
        self.head_mean = nn.Sequential(
            nn.Linear(last, out_dim)
        )

        # One variance per gene, constant across spots, so a single learnable
        # log-variance parameter per gene is enough.
        self.log_var = nn.Parameter(torch.zeros(out_dim))  # learnable log variance parameter

    def forward(self, x):
        shared  = self.backbone(x)
        mean    = self.head_mean(shared).squeeze(-1)
        log_var = self.log_var.expand_as(mean)  # same log variance for all samples, so we expand it a -> a,a,a,a,.. to match the mean vector size
        return mean, log_var


class SpotVarianceGaussianMLP(nn.Module):
    """Shared backbone, a per-spot mean head (out_dim=G), and one log_var channel
    per spot broadcast over genes.

    The spot-axis analogue of HomoscedasticGaussianMLP, which holds one learnable
    variance parameter per gene because the gene panel is a fixed enumerated set. A
    free per-spot parameter cannot do the same job: spots are the batch dimension and
    test spots are unseen. The variance is therefore predicted from x by a
    single-output head, which keeps it valid on held-out spots while sharing it
    across all genes of a spot.

    For n_genes=1 it coincides with HeteroscedasticGaussianMLP, whose variance head
    already emits one channel. The two differ only for multi-gene sets.
    """

    def __init__(self, in_dim=3072, out_dim=1, hidden_dims=None,
                 dropout=None, activation=_DEFAULT_ACT):
        super().__init__()
        hidden_dims = _DEFAULT_HIDDEN  if hidden_dims is None else hidden_dims
        dropout     = _DEFAULT_DROPOUT if dropout     is None else dropout
        self.backbone, last = build_backbone(in_dim, hidden_dims, dropout, activation)
        self.head_mean    = nn.Linear(last, out_dim)
        self.head_log_var = nn.Linear(last, 1)        # ONE variance channel per spot
        # Small-init log-var head -> variance starts near 1, like the heteroscedastic
        # Gaussian (prevents early training instability).
        nn.init.zeros_(self.head_log_var.bias)
        nn.init.normal_(self.head_log_var.weight, mean=0.0, std=0.01)

    def forward(self, x):
        shared  = self.backbone(x)
        mean    = self.head_mean(shared)                     # (B, G)
        log_var = self.head_log_var(shared).expand_as(mean)  # (B, 1) -> (B, G)
        return mean.squeeze(-1), log_var.squeeze(-1)         # (B,)/(B,G), like hetero


class PoissonMLP(nn.Module):
    """
    Mean-only MLP with a log-link head: predicts `log_rate`, the natural Poisson
    parameter. The expected count is exp(log_rate), and a Poisson has variance equal
    to its mean, so there is no separate variance head to learn.

    The training target is whichever count array the ModelSpec declares: `cp_counts`
    (round(CP_T)), `cp_counts_nr` (continuous CP_T), or `raw_counts` (the
    .layers["raw_counts"] of the preprocessed adata). The pipeline reads the right
    array automatically.

    The forward pass returns the raw, unbounded `log_rate` = f(x). No size-factor
    offset is applied anywhere, so exp(f(x)) is mu on the target's own scale, both in
    the loss and in the saved prediction. Exponentiation happens at two points only:
      (a) inside `poisson_nll_loss`, which takes the log-rate directly
          (`log_input=True`, for stability);
      (b) inside `_extract_poisson` in specs.py, which exps log_rate to produce
          `means = exp(f(x)) = mu` and the equal Poisson `pred_var`.
    Keeping log_rate raw avoids any exp() in the forward path, which matters for
    counts spanning orders of magnitude.

    `log_mu_bias_init` (optional) shifts the log-rate head's bias by that constant,
    on top of PyTorch's default random init; it does not overwrite it. train.py
    passes log(depth_scale) for the cp targets, so mu starts scaled by the common
    depth instead of at 1. Shifting rather than setting is what makes log(1.0) = 0.0
    a bit-exact no-op. None (the default) means no shift.
    """

    def __init__(self, in_dim=3072, out_dim=1, hidden_dims=None,
                 dropout=None, activation=_DEFAULT_ACT, log_mu_bias_init=None):
        super().__init__()
        hidden_dims = _DEFAULT_HIDDEN  if hidden_dims is None else hidden_dims
        dropout     = _DEFAULT_DROPOUT if dropout     is None else dropout
        backbone, last = build_backbone(in_dim, hidden_dims, dropout, activation)
        head = nn.Linear(last, out_dim)
        if log_mu_bias_init is not None:
            # Shift the default Linear bias, do not replace it: the random init is
            # preserved, and a 0.0 shift leaves the bias bit-identical.
            head.bias.data.add_(float(log_mu_bias_init))
        self.net = nn.Sequential(backbone, head)

    def forward(self, x):
        # Returns the log-rate f(x); exp(f(x)) = mu on the target's own scale.
        return self.net(x).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Negative Binomial: count models with overdispersion (var = mu + alpha*mu^2).
# The variants differ in the grain of the dispersion alpha, not in the variance,
# which always varies with mu:
#   NegativeBinomialMLP               alpha per gene, a learnable parameter.
#   NegativeBinomialHeteroFreeMLP     alpha per (spot, gene), from a head.
#   NegativeBinomialSpotDispMLP       alpha per spot, from a head, shared over genes.
#   NegativeBinomialHeteroSTImageMLP  alpha per (spot, gene), no hidden layer.
# All predict log_mu per (spot, gene), with exp(log_mu) = mu on the target's own
# scale (no size factor, no offset, as for Poisson above) and alpha fitted at that
# same scale. The (mu, alpha) -> PyTorch NB conversion lives in the loss (losses.py).
# ─────────────────────────────────────────────────────────────────────────────

class NegativeBinomialMLP(nn.Module):
    """Gene-dispersion NB: mu per (spot, gene), plus a single learnable log-alpha per
    gene, constant across spots. Constraining alpha does not make the variance
    constant: mu + alpha*mu^2 still varies with mu.

    forward returns (log_mu, log_alpha); the loss converts to PyTorch's NB
    (total_count = 1/alpha, logits = log_mu + log_alpha) and computes the NLL.

    `log_mu_bias_init` (optional) shifts the log_mu head's bias by that constant, on
    top of the default init; see PoissonMLP.
    """

    def __init__(self, in_dim=3072, out_dim=1, hidden_dims=None, dropout=None,
                 activation=_DEFAULT_ACT, log_alpha_init=-2.0, log_mu_bias_init=None):
        super().__init__()
        hidden_dims = _DEFAULT_HIDDEN  if hidden_dims is None else hidden_dims
        dropout     = _DEFAULT_DROPOUT if dropout     is None else dropout
        self.backbone, last = build_backbone(in_dim, hidden_dims, dropout, activation)
        self.head_log_mu = nn.Linear(last, out_dim)
        if log_mu_bias_init is not None:
            self.head_log_mu.bias.data.add_(float(log_mu_bias_init))   # SHIFT, not replace (see PoissonMLP)
        # Per-gene log-dispersion. log_alpha_init negative => small alpha => start near
        # Poisson (var ~ mu), letting the model add overdispersion as needed.
        self.log_alpha = nn.Parameter(torch.full((out_dim,), float(log_alpha_init)))

    def forward(self, x):
        # exp(log_mu) = mu on the target's own scale (no size factor, no offset).
        log_mu = self.head_log_mu(self.backbone(x)).squeeze(-1)
        # Right-aligned broadcast: (out_dim,) -> (B, out_dim) or (B,) for 1 gene.
        log_alpha = self.log_alpha.expand_as(log_mu)
        return log_mu, log_alpha


class NegativeBinomialHeteroFreeMLP(nn.Module):
    """Free-dispersion NB: log_mu and log_alpha both per (spot, gene), each from its
    own head, trained with the plain NB NLL and no lambda. Both pipelines use it,
    under the registry kind prob_no_lam.

    forward returns (log_mu, log_alpha); nb_nll_loss converts to PyTorch's NB
    (total_count = 1/alpha, logits = log_mu + log_alpha) and computes the NLL.

    `log_mu_bias_init` (optional) shifts the log_mu head's bias by that constant, on
    top of the default init; see PoissonMLP.
    """

    def __init__(self, in_dim=3072, out_dim=1, hidden_dims=None, dropout=None,
                 activation=_DEFAULT_ACT, log_alpha_init=-2.0, log_mu_bias_init=None):
        super().__init__()
        hidden_dims = _DEFAULT_HIDDEN  if hidden_dims is None else hidden_dims
        dropout     = _DEFAULT_DROPOUT if dropout     is None else dropout
        self.backbone, last = build_backbone(in_dim, hidden_dims, dropout, activation)
        self.head_log_mu    = nn.Linear(last, out_dim)
        if log_mu_bias_init is not None:
            self.head_log_mu.bias.data.add_(float(log_mu_bias_init))   # SHIFT, not replace (see PoissonMLP)
        self.head_log_alpha = nn.Linear(last, out_dim)
        # Tiny head weight plus a bias at log_alpha_init, so log_alpha starts near
        # log_alpha_init for every (spot, gene) (near-Poisson).
        nn.init.normal_(self.head_log_alpha.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.head_log_alpha.bias, float(log_alpha_init))

    def forward(self, x):
        shared    = self.backbone(x)
        log_mu    = self.head_log_mu(shared).squeeze(-1)      # exp = mu, target's own scale
        log_alpha = self.head_log_alpha(shared).squeeze(-1)   # per (spot,gene)
        return log_mu, log_alpha


class NegativeBinomialSpotDispMLP(nn.Module):
    """Spot-dispersion NB: mu per (spot, gene), and one log-alpha per spot broadcast
    over genes. The spot-axis analogue of NegativeBinomialMLP (one alpha per gene),
    and the count analogue of SpotVarianceGaussianMLP.

    Alpha is predicted rather than stored, for the same reason as there: spots are
    the batch dimension and test spots are unseen, so a free per-spot parameter
    cannot generalize. A single-output head keeps it valid on held-out spots while
    sharing it across all genes of a spot.

    Constraining the dispersion does not make the variance constant, and that is the
    point of the model. NB var = mu + alpha*mu^2 still varies per (spot, gene)
    through mu, so this restricts only where the overdispersion may come from, not
    whether the predictive variance is input-dependent. var_residual_corr is
    therefore well defined here, unlike for the gene-homoscedastic Gaussian.

    forward returns (log_mu, log_alpha); nb_nll_loss converts to PyTorch's NB
    (total_count = 1/alpha, logits = log_mu + log_alpha) and computes the NLL.

    `log_mu_bias_init` (optional) shifts the log_mu head's bias by that constant, on
    top of the default init; see PoissonMLP.
    """

    def __init__(self, in_dim=3072, out_dim=1, hidden_dims=None, dropout=None,
                 activation=_DEFAULT_ACT, log_alpha_init=-2.0, log_mu_bias_init=None):
        super().__init__()
        hidden_dims = _DEFAULT_HIDDEN  if hidden_dims is None else hidden_dims
        dropout     = _DEFAULT_DROPOUT if dropout     is None else dropout
        self.backbone, last = build_backbone(in_dim, hidden_dims, dropout, activation)
        self.head_log_mu    = nn.Linear(last, out_dim)
        if log_mu_bias_init is not None:
            self.head_log_mu.bias.data.add_(float(log_mu_bias_init))   # SHIFT, not replace (see PoissonMLP)
        self.head_log_alpha = nn.Linear(last, 1)      # ONE dispersion channel per spot
        # Same start as NegativeBinomialHeteroFreeMLP: tiny head weight plus a bias at
        # log_alpha_init, so every spot begins near-Poisson.
        nn.init.normal_(self.head_log_alpha.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.head_log_alpha.bias, float(log_alpha_init))

    def forward(self, x):
        shared    = self.backbone(x)
        log_mu    = self.head_log_mu(shared)                      # (B, G)
        log_alpha = self.head_log_alpha(shared).expand_as(log_mu)  # (B, 1) -> (B, G)
        return log_mu.squeeze(-1), log_alpha.squeeze(-1)          # (B,)/(B,G), like hetero


class NegativeBinomialHeteroSTImageMLP(nn.Module):
    """Free-dispersion NB with no hidden layers, reproducing STimage's published head
    shape: a bare Dense(2) per gene directly on the pooled backbone feature, with no
    hidden layer, dropout, activation or normalization. In STimage's
    CNN_NB_multiple_genes the ResNet50 backbone is frozen, so that per-gene Dense(2)
    is the only trainable weight tensor in their model.

    Here the frozen CNN is replaced by a precomputed, frozen image embedding (this
    pipeline's own patch embeddings, not STimage's ResNet50), so the equivalent head
    is `nn.Linear(in_dim, out_dim)` straight off that embedding, for both log_mu and
    log_alpha. The constructor deliberately takes no build_backbone call and no
    hidden_dims / dropout / activation parameter: build_model in specs.py forwards
    architecture keys only to constructors that accept them, so omitting them makes
    this class immune to the shared `architecture:` config block, which would
    otherwise inject the pipeline's usual hidden backbone into every model trained
    under that config.

    Identical in contract to NegativeBinomialHeteroFreeMLP otherwise: log_mu and
    log_alpha both per (spot, gene), plain NB NLL, same forward signature, same
    near-Poisson dispersion init.

    forward returns (log_mu, log_alpha); nb_nll_loss converts to PyTorch's NB
    (total_count = 1/alpha, logits = log_mu + log_alpha) and computes the NLL.
    """

    def __init__(self, in_dim=3072, out_dim=1, log_alpha_init=-2.0,
                 log_mu_bias_init=None):
        super().__init__()
        self.head_log_mu    = nn.Linear(in_dim, out_dim)
        if log_mu_bias_init is not None:
            self.head_log_mu.bias.data.add_(float(log_mu_bias_init))   # SHIFT, not replace (see PoissonMLP)
        self.head_log_alpha = nn.Linear(in_dim, out_dim)
        # Same near-Poisson start as NegativeBinomialHeteroFreeMLP: tiny head weight
        # plus a bias at log_alpha_init, so log_alpha begins there for every spot.
        nn.init.normal_(self.head_log_alpha.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.head_log_alpha.bias, float(log_alpha_init))

    def forward(self, x):
        log_mu    = self.head_log_mu(x).squeeze(-1)      # exp = mu, target's own scale
        log_alpha = self.head_log_alpha(x).squeeze(-1)   # per (spot,gene)
        return log_mu, log_alpha


# ─────────────────────────────────────────────────────────────────────────────
# Batched gene-conditioned model (GeneHyperMLP).
# ─────────────────────────────────────────────────────────────────────────────
# Instead of running a full MLP per (spot, gene) pair, this factorizes the
# computation into two towers combined by a bilinear readout:
#
#   spot backbone  phi : spot_dim -> D              (run once per spot)
#   gene hypernet  g   : gene_dim -> readout params  (run once per gene)
#   prediction     y_hat[s, j] = q_s . w_j / sqrt(D) + b_j
#
# where q_s = phi(x_s) and (w_j, b_j, ...) = g(e_j). A batch is B spots by G
# genes; the spot backbone runs B times, the hypernet G times, and the whole
# (B, G) block is one matmul, against B*G full forwards for the per-pair model.
#
# Zero-shot is preserved: the readout head w_j is generated from the gene
# embedding e_j by the shared hypernet g, so a gene never seen in training still
# gets a valid head from its embedding. None of g's parameter shapes depend on the
# number of genes; only the gene-embedding matrix you pass does.


@dataclass
class GcmHead:
    """Declarative per-gene readout for the batched GCM (GeneHyperMLP).

    Defined here rather than in specs.py: specs.py imports mlp.py to attach GcmHeads
    to its ModelSpecs, so the reverse import would be circular. Only the GCM consumes
    it; the classic pipeline ignores it, since its model class fixes the head. This is
    what makes GeneHyperMLP model-type agnostic: a new model is a GcmHead in its spec,
    with no change to GeneHyperMLP.

    Per gene, the gene hypernetwork emits `n_vectors` readout vectors (each width D)
    and `n_gene_scalars` scalars; a separate spot-side head emits `n_spot_scalars`
    per-spot scalars, which are gene-independent, so zero-shot is preserved.
    `assemble(vecs, gscalars, sscalars)`, each argument a list of (B,G) tensors,
    returns the same raw object the classic model_class.forward returns, so
    spec.extract_predictions and spec.loss_fn consume it unchanged.

    *_init_slots mark slots whose output starts at a constant: the generating rows
    are zeroed and the bias is set. var_init_slots -> 0 (Gaussian var = 1),
    disp_init_slots -> log_alpha_init (NB dispersion near Poisson). A slot is a
    ("vector" | "gene_scalar" | "spot_scalar", index) tuple. The mean / log_mu
    vector (slot 0) is never init-listed, so it keeps the default init and learns.
    """
    n_vectors: int
    n_gene_scalars: int = 0
    n_spot_scalars: int = 0
    var_init_slots:  Tuple = ()
    disp_init_slots: Tuple = ()
    assemble: Callable = None


# assemble() implementations: turn the GeneHyperMLP readout outputs into the raw
# object spec.extract_predictions / spec.loss_fn expect, which is the same shape the
# classic model_class.forward returns. Shared across the Gaussian and NB variants
# that have the same head shape: hetero Gaussian and NB-hetero both return (v0, v1).

def _gcm_assemble_mean(vecs, gscalars, sscalars):          # mse / poisson
    return vecs[0]


def _gcm_assemble_vec_pair(vecs, gscalars, sscalars):      # gaussian-hetero / NB-hetero-noanchor
    return vecs[0], vecs[1]


def _gcm_assemble_vec_genescalar(vecs, gscalars, sscalars):  # gaussian-homo / NB-homo-dispersion
    return vecs[0], gscalars[0]


def _gcm_assemble_vec_spotscalar(vecs, gscalars, sscalars):  # spot_homoscedastic
    return vecs[0], sscalars[0]


def _build_hypernet(in_dim, hidden_dims, out_dim, dropout, activation):
    """Gene hypernetwork g: gene embedding (in_dim) -> readout params (out_dim).

    LayerNorm on the input keeps the (fixed, external) gene embeddings on a stable
    scale; then the standard block stack, then a final Linear. `hidden_dims=[]`
    makes it `LayerNorm -> Linear` — a pure bilinear readout (no hypernetwork
    depth). The final Linear keeps PyTorch's default init; GeneHyperMLP zeroes the
    log-var output rows after construction so variance starts at 1."""
    body, last = build_backbone(in_dim, list(hidden_dims), dropout, activation)
    final = nn.Linear(last, out_dim)
    return nn.Sequential(nn.LayerNorm(in_dim), body, final)


class GeneHyperMLP(nn.Module):
    """Batched gene-conditioned MLP with a hypernetwork gene-head generator.

    forward(spot_x, gene_e):
        spot_x : (B, spot_dim)     gene_e : (G, gene_dim)
        returns whatever this model's GcmHead.assemble builds, which is the same raw
        object the classic model_class.forward returns (a (B,G) tensor, or a tuple
        of (B,G) tensors), so spec.extract_predictions / spec.loss_fn consume it
        unchanged.

    A GcmHead from the ModelSpec declares the per-gene readout; nothing here is
    hard-coded per model. The gene hypernetwork emits, per gene,
        head.n_vectors      readout vectors of width D -> (B,G) each (q·w/√D + b)
        head.n_gene_scalars per-gene scalars            -> broadcast over spots
    and a separate spot-side head emits
        head.n_spot_scalars per-spot scalars            -> broadcast over genes
    which are gene-independent, so zero-shot is preserved. head.assemble(vecs,
    gscalars, sscalars) arranges them into the raw output. Adding a model is a
    GcmHead in its spec; nothing changes here.

    The readout is 1/√D-scaled (attention-style) so q·w is well-conditioned for any
    interaction dim D. Variance / dispersion slots start at a constant via
    head.var_init_slots (-> 0, var=1) / head.disp_init_slots (-> log_alpha_init).
    """

    def __init__(self, spot_dim, gene_dim, hidden_dims, dropout, activation,
                 hyper_hidden, hyper_dropout, head, interaction_dim=None,
                 log_alpha_init=-2.0):
        super().__init__()
        self.head = head

        # Spot tower: spot_dim -> last_hidden; optional projection to a decoupled D.
        self.spot_backbone, last = build_backbone(spot_dim, hidden_dims, dropout, activation)
        self.D = int(interaction_dim) if interaction_dim else last
        self.spot_proj = nn.Identity() if self.D == last else nn.Linear(last, self.D)
        D = self.D

        # Per-SPOT scalar head (gene-independent -> preserves zero-shot), used by the
        # "spot" variance head. None when the model declares no spot scalars.
        self.spot_head = nn.Linear(D, head.n_spot_scalars) if head.n_spot_scalars else None

        # Gene hypernetwork output per gene: n_vectors readout vectors (each D weight
        # cols + 1 bias col, contiguous), then n_gene_scalars per-gene scalars:
        #   vector v      : weight cols [v*(D+1) : v*(D+1)+D] , bias col [v*(D+1)+D]
        #   gene scalar s : col [n_vectors*(D+1) + s]
        out_dim = head.n_vectors * (D + 1) + head.n_gene_scalars
        self.gene_hyper = _build_hypernet(gene_dim, hyper_hidden, out_dim,
                                          hyper_dropout, activation)

        # Init variance/dispersion slots to a constant (var=1 / near-Poisson). The
        # mean / log_mu vector (slot 0) is never listed, so it keeps the default init.
        with torch.no_grad():
            for slot in head.var_init_slots:
                self._init_slot(slot, 0.0)
            for slot in head.disp_init_slots:
                self._init_slot(slot, float(log_alpha_init))

    def _init_slot(self, slot, const):
        """Make slot's readout START at `const`: zero the rows that generate it (so
        the slot ignores q at init) and set the bias-generating row to `const`.
        slot = (kind, idx), kind in {"vector","gene_scalar","spot_scalar"}."""
        kind, idx = slot
        D = self.D
        final = self.gene_hyper[-1]
        if kind == "vector":
            off = idx * (D + 1)
            final.weight[off:off + D + 1].zero_()   # D weight-gen rows + 1 bias-gen row
            final.bias[off:off + D + 1].zero_()
            final.bias[off + D] = const             # vector's bias -> const (weights stay 0)
        elif kind == "gene_scalar":
            col = self.head.n_vectors * (D + 1) + idx
            final.weight[col].zero_()
            final.bias[col] = const
        elif kind == "spot_scalar":
            self.spot_head.weight[idx].zero_()
            self.spot_head.bias[idx] = const
        else:
            raise ValueError(f"unknown GcmHead slot kind {kind!r}")

    def forward(self, spot_x, gene_e):
        q = self.spot_proj(self.spot_backbone(spot_x))   # (B, D)
        p = self.gene_hyper(gene_e)                       # (G, out_dim)
        D = self.D
        scale = D ** 0.5
        H = self.head

        # Per-gene readout vectors -> (B, G) each (1/√D-scaled bilinear + bias).
        vecs = []
        for v in range(H.n_vectors):
            off = v * (D + 1)
            w = p[:, off:off + D]                         # (G, D)
            b = p[:, off + D]                             # (G,)
            vecs.append((q @ w.t()) / scale + b)          # (B, G)

        # Per-gene scalars (q-independent) broadcast over spots -> (B, G).
        base = H.n_vectors * (D + 1)
        gscalars = [p[:, base + s].unsqueeze(0).expand_as(vecs[0])
                    for s in range(H.n_gene_scalars)]

        # Per-spot scalars from q (gene-independent) broadcast over genes -> (B, G).
        sscalars = []
        if self.spot_head is not None:
            sp = self.spot_head(q)                        # (B, n_spot_scalars)
            sscalars = [sp[:, s:s + 1].expand_as(vecs[0])
                        for s in range(H.n_spot_scalars)]

        return H.assemble(vecs, gscalars, sscalars)


def build_gene_conditioned_model(
    model_type,
    spot_dim,
    gene_dim,
    device,
    *,
    arch=None,
    hyper_hidden=(256,),
    hyper_dropout=0.0,
    interaction_dim=None,
    head=None,
    log_alpha_init=-2.0,
):
    """Build the batched gene-conditioned model (GeneHyperMLP) on `device`.

    Parameters
    ----------
    model_type : str
        Used only in the error message below. `head` fully determines the readout
        structure, so this builder stays model-type agnostic like the classic one.
    head : GcmHead
        The per-gene readout descriptor (spec.gcm_head). None => the model is NOT
        available in the GCM pipeline (the STimage heads, whose bare per-gene readout
        has no counterpart for an unseen gene
        set) -> raise a clear error.
    spot_dim, gene_dim : int
        Spot-embedding dim (e.g. 3072) and gene-embedding dim (e.g. 512 network).
    arch : dict or None
        Sizes the SPOT backbone — {"hidden_dims", "dropout", "activation"} — the
        SAME block the classic pipeline uses. None -> the module defaults. The
        backbone's last hidden width is the interaction dim D (unless overridden).
    hyper_hidden : sequence of int
        Hidden dims of the gene hypernetwork. () / [] -> a bilinear readout.
    hyper_dropout : float
        Dropout inside the hypernetwork.
    interaction_dim : int or None
        Optional override of D (the spot↔gene readout dim). None -> backbone last hidden.
    log_alpha_init : float
        Initial dispersion (log α) for NB heads' disp_init_slots (near-Poisson start).
    """
    if head is None:
        raise ValueError(
            f"build_gene_conditioned_model: model_type '{model_type}' has no GcmHead "
            f"(spec.gcm_head is None) -> it is not supported in the GCM pipeline. "
            f"E.g. the STimage heads: a bare per-gene readout is undefined for a "
            f"variable / zero-shot gene set."
        )
    hidden_dims = list(arch["hidden_dims"]) if arch and arch.get("hidden_dims") else list(_DEFAULT_HIDDEN)
    dropout     = arch["dropout"] if arch and arch.get("dropout") is not None else _DEFAULT_DROPOUT
    activation  = arch["activation"] if arch and arch.get("activation") else _DEFAULT_ACT

    model = GeneHyperMLP(
        spot_dim, gene_dim, hidden_dims, dropout, activation,
        hyper_hidden=list(hyper_hidden), hyper_dropout=hyper_dropout,
        head=head, interaction_dim=interaction_dim, log_alpha_init=log_alpha_init,
    )
    return model.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Self-tests, checked against constructed expectations. CPU only, no cluster:
#   python -m src.models.mlp
# ─────────────────────────────────────────────────────────────────────────────

def _genehyper_selftest():
    torch.manual_seed(0)
    B, G, spot_dim, gene_dim = 8, 6, 32, 12
    LA = -2.0   # log_alpha_init for the NB dispersion slots

    # name -> GcmHead, covering every readout shape: mean-only, the three Gaussian
    # variance heads, and the three NB dispersion heads.
    HEADS = {
        "mean":      GcmHead(1, assemble=_gcm_assemble_mean),
        "hetero":    GcmHead(2, var_init_slots=(("vector", 1),), assemble=_gcm_assemble_vec_pair),
        "homo":      GcmHead(1, n_gene_scalars=1, var_init_slots=(("gene_scalar", 0),),
                             assemble=_gcm_assemble_vec_genescalar),
        "spot":      GcmHead(1, n_spot_scalars=1, var_init_slots=(("spot_scalar", 0),),
                             assemble=_gcm_assemble_vec_spotscalar),
        "nb_homo":   GcmHead(1, n_gene_scalars=1, disp_init_slots=(("gene_scalar", 0),),
                             assemble=_gcm_assemble_vec_genescalar),
        "nb_hetero": GcmHead(2, disp_init_slots=(("vector", 1),), assemble=_gcm_assemble_vec_pair),
        # A spot scalar addressed by a dispersion init: the one slot/init pairing the
        # other heads do not cover, and the one negativebinomial_spotdisp_cp needs.
        "nb_spot":   GcmHead(1, n_spot_scalars=1, disp_init_slots=(("spot_scalar", 0),),
                             assemble=_gcm_assemble_vec_spotscalar),
    }
    x = torch.randn(B, spot_dim)
    e = torch.randn(G, gene_dim)

    for name, head in HEADS.items():
        m = GeneHyperMLP(spot_dim, gene_dim, [16, 8], 0.0, "relu",
                         hyper_hidden=[16], hyper_dropout=0.0, head=head,
                         log_alpha_init=LA).eval()
        out = m(x, e)
        returns_tuple = isinstance(out, tuple)
        means = out[0] if returns_tuple else out
        assert means.shape == (B, G), f"{name}: bad means shape {means.shape}"

        if returns_tuple:
            second = out[1]
            assert second.shape == (B, G), f"{name}: bad second-output shape"
            # var slots start at 0 (var=1); disp slots start at log_alpha_init.
            if head.var_init_slots:
                assert torch.allclose(second, torch.zeros_like(second), atol=1e-6), \
                    f"{name}: var slot should start at 0 (var=1)"
            if head.disp_init_slots:
                assert torch.allclose(second, torch.full_like(second, LA), atol=1e-6), \
                    f"{name}: dispersion slot should start at log_alpha_init"
            if name == "homo":
                assert torch.allclose(second[0], second[1]), "homo var must be constant over spots"
            if name == "spot":
                # perturb the spot var head off its zero init, then check the
                # defining property: constant across GENES, varies across SPOTS.
                with torch.no_grad():
                    nn.init.normal_(m.spot_head.weight, std=1.0)
                    m.spot_head.bias.fill_(0.3)
                lv = m(x, e)[1]
                assert torch.allclose(lv[:, :1].expand_as(lv), lv, atol=1e-6), \
                    "spot var must be constant across genes"
                assert not torch.allclose(lv[0], lv[1], atol=1e-6), \
                    "spot var should vary across spots"

        # Permutation-equivariance: permuting genes permutes output columns only.
        # (means come from gene-dependent vectors, so this holds for every head.)
        perm = torch.randperm(G)
        out_p = m(x, e[perm])
        means_p = out_p[0] if isinstance(out_p, tuple) else out_p
        assert torch.allclose(means[:, perm], means_p, atol=1e-4), \
            f"{name}: not permutation-equivariant over genes"
    print("GeneHyperMLP self-test: PASS")


def _likelihood_models_selftest():
    """CPU checks for the count models and their losses.
        python -m src.models.mlp
    Verifies shapes, that no forward accepts an offset channel, that
    log_mu_bias_init lands on the mean head as a shift, that the per-gene and
    per-spot NB dispersions have the grain they promise, and the NB -> Poisson
    limit as alpha -> 0.
    """
    import inspect
    import math
    from src.training.losses import (
        poisson_nll_loss, nb_nll_loss,
    )
    torch.manual_seed(0)
    B, G, in_dim = 16, 4, 32
    x = torch.randn(B, in_dim)
    counts = torch.randint(0, 20, (B, G)).float()
    yX = torch.randn(B, G)

    # ── model(x) is the whole contract: no forward accepts a size-factor argument.
    for cls in (MLPMeanOnly, HeteroscedasticGaussianMLP, HomoscedasticGaussianMLP,
                SpotVarianceGaussianMLP, PoissonMLP, NegativeBinomialMLP,
                NegativeBinomialHeteroFreeMLP,
                NegativeBinomialHeteroSTImageMLP):
        assert "log_s" not in inspect.signature(cls.forward).parameters, \
            f"{cls.__name__}.forward still takes log_s"

    # ── Poisson: exp(f(x)) IS mu on the target's own scale, no offset anywhere. ──
    pm = PoissonMLP(in_dim=in_dim, out_dim=G, hidden_dims=[16], dropout=0.0).eval()
    r0 = pm(x)
    assert r0.shape == (B, G)
    assert torch.isfinite(poisson_nll_loss(r0, counts)), "poisson nll not finite"

    # log_mu_bias_init shifts the log-rate head's bias by exactly that constant, on
    # top of the default random init (train.py passes log(depth_scale)). Seed-paired
    # construction: the two models are identical except for the shift.
    bias_init = math.log(2.5)
    def _seeded_poisson(shift):
        torch.manual_seed(1234)
        return PoissonMLP(in_dim=in_dim, out_dim=G, hidden_dims=[16], dropout=0.0,
                          log_mu_bias_init=shift).eval()
    pm_0, pm_b = _seeded_poisson(None), _seeded_poisson(bias_init)
    assert torch.allclose(pm_b.net[-1].bias, pm_0.net[-1].bias + bias_init), \
        "log_mu_bias_init must SHIFT the log-rate head bias, not overwrite it"
    assert torch.allclose(pm_b(x), pm_0(x) + bias_init, atol=1e-5), \
        "log_mu_bias_init must shift the log-rate by exactly that constant"

    # depth_scale 1.0 gives log(1.0) = 0.0, and that must be a bit-exact no-op:
    # passing a zero shift has to leave the head bias exactly as passing nothing.
    pm_z = _seeded_poisson(0.0)
    assert torch.equal(pm_z.net[-1].bias, pm_0.net[-1].bias), \
        "log_mu_bias_init=0.0 must leave the head bias bit-identical to no shift"

    def _seeded_nb(shift):
        torch.manual_seed(1234)
        return NegativeBinomialHeteroFreeMLP(in_dim=in_dim, out_dim=G, hidden_dims=[16],
                                             dropout=0.0, log_mu_bias_init=shift).eval()
    nb_0, nb_b, nb_z = _seeded_nb(None), _seeded_nb(bias_init), _seeded_nb(0.0)
    assert torch.allclose(nb_b.head_log_mu.bias, nb_0.head_log_mu.bias + bias_init), \
        "log_mu_bias_init must SHIFT the NB log_mu head bias"
    assert torch.equal(nb_z.head_log_mu.bias, nb_0.head_log_mu.bias), \
        "log_mu_bias_init=0.0 must be a bit-exact no-op for the NB log_mu head"

    # ── NB homoscedastic: per-gene alpha + NB->Poisson limit ──────────────────
    nb = NegativeBinomialMLP(in_dim=in_dim, out_dim=G, hidden_dims=[16],
                             dropout=0.0, log_alpha_init=-2.0).eval()
    log_mu, log_alpha = nb(x)
    assert log_mu.shape == log_alpha.shape == (B, G)
    assert torch.allclose(log_alpha[0], log_alpha[1]), "NB homo alpha must be per-gene"
    assert torch.isfinite(nb_nll_loss((log_mu, log_alpha), counts)), "nb nll not finite"
    # alpha -> 0 (log_alpha very negative) -> NB NLL == Poisson NLL. Use float64 to
    # avoid float32 cancellation in lgamma(count + 1/alpha).
    log_mu_d = log_mu.detach().double()
    counts_d = counts.double()
    nb_lim   = nb_nll_loss((log_mu_d, torch.full((B, G), -12.0, dtype=torch.float64)), counts_d)
    poi_full = -torch.distributions.Poisson(rate=torch.exp(log_mu_d)).log_prob(counts_d).mean()
    assert torch.allclose(nb_lim, poi_full, atol=1e-3), f"NB->Poisson limit off: {nb_lim} vs {poi_full}"

    # ── NB spot-dispersion: alpha constant ACROSS GENES, varying across spots ─────
    nbs = NegativeBinomialSpotDispMLP(in_dim=in_dim, out_dim=G, hidden_dims=[16],
                                      dropout=0.0, log_alpha_init=-2.0).eval()
    log_mu_s, log_alpha_s = nbs(x)
    assert log_mu_s.shape == log_alpha_s.shape == (B, G)
    assert torch.allclose(log_alpha_s[:, :1].expand_as(log_alpha_s), log_alpha_s, atol=1e-6), \
        "NB spot-dispersion alpha must be constant across genes"
    assert not torch.allclose(log_alpha_s[0, 0], log_alpha_s[1, 0]), \
        "NB spot-dispersion alpha must vary across spots"
    assert torch.isfinite(nb_nll_loss((log_mu_s, log_alpha_s), counts)), "nb spotdisp nll not finite"
    # The mean head takes the same bias SHIFT as every other count head, so 0.0 is a no-op.
    torch.manual_seed(0)
    nbs_a = NegativeBinomialSpotDispMLP(in_dim=in_dim, out_dim=G, hidden_dims=[16], dropout=0.0)
    torch.manual_seed(0)
    nbs_b = NegativeBinomialSpotDispMLP(in_dim=in_dim, out_dim=G, hidden_dims=[16], dropout=0.0,
                                        log_mu_bias_init=0.0)
    assert torch.equal(nbs_a.head_log_mu.bias, nbs_b.head_log_mu.bias), \
        "log_mu_bias_init=0.0 must be a bit-exact no-op for the NB spot-dispersion head"

    print("likelihood-models self-test: PASS")


def _spot_homoscedastic_selftest():
    """CPU check for the classic per-spot shared-variance Gaussian.
        python -m src.models.mlp
    Verifies the (B,)/(B,G) shape convention and the defining property: log_var is
    constant across GENES but varies across SPOTS."""
    torch.manual_seed(0)
    B, in_dim = 8, 32
    x = torch.randn(B, in_dim)
    # multi-gene: one predicted variance per spot, shared across the gene axis
    m = SpotVarianceGaussianMLP(in_dim=in_dim, out_dim=5, hidden_dims=[16],
                                dropout=0.0).eval()
    with torch.no_grad():   # perturb the var head off its near-zero init so spots differ
        nn.init.normal_(m.head_log_var.weight, std=1.0); m.head_log_var.bias.fill_(0.2)
    mean, log_var = m(x)
    assert mean.shape == (B, 5) and log_var.shape == (B, 5), "bad multi-gene shapes"
    assert torch.allclose(log_var[:, :1].expand_as(log_var), log_var, atol=1e-6), \
        "spot variance must be constant across genes"
    assert not torch.allclose(log_var[0], log_var[1], atol=1e-6), \
        "spot variance should vary across spots"
    # single-gene: squeezes to (B,), coinciding with the heteroscedastic shape
    m1 = SpotVarianceGaussianMLP(in_dim=in_dim, out_dim=1, hidden_dims=[16],
                                 dropout=0.0).eval()
    mean1, log_var1 = m1(x)
    assert mean1.shape == (B,) and log_var1.shape == (B,), "bad single-gene shapes"
    print("spot-homoscedastic self-test: PASS")


if __name__ == "__main__":
    _genehyper_selftest()
    _likelihood_models_selftest()
    _spot_homoscedastic_selftest()
