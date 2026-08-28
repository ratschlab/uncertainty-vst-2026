# snake/scripts/epistemic/laplace_posthoc.py
#
# Post-hoc last-layer Laplace for one classic run. Writes a single consolidated test
# file of per-(spot, gene) uncertainty summaries. No raw draws, no figures. The output
# format matches mc_dropout_posthoc.py layer for layer.
#
# What last-layer Laplace is. A Gaussian posterior N(theta_MAP, Sigma) sits on the
# linear head only (the mean head for a Gaussian model, the log-rate head for a count
# model); the backbone stays frozen at MAP. Per gene, Sigma = (H_g + a*I)^-1 where H_g
# is the GGN data term
#   H_g = sum_n w_n phi~_n phi~_n^T,   phi~ = bias-augmented penultimate features,
#   w_n = the likelihood curvature in the head output:
#     Gaussian: precision 1/sigma^2(x)   Poisson: mu   NB: mu/(1+alpha*mu),
#   mu = exp(f(x)), the mean on the model's own fitted target scale. There is no size
#   factor to re-apply. Eigendecomposing H_g = (Q, lambda) gives the head-output
#   posterior variance for any prior a:
#       sigma_eta^2(x) = phi~^T (H_g + a I)^-1 phi~ = sum_k (q_k.phi~)^2/(lambda_k+a).
# The prior precision `a` is tuned per fold and per gene on the validation predictive NLL.
#
# Closed form, no sampling on the scoring path.
#   Gaussian (identity link): the head output is the mean, so the predictive is exact
#     and analytic. mean = mean_MAP, epistemic = sigma_eta^2, aleatoric = sigma^2(x).
#   Count (log link): eta = f(x) ~ N(eta_MAP, sigma_eta^2), so m = exp(eta) is
#     lognormal and its moments are closed form:
#       E[m]   = m_MAP * exp(sigma_eta^2 / 2)
#       Var[m] = E[m]^2 * (exp(sigma_eta^2) - 1)
#       E[m^2] = E[m]^2 + Var[m]   (= E[m]^2 * exp(sigma_eta^2))
#   Nothing ever holds an (R, n, G) stack of draws; the count prior tuning draws its R
#   samples one gene-chunk at a time.
#
# Count convention "v2_no_offset". There is no size factor, no offset and no per-spot
# exposure anywhere: every count model fits its likelihood directly on its own target
# (_cp -> round(CP_T), _cp_nr -> CP_T, _raw -> raw counts), so the mean m = exp(f(x))
# and the dispersion alpha already share one scale, the target's own. Nothing here is
# multiplied by a depth. The run's single depth constant, `depth_scale`
# (= count_common_depth/1e4 for the cp families, None for the depth-blind _raw ones),
# is needed for one thing only: read_arrays rebuilds the cp target with it. It is
# copied into uns for the reader and never enters the algebra.
#
# What is stored, per (spot, gene), on the model's own target scale. The layer names
# are the ones mc_dropout_posthoc.py writes:
#   pred (X)             Gaussian: mean_MAP        count: E[m]
#   epistemic            Gaussian: sigma_eta^2     count: Var[m]
#   y                    the observed target (adata.X for Gaussian, the count target for count)
#   + per family, the extra term needed to rebuild the aleatoric:
#       Gaussian -> aleatoric_var = sigma^2(x)          (= the aleatoric itself)
#       NB       -> mean_alpha_m2 = alpha_MAP * E[m^2]
#       Poisson  -> nothing (aleatoric == pred, since var = mean)
#
# Post-hoc reconstruction, done by the reader; the job bakes in nothing. One recipe,
# no rescaling arm:
#   pred = X ; epi = epistemic
#   aleatoric = X (Poisson) | aleatoric_var (Gaussian) | X + mean_alpha_m2 (NB)
#   total     = aleatoric + epistemic
#   The NB identity that makes this exact: X + mean_alpha_m2 = E[m] + alpha*E[m^2]
#   = E[m + alpha*m^2] = the NB variance averaged over the posterior of m. So
#   total = E[m] + alpha*E[m^2] + Var[m], the law-of-total-variance decomposition.
#
# Test only, one file. Under LOPO each donor is the held-out test donor exactly once,
# so the union of all folds' test spots is every spot once. The output is a single
# matrix .h5ad: each spot's row carries the summaries from the fold where its donor was
# held out. Validation is used only to tune the prior precision `a` and is never saved.
# Layout mirrors the pipeline's predictions: X = pred, per-(spot,gene) layers, per-spot
# obs join keys, var = genes.
#
# Output: <run-dir>/laplace/test_summaries.h5ad
#
# Usage (repo root, on the cluster; CPU is enough, the MLP forward is small and the
# algebra is numpy):
#   python -u snake/scripts/epistemic/laplace_posthoc.py --run-dir results/<...>/<model_type>

import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

# Lightweight wall-clock instrumentation: every milestone prints "[   s]" since
# start, so the .out shows where time goes (adata read vs per fold vs write).
_T0 = time.time()
def _el():
    return time.time() - _T0

# ── Args ──────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--run-dir", required=True,
               help="A classic run dir (model_fold*.pt, splits.json, metrics.json).")
p.add_argument("--data", default=None,
               help="Override the preprocessed adata path (default: metrics.json['input_adata']).")
p.add_argument("--alpha-grid", default="1e-5,1e-4,1e-3,1e-2,1e-1,1,1e1,1e2,1e3,1e4",
               help="Comma-separated prior-precision values to tune over (val NLL). "
                    "It reaches 1e-5 because some Gaussian folds tune to the bottom of the grid.")
p.add_argument("--tune-method", choices=["mixture", "moment"], default="mixture",
               help="COUNT prior tuning: 'mixture' = predictive-mixture MC NLL "
                    "(R draws, the safe default); 'moment' = moment-matched single-NB "
                    "NLL (1 eval/alpha, ~R x cheaper). Gaussian is analytic either way.")
p.add_argument("--tune-samples", type=int, default=16,
               help="COUNT 'mixture' tuning only: posterior log-rate draws (R).")
p.add_argument("--gene-chunk", type=int, default=512,
               help="COUNT tuning: genes per block (caps peak RAM of the NLL sweep).")
p.add_argument("--batch-size", type=int, default=512, help="Forward batch size.")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--device", default=None, help="cuda|cpu (default: auto).")
p.add_argument("--out-dir", default=None, help="Default: <run-dir>/laplace/")
args = p.parse_args()

DEVICE  = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
run_dir = os.path.abspath(args.run_dir)
out_dir = args.out_dir or os.path.join(run_dir, "laplace")
os.makedirs(out_dir, exist_ok=True)
alpha_grid = np.array([float(a) for a in args.alpha_grid.split(",")], dtype=np.float64)
R_TUNE = int(args.tune_samples)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)

from src.data.load_and_preprocess import read_arrays
from src.data.dataset_indexed import SharedTensors
from src.models.specs import get_spec, build_model

# ── Read the run's own description (metrics.json) ─────────────────────────────
with open(os.path.join(run_dir, "metrics.json")) as f:
    M = json.load(f)
model_type    = M["model_type"]
# gene_names is absent from the stub metrics.json a compute_final_metrics=false run
# writes: it carries n_genes but not the list. Fall back to the sibling genes.txt that
# train.py writes at startup, one gene per line.
gene_names    = M.get("gene_names")
if gene_names is None:
    with open(os.path.join(run_dir, "genes.txt")) as gf:
        gene_names = [ln.strip() for ln in gf if ln.strip()]
center        = bool(M["center"])
arch          = {k: M["architecture"][k] for k in ("hidden_dims", "dropout", "activation")}
data_path     = args.data or M["input_adata"]
target_source = M["target_source"]
spec = get_spec(model_type)
# target_source values that mark a count-likelihood model, so the curvature, tuning and
# summaries below take the Poisson/NB branch. "cp_counts" and "cp_counts_nr" are the
# common-depth variants, "raw_counts" the depth-blind one. See src/models/specs.py.
COUNT_TARGET_SOURCES = ("raw_counts", "cp_counts", "cp_counts_nr")
count_spec = (target_source in COUNT_TARGET_SOURCES)
# Handles the per-gene-independent probabilistic families only: the Gaussians
# (homoscedastic / heteroscedastic / spot) and the count models (Poisson / NB).
if not spec.is_probabilistic:
    raise NotImplementedError(
        f"laplace_posthoc supports per-gene Gaussian or count models only. Got {model_type!r}.")
model_kwargs = {"log_alpha_init": M.get("nb_log_alpha_init", -2.0)}
want_var  = (target_source == "X" and spec.is_probabilistic)        # Gaussian variance head
want_disp = ("dispersion" in spec.extra_prediction_columns)         # NB dispersion
G = len(gene_names)

# Name of the per-family extra layer, the aleatoric ingredient. Same mapping as
# mc_dropout_posthoc.py. None for Poisson, where aleatoric == pred since var = mean.
extra_layer = "aleatoric_var" if want_var else ("mean_alpha_m2" if want_disp else None)

print(f"Run dir : {run_dir}", flush=True)
print(f"Model   : {model_type} (count={count_spec}, gauss_var={want_var}, nb_disp={want_disp}), "
      f"n_genes={G}, center={center}", flush=True)
print(f"Laplace : last-layer ({'log-rate' if count_spec else 'mean'} head), closed-form, "
      f"alpha grid={list(alpha_grid)}, device={DEVICE}"
      + (f" | count tune={args.tune_method}"
         + (f" R={R_TUNE}" if args.tune_method == 'mixture' else "")
         + f", gene_chunk={args.gene_chunk}" if count_spec else ""), flush=True)

# ── Load data, extract what we need, then FREE the big adata ──────────────────
import anndata as ad
t_data = time.time()
adata = ad.read_h5ad(data_path)
obs_names_all = np.asarray(adata.obs_names)                        # join key (barcode)

# ── The run's depth constant ──────────────────────────────────────────────────
# `count_convention` is the marker every offline reader keys off. depth_scale is needed
# for one thing: read_arrays rebuilds the cp target from CP10k * depth_scale (rounded for
# the _cp family), so it must be resolved before the target is read. It never touches the
# algebra.
depth_scale = None
if count_spec:
    if M.get("count_convention") != "v2_no_offset":
        raise ValueError(
            f"{run_dir}: run with no count_convention marker in metrics.json. "
            f"Re-run the training.")
    depth_scale = M.get("depth_scale")
    if target_source in ("cp_counts", "cp_counts_nr") and depth_scale is None:
        raise KeyError(
            f"cp/cp_nr run but metrics.json has no 'depth_scale'; the CP_T target "
            f"cannot be rebuilt without it.")
    print(f"Count   : convention=v2_no_offset, target={target_source}, "
          f"depth_scale={depth_scale}", flush=True)

embeddings, y_all, sample_labels, available_genes = read_arrays(
    adata, target_source=target_source, depth_scale=depth_scale)
gene_idx = [available_genes.index(g) for g in gene_names]
shared = SharedTensors(embeddings, y_all[:, gene_idx], sample_labels, device=DEVICE)
del embeddings, y_all

# Pull out the one small per-spot obs column still needed, then drop the adata so its
# X / raw_counts / X_emb are released before the fold loop.
ground_truth_all = (adata.obs["ground_truth"].to_numpy()
                    if "ground_truth" in adata.obs else None)
del adata
print(f"[{_el():7.1f}s] data loaded + adata freed (read+prep {time.time()-t_data:.1f}s)", flush=True)

# ── Splits — train (GGN) / val (tune) / test (score+save) per fold ────────────
with open(os.path.join(run_dir, "splits.json")) as f:
    splits = json.load(f)

def _inner_idx(fo, key):
    inner = fo.get("inner_splits")
    return np.asarray(inner[0][key] if inner else fo[key], dtype=np.int64)

train_idx_by_fold = {fo["fold"]: _inner_idx(fo, "train_idx") for fo in splits["folds"]}
val_idx_by_fold   = {fo["fold"]: _inner_idx(fo, "val_idx")   for fo in splits["folds"]}
test_idx_by_fold  = {fo["fold"]: np.asarray(fo.get("test_idx", splits.get("test_idx")), dtype=np.int64)
                     for fo in splits["folds"]}

fold_ckpts = sorted(glob.glob(os.path.join(run_dir, "model_fold*.pt")))
if not fold_ckpts:
    raise FileNotFoundError(f"No model_fold*.pt in {run_dir}")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _feature_extractor(model):
    """Penultimate-feature module, robust to attribute name: Gaussian / NB expose
    `.backbone`; PoissonMLP wraps it as `.net[0]` (Sequential of backbone+head)."""
    if hasattr(model, "backbone"):
        return model.backbone
    if hasattr(model, "net"):
        return model.net[0]
    raise AttributeError(f"{type(model).__name__}: no backbone / net feature extractor")


@torch.no_grad()
def forward_run(model, idx, train_mean):
    """One deterministic forward (dropout OFF via eval()) over `idx`. Returns:
       means (n,G), pred_var (n,G), dispersion (n,G)|None, y (n,G),
       phi~ (n, H+1) — phi~ = bias-augmented penultimate features.
    means is the mean on the model's OWN target scale (m=exp(f(x)) for count,
    log1p-CP10k mean for Gaussian); pred_var is sigma^2(x) (Gaussian) / unused
    (count)."""
    idx_t  = torch.as_tensor(idx, dtype=torch.long, device=shared.emb.device)
    mean_t = (torch.as_tensor(np.asarray(train_mean), dtype=torch.float32, device=DEVICE)
              if center else None)
    feat = _feature_extractor(model)
    mu, var, disp, feats = [], [], [], []
    model.eval()
    for s0 in range(0, idx.size, args.batch_size):
        b = idx_t[s0:s0 + args.batch_size]
        xb, _ = shared.gather(b, device=DEVICE)
        if center:
            xb = xb - mean_t
        out = spec.extract_predictions(model(xb))
        mu.append(out["means"].reshape(b.numel(), G).cpu())
        var.append(out["pred_var"].reshape(b.numel(), G).cpu())
        if want_disp:
            disp.append(out["dispersion"].reshape(b.numel(), G).cpu())
        phi = feat(xb)
        ones = torch.ones(phi.shape[0], 1, device=phi.device)
        feats.append(torch.cat([phi, ones], dim=1).cpu())          # phi~ = [phi, 1]
    means = torch.cat(mu).numpy()
    pvar  = torch.cat(var).numpy()
    disp  = torch.cat(disp).numpy() if want_disp else None
    yv    = shared.tgt[idx_t].reshape(idx.size, G).cpu().numpy()
    phi   = torch.cat(feats).numpy()
    return means, pvar, disp, yv, phi


def fit_gene_eigen(phi_train, weight_train_g):
    """Eigendecompose the GGN data term H_g = sum_n w_n phi~_n phi~_n^T for one gene.
    w_n = the likelihood curvature in the head output (Gaussian precision 1/sigma^2,
    count Fisher info mu_c / mu_c(1+alpha*mu_c)). (Q, lambda) let us evaluate
    phi~^T (H_g + a I)^-1 phi~ = sum_k (q_k.phi~)^2 / (lambda_k + a) for ANY prior a."""
    w = np.sqrt(np.clip(weight_train_g, 1e-8, None))[:, None]
    Aw = phi_train * w
    lam, Q = np.linalg.eigh(Aw.T @ Aw)
    return Q, np.clip(lam, 0.0, None)


def epi_all_alphas(phi_eval, eig, grid):
    """Head-output posterior variance phi~^T (H_g + a I)^-1 phi~ for EVERY
    (spot, gene, alpha) -> (n, G, n_alpha).

    The projection P_g = phi_eval @ Q_g is independent of the prior a, so it is computed
    once per gene and the whole alpha grid is folded in as a (H+1, n_alpha) reciprocal
    matmul. Every prior shares that projection; there is no re-matmul per alpha."""
    n = phi_eval.shape[0]
    out = np.empty((n, len(eig), len(grid)), dtype=np.float32)
    g64 = np.asarray(grid, dtype=np.float64)
    for g, (Q, lam) in enumerate(eig):
        P2 = (phi_eval @ Q).astype(np.float64) ** 2                # (n, H+1) >= 0
        denom = 1.0 / (lam.astype(np.float64)[:, None] + g64[None, :])   # (H+1, n_alpha)
        out[:, g, :] = (P2 @ denom).astype(np.float32)             # sum of positives
    return out


def epi_single(phi_eval, eig, a):
    """Head-output posterior variance at prior(s) a -> (n, G), the scoring path. `a` is
    either a scalar shared across genes or an array of shape (G,), one prior per gene.
    Both are broadcast, so both work."""
    a_g = np.broadcast_to(np.asarray(a, dtype=np.float64), (len(eig),))
    out = np.empty((phi_eval.shape[0], len(eig)), dtype=np.float64)
    for g, (Q, lam) in enumerate(eig):
        P = (phi_eval @ Q).astype(np.float64)              # float64 reduction (matches epi_all_alphas)
        out[:, g] = ((P ** 2) / (lam + a_g[g])).sum(axis=1)
    return out


def gaussian_nll_sweep(y, mu, var_base, epi_all):
    """Gaussian predictive NLL per (gene, alpha), analytic. epi_all: (n, G, n_alpha).
    Predictive var = sigma^2(x) + phi~^T Sigma phi~. Returns (G, n_alpha), averaged over
    spots only so the caller picks alpha per gene. Pooling the gene axis lets well-curved
    genes dilute the fit of weakly-curved ones."""
    n, G, n_alpha = epi_all.shape
    out = np.empty((G, n_alpha))
    for ai in range(n_alpha):
        var = np.clip(var_base + epi_all[:, :, ai], 1e-8, None)             # (n, G)
        nll = 0.5 * (np.log(2 * np.pi * var) + (y - mu) ** 2 / var)         # (n, G)
        out[:, ai] = nll.mean(axis=0)                                        # -> (G,)
    return out


def _gene_chunks(n_genes, size):
    for s0 in range(0, n_genes, size):
        yield slice(s0, min(s0 + size, n_genes))


def count_mixture_nll_sweep(means, disp, y, epi_all, R, gene_chunk, seed):
    """Count predictive-mixture NLL per (gene, alpha), the default tuning method. For each
    alpha, draw mu_r = m_MAP * exp(eps_r * sqrt(epi)) and average the native Poisson/NB
    log-prob over R draws. One batched log_prob over (R, n, C) per (alpha, gene-chunk),
    in float32 and gene-chunked, so no (R, n, G) or float64 (n, G) allocation is needed.
    The R draws use common random numbers across alphas, drawn once per chunk, so the
    alpha comparison is low-variance. means and disp (n, G) are the MAP mean and the NB
    alpha, None for Poisson, on the target's own scale; y (n, G) is that same target, so
    the likelihood carries no depth factor. Returns (n_genes, A) averaged over spots only,
    so the caller tunes alpha per gene instead of pooling the panel into one alpha."""
    n, n_genes, A = epi_all.shape
    nll = np.zeros((n_genes, A), dtype=np.float64)
    rng = np.random.default_rng(seed)
    log_R = math.log(R)
    for sl in _gene_chunks(n_genes, gene_chunk):
        m0  = torch.as_tensor(means[:, sl].astype(np.float32))                 # (n, C)
        y_t = torch.as_tensor(y[:, sl].astype(np.float32)).unsqueeze(0)        # (1, n, C)
        a_t = None if disp is None else torch.as_tensor(disp[:, sl].astype(np.float32))  # (n, C)
        sig = np.sqrt(epi_all[:, sl, :]).astype(np.float32)                    # (n, C, A)
        eps = torch.as_tensor(rng.standard_normal((R, n, sl.stop - sl.start), dtype=np.float32))
        for ai in range(A):
            sig_a = torch.as_tensor(sig[:, :, ai])                             # (n, C)
            mu = m0.unsqueeze(0) * torch.exp((eps * sig_a.unsqueeze(0)).clamp_(-30, 30))  # (R,n,C)
            if a_t is None:
                # validate_args=False: cp_counts_nr targets are continuous, so y can be a
                # non-integer. PyTorch's default support check expects non-negative
                # integers for Poisson/NB and would raise on the log_prob() below.
                # losses.py::nb_nll_loss does the same during training.
                d = torch.distributions.Poisson(mu, validate_args=False)
            else:
                aa = a_t.unsqueeze(0)                                          # (1,n,C) broadcast over R
                d = torch.distributions.NegativeBinomial(
                        total_count=1.0 / aa, logits=torch.log(mu) + torch.log(aa),
                        validate_args=False)
            mix = torch.logsumexp(d.log_prob(y_t.expand_as(mu)), dim=0) - log_R  # (n, C)
            nll[sl, ai] += (-mix.to(torch.float64).sum(dim=0)).numpy()          # sum SPOTS only -> (C,)
    return nll / n


def count_moment_nll_sweep(means, disp, y, epi_all, gene_chunk):
    """Count moment-matched tuning: match the full predictive (the NB aleatoric plus the
    epistemic spread of m) to a single NB per (spot, gene, alpha) via the closed-form
    lognormal moments, one distribution eval per alpha instead of R draws. Everything is
    on the model's own target scale, with no depth factor:
        mu  = E[m];  var = mu + alpha*E[m^2] + Var[m]  (Poisson: alpha=0)
        alpha_eff = clip((var-mu)/mu^2, >=0);  nll = -mean NB(1/alpha_eff, .).log_prob(y)
    One batched NB.log_prob over (n, C, A) per gene-chunk. Consistent with the closed-form
    scoring. Returns (n_genes, A) averaged over spots only, so the caller tunes alpha per
    gene instead of pooling the panel into one alpha."""
    n, n_genes, A = epi_all.shape
    nll = np.zeros((n_genes, A), dtype=np.float64)
    for sl in _gene_chunks(n_genes, gene_chunk):
        epi   = torch.as_tensor(np.minimum(epi_all[:, sl, :], 30.0))          # (n,C,A) clip exp overflow
        m0    = torch.as_tensor(means[:, sl].astype(np.float32)).unsqueeze(-1)  # (n,C,1)
        y_t   = torch.as_tensor(y[:, sl].astype(np.float32)).unsqueeze(-1)    # (n,C,1)
        E_m   = m0 * torch.exp(0.5 * epi)                                     # (n,C,A)
        E_m2  = E_m * E_m * torch.exp(epi)
        Var_m = E_m2 - E_m * E_m
        mu    = E_m
        if disp is None:
            var = mu + Var_m                                                  # Poisson aleatoric + epistemic
        else:
            a_g = torch.as_tensor(disp[:, sl].astype(np.float32)).unsqueeze(-1)  # (n,C,1)
            var = mu + a_g * E_m2 + Var_m
        alpha_eff = torch.clamp((var - mu) / (mu * mu), min=1e-8)             # >0 (NB->Poisson as ->0)
        # validate_args=False: same reason as count_mixture_nll_sweep above -- cp_counts_nr
        # targets are continuous, so y_t can be non-integer.
        d = torch.distributions.NegativeBinomial(
                total_count=1.0 / alpha_eff, logits=torch.log(mu) + torch.log(alpha_eff),
                validate_args=False)
        nll[sl, :] += (-d.log_prob(y_t).to(torch.float64).sum(dim=0)).numpy()  # sum SPOTS -> (C, A)
    return nll / n


# ── Per fold: GGN on train, tune a on val, score+save test ────────────────────
n_total = shared.n_spots
PRED  = np.full((n_total, G), np.nan, dtype=np.float32)
EPI   = np.full((n_total, G), np.nan, dtype=np.float32)
EXTRA = np.full((n_total, G), np.nan, dtype=np.float32) if extra_layer else None
filled = np.zeros(n_total, dtype=bool)
alpha_prior_by_fold = {}

for ck in fold_ckpts:
    fold = int(os.path.basename(ck).replace("model_fold", "").replace(".pt", ""))
    # weights_only=False: our own checkpoint (trusted) carries a numpy train_mean,
    # which PyTorch 2.6's weights_only=True default refuses to unpickle.
    ckpt  = torch.load(ck, map_location=DEVICE, weights_only=False)
    # The checkpoint carries the same marker as metrics.json. Without it the mean scale is
    # not recoverable here, so stop rather than emit numbers on an unknown scale.
    if count_spec and ckpt.get("count_convention") != "v2_no_offset":
        raise ValueError(
            f"{ck}: checkpoint with no count_convention marker. "
            f"Re-run the training.")
    model = build_model(model_type, in_dim=shared.emb_dim, n_genes=G,
                        device=DEVICE, arch=arch, model_kwargs=model_kwargs)
    model.load_state_dict(ckpt["state_dict"])
    tm = ckpt["train_mean"]

    tr_idx = train_idx_by_fold[fold]
    va_idx = val_idx_by_fold[fold]
    te_idx = test_idx_by_fold[fold]
    if filled[te_idx].any():
        # Disjoint test sets are the LOPO invariant; an overlap means a non-LOPO
        # (shared-holdout) split, where one consolidated test file is ill-defined.
        print(f"WARNING fold {fold}: {int(filled[te_idx].sum())} test spots already "
              f"filled by another fold — overwriting (split is not LOPO?).", flush=True)

    # (1) GGN on TRAIN: per-(spot,gene) curvature weight -> per-gene eigendecomp. ──
    t0 = time.time()
    means_tr, var_tr, disp_tr, _, phi_tr = forward_run(model, tr_idx, tm)
    if count_spec:
        # means_tr is mu on the fitted target scale and disp_tr is the alpha fitted at that
        # same scale, so the Fisher info needs no depth factor.
        w_tr = means_tr if disp_tr is None else means_tr / (1.0 + disp_tr * means_tr)
    else:
        w_tr = 1.0 / np.clip(var_tr, 1e-8, None)                   # Gaussian precision
    eig = [fit_gene_eigen(phi_tr, w_tr[:, g]) for g in range(G)]
    t_ggn = time.time() - t0

    # (2) TUNE the prior precision a on the VALIDATION predictive NLL. ────────────
    t0 = time.time()
    means_va, var_va, disp_va, y_va, phi_va = forward_run(model, va_idx, tm)
    epi_va_all = epi_all_alphas(phi_va, eig, alpha_grid)           # (n_val, G, n_alpha); P cached
    if count_spec:
        if args.tune_method == "moment":
            nlls = count_moment_nll_sweep(means_va, disp_va, y_va, epi_va_all, args.gene_chunk)
        else:
            nlls = count_mixture_nll_sweep(means_va, disp_va, y_va, epi_va_all,
                                           R_TUNE, args.gene_chunk, args.seed + fold)
    else:
        nlls = gaussian_nll_sweep(y_va, means_va, var_va, epi_va_all)
    best_idx = np.argmin(nlls, axis=1)                     # (G,) per-gene best alpha INDEX
    best_a = alpha_grid[best_idx]                           # (G,) per-gene best alpha VALUE
    alpha_prior_by_fold[str(int(fold))] = best_a.tolist()   # per-gene list; str key: h5ad uns keys must be strings
    t_tune = time.time() - t0

    # (3) SCORE the TEST spots at a* (closed form) and fill the global arrays. ────
    t0 = time.time()
    means_te, var_te, disp_te, _, phi_te = forward_run(model, te_idx, tm)
    epi_te = epi_single(phi_te, eig, best_a)                       # head-output posterior var (n_test, G)
    if count_spec:
        # Lognormal closed form on m = exp(eta), eta ~ N(eta_MAP, sig2). No sampling.
        # Clip sig2 at 30, the same cap the moment-matched tuning sweep uses. An outlier
        # (spot, gene) can give a huge posterior variance, and exp(0.5*sig2) would then
        # overflow to +inf. Unlike NaN, inf is not skipped by nanmean, so one entry poisons
        # every downstream aggregate. sig2 = 30 already means a posterior std of about 5.5
        # on the log-rate.
        sig2  = np.minimum(epi_te, 30.0)
        E_m   = means_te * np.exp(0.5 * sig2)                      # E[m] (target scale)
        Var_m = (E_m ** 2) * np.expm1(sig2)                        # E[m]^2 (e^sig2 - 1)
        E_m2  = E_m ** 2 + Var_m                                   # = E[m]^2 e^sig2
        PRED[te_idx] = E_m.astype(np.float32)
        EPI[te_idx]  = Var_m.astype(np.float32)
        if extra_layer:                                           # NB: alpha_MAP * E[m^2]
            EXTRA[te_idx] = (disp_te * E_m2).astype(np.float32)
    else:
        # Gaussian: analytic + exact (identity link, posterior mean = MAP).
        PRED[te_idx]  = means_te.astype(np.float32)               # mean_MAP
        EPI[te_idx]   = epi_te.astype(np.float32)                 # phi~^T Sigma phi~
        EXTRA[te_idx] = var_te.astype(np.float32)                # sigma^2(x) (aleatoric_var)
    filled[te_idx] = True
    t_score = time.time() - t0

    print(f"[{_el():7.1f}s] fold {fold}: a* per-gene (min={best_a.min():.3g}, "
          f"median={np.median(best_a):.3g}, max={best_a.max():.3g})  "
          f"{te_idx.size} test spots x {G} genes  "
          f"(GGN {t_ggn:.1f}s / tune {t_tune:.1f}s / score {t_score:.1f}s)", flush=True)
    del model, ckpt, eig, phi_tr, phi_va, phi_te

# ── Assemble ONE test AnnData over the spots actually covered ─────────────────
rows = np.where(filled)[0]                                         # = all spots under LOPO
print(f"Covered {rows.size}/{n_total} spots across {len(fold_ckpts)} folds.", flush=True)

y_rows = shared.tgt[torch.as_tensor(rows, dtype=torch.long, device=shared.emb.device)] \
    .reshape(rows.size, G).cpu().numpy()

obs = pd.DataFrame(index=obs_names_all[rows])
obs["adata_row"] = rows.astype(np.int64)
obs["obs_name"]  = obs_names_all[rows]
obs["slide"]     = np.asarray(shared.groups[rows])
obs["split"]     = "test"
if ground_truth_all is not None:
    obs["ground_truth"] = ground_truth_all[rows]

layers = {"y": y_rows, "epistemic": EPI[rows]}
if extra_layer:
    layers[extra_layer] = EXTRA[rows]

out = ad.AnnData(X=PRED[rows], obs=obs,
                 var=pd.DataFrame(index=np.asarray(gene_names)), layers=layers)
out.uns["laplace"] = dict(
    model_type=model_type, target_source=target_source,
    count=bool(count_spec), n_genes=int(G), split="test",
    mean_scale=("log1p_cp10k" if not count_spec else f"count_target:{target_source}"),
    alpha_grid=[float(a) for a in alpha_grid],
    alpha_prior=alpha_prior_by_fold,   # tuned prior precision per fold, PER GENE (list of length
                                        # n_genes, gene order matches var_names/gene_names)
    tune_method=(args.tune_method if count_spec else "analytic"),
    tune_samples=(int(R_TUNE) if (count_spec and args.tune_method == "mixture") else 0),
    layers_doc=("X = pred ; y = observed target ; epistemic = posterior var of the "
                "head output ; "
                + ("aleatoric_var = sigma^2(x) (Gaussian, = aleatoric)"
                   if want_var else
                   "mean_alpha_m2 = alpha_MAP * E[m^2] (NB)"
                   if want_disp else
                   "no extra layer (Poisson: aleatoric == pred)")),
    reconstruct=("Gaussian: pred=mean_MAP (log1p-CP10k), epistemic=phi~^T Sigma phi~, "
                 "aleatoric=aleatoric_var. count: pred=E[m], epistemic=Var[m] (closed "
                 "form, on the model's own target scale), aleatoric = pred (Poisson) | "
                 "pred + mean_alpha_m2 (NB). total = aleatoric + epistemic in BOTH "
                 "cases: there is no size factor and nothing to rescale."),
)
# Count convention marker. "v2_no_offset" means X (the mean), the epistemic and extra
# layers and y all live on the model's own target scale, with no per-spot size factor
# anywhere; these files carry no obs["size_factor"]. It is always written. depth_scale is
# the single reporting constant. h5ad cannot serialize a bare Python None in uns, so the
# key is written only when it is a float and readers must use
# adata.uns.get("depth_scale"). It is absent for the depth-blind _raw models and for every
# Gaussian run.
out.uns["count_convention"] = "v2_no_offset"
if depth_scale is not None:
    out.uns["depth_scale"] = float(depth_scale)

out_path = os.path.join(out_dir, "test_summaries.h5ad")
tw = time.time()
out.write_h5ad(out_path)
print(f"[{_el():7.1f}s] DONE. {out.n_obs} spots x {out.n_vars} genes "
      f"(write {time.time()-tw:.1f}s) -> {out_path}", flush=True)
