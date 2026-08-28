# snake/scripts/epistemic/mc_dropout_posthoc.py
#
# Post-hoc MC-dropout for an existing CLASSIC run: produce ONE consolidated TEST
# file of per-(spot,gene) UNCERTAINTY SUMMARIES. No raw draws, no figures.
#
# Reads : <run-dir>/metrics.json, splits.json, model_fold*.pt, and the
#         preprocessed adata that metrics.json["input_adata"] names.
# Writes: <run-dir>/mc_dropout/test_summaries.h5ad
#
# WHAT IS A DRAW. MC-dropout leaves dropout ON at inference; each of T forward
# passes is one stochastic "model". The model mean per pass is m_t:
#   Gaussian -> E[log1p-CP10k] (the adata.X scale).
#   count    -> mu on the model's OWN fitted target scale (round(CP_T) for _cp,
#               CP_T for _cp_nr, raw counts for _raw). There is no offset and no
#               per-spot exposure: mean and dispersion share ONE scale.
#
# WHY SUMMARIES, NOT THE DRAWS. The only post-hoc quantities used are the average
# and the variance over draws (pred / epistemic / aleatoric), never the full
# predictive mixture (no mixture PIT, no per-draw g*). For those, a handful of
# running moments are SUFFICIENT statistics. They accumulate ONLINE in one pass,
# so no (T, n, G) array is ever held and storage is ~T-fold smaller. Stored per
# (spot, gene), on the model's native mean scale:
#   pred        = mean_t m_t                       (X)
#   epistemic   = var_t  m_t                        (layer "epistemic")
#   y           = the observed target               (layer "y")
#   + per family, the EXTRA term needed to rebuild the aleatoric:
#       Gaussian -> aleatoric_var = mean_t sigma^2(x)      (= the aleatoric itself)
#       NB       -> mean_alpha_m2 = mean_t (alpha_t * m_t^2)  (the alpha*mu^2 term)
#       Poisson  -> nothing (aleatoric == pred, since var = mu)
#
# POST-HOC RECONSTRUCTION (you do this; the job bakes in nothing). Everything is
# on the model's OWN target scale, with NO rescale of any kind:
#   pred = X ;  epi = epistemic
#   aleatoric = X (Poisson) | aleatoric_var (Gaussian) | X + mean_alpha_m2 (NB)
#   total     = aleatoric + epistemic
#
# TEST ONLY, ONE FILE. Under LOPO each donor is the held-out TEST donor EXACTLY
# once, so the union of all folds' test spots = every spot once. This writes a
# SINGLE matrix .h5ad: each spot's row carries the summaries from the fold where
# ITS donor was held out. Validation draws are dropped, since held-out epistemic
# does not need them. Layout mirrors the pipeline's predictions: X = pred,
# per-(spot,gene) layers, per-spot obs join keys, var = genes.
#
# Usage (repo root, on the cluster; CPU is plenty, the MLP forward is trivial):
#   python -u snake/scripts/epistemic/mc_dropout_posthoc.py --run-dir results/<...>/<model_type>

import argparse
import glob
import json
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
               help="A classic run dir (holds model_fold*.pt, splits.json, metrics.json).")
p.add_argument("--data", default=None,
               help="Override the preprocessed adata path (default: metrics.json['input_adata']).")
p.add_argument("--T", type=int, default=30, help="MC-dropout forward passes.")
p.add_argument("--batch-size", type=int, default=2048)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--device", default=None, help="cuda|cpu (default: auto).")
p.add_argument("--out-dir", default=None, help="Default: <run-dir>/mc_dropout/")
args = p.parse_args()

DEVICE = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
run_dir = os.path.abspath(args.run_dir)
out_dir = args.out_dir or os.path.join(run_dir, "mc_dropout")
os.makedirs(out_dir, exist_ok=True)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)

from src.data.load_and_preprocess import read_arrays
from src.data.dataset_indexed import SharedTensors
from src.models.specs import get_spec, build_model

# ── Read the run's own description (metrics.json) ─────────────────────────────
with open(os.path.join(run_dir, "metrics.json")) as f:
    M = json.load(f)
model_type = M["model_type"]
# gene_names is absent from the stub metrics.json a compute_final_metrics=false run
# writes: it carries n_genes but not the list. Fall back to the sibling genes.txt that
# train.py writes at startup, one gene per line.
gene_names = M.get("gene_names")
if gene_names is None:
    with open(os.path.join(run_dir, "genes.txt")) as gf:
        gene_names = [ln.strip() for ln in gf if ln.strip()]
center     = bool(M["center"])
arch       = {k: M["architecture"][k] for k in ("hidden_dims", "dropout", "activation")}
data_path  = args.data or M["input_adata"]
target_source = M["target_source"]
# target_source values that mean "this is a count-likelihood model" (Poisson/NB family). Each of
# them fits its likelihood DIRECTLY on its own target -- round(CP_T) for "cp_counts", CP_T for
# "cp_counts_nr", raw counts for "raw_counts" -- with no offset and no per-spot exposure, so the
# mean the model returns is already mu on that target's scale.
COUNT_TARGET_SOURCES = ("raw_counts", "cp_counts", "cp_counts_nr")
count_spec = (target_source in COUNT_TARGET_SOURCES)

# ── depth_scale ───────────────────────────────────────────────────────────────
# depth_scale is the ONE surviving constant: count_common_depth/1e4 (2.5 on 10x_TuPro,
# 1.0 on TLS / LUNG_CANCER) for the cp / cp_nr targets, and None for the depth-blind
# *_raw ones (None, not 1.0: raw is aligned to NO depth). It is needed here because
# read_arrays() builds the cp target itself as round(CP10k*depth_scale).
NEW_CONVENTION = "v2_no_offset"
if M.get("count_convention") != NEW_CONVENTION:
    raise ValueError(
        f"{run_dir}: run with no count_convention marker in metrics.json. "
        f"Re-run the training.")
depth_scale = M.get("depth_scale")                           # float, or None for *_raw / X targets

spec = get_spec(model_type)
model_kwargs = {"log_alpha_init": M.get("nb_log_alpha_init", -2.0)}
want_var  = (target_source == "X" and spec.is_probabilistic)        # Gaussian variance head
want_disp = ("dispersion" in spec.extra_prediction_columns)         # NB dispersion
G = len(gene_names)

# Name of the per-family EXTRA layer (the aleatoric ingredient). None for Poisson
# (and any non-probabilistic spec), whose aleatoric == pred (var = mu).
extra_layer = "aleatoric_var" if want_var else ("mean_alpha_m2" if want_disp else None)

print(f"Run dir : {run_dir}", flush=True)
print(f"Model   : {model_type} (count={count_spec}, gauss_var={want_var}, nb_disp={want_disp}), "
      f"n_genes={G}, center={center}, depth_scale={depth_scale}", flush=True)
print(f"MC      : T={args.T}, device={DEVICE} | online summaries -> {out_dir}", flush=True)

# ── Load data, extract what we need, then FREE the big adata ──────────────────
import anndata as ad
t_data = time.time()
adata = ad.read_h5ad(data_path)
obs_names_all = np.asarray(adata.obs_names)                         # join key (barcode)

embeddings, y_all, sample_labels, available_genes = read_arrays(
    adata, target_source=target_source, depth_scale=depth_scale)
gene_idx = [available_genes.index(g) for g in gene_names]
shared = SharedTensors(embeddings, y_all[:, gene_idx], sample_labels, device=DEVICE)
del embeddings, y_all

# Per-spot obs columns we still need (small) — pull them out, then drop the adata
# so its X / raw_counts / X_emb (several GB) are released before the fold loop.
ground_truth_all = (adata.obs["ground_truth"].to_numpy()
                    if "ground_truth" in adata.obs else None)
del adata
print(f"[{_el():7.1f}s] data loaded + adata freed (read+prep {time.time()-t_data:.1f}s)", flush=True)

# ── Splits — TEST spots per fold (LOPO: one held-out donor per fold) ──────────
with open(os.path.join(run_dir, "splits.json")) as f:
    splits = json.load(f)

def _test_idx(fo):
    return np.asarray(fo.get("test_idx", splits.get("test_idx")), dtype=np.int64)

test_idx_by_fold = {fo["fold"]: _test_idx(fo) for fo in splits["folds"]}

fold_ckpts = sorted(glob.glob(os.path.join(run_dir, "model_fold*.pt")))
if not fold_ckpts:
    raise FileNotFoundError(f"No model_fold*.pt in {run_dir}")


def enable_mc_dropout(model):
    """eval() everything (LayerNorm etc. deterministic), then flip ONLY the Dropout
    modules back to train so masks are resampled each forward pass."""
    model.eval()
    n = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train(); n += 1
    if n == 0:
        raise RuntimeError("No nn.Dropout modules found — MC-dropout needs dropout>0.")
    return n


@torch.no_grad()
def collect_summaries(model, idx, train_mean):
    """T dropout passes over `idx`, accumulated ONLINE. Returns float32
    (M1, V, EX) of shape (n, G) — never materializes (T, n, G):
        M1 = mean_t m_t ;  V = var_t m_t ;  EX = the family's aleatoric ingredient
             (mean_t sigma^2 for Gaussian, mean_t alpha*m^2 for NB, else None).
    Sums are kept in float64 so V = E[m^2] - E[m]^2 stays accurate."""
    idx_t  = torch.as_tensor(idx, dtype=torch.long, device=shared.emb.device)
    mean_t = (torch.as_tensor(np.asarray(train_mean), dtype=torch.float32, device=DEVICE)
              if center else None)
    n = idx.size
    s_m  = np.zeros((n, G), dtype=np.float64)
    s_m2 = np.zeros((n, G), dtype=np.float64)
    s_ex = np.zeros((n, G), dtype=np.float64) if extra_layer else None
    torch.manual_seed(args.seed)
    for _ in range(args.T):
        for s0 in range(0, n, args.batch_size):
            b = idx_t[s0:s0 + args.batch_size]
            xb, _ = shared.gather(b, device=DEVICE)
            if center:
                xb = xb - mean_t
            out = spec.extract_predictions(model(xb))
            m = out["means"].reshape(b.numel(), G).double().cpu().numpy()
            sl = slice(s0, s0 + b.numel())
            s_m[sl]  += m
            s_m2[sl] += m * m
            if want_var:
                s_ex[sl] += out["pred_var"].reshape(b.numel(), G).double().cpu().numpy()
            elif want_disp:
                a = out["dispersion"].reshape(b.numel(), G).double().cpu().numpy()
                s_ex[sl] += a * m * m
    M1 = s_m / args.T
    V  = np.maximum(s_m2 / args.T - M1 * M1, 0.0)          # clip tiny negatives (rounding)
    EX = (s_ex / args.T).astype(np.float32) if s_ex is not None else None
    return M1.astype(np.float32), V.astype(np.float32), EX


# ── Per fold: predict ITS held-out donor's TEST spots, fill the global arrays ─
n_total = shared.n_spots
PRED  = np.full((n_total, G), np.nan, dtype=np.float32)
EPI   = np.full((n_total, G), np.nan, dtype=np.float32)
EXTRA = np.full((n_total, G), np.nan, dtype=np.float32) if extra_layer else None
filled = np.zeros(n_total, dtype=bool)

for ck in fold_ckpts:
    fold = int(os.path.basename(ck).replace("model_fold", "").replace(".pt", ""))
    # weights_only=False: our own checkpoint (trusted) carries a numpy train_mean,
    # which PyTorch 2.6's weights_only=True default refuses to unpickle.
    ckpt = torch.load(ck, map_location=DEVICE, weights_only=False)
    # The checkpoint carries the same marker as metrics.json: its forward returns mu
    # directly on the target scale. An unmarked checkpoint has a mean scale that is not
    # recoverable here, so stop rather than emit numbers on an unknown scale.
    if count_spec and ckpt.get("count_convention") != NEW_CONVENTION:
        raise ValueError(
            f"{ck}: checkpoint with no count_convention marker. "
            f"Re-run the training.")
    if count_spec and ckpt.get("depth_scale") != depth_scale:
        print(f"WARNING {os.path.basename(ck)}: checkpoint depth_scale="
              f"{ckpt.get('depth_scale')!r} != run depth_scale={depth_scale!r}.", flush=True)
    model = build_model(model_type, in_dim=shared.emb_dim, n_genes=G,
                        device=DEVICE, arch=arch, model_kwargs=model_kwargs)
    model.load_state_dict(ckpt["state_dict"])
    enable_mc_dropout(model)

    te = test_idx_by_fold[fold]
    if filled[te].any():
        # Disjoint test sets are the LOPO invariant; an overlap means a non-LOPO
        # (shared-holdout) split, where one consolidated test file is ill-defined.
        print(f"WARNING fold {fold}: {int(filled[te].sum())} test spots already "
              f"filled by another fold — overwriting (split is not LOPO?).", flush=True)
    tf = time.time()
    M1, V, EX = collect_summaries(model, te, ckpt["train_mean"])
    PRED[te] = M1
    EPI[te]  = V
    if EXTRA is not None:
        EXTRA[te] = EX
    filled[te] = True
    print(f"[{_el():7.1f}s] fold {fold}: {te.size} test spots x {G} genes x T={args.T} "
          f"done in {time.time()-tf:.1f}s", flush=True)
    del model, ckpt

# ── Assemble ONE test AnnData over the spots actually covered ─────────────────
rows = np.where(filled)[0]                                   # = all spots under LOPO
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
# The marker every offline reader keys off. depth_scale is written ONLY when it is
# not None, because h5ad uns cannot serialize a bare Python None: readers use
# adata.uns.get("depth_scale"), which then yields None for the depth-blind *_raw
# models and for the X targets.
out.uns["count_convention"] = NEW_CONVENTION
if depth_scale is not None:
    out.uns["depth_scale"] = float(depth_scale)
out.uns["mc_dropout"] = dict(
    model_type=model_type, target_source=target_source, T=int(args.T),
    count=bool(count_spec), n_genes=int(G), split="test",
    mean_scale=("log1p_cp10k" if not count_spec else f"target_scale:{target_source}"),
    layers_doc=("X = pred = mean_t m_t ; y = observed target ; "
                "epistemic = var_t m_t ; "
                + ("aleatoric_var = mean_t sigma^2 (Gaussian)"
                   if want_var else
                   "mean_alpha_m2 = mean_t(alpha_t * m_t^2) (NB)"
                   if want_disp else
                   "no extra layer (Poisson: aleatoric == pred, since var = mu)")),
    reconstruct=("everything is on the model's own target scale, NO rescale: "
                 "aleatoric = pred (Poisson) | aleatoric_var (Gaussian) | "
                 "pred + mean_alpha_m2 (NB) ; total = aleatoric + epistemic."),
)

out_path = os.path.join(out_dir, "test_summaries.h5ad")
tw = time.time()
out.write_h5ad(out_path)
print(f"[{_el():7.1f}s] DONE. {out.n_obs} spots x {out.n_vars} genes "
      f"(write {time.time()-tw:.1f}s) -> {out_path}", flush=True)
