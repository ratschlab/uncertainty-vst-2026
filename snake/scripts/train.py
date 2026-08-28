# snake/scripts/train.py
#
# Classic training entrypoint: one multi-output MLP per gene set, out_dim = n_genes.
# Snakemake runs it once per (run_id_classic, dataset, gene_set, radius, model_type,
# [lambda]). It is not meant to be launched by hand.
#
# Reads the merged preprocessed adata.h5ad written by the preprocess rule, plus
# snake/config/config.yaml and snake/config/donor_slide_map.yaml. Writes into the
# directory of --out-metrics: metrics.json, atoms.parquet, splits.json, genes.txt,
# learning_curves/fold{k}.csv, predictions/preds_fold{k}.h5ad, model_fold{k}.pt, and a
# copy of fold 0's checkpoint at --out-model.
#
# Flow: resolve the ModelSpec from model_type -> read the target matrix named by
# spec.target_source (X, cp_counts, cp_counts_nr, raw_counts) -> slice it to the gene
# set -> build SharedTensors on the GPU -> donor-level outer CV, leave-one-patient-out
# by default -> per fold: train, infer val + test, write predictions and atoms, free the
# fold's memory -> aggregate the per-fold scalars -> metrics.json.
#
# Splits group by donor, metrics group by slide. Slides of one patient are near
# duplicates, so a donor straddling train and test leaks; pooling slides into one
# correlation lets between-slide mean shifts masquerade as predictive skill.
#
# Count convention "v2_no_offset": no size factor, no offset, no per-spot exposure.
# Every count model fits its likelihood directly on its own target (round(CP_T) for _cp,
# continuous CP_T for _cp_nr, raw counts for _raw) and saves the mean mu and the
# dispersion alpha on that same scale. depth_scale (count_common_depth/1e4 for the cp
# families, None for raw) is the one count constant, and it divides only before the
# log1p of the reported mean correlation (g*, the per-epoch option-A metric, *_logmean).
# PIT, pred_var, var_residual_corr and residuals stay on the model's own target scale
# with no constant applied.
#
# run_id_classic is a manual version label. Bump it whenever a parameter outside the
# output path changes (lr, batch_size, centering, ...). Snakemake reruns on a changed
# path only, so without a bump it silently reuses the previous run's outputs.
#
# Run (the form Snakemake issues):
#   python -u snake/scripts/train.py \
#       --data results/<dataset>/_preprocessed/<preproc_id>/<radius_tag>/adata.h5ad \
#       --out-metrics <out_dir>/metrics.json --out-model <out_dir>/model.pt \
#       --model-type heteroscedastic --lam 1.0 --dataset 10x_TuPro \
#       --gene-set <gene_set> --genes all --radius 0.75 --run-id-classic <label>

import argparse
import math
import os
import sys
import json
import numpy as np
import pandas as pd
import yaml
import time
import socket
import platform
from datetime import datetime

# Wall-clock start of the whole run (total duration is recorded in metrics.json).
RUN_START  = time.time()
STARTED_AT = datetime.now().astimezone().isoformat()

def _str2bool(v):
    """argparse type for a true/false flag (case-insensitive)."""
    if isinstance(v, bool):
        return v
    low = v.lower()
    if low in ("true", "1", "yes"):
        return True
    if low in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got: {v!r}")


# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--data",        required=True)
parser.add_argument("--out-metrics", required=True)
parser.add_argument("--out-model",   required=True)
parser.add_argument("--model-type",  required=True)
parser.add_argument("--lam",         required=True, type=float)
parser.add_argument("--gene-set",    required=True)
parser.add_argument("--genes",       required=True,
                    help="Comma-separated gene names, or 'all'.")
parser.add_argument("--radius",      required=True)
parser.add_argument("--dataset",     required=True)
parser.add_argument("--run-id-classic", required=True,
                    help="Manual run-version label (e.g. 'v1'). Stored in "
                         "metrics.json so each result is self-describing.")
parser.add_argument("--seed", type=int, default=None,
                    help="Deep-ensemble MEMBER seed: overrides config 'seed'. It "
                         "drives ONLY weight init + batch shuffle (the donor test "
                         "holdout and GroupKFold CV take no RNG, so every member "
                         "shares the identical test + per-fold train/val split). "
                         "Default None -> use config seed (unchanged behavior).")
# Per-run hyperparameter overrides. A flag left unpassed is None and the value comes
# from config.yaml; a flag that is passed overrides this run's in-memory value only,
# never the file. The deep-ensemble rule train_ensemble_member uses these to run its own
# training recipe without editing the shared config keys every other run reads.
parser.add_argument("--lr", type=float, default=None,
                    help="Override config 'lr'. Default None -> use config.yaml.")
parser.add_argument("--batch-size", type=int, default=None,
                    help="Override config 'batch_size'. Default None -> use config.yaml.")
parser.add_argument("--max-epochs", type=int, default=None,
                    help="Override config 'max_epochs'. Default None -> use config.yaml.")
parser.add_argument("--patience", type=int, default=None,
                    help="Override config 'patience'. Default None -> use config.yaml.")
parser.add_argument("--warmup-epochs", type=int, default=None,
                    help="Override config 'warmup_epochs'. Default None -> use config.yaml.")
parser.add_argument("--early-stop-criterion", choices=["corr", "val_loss", "mse"], default=None,
                    help="Override config 'early_stop_criterion' (corr|val_loss|mse). "
                         "Default None -> use config.yaml.")
parser.add_argument("--compute-final-metrics", type=_str2bool, default=None,
                    help="Override config 'compute_final_metrics' (true/false). "
                         "Default None -> use config.yaml.")
args = parser.parse_args()

# ── Config ────────────────────────────────────────────────────────────────────
config_path = os.path.join(os.path.dirname(__file__), "../config/config.yaml")
with open(config_path) as f:
    cfg = yaml.safe_load(f)

run_id_classic = args.run_id_classic
gene_set, radius, dataset = args.gene_set, args.radius, args.dataset
model_type, lam = args.model_type, args.lam

SEED            = args.seed if args.seed is not None else cfg.get("seed", 42)
# Outer (donor-level) CV strategy from the SHARED top-level cv: block (registry in
# src/data/splitting.py). Default lopo_next_val. CV_PARAMS feeds the strategy
# (e.g. fixed_holdout_kfold reads {k: 5}).
CV_CFG          = cfg.get("cv") or {}
CV_STRATEGY     = CV_CFG.get("strategy", "lopo_next_val")
CV_PARAMS       = dict(CV_CFG.get("params") or {})
# A passed CLI override (--lr, --batch-size, --max-epochs, --patience) wins over
# config.yaml; otherwise the config value is used.
LR              = float(args.lr) if args.lr is not None else float(cfg["lr"])
WEIGHT_DECAY    = float(cfg["weight_decay"])
BATCH_SIZE      = args.batch_size if args.batch_size is not None else cfg["batch_size"]
MAX_EPOCHS      = args.max_epochs if args.max_epochs is not None else cfg["max_epochs"]
PATIENCE        = args.patience if args.patience is not None else cfg["patience"]
CENTER          = cfg["centering_from_train"]
# Per-gene target standardization. Gated to X-target models below (count models
# train on raw counts and must not be z-scored). When inactive the per-fold stats
# are None and the whole transform path is a no-op.
TARGET_TRANSFORM = cfg.get("target_transform", "none")
TRAIN_METRIC_SPOTS = cfg.get("train_metric_spots_per_slide", None)
CORR_METRICS    = tuple(cfg.get("corr_metrics", ("pearson",)))
# Best-epoch selection criterion, the classic-pipeline counterpart of
# gcm.early_stop_criterion. "corr" (default) maximizes corr_metrics[0] on val.
# "val_loss" minimizes the validation loss, the model's own training objective, used to
# reproduce a baseline's own training recipe. "mse" minimizes the validation MSE on the
# axis the correlation is reported on (log1p(mu/depth_scale) for the count specs), so it
# means the same thing for every model family.
EARLY_STOP_CRITERION = (
    args.early_stop_criterion if args.early_stop_criterion is not None
    else cfg.get("early_stop_criterion", "corr")
)
# Donor-level test holdout (REQUIRED): explicit donor IDs held out as the fixed
# test set. CV then groups by DONOR over the rest (leakage-safe — slides of one
# patient never straddle the split). No slide-level fallback. donor->slides map:
# snake/config/donor_slide_map.yaml.
TEST_DONORS     = cfg["datasets"][dataset].get("test_donors")

# Count-model knob (Poisson / NB; a no-op for X-target models): nb_log_alpha_init
# seeds the NB dispersion.
NB_LOG_ALPHA_INIT = cfg.get("nb_log_alpha_init", -2.0)
# Common sequencing depth T (per dataset) for the cp count families: the CP10k
# target is rescaled to CP_T. depth_scale = T/1e4 is resolved from it below (None
# for datasets without it; only cp specs read it, and a clear error is raised there).
COUNT_COMMON_DEPTH = cfg["datasets"][dataset].get("count_common_depth")
# Model-specific constructor kwargs; build_model forwards only those each model
# accepts, so the same dict is safe for every model_type. log_mu_bias_init is
# appended below for the cp families.
MODEL_KWARGS      = {"log_alpha_init": NB_LOG_ALPHA_INIT}

# Large-batch stability knobs + matmul precision (all default to a no-op).
WARMUP_EPOCHS   = (
    int(args.warmup_epochs) if args.warmup_epochs is not None
    else int(cfg.get("warmup_epochs", 0))
)
GRAD_CLIP_NORM  = cfg.get("grad_clip_norm", None)
MATMUL_TF32     = bool(cfg.get("matmul_tf32", False))
# Opt-in diagnostic: also log per-epoch TEST metrics to the learning curve.
# Never affects training/selection (early stopping stays on val); just adds a
# test forward pass per epoch so the generalization gap can be plotted.
LOG_TEST_CURVE      = bool(cfg.get("log_test_curve", False))
SAVE_LEARNING_CURVES = bool(cfg.get("save_learning_curves", True))
NB_PER_EPOCH_CALIB  = bool(cfg.get("nb_per_epoch_calibration", True))
# Fold-end gating, using the same two top-level config keys the GCM reads.
# COMPUTE_FINAL_METRICS covers the final per-fold inference, the scipy/g* metrics,
# atoms.parquet and the full metrics.json headline. SAVE_PREDICTIONS covers the large
# preds_fold{k}.h5ad writes. Both default True. With compute_final_metrics=False the run
# still saves the predictions (subject to save_predictions) but writes a stub
# metrics.json and no atoms; the metrics are recomputed from predictions/ by
# snake/scripts/complete_deferred_metrics.py.
COMPUTE_FINAL_METRICS = (
    args.compute_final_metrics if args.compute_final_metrics is not None
    else bool(cfg.get("compute_final_metrics", True))
)
SAVE_PREDICTIONS      = bool(cfg.get("save_predictions", True))

# Base architecture override (config: architecture). None -> the model defaults in
# src/models/mlp.py. The dict keys (hidden_dims / dropout / activation) are passed
# to build_model via train_one_fold(arch=...). Not a path param -> bump run_id_classic.
ARCH = cfg.get("architecture") or None

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Optional TF32 / high-precision matmul (config: matmul_tf32; default false) ─
# An MLP is a stack of matmuls, so TF32 tensor cores on Ampere+ GPUs speed up training
# at a small precision cost. Off by default.
if MATMUL_TF32 and torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
# insert(0) (not append) so this checkout's own src/ wins over any editable
# `pip install -e .` pointing at a different copy of the repo.
sys.path.insert(0, project_root)

from src.data.load_and_preprocess import read_arrays
from src.data.splitting import (
    load_slide_to_donor, donor_labels as donor_labels_from_slides,
    load_donor_order, cv_donor_folds, donor_ids_to_spot_idx, is_lopo_strategy,
)
from src.data.dataset_indexed import SharedTensors
from src.models.specs import get_spec, kind_of
from src.models.mlp import _DEFAULT_HIDDEN, _DEFAULT_DROPOUT, _DEFAULT_ACT
from src.training.train_loop_indexed import train_one_fold
from src.training.inference_indexed import run_inference
from src.training.target_transform import compute_gene_target_stats

spec = get_spec(model_type)

out_metrics = args.out_metrics
out_model   = args.out_model
out_dir     = os.path.dirname(out_metrics)
os.makedirs(out_dir, exist_ok=True)

# Per-fold learning-curve CSVs, one row per epoch. Same layout as the GCM pipeline.
lc_dir = os.path.join(out_dir, "learning_curves")
if SAVE_LEARNING_CURVES:
    os.makedirs(lc_dir, exist_ok=True)

# ── Load + slice y to gene_set ────────────────────────────────────────────────
# The spec decides which expression matrix to read (.X vs .layers["raw_counts"]).
# Everything else is identical across models.
import anndata as ad
# Log the exact preprocessed adata this run trains on (carries the preproc_id
# segment) so the chosen preprocessing is visible in the log, not just metrics.json.
print(f"Input adata (preprocessed): {os.path.abspath(args.data)}")
adata = ad.read_h5ad(args.data)
# ── depth_scale, the one count constant ──────────────────────────────────────
# One float per run, dataset-dependent. For the cp families it is T/1e4 and it plays
# exactly two roles: it builds the CP_T target in read_arrays, and it divides before
# log1p when the mean correlation is reported (g*, the per-epoch option-A metric, the
# *_logmean cross-check). It is neither an offset nor a per-spot exposure. For
# raw_counts it is None, not 1.0: a raw-count model is aligned to no depth at all.
DEPTH_SCALE = None
if spec.target_source in ("cp_counts", "cp_counts_nr"):
    if COUNT_COMMON_DEPTH is None:
        raise KeyError(
            f"model_type '{model_type}' uses target_source='{spec.target_source}' but config "
            f"datasets.{dataset} has no 'count_common_depth'."
        )
    DEPTH_SCALE = float(COUNT_COMMON_DEPTH) / 1e4
    # Shift the mean head's bias by log(depth_scale) so a cp run starts at the target's
    # own order of magnitude, instead of spending part of a finite epoch budget walking
    # that bias. The model adds to the default random Linear bias rather than overwriting
    # it (see PoissonMLP), so depth_scale = 1.0 gives log(1.0) = 0.0, an exact no-op.
    # Nothing is passed for raw (depth_scale is None), and build_model filters
    # model_kwargs by signature, so the Gaussians ignore this key.
    MODEL_KWARGS["log_mu_bias_init"] = math.log(DEPTH_SCALE)
embeddings, y_all, sample_labels, available_genes = read_arrays(
    adata, target_source=spec.target_source, depth_scale=DEPTH_SCALE,
)
# Per-spot identity keys for the predictions AnnData. Both reference the UNIFIED
# merged adata (this rule's input) and are positionally aligned with the
# SharedTensors rows (read_arrays is positional and SharedTensors preserves row
# order). Written to obs in fold_predictions_adata so a prediction can be joined
# back to adata.obs / adata.obsm["spatial"]:
#   • adata_row — the row position i into this merged adata.h5ad
#   • obs_name  — the unique spot barcode (robust key; survives a re-preprocess)
obs_names = adata.obs_names.to_numpy()

requested_genes = list(available_genes) if args.genes == "all" else sorted(args.genes.split(","))
missing = [g for g in requested_genes if g not in available_genes]
if missing:
    raise ValueError(
        f"Gene set '{gene_set}' requests genes absent from the dataset's shared "
        f"panel: {missing}. Remove them from the gene_set, or check the dataset."
    )
gene_idx = [available_genes.index(g) for g in requested_genes]
y = y_all[:, gene_idx]                       # always 2-D (n_spots, n_genes)
gene_names = requested_genes
n_genes = len(gene_names)

# Full gene list -> a sibling provenance file, so neither this run's log nor the
# Snakemake command echo needs to inline thousands of names. For the all-genes /
# multi-gene sets the startup line below prints only a short preview; the
# complete list lives here (one gene per line) and in metrics.json["gene_names"].
with open(os.path.join(out_dir, "genes.txt"), "w") as f:
    f.write("\n".join(gene_names) + "\n")
genes_preview = (", ".join(gene_names) if n_genes <= 10
                 else ", ".join(gene_names[:5]) + f", ... (+{n_genes - 5} more)")

# ── Per-slide spot-count sanity flag (loud, at startup; never silent-drop) ────
slides, counts = np.unique(sample_labels, return_counts=True)
print(f"Per-slide spot counts ({len(slides)} slides):")
for s, c in zip(slides, counts):
    flag = "  <-- LOW" if c < 50 else ""
    print(f"  {s:<16} {c:>6}{flag}")
low = [str(s) for s, c in zip(slides, counts) if c < 50]

# ── Per-gene target standardization gate ──────────────────────────────────────
# Active only for X-target models (MSE + Gaussians); count models train on raw
# counts and would break their likelihood if z-scored, so the request is ignored
# (loudly) for them. The actual per-fold loc/scale are computed inside the fold
# loop on TRAIN rows only (compute_gene_target_stats), like the embedding train_mean.
DO_TARGET_TRANSFORM = TARGET_TRANSFORM != "none" and spec.target_source == "X"
if TARGET_TRANSFORM != "none" and spec.target_source != "X":
    print(f"NOTE: target_transform={TARGET_TRANSFORM!r} ignored for "
          f"target_source={spec.target_source!r} (count model); leaving target raw.")

# ── metric_scale: what the reported correlation is actually computed on ───────
# One explicit arm per legal target_source. The four arms cover every target_source a
# registered spec can carry ("cp_counts", "cp_counts_nr", "raw_counts", "X"), so the
# final else is a default for no real family: it fires only when a target_source is
# added without an arm here, and it says so loudly instead of mislabelling the run.
# Computed once and written into both the stub and the full metrics.json, so the two
# cannot disagree. The three count reporting axes:
#   _cp, _cp_nr -> log1p(CP10k)      comparable to the Gaussian models
#   _raw        -> log1p(raw counts) comparable to neither the Gaussian nor _cp
if spec.target_source == "cp_counts":
    METRIC_SCALE = (
        f"common-depth count (T={COUNT_COMMON_DEPTH}, depth_scale={DEPTH_SCALE}); the "
        f"NB/Poisson is fitted DIRECTLY on the target round(CP_T) -- no offset, no "
        f"size factor, no per-spot exposure -- so the saved X(mu) and the dispersion "
        f"alpha are BOTH on that CP_T scale. Reported corr = g*=E[log1p(Y/depth_scale)] "
        f"vs log1p(round(CP_T)/depth_scale) = log1p(CP10k), comparable to the Gaussian "
        f"models. BOTH log1p estimators are reported, as scalars AND per (slide,gene) "
        f"in atoms.parquet: {{metric}} = g* (full fitted predictive, uses mu AND the "
        f"dispersion alpha) and {{metric}}_logmean = log1p(mu/depth_scale) (MEAN-ONLY, "
        f"ignores alpha, so it sits above g* by the Jensen gap and is the scale the "
        f"per-epoch / early-stop metric uses). Calibration "
        f"(native-CDF one-sided PIT) and var_residual_corr[_partial] stay on the CP_T "
        f"scale with no constant applied. pred_var NOT persisted "
        f"(= count_pred_var(X[,dispersion]))")
elif spec.target_source == "cp_counts_nr":
    METRIC_SCALE = (
        f"common-depth count NO-ROUNDING (T={COUNT_COMMON_DEPTH}, "
        f"depth_scale={DEPTH_SCALE}); target = CONTINUOUS CP_T = CP10k*depth_scale (no "
        f"np.rint), fitted DIRECTLY by the discrete Poisson/NB via the continuous NLL "
        f"(NB validate_args=False) -- no offset, no size factor. X(mu) and alpha are "
        f"BOTH on that CP_T scale. Reported corr = g*=E[log1p(Y/depth_scale)] vs "
        f"log1p(CP_T/depth_scale) = log1p(CP10k) EXACT (Gaussian-comparable, "
        f"un-quantized). BOTH log1p estimators are reported, as scalars AND per "
        f"(slide,gene) in atoms.parquet: {{metric}} = g* (full fitted predictive, uses "
        f"mu AND alpha) and {{metric}}_logmean = log1p(mu/depth_scale) (MEAN-ONLY, "
        f"ignores alpha, above g* by the Jensen gap; the early-stop scale). "
        f"Calibration = native-CDF "
        f"one-sided PIT with the SMEARED CDF F~, on the CP_T scale. pred_var NOT "
        f"persisted (= count_pred_var(X[,dispersion]))")
elif spec.target_source == "raw_counts":
    METRIC_SCALE = (
        "depth-blind raw counts (depth_scale=None -> divisor 1.0); the NB/Poisson is "
        "fitted DIRECTLY on adata.layers['raw_counts'] -- no offset, no size factor, "
        "no per-spot exposure -- so X(mu) and alpha are BOTH on the raw-count scale. "
        "Reported corr = g*=E[log1p(Y)] vs log1p(raw_counts), i.e. the HEADLINE IS ON "
        "log1p(RAW COUNTS): it is NOT comparable to the Gaussian models and NOT "
        "comparable to the _cp / _cp_nr families, only to other _raw runs. BOTH log1p "
        "estimators are reported, as scalars AND per (slide,gene) in atoms.parquet: "
        "{metric} = g* (full fitted predictive, uses mu AND alpha) and {metric}_logmean "
        "= log1p(mu) (MEAN-ONLY, ignores alpha, above g* by the Jensen gap; the "
        "early-stop scale). Calibration and var_residual_corr[_partial] are on "
        "the raw-count scale. pred_var NOT persisted (= count_pred_var(X[,dispersion]))")
elif spec.target_source == "X":
    METRIC_SCALE = (f"per-gene standardized log-CP10k ({TARGET_TRANSFORM}, per-gene/"
                    f"per-fold; per-(slide,gene) metrics affine-invariant -> comparable "
                    f"to X)" if DO_TARGET_TRANSFORM else "X [log-CP10k]")
else:
    METRIC_SCALE = f"UNDECLARED metric scale for target_source={spec.target_source!r}"

# ── Build SharedTensors once ─────────────────────────────────────────────────
# Holds the run's data on the GPU when one is available, so a per-batch gather is pure
# on-device indexing with no host-to-device copy. SharedTensors falls back to host memory
# when there is no CUDA device.
shared = SharedTensors(embeddings, y, sample_labels, device=DEVICE,
                       depth_scale=DEPTH_SCALE)
print(f"SharedTensors: {shared.n_spots} spots x {shared.emb_dim} emb dim, "
      f"{shared.n_genes} gene(s); device={DEVICE}")
print(f"Run: run_id_classic={run_id_classic} dataset={dataset} gene_set={gene_set} "
      f"n_genes={n_genes} genes=[{genes_preview}], test_donors={TEST_DONORS}, radius={radius} "
      f"model={model_type} (target={spec.target_source}, "
      f"loss={spec.loss_display_name}, uses_lam={spec.uses_lam}, "
      f"probabilistic={spec.is_probabilistic}) lam={lam} center={CENTER} "
      f"depth_scale={DEPTH_SCALE}")
print(f"Seed: {SEED} | CV strategy: {CV_STRATEGY} | max_epochs: {MAX_EPOCHS} | "
      f"patience: {PATIENCE} | "
      f"train_metric_spots_per_slide: {TRAIN_METRIC_SPOTS}")
print(f"Training config: LR={LR}, weight_decay={WEIGHT_DECAY}, "
      f"batch_size={BATCH_SIZE}, warmup_epochs={WARMUP_EPOCHS}, "
      f"corr_metrics={CORR_METRICS}, early_stop_criterion={EARLY_STOP_CRITERION}")
print(f"Architecture: {('config '+str(ARCH)) if ARCH else 'model defaults'}")

# free the numpy copies; SharedTensors owns the tensors now
del embeddings, y, y_all


# ── Storage helpers ──────────────────────────────────────────────────────────
# Predictions are one AnnData per fold (fold_predictions_adata below), val + test spots
# stacked, with pred / y / pred_var as (n_spots x n_genes) matrices in X and layers. A
# melted (spot, gene) table would have n_spots*n_genes rows, which does not fit at the
# all-genes tier. atoms.parquet stays per-(slide, gene).
preds_dir = os.path.join(out_dir, "predictions")
os.makedirs(preds_dir, exist_ok=True)


def fold_subsample_idx(train_idx_arr, rng):
    """Fixed seeded per-slide subsample of train rows, so the per-epoch train metric
    stays cheap. Returns None when no cap is configured. Sampling is uniform within a
    slide, so tissue proportions are preserved; a slide smaller than the cap keeps all
    its spots."""
    if TRAIN_METRIC_SPOTS is None:
        return None
    g = shared.groups[train_idx_arr]
    keep = []
    for s in np.unique(g):
        rows = train_idx_arr[g == s]
        if len(rows) > TRAIN_METRIC_SPOTS:
            rows = rng.choice(rows, size=TRAIN_METRIC_SPOTS, replace=False)
        keep.append(rows)
    return np.concatenate(keep)


# The code above is model agnostic. It does not depend on which model is used.

# Below the pred dataframe and atom dataframe depend on the inference output, which depends on the model (e.g. pred_var only for Gaussian).


def fold_predictions_adata(results_by_split, row_idx_by_split, fold,
                           target_loc=None, target_scale=None):
    """One AnnData holding this fold's predictions, val + test spots stacked.

    Predictions are matrices, not a melted (spot, gene) table, so the file grows with
    n_spots and n_genes rather than with their product:
        X                  = pred (means)            (n_spots x n_genes)
        layers["y"]        = ground truth
        layers["pred_var"] = predictive variance     (Gaussian specs only; count models
                             omit it, recompute offline via count_metrics.count_pred_var)
        layers[<extra>]    = one per spec.extra_prediction_columns (e.g. NB dispersion)
        obs                = slide, obs_name, adata_row, split, fold
        var_names          = gene_names
        obs_names          = obs_name (barcode; unique within a fold)
    obs_name and adata_row are the join keys back to adata.obs, adata.obsm["spatial"] and
    ground_truth. Prefer obs_name: adata_row is positional and is only valid against the
    exact adata.h5ad this run read.

    `results_by_split` and `row_idx_by_split` are dicts {split: ...} sharing the same key
    order (val, test); their arrays are concatenated in that order so every obs column
    lines up with every matrix row.
    """
    means, ys, pred_vars = [], [], []
    slides, splits, rows = [], [], []
    extras = {col: [] for col in spec.extra_prediction_columns}
    has_pred_var = False
    for split, res in results_by_split.items():
        n = res["means"].shape[0]
        # reshape to (n_spots, n_genes) so the single-gene case (means is 1-D)
        # and the multi-gene case (already 2-D) share one path.
        means.append(res["means"].reshape(n, n_genes))
        ys.append(res["y"].reshape(n, n_genes))
        slides.append(np.asarray(res["groups"]).astype(str))
        splits.append(np.repeat(split, n))
        rows.append(np.asarray(row_idx_by_split[split]))
        if res.get("pred_var") is not None:
            has_pred_var = True
            pred_vars.append(res["pred_var"].reshape(n, n_genes))
        # Extra per-spec arrays, each written as its own (n_spots x n_genes) layer.
        # The NB specs declare "dispersion"; a spec that declares none skips this.
        for col in spec.extra_prediction_columns:
            arr = res.get(col)
            extras[col].append(arr.reshape(n, n_genes) if arr is not None
                               else np.full((n, n_genes), np.nan, dtype=np.float32))

    row_idx_all = np.concatenate(rows)
    obs = pd.DataFrame({
        "slide":     np.concatenate(slides),
        "obs_name":  obs_names[row_idx_all],
        "adata_row": row_idx_all,
        "split":     np.concatenate(splits),
        "fold":      np.repeat(fold, len(row_idx_all)),
    }, index=obs_names[row_idx_all])      # obs_names = barcode (unique within a fold)

    pred = ad.AnnData(X=np.concatenate(means).astype(np.float32), obs=obs)
    pred.var_names = gene_names
    pred.layers["y"] = np.concatenate(ys).astype(np.float32)
    # Count convention marker: "v2_no_offset" means X (mu), layers["y"] and the
    # dispersion all live on the model's own target scale, with no per-spot size factor
    # anywhere (these files carry no obs["size_factor"]). depth_scale is the single
    # reporting constant, None for the depth-blind _raw models. An offline reader should
    # key off count_convention rather than assume a scale.
    pred.uns["count_convention"] = "v2_no_offset"
    # h5ad has no writer for a bare Python None in uns, so the key is written only
    # when depth_scale is a float. Readers must therefore use
    # `adata.uns.get("depth_scale")`, which yields exactly the intended None for the
    # depth-blind _raw models (and for every X-target model).
    if DEPTH_SCALE is not None:
        pred.uns["depth_scale"] = float(DEPTH_SCALE)
    # Count models: pred_var is a deterministic function of the saved X (means) and, for
    # NB, the "dispersion" layer (Poisson var = mu; NB var = mu + alpha*mu^2), so it is
    # NOT persisted -- recompute offline via count_metrics.count_pred_var. Gaussian
    # pred_var is a separate variance HEAD (not recomputable), so it IS saved.
    if has_pred_var and spec.target_source not in (
        "raw_counts", "cp_counts", "cp_counts_nr",
    ):
        pred.layers["pred_var"] = np.concatenate(pred_vars).astype(np.float32)
    for col, lst in extras.items():
        pred.layers[col] = np.concatenate(lst).astype(np.float32)
    # When a per-gene target transform is active, X / layers["y"] / pred_var are on
    # the standardized (z) scale (the saved variance is thus SCALE-FREE — the point of
    # the transform). We additionally store this fold's per-gene loc/scale and two
    # convenience log-CP10k views — the MEAN prediction and the TARGET — so the
    # expression-plotting notebooks read the familiar scale without inverting:
    #   value_logcp10k = scale_g * z_value + loc_g
    # The predictive variance is deliberately NOT given a log-CP10k twin (keeping it
    # scale-free); recover it offline as scale_g**2 * pred_var if ever needed.
    if target_loc is not None:
        loc_row   = np.asarray(target_loc,   dtype=np.float32)      # (n_genes,)
        scale_row = np.asarray(target_scale, dtype=np.float32)
        pred.var["target_loc"]   = loc_row
        pred.var["target_scale"] = scale_row
        pred.uns["target_transform"] = TARGET_TRANSFORM
        pred.layers["means_logcp10k"] = (pred.X * scale_row + loc_row).astype(np.float32)
        pred.layers["y_logcp10k"]     = (pred.layers["y"] * scale_row + loc_row).astype(np.float32)
    return pred


def atoms_dataframe(result, fold, split):
    """Per-(slide,gene) metric atoms for one (fold, split): ONE row per
    (slide, gene), carrying a column for EVERY configured correlation metric
    (pearson, spearman, ...) — plus, for the COUNT specs, a second `{metric}_logmean`
    column holding the mean-only option-A estimator of the same quantity (see the loop
    below) — and, for probabilistic specs, the native-CDF PIT calibration columns
    (ece_pit, pit_bias, disp_signed, var_residual_corr, var_residual_corr_partial).
    Gene indices are mapped back to names.

    All metrics share the same (slide, gene) atoms (same spots, same grouping),
    so we key rows on (slide, gene) and fill one column at a time. The primary
    metric is CORR_METRICS[0] (also stored as primary_corr_metric in
    metrics.json), so a reader knows which column drove model selection.
    """
    rows = {}
    def row_for(a):
        # Seed the (slide, gene) row once. n_spots comes from the corr atoms;
        # calibration atoms don't carry it but always reuse an existing row.
        return rows.setdefault((a["slide"], a["gene"]), {
            "fold": fold, "split": split, "slide": a["slide"],
            "gene": gene_names[a["gene"]], "n_spots": a.get("n_spots"),
        })
    for name in CORR_METRICS:
        for a in result[f"{name}_per_slide_gene"]:
            row_for(a)[name] = a[name]
        # Count specs only: a second estimator of the same log1p quantity, beside the
        # headline column. `{name}` is option B, g* = E[log1p(Y/d)] from the full fitted
        # predictive, so it uses mu and the dispersion alpha. `{name}_logmean` is option
        # A, log1p(mu/d), mean only, ignoring alpha, so it sits above g* by the Jensen
        # gap. Both are kept per (slide, gene), so an offline analysis can ask where
        # routing through the full predictive helps. A non-count spec has one estimator
        # and no such column.
        logmean_atoms = result.get(f"{name}_logmean_per_slide_gene")
        if logmean_atoms is not None:
            for a in logmean_atoms:
                row_for(a)[f"{name}_logmean"] = a[f"{name}_logmean"]
    err = result.get("error_per_slide_gene")
    if err is not None:                    # MAE/MSE on the Pearson scale
        for a in err:
            r = row_for(a)
            r["mae"] = a["mae"]
            r["mse"] = a["mse"]
    cal = result.get("calibration_per_slide_gene")
    if cal is not None:                    # probabilistic specs only
        for a in cal:
            r = row_for(a)
            for k in ("ece_pit", "pit_bias", "disp_signed",
                      "var_residual_corr", "var_residual_corr_partial"):
                r[k] = a[k]
    return pd.DataFrame(list(rows.values()))


# ── Donor-level CV (registry) — LOPO by default ───────────────────────────────
# Slides of one donor are near-duplicates, so train/val/test split by DONOR. The
# registry (cv_donor_folds) returns donor-id partitions; we map them to spot rows
# here. The METRIC is unaffected — it still aggregates per SLIDE (run_inference
# uses shared.groups = slide labels). donor_per_spot drives ONLY the split.
donor_map_path = os.path.join(os.path.dirname(__file__), "../config/donor_slide_map.yaml")
slide_to_donor = load_slide_to_donor(donor_map_path, dataset)
donor_per_spot = donor_labels_from_slides(sample_labels, slide_to_donor)

# Ordered donor list — order from donor_slide_map.yaml (fixes the lopo_next_val
# val rotation); sorted fallback for 1:1 donor==slide datasets absent from the map.
present_donors = set(donor_per_spot.tolist())
donor_order = load_donor_order(donor_map_path, dataset)
if donor_order is None:
    donors = sorted(present_donors)
else:
    donors = ([d for d in donor_order if d in present_donors]
              + sorted(present_donors - set(donor_order)))

LOPO  = is_lopo_strategy(CV_STRATEGY)
folds = cv_donor_folds(donors, CV_STRATEGY, test_donors=TEST_DONORS, **CV_PARAMS)
K     = len(donors)
print(f"CV strategy: {CV_STRATEGY} (lopo={LOPO}) over {K} donors: {donors}")
print(f"  -> {len(folds)} outer fold(s); test rotation: {[f['test'] for f in folds]}")

# compute_gene_target_stats lives in src/training/target_transform.py, shared with the
# GCM so both pipelines compute the transform identically.


rng = np.random.default_rng(SEED)
fold_rows = []
atom_frames = []
cv_folds_info = []        # per OUTER fold: test donor + inner train/val rows -> splits.json
n_params = None           # parameter count, captured once from the first model

# ── Outer (donor) fold loop, then inner (train/val) split loop ────────────────
# LOPO: one outer fold per held-out test donor; one inner split (val = next donor,
# train = rest). inner_splits is a list so a FUTURE strategy can do >1 inner split
# per donor with no change here (the per-donor score then averages over inner).
for outer in folds:
    fold = outer["fold"]
    test_donor_ids   = outer["test"]
    test_donor_label = ",".join(map(str, test_donor_ids))
    test_idx     = donor_ids_to_spot_idx(donor_per_spot, test_donor_ids)
    test_slides  = sorted(set(shared.groups[test_idx].tolist()))
    n_inner      = len(outer["inner_splits"])
    inner_infos  = []

    for inner_i, inner in enumerate(outer["inner_splits"]):
        train_idx = donor_ids_to_spot_idx(donor_per_spot, inner["train"])
        val_idx   = donor_ids_to_spot_idx(donor_per_spot, inner["val"])
        # Per-fold artifact tag: plain fold{f} for the single-inner case (current
        # LOPO/k-fold), fold{f}_inner{j} only if a future strategy adds inner splits.
        tag = f"fold{fold}" if n_inner == 1 else f"fold{fold}_inner{inner_i}"
        t_fold = time.time()

        # Inner split as GLOBAL adata row indices (the predictions `adata_row`
        # coordinate), for splits.json. Explicit train + val (never reconstructed
        # by set difference).
        inner_infos.append({
            "inner_split":  inner_i,
            "train_idx":    train_idx.tolist(),
            "val_idx":      val_idx.tolist(),
            "train_donors": sorted(set(donor_per_spot[train_idx].tolist())),
            "val_slides":   sorted(set(shared.groups[val_idx].tolist())),
            "val_donors":   sorted(set(donor_per_spot[val_idx].tolist())),
        })

        # Train-fold embedding mean (used iff CENTER). Computed on the data's device
        # (GPU when GPU-resident), then pulled to host as numpy: it is saved with the
        # model and re-applied at inference.
        _tr_idx_t = torch.as_tensor(train_idx, dtype=torch.long, device=shared.emb.device)
        train_mean = shared.emb[_tr_idx_t].mean(dim=0).cpu().numpy()
        # Per-gene target standardization stats — TRAIN rows of THIS fold only (so they
        # differ across folds), saved with the checkpoint and applied (z-space) at
        # train + inference. (None, None) unless DO_TARGET_TRANSFORM. slide labels drive
        # the within-slide variance in gene_zscore.
        target_loc, target_scale = (
            compute_gene_target_stats(shared.tgt[_tr_idx_t], shared.groups[train_idx],
                                      TARGET_TRANSFORM)
            if DO_TARGET_TRANSFORM else (None, None))
        if target_loc is not None and fold == 0 and inner_i == 0:
            print(f"  target_transform={TARGET_TRANSFORM}: per-gene loc/scale "
                  f"shape={target_loc.shape} (train-only)")
        sub_idx = fold_subsample_idx(train_idx, rng)

        print(f"\n── Fold {fold} (test donor {test_donor_label}) inner {inner_i} | "
              f"train {len(train_idx)} val {len(val_idx)} test {len(test_idx)} spots ── "
              f"val slides {sorted(set(shared.groups[val_idx].tolist()))} ──")
        best_state, epoch_log, model = train_one_fold(
            shared, train_idx, val_idx, train_mean,
            model_type=model_type, lam=lam, n_genes=n_genes,
            lr=LR, weight_decay=WEIGHT_DECAY, batch_size=BATCH_SIZE,
            max_epochs=MAX_EPOCHS, patience=PATIENCE, device=DEVICE,
            center=CENTER, train_subsample_idx=sub_idx, arch=ARCH,
            corr_metrics=CORR_METRICS,
            warmup_epochs=WARMUP_EPOCHS, grad_clip_norm=GRAD_CLIP_NORM,
            test_idx=test_idx, log_test_curve=LOG_TEST_CURVE,
            model_kwargs=MODEL_KWARGS,
            target_loc=target_loc, target_scale=target_scale,
            nb_per_epoch_calibration=NB_PER_EPOCH_CALIB,
            early_stop_criterion=EARLY_STOP_CRITERION,
        )
        if n_params is None:              # identical across folds; capture once
            n_params = int(sum(p.numel() for p in model.parameters()))

        # best_epoch and stop_reason are derived from epoch_log, so train_one_fold keeps
        # its 3-tuple return. best_epoch reproduces train_one_fold's own selection scalar
        # for EARLY_STOP_CRITERION; both keep the earlier epoch on a tie, which is what
        # the first-occurrence rule of nanargmin/nanargmax gives:
        #   "corr"     -> first argmax of the primary val corr metric.
        #   "val_loss" -> first argmin of the val loss.
        #   "mse"      -> first argmin of val_mse_corr_axis, the MSE on the correlation's
        #                 own axis, not val_mse, which is on the model's target scale.
        # A criterion with no arm here does not crash. It falls through to the corr branch
        # and silently writes a best_epoch that disagrees with the weights kept, so every
        # criterion added to train_one_fold must be added here too.
        # stop_reason = early_stop if patience fired before MAX_EPOCHS, else max_epochs.
        if EARLY_STOP_CRITERION == "val_loss":
            _val_select = [e.get("val_loss", float("nan")) for e in epoch_log]
            best_epoch = (int(np.nanargmin(_val_select)) + 1
                          if np.any(np.isfinite(_val_select)) else None)
        elif EARLY_STOP_CRITERION == "mse":
            _val_select = [e.get("val_mse_corr_axis", float("nan")) for e in epoch_log]
            best_epoch = (int(np.nanargmin(_val_select)) + 1
                          if np.any(np.isfinite(_val_select)) else None)
        else:
            _val_primary = [e.get(f"val_{CORR_METRICS[0]}", float("nan")) for e in epoch_log]
            best_epoch = (int(np.nanargmax(_val_primary)) + 1
                          if np.any(np.isfinite(_val_primary)) else None)
        stop_reason = "early_stop" if len(epoch_log) < MAX_EPOCHS else "max_epochs"

        # Final val + test inference with the best model. Gated + TIMED (compute_dt) so
        # the per-fold log shows where fold-end time goes (metric compute vs file save).
        # compute_cpu_metrics=COMPUTE_FINAL_METRICS: when False, return the prediction
        # matrices WITHOUT the scipy/g* pass (save now, recompute metrics offline).
        need_inference = SAVE_PREDICTIONS or COMPUTE_FINAL_METRICS
        val_res = test_res = None
        compute_dt = save_dt = 0.0
        if need_inference:
            t_compute = time.time()
            val_res = run_inference(
                model, shared, val_idx, train_mean, spec=spec,
                batch_size=BATCH_SIZE, device=DEVICE, lam=lam, center=CENTER,
                corr_metrics=CORR_METRICS,
                target_loc=target_loc, target_scale=target_scale,
                compute_cpu_metrics=COMPUTE_FINAL_METRICS,
            )
            test_res = run_inference(
                model, shared, test_idx, train_mean, spec=spec,
                batch_size=BATCH_SIZE, device=DEVICE, lam=lam, center=CENTER,
                corr_metrics=CORR_METRICS,
                target_loc=target_loc, target_scale=target_scale,
                compute_cpu_metrics=COMPUTE_FINAL_METRICS,
            )
            compute_dt = time.time() - t_compute

        # Save model + reproduction info (train_mean AND center flag; per-gene target
        # transform name + loc/scale so inference can re-apply the same z-space
        # transform; fold/inner/test_donor identity). All train-fold quantities.
        torch.save({"state_dict": best_state, "train_mean": train_mean,
                    "center": CENTER, "gene_names": gene_names,
                    "model_type": model_type, "fold": fold,
                    "inner_split": inner_i, "test_donor": test_donor_label,
                    "target_transform": TARGET_TRANSFORM if DO_TARGET_TRANSFORM else "none",
                    "target_loc": target_loc, "target_scale": target_scale,
                    # Count convention of the weights inside: "v2_no_offset" = the
                    # model emits mu on its own target scale, with no size factor to
                    # re-apply at load time. Loaders read state_dict only, so the
                    # extra keys are backward-harmless.
                    "count_convention": "v2_no_offset",
                    "depth_scale": DEPTH_SCALE},
                   os.path.join(out_dir, f"model_{tag}.pt"))

        # Per-epoch learning curve for this fold: one row per epoch, columns are the
        # epoch-log keys (epoch, lr, train_mse, val_mse, train_pearson, val_pearson, plus
        # train_ece/val_ece for probabilistic specs, and test_* columns when
        # log_test_curve is on).
        if SAVE_LEARNING_CURVES:
            pd.DataFrame(epoch_log).to_csv(
                os.path.join(lc_dir, f"{tag}.csv"), index=False)
        # This fold's predictions as one AnnData (val + test stacked), written only when
        # SAVE_PREDICTIONS and timed as save_dt. lzf compresses fast; gzip is
        # single-threaded and dominates the fold-end write. The held-out test donor and
        # the inner split are tagged in obs.
        pred_adata = None
        if SAVE_PREDICTIONS:
            t_save = time.time()
            pred_adata = fold_predictions_adata(
                {"val": val_res, "test": test_res},
                {"val": val_idx, "test": test_idx}, fold,
                target_loc=target_loc, target_scale=target_scale,
            )
            pred_adata.obs["inner_split"] = inner_i
            pred_adata.obs["test_donor"]  = test_donor_label
            pred_adata.write_h5ad(os.path.join(preds_dir, f"preds_{tag}.h5ad"),
                                  compression="lzf")
            save_dt = time.time() - t_save
        fold_dt = time.time() - t_fold
        print(f"  fold {fold} inner {inner_i}: compute {compute_dt:.1f}s "
              f"save {save_dt:.1f}s (fold {fold_dt:.1f}s)", flush=True)
        # compute_final_metrics=False: skip atoms + per-fold headline (recompute offline
        # from predictions/); metrics.json is written as a STUB after the loop.
        if not COMPUTE_FINAL_METRICS:
            # Record the per-fold training facts anyway. They are known here and nowhere
            # else: complete_deferred_metrics.py rebuilds per_fold from the saved
            # predictions, which carry no trace of how training went, so without this a
            # deferred run's metrics.json silently loses best_epoch, stop_reason and the
            # slide lists. Only what the completion cannot recompute goes in; it merges
            # these keys without overwriting its own values. The two losses are absent
            # because the final inference that produces them is what this branch skips.
            fold_rows.append({
                "fold":                  fold,
                "inner_split":           inner_i,
                "test_donor":            test_donor_label,
                "fold_duration_seconds": round(time.time() - t_fold, 3),
                "best_epoch":            best_epoch,
                "n_epochs_ran":          len(epoch_log),
                "stop_reason":           stop_reason,
                "validation_slides":     sorted(set(shared.groups[val_idx].tolist())),
                "test_slides":           test_slides,
            })
            del best_state, model, train_mean, sub_idx
            if val_res is not None:
                del val_res, test_res
            if pred_adata is not None:
                del pred_adata
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        for split, res in (("val", val_res), ("test", test_res)):
            af = atoms_dataframe(res, fold, split)
            af["inner_split"] = inner_i
            af["test_donor"]  = test_donor_label   # the fold's held-out donor
            atom_frames.append(af)

        # Per-fold scalars for aggregation. One val_/test_ entry per configured
        # correlation metric, each named after the ACTUAL metric (val_pearson,
        # val_spearman, ...). With LOPO, test_<primary> here IS the per-donor point
        # x_i (the held-out donor's slide-aware, gene-averaged corr). ECE is
        # meaningful only for probabilistic specs (NaN otherwise).
        fold_row = {
            "fold":               fold,
            "inner_split":        inner_i,
            "test_donor":         test_donor_label,
            "fold_duration_seconds": round(time.time() - t_fold, 3),
            "best_epoch":         best_epoch,
            "n_epochs_ran":       len(epoch_log),
            "stop_reason":        stop_reason,
            "validation_slides":  sorted(set(shared.groups[val_idx].tolist())),
            "test_slides":        test_slides,
            # Native-CDF PIT calibration (ece_pit=magnitude, pit_bias=location,
            # disp_signed=over/under-confidence). NaN/None for non-probabilistic.
            "val_ece_pit":        val_res.get("ece_pit"),
            "test_ece_pit":       test_res.get("ece_pit"),
            "val_pit_bias":       val_res.get("pit_bias"),
            "test_pit_bias":      test_res.get("pit_bias"),
            "val_disp_signed":    val_res.get("disp_signed"),
            "test_disp_signed":   test_res.get("disp_signed"),
            "val_var_residual_corr_partial":  val_res.get("var_residual_corr_partial"),
            "test_var_residual_corr_partial": test_res.get("var_residual_corr_partial"),
            # Magnitude error on the log1p(CP10k) / Pearson scale (median over genes);
            # within-family only — see metric_scale.
            "val_mae_median":     val_res.get("mae_median"),
            "test_mae_median":    test_res.get("mae_median"),
            "val_mse_median":     val_res.get("mse_median"),
            "test_mse_median":    test_res.get("mse_median"),
            "val_loss":           val_res["loss"],
            "test_loss":          test_res["loss"],
        }
        for cm in CORR_METRICS:
            fold_row[f"val_{cm}"]  = val_res[cm]
            fold_row[f"test_{cm}"] = test_res[cm]
            # Count models also report the log1p(mu/depth_scale) cross-check (option A)
            # beside the g* headline (option B). run_inference sets the *_logmean keys for
            # count specs only. The scalar comes from the CPU scipy pass, so it agrees
            # with the on-device per-epoch value only to about the sixth decimal (float32
            # operands against float64). It is a cross-check, never a selection criterion.
            if f"{cm}_logmean" in val_res:
                fold_row[f"val_{cm}_logmean"]  = val_res[f"{cm}_logmean"]
                fold_row[f"test_{cm}_logmean"] = test_res[f"{cm}_logmean"]
        fold_rows.append(fold_row)

        del best_state, model, val_res, test_res, train_mean, sub_idx, pred_adata
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    cv_folds_info.append({
        "fold":         fold,
        "test_donor":   test_donor_ids,
        "test_idx":     test_idx.tolist(),
        "test_slides":  test_slides,
        "inner_splits": inner_infos,
    })

# ── splits.json — the exact CV partition as GLOBAL adata row indices ─────────
# Written ALWAYS (no metrics needed). These indices share the predictions `adata_row`
# coordinate (positional rows into the unified merged adata.h5ad), so any split joins
# straight back to adata.obs / adata.obsm["spatial"] / predictions. The test set is PER
# OUTER FOLD (each fold holds out a different donor under LOPO), so test_idx + per-fold
# train/val live under "folds".
splits = {
    "seed":        SEED,
    "cv_strategy": CV_STRATEGY,
    "cv_params":   CV_PARAMS,
    "lopo":        LOPO,
    # Split grouping is by DONOR; metric aggregation is by SLIDE (see train.py).
    "split_grouping":  "donor",
    "metric_grouping": "slide",
    "donors_ordered":  donors,
    # fixed_holdout_kfold: the fixed test-donor set (same every fold). Under LOPO
    # this is None — each fold's held-out donor is in folds[*].test_donor.
    "test_donors": (None if LOPO else (folds[0]["test"] if folds else [])),
    "folds":       cv_folds_info,
}
with open(os.path.join(out_dir, "splits.json"), "w") as f:
    json.dump(splits, f)

# ── provenance built BEFORE the metrics gate, so the STUB carries it too ──────
# architecture (the model-reload triplet the posthoc scripts need) + input data
# paths. n_params was captured from fold 0; ARCH/_DEFAULT_* are config/module
# constants. The full-metrics path below references these same names.
if ARCH:
    architecture = {
        "arch_source":  "config",
        "hidden_dims":  ARCH.get("hidden_dims", list(_DEFAULT_HIDDEN)),
        "dropout":      ARCH.get("dropout", list(_DEFAULT_DROPOUT)),
        "activation":   ARCH.get("activation", _DEFAULT_ACT),
        "in_dim":       int(shared.emb_dim),
        "out_dim":      n_genes,
        "n_parameters": n_params,
    }
else:
    architecture = {
        "arch_source":  "default",
        "hidden_dims":  list(_DEFAULT_HIDDEN),
        "dropout":      list(_DEFAULT_DROPOUT),
        "activation":   _DEFAULT_ACT,
        "in_dim":       int(shared.emb_dim),
        "out_dim":      n_genes,
        "n_parameters": n_params,
    }
# input provenance: the merged adata this run trained on + its preprocess config.
input_adata = os.path.abspath(args.data)
preprocess_config_path = os.path.join(os.path.dirname(input_adata),
                                      "preprocess_config.json")

# compute_final_metrics=False -> write a STUB metrics.json (+ copy the model.pt sentinel
# the Snakemake rule expects) and STOP; atoms/metrics/lopo are recomputed offline from
# predictions/. Mirrors the GCM fast path. (model_fold0.pt is always written in-loop.)
if not COMPUTE_FINAL_METRICS:
    import shutil
    stub = {
        "metrics_deferred": True,
        "note": ("compute_final_metrics=False - no per-fold scipy/g* was run; "
                 "recompute atoms/metrics/lopo offline from predictions/."),
        # run identity
        "run_id_classic": run_id_classic, "dataset": dataset, "gene_set": gene_set,
        "radius": radius, "model_type": model_type, "model_kind": kind_of(model_type),
        "primary_corr_metric": CORR_METRICS[0], "corr_metrics": list(CORR_METRICS),
        "n_genes": n_genes, "saved_predictions": SAVE_PREDICTIONS,
        "run_duration_seconds": round(time.time() - RUN_START, 3),
        # model-reload triplet (mc_dropout / laplace posthoc need these) + input data
        "architecture": architecture, "center": CENTER,
        "target_source": spec.target_source,
        "target_transform": TARGET_TRANSFORM if DO_TARGET_TRANSFORM else "none",
        "input_adata": input_adata,
        "preprocess_config": preprocess_config_path if os.path.exists(preprocess_config_path) else None,
        # provenance to recompute atoms/metrics/lopo offline
        "count_convention": "v2_no_offset",
        "depth_scale": DEPTH_SCALE,
        "count_common_depth": COUNT_COMMON_DEPTH,
        "metric_scale": METRIC_SCALE,
        "seed": SEED, "cv_strategy": CV_STRATEGY, "cv_params": CV_PARAMS, "lopo": LOPO,
        "donors_ordered": donors, "test_donors_per_fold": [f["test"] for f in folds],
        "n_donors": K, "n_folds": int(len(folds)),
        "test_donors": (None if LOPO else (folds[0]["test"] if folds else [])),
        "split_grouping": "donor", "metric_grouping": "slide",
        # Per-fold TRAINING facts only (best_epoch / n_epochs_ran / stop_reason / the two
        # slide lists), appended in the loop's deferred branch above. Nothing here is
        # recomputable from the saved predictions, which is exactly why it is written:
        # complete_deferred_metrics.py merges these into the per_fold rows it rebuilds,
        # never overwriting a value it computed itself.
        "per_fold": fold_rows,
        # Full hyperparameter snapshot so the completed metrics.json is self-contained
        # (complete_deferred_metrics.py merges these straight through). KEEP IN SYNC
        # with the full metrics.json block below.
        "lam": lam if spec.uses_lam else None,
        "uses_lam": spec.uses_lam, "is_probabilistic": spec.is_probabilistic,
        "learning_rate": LR, "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
        "early_stop_criterion": EARLY_STOP_CRITERION,
        "warmup_epochs": WARMUP_EPOCHS, "grad_clip_norm": GRAD_CLIP_NORM,
        "nb_log_alpha_init": NB_LOG_ALPHA_INIT, "matmul_tf32": MATMUL_TF32,
        "log_test_curve": LOG_TEST_CURVE, "save_learning_curves": SAVE_LEARNING_CURVES,
        "train_metric_spots_per_slide": TRAIN_METRIC_SPOTS,
    }
    with open(out_metrics, "w") as f:
        json.dump(stub, f, indent=2)
    shutil.copy(os.path.join(out_dir, "model_fold0.pt"), out_model)
    print(f"compute_final_metrics=False -> wrote STUB {out_metrics} + copied model.pt "
          f"(atoms/metrics/lopo deferred; recompute offline from predictions/).", flush=True)
    sys.exit(0)

# ── Full path (compute_final_metrics=True): atoms + aggregation + metrics.json ──
atoms_df = pd.concat(atom_frames, ignore_index=True)
atoms_df.to_parquet(os.path.join(out_dir, "atoms.parquet"), index=False)

# ── Aggregate across folds ───────────────────────────────────────────────────
fold_df = pd.DataFrame(fold_rows)
primary = CORR_METRICS[0]

def mean_std(col):
    series = fold_df[col]
    return float(series.mean()), float(series.std())

# NaN summary from atoms (per split).
nan_summary = {}
for split in ("val", "test"):
    sub = atoms_df[atoms_df.split == split]
    nan_rows = sub[sub[primary].isna()]
    nan_summary[split] = {
        "n_nan": int(len(nan_rows)),
        "n_total": int(len(sub)),
        "examples of NaN values": nan_rows[["slide", "gene"]].head(10).to_dict("records"),
    }

# Headline aggregates for EVERY configured correlation metric (primary first,
# each named after the metric). Spliced into `metrics` below via **corr_metric_aggs.
# With the default ("pearson",) this yields exactly val_pearson_mean/std etc.
corr_metric_aggs = {}
for cm in CORR_METRICS:
    vm, vsd = mean_std(f"val_{cm}")
    tm, tsd = mean_std(f"test_{cm}")
    corr_metric_aggs[f"val_{cm}_mean"]  = vm
    corr_metric_aggs[f"val_{cm}_std"]   = vsd
    corr_metric_aggs[f"test_{cm}_mean"] = tm
    corr_metric_aggs[f"test_{cm}_std"]  = tsd
    # log1p(mu/depth_scale) cross-check (option A) headline — count models only.
    if f"val_{cm}_logmean" in fold_df.columns:
        lvm, lvsd = mean_std(f"val_{cm}_logmean")
        ltm, ltsd = mean_std(f"test_{cm}_logmean")
        corr_metric_aggs[f"val_{cm}_logmean_mean"]  = lvm
        corr_metric_aggs[f"val_{cm}_logmean_std"]   = lvsd
        corr_metric_aggs[f"test_{cm}_logmean_mean"] = ltm
        corr_metric_aggs[f"test_{cm}_logmean_std"]  = ltsd

# ── LOPO statistics (leave-one-patient-out only) ─────────────────────────────
# The K per-donor scores x_i (each = the held-out donor's slide-aware, gene-
# averaged corr; averaged over inner splits when >1) -> mean, inter-donor SD
# (ddof=1), SEM, and Student-t 95% CI on the mean + a prediction interval for a new
# donor (df = K-1). These intervals are optimistic (folds share K-2 training donors;
# Bengio & Grandvalet 2004), so the K per-donor points are kept alongside. For
# fixed_holdout_kfold the per-fold "test" is one repeated donor, so the t-stats are
# meaningless and lopo_stats is omitted (the k-fold mean+/-std stays the headline).
lopo_stats = None
if LOPO:
    from scipy import stats as _scipy_stats
    lopo_stats = {}
    for cm in CORR_METRICS:
        per_donor = fold_df.groupby("test_donor")[f"test_{cm}"].mean()   # x_i (avg over inner)
        x = per_donor.to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        k = int(len(x))
        mean = float(np.mean(x)) if k else float("nan")
        sd   = float(np.std(x, ddof=1)) if k >= 2 else float("nan")
        sem  = float(sd / np.sqrt(k)) if k >= 2 else float("nan")
        if k >= 2:
            tstar = float(_scipy_stats.t.ppf(0.975, k - 1))
            half  = tstar * sem
            half_pi = tstar * sd * float(np.sqrt(1.0 + 1.0 / k))
            ci, pi = [mean - half, mean + half], [mean - half_pi, mean + half_pi]
        else:
            tstar, ci, pi = float("nan"), [float("nan")] * 2, [float("nan")] * 2
        lopo_stats[cm] = {
            "mean": mean, "sd": sd, "sem": sem, "K": k, "df": k - 1,
            "t_star_0975": tstar, "ci95_mean": ci, "pi95_new_donor": pi,
            "per_donor": {str(d): (None if not np.isfinite(v) else float(v))
                          for d, v in per_donor.items()},
        }
    # Convenience flat keys for the PRIMARY metric (the headline).
    _lp = lopo_stats[primary]
    corr_metric_aggs[f"lopo_{primary}_mean"] = _lp["mean"]
    corr_metric_aggs[f"lopo_{primary}_sd"]   = _lp["sd"]
    corr_metric_aggs[f"lopo_{primary}_sem"]  = _lp["sem"]

# Primary-metric scalars (used for the final stdout summary). For LOPO these equal
# the lopo mean/SD (mean over folds == mean over donors with one inner split each).
val_p,  val_p_sd  = corr_metric_aggs[f"val_{primary}_mean"],  corr_metric_aggs[f"val_{primary}_std"]
test_p, test_p_sd = corr_metric_aggs[f"test_{primary}_mean"], corr_metric_aggs[f"test_{primary}_std"]

# ── Provenance blocks spliced into metrics.json ──────────────────────────────
# environment: NO version-control info — the cluster runs synced files, not a
# git checkout. Captures the interpreter/library/GPU/scheduler context + timing.
environment = {
    "started_at":           STARTED_AT,
    "finished_at":          datetime.now().astimezone().isoformat(),
    "run_duration_seconds": round(time.time() - RUN_START, 3),
    "hostname":             socket.gethostname(),
    "slurm_job_id":         os.environ.get("SLURM_JOB_ID"),
    "slurm_nodelist":       os.environ.get("SLURM_JOB_NODELIST"),
    "argv":                 sys.argv,
    "device":               DEVICE,
    "python_version":       platform.python_version(),
    "torch_version":        torch.__version__,
    "numpy_version":        np.__version__,
    "pandas_version":       pd.__version__,
    "anndata_version":      ad.__version__,
    "cuda_version":         torch.version.cuda,
    "gpu_name":  torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
}
# `architecture`, `input_adata`, `preprocess_config_path` are built before the
# metrics gate above (so the compute_final_metrics=False stub carries them too).

metrics = {
    # ── self-describing run identity ─────────────────────────────────────────
    "run_id_classic": run_id_classic,
    "dataset": dataset, "gene_set": gene_set, "radius": radius,
    "model_type": model_type,
    "model_kind": kind_of(model_type),
    "target_source": spec.target_source,
    "uses_lam": spec.uses_lam,
    "is_probabilistic": spec.is_probabilistic,
    "lam": lam if spec.uses_lam else None,
    # What lam MEANS for this model: the beta-NLL exponent for the Gaussians,
    # unused otherwise.
    "lambda_meaning": ("gaussian_nll_variance_weight" if spec.uses_lam else None),
    "loss_function": spec.loss_display_name,
    # Count convention "v2_no_offset": no size factor / offset / per-spot exposure
    # anywhere; each count model fits its likelihood directly on its own target, and
    # depth_scale (float for the cp families, None for the depth-blind _raw ones) is
    # used ONLY as the divisor before log1p when reporting the mean correlation.
    "count_convention": "v2_no_offset",
    "depth_scale": DEPTH_SCALE,
    "count_common_depth": COUNT_COMMON_DEPTH,
    # Built once above (one explicit arm per target_source, no fall-through), so the
    # STUB and this full block always carry the identical string.
    "metric_scale": METRIC_SCALE,
    # MAE/MSE (mae_median/mse_median + per-gene + atoms) ride the SAME values as the
    # per-gene Pearson — log1p(CP10k) for X-target, g* vs log1p(y/depth_scale) for count — so
    # they are the magnitude complement to the scale-free Pearson, median over genes.
    # WITHIN-FAMILY ONLY: the count log1p comes via the g* projection, so absolute
    # error carries a native-loss asymmetry the Gaussian lacks — do NOT build a
    # cross-family MAE table; cross-family comparison stays Pearson (mean) + PIT.
    "error_metric_scale": "MAE/MSE on the per-gene Pearson scale (log1p-CP10k; g* "
                          "for count); median over genes; WITHIN-FAMILY only",
    "corr_metrics": list(CORR_METRICS),
    "primary_corr_metric": primary,
    "gene_names": gene_names, "n_genes": n_genes,
    "cv_strategy": CV_STRATEGY, "cv_params": CV_PARAMS,
    "lopo": LOPO, "n_donors": K, "n_folds": int(len(folds)),
    # CV/test split by DONOR (leakage-safe); metric still aggregated per SLIDE.
    "split_grouping": "donor", "metric_grouping": "slide",
    "donors_ordered": donors,
    # Held-out test donor(s) per fold. LOPO: each fold a different donor (the
    # per-donor test scalars are the lopo_stats points). fixed_holdout_kfold: the
    # fixed set, repeated every fold.
    "test_donors_per_fold": [f["test"] for f in folds],
    "test_donors": (None if LOPO else (folds[0]["test"] if folds else [])),
    "seed": SEED, "center": CENTER,
    # Per-gene target standardization (X-target models only): the model trains and
    # is scored in the standardized (z) space; per-(slide,gene) metrics are
    # affine-invariant, so this headline stays comparable to the X scale. The saved
    # predictions are z-scale (scale-free variance). "none" for count models / when off.
    "target_transform": TARGET_TRANSFORM if DO_TARGET_TRANSFORM else "none",
    # ── full hyperparameter snapshot (these are NOT in the output path; bumping
    #    run_id_classic is the way to re-run when these change) ──────────────
    "learning_rate": LR, "weight_decay": WEIGHT_DECAY,
    "batch_size": BATCH_SIZE,
    "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
    "early_stop_criterion": EARLY_STOP_CRITERION,
    "warmup_epochs": WARMUP_EPOCHS,
    "grad_clip_norm": GRAD_CLIP_NORM,
    "nb_log_alpha_init": NB_LOG_ALPHA_INIT,
    "matmul_tf32": MATMUL_TF32,
    "log_test_curve": LOG_TEST_CURVE,
    "save_learning_curves": SAVE_LEARNING_CURVES,
    "train_metric_spots_per_slide": TRAIN_METRIC_SPOTS,
    # ── provenance: architecture, input data, run environment ────────────────
    "architecture": architecture,
    "input_adata": input_adata,
    "preprocess_config": preprocess_config_path if os.path.exists(preprocess_config_path) else None,
    "environment": environment,
    # ── headline aggregates ──────────────────────────────────────────────────
    **corr_metric_aggs,
    # LOPO statistics (per-donor points + mean/SD/SEM + t-CI/t-PI), per corr
    # metric. None for fixed_holdout_kfold (the k-fold mean+/-std above is the
    # headline there). The HEADLINE under LOPO is lopo_stats[primary]["mean"].
    "lopo_stats": lopo_stats,
    "val_mae_median_mean":   mean_std("val_mae_median")[0],
    "test_mae_median_mean":  mean_std("test_mae_median")[0],
    "val_mse_median_mean":   mean_std("val_mse_median")[0],
    "test_mse_median_mean":  mean_std("test_mse_median")[0],
    "val_ece_pit_mean":      mean_std("val_ece_pit")[0],
    "test_ece_pit_mean":     mean_std("test_ece_pit")[0],
    "val_disp_signed_mean":  mean_std("val_disp_signed")[0],
    "test_disp_signed_mean": mean_std("test_disp_signed")[0],
    "val_pit_bias_mean":     mean_std("val_pit_bias")[0],
    "test_pit_bias_mean":    mean_std("test_pit_bias")[0],
    "val_loss_mean": mean_std("val_loss")[0],
    "test_loss_mean": mean_std("test_loss")[0],
    "per_fold": fold_rows,
    "nan_summary": nan_summary,
    "low_spot_slides": low,
    "per_slide_spot_counts": {str(s): int(c) for s, c in zip(slides, counts)},
}
with open(out_metrics, "w") as f:
    json.dump(metrics, f, indent=2)

# Snakemake declares an out-model output path: use fold 0's model there.
import shutil
shutil.copy(os.path.join(out_dir, "model_fold0.pt"), out_model)

print(f"\nDONE. val_{primary}={val_p:.4f}±{val_p_sd:.4f}  "
      f"test_{primary}={test_p:.4f}±{test_p_sd:.4f}")
if lopo_stats is not None:
    s = lopo_stats[primary]
    print(f"LOPO {primary}: mean={s['mean']:.4f} ± SD {s['sd']:.4f} (K={s['K']}); "
          f"95% CI {s['ci95_mean'][0]:.4f}..{s['ci95_mean'][1]:.4f}; "
          f"per-donor={ {d: round(v, 3) for d, v in s['per_donor'].items() if v is not None} }")
lc_note = "learning_curves/" if SAVE_LEARNING_CURVES else "(learning_curves skipped)"
print(f"Wrote {out_metrics}, atoms.parquet, splits.json, "
      f"{lc_note}, predictions/preds_fold*.h5ad")
