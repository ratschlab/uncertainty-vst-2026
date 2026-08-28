#!/usr/bin/env python
# snake/scripts/train_gcm.py
#
# GCM (GeneConditionedMLP) training entrypoint, the paired (spot, gene) counterpart of
# snake/scripts/train.py. It runs as a plain `python` process on one GPU, with no
# torchrun: the data lives in VRAM as a PairedSharedTensors, batches are built by an
# on-device gather, and each fold is trained by src/training/train_loop_gcm.py. A gene
# embedding conditions the readout, so the model can be scored on genes it never trained
# on. The output layout matches the classic pipeline, plus the gene-axis additions below.
#
# Reads the merged preprocessed adata.h5ad, the config file named by --config, the gene
# embedding CSV named by gcm.gene_embedding_sources, the gene-to-chromosome map, and
# snake/config/donor_slide_map.yaml. Writes into --output-dir: metrics.json,
# atoms.parquet, splits.json, genes.txt, learning_curves/fold{k}.csv,
# predictions/preds_fold{k}.h5ad, models/model_fold{k}.pt, model.pt (a copy of fold 0)
# and gene_embeddings_used.npz.
#
# Metrics are slide-aware: per-(slide, gene) Pearson, then averaged over slides.
#
# ── Two independent CV axes (see src/data/splitting.py) ───────────────────────
#   • Spot axis: donor-level outer CV from the shared top-level cv: block
#     (lopo_next_val by default). One model per held-out test donor.
#   • Gene axis: one fixed split (gene_single_split). Test = the held-out
#     chromosome(s); val = a seeded random fraction (or chromosome(s)) of the train
#     genes; train = the rest. The same gene split is reused for every donor fold, so
#     the cross-gene headline stays comparable across donors. Each epoch scores val
#     spots on gene-train genes (quadrant ②) and gene-val genes (quadrant ④); early
#     stopping uses gcm.early_stop_criterion (blend|seen_genes|new_genes) and both
#     quadrants are always logged. Headline = test donor × test-heldout genes,
#     aggregated across donors as LOPO stats (mean ± inter-donor SD/CI/PI).
#
# ── Count convention "v2_no_offset" (same as the classic pipeline) ────────────
#   No size factor, no offset, no per-spot exposure. Every count model fits its
#   likelihood directly on its own target (round(CP_T) for _cp, continuous CP_T for
#   _cp_nr, raw counts for _raw) and saves mu and alpha on that same scale.
#   depth_scale (count_common_depth/1e4 for the cp families, None for raw) is the one
#   count constant, and it divides only before the log1p of the reported mean
#   correlation.
#
# ── Run (the form Snakemake issues) ───────────────────────────────────────────
#   python -u snake/scripts/train_gcm.py \
#       --config snake/config/config.yaml --dataset 10x_TuPro \
#       --dev-gene-set-id <key under gcm.dev_gene_sets[dataset]> \
#       --gene-emb-id network512 --chrom-split-spec chr1_3_5_7_9 \
#       --use-projection 1 --permute-embeddings 0 \
#       --radius 0.75 --model-type heteroscedastic --lam 1.0 \
#       --preprocessed-dir results/10x_TuPro/_preprocessed/<preproc_id>/<radius_tag> \
#       --output-dir results_gcm/10x_TuPro/<run_id>/<radius_tag>/heteroscedastic/lambda_1.0

import argparse
import json
import os
import platform
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

# Wall-clock start of the whole run (recorded in metrics.json).
RUN_START  = time.time()
STARTED_AT = datetime.now().astimezone().isoformat()

# ── Make `src.*` importable when invoked from the project root ───────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import anndata as ad

from src.data.load_and_preprocess import read_arrays
from src.training.target_transform import compute_gene_target_stats
from src.data.loading import load_gene_embeddings, permute_gene_embeddings
from src.data.dataset import PairedSharedTensors, make_gene_emb_matrix
from src.data.splitting import (
    gene_single_split,
    load_slide_to_donor,
    donor_labels as donor_labels_from_slides,
    load_donor_order, cv_donor_folds, donor_ids_to_spot_idx, is_lopo_strategy,
    load_gene_chromosome_map,
    _chrom_sort_key,
)
from src.models.specs import get_spec, kind_of, MODEL_KINDS, COUNT_ROUTED_TARGETS
from src.models.mlp import _DEFAULT_HIDDEN, _DEFAULT_DROPOUT, _DEFAULT_ACT
from src.training.train_loop_gcm import train_one_fold_gcm, run_paired_inference


def log(msg):
    print(f"[train_gcm] {msg}", flush=True)


def resolve_amp_dtype(gcm_cfg):
    """Autocast dtype picked from the GPU this job actually landed on (SLURM may
    schedule a V100 one run and an Ampere card the next). bf16 needs Ampere+
    (compute capability >= 8.0); Volta/Turing (7.x) accelerate fp16 only; older
    cards stay in fp32. `gcm.amp` overrides the auto pick: auto | bf16 | fp16 | off
    (default off). Returns (mode, torch dtype or None, (major, minor) capability)."""
    mode = str(gcm_cfg.get("amp", "off")).lower()
    cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
    if mode == "auto":
        dtype = torch.bfloat16 if cap[0] >= 8 else (torch.float16 if cap[0] == 7 else None)
    elif mode == "bf16":
        dtype = torch.bfloat16
    elif mode == "fp16":
        dtype = torch.float16
    else:
        dtype = None
    return mode, dtype, cap


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Single-GPU GeneConditionedMLP training")
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--gene-emb-id', required=True,
                   help="Key into gcm.gene_embedding_sources (e.g. 'network512').")
    p.add_argument('--chrom-split-spec', required=True,
                   help="Key into gcm.chrom_split_specs (e.g. 'greedy30pctSF').")
    p.add_argument('--use-projection', type=int, choices=[0, 1], required=True)
    p.add_argument('--permute-embeddings', type=int, choices=[0, 1], required=True)
    p.add_argument('--radius', required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--dev-gene-set-id', required=True,
                   help="Key into gcm.dev_gene_sets[dataset] (the dev gene list for this run).")
    # Choices come from the registry (single source of truth) so a newly added
    # model_type never has to be re-listed here. spec-level routing decides what
    # the GCM actually supports; get_spec() raises a clear error for unknowns.
    p.add_argument('--model-type', required=True, choices=sorted(MODEL_KINDS))
    p.add_argument('--lam', type=float, required=True)
    p.add_argument('--preprocessed-dir', required=True, type=Path,
                   help="Directory holding adata.h5ad from the preprocess rule.")
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--seed', type=int, default=None,
                   help="Deep-ensemble MEMBER seed: overrides cfg['seed']. Drives "
                        "ONLY init + batch shuffle (per-fold seed+fold in the loop); "
                        "the donor spot holdout and the slide/chromosome GroupKFolds "
                        "take no RNG, so every member shares the identical splits. "
                        "Default None -> use cfg['seed'] (unchanged behavior).")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Resolve helpers
# ─────────────────────────────────────────────────────────────────────────────

def resolve_dev_genes(gcm_cfg, dataset, dev_gene_set_id, preproc_id):
    """Resolve the PER-DATASET dev gene set: gcm.dev_gene_sets[dataset][dev_gene_set_id].

    The value is either an inline gene list or a path to a one-gene-per-line file
    (the path may use `{preproc_id}`, and a relative path resolves under
    project_root). Returns a gene-name list. (Different datasets may use different
    gene lists, and a dataset may define several — each is its own run.)"""
    dev_sets = gcm_cfg.get('dev_gene_sets', {})
    ds_sets = dev_sets.get(dataset)
    if not ds_sets or dev_gene_set_id not in ds_sets:
        raise KeyError(
            f"dev_gene_set_id '{dev_gene_set_id}' not defined for dataset '{dataset}' "
            f"under gcm.dev_gene_sets. Available for this dataset: "
            f"{sorted(ds_sets) if ds_sets else '[]'}"
        )
    spec = ds_sets[dev_gene_set_id]
    if isinstance(spec, (list, tuple)):
        return [str(g).strip() for g in spec if str(g).strip()]
    gene_file = Path(str(spec).format(preproc_id=preproc_id))
    if not gene_file.is_absolute():
        gene_file = PROJECT_ROOT / gene_file
    with open(gene_file) as f:
        return [g.strip() for g in f if g.strip()]


def resolve_gene_embeddings(gene_emb_id, gene_list, permute, gcm_cfg):
    """Load gene embeddings for the requested source id; optionally permute.
    Returns (gene_embs dict, gene_dim, source_path, perm_info or None)."""
    sources = gcm_cfg['gene_embedding_sources']
    if gene_emb_id not in sources:
        raise KeyError(
            f"gene_emb_id '{gene_emb_id}' not in gcm.gene_embedding_sources. "
            f"Available: {list(sources.keys())}"
        )
    source_path = Path(sources[gene_emb_id])
    if not source_path.is_absolute():
        source_path = PROJECT_ROOT / source_path

    gene_embs, gene_dim = load_gene_embeddings(source_path, gene_list)

    perm_info = None
    if permute:
        perm_seed = int(gcm_cfg.get('permutation_seed', 42))
        gene_embs, perm_info = permute_gene_embeddings(gene_embs, seed=perm_seed)
        log(f"Gene embeddings permuted with seed={perm_seed}")

    return gene_embs, gene_dim, str(source_path), perm_info


def count_provenance(cfg, dataset, model_type, spec):
    """The count-model provenance triple (depth_scale, count_common_depth, metric_scale).

    Resolved from cfg and spec alone, so main() and write_metrics_json can call the same
    code. write_metrics_json is a separate top-level function and does not share main()'s
    locals, so deriving the triple in both places would let the two copies drift apart in
    one metrics.json.

    depth_scale is the one count constant: count_common_depth/1e4 for the cp families,
    and None (not 1.0) for the depth-blind raw-count models, which are aligned to no
    depth at all. It has exactly one role, the divisor applied before log1p when the mean
    correlation is reported (g*, the per-epoch option-A metric, the *_logmean
    cross-check). There is no offset and no per-spot size factor.

    metric_scale gets one explicit arm per legal target_source, with no fall-through
    default. The three count reporting axes:
        _cp, _cp_nr -> log1p(CP10k)      comparable to the Gaussian models
        _raw        -> log1p(raw counts) comparable to neither the Gaussian nor _cp
    """
    common_depth = cfg["datasets"][dataset].get("count_common_depth")
    depth_scale = None
    if spec.target_source in ("cp_counts", "cp_counts_nr"):
        if common_depth is None:
            raise KeyError(
                f"model_type '{model_type}' uses target_source='{spec.target_source}' but "
                f"config datasets.{dataset} has no 'count_common_depth'."
            )
        depth_scale = float(common_depth) / 1e4

    do_transform = (cfg.get("target_transform", "none") != "none"
                    and spec.target_source == "X")
    if spec.target_source == "cp_counts":
        metric_scale = (
            f"common-depth count (T={common_depth}, depth_scale={depth_scale}); the "
            f"NB/Poisson is fitted DIRECTLY on the target round(CP_T) -- no offset, no "
            f"size factor, no per-spot exposure -- so the saved X(mu) and the dispersion "
            f"alpha are BOTH on that CP_T scale. Reported corr = "
            f"g*=E[log1p(Y/depth_scale)] vs log1p(round(CP_T)/depth_scale) = log1p(CP10k), "
            f"comparable to the Gaussian models. BOTH log1p estimators are reported, as "
            f"scalars AND per (slide,gene) in atoms.parquet: {{metric}} = g* (full fitted "
            f"predictive, uses mu AND the dispersion alpha) and {{metric}}_logmean = "
            f"log1p(mu/depth_scale) (MEAN-ONLY, ignores alpha, so it sits above g* by the "
            f"Jensen gap and is the scale the per-epoch / early-stop metric uses). "
            f"Calibration (native-CDF one-sided PIT) and "
            f"var_residual_corr[_partial] stay on the CP_T scale with no constant applied. "
            f"pred_var NOT persisted (= count_pred_var(X[,dispersion]))")
    elif spec.target_source == "cp_counts_nr":
        metric_scale = (
            f"common-depth count NO-ROUNDING (T={common_depth}, depth_scale={depth_scale}); "
            f"target = CONTINUOUS CP_T = CP10k*depth_scale (no np.rint), fitted DIRECTLY by "
            f"the discrete Poisson/NB via the continuous NLL (NB validate_args=False) -- no "
            f"offset, no size factor. X(mu) and alpha are BOTH on that CP_T scale. Reported "
            f"corr = g*=E[log1p(Y/depth_scale)] vs log1p(CP_T/depth_scale) = log1p(CP10k) "
            f"EXACT (Gaussian-comparable, un-quantized). BOTH log1p estimators are "
            f"reported, as scalars AND per (slide,gene) in atoms.parquet: {{metric}} = g* "
            f"(full fitted predictive, uses mu AND alpha) and {{metric}}_logmean = "
            f"log1p(mu/depth_scale) (MEAN-ONLY, ignores alpha, above g* by the Jensen gap; "
            f"the early-stop scale). "
            f"Calibration = native-CDF one-sided PIT with the SMEARED CDF F~, on the CP_T "
            f"scale. pred_var NOT persisted (= count_pred_var(X[,dispersion]))")
    elif spec.target_source == "raw_counts":
        metric_scale = (
            "depth-blind raw counts (depth_scale=None -> divisor 1.0); the NB/Poisson is "
            "fitted DIRECTLY on adata.layers['raw_counts'] -- no offset, no size factor, no "
            "per-spot exposure -- so X(mu) and alpha are BOTH on the raw-count scale. "
            "Reported corr = g*=E[log1p(Y)] vs log1p(raw_counts), i.e. the HEADLINE IS ON "
            "log1p(RAW COUNTS): it is NOT comparable to the Gaussian models and NOT "
            "comparable to the _cp / _cp_nr families, only to other _raw runs. BOTH log1p "
            "estimators are reported, as scalars AND per (slide,gene) in atoms.parquet: "
            "{metric} = g* (full fitted predictive, uses mu AND alpha) and "
            "{metric}_logmean = log1p(mu) (MEAN-ONLY, ignores alpha, above g* by the "
            "Jensen gap; the early-stop scale). "
            "Calibration and var_residual_corr[_partial] are on "
            "the raw-count scale. pred_var NOT persisted (= count_pred_var(X[,dispersion]))")
    elif spec.target_source == "X":
        metric_scale = (
            f"per-gene standardized log-CP10k ({cfg.get('target_transform')}, per-gene/"
            f"per-fold; per-(slide,gene) metrics affine-invariant -> comparable to X)"
            if do_transform else "log1p(CP10k) = adata.X (Gaussian/MSE)")
    else:
        metric_scale = f"UNDECLARED metric scale for target_source={spec.target_source!r}"
    return depth_scale, common_depth, metric_scale


def resolve_gene_split(dev_genes, gene_chrom_map, gcm_cfg, chrom_spec_name, default_seed):
    """Dispatch gene_single_split for the named spec. The spec carries both the test-gene
    selection (heldout_chroms or greedy) and a `val:` sub-block, either a seeded random
    fraction of the train genes or whole chromosomes. Returns (train_genes, val_genes,
    test_heldout_genes, info): one fixed gene split reused across every donor fold.
    default_seed is the config-level seed used for the random val draw unless
    spec.val.seed overrides it, which keeps the split frozen across --seed members."""
    specs = gcm_cfg.get('chrom_split_specs', {})
    if chrom_spec_name not in specs:
        raise KeyError(
            f"chrom_split_spec '{chrom_spec_name}' not in gcm.chrom_split_specs. "
            f"Available: {list(specs.keys())}"
        )
    return gene_single_split(dev_genes, gene_chrom_map, specs[chrom_spec_name],
                             default_seed=default_seed)


# ─────────────────────────────────────────────────────────────────────────────
# Small numeric helpers
# ─────────────────────────────────────────────────────────────────────────────

def _nm(vals):
    """Mean over finite values only; NaN if none are finite (matches metrics file)."""
    finite = [v for v in np.asarray(vals, dtype=float).ravel() if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def _sd(vals):
    # ddof=1: the folds are a sample, and complete_deferred_metrics.py writes the same
    # metrics.json keys the same way.
    finite = [v for v in np.asarray(vals, dtype=float).ravel() if np.isfinite(v)]
    return float(np.std(finite, ddof=1)) if len(finite) >= 2 else float("nan")


def _lopo_block(per_donor):
    """LOPO statistics from a per-donor Series of scores x_i: mean, inter-donor SD
    (ddof=1), SEM, Student-t 95% CI on the mean and a prediction interval for a new
    donor (df=K-1). Keeps the K per-donor points. (Intervals are optimistic — folds
    share training donors; Bengio & Grandvalet 2004.)"""
    from scipy import stats as _st
    x = per_donor.to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    k = int(len(x))
    mean = float(np.mean(x)) if k else float("nan")
    sd   = float(np.std(x, ddof=1)) if k >= 2 else float("nan")
    sem  = float(sd / np.sqrt(k)) if k >= 2 else float("nan")
    if k >= 2:
        tstar = float(_st.t.ppf(0.975, k - 1))
        half, half_pi = tstar * sem, tstar * sd * float(np.sqrt(1.0 + 1.0 / k))
        ci, pi = [mean - half, mean + half], [mean - half_pi, mean + half_pi]
    else:
        tstar, ci, pi = float("nan"), [float("nan")] * 2, [float("nan")] * 2
    return {"mean": mean, "sd": sd, "sem": sem, "K": k, "df": k - 1,
            "t_star_0975": tstar, "ci95_mean": ci, "pi95_new_donor": pi,
            "per_donor": {str(d): (None if not np.isfinite(v) else float(v))
                          for d, v in per_donor.items()}}


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return None if np.isnan(v) else v
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and np.isnan(o):
        return None
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


# ─────────────────────────────────────────────────────────────────────────────
# Per-fold artifact builders (matrices — same layout as the classic pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def gene_role_for_fold(all_genes, gene_train_set, gene_val_set, test_heldout_set):
    """Per-gene role string for this fold, aligned to all_genes order.
    'test_heldout' (pristine, this run), 'seen' (gene-train this fold), 'new'
    (gene-val this fold). Stored in the predictions var + atoms."""
    roles = []
    for g in all_genes:
        if g in test_heldout_set:
            roles.append("test_heldout")
        elif g in gene_train_set:
            roles.append("seen")
        elif g in gene_val_set:
            roles.append("new")
        else:
            roles.append("unknown")
    return np.array(roles)


def fold_predictions_adata(results_by_split, row_idx_by_split, fold,
                           all_genes, gene_role, obs_names, spec, depth_scale=None,
                           target_loc=None, target_scale=None):
    """One AnnData holding this fold's predictions: test spots only, scored on all genes.
    Val spots drive training-time selection and already reach metrics.json and
    atoms.parquet from memory, so they are not written here.

        X                  = pred (means)            (n_spots x n_genes)
        layers["pred_var"] = predictive variance     (Gaussian specs only; a count spec
                                                       omits it, being a deterministic
                                                       function of X and the dispersion
                                                       layer, recomputed offline via
                                                       count_pred_var)
        layers[<extra>]    = one per spec.extra_prediction_columns
        obs                = slide, obs_name, adata_row, split, fold
        var_names          = all_genes ; var["gene_role"] = seen|new|test_heldout
    Ground truth is recovered offline via obs_name/adata_row joined against the shared
    preprocessed adata.h5ad, or, under a target transform, via the saved
    var["target_loc"] and var["target_scale"].
    `results_by_split` and `row_idx_by_split` share key order."""
    # Count specs never persist pred_var (same rule as the classic pipeline): it is a
    # deterministic function of the saved X (means) and, for NB, the "dispersion"
    # layer (Poisson var = mu; NB var = mu + alpha*mu^2), so it is recomputed offline
    # via count_metrics.count_pred_var. Gaussian pred_var is a separate variance HEAD
    # (not recomputable), so it IS saved.
    is_count_spec = spec.target_source in COUNT_ROUTED_TARGETS
    means, pred_vars = [], []
    slides, splits, rows = [], [], []
    extras = {col: [] for col in spec.extra_prediction_columns}
    has_pred_var = False
    n_genes = len(all_genes)
    for split, res in results_by_split.items():
        m2 = np.asarray(res["means_2d"])
        n = m2.shape[0]
        means.append(m2)
        slides.append(np.asarray(res["groups"]).astype(str))
        splits.append(np.repeat(split, n))
        rows.append(np.asarray(row_idx_by_split[split]))
        if (not is_count_spec) and res.get("pred_var_2d") is not None:
            has_pred_var = True
            pred_vars.append(np.asarray(res["pred_var_2d"]))
        for col in spec.extra_prediction_columns:
            arr = res.get(col)
            extras[col].append(arr if arr is not None
                               else np.full((n, n_genes), np.nan, dtype=np.float32))

    row_idx_all = np.concatenate(rows)
    obs = pd.DataFrame({
        "slide":     np.concatenate(slides),
        "obs_name":  obs_names[row_idx_all],
        "adata_row": row_idx_all,
        "split":     np.concatenate(splits),
        "fold":      np.repeat(fold, len(row_idx_all)),
    }, index=obs_names[row_idx_all])

    pred = ad.AnnData(X=np.concatenate(means).astype(np.float32), obs=obs)
    pred.var_names = list(all_genes)
    pred.var["gene_role"] = np.asarray(gene_role)
    # Count convention marker: "v2_no_offset" means X (mu) and the dispersion layer
    # live on the model's OWN target scale, with NO per-spot size factor anywhere
    # (there is no obs["size_factor"] in these files). depth_scale is the single
    # reporting constant; h5ad has no writer for a bare Python None in uns, so the
    # key is written only when it is a float and readers must use
    # `adata.uns.get("depth_scale")` (None for the depth-blind _raw models).
    pred.uns["count_convention"] = "v2_no_offset"
    if depth_scale is not None:
        pred.uns["depth_scale"] = float(depth_scale)
    if has_pred_var:
        pred.layers["pred_var"] = np.concatenate(pred_vars).astype(np.float32)
    for col, lst in extras.items():
        pred.layers[col] = np.concatenate(lst).astype(np.float32)
    # X-target transform: persist per-gene loc/scale so the log-CP10k scale is
    # recoverable offline (mu_y = scale*mu_z + loc); aligned to var_names (ALL_GENES).
    if target_loc is not None:
        pred.var["target_loc"]   = np.asarray(target_loc,   dtype=np.float32)
        pred.var["target_scale"] = np.asarray(target_scale, dtype=np.float32)
    return pred


def atoms_dataframe_gcm(result, fold, split, all_genes, gene_role, corr_metrics, spec):
    """Per-(slide, gene) metric atoms for one (fold, split): one row per
    (slide, gene), a column per configured corr metric (+ a `{metric}_logmean`
    column for the count specs, + ece/var_residual_corr for probabilistic specs),
    tagged with gene_role. Gene indices (into all_genes) are mapped to names.
    Matches the classic atoms.parquet schema + gene_role."""
    rows = {}

    def row_for(a):
        gi = a["gene"]
        return rows.setdefault((a["slide"], gi), {
            "fold": fold, "split": split, "slide": a["slide"],
            "gene": all_genes[gi], "gene_role": gene_role[gi],
            "n_spots": a.get("n_spots"),
        })

    for name in corr_metrics:
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
    if cal is not None:
        for a in cal:
            r = row_for(a)
            for k in ("ece_pit", "pit_bias", "disp_signed",
                      "var_residual_corr", "var_residual_corr_partial"):
                r[k] = a[k]
    return pd.DataFrame(list(rows.values()))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    use_projection     = bool(args.use_projection)
    permute_embeddings = bool(args.permute_embeddings)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Config ───────────────────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    gcm_cfg = cfg['gcm']

    # Optional TF32 / high-precision matmul (parity with classic train.py).
    if bool(cfg.get("matmul_tf32", False)) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    # Mixed precision: dtype auto-detected from the GPU this job landed on (logged
    # here + recorded in metrics.json). Autocast wraps the TRAIN forward only;
    # eval/inference stay fp32 so the saved metrics keep full precision.
    AMP_MODE, AMP_DTYPE, GPU_CAP = resolve_amp_dtype(gcm_cfg)
    log(f"AMP: mode={AMP_MODE} "
        f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'} "
        f"capability={GPU_CAP} -> autocast dtype={AMP_DTYPE}")

    # Top-level seed (NOT gcm.seed): the classic pipeline reads cfg["seed"], so
    # reading the same key here keeps both pipelines on one RNG seed => identical
    # spot holdout/CV partitions. gcm.permutation_seed stays separate (sanity check).
    SEED          = args.seed if args.seed is not None else int(cfg.get('seed', 42))
    # Outer (donor) CV from the shared top-level cv: block (registry in
    # src/data/splitting.py), the same block the classic pipeline reads, so both hold out
    # donors identically. lopo_next_val by default. The gene axis is a separate single
    # split (gene_single_split, below), independent of these spot folds.
    CV_CFG        = cfg.get('cv') or {}
    CV_STRATEGY   = CV_CFG.get('strategy', 'lopo_next_val')
    CV_PARAMS     = dict(CV_CFG.get('params') or {})
    # datasets.{ds}.test_donors — read ONLY by fixed_holdout_kfold (IGNORED under
    # lopo_next_val, where every donor is held out in turn).
    TEST_DONORS   = cfg['datasets'][args.dataset].get('test_donors')
    CORR_METRICS  = tuple(cfg.get('corr_metrics', ('pearson',)))
    PRIMARY       = CORR_METRICS[0]
    EARLY_STOP    = gcm_cfg.get('early_stop_criterion', 'blend')
    CENTER        = bool(gcm_cfg.get('gcm_centering_from_train', True))
    TRAIN_METRIC_SPOTS = gcm_cfg.get('train_metric_spots_per_slide',
                                     cfg.get('train_metric_spots_per_slide', 1000))
    SAVE_LC       = bool(cfg.get('save_learning_curves', True))
    NB_PER_EPOCH_CALIB = bool(cfg.get('nb_per_epoch_calibration', True))
    # Fold-end gating: COMPUTE_FINAL_METRICS = scipy/g* + atoms + metrics.json + lopo;
    # SAVE_PREDICTIONS = the big preds_fold{k}.h5ad writes. Both default True (full run).
    COMPUTE_FINAL_METRICS = bool(cfg.get('compute_final_metrics', True))
    SAVE_PREDICTIONS      = bool(cfg.get('save_predictions', True))

    BASE_LR       = float(gcm_cfg['gcm_lr'])
    WEIGHT_DECAY  = float(gcm_cfg.get('gcm_weight_decay', 1e-4))
    # BATCH_SIZE/LR for the batched model are set once the spot batch is read below
    # (the batch axis is SPOTS — each step scores all genes, no per-pair batch).

    MAX_EPOCHS    = int(gcm_cfg['gcm_max_epochs'])
    PATIENCE      = int(gcm_cfg['gcm_patience'])
    # GCM-only stability knobs (NOT inherited from the classic top-level, so the
    # two pipelines stay decoupled). Off by default; add gcm.warmup_epochs /
    # gcm.grad_clip_norm to enable when scaling the batch.
    WARMUP_EPOCHS = int(gcm_cfg.get('warmup_epochs', 0))
    GRAD_CLIP     = gcm_cfg.get('grad_clip_norm', None)
    # Gene hypernetwork (batched GeneHyperMLP). `use_projection` selects the readout
    # shape here: 1 gives the hypernetwork the hidden layer(s) from gcm.hyper_hidden,
    # 0 gives a bilinear readout (a single Linear). interaction_dim None means the spot
    # backbone's last hidden width.
    HYPER_HIDDEN    = list(gcm_cfg.get('hyper_hidden', [256])) if use_projection else []
    HYPER_DROPOUT   = float(gcm_cfg.get('hyper_dropout', 0.0))
    INTERACTION_DIM = gcm_cfg.get('interaction_dim', None)
    # Base architecture: GCM block if given, else the classic `architecture` block.
    ARCH          = gcm_cfg.get('architecture') or cfg.get('architecture') or None

    LOPO = is_lopo_strategy(CV_STRATEGY)

    spec = get_spec(args.model_type)
    # Fail fast if this model has no GCM head, e.g. the STimage heads, whose bare
    # per-gene readout is undefined for a zero-shot gene set.
    if spec.gcm_head is None:
        raise SystemExit(
            f"model_type '{args.model_type}' is not supported in the GCM pipeline "
            f"(its ModelSpec has gcm_head=None). Supported GCM models declare a "
            f"GcmHead in src/models/specs.py."
        )
    is_gaussian = spec.is_probabilistic
    log(f"device={DEVICE} | run: model_type={args.model_type} (kind={kind_of(args.model_type)}, "
        f"target={spec.target_source}, loss={spec.loss_display_name}, "
        f"uses_lam={spec.uses_lam}, probabilistic={spec.is_probabilistic}) lam={args.lam}")
    log(f"gene_emb_id={args.gene_emb_id}, chrom_split={args.chrom_split_spec}, "
        f"use_projection={use_projection}, permute={permute_embeddings}")
    log(f"early_stop_criterion={EARLY_STOP} | corr_metrics={CORR_METRICS} | "
        f"cv={CV_STRATEGY} | center={CENTER}")

    # ── Count-model knobs (Poisson / NB; no-ops for X-target models) ─────────
    # nb_log_alpha_init seeds the GCM NB head's dispersion near Poisson. The rest of the
    # count provenance (depth_scale, count_common_depth, metric_scale) comes from
    # count_provenance, the same helper write_metrics_json calls, so the two places that
    # build provenance cannot disagree. depth_scale is the one count constant, T/1e4 for
    # the cp families and None for the depth-blind raw models: the divisor before log1p
    # when the mean correlation is reported, never an offset or a per-spot exposure.
    NB_LOG_ALPHA_INIT  = float(cfg.get("nb_log_alpha_init", -2.0))
    DEPTH_SCALE, COUNT_COMMON_DEPTH, METRIC_SCALE = count_provenance(
        cfg, args.dataset, args.model_type, spec)
    log(f"count convention: v2_no_offset | depth_scale={DEPTH_SCALE} "
        f"(count_common_depth={COUNT_COMMON_DEPTH})")

    # Per-gene target standardization (X-target Gaussian/MSE only) — SAME knob the
    # classic pipeline reads, so when it is active both pipelines train on the same
    # transformed target and a classic-vs-GCM comparison stays fair. Count models
    # (raw_counts) are left untouched.
    TARGET_TRANSFORM    = cfg.get("target_transform", "none")
    DO_TARGET_TRANSFORM = (TARGET_TRANSFORM != "none" and spec.target_source == "X")

    # ── Dev genes (per dataset: gcm.dev_gene_sets[dataset][dev_gene_set_id]) ──
    DEV_GENES_FILE = resolve_dev_genes(gcm_cfg, args.dataset, args.dev_gene_set_id,
                                       cfg.get('preproc_id', 'v1'))
    log(f"dev_gene_set_id={args.dev_gene_set_id} for dataset={args.dataset} "
        f"({len(DEV_GENES_FILE)} genes)")

    # Batched gene-conditioned model: a STEP is `spot_batch_size` spots × ALL gene-train
    # genes (predicted in one matmul), so the batch axis is SPOTS, not pairs. The
    # effective batch (spots × genes) is already large, so lr is used as-is
    # (gcm_lr) — retune gcm_lr / gcm.warmup_epochs rather than auto-scaling.
    BATCH_SIZE = int(gcm_cfg.get('spot_batch_size', 64))
    LR = BASE_LR
    log(f"spot_batch_size={BATCH_SIZE} spots/step (each step scores all gene-train "
        f"genes) | hyper_hidden={HYPER_HIDDEN} interaction_dim={INTERACTION_DIM} "
        f"-> lr={LR:.3e}")

    # ── Load preprocessed adata (spec decides X vs raw_counts) ───────────────
    preprocessed_dir = args.preprocessed_dir
    if not preprocessed_dir.is_absolute():
        preprocessed_dir = PROJECT_ROOT / preprocessed_dir
    adata_path = preprocessed_dir / "adata.h5ad"
    log(f"Input adata (preprocessed): {adata_path}")
    adata = ad.read_h5ad(adata_path)
    embeddings, y_full, sample_labels, gene_names_all = read_arrays(
        adata, target_source=spec.target_source, depth_scale=DEPTH_SCALE,
    )
    obs_names = adata.obs_names.to_numpy()

    # Dev genes actually present in the panel.
    gene_to_idx_full = {g: i for i, g in enumerate(gene_names_all)}
    DEV_GENES_PRESENT = [g for g in DEV_GENES_FILE if g in gene_to_idx_full]
    missing = [g for g in DEV_GENES_FILE if g not in gene_to_idx_full]
    if missing:
        log(f"WARNING: dropping {len(missing)} dev genes absent from the panel: "
            f"{missing[:10]}{'...' if len(missing) > 10 else ''}")

    # ── Gene split: one fixed test/val/train split, reused across donor folds ──
    # gene_single_split gives test = the held-out chromosome(s), val = a seeded random
    # fraction (or chromosome(s)) of the train genes, train = the rest. The same
    # partition is used for every donor fold, so the cross-gene headline (test donor x
    # test-heldout genes) stays comparable across donors.
    map_path = Path(gcm_cfg['gene_chrom_map_path'])
    if not map_path.is_absolute():
        map_path = PROJECT_ROOT / map_path
    gene_chrom_map = load_gene_chromosome_map(map_path)
    # The val gene draw uses the config seed, not SEED, which is the per-member ensemble
    # --seed. The gene split is computed once here, so it is identical across donor folds,
    # and it must also be identical across deep-ensemble members. Seeding it with SEED
    # would give each member a different train/val gene split, so members would differ by
    # more than initialization and shuffle order.
    GENE_TRAIN, GENE_VAL, TEST_HELDOUT_GENES, gene_split_info = resolve_gene_split(
        DEV_GENES_PRESENT, gene_chrom_map, gcm_cfg, args.chrom_split_spec,
        int(cfg.get('seed', 42))
    )
    CV_GENES  = GENE_TRAIN + GENE_VAL                      # genes seen in CV (train+val)
    ALL_GENES = GENE_TRAIN + GENE_VAL + TEST_HELDOUT_GENES  # column order for y / gene_emb

    # Slice y to ALL_GENES (frees the full panel matrix immediately).
    all_cols_in_full = [gene_to_idx_full[g] for g in ALL_GENES]
    y = y_full[:, all_cols_in_full]
    del y_full
    gene_to_col = {g: i for i, g in enumerate(ALL_GENES)}
    # Fixed gene columns (same for every donor fold).
    GENE_TRAIN_COLS = [gene_to_col[g] for g in GENE_TRAIN]
    GENE_VAL_COLS   = [gene_to_col[g] for g in GENE_VAL]

    # ── Gene embeddings (for ALL genes; row order == y columns) ──────────────
    gene_embs, gene_dim, gene_emb_source, perm_info = resolve_gene_embeddings(
        args.gene_emb_id, ALL_GENES, permute_embeddings, gcm_cfg
    )
    gene_emb_matrix = make_gene_emb_matrix(ALL_GENES, gene_embs)   # (n_genes, gene_dim)
    spot_dim = embeddings.shape[1]

    # ── Per-slide spot-count sanity flag (loud, at startup) ──────────────────
    slides, counts = np.unique(sample_labels, return_counts=True)
    log(f"Per-slide spot counts ({len(slides)} slides):")
    for s, c in zip(slides, counts):
        log(f"  {s:<16} {c:>6}{'  <-- LOW' if c < 50 else ''}")
    low_spot_slides = [str(s) for s, c in zip(slides, counts) if c < 50]

    # ── Chromosome repartition header (compact; no full gene-list dump) ──────
    log_chrom_repartition(ALL_GENES, gene_chrom_map, TEST_HELDOUT_GENES,
                          gene_split_info)

    # ── Build the GPU-resident paired data source ONCE ───────────────────────
    pst = PairedSharedTensors(embeddings, gene_emb_matrix, y, sample_labels,
                              device=DEVICE, depth_scale=DEPTH_SCALE)
    log(f"PairedSharedTensors: {pst.n_spots} spots x {pst.spot_dim} spot-dim, "
        f"{pst.n_genes_total} genes x {pst.gene_dim} gene-dim; device={DEVICE}")
    del embeddings, y                       # tensors own the data now (keep gene_emb_matrix for npz)

    # ── Donor-level OUTER CV (registry) — LOPO by default ────────────────────
    # Same rationale as the classic pipeline: slides of one donor are near-
    # duplicates, so the spot train/val/test split is by DONOR (the registry returns
    # donor-id partitions; we map them to spot rows in the loop). The METRIC stays
    # per-slide (run_paired_inference uses pst.groups = slide). The gene split above
    # is orthogonal and FIXED across these donor folds.
    donor_map_path = PROJECT_ROOT / "snake" / "config" / "donor_slide_map.yaml"
    slide_to_donor = load_slide_to_donor(donor_map_path, args.dataset)
    donor_per_spot = donor_labels_from_slides(sample_labels, slide_to_donor)

    present_donors = set(donor_per_spot.tolist())
    donor_order = load_donor_order(donor_map_path, args.dataset)
    if donor_order is None:
        donors = sorted(present_donors)
    else:
        donors = ([d for d in donor_order if d in present_donors]
                  + sorted(present_donors - set(donor_order)))
    folds = cv_donor_folds(donors, CV_STRATEGY, test_donors=TEST_DONORS, **CV_PARAMS)
    K = len(donors)
    log(f"CV strategy: {CV_STRATEGY} (lopo={LOPO}) over {K} donors: {donors}")
    log(f"  -> {len(folds)} outer fold(s); test rotation: {[f['test'] for f in folds]}")

    # ── Output dir + provenance file ─────────────────────────────────────────
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    preds_dir = output_dir / "predictions"; preds_dir.mkdir(exist_ok=True)
    models_dir = output_dir / "models";     models_dir.mkdir(exist_ok=True)
    lc_dir = output_dir / "learning_curves"
    if SAVE_LC:
        lc_dir.mkdir(exist_ok=True)
    # Full gene list -> sibling file (the log only shows the chromosome summary).
    with open(output_dir / "genes.txt", "w") as f:
        f.write("\n".join(ALL_GENES) + "\n")

    test_heldout_set = set(TEST_HELDOUT_GENES)
    all_gene_cols = torch.arange(len(ALL_GENES), dtype=torch.long)   # final eval = all genes
    rng = np.random.default_rng(SEED)
    # gene_role (seen/new/test_heldout) is FIXED across donor folds (single gene split).
    GENE_ROLE = gene_role_for_fold(ALL_GENES, set(GENE_TRAIN), set(GENE_VAL), test_heldout_set)

    fold_summaries = []
    atom_frames = []
    cv_folds_info = []
    n_params = None

    # ── Outer (donor) fold loop, then inner (train/val) split loop ───────────
    # Spot axis = donor LOPO (registry); gene axis = the FIXED single split above
    # (GENE_TRAIN / GENE_VAL / TEST_HELDOUT, same every fold). inner_splits is a list
    # (length 1 now) so a future strategy can add inner splits per donor unchanged.
    t_run = time.time()
    for outer in folds:
        fold = outer["fold"]
        test_donor_ids   = outer["test"]
        test_donor_label = ",".join(map(str, test_donor_ids))
        test_spot_idx    = donor_ids_to_spot_idx(donor_per_spot, test_donor_ids)
        test_slides_fold = sorted(set(pst.groups[test_spot_idx].tolist()))
        n_inner = len(outer["inner_splits"])

        for inner_i, inner in enumerate(outer["inner_splits"]):
            train_spot_rows = donor_ids_to_spot_idx(donor_per_spot, inner["train"])
            val_spot_rows   = donor_ids_to_spot_idx(donor_per_spot, inner["val"])
            tag = f"fold{fold}" if n_inner == 1 else f"fold{fold}_inner{inner_i}"

            val_slides_fold = sorted(set(pst.groups[val_spot_rows].tolist()))
            log(f"{'='*72}")
            log(f"Fold {fold} (test donor {test_donor_label}) inner {inner_i} | "
                f"train {len(train_spot_rows)} val {len(val_spot_rows)} test "
                f"{len(test_spot_idx)} spots | val slides {val_slides_fold} | "
                f"gene-train {len(GENE_TRAIN)} / gene-val {len(GENE_VAL)} / "
                f"test-heldout {len(TEST_HELDOUT_GENES)}")

            train_metric_rows = fold_subsample(train_spot_rows, pst.groups, rng, TRAIN_METRIC_SPOTS)

            # Per-fold target z-transform stats — TRAIN spots only, over ALL genes (the
            # gene train/val/heldout split is orthogonal to this per-gene normalization).
            target_loc = target_scale = None
            if DO_TARGET_TRANSFORM:
                tr_t = torch.as_tensor(train_spot_rows, dtype=torch.long, device=pst.device)
                target_loc, target_scale = compute_gene_target_stats(
                    pst.y[tr_t], pst.groups[train_spot_rows], TARGET_TRANSFORM)
                if fold == 0 and inner_i == 0:
                    log(f"target_transform={TARGET_TRANSFORM}: per-gene loc/scale "
                        f"shape={target_loc.shape} (train-only)")

            t_fold = time.time()
            result = train_one_fold_gcm(
                fold=fold, model_type=args.model_type, spec=spec, lam=args.lam,
                pst=pst,
                train_spot_rows=train_spot_rows, val_spot_rows=val_spot_rows,
                train_metric_spot_rows=train_metric_rows,
                gene_train_cols=GENE_TRAIN_COLS, gene_val_cols=GENE_VAL_COLS,
                spot_dim=spot_dim, gene_dim=gene_dim,
                arch=ARCH, hyper_hidden=HYPER_HIDDEN, hyper_dropout=HYPER_DROPOUT,
                interaction_dim=INTERACTION_DIM,
                lr=LR, weight_decay=WEIGHT_DECAY, batch_size=BATCH_SIZE,
                max_epochs=MAX_EPOCHS, patience=PATIENCE,
                warmup_epochs=WARMUP_EPOCHS, grad_clip_norm=GRAD_CLIP,
                center=CENTER, early_stop_criterion=EARLY_STOP,
                corr_metrics=CORR_METRICS, seed=SEED, device=DEVICE,
                amp_dtype=AMP_DTYPE,
                log_alpha_init=NB_LOG_ALPHA_INIT,
                target_loc=target_loc, target_scale=target_scale,
                nb_per_epoch_calibration=NB_PER_EPOCH_CALIB,
            )
            fold_dt = time.time() - t_fold
            model = result["model"]
            if n_params is None:
                n_params = int(sum(p.numel() for p in model.parameters()))

            # ── Final inference + saving (both gated + TIMED so the log shows
            # where fold-end time goes: the metric compute vs the file write) ──
            spot_mean_t = (torch.as_tensor(result["spot_mean"], dtype=torch.float32,
                                           device=pst.device)
                           if result["center"] else None)
            # Prediction matrices are needed if we save them OR compute metrics from
            # them. compute_cpu_metrics=COMPUTE_FINAL_METRICS: when False we get the
            # matrices WITHOUT the scipy/g* pass (save now, recompute metrics offline).
            need_inference = SAVE_PREDICTIONS or COMPUTE_FINAL_METRICS
            val_full = test_full = None
            compute_dt = save_dt = 0.0
            if need_inference:
                t_compute = time.time()
                val_full = run_paired_inference(
                    model, pst, torch.as_tensor(val_spot_rows, dtype=torch.long),
                    all_gene_cols, spec, batch_size=BATCH_SIZE, device=DEVICE, lam=args.lam,
                    spot_mean=spot_mean_t, corr_metrics=CORR_METRICS, metrics_on_gpu=False,
                    target_loc=target_loc, target_scale=target_scale,
                    compute_cpu_metrics=COMPUTE_FINAL_METRICS,
                )
                test_full = run_paired_inference(
                    model, pst, torch.as_tensor(test_spot_idx, dtype=torch.long),
                    all_gene_cols, spec, batch_size=BATCH_SIZE, device=DEVICE, lam=args.lam,
                    spot_mean=spot_mean_t, corr_metrics=CORR_METRICS, metrics_on_gpu=False,
                    target_loc=target_loc, target_scale=target_scale,
                    compute_cpu_metrics=COMPUTE_FINAL_METRICS,
                )
                compute_dt = time.time() - t_compute

            # ── Save per-fold predictions (matrices, classic layout). lzf compresses
            # fast; gzip is single-threaded and dominates the fold-end write. ──
            if SAVE_PREDICTIONS:
                t_save = time.time()
                # Test spots only. Val is needed for training-time selection, which has
                # already run, and val_full still feeds atoms.parquet and metrics.json
                # below, so the saved artifact carries only this fold's held-out spots.
                pred_adata = fold_predictions_adata(
                    {"test": test_full},
                    {"test": test_spot_idx}, fold,
                    ALL_GENES, GENE_ROLE, obs_names, spec, depth_scale=DEPTH_SCALE,
                    target_loc=target_loc, target_scale=target_scale,
                )
                pred_adata.obs["inner_split"] = inner_i
                pred_adata.obs["test_donor"]  = test_donor_label
                pred_adata.write_h5ad(preds_dir / f"preds_{tag}.h5ad", compression="lzf")
                save_dt = time.time() - t_save
                del pred_adata

            # Atoms (per-(slide,gene)) only exist when the metrics were computed.
            if COMPUTE_FINAL_METRICS:
                for split_name, res in (("val", val_full), ("test", test_full)):
                    af = atoms_dataframe_gcm(res, fold, split_name, ALL_GENES,
                                             GENE_ROLE, CORR_METRICS, spec)
                    af["inner_split"] = inner_i
                    af["test_donor"]  = test_donor_label    # the fold's held-out donor
                    atom_frames.append(af)
            torch.save(
                {"state_dict": result["best_state"], "spot_mean": result["spot_mean"],
                 "center": result["center"], "gene_names": ALL_GENES,
                 "model_type": args.model_type, "fold": fold,
                 "inner_split": inner_i, "test_donor": test_donor_label,
                 # Count convention of the weights inside: "v2_no_offset" = the model
                 # emits mu on its own target scale, with no size factor to re-apply
                 # at load time. Loaders read state_dict only -> backward-harmless.
                 "count_convention": "v2_no_offset", "depth_scale": DEPTH_SCALE},
                models_dir / f"model_{tag}.pt",
            )
            if SAVE_LC:
                pd.DataFrame(result["epoch_log"]).to_csv(lc_dir / f"{tag}.csv", index=False)

            # best_epoch = selected epoch = first argmax of the per-epoch select metric
            # (val_<primary>_select, the blend early stopping uses); stop_reason from
            # whether patience fired before MAX_EPOCHS.
            _elog = result["epoch_log"]
            _sel = [e.get(f"val_{PRIMARY}_select", float("nan")) for e in _elog]
            _best_epoch = (int(np.nanargmax(_sel)) + 1 if np.any(np.isfinite(_sel)) else None)

            # ── Headline numbers for this fold (slide-averaged per-gene corr) — only
            # when the metrics were computed. test_<primary>_heldout_genes IS the
            # per-donor LOPO point x_i (cross-gene generalization on the held-out donor).
            if COMPUTE_FINAL_METRICS:
                ppg = test_full[f"{PRIMARY}_per_gene"]
                heldout_pos = [gene_to_col[g] for g in TEST_HELDOUT_GENES]
                cv_pos      = [gene_to_col[g] for g in CV_GENES]
                fold_summaries.append({
                    "fold": fold, "inner_split": inner_i, "test_donor": test_donor_label,
                    "best_val_select":            result["best_val_select"],
                    "best_epoch":                 _best_epoch,
                    "n_epochs_ran":               len(_elog),
                    "stop_reason":                ("early_stop" if len(_elog) < MAX_EPOCHS
                                                   else "max_epochs"),
                    f"test_{PRIMARY}_heldout_genes": _nm([ppg[i] for i in heldout_pos]),
                    f"test_{PRIMARY}_devgenes":      _nm([ppg[i] for i in cv_pos]),
                    # MAE/MSE on the per-gene Pearson scale (g* for count), median over
                    # genes; within-family only — see metric_scale.
                    "test_mae_median":            test_full.get("mae_median"),
                    "val_mae_median":             val_full.get("mae_median"),
                    "test_mse_median":            test_full.get("mse_median"),
                    "val_mse_median":             val_full.get("mse_median"),
                    "test_ece_pit":               test_full.get("ece_pit"),
                    "val_ece_pit":                val_full.get("ece_pit"),
                    "test_disp_signed":           test_full.get("disp_signed"),
                    "val_disp_signed":            val_full.get("disp_signed"),
                    "test_pit_bias":              test_full.get("pit_bias"),
                    "val_pit_bias":               val_full.get("pit_bias"),
                    "fold_duration_seconds":      round(fold_dt, 3),
                    "n_train_genes":              len(GENE_TRAIN),
                    "n_val_genes":                len(GENE_VAL),
                    "test_slides":                test_slides_fold,
                    "val_slides":                 val_slides_fold,
                })
                # Count models: the option-A log1p(mu/depth_scale) cross-check, the same
                # scale the per-epoch early-stop metric uses. It is a global slide-aware
                # scalar over all scored genes, whereas the headline above is the per-gene
                # g* restricted to the held-out genes. Stored as a diagnostic when present,
                # absent for Gaussian and MSE. It comes from the CPU scipy pass, so it
                # agrees with the on-device per-epoch value only to about the sixth decimal
                # (float32 operands against float64). It is a cross-check, never a
                # selection criterion. The per-(slide, gene) values are in atoms.parquet.
                if f"{PRIMARY}_logmean" in test_full:
                    fold_summaries[-1][f"test_{PRIMARY}_logmean"] = test_full.get(f"{PRIMARY}_logmean")
                    fold_summaries[-1][f"val_{PRIMARY}_logmean"]  = val_full.get(f"{PRIMARY}_logmean")

            cv_folds_info.append({
                "fold": fold, "inner_split": inner_i, "test_donor": test_donor_ids,
                "test_spot_idx":   test_spot_idx.tolist(),
                "slide_train_idx": train_spot_rows.tolist(),
                "slide_val_idx":   val_spot_rows.tolist(),
                "test_slides":     test_slides_fold,
                "val_slides":      val_slides_fold,
            })

            # Fold timing line: training + the fold-end compute (inference + scipy/g*)
            # vs the file write, so a glance at the log shows which one dominates.
            if COMPUTE_FINAL_METRICS:
                _head = (f"test {PRIMARY} heldout={fold_summaries[-1][f'test_{PRIMARY}_heldout_genes']:.4f} "
                         f"devgenes={fold_summaries[-1][f'test_{PRIMARY}_devgenes']:.4f}")
            else:
                _head = "metrics skipped (compute_final_metrics=False)"
            log(f"Fold {fold} done in {fold_dt/60:.1f} min | {_head} | "
                f"compute {compute_dt:.1f}s save {save_dt:.1f}s")

            del model, result
            if val_full is not None:
                del val_full, test_full
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    run_duration = time.time() - t_run

    # ── splits.json is cheap + always useful (the CV partition) ──────────────
    write_splits_json(output_dir, SEED, CV_STRATEGY, CV_PARAMS, LOPO, donors, folds,
                      GENE_TRAIN, GENE_VAL, TEST_HELDOUT_GENES, gene_split_info,
                      cv_folds_info)

    # ── LOPO stats + atoms.parquet + metrics.json: only when the per-fold metrics
    # were computed. With compute_final_metrics=False the run saved predictions (if
    # enabled) but NO metrics — recompute atoms/metrics/lopo offline from predictions/.
    lopo_stats = None
    if COMPUTE_FINAL_METRICS:
        # Per-donor points x_i = test_<primary>_heldout_genes (avg over inner splits) ->
        # mean / inter-donor SD (ddof=1) / SEM + Student-t 95% CI + new-donor PI (df=K-1).
        # Optimistic (folds share training donors; Bengio & Grandvalet 2004) — the K
        # points are kept too. None for fixed_holdout_kfold.
        if LOPO:
            sdf = pd.DataFrame(fold_summaries)
            lopo_stats = {
                "heldout_genes": _lopo_block(sdf.groupby("test_donor")[f"test_{PRIMARY}_heldout_genes"].mean()),
                "devgenes":      _lopo_block(sdf.groupby("test_donor")[f"test_{PRIMARY}_devgenes"].mean()),
                "primary_corr_metric": PRIMARY,
            }
            _hb = lopo_stats["heldout_genes"]
            log(f"LOPO {PRIMARY} (heldout genes): mean={_hb['mean']:.4f} ± SD {_hb['sd']:.4f} "
                f"(K={_hb['K']}); 95% CI {_hb['ci95_mean'][0]:.4f}..{_hb['ci95_mean'][1]:.4f}")

        atoms_df = pd.concat(atom_frames, ignore_index=True)
        atoms_df.to_parquet(output_dir / "atoms.parquet", index=False)

        write_metrics_json(
            output_dir, args, cfg, gcm_cfg, spec, DEVICE, adata_path,
            CORR_METRICS, PRIMARY, EARLY_STOP, CV_STRATEGY, CV_PARAMS, LOPO, donors, SEED,
            CENTER, LR, WEIGHT_DECAY, BATCH_SIZE, MAX_EPOCHS, PATIENCE, WARMUP_EPOCHS,
            GRAD_CLIP, HYPER_HIDDEN, HYPER_DROPOUT, INTERACTION_DIM, use_projection,
            permute_embeddings,
            ARCH, spot_dim, gene_dim, gene_emb_source, n_params,
            gene_split_info, GENE_TRAIN, GENE_VAL, CV_GENES, TEST_HELDOUT_GENES,
            len(DEV_GENES_FILE),
            fold_summaries, lopo_stats, atoms_df, slides, counts, low_spot_slides,
            run_duration, is_gaussian,
        )
    else:
        # No atoms / full metrics — but the Snakemake rule declares metrics.json as its
        # output sentinel, so write a minimal STUB (with a metrics_deferred flag) so the
        # rule completes. The real metrics are recomputed offline from predictions/.
        import json
        stub = {
            "metrics_deferred": True,
            "note": ("compute_final_metrics=False — no per-fold scipy/g* was run; "
                     "recompute atoms/metrics/lopo offline from predictions/."),
            "dataset": args.dataset, "model_type": args.model_type,
            "run_id_components": {
                "dev_gene_set_id":    args.dev_gene_set_id,
                "chrom_split_spec":   args.chrom_split_spec,
                "gene_emb_id":        args.gene_emb_id,
                "use_projection":     use_projection,
                "permute_embeddings": permute_embeddings,
            },
            "primary_corr_metric":  PRIMARY,
            "n_train_genes":        len(GENE_TRAIN),
            "n_val_genes":          len(GENE_VAL),
            "n_test_heldout_genes": len(TEST_HELDOUT_GENES),
            "saved_predictions":    bool(SAVE_PREDICTIONS),
            "run_duration_seconds": round(run_duration, 1),
            # Provenance + full hyperparameter snapshot so the completed metrics.json is
            # self-contained (complete_deferred_metrics.py merges these through; cv/donors
            # + gene split come from splits.json). KEEP IN SYNC with write_metrics_json.
            "radius": args.radius, "model_kind": kind_of(args.model_type),
            "target_source": spec.target_source, "uses_lam": spec.uses_lam,
            "is_probabilistic": spec.is_probabilistic,
            "lam": args.lam if spec.uses_lam else None,
            "count_convention": "v2_no_offset",
            "depth_scale": DEPTH_SCALE,
            "count_common_depth": COUNT_COMMON_DEPTH,
            "metric_scale": METRIC_SCALE,
            "target_transform": (cfg.get("target_transform", "none")
                                 if spec.target_source == "X" else None),
            "corr_metrics": list(CORR_METRICS), "early_stop_criterion": EARLY_STOP,
            "seed": SEED, "center": CENTER,
            "cv_strategy": CV_STRATEGY, "cv_params": CV_PARAMS, "lopo": LOPO,
            "n_donors": len(donors), "n_folds": len(folds),
            "donors_ordered": donors,
            "test_donors_per_fold": [f["test"] for f in folds],
            "split_grouping": "donor", "metric_grouping": "slide",
            "learning_rate": LR, "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE, "spot_batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
            "warmup_epochs": WARMUP_EPOCHS, "grad_clip_norm": GRAD_CLIP,
            "hyper_hidden": list(HYPER_HIDDEN), "hyper_dropout": HYPER_DROPOUT,
            "interaction_dim": INTERACTION_DIM,
            "spot_dim": int(spot_dim), "gene_dim": int(gene_dim),
            "gene_emb_source": gene_emb_source,
            "train_metric_spots_per_slide": gcm_cfg.get('train_metric_spots_per_slide',
                                                        cfg.get('train_metric_spots_per_slide')),
            "input_adata": str(adata_path),
        }
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(stub, f, indent=2)
        log("compute_final_metrics=False -> wrote metrics.json STUB (metrics_deferred); "
            "skipped atoms.parquet + lopo (recompute offline from predictions/).")

    # gene_embeddings_used.npz (GCM-specific reproducibility provenance)
    ge_payload = {
        "gene_emb_matrix": gene_emb_matrix,
        "gene_names":      np.array(ALL_GENES),
        "source_path":     np.array(gene_emb_source),
        "permuted":        np.array(permute_embeddings),
    }
    if perm_info is not None:
        ge_payload["permutation_seed"] = np.array(perm_info['seed'])
        ge_payload["permutation_donors"] = np.array(
            [perm_info['permutation_map'][g] for g in ALL_GENES]
        )
    np.savez(output_dir / "gene_embeddings_used.npz", **ge_payload)

    # Top-level model.pt = fold-0 copy (parity with classic).
    import shutil
    shutil.copy(models_dir / "model_fold0.pt", output_dir / "model.pt")

    if COMPUTE_FINAL_METRICS:
        hg = _nm([f[f"test_{PRIMARY}_heldout_genes"] for f in fold_summaries])
        log(f"All folds done in {run_duration/60:.1f} min. "
            f"PRIMARY test {PRIMARY} (heldout genes) = {hg:.4f}")
    else:
        log(f"All folds done in {run_duration/60:.1f} min. "
            f"(compute_final_metrics=False — no headline; metrics deferred)")
    log("Done.")


# ─────────────────────────────────────────────────────────────────────────────
# Logging / writing helpers
# ─────────────────────────────────────────────────────────────────────────────

def fold_subsample(train_rows, groups, rng, cap):
    """Fixed seeded per-slide subsample of train spot rows for the per-epoch
    TRAIN metric (training itself uses all rows). None cap -> all rows."""
    if cap is None:
        return train_rows
    g = groups[train_rows]
    keep = []
    for s in np.unique(g):
        rows = train_rows[g == s]
        if len(rows) > cap:
            rows = rng.choice(rows, size=cap, replace=False)
        keep.append(rows)
    return np.concatenate(keep)


def log_chrom_repartition(all_genes, gene_chrom_map, test_heldout_genes,
                          gene_split_info):
    """Compact per-chromosome count of the current gene set at the top of the log
    (no full gene-list dump). Test-heldout chromosomes are bracketed + starred, and
    the single train/val/test-heldout gene split is summarized."""
    from collections import Counter
    chrom_of = {g: str(gene_chrom_map.get(g, 'unknown')) for g in all_genes}
    counts = Counter(chrom_of[g] for g in all_genes)
    heldout_chroms = set(map(str, gene_split_info['test_heldout_chroms']))
    ordered = sorted(counts, key=_chrom_sort_key)
    parts = [f"[{c}:{counts[c]}]*" if c in heldout_chroms else f"{c}:{counts[c]}"
             for c in ordered]
    pct = 100 * len(test_heldout_genes) / max(len(all_genes), 1)
    log(f"Dev gene set: {len(all_genes)} present genes / {len(counts)} chromosomes")
    log("  " + "  ".join(parts))
    log(f"  * = TEST-HELDOUT ({len(heldout_chroms)} chroms / "
        f"{len(test_heldout_genes)} genes / {pct:.1f}%)")
    log(f"  gene split (fixed across donor folds): train={gene_split_info['n_train_genes']} "
        f"val={gene_split_info['n_val_genes']} test_heldout={gene_split_info['n_test_heldout_genes']} "
        f"| val={gene_split_info['val']}")


def write_splits_json(output_dir, seed, cv_strategy, cv_params, lopo, donors, folds,
                      gene_train, gene_val, test_heldout_genes, gene_split_info,
                      cv_folds_info):
    """splits.json — the SPOT axis (donor LOPO, per-fold rows as GLOBAL adata_row
    indices) + the FIXED gene axis (train/val/test-heldout gene names, same for
    every fold), so any split joins straight back to adata.obs / predictions."""
    splits = {
        "seed": seed, "cv_strategy": cv_strategy, "cv_params": cv_params, "lopo": lopo,
        "split_grouping": "donor", "metric_grouping": "slide",
        "donors_ordered": donors,
        # fixed_holdout_kfold: the fixed test-donor set (same every fold). Under LOPO
        # this is None — each fold's held-out donor is in folds[*].test_donor.
        "test_donors": (None if lopo else (folds[0]["test"] if folds else [])),
        # Gene axis is a SINGLE split, fixed across all donor folds.
        "gene_split": {
            "gene_train": gene_train, "gene_val": gene_val,
            "test_heldout_genes": test_heldout_genes, "info": gene_split_info,
        },
        "folds": cv_folds_info,
    }
    with open(output_dir / "splits.json", "w") as f:
        json.dump(splits, f, default=_json_default)


def write_metrics_json(output_dir, args, cfg, gcm_cfg, spec, device, adata_path,
                       corr_metrics, primary, early_stop, cv_strategy, cv_params,
                       lopo, donors, seed, center, lr, weight_decay, batch_size,
                       max_epochs, patience, warmup_epochs, grad_clip,
                       hyper_hidden, hyper_dropout, interaction_dim,
                       use_projection, permute_embeddings, arch, spot_dim, gene_dim,
                       gene_emb_source, n_params, gene_split_info, gene_train, gene_val,
                       cv_genes, test_heldout_genes, n_dev_genes_file, fold_summaries,
                       lopo_stats, atoms_df, slides, counts, low_spot_slides,
                       run_duration, is_gaussian):
    """metrics.json — classic provenance blocks + GCM-specific blocks. PRIMARY
    number = test_{primary}_heldout_genes (cross-gene generalization)."""
    # Batched GeneHyperMLP: arch sizes the spot backbone, hyper_hidden and hyper_dropout
    # size the gene hypernetwork, and the interaction dim defaults to the backbone's last
    # hidden width. Spot and gene inputs are never concatenated, so there is no combined
    # input dim to record.
    architecture = {
        "gene_head":   "hypernet" if hyper_hidden else "bilinear",
        "arch_source": "config" if arch else "default",
        "spot_hidden_dims": (arch.get("hidden_dims", list(_DEFAULT_HIDDEN)) if arch
                             else list(_DEFAULT_HIDDEN)),
        "dropout":     (arch.get("dropout", list(_DEFAULT_DROPOUT)) if arch
                        else list(_DEFAULT_DROPOUT)),
        "activation":  (arch.get("activation", _DEFAULT_ACT) if arch else _DEFAULT_ACT),
        "hyper_hidden":    list(hyper_hidden),
        "hyper_dropout":   hyper_dropout,
        "interaction_dim": interaction_dim,    # null => spot backbone's last hidden dim
        "spot_dim": int(spot_dim), "gene_dim": int(gene_dim),
        "out_dim": 1, "n_parameters": n_params,
    }

    amp_mode, amp_dtype, gpu_cap = resolve_amp_dtype(gcm_cfg)
    environment = {
        "started_at": STARTED_AT,
        "finished_at": datetime.now().astimezone().isoformat(),
        "run_duration_seconds": round(run_duration, 3),
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_nodelist": os.environ.get("SLURM_JOB_NODELIST"),
        "argv": sys.argv, "device": device,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__, "numpy_version": np.__version__,
        "pandas_version": pd.__version__, "anndata_version": ad.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_capability": list(gpu_cap) if torch.cuda.is_available() else None,
        "amp_mode": amp_mode,
        "amp_dtype": str(amp_dtype) if amp_dtype is not None else None,
    }

    # NaN summary from atoms (per split), on the primary metric (classic style).
    nan_summary = {}
    for split in ("val", "test"):
        sub = atoms_df[atoms_df.split == split]
        nan_rows = sub[sub[primary].isna()]
        nan_summary[split] = {
            "n_nan": int(len(nan_rows)), "n_total": int(len(sub)),
            "examples of NaN values": nan_rows[["slide", "gene"]].head(10).to_dict("records"),
        }

    preprocess_config_path = os.path.join(os.path.dirname(adata_path), "preprocess_config.json")

    # Count-model provenance. This function is top level and does not share main()'s
    # locals, so it calls the same helper main() calls. Re-deriving the formulas here
    # would let the two blocks drift apart.
    DEPTH_SCALE, COUNT_COMMON_DEPTH, METRIC_SCALE = count_provenance(
        cfg, args.dataset, args.model_type, spec)

    metrics = {
        "run_id_components": {
            "dev_gene_set_id":    args.dev_gene_set_id,
            "chrom_split_spec":   args.chrom_split_spec,
            "gene_emb_id":        args.gene_emb_id,
            "use_projection":     use_projection,
            "permute_embeddings": permute_embeddings,
        },
        "dataset": args.dataset, "radius": args.radius,
        "model_type": args.model_type, "model_kind": kind_of(args.model_type),
        "target_source": spec.target_source, "uses_lam": spec.uses_lam,
        "is_probabilistic": spec.is_probabilistic,
        "lam": args.lam if spec.uses_lam else None,
        "loss_function": spec.loss_display_name,
        # Count convention "v2_no_offset": no size factor / offset / per-spot exposure
        # anywhere; each count model fits its likelihood directly on its own target,
        # and depth_scale (float for the cp families, None for the depth-blind _raw
        # ones) is used ONLY as the divisor before log1p when reporting the mean corr.
        "count_convention": "v2_no_offset",
        "depth_scale": DEPTH_SCALE,
        "count_common_depth": COUNT_COMMON_DEPTH,
        "target_transform": (cfg.get("target_transform", "none")
                             if spec.target_source == "X" else None),
        "metric_scale": METRIC_SCALE,
        # MAE/MSE ride the per-gene Pearson values (g* for count, model-output for
        # Gaussian), median over genes. WITHIN-FAMILY only — do NOT cross-compare.
        "error_metric_scale": "MAE/MSE on the per-gene Pearson scale (log1p-CP10k; g* "
                              "for count); median over genes; WITHIN-FAMILY only",
        "corr_metrics": list(corr_metrics), "primary_corr_metric": primary,
        "early_stop_criterion": early_stop,
        "cv_strategy": cv_strategy, "cv_params": cv_params, "lopo": lopo,
        "n_donors": len(donors), "n_folds": len(fold_summaries),
        "donors_ordered": donors,
        "test_donors_per_fold": [f["test_donor"] for f in fold_summaries],
        "split_grouping": "donor", "metric_grouping": "slide",
        "seed": seed, "center": center,
        # hyperparameter snapshot (not in the output path -> versioned by run_id)
        "learning_rate": lr, "weight_decay": weight_decay, "batch_size": batch_size,
        "max_epochs": max_epochs, "patience": patience,
        "warmup_epochs": warmup_epochs, "grad_clip_norm": grad_clip,
        "spot_batch_size": batch_size,
        "hyper_hidden": list(hyper_hidden), "hyper_dropout": hyper_dropout,
        "interaction_dim": interaction_dim,
        "train_metric_spots_per_slide": gcm_cfg.get('train_metric_spots_per_slide',
                                                    cfg.get('train_metric_spots_per_slide')),
        # provenance
        "architecture": architecture,
        "spot_dim": int(spot_dim), "gene_dim": int(gene_dim),
        "gene_emb_source": gene_emb_source,
        "input_adata": str(adata_path),
        "preprocess_config": preprocess_config_path if os.path.exists(preprocess_config_path) else None,
        "environment": environment,
        # GCM gene-split block — ONE fixed split (test/val/train), same every donor
        # fold. test_heldout genes carry the cross-gene headline.
        "gene_split_info": {
            **gene_split_info,
            "test_heldout_chroms": list(gene_split_info['test_heldout_chroms']),
            "train_chroms":        list(gene_split_info['train_chroms']),
        },
        "genes": {
            "GENE_TRAIN": gene_train, "GENE_VAL": gene_val,
            "CV_GENES": cv_genes, "TEST_HELDOUT_GENES": test_heldout_genes,
            "n_gene_train": len(gene_train), "n_gene_val": len(gene_val),
            "n_cv_genes": len(cv_genes), "n_test_heldout_genes": len(test_heldout_genes),
            "n_dropped_dev_genes": n_dev_genes_file - (len(cv_genes) + len(test_heldout_genes)),
        },
        # per-fold + headline
        "per_fold": fold_summaries,
        # LOPO statistics over donors (per-donor points + mean/SD/SEM + t-CI/t-PI on
        # both the heldout-gene headline and the devgenes side). None for
        # fixed_holdout_kfold. HEADLINE under LOPO = lopo_stats["heldout_genes"]["mean"].
        "lopo_stats": lopo_stats,
        "summary": {
            f"test_{primary}_heldout_genes":     _nm([f[f"test_{primary}_heldout_genes"] for f in fold_summaries]),
            f"test_{primary}_heldout_genes_std": _sd([f[f"test_{primary}_heldout_genes"] for f in fold_summaries]),
            f"test_{primary}_devgenes":          _nm([f[f"test_{primary}_devgenes"] for f in fold_summaries]),
            f"test_{primary}_devgenes_std":      _sd([f[f"test_{primary}_devgenes"] for f in fold_summaries]),
            "mean_best_val_select":              _nm([f["best_val_select"] for f in fold_summaries]),
            "mean_test_mae_median":  _nm([f["test_mae_median"] for f in fold_summaries]),
            "mean_test_mse_median":  _nm([f["test_mse_median"] for f in fold_summaries]),
            "mean_test_ece_pit":     _nm([f["test_ece_pit"] for f in fold_summaries]) if is_gaussian else None,
            "mean_test_disp_signed": _nm([f["test_disp_signed"] for f in fold_summaries]) if is_gaussian else None,
            # Count models: global option-A (log1p(mu/depth_scale)) cross-check of the headline.
            **({f"test_{primary}_logmean": _nm([f.get(f"test_{primary}_logmean") for f in fold_summaries]),
                f"val_{primary}_logmean":  _nm([f.get(f"val_{primary}_logmean") for f in fold_summaries])}
               if any(f.get(f"test_{primary}_logmean") is not None for f in fold_summaries) else {}),
        },
        "nan_summary": nan_summary,
        "low_spot_slides": low_spot_slides,
        "per_slide_spot_counts": {str(s): int(c) for s, c in zip(slides, counts)},
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=_json_default)
    log("Wrote metrics.json, atoms.parquet, splits.json, genes.txt, "
        "predictions/preds_fold*.h5ad, models/, model.pt, gene_embeddings_used.npz")


if __name__ == "__main__":
    main()
