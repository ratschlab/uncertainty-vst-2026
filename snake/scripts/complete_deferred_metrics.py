#!/usr/bin/env python
"""Complete the metrics a `compute_final_metrics=False` run deferred, from its saved
predictions. CPU-only; handles both pipelines (classic `results/` + GCM `results_gcm/`).

A run launched with `compute_final_metrics=False` trains, saves
`predictions/preds_fold*.h5ad` + `splits.json`, and writes only a stub `metrics.json`
(`"metrics_deferred": true`): it skips the per-(slide,gene) scipy correlation, the
native-CDF PIT and the count `g*` aggregation. This script runs that aggregation offline by
calling the same `src/evaluation` functions the training final inference calls, then writes
the real `atoms.parquet` and a full `metrics.json` in place of the stub. The stub is backed
up to `metrics.stub.json`.

No GPU and no model forward: every metric comes from the saved arrays (`X`=means,
`layers["y"]`, `layers["pred_var"]` [Gaussian], `layers["dispersion"]` [NB],
`uns["count_convention"]` + `uns["depth_scale"]` [count],
`obs[slide/split/fold/inner_split/test_donor]`, `var_names`, GCM `var["gene_role"]`).
Count models do not persist `pred_var`; it is recomputed here as
`count_pred_var(X, dispersion)`, mirroring `benchmark_summary`.

Count scale. A count model fits its likelihood directly on its own target, with no size
factor and no offset, so the saved `X` (mu) and `layers["dispersion"]` (alpha) are both on
that target's scale. `depth_scale` serves exactly one purpose: the divisor applied before
log1p when reporting the mean correlation. Predictions declare this with
`uns["count_convention"] = "v2_no_offset"` (+ `uns["depth_scale"]`, absent -> None for the
depth-blind *_raw models). On a file without that marker `resolve_count_scale` raises
instead of guessing a scale.

GCM predictions hold test spots only and carry no `layers["y"]`
(train_gcm.py::fold_predictions_adata). When that layer is missing, `y` is rebuilt from the
shared preprocessed adata the run trained on (metrics.json["input_adata"]), joined back via
`obs["adata_row"]` + `var_names` (`target_values_for_predictions`).

Submit as a plain CPU job (no `-p gpu`, no `--gres` -> lands on the `compute` partition).
Size `--time` for the aggregation, which is the whole cost the run deferred. For a count
model it also covers `g*`, unless a `gstar_fold{k}.h5ad` companion sits next to each
predictions file: those are read instead. Recomputing `g*` here runs on CPU and is far
slower than the GPU job that writes the companions.

Usage
-----
    python snake/scripts/complete_deferred_metrics.py <run_dir> [<run_dir> ...]
    python snake/scripts/complete_deferred_metrics.py --glob "results_gcm/*/*/*/*/*"
    python snake/scripts/complete_deferred_metrics.py --glob "results/*/*/*/*/*" --dry-run

A <run_dir> is a directory holding a (stub) `metrics.json` + `predictions/`. Classic vs GCM
is auto-detected. Runs whose `metrics.json` is already complete (no `metrics_deferred`), or
that saved no predictions, are skipped (use `--force` to recompute a complete one anyway).
"""
import argparse
import glob as globmod
import json
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats as scipy_stats

# Project root on sys.path — the repository is not necessarily pip-installed (same shim
# the training scripts use). insert(0) so this tree's src/ wins over any editable install.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.models.specs import get_spec                                # noqa: E402
from src.evaluation.metrics_slide_aware import compute_metrics       # noqa: E402
from src.evaluation.count_metrics import count_metrics_final, count_pred_var  # noqa: E402
from src.data.load_and_preprocess import target_values_for_predictions  # noqa: E402

# The native-CDF PIT calibration atom columns (same set atoms_dataframe writes).
CALIB_ATOM_COLS = ("ece_pit", "pit_bias", "disp_signed",
                   "var_residual_corr", "var_residual_corr_partial")
# metrics.json keys that mark the STUB itself (dropped before merging its provenance).
STUB_MARKERS = ("metrics_deferred", "note", "saved_predictions")


def _nm(v):
    """nanmean of a list, ignoring None; nan if empty/all-None (mirrors trainers' _nm)."""
    a = np.array([x for x in v if x is not None], dtype=float)
    return float(np.nanmean(a)) if a.size and np.isfinite(a).any() else float("nan")


def _sd(v):
    a = np.array([x for x in v if x is not None], dtype=float)
    return float(np.nanstd(a, ddof=1)) if a.size >= 2 else float("nan")


def _dense(a):
    """Predictions matrices are dense, but be robust to a sparse X/layer."""
    return np.asarray(a.todense() if hasattr(a, "todense") else a, dtype=float)


def _json_default(o):
    """numpy-aware JSON fallback: numbers stay numbers instead of being stringified.
    Mirrors the trainers' _json_default so the completed metrics.json keeps their types."""
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def resolve_count_scale(ad, path):
    """The count scale of one predictions file -> depth_scale.

    `uns["count_convention"] == "v2_no_offset"` says `X` is already mu on the model's own
    fitted target scale, so nothing is rescaled. The reporting constant is
    `uns["depth_scale"]`, written only when it is not None, so `.get` yields None for the
    depth-blind *_raw models.
    """
    if ad.uns.get("count_convention") != "v2_no_offset":
        raise ValueError(
            f"{path}: artifact with no count_convention marker. "
            f"Re-run the precompute.")
    ds = ad.uns.get("depth_scale")
    return None if ds is None else float(ds)


# ── per-(fold, split) metric recompute — mirrors run_inference's FINAL path ──────────────
def recompute_result(ad_split, spec, corr_metrics, adata_path=None, depth_scale=None,
                     gstar=None):
    """Rebuild the `result` dict (aggregated corr/calibration scalars + per-(slide,gene)
    atoms) for one (fold, split) from a saved predictions AnnData slice, by calling the
    same functions the training final inference calls (inference_indexed.run_inference /
    train_loop_gcm.run_paired_inference). Count spec -> count_metrics_final (mu and alpha
    on the model's own target scale, g* on the log1p(y/depth_scale) reporting axis, native
    PIT on the target scale); else -> compute_metrics.

    `depth_scale` comes from resolve_count_scale (count specs only).

    `y` comes from `layers["y"]`. GCM predictions have no such layer
    (train_gcm.py::fold_predictions_adata), so there `adata_path`
    (metrics.json["input_adata"]) is required to rebuild it from the shared preprocessed
    adata.
    """
    means = _dense(ad_split.X)                        # mu on the model's own fitted target scale
    if "y" in ad_split.layers:
        y = _dense(ad_split.layers["y"])
    else:
        if adata_path is None:
            raise ValueError(
                "predictions carry no layers['y'] and no adata_path was given to "
                "recover it (metrics.json['input_adata'])."
            )
        loc   = (ad_split.var["target_loc"].to_numpy()
                if "target_loc" in ad_split.var else None)
        scale = (ad_split.var["target_scale"].to_numpy() if loc is not None else None)
        y = target_values_for_predictions(
            adata_path, spec.target_source, list(ad_split.var_names),
            ad_split.obs["adata_row"].to_numpy(),
            depth_scale=depth_scale,       # REQUIRED for cp_counts/cp_counts_nr targets
            target_loc=loc, target_scale=scale,
        )
    groups = ad_split.obs["slide"].to_numpy()
    is_count = spec.target_source in ("raw_counts", "cp_counts", "cp_counts_nr")

    # NB dispersion layer; Poisson has none. Gaussian heads persist pred_var, count models
    # do not, so for them it is recomputed as count_pred_var(X, dispersion). Deterministic
    # (MSE) heads have neither.
    alpha = _dense(ad_split.layers["dispersion"]) if "dispersion" in ad_split.layers else None
    if is_count and spec.is_probabilistic:
        pred_var = count_pred_var(means, alpha)         # Poisson: X ; NB: X + alpha*X^2
    elif "pred_var" in ad_split.layers:
        pred_var = _dense(ad_split.layers["pred_var"])
    else:
        pred_var = None

    result = {"means": means, "y": y, "groups": groups, "pred_var": pred_var}
    if is_count:
        # depth_scale is a scalar (None for *_raw), never a per-spot column: it moves only
        # the g* / *_logmean reporting axis; PIT and pred_var stay on the target scale.
        # device="cpu" is the slow path for g*. When `gstar` is supplied (a companion file
        # precomputed on the GPU, one job per fold) count_metrics_final skips the truncated
        # sum and never reaches `device`.
        result.update(count_metrics_final(means, y, depth_scale, groups, corr_metrics,
                                          alpha=alpha, pred_var=pred_var, device="cpu",
                                          smear=(spec.target_source == "cp_counts_nr"),
                                          gstar=gstar))
    else:
        result.update(compute_metrics(means, y, groups=groups, pred_var=pred_var,
                                      corr_metrics=corr_metrics))
    result["pred_var"] = pred_var
    return result


def build_atoms(result, fold, split, gene_names, corr_metrics, gene_role=None):
    """One row per (slide, gene): a column per corr metric (+ mae/mse + native-PIT
    calibration for probabilistic specs), gene indices -> names. gene_role != None adds the
    GCM tag. Mirrors atoms_dataframe (classic) / atoms_dataframe_gcm (GCM) exactly."""
    rows = {}

    def row_for(a):
        gi = a["gene"]
        r = rows.setdefault((a["slide"], gi), {
            "fold": fold, "split": split, "slide": a["slide"],
            "gene": gene_names[gi], "n_spots": a.get("n_spots"),
        })
        if gene_role is not None:
            r["gene_role"] = gene_role[gi]
        return r

    for name in corr_metrics:
        for a in result[f"{name}_per_slide_gene"]:
            row_for(a)[name] = a[name]
        # Count models only: option A (log1p(mu/depth_scale), mean-only, ignores alpha)
        # next to the option-B g* column above, so this path writes the same two estimators
        # the training writers emit.
        logmean_atoms = result.get(f"{name}_logmean_per_slide_gene")
        if logmean_atoms is not None:
            for a in logmean_atoms:
                row_for(a)[f"{name}_logmean"] = a[f"{name}_logmean"]
    err = result.get("error_per_slide_gene")
    if err is not None:
        for a in err:
            r = row_for(a)
            r["mae"] = a["mae"]
            r["mse"] = a["mse"]
    cal = result.get("calibration_per_slide_gene")
    if cal is not None:
        for a in cal:
            r = row_for(a)
            for k in CALIB_ATOM_COLS:
                r[k] = a[k]
    return pd.DataFrame(list(rows.values()))


def lopo_block(per_donor):
    """mean / inter-donor SD (ddof=1) / SEM + Student-t 95% CI + new-donor PI (df=K-1)
    over the K per-donor points. Mirrors train.py's lopo_stats block exactly."""
    x = per_donor.to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    k = int(len(x))
    mean = float(np.mean(x)) if k else float("nan")
    sd = float(np.std(x, ddof=1)) if k >= 2 else float("nan")
    sem = float(sd / np.sqrt(k)) if k >= 2 else float("nan")
    if k >= 2:
        tstar = float(scipy_stats.t.ppf(0.975, k - 1))
        half = tstar * sem
        half_pi = tstar * sd * float(np.sqrt(1.0 + 1.0 / k))
        ci, pi = [mean - half, mean + half], [mean - half_pi, mean + half_pi]
    else:
        tstar, ci, pi = float("nan"), [float("nan")] * 2, [float("nan")] * 2
    return {"mean": mean, "sd": sd, "sem": sem, "K": k, "df": k - 1,
            "t_star_0975": tstar, "ci95_mean": ci, "pi95_new_donor": pi,
            "per_donor": {str(d): (None if not np.isfinite(v) else float(v))
                          for d, v in per_donor.items()}}


def metric_scale_string(spec, depth_scale, common_depth, target_transform):
    """`metric_scale` provenance, rebuilt post-hoc: one explicit arm per target_source.
    train.py and train_gcm.py::count_provenance hold the other two copies of these strings
    and must agree with this one. The arms cover every target_source a registered spec can
    carry ("cp_counts", "cp_counts_nr", "raw_counts", "X"); the final return fires only for
    a target_source with no arm here, and says so rather than mislabelling the run. The
    three count reporting axes:
        _cp, _cp_nr -> log1p(CP10k)      comparable to the Gaussian models
        _raw        -> log1p(raw counts) comparable to neither the Gaussian nor _cp
    """
    if spec.target_source == "cp_counts":
        return (f"common-depth count (T={common_depth}, depth_scale={depth_scale}); the "
                f"NB/Poisson is fitted DIRECTLY on the target round(CP_T) (no offset, no "
                f"size factor, no per-spot exposure), so the saved X(mu) and the dispersion "
                f"alpha are BOTH on that CP_T scale. Reported corr = "
                f"g*=E[log1p(Y/depth_scale)] vs log1p(round(CP_T)/depth_scale) = "
                f"log1p(CP10k), comparable to the Gaussian models; cross-check "
                f"log1p(mu/depth_scale) under *_logmean. Calibration (native-CDF one-sided "
                f"PIT) and var_residual_corr[_partial] stay on the CP_T scale with no "
                f"constant applied")
    if spec.target_source == "cp_counts_nr":
        return (f"common-depth count NO-ROUNDING (T={common_depth}, "
                f"depth_scale={depth_scale}); target = CONTINUOUS CP_T = CP10k*depth_scale "
                f"(no np.rint), fitted DIRECTLY by the discrete Poisson/NB via the "
                f"continuous NLL (no offset, no size factor). X(mu) and alpha are BOTH on "
                f"that CP_T scale. Reported corr = g*=E[log1p(Y/depth_scale)] vs "
                f"log1p(CP_T/depth_scale) = log1p(CP10k) EXACT (Gaussian-comparable, "
                f"un-quantized); cross-check under *_logmean. Calibration = native-CDF "
                f"one-sided PIT with the SMEARED CDF F~, on the CP_T scale")
    if spec.target_source == "raw_counts":
        return ("depth-blind raw counts (depth_scale=None -> divisor 1.0); the NB/Poisson "
                "is fitted DIRECTLY on adata.layers['raw_counts'] (no offset, no size "
                "factor, no per-spot exposure), so X(mu) and alpha are BOTH on the "
                "raw-count scale. Reported corr = g*=E[log1p(Y)] vs log1p(raw_counts), "
                "i.e. the HEADLINE IS ON log1p(RAW COUNTS): it is NOT comparable to the "
                "Gaussian models and NOT comparable to the _cp / _cp_nr families, only to "
                "other _raw runs. Cross-check log1p(mu) under *_logmean. Calibration and "
                "var_residual_corr[_partial] are on the raw-count scale")
    if spec.target_source == "X":
        if target_transform and target_transform != "none":
            return (f"per-gene standardized log-CP10k ({target_transform}, per-gene/"
                    f"per-fold; per-(slide,gene) metrics affine-invariant -> comparable to X)")
        return "log1p(CP10k) = adata.X (Gaussian/MSE)"
    return f"UNDECLARED metric scale for target_source={spec.target_source!r}"


def _environment(run_duration_s):
    """Recompute-time environment. The training-time timing and environment cannot be
    recovered from the predictions; they stay in the metrics.stub.json backup."""
    import torch
    return {
        "recomputed_posthoc": True,
        "recomputed_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "recompute_duration_seconds": round(run_duration_s, 3),
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "device": "cpu",
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "torch_version": torch.__version__,
        "note": ("metrics recomputed offline from predictions/ (compute_final_metrics=False "
                 "run); original run-time timing/environment is in metrics.stub.json"),
    }


# ── iterate a run's folds -> per-split result dicts + fold identity ──────────────────────
def load_gstar_companion(pred_file, ad):
    """Per-split precomputed g* for one predictions file -> {split: (n_split, G) array}.

    Returns {} when there is no companion; recompute_result then computes g* itself, so a
    run whose g* was never precomputed still completes correctly, just slowly.

    The companion is `gstar_fold{k}.h5ad`, written next to `preds_fold{k}.h5ad` by
    epistemic/DE_gstar_precompute.py with X = g* and `obs`/`var` copied verbatim from the
    rows it kept. It may hold fewer splits than the predictions file: that script defaults
    to test-only and needs `--splits val,test` for a classic run. A split it does not cover
    is absent from the returned dict and falls back.

    Rows are aligned by obs_name, never by position: the companion is a subset of the
    predictions rows, so trusting order would silently mis-pair spots the day the mask or
    the split order changes. Genes are checked, not aligned: a mismatch means the companion
    belongs to a different run, so it aborts instead of reindexing.
    """
    gpath = pred_file.parent / pred_file.name.replace("preds_fold", "gstar_fold")
    if gpath == pred_file or not gpath.is_file():
        return {}
    G = sc.read_h5ad(gpath)
    if list(G.var_names) != list(ad.var_names):
        raise ValueError(f"{gpath}: gene panel differs from {pred_file.name} "
                         f"({G.n_vars} vs {ad.n_vars} genes) -- not this run's companion.")
    have = set(G.obs_names)
    out = {}
    for split in ("val", "test"):
        names = ad.obs_names[ad.obs["split"] == split]
        if len(names) and set(names).issubset(have):
            out[split] = _dense(G[list(names)].X)      # reindexed into the slice's own order
    return out


def iter_folds(pred_files, spec, corr_metrics, adata_path=None):
    """Yield (fold, inner_split, test_donor, {split: result}, gene_names, gene_role,
    depth_scale) per predictions file. gene_role is None for classic runs (no
    var['gene_role']); depth_scale is None for every non-count spec.
    `adata_path` is only used as a fallback to rebuild `y` for predictions files that
    carry no `layers["y"]` (the GCM test-only files); see recompute_result."""
    is_count = spec.target_source in ("raw_counts", "cp_counts", "cp_counts_nr")
    for f in pred_files:
        ad = sc.read_h5ad(f)
        depth_scale = resolve_count_scale(ad, f) if is_count else None
        gene_names = list(ad.var_names)
        gene_role = (ad.var["gene_role"].to_numpy() if "gene_role" in ad.var else None)
        fold = int(ad.obs["fold"].iloc[0])
        inner_i = int(ad.obs["inner_split"].iloc[0]) if "inner_split" in ad.obs else 0
        test_donor = str(ad.obs["test_donor"].iloc[0]) if "test_donor" in ad.obs else str(fold)
        # Precomputed g*, per split, when a companion file sits next to this one (count
        # specs only, nothing else has a g*). Absent => recomputed inline.
        gstars = load_gstar_companion(f, ad) if is_count else {}
        if gstars:
            print(f"  {f.name}: using precomputed g* for split(s) "
                  f"{', '.join(sorted(gstars))}", flush=True)
        results = {}
        for split in ("val", "test"):
            sub = ad[ad.obs["split"] == split]
            if sub.n_obs == 0:
                continue
            results[split] = recompute_result(sub, spec, corr_metrics, adata_path=adata_path,
                                              depth_scale=depth_scale,
                                              gstar=gstars.get(split))
        yield fold, inner_i, test_donor, results, gene_names, gene_role, depth_scale


# ── CLASSIC ─────────────────────────────────────────────────────────────────────────────
def complete_classic(run_dir, stub, splits, spec, pred_files, t0):
    # Robust to OLD lean stubs (pre-enrichment) that may lack corr_metrics.
    corr_metrics = tuple(stub.get("corr_metrics") or (stub.get("primary_corr_metric", "pearson"),))
    primary = stub.get("primary_corr_metric", corr_metrics[0])
    lopo = bool(stub.get("lopo", splits.get("lopo", False)))
    adata_path = stub.get("input_adata")

    # Per-fold training facts the stub carries (best_epoch / n_epochs_ran / stop_reason /
    # the two slide lists, written by train.py's deferred branch). Nothing in the saved
    # predictions can reproduce them, so they are merged into the rows rebuilt below with
    # setdefault: a recomputed value is never overwritten by a stub one. Keyed by
    # (fold, inner_split) rather than by position, because the stub's order is the training
    # order and the rebuilt order is the sorted preds_fold* order, which need not agree. A
    # stub without per_fold yields an empty map and nothing is merged.
    stub_fold_rows = {(r.get("fold"), r.get("inner_split", 0)): r
                      for r in (stub.get("per_fold") or []) if isinstance(r, dict)}

    atom_frames, fold_rows, gene_names, depth_scale = [], [], None, None
    for fold, inner_i, test_donor, results, g_names, _, depth_scale in iter_folds(
            pred_files, spec, corr_metrics, adata_path=adata_path):
        gene_names = g_names
        for split, res in results.items():
            af = build_atoms(res, fold, split, gene_names, corr_metrics)
            af["inner_split"] = inner_i
            af["test_donor"] = test_donor
            atom_frames.append(af)
        val_res, test_res = results.get("val", {}), results.get("test", {})
        row = {"fold": fold, "inner_split": inner_i, "test_donor": test_donor,
               "val_loss": np.nan, "test_loss": np.nan}   # NLL not recomputed (see docstring)
        for split, res in (("val", val_res), ("test", test_res)):
            row[f"{split}_ece_pit"] = res.get("ece_pit")
            row[f"{split}_pit_bias"] = res.get("pit_bias")
            row[f"{split}_disp_signed"] = res.get("disp_signed")
            row[f"{split}_var_residual_corr_partial"] = res.get("var_residual_corr_partial")
            row[f"{split}_mae_median"] = res.get("mae_median")
            row[f"{split}_mse_median"] = res.get("mse_median")
        for cm in corr_metrics:
            row[f"val_{cm}"] = val_res.get(cm)
            row[f"test_{cm}"] = test_res.get(cm)
            if f"{cm}_logmean" in val_res:                # count models (option A cross-check)
                row[f"val_{cm}_logmean"] = val_res.get(f"{cm}_logmean")
                row[f"test_{cm}_logmean"] = test_res.get(f"{cm}_logmean")
        for k, v in stub_fold_rows.get((fold, inner_i), {}).items():
            row.setdefault(k, v)          # training-only keys; recomputed ones already set
        fold_rows.append(row)

    atoms_df = pd.concat(atom_frames, ignore_index=True)
    fold_df = pd.DataFrame(fold_rows)

    def mean_std(col):
        s = pd.to_numeric(fold_df[col], errors="coerce") if col in fold_df else pd.Series([np.nan])
        return float(s.mean()), float(s.std())

    nan_summary = {}
    for split in ("val", "test"):
        sub = atoms_df[atoms_df.split == split]
        nan_rows = sub[sub[primary].isna()]
        nan_summary[split] = {
            "n_nan": int(len(nan_rows)), "n_total": int(len(sub)),
            "examples of NaN values": nan_rows[["slide", "gene"]].head(10).to_dict("records"),
        }

    aggs = {}
    for cm in corr_metrics:
        aggs[f"val_{cm}_mean"], aggs[f"val_{cm}_std"] = mean_std(f"val_{cm}")
        aggs[f"test_{cm}_mean"], aggs[f"test_{cm}_std"] = mean_std(f"test_{cm}")
        if f"val_{cm}_logmean" in fold_df.columns:
            aggs[f"val_{cm}_logmean_mean"], aggs[f"val_{cm}_logmean_std"] = mean_std(f"val_{cm}_logmean")
            aggs[f"test_{cm}_logmean_mean"], aggs[f"test_{cm}_logmean_std"] = mean_std(f"test_{cm}_logmean")

    lopo_stats = None
    if lopo:
        lopo_stats = {}
        for cm in corr_metrics:
            per_donor = fold_df.groupby("test_donor")[f"test_{cm}"].mean()
            lopo_stats[cm] = lopo_block(per_donor)
        aggs[f"lopo_{primary}_mean"] = lopo_stats[primary]["mean"]
        aggs[f"lopo_{primary}_sd"] = lopo_stats[primary]["sd"]
        aggs[f"lopo_{primary}_sem"] = lopo_stats[primary]["sem"]

    # The count provenance the recompute ACTUALLY used (overrides whatever the stub said).
    common_depth = stub.get("count_common_depth")
    if common_depth is None and depth_scale is not None:
        common_depth = int(round(depth_scale * 1e4))
    prov = {k: v for k, v in stub.items() if k not in STUB_MARKERS}
    metrics = {
        **prov,
        "metrics_recomputed_posthoc": True,
        "gene_names": gene_names, "n_genes": len(gene_names),
        "loss_function": spec.loss_display_name,
        "depth_scale": depth_scale,
        "count_common_depth": common_depth,
        "metric_scale": metric_scale_string(spec, depth_scale, common_depth,
                                            stub.get("target_transform")),
        "error_metric_scale": ("MAE/MSE on the per-gene Pearson scale (log1p-CP10k; g* for "
                               "count); median over genes; WITHIN-FAMILY only"),
        "environment": _environment((datetime.now(timezone.utc).timestamp()) - t0),
        **aggs,
        "lopo_stats": lopo_stats,
        "val_mae_median_mean": mean_std("val_mae_median")[0],
        "test_mae_median_mean": mean_std("test_mae_median")[0],
        "val_mse_median_mean": mean_std("val_mse_median")[0],
        "test_mse_median_mean": mean_std("test_mse_median")[0],
        "val_ece_pit_mean": mean_std("val_ece_pit")[0],
        "test_ece_pit_mean": mean_std("test_ece_pit")[0],
        "val_disp_signed_mean": mean_std("val_disp_signed")[0],
        "test_disp_signed_mean": mean_std("test_disp_signed")[0],
        "val_pit_bias_mean": mean_std("val_pit_bias")[0],
        "test_pit_bias_mean": mean_std("test_pit_bias")[0],
        "per_fold": fold_rows,
        "nan_summary": nan_summary,
    }
    headline = (lopo_stats[primary]["mean"] if lopo_stats else aggs[f"test_{primary}_mean"])
    return atoms_df, metrics, f"test {primary} = {headline:.4f}"


# ── GCM ─────────────────────────────────────────────────────────────────────────────────
def complete_gcm(run_dir, stub, splits, spec, pred_files, t0):
    # Robust to OLD lean stubs (pre-enrichment) that may lack corr_metrics.
    corr_metrics = tuple(stub.get("corr_metrics") or (stub.get("primary_corr_metric", "pearson"),))
    primary = stub.get("primary_corr_metric", corr_metrics[0])
    lopo = bool(stub.get("lopo", splits.get("lopo", False)))
    adata_path = stub.get("input_adata")

    atom_frames, fold_summaries, gene_names, depth_scale = [], [], None, None
    for fold, inner_i, test_donor, results, g_names, gene_role, depth_scale in iter_folds(
            pred_files, spec, corr_metrics, adata_path=adata_path):
        gene_names = g_names
        if gene_role is None:
            raise ValueError(f"{run_dir}: GCM run but predictions lack var['gene_role'].")
        heldout_mask = gene_role == "test_heldout"          # cross-gene headline
        devgenes_mask = ~heldout_mask                        # CV_GENES = seen + new
        for split, res in results.items():
            af = build_atoms(res, fold, split, gene_names, corr_metrics, gene_role=gene_role)
            af["inner_split"] = inner_i
            af["test_donor"] = test_donor
            atom_frames.append(af)
        val_res, test_res = results.get("val", {}), results.get("test", {})
        ppg = np.asarray(test_res[f"{primary}_per_gene"], dtype=float)
        summ = {
            "fold": fold, "inner_split": inner_i, "test_donor": test_donor,
            f"test_{primary}_heldout_genes": _nm(ppg[heldout_mask].tolist()),
            f"test_{primary}_devgenes": _nm(ppg[devgenes_mask].tolist()),
            "test_mae_median": test_res.get("mae_median"), "val_mae_median": val_res.get("mae_median"),
            "test_mse_median": test_res.get("mse_median"), "val_mse_median": val_res.get("mse_median"),
            "test_ece_pit": test_res.get("ece_pit"), "val_ece_pit": val_res.get("ece_pit"),
            "test_disp_signed": test_res.get("disp_signed"), "val_disp_signed": val_res.get("disp_signed"),
            "test_pit_bias": test_res.get("pit_bias"), "val_pit_bias": val_res.get("pit_bias"),
        }
        if f"{primary}_logmean" in test_res:                # count models
            summ[f"test_{primary}_logmean"] = test_res.get(f"{primary}_logmean")
            summ[f"val_{primary}_logmean"] = val_res.get(f"{primary}_logmean")
        fold_summaries.append(summ)

    atoms_df = pd.concat(atom_frames, ignore_index=True)
    sdf = pd.DataFrame(fold_summaries)

    nan_summary = {}
    for split in ("val", "test"):
        sub = atoms_df[atoms_df.split == split]
        nan_rows = sub[sub[primary].isna()]
        nan_summary[split] = {
            "n_nan": int(len(nan_rows)), "n_total": int(len(sub)),
            "examples of NaN values": nan_rows[["slide", "gene"]].head(10).to_dict("records"),
        }

    lopo_stats = None
    if lopo:
        lopo_stats = {
            "heldout_genes": lopo_block(sdf.groupby("test_donor")[f"test_{primary}_heldout_genes"].mean()),
            "devgenes": lopo_block(sdf.groupby("test_donor")[f"test_{primary}_devgenes"].mean()),
            "primary_corr_metric": primary,
        }

    is_gaussian = spec.is_probabilistic
    has_logmean = any(f"test_{primary}_logmean" in f for f in fold_summaries)
    summary = {
        f"test_{primary}_heldout_genes": _nm([f[f"test_{primary}_heldout_genes"] for f in fold_summaries]),
        f"test_{primary}_heldout_genes_std": _sd([f[f"test_{primary}_heldout_genes"] for f in fold_summaries]),
        f"test_{primary}_devgenes": _nm([f[f"test_{primary}_devgenes"] for f in fold_summaries]),
        f"test_{primary}_devgenes_std": _sd([f[f"test_{primary}_devgenes"] for f in fold_summaries]),
        "mean_test_mae_median": _nm([f["test_mae_median"] for f in fold_summaries]),
        "mean_test_mse_median": _nm([f["test_mse_median"] for f in fold_summaries]),
        "mean_test_ece_pit": _nm([f["test_ece_pit"] for f in fold_summaries]) if is_gaussian else None,
        "mean_test_disp_signed": _nm([f["test_disp_signed"] for f in fold_summaries]) if is_gaussian else None,
    }
    if has_logmean:
        summary[f"test_{primary}_logmean"] = _nm([f.get(f"test_{primary}_logmean") for f in fold_summaries])
        summary[f"val_{primary}_logmean"] = _nm([f.get(f"val_{primary}_logmean") for f in fold_summaries])

    # Gene-split provenance comes from splits.json (always written), not the predictions.
    gs = splits.get("gene_split", {})
    gene_train, gene_val, heldout = gs.get("gene_train", []), gs.get("gene_val", []), gs.get("test_heldout_genes", [])
    gene_split_info = gs.get("info", {})
    # The count provenance the recompute ACTUALLY used (overrides whatever the stub said).
    common_depth = stub.get("count_common_depth")
    if common_depth is None and depth_scale is not None:
        common_depth = int(round(depth_scale * 1e4))
    prov = {k: v for k, v in stub.items() if k not in STUB_MARKERS}
    metrics = {
        **prov,
        "metrics_recomputed_posthoc": True,
        "loss_function": spec.loss_display_name,
        "depth_scale": depth_scale,
        "count_common_depth": common_depth,
        "metric_scale": metric_scale_string(spec, depth_scale, common_depth,
                                            stub.get("target_transform")),
        "error_metric_scale": ("MAE/MSE on the per-gene Pearson scale (log1p-CP10k; g* for "
                               "count); median over genes; WITHIN-FAMILY only"),
        "environment": _environment((datetime.now(timezone.utc).timestamp()) - t0),
        "gene_split_info": gene_split_info,
        "genes": {
            "GENE_TRAIN": gene_train, "GENE_VAL": gene_val,
            "CV_GENES": list(gene_train) + list(gene_val), "TEST_HELDOUT_GENES": heldout,
            "n_gene_train": len(gene_train), "n_gene_val": len(gene_val),
            "n_cv_genes": len(gene_train) + len(gene_val), "n_test_heldout_genes": len(heldout),
        },
        "per_fold": fold_summaries,
        "lopo_stats": lopo_stats,
        "summary": summary,
        "nan_summary": nan_summary,
    }
    headline = (lopo_stats["heldout_genes"]["mean"] if lopo_stats
                else summary[f"test_{primary}_heldout_genes"])
    return atoms_df, metrics, f"test {primary} (heldout genes) = {headline:.4f}"


# ── driver ──────────────────────────────────────────────────────────────────────────────
def complete_one(run_dir, force=False, dry_run=False):
    run_dir = Path(run_dir)
    mpath = run_dir / "metrics.json"
    if not mpath.exists():
        print(f"[skip] {run_dir}: no metrics.json"); return
    stub = json.loads(mpath.read_text())
    if not stub.get("metrics_deferred") and not force:
        print(f"[skip] {run_dir}: metrics.json already complete (use --force to redo)"); return
    if stub.get("saved_predictions") is False:
        print(f"[skip] {run_dir}: run saved no predictions (nothing to recompute from)"); return
    pred_files = sorted((run_dir / "predictions").glob("preds_fold*.h5ad"))
    if not pred_files:
        print(f"[skip] {run_dir}: no predictions/preds_fold*.h5ad on disk"); return

    is_gcm = ("results_gcm" in str(run_dir).replace("\\", "/")) or ("run_id_components" in stub)
    model_type = stub["model_type"]
    spec = get_spec(model_type)
    splits_path = run_dir / "splits.json"
    splits = json.loads(splits_path.read_text()) if splits_path.exists() else {}

    kind = "GCM" if is_gcm else "classic"
    print(f"[run ] {run_dir}  ({kind}, {model_type}, {len(pred_files)} folds)"
          + ("  [dry-run]" if dry_run else ""))
    if dry_run:
        return

    t0 = datetime.now(timezone.utc).timestamp()
    fn = complete_gcm if is_gcm else complete_classic
    atoms_df, metrics, headline = fn(run_dir, stub, splits, spec, pred_files, t0)

    # Back up the stub, then write atoms.parquet + the completed metrics.json in place.
    if not (run_dir / "metrics.stub.json").exists():
        (run_dir / "metrics.stub.json").write_text(json.dumps(stub, indent=2))
    atoms_df.to_parquet(run_dir / "atoms.parquet", index=False)
    mpath.write_text(json.dumps(metrics, indent=2, default=_json_default))
    dt = datetime.now(timezone.utc).timestamp() - t0
    print(f"[done] {run_dir}  {headline}  ({len(atoms_df)} atoms, {dt/60:.1f} min)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="*", help="run directories to complete")
    ap.add_argument("--glob", help="glob of run dirs (e.g. 'results_gcm/*/*/*/*/*')")
    ap.add_argument("--force", action="store_true",
                    help="recompute even if metrics.json is already complete")
    ap.add_argument("--dry-run", action="store_true",
                    help="list what would be completed, write nothing")
    args = ap.parse_args()

    dirs = list(args.run_dirs)
    if args.glob:
        dirs += [d for d in globmod.glob(args.glob) if os.path.isdir(d)]
    if not dirs:
        ap.error("give one or more run_dirs, or --glob")
    for d in sorted(set(dirs)):
        complete_one(d, force=args.force, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
