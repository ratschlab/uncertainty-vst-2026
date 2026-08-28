# snake/scripts/epistemic/DE_gstar_precompute.py
#
# Compute g* = E[log1p(Y/depth_scale)] for ONE (seed, fold) prediction file of a count model.
#
# Reads : one preds_fold{k}.h5ad (--pred-file), plus the metrics.json of the run dir holding it.
# Writes: a companion .h5ad, by default gstar_fold{k}.h5ad in the same predictions/ directory.
#
# It is its own step because g* is the one expensive quantity of the count post-hoc chain, a
# windowed recurrence over the NB/Poisson PMF (src/evaluation/count_metrics.py). One prediction
# file is the natural unit: each is independent and already on disk, so the sweep is N small
# concurrent jobs. deep_ensemble_combine.py then only averages the M per-seed g* arrays, which is
# exact: a deep ensemble's predictive law is the equal-weight mixture of its members, and g* is an
# expectation under that law, so mean_s(g*_s) IS the mixture's g*. The mixture is never built.
#
# ── Scale ────────────────────────────────────────────────────────────────────────────────────
# A count model fits its likelihood directly on its own target, so a prediction file already holds
# everything on ONE scale: X = mu, layers["y"] = that same target, layers["dispersion"] = alpha
# fitted at that same scale. mu is used as-is; nothing is multiplied by anything.
#
# `depth_scale` is the divisor applied before log1p when reporting: one float per run, read from
# uns["depth_scale"]. It is `count_common_depth / 1e4` for the cp targets (2.5 at a common depth of
# 25000, 1.0 at 10000) and None for raw counts, which are aligned to no depth. None is resolved to
# a divisor of 1.0 once, inside count_expected_log1p, never here. Reporting axes follow:
# cp_counts and cp_counts_nr land on log1p(CP10k), comparable to the Gaussian models; raw_counts
# lands on log1p(raw counts), comparable only to other *_raw runs.
#
# A file without uns["count_convention"] = "v2_no_offset" is a hard failure: read under this
# convention it would emit numbers on an unknown scale.
#
# Rounded (cp_counts) and continuous (cp_counts_nr) targets are treated alike: count_expected_log1p
# sums the PMF over integer k either way, so g* is the same quantity count_metrics_final reports.
#
# ── Output ───────────────────────────────────────────────────────────────────────────────────
# ONE companion .h5ad per input, restricted to --splits (default "test"). Default path: alongside
# the input, preds_fold{k}.h5ad -> gstar_fold{k}.h5ad. Schema:
#   X                        = g* (float32), same row and gene order as the input
#   obs                      = a copy of the input's obs for those rows, so the file is alignable
#                              by position or by obs_name / adata_row
#   var                      = a copy of the input's var
#   uns["gstar_precompute"]  = source_pred_file, depth_scale (omitted when None),
#                              count_convention, target_source, residual_mass_max
# Written with compression="lzf", like train.py's predictions writer.
#
# The script reads ONE prediction file and computes that file's own g*, so it applies to a single
# non-ensemble run as well, classic or GCM. The DE_ prefix names the usual caller.
#
# Usage (repo root, on the cluster; one job per (seed, fold) prediction file, meant to be launched
# as many small, independent, concurrent jobs. Prefer a GPU job: count_expected_log1p is much
# slower on CPU):
#   python -u snake/scripts/epistemic/DE_gstar_precompute.py \
#       --pred-file results/<dataset>/<run_id>/<gene_set>/r75/STI_nb_cp/seed_0/predictions/preds_fold0.h5ad
#   python -u snake/scripts/epistemic/DE_gstar_precompute.py \
#       --pred-file .../seed_0/predictions/preds_fold0.h5ad \
#       --out-file /custom/path/gstar_fold0.h5ad

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import scanpy as sc
import anndata as ad
import torch

_T0 = time.time()
def _el():
    return time.time() - _T0

# Project root on sys.path (the repository is not necessarily pip-installed) -- same shim
# every other snake/scripts/*.py uses.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(PROJECT_ROOT)
from src.models.specs import get_spec                     # noqa: E402
from src.evaluation.count_metrics import count_expected_log1p  # noqa: E402

# The count target_sources this script reports g* for. A target_source absent from this tuple is
# rejected rather than reported on an unknown scale.
#
SUPPORTED_TARGET_SOURCES = ("cp_counts", "cp_counts_nr", "raw_counts")

# The marker every offline reader keys off to confirm a prediction file's scale convention.
COUNT_CONVENTION = "v2_no_offset"


def _dense(a):
    """Predictions matrices are dense in this pipeline, but be robust to a sparse layer
    (mirrors complete_deferred_metrics.py's own `_dense` helper)."""
    return np.asarray(a.todense() if hasattr(a, "todense") else a, dtype=np.float64)


def _resolve_scale(a_test, pred_file, target_source):
    """The g* reporting divisor of this prediction file: float, or None for the depth-blind
    target "raw_counts".

    X is used as-is (it IS mu on the model's own fitted target scale), so this reads exactly one
    thing: uns["depth_scale"], gated on the convention marker.
    """
    if a_test.uns.get("count_convention") != COUNT_CONVENTION:
        raise ValueError(
            f"{pred_file}: artifact with no count_convention marker. "
            f"Re-run the precompute."
        )
    depth_scale = a_test.uns.get("depth_scale")     # absent => None (raw_counts)
    if depth_scale is None and target_source in ("cp_counts", "cp_counts_nr"):
        raise ValueError(
            f"{pred_file}: predictions file carries count_convention={COUNT_CONVENTION!r} but "
            f"no uns['depth_scale'], and target_source={target_source!r} REQUIRES one "
            f"(count_common_depth / 1e4, e.g. 2.5 on 10x_TuPro). Refusing to assume 1.0, "
            f"which would silently mis-scale every reported g* on a non-1e4 dataset."
        )
    return None if depth_scale is None else float(depth_scale)


# -- Args ---------------------------------------------------------------------------------------
# No description=__doc__: this file's header is a `#`-comment block, not a module docstring, so
# __doc__ is None.
p = argparse.ArgumentParser()
p.add_argument("--pred-file", required=True,
              help="One seed/fold predictions file, e.g. .../seed_0/predictions/preds_fold0.h5ad.")
p.add_argument("--out-file", default=None,
              help="Output path. Default: sibling gstar_fold{k}.h5ad next to --pred-file "
                   "(preds_fold{k}.h5ad -> gstar_fold{k}.h5ad, same predictions/ directory).")
p.add_argument("--device", default=None, help="cuda|cpu (default: auto). count_expected_log1p's "
              "own docstring documents CUDA float32 marching as ~32x the CPU float64 path's "
              "speed -- matches the device the PRODUCTION callers use (inference_indexed.py /"
              "train_loop_gcm.py pass the live GPU-resident SharedTensors' device); this "
              "precompute script has no live tensors to inherit a device from (it's a fresh "
              "process reading one .h5ad off disk), so it auto-detects instead.")
p.add_argument("--splits", default="test",
              help="Comma-separated obs['split'] values to keep. DEFAULT 'test' -- the deep-ensemble "
                   "convention, which deep_ensemble_combine.py relies on row-for-row; do not change "
                   "it for ensemble runs. Pass 'val,test' for a CLASSIC deferred-metrics run, whose "
                   "preds_fold{k}.h5ad holds both splits and whose metrics.json reports both.")
args = p.parse_args()

DEVICE = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
pred_file = os.path.abspath(args.pred_file)
pred_dir = os.path.dirname(pred_file)              # .../predictions
run_dir = os.path.dirname(pred_dir)                 # the seed dir: sibling of predictions/, holds metrics.json

if args.out_file is not None:
    out_file = os.path.abspath(args.out_file)
else:
    out_file = os.path.join(pred_dir, os.path.basename(pred_file).replace("preds_", "gstar_", 1))

print(f"[{_el():7.1f}s] [run ] pred_file={pred_file}", flush=True)
print(f"[{_el():7.1f}s]       out_file ={out_file}", flush=True)
print(f"[{_el():7.1f}s]       device   ={DEVICE}", flush=True)

# -- model_type / target_source -- mirrors complete_deferred_metrics.py::complete_one() ----------
# Read the run's own metrics.json for model_type (present even in a compute_final_metrics=False
# STUB), then resolve target_source from the SPEC (the single source of truth), not from
# metrics.json directly -- the STUB does not carry target_source at all.
metrics_path = os.path.join(run_dir, "metrics.json")
with open(metrics_path) as f:
    metrics_stub = json.load(f)
model_type = metrics_stub["model_type"]
spec = get_spec(model_type)
target_source = spec.target_source
if target_source not in SUPPORTED_TARGET_SOURCES:
    raise ValueError(
        f"{run_dir}: model_type={model_type!r} has target_source={target_source!r}, not in "
        f"{SUPPORTED_TARGET_SOURCES} -- DE_gstar_precompute.py only handles the count targets "
        f"listed there; see this file's header SCALE block."
    )
want_disp = "dispersion" in spec.extra_prediction_columns   # NB (alpha) vs Poisson (alpha=None)
print(f"[{_el():7.1f}s] model_type={model_type} target_source={target_source} "
      f"family={'NB' if want_disp else 'Poisson'}", flush=True)

# -- Load predictions, restrict to the requested splits ------------------------------------------
# The default --splits test matches deep_ensemble_combine.py::combine_test_fold, which reads the
# same preds_fold{k}.h5ad with `a.obs["split"] == "test"`. This file's rows must line up with that
# one row for row, so do not change the default: the deep-ensemble chain depends on it.
#
# --splits val,test serves the classic deferred-metrics chain, whose preds_fold{k}.h5ad stacks val
# AND test and whose metrics.json reports both. Restricting to test there leaves the val headline
# with no precomputed g*, forcing a partial recompute. Subsetting carries uns over, so the marker
# and depth_scale stay readable off a_test.
a = sc.read_h5ad(pred_file)
want_splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
if "split" in a.obs.columns:
    test_mask = np.isin(a.obs["split"].to_numpy(), want_splits)
    if not test_mask.any():
        raise SystemExit(f"{pred_file}: no rows with split in {want_splits} "
                         f"(present: {sorted(set(a.obs['split'].to_numpy()))})")
    a_test = a[test_mask].copy()
else:
    print(f"[{_el():7.1f}s] NOTE: no obs['split'] column in {pred_file} -- treating every row "
          f"as test (classic predictions files always carry 'split'; this is a defensive "
          f"fallback, not the expected path).", flush=True)
    a_test = a.copy()
n_test = a_test.n_obs
print(f"[{_el():7.1f}s] {n_test}/{a.n_obs} rows kept (splits={','.join(want_splits)})", flush=True)

means = _dense(a_test.X)                              # (n_test, G) predictive mean, float64
# g* is a property of the fitted predictive distribution alone: count_expected_log1p takes mu and
# alpha and has no y term. y is loaded only as a sanity check. train.py::fold_predictions_adata
# writes layers["y"] straight from read_arrays()'s own target array, the model's own count target
# on the same scale as X, so a non-negative check catches a corrupted or mis-scaled predictions
# file before it can silently feed a wrong g*.
y = _dense(a_test.layers["y"])                        # (n_test, G) the model's own training target
if np.nanmin(y) < 0:
    raise ValueError(f"{pred_file}: layers['y'] has negative values (min={np.nanmin(y):.3g}) -- "
                     f"not a valid count-scale target for target_source={target_source!r}.")
if want_disp:
    if "dispersion" not in a_test.layers:
        raise KeyError(f"{pred_file}: spec for {model_type!r} declares 'dispersion' but the "
                       f"predictions file has no layers['dispersion'].")
    # alpha is fitted at the SAME scale as mu (mean and dispersion always share one scale), so
    # like mu it is used as-is: no rescaling anywhere.
    alpha = _dense(a_test.layers["dispersion"])        # (n_test, G) NB dispersion, float64
else:
    alpha = None
G = means.shape[1]

# -- depth_scale -- see the SCALE block in the header --------------------------------------------
depth_scale = _resolve_scale(a_test, pred_file, target_source)
print(f"[{_el():7.1f}s] depth_scale={depth_scale}", flush=True)

# -- g* = E[log1p(Y/depth_scale)] -- the expensive windowed truncated-sum recurrence, once -------
t_g = time.time()
gstar, residual_mass = count_expected_log1p(means, depth_scale, alpha=alpha, device=DEVICE)
print(f"[{_el():7.1f}s] g* computed: {n_test} test spots x {G} genes in "
      f"{time.time() - t_g:.1f}s (residual_mass={residual_mass:.2e})", flush=True)
if residual_mass > 1e-4:
    # Mirrors count_metrics_final's own diagnostic warning (src/evaluation/count_metrics.py) --
    # informational only, does not gate the job or invalidate the saved g* (see the truncation
    # note in src/evaluation/count_metrics.py's own docstring: residual is self-normalized away).
    print(f"[gstar_precompute] WARNING: g* residual tail mass {residual_mass:.2e} (>1e-4) -- "
          f"the windowed recurrence hit its k_cap safety bound before the tail closed for at "
          f"least one (spot, gene) cell. Diagnostic only.", flush=True)

# -- Write the companion .h5ad -------------------------------------------------------------------
out = ad.AnnData(
    X=gstar.astype(np.float32),
    obs=a_test.obs.copy(),          # every column verbatim: obs_name/adata_row/slide/split/fold/...
    var=a_test.var.copy(),          # var index == a_test.var_names -> same gene order preserved
)
prov = {
    "source_pred_file": pred_file,
    "count_convention": COUNT_CONVENTION,        # verified present by _resolve_scale
    "target_source": target_source,
    "residual_mass_max": float(residual_mass),
    # Harmless extra provenance beyond the required contract fields above.
    "model_type": model_type,
    "n_test_spots": int(n_test),
    "n_genes": int(G),
    "computed_at": datetime.now(timezone.utc).astimezone().isoformat(),
}
if depth_scale is not None:
    # Written only when not None: h5ad uns cannot serialize a bare Python None. Readers must use
    # .get("depth_scale"), which correctly yields None for the depth-blind targets.
    prov["depth_scale"] = float(depth_scale)
out.uns["gstar_precompute"] = prov

out_file_dir = os.path.dirname(out_file)
if out_file_dir:                      # guard a bare relative filename (empty dirname)
    os.makedirs(out_file_dir, exist_ok=True)
t_w = time.time()
out.write_h5ad(out_file, compression="lzf")
print(f"[{_el():7.1f}s] [ok  ] wrote {out_file} ({time.time() - t_w:.1f}s)", flush=True)
