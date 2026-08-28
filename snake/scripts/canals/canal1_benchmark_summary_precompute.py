# snake/scripts/canals/canal1_benchmark_summary_precompute.py
#
# Per-run CLI wrapper around src/evaluation/benchmark_summary.py. Writes one model's compact
# per-(held-out donor, gene) performance and calibration table next to the run, so a cross-model
# comparison reads small files instead of the heavy predictions.
#
# ── Two modes, mutually exclusive ────────────────────────────────────────────────────────────
#   --run-dir <dir>       a classic LOPO training run dir -> export_summary(run_dir)
#   --ensemble-dir <dir>  a deep ensemble's '.../ensemble/' dir, holding the test_summaries.h5ad
#                         that epistemic/deep_ensemble_combine.py writes ->
#                         export_summary_ensemble(ensemble_dir)
# What each function computes, and why the ensemble mode reports fewer columns, is in their
# docstrings in src/evaluation/benchmark_summary.py.
#
# Both modes cache benchmark_summary.{parquet,json} in the directory they are given and skip the
# work when those exist; --overwrite forces a recompute.
#
# Classic paths only, in both modes. parse_run_identity() assumes the classic depth
# results/{dataset}/{run_id}/{gene_set}/{radius}/{model_type}/[lambda_{lam}], and an ensemble
# dir's parent sits at exactly that depth. results_gcm/ has no gene_set segment, so the identity
# fields would shift by one level and be wrong with no error raised.
#
# Usage (repo root, on the cluster; CPU only):
#   python -u snake/scripts/canals/canal1_benchmark_summary_precompute.py \
#       --run-dir results/{dataset}/{run_id}/{gene_set}/{radius}/heteroscedastic/lambda_0.0
#   python -u snake/scripts/canals/canal1_benchmark_summary_precompute.py --run-dir ... --overwrite
#   python -u snake/scripts/canals/canal1_benchmark_summary_precompute.py \
#       --ensemble-dir results/{dataset}/{run_id}/{gene_set}/{radius}/{model_type}/ensemble

import argparse
import os
import sys
import time

_T0 = time.time()
def _el():
    return time.time() - _T0

p = argparse.ArgumentParser()
_mode = p.add_mutually_exclusive_group(required=True)
_mode.add_argument("--run-dir", default=None,
               help="A classic run dir (holds metrics.json, atoms.parquet, predictions/).")
_mode.add_argument("--ensemble-dir", default=None,
               help="A deep-ensemble '.../ensemble/' dir (holds test_summaries.h5ad, produced by "
                    "epistemic/deep_ensemble_combine.py with g*/PIT precomputed). See header comment.")
p.add_argument("--overwrite", action="store_true",
               help="Recompute even if benchmark_summary.{parquet,json} already exist.")
args = p.parse_args()

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(PROJECT_ROOT)
from src.evaluation.benchmark_summary import export_summary, export_summary_ensemble

if args.run_dir is not None:
    run_dir = os.path.abspath(args.run_dir)
    print(f"run_dir: {run_dir}", flush=True)
    per_dg, meta = export_summary(run_dir, overwrite=args.overwrite)
    # report_axis and headline_pearson_logmean are echoed because the headline alone is easy to
    # misread. A depth-blind *_raw run's pearson sits on log1p(raw counts), report_axis
    # "log1p_raw_counts", and is not comparable to a _cp or Gaussian one. pearson_logmean is the
    # count-only mean-only estimator of the same quantity, above g* by the Jensen gap; it is None
    # for a Gaussian run, which has a single estimator.
    print(f"[{_el():6.1f}s] done. dataset={meta['dataset']} model_type={meta['model_type']} "
          f"lambda={meta['lambda']} n_genes={meta['n_genes']} n_donors={meta['n_donors']} "
          f"is_hetero={meta['is_hetero']} report_axis={meta['report_axis']} "
          f"headline_pearson={meta['headline'].get('pearson')} "
          f"headline_pearson_logmean={meta['headline'].get('pearson_logmean')}",
          flush=True)
else:
    ensemble_dir = os.path.abspath(args.ensemble_dir)
    print(f"ensemble_dir: {ensemble_dir}", flush=True)
    per_dg, meta = export_summary_ensemble(ensemble_dir, overwrite=args.overwrite)
    # headline_pearson_logmean is the ensemble's mean-only estimator log1p(mu/div), echoed next to
    # g* as in the --run-dir branch. Count families only: a continuous ensemble has one estimator,
    # so the key is absent and this prints None.
    print(f"[{_el():6.1f}s] done. dataset={meta['dataset']} model_type={meta['model_type']} "
          f"lambda={meta['lambda']} n_genes={meta['n_genes']} n_donors={meta['n_donors']} "
          f"is_ensemble={meta['is_ensemble']} n_members={meta['n_members']} "
          f"report_axis={meta['report_axis']} "
          f"headline_pearson={meta['headline'].get('pearson')} "
          f"headline_pearson_logmean={meta['headline'].get('pearson_logmean')}",
          flush=True)
