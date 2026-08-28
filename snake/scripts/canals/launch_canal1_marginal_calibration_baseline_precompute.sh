#!/usr/bin/env bash
# snake/scripts/canals/launch_canal1_marginal_calibration_baseline_precompute.sh
#
# Submit the marginal, no-image calibration baseline precompute
# (canal1_marginal_calibration_baseline_precompute.py). One SLURM job per dataset, not per run:
# the baseline is model-independent and uses one cp reference run per dataset only to recover the
# folds, the gene panel, the adata and depth_scale. CPU-only.
#
# Each job fits the three family marginals (gaussian, poisson, nb) per LOPO fold and caches
#   {out}/canal1_marginal_baseline/{dataset}.parquet (+ .json)
# next to the other _analysis precomputes.
#
# Usage (cluster login node, repo root):
#   bash snake/scripts/canals/launch_canal1_marginal_calibration_baseline_precompute.sh            # submit missing datasets
#   DRYRUN=1 bash snake/scripts/canals/launch_canal1_marginal_calibration_baseline_precompute.sh   # preview only
#   FORCE=1  bash snake/scripts/canals/launch_canal1_marginal_calibration_baseline_precompute.sh   # --overwrite + resubmit
set -euo pipefail

# ── Knobs (env overrides) ─────────────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PROJ="${PROJ:-$PROJECT_ROOT}"
RUN_ID="${RUN_ID:-pr_classic3}"
RADIUS="${RADIUS:-r75}"
DATASETS="${DATASETS:-10x_TuPro TLS LUNG_CANCER}"
OUT_DIR="${OUT_DIR:-$PROJ/results/_analysis/canal1_marginal_baseline}"
CONDA_ENV="${CONDA_ENV:-mtenv0}"                  # env with scipy + anndata + src
CONDA_SH="${CONDA_SH:-$(conda info --base)/etc/profile.d/conda.sh}"
DRYRUN="${DRYRUN:-0}"
FORCE="${FORCE:-0}"

# CPU-only: no -p/--gres is passed. The job reads the merged preprocessed adata once, an all-gene
# transient, then runs a per-(slide, gene) calibration loop over 3 families x K folds.
CPUS="${CPUS:-4}"; MEM="${MEM:-64000}"; TIME="${TIME:-01:30:00}"

LOGDIR="$PROJECT_ROOT/snake/logs"
mkdir -p "$LOGDIR"

n=0
for ds in $DATASETS; do
    extra=""
    [[ "$FORCE" == "1" ]] && extra="--overwrite"
    if [[ "$FORCE" != "1" && -f "$OUT_DIR/$ds.parquet" && -f "$OUT_DIR/$ds.json" ]]; then
        echo "skip (already done): $OUT_DIR/$ds.parquet"
        continue
    fi
    if [[ "$DRYRUN" == "1" ]]; then
        echo "WOULD SUBMIT  dataset=$ds  ${extra:+[$extra]}"
        n=$((n+1)); continue
    fi
    sbatch --job-name="canal1_baseline_${ds}" --time="$TIME" -c "$CPUS" --mem="${MEM}M" \
        --output="$LOGDIR/canal1_baseline_${ds}_%j.out" \
        --error="$LOGDIR/canal1_baseline_${ds}_%j.err" \
        --wrap ". '$CONDA_SH' && conda activate $CONDA_ENV && cd '$PROJECT_ROOT' || exit 1; python -u snake/scripts/canals/canal1_marginal_calibration_baseline_precompute.py --dataset '$ds' --proj-root '$PROJ' --run-id '$RUN_ID' --radius '$RADIUS' --out-dir '$OUT_DIR' $extra"
    echo "submitted     dataset=$ds  ${extra:+[$extra]}"
    n=$((n+1))
done
echo "------"
echo "${DRYRUN:+(dry-run) }matched $n dataset(s)  [RUN_ID=$RUN_ID OUT_DIR=$OUT_DIR]"
