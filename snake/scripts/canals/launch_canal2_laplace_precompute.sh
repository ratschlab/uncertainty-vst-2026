#!/usr/bin/env bash
# snake/scripts/canals/launch_canal2_laplace_precompute.sh
#
# Submit canal-2 Laplace precompute jobs (canal2_uncertainty_precompute.py --laplace-dir) for
# the classic runs that carry last-layer Laplace summaries (<run-dir>/laplace/test_summaries.h5ad,
# written by snake/scripts/epistemic/laplace_posthoc.py). One SLURM job per (dataset, model),
# CPU-only, all submitted at once.
#
# It has its own launcher rather than a channel arm of the MC-dropout one because the project
# convention is one launcher per epistemic method, so a rerun of one never re-submits the other.
#
# Input layout. The launcher points at a single training sweep (RUN_ID below), so the
# own-head aleatoric, the MC-dropout and the Laplace estimates all come off the same checkpoints
# and a difference between them is the estimator, not the training run. The Gaussian heads are
# taken at the lambda_0.0 leaf, the proper-NLL baseline.
#
# Usage (cluster login node, repo root):
#   bash snake/scripts/canals/launch_canal2_laplace_precompute.sh                    # epistemic channel (default)
#   CHANNEL=total     bash snake/scripts/canals/launch_canal2_laplace_precompute.sh  # aleatoric+epistemic (like-for-like vs a single model's own head)
#   CHANNEL=aleatoric bash snake/scripts/canals/launch_canal2_laplace_precompute.sh  # the aleatoric ingredient alone (same mu/checkpoint contrast)
#   DRYRUN=1 bash snake/scripts/canals/launch_canal2_laplace_precompute.sh           # preview only, no sbatch
#   FORCE=1  bash snake/scripts/canals/launch_canal2_laplace_precompute.sh           # resubmit even already-done
set -euo pipefail

# ── Knobs (env overrides) ─────────────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
RESULTS_BASE="${RESULTS_BASE:-$PROJECT_ROOT/results}"
RUN_ID="${RUN_ID:-pr_classic3}"      # run_id_classic segment inside results/{dataset}/
RADIUS="${RADIUS:-r75}"
# CHANNEL = epistemic (default) | aleatoric | total. All three read the SAME
# laplace/test_summaries.h5ad (--uncertainty-source under the hood). The out-subdir is the
# script's own default for that channel (canal2_laplace_<source>/), kept distinct from the
# MC-dropout subdirs so both estimators coexist in one run dir.
CHANNEL="${CHANNEL:-epistemic}"
case "$CHANNEL" in
    epistemic|aleatoric|total) ;;
    *) echo "CHANNEL must be 'epistemic', 'aleatoric', or 'total' (got '$CHANNEL')"; exit 1 ;;
esac
OUT_SUBDIR="${OUT_SUBDIR:-canal2_laplace_$CHANNEL}"   # must match the script's --out-dir default
JOB_PREFIX="canal2lap_${CHANNEL:0:3}"
FORCE="${FORCE:-0}"                            # 1 = resubmit even if summary.json already exists
DATASETS="${DATASETS:-10x_TuPro TLS LUNG_CANCER}"
# The heads to submit. The leaf carries the lambda_ segment for the Gaussian family and nothing
# for the count family, exactly as the results tree stores it. The *_raw pair is included: it is
# measurable, it just reports on the raw-count axis, so downstream it belongs in its own table
# and never on a shared axis with the _cp and Gaussian heads.
MODELS_DEFAULT=(
    "heteroscedastic/lambda_0.0"
    "homoscedastic/lambda_0.0"
    "spot_homoscedastic/lambda_0.0"
    "poisson_cp"
    "negativebinomial_hetero_noanchor_cp"
    "poisson_raw"
    "negativebinomial_hetero_noanchor_raw"
)
# MODELS="a/lambda_0.0 b" to restrict; defaults to the array above.
if [[ -n "${MODELS:-}" ]]; then read -r -a MODELS_ARR <<< "$MODELS"; else MODELS_ARR=("${MODELS_DEFAULT[@]}"); fi
CONDA_ENV="${CONDA_ENV:-mtenv0}"
CONDA_SH="${CONDA_SH:-$(conda info --base)/etc/profile.d/conda.sh}"
DRYRUN="${DRYRUN:-0}"
# CPU-only: passing no -p and no --gres puts the job on the compute partition. The heaviest step
# is the whole-panel linear pass plus the long per-spot table, which is what the memory request
# below is sized for.
CPUS="${CPUS:-4}"; MEM="${MEM:-128000}"; TIME="${TIME:-04:00:00}"

LOGDIR="$PROJECT_ROOT/snake/logs"
mkdir -p "$LOGDIR"

submit() {
    local run_dir="$1"
    local tag; tag="$(echo "${run_dir#"$RESULTS_BASE"/}" | tr '/' '_')"

    if [[ "$FORCE" != "1" && -f "$run_dir/$OUT_SUBDIR/summary.json" ]]; then
        echo "skip (already done): $run_dir"
        return
    fi
    if [[ "$DRYRUN" == "1" ]]; then
        echo "WOULD SUBMIT  $run_dir"
        return
    fi

    sbatch --job-name="${JOB_PREFIX}_${tag}" --time="$TIME" -c "$CPUS" --mem="${MEM}M" \
        --output="$LOGDIR/${JOB_PREFIX}_${tag}_%j.out" \
        --error="$LOGDIR/${JOB_PREFIX}_${tag}_%j.err" \
        --wrap ". '$CONDA_SH' && conda activate $CONDA_ENV && cd '$PROJECT_ROOT' || exit 1; python -u snake/scripts/canals/canal2_uncertainty_precompute.py --laplace-dir '$run_dir' --uncertainty-source '$CHANNEL'"
    echo "submitted     $run_dir"
}

n=0
for ds in $DATASETS; do
    for leaf in "${MODELS_ARR[@]}"; do
        run_dir="$RESULTS_BASE/$ds/$RUN_ID/_${ds}_hvg_top3000/$RADIUS/$leaf"
        if [[ ! -f "$run_dir/laplace/test_summaries.h5ad" ]]; then
            echo "skip (no Laplace summaries): $run_dir/laplace/test_summaries.h5ad"
            continue
        fi
        if [[ ! -f "$run_dir/metrics.json" ]]; then
            echo "skip (no metrics.json): $run_dir"
            continue
        fi
        submit "$run_dir"; n=$((n+1))
    done
done
echo "------"
echo "${DRYRUN:+(dry-run) }matched $n run-dir(s)  [RESULTS_BASE=$RESULTS_BASE RUN_ID=$RUN_ID DATASETS='$DATASETS' CHANNEL=$CHANNEL]"
