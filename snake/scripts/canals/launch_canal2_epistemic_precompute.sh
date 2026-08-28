#!/usr/bin/env bash
# snake/scripts/canals/launch_canal2_epistemic_precompute.sh
#
# Submit Canal 2 jobs against MC-dropout post-hoc summaries (canal2_uncertainty_precompute.py
# --mc-dropout-dir) for the models that have them. Reads <run-dir>/mc_dropout/test_summaries.h5ad,
# so snake/scripts/epistemic/mc_dropout_posthoc.py must run first. One SLURM job per
# (dataset, model), CPU-only, all concurrent. CHANNEL picks which uncertainty the job scores.
#
# Targets are listed explicitly in the SPECS array below, one entry per "label:run_id:leaf",
# so entries from different run directories can coexist.
#
# Usage (cluster login node, repo root):
#   bash snake/scripts/canals/launch_canal2_epistemic_precompute.sh                    # epistemic channel (default)
#   CHANNEL=aleatoric bash snake/scripts/canals/launch_canal2_epistemic_precompute.sh  # aleatoric-from-MC (clean contrast, same mu/checkpoint)
#   CHANNEL=total bash snake/scripts/canals/launch_canal2_epistemic_precompute.sh      # aleatoric+epistemic (like-for-like vs a single model's own uncertainty)
#   DRYRUN=1 bash snake/scripts/canals/launch_canal2_epistemic_precompute.sh           # preview only, no sbatch
#   FORCE=1 bash snake/scripts/canals/launch_canal2_epistemic_precompute.sh            # resubmit even already-done
set -euo pipefail

# ── Knobs (env overrides) ─────────────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
RESULTS_BASE="${RESULTS_BASE:-$PROJECT_ROOT/results}"
RADIUS="${RADIUS:-r75}"
# CHANNEL = epistemic (default) | aleatoric | total. All three read the SAME
# mc_dropout/test_summaries.h5ad (--uncertainty-source under the hood); 'aleatoric' gives a clean
# channel-only contrast (same mu/checkpoint), 'total' = aleatoric+epistemic (like-for-like vs a
# single model's own uncertainty head). Sets the out-subdir + the job-name prefix so the three
# channels never collide.
CHANNEL="${CHANNEL:-epistemic}"
case "$CHANNEL" in
    epistemic) OUT_SUBDIR_DEFAULT="canal2_epistemic";    JOB_PREFIX="canal2epi" ;;
    aleatoric) OUT_SUBDIR_DEFAULT="canal2_aleatoric_mc"; JOB_PREFIX="canal2alemc" ;;
    total)     OUT_SUBDIR_DEFAULT="canal2_total_mc";     JOB_PREFIX="canal2totmc" ;;
    *) echo "CHANNEL must be 'epistemic', 'aleatoric', or 'total' (got '$CHANNEL')"; exit 1 ;;
esac
OUT_SUBDIR="${OUT_SUBDIR:-$OUT_SUBDIR_DEFAULT}"   # must match the script's --out-dir default for this channel
FORCE="${FORCE:-0}"                            # 1 = resubmit even if summary.json already exists
DATASETS="${DATASETS:-10x_TuPro TLS LUNG_CANCER}"
# One entry per valid model: "label:run_id:leaf".
#
# Every active entry points at one training arm, and that is the point: it carries both post-hoc
# estimators (mc_dropout/ and laplace/) for the same heads on the same cohorts, so reading the whole
# epistemic column off it makes "own head vs MC-dropout vs Laplace" a comparison of estimators
# rather than of checkpoints. Its Gaussian leaf is lambda_0.0, the proper-NLL baseline.
# The *_raw pair is included: measurable, but on the raw-count axis, so it belongs in its own table
# downstream and never on a shared axis with the _cp / Gaussian heads.
SPECS=(
    "pr_classic3:pr_classic3:heteroscedastic/lambda_0.0"
    "pr_classic3:pr_classic3:homoscedastic/lambda_0.0"
    "pr_classic3:pr_classic3:spot_homoscedastic/lambda_0.0"
    "pr_classic3:pr_classic3:poisson_cp"
    "pr_classic3:pr_classic3:negativebinomial_hetero_noanchor_cp"
    "pr_classic3:pr_classic3:poisson_raw"
    "pr_classic3:pr_classic3:negativebinomial_hetero_noanchor_raw"
)
CONDA_ENV="${CONDA_ENV:-mtenv0}"
CONDA_SH="${CONDA_SH:-$(conda info --base)/etc/profile.d/conda.sh}"
DRYRUN="${DRYRUN:-0}"
# CPU-only: passing no -p/--gres lands the job on the compute partition. Heaviest step is the long
# per-spot table plus the rank-based groupby cascade, and the whole-panel linear-error pass over
# every (spot, gene) entry, zeros included, which is what the memory here is sized for. Bump TIME
# if a run times out.
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
        --wrap ". '$CONDA_SH' && conda activate $CONDA_ENV && cd '$PROJECT_ROOT' || exit 1; python -u snake/scripts/canals/canal2_uncertainty_precompute.py --mc-dropout-dir '$run_dir' --uncertainty-source '$CHANNEL'"
    echo "submitted     $run_dir"
}

n=0
for ds in $DATASETS; do
    for spec in "${SPECS[@]}"; do
        IFS=: read -r label run_id leaf <<< "$spec"
        run_dir="$RESULTS_BASE/$ds/$run_id/_${ds}_hvg_top3000/$RADIUS/$leaf"
        if [[ ! -f "$run_dir/mc_dropout/test_summaries.h5ad" ]]; then
            echo "skip (no MC summaries): $run_dir/mc_dropout/test_summaries.h5ad"
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
echo "${DRYRUN:+(dry-run) }matched $n run-dir(s)  [RESULTS_BASE=$RESULTS_BASE DATASETS='$DATASETS']"
