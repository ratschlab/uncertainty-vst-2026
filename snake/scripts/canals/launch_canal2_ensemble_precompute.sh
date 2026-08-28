#!/usr/bin/env bash
# snake/scripts/canals/launch_canal2_ensemble_precompute.sh
#
# Submit Canal 2 precompute jobs for deep-ensemble models (canal2_uncertainty_precompute.py
# --ensemble-dir), reading the <model-dir>/ensemble/test_summaries.h5ad that
# snake/scripts/epistemic/deep_ensemble_combine.py writes. That script must run first. One SLURM
# job per (dataset, spec), CPU-only, all concurrent.
#
# The --run-dir sweep walks one fixed RUN_ID under one results/ tree; ensemble models can
# instead sit under several run directories, so targets are listed explicitly in the SPECS
# array below, one entry per "run_id:leaf".
#
# Usage (cluster login node, repo root):
#   bash snake/scripts/canals/launch_canal2_ensemble_precompute.sh                    # total source (default)
#   UNCERTAINTY_SOURCE=epistemic bash snake/scripts/canals/launch_canal2_ensemble_precompute.sh
#   UNCERTAINTY_SOURCE=aleatoric bash snake/scripts/canals/launch_canal2_ensemble_precompute.sh
#   DRYRUN=1 bash snake/scripts/canals/launch_canal2_ensemble_precompute.sh           # preview only, no sbatch
#   FORCE=1 bash snake/scripts/canals/launch_canal2_ensemble_precompute.sh            # resubmit even already-done
set -euo pipefail

# ── Knobs (env overrides) ─────────────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
RESULTS_BASE="${RESULTS_BASE:-$PROJECT_ROOT/results}"
RADIUS="${RADIUS:-r75}"
# UNCERTAINTY_SOURCE = total (default) | epistemic | aleatoric. `total` is the like-for-like
# comparison to a --run-dir model's own uncertainty head, which already conflates the two sources.
# It sets the out-subdir and the job-name prefix, so the three sources never collide.
UNCERTAINTY_SOURCE="${UNCERTAINTY_SOURCE:-total}"
case "$UNCERTAINTY_SOURCE" in
    total)     JOB_PREFIX="canal2ens_tot" ;;
    epistemic) JOB_PREFIX="canal2ens_epi" ;;
    aleatoric) JOB_PREFIX="canal2ens_ale" ;;
    *) echo "UNCERTAINTY_SOURCE must be 'total', 'epistemic', or 'aleatoric' (got '$UNCERTAINTY_SOURCE')"; exit 1 ;;
esac
FORCE="${FORCE:-0}"                            # 1 = resubmit even if summary.json already exists
DATASETS="${DATASETS:-10x_TuPro TLS LUNG_CANCER}"
RUN_IDS="${RUN_IDS:-}"                           # optional: restrict to these run ids (space-sep,
                                               # e.g. RUN_IDS="de_6"). Empty => every SPECS entry.
                                               # Pair it with FORCE=1, which is the only way to
                                               # refresh an existing artifact: without RUN_IDS,
                                               # FORCE resubmits every entry in SPECS, the
                                               # selection-rule side-experiments included.
# One entry per ensemble-combined model: "run_id:leaf". Add new comparisons here
# as more deep ensembles are built.
#
# STI_nb_cp trains on cp_counts = round(CP_T), the same target as
# negativebinomial_hetero_noanchor_cp, which makes that pair an architecture-only comparison: a
# bare Linear head against an MLP.
#
# de_6 is the corr-selected ensemble zoo, several heads and not just the NB, so it carries the
# deep-ensemble epistemic term on more than one model family. msemeanonly is absent on purpose: it
# has no uncertainty estimate at all, so canal 2 exits gracefully and the job is a guaranteed no-op.
#
# Each entry is "run_id:model_type". The model_type is the leaf directory of the run tree. The
# ensemble path carries no lambda_ level, unlike the single-model tree, because ensemble.lambda is
# constant across members (0.0, the proper NLL). An entry whose test_summaries.h5ad is absent skips
# itself, so a head that has not been built yet can sit beside a built one.
SPECS=(
    "sti_v1:STI_nb_cp"
    "sti_v1:STI_nb_raw"
    "de_6:heteroscedastic"
    "de_6:homoscedastic"
    "de_6:spot_homoscedastic"
    "de_6:poisson_cp"
    "de_6:negativebinomial_hetero_noanchor_cp"
    "de_6:poisson_raw"
    "de_6:negativebinomial_hetero_noanchor_raw"
)
CONDA_ENV="${CONDA_ENV:-mtenv0}"
CONDA_SH="${CONDA_SH:-$(conda info --base)/etc/profile.d/conda.sh}"
DRYRUN="${DRYRUN:-0}"
# CPU-only: passing no -p/--gres lands the job on the compute partition. Same per-run cost profile
# as the own-head Canal 2 (long per-spot table + rank-based groupby cascade), plus the whole-panel
# linear-error pass over every (spot, gene) entry, zeros included, which is what the memory here is
# sized for. Bump TIME if a run times out.
CPUS="${CPUS:-4}"; MEM="${MEM:-128000}"; TIME="${TIME:-04:00:00}"

LOGDIR="$PROJECT_ROOT/snake/logs"
mkdir -p "$LOGDIR"

submit() {
    local ensemble_dir="$1"
    local tag; tag="$(echo "${ensemble_dir#"$RESULTS_BASE"/}" | tr '/' '_')"

    if [[ "$FORCE" != "1" && -f "$ensemble_dir/canal2_uncertainty/$UNCERTAINTY_SOURCE/summary.json" ]]; then
        echo "skip (already done): $ensemble_dir"
        return
    fi
    if [[ "$DRYRUN" == "1" ]]; then
        echo "WOULD SUBMIT  $ensemble_dir"
        return
    fi

    sbatch --job-name="${JOB_PREFIX}_${tag}" --time="$TIME" -c "$CPUS" --mem="${MEM}M" \
        --output="$LOGDIR/${JOB_PREFIX}_${tag}_%j.out" \
        --error="$LOGDIR/${JOB_PREFIX}_${tag}_%j.err" \
        --wrap ". '$CONDA_SH' && conda activate $CONDA_ENV && cd '$PROJECT_ROOT' || exit 1; python -u snake/scripts/canals/canal2_uncertainty_precompute.py --ensemble-dir '$ensemble_dir' --uncertainty-source '$UNCERTAINTY_SOURCE'"
    echo "submitted     $ensemble_dir"
}

n=0
for ds in $DATASETS; do
    for spec in "${SPECS[@]}"; do
        IFS=: read -r run_id leaf <<< "$spec"
        if [[ -n "$RUN_IDS" ]]; then case " $RUN_IDS " in *" $run_id "*) ;; *) continue ;; esac; fi
        model_dir="$RESULTS_BASE/$ds/$run_id/_${ds}_hvg_top3000/$RADIUS/$leaf"
        ensemble_dir="$model_dir/ensemble"
        if [[ ! -f "$ensemble_dir/test_summaries.h5ad" ]]; then
            echo "skip (no ensemble summaries): $ensemble_dir/test_summaries.h5ad"
            continue
        fi
        submit "$ensemble_dir"; n=$((n+1))
    done
done
echo "------"
echo "matched $n ensemble-dir(s)  [RESULTS_BASE=$RESULTS_BASE DATASETS='$DATASETS' RUN_IDS='$RUN_IDS' UNCERTAINTY_SOURCE=$UNCERTAINTY_SOURCE]"
