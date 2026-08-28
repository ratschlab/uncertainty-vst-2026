#!/usr/bin/env bash
# snake/scripts/canals/launch_canal1_ensemble_precompute.sh
#
# Submit benchmark-summary precompute jobs for deep-ensemble models
# (canal1_benchmark_summary_precompute.py --ensemble-dir), against the
# <model-dir>/ensemble/test_summaries.h5ad that snake/scripts/epistemic/deep_ensemble_combine.py
# writes, with the PIT bounds already computed and, for the count families, g* as well. One SLURM
# job per (dataset, spec), CPU-only, all concurrent, following the same pattern as
# launch_canal2_ensemble_precompute.sh.
#
# Ensembles can sit under several run directories, so they cannot be found from a single
# RUN_ID the way the --run-dir launcher finds its runs. The SPECS array below names them
# instead, one entry per "run_id:leaf".
#
# There is no UNCERTAINTY_SOURCE knob here, unlike the Canal 2 ensemble launcher: g* and the PIT
# are single quantities of the ensemble mixture, not total, aleatoric and epistemic variants.
#
# Usage (cluster login node, repo root):
#   bash snake/scripts/canals/launch_canal1_ensemble_precompute.sh                    # submit all
#   DRYRUN=1 bash snake/scripts/canals/launch_canal1_ensemble_precompute.sh           # preview only, no sbatch
#   FORCE=1 bash snake/scripts/canals/launch_canal1_ensemble_precompute.sh            # resubmit even already-done
set -euo pipefail

# ── Knobs (env overrides) ─────────────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
RESULTS_BASE="${RESULTS_BASE:-$PROJECT_ROOT/results}"
RADIUS="${RADIUS:-r75}"
FORCE="${FORCE:-0}"                            # 1 = resubmit even if benchmark_summary.json already exists
DATASETS="${DATASETS:-10x_TuPro TLS LUNG_CANCER}"
# One entry per ensemble-combined model. Add new comparisons here as more ensembles are built.
#
# The STI_nb_cp leaf trains on cp_counts = round(CP_T), the same target as
# negativebinomial_hetero_noanchor_cp, which makes that pair an architecture-only comparison: a
# bare Linear head against an MLP.
# Each entry is "run_id:model_type". The model_type is the leaf directory of the run tree; the
# ensemble path carries no lambda_ level, unlike the single-model tree, because ensemble.lambda is
# constant across members. An entry whose test_summaries.h5ad is absent skips itself, so a head
# that has not been built yet can sit beside a built one.
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
# CPU-only: no -p/--gres, so the job lands on the compute partition. Each run rebuilds the
# per-(slide, gene) atoms from the full test_summaries.h5ad matrices, an ensemble having no
# atoms.parquet; raise TIME if a run hits the limit.
CPUS="${CPUS:-4}"; MEM="${MEM:-64000}"; TIME="${TIME:-01:00:00}"

LOGDIR="$PROJECT_ROOT/snake/logs"
mkdir -p "$LOGDIR"

submit() {
    local ensemble_dir="$1"
    local tag; tag="$(echo "${ensemble_dir#"$RESULTS_BASE"/}" | tr '/' '_')"

    if [[ "$FORCE" != "1" && -f "$ensemble_dir/benchmark_summary.json" ]]; then
        echo "skip (already done): $ensemble_dir"
        return
    fi
    if [[ "$DRYRUN" == "1" ]]; then
        echo "WOULD SUBMIT  $ensemble_dir"
        return
    fi

    sbatch --job-name="canal1ens_${tag}" --time="$TIME" -c "$CPUS" --mem="${MEM}M" \
        --output="$LOGDIR/canal1ens_${tag}_%j.out" \
        --error="$LOGDIR/canal1ens_${tag}_%j.err" \
        --wrap ". '$CONDA_SH' && conda activate $CONDA_ENV && cd '$PROJECT_ROOT' || exit 1; python -u snake/scripts/canals/canal1_benchmark_summary_precompute.py --ensemble-dir '$ensemble_dir'"
    echo "submitted     $ensemble_dir"
}

n=0
for ds in $DATASETS; do
    for spec in "${SPECS[@]}"; do
        IFS=: read -r run_id leaf <<< "$spec"
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
echo "${DRYRUN:+(dry-run) }matched $n ensemble-dir(s)  [RESULTS_BASE=$RESULTS_BASE DATASETS='$DATASETS']"
