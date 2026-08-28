#!/usr/bin/env bash
# snake/scripts/canals/launch_epi_ale_density_precompute.sh
#
# Submit the epistemic-vs-aleatoric density precompute (epi_ale_density_precompute.py): one SLURM
# job per (dataset, head, method), CPU-only, all concurrent, in the house style of
# launch_canal2_precompute.sh. Each job reads exactly one test_summaries.h5ad and writes
# histograms, so the jobs share nothing.
#
# The three estimators and where they live. This is the thing to get right, because the deep
# ensemble sits under a different run directory and a different leaf shape:
#   MC-dropout     $RESULTS_BASE/<ds>/$POSTHOC_RUN_ID/.../<head>[/lambda_0.0]/mc_dropout/test_summaries.h5ad
#   Laplace        $RESULTS_BASE/<ds>/$POSTHOC_RUN_ID/.../<head>[/lambda_0.0]/laplace/test_summaries.h5ad
#   deep ensemble  $RESULTS_BASE/<ds>/$ENS_RUN_ID/.../<head>/ensemble/test_summaries.h5ad   <- no lambda_ level
# The single-model tree has lambda_0.0 as a real directory level for the Gaussian heads: 0.0 is
# the plain NLL and the only lambda with post-hoc summaries on disk. The ensemble tree has no such
# level, because the ensemble lambda is constant across members and is not in the path.
#
# Why these heads. They are the intersection: the heads for which all three estimators exist, on
# the shared reporting axis. The ensemble sweep also carries msemeanonly (no aleatoric head, so no
# x axis) and the two *_raw heads (a different reporting axis, so they need their own figure);
# both are deliberately out. To add them, uncomment below and expect the *_raw pair to be plotted
# separately, never beside the _cp or Gaussian ones.
#
# Usage (cluster login node, repo root):
#   bash snake/scripts/canals/launch_epi_ale_density_precompute.sh
#   DRYRUN=1 bash snake/scripts/canals/launch_epi_ale_density_precompute.sh   # preview, no sbatch
#   FORCE=1  bash snake/scripts/canals/launch_epi_ale_density_precompute.sh   # resubmit even done
#   METHODS="laplace" bash snake/scripts/canals/launch_epi_ale_density_precompute.sh
# Other env knobs, all defaulted in the block below: RESULTS_BASE, RADIUS, DATASETS,
# POSTHOC_RUN_ID, ENS_RUN_ID, CONDA_ENV, CONDA_SH, CPUS, MEM, TIME.
set -euo pipefail

# ── Knobs (env overrides) ─────────────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
RESULTS_BASE="${RESULTS_BASE:-$PROJECT_ROOT/results}"
RADIUS="${RADIUS:-r75}"
DATASETS="${DATASETS:-10x_TuPro TLS LUNG_CANCER}"
METHODS="${METHODS:-mc_dropout laplace deep_ensemble}"
FORCE="${FORCE:-0}"
DRYRUN="${DRYRUN:-0}"

# The run id of the single-model arm (MC-dropout + Laplace) and of the ensemble arm.
POSTHOC_RUN_ID="${POSTHOC_RUN_ID:-pr_classic3}"
ENS_RUN_ID="${ENS_RUN_ID:-de_6}"

# One entry per head, "model_type:gaussian?". The flag says whether the single-model leaf carries
# a lambda_0.0 level; the ensemble leaf never does.
HEADS=(
    "homoscedastic:1"
    "heteroscedastic:1"
    "spot_homoscedastic:1"
    "poisson_cp:0"
    "negativebinomial_hetero_noanchor_cp:0"
    # OUT ON PURPOSE (see header):
    # "msemeanonly:1"                            # no aleatoric head -> no x axis
    # "poisson_raw:0"                            # raw-count axis -> its own figure
    # "negativebinomial_hetero_noanchor_raw:0"   # raw-count axis -> its own figure
)

CONDA_ENV="${CONDA_ENV:-mtenv0}"
CONDA_SH="${CONDA_SH:-$(conda info --base)/etc/profile.d/conda.sh}"
# CPU-only: no -p / --gres, so the job lands on the `compute` partition. Peak RAM is the h5ad
# itself, one float32 value per (spot, gene) for each of a few layers; the histograms are
# negligible. Far lighter than canal 2: no long per-spot dataframe, no rank cascade.
CPUS="${CPUS:-2}"; MEM="${MEM:-96000}"; TIME="${TIME:-01:00:00}"

LOGDIR="$PROJECT_ROOT/snake/logs"
mkdir -p "$LOGDIR"

n=0; skipped=0
for ds in $DATASETS; do
    for entry in "${HEADS[@]}"; do
        IFS=: read -r head is_gauss <<< "$entry"
        for method in $METHODS; do
            case "$method" in
                mc_dropout|laplace)
                    if [[ "$is_gauss" == "1" ]]; then leaf="$head/lambda_0.0"; else leaf="$head"; fi
                    sub="$method"
                    base="$RESULTS_BASE/$ds/$POSTHOC_RUN_ID/_${ds}_hvg_top3000/$RADIUS/$leaf"
                    ;;
                deep_ensemble)
                    leaf="$head"                     # ensemble leaf has NO lambda_ level
                    sub="ensemble"
                    base="$RESULTS_BASE/$ds/$ENS_RUN_ID/_${ds}_hvg_top3000/$RADIUS/$leaf"
                    ;;
                *) echo "unknown method '$method'"; exit 1 ;;
            esac
            f="$base/$sub/test_summaries.h5ad"
            out="$base/$sub/epi_ale_density"

            if [[ ! -f "$f" ]]; then
                echo "skip (no summaries): $f"; skipped=$((skipped+1)); continue
            fi
            if [[ "$FORCE" != "1" && -f "$out/summary.json" ]]; then
                echo "skip (already done): $out"; skipped=$((skipped+1)); continue
            fi
            tag="$(echo "${base#"$RESULTS_BASE"/}" | tr '/' '_')_$method"
            if [[ "$DRYRUN" == "1" ]]; then
                echo "WOULD SUBMIT  $f"; n=$((n+1)); continue
            fi
            sbatch --job-name="epiale_${tag}" --time="$TIME" -c "$CPUS" --mem="${MEM}M" \
                --output="$LOGDIR/epiale_${tag}_%j.out" \
                --error="$LOGDIR/epiale_${tag}_%j.err" \
                --wrap ". '$CONDA_SH' && conda activate $CONDA_ENV && cd '$PROJECT_ROOT' || exit 1; python -u snake/scripts/canals/epi_ale_density_precompute.py --summary-file '$f' --dataset '$ds' --model-type '$head' --method '$method'"
            echo "submitted     $f"
            n=$((n+1))
        done
    done
done
echo "------"
echo "${DRYRUN:+(dry-run) }submitted $n job(s), skipped $skipped  [RESULTS_BASE=$RESULTS_BASE DATASETS='$DATASETS' METHODS='$METHODS']"
