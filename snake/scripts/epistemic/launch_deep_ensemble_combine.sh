#!/usr/bin/env bash
# snake/scripts/epistemic/launch_deep_ensemble_combine.sh
#
# Submit deep_ensemble_combine.py as ONE SLURM job PER (dataset, model). The script's CLI takes
# several --model-dir values, but bundling them serializes work that is independent:
# combine_model_dir() shares no state across model dirs (own read, own accumulation, own
# write_h5ad), and main() loops over them one after another inside a single job.
#
# CPU-only, modest resources -- this step is pure read + numpy, so no GPU is requested. Contrast
# DE_gstar_precompute.py, whose count_expected_log1p wants one.
#
# Run from the root of the tree whose results/ holds the model(s) wanted. Two model types can
# sit under two different run directories, with no single results/ tree holding both, so this
# launcher is run once per tree. A MODELS entry with no match under the current results/ tree
# is skipped.
#
# Usage (cluster login node, repo root):
#   bash snake/scripts/epistemic/launch_deep_ensemble_combine.sh                # submit missing only
#   DRYRUN=1 bash snake/scripts/epistemic/launch_deep_ensemble_combine.sh       # preview only
#   FORCE=1 bash snake/scripts/epistemic/launch_deep_ensemble_combine.sh        # resubmit even done ones
#   MODELS="STI_nb_cp" bash snake/scripts/epistemic/launch_deep_ensemble_combine.sh
set -euo pipefail

# ── Knobs (env overrides) ─────────────────────────────────────────────────────
RESULTS_BASE="${RESULTS_BASE:-results}"            # relative to cwd -- run from the repo root
RADIUS="${RADIUS:-r75}"
FORCE="${FORCE:-0}"                                 # 1 = resubmit even if ensemble/test_summaries.h5ad exists
DATASETS="${DATASETS:-10x_TuPro TLS LUNG_CANCER}"
# The two default models share one target, cp_counts = round(CP_T), so comparing them is
# architecture-only: STI_nb_cp has a bare Linear head, negativebinomial_hetero_noanchor_cp an MLP
# head.
MODELS="${MODELS:-STI_nb_cp negativebinomial_hetero_noanchor_cp}"
SEEDS="${SEEDS:-0,1,2,3,4}"
CONDA_ENV="${CONDA_ENV:-mtenv0}"
CONDA_SH="${CONDA_SH:-$(conda info --base)/etc/profile.d/conda.sh}"
DRYRUN="${DRYRUN:-0}"

CPUS="${CPUS:-2}"; MEM="${MEM:-32000}"; TIME="${TIME:-00:30:00}"   # per DATASET now, not per 3-dataset bundle

PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
LOGDIR="$PROJECT_ROOT/snake/logs"
mkdir -p "$LOGDIR"

submit() {
    local ds="$1" mt="$2"
    local resolved
    resolved="$(find "$RESULTS_BASE/$ds" -mindepth 4 -maxdepth 4 -type d -path "*/$RADIUS/$mt" 2>/dev/null)"
    local n_matches; n_matches="$(echo -n "$resolved" | grep -c . || true)"
    if [[ "$n_matches" -eq 0 ]]; then
        echo "skip (no match): $RESULTS_BASE/$ds/*/*/$RADIUS/$mt"
        return
    fi
    if [[ "$n_matches" -gt 1 ]]; then
        echo "SKIP $ds/$mt: expected exactly 1 match, got $n_matches -- resolve ambiguity manually:"
        echo "$resolved"
        return
    fi

    local out_file="$resolved/ensemble/test_summaries.h5ad"
    if [[ "$FORCE" != "1" && -f "$out_file" ]]; then
        echo "skip (already done): $out_file"
        return
    fi

    if [[ "$DRYRUN" == "1" ]]; then
        echo "WOULD SUBMIT  $ds/$mt  ($resolved)"
        return
    fi

    sbatch --job-name="combine_${ds}_${mt}" --time="$TIME" -c "$CPUS" --mem="${MEM}M" \
        --output="$LOGDIR/combine_${ds}_${mt}_%j.out" \
        --error="$LOGDIR/combine_${ds}_${mt}_%j.err" \
        --wrap ". '$CONDA_SH' && conda activate $CONDA_ENV && cd '$PROJECT_ROOT' || exit 1; python -u snake/scripts/epistemic/deep_ensemble_combine.py --model-dir '$resolved' --seeds '$SEEDS'"
    echo "submitted     $ds/$mt  ($resolved)"
}

n=0
for ds in $DATASETS; do
    [[ -d "$RESULTS_BASE/$ds" ]] || { echo "skip (no dataset dir): $RESULTS_BASE/$ds"; continue; }
    for mt in $MODELS; do
        submit "$ds" "$mt"; n=$((n+1))
    done
done
echo "------"
echo "${DRYRUN:+(dry-run) }considered $n (dataset, model) pair(s)  [RESULTS_BASE=$RESULTS_BASE MODELS='$MODELS' DATASETS='$DATASETS']"
