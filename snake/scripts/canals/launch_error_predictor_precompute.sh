#!/usr/bin/env bash
# snake/scripts/canals/launch_error_predictor_precompute.sh
#
# Submit error_predictor_gene_precompute.py (the reported version): one CPU job per (run,
# held-out donor), all concurrent. POOLED=1 runs the earlier pooled version,
# error_predictor_precompute.py, instead. The fold count is read from each run's splits.json.
#
# Usage (cluster login node, repo root):
#   bash snake/scripts/canals/launch_error_predictor_precompute.sh            # submit missing folds only
#   DRYRUN=1 bash snake/scripts/canals/launch_error_predictor_precompute.sh   # preview only
#   FORCE=1 bash snake/scripts/canals/launch_error_predictor_precompute.sh    # resubmit done folds too
#   MODELS="poisson_cp" DATASETS="TLS" bash snake/scripts/canals/launch_error_predictor_precompute.sh
#   POOLED=1 bash snake/scripts/canals/launch_error_predictor_precompute.sh    # earlier pooled version
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
RESULTS_BASE="${RESULTS_BASE:-$PROJECT_ROOT/results}"
RUN_ID="${RUN_ID:-pr_classic3}"
RADIUS="${RADIUS:-r75}"
POOLED="${POOLED:-0}"
if [[ "$POOLED" == "1" ]]; then
    SCRIPT="error_predictor_precompute.py";      OUT_SUBDIR="${OUT_SUBDIR:-error_predictor}";      JOB="errpred"
else
    SCRIPT="error_predictor_gene_precompute.py"; OUT_SUBDIR="${OUT_SUBDIR:-error_predictor_gene}"; JOB="errpredg"
fi
DATASETS="${DATASETS:-10x_TuPro TLS LUNG_CANCER}"
# poisson_cp and homoscedastic are sanity heads: B equals A up to fitting noise.
MODELS="${MODELS:-homoscedastic heteroscedastic spot_homoscedastic poisson_cp negativebinomial_hetero_noanchor_cp}"
LAMBDAS="${LAMBDAS:-0.0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
FORCE="${FORCE:-0}"
DRYRUN="${DRYRUN:-0}"
CONDA_ENV="${CONDA_ENV:-mtenv0}"
CONDA_SH="${CONDA_SH:-$(conda info --base)/etc/profile.d/conda.sh}"

# CPU only; the forests use all requested cores.
CPUS="${CPUS:-8}"; MEM="${MEM:-64000}"; TIME="${TIME:-06:00:00}"

LOGDIR="$PROJECT_ROOT/snake/logs"
mkdir -p "$LOGDIR"

submit() {
    local run_dir="$1" fold="$2"
    local tag; tag="$(echo "${run_dir#"$RESULTS_BASE"/}" | tr '/' '_')_fold${fold}"
    if [[ "$FORCE" != "1" && -f "$run_dir/$OUT_SUBDIR/fold${fold}.json" ]]; then
        echo "skip (already done): $run_dir fold $fold"
        return
    fi
    if [[ "$FORCE" != "1" ]] && squeue -u "$(whoami)" -h -o '%j' | grep -qx "${JOB}_${tag}"; then
        echo "skip (already queued or running): $run_dir fold $fold"
        return
    fi
    if [[ "$DRYRUN" == "1" ]]; then
        echo "WOULD SUBMIT  $run_dir fold $fold"
        return
    fi
    sbatch --job-name="${JOB}_${tag}" --time="$TIME" -c "$CPUS" --mem="${MEM}M" \
        --output="$LOGDIR/${JOB}_${tag}_%j.out" \
        --error="$LOGDIR/${JOB}_${tag}_%j.err" \
        --wrap ". '$CONDA_SH' && conda activate $CONDA_ENV && cd '$PROJECT_ROOT' || exit 1; python -u snake/scripts/canals/$SCRIPT --run-dir '$run_dir' --fold $fold $EXTRA_ARGS"
    echo "submitted     $run_dir fold $fold"
}

n=0
for ds in $DATASETS; do
    base="$RESULTS_BASE/$ds/$RUN_ID"
    [[ -d "$base" ]] || { echo "skip (no dir): $base"; continue; }
    while IFS= read -r mj; do
        d="$(dirname "$mj")"
        [[ -d "$d/predictions" && -f "$d/splits.json" ]] || continue
        b="$(basename "$d")"
        if [[ "$b" == lambda_* ]]; then mt="$(basename "$(dirname "$d")")"; else mt="$b"; fi
        [[ "$d" == *"/$RADIUS/"* ]] || continue
        case " $MODELS " in *" $mt "*) ;; *) continue ;; esac
        if [[ -n "$LAMBDAS" && "$b" == lambda_* ]]; then
            lam="${b#lambda_}"
            case " $LAMBDAS " in *" $lam "*) ;; *) continue ;; esac
        fi
        k="$(grep -o '"fold": [0-9]*' "$d/splits.json" | wc -l)"
        [[ "$k" -ge 3 ]] || { echo "skip (only $k folds in splits.json): $d"; continue; }
        for ((j = 0; j < k; j++)); do
            submit "$d" "$j"; n=$((n+1))
        done
    done < <(find "$base" -name metrics.json)
done
echo "------"
[[ "$DRYRUN" == "1" ]] && pfx="(dry-run) " || pfx=""
echo "${pfx}matched $n (run, fold) job(s)  [$SCRIPT -> $OUT_SUBDIR; RESULTS_BASE=$RESULTS_BASE RUN_ID=$RUN_ID MODELS='$MODELS' LAMBDAS='$LAMBDAS']"
