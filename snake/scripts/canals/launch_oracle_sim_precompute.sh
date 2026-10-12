#!/usr/bin/env bash
# snake/scripts/canals/launch_oracle_sim_precompute.sh
#
# Submit oracle_sim_precompute.py: one CPU job per (run, seed), all concurrent. REAL=1 instead
# submits one --no-sim job per run, the check against canal 2's own output.
#
# Usage (cluster login node, repo root):
#   REAL=1 bash snake/scripts/canals/launch_oracle_sim_precompute.sh      # equality check first
#   bash snake/scripts/canals/launch_oracle_sim_precompute.sh             # simulation, SEEDS
#   SEEDS="0 1 2" bash snake/scripts/canals/launch_oracle_sim_precompute.sh
#   DRYRUN=1 bash snake/scripts/canals/launch_oracle_sim_precompute.sh    # preview only
#   DIAG=1 SEEDS="0 1 2" bash snake/scripts/canals/launch_oracle_sim_precompute.sh  # only diag_genes.parquet
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
RESULTS_BASE="${RESULTS_BASE:-$PROJECT_ROOT/results}"
RUN_ID="${RUN_ID:-pr_classic3}"
RADIUS="${RADIUS:-r75}"
DATASETS="${DATASETS:-10x_TuPro TLS LUNG_CANCER}"
MODELS="${MODELS:-heteroscedastic poisson_cp negativebinomial_hetero_noanchor_cp}"
LAMBDAS="${LAMBDAS:-0.0}"
SEEDS="${SEEDS:-0}"
REAL="${REAL:-0}"
DIAG="${DIAG:-0}"
FORCE="${FORCE:-0}"
DRYRUN="${DRYRUN:-0}"
CONDA_ENV="${CONDA_ENV:-mtenv0}"
CONDA_SH="${CONDA_SH:-$(conda info --base)/etc/profile.d/conda.sh}"
CPUS="${CPUS:-4}"; MEM="${MEM:-128000}"; TIME="${TIME:-04:00:00}"

LOGDIR="$PROJECT_ROOT/snake/logs"
mkdir -p "$LOGDIR"

submit() {
    local run_dir="$1" sub="$2" flag="$3" done_file="$4"
    local tag; tag="$(echo "${run_dir#"$RESULTS_BASE"/}" | tr '/' '_')_${sub}"
    [[ "$DIAG" == "1" ]] && tag="${tag}_diag"
    if [[ "$FORCE" != "1" && -f "$run_dir/oracle_sim/$sub/$done_file" ]]; then
        echo "skip (already done): $run_dir [$sub]"
        return
    fi
    if [[ "$DRYRUN" == "1" ]]; then
        echo "WOULD SUBMIT  $run_dir [$sub]"
        return
    fi
    sbatch --job-name="oraclesim_${tag}" --time="$TIME" -c "$CPUS" --mem="${MEM}M" \
        --output="$LOGDIR/oraclesim_${tag}_%j.out" \
        --error="$LOGDIR/oraclesim_${tag}_%j.err" \
        --wrap ". '$CONDA_SH' && conda activate $CONDA_ENV && cd '$PROJECT_ROOT' || exit 1; python -u snake/scripts/canals/oracle_sim_precompute.py --run-dir '$run_dir' $flag"
    echo "submitted     $run_dir [$sub]"
}

n=0
for ds in $DATASETS; do
    base="$RESULTS_BASE/$ds/$RUN_ID"
    [[ -d "$base" ]] || { echo "skip (no dir): $base"; continue; }
    while IFS= read -r mj; do
        d="$(dirname "$mj")"
        [[ -d "$d/predictions" ]] || continue
        b="$(basename "$d")"
        if [[ "$b" == lambda_* ]]; then mt="$(basename "$(dirname "$d")")"; else mt="$b"; fi
        [[ "$d" == *"/$RADIUS/"* ]] || continue
        case " $MODELS " in *" $mt "*) ;; *) continue ;; esac
        if [[ -n "$LAMBDAS" && "$b" == lambda_* ]]; then
            lam="${b#lambda_}"
            case " $LAMBDAS " in *" $lam "*) ;; *) continue ;; esac
        fi
        if [[ "$REAL" == "1" ]]; then
            submit "$d" "real" "--no-sim" "summary.json"; n=$((n+1))
        elif [[ "$DIAG" == "1" ]]; then
            for s in $SEEDS; do submit "$d" "seed$s" "--seed $s --diag-only" "diag_genes.parquet"; n=$((n+1)); done
        else
            for s in $SEEDS; do submit "$d" "seed$s" "--seed $s" "summary.json"; n=$((n+1)); done
        fi
    done < <(find "$base" -name metrics.json -not -path "*/oracle_sim/*" -not -path "*/canal2*")
done
echo "------"
[[ "$DRYRUN" == "1" ]] && pfx="(dry-run) " || pfx=""
echo "${pfx}matched $n job(s)  [RUN_ID=$RUN_ID MODELS='$MODELS' LAMBDAS='$LAMBDAS' SEEDS='$SEEDS' REAL=$REAL DIAG=$DIAG]"
