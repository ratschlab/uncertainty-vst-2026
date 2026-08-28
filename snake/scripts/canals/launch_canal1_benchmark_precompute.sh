#!/usr/bin/env bash
# snake/scripts/canals/launch_canal1_benchmark_precompute.sh
#
# Submit one benchmark-summary precompute job per classic run
# (canal1_benchmark_summary_precompute.py --run-dir, a wrapper around
# src/evaluation/benchmark_summary.py::export_summary), for every run found under RESULTS_BASE
# whose model_type is listed in MODELS, across the datasets in DATASETS. One SLURM job per run
# dir, CPU-only, all concurrent.
#
# No model type is excluded on cost grounds, msemeanonly included: export_summary skips the
# prediction read for a non-probabilistic run, which makes that head the cheapest case.
#
# One sbatch call per task, no job array, matching the other launchers in this directory.
#
# Usage (cluster login node, repo root):
#   bash snake/scripts/canals/launch_canal1_benchmark_precompute.sh                 # submit missing runs only
#   DRYRUN=1 bash snake/scripts/canals/launch_canal1_benchmark_precompute.sh        # preview only, no sbatch
#   FORCE=1 bash snake/scripts/canals/launch_canal1_benchmark_precompute.sh         # recompute (--overwrite) + resubmit even already-done runs
set -euo pipefail

# ── Knobs (env overrides) ─────────────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
RESULTS_BASE="${RESULTS_BASE:-$PROJECT_ROOT/results}"
RUN_ID="${RUN_ID:-pr_classic3}"
RADIUS="${RADIUS:-r75}"
DATASETS="${DATASETS:-10x_TuPro TLS LUNG_CANCER}"
MODELS="${MODELS:-heteroscedastic homoscedastic msemeanonly spot_homoscedastic poisson_cp negativebinomial_hetero_noanchor_cp negativebinomial_spotdisp_cp negativebinomial_genedisp_cp}"
CONDA_ENV="${CONDA_ENV:-mtenv0}"                  # env with torch + scipy + anndata + src
CONDA_SH="${CONDA_SH:-$(conda info --base)/etc/profile.d/conda.sh}"
DRYRUN="${DRYRUN:-0}"
FORCE="${FORCE:-0}"                               # 1 = --overwrite + resubmit even already-done runs

# CPU-only: no -p/--gres is passed, so the job lands on the compute partition. export_summary
# always reads atoms.parquet and, for a probabilistic run, the predictions once, for the
# sigma-vs-mean correlation and, on a heteroscedastic run, the per-gene e_aurc loop. It builds no
# per-spot table, so the request below has headroom.
CPUS="${CPUS:-4}"; MEM="${MEM:-32000}"; TIME="${TIME:-00:45:00}"

LOGDIR="$PROJECT_ROOT/snake/logs"
mkdir -p "$LOGDIR"

submit() {
    local run_dir="$1"
    local extra_args=""
    [[ "$FORCE" == "1" ]] && extra_args="--overwrite"

    # export_summary's own cache marker lives directly in run_dir (no subfolder).
    if [[ "$FORCE" != "1" && -f "$run_dir/benchmark_summary.json" ]]; then
        echo "skip (already done): $run_dir"
        return
    fi

    local tag; tag="$(echo "${run_dir#"$RESULTS_BASE"/}" | tr '/' '_')"
    if [[ "$DRYRUN" == "1" ]]; then
        echo "WOULD SUBMIT  $run_dir  ${extra_args:+[$extra_args]}"
        return
    fi

    sbatch --job-name="canal1_bench_${tag}" --time="$TIME" -c "$CPUS" --mem="${MEM}M" \
        --output="$LOGDIR/canal1_bench_${tag}_%j.out" \
        --error="$LOGDIR/canal1_bench_${tag}_%j.err" \
        --wrap ". '$CONDA_SH' && conda activate $CONDA_ENV && cd '$PROJECT_ROOT' || exit 1; python -u snake/scripts/canals/canal1_benchmark_summary_precompute.py --run-dir '$run_dir' $extra_args"
    echo "submitted     $run_dir  ${extra_args:+[$extra_args]}"
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
        submit "$d"; n=$((n+1))
    done < <(find "$base" -name metrics.json)
done
echo "------"
echo "${DRYRUN:+(dry-run) }matched $n run-dir(s)  [RESULTS_BASE=$RESULTS_BASE RUN_ID=$RUN_ID MODELS='$MODELS']"
