#!/usr/bin/env bash
# snake/scripts/canals/launch_canal3_precompute.sh
#
# Submit canal-3 zero-detection AUROC precompute jobs (canal3_zero_detection_precompute.py) for
# every discovered classic run dir of one sweep, across the 3 datasets. One SLURM job per run dir,
# all running concurrently, CPU-only. One `sbatch` per task, the house style here, as in
# launch_epistemic.sh; no job arrays.
#
# Unlike launch_canal2_precompute.sh, msemeanonly is included here: zero detection needs only a
# monotone proxy score (-X), not a real uncertainty estimate.
#
# Usage (cluster login node, repo root):
#   bash snake/scripts/canals/launch_canal3_precompute.sh                 # submit missing runs only
#   DRYRUN=1 bash snake/scripts/canals/launch_canal3_precompute.sh        # preview only, no sbatch
#   FORCE=1 bash snake/scripts/canals/launch_canal3_precompute.sh         # resubmit even already-done runs
#   MODELS="heteroscedastic poisson_cp" bash snake/scripts/canals/launch_canal3_precompute.sh
# Other env knobs, all defaulted in the block below: RESULTS_BASE, RUN_ID, RADIUS, DATASETS,
# OUT_SUBDIR, CONDA_ENV, CONDA_SH, CPUS, MEM, TIME.
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
OUT_SUBDIR="${OUT_SUBDIR:-canal3_zero_detection}"   # must match the script's --out-dir default
FORCE="${FORCE:-0}"                                 # 1 = resubmit even if summary.json already exists

# CPU-only: no -p / --gres is passed at all, so SLURM lands the job on the `compute` partition.
# Canal 3 is lighter per run than canal 2: no long per-spot table and no rank cascade, just a
# per-slide vectorized AUROC. The values below are generous headroom.
CPUS="${CPUS:-4}"; MEM="${MEM:-48000}"; TIME="${TIME:-01:00:00}"

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

    sbatch --job-name="canal3_${tag}" --time="$TIME" -c "$CPUS" --mem="${MEM}M" \
        --output="$LOGDIR/canal3_${tag}_%j.out" \
        --error="$LOGDIR/canal3_${tag}_%j.err" \
        --wrap ". '$CONDA_SH' && conda activate $CONDA_ENV && cd '$PROJECT_ROOT' || exit 1; python -u snake/scripts/canals/canal3_zero_detection_precompute.py --run-dir '$run_dir'"
    echo "submitted     $run_dir"
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
