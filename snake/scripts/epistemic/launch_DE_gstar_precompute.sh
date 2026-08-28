#!/usr/bin/env bash
# snake/scripts/epistemic/launch_DE_gstar_precompute.sh
#
# Submit g* precompute jobs (DE_gstar_precompute.py) for every discovered (seed, fold)
# prediction file of the in-scope count models. One SLURM job PER preds_fold{k}.h5ad,
# finer-grained than the per-run canal launchers, because g* is a per-(seed,fold) quantity
# and needs the expensive windowed truncated-sum recurrence count_expected_log1p rather
# than the closed-form quantities deep_ensemble_combine.py computes inline.
#
# Run from INSIDE the directory whose results/ tree holds the model(s) you want: two model
# types can sit under two different results trees, with no single one holding both.
# RESULTS_BASE defaults to "results", relative to cwd, precisely so the SAME launcher works
# unchanged from either root -- just cd in and run it.
#
# Usage (cluster login node, repo root):
#   bash snake/scripts/epistemic/launch_DE_gstar_precompute.sh                # submit missing folds only
#   DRYRUN=1 bash snake/scripts/epistemic/launch_DE_gstar_precompute.sh       # preview only, no sbatch
#   FORCE=1 bash snake/scripts/epistemic/launch_DE_gstar_precompute.sh        # resubmit even already-done folds
#   MODELS="STI_nb_cp" bash snake/scripts/epistemic/launch_DE_gstar_precompute.sh
#   GPUS=0 bash snake/scripts/epistemic/launch_DE_gstar_precompute.sh         # force CPU (slow, see below)
set -euo pipefail

# ── Knobs (env overrides) ─────────────────────────────────────────────────────
RESULTS_BASE="${RESULTS_BASE:-results}"            # relative to cwd -- run from the repo root
RADIUS="${RADIUS:-r75}"
FORCE="${FORCE:-0}"                                 # 1 = resubmit even if gstar_fold{k}.h5ad exists
DATASETS="${DATASETS:-10x_TuPro TLS LUNG_CANCER}"
# The count models in scope. DE_gstar_precompute.py's SUPPORTED_TARGET_SOURCES guard hard-fails
# on any other target_source rather than silently apply an unreviewed scale convention -- see
# that script's header.
#
# The two defaults share one target, cp_counts = round(CP_T), so comparing them is
# architecture-only: STI_nb_cp has a bare Linear head, negativebinomial_hetero_noanchor_cp an MLP
# head.
MODELS="${MODELS:-STI_nb_cp negativebinomial_hetero_noanchor_cp}"
CONDA_ENV="${CONDA_ENV:-mtenv0}"                    # env with torch + scipy + anndata + src
CONDA_SH="${CONDA_SH:-$(conda info --base)/etc/profile.d/conda.sh}"
DRYRUN="${DRYRUN:-0}"

# GPU by default (PART/GPUS, same knobs/placement convention as launch_epistemic.sh).
# count_expected_log1p (src/evaluation/count_metrics.py) marches in float32 on CUDA and float64
# on CPU, and the CPU path is much slower, which is why this launcher asks for a GPU where the
# other post-hoc launchers in this directory do not. GPUS=0 forces CPU, for debugging without a
# free GPU; DE_gstar_precompute.py auto-detects CUDA either way.
PART="${PART:-gpu}"; GPUS="${GPUS:-1}"
# TIME is a rough estimate: bump it if a fold times out (the job's .err lands in snake/logs).
CPUS="${CPUS:-2}"; MEM="${MEM:-32000}"; TIME="${TIME:-00:30:00}"

PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
LOGDIR="$PROJECT_ROOT/snake/logs"
mkdir -p "$LOGDIR"

submit() {
    local pred_file="$1"
    local out_file; out_file="$(dirname "$pred_file")/$(basename "$pred_file" | sed 's/^preds_/gstar_/')"
    local tag; tag="$(echo "${pred_file#"$RESULTS_BASE"/}" | tr '/' '_' | sed 's/\.h5ad$//')"

    if [[ "$FORCE" != "1" && -f "$out_file" ]]; then
        echo "skip (already done): $pred_file"
        return
    fi

    if [[ "$DRYRUN" == "1" ]]; then
        echo "WOULD SUBMIT  $pred_file"
        return
    fi

    local placement=""
    [[ "${GPUS:-0}" != "0" ]] && placement="-p $PART --gres=gpu:$GPUS"

    local monitor=""
    if [[ "${GPUS:-0}" != "0" ]]; then
        local gpu_raw="$LOGDIR/gstar_${tag}_gpu.csv"
        monitor="nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv -l 60 > '$gpu_raw' & NVSMI_PID=\$!; trap 'kill \$NVSMI_PID 2>/dev/null' EXIT; "
    fi

    sbatch --job-name="gstar_${tag}" --time="$TIME" -c "$CPUS" $placement --mem="${MEM}M" \
        --output="$LOGDIR/gstar_${tag}_%j.out" \
        --error="$LOGDIR/gstar_${tag}_%j.err" \
        --wrap ". '$CONDA_SH' && conda activate $CONDA_ENV && cd '$PROJECT_ROOT' || exit 1; ${monitor}python -u snake/scripts/epistemic/DE_gstar_precompute.py --pred-file '$pred_file'"
    echo "submitted     $pred_file"
}

n=0
for ds in $DATASETS; do
    base="$RESULTS_BASE/$ds"
    [[ -d "$base" ]] || { echo "skip (no dataset dir): $base"; continue; }
    for mt in $MODELS; do
        while IFS= read -r pf; do
            submit "$pf"; n=$((n+1))
        done < <(find "$base" -path "*/$RADIUS/$mt/seed_*/predictions/preds_fold*.h5ad" 2>/dev/null)
    done
done
echo "------"
echo "${DRYRUN:+(dry-run) }matched $n prediction file(s)  [RESULTS_BASE=$RESULTS_BASE MODELS='$MODELS']"
