#!/usr/bin/env bash
# snake/scripts/canals/launch_hvg_overlap_precompute.sh
#
# Submit hvg_overlap_precompute.py, CPU only: one panel job per dataset, then one spot-axis job
# per (dataset, model) that waits for it. Output: $OUT_DIR/{dataset}/.
#
# Usage (cluster login node, repo root):
#   bash snake/scripts/canals/launch_hvg_overlap_precompute.sh
#   DRYRUN=1 bash snake/scripts/canals/launch_hvg_overlap_precompute.sh
#   DATASETS=TLS bash snake/scripts/canals/launch_hvg_overlap_precompute.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PROJ="${PROJ:-$PROJECT_ROOT}"
RUN_ID="${RUN_ID:-pr_classic3}"
DATASETS="${DATASETS:-10x_TuPro TLS LUNG_CANCER}"
SPOT_MODELS="${SPOT_MODELS:-homoscedastic heteroscedastic spot_homoscedastic poisson_cp negativebinomial_hetero_noanchor_cp}"
OUT_DIR="${OUT_DIR:-$PROJ/results/_analysis/rebuttal_hvg_overlap}"
CONDA_ENV="${CONDA_ENV:-mtenv0}"
CONDA_SH="${CONDA_SH:-$(conda info --base)/etc/profile.d/conda.sh}"
DRYRUN="${DRYRUN:-0}"

PANEL_CPUS="${PANEL_CPUS:-4}"; PANEL_MEM="${PANEL_MEM:-64000}"; PANEL_TIME="${PANEL_TIME:-02:00:00}"
SPOT_CPUS="${SPOT_CPUS:-2}";   SPOT_MEM="${SPOT_MEM:-64000}";   SPOT_TIME="${SPOT_TIME:-02:00:00}"

LOGDIR="$PROJECT_ROOT/snake/logs"
mkdir -p "$LOGDIR"
SCRIPT="snake/scripts/canals/hvg_overlap_precompute.py"
PRE=". '$CONDA_SH' && conda activate $CONDA_ENV && cd '$PROJECT_ROOT' || exit 1;"

for ds in $DATASETS; do
    common="--dataset '$ds' --proj-root '$PROJ' --run-id '$RUN_ID' --out-dir '$OUT_DIR'"
    if [[ "$DRYRUN" == "1" ]]; then
        echo "WOULD SUBMIT  panel  dataset=$ds"
        for m in $SPOT_MODELS; do echo "WOULD SUBMIT  spot   dataset=$ds model=$m  (afterok panel)"; done
        continue
    fi
    jid=$(sbatch --parsable --job-name="hvgov_panel_${ds}" --time="$PANEL_TIME" -c "$PANEL_CPUS" \
        --mem="${PANEL_MEM}M" \
        --output="$LOGDIR/hvgov_panel_${ds}_%j.out" --error="$LOGDIR/hvgov_panel_${ds}_%j.err" \
        --wrap "$PRE python -u $SCRIPT $common")
    echo "submitted     panel  dataset=$ds  job=$jid"
    for m in $SPOT_MODELS; do
        sjid=$(sbatch --parsable --dependency="afterok:$jid" --job-name="hvgov_spot_${ds}_${m}" \
            --time="$SPOT_TIME" -c "$SPOT_CPUS" --mem="${SPOT_MEM}M" \
            --output="$LOGDIR/hvgov_spot_${ds}_${m}_%j.out" --error="$LOGDIR/hvgov_spot_${ds}_${m}_%j.err" \
            --wrap "$PRE python -u $SCRIPT $common --spot-model '$m'")
        echo "submitted     spot   dataset=$ds model=$m  job=$sjid (afterok:$jid)"
    done
done
echo "------"
echo "OUT_DIR=$OUT_DIR  RUN_ID=$RUN_ID"
