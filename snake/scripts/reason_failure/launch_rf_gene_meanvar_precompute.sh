#!/usr/bin/env bash
# snake/scripts/reason_failure/launch_rf_gene_meanvar_precompute.sh
#
# Submit D1 (rf_gene_meanvar_precompute.py): the mean-variance coupling in the data, per
# (slide, gene), on the two target scales. Model-free, it reads only the merged preprocessed
# adata, so it does not depend on any run_id and is computed once for the three cohorts.
#
# ONE job for all three datasets, unlike the other launchers here: the three write into the same
# parquets under results/_analysis/, so splitting them would need a merge step and buy nothing.
# The pass is I/O bound, one contiguous slide slice at a time.
#
# CPU-only: no -p / --gres at all, so SLURM lands the job on the `compute` partition, the
# convention on this cluster. Peak RSS is one slide's rows x the full gene axis, a few
# hundred MB, so 32 GB is already generous.
#
# Usage (cluster login node, repo root):
#   bash snake/scripts/reason_failure/launch_rf_gene_meanvar_precompute.sh
#   DRYRUN=1 bash snake/scripts/reason_failure/launch_rf_gene_meanvar_precompute.sh
#   DATASETS="10x_TuPro TLS" bash snake/scripts/reason_failure/launch_rf_gene_meanvar_precompute.sh
# Other knobs: PROJ, PREPROC_ID, RADIUS, PANEL, CONDA_ENV, CONDA_SH, CPUS, MEM, TIME.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PROJ="${PROJ:-$PROJECT_ROOT}"
DATASETS="${DATASETS:-10x_TuPro TLS LUNG_CANCER}"
PREPROC_ID="${PREPROC_ID:-v5}"
RADIUS="${RADIUS:-r75}"
PANEL="${PANEL:-hvg_top3000.txt}"

CONDA_ENV="${CONDA_ENV:-mtenv0}"
CONDA_SH="${CONDA_SH:-$(conda info --base)/etc/profile.d/conda.sh}"
DRYRUN="${DRYRUN:-0}"
CPUS="${CPUS:-4}"; MEM="${MEM:-32000}"; TIME="${TIME:-02:00:00}"

LOGDIR="$PROJECT_ROOT/snake/logs"
mkdir -p "$LOGDIR"

CMD="python -u snake/scripts/reason_failure/rf_gene_meanvar_precompute.py \
--project-root '$PROJ' --datasets $DATASETS --preproc-id '$PREPROC_ID' --radius '$RADIUS' \
--panel '$PANEL'"

if [[ "$DRYRUN" == "1" ]]; then
    echo "WOULD SUBMIT (1 job): $CMD"
    exit 0
fi

sbatch --job-name="rf_meanvar" --time="$TIME" -c "$CPUS" --mem="${MEM}M" \
    --output="$LOGDIR/rf_meanvar_%j.out" \
    --error="$LOGDIR/rf_meanvar_%j.err" \
    --wrap ". '$CONDA_SH' && conda activate $CONDA_ENV && cd '$PROJECT_ROOT' || exit 1; $CMD"
echo "submitted  datasets='$DATASETS'  preproc_id=$PREPROC_ID  radius=$RADIUS  panel=$PANEL"
echo "outputs -> $PROJ/results/_analysis/rf_gene_meanvar.parquet"
echo "           $PROJ/results/_analysis/rf_meanvar_spearman.parquet"
