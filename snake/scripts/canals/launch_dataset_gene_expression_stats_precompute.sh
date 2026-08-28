#!/usr/bin/env bash
# snake/scripts/canals/launch_dataset_gene_expression_stats_precompute.sh
#
# Submit one CPU-only SLURM job that builds both per-gene expression parquets, mean expression and
# the zero fraction, for the datasets in DATASETS, via
# dataset_gene_expression_stats_precompute.py:
#   results/_analysis/gene_expression_stats.parquet            (dataset, gene)
#   results/_analysis/gene_expression_stats_per_donor.parquet  (dataset, donor, gene)
# The per-donor table is what lets an expression panel carry a per-donor Student-t interval, from
# K per-donor points; the dataset-wide table collapses all patients and cannot provide them.
#
# The statistics are model-independent, a property of the preprocessed data and not of any run, so
# this reruns no training or benchmark job: it reads each dataset's merged preprocessed adata once.
#
# CPU-only: no -p/--gres is passed, so the job lands on the default `compute` partition. The
# merged adata is large and dense, hence the RAM request below.
#
# Usage (cluster login node, repo root):
#   bash snake/scripts/canals/launch_dataset_gene_expression_stats_precompute.sh
#   DRYRUN=1 bash snake/scripts/canals/launch_dataset_gene_expression_stats_precompute.sh   # preview only
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-mtenv0}"
CONDA_SH="${CONDA_SH:-$(conda info --base)/etc/profile.d/conda.sh}"
DATASETS="${DATASETS:-10x_TuPro TLS LUNG_CANCER}"
# TIME covers two reduction passes per dataset, dataset-wide and per-donor; the dominant cost is
# reading the merged adata.
CPUS="${CPUS:-4}"; MEM="${MEM:-64000}"; TIME="${TIME:-01:15:00}"
DRYRUN="${DRYRUN:-0}"

PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
LOGDIR="$PROJECT_ROOT/snake/logs"
mkdir -p "$LOGDIR"

CMD=". '$CONDA_SH' && conda activate $CONDA_ENV && cd '$PROJECT_ROOT' || exit 1; \
python -u snake/scripts/canals/dataset_gene_expression_stats_precompute.py --datasets $DATASETS"

if [[ "$DRYRUN" == "1" ]]; then
    echo "WOULD SUBMIT (CPU-only, compute partition):"
    echo "  sbatch --job-name=gene_expr_stats --time=$TIME -c $CPUS --mem=${MEM}M"
    echo "  $CMD"
    exit 0
fi

sbatch --job-name="gene_expr_stats" --time="$TIME" -c "$CPUS" --mem="${MEM}M" \
    --output="$LOGDIR/gene_expr_stats_%j.out" \
    --error="$LOGDIR/gene_expr_stats_%j.err" \
    --wrap "$CMD"
echo "submitted gene_expr_stats  [DATASETS='$DATASETS' MEM=${MEM}M TIME=$TIME]"
