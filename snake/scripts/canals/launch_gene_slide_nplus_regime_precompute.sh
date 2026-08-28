#!/usr/bin/env bash
# snake/scripts/canals/launch_gene_slide_nplus_regime_precompute.sh
#
# Submit one CPU-only SLURM job that builds, via gene_slide_nplus_regime_precompute.py:
#   results/_analysis/gene_slide_nplus_regime.parquet   (dataset, slide, gene, n_plus, regime)
#
# n_plus is how many spots of a slide detect a gene, and regime is the label that says what a
# statistic computed inside that pair can afford. The kNN precompute reads it to decide how many
# bins to stratify each (gene, slide) into, so this has to run BEFORE
# snake/scripts/epistemic/launch_knn_uncertainty_precompute.sh.
#
# The counts are model-independent, a property of the preprocessed data and not of any run, so
# this reruns no training job: it reads each dataset's merged preprocessed adata once.
#
# CPU-only: no -p/--gres is passed, so the job lands on the default `compute` partition. The
# merged adata is large, hence the RAM request below.
#
# Usage (cluster login node, repo root):
#   bash snake/scripts/canals/launch_gene_slide_nplus_regime_precompute.sh
#   DRYRUN=1 bash snake/scripts/canals/launch_gene_slide_nplus_regime_precompute.sh   # preview only
#   OVERWRITE=1 bash snake/scripts/canals/launch_gene_slide_nplus_regime_precompute.sh
set -euo pipefail
CONDA_ENV="${CONDA_ENV:-mtenv0}"
CONDA_SH="${CONDA_SH:-$(conda info --base)/etc/profile.d/conda.sh}"
DATASETS="${DATASETS:-10x_TuPro TLS LUNG_CANCER}"
OVERWRITE="${OVERWRITE:-0}"
# One pass over each merged adata, counting non-zeros per slide. Cheaper than the expression
# stats job next door, which also computes means through an expm1 pass.
CPUS="${CPUS:-4}"; MEM="${MEM:-64000}"; TIME="${TIME:-01:00:00}"
DRYRUN="${DRYRUN:-0}"

PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
LOGDIR="$PROJECT_ROOT/snake/logs"
mkdir -p "$LOGDIR"

EXTRA=""
[[ "$OVERWRITE" == "1" ]] && EXTRA=" --overwrite"

CMD=". '$CONDA_SH' && conda activate $CONDA_ENV && cd '$PROJECT_ROOT' || exit 1; \
python -u snake/scripts/canals/gene_slide_nplus_regime_precompute.py --datasets $DATASETS$EXTRA"

if [[ "$DRYRUN" == "1" ]]; then
    echo "WOULD SUBMIT (CPU-only, compute partition):"
    echo "  sbatch --job-name=nplus_regime --time=$TIME -c $CPUS --mem=${MEM}M"
    echo "  $CMD"
    exit 0
fi

sbatch --job-name="nplus_regime" --time="$TIME" -c "$CPUS" --mem="${MEM}M" \
    --output="$LOGDIR/nplus_regime_%j.out" \
    --error="$LOGDIR/nplus_regime_%j.err" \
    --wrap "$CMD"
echo "submitted nplus_regime  [DATASETS='$DATASETS' MEM=${MEM}M TIME=$TIME]"
