#!/usr/bin/env bash
# snake/scripts/bio/launch_gene_variance_precompute.sh
#
# Per-gene variance references from the merged preprocessed adata (gene_variance_precompute.py).
# One SLURM job per dataset, all concurrent: the three cohorts share nothing, so there is no reason
# to run them in sequence.
#
# CPU only: no -p and no --gres, so SLURM lands the job on the `compute` partition. The cost is
# reading one merged adata and one seurat_v3 HVG pass over it, both memory-bound, hence the large
# --mem and the modest core count.
#
# Usage (cluster login node, repo root):
#   DRYRUN=1 bash snake/scripts/bio/launch_gene_variance_precompute.sh
#   bash snake/scripts/bio/launch_gene_variance_precompute.sh
#   PREPROC_ID=<preproc id> FORCE=1 bash snake/scripts/bio/launch_gene_variance_precompute.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PROJ="${PROJ:-$PROJECT_ROOT}"
PREPROC_ID="${PREPROC_ID:-v5}"
RADIUS="${RADIUS:-r75}"
N_TOP="${N_TOP:-3000}"
DATASETS="${DATASETS:-10x_TuPro TLS LUNG_CANCER}"
OUT="${OUT:-$PROJ/results/_analysis/gene_variance}"
FORCE="${FORCE:-0}"

CONDA_ENV="${CONDA_ENV:-mtenv0}"
CONDA_SH="${CONDA_SH:-$(conda info --base)/etc/profile.d/conda.sh}"
DRYRUN="${DRYRUN:-0}"

CPUS="${CPUS:-4}"; MEM="${MEM:-128000}"; TIME="${TIME:-02:00:00}"

LOGDIR="$PROJECT_ROOT/snake/logs"
mkdir -p "$LOGDIR"

n=0
for ds in $DATASETS; do
    adata="$PROJ/results/$ds/_preprocessed/$PREPROC_ID/$RADIUS/adata.h5ad"
    if [[ ! -f "$adata" ]]; then echo "skip (no adata): $adata"; continue; fi
    if [[ "$FORCE" != "1" && -f "$OUT/gene_variance_$ds.parquet" ]]; then
        echo "skip (already done): $ds"; continue
    fi
    if [[ "$DRYRUN" == "1" ]]; then echo "WOULD SUBMIT  $ds  ($adata)"; n=$((n+1)); continue; fi
    sbatch --job-name="genevar_${ds}" --time="$TIME" -c "$CPUS" --mem="${MEM}M" \
        --output="$LOGDIR/genevar_${ds}_%j.out" \
        --error="$LOGDIR/genevar_${ds}_%j.err" \
        --wrap ". '$CONDA_SH' && conda activate $CONDA_ENV && cd '$PROJECT_ROOT' || exit 1; python -u snake/scripts/bio/gene_variance_precompute.py --proj '$PROJ' --dataset '$ds' --preproc-id '$PREPROC_ID' --radius '$RADIUS' --n-top '$N_TOP' --out '$OUT'"
    echo "submitted     $ds"
    n=$((n+1))
done
echo "------"
echo "${DRYRUN:+(dry-run) }launched $n dataset(s)  [PREPROC_ID=$PREPROC_ID RADIUS=$RADIUS N_TOP=$N_TOP]"
echo "out: $OUT"
