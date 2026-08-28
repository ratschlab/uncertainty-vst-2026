#!/usr/bin/env bash
# snake/scripts/embeddings_extraction/launch_clean_extraction.sh
#
# Submit ONE SLURM GPU job that extracts the CLEAN patch embeddings for one dataset: the
# embeddings every training run consumes. This is step one of the pipeline, so run it
# before the Snakemake targets; preprocessing reads its output.
#
# Outputs land in `embeddings_radius_factor_{RADIUS}_swapxy_True${SUFFIX}/` under the
# dataset's patch_embeddings dir. SUFFIX must match datasets.{ds}.emb_subdir in
# snake/config/config.yaml, or preprocessing will not find the .pt files: 10x_TuPro and
# TLS are configured to read `_2`, LUNG_CANCER the unsuffixed dir. SKIP_EXISTING (default)
# skips slides whose .pt is already there, so a re-run resumes.
#
# LUNG_CANCER is extracted as two launches, LUNG_CANCER_LUAD and LUNG_CANCER_LUSC: the two
# sub-cohorts have different source trees but write into the same merged output dir, which
# is the one the LUNG_CANCER dataset reads.
#
# Run from the cluster checkout that holds src/: the loop itself is
# src/embedding_extraction/patch_embed_extraction.py::run_extraction.
#
# Usage (cluster login node, repo root):
#   # TLS, the configured _2 extraction:
#   DATASET=TLS SUFFIX=_2 bash snake/scripts/embeddings_extraction/launch_clean_extraction.sh
#   # TuPro:
#   DATASET=10x_TuPro SUFFIX=_2 bash snake/scripts/embeddings_extraction/launch_clean_extraction.sh
#   # NSCLC, both sub-cohorts into the merged dir:
#   DATASET=LUNG_CANCER_LUAD bash snake/scripts/embeddings_extraction/launch_clean_extraction.sh
#   DATASET=LUNG_CANCER_LUSC bash snake/scripts/embeddings_extraction/launch_clean_extraction.sh
#   # preview, or force a full re-extraction:
#   DRYRUN=1 bash snake/scripts/embeddings_extraction/launch_clean_extraction.sh
#   SKIP_EXISTING=0 bash snake/scripts/embeddings_extraction/launch_clean_extraction.sh
set -euo pipefail

# ── Knobs (env overrides) ─────────────────────────────────────────────────────
DATASET="${DATASET:-TLS}"            # TLS | 10x_TuPro | LUNG_CANCER_LUAD | LUNG_CANCER_LUSC
RADIUS="${RADIUS:-0.75}"             # crop half-width = round(spot_diam * RADIUS)
SUFFIX="${SUFFIX:-}"                 # output-dir suffix; must match emb_subdir in config.yaml
SKIP_EXISTING="${SKIP_EXISTING:-1}"  # 1 = skip slides whose .pt already exists (resumable)
CONDA_ENV="${CONDA_ENV:-mtenv0}"     # env with torch + transformers + src
CONDA_SH="${CONDA_SH:-$(conda info --base)/etc/profile.d/conda.sh}"
DRYRUN="${DRYRUN:-0}"

# SLURM resources. Peak RAM is one full-res slide plus its crops (TuPro 50k x 54k x 3 uint8
# is about 8 GB; TLS about 0.3 GB), so 40 GB is generous. One ViT pass per slide, hence a
# shorter budget than the radius sweep, which runs one pass per radius.
PART="${PART:-gpu}"; GPUS="${GPUS:-1}"; CPUS="${CPUS:-4}"; MEM="${MEM:-40000}"; TIME="${TIME:-04:00:00}"

PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
LOGDIR="$PROJECT_ROOT/snake/logs"
mkdir -p "$LOGDIR"

tag="cleanemb_${DATASET}"
gpu_raw="$LOGDIR/${tag}_gpu.csv"
# nvidia-smi monitor (same as snake/Snakefile): polls every 60s; the EXIT trap kills it.
monitor="nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv -l 60 > '$gpu_raw' & NVSMI_PID=\$!; trap 'kill \$NVSMI_PID 2>/dev/null' EXIT; "

skip_flag=""
[[ "$SKIP_EXISTING" == "0" ]] && skip_flag=" --no-skip-existing"

py="python -u snake/scripts/embeddings_extraction/extract_clean_embeddings.py \
--dataset $DATASET --radius $RADIUS --suffix '$SUFFIX'${skip_flag}"

echo "dataset=$DATASET  radius=$RADIUS  suffix='$SUFFIX'  skip_existing=$SKIP_EXISTING"
echo "placement: -p $PART --gres=gpu:$GPUS  (mem=${MEM}M, time=$TIME, cpus=$CPUS)"

if [[ "$DRYRUN" == "1" ]]; then
    echo "WOULD SUBMIT  [$tag]"
    echo "  $py"
    exit 0
fi

sbatch --job-name="$tag" --time="$TIME" -c "$CPUS" \
    -p "$PART" --gres=gpu:"$GPUS" --mem="${MEM}M" \
    --output="$LOGDIR/${tag}_%j.out" \
    --error="$LOGDIR/${tag}_%j.err" \
    --wrap ". '$CONDA_SH' && conda activate $CONDA_ENV && cd '$PROJECT_ROOT' || exit 1; ${monitor}${py}"
echo "submitted     [$tag]"
