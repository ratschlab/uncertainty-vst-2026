#!/usr/bin/env bash
# snake/scripts/embeddings_extraction/launch_radius_extraction.sh
#
# Submit ONE SLURM GPU job that extracts patch embeddings across a sweep of crop radii
# (field-of-view distribution-shift study). The whole radius sweep x all slides runs in one
# process, slide-outer: each slide's full-res .tif is read once, then every radius re-crops,
# resizes and runs the ViT before moving to the next slide.
#
# Outputs land in plain `embeddings_radius_factor_{rf}_swapxy_True/` dirs, same naming and
# .pt format as the clean 0.5/0.75 extractions. SKIP_EXISTING (default) reuses an
# already-extracted radius (e.g. 0.5) and makes re-runs resumable. The 0.75 training set
# stays in its `..._swapxy_True_2/` dir; this sweep writes a plain 0.75 next to it.
#
# Run from the cluster checkout that holds src/: the sweep itself is
# src/embedding_extraction/patch_embed_extraction.py::run_radius_sweep. Each dataset is one
# launch:
#
# Usage (cluster login node, repo root):
#   # TuPro, full grid (default):
#   bash snake/scripts/embeddings_extraction/launch_radius_extraction.sh
#   # TLS:
#   DATASET=TLS bash snake/scripts/embeddings_extraction/launch_radius_extraction.sh
#   # custom sweep / preview:
#   RADII="1.0 1.5 2.0" DRYRUN=1 bash snake/scripts/embeddings_extraction/launch_radius_extraction.sh
#   # force re-extract everything (ignore existing .pt):
#   SKIP_EXISTING=0 bash snake/scripts/embeddings_extraction/launch_radius_extraction.sh
set -euo pipefail

# ── Knobs (env overrides) ─────────────────────────────────────────────────────
DATASET="${DATASET:-10x_TuPro}"               # 10x_TuPro | TLS (see the driver's DATASETS)
RADII="${RADII:-0.25 0.5 0.75 1.0 1.25 1.5}"  # crop radius_factor sweep
SKIP_EXISTING="${SKIP_EXISTING:-1}"           # 1 = skip radii whose .pt already exist (reuse 0.5, resumable)
CONDA_ENV="${CONDA_ENV:-mtenv0}"              # env with torch + transformers + src
CONDA_SH="${CONDA_SH:-$(conda info --base)/etc/profile.d/conda.sh}"
DRYRUN="${DRYRUN:-0}"

# SLURM resources. Peak RAM is one full-res slide held across the radius loop (TuPro
# 50k x 54k x 3 uint8 ~ 8 GB; TLS ~10k x 10k ~ 0.3 GB) + the transient per-radius crops;
# 40 GB is generous. TIME covers all slides x up to 6 radii (each radius re-crops + one ViT
# pass); the radius sweep does more ViT passes than salt-and-pepper, so 06:00:00.
PART="${PART:-gpu}"; GPUS="${GPUS:-1}"; CPUS="${CPUS:-4}"; MEM="${MEM:-40000}"; TIME="${TIME:-06:00:00}"

PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
LOGDIR="$PROJECT_ROOT/snake/logs"
mkdir -p "$LOGDIR"

tag="radiusweep_${DATASET}"
gpu_raw="$LOGDIR/${tag}_gpu.csv"
# nvidia-smi monitor (same as snake/Snakefile): polls every 60s; the EXIT trap kills it.
monitor="nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv -l 60 > '$gpu_raw' & NVSMI_PID=\$!; trap 'kill \$NVSMI_PID 2>/dev/null' EXIT; "

skip_flag=""
[[ "$SKIP_EXISTING" == "0" ]] && skip_flag=" --no-skip-existing"

py="python -u snake/scripts/embeddings_extraction/extract_radius_embeddings.py \
--dataset $DATASET --radii $RADII${skip_flag}"

echo "dataset=$DATASET  radii=[$RADII]  skip_existing=$SKIP_EXISTING"
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
