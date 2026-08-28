#!/usr/bin/env bash
# snake/scripts/embeddings_extraction/launch_blur_extraction.sh
#
# Submit ONE SLURM GPU job that extracts patch embeddings from a dataset's slides degraded
# with a whole-slide Gaussian blur (slide-quality distribution-shift study), the blur
# counterpart of launch_saltpepper_extraction.sh. The whole blur sweep x all slides runs in
# one process, slide-outer: each slide's full-res .tif is read once, then every level blurs
# a copy of it, re-crops, resizes and runs the ViT before moving to the next slide.
#
# sigma_px = blur_factor * spot_diameter_fullres (per slide), applied to the whole slide
# before cropping, so a given blur_factor smears the same physical extent on slides of
# different resolution. bf=0.0 skips the filter (clean control). Outputs land in isolated
# `..._blurry_bf{bf}/` dirs (clean training embeddings untouched), same .pt format as the
# clean extractions. SKIP_EXISTING (default) makes re-runs resumable.
#
# LUNG: the merged LUNG_CANCER has no image/meta tree of its own, so it is launched as its
# two source subtypes, LUNG_CANCER_LUAD (22 slides) + LUNG_CANCER_LUSC (14), and both write
# into the merged patch_embeddings/LUNG_CANCER/... dir. Slide names are disjoint, so the two
# jobs together fill the 36-slide dir directly (no symlinking afterwards).
#
# Run from the cluster checkout that holds src/: the sweep itself is
# src/embedding_extraction/patch_embed_extraction.py::run_blur_sweep. Each dataset is one
# launch:
#
# Usage (cluster login node, repo root):
#   # merged lung, both subtypes (the two jobs fill patch_embeddings/LUNG_CANCER/):
#   DATASET=LUNG_CANCER_LUAD bash snake/scripts/embeddings_extraction/launch_blur_extraction.sh
#   DATASET=LUNG_CANCER_LUSC bash snake/scripts/embeddings_extraction/launch_blur_extraction.sh
#   # TuPro / TLS (re-extraction; already on disk from notebooks 1b.1/1b.2):
#   DATASET=10x_TuPro MEM=64000 bash snake/scripts/embeddings_extraction/launch_blur_extraction.sh
#   # custom sweep / preview:
#   BLUR_FACTORS="0.02 0.05" DRYRUN=1 bash snake/scripts/embeddings_extraction/launch_blur_extraction.sh
#   # force re-extract everything (ignore existing .pt):
#   SKIP_EXISTING=0 bash snake/scripts/embeddings_extraction/launch_blur_extraction.sh
set -euo pipefail

# ── Knobs (env overrides) ─────────────────────────────────────────────────────
DATASET="${DATASET:-LUNG_CANCER_LUAD}"                 # 10x_TuPro | TLS | LUNG_CANCER_LUAD | LUNG_CANCER_LUSC
BLUR_FACTORS="${BLUR_FACTORS:-0.0 0.005 0.01 0.02 0.05}"  # sigma = bf * spot_diam (0.0 = clean control)
SKIP_EXISTING="${SKIP_EXISTING:-1}"           # 1 = skip (slide, bf) whose .pt already exists (resumable)
CONDA_ENV="${CONDA_ENV:-mtenv0}"              # env with torch + transformers + src
CONDA_SH="${CONDA_SH:-$(conda info --base)/etc/profile.d/conda.sh}"
DRYRUN="${DRYRUN:-0}"

# SLURM resources. Unlike salt-and-pepper (noise on the 224 patch), the blur is a FULL-SLIDE
# filter, so peak RAM ~ 2.33x the slide: pristine + blurred copy + one float32 channel.
# LUNG ~17.6k^2 x3 ~ 0.9 GB -> ~3 GB; TLS ~10k^2 -> ~1 GB; TuPro 50k x 54k x3 ~ 8 GB -> ~27 GB
# (pass MEM=64000 for TuPro). TIME covers all slides x 5 levels (ViT once per level).
PART="${PART:-gpu}"; GPUS="${GPUS:-1}"; CPUS="${CPUS:-4}"; MEM="${MEM:-48000}"; TIME="${TIME:-08:00:00}"

PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
LOGDIR="$PROJECT_ROOT/snake/logs"
mkdir -p "$LOGDIR"

tag="blur_${DATASET}"
gpu_raw="$LOGDIR/${tag}_gpu.csv"
# nvidia-smi monitor (same as snake/Snakefile): polls every 60s; the EXIT trap kills it.
monitor="nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv -l 60 > '$gpu_raw' & NVSMI_PID=\$!; trap 'kill \$NVSMI_PID 2>/dev/null' EXIT; "

skip_flag=""
[[ "$SKIP_EXISTING" == "0" ]] && skip_flag=" --no-skip-existing"

py="python -u snake/scripts/embeddings_extraction/extract_blur_embeddings.py \
--dataset $DATASET --blur-factors $BLUR_FACTORS${skip_flag}"

echo "dataset=$DATASET  blur_factors=[$BLUR_FACTORS]  skip_existing=$SKIP_EXISTING"
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
