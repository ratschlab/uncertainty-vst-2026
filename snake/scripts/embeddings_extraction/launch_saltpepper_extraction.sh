#!/usr/bin/env bash
# snake/scripts/embeddings_extraction/launch_saltpepper_extraction.sh
#
# Submit ONE SLURM GPU job that extracts patch embeddings from a dataset's slides
# corrupted with salt-and-pepper noise (the counterpart of the Gaussian-blur
# extraction). The whole amount sweep x all slides runs in one process, slide-outer:
# each slide is read and resized once, then swept over amounts.
#
# The noise is applied to the 224 model input and drawn per (seed, spot index), so a
# job is reproducible. Outputs land in isolated
# `..._saltpepper_a{amount}_sp{min}-{max}/` dirs (clean training embeddings untouched).
# There is no SKIP_EXISTING knob here, unlike the blur and radius launchers: the sweep
# re-extracts every (slide, amount), so a re-run recomputes everything.
#
# Run from the cluster checkout that holds src/: the sweep itself is
# src/embedding_extraction/patch_embed_extraction.py::run_saltpepper_sweep. Each
# (dataset x variant) is one launch:
#
# Usage (cluster login node, repo root):
#   # TuPro, monopixel salt-and-pepper (defaults: DATASET=10x_TuPro, speck 1..1):
#   bash snake/scripts/embeddings_extraction/launch_saltpepper_extraction.sh
#   # TuPro, small dust blobs (2..4 px):
#   SPECK_MIN=2 SPECK_MAX=4 bash snake/scripts/embeddings_extraction/launch_saltpepper_extraction.sh
#   # TLS, monopixel + dust:
#   DATASET=TLS bash snake/scripts/embeddings_extraction/launch_saltpepper_extraction.sh
#   DATASET=TLS SPECK_MIN=2 SPECK_MAX=4 bash snake/scripts/embeddings_extraction/launch_saltpepper_extraction.sh
#   # merged lung: the two subtypes both write into patch_embeddings/LUNG_CANCER/ (disjoint
#   # slide names -> the 36-slide dir is filled directly, no symlinking). 4 jobs for both variants:
#   DATASET=LUNG_CANCER_LUAD bash snake/scripts/embeddings_extraction/launch_saltpepper_extraction.sh
#   DATASET=LUNG_CANCER_LUSC bash snake/scripts/embeddings_extraction/launch_saltpepper_extraction.sh
#   DATASET=LUNG_CANCER_LUAD SPECK_MIN=2 SPECK_MAX=4 bash snake/scripts/embeddings_extraction/launch_saltpepper_extraction.sh
#   DATASET=LUNG_CANCER_LUSC SPECK_MIN=2 SPECK_MAX=4 bash snake/scripts/embeddings_extraction/launch_saltpepper_extraction.sh
#   # custom sweep / preview:
#   AMOUNTS="0.05 0.2" DRYRUN=1 bash snake/scripts/embeddings_extraction/launch_saltpepper_extraction.sh
set -euo pipefail

# ── Knobs (env overrides) ─────────────────────────────────────────────────────
DATASET="${DATASET:-10x_TuPro}"               # 10x_TuPro | TLS | LUNG_CANCER_LUAD | LUNG_CANCER_LUSC (driver's DATASETS)
AMOUNTS="${AMOUNTS:-0.01 0.05 0.1 0.2}"       # fraction of 224 model-input pixels corrupted per level (0.0=clean, dropped)
SALT_RATIO="${SALT_RATIO:-0.5}"               # white vs black split of the specks
SPECK_MIN="${SPECK_MIN:-1}"                    # speck side px (1 = monopixel)
SPECK_MAX="${SPECK_MAX:-1}"                    # speck side px (4 for 2..4 dust blobs)
SEED="${SEED:-0}"                              # base seed (noise per (seed, spot idx))
CONDA_ENV="${CONDA_ENV:-mtenv0}"               # env with torch + transformers + src
CONDA_SH="${CONDA_SH:-$(conda info --base)/etc/profile.d/conda.sh}"
DRYRUN="${DRYRUN:-0}"

# SLURM resources. Peak RAM is one full-res slide (TuPro 50k×54k×3 uint8 ≈ 8 GB; TLS
# ~10k×10k ≈ 0.3 GB) + the ~0.7 GB 224 cache; no full-slide float blur (noise is on the
# 224 patch), so 32 GB is generous. TIME covers all slides × N amounts.
PART="${PART:-gpu}"; GPUS="${GPUS:-1}"; CPUS="${CPUS:-4}"; MEM="${MEM:-32000}"; TIME="${TIME:-04:00:00}"

PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
LOGDIR="$PROJECT_ROOT/snake/logs"
mkdir -p "$LOGDIR"

tag="saltpepper_${DATASET}_sp${SPECK_MIN}-${SPECK_MAX}_seed${SEED}"
gpu_raw="$LOGDIR/${tag}_gpu.csv"
# nvidia-smi monitor (same as snake/Snakefile): polls every 60s; the EXIT trap kills it.
monitor="nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv -l 60 > '$gpu_raw' & NVSMI_PID=\$!; trap 'kill \$NVSMI_PID 2>/dev/null' EXIT; "

py="python -u snake/scripts/embeddings_extraction/extract_saltpepper_embeddings.py \
--dataset $DATASET --amounts $AMOUNTS --salt-ratio $SALT_RATIO \
--speck-min $SPECK_MIN --speck-max $SPECK_MAX --seed $SEED"

echo "dataset=$DATASET  amounts=[$AMOUNTS]  specks=${SPECK_MIN}-${SPECK_MAX}px  salt_ratio=$SALT_RATIO  seed=$SEED"
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
