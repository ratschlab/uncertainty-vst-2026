#!/usr/bin/env bash
# snake/scripts/epistemic/launch_artifact_uncertainty_precompute.sh
#
# Submit the artifact dose-response precompute: does the model's uncertainty react to image
# artifacts (blur, salt-and-pepper)? One GPU job per (dataset, method); the jobs are fully
# independent and all submitted at once.
#
# Method -> run dir:
#   mc       = MC-dropout on one heteroscedastic model  -> $MC_RUN_ID/.../heteroscedastic/lambda_$LAM
#   ensemble = the deep ensemble's seed_*/ members      -> $ENS_RUN_ID/.../heteroscedastic
#              The ensemble path has no lambda_ segment: ensemble.lambda is constant across
#              members, so the Snakefile's member rule leaves it out of the path.
#
# Usage (cluster login node, repo root):
#   bash snake/scripts/epistemic/launch_artifact_uncertainty_precompute.sh            # every dataset x method
#   DRYRUN=1 bash snake/scripts/epistemic/launch_artifact_uncertainty_precompute.sh   # preview
#   DATASETS="TLS" METHODS="ensemble" bash snake/scripts/epistemic/launch_artifact_uncertainty_precompute.sh
set -euo pipefail

# ── Knobs (env overrides) ─────────────────────────────────────────────────────
DATASETS="${DATASETS:-10x_TuPro TLS LUNG_CANCER}"
METHODS="${METHODS:-mc ensemble}"
MC_RUN_ID="${MC_RUN_ID:-pr_classic3}"      # single-model run (corr-selected headline)
ENS_RUN_ID="${ENS_RUN_ID:-de_6}"           # deep ensemble paired with it
# A run's outputs can live under a per-run results root
# The results root can be overridden per source with MC_ROOT and ENS_ROOT.
# usually the run_id, but that is not enforced: override MC_ROOT / ENS_ROOT when it is not,
# or point them at the central results/.
MC_ROOT="${MC_ROOT:-results}"
ENS_ROOT="${ENS_ROOT:-results}"
MODEL_TYPE="${MODEL_TYPE:-heteroscedastic}"
LAM="${LAM:-0.0}"                          # lambda_ segment of the SINGLE-model run only
GENE_SET_TPL="${GENE_SET_TPL:-_%s_hvg_top3000}"   # %s = dataset
RADIUS="${RADIUS:-r75}"
T="${T:-30}"                               # MC-dropout forward passes
# Genes also written per (slide, gene, severity), the only view that carries both axes. One
# shared list for the three cohorts: a gene absent from a panel is named and skipped.
TRACE_GENES="${TRACE_GENES:-}"
BLUR="${BLUR:-0.005 0.01 0.02 0.05}"       # severity 0 = the clean X_emb, always added
SP_AMOUNTS="${SP_AMOUNTS:-0.01 0.05 0.1 0.2}"
SP_SPECKS="${SP_SPECKS:-1-1 2-4}"
CONDA_ENV="${CONDA_ENV:-mtenv0}"
CONDA_SH="${CONDA_SH:-$(conda info --base)/etc/profile.d/conda.sh}"
DRYRUN="${DRYRUN:-0}"

# GPU: the sweep is n_draws x n_spots x n_genes forward passes per severity and per fold,
# fast on a GPU and slow on CPU. Peak host RAM is the preprocessed adata plus the corrupted
# .pt files of one donor's slides.
PART="${PART:-gpu}"; GPUS="${GPUS:-1}"; CPUS="${CPUS:-4}"; MEM="${MEM:-48000}"; TIME="${TIME:-04:00:00}"

PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
LOGDIR="$PROJECT_ROOT/snake/logs"
mkdir -p "$LOGDIR"

for ds in $DATASETS; do
  gene_set="$(printf "$GENE_SET_TPL" "$ds")"
  for method in $METHODS; do
    if [[ "$method" == "mc" ]]; then
      run_dir="$MC_ROOT/$ds/$MC_RUN_ID/$gene_set/$RADIUS/$MODEL_TYPE/lambda_$LAM"
    else
      run_dir="$ENS_ROOT/$ds/$ENS_RUN_ID/$gene_set/$RADIUS/$MODEL_TYPE"
    fi

    # The emb dataset defaults to the run's own dataset inside the python script; both lung
    # sub-cohorts write into the merged patch_embeddings/LUNG_CANCER/, so that default is right
    # for all three cohorts and there is nothing to special-case here.
    py="python -u snake/scripts/epistemic/artifact_uncertainty_precompute.py \
--method $method --run-dir $run_dir --T $T \
--blur-factors $BLUR --sp-amounts $SP_AMOUNTS --sp-specks $SP_SPECKS"
    [[ -n "$TRACE_GENES" ]] && py="$py --trace-genes $TRACE_GENES"

    tag="artifactunc_${ds}_${method}"
    if [[ ! -d "$PROJECT_ROOT/$run_dir" ]]; then
      echo "skip [$tag]: run dir not found"
      echo "     $run_dir   (set MC_ROOT / ENS_ROOT / *_RUN_ID if the run lives elsewhere)"
      continue
    fi
    if [[ "$DRYRUN" == "1" ]]; then
      echo "WOULD SUBMIT  [$tag]"
      echo "  run_dir: $run_dir"
      echo "  $py"
      continue
    fi

    sbatch --job-name="$tag" --time="$TIME" -c "$CPUS" \
        -p "$PART" --gres=gpu:"$GPUS" --mem="${MEM}M" \
        --output="$LOGDIR/${tag}_%j.out" \
        --error="$LOGDIR/${tag}_%j.err" \
        --wrap ". '$CONDA_SH' && conda activate $CONDA_ENV && cd '$PROJECT_ROOT' || exit 1; $py"
    echo "submitted     [$tag]  $run_dir"
  done
done
