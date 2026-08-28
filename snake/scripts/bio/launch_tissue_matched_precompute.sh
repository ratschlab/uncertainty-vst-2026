#!/usr/bin/env bash
# snake/scripts/bio/launch_tissue_matched_precompute.sh
#
# Submit the matched tissue-type contrast (tissue_matched_precompute.py) for every discovered
# classic run dir. One SLURM job per run dir, all concurrent.
#
# Only the two annotated cohorts are swept: LUNG_CANCER carries no per-spot tissue annotation and
# the script exits cleanly on it, so there is no reason to spend a job.
#
# The model list is the same ladder as launch_tissue_sigma_precompute.sh, and for the same reason:
#   heteroscedastic     the head the section is about
#   spot_homoscedastic  one uncertainty per spot, shared by every gene, so any contrast it shows
#                       is a spot-level effect and not a gene-specific one
#   homoscedastic       one uncertainty per gene, constant across spots, so its matched contrast
#                       is exactly zero. Built-in negative control: if it is not zero, the pairing
#                       is wrong, not the biology.
#   negativebinomial_hetero_noanchor_cp   the count arm
#
# Usage (cluster login node, repo root):
#   DRYRUN=1 bash snake/scripts/bio/launch_tissue_matched_precompute.sh
#   bash snake/scripts/bio/launch_tissue_matched_precompute.sh
#   CALIPER=0.05 FORCE=1 bash snake/scripts/bio/launch_tissue_matched_precompute.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PROJ="${PROJ:-$PROJECT_ROOT}"
RUN_ID="${RUN_ID:-pr_classic3}"
RESULTS_BASE="${RESULTS_BASE:-$PROJ/results}"
RADIUS="${RADIUS:-r75}"
OUT_SUBDIR="${OUT_SUBDIR:-tissue_matched}"
FORCE="${FORCE:-0}"
DATASETS="${DATASETS:-10x_TuPro TLS}"
MODELS="${MODELS:-heteroscedastic spot_homoscedastic homoscedastic negativebinomial_hetero_noanchor_cp}"
LAMBDAS="${LAMBDAS:-0.0}"
CALIPER="${CALIPER:-0.10}"            # fraction of the SD of the predicted mean, per (gene, slide)

CONDA_ENV="${CONDA_ENV:-mtenv0}"
CONDA_SH="${CONDA_SH:-$(conda info --base)/etc/profile.d/conda.sh}"
DRYRUN="${DRYRUN:-0}"

# CPU only: no -p and no --gres, so SLURM lands the job on the `compute` partition. Each fold's
# test block is materialised whole, plus one sorted copy of the reference tissue per pairing, in
# float32. The matching itself is a single vectorised searchsorted per (slide, tissue).
CPUS="${CPUS:-4}"; MEM="${MEM:-64000}"; TIME="${TIME:-03:00:00}"

LOGDIR="$PROJECT_ROOT/snake/logs"
mkdir -p "$LOGDIR"

submit() {
    local run_dir="$1"
    local tag; tag="$(echo "${run_dir#"$RESULTS_BASE"/}" | tr '/' '_')"
    if [[ "$FORCE" != "1" && -f "$run_dir/$OUT_SUBDIR/matched.parquet" ]]; then
        echo "skip (already done): $run_dir"; return
    fi
    if [[ "$DRYRUN" == "1" ]]; then echo "WOULD SUBMIT  $run_dir"; return; fi
    sbatch --job-name="tismatch_${tag}" --time="$TIME" -c "$CPUS" --mem="${MEM}M" \
        --output="$LOGDIR/tismatch_${tag}_%j.out" \
        --error="$LOGDIR/tismatch_${tag}_%j.err" \
        --wrap ". '$CONDA_SH' && conda activate $CONDA_ENV && cd '$PROJECT_ROOT' || exit 1; python -u snake/scripts/bio/tissue_matched_precompute.py --run-dir '$run_dir' --caliper-sd '$CALIPER'"
    echo "submitted     $run_dir"
}

n=0
for ds in $DATASETS; do
    base="$RESULTS_BASE/$ds/$RUN_ID"
    [[ -d "$base" ]] || { echo "skip (no dir): $base"; continue; }
    while IFS= read -r mj; do
        d="$(dirname "$mj")"
        [[ -d "$d/predictions" ]] || continue
        b="$(basename "$d")"
        if [[ "$b" == lambda_* ]]; then mt="$(basename "$(dirname "$d")")"; else mt="$b"; fi
        [[ "$d" == *"/$RADIUS/"* ]] || continue
        case " $MODELS " in *" $mt "*) ;; *) continue ;; esac
        if [[ -n "$LAMBDAS" && "$b" == lambda_* ]]; then
            lam="${b#lambda_}"
            case " $LAMBDAS " in *" $lam "*) ;; *) continue ;; esac
        fi
        submit "$d"; n=$((n+1))
    done < <(find "$base" -name metrics.json)
done
echo "------"
echo "${DRYRUN:+(dry-run) }matched $n run-dir(s)  [RUN_ID=$RUN_ID MODELS='$MODELS' CALIPER=$CALIPER]"
