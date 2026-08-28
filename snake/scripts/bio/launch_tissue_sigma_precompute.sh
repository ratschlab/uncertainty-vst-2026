#!/usr/bin/env bash
# snake/scripts/bio/launch_tissue_sigma_precompute.sh
#
# Submit the per-(gene, slide, tissue) uncertainty export (tissue_sigma_precompute.py) for every
# discovered classic run dir. One SLURM job per run dir, all concurrent.
#
# The default model list is a ladder, and each rung is there for a reason:
#   heteroscedastic       sigma per (spot, gene), the head the analysis is about
#   spot_homoscedastic    sigma per spot, shared across genes, so any tissue contrast it shows is a
#                         pure spot-level effect, identical for every gene
#   homoscedastic         sigma per gene, constant across spots, so its tissue contrast is exactly
#                         zero by construction. Built-in negative control; if it is not zero, the
#                         pipeline is wrong, not the biology.
#   negativebinomial_hetero_noanchor_cp   carries `dispersion`, so alpha is exported too. For the
#                         NB, cv^2 = 1/mu + alpha, so alpha is the mean-free part of the relative
#                         spread. It is the one quantity here that needs no de-confounding.
# LUNG_CANCER stays in DATASETS even though it has no per-spot tissue annotation; the note above
# DATASETS says why.
#
# Usage (cluster login node, repo root):
#   DRYRUN=1 bash snake/scripts/bio/launch_tissue_sigma_precompute.sh
#   bash snake/scripts/bio/launch_tissue_sigma_precompute.sh
#   FORCE=1 MODELS="heteroscedastic" bash snake/scripts/bio/launch_tissue_sigma_precompute.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PROJ="${PROJ:-$PROJECT_ROOT}"
RUN_ID="${RUN_ID:-pr_classic3}"
RESULTS_BASE="${RESULTS_BASE:-$PROJ/results}"
RADIUS="${RADIUS:-r75}"
OUT_SUBDIR="${OUT_SUBDIR:-tissue_sigma}"
FORCE="${FORCE:-0}"
# LUNG_CANCER is included even though it has no per-spot tissue annotation: the variance shares and
# the within-gene / within-spot spreads need no label. Only its per-(gene, slide, tissue) table
# comes out empty.
DATASETS="${DATASETS:-10x_TuPro TLS LUNG_CANCER}"
MODELS="${MODELS:-heteroscedastic spot_homoscedastic homoscedastic negativebinomial_hetero_noanchor_cp}"
LAMBDAS="${LAMBDAS:-0.0}"                 # lambda_0.0 IS the proper NLL; count heads have no lambda leaf

CONDA_ENV="${CONDA_ENV:-mtenv0}"
CONDA_SH="${CONDA_SH:-$(conda info --base)/etc/profile.d/conda.sh}"
DRYRUN="${DRYRUN:-0}"

# CPU only: no -p and no --gres, so SLURM lands the job on the `compute` partition. Each fold's
# full test block is materialised (n_test x n_genes) in float32 with float64 accumulators, plus one
# int32 rank matrix per slide.
CPUS="${CPUS:-4}"; MEM="${MEM:-64000}"; TIME="${TIME:-02:00:00}"

LOGDIR="$PROJECT_ROOT/snake/logs"
mkdir -p "$LOGDIR"

submit() {
    local run_dir="$1"
    local tag; tag="$(echo "${run_dir#"$RESULTS_BASE"/}" | tr '/' '_')"
    if [[ "$FORCE" != "1" && -f "$run_dir/$OUT_SUBDIR/per_gene_tissue.parquet" ]]; then
        echo "skip (already done): $run_dir"; return
    fi
    if [[ "$DRYRUN" == "1" ]]; then echo "WOULD SUBMIT  $run_dir"; return; fi
    sbatch --job-name="tissig_${tag}" --time="$TIME" -c "$CPUS" --mem="${MEM}M" \
        --output="$LOGDIR/tissig_${tag}_%j.out" \
        --error="$LOGDIR/tissig_${tag}_%j.err" \
        --wrap ". '$CONDA_SH' && conda activate $CONDA_ENV && cd '$PROJECT_ROOT' || exit 1; python -u snake/scripts/bio/tissue_sigma_precompute.py --run-dir '$run_dir'"
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
echo "${DRYRUN:+(dry-run) }matched $n run-dir(s)  [RESULTS_BASE=$RESULTS_BASE RUN_ID=$RUN_ID MODELS='$MODELS']"
