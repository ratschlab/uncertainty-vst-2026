#!/usr/bin/env bash
# snake/scripts/bio/launch_bio_axis_precompute.sh
#
# Submit the biological-axes export (bio_axis_precompute.py) for every discovered classic run dir
# of one sweep, across the 3 datasets. One SLURM job per run dir, all concurrent: the run dirs are
# independent, so there is no reason to bundle them into one sequential job.
#
# The default scope is deliberately narrow: heteroscedastic at lambda_0.0 only, the plain NLL
# Gaussian, which is the head the biology section reports. It is the only head with a genuinely
# per-(spot, gene) variance (homoscedastic learns one log-variance per gene, spot_homoscedastic one
# per spot), so it is the only one whose sigma can vary within a gene across a tissue. Widen with
# MODELS=... for the analytic controls (homoscedastic must give sigma_z == 0 everywhere) or the
# count heads, but note the count sigma is on a different axis (CP10k vs log1p-CP10k) and must
# never share a sigma axis with the Gaussians.
#
# Usage (cluster login node, repo root):
#   bash snake/scripts/bio/launch_bio_axis_precompute.sh              # missing runs only
#   DRYRUN=1 bash snake/scripts/bio/launch_bio_axis_precompute.sh     # preview, no sbatch
#   FORCE=1  bash snake/scripts/bio/launch_bio_axis_precompute.sh     # resubmit done runs
#   DATASETS="10x_TuPro TLS" bash snake/scripts/bio/launch_bio_axis_precompute.sh
#   MODELS="heteroscedastic homoscedastic" bash snake/scripts/bio/launch_bio_axis_precompute.sh
#   RUN_ID=<run_id> RESULTS_BASE=<path>/results bash ...
set -euo pipefail

# ── Knobs (env overrides) ─────────────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PROJ="${PROJ:-$PROJECT_ROOT}"
RUN_ID="${RUN_ID:-pr_classic3}"
RESULTS_BASE="${RESULTS_BASE:-$PROJ/results}"   # where the runs were written
RADIUS="${RADIUS:-r75}"
OUT_SUBDIR="${OUT_SUBDIR:-bio_axis}"             # must match the script's --out-dir default
FORCE="${FORCE:-0}"
DATASETS="${DATASETS:-10x_TuPro TLS LUNG_CANCER}"
MODELS="${MODELS:-heteroscedastic}"
LAMBDAS="${LAMBDAS:-0.0}"                        # lambda_0.0 IS the proper NLL (beta-NLL exponent)

CONDA_ENV="${CONDA_ENV:-mtenv0}"
CONDA_SH="${CONDA_SH:-$(conda info --base)/etc/profile.d/conda.sh}"
DRYRUN="${DRYRUN:-0}"

# CPU only: no -p and no --gres, so SLURM lands the job on the `compute` partition. Nothing here
# calls count_expected_log1p, so there is no GPU case. The script needs every gene, so a fold's
# whole test block is materialised: (n_test x n_genes) x a few layers in float32 with float64
# reductions. Drop MEM to 32000 if the queue is tight, and read the .err before raising it.
CPUS="${CPUS:-4}"; MEM="${MEM:-64000}"; TIME="${TIME:-02:00:00}"

LOGDIR="$PROJECT_ROOT/snake/logs"
mkdir -p "$LOGDIR"

submit() {
    local run_dir="$1"
    local tag; tag="$(echo "${run_dir#"$RESULTS_BASE"/}" | tr '/' '_')"

    # per_gene_tissue.parquet is the "already done" sentinel. FORCE=1 ignores it.
    if [[ "$FORCE" != "1" && -f "$run_dir/$OUT_SUBDIR/per_gene_tissue.parquet" ]]; then
        echo "skip (already done): $run_dir"
        return
    fi
    if [[ "$DRYRUN" == "1" ]]; then
        echo "WOULD SUBMIT  $run_dir"
        return
    fi

    sbatch --job-name="bioaxis_${tag}" --time="$TIME" -c "$CPUS" --mem="${MEM}M" \
        --output="$LOGDIR/bioaxis_${tag}_%j.out" \
        --error="$LOGDIR/bioaxis_${tag}_%j.err" \
        --wrap ". '$CONDA_SH' && conda activate $CONDA_ENV && cd '$PROJECT_ROOT' || exit 1; python -u snake/scripts/bio/bio_axis_precompute.py --run-dir '$run_dir'"
    echo "submitted     $run_dir"
}

n=0
for ds in $DATASETS; do
    base="$RESULTS_BASE/$ds/$RUN_ID"
    [[ -d "$base" ]] || { echo "skip (no dir): $base"; continue; }
    # Every leaf run dir holds metrics.json + a predictions/ dir. The model_type is that dir's
    # name, or its parent's when the leaf is a lambda_ sub-dir.
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
echo "${DRYRUN:+(dry-run) }matched $n run-dir(s)  [RESULTS_BASE=$RESULTS_BASE RUN_ID=$RUN_ID MODELS='$MODELS' LAMBDAS='$LAMBDAS']"
