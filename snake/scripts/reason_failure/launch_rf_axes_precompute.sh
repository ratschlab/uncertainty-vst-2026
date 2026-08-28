#!/usr/bin/env bash
# snake/scripts/reason_failure/launch_rf_axes_precompute.sh
#
# Submit D3 / D5 / D6 (rf_axes_precompute.py) for every discovered CLASSIC run dir of one sweep,
# across the 3 datasets. ONE SLURM job per run dir, all concurrent: the run dirs are independent,
# so there is no reason to bundle them into one sequential job.
#
# Default scope = the five heads of the axes figure, at lambda_0.0 (the plain NLL; the lambda
# sweep is beta-NLL and 0.0 is the proper likelihood):
#   heteroscedastic                      the principal head, the only Gaussian whose sigma^2(x) is
#                                        a free output with no imposed link to the mean, hence the
#                                        only one that can carry the argument
#   homoscedastic / spot_homoscedastic   the analytic controls: one sigma per GENE, one per SPOT.
#                                        homoscedastic must return share_gene = 1 on D5's
#                                        log_sigma, the built-in sanity check of the section
#   poisson_cp / negativebinomial_hetero_noanchor_cp   the count controls, whose variance is tied
#                                        to the mean by the shape of the likelihood
# msemeanonly is skipped by the script itself (no uncertainty head), with a summary.json saying so.
#
# A count head's sigma is on its own count scale and a Gaussian's on log1p-CP10k. Every statistic
# produced here is a rank (D3), a ratio (D6) or a variance share of a log (D5), all invariant to a
# per-run constant, so they are comparable across families. The sigma values are not, and must
# never share an axis.
#
# CPU-only: no -p / --gres at all, so SLURM lands the job on the `compute` partition, the
# convention on this cluster. Nothing here calls count_expected_log1p, so there is no GPU case.
# Each fold's test block is materialised whole (n_test x n_genes, four float32 arrays) and then
# worked on one slide at a time in float64; 64 GB is generous headroom.
#
# Usage (cluster login node, repo root):
#   bash snake/scripts/reason_failure/launch_rf_axes_precompute.sh              # missing runs only
#   DRYRUN=1 bash snake/scripts/reason_failure/launch_rf_axes_precompute.sh     # preview, no sbatch
#   FORCE=1  bash snake/scripts/reason_failure/launch_rf_axes_precompute.sh     # resubmit done runs
#   DATASETS="10x_TuPro" MODELS="heteroscedastic" bash snake/scripts/reason_failure/launch_rf_axes_precompute.sh
# Other knobs: PROJ, RUN_ID, RESULTS_BASE, RADIUS, LAMBDAS, OUT_SUBDIR, CONDA_ENV, CONDA_SH,
# CPUS, MEM, TIME.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PROJ="${PROJ:-$PROJECT_ROOT}"
RUN_ID="${RUN_ID:-pr_classic3}"
RESULTS_BASE="${RESULTS_BASE:-$PROJ/results}"   # where the runs were written
RADIUS="${RADIUS:-r75}"
OUT_SUBDIR="${OUT_SUBDIR:-reason_failure}"       # must match the script's --out-dir default
FORCE="${FORCE:-0}"
DATASETS="${DATASETS:-10x_TuPro TLS LUNG_CANCER}"
MODELS="${MODELS:-heteroscedastic homoscedastic spot_homoscedastic poisson_cp negativebinomial_hetero_noanchor_cp}"
LAMBDAS="${LAMBDAS:-0.0}"

CONDA_ENV="${CONDA_ENV:-mtenv0}"
CONDA_SH="${CONDA_SH:-$(conda info --base)/etc/profile.d/conda.sh}"
DRYRUN="${DRYRUN:-0}"
CPUS="${CPUS:-4}"; MEM="${MEM:-64000}"; TIME="${TIME:-04:00:00}"

LOGDIR="$PROJECT_ROOT/snake/logs"
mkdir -p "$LOGDIR"

submit() {
    local run_dir="$1"
    local tag; tag="$(echo "${run_dir#"$RESULTS_BASE"/}" | tr '/' '_')"

    # Sentinel = spot_slide.parquet, the last parquet the script writes, so a job killed mid-write
    # is re-submitted rather than skipped as "already done".
    if [[ "$FORCE" != "1" && -f "$run_dir/$OUT_SUBDIR/spot_slide.parquet" ]]; then
        echo "skip (already done): $run_dir"
        return
    fi
    if [[ "$DRYRUN" == "1" ]]; then
        echo "WOULD SUBMIT  $run_dir"
        return
    fi

    sbatch --job-name="rfaxes_${tag}" --time="$TIME" -c "$CPUS" --mem="${MEM}M" \
        --output="$LOGDIR/rfaxes_${tag}_%j.out" \
        --error="$LOGDIR/rfaxes_${tag}_%j.err" \
        --wrap ". '$CONDA_SH' && conda activate $CONDA_ENV && cd '$PROJECT_ROOT' || exit 1; python -u snake/scripts/reason_failure/rf_axes_precompute.py --run-dir '$run_dir'"
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
