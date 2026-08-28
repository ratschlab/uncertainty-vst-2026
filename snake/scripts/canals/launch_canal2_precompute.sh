#!/usr/bin/env bash
# snake/scripts/canals/launch_canal2_precompute.sh
#
# Submit Canal 2 uncertainty-vs-error precompute jobs (canal2_uncertainty_precompute.py --run-dir)
# for every classic model run found under RESULTS_BASE/<dataset>/RUN_ID: any directory holding both
# metrics.json and predictions/, kept only if its path carries the RADIUS segment and its model_type
# is in MODELS. One SLURM job per run-dir, all concurrent, CPU-only; each job writes
# <run-dir>/$OUT_SUBDIR/. Every knob below is an environment-variable override.
#
# Usage (cluster login node, repo root):
#   bash snake/scripts/canals/launch_canal2_precompute.sh                 # submit missing runs only
#   DRYRUN=1 bash snake/scripts/canals/launch_canal2_precompute.sh        # preview only, no sbatch
#   FORCE=1 bash snake/scripts/canals/launch_canal2_precompute.sh         # resubmit even already-done runs
#   MODELS="heteroscedastic poisson" bash snake/scripts/canals/launch_canal2_precompute.sh
set -euo pipefail

# ── Knobs (env overrides) ─────────────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
RESULTS_BASE="${RESULTS_BASE:-$PROJECT_ROOT/results}"
RUN_ID="${RUN_ID:-pr_classic3}"
RADIUS="${RADIUS:-r75}"
OUT_SUBDIR="${OUT_SUBDIR:-canal2_uncertainty}"   # must match the script's --out-dir default
CONVENTION="${CONVENTION:-v3_native_scale}"      # a summary.json without it is STALE, not done
FORCE="${FORCE:-0}"                              # 1 = resubmit even an up-to-date summary.json
DATASETS="${DATASETS:-10x_TuPro TLS LUNG_CANCER}"
# msemeanonly is deliberately excluded: is_probabilistic=False (src/models/specs.py), no pred_var
# layer is ever saved for it, so there is no uncertainty to test. The script no-ops gracefully if
# launched on it directly, but the job would be a guaranteed no-op.
# homoscedastic is out of the default list for a weaker reason: its per-gene constant variance is
# degenerate on the gene axis only, and the script's spot axis (one group per (slide, spot),
# spot_partial.parquet) is precisely where that head can be measured. Same for
# negativebinomial_genedisp_cp. Pass MODELS="... homoscedastic negativebinomial_genedisp_cp ..."
# when the spot axis is wanted.
MODELS="${MODELS:-heteroscedastic spot_homoscedastic poisson_cp negativebinomial_hetero_noanchor_cp negativebinomial_spotdisp_cp negativebinomial_genedisp_cp}"
LAMBDAS="${LAMBDAS:-}"                            # optional: restrict which lambda_ sub-dirs (space-sep
                                                  # values, e.g. "0.0" or "0.0 1.0"). Empty => all lambdas.
                                                  # Count models have no lambda_ leaf and ALWAYS pass this
                                                  # filter (it only gates the lambda-bearing Gaussian dirs).
                                                  # Same knob, same semantics as launch_epistemic.sh's.
                                                  # Usually worth setting: a lambda sweep yields one run-dir
                                                  # per (Gaussian head, lambda), so an unfiltered launch
                                                  # multiplies the jobs. To pair this against a post-hoc
                                                  # epistemic campaign, pass the one lambda those summaries
                                                  # were written for; otherwise the own-head column covers
                                                  # lambdas the epistemic column does not, and the two are
                                                  # not the same models.
CONDA_ENV="${CONDA_ENV:-mtenv0}"                  # env with torch + scipy + anndata + src
# conda.sh to source INSIDE the job (the same one snake/Snakefile and
# launch_epistemic.sh use). The job runs under /bin/sh, so we source it with `.` (POSIX).
CONDA_SH="${CONDA_SH:-$(conda info --base)/etc/profile.d/conda.sh}"
DRYRUN="${DRYRUN:-0}"

# CPU-only: passing no -p/--gres at all lands the job on the compute partition. The heaviest
# step per run is building a long per-spot table (tens of millions of rows on the densest
# dataset) plus a rank-based groupby cascade, so memory is generous and cores are few (pandas
# groupby is mostly single-threaded; extra cores mainly help numpy elementwise ops). Bump TIME
# if a run times out, after reading its .err.
CPUS="${CPUS:-4}"; MEM="${MEM:-128000}"; TIME="${TIME:-04:00:00}"

LOGDIR="$PROJECT_ROOT/snake/logs"
mkdir -p "$LOGDIR"

# The n+ regime is derived inside canal2_uncertainty_precompute.py from each run's own
# positive-spot counts. This launcher has no external regime-table dependency.

submit() {
    local run_dir="$1"
    local tag; tag="$(echo "${run_dir#"$RESULTS_BASE"/}" | tr '/' '_')"

    local sj="$run_dir/$OUT_SUBDIR/summary.json"
    if [[ "$FORCE" != "1" && -f "$sj" ]]; then
        # "Done" means done under the current convention. A summary.json that lacks $CONVENTION
        # holds different column names on a converted scale and every reader discards it, so
        # skipping it would leave a hole that looks exactly like a run that was never launched.
        # That is how a whole sweep silently goes missing from a table. Resubmit instead.
        if grep -q "$CONVENTION" "$sj"; then
            echo "skip (already done): $run_dir"
            return
        fi
        echo "stale (no $CONVENTION): $run_dir"
    fi

    if [[ "$DRYRUN" == "1" ]]; then
        echo "WOULD SUBMIT  $run_dir"
        return
    fi

    sbatch --job-name="canal2_${tag}" --time="$TIME" -c "$CPUS" --mem="${MEM}M" \
        --output="$LOGDIR/canal2_${tag}_%j.out" \
        --error="$LOGDIR/canal2_${tag}_%j.err" \
        --wrap ". '$CONDA_SH' && conda activate $CONDA_ENV && cd '$PROJECT_ROOT' || exit 1; python -u snake/scripts/canals/canal2_uncertainty_precompute.py --run-dir '$run_dir'"
    echo "submitted     $run_dir"
}

n=0
for ds in $DATASETS; do
    base="$RESULTS_BASE/$ds/$RUN_ID"
    [[ -d "$base" ]] || { echo "skip (no dir): $base"; continue; }
    # Every leaf run-dir holds metrics.json + a predictions/ dir. The model_type is the
    # dir name (or its parent when the leaf is a lambda_ sub-dir, e.g. heteroscedastic).
    while IFS= read -r mj; do
        d="$(dirname "$mj")"
        [[ -d "$d/predictions" ]] || continue
        b="$(basename "$d")"
        if [[ "$b" == lambda_* ]]; then mt="$(basename "$(dirname "$d")")"; else mt="$b"; fi
        # require the radius segment + a requested model_type
        [[ "$d" == *"/$RADIUS/"* ]] || continue
        case " $MODELS " in *" $mt "*) ;; *) continue ;; esac
        # LAMBDAS filter (Gaussian lambda_ dirs only): skip a lambda not in the list. A count leaf
        # has no lambda_ segment and always passes.
        if [[ -n "$LAMBDAS" && "$b" == lambda_* ]]; then
            lam="${b#lambda_}"
            case " $LAMBDAS " in *" $lam "*) ;; *) continue ;; esac
        fi
        submit "$d"; n=$((n+1))
    done < <(find "$base" -name metrics.json)
done
echo "------"
# Not ${DRYRUN:+...}: DRYRUN defaults to the string "0", which is non-empty, so that form prints
# "(dry-run)" on a real launch. This line must never lie about what happened.
[[ "$DRYRUN" == "1" ]] && pfx="(dry-run) " || pfx=""
echo "${pfx}matched $n run-dir(s)  [RESULTS_BASE=$RESULTS_BASE RUN_ID=$RUN_ID MODELS='$MODELS' LAMBDAS='$LAMBDAS']"
