#!/usr/bin/env bash
# snake/scripts/canals/launch_canal2_uform_precompute.sh
#
# Canal 2 under the two alternative uncertainty forms, for the robustness appendix:
#
#   cv              u = sqrt(variance) / mu, the coefficient of variation
#   one_plus_alpha  u = 1 + alpha, the NB dispersion read alone
#
# Its own launcher rather than a knob on launch_canal2_precompute.sh because the two forms do not
# share a model list. cv applies to every head that emits a variance, while one_plus_alpha needs a
# fitted dispersion and therefore exists for the NB heads only. The script writes a graceful
# not_applicable sentinel for the others, but there is no reason to spend a job slot on a
# guaranteed no-op. Keeping the pairing (form -> heads) here makes the launch self-documenting.
#
# Each job writes to its own directory, <run-dir>/canal2_uncertainty_<form>/, so the reference
# artifact <run-dir>/canal2_uncertainty/ that every figure reads is never touched.
#
# One SLURM job per (dataset, head, form), all concurrent, CPU-only, same resource sizing as
# launch_canal2_precompute.sh. Every knob below is an environment-variable override.
#
# Usage (cluster login node, repo root):
#   bash snake/scripts/canals/launch_canal2_uform_precompute.sh                 # submit missing only
#   DRYRUN=1 bash snake/scripts/canals/launch_canal2_uform_precompute.sh        # preview, no sbatch
#   FORCE=1  bash snake/scripts/canals/launch_canal2_uform_precompute.sh        # resubmit done ones
#   FORMS="cv" bash snake/scripts/canals/launch_canal2_uform_precompute.sh      # one form only
set -euo pipefail

# ── Knobs (env overrides) ─────────────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
RESULTS_BASE="${RESULTS_BASE:-$PROJECT_ROOT/results}"
RUN_ID="${RUN_ID:-pr_classic3}"
RADIUS="${RADIUS:-r75}"
DATASETS="${DATASETS:-10x_TuPro TLS LUNG_CANCER}"
FORMS="${FORMS:-cv one_plus_alpha}"
FORCE="${FORCE:-0}"
DRYRUN="${DRYRUN:-0}"

# The five heads the reference sd runs cover, so the cv row of the table has exactly the rows the
# reference row has. Three of them are degenerate under cv for the same reason they are degenerate
# under sd: u stays a monotone function of mu inside the group. They are still submitted, because a
# missing row would read as missing data rather than as a degenerate head.
CV_MODELS="${CV_MODELS:-homoscedastic heteroscedastic spot_homoscedastic poisson_cp negativebinomial_hetero_noanchor_cp}"
# NB heads only: 1 + alpha needs layers["dispersion"].
ALPHA_MODELS="${ALPHA_MODELS:-negativebinomial_hetero_noanchor_cp}"

# The Gaussians carry a beta-NLL sweep; lambda_0.0 is the proper NLL and the one the figures read.
# The beta variation of the robustness table comes from the sd artifacts already computed at the
# other lambdas, so re-sweeping beta here would multiply the launch for nothing.
LAMBDAS="${LAMBDAS:-0.0}"

CONDA_ENV="${CONDA_ENV:-mtenv0}"
CONDA_SH="${CONDA_SH:-$(conda info --base)/etc/profile.d/conda.sh}"
CPUS="${CPUS:-4}"; MEM="${MEM:-128000}"; TIME="${TIME:-04:00:00}"

LOGDIR="$PROJECT_ROOT/snake/logs"
mkdir -p "$LOGDIR"

models_for_form() {
    case "$1" in
        cv)             echo "$CV_MODELS" ;;
        one_plus_alpha) echo "$ALPHA_MODELS" ;;
        *) echo "unknown form '$1' (expected cv or one_plus_alpha)" >&2; exit 1 ;;
    esac
}

n=0
for form in $FORMS; do
    models="$(models_for_form "$form")"
    out_sub="canal2_uncertainty_${form}"
    for ds in $DATASETS; do
        base="$RESULTS_BASE/$ds/$RUN_ID"
        [[ -d "$base" ]] || { echo "skip (no dir): $base"; continue; }
        while IFS= read -r mj; do
            d="$(dirname "$mj")"
            [[ -d "$d/predictions" ]] || continue
            b="$(basename "$d")"
            if [[ "$b" == lambda_* ]]; then mt="$(basename "$(dirname "$d")")"; else mt="$b"; fi
            [[ "$d" == *"/$RADIUS/"* ]] || continue
            case " $models " in *" $mt "*) ;; *) continue ;; esac
            if [[ -n "$LAMBDAS" && "$b" == lambda_* ]]; then
                lam="${b#lambda_}"
                case " $LAMBDAS " in *" $lam "*) ;; *) continue ;; esac
            fi
            if [[ "$FORCE" != "1" && -f "$d/$out_sub/summary.json" ]]; then
                echo "skip (already done): [$form] $d"
                continue
            fi
            tag="$(echo "${d#"$RESULTS_BASE"/}" | tr '/' '_')"
            if [[ "$DRYRUN" == "1" ]]; then
                echo "WOULD SUBMIT  [$form] $d"
                n=$((n+1)); continue
            fi
            sbatch --job-name="canal2${form}_${tag}" --time="$TIME" -c "$CPUS" --mem="${MEM}M" \
                --output="$LOGDIR/canal2_${form}_${tag}_%j.out" \
                --error="$LOGDIR/canal2_${form}_${tag}_%j.err" \
                --wrap ". '$CONDA_SH' && conda activate $CONDA_ENV && cd '$PROJECT_ROOT' || exit 1; python -u snake/scripts/canals/canal2_uncertainty_precompute.py --run-dir '$d' --uncertainty-form '$form'"
            echo "submitted     [$form] $d"
            n=$((n+1))
        done < <(find "$base" -name metrics.json)
    done
done
echo "------"
# Not ${DRYRUN:+...}: DRYRUN defaults to the string "0", which is non-empty, so that form prints
# "(dry-run)" on a real launch. This line must never lie about what happened.
[[ "$DRYRUN" == "1" ]] && pfx="(dry-run) " || pfx=""
echo "${pfx}matched $n job(s)  [RESULTS_BASE=$RESULTS_BASE RUN_ID=$RUN_ID FORMS='$FORMS' LAMBDAS='$LAMBDAS']"
