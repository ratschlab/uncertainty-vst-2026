#!/usr/bin/env bash
# snake/scripts/epistemic/launch_epistemic.sh
#
# Submit post-hoc EPISTEMIC jobs (MC-dropout + last-layer Laplace) for CLASSIC runs.
# One SLURM job PER (leaf run-dir, method) PAIR -- mc and laplace are ALWAYS
# submitted as INDEPENDENT jobs (METHODS selects which methods to submit, default
# "mc laplace" = both, each its own job), writing <run-dir>/mc_dropout/ and/or
# <run-dir>/laplace/. The GPU is OPTIONAL: GPUS=0 -> a plain CPU job (no --gres, no
# nvidia-smi), the right shape for both, whose forward and algebra are CPU-cheap.
#
# WHY SEPARATE JOBS, never chained with &&: MC-dropout and Laplace have wildly
# different, and very different-variance, wall-clock costs. MC-dropout is cheap.
# Laplace's dominant cost is a per-gene GGN fit (an eigendecomposition plus a
# (H+1, n_train)x(n_train, H+1) matmul, once per gene, in a serial Python loop)
# whose time scales with n_train and is markedly worse for NB (dispersion head)
# than Poisson, and for the datasets with more train spots. Chained, (a) the job
# sits blocked on Laplace long after MC finished, wasting the allocation, and
# (b) a Laplace timeout LOSES the entire Laplace output, since its
# test_summaries.h5ad is written ONCE after ALL folds, even though MC's own output
# was already written safely earlier in the SAME job.
#
# Handles the Gaussian and the count families alike -- the scripts auto-detect the
# family from metrics.json. The model_type is read from the directory tree, so the
# lambda_ sub-dir of the heteroscedastic-family models is matched correctly.
#
# IMPORTANT — run from the repo root, the one holding the src/ and
# scripts to run; --run-dir then points at wherever the results actually
# live. The script CODE comes from this tree, the checkpoints and data from
# RESULTS_BASE — they are independent.
#
# Usage (cluster login node, repo root):
#   # central results, both methods, each its own job with its own default TIME:
#   bash snake/scripts/epistemic/launch_epistemic.sh
#   # results written somewhere else:
#   RESULTS_BASE=/path/to/results bash snake/scripts/epistemic/launch_epistemic.sh
#   # preview only (no sbatch):
#   DRYRUN=1 bash snake/scripts/epistemic/launch_epistemic.sh
#   # MC-dropout ONLY, on CPU (no GPU, no -p needed — SLURM picks a compute node):
#   METHODS=mc GPUS=0 bash snake/scripts/epistemic/launch_epistemic.sh
#   # override either method's time budget explicitly:
#   TIME_LAPLACE=72:00:00 bash snake/scripts/epistemic/launch_epistemic.sh
set -euo pipefail

# ── Knobs (env overrides) ─────────────────────────────────────────────────────
RESULTS_BASE="${RESULTS_BASE:-results}"          # central results/, or any other results tree
RUN_ID="${RUN_ID:-pr_classic3}"
RADIUS="${RADIUS:-r75}"
DATASETS="${DATASETS:-10x_TuPro TLS LUNG_CANCER}"
MODELS="${MODELS:-heteroscedastic poisson_cp negativebinomial_hetero_noanchor_cp}"
LAMBDAS="${LAMBDAS:-}"                            # optional: restrict which lambda_ sub-dirs (space-sep
                                                  # values, e.g. "0.0" or "0.0 1.0"). Empty => all lambdas.
                                                  # Count models have no lambda_ leaf and ALWAYS pass this
                                                  # filter (it only gates the lambda-bearing Gaussian dirs).
METHODS="${METHODS:-mc laplace}"                 # which post-hoc scripts (space-sep): mc | laplace.
                                                  # EACH is submitted as its OWN separate SLURM job
                                                  # per run-dir (never chained) -- see header.
CONDA_ENV="${CONDA_ENV:-mtenv0}"                  # env with torch + scipy + anndata + src
# conda.sh to source INSIDE the job (the same one snake/Snakefile and
# use). The job runs under /bin/sh, so we source it with `.` (POSIX), not bash `source`.
CONDA_SH="${CONDA_SH:-$(conda info --base)/etc/profile.d/conda.sh}"
DRYRUN="${DRYRUN:-0}"

# SLURM resources. The jobs only compute and save the summary outputs (no metrics,
# no figures), so peak RAM is modest and 96G is generous headroom; drop it after
# checking MaxRSS. GPUS=0 -> CPU-only job: drops -p, --gres AND the nvidia-smi
# monitor, so SLURM places it on a compute node (like an interactive
# `srun -c .. --mem ..` with no -p). MEM/CPUS are shared by both methods; TIME is
# PER METHOD (see below), since mc and laplace cost very differently.
PART="${PART:-gpu}"; GPUS="${GPUS:-1}"; CPUS="${CPUS:-4}"; MEM="${MEM:-96000}"

# Per-method TIME budgets:
#   mc      : T=30 dropout passes is cheap, so 2h is generous headroom.
#   laplace : the per-gene GGN fit dominates. It is markedly worse for NB
#             (dispersion head) than Poisson, and for the datasets with more train
#             spots. 48h leaves real margin; raise it (TIME_LAPLACE=72:00:00 or
#             more) for a run with more genes, folds or donors.
# Override either independently (TIME_MC=.., TIME_LAPLACE=..). Do not shrink
# TIME_LAPLACE on the assumption that CPU work is cheap: a Laplace timeout loses
# every fold, because test_summaries.h5ad is written once at the end.
TIME_MC="${TIME_MC:-02:00:00}"
TIME_LAPLACE="${TIME_LAPLACE:-48:00:00}"

PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
LOGDIR="$PROJECT_ROOT/snake/logs"
mkdir -p "$LOGDIR"

submit() {
    local run_dir="$1"
    local tag; tag="$(echo "${run_dir#"$RESULTS_BASE"/}" | tr '/' '_')"

    # Placement (GPU vs CPU) is shared by both methods; computed once per run-dir.
    local placement=""
    if [[ "${GPUS:-0}" != "0" ]]; then
        placement="-p $PART --gres=gpu:$GPUS"
    fi
    local place_desc="${placement:-cpu (no -p/--gres)}"

    # ONE independent sbatch call PER METHOD -- never chained with && (see header
    # "WHY SEPARATE JOBS"). Each gets its own job name, its own log files (the
    # method name is part of the filename so mc and laplace never collide), and its
    # own TIME budget (TIME_MC vs TIME_LAPLACE).
    local meth cmd job_time monitor gpu_raw
    for meth in $METHODS; do
        case "$meth" in
            mc)      cmd="python -u snake/scripts/epistemic/mc_dropout_posthoc.py --run-dir '$run_dir'"
                     job_time="$TIME_MC" ;;
            laplace) cmd="python -u snake/scripts/epistemic/laplace_posthoc.py   --run-dir '$run_dir'"
                     job_time="$TIME_LAPLACE" ;;
            *) echo "skip unknown method '$meth'"; continue ;;
        esac

        if [[ "$DRYRUN" == "1" ]]; then
            echo "WOULD SUBMIT  [$meth]  $run_dir  [time=$job_time | $place_desc]"
            continue
        fi

        monitor=""
        if [[ "${GPUS:-0}" != "0" ]]; then
            gpu_raw="$LOGDIR/epistemic_${meth}_${tag}_gpu.csv"
            monitor="nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv -l 60 > '$gpu_raw' & NVSMI_PID=\$!; trap 'kill \$NVSMI_PID 2>/dev/null' EXIT; "
        fi

        # Export BLAS thread-count vars BEFORE python starts. Without this, src/data/
        # load_and_preprocess.py's os.environ.setdefault(..., "1") -- written to stop ITS
        # OWN parallel preprocessing workers from oversubscribing, unrelated to this
        # script -- silently caps OpenBLAS/MKL to 1 thread as a side effect of the
        # laplace_posthoc.py/mc_dropout_posthoc.py -> read_arrays import chain, leaving
        # most of a `-c 4` allocation idle. setdefault() only sets a var if UNSET, so
        # exporting it first (tied to $CPUS, so it scales with whatever -c you request)
        # makes it a no-op.
        sbatch --job-name="epi_${meth}_${tag}" --time="$job_time" -c "$CPUS" \
            $placement --mem="${MEM}M" \
            --output="$LOGDIR/epistemic_${meth}_${tag}_%j.out" \
            --error="$LOGDIR/epistemic_${meth}_${tag}_%j.err" \
            --wrap ". '$CONDA_SH' && conda activate $CONDA_ENV && cd '$PROJECT_ROOT' || exit 1; export OMP_NUM_THREADS=$CPUS OPENBLAS_NUM_THREADS=$CPUS MKL_NUM_THREADS=$CPUS NUMEXPR_NUM_THREADS=$CPUS; ${monitor}${cmd}"
        echo "submitted     [$meth]  $run_dir  [time=$job_time | $place_desc]"
    done
}

n=0
for ds in $DATASETS; do
    base="$RESULTS_BASE/$ds/$RUN_ID"
    [[ -d "$base" ]] || { echo "skip (no dir): $base"; continue; }
    # Every leaf run-dir holds metrics.json + model_fold0.pt. The model_type is the
    # dir name (or its parent when the leaf is a lambda_ sub-dir, e.g. heteroscedastic).
    while IFS= read -r mj; do
        d="$(dirname "$mj")"
        [[ -f "$d/model_fold0.pt" ]] || continue
        b="$(basename "$d")"
        if [[ "$b" == lambda_* ]]; then
            mt="$(basename "$(dirname "$d")")"
            # LAMBDAS filter (Gaussian lambda_ dirs only): skip a lambda not in the list.
            if [[ -n "$LAMBDAS" ]]; then
                lam="${b#lambda_}"
                case " $LAMBDAS " in *" $lam "*) ;; *) continue ;; esac
            fi
        else
            mt="$b"
        fi
        # require the radius segment + a requested model_type
        [[ "$d" == *"/$RADIUS/"* ]] || continue
        case " $MODELS " in *" $mt "*) ;; *) continue ;; esac
        submit "$d"; n=$((n+1))
    done < <(find "$base" -name metrics.json)
done
echo "------"
# "${DRYRUN:+word}" would be wrong here: DRYRUN always holds some non-empty value,
# "0" by default, so bash's :+ always fires and mislabels real launches as
# "(dry-run)". Hence the explicit string-equality check below, matching submit()'s
# own "[[ "$DRYRUN" == "1" ]]" gate.
dryrun_label=""
[[ "$DRYRUN" == "1" ]] && dryrun_label="(dry-run) "
echo "${dryrun_label}matched $n run-dir(s) x methods='$METHODS' (each its own job)  [RESULTS_BASE=$RESULTS_BASE RUN_ID=$RUN_ID MODELS='$MODELS' TIME_MC=$TIME_MC TIME_LAPLACE=$TIME_LAPLACE]"
