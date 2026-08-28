#!/usr/bin/env bash
# snake/scripts/epistemic/launch_knn_uncertainty_precompute.sh
#
# Submit the kNN local-uncertainty precompute jobs (knn_uncertainty_precompute.py, which does all
# the heavy work: the reference index build, the neighbor queries, the gene-chunked
# (spot x gene x k) signal grid and the OOD embedding queries). The signals themselves are defined
# in src/evaluation/knn_uncertainty.py.
#
# CPU-only: no -p and no --gres is passed to sbatch, which is what places the job on the plain
# `compute` partition instead of `gpu`. sklearn NearestNeighbors plus numpy/pandas/torch-cpu need no
# GPU here.
#
# One SLURM job per (donor, stage), because knn_uncertainty_precompute.py takes exactly one --donor
# and one --stage per invocation. The donor list is read, never hardcoded and never re-derived from
# a separate map: it comes from the baseline run's own splits.json["donors_ordered"], the same file
# the precompute script reads to build the donor-excluded reference pool. A donor can hold several
# slides, so the unit is the DONOR, and letting this run's own LOPO fold list be authoritative is
# what guarantees every submitted donor has a fold to read.
#
# One-time pre-flight before the first launch, on the login node. The unsupervised `leiden` cluster
# scheme needs scanpy's igraph backend: the python-igraph package, not leidenalg, and scanpy >= 1.10.
#   conda activate mtenv0 && python -c "import scanpy, igraph; print(scanpy.__version__, igraph.__version__)"
# If igraph is missing the code falls back to the leidenalg backend with a printed reason, and a
# leiden failure is a per-scheme skip rather than an abort, so a cohort with a usable ground_truth
# column still finishes. A cohort where leiden is the only applicable scheme fails on purpose.
#
# Stage scope. ood_corruption needs the blur / salt-and-pepper extractions to exist under
# patch_embeddings/<EMB_DATASET or dataset>/; submit() below skips a (dataset, ood_corruption) pair
# whose bf0.0 dir is missing rather than queueing a job that cannot succeed. ood_crossdataset is
# skipped when CROSS_DATASET equals the dataset itself, since querying a cohort against its own index
# is not a population-shift control.
#
# Usage (cluster login node, repo root):
#   bash snake/scripts/epistemic/launch_knn_uncertainty_precompute.sh                        # stage=signals, every donor of every dataset in DATASETS
#   STAGES="signals ood_corruption ood_crossdataset" bash snake/scripts/epistemic/launch_knn_uncertainty_precompute.sh
#   DRYRUN=1 bash snake/scripts/epistemic/launch_knn_uncertainty_precompute.sh               # preview only, no sbatch
#   FORCE=1 bash snake/scripts/epistemic/launch_knn_uncertainty_precompute.sh                # --overwrite + resubmit even already-done (donor,stage)
#   bash snake/scripts/epistemic/launch_knn_uncertainty_precompute.sh MACEGEJ SCC2           # only these donors (positional args), skip discovery
#   EXTRA_ARGS="--k-headline 50 --tau 2.0" bash snake/scripts/epistemic/launch_knn_uncertainty_precompute.sh  # forward any extra flag verbatim
set -euo pipefail

# ── Knobs (env overrides) ─────────────────────────────────────────────────────
RUN_ID="${RUN_ID:-pr_classic3}"              # the run_id_classic SEGMENT inside results/{dataset}/
                                             # The kNN reference pool is fixed by the donor list, not
                                             # by the run, so pointing this at the run being scored is
                                             # what keeps adata_row alignment true by construction.
# All three cohorts. Which d_norm cluster schemes apply differs per cohort: TLS has many UNASSIGNED
# spots so its `gt` scheme is skipped and the imputed gt_spatial15 / gt_emb10 ones are used instead,
# and LUNG_CANCER has no ground_truth column at all so only the unsupervised `leiden` scheme applies
# there. Each job prints which schemes it kept and which it skipped, and why.
DATASETS="${DATASETS:-10x_TuPro TLS LUNG_CANCER}"   # dataset(s) whose baseline run's donors are discovered + submitted
GENE_SET="${GENE_SET:-}"                     # empty -> per-dataset default _<dataset>_hvg_top3000 (mirrors the python script's own default)
RADIUS="${RADIUS:-r75}"
MODEL_TYPE="${MODEL_TYPE:-heteroscedastic}"  # baseline MUST be a continuous/Gaussian spec with a saved pred_var -- see the SCOPE note in knn_uncertainty_precompute.py
LAMBDA_DIR="${LAMBDA_DIR-lambda_0.0}"        # set to "" to omit the lambda_ subdir (passed through as --lambda-dir none)
                                             # lambda_0.0 is the proper-NLL Gaussian.
STAGES="${STAGES:-signals}"                  # space-sep subset of: signals ood_corruption ood_crossdataset
CROSS_DATASET="${CROSS_DATASET:-TLS}"        # stage=ood_crossdataset: the FOREIGN cohort queried against
                                             # this one's index. A (dataset, stage) pair where the two
                                             # are equal is skipped below -- that is not a shift control.
EMB_DATASET="${EMB_DATASET:-}"               # stage=ood_corruption: patch_embeddings/<name>/ holding the
                                             # corrupted extractions, when that dir is not named after
                                             # the dataset itself. Applies to every dataset in
                                             # DATASETS, so set it with a single-cohort launch.
RADIUS_FACTOR="${RADIUS_FACTOR:-0.75}"       # only NAMES the corrupted-embedding dirs (extraction template)
SWAP_XY="${SWAP_XY:-True}"
CONDA_ENV="${CONDA_ENV:-mtenv0}"             # env with sklearn + scanpy + anndata + torch + src
CONDA_SH="${CONDA_SH:-$(conda info --base)/etc/profile.d/conda.sh}"
DRYRUN="${DRYRUN:-0}"
FORCE="${FORCE:-0}"                          # 1 = --overwrite + resubmit even already-done (donor,stage)
EXTRA_ARGS="${EXTRA_ARGS:-}"                 # forwarded verbatim to every job, e.g. "--k-headline 50 --n-corruption-spots 800"

# CPU-only, sized for the heaviest stage: signals, which runs a k_max=100 index query and the
# gene-chunked (spot x gene x k) grid over the HVG panel. ood_corruption and ood_crossdataset fit
# well inside the same budget. The budget is per cohort, set in submit() below. The leiden scheme
# adds a full sc.pp.neighbors + Leiden pass over the reference pool, cached per donor so only the
# first stage of a donor pays it. A timeout loses the whole donor: summary.json is written last, so
# the sentinel then forces a recompute from the index build up. TIME and MEM env overrides win for
# every cohort when set.
CPUS="${CPUS:-4}"; MEM="${MEM:-}"; TIME="${TIME:-}"

PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
# knn_uncertainty_precompute.py derives its own RESULTS_ROOT as PROJECT_ROOT/results (from
# --project-root, defaulted from __file__ after the `cd "$PROJECT_ROOT"` below). Recomputing it
# identically here is what keeps donor discovery and the submitted job looking at the same tree.

# OUT_ROOT below is the other env knob: where the donor dirs are written. Its default must stay in
# step with launch_canal2_knn_precompute.sh's KNN_ROOT, which is the reader of those dirs.
RESULTS_ROOT="$PROJECT_ROOT/results"
OUT_ROOT="${OUT_ROOT:-$PROJECT_ROOT/results/_analysis/knn_uncertainty}"
LOGDIR="$PROJECT_ROOT/snake/logs"
mkdir -p "$LOGDIR"

gene_set_for() { local ds="$1"; echo "${GENE_SET:-_${ds}_hvg_top3000}"; }

run_dir_for() {
    local ds="$1" gs; gs="$(gene_set_for "$ds")"
    local d="$RESULTS_ROOT/$ds/$RUN_ID/$gs/$RADIUS/$MODEL_TYPE"
    [[ -n "$LAMBDA_DIR" ]] && d="$d/$LAMBDA_DIR"
    echo "$d"
}

donors_for_run() {
    # donors_ordered is THIS run's own LOPO donor list (splits.json, written by train.py --
    # see cv_donor_folds in src/data/splitting.py). Reading it straight from the file the
    # precompute script itself trusts is what guarantees every submitted donor actually has a
    # fold to read (no drift vs. an independently maintained donor list). Only the stdlib `json`
    # module is needed; CONDA_ENV is activated anyway for a consistent python across this repo.
    # `set +u` around the activation: several conda envs' activate.d hooks (e.g. the
    # binutils_linux-64 hook exporting ADDR2LINE) reference variables they never initialize,
    # which trips this script's own `set -u` and aborts the subshell -- the trailing `|| true`
    # then hides that failure, leaving `donors` silently empty. Nounset must be off while conda's
    # own scripts run; harmless here since python's own script below is what matters.
    local splits_json="$1/splits.json"
    [[ -f "$splits_json" ]] || return 0
    ( set +u; . "$CONDA_SH" && conda activate "$CONDA_ENV" && python -c "
import json
d = json.load(open('$splits_json'))
print(' '.join(str(x) for x in d['donors_ordered']))
" ) || true
}

submit() {
    local ds="$1" donor="$2" stage="$3"
    local gs; gs="$(gene_set_for "$ds")"

    # ── Stage applicability, checked BEFORE sbatch so no job is queued that cannot succeed ──
    if [[ "$stage" == "ood_corruption" ]]; then
        local emb_ds="${EMB_DATASET:-$ds}"
        local blur0="$PROJECT_ROOT/patch_embeddings/$emb_ds/embeddings_radius_factor_${RADIUS_FACTOR}_swapxy_${SWAP_XY}_blurry_bf0.0"
        if [[ ! -d "$blur0" ]]; then
            echo "skip dataset=$ds donor=$donor stage=ood_corruption (no corrupted embeddings:"
            echo "     $blur0 -- extract them with launch_blur_extraction.sh / launch_saltpepper_extraction.sh,"
            echo "     or set EMB_DATASET (the merged LUNG_CANCER has none; use LUNG_CANCER_LUAD / _LUSC))"
            return
        fi
    fi
    if [[ "$stage" == "ood_crossdataset" && "$CROSS_DATASET" == "$ds" ]]; then
        echo "skip dataset=$ds donor=$donor stage=ood_crossdataset (CROSS_DATASET == the dataset:"
        echo "     querying a cohort against its own reference index gives AUROC ~0.5 by construction."
        echo "     Relaunch as CROSS_DATASET=<other cohort> DATASETS=$ds ...)"
        return
    fi

    # Per-cohort budget (see the CPUS/MEM/TIME block); an explicit env TIME/MEM overrides it.
    local _t _m
    # Sized by spots per DONOR, which is what drives the run: the SIG arrays are
    # n_query x n_genes x |K_GRID| x 4 signals, and n_query is one donor's test spots. Total cohort
    # size is not the driver, since a cohort with many donors splits into small query sets.
    case "$ds" in
        10x_TuPro)   _t="04:00:00"; _m="128000" ;;
        LUNG_CANCER) _t="04:00:00"; _m="128000" ;;
        TLS)         _t="02:00:00"; _m="64000"  ;;
        *)           _t="04:00:00"; _m="128000" ;;   # unknown cohort: assume the heavy end
    esac
    [[ -n "$TIME" ]] && _t="$TIME"
    [[ -n "$MEM"  ]] && _m="$MEM"

    local run_tag="${RUN_ID}__${gs}__${RADIUS}__${MODEL_TYPE}"
    [[ -n "$LAMBDA_DIR" ]] && run_tag="${run_tag}__${LAMBDA_DIR}"
    local out_dir="$OUT_ROOT/$ds/$run_tag/donor=$donor"
    [[ "$stage" != "signals" ]] && out_dir="$out_dir/$stage"

    local extra_args="$EXTRA_ARGS"
    [[ "$FORCE" == "1" ]] && extra_args="$extra_args --overwrite"

    # knn_uncertainty_precompute.py's cache marker (summary.json) lives directly in its output dir,
    # the same skip convention as launch_canal1_benchmark_precompute.sh's benchmark_summary.json.
    # The test is stage-aware and must stay in sync with the identical rule in the python script: for
    # stage=signals the marker is summary.json AND knn_signals.h5ad, since summary.json is written
    # last. Without the second test, a re-run meant to produce the missing h5ad silently skips every
    # donor and writes nothing.
    if [[ "$FORCE" != "1" && -f "$out_dir/summary.json" ]] \
       && { [[ "$stage" != "signals" ]] || [[ -f "$out_dir/knn_signals.h5ad" ]]; }; then
        echo "skip (already done): dataset=$ds donor=$donor stage=$stage  ($out_dir)"
        return
    fi

    local tag="${ds}_${donor}_${stage}"
    if [[ "$DRYRUN" == "1" ]]; then
        echo "WOULD SUBMIT  dataset=$ds donor=$donor stage=$stage  ${extra_args:+[$extra_args]}"
        return
    fi

    sbatch --job-name="knn_unc_${tag}" --time="$_t" -c "$CPUS" --mem="${_m}M" \
        --output="$LOGDIR/knn_unc_${tag}_%j.out" \
        --error="$LOGDIR/knn_unc_${tag}_%j.err" \
        --wrap ". '$CONDA_SH' && conda activate $CONDA_ENV && cd '$PROJECT_ROOT' || exit 1; python -u snake/scripts/epistemic/knn_uncertainty_precompute.py --dataset '$ds' --donor '$donor' --stage '$stage' --run-id '$RUN_ID' --gene-set '$gs' --radius '$RADIUS' --model-type '$MODEL_TYPE' --lambda-dir '${LAMBDA_DIR:-none}' --out-root '$OUT_ROOT' --cross-dataset '$CROSS_DATASET' --radius-factor '$RADIUS_FACTOR' --swap-xy '$SWAP_XY' ${EMB_DATASET:+--emb-dataset '$EMB_DATASET'} $extra_args"
    echo "submitted     dataset=$ds donor=$donor stage=$stage (time=$_t mem=${_m}M) ${extra_args:+[$extra_args]}"
}

n=0
for ds in $DATASETS; do
    run_dir="$(run_dir_for "$ds")"
    if [[ ! -f "$run_dir/splits.json" ]]; then
        echo "skip dataset $ds (no splits.json under $run_dir -- run/land the baseline $MODEL_TYPE $RUN_ID training first)"
        continue
    fi
    if [[ $# -gt 0 ]]; then
        donors=("$@")
    else
        donors=($(donors_for_run "$run_dir"))
    fi
    if [[ ${#donors[@]} -eq 0 ]]; then
        echo "skip dataset $ds (no donors found in $run_dir/splits.json)"
        continue
    fi
    for donor in "${donors[@]}"; do
        for stage in $STAGES; do
            submit "$ds" "$donor" "$stage"
            n=$((n+1))
        done
    done
done
echo "------"
echo "${DRYRUN:+(dry-run) }matched $n (donor,stage) job(s)  [RUN_ID=$RUN_ID DATASETS='$DATASETS' MODEL_TYPE=$MODEL_TYPE LAMBDA_DIR='$LAMBDA_DIR' STAGES='$STAGES']"
