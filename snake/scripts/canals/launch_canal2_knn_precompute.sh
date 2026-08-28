#!/usr/bin/env bash
# snake/scripts/canals/launch_canal2_knn_precompute.sh
#
# Submit the canal 2 kNN jobs (canal2_uncertainty_precompute.py --knn-dir + --run-dir): the
# training-free kNN signal scored against each MODEL's error, with canal 2's own error definition,
# grouping axes, floor policy and donor aggregation, so the numbers land in the same table as the
# aleatoric column. One SLURM job per (dataset, model), CPU-only, all concurrent.
#
# The kNN signal is a property of the COHORT, not of a model: it comes from the frozen H&E
# embeddings, the gene panel and the LOPO donor split. One kNN precompute per dataset therefore
# feeds every model of that dataset, and a job here pairs that one kNN dir with one model's run
# dir. Run snake/scripts/epistemic/launch_knn_uncertainty_precompute.sh first, once per dataset;
# without its donor=*/knn_signals.h5ad this launcher prints "skip dataset ..." and submits nothing.
#
# The signals, and what each one can answer:
#   sigma_knn  per-(spot,gene) local dispersion, the aleatoric analog. It varies along both axes,
#              so it produces a row on the gene axis and one on the spot axis.
#   d_mean     per-SPOT coverage score, the epistemic analog. Broadcast over genes, hence constant
#              within a spot-axis group: the spot axis degenerates exactly the way a
#              gene-homoscedastic variance degenerates on the gene axis. That is a property of the
#              signal, not missing data.
#   d_norm     the same, divided by the tissue-type cluster radius, which removes the type-rarity
#              confound: a rare tissue is far from its neighbors merely because it is sparsely
#              sampled. Same spot-axis degeneracy as d_mean.
#
# All three cohorts, but not with the same signals, since the coverage score normalizes by a
# cluster radius and the clusters a cohort can define differ:
#   10x_TuPro     fully labeled           -> d_norm_gt (primary), d_norm_leiden
#   TLS           many UNASSIGNED spots   -> d_norm_gt_spatial15 (primary), d_norm_gt_emb10, d_norm_leiden
#   LUNG_CANCER   no ground_truth at all  -> d_norm_leiden (primary)
# sigma_knn and d_mean need no clusters and exist everywhere; plain d_norm always aliases that
# cohort's primary scheme, so SIGNAL=d_norm runs on all three without special-casing. The producer
# prints which schemes it kept per donor, and records them in summary.json["cluster_schemes"].
#
# KNN_RUN_ID below only says which run's splits.json the kNN precompute read. A fold's reference
# pool is every spot except the held-out donor's, a set fixed by the donor list alone, so it is the
# same whichever run supplied the split. A different preproc_id would break the correspondence, and
# that fails on the adata_row alignment guard rather than passing silently.
#
# Usage (cluster login node, repo root):
#   bash snake/scripts/canals/launch_canal2_knn_precompute.sh                 # sigma_knn (default)
#   SIGNAL=d_norm bash snake/scripts/canals/launch_canal2_knn_precompute.sh   # the coverage signal
#   DRYRUN=1 bash snake/scripts/canals/launch_canal2_knn_precompute.sh        # preview only, no sbatch
#   FORCE=1  bash snake/scripts/canals/launch_canal2_knn_precompute.sh        # resubmit even already-done
#   MODELS="heteroscedastic/lambda_0.0" bash .../launch_canal2_knn_precompute.sh   # restrict
set -euo pipefail

# ── Knobs (env overrides) ─────────────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
RESULTS_BASE="${RESULTS_BASE:-$PROJECT_ROOT/results}"
RUN_ID="${RUN_ID:-pr_classic3}"      # run_id_classic segment inside results/{dataset}/
RADIUS="${RADIUS:-r75}"

SIGNAL="${SIGNAL:-sigma_knn}"
case "$SIGNAL" in
    sigma_knn|d_mean|d_norm|d_norm_gt|d_norm_gt_spatial15|d_norm_gt_emb10|d_norm_leiden) ;;
    *) echo "SIGNAL must be one of: sigma_knn | d_mean | d_norm | d_norm_gt |"
       echo "  d_norm_gt_spatial15 | d_norm_gt_emb10 | d_norm_leiden   (got '$SIGNAL')"
       echo "d_norm is an alias for the cohort's primary scheme; which d_norm_<scheme> variants"
       echo "actually exist depends on the cohort (gt needs a fully labeled pool, gt_* need the"
       echo "imputed columns, leiden is always available). See summary.json['cluster_schemes']."
       exit 1 ;;
esac
OUT_SUBDIR="${OUT_SUBDIR:-canal2_knn_$SIGNAL}"   # must match the script's --out-dir default
# Full signal name, not a 3-char prefix: d_norm_gt and d_norm_leiden would otherwise share a job
# name and be indistinguishable in squeue and in the log filenames.
JOB_PREFIX="canal2knn_${SIGNAL}"

# Which kNN dir to read: the kNN precompute's out-root (KNN_ROOT, derived below next to
# PROJECT_ROOT), then one dir per (dataset, baseline run) named by the KNN_* knobs here. They name
# the baseline the kNN was computed against, which is NOT necessarily the run being scored (that is
# the whole point: one kNN dir serves every model of the cohort).
KNN_RUN_ID="${KNN_RUN_ID:-pr_classic3}"          # MUST match RUN_ID/MODEL_TYPE/LAMBDA_DIR/GENE_SET of
KNN_MODEL_TYPE="${KNN_MODEL_TYPE:-heteroscedastic}"   # the launch_knn_uncertainty_precompute.sh call
KNN_LAMBDA_DIR="${KNN_LAMBDA_DIR-lambda_0.0}"    # that produced the signals ("" to omit the segment)
KNN_GENE_SET="${KNN_GENE_SET:-}"                 # empty -> per-dataset _<dataset>_hvg_top3000, the
                                                 # same default as the producer's GENE_SET knob --
                                                 # these four only NAME the directory, they do not
                                                 # change what is scored.

FORCE="${FORCE:-0}"                            # 1 = resubmit even if summary.json already exists
# All three cohorts, matching launch_knn_uncertainty_precompute.sh's own default. Which SIGNALS
# exist differs per cohort, and that is the thing to check before choosing SIGNAL: sigma_knn and
# d_mean everywhere; d_norm_gt on 10x_TuPro only (it needs a fully labeled reference pool);
# d_norm_gt_spatial15 / d_norm_gt_emb10 on TLS (~52% UNASSIGNED, so the imputed annotations are
# used rather than dropping half the pool); d_norm_leiden everywhere (unsupervised, and the only
# one LUNG_CANCER supports since it carries no ground_truth column at all). Plain d_norm is always
# valid: it aliases whichever scheme that cohort made primary. A dataset with no signals on disk
# is skipped with a message rather than submitted.
DATASETS="${DATASETS:-10x_TuPro TLS LUNG_CANCER}"
# The same 7 model types as the Laplace and MC-dropout launchers, but not necessarily the same
# trained models: they can be pointed at a different RUN_ID, that is, other checkpoints. Do not
# put the kNN column and an epistemic column from another launcher in one table unless both were
# produced under the same RUN_ID.
# msemeanonly appears here and in no other canal-2 launcher: it has no uncertainty head of its own,
# so canal 2 skips it in every other mode, but here the kNN supplies the uncertainty externally. That
# makes "can a training-free signal rank the errors of a model that emits no uncertainty at all" a
# question only this mode can ask. Drop it from MODELS if you do not want it.
MODELS_DEFAULT=(
    "heteroscedastic/lambda_0.0"
    "homoscedastic/lambda_0.0"
    "spot_homoscedastic/lambda_0.0"
    "poisson_cp"
    "negativebinomial_hetero_noanchor_cp"
    "poisson_raw"
    "negativebinomial_hetero_noanchor_raw"
    "msemeanonly"
)
# MODELS="a/lambda_0.0 b" to restrict; defaults to the array above.
if [[ -n "${MODELS:-}" ]]; then read -r -a MODELS_ARR <<< "$MODELS"; else MODELS_ARR=("${MODELS_DEFAULT[@]}"); fi
CONDA_ENV="${CONDA_ENV:-mtenv0}"
CONDA_SH="${CONDA_SH:-$(conda info --base)/etc/profile.d/conda.sh}"
DRYRUN="${DRYRUN:-0}"
# CPU-only (no -p/--gres -> compute partition). Same budget as the other canal-2 launchers, plus
# headroom: this mode additionally holds the concatenated kNN signals in memory while aligning them.
CPUS="${CPUS:-4}"; MEM="${MEM:-128000}"; TIME="${TIME:-04:00:00}"

LOGDIR="$PROJECT_ROOT/snake/logs"
mkdir -p "$LOGDIR"

# Derived from PROJECT_ROOT, exactly like the PRODUCER's OUT_ROOT
# (launch_knn_uncertainty_precompute.sh: $PROJECT_ROOT/results/_analysis/knn_uncertainty). It must
# be the same shape on both sides, or running either script against a relocated results tree
# would put the signals somewhere this launcher never looks -- with no symptom beyond the generic
# "skip dataset" message. Override BOTH together (KNN_ROOT here, OUT_ROOT there) if you relocate.
KNN_ROOT="${KNN_ROOT:-$PROJECT_ROOT/results/_analysis/knn_uncertainty}"

knn_dir_for() {
    local ds="$1" gs
    gs="${KNN_GENE_SET:-_${ds}_hvg_top3000}"
    # Reproduces knn_uncertainty_precompute.py's run_tag character for character:
    #   f"{run_id}__{gene_set}__{radius}__{model_type}" + ("__{lambda_dir}" if lambda_dir else "")
    # The triple underscore emerges from gene_set's own leading "_", it is not typed literally.
    local tag="${KNN_RUN_ID}__${gs}__${RADIUS}__${KNN_MODEL_TYPE}"
    [[ -n "$KNN_LAMBDA_DIR" ]] && tag="${tag}__${KNN_LAMBDA_DIR}"
    echo "$KNN_ROOT/$ds/$tag"
}

submit() {
    local run_dir="$1" knn_dir="$2"
    local tag; tag="$(echo "${run_dir#"$RESULTS_BASE"/}" | tr '/' '_')"

    if [[ "$FORCE" != "1" && -f "$run_dir/$OUT_SUBDIR/summary.json" ]]; then
        echo "skip (already done): $run_dir"
        return
    fi
    if [[ "$DRYRUN" == "1" ]]; then
        echo "WOULD SUBMIT  $run_dir  [knn: $knn_dir]"
        return
    fi

    sbatch --job-name="${JOB_PREFIX}_${tag}" --time="$TIME" -c "$CPUS" --mem="${MEM}M" \
        --output="$LOGDIR/${JOB_PREFIX}_${tag}_%j.out" \
        --error="$LOGDIR/${JOB_PREFIX}_${tag}_%j.err" \
        --wrap ". '$CONDA_SH' && conda activate $CONDA_ENV && cd '$PROJECT_ROOT' || exit 1; python -u snake/scripts/canals/canal2_uncertainty_precompute.py --run-dir '$run_dir' --knn-dir '$knn_dir' --knn-signal '$SIGNAL'"
    echo "submitted     $run_dir"
}

n=0
for ds in $DATASETS; do
    knn_dir="$(knn_dir_for "$ds")"
    # Checked ONCE per dataset, not per model: the same kNN dir feeds every model below, so a
    # missing one is a per-cohort problem and repeating the message per model would only hide it.
    if ! compgen -G "$knn_dir/donor=*/knn_signals.h5ad" > /dev/null; then
        echo "skip dataset $ds (no donor=*/knn_signals.h5ad under $knn_dir -- run"
        echo "     snake/scripts/epistemic/launch_knn_uncertainty_precompute.sh first; an older kNN"
        echo "     output does not carry that file, and the DATASETS default there is 10x_TuPro only)"
        continue
    fi
    # Partial-coverage warning. The check above fires on the first matching donor, but canal 2
    # requires the kNN dir to cover every test spot of the run, i.e. every donor: a donor whose kNN
    # job timed out lets all the model jobs below start, load, and only then die on the adata_row
    # assert. Counting the donor dirs that did produce the file against those that exist catches a
    # launched-but-unfinished donor before a single job is submitted; a donor that was never launched
    # leaves no dir and is still only caught by that assert.
    # Scheme gate. A scheme-suffixed SIGNAL exists only on cohorts whose reference pool can define
    # those clusters: d_norm_gt needs a fully labeled pool, d_norm_gt_* need the imputed columns,
    # d_norm_leiden is available everywhere. Without this gate a SIGNAL=d_norm_gt sweep over the
    # default DATASETS queues a job per unsupported (dataset, model) pair, each of which loads the
    # model's predictions and a large knn_signals.h5ad before dying. Checked against the producer's
    # own record, and only when that record is present; a kNN dir whose summary.json carries no
    # cluster_schemes key is left to the python pre-flight, which fails in seconds anyway.
    if [[ "$SIGNAL" == d_norm_?* ]]; then
        _scheme="${SIGNAL#d_norm_}"
        _sum="$( (compgen -G "$knn_dir/donor=*/summary.json" || true) | head -n1 )"
        if [[ -n "$_sum" ]] && grep -q '"cluster_schemes"' "$_sum" && ! grep -q "\"$_scheme\"" "$_sum"; then
            echo "skip dataset $ds (cluster scheme '$_scheme' was not produced for this cohort --"
            echo "     see $_sum ['cluster_schemes'])"
            continue
        fi
    fi
    n_have=$(compgen -G "$knn_dir/donor=*/knn_signals.h5ad" | wc -l)
    n_dirs=$(find "$knn_dir" -maxdepth 1 -type d -name 'donor=*' | wc -l)
    if [[ "$n_have" -lt "$n_dirs" ]]; then
        echo "WARNING dataset $ds: only $n_have of $n_dirs donor dir(s) under $knn_dir carry"
        echo "     knn_signals.h5ad -- an incomplete kNN sweep will fail every job below on the"
        echo "     adata_row coverage assert. Finish the kNN precompute first."
    fi
    for leaf in "${MODELS_ARR[@]}"; do
        run_dir="$RESULTS_BASE/$ds/$RUN_ID/_${ds}_hvg_top3000/$RADIUS/$leaf"
        if [[ ! -f "$run_dir/metrics.json" ]]; then
            echo "skip (no metrics.json): $run_dir"
            continue
        fi
        if ! compgen -G "$run_dir/predictions/preds_fold*.h5ad" > /dev/null; then
            echo "skip (no predictions): $run_dir/predictions/"
            continue
        fi
        submit "$run_dir" "$knn_dir"; n=$((n+1))
    done
done
echo "------"
echo "${DRYRUN:+(dry-run) }matched $n run-dir(s)  [RESULTS_BASE=$RESULTS_BASE RUN_ID=$RUN_ID DATASETS='$DATASETS' SIGNAL=$SIGNAL KNN_RUN_ID=$KNN_RUN_ID]"
