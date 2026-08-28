# snake/scripts/canals/canal2_uncertainty_precompute.py
#
# Canal 2 -- is the predicted uncertainty informative about the prediction ERROR, beyond what the
# predicted MEAN already says?
#
# ══ NO CHANGE OF SCALE ═════════════════════════════════════════════════════════════════════════
# Everything is taken on the model's OWN output scale, exactly as the artifact stores it:
#     mu = X                    log1p-CP10k for the Gaussians; the fitted count target for the
#                               count heads (CP_T for cp_counts / cp_counts_nr, raw counts for
#                               raw_counts)
#     y  = layers["y"]          the same axis as mu, by construction
#     u  = sqrt(variance)       a standard deviation (the raw signal in kNN mode -- see below)
# No expm1, no /depth_scale, no delta method, no size factor, no per-spot exposure. Every reported
# number is rank-based (Spearman) or a mean of one error column, so a per-run constant divisor would
# change nothing but the honesty of a column name.
#
# One consequence: for a Gaussian, log_pos below is a log of a log1p quantity. It moves the VALUE of
# the log error and nothing else, since mu_cp10k = expm1(X) is <= 0 exactly when X <= 0 and
# y_cp10k = expm1(y_log1p) is 0 exactly when y_log1p is 0: the y > 0 population and the entries
# where the prediction floor fires are the same either way. Correlations stay comparable across
# families, a rank correlation being invariant under a monotone transform of each variable. Only the
# mean-error SCALARS (mean_err_*) are scale-bound, and those are never in a cross-family table.
#
# ══ THREE ERROR CASES ══════════════════════════════════════════════════════════════════════════
#   lin_all   |y - mu|              EVERY (spot, gene) entry, zeros included
#   lin_pos   |y - mu|              y > 0 only
#   log_pos   |log y - log mu|      y > 0 only, mu SUBSTITUTED where log(mu) is undefined (mu <= 0)
# lin_pos shares its DEFINITION with lin_all and its POPULATION with log_pos, which is what makes a
# gap against either attributable.
# Read lin_all's NAIVE number with care: at y = 0 the error IS the prediction (|mu - 0| = |mu|), so
# on the majority of entries the "error" is a function of mu alone and the naive correlation
# approaches the mu-vs-u coupling mechanically. The partial is the number to read there.
#
# ══ THE METRIC ═════════════════════════════════════════════════════════════════════════════════
# Per group, four rank statistics over the group's entries:
#     r_uD = Spearman(u, Delta)        the NAIVE association
#     r_um = Spearman(u, mu)           the u-vs-mu coupling, WITHIN the group
#     r_Dm = Spearman(Delta, mu)
#     r_partial = (r_uD - r_um*r_Dm) / sqrt((1 - r_um^2)(1 - r_Dm^2))
# r_partial is src.evaluation.metrics_slide_aware._safe_partial_spearman(a=u, b=Delta, c=mu). Naive
# and controlled differ by the control and by nothing else: same groups, entries and aggregation.
#
# ══ TWO GROUPING AXES ══════════════════════════════════════════════════════════════════════════
#   GENE axis  group = (gene, slide)  -- per-SPOT resolution at fixed gene
#   SPOT axis  group = (slide, spot)  -- per-GENE resolution at fixed spot, its exact dual
# A head is DEGENERATE on the axis along which its variance parameter is CONSTANT: there u is a
# constant or a monotone function of mu, r_um is +-1, the denominator collapses and r_partial is
# NaN by construction, which is the correct answer rather than a failure.
#     gene-homoscedastic Gaussian   one sigma^2 per GENE -> GENE axis dead, SPOT axis alive
#     spot_homoscedastic Gaussian   one sigma^2 per SPOT -> mirror image
#     negativebinomial_genedisp_cp  alpha per GENE       -> the count analogue of the first
#     negativebinomial_spotdisp_cp  alpha per SPOT       -> the count analogue of the second
#     Poisson (u == mu)             no free parameter    -> dead on BOTH axes
#     heteroscedastic / per-(spot,gene) NB               -> both axes alive
# Both are therefore reported: one axis alone would leave such a head unmeasured for a reason about
# the axis, not about the head. The SPOT axis carries the weaker control, since at a fixed spot gene
# identity brings its own dispersion and detection rate, which partialling out mu does not remove.
#
# ══ AGGREGATION ════════════════════════════════════════════════════════════════════════════════
# GENE axis: keep n >= 60, drop NaN -> per gene, mean over its slides WEIGHTED by n -> simple mean
#            over genes. SPOT axis: keep n >= 60, drop NaN -> plain mean over spots, NO weighting.
# The asymmetry follows the data: the gene axis folds a dimension (spots -> gene) and weights by how
# many, while a spot sits in exactly one slide and hence one donor, so there is nothing to fold.
# The DONOR level (mean over a donor's genes/spots -> K donor points -> Student-t interval) is the
# notebook's job: the per-group parquets carry the slide, and slide -> donor is a config lookup.
#
# ══ ALSO COMPUTED: the SPARSIFICATION curves, on ALL THREE cases ═══════════════════════════════
# ── THREE PROTOCOLS, i.e. three answers to "removal inside WHAT?" ──────────────────────────────
#   gene_slide      inside (gene, slide). For one gene on one slide, drop the f most uncertain of
#                   ITS spots. The operational analogue of the NAIVE gene-axis correlation.
#   gene_slide_bin  inside (gene, slide, mu-bin). Same, but the fraction comes out of each mu-bin
#                   separately, so the predicted-mean composition of the remainder is fixed. The
#                   operational analogue of the PARTIAL gene-axis correlation.
#   pooled          one sort over EVERY entry of the cohort, no grouping at all. It lets a head win
#                   by knowing which GENES are hard rather than which spots are wrong, so it goes in
#                   the table and stays out of the figure.
# The first two are aggregated like every other number here: one curve per (gene, slide), normalized
# by that group's own f = 0 value (its mean error), folded over slides WEIGHTED BY n, then averaged
# over genes. That fold acts on already-normalized RATIOS, unlike gene_axis_mean, which weights raw
# statistics. A __median-over-genes twin sits next to each mean, since a relative curve is not
# bounded by 1 and a few genes can carry the mean. `pooled` is normalized once, globally, so it does
# not belong on a panel with the other two.
#
# All three run on lin_all, lin_pos AND log_pos, each under FOUR removal scores:
#     pred     the uncertainty          the thing being tested
#     oracle   the true error           the bound on what any score could achieve
#     mu_high  the predicted mean, largest removed first
#     mu_low   the predicted mean, smallest removed first
# The two mu scores are the baseline: the predicted mean is what a practitioner has without any
# uncertainty head, so a head that does not beat mu is redundant with it. Both directions, because
# which end of mu carries the error depends on the error (|y - mu| grows with mu, |log y - log mu|
# is largest where mu is smallest). There is no random curve: random removal is unbiased for the
# mean of the kept set, hence exactly flat at the f = 0 value.
# Read lin_all's pooled curve WITH ITS mu_high twin: there the error at y = 0 IS the prediction, so
# any score tracking mu partly sorts by error, and the twin measures how much. The mu-BINNED
# Spearman stays y > 0 only, needing the regime groups built from the positive-entry counts.
#
# ══ FIVE MODES (four mutually exclusive, plus --knn-dir which COMBINES with --run-dir) ══════════
#   --run-dir <dir>          the run's OWN uncertainty head (pred_var, or count_pred_var from
#                            layers["dispersion"]) against its own error. No --uncertainty-source:
#                            such a model has exactly one estimate.
#   --ensemble-dir <dir>     a deep ensemble's '.../ensemble/' dir (test_summaries.h5ad from
#                            epistemic/deep_ensemble_combine.py). Requires --uncertainty-source.
#   --mc-dropout-dir <dir>   a model dir holding mc_dropout/test_summaries.h5ad + metrics.json.
#                            --uncertainty-source optional, defaults to "epistemic".
#   --laplace-dir <dir>      same, for laplace/test_summaries.h5ad. The same code path, since
#                            laplace_posthoc.py writes the identical layer layout; only the
#                            subdirectory and the estimator behind layers["epistemic"] differ, and
#                            that is a property of the input file, not of this script.
#   --knn-dir <dir>          + --run-dir. The training-free kNN signal (donor=*/knn_signals.h5ad),
#                            scored against the error of the model given by --run-dir. It supplies
#                            ONLY the uncertainty, so it combines rather than excludes, and one kNN
#                            dir serves every model of a cohort. --knn-signal picks the signal:
#                            sigma_knn is per (spot, gene); d_mean / d_norm[_scheme] are per SPOT
#                            and broadcast over genes, hence CONSTANT within a spot-axis group and
#                            degenerate there. NO_UNCERTAINTY_MODELS is not enforced: whether a
#                            training-free signal can rank the errors of a model that emits no
#                            uncertainty is a question only this mode asks.
#
# UNCERTAINTY SOURCES for the three post-hoc modes, from each mode's own test_summaries.h5ad:
#   epistemic = layers["epistemic"]           the exact mixture / draw-disagreement variance
#   aleatoric = X                             (Poisson-family: var == mean)
#             | layers["aleatoric_var"]       (Gaussian)
#             | X + layers["mean_alpha_m2"]   (NB-family)
#   total     = aleatoric + epistemic
# Ensemble mode picks the count formula from the STATIC taxonomy (POISSON_MODELS / NB_MODELS);
# mc-dropout and laplace use a RUNTIME check ("mean_alpha_m2" in B.layers), which covers the whole
# COUNT_MODELS set without whitelisting each type. The two model sets have different scopes and are
# not interchangeable.
# `total` is the like-for-like comparison to a --run-dir model's own head, which already conflates
# the two; `aleatoric` and `epistemic` alone ask which source does the ranking work.
#
# Run identity (dataset, model_type) is read from metrics.json, never parsed from the path. Ensemble
# mode reads <ensemble-dir>/../seed_0/metrics.json (there is none at the ensemble/ level) and
# cross-checks it against the file's own uns["deep_ensemble"]["model_type"].
#
# ══ THREE UNCERTAINTY FORMS (--uncertainty-form, --run-dir ONLY) ═══════════════════════════════
#   sd              u = sqrt(variance). THE definition of this canal, the default, and what every
#                   figure of the paper reads. Bit-exact no-op.
#   cv              u = sqrt(variance) / mu, the coefficient of variation.
#   one_plus_alpha  u = 1 + alpha, the NB dispersion alone (family 'nb'; graceful not_applicable
#                   sentinel on any other family).
# The two alternatives serve the robustness appendix: they show that the collapse of the controlled
# correlation is not an artifact of calling a standard deviation "the uncertainty". They break the
# sigma-follows-mu link in opposite ways, cv dividing the mean out and one_plus_alpha never letting
# it in. Both write to canal2_uncertainty_<form>/, leaving the reference artifact untouched, and
# both stamp summary["uncertainty_form"] -- gate on it, the three are different quantities. cv is
# the one quantity here that a mean touches; it does not resurrect a degenerate head, u/mu staying a
# monotone function of mu wherever u was.
#
# ══ OUTPUT ═════════════════════════════════════════════════════════════════════════════════════
# Under <run-dir>/canal2_uncertainty/ (run-dir; canal2_uncertainty_<form>/ for a non-default
# --uncertainty-form), <ensemble-dir>/canal2_uncertainty/<source>/,
# <mc-dropout-dir>/canal2_{epistemic,aleatoric_mc,total_mc}/, <laplace-dir>/canal2_laplace_<source>/
# or <run-dir>/canal2_knn_<signal>/ (kNN -- under the MODEL's dir, since one kNN dir is shared by
# every model of the cohort). --out-dir, if given, is used literally in any mode.
#
#   pair_partial.parquet   gene, slide, n_plus_all, n_plus_pos,
#                          {r_uD,r_um,r_Dm,r_partial}_{lin_all,lin_pos,log_pos}
#   spot_partial.parquet   slide, spot, the same 12 stat columns + the same two counts
#                          (`spot` is obs["adata_row"], joinable to adata.obs / obsm["spatial"])
#   bin_stats.parquet      gene, slide, bin, n, regime, spearman_lin_pos, spearman_log_pos
#   sparsification.parquet frac, then curve_{score}_{protocol}_{case} for every
#                          score    in {pred, oracle, mu_high, mu_low}
#                          protocol in {pooled, gene_slide, gene_slide_bin}
#                          case     in {lin_all, lin_pos, log_pos}
#                          plus a __median twin of each per-group curve (see THREE PROTOCOLS).
#   per_group_curves.parquet  the (gene, slide) curves behind those aggregates: case, protocol,
#                          gene, n, mean_err (the f=0 divisor that was normalized out), then
#                          f00_{score} .. f90_{score}. Kept so the SPREAD across genes is
#                          inspectable without a rerun.
#   long_df.parquet        gene, slide, spot, bin, regime, mu, y, uncertainty, err_lin, err_log,
#                          mu_substituted (y > 0 AND regime groups only -- exactly the population
#                          the curves above run on, mu-bin included, so any variant of them is an
#                          offline re-derivation and never another sweep)
#   summary.json           identity + provenance + the dataset-level scalars (see the assembled
#                          dict at the bottom). Gate on canal2_convention == "v3_native_scale":
#                          an artifact tagged otherwise was produced on another scale under other
#                          column names and is NOT comparable.
#
# Two n_plus columns because the group SIZE differs by population: n_plus_all counts every entry of
# the group, n_plus_pos only the detected ones. The n >= 60 floor applies to the count of the case
# being computed.
#
# ══ lin_all-ONLY MODE (target_transform != "none") ═════════════════════════════════════════════
# A run trained with a per-gene target transform (gene_zscore / gene_minmax) has its y and mu in the
# STANDARDIZED space, where the zero has moved and "y > 0" no longer means "detected". The script
# detects this from metrics.json and computes lin_all ALONE -- see the long note at LIN_ALL_ONLY.
# summary["lin_all_only"] is the gate; summary["target_transform"] says which transform. Then:
# pair_partial / spot_partial carry their lin_all columns only, sparsification carries only the
# eight lin_all curves, and bin_stats.parquet / long_df.parquet are NOT WRITTEN (an empty file
# would falsely read as "computed, nothing passed the gate"). Column names never change.
#
# All heavy work stays here, so the notebooks only read these small aggregates and plot. One run per
# process: the sweep is launched as N concurrent CPU SLURM jobs (see launch_canal2_precompute.sh).
#
# Usage (repo root, on the cluster; CPU only):
#   python -u snake/scripts/canals/canal2_uncertainty_precompute.py --run-dir results/10x_TuPro/pr_classic3/_10x_TuPro_hvg_top3000/r75/heteroscedastic/lambda_0.0
#   python -u snake/scripts/canals/canal2_uncertainty_precompute.py \
#       --ensemble-dir results/.../negativebinomial_hetero_noanchor_cp/ensemble --uncertainty-source epistemic
#   python -u snake/scripts/canals/canal2_uncertainty_precompute.py \
#       --laplace-dir results/.../heteroscedastic/lambda_0.0 --uncertainty-source epistemic
#   python -u snake/scripts/canals/canal2_uncertainty_precompute.py \
#       --run-dir results/.../negativebinomial_hetero_noanchor_cp --uncertainty-form one_plus_alpha

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
import anndata as ad

_T0 = time.time()
def _el():
    return time.time() - _T0

# ── Args ──────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
_mode = p.add_mutually_exclusive_group(required=True)
_mode.add_argument("--run-dir", default=None,
               help="A classic run dir (holds predictions/preds_fold*.h5ad, metrics.json).")
_mode.add_argument("--ensemble-dir", default=None,
               help="A deep-ensemble '.../ensemble/' dir (test_summaries.h5ad from "
                    "epistemic/deep_ensemble_combine.py). Requires --uncertainty-source.")
_mode.add_argument("--mc-dropout-dir", default=None,
               help="A model dir holding mc_dropout/test_summaries.h5ad + metrics.json.")
_mode.add_argument("--laplace-dir", default=None,
               help="A model dir holding laplace/test_summaries.h5ad + metrics.json. Same semantics "
                    "as --mc-dropout-dir: that writer emits the identical layer layout.")
p.add_argument("--knn-dir", default=None,
               help="A kNN signals dir holding donor=*/knn_signals.h5ad. NOT part of the mutually "
                    "exclusive group: it must be combined with --run-dir, because the kNN supplies "
                    "only the UNCERTAINTY while mu, y and therefore the error come from the model "
                    "being scored. Select the signal with --knn-signal.")
p.add_argument("--knn-signal", default=None,
               help="Which kNN signal to score; only meaningful with --knn-dir. Free-form rather "
                    "than a fixed choice list, because the available d_norm variants depend on which "
                    "cluster schemes the cohort supports; an unknown name fails with the list found "
                    "in the file. sigma_knn = per-(spot,gene) local dispersion. d_mean / d_norm / "
                    "d_norm_<scheme> = per-SPOT coverage scores, where <scheme> is gt, gt_spatial15 "
                    "/ gt_emb10 (TLS) or leiden (the only one LUNG_CANCER supports); plain d_norm is "
                    "an alias for that cohort's primary scheme. Defaults to 'sigma_knn'.")
p.add_argument("--uncertainty-source", choices=["aleatoric", "epistemic", "total"], default=None,
               help="Which post-hoc uncertainty to score against the error. Required with "
                    "--ensemble-dir; defaults to 'epistemic' with --mc-dropout-dir/--laplace-dir; "
                    "rejected with --run-dir, which only ever has one uncertainty estimate.")
p.add_argument("--uncertainty-form", choices=["sd", "cv", "one_plus_alpha"], default="sd",
               help="WHICH FUNCTION OF THE PREDICTIVE is scored as 'the uncertainty'. 'sd' (the "
                    "default) is this canal's standing definition and a bit-exact no-op. 'cv' is "
                    "the coefficient of variation sd/mu, and 'one_plus_alpha' is 1 + alpha, the NB "
                    "dispersion read alone. The two alternatives exist for ONE reason: the paper's "
                    "robustness appendix has to show that the collapse of the controlled "
                    "correlation is not an artifact of defining the uncertainty as a standard "
                    "deviation. --run-dir ONLY: they are statements about the HEAD's own "
                    "parameters, and 1 + alpha needs layers['dispersion'], which no post-hoc "
                    "summary file carries (an ensemble stores mean(alpha*m^2), not alpha).")
p.add_argument("--out-dir", default=None,
               help="Default: <run-dir>/canal2_uncertainty/, "
                    "<ensemble-dir>/canal2_uncertainty/<uncertainty-source>/, "
                    "<mc-dropout-dir>/canal2_{epistemic,aleatoric_mc,total_mc}/, "
                    "<laplace-dir>/canal2_laplace_<uncertainty-source>/ or "
                    "<run-dir>/canal2_knn_<knn-signal>/.")
args = p.parse_args()

if args.run_dir is not None and args.knn_dir is not None:
    mode = "knn"          # --run-dir supplies mu/y/error, --knn-dir supplies the uncertainty
elif args.run_dir is not None:
    mode = "run_dir"
elif args.ensemble_dir is not None:
    mode = "ensemble"
elif args.mc_dropout_dir is not None:
    mode = "mc_dropout"
else:
    mode = "laplace"

# mc_dropout and laplace behave identically everywhere below (same loader, same formulas, same
# default source); they differ only in the subdirectory read and the one written. POSTHOC_MODES is
# that shared membership, named once so a future fifth mode is one list entry and not a hunt for
# every `in ("mc_dropout", "laplace")` test.
POSTHOC_MODES = ("mc_dropout", "laplace")

if mode in POSTHOC_MODES and args.uncertainty_source is None:
    args.uncertainty_source = "epistemic"
    print(f"--uncertainty-source not given with --{mode.replace('_', '-')}-dir -- defaulting to "
          "'epistemic'.", flush=True)
if mode == "ensemble" and args.uncertainty_source is None:
    p.error("--uncertainty-source is required with --ensemble-dir.")
if mode == "run_dir" and args.uncertainty_source is not None:
    p.error("--uncertainty-source is only meaningful with --ensemble-dir/--mc-dropout-dir/"
            "--laplace-dir (a --run-dir model has exactly one uncertainty estimate).")
if args.uncertainty_form != "sd" and mode != "run_dir":
    p.error(f"--uncertainty-form {args.uncertainty_form!r} is --run-dir ONLY: both alternatives are "
            "functions of the HEAD's own predictive (sd/mu needs the head's mu, 1+alpha needs "
            "layers['dispersion']), and no post-hoc or kNN input file carries them.")

# True wherever the uncertainty can legitimately be UNDEFINED on some entries, which changes the
# population and therefore has to be dropped ONCE, at the two points where a per-entry frame is
# born. Two causes, same treatment: a kNN signal missing where a (spot, gene) had too few detected
# neighbours, and cv = sd/mu missing where mu <= 0 (possible for a Gaussian on log1p-CP10k, never
# for a count head, whose mu is an exp). The dropped count is printed and lands in summary.json --
# read it before comparing an n_plus count of this run with another form's.
U_MAY_BE_UNDEFINED = (mode == "knn") or (args.uncertainty_form == "cv")


def knn_uncertainty_scale(sig):
    """One-line description of the axis a kNN signal lives on, for summary.json.

    A TOTAL function: --knn-signal is free-form by design (the available d_norm
    variants depend on which cluster schemes the cohort supports), so it must NEVER be used as a
    dict key without a default. A missing key would raise while ASSEMBLING the summary, that is,
    after the whole multi-hour job had succeeded and before a single output file was written."""
    if sig == "sigma_knn":
        return "log1p-CP10k standard deviation of the local neighbourhood"
    if sig == "d_mean":
        return "H&E embedding distance (not a dispersion)"
    if sig.startswith("d_norm"):
        scheme = sig[len("d_norm_"):] if sig.startswith("d_norm_") else "primary (cohort alias)"
        return f"H&E embedding distance / cluster radius, scheme={scheme} (not a dispersion)"
    return f"unknown kNN signal {sig!r}"


# Resolved from the FIRST donor file in load_knn_signal (which asserts every donor agrees) and
# echoed into summary.json; stays None in every non-kNN mode.
KNN_PRIMARY_SCHEME = None

if mode == "knn":
    if args.uncertainty_source is not None:
        p.error("--uncertainty-source is not used in kNN mode; pick the signal with --knn-signal.")
    if args.knn_signal is None:
        args.knn_signal = "sigma_knn"
        print("--knn-signal not given with --knn-dir -- defaulting to 'sigma_knn'.", flush=True)
    # PRE-FLIGHT, here rather than in load_knn_signal (which only runs after every preds_fold*.h5ad
    # has been concatenated and one full donor file has been read into RAM): a typo, or asking for a
    # scheme this cohort never produced (d_norm_gt on TLS, where the gt scheme is legitimately
    # skipped), otherwise costs a whole job slot before saying so. Backed read, milliseconds.
    _f0 = sorted(Path(args.knn_dir).glob("donor=*/knn_signals.h5ad"))
    assert _f0, (f"no donor=*/knn_signals.h5ad under {args.knn_dir} -- run "
                 f"epistemic/knn_uncertainty_precompute.py --stage signals first.")
    try:
        _h0 = sc.read_h5ad(_f0[0], backed="r")
        _have = (args.knn_signal in _h0.obs.columns) or (args.knn_signal in _h0.layers)
        _msg = (f"{_f0[0]}: kNN signal {args.knn_signal!r} is neither an obs column nor a layer.\n"
                f"  per-spot columns available: "
                f"{sorted(c for c in _h0.obs.columns if c.startswith('d_'))}\n"
                f"  per-(spot,gene) layers available: {sorted(_h0.layers)}\n"
                f"  cluster schemes this cohort produced: "
                f"{_h0.uns.get('cluster_schemes', 'not recorded')}")
        del _h0
    except OSError as _e:      # the backed peek is an OPTIMIZATION, never the authority
        _have, _msg = True, ""
        print(f"kNN pre-flight skipped ({type(_e).__name__}: {_e}); the signal is still validated "
              f"in load_knn_signal.", flush=True)
    assert _have, _msg
elif args.knn_dir is not None:
    p.error("--knn-dir requires --run-dir (the kNN supplies only the uncertainty; the mean and the "
            "error come from the model being scored).")
elif args.knn_signal is not None:
    p.error("--knn-signal is only meaningful with --knn-dir.")

run_dir = os.path.abspath({"run_dir": args.run_dir, "ensemble": args.ensemble_dir,
                           "mc_dropout": args.mc_dropout_dir, "laplace": args.laplace_dir,
                           "knn": args.run_dir}[mode])   # knn: the MODEL dir, as in run_dir mode

if mode == "knn":
    # Own namespace under the MODEL's dir, keyed by the signal: one model is scored against several
    # kNN signals, and (unlike the post-hoc modes) the kNN dir is shared by every model of the
    # cohort, so the output cannot live there without runs of different models colliding.
    out_dir = args.out_dir or os.path.join(run_dir, f"canal2_knn_{args.knn_signal}")
elif mode == "ensemble":
    out_dir = args.out_dir or os.path.join(run_dir, "canal2_uncertainty", args.uncertainty_source)
elif mode == "mc_dropout":
    _default_sub = {"epistemic": "canal2_epistemic", "aleatoric": "canal2_aleatoric_mc",
                    "total": "canal2_total_mc"}[args.uncertainty_source]
    out_dir = args.out_dir or os.path.join(run_dir, _default_sub)
elif mode == "laplace":
    # Own namespace, NOT the mc-dropout one: pr_classic1 carries both estimators in the same run
    # dir, so sharing the subdirectory would make whichever ran second silently replace the first.
    out_dir = args.out_dir or os.path.join(run_dir, f"canal2_laplace_{args.uncertainty_source}")
else:
    # An alternative uncertainty FORM gets its own namespace for the same reason a post-hoc source
    # does: the reference form is the paper's figure, and a re-run must never overwrite it.
    _sub = ("canal2_uncertainty" if args.uncertainty_form == "sd"
            else f"canal2_uncertainty_{args.uncertainty_form}")
    out_dir = args.out_dir or os.path.join(run_dir, _sub)
os.makedirs(out_dir, exist_ok=True)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(PROJECT_ROOT)
from src.evaluation.count_metrics import count_pred_var
from src.evaluation.sparsification import (SparsificationPrecisionError, spars_scores,
                                           pooled_curves, per_group_curves)

# ── Constants ─────────────────────────────────────────────────────────────────
# The three error cases, and the population each is defined on. THE definition of canal 2.
CASES = {
    "lin_all": "|y - mu|, every (spot, gene) entry, zeros included",
    "lin_pos": "|y - mu|, y > 0 only",
    "log_pos": "|log y - log mu|, y > 0 only, mu substituted where log(mu) is undefined",
}
STATS = ["r_uD", "r_um", "r_Dm", "r_partial"]

N_PLUS_MIN = 60          # inclusion floor for a group, on the count of the case being computed
EPS = 1e-12              # denominator guard, identical to _safe_partial_spearman / _safe_spearman
# Bin count for the SECONDARY mu-binned Spearman, from the group's positive-entry count.
REGIME_TO_NBINS = {"terciles": 3, "quintiles": 5, "deciles": 10}
FRACS = np.linspace(0, 0.9, 19)          # sparsification removal fractions

# The per-(gene, slide) sparsification curves that could NOT be produced, and why. Filled by the two
# per_group_curves call sites, echoed into summary.json -- so a reader of an output whose curve_*
# per-group columns are missing is TOLD why instead of inferring it. Empty on a healthy run.
SPARS_SKIPPED = []

# Carried in every SPARS_SKIPPED entry. The POOLED curves of a skipped case are still written --
# they never accumulate across groups, so they do not hit the failure -- but they are means of the
# same error array, so the tail that broke the per-group accumulation dominates them just as much.
# Numerically computable is not the same as trustworthy, and the difference has to be IN the file.
SPARS_SKIPPED_NOTE = ("the pooled curves of this case are still written but are means of the same "
                      "error array, so they are dominated by the same tail -- do not report them")


# The dtype of the three big (N, G) matrices, and of them ONLY. The .h5ad stores float32, so
# float64 here doubled the largest allocation of the script for nothing: 232M entries on
# 10x_TuPro is 1.85 GB per matrix in float64 against 0.93 GB in float32, times three matrices.
#
# Where float32 is safe, and where it would not be. Every reported number of this canal is rank-based,
# and the ranks are not float32: pandas' rank() returns float64 whatever it is given, and
# np.bincount(..., weights=w) accumulates in float64. So the correlation cascade keeps its
# precision for free. The only float32 risk is the value -> rank step itself, where two entries
# that differ in float64 could tie in float32; at 7 significant digits against expression data
# that is far below the noise of the experiment, and a tie is handled by average ranking rather
# than silently dropped.
#
# What must NOT follow this constant: any long ACCUMULATION. A cumsum over tens of millions of
# entries in float32 drifts badly, so the sparsification sums below are cast to float64 first.
# Same rule as the on-device metric twins in src/evaluation/metrics_slide_aware.py: operands
# float32, reductions float64.
DT = np.float32

# Shared by both family taxonomies below.
CONTINUOUS_MODELS = {"msemeanonly", "homoscedastic", "heteroscedastic", "spot_homoscedastic"}

# ── Taxonomy #1: model_family() -- 3-way. Used ONLY for --run-dir (native aleatoric) and
# --ensemble-dir, where the aleatoric formula genuinely differs between Poisson (var == mu) and NB
# (var = mu + alpha*mu^2, needs a dispersion layer). The *_cp_nr variants are IN and treated exactly
# like their _cp twins: same predictive family, same fitted target scale, and every formula below
# reads mu and alpha off the MODEL, never off the target -- only whether the target was rounded
# differs. The depth-blind *_raw models are IN for the same reason; what differs for them is only
# the axis their numbers live on, which does not matter here: nothing is converted, and every
# correlation is rank-invariant. ──
POISSON_MODELS    = {"poisson_cp", "poisson_raw", "poisson_cp_nr"}
NB_MODELS         = {"negativebinomial_hetero_noanchor_cp",
                     "negativebinomial_hetero_noanchor_raw",
                     # Restricted-dispersion cp NB (alpha per spot / per gene): same NB formula
                     # var = mu + alpha*mu^2, only the GRAIN of alpha differs, and the formula reads
                     # alpha per (spot,gene) after the model broadcasts it.
                     "negativebinomial_spotdisp_cp", "negativebinomial_genedisp_cp",
                     "negativebinomial_hetero_noanchor_cp_nr",
                     "negativebinomial_spotdisp_cp_nr", "negativebinomial_genedisp_cp_nr",
                     "STI_nb_cp", "STI_nb_raw"}   # STimage bare-Linear heads: same NB formula

# ── Taxonomy #2: family_scale() -- 2-way. Used for the post-hoc and kNN modes, where EPISTEMIC
# uncertainty is layers["epistemic"] regardless of count sub-family and the aleatoric count formula
# is picked by a RUNTIME layer-presence check -- so this set only has to answer "is this a count
# family at all". Deliberately NOT unioned with the two sets above: they answer different questions,
# and a future family may be a count without having a settled aleatoric formula. ──
COUNT_MODELS = {"poisson_cp", "poisson_cp_nr", "poisson_raw",
                "negativebinomial_hetero_noanchor_cp",
                "negativebinomial_hetero_noanchor_cp_nr",
                "negativebinomial_hetero_noanchor_raw",
                "negativebinomial_spotdisp_cp", "negativebinomial_genedisp_cp",
                "negativebinomial_spotdisp_cp_nr", "negativebinomial_genedisp_cp_nr",
                "STI_nb_cp", "STI_nb_raw"}

# msemeanonly has is_probabilistic=False (src/models/specs.py) -- no pred_var layer is ever saved,
# so it has no uncertainty estimate at all and canal 2 is structurally not applicable. It is the
# ONLY early exit, and only in the two modes that read the model's OWN head: post-hoc epistemic
# uncertainty exists for it, and so does a kNN signal. homoscedastic is NOT exited on -- its
# variance is constant within a (gene,slide) pair, so its gene-axis columns come out NaN by
# construction, but the SPOT axis is exactly the axis it CAN be measured on.
NO_UNCERTAINTY_MODELS = {"msemeanonly"}


def model_family(model_type):
    if model_type in CONTINUOUS_MODELS:
        return "continuous"
    if model_type in POISSON_MODELS:
        return "poisson"
    if model_type in NB_MODELS:
        return "nb"
    raise ValueError(f"unknown model family for model_type={model_type!r}")

def family_scale(model_type):
    if model_type in CONTINUOUS_MODELS:
        return "continuous"
    if model_type in COUNT_MODELS:
        return "count"
    raise ValueError(f"unknown family for model_type={model_type!r} -- add it to one of the sets above")

def _dense(a):
    return a.toarray() if sp.issparse(a) else np.asarray(a)


# ── Run identity — read from metrics.json (authoritative), never parsed from the path ──
# Ensemble mode: metrics.json does NOT live at the ensemble/ dir level (one level down, per
# seed_{s}/), and the file's own uns["deep_ensemble"] carries model_type but not dataset, so
# seed_0/metrics.json is the primary source for both fields (read once, cheap, before the h5ad).
if mode == "ensemble":
    ensemble_dir_p = Path(run_dir)
    model_dir_p = ensemble_dir_p.parent            # dir containing seed_{s}/, == .../<model_type>[/lambda_x]/
    metrics_json_path = model_dir_p / "seed_0" / "metrics.json"
    lambda_dir = model_dir_p.name if model_dir_p.name.startswith("lambda_") else None
else:
    metrics_json_path = Path(run_dir) / "metrics.json"
    lambda_dir = os.path.basename(run_dir) if os.path.basename(run_dir).startswith("lambda_") else None
with open(metrics_json_path) as f:
    M = json.load(f)
dataset = M["dataset"]
model_type = M["model_type"]

# ── lin_all-ONLY mode, forced by a per-gene TARGET TRANSFORM ─────────────────────────────────
# When the run was trained with target_transform != "none" (gene_zscore / gene_minmax), X and
# layers["y"] are in the STANDARDIZED space: y_z = (y - loc_g) / scale_g, a per-gene AFFINE map.
# That is precisely why the headline correlation and the PIT stay valid there -- but an affine map
# MOVES THE ZERO, and the y>0 population is the one thing in this canal that depends on where zero
# sits. A non-detected entry (y = 0) becomes y_z = -loc_g / scale_g < 0, so:
#   - "y > 0" stops meaning "detected" and starts meaning "above the gene's TRAIN mean" -- a
#     different, undocumented population (roughly half the panel instead of its detected fraction),
#     which would silently redefine lin_pos;
#   - log(y_z) is undefined on every entry below that mean, so log_pos is not merely a different
#     population but an ill-posed quantity, and the floor substitution cannot repair it (the floor
#     is a quantile of the OBSERVATIONS, which are negative there).
# So both y>0 cases are SKIPPED and only lin_all is computed: |y - mu| is defined at every entry
# and is affine-EQUIVARIANT, i.e. it is the one case whose meaning survives the transform. It is
# then expressed in z units, which matters only for the scale-BOUND numbers (mean_err) and not for
# the rank statistics that are the point of this canal.
# What disappears with the y>0 population: bin_stats.parquet, long_df.parquet, the _pos
# sparsification curves, mean_err, floor_sensitivity, residual_confound, and the _lin_pos /
# _log_pos columns of pair_partial / spot_partial. Everything that survives keeps its EXACT column
# names, so a reader gates on summary["lin_all_only"] rather than on a missing key.
TARGET_TRANSFORM = str(M.get("target_transform", "none"))
LIN_ALL_ONLY = TARGET_TRANSFORM != "none"

def exit_not_applicable(reason):
    """Write the sentinel summary.json and leave with status 0. NOT a failure.

    Two situations reach it, and both are facts about the HEAD rather than about the job: a model
    that emits no uncertainty at all, and an uncertainty FORM that does not exist for this family
    (1 + alpha on anything but an NB). A launcher sweeps heads, so the alternative would be a
    directory of crashed jobs that a reader cannot tell apart from a real failure."""
    print(f"model_type={model_type!r}: {reason} -- Canal 2 does not apply, skipping gracefully "
          f"(not a failure).", flush=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        # Same key SET as the full summary below, so a reader can gate uniformly instead of having
        # to .get() every field on the not-applicable artifacts.
        json.dump(dict(dataset=dataset, model_type=model_type, lambda_dir=lambda_dir, mode=mode,
                       canal2_convention="v3_native_scale",
                       is_ensemble=(mode == "ensemble"), is_mc_dropout=False, is_laplace=False,
                       is_knn=False, knn_signal=None, knn_dir=None,
                       uncertainty_source=(args.uncertainty_source if mode == "ensemble" else None),
                       uncertainty_form=args.uncertainty_form,
                       target_transform=TARGET_TRANSFORM, lin_all_only=LIN_ALL_ONLY,
                       not_applicable=True, reason=reason), f, indent=2)
    sys.exit(0)


if mode in ("run_dir", "ensemble") and model_type in NO_UNCERTAINTY_MODELS:
    exit_not_applicable("no uncertainty estimate at all (is_probabilistic=False, no pred_var layer)")

family = model_family(model_type) if mode in ("run_dir", "ensemble") else family_scale(model_type)

if args.uncertainty_form == "one_plus_alpha" and family != "nb":
    exit_not_applicable(f"--uncertainty-form one_plus_alpha needs a fitted dispersion, and family="
                        f"{family!r} has none (the Gaussians carry sigma^2, the Poisson carries "
                        f"nothing free)")

_mode_desc = {"run_dir": "run-dir",
              "ensemble": f"ensemble (source={args.uncertainty_source})",
              "mc_dropout": f"mc-dropout (source={args.uncertainty_source})",
              "laplace": f"laplace (source={args.uncertainty_source})",
              "knn": f"knn (signal={args.knn_signal}, dir={args.knn_dir})",
              }.get(mode, mode)   # .get, not [], so a future sixth mode cannot crash on a print
print(f"run_dir     : {run_dir}", flush=True)
print(f"mode        : {_mode_desc}", flush=True)
print(f"uncert. form: {args.uncertainty_form}"
      + ("  (the canal's standing definition)" if args.uncertainty_form == "sd"
         else "  -> ALTERNATIVE definition, robustness appendix; own output dir"), flush=True)
print(f"dataset     : {dataset}  model_type: {model_type} ({family})  lambda_dir: {lambda_dir}", flush=True)
print(f"target_transform: {TARGET_TRANSFORM}"
      + ("  -> lin_all ONLY (y>0 cases skipped, see the note above)" if LIN_ALL_ONLY else ""),
      flush=True)

# The cases actually COMPUTED by this invocation. CASES stays the full 3-way definition of the
# canal (it is documentation); this is what the loops iterate and what summary.json reports.
CASES_RUN = {"lin_all": CASES["lin_all"]} if LIN_ALL_ONLY else dict(CASES)


# ══════════════════════════════════════════════════════════════════════════════
# Loading, and the extraction of (mu, u, y) on the model's OWN scale.
# ══════════════════════════════════════════════════════════════════════════════

def load_test_predictions(run_dir):
    pred_files = sorted((run_dir / "predictions").glob("preds_fold*.h5ad"))
    assert pred_files, f"no predictions in {run_dir / 'predictions'}"
    parts = []
    for f in pred_files:
        p = sc.read_h5ad(f, backed="r")           # lazy -- avoids reading the "val" rows we drop
        parts.append(p[p.obs["split"].to_numpy() == "test"].to_memory())
    B = ad.concat(parts, axis=0, merge="same")
    assert np.unique(B.obs["adata_row"]).size == B.n_obs, "test spots overlap across folds -- non-LOPO?"
    return B


def load_test_predictions_ensemble(ensemble_dir):
    """The ONE consolidated test_summaries.h5ad (deep_ensemble_combine.py already did the
    fold-by-fold reduction over the seed members). Under LOPO every spot's row already IS its
    single held-out-fold prediction, so there is no "val" split left to filter -- DEFENSIVELY
    ASSERT obs['split'] is constant 'test' rather than filtering, so a future writer emitting
    non-test rows surfaces loudly instead of being silently dropped."""
    path = ensemble_dir / "test_summaries.h5ad"
    assert path.exists(), f"no test_summaries.h5ad at {path} -- run deep_ensemble_combine.py first"
    B = sc.read_h5ad(path)
    if "split" in B.obs.columns:
        assert (B.obs["split"].to_numpy() == "test").all(), \
            f"{path}: obs['split'] is not constant 'test'"
    assert np.unique(B.obs["adata_row"]).size == B.n_obs, f"{path}: test spots overlap -- non-LOPO?"
    return B


def load_posthoc_summaries(model_dir, subdir):
    """MC-dropout / Laplace: the consolidated post-hoc TEST summaries -- one row per spot (LOPO
    covers each spot once), so no per-fold concat. `subdir` ("mc_dropout" or "laplace") is the ONLY
    difference between the two callers: laplace_posthoc.py writes the same X / y / epistemic /
    per-family extra layers as mc_dropout_posthoc.py, so one loader serves both."""
    path = model_dir / subdir / "test_summaries.h5ad"
    assert path.is_file(), f"no {subdir} summaries at {path}"
    B = sc.read_h5ad(path)
    assert "epistemic" in B.layers, f"no layers['epistemic'] in {path}"
    assert np.unique(B.obs["adata_row"].to_numpy()).size == B.n_obs, "test spots overlap -- non-LOPO?"
    return B


def load_knn_signal(knn_dir, B, signal):
    """Return an (n_obs, n_vars) array aligned to B, holding the requested kNN signal for every
    (spot, gene) of B.

    Unlike every other mode the uncertainty comes from a DIFFERENT directory than the mean: the kNN
    signal is a property of the cohort (H&E embeddings, gene panel, LOPO split), not of the model,
    so ONE kNN dir serves every model of that cohort and B decides which spots and genes are scored.
    Alignment is therefore explicit and asserted, never positional: spots by obs["adata_row"], genes
    by name. The per-donor files are globbed and concatenated the way run_dir mode globs
    preds_fold*.h5ad (one producer process per held-out donor).

    d_mean / d_norm are per-SPOT obs columns, broadcast over the gene axis here: a coverage signal
    gives every gene of a spot the same score. That is a property of the signal, not a defect, and
    it is why such a signal degenerates on the SPOT axis (constant within the group)."""
    files = sorted(knn_dir.glob("donor=*/knn_signals.h5ad"))
    assert files, (f"no donor=*/knn_signals.h5ad under {knn_dir} -- run "
                   f"epistemic/knn_uncertainty_precompute.py --stage signals first.")
    # Read one donor at a time and keep ONLY what this signal needs. A knn_signals.h5ad carries FOUR
    # (spots x genes) matrices and exactly one is ever used, so reading them all and letting
    # ad.concat duplicate the lot would cost ~4x the necessary peak on the heaviest canal-2 mode.
    global KNN_PRIMARY_SCHEME
    obs_parts, mat_parts, var_names, schemes = [], [], None, []
    for f in files:
        p = sc.read_h5ad(f)
        # Which cluster scheme this donor's unsuffixed `d_norm` alias resolves to. The producer
        # decides it PER FOLD (a scheme is skipped when that fold's reference pool has unusable
        # labels), so two donors of one cohort can disagree -- and then `d_norm` would carry two
        # different divisors in one column. Only meaningful for the alias.
        schemes.append(p.uns.get("primary_cluster_scheme"))
        if var_names is None:
            var_names = np.asarray(p.var_names)
        else:
            assert np.array_equal(np.asarray(p.var_names), var_names), (
                f"{f}: var_names differ from {files[0]} -- these donor files are not one cohort.")
        obs_parts.append(p.obs.copy())                       # per-spot, tiny
        if signal not in p.obs.columns:
            assert signal in p.layers, (
                f"{f}: kNN signal {signal!r} is neither an obs column nor a layer.\n"
                f"  per-spot columns available: {sorted(c for c in p.obs.columns if c.startswith('d_'))}\n"
                f"  per-(spot,gene) layers available: {sorted(p.layers)}")
            mat_parts.append(_dense(p.layers[signal]))       # native float32 until the slice
        del p
    # The per-file test above and the concatenated test further down are made on DIFFERENT objects,
    # so they can disagree in exactly one way: a signal that is an obs column in one donor file and
    # a layer in another. The per-spot branch would then win while mat_parts holds only SOME donors'
    # matrices, i.e. rows that no longer line up. All-or-nothing makes them provably agree.
    assert len(mat_parts) in (0, len(files)), (
        f"{knn_dir}: signal {signal!r} is an obs column in some donor files and a layer in others "
        f"({len(mat_parts)}/{len(files)} matrices collected) -- these files are not one cohort.")
    _uniq_schemes = sorted({s for s in schemes if s is not None})
    KNN_PRIMARY_SCHEME = _uniq_schemes[0] if len(_uniq_schemes) == 1 else None
    assert not (signal == "d_norm" and len(_uniq_schemes) > 1), (
        f"{knn_dir}: the unsuffixed 'd_norm' alias resolves to DIFFERENT cluster schemes across "
        f"donor files ({_uniq_schemes}) -- use the explicit d_norm_<scheme> name instead.")
    if signal.startswith("d_norm"):
        print(f"[{_el():6.1f}s] kNN cluster scheme(s) reported by the donor files: "
              f"{_uniq_schemes or 'not recorded'}", flush=True)

    k_obs = pd.concat(obs_parts, ignore_index=True)
    del obs_parts
    _donors = [f.parent.name for f in files]
    assert np.unique(k_obs["adata_row"].to_numpy()).size == len(k_obs), (
        f"{knn_dir}: kNN query spots overlap across donors -- non-LOPO? ({len(files)} donor "
        f"file(s): {_donors})")
    assert pd.Index(var_names).is_unique, f"{knn_dir}: duplicate gene names in var_names"

    # Spots: reindex the kNN rows onto B's rows by adata_row. Every spot of B must be present -- a
    # missing one means the kNN dir was built from a different split (or, more commonly, that the
    # kNN sweep is INCOMPLETE, which the glob accepts silently), hence the donor list in the message.
    k_rows = pd.Index(k_obs["adata_row"].to_numpy())
    row_pos = k_rows.get_indexer(pd.Index(B.obs["adata_row"].to_numpy()))
    n_missing_spots = int((row_pos < 0).sum())
    assert n_missing_spots == 0, (
        f"{knn_dir}: {n_missing_spots:,} of {B.n_obs:,} scored spots have no kNN row (adata_row "
        f"mismatch) -- different split, or an incomplete kNN sweep. Donor file(s): {_donors}.")

    col_pos = pd.Index(var_names).get_indexer(pd.Index(np.asarray(B.var_names)))
    missing_genes = np.asarray(B.var_names)[col_pos < 0]
    assert missing_genes.size == 0, (
        f"{knn_dir}: {missing_genes.size:,} of {B.n_vars:,} scored genes absent from the kNN "
        f"signals (e.g. {list(missing_genes[:5])}) -- different gene set.")

    if signal in k_obs.columns:      # any per-SPOT coverage score
        # Broadcast AFTER the row selection: the value is one float per SPOT, so materializing the
        # full (n_knn_spots x n_genes) matrix first would allocate n_genes copies of a per-spot
        # column only to immediately slice it back down.
        per_spot = k_obs[signal].to_numpy(dtype=DT)[row_pos]
        return np.repeat(per_spot[:, None], len(col_pos), axis=1)

    src = mat_parts[0] if len(mat_parts) == 1 else np.concatenate(mat_parts, axis=0)
    del mat_parts
    return src[np.ix_(row_pos, col_pos)].astype(DT)


def assert_count_convention(B, M):
    """A count artifact must carry count_convention == "v2_no_offset". Returns it.

    Nothing here rescales anything, so this is NOT about a divisor: it is about what mu and alpha
    MEAN. An artifact without the marker was fitted with a per-spot offset, mu = s_i * exp(f(x)), and a
    dispersion at a different scale, so count_pred_var(mu, alpha) = mu + alpha*mu^2 would
    pair two incompatible parameters. Such a file is REJECTED, never guessed at.

    The marker is read from metrics.json in every mode but ensemble; an ensemble summary carries
    its own, stamped by deep_ensemble_combine.py, and has no run-level metrics.json."""
    conv = (B.uns.get("count_convention") if mode == "ensemble"
            else (M.get("count_convention") or B.uns.get("count_convention")))
    if conv != "v2_no_offset":
        raise ValueError(f"{run_dir}: count artifact with the wrong marker (count_convention={conv!r}, "
                         f"expected 'v2_no_offset'). Re-run the training: such a file is "
                         f"rejected rather than reinterpreted.")
    return conv


def native_axis(family, M):
    """The axis mu, y and the errors live on -- recorded, never converted. See the header.

    A per-gene target transform is part of that axis and must be NAMED here: under gene_zscore the
    Gaussian's output is not log1p-CP10k but its per-gene standardization, so the plain label would
    be a false provenance string on the one artifact a reader trusts for provenance. Same wording as
    complete_deferred_metrics.py::metric_scale_string, so the two agree."""
    if family == "continuous":
        tt = str(M.get("target_transform", "none"))
        if tt != "none":
            return (f"per-gene standardized log1p-CP10k ({tt}; loc/scale from the fold's TRAIN rows, "
                    f"in var['target_loc'/'target_scale'])")
        return "log1p-CP10k (the Gaussian's own output scale)"
    ts = M.get("target_source")
    return {"cp_counts": "CP_T counts = round(CP10k * depth_scale), the fitted target",
            "cp_counts_nr": "CP_T counts, unrounded, the fitted target",
            "raw_counts": "raw counts (depth-blind), the fitted target",
            }.get(ts, f"the fitted count target ({ts!r})")


def _sigma(variance):
    """sqrt of a variance, with the one thing that can go wrong stated rather than hidden.

    A negative entry would make sqrt NaN, and a NaN uncertainty is DROPPED further down -- so it
    would silently shrink the population instead of failing. None of the four formulas above can
    produce one, hence the loud warning rather than a clamp: a clamp would repair a symptom whose
    cause is worth knowing about."""
    neg = int((variance < 0).sum())
    if neg:
        print(f"WARNING: {neg:,} of {variance.size:,} variance entries are NEGATIVE. sqrt makes them "
              f"NaN and they will be dropped from every statistic -- this should be impossible, "
              f"investigate the artifact before using these numbers.", flush=True)
    return np.sqrt(variance)


def predictions_and_uncertainty(B, mode, family, uncertainty_source=None):
    """(mu, u), both (N, G), on the model's OWN scale -- NOTHING is converted.

    mu = X, verbatim. u is a STANDARD DEVIATION, sqrt of whichever variance the mode supplies:
      run_dir   the run's own head: pred_var (Gaussian) or count_pred_var(X[, dispersion]).
      ensemble / mc_dropout / laplace: one of three post-hoc sources --
        epistemic = layers["epistemic"] as-is (already the mixture/draw-disagreement variance)
        aleatoric = X (Poisson-family) | layers["aleatoric_var"] (Gaussian)
                    | X + layers["mean_alpha_m2"] (NB-family)
        total     = aleatoric + epistemic
      Ensemble picks the count formula from the STATIC taxonomy; the two post-hoc modes use a
      RUNTIME layer-presence check. That difference is deliberate (see the header).
      knn       the kNN signal AS IS -- NO sqrt. It is already a standard deviation (sigma_knn) or a
                distance (d_mean / d_norm), never a variance, so taking a root would be a second
                unlabelled transform of a quantity this canal is supposed to carry raw. Every
                consumer here is rank-based, so it changes no reported correlation either way; the
                axis it does live on is recorded as summary["uncertainty_scale"]."""
    X = _dense(B.X).astype(DT)
    if mode == "knn":
        # KNN_SIGNAL is aligned to the FULL B (all spots x all genes), so this is only valid when
        # called on the full B -- which is the only way it is ever called.
        return X, KNN_SIGNAL
    if mode == "run_dir":
        if family == "continuous":
            variance = _dense(B.layers["pred_var"]).astype(DT)
        elif family == "poisson":
            variance = count_pred_var(X)                                    # var == mu exactly
        elif family == "nb":
            alpha = _dense(B.layers["dispersion"]).astype(DT)       # fitted at X's scale
            variance = count_pred_var(X, dispersion=alpha)                  # mu + alpha*mu^2
        else:
            raise ValueError(family)
        return X, _sigma(variance)

    epistemic = _dense(B.layers["epistemic"]).astype(DT)
    if family == "continuous":
        aleatoric = _dense(B.layers["aleatoric_var"]).astype(DT)
    elif mode == "ensemble":
        if family == "poisson":
            aleatoric = X                                                   # var == mean, no layer
        elif family == "nb":
            aleatoric = X + _dense(B.layers["mean_alpha_m2"]).astype(DT)
        else:
            raise ValueError(family)
    else:                                        # mc_dropout / laplace, family == "count"
        aleatoric = (X + _dense(B.layers["mean_alpha_m2"]).astype(DT)
                     if "mean_alpha_m2" in B.layers else X)
    # Branched rather than indexed out of a dict: a dict literal would evaluate aleatoric+epistemic
    # for every source, i.e. allocate one more (N, G) float64 array on the two runs that never use it.
    if uncertainty_source == "epistemic":
        variance = epistemic
    elif uncertainty_source == "aleatoric":
        variance = aleatoric
    else:
        variance = aleatoric + epistemic
    return X, _sigma(variance)


def apply_uncertainty_form(mu, u, B):
    """The ALTERNATIVE definitions of 'the uncertainty', for the robustness appendix. run_dir only.

    'sd' never reaches here (the caller skips it), so this function is exactly the deviation from
    the canal's standing definition and there is one place to read to know what it is.

      cv              u / mu, the coefficient of variation. It is the one alternative that the
                      header's no-change-of-scale rule does not cover, by design: it
                      divides by the MEAN, which is the very covariate the partial correlation
                      controls for. That is the point of the check -- an uncertainty that has
                      already been divided by the mean is the strongest form of the objection
                      'you only found a collapse because sigma tracks mu'. Note it does NOT
                      resurrect a degenerate head: where u is a monotone function of mu inside the
                      group (Poisson on both axes, a homoscedastic sigma on its own axis), so is
                      u/mu, r_um stays +-1 and the partial stays NaN by construction.
                      mu <= 0 leaves it UNDEFINED (NaN), never floored: the floor of build_long_df
                      repairs a log, which is a different quantity, and a fabricated ratio would
                      enter the ranking as a real value. Those entries are dropped downstream via
                      U_MAY_BE_UNDEFINED, and the count is reported.
      one_plus_alpha  1 + alpha, the NB dispersion read alone (family == 'nb', enforced above).
                      No mu enters it at all, so it is the mirror image of cv: cv removes the mean
                      from the uncertainty by division, this one never lets it in.

    Both are returned RAW, with no rescaling of any kind -- every statistic downstream is rank-based
    (see the header), so a per-run constant would change nothing and only make a name lie."""
    if args.uncertainty_form == "cv":
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(mu > 0, u / mu, np.nan).astype(DT)
        n_undef = int((mu <= 0).sum())
        print(f"uncertainty form 'cv' = sd/mu: undefined (mu <= 0) on {n_undef:,} of {mu.size:,} "
              f"entries ({100 * n_undef / mu.size:.4f}%) -- those are DROPPED, not floored.",
              flush=True)
        return out
    alpha = _dense(B.layers["dispersion"]).astype(DT)     # fitted at mu's scale, NB only
    print(f"uncertainty form 'one_plus_alpha' = 1 + alpha, read from layers['dispersion'] "
          f"(alpha in [{np.nanmin(alpha):.4g}, {np.nanmax(alpha):.4g}]).", flush=True)
    return (1.0 + alpha).astype(DT)


# ══════════════════════════════════════════════════════════════════════════════
# The rank cascade.
# ══════════════════════════════════════════════════════════════════════════════

def _corr(n, sx, sy, sxy, sxx, syy):
    """Pearson from grouped sums. On WITHIN-GROUP RANKS this is Spearman (ties averaged, as
    scipy.stats.rankdata does), so it matches _safe_spearman up to floating point."""
    num = n * sxy - sx * sy
    den = np.sqrt((n * sxx - sx ** 2) * (n * syy - sy ** 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.isfinite(den) & (den > EPS), num / den, np.nan)


def partial_spearman_per_group(df, group_cols, err_col):
    """The four statistics per group: r_uD, r_um, r_Dm and the partial

        r_partial = (r_uD - r_um*r_Dm) / sqrt((1 - r_um^2)(1 - r_Dm^2))

    identical to src.evaluation.metrics_slide_aware._safe_partial_spearman(a=u, b=Delta, c=mu).
    Returns group_cols + ["n", "r_uD", "r_um", "r_Dm", "r_partial"], one row per present group.

    ONE PASS. The three columns are ranked ONCE each within the group (Spearman == Pearson on
    within-group ranks) and the ten grouped sums come from np.bincount over the group codes. Ranking
    each column once instead of once per PAIR halves the sorts, and the groupby factorization -- the
    real cost on tens of millions of object-keyed rows -- happens once instead of three times. It
    also makes the three correlations provably share ONE population, which is the only condition
    under which the partial formula means anything.

    NaN: rows where any of the three is non-finite are dropped BEFORE ranking. Only the kNN signal
    is ever non-finite, and that mode already drops those entries upstream, so this is a guard.

    r_partial is NaN where a control COLLAPSES (|r_um| or |r_Dm| == 1): a head whose uncertainty is
    constant, or a monotone function of mu, along the grouping axis' free dimension carries no
    uncertainty degree of freedom independent of mu there. That is the correct answer, not a
    failure. The guard tests 1 - r^2 on EACH control separately, not only their product under the
    root: a degenerate control landing at 1 - 1ulp leaves 1 - r^2 ~ 2e-16 but a denominator ~1e-8,
    which would clear an EPS placed only on the denominator and return large finite garbage."""
    g = list(group_cols)
    cols = g + ["uncertainty", "mu", err_col]
    d = df[cols]
    ok = (np.isfinite(d["uncertainty"].to_numpy()) & np.isfinite(d["mu"].to_numpy())
          & np.isfinite(d[err_col].to_numpy()))
    if not ok.all():
        d = d[ok]
    if d.empty:
        return pd.DataFrame(columns=g + ["n"] + STATS)

    gb = d.groupby(g, sort=True)
    keys = gb.size().reset_index(name="n")     # group keys, in ngroup order (sort=True)
    code = gb.ngroup().to_numpy()
    k = len(keys)
    ru = gb["uncertainty"].rank().to_numpy()
    rm = gb["mu"].rank().to_numpy()
    rd = gb[err_col].rank().to_numpy()
    del d, gb

    def S(w):
        return np.bincount(code, weights=w, minlength=k)

    n = keys["n"].to_numpy().astype(np.float64)
    # ngroup() numbers groups in the order a sort=True groupby iterates them, which is the order
    # size() indexes them -- so row i of `keys` IS code i. Everything below attaches a bincount sum
    # to a key by that correspondence, so it is CHECKED (the per-code counts must reproduce the
    # per-key sizes) rather than assumed; a silent reordering would mislabel every group.
    assert np.array_equal(np.bincount(code, minlength=k), keys["n"].to_numpy()), \
        "ngroup() and size() disagree -- the grouped sums would be attached to the wrong keys"
    su, sm, sd = S(ru), S(rm), S(rd)
    suu, smm, sdd = S(ru * ru), S(rm * rm), S(rd * rd)
    r_uD = _corr(n, su, sd, S(ru * rd), suu, sdd)
    r_um = _corr(n, su, sm, S(ru * rm), suu, smm)
    r_Dm = _corr(n, sd, sm, S(rd * rm), sdd, smm)
    del ru, rm, rd, code

    one_um = 1.0 - r_um ** 2
    one_Dm = 1.0 - r_Dm ** 2
    den = np.sqrt(one_um * one_Dm)
    with np.errstate(divide="ignore", invalid="ignore"):
        r_partial = (r_uD - r_um * r_Dm) / den
    bad = (~np.isfinite(den) | (den < EPS) | (one_um < EPS) | (one_Dm < EPS))
    keys["r_uD"], keys["r_um"], keys["r_Dm"] = r_uD, r_um, r_Dm
    keys["r_partial"] = np.where(bad, np.nan, r_partial)
    return keys


def rankcorr_per_group(df, group_cols, col_x, col_y):
    """A single grouped Spearman(col_x, col_y). Same closed form as above, for the SECONDARY
    diagnostics (the mu-binned Spearman and the residual-confound check), which need one
    correlation rather than the four. Returns group_cols + ["n", "spearman"]."""
    g = list(group_cols)
    d = df[g + [col_x, col_y]]
    ok = np.isfinite(d[col_x].to_numpy()) & np.isfinite(d[col_y].to_numpy())
    if not ok.all():
        d = d[ok]
    if d.empty:
        return pd.DataFrame(columns=g + ["n", "spearman"])
    gb = d.groupby(g, sort=True)
    keys = gb.size().reset_index(name="n")
    code = gb.ngroup().to_numpy()
    k = len(keys)
    rx = gb[col_x].rank().to_numpy()
    ry = gb[col_y].rank().to_numpy()
    n = keys["n"].to_numpy().astype(np.float64)
    assert np.array_equal(np.bincount(code, minlength=k), keys["n"].to_numpy()), \
        "ngroup() and size() disagree -- the grouped sums would be attached to the wrong keys"
    sx, sy = np.bincount(code, weights=rx, minlength=k), np.bincount(code, weights=ry, minlength=k)
    keys["spearman"] = _corr(n, sx, sy,
                             np.bincount(code, weights=rx * ry, minlength=k),
                             np.bincount(code, weights=rx * rx, minlength=k),
                             np.bincount(code, weights=ry * ry, minlength=k))
    return keys


def gene_axis_mean(pair, col):
    """GENE axis: group -> gene (mean over the gene's slides, WEIGHTED by n) -> simple mean over
    genes. Groups below the floor and NaN values are dropped first (nanmean convention)."""
    v = pair[(pair["n"] >= N_PLUS_MIN) & pair[col].notna()]
    if v.empty:
        return float("nan")
    w = pd.DataFrame({"gene": v["gene"].to_numpy(),
                      "w": v[col].to_numpy() * v["n"].to_numpy(),
                      "n": v["n"].to_numpy()}).groupby("gene", as_index=False).sum()
    return float((w["w"] / w["n"]).mean())


def spot_axis_mean(spot, col):
    """SPOT axis: a spot sits in exactly one slide and hence one donor, so there is no intermediate
    level to fold -- a plain mean over the included spots, no weighting."""
    v = spot[spot["n"] >= N_PLUS_MIN][col].dropna()
    return float(v.mean()) if len(v) else float("nan")


def axis_stats(frame, axis):
    """Everything reported for ONE (axis, error case): the naive value, the controlled value, and
    the counts a reader needs to judge them. n_defined_partial is not optional detail -- on lin_all
    the spot axis it is the number that says whether the collapse discussed in the header happened."""
    agg = gene_axis_mean if axis == "gene" else spot_axis_mean
    inc = frame[frame["n"] >= N_PLUS_MIN] if len(frame) else frame
    return dict(
        naive=agg(frame, "r_uD"),
        partial=agg(frame, "r_partial"),
        partial_median=(float(inc["r_partial"].median()) if len(inc) else float("nan")),
        n_groups=int(len(frame)),
        n_included=int(len(inc)),
        n_defined_naive=(int(inc["r_uD"].notna().sum()) if len(inc) else 0),
        n_defined_partial=(int(inc["r_partial"].notna().sum()) if len(inc) else 0),
    )


# ══════════════════════════════════════════════════════════════════════════════
# The whole-panel pass (lin_all) and the y>0 long table.
# ══════════════════════════════════════════════════════════════════════════════

def fold_slides_then_genes(C, gene_of_group, n_of_group, n_genes):
    """(gene, slide) curves -> one curve per gene -> one curve per model.

    Slides are folded into a gene WEIGHTED BY n, matching gene_axis_mean; genes are then averaged
    unweighted. NOTE the weighting acts on already-normalized RATIOS here, whereas gene_axis_mean
    weights raw statistics. The paper states that as its own sentence.

    Returns (per_gene, mean_over_genes, median_over_genes). The MEDIAN is not decoration: a group's
    relative curve is not bounded by 1 (removing the most uncertain entries can remove the SMALLEST
    errors), so a handful of genes can carry the mean. If the two diverge, that is a result to
    report, not a reason to switch metric.

    np.bincount cannot fold a 2-D block -- it sums scalars -- so the per-fraction columns are
    accumulated with np.add.at."""
    ok = np.isfinite(C)
    w = np.where(ok, n_of_group[:, None].astype(np.float64), 0.0)
    num = np.zeros((n_genes, C.shape[1]), dtype=np.float64)
    den = np.zeros((n_genes, C.shape[1]), dtype=np.float64)
    np.add.at(num, gene_of_group, np.where(ok, C, 0.0) * w)
    np.add.at(den, gene_of_group, w)
    with np.errstate(invalid="ignore", divide="ignore"):
        per_gene = np.where(den > 0, num / den, np.nan)
    return (per_gene,
            np.nanmean(per_gene, axis=0) if np.isfinite(per_gene).any() else np.full(C.shape[1], np.nan),
            np.nanmedian(per_gene, axis=0) if np.isfinite(per_gene).any() else np.full(C.shape[1], np.nan))


def pooled_whole_panel(mu_all, y_all, u_all, fracs):
    """The POOLED lin_all curves: one global sort over the WHOLE panel, zeros included.

    NOT inside whole_panel_pass. That function loops over slides because its GROUPED statistics
    (per (gene, slide) and per (slide, spot)) need a per-entry DataFrame, which is what costs the
    memory; a pooled curve has no group at all and reads the three (N, G) matrices directly, and
    those are resident for the whole script either way. The extra cost is one |mu - y| copy plus
    one argsort index per score.

    READ THE mu_high CURVE FIRST HERE. On the whole panel the error at an undetected entry IS the
    prediction (|mu - 0| = |mu|), so sorting by anything that tracks mu largely sorts by error.
    That is not a reason to withhold the curve, it is the reason to print the mu-only baseline next
    to it: the gap between `pred` and `mu_high` is what the uncertainty head adds over the mean it
    is already conditioned on, and the curve states that mechanism instead of asserting it.

    Returns (curves, mean_err). The mean is the f = 0 value, kept for the summary so the agreement
    control covers lin_all like the two y > 0 cases."""
    err = np.abs(mu_all - y_all).ravel().astype(np.float64)
    values = {"uncertainty": u_all.ravel(), "mu": mu_all.ravel(), "err_lin": err}
    if U_MAY_BE_UNDEFINED:
        # kNN / cv form only, same drop as in whole_panel_pass: an undefined uncertainty has no
        # place in a sort. A strict no-op otherwise.
        keep = np.isfinite(values["uncertainty"])
        if not keep.all():
            values = {k: v[keep] for k, v in values.items()}
            err = values["err_lin"]
    return pooled_curves(err, values, spars_scores("err_lin"), fracs), float(err.mean())


def whole_panel_pass(mu_all, y_all, u_all, slides, gene_code, slide_code, adata_rows):
    """lin_all on BOTH axes: |y - mu| over EVERY (spot, gene) entry, zeros included.

    A separate pass, because the log error needs y > 0, which discards 62 % of the entries on
    10x_TuPro and 90 % on TLS -- exactly the regime where the count families differ. The linear
    error is defined at y = 0, so this is the only view of the whole panel the analysis has.

    MEMORY. Grouping is by (gene, slide) or (slide, spot) and a slide's spots never mix with
    another's, so the pass is LOOPED OVER SLIDES: the peak is one slide's (spots x genes) block, not
    the cohort's, and the result is EXACT, not an approximation of the single-shot version. Only the
    per-group frames come back (a few rows per gene / per spot); nothing per-entry is kept.

    The per-slide frame keys groups by the LOCAL gene column index and the LOCAL row, then maps them
    on the small result to the GLOBAL gene code and to obs["adata_row"] -- so the returned keys join
    to the y>0 frames, without ever putting an object column in the big per-entry frame.

    The grouping stays on the LOCAL column index rather than on the global code: it
    keeps the returned rows in var_names order, which is the order they were in before the codes
    existed, so the float summation order of the aggregation downstream is untouched."""
    gene_parts, spot_parts = [], []
    # Per-(gene, slide) sparsification curves for lin_all, collected slide by slide. A (gene,
    # slide) group holds every spot of ONE slide and never crosses into another, so computing the
    # curves inside this loop and stacking the blocks afterwards is EXACT, not an approximation of
    # a single-shot pass. Each entry is (curves_by_score, divisors, global_gene_codes, n_per_group)
    # for one slide; the fold to gene then model happens once, after the loop.
    spars_blocks = []
    spars_ok = True          # cleared by the first slide that cannot be curved; see the except below
    for sl in pd.unique(slides):
        rows = np.flatnonzero(slides == sl)
        sl_code = int(slide_code[rows[0]])       # constant over the slice, by construction
        mu, y, u = mu_all[rows], y_all[rows], u_all[rows]
        n_s, n_g = mu.shape
        d = pd.DataFrame({"gene": np.tile(np.arange(n_g, dtype=np.int32), n_s),
                          "spot": np.repeat(np.arange(n_s, dtype=np.int32), n_g),
                          "mu": mu.ravel(), "uncertainty": u.ravel(),
                          "err_lin": np.abs(mu - y).ravel()})
        del mu, y, u
        # kNN signal / cv form only: this frame never goes through the long table, so the entries
        # where the uncertainty is undefined have to be dropped here too. A strict no-op otherwise.
        if U_MAY_BE_UNDEFINED:
            keep = np.isfinite(d["uncertainty"].to_numpy())
            if not keep.all():
                d = d[keep]
            if d.empty:
                continue
        pg = partial_spearman_per_group(d, ["gene"], "err_lin")
        pg["gene"] = gene_code[pg["gene"].to_numpy()]     # local column index -> global gene code
        pg["slide"] = sl_code
        gene_parts.append(pg)
        ps = partial_spearman_per_group(d, ["spot"], "err_lin")
        ps["spot"] = adata_rows[rows][ps["spot"].to_numpy()]
        ps["slide"] = sl_code
        spot_parts.append(ps)
        # A (gene, slide) group on the whole panel holds every spot of the slide, so all of this
        # slide's groups share ONE size and therefore one bin count. Read off d rather than off
        # len(rows), because kNN mode has just dropped the entries with no defined signal and the
        # groups are no longer full slides there. Below the smallest regime the slide contributes
        # nothing rather than being binned into groups too small to rank.
        n_grp = int(len(d) // max(1, d["gene"].nunique()))
        nb = (REGIME_TO_NBINS["deciles"] if n_grp >= 500 else
              REGIME_TO_NBINS["quintiles"] if n_grp >= 150 else
              REGIME_TO_NBINS["terciles"] if n_grp >= 60 else 0)
        if nb and spars_ok:
            # Bins: equal-count quantile bins of the predicted mean WITHIN each gene of this slide,
            # from a rank rather than qcut (one pass, no per-group Python loop). Same rule as the
            # y>0 cases; it simply always lands on 10 here, because a lin_all group is a whole
            # slide and every slide holds thousands of spots.
            g = d.groupby(["gene"], sort=False)
            pct_mu = g["mu"].rank(pct=True, method="first").to_numpy()
            d = d.assign(bin=np.minimum((pct_mu * nb).astype(np.int16), nb - 1),
                         slide=sl_code)
            # The (gene, slide) code IS the local gene index here: one slide per block.
            a = d["gene"].to_numpy().astype(np.int64)
            n_a = np.bincount(a, minlength=n_g).astype(np.float64)
            err_sl = d["err_lin"].to_numpy().astype(np.float64)
            gkeys = {k: d[k].to_numpy() for k in ("gene", "slide", "bin")}
            vals = {"uncertainty": d["uncertainty"].to_numpy(), "mu": d["mu"].to_numpy(),
                    "err_lin": err_sl}          # the float64 one, see per_group_curves
            block = {}
            try:
                for tag, rm in (("gene_slide", ["gene", "slide"]),
                                ("gene_slide_bin", ["gene", "slide", "bin"])):
                    block[tag] = per_group_curves(gkeys, vals, err_sl, spars_scores("err_lin"),
                                                  FRACS, rm, a.astype(np.float64), n_g)
            except SparsificationPrecisionError as exc:
                # ONE slide failing drops the WHOLE lin_all case, both protocols. The blocks are
                # stacked into a single (gene, slide) table below, so keeping the slides that
                # happened to survive would fold a silently smaller slide set into every gene
                # average -- a number that looks like the others and is not the same quantity.
                spars_ok = False
                spars_blocks.clear()
                SPARS_SKIPPED.append(dict(case="lin_all",
                                          protocols=["gene_slide", "gene_slide_bin"],
                                          raised_on=f"slide {sl}, protocol {tag}", error=str(exc),
                                          note=SPARS_SKIPPED_NOTE))
                print(f"[{_el():6.1f}s] (!) lin_all per-(gene, slide) sparsification SKIPPED "
                      f"(slide {sl}, {tag}): {exc}", flush=True)
            else:
                spars_blocks.append((block, gene_code[np.arange(n_g)], n_a))
        del d
    cols = ["n"] + STATS
    pair = (pd.concat(gene_parts, ignore_index=True)[["gene", "slide"] + cols]
            if gene_parts else pd.DataFrame(columns=["gene", "slide"] + cols))
    spot = (pd.concat(spot_parts, ignore_index=True)[["slide", "spot"] + cols]
            if spot_parts else pd.DataFrame(columns=["slide", "spot"] + cols))
    # EMPTY fallback only: pd.DataFrame(columns=...) gives object-typed key columns, and merging
    # those against the int-keyed y>0 frames raises. The populated path is left untouched on
    # purpose -- coercing there would silently widen `spot` (obs["adata_row"]'s own dtype) to int64
    # and change a column of the saved artifact for no reason. Only reachable in kNN mode, where
    # every slide's signal can be undefined.
    if pair.empty:
        pair = pair.astype({"gene": "int64", "slide": "int64"})
    if spot.empty:
        spot = spot.astype({"slide": "int64", "spot": "int64"})
    # Stack the per-slide blocks into ONE (gene, slide) x fraction table per (protocol, score),
    # then hand them to the caller unfolded: the fold to gene and to model is shared with the y>0
    # cases and lives in one place.
    stacked = {}
    if spars_blocks:
        gene_of = np.concatenate([b[1] for b in spars_blocks])
        n_of = np.concatenate([b[2] for b in spars_blocks])
        for tag in ("gene_slide", "gene_slide_bin"):
            curves = {s: np.vstack([b[0][tag][0][s] for b in spars_blocks])
                      for s in spars_scores("err_lin")}
            divisors = np.concatenate([b[0][tag][1] for b in spars_blocks])
            stacked[tag] = (curves, divisors, gene_of, n_of)
    return pair, spot, stacked


def build_long_df(mu_all, y_all, u_all, slide_code, gene_code, adata_rows):
    """One row per (POSITIVE spot, gene): gene, slide, spot, mu, y, uncertainty, err_lin, err_log,
    mu_substituted. `spot` is obs["adata_row"], the project's canonical stable spot key, so a
    spot-axis result joins back to adata.obs / obsm["spatial"] like any other prediction.

    `gene` and `slide` are INTEGER CODES here, not names -- see the factorize call in the main body
    for why, and for why the names come back only at write time.

    ── SUBSTITUTION, not clamping ──────────────────────────────────────────────────────────────
    The prediction is substituted ONLY where log(mu) does not exist, i.e. mu <= 0. Everywhere else
    the model's own mean is used, however small it is.

    Substituting every 0 < mu < floor as well would rewrite the middle of the distribution rather
    than a tail: on this sweep such a rule fires on 12-35 % of the continuous entries and 22-47 %
    of the count entries, all of them entries whose logarithm exists, and the error it suppresses
    has a median of 0.57-0.85 against a mean |Delta| of 0.76-1.73. On a count head the floor also
    equals the smallest observable positive value, so it would set Delta to EXACTLY ZERO on 12-15 %
    of all entries, collapsing that population to one rank under the average-rank convention and
    attenuating the very rank statistic this canal reports.

    WHY THE 1st PERCENTILE as the substituted value. mu <= 0 is an out-of-support prediction, so no
    positive factor maps it onto a positive observation and the honest log-ratio is unbounded: every
    finite substitute understates. The 1st percentile of the observed positives ESTIMATES something
    -- the low end of what the assay reports in this cohort -- rather than being a chosen penalty;
    it is derived from the OBSERVATIONS alone, so every head of a family is scored by the same
    number, and it is robust where the smallest observed positive is a single extreme order
    statistic fixed by one deeply-sequenced spot. It understates, which is the direction that cannot
    be accused of manufacturing errors against the model. The sensitivity of the headline to it is
    MEASURED below (floor_sensitivity) rather than argued.

    Note the floor is a quantile of the observations, so it follows the native scale automatically:
    the 1st percentile of expm1(y)[y>0] is expm1 of the 1st percentile of y[y>0] (a quantile commutes
    with a monotone map). Nothing about the substitution changes with this rewrite -- neither which
    entries fire nor where the substitute sits in the distribution."""
    is_pos = y_all > 0
    n_pos = int(is_pos.sum())
    assert n_pos > 0, "no positive observation anywhere -- nothing to score on y > 0"
    floor = float(np.nanpercentile(y_all[is_pos], 1))

    spot_i, gene_j = np.nonzero(is_pos)
    mu = mu_all[spot_i, gene_j]
    y = y_all[spot_i, gene_j]
    sub = mu <= 0
    n_sub = int(sub.sum())
    print(f"floor diagnostic ({family}, y>0 entries, n={n_pos:,}): substituted mu<=0: {n_sub:,} "
          f"({100 * n_sub / n_pos:.4f}%); floor={floor:.6g}; "
          f"kept although below the floor: {int((mu < floor).sum()) - n_sub:,}", flush=True)

    mu_hat = np.where(sub, floor, mu)                # substitute the PREDICTION only; y untouched
    return pd.DataFrame({
        "gene": gene_code[gene_j], "slide": slide_code[spot_i], "spot": adata_rows[spot_i],
        "mu": mu, "y": y, "uncertainty": u_all[spot_i, gene_j],
        "err_lin": np.abs(mu - y),
        "err_log": np.abs(np.log(mu_hat) - np.log(y)),
        "mu_substituted": sub,
    }), floor, n_sub, n_pos


# ══════════════════════════════════════════════════════════════════════════════
# Secondary: the mu-binned control and the sparsification curves (y > 0 only).
# ══════════════════════════════════════════════════════════════════════════════

def assign_regime(pair):
    """From a (gene, slide) group's positive-entry count, its regime and bin count:
    deciles n >= 500 (B=10), quintiles 150 <= n < 500 (B=5), terciles 60 <= n < 150 (B=3).
    Below 60 there is no regime, which is the same floor the central metric applies."""
    n = pair["n"].to_numpy()
    # object array so the "no regime" sentinel stays a real None (np.select on string choices would
    # cast the default to the literal string "None"); overwrite in ascending tiers.
    regime = np.full(len(n), None, dtype=object)
    regime[n >= N_PLUS_MIN] = "terciles"
    regime[n >= 150] = "quintiles"
    regime[n >= 500] = "deciles"
    reg = pd.DataFrame({"gene": pair["gene"].to_numpy(), "slide": pair["slide"].to_numpy(),
                        "n_plus": n, "regime": regime})
    reg["n_bins"] = reg["regime"].map(REGIME_TO_NBINS)     # NaN where regime is None
    return reg


def add_mu_bins(df, reg_lookup):
    """Inner-join the regime groups onto the long df and assign each positive entry to a mu-quantile
    bin WITHIN its (gene, slide) group (B = that group's regime bin count)."""
    out = df.merge(reg_lookup, on=["gene", "slide"], how="inner")
    out["mu_rank_pct"] = out.groupby(["gene", "slide"])["mu"].rank(pct=True)
    out["bin"] = np.minimum((out["mu_rank_pct"] * out["n_bins"]).astype(int), out["n_bins"] - 1)
    return out.drop(columns=["mu_rank_pct"])


def binned_to_dataset(bin_xy, reg_lookup):
    """bin -> group (n-weighted) -> gene -> dataset: the binned analogue of the partial's cascade,
    so the two controls can be compared without the aggregation being a difference between them."""
    b = bin_xy[bin_xy["spearman"].notna()]
    if b.empty:
        return float("nan")
    pair = pd.DataFrame({"gene": b["gene"].to_numpy(), "slide": b["slide"].to_numpy(),
                         "w": b["spearman"].to_numpy() * b["n"].to_numpy(),
                         "sn": b["n"].to_numpy()}).groupby(["gene", "slide"], as_index=False).sum()
    pair["r"] = pair["w"] / pair["sn"]
    pair = pair[["gene", "slide", "r"]].merge(reg_lookup[["gene", "slide", "n_plus"]],
                                              on=["gene", "slide"]).rename(columns={"n_plus": "n"})
    return gene_axis_mean(pair, "r")


def sparsification_curves(df_binned, case, err_col, fracs):
    """The three protocols for ONE error column, all on the SAME population (y > 0, regime groups).

    Returns (pooled, per_group), where pooled is {score: curve} and per_group is
    {protocol: (curves_by_score, divisors, gene_of_group, n_of_group)} left UNFOLDED, so the fold
    to gene and to model is done once by the caller and is identical for lin_all.

    Both per-group protocols read THE SAME frame. That is not incidental: computing gene_slide on
    the full long table (it needs no bins) and gene_slide_bin on the binned subset would put the
    two rows of the figure on different gene sets with different starting errors.

    The ORACLE (removal ordered by the TRUE error) bounds what any score could achieve and is one
    of the scores; the mu-only baselines are the other two. See spars_scores for what each means
    and for why `random` is a drawn line rather than a computed curve."""
    # float64 HERE and not upstream: the operand matrices are float32 (see DT), but every line
    # below is a long accumulation over tens of millions of entries, where float32 drifts.
    err = df_binned[err_col].to_numpy().astype(np.float64)
    scores = spars_scores(err_col)
    values = {"uncertainty": df_binned["uncertainty"].to_numpy(),
              "mu": df_binned["mu"].to_numpy(),
              err_col: err}                    # the float64 one, see pooled_curves
    # (gene, slide) -> a contiguous code, and the gene / size of each such group.
    keys = df_binned[["gene", "slide"]]
    a, uniq = pd.factorize(pd.MultiIndex.from_frame(keys), sort=False)
    n_agg = len(uniq)
    gene_of = keys["gene"].to_numpy()[np.unique(a, return_index=True)[1]]
    n_of = np.bincount(a, minlength=n_agg).astype(np.float64)
    a = a.astype(np.float64)

    gkeys = {k: df_binned[k].to_numpy() for k in ("gene", "slide", "bin")}
    per_group = {}
    try:
        for tag, rm in (("gene_slide", ["gene", "slide"]),
                        ("gene_slide_bin", ["gene", "slide", "bin"])):
            curves, divisors = per_group_curves(gkeys, values, err, scores, fracs, rm, a, n_agg)
            per_group[tag] = (curves, divisors, gene_of, n_of)
    except SparsificationPrecisionError as exc:
        # BOTH protocols go, not only the one that raised: the paper reads them side by side as the
        # same curve under two removal units, so half a case would be a comparison it is not. The
        # POOLED curves below still run -- they re-sum the kept tail with .mean() and never
        # accumulate across groups, so the failure mode does not reach them.
        per_group = {}
        SPARS_SKIPPED.append(dict(case=case, protocols=["gene_slide", "gene_slide_bin"],
                                  raised_on=f"protocol {tag}", error=str(exc),
                                  note=SPARS_SKIPPED_NOTE))
        print(f"[{_el():6.1f}s] (!) {case} per-(gene, slide) sparsification SKIPPED ({tag}): {exc}",
              flush=True)
    return pooled_curves(err, values, scores, fracs), per_group


# ══════════════════════════════════════════════════════════════════════════════
# Main body — for THIS one run.
# ══════════════════════════════════════════════════════════════════════════════

run_dir_p = Path(run_dir)   # in the non-run_dir modes this IS the --ensemble-dir / --*-dir path

print(f"[{_el():6.1f}s] loading test predictions ...", flush=True)
if mode == "ensemble":
    B = load_test_predictions_ensemble(run_dir_p)
    assert "deep_ensemble" in B.uns, f"{run_dir_p}: missing uns['deep_ensemble']"
    ens_model_type = str(B.uns["deep_ensemble"]["model_type"])
    assert ens_model_type == model_type, (
        f"test_summaries.h5ad model_type={ens_model_type!r} != {model_type!r} from "
        f"{metrics_json_path} -- stale ensemble file? re-run deep_ensemble_combine.py.")
elif mode in POSTHOC_MODES:
    B = load_posthoc_summaries(run_dir_p, mode)   # the mode name IS the subdirectory name
else:
    B = load_test_predictions(run_dir_p)          # run_dir AND knn: the model's own predictions
print(f"[{_el():6.1f}s] loaded {B.n_obs:,} test spots x {B.n_vars:,} genes.", flush=True)

# kNN mode: the uncertainty, aligned to B, resolved ONCE (a module-level global read by
# predictions_and_uncertainty). NaN is expected and is not an error: sigma_knn is missing wherever a
# (spot, gene) had fewer than min_pos_neighbors detected neighbours, a real property of the cohort's
# zero rate. It is never FILLED -- those entries are DROPPED, once, right after the long table is
# built (and inside the whole-panel loop), because they do not drop out on their own: pandas' rank()
# keeps NaN. The rate is printed AND recorded in summary.json.
KNN_SIGNAL = None
if mode == "knn":
    KNN_SIGNAL = load_knn_signal(Path(args.knn_dir), B, args.knn_signal)
    # NON-finite, not just NaN: the drop below is np.isfinite, so counting only NaN here would
    # report "0.00%" for a signal carrying infinities that then vanish silently from the population.
    _n_nan = int((~np.isfinite(KNN_SIGNAL)).sum())
    print(f"[{_el():6.1f}s] kNN signal '{args.knn_signal}' aligned to B: {KNN_SIGNAL.shape}, "
          f"non-finite on {_n_nan:,} / {KNN_SIGNAL.size:,} entries "
          f"({100 * _n_nan / KNN_SIGNAL.size:.2f}%)", flush=True)

count_convention = None if family == "continuous" else assert_count_convention(B, M)
report_axis = native_axis(family, M)
print(f"[{_el():6.1f}s] scale: NOTHING is converted; mu, y and the errors are on {report_axis}.",
      flush=True)

# The three (N, G) matrices, built ONCE and shared by both passes. B is freed immediately after.
MU, U = predictions_and_uncertainty(B, mode, family, uncertainty_source=args.uncertainty_source)
if args.uncertainty_form != "sd":
    # Here rather than inside predictions_and_uncertainty, so that function keeps ONE job (which
    # variance does this mode supply) and the deviation from the canal's definition sits at the
    # top level where it is impossible to miss. B is still alive: one_plus_alpha reads a layer.
    U = apply_uncertainty_form(MU, U, B)
Y = _dense(B.layers["y"]).astype(DT)
# .astype(str) HERE and nowhere else, so the slide has ONE representation: whole_panel_pass masks
# rows with it while SLIDE_CODE below is factorized from it, and two raw values that happened to
# share a str() would otherwise get one code but two masks -- i.e. duplicate (gene, slide) keys that
# the final merge would silently fan out.
SLIDES = B.obs["slide"].to_numpy().astype(str)
ADATA_ROWS = B.obs["adata_row"].to_numpy()
GENES = np.asarray(B.var_names)
n_entries_total = int(MU.size)
del B

# ── The group keys travel as INTEGER CODES, never as strings ──────────────────────────────────
# Every cascade below groups by (gene, slide) or (slide, spot) over tens of millions of rows, and
# pandas factorizes the key columns on each groupby. On an object column that means hashing Python
# string objects row by row, which is the single most expensive step of the whole script; on int64
# it is a fast contiguous pass. The long table also shrinks (4 bytes per key instead of an object
# pointer), which matters at the memory ceiling this job runs at. The names come back exactly once,
# at write time, so the parquets are byte-for-byte what they were.
#
# sort=True is NOT cosmetic: it assigns code i to the i-th smallest NAME, so grouping by code puts
# the groups in the same order grouping by name did. That preserves the float summation order in
# gene_axis_mean, which is what makes this optimisation bit-identical rather than merely equivalent.
GENE_CODE, GENE_NAMES = pd.factorize(GENES, sort=True)
SLIDE_CODE, SLIDE_NAMES = pd.factorize(SLIDES, sort=True)     # SLIDES is already the str form
print(f"[{_el():6.1f}s] group keys encoded: {len(GENE_NAMES):,} genes, {len(SLIDE_NAMES)} slides.",
      flush=True)

PAIR, SPOT = {}, {}      # case -> the per-group frame, gene axis / spot axis

# ── lin_all: the whole panel, zeros included. Runs FIRST, before the long table exists. ──
print(f"[{_el():6.1f}s] lin_all: whole panel (zeros included), both axes ...", flush=True)
PAIR["lin_all"], SPOT["lin_all"], SPARS_LIN_ALL = whole_panel_pass(MU, Y, U, SLIDES, GENE_CODE,
                                                                   SLIDE_CODE, ADATA_ROWS)
# The POOLED lin_all curves, straight off the resident matrices -- see pooled_whole_panel for why
# they are not accumulated inside the slide loop above.
SPARS_LIN_ALL_POOLED, MEAN_ERR_LIN_ALL = pooled_whole_panel(MU, Y, U, FRACS)
print(f"[{_el():6.1f}s] lin_all pooled curves: mean |y - mu| = {MEAN_ERR_LIN_ALL:.6g} at f=0, "
      f"pred {SPARS_LIN_ALL_POOLED['pred'][-1] / MEAN_ERR_LIN_ALL:.3f} vs "
      f"mu_high {SPARS_LIN_ALL_POOLED['mu_high'][-1] / MEAN_ERR_LIN_ALL:.3f} vs "
      f"oracle {SPARS_LIN_ALL_POOLED['oracle'][-1] / MEAN_ERR_LIN_ALL:.3f} at f=0.9 (relative).",
      flush=True)


PER_GROUP_ROWS = []      # the (gene, slide) curves, written to per_group_curves.parquet


def emit_curves(case, pooled, per_group, into):
    """Fold the per-(gene, slide) curves to gene then to model, name the columns, and record the
    diagnostics. Shared by lin_all and the two y>0 cases so the naming and the fold exist once.

    Prints, per protocol: the SUPPORT (how many groups are defined at each fraction) and the MEAN
    vs MEDIAN over genes. The support is the cheap version of "is the gene set the same at every
    fraction": if it is flat there is nothing to worry about, and if it falls you see it here
    instead of it hiding inside a nanmean. Mean vs median matters because a group's relative curve
    is not bounded by 1 -- removing the most uncertain entries can remove the SMALLEST errors -- so
    a few genes can carry the mean."""
    for name, c in pooled.items():
        into[f"curve_{name}_pooled_{case}"] = c
    for tag, (curves, divisors, gene_of, n_of) in per_group.items():
        for name, C in curves.items():
            per_gene, mean_c, med_c = fold_slides_then_genes(C, gene_of, n_of, len(GENE_NAMES))
            into[f"curve_{name}_{tag}_{case}"] = mean_c
            into[f"curve_{name}_{tag}_{case}__median"] = med_c
            if name == "pred":
                sup = np.isfinite(C).sum(axis=0)
                print(f"[{_el():6.1f}s] {case}/{tag}: {C.shape[0]:,} (gene,slide) groups; "
                      f"support {sup[0]:,} at f=0 -> {sup[-1]:,} at f=0.9"
                      f"{'  (CONSTANT)' if sup[0] == sup[-1] else '  (!) SHRINKS'}; "
                      f"mean/median at f=0.5 = {mean_c[len(mean_c)//2]:.3f} / "
                      f"{med_c[len(med_c)//2]:.3f}", flush=True)
                # How degenerate is this score? A removal unit with a single distinct value has no
                # internal order at all, and that -- not the curve's shape -- is what says a head
                # carries no within-group information.
                flat = int((np.nanmax(C, axis=1) == np.nanmin(C, axis=1)).sum())
                print(f"           groups whose pred curve never moves: {flat:,}/{C.shape[0]:,}",
                      flush=True)
        # The divisors ARE each group's own mean error, i.e. the f=0 value that was divided out.
        # Saved because the per-group normalization leaves every curve starting at 1.0 by
        # construction, so the value itself is no longer readable off the curve. With the divisors on
        # disk the normalization stays auditable without a rerun (their n-weighted mean over a
        # gene, then the mean over genes, must reproduce the reported mean error).
        PER_GROUP_ROWS.append(pd.DataFrame({
            "case": case, "protocol": tag, "gene": gene_of, "n": n_of, "mean_err": divisors,
            **{f"f{int(round(100*f)):02d}_{name}": curves[name][:, q].astype(np.float32)
               for name in curves for q, f in enumerate(FRACS)}}))

# ── The y>0 long table, shared by lin_pos and log_pos. SKIPPED under LIN_ALL_ONLY. ──
u_undef_n_dropped, u_undef_pct_dropped = 0, 0.0
if LIN_ALL_ONLY:
    # Nothing below this point may read `df`: in a standardized target space the y>0 population is
    # not the detected population (see the note where LIN_ALL_ONLY is set). The whole-panel pass
    # above has already produced everything this mode reports.
    del MU, U, Y
    df, floor, n_substituted, n_positive = None, float("nan"), 0, 0
    print(f"[{_el():6.1f}s] lin_all ONLY (target_transform={TARGET_TRANSFORM!r}): skipping the y>0 "
          f"long table, lin_pos, log_pos and every artifact derived from them.", flush=True)
else:
    df, floor, n_substituted, n_positive = build_long_df(MU, Y, U, SLIDE_CODE, GENE_CODE, ADATA_ROWS)
    del MU, U, Y

    if U_MAY_BE_UNDEFINED:
        # Dropped at the single point where the long frame is born, so every consumer downstream sees
        # ONE population and r_uD / r_um / r_Dm are computed over identical rows (a per-column drop would
        # make the partial formula combine three different samples). Consequence when reading the
        # output: n_plus_pos then means "entries with a DEFINED uncertainty", which is exactly what the
        # N_PLUS_MIN gate should be filtering on.
        _n0 = len(df)
        df = df[np.isfinite(df["uncertainty"].to_numpy())].reset_index(drop=True)
        u_undef_n_dropped = _n0 - len(df)
        u_undef_pct_dropped = float(100.0 * u_undef_n_dropped / _n0) if _n0 else float("nan")
        _what = (f"kNN signal {args.knn_signal!r}" if mode == "knn"
                 else f"uncertainty form {args.uncertainty_form!r}")
        assert len(df) > 0, (f"{_what} is undefined on ALL {_n0:,} positive (spot, gene) entries -- "
                             f"nothing left to score.")
        # Recount on the SURVIVING frame: build_long_df returned these over the pre-drop population,
        # while every other count in summary.json is post-drop.
        n_substituted, n_positive = int(df["mu_substituted"].sum()), len(df)
        print(f"[{_el():6.1f}s] {_what}: dropped {u_undef_n_dropped:,}/{_n0:,} "
              f"({u_undef_pct_dropped:.2f}%) positive entries where it is undefined; "
              f"{len(df):,} rows kept.", flush=True)

    print(f"[{_el():6.1f}s] long df: {len(df):,} (positive spot, gene) rows "
          f"of {n_entries_total:,} entries ({100 * len(df) / n_entries_total:.1f}%).", flush=True)

    for case, err_col in (("lin_pos", "err_lin"), ("log_pos", "err_log")):
        print(f"[{_el():6.1f}s] {case}: gene axis ...", flush=True)
        PAIR[case] = partial_spearman_per_group(df, ["gene", "slide"], err_col)
        print(f"[{_el():6.1f}s] {case}: spot axis ...", flush=True)
        SPOT[case] = partial_spearman_per_group(df, ["slide", "spot"], err_col)

AXIS = {}
for case in CASES_RUN:
    AXIS[f"gene_{case}"] = axis_stats(PAIR[case], "gene")
    AXIS[f"spot_{case}"] = axis_stats(SPOT[case], "spot")
for key, s in AXIS.items():
    print(f"[{_el():6.1f}s] {key:<16s} partial={s['partial']:+.4f} naive={s['naive']:+.4f}  "
          f"({s['n_defined_partial']:,}/{s['n_included']:,} groups with a defined partial, "
          f"{s['n_groups']:,} present)", flush=True)

if LIN_ALL_ONLY:
    # Everything from here to the assembly is defined on the y>0 population, which does not exist in
    # this mode. The neutral values below are what the assembly reads; the ONE real quantity that
    # survives is the lin_all curve set, which came out of the whole-panel pass and its pooled twin.
    mean_err_positive, binned_by_regime, residual_confound = {}, {}, {}
    bin_stats, df_binned = None, None
    sparsification = {"frac": FRACS}
    emit_curves("lin_all", SPARS_LIN_ALL_POOLED, SPARS_LIN_ALL, sparsification)
    print(f"[{_el():6.1f}s] lin_all ONLY: no binned control, no _pos sparsification curves, "
          f"no bin_stats/long_df parquet.", flush=True)

# ── Sensitivity of log_pos to the SUBSTITUTED VALUE (see build_long_df) ──────────────────────
# The substituted entries all carry Delta = |log(c) - log(y)|, so changing the substitute c shifts
# that whole block by the constant log(c1/c2) and nothing else. Rather than argue for one value, the
# gene-axis headline is recomputed at half and double the floor and the spread is reported. Both
# alternatives are derived from the same observed quantile, so no per-family constant enters. It
# runs only when something was actually substituted -- dead, and free, for every count family, whose
# mean is exp(f(x)) > 0 by construction.
floor_sensitivity = {}
if not LIN_ALL_ONLY and n_substituted > 0:
    sub = df["mu_substituted"].to_numpy()
    y_sub = df["y"].to_numpy()[sub]
    base = df["err_log"].to_numpy()
    for name, c in (("floor_half", floor / 2.0), ("floor_double", floor * 2.0)):
        alt = base.copy()
        alt[sub] = np.abs(np.log(c) - np.log(y_sub))
        d2 = pd.DataFrame({"gene": df["gene"].to_numpy(), "slide": df["slide"].to_numpy(),
                           "mu": df["mu"].to_numpy(), "uncertainty": df["uncertainty"].to_numpy(),
                           "alt": alt})
        floor_sensitivity[name] = gene_axis_mean(
            partial_spearman_per_group(d2, ["gene", "slide"], "alt"), "r_partial")
        del d2, alt
    print(f"[{_el():6.1f}s] floor sensitivity of log_pos ({n_substituted:,} substituted entries, "
          f"gene-axis partial {AXIS['gene_log_pos']['partial']:+.4f}): "
          f"{ {k: round(v, 4) for k, v in floor_sensitivity.items()} }", flush=True)

# ── SECONDARY: the mu-binned control + the sparsification curves, on y > 0 ───────────────────
# The whole block is y>0-only, hence the guard: the mu-BINS themselves come from assign_regime on
# PAIR["lin_pos"], which does not exist under LIN_ALL_ONLY.
if not LIN_ALL_ONLY:
    print(f"[{_el():6.1f}s] binned control + sparsification ...", flush=True)
    reg = assign_regime(PAIR["lin_pos"])                      # the y>0 group sizes decide the bin count
    reg_lookup = reg[reg["regime"].notna()][["gene", "slide", "n_plus", "n_bins", "regime"]]
    # Taken BEFORE df is released: this is the mean over EVERY positive entry, whereas everything below
    # lives on the regime-group subset (the population the curves start from). Two different means of
    # the same column, so both are reported under names that say which.
    mean_err_positive = {c: float(df[e].to_numpy().mean(dtype=np.float64))
                         for c, e in (("lin_pos", "err_lin"), ("log_pos", "err_log"))}
    df_binned = add_mu_bins(df, reg_lookup)
    del df                        # the merge above copied everything it needs; holding both doubles the peak
    print(f"[{_el():6.1f}s] binned population: {len(df_binned):,} entries in "
          f"{len(reg_lookup):,} regime groups.", flush=True)

    binned_by_regime, sparsification, bin_frames = {}, {"frac": FRACS}, {}
    # lin_all: per-(gene, slide) from the whole-panel pass, pooled from pooled_whole_panel.
    emit_curves("lin_all", SPARS_LIN_ALL_POOLED, SPARS_LIN_ALL, sparsification)
    for case, err_col in (("lin_pos", "err_lin"), ("log_pos", "err_log")):
        b = rankcorr_per_group(df_binned, ["gene", "slide", "bin"], "uncertainty", err_col)
        b = b.merge(reg_lookup[["gene", "slide", "regime"]], on=["gene", "slide"])
        for R in REGIME_TO_NBINS:
            binned_by_regime[f"{R}_{case}"] = binned_to_dataset(b[b["regime"] == R], reg_lookup)
        bin_frames[case] = b.rename(columns={"spearman": f"spearman_{case}"})
        pooled, per_group = sparsification_curves(df_binned, case, err_col, FRACS)
        emit_curves(case, pooled, per_group, sparsification)

    # The two errors are ranked over the SAME bins (both are finite wherever y > 0), so the outer merge
    # is a formality that would only show up if that ever stopped being true.
    bin_stats = bin_frames["lin_pos"][["gene", "slide", "bin", "n", "regime", "spearman_lin_pos"]].merge(
        bin_frames["log_pos"][["gene", "slide", "bin", "spearman_log_pos"]],
        on=["gene", "slide", "bin"], how="outer")

    # ── RESIDUAL CONFOUND: does mu still rank the uncertainty / the error INSIDE a mu-bin? ───────
    # If it does, the binning has not removed the confound and the binned control above is optimistic.
    # Same cascade as the binned Spearman, so the three numbers are directly comparable.
    residual_confound = dict(
        mu_vs_uncertainty=binned_to_dataset(
            rankcorr_per_group(df_binned, ["gene", "slide", "bin"], "mu", "uncertainty"), reg_lookup),
        mu_vs_err_lin_pos=binned_to_dataset(
            rankcorr_per_group(df_binned, ["gene", "slide", "bin"], "mu", "err_lin"), reg_lookup),
        mu_vs_err_log_pos=binned_to_dataset(
            rankcorr_per_group(df_binned, ["gene", "slide", "bin"], "mu", "err_log"), reg_lookup),
    )

# ══ Assemble and write ════════════════════════════════════════════════════════
def _tag(frame, case, keys, n_name=None):
    """Rename one case's four statistics with its suffix, so the three cases can share one file
    without a column ever holding a value its name disagrees with."""
    out = frame.rename(columns={c: f"{c}_{case}" for c in STATS})
    cols = keys + [f"{c}_{case}" for c in STATS]
    if n_name is not None:
        out = out.rename(columns={"n": n_name})
        cols = keys + [n_name] + [f"{c}_{case}" for c in STATS]
    return out[cols]


# lin_all's group set is a SUPERSET of the y>0 one (a group with no detected entry has no _pos row
# at all), hence the outer merge. log_pos shares lin_pos's population exactly, so it contributes no
# second count column. Under LIN_ALL_ONLY the two merges simply do not happen, so the files carry the
# lin_all columns alone -- same names, fewer of them.
pair_out = _tag(PAIR["lin_all"], "lin_all", ["gene", "slide"], "n_plus_all")
spot_out = _tag(SPOT["lin_all"], "lin_all", ["slide", "spot"], "n_plus_all")
if not LIN_ALL_ONLY:
    pair_out = (pair_out
                .merge(_tag(PAIR["lin_pos"], "lin_pos", ["gene", "slide"], "n_plus_pos"),
                       on=["gene", "slide"], how="outer")
                .merge(_tag(PAIR["log_pos"], "log_pos", ["gene", "slide"]),
                       on=["gene", "slide"], how="outer"))
    spot_out = (spot_out
                .merge(_tag(SPOT["lin_pos"], "lin_pos", ["slide", "spot"], "n_plus_pos"),
                       on=["slide", "spot"], how="outer")
                .merge(_tag(SPOT["log_pos"], "log_pos", ["slide", "spot"]),
                       on=["slide", "spot"], how="outer"))

summary = dict(
    dataset=dataset, model_type=model_type, lambda_dir=lambda_dir, mode=mode,
    # Gate on THIS, not on the absence of a key: an artifact tagged with another convention carries
    # different column names on another scale and is not comparable to this one.
    canal2_convention="v3_native_scale",
    is_ensemble=(mode == "ensemble"), is_mc_dropout=(mode == "mc_dropout"),
    is_laplace=(mode == "laplace"), is_knn=(mode == "knn"),
    uncertainty_source=(args.uncertainty_source if mode not in ("run_dir", "knn") else None),
    # WHICH FUNCTION of the predictive was scored: "sd" is the canal's standing definition, "cv" and
    # "one_plus_alpha" are the robustness-appendix alternatives. Gate on it before putting a number
    # beside a reference one -- the three are different quantities, not three estimates of one.
    uncertainty_form=args.uncertainty_form,
    knn_signal=(args.knn_signal if mode == "knn" else None),
    knn_dir=(os.path.abspath(args.knn_dir) if mode == "knn" else None),
    # How many positive entries were dropped because the UNCERTAINTY is undefined there: a missing
    # kNN signal, or cv = sd/mu at mu <= 0. 0 whenever it cannot happen. Read it before comparing an
    # n_plus count with another run's: the populations differ.
    uncertainty_undefined_n_entries_dropped=int(u_undef_n_dropped),
    uncertainty_undefined_pct_entries_dropped=float(u_undef_pct_dropped),
    # Two kNN-specific aliases of the two counts above. They are 0 outside kNN mode.
    knn_n_entries_dropped=int(u_undef_n_dropped if mode == "knn" else 0),
    knn_pct_entries_dropped=float(u_undef_pct_dropped if mode == "knn" else 0.0),
    # The axis the UNCERTAINTY lives on, which in kNN mode is NOT report_axis (that one describes mu
    # and y, which do come from the scored model). Recorded rather than converted: it must never be
    # compared in MAGNITUDE with a model's sigma, and every statistic here is rank-based anyway.
    uncertainty_scale=(knn_uncertainty_scale(args.knn_signal) if mode == "knn"
                       else {"sd": "sqrt(variance), on the model's own output scale",
                             "cv": "sqrt(variance) / mu, dimensionless (coefficient of variation)",
                             "one_plus_alpha": "1 + alpha, the NB dispersion alone, dimensionless",
                             }[args.uncertainty_form]),
    knn_cluster_scheme=(KNN_PRIMARY_SCHEME if mode == "knn" and args.knn_signal.startswith("d_norm")
                        else None),
    # d_norm is normalized PER LOPO FOLD. The per-group statistics are safe (a slide belongs to
    # exactly one donor, so every rank comparison happens inside one fold), but the sparsification
    # curves rank across all donors and therefore mix per-fold divisors.
    knn_pooled_stats_caveat=("d_norm is normalized per LOPO fold; the sparsification curves mix "
                             "per-fold cluster radii -- prefer the per-group partial"
                             if mode == "knn" and args.knn_signal.startswith("d_norm") else None),
    # Provenance of the scale that is NOT applied. report_axis names the axis mu / y / the errors
    # sit on; count_convention is the guard that the count mean and dispersion share it.
    report_axis=report_axis, count_convention=count_convention,
    # THE gate for a reader: lin_all_only says the y>0 half of this canal was never computed, and
    # target_transform says why. When true, mu / y / the errors are in the STANDARDIZED space, so
    # every rank statistic below is still valid but the scale-bound ones (mean_err) are in z units
    # and never comparable to an untransformed run's.
    target_transform=TARGET_TRANSFORM, lin_all_only=LIN_ALL_ONLY,
    error_cases=CASES_RUN,
    error_floor_policy="v2_substitute_where_undefined",
    floor=float(floor), n_substituted=int(n_substituted),
    pct_substituted=(float(100.0 * n_substituted / n_positive) if n_positive else float("nan")),
    floor_sensitivity=floor_sensitivity,
    n_entries_total=n_entries_total,
    # None, NOT 0, under LIN_ALL_ONLY. The y>0 population was never COUNTED there, and a literal 0
    # would read as "this run has no detected entry" -- a statement about the cohort instead of
    # about what was computed, and the kind of false number a provenance block exists to prevent.
    n_entries_positive=(None if LIN_ALL_ONLY else int(n_positive)),
    pct_positive=(None if LIN_ALL_ONLY else float(100.0 * n_positive / n_entries_total)),
    # THE headline block: one entry per (axis, case), each with its naive value, its controlled
    # value and the counts behind them. Keys are f"{axis}_{case}".
    axis_stats=AXIS,
    # Scale-BOUND, unlike every correlation above: a mean error carries the units of report_axis, so
    # it compares across heads of ONE family and never across families. Two populations per case,
    # named apart: `*_positive` is every y > 0 entry, `*_sparsification_population` is the
    # regime-group subset the curves run on -- so a curve's f = 0 value must equal the latter
    # exactly, which is the cheapest check that the two are reading the same data.
    mean_err=(dict(lin_all_sparsification_population=MEAN_ERR_LIN_ALL) if LIN_ALL_ONLY else dict(
        # lin_all's population is the WHOLE panel, so it has no `*_positive` counterpart: the two
        # means would be the same number. Present in BOTH modes, unlike everything else here,
        # because the lin_all curves are the one thing LIN_ALL_ONLY still computes.
        lin_all_sparsification_population=MEAN_ERR_LIN_ALL,
        lin_pos_positive=mean_err_positive["lin_pos"],
        log_pos_positive=mean_err_positive["log_pos"],
        lin_pos_sparsification_population=float(df_binned["err_lin"].to_numpy().mean(dtype=np.float64)),
        log_pos_sparsification_population=float(df_binned["err_log"].to_numpy().mean(dtype=np.float64)),
    )),
    # EMPTY on a healthy run. Non-empty means the per-(gene, slide) curve_* columns of the listed
    # cases are ABSENT from sparsification.parquet and per_group_curves.parquet -- and says why,
    # so their absence is never read as "this head has no signal". The pooled curves and every rank
    # statistic are still there and still valid; see SparsificationPrecisionError.
    sparsification_skipped=SPARS_SKIPPED,
    binned_spearman=binned_by_regime,
    residual_confound=residual_confound,
)

def decode_keys(frame):
    """Integer codes -> the gene / slide NAMES, applied ONCE, immediately before writing.

    Everything upstream groups and merges on the codes -- that is the whole point of them -- but
    every reader of these files joins on the names, so the artifacts are unchanged. Only the columns
    that exist are touched, so one helper serves the per-group frames and the long table. The frames
    passed here are all freshly built (a merge or a column selection), so writing into them in place
    aliases nothing."""
    if "gene" in frame.columns:
        frame["gene"] = GENE_NAMES[frame["gene"].to_numpy()]
    if "slide" in frame.columns:
        frame["slide"] = SLIDE_NAMES[frame["slide"].to_numpy()]
    return frame


print(f"[{_el():6.1f}s] writing outputs to {out_dir} ...", flush=True)
decode_keys(pair_out).to_parquet(os.path.join(out_dir, "pair_partial.parquet"), index=False)
decode_keys(spot_out).to_parquet(os.path.join(out_dir, "spot_partial.parquet"), index=False)
pd.DataFrame(sparsification).to_parquet(os.path.join(out_dir, "sparsification.parquet"), index=False)
# The per-(gene, slide) curves behind the aggregates above, so the SPREAD is inspectable without a
# rerun -- which is what points 1 and 5 of the design review actually need: whether the gene set is
# stable across fractions, and whether a few genes carry the mean. float32 because these are
# already-normalized ratios read for shape, never re-accumulated.
if PER_GROUP_ROWS:
    pg = pd.concat(PER_GROUP_ROWS, ignore_index=True)
    pg["gene"] = GENE_NAMES[pg["gene"].to_numpy()]
    pg.to_parquet(os.path.join(out_dir, "per_group_curves.parquet"), index=False)
    print(f"[{_el():6.1f}s] per_group_curves.parquet: {len(pg):,} rows "
          f"({pg['case'].nunique()} case(s) x {pg['protocol'].nunique()} protocols).", flush=True)
elif SPARS_SKIPPED:
    # Say it here too, not only in the JSON: a missing artifact is the kind of thing a reader
    # attributes to the wrong cause when the log is silent about it.
    print(f"[{_el():6.1f}s] per_group_curves.parquet NOT written -- every case was skipped "
          f"({len(SPARS_SKIPPED)} record(s) under summary.json['sparsification_skipped']).",
          flush=True)
# bin_stats and long_df are the two artifacts that exist ONLY on the y>0 population, so under
# LIN_ALL_ONLY they are not written at all rather than written empty: an empty parquet would read as
# "computed, nothing passed the gate", which is a different and false statement.
if not LIN_ALL_ONLY:
    decode_keys(bin_stats).to_parquet(os.path.join(out_dir, "bin_stats.parquet"), index=False)
    # `bin` is persisted: with the mu-bin on disk, the gene_slide_bin curves and the binned control
    # are an offline re-derivation over this parquet instead of a reconstruction that has to
    # re-guess the regime and the tie convention.
    decode_keys(df_binned)     # in place, and this is the frame's last use: everything it fed
                               # (the curves, the binned control, the mean_err scalars) is already done
    df_binned[["gene", "slide", "spot", "bin", "regime", "mu", "y", "uncertainty",
               "err_lin", "err_log", "mu_substituted"]].to_parquet(
        os.path.join(out_dir, "long_df.parquet"), index=False)
with open(os.path.join(out_dir, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2, default=float)

print(f"[{_el():6.1f}s] done. {json.dumps(summary, indent=2, default=float)}", flush=True)
