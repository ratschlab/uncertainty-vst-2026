# snake/scripts/canals/canal3_zero_detection_precompute.py
#
# Canal 3: does the model know which (spot, gene) cells are true zeros? For one run it scores,
# per (gene, slide) pair, the AUROC of a zero-likelihood score against the observed zero locus.
# One run per job, so a sweep runs as concurrent SLURM jobs (launch_canal3_precompute.sh).
#
# Two mutually exclusive input modes, selected by which flag is passed.
#
#   --run-dir <classic run dir>   reads predictions/preds_fold*.h5ad + metrics.json and scores a
#       single model: continuous -> -X ; poisson -> -mu ; nb -> the exact P(Y=0) from mu and the
#       dispersion layer.
#
#   --ensemble-dir <.../<model_type>/ensemble/>   reads the one consolidated test_summaries.h5ad
#       written by snake/scripts/epistemic/deep_ensemble_combine.py. That file already holds one
#       row per spot, LOPO consolidated, so there is no glob and no per-fold concat. It scores the
#       ensemble's own predictive mixture, not a per-member model:
#         continuous     score = -X, X = the mean over members of the member means.
#         nb / poisson   score = layers["zero_prob"], read as-is. That layer is already the exact
#           member-averaged mixture P(Y=0): a PMF value at one point is an expectation under each
#           member's own predictive, so the equal-weight mixture value is exactly the average of
#           the members' values. Nothing is recomputed from mu/dispersion in this mode.
#       A missing zero_prob layer raises instead of falling back to a -mu proxy. P(Y=0) is convex
#       in mu, so a mean-mu proxy under-estimates the mixture P(Y=0) by Jensen's inequality, and
#       the bias grows with cross-member disagreement, which is exactly the regime this analysis
#       studies. The fix is to re-run deep_ensemble_combine.py for that model dir with its
#       per-(seed, fold) gstar_fold*.h5ad files present.
#       Run identity: uns["deep_ensemble"] carries model_type but not dataset, so dataset comes
#       from the first member's seed_{s}/metrics.json, one directory above ensemble/, and
#       model_type is cross-checked against uns as a consistency guard.
#
# msemeanonly is in scope here, unlike canal 2: zero detection needs only a monotone proxy score
# (-X), not a real uncertainty estimate, so it works for any model.
#
# Output (under <run-dir>/canal3_zero_detection/ or <ensemble-dir>/canal3_zero_detection/):
#   pair_auroc.parquet     -- slide, gene, n_zero, n_pos, auroc, one row per (gene, slide) pair
#   gene_zero_rate.parquet -- gene, zero_rate. A property of the dataset, so every run of it gives
#                             the same values; written per output so each output stands alone.
#   summary.json           -- dataset, model_type, lambda_dir, n_genes_evaluated, n_pairs,
#                             auroc_dataset_mean, auroc_dataset_median, plus is_ensemble and
#                             n_members in ensemble mode only.
#
# Not applicable under a per-gene target transform (--run-dir mode): such a run stores y
# standardized per gene, which moves the zero, so (y == 0) is not the non-detected locus. The
# script then writes summary.json {not_applicable: true, reason, target_transform}, writes no
# parquet and exits 0. See the guard right after the run identity.
#
# Usage (repo root, on the cluster; CPU only):
#   python -u snake/scripts/canals/canal3_zero_detection_precompute.py --run-dir <classic run dir>
#   python -u snake/scripts/canals/canal3_zero_detection_precompute.py \
#       --ensemble-dir <results>/<dataset>/<run_id>/<gene_set>/<radius>/<model_type>/ensemble

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
import torch

_T0 = time.time()
def _el():
    return time.time() - _T0

# ── Args ──────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
mode = p.add_mutually_exclusive_group(required=True)
mode.add_argument("--run-dir", default=None,
                  help="A classic run dir (holds predictions/preds_fold*.h5ad, metrics.json).")
mode.add_argument("--ensemble-dir", default=None,
                  help="A deep-ensemble .../<model_type>/ensemble/ dir (holds test_summaries.h5ad, "
                       "written by deep_ensemble_combine.py) -- scores the ENSEMBLE-COMBINED "
                       "prediction directly, no per-seed loop.")
p.add_argument("--out-dir", default=None,
               help="Default: <run-dir>/canal3_zero_detection/ or <ensemble-dir>/canal3_zero_detection/")
args = p.parse_args()

IS_ENSEMBLE = args.ensemble_dir is not None
run_dir = os.path.abspath(args.run_dir) if args.run_dir else None
ensemble_dir = os.path.abspath(args.ensemble_dir) if args.ensemble_dir else None
base_dir = ensemble_dir if IS_ENSEMBLE else run_dir
out_dir = args.out_dir or os.path.join(base_dir, "canal3_zero_detection")
os.makedirs(out_dir, exist_ok=True)

MIN_SIDE = 20   # per (gene, slide) pair: need >= this many zeros AND >= this many positives

CONTINUOUS_MODELS = {"msemeanonly", "homoscedastic", "heteroscedastic", "spot_homoscedastic"}
POISSON_MODELS    = {"poisson_cp", "poisson_cp_nr", "poisson_raw"}
NB_MODELS         = {"negativebinomial_hetero_noanchor_cp",
                     "negativebinomial_hetero_noanchor_cp_nr",
                     "negativebinomial_hetero_noanchor_raw",
                     # Restricted-dispersion cp NB (alpha per spot / per gene): same zero
                     # probability P(Y=0) = (1 + alpha*mu)^(-1/alpha), same cp_counts target,
                     # only the grain of alpha differs. The _nr twins sit on the unrounded CP_T
                     # target, which changes nothing here: P(Y=0) is a property of the
                     # predictive, not of the target.
                     "negativebinomial_spotdisp_cp", "negativebinomial_genedisp_cp",
                     "negativebinomial_spotdisp_cp_nr", "negativebinomial_genedisp_cp_nr",
                     "STI_nb_cp",        # STimage-fidelity NB benchmark, cp_counts target
                     "STI_nb_raw"}       # STimage head on raw counts (== nb_noanchor_raw, bare head)

# No reporting-axis tag here, unlike canal 1/2. Everything reported is the AUROC of a
# per-(gene,slide) ranking, and both scores are invariant to the axis: -mu / -X is monotone in mu
# under any positive rescale, and the NB P(Y=0) is evaluated from mu and alpha as the model fitted
# them. A depth-blind *_raw run (depth_scale None, no divisor anywhere) is therefore directly
# comparable to a _cp or a Gaussian run, and count_mean applies no depth_scale at all, which is
# why the None case needs no special handling below.

def model_family(model_type):
    if model_type in CONTINUOUS_MODELS:
        return "continuous"
    if model_type in POISSON_MODELS:
        return "poisson"
    if model_type in NB_MODELS:
        return "nb"
    raise ValueError(f"unknown model family for model_type={model_type!r} -- add it to one of the sets above")

def _dense(a):
    return a.toarray() if sp.issparse(a) else np.asarray(a)


if not IS_ENSEMBLE:
    # ── Run identity — read from metrics.json (authoritative), never parsed from the path ──
    with open(os.path.join(run_dir, "metrics.json")) as f:
        M = json.load(f)
    dataset = M["dataset"]
    model_type = M["model_type"]
    family = model_family(model_type)
    lambda_dir = os.path.basename(run_dir) if os.path.basename(run_dir).startswith("lambda_") else None
    print(f"run_dir     : {run_dir}", flush=True)
    print(f"dataset     : {dataset}  model_type: {model_type} ({family})  lambda_dir: {lambda_dir}", flush=True)

    # ── A per-gene TARGET TRANSFORM makes this whole canal undefined ──────────────────────────
    # Everything here rests on one predicate: is_zero = (layers["y"] == 0), "the assay detected
    # nothing at this (spot, gene)". A run trained with target_transform != "none" (gene_zscore /
    # gene_minmax) stores y standardized, y_z = (y - loc_g)/scale_g, an affine map that moves the
    # zero: a non-detected entry becomes y_z = -loc_g/scale_g != 0. Then (y == 0) selects nothing,
    # every pair fails the MIN_SIDE gate, and the run writes a summary.json full of NaN, a silent
    # wrong answer. Exit gracefully instead, with the not_applicable shape canal2 also uses: the
    # sentinel is still produced, so the Snakemake chain completes and nothing downstream has to
    # special-case a missing file. Scoring such a run would mean recomputing the zero locus from
    # layers["y_logcp10k"], which train.py writes for exactly this purpose; not done here, because
    # it would mix two scales inside one artifact.
    target_transform = str(M.get("target_transform", "none"))
    if target_transform != "none":
        reason = (f"target_transform={target_transform!r}: layers['y'] is standardized per gene, so "
                  f"(y == 0) is not the non-detected locus and zero detection is undefined")
        print(f"{reason} -- Canal 3 does not apply, skipping gracefully (not a failure).", flush=True)
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(dict(dataset=dataset, model_type=model_type, lambda_dir=lambda_dir,
                           target_transform=target_transform,
                           not_applicable=True, reason=reason), f, indent=2)
        sys.exit(0)
else:
    print(f"ensemble_dir: {ensemble_dir}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# --run-dir mode only: load the per-fold test predictions and turn them into a
# zero-likelihood score.
# ══════════════════════════════════════════════════════════════════════════════

def load_test_predictions(run_dir):
    pred_files = sorted((run_dir / "predictions").glob("preds_fold*.h5ad"))
    assert pred_files, f"no predictions in {run_dir / 'predictions'}"
    parts = []
    for f in pred_files:
        p = sc.read_h5ad(f, backed="r")           # lazy -- avoids reading the "val" rows we're about to drop
        parts.append(p[p.obs["split"].to_numpy() == "test"].to_memory())  # materialize only these rows off disk
    B = ad.concat(parts, axis=0, merge="same")
    assert np.unique(B.obs["adata_row"]).size == B.n_obs, "test spots overlap across folds -- non-LOPO?"
    return B

def count_mean(B):
    """mu on the scale the count likelihood was fitted on, which is this file's X untouched.

    X is mu on the model's own target scale (cp_counts / cp_counts_nr / raw_counts) and
    layers["dispersion"] is the alpha fitted at that same scale, so the P(Y=0) below needs no
    constant applied to it. Returns X after checking the file carries the convention marker.
    """
    # The marker is read from metrics.json (M) first and the predictions uns second; metrics.json
    # is the authoritative run provenance. count_mean runs only in run-dir mode, where the
    # module-level M is defined.
    if (M.get("count_convention") or B.uns.get("count_convention")) != "v2_no_offset":
        raise ValueError(f"{run_dir}: artifact with no count_convention marker. "
                         f"Re-run the precompute.")
    return _dense(B.X).astype(np.float64)                               # (N, G)

def zero_scores(B, family):
    """Higher score = model thinks this (spot, gene) cell is more likely a true zero."""
    if family == "continuous":
        return -_dense(B.X).astype(np.float64)
    mu = count_mean(B)                                                  # fitted-scale mean (N, G)
    if family == "poisson":
        return -mu                                                      # monotone in exp(-mu) -> same AUROC, no overflow
    if family == "nb":
        alpha = _dense(B.layers["dispersion"]).astype(np.float64)       # (N, G)
        mu_t, a_t = torch.as_tensor(mu), torch.as_tensor(alpha)
        r_t = 1.0 / a_t
        dist = torch.distributions.NegativeBinomial(total_count=r_t, logits=torch.log(mu_t) + torch.log(a_t))
        return dist.log_prob(torch.zeros_like(mu_t)).exp().numpy()      # exact P(Y=0)
    raise ValueError(family)


# ══════════════════════════════════════════════════════════════════════════════
# --ensemble-dir mode. Only how B and `score` are obtained differs from --run-dir
# mode; the aggregation below (auroc_matrix / per_pair_auroc / weighted_mean) is shared.
# ══════════════════════════════════════════════════════════════════════════════

def load_ensemble_test_summaries(ensemble_dir_p):
    """The deep-ensemble consolidated TEST summaries: one row per spot (LOPO covers each spot
    exactly once), so no per-fold concat is needed (unlike --run-dir's preds_fold*.h5ad)."""
    path = ensemble_dir_p / "test_summaries.h5ad"
    assert path.is_file(), f"no ensemble summaries at {path}"
    B = sc.read_h5ad(path)
    if "split" in B.obs.columns:                                       # defensive: LOPO test-only file
        vals = set(pd.unique(B.obs["split"]))
        assert vals == {"test"}, f"expected obs['split'] constant 'test' in {path}, got {vals}"
    assert np.unique(B.obs["adata_row"]).size == B.n_obs, "test spots overlap -- non-LOPO?"
    return B

def ensemble_run_identity(ensemble_dir_p, B):
    """dataset / model_type / lambda_dir for --ensemble-dir mode.

    B.uns["deep_ensemble"] carries model_type, n_members, seeds and more, but not dataset, so both
    come from the first member's seed_{s}/metrics.json one directory above "ensemble/", and
    model_type is cross-checked against uns. Identity is never parsed from the path. lambda_dir
    uses the same basename().startswith("lambda_") rule as --run-dir mode, applied to that parent
    directory, since ensemble/ is one extra segment after the model_type[/lambda_X] dir."""
    assert "deep_ensemble" in B.uns, (
        f"{ensemble_dir_p / 'test_summaries.h5ad'} has no uns['deep_ensemble'] -- "
        f"not a deep_ensemble_combine.py output?")
    de = B.uns["deep_ensemble"]
    model_dir = ensemble_dir_p.parent                      # .../<model_type>[/lambda_X], parent of "ensemble"
    seed0 = int(np.asarray(de["seeds"]).ravel()[0])
    with open(model_dir / f"seed_{seed0}" / "metrics.json") as f:
        M = json.load(f)
    dataset = M["dataset"]
    model_type = M["model_type"]
    assert model_type == str(de["model_type"]), (
        f"model_type mismatch: seed_{seed0}/metrics.json={model_type!r} vs "
        f"uns['deep_ensemble']['model_type']={de['model_type']!r}")
    lambda_dir = model_dir.name if model_dir.name.startswith("lambda_") else None
    return dataset, model_type, lambda_dir

def zero_scores_ensemble(B, family):
    """Ensemble-mode score. Higher = this (spot, gene) cell is more likely a true zero, the same
    convention as zero_scores().

    continuous: score = -X, X = the ensemble mean of the per-member means. No ensemble-specific
      correction is needed: -X is a ranking proxy, not a probability estimate, and negation is
      linear, so mean_s(-m_s) == -mean_s(m_s) exactly. There is no Jensen gap here, unlike the NB
      case below.

    nb / poisson: score = layers["zero_prob"], consumed as-is. It is already the exact
      member-averaged mixture P(Y=0) (see this file's header for the linearity argument), written
      by deep_ensemble_combine.py, so no dispersion access happens here at all. If the layer is
      absent this raises rather than falling back to a -mu proxy: P(Y=0) is convex in mu, so a
      mean-mu proxy under-estimates the mixture P(Y=0) by Jensen's inequality, worst exactly where
      members disagree.

      Both NB and Poisson get the layer from the same code path in deep_ensemble_combine.py, which
      decides per member which formula applies ((r/(r+mu_c))^r for NB, exp(-mu_c) for Poisson)
      from whether that member's own alpha is available. So an absent layer means that script must
      be run for this model dir with its gstar_fold*.h5ad precompute files present.
    """
    if family == "continuous":
        return -_dense(B.X).astype(np.float64)
    if family in ("nb", "poisson"):
        if "zero_prob" not in B.layers:
            raise KeyError(
                f"ensemble mode requires layers['zero_prob'] for family={family!r} (the exact "
                f"per-seed-averaged mixture P(Y=0)) -- missing from this test_summaries.h5ad. "
                f"Refusing to fall back to a naive -mu proxy: P(Y=0) is CONVEX in mu, so a "
                f"mean-mu proxy systematically UNDER-estimates the true mixture P(Y=0) (Jensen's "
                f"inequality), with the bias growing with cross-seed disagreement (epistemic "
                f"uncertainty) -- exactly the regime this analysis exists to study. "
                f"Re-run deep_ensemble_combine.py for this model dir with its gstar_fold*.h5ad "
                f"precompute files present (want_pit needs want_gstar too -- see that script's "
                f"header COUNT-CANAL EXTENSION note), or re-combine if this file predates that "
                f"extension."
            )
        return _dense(B.layers["zero_prob"]).astype(np.float64)
    raise ValueError(family)


# ══════════════════════════════════════════════════════════════════════════════
# Shared by both modes: the per-(gene, slide) AUROC. Only how B and the score are
# obtained differs above.
# ══════════════════════════════════════════════════════════════════════════════

def auroc_matrix(is_zero, score):
    """Vectorized per-column AUROC via the rank-sum identity (Mann-Whitney U). is_zero/score: (n_spots, G).
    Returns (auroc, n_zero, n_pos), each shape (G,). Ties are NOT rank-averaged (fast path);
    negligible for continuous scores."""
    n_zero = is_zero.sum(axis=0)
    n_pos = (~is_zero).sum(axis=0)
    order = np.argsort(score, axis=0)
    ranks = np.empty_like(order, dtype=np.float64)
    cols = np.arange(score.shape[1])
    ranks[order, cols] = np.arange(1, score.shape[0] + 1)[:, None]
    sum_ranks_zero = np.where(is_zero, ranks, 0.0).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        auroc = (sum_ranks_zero - n_zero * (n_zero + 1) / 2.0) / (n_zero * n_pos)
    return auroc, n_zero, n_pos

def per_pair_auroc(B, score):
    y = _dense(B.layers["y"]).astype(np.float64)          # raw counts (count models) or log1p-CP10k (continuous) -- same zero locus
    is_zero = (y == 0)
    slides = B.obs["slide"].to_numpy()
    genes = np.asarray(B.var_names)
    parts = []
    for sl in pd.unique(slides):
        m = slides == sl
        auroc, n_zero, n_pos = auroc_matrix(is_zero[m], score[m])
        ok = (n_zero >= MIN_SIDE) & (n_pos >= MIN_SIDE)
        parts.append(pd.DataFrame({
            "slide": sl, "gene": genes[ok],
            "n_zero": n_zero[ok].astype(int), "n_pos": n_pos[ok].astype(int), "auroc": auroc[ok],
        }))
    return pd.concat(parts, ignore_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# The one run, or ensemble. The mode branch covers only how B, the run identity and
# the score are obtained; everything downstream is identical in the two modes.
# ══════════════════════════════════════════════════════════════════════════════

print(f"[{_el():6.1f}s] loading {'ensemble test summaries' if IS_ENSEMBLE else 'test predictions'} ...", flush=True)
if IS_ENSEMBLE:
    ensemble_dir_p = Path(ensemble_dir)
    B = load_ensemble_test_summaries(ensemble_dir_p)
    dataset, model_type, lambda_dir = ensemble_run_identity(ensemble_dir_p, B)
    family = model_family(model_type)
    n_members = int(np.asarray(B.uns["deep_ensemble"]["n_members"]))
    print(f"dataset     : {dataset}  model_type: {model_type} ({family})  lambda_dir: {lambda_dir}  "
          f"(ensemble, n_members={n_members})", flush=True)
else:
    run_dir_p = Path(run_dir)
    B = load_test_predictions(run_dir_p)
print(f"[{_el():6.1f}s] loaded {B.n_obs:,} test spots x {B.n_vars:,} genes", flush=True)

y = _dense(B.layers["y"]).astype(np.float64)
gene_zero_rate = pd.Series((y == 0).mean(axis=0), index=B.var_names, name="zero_rate")

score = zero_scores_ensemble(B, family) if IS_ENSEMBLE else zero_scores(B, family)
pair_df = per_pair_auroc(B, score)
del B, score
print(f"[{_el():6.1f}s] {len(pair_df):,} (gene,slide) AUROC pairs computed", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Per-gene weighted aggregate, then the dataset headline.
# ══════════════════════════════════════════════════════════════════════════════

def weighted_mean(values, weights):
    return np.average(values, weights=weights)

pair_df_w = pair_df.assign(weight=np.minimum(pair_df["n_zero"], pair_df["n_pos"]))
per_gene = (
    pair_df_w.groupby("gene")
    .apply(lambda g: weighted_mean(g["auroc"], g["weight"]), include_groups=False)
    .rename("auroc_gene")
)

summary = dict(
    dataset=dataset, model_type=model_type, lambda_dir=lambda_dir,
    n_genes_evaluated=int(per_gene.size), n_pairs=int(len(pair_df)),
    auroc_dataset_mean=float(per_gene.mean()), auroc_dataset_median=float(per_gene.median()),
)
if IS_ENSEMBLE:
    # Ensemble-only keys. --run-dir mode writes neither, rather than is_ensemble=false, so a
    # reader tells the two summary.json shapes apart by presence.
    summary["is_ensemble"] = True
    summary["n_members"] = n_members

print(f"[{_el():6.1f}s] writing outputs to {out_dir} ...", flush=True)
pair_df.to_parquet(os.path.join(out_dir, "pair_auroc.parquet"), index=False)
gene_zero_rate.reset_index().rename(columns={"index": "gene"}).to_parquet(
    os.path.join(out_dir, "gene_zero_rate.parquet"), index=False)
with open(os.path.join(out_dir, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"[{_el():6.1f}s] done. {summary}", flush=True)
