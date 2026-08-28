"""Per-model benchmark summary, one row per (held-out donor, gene).

Builds a compact per-(held-out donor, gene) table plus scalar metadata for one classic LOPO run and
writes it next to the run, so a cross-model benchmark aggregates light files instead of re-reading
the heavy ``predictions/preds_fold{k}.h5ad``. This module is the single source of truth for the
per-gene metrics that benchmark uses.

Two entry points. ``export_summary(run_dir)`` reads a classic training run's ``atoms.parquet``.
``export_summary_ensemble(ensemble_dir)`` handles a deep ensemble, which has no ``atoms.parquet``
because it is not a training run, so it rebuilds the per-(slide, gene) atoms from
``snake/scripts/epistemic/deep_ensemble_combine.py``'s consolidated ``test_summaries.h5ad`` using
the same ``per_slide_gene_corr`` / ``per_slide_gene_calibration`` primitives.

Why the donor axis is kept. Under leave-one-patient-out each fold is a different held-out donor with
disjoint test slides, so a mean over folds averages nothing: it pools all donors into one number and
destroys the per-donor structure. The headline is instead the per-donor point x_i (that donor's
slide-aware, gene-averaged metric), then the mean over donors with an inter-donor Student-t CI,
df = K-1. This mirrors ``train.py``'s ``lopo_stats``. The per-(donor, gene) table keeps both axes so
a consumer can show per-donor figures before averaging.

Conventions:
- TEST split only. Validation drives early stopping, so scoring it would leak.
- Slide-aware reduction within each held-out donor: per-(test_donor, inner_split, slide, gene) atom
  -> mean over inner_split and slides -> one value per (test_donor, gene). NaN-skipping.
- Calibration is native-CDF one-sided PIT: ``ece_pit``, ``pit_bias``, ``disp_signed``.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.count_metrics import count_pred_var
from src.evaluation.metrics_slide_aware import (
    per_slide_gene_corr, _aggregate_corr, per_slide_gene_calibration, _aggregate_calibration,
    _safe_pearson,
)
import scanpy as sc

# Read straight from atoms.parquet. export_summary drops any column a run lacks, so listing the full
# PIT set is safe even for a mean-only run, which carries no calibration columns.
# "pearson_logmean" exists for count runs only. It is the second estimator of the same log1p
# quantity train.py's atoms_dataframe writes next to "pearson": {metric} = g* = E[log1p(Y/div)] uses
# mu and alpha, {metric}_logmean = log1p(mu/div) uses the mean alone and sits above g* by the Jensen
# gap. Listing it here carries that comparison into the per-(donor, gene) table and the headline
# with no recomputation. Gaussian runs have a single estimator, so the column is absent there.
ATOM_METRICS = ["pearson", "pearson_logmean", "mae", "mse", "ece_pit", "pit_bias", "disp_signed",
                "var_residual_corr", "var_residual_corr_partial"]
COVERAGE_GRID = np.linspace(0.0, 1.0, 101)[1:]           # 0.01 .. 1.0 risk-coverage grid

# target_source values that mean "this run fits a count likelihood (Poisson / NB) directly on its
# own target". Their predictive variance is recomputed here as count_pred_var(X, dispersion) on that
# same fitted scale, never read from the file (see iter_fold_predictions).
# This asks a reading-side question of a saved run. specs.COUNT_ROUTED_TARGETS holds the same values
# but is the import-time guard on which (loss, target_source) pairs a spec may declare. The two
# answer different questions and stay two separate definitions.
COUNT_TARGET_SOURCES = ("raw_counts", "cp_counts", "cp_counts_nr")


def report_axis(target_source, depth_scale):
    """Which axis this run's headline correlation lives on: "log1p_cp10k" or "log1p_raw_counts".

    Recorded in meta so a cross-model table cannot silently put a depth-blind *_raw count model
    (depth_scale None, divisor 1.0, so it reports on log1p(raw counts)) on the same axis as a _cp or
    a Gaussian model, both of which report on log1p(CP10k). Those numbers are not comparable: group
    or split by this tag before ranking anything.

    Pure classifier. The count-convention marker is checked upstream, before this is reached: in
    load_run for a run dir, in export_summary_ensemble for a deep ensemble.
    """
    if target_source not in COUNT_TARGET_SOURCES:
        return "log1p_cp10k"                    # Gaussian / MSE: the target IS log1p(CP10k)
    return "log1p_cp10k" if depth_scale is not None else "log1p_raw_counts"


# --------------------------------------------------------------------------- run loading (cheap)
def load_run(run_dir):
    """metrics.json + TEST atoms (cheap); lists the preds_fold files lazily (the heavy part)."""
    run_dir = Path(run_dir)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    atoms = pd.read_parquet(run_dir / "atoms.parquet")
    atoms_test = atoms[atoms["split"] == "test"].copy()
    pred_files = sorted((run_dir / "predictions").glob("preds_fold*.h5ad"))
    model_type = metrics.get(
        "model_type", run_dir.parent.name if run_dir.name.startswith("lambda_") else run_dir.name
    )
    # heteroscedastic <=> per-spot variance <=> var_residual_corr defined (not all-NaN)
    is_hetero = atoms_test["var_residual_corr"].notna().any() if "var_residual_corr" in atoms_test else False
    # A count-likelihood run is identified by its target_source, which every metrics.json records.
    if "target_source" not in metrics:
        raise ValueError(
            f"{run_dir}: metrics.json has no 'target_source', so the reporting axis cannot "
            f"be determined. Re-run the precompute for this run.")
    target_source = metrics["target_source"]
    is_count = target_source in COUNT_TARGET_SOURCES
    # The single convention check of this module's run path. It lives here because load_run is the
    # one entry point every consumer goes through, including notebooks that call
    # iter_fold_predictions directly. It fires before any prediction file is opened, so a count run
    # whose X sits on a different scale can never reach count_pred_var or e_aurc.
    if is_count and metrics.get("count_convention") != "v2_no_offset":
        raise ValueError(
            f"{run_dir}: metrics.json has no count_convention marker, so the scale of the "
            f"saved mean and dispersion cannot be determined. Re-run the precompute.")
    return {
        "run_dir": run_dir,
        "metrics": metrics,
        "atoms": atoms_test,
        "pred_files": pred_files,
        "model_type": model_type,
        "target_source": target_source,
        "is_count": is_count,
        "is_hetero": bool(is_hetero),
        "is_lopo": bool(metrics.get("lopo", False)),
        "primary_corr_metric": metrics.get("primary_corr_metric", "pearson"),
    }


def reduce_atoms_per_donor(atoms, metric):
    """Slide-aware reduction within each held-out donor: one value per (test_donor, gene).

    Mean over inner_split, then over slides, NaN-skipping throughout. Returns a Series indexed by
    (test_donor, gene). Under LOPO each fold is one held-out donor, so grouping by ``test_donor`` is
    the per-fold view; averaging over folds instead would pool the donors together.
    """
    per_slide_gene = atoms.groupby(["test_donor", "slide", "gene"])[metric].mean()   # over inner_split
    return per_slide_gene.groupby(["test_donor", "gene"]).mean()                      # over slides


# --------------------------------------------------------------------------- predictions (heavy)
def iter_fold_predictions(run):
    """Yield (donor, mu, y, var, genes, slides, gene_mean) for each fold's TEST predictions.

    Under LOPO each fold's test spots are one held-out donor and the folds are disjoint, so mu and
    var are never averaged across folds. Spots are sorted by obs_name for determinism.

    gene_mean is the per-gene mean of the target, paired with the sigma of the same run, so the
    across-gene sigma-vs-mean Pearson is a within-run diagnostic:
      - Gaussian runs: X, y and pred_var are all on the log1p(CP10k) axis, so gene_mean = mean of y
        is log1p(CP10k) as well and both sides share that scale.
      - Gaussian runs under a target transform (z-score / min-max): X, y and pred_var are on the
        standardized scale, and the file carries a y_logcp10k layer holding the original target,
        which is what gene_mean reads. pred_var has no such twin, so here the two sides of the
        correlation sit on different scales.
      - Count runs: no y_logcp10k layer, so gene_mean is the mean of y, the model's own count target
        (round(CP_T) / CP_T / raw counts) on a linear count scale, not log1p(CP10k). Here
        sigma = sqrt(count_pred_var(mu, alpha)) lives on that same count scale.
    So sigma_mean_corr is a per-run diagnostic on the model's own axis. It is not harmonized to
    log1p(CP10k) the way the headline g* correlation is, so a Gaussian's log-vs-log value and a count
    model's countscale-vs-countscale value must not be ranked against each other.
    """
    for f in run["pred_files"]:
        ad = sc.read_h5ad(f)
        ad = ad[ad.obs["split"] == "test"]
        donor = str(ad.obs["test_donor"].iloc[0]) if "test_donor" in ad.obs else Path(f).stem
        ad = ad[np.argsort(ad.obs_names.to_numpy())]
        y = np.asarray(ad.layers["y"], dtype=float)
        mu = np.asarray(ad.X, dtype=float)
        # A count run's predictive variance is recomputed from X and the dispersion on the model's
        # own fitted scale (Poisson var = mu, NB var = mu + alpha*mu^2). Count prediction files
        # carry no pred_var layer.
        if run["is_count"]:
            var = count_pred_var(mu, ad.layers["dispersion"] if "dispersion" in ad.layers else None)
        elif "pred_var" in ad.layers:
            var = np.asarray(ad.layers["pred_var"], dtype=float)
        else:
            raise KeyError(
                f"{f}: no layers['pred_var'] and target_source={run['target_source']!r} is not a "
                "count target, so the predictive variance cannot be recovered."
            )
        genes = ad.var_names.to_numpy()
        slides = ad.obs["slide"].to_numpy()
        ylog = (np.asarray(ad.layers["y_logcp10k"], dtype=float)
                if "y_logcp10k" in ad.layers else y)
        yield donor, mu, y, var, genes, slides, ylog.mean(axis=0)


def _rc_curve(resid_sq, score):
    """Risk-coverage for ONE (slide, gene): rank spots by score ascending (most-confident first),
    cumulative mean residual^2, returned on COVERAGE_GRID."""
    order = np.argsort(score)
    running = np.cumsum(resid_sq[order]) / np.arange(1, len(resid_sq) + 1)
    cov = np.arange(1, len(resid_sq) + 1) / len(resid_sq)
    return np.interp(COVERAGE_GRID, cov, running)


def eaurc_per_gene(mu, y, var, slides, genes):
    """Per-gene E-AURC (slide-aware) as a Series indexed by gene, for ONE donor's test spots.

    Per (slide, gene): AURC = mean of the normalized risk-coverage curve, ranked by sigma (model)
    and by |residual| (oracle); average over slides; E-AURC_g = AURC_model - AURC_oracle.
    Lower = sigma ranks spots more like the true error.
    """
    resid_sq = (y - mu) ** 2
    sigma = np.sqrt(var)
    abs_resid = np.abs(y - mu)
    slide_ids = np.unique(slides)
    out = {}
    for g in range(mu.shape[1]):
        m_slides, o_slides = [], []
        for s in slide_ids:
            rows = slides == s
            full = resid_sq[rows, g].mean()
            if not np.isfinite(full) or full == 0:
                continue
            m_slides.append(_rc_curve(resid_sq[rows, g], sigma[rows, g]).mean() / full)
            o_slides.append(_rc_curve(resid_sq[rows, g], abs_resid[rows, g]).mean() / full)
        if m_slides:
            out[genes[g]] = float(np.mean(m_slides) - np.mean(o_slides))
    return pd.Series(out, name="e_aurc")


def sigma_mean_corr(gene_mean, var):
    """Pearson across genes between per-gene mean predicted sigma and the gene's expression level.

    The "uncertainty just follows the gene mean" diagnostic, for one donor's test spots: per gene,
    mean predicted sigma against that gene's mean target, correlated across genes. Near 1 means sigma
    mostly encodes each gene's expression scale, near 0 means sigma is decoupled from it. Both inputs
    come from iter_fold_predictions and carry the per-family scale documented there, so the value is
    a within-run diagnostic. Defined for any probabilistic run. Returns a scalar.
    """
    sigma_g = np.sqrt(var).mean(axis=0)      # per-gene mean predicted sigma (over this donor's spots)
    gene_mean = np.asarray(gene_mean, dtype=float)
    ok = np.isfinite(sigma_g) & np.isfinite(gene_mean)
    if ok.sum() < 3:
        return float("nan")
    return float(np.corrcoef(sigma_g[ok], gene_mean[ok])[0, 1])


# --------------------------------------------------------------------------- run identity (from path)
def parse_run_identity(run_dir):
    """(dataset, run_id, gene_set, radius, model_type, lambda) from the results path layout
    results/{dataset}/{run_id}/{gene_set}/{radius}/{model_type}/[lambda_{lam}]."""
    run_dir = Path(run_dir)
    if run_dir.name.startswith("lambda_"):
        lam = run_dir.name[len("lambda_"):]
        base = run_dir.parent
    else:
        lam = None
        base = run_dir
    return {
        "model_type": base.name,
        "radius": base.parent.name,
        "gene_set": base.parents[1].name,
        "run_id": base.parents[2].name,
        "dataset": base.parents[3].name,
        "lambda": lam,
    }


# --------------------------------------------------------------------------- LOPO donor aggregation
def donor_lopo_stats(per_donor_values):
    """K per-donor points -> mean, inter-donor SD (ddof=1), SEM, Student-t 95% CI on the mean + a
    prediction interval for a new donor (df = K-1). Mirrors train.py's lopo_stats exactly.

    These intervals are optimistic (folds share K-2 training donors; Bengio & Grandvalet 2004), so
    the per-donor points are kept alongside. Returns a dict (NaN-padded for K<2).
    """
    from scipy import stats as _scipy_stats
    x = np.asarray(list(per_donor_values.values()) if isinstance(per_donor_values, dict)
                   else per_donor_values, dtype=float)
    keys = (list(per_donor_values.keys()) if isinstance(per_donor_values, dict)
            else [str(i) for i in range(len(x))])
    finite = np.isfinite(x)
    xv = x[finite]
    k = int(len(xv))
    mean = float(np.mean(xv)) if k else float("nan")
    sd   = float(np.std(xv, ddof=1)) if k >= 2 else float("nan")
    sem  = float(sd / np.sqrt(k)) if k >= 2 else float("nan")
    if k >= 2:
        tstar = float(_scipy_stats.t.ppf(0.975, k - 1))
        half, half_pi = tstar * sem, tstar * sd * float(np.sqrt(1.0 + 1.0 / k))
        ci, pi = [mean - half, mean + half], [mean - half_pi, mean + half_pi]
    else:
        tstar, ci, pi = float("nan"), [float("nan")] * 2, [float("nan")] * 2
    return {
        "mean": mean, "sd": sd, "sem": sem, "K": k, "df": k - 1,
        "t_star_0975": tstar, "ci95_mean": ci, "pi95_new_donor": pi,
        "per_donor": {str(d): (None if not np.isfinite(v) else float(v)) for d, v in zip(keys, x)},
    }


def gene_bootstrap_stats(per_gene_values, n_boot=1000, seed=0, draws=None):
    """N per-gene values -> mean + percentile 95% CI + bootstrap SE over resamples of the GENES.

    Companion to ``donor_lopo_stats``, and a DIFFERENT ESTIMAND -- not a competing estimate of the
    same number, so the two are never interchangeable. Writing each per-(donor, gene) score as
    ``r_dg = theta + a_d + b_g + e_dg``:
      - ``donor_lopo_stats`` resamples PATIENTS with the gene panel fixed, so the common gene term
        cancels in the between-donor spread and it targets sigma2_don/K + sigma2_res/(K*N): the
        generalization-to-a-new-patient component. That is the headline error bar.
      - this function resamples genes with the donor cohort fixed, so the common donor term cancels
        instead and it targets sigma2_gene/N + sigma2_res/(K*N), containing zero inter-donor
        variance. It answers "is the mean over this gene set stable?", never "does this generalize to
        a new patient". Label any reported value as gene-panel variability, patients fixed.

    ``per_gene_values``: the per-gene scores of one cutoff, already averaged over donors, for example
    a row mean of ``gene_donor_matrix``. The set is fixed, with no re-selection of a top-n inside the
    bootstrap, so the interval is conditional on that set and does not account for the selection that
    produced it. A top-n ranked on the plotted metric stays optimistic and no bootstrap fixes that.

    ``draws``: optional pre-drawn ``(n_boot, N)`` index array, sized on the number of gene positions,
    NaNs included, since nanmean skips them. Pass the same draws across models so a model-vs-model
    comparison is paired, which cancels each gene's own difficulty.

    The percentile CI is preferred over ``mean +/- 1.96*se_boot`` because it is not forced to be
    symmetric: for a right-skewed quantity the two bounds genuinely differ. Returns a dict,
    NaN-padded for fewer than 2 finite values.
    """
    v = np.asarray(list(per_gene_values.values()) if isinstance(per_gene_values, dict)
                   else per_gene_values, dtype=float)
    n = int(v.size)
    n_finite = int(np.isfinite(v).sum())
    mean = float(np.nanmean(v)) if n_finite else float("nan")
    if n_finite < 2:
        return {"mean": mean, "se_boot": float("nan"), "ci95_percentile": [float("nan")] * 2,
                "N": n, "N_finite": n_finite, "n_boot": 0, "seed": seed}
    if draws is None:
        draws = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    # One vectorized gather instead of a python loop over draws. This assumes the (n_boot, N) index
    # array is small enough to materialize in one go, which holds at the gene-set sizes used here.
    boot_means = np.nanmean(v[draws], axis=1)
    return {
        "mean": mean,
        "se_boot": float(np.std(boot_means, ddof=1)),          # gene-panel SE, patients FIXED
        "ci95_percentile": [float(np.percentile(boot_means, 2.5)),
                            float(np.percentile(boot_means, 97.5))],
        "N": n, "N_finite": n_finite, "n_boot": int(boot_means.size), "seed": seed,
    }


def donor_and_gene_stats(mat, n_boot=1000, seed=0, draws=None):
    """(gene x donor) matrix for ONE metric -> every available uncertainty, as one flat dict.

    Merges the two families, which are different estimands and must never be swapped (variance
    decomposition in ``gene_bootstrap_stats``):
      ``donor_*`` average over genes first -> K donor points -> ``donor_lopo_stats``: Student-t on
                  the mean, patients resampled and gene panel fixed, the generalization-to-a-new-
                  patient bar. Plus ``donor_pi_*``, where a new donor would itself land, much wider
                  than the CI and a different question, not an error bar on the mean.
      ``gene_*``  average over donors first -> N gene points -> ``gene_bootstrap_stats``: percentile
                  CI resampling genes with the donor cohort fixed, so it holds zero inter-donor
                  variance. Gene-panel stability only, and typically far narrower, N genes against K
                  donors. Never report it as the headline error bar.
    ``donor_sem`` and ``donor_ci_*`` are the same estimand (ci = mean +/- t*·sem, df = K-1); both are
    emitted because t* moves with K.

    ``draws``: pass the SAME pre-drawn indices across models to make the gene bootstrap PAIRED.
    ``per_donor`` (the K raw points) rides along: always show it next to a LOPO interval, which is
    optimistic because folds share K-2 training donors (Bengio & Grandvalet 2004).
    """
    d = donor_lopo_stats(mat.mean(axis=0, skipna=True).to_dict())        # over genes  -> K points
    g = gene_bootstrap_stats(mat.mean(axis=1, skipna=True).to_numpy(),   # over donors -> N points
                             n_boot=n_boot, seed=seed, draws=draws)
    return {
        "mean": d["mean"], "K": d["K"], "df": d["df"], "t_star": d["t_star_0975"],
        "donor_sd": d["sd"], "donor_sem": d["sem"],
        "donor_ci_lo": d["ci95_mean"][0], "donor_ci_hi": d["ci95_mean"][1],
        "donor_pi_lo": d["pi95_new_donor"][0], "donor_pi_hi": d["pi95_new_donor"][1],
        "gene_N": g["N"], "gene_se_boot": g["se_boot"], "n_boot": g["n_boot"],
        "gene_ci_lo": g["ci95_percentile"][0], "gene_ci_hi": g["ci95_percentile"][1],
        "per_donor": d["per_donor"],
    }


REGISTRY_METRICS = ("pearson", "pearson_logmean", "ece_pit", "pit_bias", "disp_signed",
                    "var_residual_corr", "var_residual_corr_partial")


def headline_registry(summaries, metrics=REGISTRY_METRICS, n_boot=1000, seed=0):
    """Complete headline record over the FULL shared gene panel -> (stats_df, donor_points_df).

    ``summaries``: iterable of dicts with keys ``dataset``, ``gene_set``, ``label``, ``per_long``,
    the shape ``discover_run_summaries`` returns. Models are grouped by (dataset, gene_set) and
    restricted to the gene panel they share, so every model in a block is scored on the same genes.

    No top-n cutoff, and that is the point: a cumulative top-n curve ranks genes with the very metric
    it then reports, and neither interval family corrects for that selection. Here the panel is the
    full shared set, selected by nothing.

    One set of bootstrap draws per (dataset, gene_set), reused across models and metrics, so any
    model-vs-model gene-bootstrap comparison is paired on the same resampled genes, which cancels
    each gene's own difficulty.
    """
    rows, donor_rows = [], []
    for key in sorted({(s["dataset"], s["gene_set"]) for s in summaries}):
        group = [s for s in summaries if (s["dataset"], s["gene_set"]) == key]
        shared = None
        for s in group:
            idx = set(gene_donor_matrix(s["per_long"], "pearson").index)
            shared = idx if shared is None else (shared & idx)
        shared = sorted(shared or [])
        if not shared:
            continue
        draws = np.random.default_rng(seed).integers(0, len(shared), size=(n_boot, len(shared)))
        for s in group:
            for m in metrics:
                if m not in s["per_long"].columns:
                    continue                    # e.g. pearson_logmean: COUNT models only
                mat = gene_donor_matrix(s["per_long"], m).reindex(shared)
                if not np.isfinite(mat.to_numpy()).any():
                    continue                    # all-NaN (e.g. var_residual_corr on a mean-only run)
                st = donor_and_gene_stats(mat, n_boot=n_boot, seed=seed, draws=draws)
                per_donor = st.pop("per_donor")
                rows.append(dict(dataset=key[0], gene_set=key[1], model=s["label"], metric=m, **st))
                for donor, v in per_donor.items():
                    donor_rows.append(dict(dataset=key[0], model=s["label"], metric=m,
                                           donor=donor, value=v))
    return pd.DataFrame(rows), pd.DataFrame(donor_rows)


def format_headline_registry(stats_df, datasets=None, metrics=REGISTRY_METRICS):
    """``headline_registry``'s stats frame -> printable string, one block per (dataset, metric)."""
    def _row(r):
        return (f"{r['mean']:+.4f}  SE {r['donor_sem']:.4f}"
                f"  t-CI [{r['donor_ci_lo']:+.4f},{r['donor_ci_hi']:+.4f}]"
                f" (df {int(r['df'])}, t* {r['t_star']:.2f})"
                f"  PI-new-donor [{r['donor_pi_lo']:+.4f},{r['donor_pi_hi']:+.4f}]"
                f"  gene-boot [{r['gene_ci_lo']:+.4f},{r['gene_ci_hi']:+.4f}]"
                f" (SE {r['gene_se_boot']:.4f})")
    out = ["=" * 118,
           "HEADLINE -- full shared gene panel. THREE DIFFERENT THINGS, do not swap them:",
           "  SE / t-CI     spread of the MEAN over donors (patients resampled, gene panel fixed).",
           "                One estimand, two presentations: t-CI = mean +/- t*·SE.  <- THE headline bar.",
           "  PI-new-donor  where a NEW patient would land. Wider, and a different question entirely.",
           "  gene-boot     genes resampled, donor cohort FIXED -> ZERO inter-donor variance.",
           "                Gene-panel stability only; far narrower (N genes vs K donors).",
           "LOPO folds share K-2 training donors (Bengio & Grandvalet 2004), so these intervals are",
           "optimistic -- always read them next to the per-donor points (donor_points).",
           "=" * 118]
    for ds in (datasets if datasets is not None else sorted(stats_df["dataset"].unique())):
        for m in metrics:
            sub = stats_df[(stats_df["dataset"] == ds) & (stats_df["metric"] == m)]
            if sub.empty:
                continue
            out.append(f"\n--- {ds} | {m}  (K={int(sub['K'].iloc[0])} donors, "
                       f"N={int(sub['gene_N'].iloc[0])} genes) ---")
            for _, r in sub.sort_values("mean", ascending=False).iterrows():
                out.append(f"  {r['model']:<42s} {_row(r)}")
    return "\n".join(out)


def paired_donor_deltas(donor_points, reference, metric="pearson", datasets=None):
    """Per-donor PAIRED differences against a reference model -> tidy frame with a CI on the DIFFERENCE.

    Why paired. Every model is scored on the same donors, folds and genes, and the donor effect
    dominates the spread. Comparing two models through their separate per-model intervals leaves that
    shared donor effect inside both, so the intervals come out wide and overlap even when one model
    is better on nearly every donor. Differencing donor by donor cancels it exactly, and the CI on
    the mean difference is correspondingly narrower.

    Consequence for how results are phrased: two overlapping per-model intervals do not license the
    claim "no difference", which is absence of evidence. This function gives the instrument that
    does, a bound on the magnitude of the difference.

    Optimistic in the same way as every LOPO interval here, since folds share K-2 training donors
    (Bengio & Grandvalet 2004): read ``delta_ci_*`` as a bound on the effect, not as an exact
    p-value. ``n_donors_better`` / ``n_donors_worse`` sit next to it because a consistent sign across
    K donors is often more convincing than the interval itself at these small K.

    Accepts either the full-panel ``donor_points`` from ``headline_registry`` or the per-cutoff frame
    from ``cutoff_donor_points``: when an ``n`` column is present the deltas are computed separately
    at each cutoff.
    """
    has_n = "n" in donor_points.columns
    rows = []
    for ds in (datasets if datasets is not None
               else sorted(donor_points["dataset"].unique())):
        base = donor_points[(donor_points["dataset"] == ds) & (donor_points["metric"] == metric)]
        if base.empty:
            continue
        for n_val in (sorted(base["n"].unique()) if has_n else [None]):
            sub = base[base["n"] == n_val] if has_n else base
            piv = sub.pivot_table(index="donor", columns="model", values="value")
            if reference not in piv.columns:
                continue
            ref_mean = float(piv[reference].mean(skipna=True))
            for model in piv.columns:
                if model == reference:
                    continue
                d = (piv[model] - piv[reference]).dropna()  # paired: donor effect cancels
                if d.size < 2:
                    continue
                st = donor_lopo_stats(d.to_dict())
                lo, hi = st["ci95_mean"]
                row = dict(
                    dataset=ds, metric=metric, reference=reference, model=model,
                    reference_mean=ref_mean, delta_mean=st["mean"],
                    delta_sd=st["sd"], delta_sem=st["sem"], K=st["K"], df=st["df"],
                    delta_ci_lo=lo, delta_ci_hi=hi,
                    # Relative to the reference, the form to quote in prose ("costs X% of the
                    # baseline"). It divides by the reference, so it is unstable when the baseline is
                    # near zero: quote the absolute delta in that case.
                    delta_rel_pct=(100.0 * st["mean"] / ref_mean) if ref_mean else float("nan"),
                    worst_case_rel_pct=(100.0 * min(lo, hi) / ref_mean) if ref_mean else float("nan"),
                    n_donors_better=int((d > 0).sum()), n_donors_worse=int((d < 0).sum()),
                    excludes_zero=bool(lo > 0 or hi < 0),
                )
                if has_n:
                    row["n"] = int(n_val)
                rows.append(row)
    return pd.DataFrame(rows)


def cutoff_donor_points(summaries, ns=(50, 100, 500, 1000, 2000, None),
                        metrics=("pearson", "pearson_logmean"), rank_metric="pearson"):
    """Per-donor points at each cumulative top-n cutoff -> tidy long frame.

    Genes are ranked once per (dataset, gene_set) by the cross-model donor-averaged ``rank_metric``,
    and that single ranking is reused for every model, every metric and every cutoff, so a given n
    means the same gene set on all curves. ``n=None`` means the full shared panel, stored as its
    actual gene count.

    Selection caveat: a top-n set is chosen using the very metric then reported on it, so absolute
    values at small n are optimistic and no interval here corrects for that. Differences between
    models at a fixed n are far less affected, since all models share the same ranking and the same
    genes, which is why the paired deltas are the safer quantity to quote at small n.
    """
    rows = []
    for key in sorted({(s["dataset"], s["gene_set"]) for s in summaries}):
        group = [s for s in summaries if (s["dataset"], s["gene_set"]) == key]
        shared, pieces = None, []
        for s in group:
            mat = gene_donor_matrix(s["per_long"], rank_metric)
            shared = set(mat.index) if shared is None else (shared & set(mat.index))
            pieces.append(mat.mean(axis=1))
        shared = sorted(shared or [])
        if not shared:
            continue
        order = (pd.concat(pieces, axis=1).loc[shared].mean(axis=1)
                 .sort_values(ascending=False).index)
        cutoffs = sorted({(len(order) if n is None else min(int(n), len(order))) for n in ns})
        for s in group:
            for m in metrics:
                if m not in s["per_long"].columns:
                    continue               # pearson_logmean: COUNT models only
                mat = gene_donor_matrix(s["per_long"], m).reindex(order)
                if not np.isfinite(mat.to_numpy()).any():
                    continue
                for n in cutoffs:
                    per_donor = mat.iloc[:n].mean(axis=0, skipna=True)   # genes -> one point per donor
                    for donor, v in per_donor.items():
                        rows.append(dict(dataset=key[0], gene_set=key[1], model=s["label"],
                                         metric=m, n=int(n), donor=str(donor),
                                         value=(None if not np.isfinite(v) else float(v))))
    return pd.DataFrame(rows)


def paired_deltas_full(summaries, reference, metrics=("pearson", "pearson_logmean"),
                       ns=(50, 100, 500, 1000, 2000, None), rank_metric="pearson",
                       n_boot=1000, seed=0):
    """Paired model-minus-reference deltas with ALL THREE uncertainty readings, at every cutoff.

    The construction that makes this coherent: for one (model, reference, metric, cutoff) the
    difference matrix ``D = mat_model - mat_reference`` of shape (genes x donors) is built once, then
    collapsed along either axis. Both readings describe the same quantity, and their point estimates
    agree by construction (``D.mean(0).mean() == D.mean(1).mean()``):
      - collapse over genes  -> K per-donor deltas -> ``donor_lopo_stats``: ``delta_sem`` (1 SE) and
        ``delta_ci_*`` (Student-t 95%, df = K-1). Resamples patients, so this is the generalization
        bar. The two are the same estimand (ci = mean +/- t*·sem); both are emitted so a report may
        quote either.
      - collapse over donors -> N per-gene deltas -> ``gene_bootstrap_stats``: ``gene_se_boot`` and
        ``gene_ci_*`` (percentile 95%). Resamples genes with the donor cohort fixed, so it is
        gene-panel stability only, zero inter-donor variance. Far narrower, never the headline bar.

    Pairing is what makes any of this sharp: the donor effect dominates the raw spread and is
    identical for both models, so it cancels exactly in D. Two overlapping per-model intervals
    therefore say much less than one interval on D.

    Bootstrap draws are shared across models at a given (dataset, cutoff), so model-vs-model gene
    comparisons are paired there too. Genes are ranked once by ``rank_metric`` and ``n=None`` is the
    full shared panel.
    """
    rows = []
    for key in sorted({(s["dataset"], s["gene_set"]) for s in summaries}):
        group = [s for s in summaries if (s["dataset"], s["gene_set"]) == key]
        by_label = {s["label"]: s for s in group}
        ref_label = next((l for l in by_label if l == reference or l.startswith(reference + "|")), None)
        if ref_label is None:
            continue
        shared, pieces = None, []
        for s in group:
            mat = gene_donor_matrix(s["per_long"], rank_metric)
            shared = set(mat.index) if shared is None else (shared & set(mat.index))
            pieces.append(mat.mean(axis=1))
        shared = sorted(shared or [])
        if not shared:
            continue
        order = (pd.concat(pieces, axis=1).loc[shared].mean(axis=1)
                 .sort_values(ascending=False).index)
        cutoffs = sorted({(len(order) if n is None else min(int(n), len(order))) for n in ns})

        for m in metrics:
            if m not in by_label[ref_label]["per_long"].columns:
                continue                      # reference lacks this metric (A is count-only)
            ref_mat = gene_donor_matrix(by_label[ref_label]["per_long"], m).reindex(order)
            for n in cutoffs:
                draws = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
                for label, s in by_label.items():
                    if label == ref_label or m not in s["per_long"].columns:
                        continue
                    mat = gene_donor_matrix(s["per_long"], m).reindex(order)
                    D = (mat.iloc[:n] - ref_mat.iloc[:n])          # (genes x donors), paired
                    d = donor_lopo_stats(D.mean(axis=0, skipna=True).to_dict())   # over genes
                    g = gene_bootstrap_stats(D.mean(axis=1, skipna=True).to_numpy(),  # over donors
                                             n_boot=n_boot, seed=seed, draws=draws)
                    ref_mean = float(ref_mat.iloc[:n].mean(axis=0, skipna=True).mean())
                    lo, hi = d["ci95_mean"]
                    glo, ghi = g["ci95_percentile"]
                    per_donor = D.mean(axis=0, skipna=True)
                    rows.append(dict(
                        dataset=key[0], metric=m, n=int(n), reference=ref_label, model=label,
                        reference_mean=ref_mean, delta_mean=d["mean"],
                        delta_sd=d["sd"], delta_sem=d["sem"], K=d["K"], df=d["df"],
                        delta_ci_lo=lo, delta_ci_hi=hi,
                        delta_pi_lo=d["pi95_new_donor"][0], delta_pi_hi=d["pi95_new_donor"][1],
                        gene_N=g["N"], gene_se_boot=g["se_boot"],
                        gene_ci_lo=glo, gene_ci_hi=ghi, n_boot=g["n_boot"],
                        delta_rel_pct=(100.0 * d["mean"] / ref_mean) if ref_mean else float("nan"),
                        n_donors_better=int((per_donor > 0).sum()),
                        n_donors_worse=int((per_donor < 0).sum()),
                        excludes_zero_t=bool(lo > 0 or hi < 0),
                        excludes_zero_gene=bool(glo > 0 or ghi < 0),
                    ))
    return pd.DataFrame(rows)


def estimator_gap(summaries, metric_b="pearson", metric_a="pearson_logmean",
                  ns=(50, 100, 500, 1000, 2000, None), rank_metric="pearson",
                  n_boot=1000, seed=0):
    """Paired B - A gap (the Jensen gap) WITHIN each model, at every cutoff, all three readings.

    Why within a model and not against the baseline: A = log1p(mu/div) is defined for count models
    only, since a Gaussian predicts on the log1p axis directly and so has a single estimator. A delta
    in A against the mean-only baseline is therefore undefined, the baseline having no A. The
    meaningful A-vs-B comparison is within one count model, and it is exactly the Jensen gap of the
    concave log(1+.): B routes the prediction through the full fitted predictive (mu and alpha), A
    uses the mean alone, and the gap grows with the predictive spread rather than being a constant
    offset (see count_metrics.py).

    Same construction as ``paired_deltas_full``: build D = mat_B - mat_A (genes x donors) once, then
    collapse over genes (K donor points, Student-t) or over donors (N gene points, percentile
    bootstrap). A positive gap means routing through the full predictive helps.
    """
    rows = []
    for key in sorted({(s["dataset"], s["gene_set"]) for s in summaries}):
        group = [s for s in summaries if (s["dataset"], s["gene_set"]) == key]
        shared, pieces = None, []
        for s in group:
            mat = gene_donor_matrix(s["per_long"], rank_metric)
            shared = set(mat.index) if shared is None else (shared & set(mat.index))
            pieces.append(mat.mean(axis=1))
        shared = sorted(shared or [])
        if not shared:
            continue
        order = (pd.concat(pieces, axis=1).loc[shared].mean(axis=1)
                 .sort_values(ascending=False).index)
        cutoffs = sorted({(len(order) if n is None else min(int(n), len(order))) for n in ns})
        for s in group:
            cols = s["per_long"].columns
            if metric_a not in cols or metric_b not in cols:
                continue                       # Gaussian / mean-only: a single estimator, no gap
            mb = gene_donor_matrix(s["per_long"], metric_b).reindex(order)
            ma = gene_donor_matrix(s["per_long"], metric_a).reindex(order)
            for n in cutoffs:
                draws = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
                D = (mb.iloc[:n] - ma.iloc[:n])
                d = donor_lopo_stats(D.mean(axis=0, skipna=True).to_dict())
                g = gene_bootstrap_stats(D.mean(axis=1, skipna=True).to_numpy(),
                                         n_boot=n_boot, seed=seed, draws=draws)
                per_donor = D.mean(axis=0, skipna=True)
                lo, hi = d["ci95_mean"]
                glo, ghi = g["ci95_percentile"]
                rows.append(dict(
                    dataset=key[0], model=s["label"], n=int(n),
                    B_mean=float(mb.iloc[:n].mean(axis=0, skipna=True).mean()),
                    A_mean=float(ma.iloc[:n].mean(axis=0, skipna=True).mean()),
                    gap_mean=d["mean"], gap_sd=d["sd"], gap_sem=d["sem"], K=d["K"], df=d["df"],
                    gap_ci_lo=lo, gap_ci_hi=hi,
                    gene_se_boot=g["se_boot"], gene_ci_lo=glo, gene_ci_hi=ghi,
                    n_donors_positive=int((per_donor > 0).sum()),
                    n_donors_negative=int((per_donor < 0).sum()),
                    excludes_zero_t=bool(lo > 0 or hi < 0),
                    excludes_zero_gene=bool(glo > 0 or ghi < 0),
                ))
    return pd.DataFrame(rows)


def discover_run_summaries(results_root, run_id, datasets, radius="r75",
                           models=None, lambda_keep=0.0, gene_set_suffix="_hvg_top3000"):
    """Directory-scan a classic run tree -> the ``summaries`` list the functions above consume.

    Keeps the notebook thin: one call replaces the discovery + parquet-loading boilerplate.
    ``models=None`` keeps every model_type found. ``lambda_keep`` filters the beta-NLL sweep by
    VALUE (0.0 = the proper-NLL default); models with no lambda axis are always kept, and
    ``lambda_keep=None`` keeps the whole sweep.
    """
    from pathlib import Path as _Path
    results_root = _Path(results_root)
    out = []
    for ds in datasets:
        gene_set = f"_{ds}{gene_set_suffix}"
        base = results_root / ds / run_id / gene_set / radius
        if not base.is_dir():
            continue
        for model_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            for leaf in [model_dir] + sorted(model_dir.glob("lambda_*")):
                pq = leaf / "benchmark_summary.parquet"
                if not pq.is_file():
                    continue
                lam_dir = leaf.name if leaf.name.startswith("lambda_") else None
                if models is not None and model_dir.name not in models:
                    continue
                if lambda_keep is not None and lam_dir is not None:
                    if not np.isclose(float(lam_dir[len("lambda_"):]), lambda_keep):
                        continue
                label = model_dir.name + (f"|lam{lam_dir[len('lambda_'):]}" if lam_dir else "")
                out.append(dict(dataset=ds, gene_set=gene_set, model_type=model_dir.name,
                                lambda_dir=lam_dir, label=label,
                                per_long=pd.read_parquet(pq)))
    return out


def format_paired_deltas(delta_df):
    """``paired_donor_deltas`` output -> printable string, one block per dataset."""
    if delta_df.empty:
        return "(no paired deltas: reference model absent from donor_points)"
    ref = delta_df["reference"].iloc[0]
    metric = delta_df["metric"].iloc[0]
    out = ["=" * 112,
           f"PAIRED per-donor deltas vs '{ref}'  [metric: {metric}]",
           "  Differences are taken donor by donor, so the donor effect (which dominates the raw",
           "  spread) cancels and the CI is ~10x narrower than the per-model intervals.",
           "  '*' marks a CI excluding 0. sign = how many donors favor the model vs the reference.",
           "  Read these as a BOUND on the magnitude, not as exact p-values (LOPO folds overlap).",
           "=" * 112]
    for ds, sub in delta_df.groupby("dataset", sort=False):
        ref_mean = sub["reference_mean"].iloc[0]
        out.append(f"\n--- {ds}  (reference mean = {ref_mean:+.4f}, K = {int(sub['K'].iloc[0])} donors) ---")
        for _, r in sub.sort_values("delta_mean", ascending=False).iterrows():
            star = "*" if r["excludes_zero"] else " "
            out.append(f" {star} {r['model']:<40s} delta {r['delta_mean']:+.4f} "
                       f"[{r['delta_ci_lo']:+.4f},{r['delta_ci_hi']:+.4f}] "
                       f"({r['delta_rel_pct']:+.1f}% of baseline, worst case {r['worst_case_rel_pct']:+.1f}%)  "
                       f"donors {int(r['n_donors_better'])}+/{int(r['n_donors_worse'])}-")
    return "\n".join(out)


# --------------------------------------------------------------------------- export
def export_summary(run_dir, overwrite=False):
    """Write benchmark_summary.parquet (+ .json) into run_dir and return (per_donor_gene_df, meta).

    per_donor_gene_df: a LONG table with columns [test_donor, gene, pearson, ece_pit, pit_bias,
    disp_signed, var_residual_corr(+_partial), (+ e_aurc for hetero runs)] — one row per (held-out
    donor, gene). Reads atoms (cheap) for the atom metrics and the preds ONCE for e_aurc + the
    sigma-vs-mean diagnostic. Skips work if both files already exist unless overwrite=True.

    meta.headline[m] = LOPO mean over donors of the per-donor gene-mean (= train.py's lopo_stats
    mean). meta.lopo[m] = the full donor_lopo_stats block (mean/SD/SEM/CI/PI + per-donor points).
    """
    run_dir = Path(run_dir)
    pq, js = run_dir / "benchmark_summary.parquet", run_dir / "benchmark_summary.json"
    if pq.exists() and js.exists() and not overwrite:
        return pd.read_parquet(pq), json.loads(js.read_text())

    run = load_run(run_dir)
    if not run["is_lopo"]:
        raise ValueError(
            f"{run_dir}: metrics.json says lopo=False. This module reads leave-one-patient-out "
            "runs only, because its statistics are per-donor."
        )
    present = [m for m in ATOM_METRICS if m in run["atoms"].columns]   # mean-only runs lack calibration
    # per-(test_donor, gene) table from the atoms — slide-aware within each held-out donor.
    per_dg = pd.DataFrame({m: reduce_atoms_per_donor(run["atoms"], m) for m in present})

    # Load the test predictions ONCE if probabilistic: e_aurc (hetero only) + sigma-vs-mean
    # diagnostic (any probabilistic run), computed PER DONOR (each fold = one held-out donor).
    is_prob = bool(run["metrics"].get("is_probabilistic", run["is_hetero"]))
    sig_per_donor = {}
    if is_prob and run["pred_files"]:
        eaurc_pieces = []                                   # (test_donor, gene) -> e_aurc
        for donor, mu, y, var, genes, slides, gene_mean in iter_fold_predictions(run):
            sig_per_donor[donor] = sigma_mean_corr(gene_mean, var)
            if run["is_hetero"]:
                ea = eaurc_per_gene(mu, y, var, slides, genes)
                ea.index = pd.MultiIndex.from_product([[donor], ea.index], names=["test_donor", "gene"])
                eaurc_pieces.append(ea)
        if eaurc_pieces:
            per_dg["e_aurc"] = pd.concat(eaurc_pieces).reindex(per_dg.index)
    per_dg.index.names = ["test_donor", "gene"]

    # LOPO aggregation: per-donor point = nanmean over genes within the donor; then over donors.
    metric_cols = list(per_dg.columns)
    lopo = {}
    headline = {}
    for m in metric_cols:
        per_donor_point = per_dg[m].groupby("test_donor").mean()          # x_i (gene-averaged)
        lopo[m] = donor_lopo_stats(per_donor_point.to_dict())
        headline[m] = lopo[m]["mean"]

    donors = sorted(per_dg.index.get_level_values("test_donor").unique().tolist())
    meta = {
        **parse_run_identity(run_dir),
        "lopo": lopo,                                        # per-metric mean/SD/SEM/CI/PI + per-donor
        "headline": headline,                                # mean over donors (the LOPO headline)
        "n_genes": int(per_dg.index.get_level_values("gene").nunique()),
        "n_donors": len(donors),
        "donors": donors,
        "is_hetero": run["is_hetero"],
        "is_lopo": run["is_lopo"],
        "primary_corr_metric": run["primary_corr_metric"],
        # sigma-vs-gene-mean decoupling diagnostic, per donor + mean over donors.
        "sigma_mean_corr_per_donor": {str(d): v for d, v in sig_per_donor.items()},
        "sigma_mean_corr": (float(np.nanmean(list(sig_per_donor.values())))
                            if sig_per_donor else float("nan")),
        "target_transform": run["metrics"].get("target_transform", "none"),
        # Count identity + the reporting axis (see report_axis). Carried so a cross-model table
        # never has to re-derive them, and so a depth-blind *_raw run (report_axis
        # "log1p_raw_counts") cannot be ranked against a _cp / Gaussian one ("log1p_cp10k").
        "target_source": run["target_source"],
        "count_convention": run["metrics"].get("count_convention"),
        "depth_scale": run["metrics"].get("depth_scale"),
        "report_axis": report_axis(run["target_source"], run["metrics"].get("depth_scale")),
        "run_dir": str(run_dir),
    }
    per_dg.reset_index().to_parquet(pq)                      # LONG table: test_donor, gene, metrics...
    js.write_text(json.dumps(meta, indent=2))
    return pd.read_parquet(pq), meta


# --------------------------------------------------------------------------- export (deep ensemble)
def export_summary_ensemble(ensemble_dir, overwrite=False):
    """Deep-ensemble sibling of export_summary: write benchmark_summary.parquet (+ .json) into
    ensemble_dir and return (per_donor_gene_df, meta), for a deep ensemble's consolidated
    test_summaries.h5ad (snake/scripts/epistemic/deep_ensemble_combine.py's own output).

    An ensemble has no atoms.parquet, because it is not a training run, so the per-(slide, gene)
    atoms are built here from test_summaries.h5ad's layers using the same primitives export_summary
    reads pre-aggregated: per_slide_gene_corr / per_slide_gene_calibration and their _aggregate_*
    reducers in src/evaluation/metrics_slide_aware.py.

    Scored quantity: layers["gstar"], the ensemble-mixture average of each member's own
    g* = E[log1p(Y/div)], against the reconstructed true log1p(y/div). That is the same comparison a
    single count run's count_metrics_final makes at final inference (see count_metrics.py), so the
    "pearson" column here is directly comparable to a single count model's headline pearson. Next to
    it, "pearson_logmean" is the correlation of the mean-only estimator log1p(mu/div), with mu = B.X
    the ensemble mean of member means, against the same truth. It is the ensemble analog of the
    single model's {metric}_logmean atom column: g* uses mu and alpha, option A uses mu alone, and
    their difference is the Jensen gap.
    `div` must be the same divisor deep_ensemble_combine.py used to build g*, so it is read off the
    file's own uns["deep_ensemble"]["depth_scale"], the one constant involved, absent meaning 1.0 for
    the depth-blind *_raw case, and gated on uns["count_convention"] = "v2_no_offset". An unmarked
    file raises: reading it under another convention would silently compare g* against the wrong
    ground truth.

    Calibration comes from the precomputed native-CDF PIT bounds (layers["pit_flo"/"pit_fhi"]):
    ece_pit, pit_bias and disp_signed only. var_residual_corr[_partial] are deliberately omitted. The
    ensemble's epistemic and aleatoric-ingredient layers live on the cp10k mean scale while
    pred_log1p and truth_log1p, which drive g* and the PIT here, live on the log1p scale, so a
    var_residual_corr against them would silently compare mismatched scales. A single count model
    does not have that problem: its own var_residual_corr rides the exact (means, pred_var) pair used
    for its PIT bounds, see count_metrics_final. per_slide_gene_calibration is still called, since
    its PIT scalars use only F_lo/F_hi, with an all-zero pred_var placeholder that makes its
    var_residual_corr atom a std<eps NaN by construction; that column is then dropped, so no
    scale-mismatched number can escape.

    Two arms, by family. Count (target_source raw_counts / cp_counts / cp_counts_nr) scores
    layers["gstar"] against log1p(y/depth_scale), plus option A, log1p(mu/depth_scale), as the
    mean-only twin. Continuous (target_source "X", the mean-only baseline and the Gaussian heads)
    already has X and layers["y"] on the log1p-CP10k reporting axis, so it scores them directly with
    no divisor and no log1p, and emits no pearson_logmean at all, there being no count likelihood to
    integrate. That column is omitted rather than NaN-filled: every downstream consumer gates on its
    presence, so a NaN column would pass those gates and manufacture a Jensen gap for a family that
    has none.

    Calibration is emitted only when layers["pit_flo"] and ["pit_fhi"] exist. g* is legitimately
    count-only, being the predictive mean of a transform where a Gaussian already predicts on the
    reporting axis, but the PIT is not: it is uniform on [0,1] whatever the target scale, which is
    what makes a continuous and a discrete head comparable on one calibration axis. A count file
    missing those layers is rejected by the guard below, so a file that reaches here without them is
    a continuous one and yields correlation only.

    meta mirrors export_summary's field names (dataset/run_id/gene_set/radius/model_type/lambda via
    parse_run_identity, lopo, headline, n_genes, n_donors, donors, run_dir) plus is_ensemble=True and
    n_members, so a downstream reader distinguishes the two without two schemas.
    """
    ensemble_dir = Path(ensemble_dir)
    model_dir = ensemble_dir.parent            # holds seed_{s}/ + ensemble/ (deep_ensemble_combine.py's out_dir)
    pq, js = ensemble_dir / "benchmark_summary.parquet", ensemble_dir / "benchmark_summary.json"
    if pq.exists() and js.exists() and not overwrite:
        return pd.read_parquet(pq), json.loads(js.read_text())

    B = sc.read_h5ad(ensemble_dir / "test_summaries.h5ad")
    if "deep_ensemble" not in B.uns:
        raise ValueError(
            f"{ensemble_dir}/test_summaries.h5ad: missing uns['deep_ensemble'] -- not a "
            "deep_ensemble_combine.py output?"
        )
    ue = B.uns["deep_ensemble"]
    target_source = str(ue["target_source"])
    count_convention = B.uns.get("count_convention")
    if count_convention != "v2_no_offset":
        raise ValueError(
            f"{ensemble_dir}: no count_convention marker, so the scale of the saved mean and "
            f"dispersion cannot be determined. Re-run the precompute.")

    # The layer requirements are per-family, so target_source must be resolved before the guards
    # below run; guarding first would reject every non-count ensemble.
    # g* is legitimately count-only: it is the predictive mean of log1p(Y/depth_scale), and a
    # Gaussian head already predicts on the log1p-CP10k axis, so there is no transform whose
    # expectation to take. The PIT is not count-only. It is uniform on [0,1] whatever the target
    # scale, which is what makes a continuous and a discrete head comparable on one calibration
    # axis, so for a count family the guard below requires it, while a continuous file without pit
    # layers simply gets correlation only.
    is_count = target_source in COUNT_TARGET_SOURCES
    if is_count:
        if "gstar" not in B.layers:
            raise ValueError(
                f"{ensemble_dir}/test_summaries.h5ad: no layers['gstar'] -- g* was never precomputed "
                "for this model. Run snake/scripts/epistemic/DE_gstar_precompute.py, then re-run "
                "deep_ensemble_combine.py, before calling export_summary_ensemble."
            )
        if "pit_flo" not in B.layers or "pit_fhi" not in B.layers:
            raise ValueError(
                f"{ensemble_dir}/test_summaries.h5ad: has 'gstar' but is missing 'pit_flo'/'pit_fhi' -- "
                "deep_ensemble_combine.py gates these together with gstar for COUNT families, so "
                "this file is unexpected/stale. Re-run deep_ensemble_combine.py."
            )
    has_pit = ("pit_flo" in B.layers and "pit_fhi" in B.layers)
    # The prediction/truth pair, per family. Running the count arm on a continuous model would not
    # crash but would be silently wrong: the combine writes uns["deep_ensemble"]["depth_scale"] only
    # for count families, so a Gaussian would get div = 1.0 and then have log1p applied to an
    # already-log1p target on both axes. log1p is monotone and near-linear over the range these
    # values live in, so the correlation would stay high and plausible while being wrong.
    if is_count:
        # layers["gstar"] is E[log1p(Y/depth_scale)] on the model's own fitted target scale, so the
        # ground truth it must be compared against is log1p(y/depth_scale) with the same constant.
        # depth_scale is absent for the depth-blind *_raw models, since h5ad uns cannot hold a bare
        # None, and that absence is exactly the divisor 1.0 case.
        depth_scale = ue.get("depth_scale")
        div = 1.0 if depth_scale is None else float(depth_scale)
        truth_log1p = np.log1p(np.asarray(B.layers["y"], dtype=float) / div)
        pred_log1p = np.asarray(B.layers["gstar"], dtype=float)
        # Option A (mean-only) next to g* (option B). B.X is the ensemble mean mu = mean_s(member
        # mean); log1p(mu/div) is the same transform-of-mean the single-model path stores as
        # {metric}_logmean in atoms.parquet. Both are correlated against the SAME truth_log1p, so
        # g* minus option A isolates the Jensen contribution of routing through the full predictive
        # (mu AND alpha) versus the mean alone.
        optA_log1p = np.log1p(np.asarray(B.X, dtype=float) / div)
    else:
        # target_source "X": the head predicts log1p(CP10k) directly, so BOTH X and layers["y"]
        # already live on the reporting axis. No divisor, no log1p, and no option A -- there is no
        # count likelihood to integrate, hence a single estimator. optA stays None so the caller
        # below OMITS the pearson_logmean column rather than filling it with NaN: every downstream
        # consumer gates on the column being PRESENT, so a NaN column would pass those gates and
        # manufacture a Jensen gap for a family that has only one estimator.
        depth_scale = None
        truth_log1p = np.asarray(B.layers["y"], dtype=float)
        pred_log1p = np.asarray(B.X, dtype=float)
        optA_log1p = None
    # Read only when present. A placeholder (zeros) would NOT be a harmless sentinel: with
    # F_lo == F_hi == 0 the coverage curve takes its continuous branch at every level, giving
    # coverage 1.0 throughout, hence ece_pit ~ 0.5 and pit_bias ~ +0.5 -- which reads as a
    # quotable "this model is badly calibrated" rather than as missing data.
    flo_all = np.asarray(B.layers["pit_flo"], dtype=float) if has_pit else None
    fhi_all = np.asarray(B.layers["pit_fhi"], dtype=float) if has_pit else None

    slides = B.obs["slide"].to_numpy()
    test_donors = np.asarray([str(d) for d in B.obs["test_donor"].to_numpy()])
    genes = B.var_names.to_numpy()
    n_genes = len(genes)

    per_dg_parts = []
    # The metric set is built from what this family ACTUALLY has, not fixed at five keys: a column
    # is either present and meaningful or absent, never present and NaN (see the optA comment above).
    metric_keys = (["pearson"]
                   + (["pearson_logmean"] if optA_log1p is not None else [])
                   + (["ece_pit", "pit_bias", "disp_signed"] if has_pit else []))
    lopo_input = {m: {} for m in metric_keys}
    for donor in sorted(set(test_donors.tolist())):
        rows = test_donors == donor
        pred_d, truth_d, slide_d = pred_log1p[rows], truth_log1p[rows], slides[rows]
        cols = {"test_donor": donor, "gene": genes}

        corr_atoms, _, _ = per_slide_gene_corr(pred_d, truth_d, slide_d, _safe_pearson, key="pearson")
        lopo_input["pearson"][donor], cols["pearson"] = _aggregate_corr(corr_atoms, n_genes, key="pearson")

        if optA_log1p is not None:
            # Option A (mean-only), same slide-aware primitives, same truth_d. Count families only.
            lm_atoms, _, _ = per_slide_gene_corr(optA_log1p[rows], truth_d, slide_d,
                                                 _safe_pearson, key="pearson_logmean")
            lopo_input["pearson_logmean"][donor], cols["pearson_logmean"] = _aggregate_corr(
                lm_atoms, n_genes, key="pearson_logmean")

        if has_pit:
            # var_residual_corr[_partial] not needed here (see docstring) -- the all-zero pred_var
            # placeholder makes those two atom columns a provable std<eps NaN, never a real number.
            calib_atoms, _, _ = per_slide_gene_calibration(
                pred_d, truth_d, np.zeros_like(pred_d), flo_all[rows], fhi_all[rows], slide_d)
            calib = _aggregate_calibration(calib_atoms, n_genes)
            for m in ("ece_pit", "pit_bias", "disp_signed"):
                cols[m], lopo_input[m][donor] = calib[f"{m}_per_gene"], calib[m]

        per_dg_parts.append(pd.DataFrame(cols))

    per_dg = pd.concat(per_dg_parts, ignore_index=True)
    lopo = {m: donor_lopo_stats(vals) for m, vals in lopo_input.items()}
    headline = {m: lopo[m]["mean"] for m in lopo}
    donors = sorted(per_dg["test_donor"].unique().tolist())

    meta = {
        **parse_run_identity(model_dir),
        "lopo": lopo,                                        # per-metric mean/SD/SEM/CI/PI + per-donor
        "headline": headline,                                # mean over donors (the LOPO headline)
        "n_genes": int(per_dg["gene"].nunique()),
        "n_donors": len(donors),
        "donors": donors,
        "is_ensemble": True,
        "n_members": int(ue["n_members"]),
        "target_source": target_source,
        "mean_scale": str(ue.get("mean_scale")),
        # which convention this file was read under, and the ONE constant used before log1p.
        "count_convention": str(count_convention),
        "depth_scale": (float(depth_scale) if depth_scale is not None else None),
        # Reporting axis, same tag vocabulary as export_summary's report_axis (group/split a
        # cross-model table by it; never rank across two values of it). The only non-CP10k case
        # is a depth-BLIND *_raw model: depth_scale None -> div 1.0 -> log1p(raw counts),
        # comparable only to other *_raw runs.
        "report_axis": report_axis(target_source, depth_scale),
        "run_dir": str(ensemble_dir),
    }
    per_dg.to_parquet(pq, index=False)                        # LONG table: test_donor, gene, metrics...
    js.write_text(json.dumps(meta, indent=2))
    return pd.read_parquet(pq), meta


# --------------------------------------------------------------------------- aggregation helpers (6.3)




def gene_donor_matrix(per_long, metric):
    """Pivot to a (gene x donor) matrix for one metric. Feeds the inter-donor band of the cumulative
    top-n curve: per n, a per-donor mean over the top-n genes gives K points, then a t-CI."""
    return per_long.pivot_table(index="gene", columns="test_donor", values=metric)


# --------------------------------------------------------------------------- gene-bootstrap helpers
# A gene-level bootstrap: resamples genes with the donor cohort fixed, so it carries zero inter-donor
# variance. Secondary to the inter-donor helpers above, which give the headline error bar. Some are
# still imported by existing analysis notebooks.
def gene_boot_draws(n_genes, n_boot=1000, seed=0):
    """Pre-draw bootstrap resamples of gene POSITIONS; reuse the SAME draws across models -> paired."""
    rng = np.random.default_rng(seed)
    return [rng.integers(0, n_genes, size=n_genes) for _ in range(n_boot)]


def boot_ci(values, draws):
    """mean + 95% CI over pre-drawn gene resamples (NaN-skipping)."""
    v = np.asarray(values, dtype=float)
    bm = [np.nanmean(v[idx]) for idx in draws]
    return {"mean": float(np.nanmean(v)), "lo": float(np.percentile(bm, 2.5)), "hi": float(np.percentile(bm, 97.5))}


