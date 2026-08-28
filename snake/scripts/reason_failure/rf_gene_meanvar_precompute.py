# snake/scripts/reason_failure/rf_gene_meanvar_precompute.py
#
# D1 of the "reason for failure" analysis: the mean-variance coupling in the data. It writes
# parquets; the plotting happens downstream.
#
# The question. Does the observed spread of expression rise with the expression level? No model is
# involved: this is a property of the measurements alone, which is what makes it the right
# starting point. If the coupling were absent from the data, no amount of fitting could put it
# into sigma.
#
# What is computed. For each (slide, gene) over the HVG panel, the mean and the variance across
# the spots of that slide, on the two scales the two model families are trained on:
#
#   scale "log1p_cp10k"   y = adata.X                                  the Gaussian heads' target
#   scale "count_T"       c = round(expm1(adata.X) * depth_scale)      the *_cp heads' target
#                         depth_scale = datasets.{ds}.count_common_depth / 1e4
#                         (10x_TuPro 2.5, TLS 1.0, LUNG_CANCER 1.0)
#
# Two scales and not one, because sigma is never moved between them: the expm1 Jacobian is
# (1 + mu), a function of the mean, so applying it would manufacture the very coupling under
# measurement. Each family's predicted spread is compared to the data spread on its own axis, and
# the two panels are never merged.
#
# Per slide, not per donor and not pooled. Pooling slides would add the between-slide mean shift
# to every gene's variance, the same pooling bug the slide-aware metric exists to avoid.
#
# The limit to declare, in the text and not in a footnote. v_g here is the marginal within-slide
# variance, which decomposes as E_s[Var(Y | x_s)] + Var_s(E[Y | x_s]). The second term is the
# spatially predictable variation, bounded by the attainable R^2, so v_g stands in for the mean
# conditional variance only to the extent that R^2 is small. Saying so closes the objection rather
# than inviting it.
#
# Two axes, the same duality the rest of the analysis uses.
#   GENE AXIS (the estimand above): one point per gene, mean and variance across the spots of the
#            slide, Spearman over the panel. This is the coupling in its textbook form.
#   SPOT AXIS: one point per spot, read across the genes of that spot. A variance needs repetitions
#            of the same random variable and a spot gives one draw per gene, so the level and the
#            spread must come from different places: the level is y_sg, the spot's own value, and
#            the spread is v_g, that gene's variance across the spots of the same slide. Their
#            Spearman across the spot's detected genes is the data-side twin of the model's
#            sigma-vs-mu coupling on the spot axis. The naive dual (mean and variance taken across
#            the genes of a spot) is exported next to it, but it measures the dispersion of that
#            spot's gene composition, not a conditional variance. Read log1p as primary here: on
#            the count scale a spot's detected values are heavily tied at small integers, which
#            attenuates a rank correlation.
#
# Outputs (under <project-root>/results/_analysis/):
#   rf_gene_meanvar.parquet       dataset, slide, donor, gene, n_spots, then per scale:
#                                 mean_{scale}, var_{scale}; plus pct_zero on both scales
#                                 (pct_zero_log1p_cp10k and pct_zero_count_T, whose difference is
#                                 the round-to-zero corruption of the count target).
#   rf_spot_meanvar.parquet       dataset, slide, donor, adata_row, then per scale: n_det_{scale},
#                                 rho_level_spread_{scale} (the spot-axis statistic),
#                                 mean_det_{scale}, var_det_{scale} (the naive dual).
#   rf_meanvar_spearman.parquet   dataset, slide, donor, scale, axis, spearman, slope_loglog,
#                                 n_genes, n_genes_loglog, n_spots, n_units: the per-slide summary
#                                 that is aggregated to donor then cohort downstream. Filter on
#                                 `axis`, the gene rows and the spot rows are different estimands.
#                                 On the spot rows `spearman` is the median over the slide's spots
#                                 of the per-spot value, and slope_loglog is NaN (a slope needs the
#                                 gene-axis pair of a mean and a variance of the same variable).
#   rf_gene_meanvar_summary.json  identity, resolved depth_scale, panel size, row counts.
#
# Cost. One backed read per dataset, one contiguous slide slice at a time, so peak RSS is one
# slide's rows x the full gene axis (a few hundred MB), never the whole merged matrix. CPU only.
#
# Usage (repo root, on the cluster; CPU only, omit -p/--gres so it lands on `compute`):
#   conda activate mtenv0
#   python -u snake/scripts/reason_failure/rf_gene_meanvar_precompute.py
#   # scope flags: --datasets, --preproc-id, --radius, --panel (--help lists the rest)
#   # whole thing as a SLURM job: bash snake/scripts/reason_failure/launch_rf_gene_meanvar_precompute.sh

import argparse
import json
import os
import sys
import time

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml
from scipy import stats

_T0 = time.time()
def _el():
    return time.time() - _T0

# Code/config sits next to this script; --project-root
# points at the DATA tree, which may be a different place.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
DEFAULT_PROJECT_ROOT = REPO_ROOT

p = argparse.ArgumentParser()
p.add_argument("--project-root", default=DEFAULT_PROJECT_ROOT)
p.add_argument("--datasets", nargs="+", default=["10x_TuPro", "TLS", "LUNG_CANCER"])
p.add_argument("--preproc-id", default="v5")
p.add_argument("--radius", default="r75")
p.add_argument("--panel", default="hvg_top3000.txt",
               help="Gene-list file under _preprocessed/{preproc_id}/{radius}/hvg/. This is the "
                    "panel the models were trained on; D1 must be read on the same genes.")
p.add_argument("--config", default=os.path.join(REPO_ROOT, "snake/config/config.yaml"),
               help="Source of datasets.{ds}.count_common_depth, hence of depth_scale.")
p.add_argument("--donor-map", default=os.path.join(REPO_ROOT, "snake/config/donor_slide_map.yaml"),
               help="{donor: [slides]} map, the SAME file the CV splits use.")
p.add_argument("--min-detected-genes", type=int, default=60,
               help="SPOT AXIS only: a spot needs this many DETECTED genes for its Spearman to be "
                    "computed, else NaN. Same floor as the 60-entry group floor used everywhere "
                    "else in this analysis.")
p.add_argument("--out-dir", default=None,
               help="Default <project-root>/results/_analysis/.")
args = p.parse_args()

sys.path.append(REPO_ROOT)
from src.data.splitting import load_slide_to_donor, donor_labels   # noqa: E402  (needs sys.path)

project_root = args.project_root.rstrip("/") + "/"
out_dir = args.out_dir or f"{project_root}results/_analysis"
os.makedirs(out_dir, exist_ok=True)

with open(args.config) as fh:
    CFG = yaml.safe_load(fh)

SCALES = ["log1p_cp10k", "count_T"]


def _dense(a):
    return a.toarray() if sp.issparse(a) else np.asarray(a)


def depth_scale_of(ds):
    """The reporting constant depth_scale, resolved as train.py resolves it for the cp families."""
    d = CFG["datasets"][ds].get("count_common_depth")
    if d is None:
        raise ValueError(f"{ds}: no count_common_depth in {args.config} -- the count_T scale of D1 "
                         "is undefined without it.")
    return float(d) / 1e4


def one_dataset(ds):
    """Per-(slide, gene) mean/variance on both scales, plus the per-slide summary."""
    path = f"{project_root}results/{ds}/_preprocessed/{args.preproc_id}/{args.radius}/adata.h5ad"
    panel_path = (f"{project_root}results/{ds}/_preprocessed/{args.preproc_id}/{args.radius}"
                  f"/hvg/{args.panel}")
    ds_scale = depth_scale_of(ds)
    print(f"[{_el():6.1f}s] {ds}: {path}", flush=True)
    print(f"[{_el():6.1f}s] {ds}: panel {panel_path}   depth_scale={ds_scale}", flush=True)

    panel = [g.strip() for g in open(panel_path) if g.strip()]

    # Backed: obs, var and the slide layout come off disk without materialising X. X is then read
    # ONE SLIDE AT A TIME, which is what keeps this a few hundred MB instead of tens of GB.
    A = ad.read_h5ad(path, backed="r")
    genes_all = np.asarray(A.var_names).astype(str)
    # The merged adata's slide column is obs["sample_id"] (obs["slide"] exists only in the
    # prediction files) -- the same trap dataset_gene_expression_stats_precompute.py documents.
    sample = A.obs["sample_id"].astype(str).to_numpy()

    pos = {g: i for i, g in enumerate(genes_all)}
    missing = [g for g in panel if g not in pos]
    if missing:
        raise ValueError(f"{ds}: {len(missing)} panel gene(s) absent from the adata, e.g. "
                         f"{missing[:5]} -- the panel and the adata do not match.")
    gidx = np.array([pos[g] for g in panel], dtype=np.int64)
    genes = np.asarray(panel, dtype=object)

    slide_to_donor = load_slide_to_donor(args.donor_map, ds)
    rows_out, summ_out, spot_out = [], [], []
    for slide in pd.unique(sample):
        idx = np.flatnonzero(sample == slide)
        # ad.concat lays the slides out contiguously, so a slice is a plain h5 read. Asserting it
        # rather than assuming it: a fancy index on a backed dense X is orders of magnitude slower,
        # and silently taking that path would look like a hang, not like a bug.
        assert idx[-1] - idx[0] + 1 == idx.size, f"{ds}/{slide}: slide rows are not contiguous"
        blk = _dense(A.X[idx[0]:idx[-1] + 1])[:, gidx].astype(np.float64)
        donor = donor_labels(np.array([slide]), slide_to_donor)[0]

        y = blk                                            # log1p(CP10k), the Gaussian target
        c = np.rint(np.expm1(blk) * ds_scale)              # round(CP_T), the *_cp target
        rec = {"dataset": ds, "slide": slide, "donor": donor, "gene": genes,
               "n_spots": blk.shape[0]}
        for name, S in (("log1p_cp10k", y), ("count_T", c)):
            rec[f"mean_{name}"] = S.mean(axis=0)
            rec[f"var_{name}"] = S.var(axis=0, ddof=1)
            rec[f"pct_zero_{name}"] = (S == 0).mean(axis=0)
        rows_out.append(pd.DataFrame(rec))

        # ---- GENE AXIS: one point per gene, the coupling read across the genes of the slide ----
        for name in SCALES:
            m, v = rec[f"mean_{name}"], rec[f"var_{name}"]
            ok = np.isfinite(m) & np.isfinite(v)
            rho = float(stats.spearmanr(m[ok], v[ok])[0]) if ok.sum() > 2 else np.nan
            # The log-log slope is the SHAPE of the coupling, next to the Spearman which is only
            # its monotonicity: a Poisson gives 1, a quadratic mean-variance relation gives 2 in
            # the high-mean regime. Genes with a zero mean or a zero variance are dropped from the
            # slope only (they carry no log), and the count is reported so the drop is visible.
            lg = ok & (m > 0) & (v > 0)
            slope = (float(np.polyfit(np.log(m[lg]), np.log(v[lg]), 1)[0])
                     if lg.sum() > 2 else np.nan)
            summ_out.append(dict(dataset=ds, slide=slide, donor=donor, scale=name, axis="gene",
                                 spearman=rho, slope_loglog=slope,
                                 n_genes=int(ok.sum()), n_genes_loglog=int(lg.sum()),
                                 n_spots=int(blk.shape[0]), n_units=int(ok.sum())))

        # ---- SPOT AXIS: the dual, one point per SPOT, read across the genes of that spot -------
        # Why a borrowed variance, and why this is the honest dual. A variance needs repetitions of
        # the same random variable; a spot supplies one draw of each gene, so "the variance across
        # the genes of a spot" is the dispersion of that spot's gene composition and estimates no
        # conditional variance at all. So the level and the spread are taken from different places:
        #   level  = y_sg, this spot's own observed value for gene g;
        #   spread = v_g, that gene's variance across the spots of the same slide (computed above).
        # rho_level_spread is then the Spearman between the two across the genes of the spot, the
        # data-side twin of comparing a model's sigma to its mu across the genes of one spot: sigma
        # varies there because gene identity varies, and v_g is the empirical version of "what this
        # gene's spread is".
        #
        # Detected genes only. Most of a spot's genes read exactly zero, so on the full panel the
        # statistic would be a detection statistic with a huge tie block at the bottom. Restricting
        # to the scale's own positive entries is also convention 2 of the analysis. Below the floor
        # the spot returns NaN rather than a noisy value.
        #
        # The naive dual is exported alongside (mean_det / var_det, across the spot's detected
        # genes) because it is what a reader will ask for; its caveat is the paragraph above, and
        # summary.json repeats it. A python loop over spots is the readable form here: the detected
        # gene set differs from spot to spot, so there is no vectorised Spearman to use.
        n_s = blk.shape[0]
        srec = {"dataset": ds, "slide": slide, "donor": donor,
                "adata_row": np.arange(idx[0], idx[-1] + 1, dtype=np.int64)}
        for name, S in (("log1p_cp10k", y), ("count_T", c)):
            v_g = rec[f"var_{name}"]
            det = S > 0
            n_det = det.sum(axis=1)
            rho_s = np.full(n_s, np.nan)
            mean_s = np.full(n_s, np.nan)
            var_s = np.full(n_s, np.nan)
            for i in range(n_s):
                d_i = det[i]
                if n_det[i] < args.min_detected_genes:
                    continue
                vals = S[i, d_i]
                mean_s[i] = vals.mean()
                var_s[i] = vals.var(ddof=1)
                # A constant input has no rank correlation. It is not hypothetical on the count
                # scale: a shallow spot can have every one of its detected genes at exactly 1
                # count. Skipped explicitly rather than left to emit a ConstantInputWarning per
                # spot, and it is why the count scale is the SECONDARY read on this axis (see the
                # tie caveat in summary.json).
                spread = v_g[d_i]
                if np.ptp(vals) > 0 and np.ptp(spread) > 0:
                    rho_s[i] = float(stats.spearmanr(vals, spread)[0])
            srec[f"n_det_{name}"] = n_det.astype(np.int64)
            srec[f"rho_level_spread_{name}"] = rho_s
            srec[f"mean_det_{name}"] = mean_s
            srec[f"var_det_{name}"] = var_s
            ok_s = np.isfinite(rho_s)
            summ_out.append(dict(dataset=ds, slide=slide, donor=donor, scale=name, axis="spot",
                                 spearman=float(np.median(rho_s[ok_s])) if ok_s.any() else np.nan,
                                 slope_loglog=np.nan,
                                 n_genes=int(np.median(n_det)), n_genes_loglog=0,
                                 n_spots=n_s, n_units=int(ok_s.sum())))
        spot_out.append(pd.DataFrame(srec))

        # summ_out order per slide: [-4] gene/log1p, [-3] gene/count_T, [-2] spot/log1p,
        # [-1] spot/count_T -- SCALES drives the first pair, the (name, S) tuple the second.
        print(f"[{_el():6.1f}s]   {slide:<12s} donor {donor:<10s} {blk.shape[0]:>6,d} spots  "
              f"gene axis rho(log1p)={summ_out[-4]['spearman']:+.3f} "
              f"rho(count_T)={summ_out[-3]['spearman']:+.3f}   "
              f"spot axis rho(log1p)={summ_out[-2]['spearman']:+.3f} "
              f"rho(count_T)={summ_out[-1]['spearman']:+.3f}", flush=True)
        del blk, y, c

    A.file.close()
    return (pd.concat(rows_out, ignore_index=True), pd.DataFrame(summ_out),
            pd.concat(spot_out, ignore_index=True))


triples = [one_dataset(ds) for ds in args.datasets]
per_gene = pd.concat([a for a, _b, _c in triples], ignore_index=True)
per_slide = pd.concat([b for _a, b, _c in triples], ignore_index=True)
per_spot = pd.concat([c for _a, _b, c in triples], ignore_index=True)

f_gene = os.path.join(out_dir, "rf_gene_meanvar.parquet")
f_slide = os.path.join(out_dir, "rf_meanvar_spearman.parquet")
f_spot = os.path.join(out_dir, "rf_spot_meanvar.parquet")
per_gene.to_parquet(f_gene, index=False)
per_slide.to_parquet(f_slide, index=False)
per_spot.to_parquet(f_spot, index=False)

summary = dict(
    datasets=list(args.datasets),
    preproc_id=args.preproc_id, radius=args.radius, panel=args.panel,
    depth_scale={ds: depth_scale_of(ds) for ds in args.datasets},
    scales=SCALES, axes=["gene", "spot"],
    min_detected_genes=int(args.min_detected_genes),
    n_rows_per_gene=int(len(per_gene)), n_rows_per_slide=int(len(per_slide)),
    n_rows_per_spot=int(len(per_spot)),
    spot_axis_note=("SPOT AXIS: level = y_sg (the spot's own value), spread = v_g (that gene's "
                    "variance across the spots of the SAME slide), Spearman over the spot's "
                    "DETECTED genes. A spot gives one draw per gene, so no within-spot variance "
                    "exists and the spread has to be borrowed from the gene; mean_det / var_det "
                    "are the naive dual and measure the dispersion of the spot's gene COMPOSITION, "
                    "not a conditional variance. Read log1p_cp10k as primary on this axis: on "
                    "count_T a spot's detected values are heavily tied at small integers, which "
                    "attenuates a rank correlation, and a spot whose detected values are ALL equal "
                    "returns NaN (counted in n_units)."),
    note=("v_g is the MARGINAL within-slide variance = E_s[Var(Y|x_s)] + Var_s(E[Y|x_s]); the "
          "second term is bounded by the attainable R^2 (a few percent here), so v_g stands in "
          "for the mean conditional variance. No model is involved in this file."),
)
with open(os.path.join(out_dir, "rf_gene_meanvar_summary.json"), "w") as fh:
    json.dump(summary, fh, indent=2)

print(f"[{_el():6.1f}s] wrote {len(per_gene):,} (dataset, slide, gene) rows -> {f_gene}", flush=True)
print(f"[{_el():6.1f}s] wrote {len(per_spot):,} (dataset, slide, spot) rows -> {f_spot}", flush=True)
print(f"[{_el():6.1f}s] wrote {len(per_slide):,} (dataset, slide, scale, axis) rows -> {f_slide}",
      flush=True)
print()
print("MEDIAN OVER SLIDES of the per-slide statistic, BY AXIS (the two are different estimands):")
print(per_slide.pivot_table(index=["dataset", "axis"], columns="scale",
                            values=["spearman", "slope_loglog"], aggfunc="median")
                .round(3).to_string())
print()
print("SPOT AXIS coverage: spots with a defined value / spots on the slide (a spot below the "
      f"{args.min_detected_genes}-detected-gene floor, or with all its detected values equal, has none):")
_sp = per_slide[per_slide["axis"] == "spot"]
print(_sp.groupby(["dataset", "scale"])[["n_units", "n_spots"]].sum()
         .assign(pct=lambda t: (100 * t["n_units"] / t["n_spots"]).round(1)).to_string())
