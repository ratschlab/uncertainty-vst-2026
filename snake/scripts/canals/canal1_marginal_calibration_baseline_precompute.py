# snake/scripts/canals/canal1_marginal_calibration_baseline_precompute.py
#
# Marginal calibration baseline, one file per dataset: how well a predictive distribution
# calibrates when it never sees the image.
#
# An absolute ece_pit means nothing on its own, so each model needs a reference. This fits a
# distribution directly on the true expression of each gene, without the image and therefore
# constant across spots, and scores it with the same native-CDF one-sided PIT and the same
# slide-aware, per-(held-out donor, gene) aggregation as the models. The gap model minus marginal
# of the same family is what the histology adds to calibration. A well-fit marginal is well
# calibrated and yet has near-zero Pearson, predicting the same value everywhere, which is the
# demonstration that calibration is not spatial utility.
#
# Three family baselines, one curve each; a model is compared to its own family:
#   gaussian : per gene mu_g, sigma_g of adata.X (log1p-CP10k);  PIT via the Gaussian CDF.
#   poisson  : per gene rate mu_g = mean(y_cp);                  PIT via the Poisson CDF.
#   nb       : per gene mu_g + method-of-moments dispersion a_g; PIT via the NB CDF.
#
# Count scale, matching poisson_cp and the NB cp heads. The _cp models train and score on the
# target y_cp = round(CP10k*depth_scale) = round(CP_T), where depth_scale = T/1e4 is one
# dataset-dependent constant. There is no size factor, no offset and no per-spot exposure: the
# likelihood is fitted directly on y_cp. The count baseline is therefore a plain per-gene fit on
# y_cp with mu constant across spots. The PIT is the native discrete Poisson or NB CDF at the
# integer y_cp with smear=False, cp_counts being rounded. depth_scale is read from a cp run's own
# predictions (uns["depth_scale"]), and y_cp is rebuilt via
# read_arrays(..., "cp_counts", depth_scale), so it is identical to the models' target.
#
# Leakage. Each fold is fit on the train+val (non-held-out) spots and the PIT is evaluated on that
# fold's held-out donor test spots, the exact spots the model is scored on. That yields K
# per-donor points per family, to overlay on the model curves.
#
# Output, next to the other _analysis precomputes:
#   {out_dir}/{dataset}.parquet  long: family, test_donor, gene, ece_pit, pit_bias, disp_signed
#   {out_dir}/{dataset}.json     meta: depth_scale, n_genes, donors, source run
#
# Usage (cluster login node, CPU only):
#   python -u snake/scripts/canals/canal1_marginal_calibration_baseline_precompute.py --dataset 10x_TuPro
#   python -u snake/scripts/canals/canal1_marginal_calibration_baseline_precompute.py --dataset TLS --overwrite

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import scanpy as sc

_T0 = time.time()
def _el():
    return time.time() - _T0

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
p = argparse.ArgumentParser()
p.add_argument("--dataset", required=True, help="10x_TuPro | TLS | LUNG_CANCER")
p.add_argument("--proj-root", default=PROJECT_ROOT)
p.add_argument("--run-id", default="pr_classic3")
p.add_argument("--radius", default="r75")
p.add_argument("--results-base", default=None, help="default {proj}/results")
p.add_argument("--ref-model", default="poisson_cp",
               help="a cp run to read splits/genes/adata/depth_scale from (its splits+genes are "
                    "model-independent; depth_scale comes from its predictions uns).")
p.add_argument("--out-dir", default=None,
               help="default {proj}/results/_analysis/canal1_marginal_baseline")
p.add_argument("--overwrite", action="store_true")
args = p.parse_args()

sys.path.append(PROJECT_ROOT)
from src.data.load_and_preprocess import read_arrays
from src.evaluation.metrics_slide_aware import per_slide_gene_calibration
from src.evaluation.calibration_pit import (
    gaussian_pit_bounds, poisson_pit_bounds, nb_pit_bounds,
)

RESULTS_BASE = args.results_base or f"{args.proj_root}/results"
OUT_DIR      = args.out_dir      or f"{args.proj_root}/results/_analysis/canal1_marginal_baseline"
os.makedirs(OUT_DIR, exist_ok=True)
out_pq = os.path.join(OUT_DIR, f"{args.dataset}.parquet")
out_js = os.path.join(OUT_DIR, f"{args.dataset}.json")
if os.path.exists(out_pq) and os.path.exists(out_js) and not args.overwrite:
    print(f"skip (exists): {out_pq}  (pass --overwrite to recompute)")
    sys.exit(0)

# ── Reference cp run: splits.json (folds), atoms.parquet (gene panel), metrics.json
#    (input_adata) and predictions (depth_scale). All model-independent for our purposes. ──
ref_glob = os.path.join(RESULTS_BASE, args.dataset, args.run_id, "*", args.radius,
                        args.ref_model, "metrics.json")
ref_hits = glob.glob(ref_glob)
assert len(ref_hits) == 1, f"expected exactly one ref run, got {len(ref_hits)}: {ref_glob}\n{ref_hits}"
ref_dir = os.path.dirname(ref_hits[0])
print(f"ref run: {ref_dir}", flush=True)

metrics = json.loads(open(os.path.join(ref_dir, "metrics.json")).read())
input_adata = metrics["input_adata"]
splits = json.loads(open(os.path.join(ref_dir, "splits.json")).read())
assert splits.get("lopo"), "baseline assumes LOPO (each donor is the held-out TEST once)."

gene_list = sorted(pd.read_parquet(os.path.join(ref_dir, "atoms.parquet"),
                                   columns=["gene"])["gene"].unique().tolist())
G = len(gene_list)

# depth_scale is read from the reference cp run's own predictions: uns["depth_scale"] is the
# constant. Its absence means a depth-blind *_raw run, which cannot serve as the reference for a
# cp baseline, so that case is rejected.
pred_files = sorted(glob.glob(os.path.join(ref_dir, "predictions", "preds_fold*.h5ad")))
assert pred_files, f"no predictions under {ref_dir}/predictions"
p0 = sc.read_h5ad(pred_files[0])
if p0.uns.get("count_convention") != "v2_no_offset":
    raise ValueError(f"{pred_files[0]}: artifact with no count_convention marker. "
                     f"Re-run the precompute.")
depth_scale = p0.uns.get("depth_scale")
assert depth_scale is not None, (
    f"{pred_files[0]} is marked v2_no_offset but carries no uns['depth_scale'] -- that means "
    f"a depth-blind *_raw run, which cannot serve as the reference for this cp baseline. "
    f"Point --ref-model at a _cp run.")
depth_scale = float(depth_scale)
del p0
print(f"[{_el():6.1f}s] {G} genes, depth_scale={depth_scale}, {len(splits['folds'])} LOPO folds",
      flush=True)

# ── Targets, restricted to the model's gene panel. read_arrays is the same target
#    reconstruction train.py uses, so y_cp is identical to what the _cp models saw. ──
adata = sc.read_h5ad(input_adata)
_, y_cp_all, slides_all, gene_names = read_arrays(adata, target_source="cp_counts", depth_scale=depth_scale)
_, y_X_all,  _,          _          = read_arrays(adata, target_source="X")
slides_all = np.asarray(slides_all)
name_to_col = {g: i for i, g in enumerate(gene_names)}
missing = [g for g in gene_list if g not in name_to_col]
assert not missing, f"{len(missing)} model genes absent from adata, e.g. {missing[:5]}"
cols = np.asarray([name_to_col[g] for g in gene_list], dtype=np.int64)
y_cp = np.asarray(y_cp_all[:, cols], dtype=float); del y_cp_all
y_X  = np.asarray(y_X_all[:, cols],  dtype=float); del y_X_all
print(f"[{_el():6.1f}s] loaded adata: {y_cp.shape[0]} spots x {G} genes", flush=True)


# ── Per-family fit on the `fit` rows, native-CDF PIT bounds on the `test` rows. Each returns
#    (F_lo, F_hi, means, pred_var, y_test). mu, sigma and alpha are constant per gene, so the
#    only difference from a real model is that mu is a per-gene constant and not f(x).
#    means and pred_var are passed only because per_slide_gene_calibration also emits
#    var_residual_corr, ignored here; the calibration scalars come from F_lo and F_hi alone. ──
def fit_gaussian(fit, test):
    mu = np.nanmean(y_X[fit], axis=0)
    sd = np.nanstd(y_X[fit], axis=0, ddof=1)
    sd = np.where(sd > 1e-8, sd, 1e-8)                      # floor: avoid a degenerate 0-width CDF
    yt = y_X[test]
    F_lo, F_hi = gaussian_pit_bounds(mu[None, :], sd[None, :], yt)
    return F_lo, F_hi, np.broadcast_to(mu, yt.shape), np.broadcast_to(sd ** 2, yt.shape), yt

def fit_poisson(fit, test):
    mu = np.nanmean(y_cp[fit], axis=0)
    mu = np.where(mu > 1e-6, mu, 1e-6)                      # floor: keep the Poisson CDF well-defined
    yt = y_cp[test]
    mu_bc = np.broadcast_to(mu, yt.shape)
    F_lo, F_hi = poisson_pit_bounds(mu_bc, yt)              # smear=False (cp_counts is rounded)
    return F_lo, F_hi, mu_bc, mu_bc, yt                     # Poisson var = mu

def fit_nb(fit, test):
    mu  = np.nanmean(y_cp[fit], axis=0)
    var = np.nanvar(y_cp[fit], axis=0, ddof=1)
    mu = np.where(mu > 1e-6, mu, 1e-6)
    alpha = np.clip((var - mu) / (mu ** 2), 1e-6, 1e6)      # MoM; underdispersed -> ~Poisson
    yt = y_cp[test]
    mu_bc, al_bc = np.broadcast_to(mu, yt.shape), np.broadcast_to(alpha, yt.shape)
    F_lo, F_hi = nb_pit_bounds(mu_bc, al_bc, yt)            # smear=False
    return F_lo, F_hi, mu_bc, mu_bc + al_bc * mu_bc ** 2, yt

FAMILY = {"gaussian": fit_gaussian, "poisson": fit_poisson, "nb": fit_nb}

# ── LOPO folds: fit each family on the non-held-out spots, PIT on the held-out donor's test
#    spots, slide-aware reduce (one atom per (slide, gene), then nanmean over slides) -> one
#    value per (family, donor, gene), the long shape a consumer reduces to K per-donor points
#    for donor_lopo_stats. ──
rows, donors = [], []
for fold in splits["folds"]:
    donor = ",".join(map(str, fold["test_donor"]))
    test = np.asarray(fold["test_idx"], dtype=np.int64)
    fit = np.unique(np.concatenate(
        [np.asarray(ins["train_idx"], dtype=np.int64) for ins in fold["inner_splits"]] +
        [np.asarray(ins["val_idx"],   dtype=np.int64) for ins in fold["inner_splits"]]))
    test_slides = slides_all[test]
    donors.append(donor)
    for fam, fit_fn in FAMILY.items():
        F_lo, F_hi, means, pv, yt = fit_fn(fit, test)
        atoms, _, _ = per_slide_gene_calibration(means, yt, pv, F_lo, F_hi, test_slides)
        red = (pd.DataFrame(atoms)
               .groupby("gene")[["ece_pit", "pit_bias", "disp_signed"]].mean())   # over slides
        for gidx, r in red.iterrows():
            rows.append(dict(family=fam, test_donor=donor, gene=gene_list[int(gidx)],
                             ece_pit=r["ece_pit"], pit_bias=r["pit_bias"],
                             disp_signed=r["disp_signed"]))
    print(f"[{_el():6.1f}s] donor={donor}: fit={len(fit)} test={len(test)} spots "
          f"({len(np.unique(test_slides))} slides)", flush=True)

df = pd.DataFrame(rows)
df.to_parquet(out_pq)
meta = dict(dataset=args.dataset, depth_scale=depth_scale,
            n_genes=G, families=list(FAMILY),
            n_donors=len(donors), donors=donors, ref_run_dir=ref_dir, input_adata=input_adata,
            run_id=args.run_id, radius=args.radius)
open(out_js, "w").write(json.dumps(meta, indent=2))
print(f"[{_el():6.1f}s] wrote {out_pq} ({len(df):,} rows: {len(FAMILY)} families x "
      f"{len(donors)} donors x {G} genes) + {out_js}", flush=True)
