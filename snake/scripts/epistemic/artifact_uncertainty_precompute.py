# snake/scripts/epistemic/artifact_uncertainty_precompute.py
#
# Dose-response of aleatoric and epistemic uncertainty against image-artifact severity
# (blur, salt-and-pepper), for the two epistemic methods that share one decomposition:
# MC-dropout and the deep ensemble.
#
# Both methods are T stochastic models; only the source of the draws differs (dropout masks
# resampled per forward pass, or M independently-seeded members). The accumulation is
# therefore identical, and per (spot, gene):
#     pred      = mean_t m_t
#     epistemic = var_t  m_t                  <- disagreement between draws
#     aleatoric = mean_t sigma^2_t(x)         <- the variance head's own output
# Gaussian heads only (target_source == "X" plus a variance head): they live on log1p-CP10k,
# so nothing here touches depth_scale, g* or any count-scale caveat. A count head would need
# its own aleatoric formula and its own reporting axis.
#
# What varies. The model is frozen, trained on clean slides. Only its input changes: the same
# test spots are re-embedded from artifact-corrupted slides
# (patch_embeddings/<ds>/embeddings_..._blurry_bf{X}/ and ..._saltpepper_a{A}_sp{min}-{max}/,
# produced by snake/scripts/embeddings_extraction/launch_{blur,saltpepper}_extraction.sh).
# Severity 0 is not a re-extracted dir: it is adata.obsm["X_emb"], the exact embeddings the
# model was trained and evaluated on (whichever extraction the dataset's emb_subdir names in
# config.yaml: the second extraction for 10x_TuPro and TLS, the plain one for LUNG_CANCER).
# So the clean point reproduces the standard post-hoc numbers and every corruption family
# shares one reference.
#
# LOPO. Each donor is the held-out test donor exactly once, so a fold's test spots are that
# donor's spots, scored by the model that never saw them. The donor is the unit of
# replication: the output keeps one row per (corruption, severity, fold), and an analysis
# takes mean +- SD over donors.
#
# Output, by default under <run-dir>/artifact_uncertainty/. Every matrix is reduced on the
# device, so nothing per-(spot, gene) is written and the files stay small.
#   artifact_uncertainty.parquet           one row per (corruption, severity, fold)
#     + n_spots, n_genes, n_draws, pred_mean, aleatoric_mean, epistemic_mean, total_mean,
#       abs_resid_mean, rmse, and three columns that put the artifact on the same axes as the
#       rest of the analysis:
#         pearson               slide-aware Pearson (per (slide, gene) -> nanmean slides ->
#                               nanmean genes). Same aggregation as the on-device twin used for
#                               early stopping, not the canal-2 cascade (no 60-entry floor, no
#                               donor-weighted fold-back), so it is not a canal-2 number.
#         unc_err_spearman_gene error-uncertainty rank correlation on the gene axis, across
#                               the spots of one gene on one slide, aggregated the same way.
#         pred_sd               SD of the prediction across spots, per gene, averaged over genes.
#                               Goes to zero if the artifact collapses every spot onto the same
#                               embedding, which would force a constant prediction.
#   artifact_uncertainty_per_spot.parquet  one row per (corruption, severity, spot), the mean
#     over genes, for the un-aggregated view. Columns: obs_name, slide, pred, aleatoric,
#     epistemic, abs_resid. obs_name joins these rows to any other per-spot table.
#   artifact_uncertainty_per_gene.parquet  one row per (corruption, severity, gene, fold), the
#     mean over spots, plus a per-gene `pearson` (the slide-aware atoms averaged over slides
#     only). The clean severity of that column ranks the genes by how well they are predicted,
#     so an analysis can read the collapse on the best-predicted genes rather than on a
#     3000-gene average in which most genes were never predicted at all.
#   artifact_uncertainty_trace_genes.parquet  one row per (corruption, severity, slide, gene)
#     for the --trace-genes list, the only view carrying both axes at once. Written only when
#     that list names at least one gene present in the panel.
#   summary.json                           the run's parameters and the severity grid.
#
# Usage (repo root, cluster; GPU strongly preferred, the sweep is many forward passes):
#   # MC-dropout on a single heteroscedastic model:
#   python -u snake/scripts/epistemic/artifact_uncertainty_precompute.py --method mc \
#       --run-dir results/10x_TuPro/<run_id>/_10x_TuPro_hvg_top3000/r75/heteroscedastic/lambda_0.0
#   # Deep ensemble. --run-dir is the model dir holding the seed_*/ members, and it has no
#   # lambda_ segment: the ensemble lambda is constant across members and is not in the path.
#   python -u snake/scripts/epistemic/artifact_uncertainty_precompute.py --method ensemble \
#       --run-dir results/10x_TuPro/<run_id>/_10x_TuPro_hvg_top3000/r75/heteroscedastic

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

_T0 = time.time()
def _el():
    return time.time() - _T0

# ── Args ──────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--run-dir", required=True,
               help="method=mc: a classic run dir (model_fold*.pt + splits.json + metrics.json). "
                    "method=ensemble: the model dir HOLDING the seed_*/ member dirs.")
p.add_argument("--method", required=True, choices=["mc", "ensemble"])
p.add_argument("--emb-dataset", default=None,
               help="patch_embeddings/<name>/ holding the corrupted dirs. Default: the run's "
                    "own dataset (metrics.json). The merged LUNG_CANCER dir is filled by the "
                    "two LUAD/LUSC extraction jobs, so the default works there too.")
p.add_argument("--radius-factor", default="0.75")
p.add_argument("--swap-xy", default="True")
p.add_argument("--blur-factors", type=float, nargs="*", default=[0.005, 0.01, 0.02, 0.05],
               help="Severity 0 is the clean X_emb and is always added, so do NOT list 0.0.")
p.add_argument("--sp-amounts", type=float, nargs="*", default=[0.01, 0.05, 0.1, 0.2])
p.add_argument("--sp-specks", nargs="*", default=["1-1", "2-4"],
               help="Speck-size variants, as '{min}-{max}' (matches the extraction dir name).")
p.add_argument("--T", type=int, default=30, help="method=mc: number of dropout forward passes.")
p.add_argument("--batch-size", type=int, default=4096)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--device", default=None, help="cuda|cpu (default: auto).")
p.add_argument("--trace-genes", nargs="*", default=[],
               help="Gene names to also write per (slide, gene, severity). Genes absent from a "
                    "cohort's panel are skipped with a warning, so one shared list works for all "
                    "three cohorts. Empty -> that file is not written.")
p.add_argument("--out-dir", default=None, help="Default: <run-dir>/artifact_uncertainty/")
args = p.parse_args()

DEVICE  = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
run_dir = os.path.abspath(args.run_dir)
out_dir = args.out_dir or os.path.join(run_dir, "artifact_uncertainty")
os.makedirs(out_dir, exist_ok=True)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(PROJECT_ROOT)

import anndata as ad
from src.data.load_and_preprocess import read_arrays
from src.evaluation.metrics_slide_aware import (
    corr_scalars_gpu, _nanmean_t, _pearson_cols, _slide_row_indices)
from src.models.specs import get_spec, build_model

# ── Members: one dir for MC-dropout, the seed_*/ dirs for the ensemble ─────────
# A member is a normal run with an extra /seed_{seed}/ segment, so every member dir carries
# its own model_fold*.pt + train_mean; splits.json / metrics.json are identical across
# members (same donor CV, same data), so the first member describes the whole ensemble.
if args.method == "ensemble":
    member_dirs = sorted(glob.glob(os.path.join(run_dir, "seed_*")))
    if not member_dirs:
        raise FileNotFoundError(f"No seed_*/ member dirs in {run_dir} (is this the ensemble model dir?)")
else:
    member_dirs = [run_dir]
meta_dir = member_dirs[0]

with open(os.path.join(meta_dir, "metrics.json")) as f:
    M = json.load(f)
model_type    = M["model_type"]
target_source = M["target_source"]
center        = bool(M["center"])
arch          = {k: M["architecture"][k] for k in ("hidden_dims", "dropout", "activation")}
data_path     = M["input_adata"]
dataset       = M["dataset"]
spec          = get_spec(model_type)

# Gaussian variance head only — the aleatoric here IS the head's sigma^2, on log1p-CP10k.
if not (target_source == "X" and spec.is_probabilistic):
    raise ValueError(
        f"{model_type}: this script is Gaussian-only (target_source == 'X' + a variance head). "
        f"A count head's aleatoric is mu (Poisson) or mu + alpha*mu^2 (NB), on its own target "
        f"scale — a different formula and a different reporting axis, deliberately out of scope.")

# Ensemble members run with compute_final_metrics=false, which writes a stub metrics.json
# carrying the model-reload fields but not gene_names. Recover them from a saved prediction
# file, read backed so only var_names is loaded and never the matrix.
gene_names = M.get("gene_names")
if gene_names is None:
    pred_files = sorted(glob.glob(os.path.join(meta_dir, "predictions", "preds_fold*.h5ad")))
    if not pred_files:
        raise FileNotFoundError(f"No gene_names in {meta_dir}/metrics.json and no predictions/ to read them from.")
    gene_names = list(ad.read_h5ad(pred_files[0], backed="r").var_names)
G = len(gene_names)

print(f"Run dir : {run_dir}", flush=True)
print(f"Method  : {args.method} ({len(member_dirs)} member dir(s))"
      + (f", T={args.T} dropout passes" if args.method == "mc" else ""), flush=True)
print(f"Model   : {model_type} on {dataset}, n_genes={G}, center={center}, device={DEVICE}", flush=True)

# ── Corruption levels: (corruption, severity, emb_dir or None for the clean X_emb) ────
EMB_BASE = os.path.join(PROJECT_ROOT, "patch_embeddings", args.emb_dataset or dataset)
_stem    = f"embeddings_radius_factor_{args.radius_factor}_swapxy_{args.swap_xy}"

levels = [("blur", 0.0, None)]
levels += [("blur", bf, os.path.join(EMB_BASE, f"{_stem}_blurry_bf{bf}")) for bf in args.blur_factors]
for sp in args.sp_specks:
    levels.append((f"saltpepper_sp{sp}", 0.0, None))
    levels += [(f"saltpepper_sp{sp}", a, os.path.join(EMB_BASE, f"{_stem}_saltpepper_a{a}_sp{sp}"))
               for a in args.sp_amounts]

# Check every dir up front: a missing extraction should name itself now, not surface as a
# bare FileNotFoundError from torch.load halfway through the sweep.
missing = sorted({d for _, _, d in levels if d is not None and not os.path.isdir(d)})
if missing:
    raise FileNotFoundError(
        "Missing corrupted embedding dir(s):\n  " + "\n  ".join(missing) +
        "\nExtract them with snake/scripts/embeddings_extraction/launch_{blur,saltpepper}_extraction.sh "
        "(DATASET in 10x_TuPro | TLS | LUNG_CANCER_LUAD | LUNG_CANCER_LUSC).")
print(f"Levels  : {len(levels)} = 1 clean + {len(args.blur_factors)} blur + "
      f"{len(args.sp_specks)}x({1}+{len(args.sp_amounts)}) salt-and-pepper", flush=True)

# ── Data: clean embeddings + targets + the join keys ───────────────────────────
t_data = time.time()
adata = ad.read_h5ad(data_path)
obs_names_all = np.asarray(adata.obs_names).astype(str)
emb_clean, y_all, slide_all, available_genes = read_arrays(adata, target_source=target_source,
                                                           depth_scale=M.get("depth_scale"))
gene_idx = [available_genes.index(g) for g in gene_names]
y_all    = np.asarray(y_all[:, gene_idx], dtype=np.float32)
emb_clean = np.asarray(emb_clean, dtype=np.float32)
slide_all = np.asarray(slide_all).astype(str)
del adata
D = emb_clean.shape[1]
print(f"[{_el():7.1f}s] data loaded: {emb_clean.shape[0]} spots x {D} emb dims, "
      f"{G} genes (read {time.time()-t_data:.1f}s)", flush=True)

with open(os.path.join(meta_dir, "splits.json")) as f:
    splits = json.load(f)
folds = {fo["fold"]: (np.asarray(fo["test_idx"], dtype=np.int64), fo.get("test_donor"))
         for fo in splits["folds"]}


def load_corrupted(emb_dir, rows):
    """The SAME spots as `rows`, re-embedded from the corrupted slides. Joined by barcode
    (obs_name is '{barcode}-{slide}'; the .pt stores the raw barcodes), never by position."""
    out = np.empty((rows.size, D), dtype=np.float32)
    for sl in pd.unique(slide_all[rows]):
        m = slide_all[rows] == sl
        blob = torch.load(os.path.join(emb_dir, f"{sl}.pt"), map_location="cpu", weights_only=False)
        e     = blob["embeddings"].float().numpy()
        names = np.asarray(blob["obs_names"]).astype(str)
        want  = np.array([n[:-(len(sl) + 1)] for n in obs_names_all[rows][m]])
        order = pd.Index(names).get_indexer(want)
        assert (order >= 0).all(), f"barcode mismatch loading {emb_dir}/{sl}.pt"
        out[m] = e[order]
    return out


@torch.no_grad()
def summarize(draws, X, y_t, slide_codes, n_slides):
    """Accumulate the draw moments ONLINE over `draws` = [(model, train_mean), ...], never
    holding (n_draws, n, G). Sums in float64 so var = E[m^2] - E[m]^2 stays accurate. The
    per-(spot, gene) matrices stay on the device and are reduced there into THREE views:
    the scalars, the per-SPOT means (mean over genes) and the per-GENE means (mean over
    spots). Only those reduced vectors ever cross to the host."""
    n = X.shape[0]
    X_t  = torch.as_tensor(X, device=DEVICE)
    s_m  = torch.zeros((n, G), dtype=torch.float64, device=DEVICE)
    s_m2 = torch.zeros((n, G), dtype=torch.float64, device=DEVICE)
    s_v  = torch.zeros((n, G), dtype=torch.float64, device=DEVICE)
    for model, train_mean in draws:
        mean_t = (torch.as_tensor(np.asarray(train_mean), dtype=torch.float32, device=DEVICE)
                  if center else None)
        for b0 in range(0, n, args.batch_size):
            xb = X_t[b0:b0 + args.batch_size]
            if center:
                xb = xb - mean_t
            out = spec.extract_predictions(model(xb))
            sl  = slice(b0, b0 + xb.shape[0])
            m   = out["means"].reshape(xb.shape[0], G).double()
            s_m[sl]  += m
            s_m2[sl] += m * m
            s_v[sl]  += out["pred_var"].reshape(xb.shape[0], G).double()
    nd    = len(draws)
    pred  = s_m / nd
    epi   = torch.clamp(s_m2 / nd - pred * pred, min=0.0)    # clip tiny negative rounding
    ale   = s_v / nd
    absr  = (pred - y_t).abs()
    # Slide-aware Pearson: the metric the rest of this repository reports, so the artifact numbers
    # sit on the same axis as every other result. Aggregation is the on-device twin's --
    # one atom per (slide, gene), nanmean over slides, nanmean over genes -- NOT the canal-2
    # cascade (no 60-entry floor, no donor-weighted fold-back), so it is not a canal-2 number.
    pearson = corr_scalars_gpu(pred.float(), y_t.float(), slide_codes, n_slides,
                               corr_metrics=("pearson",))["pearson"]
    # Error-uncertainty rank correlation on the GENE axis, i.e. across the spots of one gene on
    # one slide: the same quantity canal 2 reports, recomputed here at each severity. Same
    # aggregation as the Pearson above.
    unc_err = corr_scalars_gpu((ale + epi).float(), absr.float(), slide_codes, n_slides,
                               corr_metrics=("spearman",))["spearman"]
    scalars = dict(
        pred_mean       = float(pred.mean()),
        # SD across spots, per gene, then averaged over genes. If the artifact collapses the
        # embeddings onto one point, the prediction becomes constant and this goes to zero.
        pred_sd         = float(pred.std(0).mean()),
        pearson         = float(pearson),
        unc_err_spearman_gene = float(unc_err),
        aleatoric_mean  = float(ale.mean()),
        epistemic_mean  = float(epi.mean()),
        total_mean      = float((ale + epi).mean()),
        abs_resid_mean  = float(absr.mean()),
        rmse            = float(torch.sqrt(((pred - y_t) ** 2).mean())),
    )
    # dim=1 -> one value per SPOT (mean over genes) ; dim=0 -> one per GENE (mean over spots)
    # `pred` is carried per spot as well: it is what makes the (mean, aleatoric) trajectory
    # readable PER SLIDE, i.e. whether sigma moves along mu (coupled) or on its own (decoupled),
    # and whether the slides move in the same direction at all.
    per_spot = {k: v.cpu().numpy() for k, v in
                dict(pred=pred.mean(1), aleatoric=ale.mean(1), epistemic=epi.mean(1),
                     abs_resid=absr.mean(1)).items()}
    per_gene = {k: v.cpu().numpy() for k, v in
                dict(pred=pred.mean(0), aleatoric=ale.mean(0), epistemic=epi.mean(0),
                     abs_resid=absr.mean(0)).items()}
    # Per-gene Pearson: the SAME atoms as the scalar above, stopped one step earlier (nanmean
    # over slides only, no nanmean over genes). It is what lets an analysis rank the genes by
    # how well they are predicted on clean input, then read the collapse on the best-predicted
    # ones instead of on a 3000-gene average dominated by genes the model never predicted.
    # The helpers are imported from metrics_slide_aware, so there is one implementation of the
    # formula, not two.
    pg = torch.full((n_slides, G), float("nan"), dtype=torch.float64, device=DEVICE)
    for s, rows in enumerate(_slide_row_indices(slide_codes, n_slides, DEVICE)):
        if rows.numel() >= 2:
            pg[s] = _pearson_cols(pred.float().index_select(0, rows),
                                  y_t.float().index_select(0, rows), 1e-12)
    per_gene["pearson"] = _nanmean_t(pg, dim=0).cpu().numpy()
    # The ONE view that carries both axes at once: for a handful of named genes, the mean over
    # the spots of each SLIDE. Neither of the two views above can give it -- per_spot is already
    # averaged over genes, per_gene over spots -- and the full (slide, gene) grid would be a
    # different object entirely. Empty unless --trace-genes is passed.
    trace = []
    if trace_gene_idx:
        for s, rows in enumerate(_slide_row_indices(slide_codes, n_slides, DEVICE)):
            if rows.numel() == 0:
                continue
            for name, gi in trace_gene_idx.items():
                trace.append(dict(slide_code=int(s), gene=name,
                                  pred=float(pred[rows, gi].mean()),
                                  aleatoric=float(ale[rows, gi].mean()),
                                  epistemic=float(epi[rows, gi].mean()),
                                  abs_resid=float(absr[rows, gi].mean())))
    return scalars, per_spot, per_gene, trace


def load_draws(fold):
    """The fold's draw list. MC-dropout: ONE model, dropout left on, repeated T times (each
    forward resamples its masks). Ensemble: the M members' checkpoints for this fold, all in
    plain eval mode. Same downstream accumulation either way."""
    models = []
    for md in member_dirs:
        ck = os.path.join(md, f"model_fold{fold}.pt")
        # weights_only=False: our own checkpoint (trusted) carries a numpy train_mean.
        ckpt  = torch.load(ck, map_location=DEVICE, weights_only=False)
        model = build_model(model_type, in_dim=D, n_genes=G, device=DEVICE, arch=arch,
                            model_kwargs={"log_alpha_init": M.get("nb_log_alpha_init", -2.0)})
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        models.append((model, ckpt["train_mean"]))
    if args.method == "mc":
        (model, _) = models[0]
        n_do = 0
        for mod in model.modules():                 # eval() everything, dropout back to train
            if isinstance(mod, torch.nn.Dropout):
                mod.train(); n_do += 1
        if n_do == 0:
            raise RuntimeError("No nn.Dropout modules — MC-dropout needs dropout > 0.")
        torch.manual_seed(args.seed)
        return models * args.T
    return models


# ── Sweep: fold OUTER (checkpoints loaded once), severity INNER ────────────────
# The traced genes, resolved to column indices once. A gene the cohort's panel does not carry
# is named and skipped, so the same list can be passed to all three cohorts.
trace_gene_idx = {g: gene_names.index(g) for g in args.trace_genes if g in gene_names}
if args.trace_genes:
    missing = [g for g in args.trace_genes if g not in gene_names]
    print(f"[trace] {len(trace_gene_idx)} gene(s) traced per slide: {list(trace_gene_idx)}"
          + (f" | absent from this panel: {missing}" if missing else ""), flush=True)

rows_out, spot_out, gene_out, trace_out = [], [], [], []
for fold in sorted(folds):
    te, donor = folds[fold]
    donor_id  = donor[0] if isinstance(donor, list) else donor
    draws = load_draws(fold)
    y_t   = torch.as_tensor(y_all[te], dtype=torch.float64, device=DEVICE)
    # Slide id per spot, as consecutive codes: what corr_scalars_gpu wants for the per-(slide,
    # gene) atoms. A donor can hold several slides, and the metric groups by SLIDE, never donor.
    codes, slide_uniques = pd.factorize(slide_all[te])
    slide_codes = torch.as_tensor(codes, dtype=torch.long, device=DEVICE)
    n_slides    = int(codes.max()) + 1
    for corruption, severity, emb_dir in levels:
        X = emb_clean[te] if emb_dir is None else load_corrupted(emb_dir, te)
        stats, per_spot, per_gene, trace = summarize(draws, X, y_t, slide_codes, n_slides)
        keys = dict(dataset=dataset, model_type=model_type, method=args.method,
                    corruption=corruption, severity=float(severity),
                    fold=int(fold), test_donor=donor_id)
        rows_out.append(dict(**keys, n_spots=int(te.size), n_genes=int(G),
                             n_draws=int(len(draws)), **stats))
        # The two un-aggregated views, from the same matrices reduced on the other axis:
        # one value per spot (mean over genes) and one per gene (mean over spots).
        spot_out.append(pd.DataFrame(dict(obs_name=obs_names_all[te], slide=slide_all[te],
                                          **keys, **per_spot)))
        gene_out.append(pd.DataFrame(dict(gene=np.asarray(gene_names), **keys, **per_gene)))
        if trace:
            t = pd.DataFrame(trace)
            t["slide"] = slide_uniques[t["slide_code"].to_numpy()]   # code -> slide name
            trace_out.append(t.drop(columns="slide_code").assign(**keys))
        print(f"[{_el():7.1f}s] fold {fold} ({donor_id}) {corruption} sev={severity}: "
              f"r={stats['pearson']:.4f} ale={stats['aleatoric_mean']:.5f} "
              f"epi={stats['epistemic_mean']:.6f} |res|={stats['abs_resid_mean']:.5f} "
              f"pred_sd={stats['pred_sd']:.5f}", flush=True)
    del y_t
    del draws
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

df = pd.DataFrame(rows_out)
out_path = os.path.join(out_dir, "artifact_uncertainty.parquet")
df.to_parquet(out_path, index=False)

for frames, name in ((spot_out, "artifact_uncertainty_per_spot.parquet"),
                     (gene_out, "artifact_uncertainty_per_gene.parquet"),
                     (trace_out, "artifact_uncertainty_trace_genes.parquet")):
    if not frames:
        continue
    t = pd.concat(frames, ignore_index=True)
    for c in ("dataset", "model_type", "method", "corruption", "test_donor", "slide", "gene"):
        if c in t.columns:
            t[c] = t[c].astype("category")       # dictionary-encoded -> the file stays small
    t.to_parquet(os.path.join(out_dir, name), index=False)
    print(f"[{_el():7.1f}s] {name}: {len(t)} rows", flush=True)

# Donor-averaged read-off, printed so the .out alone answers "does uncertainty react?".
# Two blocks, because the two families of column are not read the same way. The uncertainty and
# error columns are ratios to the clean point (1.0 = did not move). The correlations and the
# prediction spread are printed RAW: a ratio of correlations is not interpretable, and pred_sd
# is meaningful in its own units.
RATIO_COLS = ["aleatoric_mean", "epistemic_mean", "abs_resid_mean"]
RAW_COLS   = ["pearson", "unc_err_spearman_gene", "pred_sd"]
agg = df.groupby(["corruption", "severity"])[RATIO_COLS + RAW_COLS].mean()
for corruption in df["corruption"].unique():
    sub = agg.loc[corruption]
    print(f"\n{corruption}  --  ratio to clean")
    print((sub[RATIO_COLS] / sub[RATIO_COLS].loc[0.0]).round(3).to_string())
    print(f"{corruption}  --  raw (pearson and the error-uncertainty rank correlation are "
          f"slide-aware, gene-averaged)")
    print(sub[RAW_COLS].round(4).to_string())

with open(os.path.join(out_dir, "summary.json"), "w") as f:
    json.dump(dict(run_dir=run_dir, method=args.method, dataset=dataset, model_type=model_type,
                   emb_dataset=args.emb_dataset or dataset, n_members=len(member_dirs),
                   T=(args.T if args.method == "mc" else None), n_genes=G,
                   n_folds=len(folds), levels=[[c, s] for c, s, _ in levels],
                   uncertainty_scale="log1p_cp10k (Gaussian variance head, no rescale)",
                   rows=len(df), out=out_path,
                   also=["artifact_uncertainty_per_spot.parquet",
                         "artifact_uncertainty_per_gene.parquet"]), f, indent=2)
print(f"\n[{_el():7.1f}s] DONE. {len(df)} rows -> {out_path}", flush=True)
