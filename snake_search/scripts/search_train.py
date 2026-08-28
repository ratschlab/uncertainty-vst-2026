#!/usr/bin/env python
# snake_search/scripts/search_train.py
#
# Train ONE config over ALL CV folds, IN THE PRODUCTION SETTING:
#   • MULTI-OUTPUT model: one mean-only MLP predicts the WHOLE 3000-HVG panel
#     (out_dim = n_genes), exactly like the classic pipeline — not one model per
#     gene, so the search tunes the architecture the pipeline actually trains.
#   • DONOR-grouped 5-fold GroupKFold over ALL donors (no test holdout). Grouping
#     by SLIDE would leak, since slides of one donor are near-
#     duplicates; production splits by donor (snake/scripts/train.py +
#     snake/config/donor_slide_map.yaml), so the search must too. This is
#     SELECTION cross-validation only — its val Pearson is a ranking signal, never
#     a reported generalization number (that comes later from LOPO on the frozen
#     pipeline).
#   • METRIC is DONOR-WEIGHTED slide-aware Pearson — the donor is the independent
#     unit, so a multi-slide donor is NOT counted as several observations:
#         r(slide,gene) -> mean over a donor's slides -> mean over donors
#                       -> mean over genes -> mean over folds.
#     (This differs from production's flat slide-mean, which weights donors by
#     their slide count. The flat slide-mean is also saved, as a diagnostic.)
#     SharedTensors.groups = slide (drives the per-(slide,gene) atoms); the donor
#     map regroups slides->donors for both the SPLIT and this METRIC.
#
# ONE JOB = ONE CONFIG, all folds. The adata is loaded ONCE and the GPU-resident
# SharedTensors reused across folds (the load — read + densify + GPU upload — is
# the expensive part; re-doing it per fold would waste it). The Snakefile
# wildcards {config_id} only; search_aggregate ranks configs by their mean
# val Pearson over folds.
#
# Reuses the SAME src functions as the classic pipeline (read_arrays,
# train_one_fold, run_inference, slide-aware metrics) so the picked architecture
# is the one the classic pipeline will actually train with.

import argparse, os, sys, json, time
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--configs", required=True)      # configs_v2.json from sample_configs
p.add_argument("--config-id", required=True)
p.add_argument("--n-folds", type=int, default=5)
p.add_argument("--dataset", required=True)
p.add_argument("--model-type", default="msemeanonly")
p.add_argument("--project-root", required=True)
p.add_argument("--adata", required=True)
p.add_argument("--gene-list-file", required=True,        # the HVG-top3000 list (one gene/line)
               help="text file, one gene per line — the multi-output target panel")
p.add_argument("--donor-map", required=True,             # snake/config/donor_slide_map.yaml
               help="donor->slide map; absent dataset => donor == slide (1:1)")
p.add_argument("--center", type=int, required=True,      # 0/1, match classic config
               help="subtract the train-fold embedding mean")
p.add_argument("--out", required=True)
a = p.parse_args()
t0 = time.time()                  # end-to-end wall time for THIS config (all folds)

sys.path.append(a.project_root)
import torch
from src.data.load_and_preprocess import read_arrays
from src.data.splitting import get_cv_splitter, load_slide_to_donor, donor_labels
from src.data.dataset_indexed import SharedTensors
from src.models.specs import get_spec
from src.training.train_loop_indexed import train_one_fold
from src.training.inference_indexed import run_inference

with open(a.configs) as f:
    blob = json.load(f)
fixed = blob["fixed"]
cfg = next(c for c in blob["configs"] if c["config_id"] == a.config_id)

SEED = fixed["seed"]
torch.manual_seed(SEED); np.random.seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CENTER = bool(a.center)

# The spec drives target source + loss + prediction extraction. msemeanonly ->
# adata.X (log-CP10k); the search tunes the shared backbone on this clean proxy.
spec = get_spec(a.model_type)

# Architecture override for the searched model (the search's main lever). The new
# `normalization` axis ({layernorm, none}) is threaded via arch -> build_model ->
# MLPMeanOnly (see src/models/specs.py::build_model).
ARCH = {"hidden_dims": cfg["hidden_dims"], "dropout": cfg["dropout"],
        "activation": cfg["activation"],
        "normalization": cfg.get("normalization", "layernorm")}

# ── load + slice to the MULTI-OUTPUT gene panel (ONCE for all folds) ──────────
import anndata as ad
adata = ad.read_h5ad(a.adata)
embeddings, y_all, sample_labels, available_genes = read_arrays(
    adata, target_source=spec.target_source,
)

with open(a.gene_list_file) as f:
    requested = [ln.strip() for ln in f if ln.strip()]
avail_set = set(available_genes)
gene_names = [g for g in requested if g in avail_set]      # keep panel order, drop absent
missing = [g for g in requested if g not in avail_set]
if missing:
    print(f"[warn] {len(missing)} of {len(requested)} HVG genes absent from the "
          f"panel (skipped), e.g. {missing[:5]}", flush=True)
if not gene_names:
    raise ValueError(f"no genes from {a.gene_list_file} are present in the dataset panel")
gene_idx = [available_genes.index(g) for g in gene_names]
y = y_all[:, gene_idx]                                     # (n_spots, n_genes)
n_genes = len(gene_names)

# GPU-resident data: per-batch gathers become pure on-device indexing. groups =
# slide label (drives the slide-aware metric), as in production. Built ONCE and
# reused across every fold below.
shared = SharedTensors(embeddings, y, sample_labels, device=DEVICE)
del adata, embeddings, y, y_all
print(f"[load] {shared.n_spots} spots x {shared.emb_dim}-d emb; "
      f"{n_genes}-gene panel; device {DEVICE}", flush=True)
load_sec = time.time() - t0       # adata read + densify + GPU upload (ONCE; EXCLUDED from train time)

# ── donor-grouped CV over ALL donors (selection only, no test holdout) ────────
slide_to_donor = load_slide_to_donor(a.donor_map, a.dataset)   # None => donor == slide
donor_per_spot = donor_labels(sample_labels, slide_to_donor)
n_donors = len(set(donor_per_spot.tolist()))
if n_donors < a.n_folds:
    raise ValueError(
        f"{a.n_folds}-fold donor CV needs >= {a.n_folds} donors, but dataset "
        f"'{a.dataset}' has only {n_donors}. Lower n_folds or pick another dataset."
    )

all_idx = np.arange(shared.n_spots)
cv = get_cv_splitter(f"{a.n_folds}fold")                       # GroupKFold (deterministic)
# GroupKFold is deterministic given the group array, so these folds are identical
# across every config -> a fair, paired comparison.
splits = list(cv.split(all_idx, groups=donor_per_spot))


def donor_weighted_score(atoms, slide_to_donor):
    """Donor-weighted slide-aware Pearson from per-(slide,gene) atoms:
        r(slide,gene) -> mean over a donor's slides -> mean over donors -> mean over genes.
    The donor is the independent unit, weighted equally (a multi-slide donor is
    collapsed first, so it does not count as several observations). NaN atoms
    (constant pred/target, <2 spots) are skipped at every mean, never zeroed.
    `atoms` = run_inference's pearson_per_slide_gene list of {slide, gene, pearson}.
    slide_to_donor None => donor == slide (1:1 dataset)."""
    per_gene = {}                                  # gene -> donor -> [pearson over its slides]
    for at in atoms:
        r = at["pearson"]
        if r != r:                                 # skip NaN
            continue
        slide = at["slide"]
        donor = slide_to_donor.get(slide, slide) if slide_to_donor else slide
        per_gene.setdefault(at["gene"], {}).setdefault(donor, []).append(r)
    gene_scores = []
    for donors in per_gene.values():
        donor_means = [float(np.mean(v)) for v in donors.values()]   # mean over a donor's slides
        gene_scores.append(float(np.mean(donor_means)))              # mean over donors (equal weight)
    return float(np.mean(gene_scores)) if gene_scores else float("nan")


train_start = time.time()
fold_val, fold_val_slidemean, fold_epochs, fold_ceiling = [], [], [], []
atom_rows = []                                     # per-(fold,slide,gene) atoms -> saved parquet
for fold, (tr_local, va_local) in enumerate(splits):
    train_idx, val_idx = all_idx[tr_local], all_idx[va_local]
    print(f"[fold {fold}/{a.n_folds}] train={len(train_idx)} val={len(val_idx)} "
          f"| val donors={sorted(set(donor_per_spot[val_idx].tolist()))}", flush=True)

    # train-fold embedding mean, on the data's device (emb is GPU-resident).
    _tr = torch.as_tensor(train_idx, dtype=torch.long, device=shared.emb.device)
    train_mean = shared.emb[_tr].mean(dim=0).cpu().numpy()

    _, epoch_log, model = train_one_fold(
        shared, train_idx, val_idx, train_mean,
        model_type=a.model_type, lam=1.0, n_genes=n_genes,
        lr=cfg["lr"], weight_decay=cfg["weight_decay"],
        batch_size=cfg["batch_size"], max_epochs=fixed["max_epochs"],
        patience=fixed["patience"], device=DEVICE,
        center=CENTER, train_subsample_idx=None, arch=ARCH,
        warmup_epochs=cfg.get("warmup_epochs", 0),
    )
    vr = run_inference(model, shared, val_idx, train_mean, spec=spec,
                       batch_size=cfg["batch_size"], device=DEVICE,
                       lam=1.0, center=CENTER)
    # Donor-weighted slide-aware score (the SELECTION metric) from the per-(slide,
    # gene) atoms; vr["pearson"] (flat slide-mean) is kept only as a diagnostic.
    atoms = vr["pearson_per_slide_gene"]            # [{slide, gene, pearson, n_spots}, ...]
    fold_val.append(donor_weighted_score(atoms, slide_to_donor))
    fold_val_slidemean.append(float(vr["pearson"]))
    for at in atoms:                                # accumulate for offline re-aggregation
        atom_rows.append({"fold": fold, "slide": at["slide"],
                          "gene": gene_names[at["gene"]], "pearson": at["pearson"],
                          "n_spots": at["n_spots"]})
    # Epoch-ceiling check: epoch_log is one row per epoch run, so
    # len() is epochs run. Many folds hitting max_epochs => early stopping never
    # bound => raise max_epochs (slow configs would be unfairly truncated).
    n_ep = len(epoch_log) if epoch_log is not None else None
    fold_epochs.append(n_ep)
    fold_ceiling.append(bool(n_ep is not None and n_ep >= fixed["max_epochs"]))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def nanmean(x):
    x = [v for v in x if v == v]
    return float(np.mean(x)) if x else float("nan")

def nanstd(x):
    x = [v for v in x if v == v]
    return float(np.std(x)) if x else float("nan")

train_sec = time.time() - train_start     # all folds' train + inference (load EXCLUDED)

# Save the per-(fold,slide,gene) atoms next to result.json so ANY future metric
# re-aggregation (donor-weighted, flat, gene subset, ...) is a pure offline step —
# no re-training needed. Mirrors production's atoms.parquet.
os.makedirs(os.path.dirname(a.out), exist_ok=True)
import pandas as pd
pd.DataFrame(atom_rows).to_parquet(
    os.path.join(os.path.dirname(a.out), "atoms.parquet"), index=False)

out = {
    "config_id": a.config_id, "n_folds": a.n_folds,
    "dataset": a.dataset, "model_type": a.model_type,
    "n_genes": n_genes,
    "metric": ("donor_weighted_slide_aware_pearson: r(slide,gene) -> mean over a "
               "donor's slides -> mean over donors -> mean over genes -> mean over folds"),
    "mean_val_pearson": nanmean(fold_val),   # config SCORE (donor-weighted, mean over folds)
    "std_val_pearson":  nanstd(fold_val),    # across-fold spread (noise floor)
    "per_fold_val_pearson": fold_val,        # donor-weighted, per fold (the selection metric)
    "per_fold_val_pearson_slidemean": fold_val_slidemean,  # flat slide-mean, diagnostic only
    "per_fold_n_epochs": fold_epochs,
    "epoch_ceiling_hits": fold_ceiling,      # one bool per fold
    "max_epochs": fixed["max_epochs"],
    "load_sec": round(load_sec, 2),          # ONCE (read + densify + GPU upload)
    "train_sec": round(train_sec, 2),        # all folds (load EXCLUDED)
    "elapsed_sec": round(time.time() - t0, 2),
    "config": {k: cfg[k] for k in (
        "hidden_dims", "dropout", "activation", "normalization",
        "weight_decay", "base_lr", "batch_size", "lr", "warmup_epochs")
        if k in cfg},
}
with open(a.out, "w") as f:
    json.dump(out, f, indent=2)
print(f"{a.config_id}: mean_val(donor-wtd)={out['mean_val_pearson']:.4f}±{out['std_val_pearson']:.4f} "
      f"(slide-mean diag {nanmean(fold_val_slidemean):.4f}) "
      f"over {a.n_folds} folds ({n_genes} genes) train={train_sec:.1f}s load={load_sec:.1f}s")
