# snake/scripts/canals/dataset_gene_expression_stats_precompute.py
#
# Per-gene expression statistics, model-independent: nothing here depends on a trained model.
#   - mean expression per gene, as log1p-CP10k, as CP10k, and as CP10k over detected spots only
#   - fraction of spots where the gene is not detected
#
# Two outputs, for two groupings:
#   (1) dataset-wide, one row per (dataset, gene), reduced over every spot of every slide.
#   (2) per donor, one row per (dataset, donor, gene), the same reductions within one patient.
#       Needed to put a Student-t interval on an expression panel: that interval is built from K
#       per-donor points, which the dataset-wide table cannot provide, collapsing all patients
#       into a single number per gene. The interval is descriptive, how much expression level
#       varies between patients, and is not a statement about model generalization.
#
#       Under LOPO each donor is the held-out test donor exactly once and all of that donor's
#       spots are that fold's test set, so "all spots of donor d" is "donor d's test spots" and
#       these per-donor means line up with a model's per-donor points with no population mismatch.
#
# Why a precompute rather than inline work: the merged adata is large and dense, so a full read
# plus column reduction can exhaust an interactive shell. The result is a property of the data and
# not of any run, so it is computed once per dataset from the shared preprocessed adata and
# cached, and every consumer then loads these small parquets.
#
# Scale note. Raw CP10k is strongly right-skewed, so a symmetric Student-t interval places both of
# its bounds badly there. log1p compresses the tail, which makes `mean_log1p_cp10k` the column to
# build a t interval on. `mean_cp10k` is kept for a bootstrap-percentile interval, which is free
# to be asymmetric.
#
# Outputs, derived analysis artifacts that no pipeline rule reads:
#   results/_analysis/gene_expression_stats.parquet            (dataset, gene, ...)
#   results/_analysis/gene_expression_stats_per_donor.parquet  (dataset, donor, gene, ...)
# both with columns: n_spots, pct_zeros, mean_log1p_cp10k, mean_cp10k, mean_cp10k_nonzero
#
# Usage (repo root, on the cluster; CPU only, so omit -p/--gres and land on `compute`):
#   conda activate mtenv0
#   python -u snake/scripts/canals/dataset_gene_expression_stats_precompute.py
#   # or a subset: --datasets 10x_TuPro TLS --preproc-id <id> --radius <radius>

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc

_T0 = time.time()
def _el():
    return time.time() - _T0

# Code and config live next to this script, so a copy of the tree works too, whereas
# --project-root points at the data tree.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
DEFAULT_PROJECT_ROOT = REPO_ROOT

p = argparse.ArgumentParser()
p.add_argument("--project-root", default=DEFAULT_PROJECT_ROOT)
p.add_argument("--datasets", nargs="+", default=["10x_TuPro", "TLS", "LUNG_CANCER"])
p.add_argument("--preproc-id", default="v5")
p.add_argument("--radius", default="r75")
p.add_argument("--donor-map", default=os.path.join(REPO_ROOT, "snake/config/donor_slide_map.yaml"),
               help="{donor: [slides]} map; the SAME file the CV splits use, so the per-donor "
                    "grouping here matches the folds exactly.")
p.add_argument("--out", default=None,
               help="Default: <project-root>/results/_analysis/gene_expression_stats.parquet")
p.add_argument("--out-per-donor", default=None,
               help="Default: <project-root>/results/_analysis/gene_expression_stats_per_donor.parquet")
p.add_argument("--chunk", type=int, default=20000,
               help="Row (spot) chunk for the expm1 pass -- bounds peak memory (no full expm1(X) copy).")
args = p.parse_args()

sys.path.append(REPO_ROOT)
from src.data.splitting import load_slide_to_donor, donor_labels   # noqa: E402  (needs sys.path)

project_root = args.project_root.rstrip("/") + "/"
out_path = args.out or f"{project_root}results/_analysis/gene_expression_stats.parquet"
out_path_donor = (args.out_per_donor
                  or f"{project_root}results/_analysis/gene_expression_stats_per_donor.parquet")


def _dense(a):
    return a.toarray() if sp.issparse(a) else np.asarray(a)


def _reduce_rows(X, idx, chunk):
    """Per-gene reductions over a subset of X's rows -> dict of (n_genes,) arrays + the row count.

    X is the merged adata's dense (n_spots, n_genes) log1p(CP10k) matrix and CP10k = expm1(X). The
    zero locus is the same on every scale, a count of 0 staying 0 throughout. idx=None means all
    rows, and the chunk is then a plain slice, a view rather than a copy.

    The CP10k mean goes through a chunked expm1 pass so that no full expm1(X) copy is
    materialized: a second dense (n_spots, n_genes) array would double peak memory. expm1(0) = 0,
    so the per-gene CP10k sum is the sum over detected spots and the nonzero mean is
    sum / n_nonzero.
    """
    n_rows = X.shape[0] if idx is None else int(idx.size)
    n_genes = X.shape[1]
    sum_log1p = np.zeros(n_genes, dtype=np.float64)
    n_nonzero = np.zeros(n_genes, dtype=np.float64)
    sum_cp = np.zeros(n_genes, dtype=np.float64)
    for i in range(0, n_rows, chunk):
        blk = X[i:i + chunk] if idx is None else X[idx[i:i + chunk]]
        sum_log1p += blk.sum(axis=0)
        n_nonzero += (blk != 0).sum(axis=0)
        sum_cp += np.expm1(blk).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_cp_nonzero = np.where(n_nonzero > 0, sum_cp / n_nonzero, 0.0)
    return {
        "n_spots": n_rows,
        "pct_zeros": 1.0 - n_nonzero / n_rows,
        "mean_log1p_cp10k": sum_log1p / n_rows,
        "mean_cp10k": sum_cp / n_rows,
        "mean_cp10k_nonzero": mean_cp_nonzero,
    }


def gene_stats_one_dataset(ds):
    """(dataset-wide df, per-donor df) for one dataset, from the merged adata's X."""
    path = f"{project_root}results/{ds}/_preprocessed/{args.preproc_id}/{args.radius}/adata.h5ad"
    print(f"[{_el():6.1f}s] {ds}: reading {path}", flush=True)
    adata = sc.read_h5ad(path)
    X = _dense(adata.X).astype(np.float64)              # (n_spots, n_genes), log1p(CP10k)
    genes = np.asarray(adata.var_names)

    wide = pd.DataFrame({"dataset": ds, "gene": genes, **_reduce_rows(X, None, args.chunk)})
    print(f"[{_el():6.1f}s] {ds}: dataset-wide done ({len(genes):,} genes x {X.shape[0]:,} spots)",
          flush=True)

    # Per donor. The merged adata's slide column is obs["sample_id"]; obs["slide"] exists only in
    # the prediction files, so "slide" here would raise KeyError. donor_labels() maps slide ->
    # donor through the shared donor_slide_map.yaml, identity for a 1:1 dataset, and raises on an
    # unmapped slide, so a typo fails loudly instead of silently mis-grouping patients.
    donors = donor_labels(adata.obs["sample_id"].astype(str).to_numpy(),
                          load_slide_to_donor(args.donor_map, ds))
    pieces = []
    for d in sorted(set(donors)):
        idx = np.flatnonzero(donors == d)
        st = _reduce_rows(X, idx, args.chunk)
        pieces.append(pd.DataFrame({"dataset": ds, "donor": d, "gene": genes, **st}))
        print(f"[{_el():6.1f}s]   donor {d}: {st['n_spots']:,} spots", flush=True)
    per_donor = pd.concat(pieces, ignore_index=True)

    n_spots_sum = int(per_donor.groupby("donor")["n_spots"].first().sum())
    assert n_spots_sum == X.shape[0], (
        f"{ds}: per-donor spot counts sum to {n_spots_sum} but the adata has {X.shape[0]} spots -- "
        "the donor grouping is not a partition of the spots.")
    return wide, per_donor


pairs = [gene_stats_one_dataset(ds) for ds in args.datasets]
combined = pd.concat([w for w, _ in pairs], ignore_index=True)
combined_donor = pd.concat([d for _, d in pairs], ignore_index=True)

os.makedirs(os.path.dirname(out_path), exist_ok=True)
combined.to_parquet(out_path, index=False)
combined_donor.to_parquet(out_path_donor, index=False)
print(f"[{_el():6.1f}s] wrote {len(combined):,} (dataset, gene) rows -> {out_path}", flush=True)
print(f"[{_el():6.1f}s] wrote {len(combined_donor):,} (dataset, donor, gene) rows -> {out_path_donor}",
      flush=True)
for ds, g in combined.groupby("dataset"):
    n_don = combined_donor.loc[combined_donor["dataset"] == ds, "donor"].nunique()
    print(f"   {ds:14s}: {len(g):,} genes | {n_don} donors | mean CP10k median={g['mean_cp10k'].median():.3f} "
          f"| pct_zeros median={g['pct_zeros'].median():.3f}", flush=True)
