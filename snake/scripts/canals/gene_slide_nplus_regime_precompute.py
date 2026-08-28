# snake/scripts/canals/gene_slide_nplus_regime_precompute.py
#
# How many spots detect each gene, per (gene, slide), and what that allows.
#
# n+(gene, slide) is the number of spots of that slide where the gene is non-zero. It is the
# sample size available to any statistic computed inside that one (gene, slide) pair, and it
# varies by three orders of magnitude across the panel. A pair with 4 detected spots and a pair
# with 900 cannot carry the same statistic, so the pair is labelled with a REGIME and every
# consumer reads the label rather than choosing a threshold of its own.
#
#   regime        n+        what it allows
#   exclu_vide    0         nothing; the gene is never detected on that slide
#   exclu         1-19      too few points, and pooling weak slides is unlikely to help
#   poole         20-59     not enough alone, but pooling this gene's weak slides may reach 60
#   terciles      60-149    a coarse within-pair statistic, 3 bins
#   quintiles     150-499   5 bins
#   deciles       >= 500    10 bins
#
# The count is scale-invariant: expm1(log1p(CP10k)) has the same zero locus as the raw counts,
# so "detected" means the same thing on every scale the models are trained on. The test is run
# on adata.X directly, since log1p(x) == 0 exactly when x == 0.
#
# Consumers:
#   snake/scripts/epistemic/knn_uncertainty_precompute.py  keeps the three usable regimes and
#       maps them to a bin count, then weights the aggregation cascade by n_plus.
#   notebooks/.../canal1_STimage_vs_DE.ipynb  reports how many genes have no usable pair at all
#       and how many are rescued by pooling.
#
# Usage (repo root, on the cluster; CPU only):
#   python -u snake/scripts/canals/gene_slide_nplus_regime_precompute.py
#   python -u snake/scripts/canals/gene_slide_nplus_regime_precompute.py --datasets TLS
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc

_T0 = time.time()
def _el():
    return time.time() - _T0

# Code and config live next to this script, so a frozen copy works too, whereas --project-root
# points at the data tree.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))

p = argparse.ArgumentParser()
p.add_argument("--project-root", default=REPO_ROOT)
p.add_argument("--datasets", nargs="+", default=["10x_TuPro", "TLS", "LUNG_CANCER"])
p.add_argument("--preproc-id", default="v5")
p.add_argument("--radius", default="r75")
p.add_argument("--out", default=None,
               help="Default: <project-root>/results/_analysis/gene_slide_nplus_regime.parquet")
p.add_argument("--overwrite", action="store_true")
args = p.parse_args()

PROJECT_ROOT = args.project_root.rstrip("/")
OUT = args.out or os.path.join(PROJECT_ROOT, "results", "_analysis",
                               "gene_slide_nplus_regime.parquet")

# The boundaries are the whole content of this file: everything else is a count. They are kept
# here, in one place, rather than in each consumer, so that two consumers cannot drift apart.
EDGES  = [0, 1, 20, 60, 150, 500]
LABELS = ["exclu_vide", "exclu", "poole", "terciles", "quintiles", "deciles"]


def regime_of(n_plus):
    """Label each n+ by the bracket it falls in. np.searchsorted keeps this O(n log k)."""
    return np.asarray(LABELS, dtype=object)[np.searchsorted(EDGES, n_plus, side="right") - 1]


def one_dataset(ds):
    path = os.path.join(PROJECT_ROOT, "results", ds, "_preprocessed",
                        args.preproc_id, args.radius, "adata.h5ad")
    adata = sc.read_h5ad(path)
    genes = adata.var_names.to_numpy()
    # The merged adata's slide column is obs["sample_id"]; obs["slide"] exists only in the
    # per-slide files that preprocessing reads.
    slides = adata.obs["sample_id"].astype(str).to_numpy()
    X = adata.X

    rows = []
    for s in pd.unique(slides):
        Xs = X[slides == s]
        # Count non-zeros per gene without densifying: getnnz on a sparse matrix, a boolean sum
        # otherwise. Either way this is one pass over the slide's rows.
        n_plus = Xs.getnnz(axis=0) if sp.issparse(Xs) else (np.asarray(Xs) != 0).sum(axis=0)
        rows.append(pd.DataFrame({"dataset": ds, "slide": s, "gene": genes,
                                  "n_plus": np.asarray(n_plus).ravel().astype(np.int32)}))
    out = pd.concat(rows, ignore_index=True)
    out["regime"] = regime_of(out["n_plus"].to_numpy())
    print(f"[{_el():6.1f}s] {ds:<12s} {out['slide'].nunique():>2d} slides x {len(genes):>5d} genes "
          f"= {len(out):>8,} pairs", flush=True)
    for lab in LABELS:
        k = int((out["regime"] == lab).sum())
        print(f"                 {lab:<12s} {k:>8,}  ({100 * k / len(out):5.1f} %)", flush=True)
    return out


if os.path.isfile(OUT) and not args.overwrite:
    raise SystemExit(f"exists, use --overwrite: {OUT}")

df = pd.concat([one_dataset(ds) for ds in args.datasets], ignore_index=True)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
df.to_parquet(OUT, index=False)
print(f"[{_el():6.1f}s] wrote {len(df):,} rows -> {os.path.relpath(OUT, PROJECT_ROOT)}", flush=True)
