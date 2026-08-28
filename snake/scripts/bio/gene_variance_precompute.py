#!/usr/bin/env python
"""Per-gene variance references, computed from the data alone, with no model involved.

The question this feeds: is the model's per-gene predicted uncertainty anything more than a static
property of the gene, namely how variable that gene is? A reference that needs no image, no
training and no split would make the per-gene uncertainty a lookup table.

Three references are written, on three deliberately different scales, because they are not
interchangeable:

  variances_norm  the seurat_v3 standardised variance, the quantity the HVG selection ranks on.
                  Computed on raw counts with the mean-variance trend already regressed out, so it
                  is the closest thing to an expression-free variability the data offers.
  variances       the raw-count variance it derives from, so that what the standardisation does is
                  visible rather than assumed.
  var_logcp       the variance of adata.X, that is of log1p(CP10k). This is the scale the Gaussian
                  models predict on, so it is the only strictly comparable reference.

They have to be recomputed: load_and_preprocess.py drops means / variances / variances_norm from
the saved adata and keeps only the boolean highly_variable_top{N} flags. The parameters below are
copied from it so the numbers are the ones the panel was actually selected with: flavor seurat_v3,
batch_key sample_id, span 0.3, layer raw_counts.

Two aggregations are written for the log1p-CP10k quantities, because they answer different
questions:
  *_pooled  over every spot of the cohort at once.
  *_slide   per slide, then the median over slides, so between-slide mean shifts do not enter it.

Reads <proj>/results/<dataset>/_preprocessed/<preproc-id>/<radius>/adata.h5ad.
Writes <out>/gene_variance_{dataset}.parquet, one row per gene of the merged panel, plus a matching
.json recording the adata path, the slide count and the HVG parameters used.

Usage (cluster):
  python -u snake/scripts/bio/gene_variance_precompute.py --dataset TLS
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

# copied from src/data/load_and_preprocess.py, do not "improve" them independently
HVG_FLAVOR = "seurat_v3"
HVG_BATCH_KEY = "sample_id"
HVG_SPAN = 0.3
RAW_LAYER = "raw_counts"


def col_mean_var(M):
    """Per-column mean and population variance, sparse or dense.

    ddof=0, numpy's default, so these line up with a plain np.var computed elsewhere.
    """
    if sp.issparse(M):
        m = np.asarray(M.mean(axis=0), dtype=np.float64).ravel()
        m2 = np.asarray(M.multiply(M).mean(axis=0), dtype=np.float64).ravel()
        return m, np.maximum(m2 - m * m, 0.0)          # clamp the float cancellation
    X = np.asarray(M, dtype=np.float64)
    return X.mean(axis=0), X.var(axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj", default=str(Path(__file__).resolve().parents[3]))
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--preproc-id", default="v5")
    ap.add_argument("--radius", default="r75")
    ap.add_argument("--n-top", type=int, default=3000,
                    help="the panel size the models were trained on; only sets which genes are "
                         "flagged, every gene of the merged panel is written")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    proj = Path(args.proj)
    adata_path = (proj / "results" / args.dataset / "_preprocessed" / args.preproc_id /
                  args.radius / "adata.h5ad")
    out_dir = Path(args.out) if args.out else proj / "results" / "_analysis" / "gene_variance"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"reading {adata_path}", flush=True)
    adata = sc.read_h5ad(adata_path)
    print(f"  {adata.n_obs:,} spots x {adata.n_vars:,} genes", flush=True)
    if RAW_LAYER not in adata.layers:
        raise SystemExit(f"layer '{RAW_LAYER}' missing; seurat_v3 needs raw counts. "
                         f"Re-preprocess or point at a newer preproc_id.")
    if HVG_BATCH_KEY not in adata.obs:
        raise SystemExit(f"obs['{HVG_BATCH_KEY}'] missing, cannot reproduce the HVG batching.")

    # ── the seurat_v3 quantities, recomputed with the preprocessing parameters ──────────────
    print(f"seurat_v3 HVG: n_top={args.n_top}, batch_key={HVG_BATCH_KEY}, span={HVG_SPAN}",
          flush=True)
    sc.pp.highly_variable_genes(adata, flavor=HVG_FLAVOR, n_top_genes=int(args.n_top),
                                batch_key=HVG_BATCH_KEY, span=HVG_SPAN, layer=RAW_LAYER)
    df = pd.DataFrame({"gene": np.asarray(adata.var_names).astype(str)})
    for c in ("means", "variances", "variances_norm"):
        if c not in adata.var.columns:
            raise SystemExit(f"scanpy did not return var['{c}']; check the scanpy version.")
        df[c] = adata.var[c].to_numpy(dtype=np.float64)
    df["is_hvg_top"] = adata.var["highly_variable"].to_numpy(dtype=bool)
    print(f"  flagged {int(df['is_hvg_top'].sum()):,} genes", flush=True)

    # ── the log1p-CP10k quantities, on the scale the Gaussian models predict ────────────────
    X = adata.X
    m_pool, v_pool = col_mean_var(X)
    df["mean_logcp_pooled"] = m_pool
    df["var_logcp_pooled"] = v_pool

    slides = adata.obs[HVG_BATCH_KEY].astype(str).to_numpy()
    order = np.unique(slides)
    if sp.issparse(X):
        X = X.tocsr()
    means, vars_ = [], []
    for sl in order:
        idx = np.flatnonzero(slides == sl)
        m, v = col_mean_var(X[idx])
        means.append(m)
        vars_.append(v)
    # median over slides: the slide-aware reading, immune to between-slide mean shifts
    df["mean_logcp_slide"] = np.median(np.vstack(means), axis=0)
    df["var_logcp_slide"] = np.median(np.vstack(vars_), axis=0)
    print(f"  slide-aware over {len(order)} slides", flush=True)

    # ── detection rate, the third gene property that tracks all of the above ───────────────
    R = adata.layers[RAW_LAYER]
    if sp.issparse(R):
        df["detection_rate"] = np.asarray((R > 0).mean(axis=0), dtype=np.float64).ravel()
    else:
        df["detection_rate"] = (np.asarray(R) > 0).mean(axis=0)

    out = out_dir / f"gene_variance_{args.dataset}.parquet"
    df.to_parquet(out, index=False)
    meta = dict(dataset=args.dataset, adata=str(adata_path), n_spots=int(adata.n_obs),
                n_genes=int(adata.n_vars), n_slides=int(len(order)), n_top=int(args.n_top),
                hvg_flavor=HVG_FLAVOR, hvg_batch_key=HVG_BATCH_KEY, hvg_span=HVG_SPAN,
                raw_layer=RAW_LAYER, scanpy=sc.__version__)
    (out_dir / f"gene_variance_{args.dataset}.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out}  ({len(df):,} genes)", flush=True)
    print(df.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
