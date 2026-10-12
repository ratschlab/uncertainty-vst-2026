# snake/scripts/canals/hvg_overlap_precompute.py
#
# Rebuttal check: the 3,000-HVG panel was selected on all spots of a cohort, held-out donors
# included. Without retraining:
#   1. per LOPO fold j, rerun the selection on the train + val spots only (G_j) and report the
#      overlap o_j = |G n G_j| / |G|, G being the same selection rerun on all spots;
#   2. recompute the paper summaries per donor on (paper panel n G_j) and compare them to the full
#      panel: accuracy and calibration (benchmark_summary.parquet), gene axis (canal2
#      pair_partial.parquet), spot axis (--spot-model, recomputed from the predictions).
# Selection settings as src/data/load_and_preprocess.py: seurat_v3, 3000 genes, batch = slide,
# span 0.3, raw counts.
#
# Usage (CPU; the launcher runs the panel job first, then one spot job per head):
#   python -u snake/scripts/canals/hvg_overlap_precompute.py --dataset TLS
#   python -u snake/scripts/canals/hvg_overlap_precompute.py --dataset TLS --spot-model heteroscedastic

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import scanpy as sc
import yaml
from scipy.stats import rankdata

_T0 = time.time()
def _el():
    return time.time() - _T0

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(PROJECT_ROOT)
from src.evaluation.benchmark_summary import donor_lopo_stats
from src.evaluation.count_metrics import count_pred_var

p = argparse.ArgumentParser()
p.add_argument("--dataset", required=True, help="10x_TuPro | TLS | LUNG_CANCER")
p.add_argument("--proj-root", default=PROJECT_ROOT)
p.add_argument("--run-id", default="pr_classic3")
p.add_argument("--radius", default="r75")
p.add_argument("--spot-model", default=None,
               help="spot-axis mode for one model_type (needs the panel job's genes.parquet)")
p.add_argument("--out-dir", default=None, help="default {proj}/results/_analysis/rebuttal_hvg_overlap")
args = p.parse_args()

DS = args.dataset
RUN_BASE = f"{args.proj_root}/results/{DS}/{args.run_id}/_{DS}_hvg_top3000/{args.radius}"
OUT = os.path.join(args.out_dir or f"{args.proj_root}/results/_analysis/rebuttal_hvg_overlap", DS)
os.makedirs(OUT, exist_ok=True)

N_TOP, BATCH_KEY, HVG_SPAN = 3000, "sample_id", 0.3

# Model lists of the paper notebooks: *_RANK defines the gene panel averaged over, *_REPORT is drawn.
ACC_RANK   = ["msemeanonly", "homoscedastic", "heteroscedastic", "spot_homoscedastic", "poisson_cp",
              "negativebinomial_hetero_noanchor_cp", "negativebinomial_spotdisp_cp",
              "negativebinomial_genedisp_cp"]
ACC_REPORT = ACC_RANK[:6]
CAL_RANK   = ACC_RANK[:6]
CAL_REPORT = ACC_RANK[1:6]
AXIS_MODELS = ACC_RANK[1:6]
REFERENCE  = "msemeanonly"
N_PLUS_MIN = 60  # as canal2
EPS = 1e-12

DONOR_MAP = yaml.safe_load(open(f"{PROJECT_ROOT}/snake/config/donor_slide_map.yaml"))
S2D = {sl: d for d, slides in (DONOR_MAP.get(DS) or {}).items() for sl in slides}


def run_dir(model):
    """Gaussian heads are read at lambda 0.0, as in the paper."""
    base = os.path.join(RUN_BASE, model)
    lam = [d for d in sorted(glob.glob(os.path.join(base, "lambda_*")))
           if np.isclose(float(os.path.basename(d)[len("lambda_"):]), 0.0)]
    return lam[0] if lam else base


def compare(summary, model, stat, full, shared, rows, donor_rows):
    donors = sorted(set(full) & set(shared))
    diff = {d: shared[d] - full[d] for d in donors}
    for scope, pts in (("full", full), ("shared", shared), ("diff", diff)):
        st = donor_lopo_stats(pts)
        rows.append(dict(summary=summary, model=model, stat=stat, scope=scope, mean=st["mean"],
                         ci_lo=st["ci95_mean"][0], ci_hi=st["ci95_mean"][1], K=st["K"]))
    for d in sorted(set(full) | set(shared)):
        donor_rows.append(dict(summary=summary, model=model, stat=stat, test_donor=d,
                               full=full.get(d, np.nan), shared=shared.get(d, np.nan)))


# --- spot-axis mode ---
def corr(n, sx, sy, sxy, sxx, syy):
    num = n * sxy - sx * sy
    den = np.sqrt((n * sxx - sx ** 2) * (n * syy - sy ** 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.isfinite(den) & (den > EPS), num / den, np.nan)


def dense(a):
    return np.asarray(a.toarray() if hasattr(a, "toarray") else a, dtype=np.float32)


def spot_partial(mu, u, err):
    """Per spot (row): naive and partial Spearman, as canal2."""
    ru, rm, rd = (rankdata(a, axis=1) for a in (u, mu, err))
    n = float(mu.shape[1])
    S = lambda a: a.sum(axis=1)
    su, sm, sd = S(ru), S(rm), S(rd)
    suu, smm, sdd = S(ru * ru), S(rm * rm), S(rd * rd)
    r_uD = corr(n, su, sd, S(ru * rd), suu, sdd)
    r_um = corr(n, su, sm, S(ru * rm), suu, smm)
    r_Dm = corr(n, sd, sm, S(rd * rm), sdd, smm)
    one_um, one_Dm = 1.0 - r_um ** 2, 1.0 - r_Dm ** 2
    den = np.sqrt(one_um * one_Dm)
    with np.errstate(divide="ignore", invalid="ignore"):
        r_partial = (r_uD - r_um * r_Dm) / den
    bad = ~np.isfinite(den) | (den < EPS) | (one_um < EPS) | (one_Dm < EPS)
    return r_uD, np.where(bad, np.nan, r_partial)


if args.spot_model is not None:
    model = args.spot_model
    assert model in AXIS_MODELS, f"{model} not a paper spot-axis head {AXIS_MODELS}"
    rd_ = run_dir(model)
    genes_df = pd.read_parquet(os.path.join(OUT, "genes.parquet"))
    shared_by_donor = {d: set(g.loc[g["in_paper"] & g["in_fold"], "gene"])
                       for d, g in genes_df.groupby("test_donor")}
    print(f"run: {rd_}", flush=True)

    per_spot = []
    for f in sorted(glob.glob(os.path.join(rd_, "predictions", "preds_fold*.h5ad"))):
        B = sc.read_h5ad(f, backed="r")
        B = B[B.obs["split"].to_numpy() == "test"].to_memory()
        donor = str(B.obs["test_donor"].iloc[0])
        assert B.obs["test_donor"].nunique() == 1 and donor in shared_by_donor, f"{f}: donor {donor}"
        X, Y = dense(B.X), dense(B.layers["y"])
        # u = predictive SD, as canal2
        if model == "poisson_cp":
            U = np.sqrt(count_pred_var(X))
        elif model.startswith("negativebinomial"):
            U = np.sqrt(count_pred_var(X, dispersion=dense(B.layers["dispersion"])))
        else:
            U = np.sqrt(dense(B.layers["pred_var"]))
        if model in ("poisson_cp", "negativebinomial_hetero_noanchor_cp"):
            assert B.uns.get("count_convention") == "v2_no_offset", f"{f}: pre-refactor count artifact"
        E = np.abs(X - Y)
        keep = np.isin(np.asarray(B.var_names), list(shared_by_donor[donor]))
        for scope, cols in (("full", np.ones(B.n_vars, dtype=bool)), ("shared", keep)):
            r_uD, r_par = spot_partial(X[:, cols], U[:, cols], E[:, cols])
            per_spot.append(pd.DataFrame({
                "scope": scope, "test_donor": donor, "slide": B.obs["slide"].astype(str).to_numpy(),
                "spot": B.obs["adata_row"].to_numpy(), "n": int(cols.sum()),
                "r_uD": r_uD, "r_partial": r_par}))
        print(f"[{_el():6.1f}s] {os.path.basename(f)} donor={donor}: {B.n_obs:,} spots, "
              f"{B.n_vars} genes -> {int(keep.sum())} shared", flush=True)
        del B, X, Y, U, E
    per_spot = pd.concat(per_spot, ignore_index=True)

    # control: full-panel values must match canal2 spot_partial.parquet
    c2 = pd.read_parquet(os.path.join(rd_, "canal2_uncertainty", "spot_partial.parquet"),
                         columns=["spot", "r_uD_lin_all", "r_partial_lin_all"])
    m = per_spot[per_spot["scope"] == "full"].merge(c2, on="spot", how="inner")
    maxdiff = lambda a, b: float(np.abs(a - b).max()) if (a - b).notna().any() else float("nan")
    control = dict(n_spots_recomputed=int((per_spot["scope"] == "full").sum()),
                   n_spots_matched=int(len(m)),
                   max_abs_diff_r_uD=maxdiff(m["r_uD"], m["r_uD_lin_all"]),
                   max_abs_diff_r_partial=maxdiff(m["r_partial"], m["r_partial_lin_all"]),
                   nan_pattern_equal=bool((m["r_uD"].isna() == m["r_uD_lin_all"].isna()).all()
                                          and (m["r_partial"].isna() == m["r_partial_lin_all"].isna()).all()))
    print(f"[{_el():6.1f}s] CONTROL vs canal2 spot_partial: {control}", flush=True)

    # per-donor mean over spots with n >= 60
    rows, donor_rows = [], []
    for stat, tag in (("r_uD", "naive"), ("r_partial", "partial")):
        pts = {}
        for scope in ("full", "shared"):
            d = per_spot[(per_spot["scope"] == scope) & (per_spot["n"] >= N_PLUS_MIN)
                         & per_spot[stat].notna()]
            pts[scope] = d.groupby("test_donor")[stat].mean().to_dict()
        if pts["full"]:
            compare("spot_axis", model, tag, pts["full"], pts["shared"], rows, donor_rows)
    pd.DataFrame(rows).to_parquet(os.path.join(OUT, f"spot_{model}.parquet"), index=False)
    pd.DataFrame(donor_rows).to_parquet(os.path.join(OUT, f"spot_{model}_per_donor.parquet"), index=False)
    with open(os.path.join(OUT, f"spot_{model}.json"), "w") as fh:
        json.dump(dict(dataset=DS, model=model, run_dir=rd_, control_vs_canal2=control), fh, indent=2)
    print(f"[{_el():6.1f}s] done: {OUT}/spot_{model}.parquet", flush=True)
    sys.exit(0)


# --- panel mode, part 1: overlap ---
def hvg_ranked(a):
    """seurat_v3 as in preprocessing; returns the selected genes and their highly_variable_rank."""
    sc.pp.highly_variable_genes(a, flavor="seurat_v3", n_top_genes=N_TOP,
                                batch_key=BATCH_KEY, span=HVG_SPAN)
    v = a.var
    sel = v.index[v["highly_variable"].to_numpy()]
    return set(sel), v.loc[sel, "highly_variable_rank"].to_dict()


ref_dir = run_dir(REFERENCE)
metrics = json.load(open(os.path.join(ref_dir, "metrics.json")))
splits = json.load(open(os.path.join(ref_dir, "splits.json")))
assert splits.get("lopo"), "LOPO only: each donor is the held-out test donor once"
for m in AXIS_MODELS:
    s = json.load(open(os.path.join(run_dir(m), "splits.json")))
    assert [f["test_donor"] for f in s["folds"]] == [f["test_donor"] for f in splits["folds"]], m

adata = sc.read_h5ad(metrics["input_adata"], backed="r")
print(f"[{_el():6.1f}s] {metrics['input_adata']}: {adata.n_obs:,} spots x {adata.n_vars:,} genes",
      flush=True)
G_paper = set(adata.var_names[adata.var[f"highly_variable_top{N_TOP}"].to_numpy()])
slim = sc.AnnData(X=adata.layers["raw_counts"], obs=adata.obs[[BATCH_KEY]].copy(),
                  var=pd.DataFrame(index=adata.var_names.copy()))
del adata

# G = the selection rerun on all spots; it can differ from the saved panel by a few cutoff ties.
G, rank_full = hvg_ranked(slim.copy())
n_mismatch = len(G_paper ^ G)
print(f"[{_el():6.1f}s] all spots rerun vs saved paper panel: symmetric difference = {n_mismatch}",
      flush=True)

ov_rows, gene_rows, panels = [], [], {}
for fold in splits["folds"]:
    donor = ",".join(map(str, fold["test_donor"]))
    inner = fold["inner_splits"][0]
    keep = np.asarray(sorted(inner["train_idx"] + inner["val_idx"]), dtype=np.int64)
    test = np.asarray(fold["test_idx"], dtype=np.int64)
    assert not np.intersect1d(keep, test).size, f"{donor}: train/val overlap the test spots"
    assert keep.size + test.size == slim.n_obs, f"{donor}: train+val+test != all spots"
    Gj, rank_fold = hvg_ranked(slim[keep].copy())
    panels[donor] = Gj
    out_, in_ = G - Gj, Gj - G
    ov_rows.append(dict(fold=fold["fold"], test_donor=donor, n_spots_selection=int(keep.size),
                        n_full=len(G), n_fold=len(Gj), n_shared=len(G & Gj),
                        overlap=len(G & Gj) / len(G),
                        n_swapped=len(out_),
                        out_rank_full_median=float(np.nanmedian([rank_full[g] for g in out_])) if out_ else np.nan,
                        out_rank_full_min=float(np.nanmin([rank_full[g] for g in out_])) if out_ else np.nan,
                        in_rank_fold_median=float(np.nanmedian([rank_fold[g] for g in in_])) if in_ else np.nan,
                        in_rank_fold_min=float(np.nanmin([rank_fold[g] for g in in_])) if in_ else np.nan))
    for g in sorted(G | Gj):
        gene_rows.append(dict(test_donor=donor, gene=g, in_full=g in G, in_fold=g in Gj, in_paper=g in G_paper,
                              rank_full=rank_full.get(g, np.nan), rank_fold=rank_fold.get(g, np.nan)))
    print(f"[{_el():6.1f}s] fold {fold['fold']} donor={donor}: overlap {ov_rows[-1]['overlap']:.4f} "
          f"({len(out_)} swapped)", flush=True)
del slim

overlap = pd.DataFrame(ov_rows)
genes_df = pd.DataFrame(gene_rows)
overlap.to_parquet(os.path.join(OUT, "overlap.parquet"), index=False)
genes_df.to_parquet(os.path.join(OUT, "genes.parquet"), index=False)
in_all_folds = G.intersection(*panels.values())

# --- panel mode, part 2: summaries on G n G_j ---
shared_by_donor = {d: G_paper & Gj for d, Gj in panels.items()}
rows, donor_rows = [], []

def donor_gene_means(mat, genes_by_donor=None):
    out = {}
    for d in mat.columns:
        col = mat[d] if genes_by_donor is None else mat.loc[mat.index.isin(genes_by_donor[str(d)]), d]
        out[str(d)] = float(col.mean(skipna=True))
    return out

per_long = {m: pd.read_parquet(os.path.join(run_dir(m), "benchmark_summary.parquet")) for m in ACC_RANK}
for summary, rank_models, report, stats in (("accuracy", ACC_RANK, ACC_REPORT, ["pearson"]),
                                            ("calibration", CAL_RANK, CAL_REPORT, ["ece_pit", "disp_signed"])):
    panel = set.intersection(*(set(per_long[m]["gene"]) for m in rank_models))
    for stat in stats:
        mats = {m: per_long[m].pivot_table(index="gene", columns="test_donor", values=stat).loc[sorted(panel)]
                for m in report if stat in per_long[m].columns}
        for m, mat in mats.items():
            compare(summary, m, stat, donor_gene_means(mat), donor_gene_means(mat, shared_by_donor),
                    rows, donor_rows)
        if summary == "accuracy":  # paired delta vs the mean-only head
            for m, mat in mats.items():
                if m == REFERENCE:
                    continue
                D = mat - mats[REFERENCE]
                compare(summary, m, f"{stat}_delta_vs_{REFERENCE}", donor_gene_means(D),
                        donor_gene_means(D, shared_by_donor), rows, donor_rows)

# gene axis: (gene, slide) -> (gene, donor) weighted by n, then mean over the donor's genes
for m in AXIS_MODELS:
    pq = os.path.join(run_dir(m), "canal2_uncertainty", "pair_partial.parquet")
    pair = pd.read_parquet(pq, columns=["gene", "slide", "n_plus_all", "r_uD_lin_all", "r_partial_lin_all"])
    pair["donor"] = pair["slide"].map(lambda s: S2D.get(s, s))
    for stat, tag in (("r_uD_lin_all", "naive"), ("r_partial_lin_all", "partial")):
        d = pair[(pair["n_plus_all"] >= N_PLUS_MIN) & pair[stat].notna()]
        if d.empty:
            continue
        g = pd.DataFrame({"gene": d["gene"].to_numpy(), "donor": d["donor"].to_numpy(),
                          "w": d[stat].to_numpy() * d["n_plus_all"].to_numpy(),
                          "n": d["n_plus_all"].to_numpy()}).groupby(["gene", "donor"], as_index=False).sum()
        g["v"] = g["w"] / g["n"]
        full = g.groupby("donor")["v"].mean().to_dict()
        shared = {dn: float(gd.loc[gd["gene"].isin(shared_by_donor[dn]), "v"].mean())
                  for dn, gd in g.groupby("donor")}
        compare("gene_axis", m, tag, full, shared, rows, donor_rows)

pd.DataFrame(rows).to_parquet(os.path.join(OUT, "scores.parquet"), index=False)
pd.DataFrame(donor_rows).to_parquet(os.path.join(OUT, "scores_per_donor.parquet"), index=False)
with open(os.path.join(OUT, "panel.json"), "w") as fh:
    json.dump(dict(dataset=DS, run_id=args.run_id, input_adata=metrics["input_adata"],
                   hvg=dict(flavor="seurat_v3", n_top_genes=N_TOP, batch_key=BATCH_KEY, span=HVG_SPAN,
                            layer="raw_counts"),
                   scanpy_version=sc.__version__, n_paper_vs_all_spots_symmetric_difference=n_mismatch,
                   n_folds=len(panels), overlap_mean=float(overlap["overlap"].mean()),
                   overlap_min=float(overlap["overlap"].min()),
                   n_in_all_folds=len(in_all_folds), overlap_all_folds=len(in_all_folds) / len(G)),
              fh, indent=2)
print(f"[{_el():6.1f}s] done: {OUT}  overlap mean {overlap['overlap'].mean():.4f}, "
      f"min {overlap['overlap'].min():.4f}, all folds {len(in_all_folds) / len(G):.4f}", flush=True)
