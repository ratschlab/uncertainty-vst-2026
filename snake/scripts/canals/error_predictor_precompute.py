# snake/scripts/canals/error_predictor_precompute.py
#
# EARLIER POOLED VERSION, superseded by error_predictor_gene_precompute.py and not the reported
# result: one forest over all genes, scored by rank correlation. Kept for the record.
#
# Rebuttal check: does the predicted uncertainty add to the predicted mean for predicting the
# error of a held-out donor?
#   A: random forest  rank(|err|) ~ rank(mu)
#   B: random forest  rank(|err|) ~ rank(mu) + rank(u)
# mu, u and err are canal 2's all-entries case (lin_all), ranked within (gene, slide), n >= 60.
#
# One job = one LOPO fold j. preds_fold{j}.h5ad holds the predictions of the fold-j model on its
# validation donor (fits the forests and their random search) and on its held-out donor (scored
# only), neither of which it was trained on. The held-out donor enters no step before scoring.
# A and B share the same search configurations. For Poisson and gene-homoscedastic heads, u is a
# monotone function of mu or constant within a group, so B should equal A (exactly for Poisson).
#
# Output (<run-dir>/error_predictor/): fold{j}.parquet (gene, slide, n, rho_A, rho_B, r_mu, r_u)
# and fold{j}.json (donors, search trials, chosen configurations).
#
# Usage (CPU):
#   python -u snake/scripts/canals/error_predictor_precompute.py --run-dir <run-dir> --fold 0
#   (or POOLED=1 with launch_error_predictor_precompute.sh)
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
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestRegressor

_T0 = time.time()
def _el():
    return time.time() - _T0

p = argparse.ArgumentParser()
p.add_argument("--run-dir", required=True, help="A classic run dir (predictions/, metrics.json, splits.json).")
p.add_argument("--fold", type=int, required=True, help="LOPO fold j (= held-out donor).")
p.add_argument("--n-fit", type=int, default=2_000_000, help="V entries for the final fit")
p.add_argument("--n-search", type=int, default=500_000, help="V entries (fit genes) per search trial")
p.add_argument("--n-trials", type=int, default=20)
p.add_argument("--n-select-genes", type=int, default=300, help="select genes scored per search trial")
p.add_argument("--n-estimators", type=int, default=100)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--n-jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
p.add_argument("--out-dir", default=None)
args = p.parse_args()

run_dir = os.path.abspath(args.run_dir)
out_dir = args.out_dir or os.path.join(run_dir, "error_predictor")
os.makedirs(out_dir, exist_ok=True)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(PROJECT_ROOT)
from src.evaluation.count_metrics import count_pred_var

N_PLUS_MIN = 60
EPS = 1e-12
DT = np.float32

# family taxonomy and (mu, u) as canal2_uncertainty_precompute.py
CONTINUOUS_MODELS = {"msemeanonly", "homoscedastic", "heteroscedastic", "spot_homoscedastic",
                     "gaussian_lowrank", "gaussian_fullcov"}
POISSON_MODELS    = {"poisson", "poisson_cp", "poisson_raw", "poisson_cp_nr"}
NB_MODELS         = {"negativebinomial", "negativebinomial_hetero",
                     "negativebinomial_hetero_noanchor", "negativebinomial_hetero_noanchor_cp",
                     "negativebinomial_hetero_noanchor_raw",
                     "negativebinomial_spotdisp_cp", "negativebinomial_genedisp_cp",
                     "negativebinomial_hetero_noanchor_cp_nr",
                     "negativebinomial_spotdisp_cp_nr", "negativebinomial_genedisp_cp_nr",
                     "STI_nb_cp", "STI_nb_raw"}
NO_UNCERTAINTY_MODELS = {"msemeanonly"}


def model_family(model_type):
    if model_type in CONTINUOUS_MODELS:
        return "continuous"
    if model_type in POISSON_MODELS:
        return "poisson"
    if model_type in NB_MODELS:
        return "nb"
    raise ValueError(f"unknown model family for model_type={model_type!r}")

def _dense(a):
    return a.toarray() if sp.issparse(a) else np.asarray(a)


with open(Path(run_dir) / "metrics.json") as f:
    M = json.load(f)
dataset = M["dataset"]
model_type = M["model_type"]
assert model_type not in NO_UNCERTAINTY_MODELS, f"{model_type}: no uncertainty head, test does not apply"
family = model_family(model_type)

with open(Path(run_dir) / "splits.json") as f:
    SPLITS = json.load(f)
assert SPLITS.get("cv_strategy", "lopo_next_val") == "lopo_next_val", \
    f"splits.json: cv_strategy={SPLITS.get('cv_strategy')!r}, this test assumes LOPO"
FOLD = next(fd for fd in SPLITS["folds"] if fd["fold"] == args.fold)
assert len(FOLD["inner_splits"]) == 1, "one inner split expected (lopo_next_val)"
INNER = FOLD["inner_splits"][0]
TEST_DONOR = ",".join(map(str, FOLD["test_donor"]))
VAL_DONOR = ",".join(map(str, INNER["val_donors"]))
print(f"run_dir : {run_dir}\ndataset : {dataset}  model_type: {model_type} ({family})\n"
      f"fold {args.fold}: fit on val donor {VAL_DONOR}, score on test donor {TEST_DONOR}", flush=True)


def assert_count_convention(B, M):
    conv = M.get("count_convention") or B.uns.get("count_convention")
    if conv != "v2_no_offset":
        raise ValueError(f"{run_dir}: pre-refactor count artifact (count_convention={conv!r}, "
                         f"expected 'v2_no_offset'). Re-run the training; legacy reading was "
                         f"removed on purpose.")
    return conv


def _sigma(variance):
    neg = int((variance < 0).sum())
    if neg:
        print(f"WARNING: {neg:,} of {variance.size:,} variance entries are NEGATIVE. sqrt makes them "
              f"NaN and they will be dropped from every statistic -- this should be impossible, "
              f"investigate the artifact before using these numbers.", flush=True)
    return np.sqrt(variance)


def predictions_and_uncertainty(B, family):
    X = _dense(B.X).astype(DT)
    if family == "continuous":
        variance = _dense(B.layers["pred_var"]).astype(DT)
    elif family == "poisson":
        variance = count_pred_var(X)
    elif family == "nb":
        alpha = _dense(B.layers["dispersion"]).astype(DT)
        variance = count_pred_var(X, dispersion=alpha)
    else:
        raise ValueError(family)
    return X, _sigma(variance)


# fold j predictions: validation donor (split "val") and held-out donor (split "test")
print(f"[{_el():6.1f}s] loading preds_fold{args.fold}.h5ad ...", flush=True)
B = sc.read_h5ad(Path(run_dir) / "predictions" / f"preds_fold{args.fold}.h5ad")
if family != "continuous":
    assert_count_convention(B, M)
MU, U = predictions_and_uncertainty(B, family)
Y = _dense(B.layers["y"]).astype(DT)
SPLIT = B.obs["split"].to_numpy().astype(str)
SLIDES = B.obs["slide"].to_numpy().astype(str)
ADATA_ROWS = B.obs["adata_row"].to_numpy().astype(np.int64)
GENES = np.asarray(B.var_names)
del B
N, G = MU.shape
assert np.isfinite(U).all(), f"{int((~np.isfinite(U)).sum()):,} non-finite uncertainty entries"
for role, idx in (("val", INNER["val_idx"]), ("test", FOLD["test_idx"])):
    assert set(ADATA_ROWS[SPLIT == role].tolist()) == set(idx), \
        f"{role} spots of preds_fold{args.fold}.h5ad differ from splits.json"
print(f"[{_el():6.1f}s] {int((SPLIT == 'val').sum()):,} val + {int((SPLIT == 'test').sum()):,} test "
      f"spots x {G:,} genes.", flush=True)

# within-(gene, slide) percentiles: one column of one slide's row block
P_MU = np.empty((N, G), dtype=DT)
P_U = np.empty((N, G), dtype=DT)
P_ERR = np.empty((N, G), dtype=DT)
SLIDE_ROWS = {}
for sl in pd.unique(SLIDES):
    rows = np.flatnonzero(SLIDES == sl)
    n_s = len(rows)
    assert len(set(SPLIT[rows])) == 1, f"slide {sl} is split across val and test"
    if n_s < N_PLUS_MIN:
        print(f"  slide {sl}: {n_s} spots < {N_PLUS_MIN}, excluded", flush=True)
        continue
    SLIDE_ROWS[sl] = rows
    P_MU[rows] = rankdata(MU[rows], axis=0) / n_s
    P_U[rows] = rankdata(U[rows], axis=0) / n_s
    P_ERR[rows] = rankdata(np.abs(MU[rows] - Y[rows]), axis=0) / n_s
del MU, U, Y
print(f"[{_el():6.1f}s] percentiles done ({len(SLIDE_ROWS)} slides).", flush=True)


def col_corr(a, b):
    """Column-wise Pearson of two (n, G) blocks; on ranks this is the per-(gene, slide) Spearman."""
    a = a.astype(np.float64) - a.mean(axis=0, dtype=np.float64)
    b = b.astype(np.float64) - b.mean(axis=0, dtype=np.float64)
    den = np.sqrt((a * a).sum(axis=0) * (b * b).sum(axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > EPS, (a * b).sum(axis=0) / den, np.nan)


def features(rows, genes, with_u):
    """(len(rows)*len(genes), 1 or 2) feature matrix, spot-major (reshapes to (n, g))."""
    cols = [P_MU[np.ix_(rows, genes)].ravel()]
    if with_u:
        cols.append(P_U[np.ix_(rows, genes)].ravel())
    return np.column_stack(cols)


def score(rf, with_u, split, genes):
    """Per (gene, slide) Spearman(predicted, err) over the slides of `split`."""
    parts = []
    for sl, rows in SLIDE_ROWS.items():
        if SPLIT[rows[0]] != split:
            continue
        pred = rf.predict(features(rows, genes, with_u)).reshape(len(rows), len(genes))
        parts.append(pd.DataFrame({"gene": genes, "slide": sl, "n": len(rows),
                                   "rho": col_corr(rankdata(pred, axis=0), P_ERR[np.ix_(rows, genes)])}))
    return pd.concat(parts, ignore_index=True)


def donor_value(frame, col):
    """One donor's frame -> (gene) n-weighted mean over its slides -> mean over genes."""
    d = frame[(frame["n"] >= N_PLUS_MIN) & frame[col].notna()]
    g = d.assign(w=d[col] * d["n"]).groupby("gene")[["w", "n"]].sum()
    return float((g["w"] / g["n"]).mean())


def sample_entries(rows, genes, n, rng):
    """n random (spot, gene) entries of rows x genes -> (X with both features, y)."""
    flat = rng.choice(len(rows) * len(genes), size=min(n, len(rows) * len(genes)), replace=False)
    r, g = rows[flat // len(genes)], genes[flat % len(genes)]
    return np.column_stack([P_MU[r, g], P_U[r, g]]), P_ERR[r, g]


# validation donor: genes split for the search (2/3 fit, 1/3 select), then the final fit sample
rng = np.random.default_rng(args.seed)
val_rows = np.concatenate([r for r in SLIDE_ROWS.values() if SPLIT[r[0]] == "val"])
perm = rng.permutation(G)
FIT_GENES = np.sort(perm[: (2 * G) // 3])
SELECT_GENES = np.sort(rng.choice(perm[(2 * G) // 3:], size=min(args.n_select_genes, G - (2 * G) // 3),
                                  replace=False))
X_S, Y_S = sample_entries(val_rows, FIT_GENES, args.n_search, rng)
X_FIT, Y_FIT = sample_entries(val_rows, np.arange(G), args.n_fit, rng)
print(f"[{_el():6.1f}s] V: {len(val_rows):,} spots; search fit {len(Y_S):,} entries on "
      f"{len(FIT_GENES)} genes, select on {len(SELECT_GENES)} genes; final fit {len(Y_FIT):,} "
      f"entries.", flush=True)

# random search, same configurations for A and B
CONFIGS = [dict(min_samples_leaf=int(round(10 ** rng.uniform(np.log10(20), np.log10(20000)))),
                max_depth=[8, 12, 16, None][rng.integers(4)],
                max_samples=[0.1, 0.3, 1.0][rng.integers(3)],
                max_features=[1, 2][rng.integers(2)])
           for _ in range(args.n_trials)]


def make_rf(cfg, with_u):
    return RandomForestRegressor(n_estimators=args.n_estimators,
                                 min_samples_leaf=cfg["min_samples_leaf"], max_depth=cfg["max_depth"],
                                 max_samples=cfg["max_samples"],
                                 max_features=(cfg["max_features"] if with_u else 1),
                                 n_jobs=args.n_jobs, random_state=args.seed)


TRIALS, BEST, FINAL = [], {}, {}
for name, with_u in (("A", False), ("B", True)):
    nf = 2 if with_u else 1
    best = None
    for t, cfg in enumerate(CONFIGS):
        t0 = time.time()
        rf = make_rf(cfg, with_u).fit(X_S[:, :nf], Y_S)
        v = donor_value(score(rf, with_u, "val", SELECT_GENES), "rho")
        TRIALS.append(dict(model=name, trial=t, **cfg, select_score=v, seconds=time.time() - t0))
        print(f"[{_el():6.1f}s] {name} trial {t:2d} {cfg} -> select {v:.4f}", flush=True)
        if best is None or not np.isfinite(best[1]) or v > best[1]:
            best = (cfg, v)
    BEST[name] = dict(config=best[0], select_score=best[1])
    t0 = time.time()
    FINAL[name] = make_rf(best[0], with_u).fit(X_FIT[:, :nf], Y_FIT)
    print(f"[{_el():6.1f}s] {name} refit on {len(Y_FIT):,} V entries with {best[0]} "
          f"({time.time() - t0:.0f}s)", flush=True)
del X_S, Y_S, X_FIT, Y_FIT

# score on the held-out donor, all genes
ALL_GENES = np.arange(G)
fa = score(FINAL["A"], False, "test", ALL_GENES).rename(columns={"rho": "rho_A"})
fb = score(FINAL["B"], True, "test", ALL_GENES)
assert (fa[["gene", "slide"]].to_numpy() == fb[["gene", "slide"]].to_numpy()).all()
out = fa.assign(rho_B=fb["rho"].to_numpy())
test_blocks = [rows for rows in SLIDE_ROWS.values() if SPLIT[rows[0]] == "test"]
out["r_mu"] = np.concatenate([col_corr(P_MU[r], P_ERR[r]) for r in test_blocks])
out["r_u"] = np.concatenate([col_corr(P_U[r], P_ERR[r]) for r in test_blocks])
out["gene"] = GENES[out["gene"].to_numpy()]
out.to_parquet(os.path.join(out_dir, f"fold{args.fold}.parquet"), index=False)

summary = dict(dataset=dataset, model_type=model_type, family=family, run_dir=run_dir,
               fold=args.fold, test_donor=TEST_DONOR, val_donor=VAL_DONOR, case="lin_all",
               n_val_spots=int(len(val_rows)), n_test_spots=int(sum(len(r) for r in test_blocks)),
               n_genes=int(G), n_fit=int(min(args.n_fit, len(val_rows) * G)),
               n_search=int(min(args.n_search, len(val_rows) * len(FIT_GENES))),
               n_fit_genes=int(len(FIT_GENES)), n_select_genes=int(len(SELECT_GENES)),
               n_estimators=args.n_estimators, seed=args.seed, best=BEST, trials=TRIALS,
               donor_value=dict(rho_A=donor_value(out, "rho_A"), rho_B=donor_value(out, "rho_B"),
                                r_mu=donor_value(out, "r_mu"), r_u=donor_value(out, "r_u")),
               seconds=time.time() - _T0)
with open(os.path.join(out_dir, f"fold{args.fold}.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"[{_el():6.1f}s] donor {TEST_DONOR}: A {summary['donor_value']['rho_A']:.4f}  "
      f"B {summary['donor_value']['rho_B']:.4f}  (r_mu {summary['donor_value']['r_mu']:.4f}, "
      f"r_u {summary['donor_value']['r_u']:.4f}) -> {out_dir}", flush=True)
