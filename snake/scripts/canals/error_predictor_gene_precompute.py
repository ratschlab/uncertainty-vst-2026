# snake/scripts/canals/error_predictor_gene_precompute.py
#
# Rebuttal check, the REPORTED version: does the predicted uncertainty add to the predicted mean
# for withholding the worst predictions of a held-out donor? (error_predictor_precompute.py is an
# earlier pooled version, superseded by this one.)
#   A: per-gene random forest  |err| ~ pct(mu)
#   B: per-gene random forest  |err| ~ pct(mu) + pct(u)
# pct = percentile within (gene, slide); the target is the raw |err|, so the forest estimates the
# expected error, the right score for the mean error of what is kept. Cases: lin_all (all
# entries) and lin_pos (y > 0), as in Table 4.
#
# One job = one LOPO fold j. Forests are tuned and fitted on the fold-j model's validation donor
# and scored on its held-out donor, which is touched last. The leaf size is chosen per (case,
# forest) on an 80/20 split of the validation spots, on a random subset of genes, then refitted
# per gene on all validation spots.
#
# Score: Table 4's withholding curves (highest score withheld first, 19 fractions, normalized at
# f = 0) for A, B and the references u, mu_high, mu_low, oracle. A tied block that a cut splits
# is withheld proportionally (the expectation over orderings of the tie); without ties this is
# canal 2's curve. Spearman(prediction, err) is kept as a second score. The cascade over genes
# and donors is done in the notebook.
#
# Output (<run-dir>/error_predictor_gene/): fold{j}.parquet (per case, gene, slide: n, mean_err,
# curves c_{score}_{q}, rho_A, rho_B, r_mu, r_u) and fold{j}.json (donors, tuning, leaf sizes).
#
# Usage (CPU):
#   python -u snake/scripts/canals/error_predictor_gene_precompute.py --run-dir <run-dir> --fold 0
import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
from joblib import Parallel, delayed
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestRegressor

_T0 = time.time()
def _el():
    return time.time() - _T0

p = argparse.ArgumentParser()
p.add_argument("--run-dir", required=True, help="A classic run dir (predictions/, metrics.json, splits.json).")
p.add_argument("--fold", type=int, required=True, help="LOPO fold j (= held-out donor).")
p.add_argument("--cases", nargs="+", default=["lin_all", "lin_pos"])
p.add_argument("--leaves", type=int, nargs="+", default=[10, 30, 100, 300], help="min_samples_leaf grid")
p.add_argument("--n-tune-genes", type=int, default=300)
p.add_argument("--sel-frac", type=float, default=0.2, help="share of V's spots used to select the leaf size")
p.add_argument("--n-estimators", type=int, default=50)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--n-jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
p.add_argument("--out-dir", default=None)
args = p.parse_args()

run_dir = os.path.abspath(args.run_dir)
out_dir = args.out_dir or os.path.join(run_dir, "error_predictor_gene")
os.makedirs(out_dir, exist_ok=True)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(PROJECT_ROOT)
from src.evaluation.count_metrics import count_pred_var

N_PLUS_MIN = 60
FRACS = np.linspace(0, 0.9, 19)
EPS = 1e-12
DT = np.float32
warnings.filterwarnings("ignore", message="Mean of empty slice")

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
assert str(M.get("target_transform", "none")) == "none", "standardized target: y > 0 is not 'detected'"
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
ERR = np.asfortranarray(np.abs(MU.astype(np.float64) - Y.astype(np.float64)))

SLIDE_ROWS = {}
for sl in pd.unique(SLIDES):
    rows = np.flatnonzero(SLIDES == sl)
    assert len(set(SPLIT[rows])) == 1, f"slide {sl} is split across val and test"
    if len(rows) < N_PLUS_MIN:
        print(f"  slide {sl}: {len(rows)} spots < {N_PLUS_MIN}, excluded", flush=True)
        continue
    SLIDE_ROWS[sl] = rows
VAL_SLIDES = {sl: r for sl, r in SLIDE_ROWS.items() if SPLIT[r[0]] == "val"}
TEST_SLIDES = {sl: r for sl, r in SLIDE_ROWS.items() if SPLIT[r[0]] == "test"}
print(f"[{_el():6.1f}s] V: {sum(map(len, VAL_SLIDES.values())):,} spots on {len(VAL_SLIDES)} slide(s); "
      f"A: {sum(map(len, TEST_SLIDES.values())):,} spots on {len(TEST_SLIDES)} slide(s); "
      f"{G:,} genes.", flush=True)


# --- ranks, correlation, withholding curve ---
def pop_rank(x, pop):
    """Per column, percentile over the population entries (ties averaged); NaN elsewhere."""
    r = rankdata(np.where(pop, x, np.inf), axis=0)
    return np.where(pop, r / np.maximum(pop.sum(axis=0), 1), np.nan)


def col_corr_pop(a, b, pop):
    """Column-wise Pearson over the population; on ranks, the per-(gene, slide) Spearman."""
    a = np.where(pop, a, np.nan)
    b = np.where(pop, b, np.nan)
    a = a - np.nanmean(a, axis=0)
    b = b - np.nanmean(b, axis=0)
    den = np.sqrt(np.nansum(a * a, axis=0) * np.nansum(b * b, axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > EPS, np.nansum(a * b, axis=0) / den, np.nan)


def curves(score, err, pop):
    """One slide block (n_rows, g) -> (g, 19) withholding curves, not normalized. floor(f * n)
    entries are withheld; a tied block split by the cut is withheld proportionally."""
    n_rows, g = score.shape
    bad = ~np.isfinite(np.where(pop, score, 0.0)).all(axis=0)
    S = np.where(pop & ~bad, score, -np.inf).astype(np.float64)
    E = np.where(pop, err, 0.0)
    order = np.argsort(-S, axis=0, kind="stable")
    Ss = np.take_along_axis(S, order, axis=0)
    Es = np.take_along_axis(E, order, axis=0)
    # tie blocks numbered over all columns; each entry replaced by its block mean
    new = np.ones(Ss.shape, dtype=bool)
    new[1:] = Ss[1:] != Ss[:-1]
    block = np.cumsum(new, axis=0) - 1 + np.arange(g) * n_rows
    sums = np.bincount(block.ravel(), weights=Es.ravel(), minlength=n_rows * g)
    cnts = np.bincount(block.ravel(), minlength=n_rows * g)
    cum = np.vstack([np.zeros((1, g)), np.cumsum((sums / np.maximum(cnts, 1))[block], axis=0)])
    n = pop.sum(axis=0)
    cols = np.arange(g)
    C = np.empty((g, len(FRACS)))
    with np.errstate(divide="ignore", invalid="ignore"):
        for q, f in enumerate(FRACS):
            n_rm = np.floor(f * n + 1e-9).astype(np.int64)
            C[:, q] = (cum[n, cols] - cum[n_rm, cols]) / (n - n_rm)
    C[bad | (n < N_PLUS_MIN)] = np.nan
    return C


def score_slides(slides, preds, genes):
    """One row per (gene, slide): n, mean_err, the normalized curve of every score, and Spearman."""
    parts, off = [], 0
    for sl, rows in slides.items():
        ix = np.ix_(rows, genes)
        pop, err, pm, pu = POP[ix], ERR[ix], P_MU[ix], P_U[ix]
        scores = {k: v[off:off + len(rows)] for k, v in preds.items()}
        scores.update(u=pu, mu_high=pm, mu_low=-pm, oracle=err)
        off += len(rows)
        n = pop.sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            mean_err = np.where(pop, err, 0.0).sum(axis=0) / n
        d = dict(gene=GENES[genes], slide=sl, n=n, mean_err=mean_err)
        for k, s in scores.items():
            with np.errstate(divide="ignore", invalid="ignore"):
                C = curves(s, err, pop) / mean_err[:, None]
            for q in range(len(FRACS)):
                d[f"c_{k}_{q:02d}"] = C[:, q]
        r_err = pop_rank(err, pop)
        for k in preds:
            d[f"rho_{k}"] = col_corr_pop(pop_rank(scores[k], pop), r_err, pop)
        d["r_mu"] = col_corr_pop(pm, r_err, pop)
        d["r_u"] = col_corr_pop(pu, r_err, pop)
        parts.append(pd.DataFrame(d))
    return pd.concat(parts, ignore_index=True)


def pooled_share(tab, score):
    """Table 4 cascade: slides n-weighted per gene, mean over genes, AUSE, share vs the oracle."""
    t = tab[(tab["n"] >= N_PLUS_MIN) & tab[f"c_{score}_00"].notna()]

    def ause(s):
        cols = [f"c_{s}_{q:02d}" for q in range(len(FRACS))]
        per_gene = t[cols].mul(t["n"], axis=0).groupby(t["gene"]).sum().div(
            t.groupby("gene")["n"].sum(), axis=0)
        return float(per_gene.mean(axis=0).mean())

    return (1 - ause(score)) / (1 - ause("oracle"))


def forest_predictions(genes, leaf, with_u, fit_rows, slides):
    """Per gene: fit on the population entries of fit_rows, predict those of `slides` (NaN if not fitted)."""
    pred_rows = np.concatenate(list(slides.values()))
    feats = [P_MU, P_U] if with_u else [P_MU]

    def one(g):
        out = np.full(len(pred_rows), np.nan)
        r = fit_rows[POP[fit_rows, g]]
        if len(r) < N_PLUS_MIN:
            return out
        rf = RandomForestRegressor(n_estimators=args.n_estimators, min_samples_leaf=leaf,
                                   max_features=None, n_jobs=1, random_state=args.seed)
        rf.fit(np.column_stack([F[r, g] for F in feats]), ERR[r, g])
        m = POP[pred_rows, g]
        if m.any():
            out[m] = rf.predict(np.column_stack([F[pred_rows[m], g] for F in feats]))
        return out

    return np.column_stack(Parallel(n_jobs=args.n_jobs, prefer="threads")(delayed(one)(g) for g in genes))


# --- per case: percentiles, tuning and final fit on the validation donor, scoring on the held-out donor ---
rng = np.random.default_rng(args.seed)
# 80/20 split of the validation spots per slide, shared by both cases and forests
V_FIT_ROWS, V_SEL_SLIDES = [], {}
for sl, rows in VAL_SLIDES.items():
    sh = rng.permutation(rows)
    k = int(round(args.sel_frac * len(rows)))
    V_SEL_SLIDES[sl] = np.sort(sh[:k])
    V_FIT_ROWS.append(sh[k:])
V_FIT_ROWS = np.sort(np.concatenate(V_FIT_ROWS))
V_ALL_ROWS = np.concatenate(list(VAL_SLIDES.values()))
TUNE_GENES = np.sort(rng.choice(G, size=min(args.n_tune_genes, G), replace=False))
ALL_GENES = np.arange(G)

TABS, TUNING, BEST, CASE_INFO = [], [], {}, {}
for case in args.cases:
    POP = np.asfortranarray(np.ones((N, G), dtype=bool) if case == "lin_all" else (Y > 0))
    P_MU = np.full((N, G), np.nan, dtype=DT, order="F")
    P_U = np.full((N, G), np.nan, dtype=DT, order="F")
    for sl, rows in SLIDE_ROWS.items():
        P_MU[rows] = pop_rank(MU[rows], POP[rows])
        P_U[rows] = pop_rank(U[rows], POP[rows])
    print(f"[{_el():6.1f}s] {case}: percentiles done, population {POP.mean():.1%} of entries.", flush=True)

    # tuning: leaf size by the share on the 20%
    BEST[case] = {}
    for name, with_u in (("A", False), ("B", True)):
        best = None
        for leaf in args.leaves:
            t0 = time.time()
            P = forest_predictions(TUNE_GENES, leaf, with_u, V_FIT_ROWS, V_SEL_SLIDES)
            s = pooled_share(score_slides(V_SEL_SLIDES, {name: P}, TUNE_GENES), name)
            TUNING.append(dict(case=case, forest=name, leaf=leaf, share_sel=s, seconds=time.time() - t0))
            print(f"[{_el():6.1f}s] {case} tune {name} leaf {leaf:>4d}: share on V 20% = {s:+.4f}", flush=True)
            if best is None or (np.isfinite(s) and (not np.isfinite(best[1]) or s > best[1])):
                best = (leaf, s)
        BEST[case][name] = dict(leaf=best[0], share_sel=best[1])

    # final fit per gene on all validation spots, scored on the held-out donor
    t0 = time.time()
    PA = forest_predictions(ALL_GENES, BEST[case]["A"]["leaf"], False, V_ALL_ROWS, TEST_SLIDES)
    PB = forest_predictions(ALL_GENES, BEST[case]["B"]["leaf"], True, V_ALL_ROWS, TEST_SLIDES)
    tab = score_slides(TEST_SLIDES, {"A": PA, "B": PB}, ALL_GENES)
    n_fit = POP[V_ALL_ROWS].sum(axis=0)
    tab["fitted"] = np.tile(n_fit >= N_PLUS_MIN, len(TEST_SLIDES))
    tab.insert(0, "case", case)
    TABS.append(tab)
    shares = {k: pooled_share(tab[tab["fitted"]], k) for k in ("A", "B", "u", "mu_high", "mu_low")}
    CASE_INFO[case] = dict(n_genes_fitted=int((n_fit >= N_PLUS_MIN).sum()),
                           population_frac=float(POP.mean()), share_this_donor=shares,
                           seconds_final=time.time() - t0)
    print(f"[{_el():6.1f}s] {case} donor {TEST_DONOR}: share A {shares['A']:+.4f}  B {shares['B']:+.4f}  "
          f"(u {shares['u']:+.4f}, mu_high {shares['mu_high']:+.4f}, mu_low {shares['mu_low']:+.4f}); "
          f"{CASE_INFO[case]['n_genes_fitted']:,}/{G:,} genes fitted", flush=True)
    del POP, P_MU, P_U, PA, PB

pd.concat(TABS, ignore_index=True).to_parquet(os.path.join(out_dir, f"fold{args.fold}.parquet"), index=False)
summary = dict(dataset=dataset, model_type=model_type, family=family, run_dir=run_dir,
               fold=args.fold, test_donor=TEST_DONOR, val_donor=VAL_DONOR, cases=args.cases,
               n_val_spots=int(len(V_ALL_ROWS)), n_val_fit_spots=int(len(V_FIT_ROWS)),
               n_val_sel_spots=int(sum(map(len, V_SEL_SLIDES.values()))),
               n_test_spots=int(sum(map(len, TEST_SLIDES.values()))), n_genes=int(G),
               n_tune_genes=int(len(TUNE_GENES)), leaves=args.leaves, n_estimators=args.n_estimators,
               seed=args.seed, best=BEST, tuning=TUNING, per_case=CASE_INFO,
               seconds=time.time() - _T0)
with open(os.path.join(out_dir, f"fold{args.fold}.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"[{_el():6.1f}s] done -> {out_dir}", flush=True)
