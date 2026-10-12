# snake/scripts/canals/oracle_sim_precompute.py
#
# Rebuttal check: how much of the oracle's withholding gain is hindsight alone? The oracle ranks
# realized errors, so even a correct uncertainty cannot reach it. Per held-out (spot, gene), one
# measurement y* is drawn from the head's own fitted predictive (Gaussian N(mu, sigma^2),
# Poisson(mu) or NB(mu, alpha)); the prediction is mu and the uncertainty the head's own sd, so it
# is correct by construction. The withholding curves of canal 2 (lin_all and lin_pos; pooled,
# within (gene, slide), within mean bins) are then computed on |y* - mu|. Seeds repeat the whole
# simulation; only the final share is averaged across seeds, in the notebook.
#
# canal2_uncertainty_precompute.py cannot be imported, so its sparsification functions are copied
# here. --no-sim runs the same code on the real y and compares the result to that run's
# canal2_uncertainty/sparsification.parquet (summary.json, "equality_check").
#
# Output: <run-dir>/oracle_sim/seed{S}/ (or oracle_sim/real/ with --no-sim): sparsification.parquet
# (canal 2 column names), summary.json, and diag_genes.parquet (per gene, real y vs y*: zero
# fraction, mean, variance), which checks that the draw keeps the observed sparsity and level.
#
# Usage (CPU):
#   python -u snake/scripts/canals/oracle_sim_precompute.py --run-dir <run-dir> --seed 0
#   python -u snake/scripts/canals/oracle_sim_precompute.py --run-dir <run-dir> --no-sim
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
import anndata as ad

_T0 = time.time()
def _el():
    return time.time() - _T0

p = argparse.ArgumentParser()
p.add_argument("--run-dir", required=True,
               help="A classic run dir (predictions/preds_fold*.h5ad + metrics.json).")
p.add_argument("--seed", type=int, default=0, help="Seed of the simulated measurement.")
p.add_argument("--no-sim", action="store_true",
               help="Use the REAL y (no draw) and compare to canal2_uncertainty/sparsification.parquet.")
p.add_argument("--diag-only", action="store_true",
               help="Write only diag_genes.parquet (sparsity / expression level of y* vs y), then exit.")
p.add_argument("--out-dir", default=None,
               help="Default: <run-dir>/oracle_sim/seed{S}/, or <run-dir>/oracle_sim/real/ with --no-sim.")
args = p.parse_args()
assert not (args.diag_only and args.no_sim), "--diag-only compares y* to y: it needs the simulated mode"

run_dir = Path(os.path.abspath(args.run_dir))
out_dir = Path(args.out_dir or (run_dir / "oracle_sim" / ("real" if args.no_sim else f"seed{args.seed}")))
out_dir.mkdir(parents=True, exist_ok=True)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(PROJECT_ROOT)
from src.evaluation.count_metrics import count_pred_var

FRACS = np.linspace(0, 0.9, 19)
N_PLUS_MIN = 60
REGIME_TO_NBINS = {"terciles": 3, "quintiles": 5, "deciles": 10}
DT = np.float32
ROW_CHUNK = 5000

FAMILY = {"heteroscedastic": "continuous", "spot_homoscedastic": "continuous",
          "poisson_cp": "poisson", "poisson_cp_nr": "poisson",
          "negativebinomial_hetero_noanchor_cp": "nb", "negativebinomial_hetero_noanchor_cp_nr": "nb"}

SPARS_SKIPPED = []


class SparsificationPrecisionError(RuntimeError):
    """f = 0 values of the scores disagree (cumsum cancellation), as canal 2."""


with open(run_dir / "metrics.json") as f:
    M = json.load(f)
dataset, model_type = M["dataset"], M["model_type"]
assert model_type in FAMILY, f"model_type={model_type!r} not supported (supported: {sorted(FAMILY)})"
family = FAMILY[model_type]
target_transform = str(M.get("target_transform", "none"))
assert target_transform == "none", (
    f"target_transform={target_transform!r}: y > 0 is not 'detected' in a standardized space")
if family != "continuous":
    assert M.get("count_convention") == "v2_no_offset", (
        f"pre-refactor count artifact (count_convention={M.get('count_convention')!r})")

print(f"run_dir : {run_dir}", flush=True)
print(f"dataset : {dataset}  model_type: {model_type} ({family})", flush=True)
print(f"mode    : {'REAL y (equality check vs canal 2)' if args.no_sim else f'simulated y*, seed {args.seed}'}",
      flush=True)


def _dense(a):
    return a.toarray() if sp.issparse(a) else np.asarray(a)


# --- copied from canal 2 ---
def load_test_predictions(run_dir):
    pred_files = sorted((run_dir / "predictions").glob("preds_fold*.h5ad"))
    assert pred_files, f"no predictions in {run_dir / 'predictions'}"
    parts = []
    for f in pred_files:
        p = sc.read_h5ad(f, backed="r")
        parts.append(p[p.obs["split"].to_numpy() == "test"].to_memory())
    B = ad.concat(parts, axis=0, merge="same")
    assert np.unique(B.obs["adata_row"]).size == B.n_obs, "test spots overlap across folds -- non-LOPO?"
    return B


def spars_scores(err_col):
    return {"pred":    ("uncertainty", True),
            "oracle":  (err_col,       True),
            "mu_high": ("mu",          True),
            "mu_low":  ("mu",          False)}


def pooled_curves(err, values, scores, fracs):
    n = len(err)
    out = {}
    for name, (col, desc) in scores.items():
        v = values[col]
        e = err[np.argsort(-v if desc else v)]
        out[name] = np.array([e[int(f * n):].mean() for f in fracs])
    return out


def per_group_curves(keys, values, err, scores, fracs, rm_cols, agg_code, n_agg):
    out, divisors = {}, None
    gidx = np.arange(n_agg, dtype=np.float64)
    rm_keys = [keys[k] for k in rm_cols]
    for name, (col, desc) in scores.items():
        pct = pd.Series(values[col]).groupby(rm_keys, sort=False).rank(
            ascending=not desc, pct=True).to_numpy()
        key = agg_code + pct
        order = np.argsort(key)
        ks = key[order]
        assert np.all(np.diff(agg_code[order]) >= 0), "groups not contiguous after the key sort"
        cum = np.concatenate([[0.0], np.cumsum(err[order])])
        end = np.searchsorted(ks, gidx + 1.0, side="right")
        C = np.empty((n_agg, len(fracs)), dtype=np.float64)
        for q, f in enumerate(fracs):
            cut = np.searchsorted(ks, gidx + f, side="right")
            n_kept = end - cut
            with np.errstate(invalid="ignore", divide="ignore"):
                C[:, q] = np.where(n_kept > 0, (cum[end] - cum[cut]) / n_kept, np.nan)
        if divisors is None:
            divisors = C[:, 0].copy()
        else:
            if not np.allclose(C[:, 0], divisors, rtol=1e-9, equal_nan=True):
                with np.errstate(invalid="ignore", divide="ignore"):
                    dev = np.nanmax(np.abs(C[:, 0] - divisors) / np.abs(divisors))
                raise SparsificationPrecisionError(
                    f"score {name!r} starts from a different f=0 value (max relative deviation "
                    f"{dev:.3g})")
        bad = ~np.isfinite(divisors) | (divisors <= 0)
        C[bad, :] = np.nan
        C /= np.where(bad, 1.0, divisors)[:, None]
        out[name] = C
    return out, divisors


def fold_slides_then_genes(C, gene_of_group, n_of_group, n_genes):
    ok = np.isfinite(C)
    w = np.where(ok, n_of_group[:, None].astype(np.float64), 0.0)
    num = np.zeros((n_genes, C.shape[1]), dtype=np.float64)
    den = np.zeros((n_genes, C.shape[1]), dtype=np.float64)
    np.add.at(num, gene_of_group, np.where(ok, C, 0.0) * w)
    np.add.at(den, gene_of_group, w)
    with np.errstate(invalid="ignore", divide="ignore"):
        per_gene = np.where(den > 0, num / den, np.nan)
    return (per_gene,
            np.nanmean(per_gene, axis=0) if np.isfinite(per_gene).any() else np.full(C.shape[1], np.nan),
            np.nanmedian(per_gene, axis=0) if np.isfinite(per_gene).any() else np.full(C.shape[1], np.nan))


def assign_regime(pair):
    n = pair["n"].to_numpy()
    regime = np.full(len(n), None, dtype=object)
    regime[n >= N_PLUS_MIN] = "terciles"
    regime[n >= 150] = "quintiles"
    regime[n >= 500] = "deciles"
    reg = pd.DataFrame({"gene": pair["gene"].to_numpy(), "slide": pair["slide"].to_numpy(),
                        "n_plus": n, "regime": regime})
    reg["n_bins"] = reg["regime"].map(REGIME_TO_NBINS)
    return reg


def add_mu_bins(df, reg_lookup):
    out = df.merge(reg_lookup, on=["gene", "slide"], how="inner")
    out["mu_rank_pct"] = out.groupby(["gene", "slide"])["mu"].rank(pct=True)
    out["bin"] = np.minimum((out["mu_rank_pct"] * out["n_bins"]).astype(int), out["n_bins"] - 1)
    return out.drop(columns=["mu_rank_pct"])


# lin_all: canal 2 whole_panel_pass, sparsification part only
def lin_all_per_group(mu_all, y_all, u_all, slides, gene_code, slide_code):
    blocks = []
    for sl in pd.unique(slides):
        rows = np.flatnonzero(slides == sl)
        sl_code = int(slide_code[rows[0]])
        mu, y, u = mu_all[rows], y_all[rows], u_all[rows]
        n_s, n_g = mu.shape
        d = pd.DataFrame({"gene": np.tile(np.arange(n_g, dtype=np.int32), n_s),
                          "spot": np.repeat(np.arange(n_s, dtype=np.int32), n_g),
                          "mu": mu.ravel(), "uncertainty": u.ravel(),
                          "err_lin": np.abs(mu - y).ravel()})
        del mu, y, u
        n_grp = int(len(d) // max(1, d["gene"].nunique()))
        nb = (REGIME_TO_NBINS["deciles"] if n_grp >= 500 else
              REGIME_TO_NBINS["quintiles"] if n_grp >= 150 else
              REGIME_TO_NBINS["terciles"] if n_grp >= 60 else 0)
        if not nb:
            continue
        g = d.groupby(["gene"], sort=False)
        pct_mu = g["mu"].rank(pct=True, method="first").to_numpy()
        d = d.assign(bin=np.minimum((pct_mu * nb).astype(np.int16), nb - 1), slide=sl_code)
        a = d["gene"].to_numpy().astype(np.int64)
        n_a = np.bincount(a, minlength=n_g).astype(np.float64)
        err_sl = d["err_lin"].to_numpy().astype(np.float64)
        gkeys = {k: d[k].to_numpy() for k in ("gene", "slide", "bin")}
        vals = {"uncertainty": d["uncertainty"].to_numpy(), "mu": d["mu"].to_numpy(),
                "err_lin": err_sl}
        block = {}
        try:
            for tag, rm in (("gene_slide", ["gene", "slide"]),
                            ("gene_slide_bin", ["gene", "slide", "bin"])):
                block[tag] = per_group_curves(gkeys, vals, err_sl, spars_scores("err_lin"),
                                              FRACS, rm, a.astype(np.float64), n_g)
        except SparsificationPrecisionError as exc:
            SPARS_SKIPPED.append(dict(case="lin_all", raised_on=f"slide {sl}, protocol {tag}",
                                      error=str(exc)))
            print(f"[{_el():6.1f}s] (!) lin_all per-group curves SKIPPED (slide {sl}): {exc}", flush=True)
            return {}
        blocks.append((block, gene_code[np.arange(n_g)], n_a))
        del d
    stacked = {}
    if blocks:
        gene_of = np.concatenate([b[1] for b in blocks])
        n_of = np.concatenate([b[2] for b in blocks])
        for tag in ("gene_slide", "gene_slide_bin"):
            curves = {s: np.vstack([b[0][tag][0][s] for b in blocks]) for s in spars_scores("err_lin")}
            divisors = np.concatenate([b[0][tag][1] for b in blocks])
            stacked[tag] = (curves, divisors, gene_of, n_of)
    return stacked


def lin_all_pooled(mu_all, y_all, u_all):
    err = np.abs(mu_all - y_all).ravel().astype(np.float64)
    values = {"uncertainty": u_all.ravel(), "mu": mu_all.ravel(), "err_lin": err}
    return pooled_curves(err, values, spars_scores("err_lin"), FRACS), float(err.mean())


# lin_pos: canal 2 build_long_df (err_lin only), regimes, bins, curves
def lin_pos_curves(mu_all, y_all, u_all, slide_code, gene_code):
    is_pos = y_all > 0
    spot_i, gene_j = np.nonzero(is_pos)
    df = pd.DataFrame({"gene": gene_code[gene_j], "slide": slide_code[spot_i],
                       "mu": mu_all[spot_i, gene_j], "uncertainty": u_all[spot_i, gene_j],
                       "err_lin": np.abs(mu_all[spot_i, gene_j] - y_all[spot_i, gene_j])})
    n_pos = len(df)
    ok = (np.isfinite(df["uncertainty"].to_numpy()) & np.isfinite(df["mu"].to_numpy())
          & np.isfinite(df["err_lin"].to_numpy()))
    pair = df[ok].groupby(["gene", "slide"], sort=True).size().reset_index(name="n")
    reg = assign_regime(pair)
    reg_lookup = reg[reg["regime"].notna()][["gene", "slide", "n_plus", "n_bins", "regime"]]
    df_binned = add_mu_bins(df, reg_lookup)
    del df
    print(f"[{_el():6.1f}s] lin_pos: {n_pos:,} entries with y > 0; {len(df_binned):,} in "
          f"{len(reg_lookup):,} regime groups.", flush=True)

    err = df_binned["err_lin"].to_numpy().astype(np.float64)
    scores = spars_scores("err_lin")
    values = {"uncertainty": df_binned["uncertainty"].to_numpy(),
              "mu": df_binned["mu"].to_numpy(), "err_lin": err}
    keys = df_binned[["gene", "slide"]]
    a, uniq = pd.factorize(pd.MultiIndex.from_frame(keys), sort=False)
    n_agg = len(uniq)
    gene_of = keys["gene"].to_numpy()[np.unique(a, return_index=True)[1]]
    n_of = np.bincount(a, minlength=n_agg).astype(np.float64)
    a = a.astype(np.float64)
    gkeys = {k: df_binned[k].to_numpy() for k in ("gene", "slide", "bin")}
    per_group = {}
    try:
        for tag, rm in (("gene_slide", ["gene", "slide"]),
                        ("gene_slide_bin", ["gene", "slide", "bin"])):
            curves, divisors = per_group_curves(gkeys, values, err, scores, FRACS, rm, a, n_agg)
            per_group[tag] = (curves, divisors, gene_of, n_of)
    except SparsificationPrecisionError as exc:
        per_group = {}
        SPARS_SKIPPED.append(dict(case="lin_pos", raised_on=f"protocol {tag}", error=str(exc)))
        print(f"[{_el():6.1f}s] (!) lin_pos per-group curves SKIPPED ({tag}): {exc}", flush=True)
    pooled = pooled_curves(err, values, scores, FRACS)
    return pooled, per_group, n_pos, float(err.mean())


# --- main ---
print(f"[{_el():6.1f}s] loading test predictions ...", flush=True)
B = load_test_predictions(run_dir)
print(f"[{_el():6.1f}s] loaded {B.n_obs:,} test spots x {B.n_vars:,} genes.", flush=True)

# mu and u as canal 2
MU = _dense(B.X).astype(DT)
if family == "continuous":
    VAR = _dense(B.layers["pred_var"]).astype(DT)
    ALPHA = None
elif family == "poisson":
    VAR = count_pred_var(MU)
    ALPHA = None
else:
    ALPHA = _dense(B.layers["dispersion"]).astype(DT)
    VAR = count_pred_var(MU, dispersion=ALPHA)
assert int((VAR < 0).sum()) == 0, "negative predictive variance"
U = np.sqrt(VAR)
Y_REAL = _dense(B.layers["y"]).astype(DT)
SLIDES = B.obs["slide"].to_numpy().astype(str)
GENES = np.asarray(B.var_names)
del B

# one draw y* per entry from the head's own predictive
if args.no_sim:
    Y = Y_REAL
else:
    rng = np.random.default_rng(args.seed)
    Y = np.empty_like(MU)
    for r0 in range(0, MU.shape[0], ROW_CHUNK):
        r1 = min(r0 + ROW_CHUNK, MU.shape[0])
        m = MU[r0:r1].astype(np.float64)
        if family == "continuous":
            Y[r0:r1] = m + np.sqrt(VAR[r0:r1].astype(np.float64)) * rng.standard_normal(m.shape)
        elif family == "poisson":
            Y[r0:r1] = rng.poisson(m)
        else:
            # NB as gamma-Poisson: lambda ~ Gamma(1/alpha, alpha*mu)
            al = ALPHA[r0:r1].astype(np.float64)
            assert (al > 0).all(), "NB dispersion must be > 0"
            Y[r0:r1] = rng.poisson(rng.gamma(1.0 / al, al * m))
    print(f"[{_el():6.1f}s] drew y* ({family}, seed {args.seed}): y* > 0 on {int((Y > 0).sum()):,} "
          f"entries vs real y > 0 on {int((Y_REAL > 0).sum()):,} (of {Y.size:,}).", flush=True)
del VAR, ALPHA
n_pos_real = int((Y_REAL > 0).sum())

# per-gene diagnostic: sparsity and level of y* vs y
if not args.no_sim:
    diag = {"gene": GENES, "mean_mu": MU.mean(axis=0, dtype=np.float64)}
    for tag, A in (("real", Y_REAL), ("sim", Y)):
        det = A > 0
        n_det = det.sum(axis=0)
        diag[f"undetected_{tag}"] = 1.0 - n_det / A.shape[0]
        diag[f"mean_{tag}"] = A.mean(axis=0, dtype=np.float64)
        diag[f"var_{tag}"] = A.var(axis=0, dtype=np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            diag[f"mean_detected_{tag}"] = np.where(det, A, 0).sum(axis=0, dtype=np.float64) / n_det
        del det
    pd.DataFrame(diag).to_parquet(out_dir / "diag_genes.parquet", index=False)
    print(f"[{_el():6.1f}s] diag: undetected fraction real {1 - n_pos_real / Y.size:.3f} vs y* "
          f"{float((Y <= 0).mean()):.3f} -> {out_dir / 'diag_genes.parquet'}", flush=True)
    del Y_REAL
    if args.diag_only:
        sys.exit(0)

GENE_CODE, GENE_NAMES = pd.factorize(GENES, sort=True)
SLIDE_CODE, SLIDE_NAMES = pd.factorize(SLIDES, sort=True)

print(f"[{_el():6.1f}s] lin_all per-group curves ...", flush=True)
PG_LIN_ALL = lin_all_per_group(MU, Y, U, SLIDES, GENE_CODE, SLIDE_CODE)
POOLED_LIN_ALL, MEAN_ERR_LIN_ALL = lin_all_pooled(MU, Y, U)
print(f"[{_el():6.1f}s] lin_pos curves ...", flush=True)
POOLED_LIN_POS, PG_LIN_POS, n_pos_used, MEAN_ERR_LIN_POS = lin_pos_curves(MU, Y, U, SLIDE_CODE, GENE_CODE)
n_entries = int(MU.size)
del MU, Y, U

# column names as canal 2
sparsification = {"frac": FRACS}
for case, pooled, per_group in (("lin_all", POOLED_LIN_ALL, PG_LIN_ALL),
                                ("lin_pos", POOLED_LIN_POS, PG_LIN_POS)):
    for name, c in pooled.items():
        sparsification[f"curve_{name}_pooled_{case}"] = c
    for tag, (curves, divisors, gene_of, n_of) in per_group.items():
        for name, C in curves.items():
            _, mean_c, med_c = fold_slides_then_genes(C, gene_of, n_of, len(GENE_NAMES))
            sparsification[f"curve_{name}_{tag}_{case}"] = mean_c
            sparsification[f"curve_{name}_{tag}_{case}__median"] = med_c
SPARS = pd.DataFrame(sparsification)

# --no-sim: compare to canal 2's output for this run
equality_check = None
if args.no_sim:
    ref_path = run_dir / "canal2_uncertainty" / "sparsification.parquet"
    if not ref_path.is_file():
        equality_check = dict(reference=str(ref_path), found=False)
        print(f"(!) no reference at {ref_path}; equality check not done.", flush=True)
    else:
        ref = pd.read_parquet(ref_path)
        missing = [c for c in SPARS.columns if c not in ref.columns]
        diffs = {c: float(np.nanmax(np.abs(SPARS[c].to_numpy() - ref[c].to_numpy())))
                 for c in SPARS.columns if c in ref.columns}
        nan_mismatch = [c for c in diffs if not np.array_equal(np.isnan(SPARS[c].to_numpy()),
                                                               np.isnan(ref[c].to_numpy()))]
        max_diff = max(diffs.values()) if diffs else float("nan")
        equality_check = dict(reference=str(ref_path), found=True, n_columns=len(diffs),
                              missing_in_reference=missing, nan_pattern_mismatch=nan_mismatch,
                              max_abs_diff=max_diff, exact=(max_diff == 0.0 and not nan_mismatch))
        print(f"[{_el():6.1f}s] EQUALITY CHECK vs canal 2: {len(diffs)} columns, max |diff| = "
              f"{max_diff:.3g}, NaN-pattern mismatches: {len(nan_mismatch)}, missing in reference: "
              f"{len(missing)}  -> {'EXACT' if equality_check['exact'] else '(!) NOT EXACT'}",
              flush=True)
        if max_diff > 1e-12 or nan_mismatch:
            worst = sorted(diffs.items(), key=lambda kv: -kv[1])[:5]
            print(f"    largest differences: {worst}", flush=True)

summary = dict(dataset=dataset, model_type=model_type, family=family, run_dir=str(run_dir),
               simulated=not args.no_sim, seed=(None if args.no_sim else args.seed),
               detected_rule=("real y > 0" if args.no_sim else "y* > 0 (literal rule on the draw)"),
               n_entries=n_entries, n_positive_real=n_pos_real, n_positive_used=n_pos_used,
               mean_err=dict(lin_all=MEAN_ERR_LIN_ALL, lin_pos_sparsification_population=MEAN_ERR_LIN_POS),
               sparsification_skipped=SPARS_SKIPPED, equality_check=equality_check)

SPARS.to_parquet(out_dir / "sparsification.parquet", index=False)
with open(out_dir / "summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=float)
print(f"[{_el():6.1f}s] done -> {out_dir}\n{json.dumps(summary, indent=2, default=float)}", flush=True)
