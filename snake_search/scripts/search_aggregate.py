#!/usr/bin/env python
# snake_search/scripts/search_aggregate.py
#
# Gather all per-CONFIG result JSONs (each already holds its per-fold val Pearson)
# into a ranked table.
#
# Selection is on VALIDATION Pearson (donor-grouped CV; SELECTION only, never a
# reported generalization number). Config score = mean over folds of the per-fold
# DONOR-WEIGHTED slide-aware val Pearson (search_train computes per fold:
# r(slide,gene) -> mean over a donor's slides -> mean over donors -> mean over
# genes). We also report the across-fold spread (the noise floor for comparing
# configs) and a fold-by-fold rank table so you can judge whether the top configs
# stay on top across folds (stability). The per-config atoms.parquet next to each
# result.json lets you re-aggregate any other way offline (no re-run).
#
# Decision gate: compare the BAND WIDTH (best - worst config
# score) to the NOISE FLOOR (typical across-fold std). If band <= a few x noise
# and the top ranking is stable -> architecture is in the noise: pick the winner
# (or the cheaper equivalent) and report the band width as the optimism bound.
#
# Outputs:
#   ranking.csv   — one row per config, sorted best-first by mean val Pearson,
#                   with mean/std over folds, per-fold val columns, per-fold rank
#                   columns, mean train/load time, and the config hyperparameters
#   best.json     — the winning config (for the classic pipeline to adopt) + the
#                   band-width / noise-floor summary that justifies the choice

import argparse, json, glob, os
import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--results-glob", required=True, help="glob for per-config JSONs")
p.add_argument("--out-csv", required=True)
p.add_argument("--out-best", required=True)
a = p.parse_args()

rows = []
for fp in glob.glob(a.results_glob):
    with open(fp) as f:
        rows.append(json.load(f))
if not rows:
    raise SystemExit(f"no result files matched {a.results_glob}")

# Each json = one config with its per-fold val Pearson list. Explode the folds into
# a wide (config x fold) val table so we can rank per fold (stability) and average.
n_folds = max(len(r.get("per_fold_val_pearson", [])) for r in rows)
val_wide = pd.DataFrame(
    {r["config_id"]: pd.Series(r.get("per_fold_val_pearson", []),
                               index=range(len(r.get("per_fold_val_pearson", []))))
     for r in rows}
).T                                            # index = config_id, columns = fold
val_wide = val_wide.reindex(columns=range(n_folds))

summary = pd.DataFrame(index=val_wide.index)
summary["mean_val"]        = val_wide.mean(axis=1, skipna=True)
summary["std_val"]         = val_wide.std(axis=1, skipna=True)   # across folds
summary["n_folds_done"]    = val_wide.notna().sum(axis=1)
# timing + config hyperparameters, straight off each per-config json
meta = pd.DataFrame(rows).set_index("config_id")
summary["mean_train_time"] = meta["train_sec"]
summary["mean_load_time"]  = meta["load_sec"]

# per-fold RANK of each config (1 = best config for that fold) — stability check.
rank_wide = val_wide.rank(axis=0, ascending=False, method="min")
rank_wide.columns = [f"rank_fold{c}" for c in rank_wide.columns]

cfg_cols = meta["config"].apply(pd.Series)

table = summary.join(cfg_cols).join(rank_wide).join(
    val_wide.add_prefix("val_fold")
)
table = table.sort_values("mean_val", ascending=False)

os.makedirs(os.path.dirname(a.out_csv) or ".", exist_ok=True)
table.to_csv(a.out_csv)

# Epoch-ceiling rate: fraction of (config,fold) runs that hit
# max_epochs. If high, early stopping rarely bound -> raise max_epochs and re-run.
all_ceiling = [bool(b) for r in rows for b in r.get("epoch_ceiling_hits", [])]
ceiling_frac = float(np.mean(all_ceiling)) if all_ceiling else float("nan")

# ── decision-gate summary: band width vs noise floor ──────────────────────────
band_width  = float(table["mean_val"].max() - table["mean_val"].min())
noise_floor = float(table["std_val"].median())   # typical across-fold spread
# top-5 stability: how often the overall top-5 configs also rank top-5 per fold.
top5 = list(table.index[:5])
stab = []
for c in rank_wide.columns:
    in_top5 = rank_wide.loc[top5, c] <= 5
    stab.append(float(in_top5.mean()))
top5_stability = float(np.mean(stab)) if stab else float("nan")

def _json_default(o):
    if isinstance(o, np.integer):  return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.ndarray):  return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

best_id = table.index[0]
best = {
    "config_id": best_id,
    "selection_metric": "mean_over_folds_of_donor_weighted_slide_aware_val_pearson",
    "mean_val":  float(table.loc[best_id, "mean_val"]),
    "std_val":   float(table.loc[best_id, "std_val"]),
    "config":    {k: (list(v) if isinstance(v, (list, np.ndarray)) else v)
                  for k, v in cfg_cols.loc[best_id].items()},
    "n_configs": int(len(table)),
    "n_folds":   int(n_folds),
    "decision_gate": {
        "band_width_best_minus_worst": band_width,
        "noise_floor_median_across_fold_std": noise_floor,
        "band_over_noise": (band_width / noise_floor if noise_floor else float("inf")),
        "top5_stability_frac": top5_stability,   # 1.0 = top-5 always top-5 per fold
        "epoch_ceiling_frac": ceiling_frac,      # frac of runs that hit max_epochs
        "note": "if band_width is small vs noise_floor and top5_stability is high, "
                "architecture is in the noise -> pick winner, report band_width as "
                "the selection-optimism bound; else do nested selection. "
                "If epoch_ceiling_frac is high, raise max_epochs and re-run.",
    },
    "mean_train_time_sec": float(table.loc[best_id, "mean_train_time"]),
    "mean_load_time_sec":  float(table.loc[best_id, "mean_load_time"]),
    "timing_note": "train_time EXCLUDES data load (adata read + GPU upload, done once per config).",
}
with open(a.out_best, "w") as f:
    json.dump(best, f, indent=2, default=_json_default)

print(f"Ranked {len(table)} configs over {n_folds} donor-grouped folds (select on val).")
print(f"Best: {best_id}  mean_val={best['mean_val']:.4f}±{best['std_val']:.4f}  "
      f"config={best['config']}")
print(f"Band width={band_width:.4f}  noise floor (median across-fold std)={noise_floor:.4f}  "
      f"band/noise={best['decision_gate']['band_over_noise']:.2f}  "
      f"top5 stability={top5_stability:.2f}  epoch-ceiling frac={ceiling_frac:.2f}")
print(f"Wrote {a.out_csv} and {a.out_best}")
