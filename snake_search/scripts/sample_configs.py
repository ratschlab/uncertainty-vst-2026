#!/usr/bin/env python
# snake_search/scripts/sample_configs.py
#
# Sample N hyperparameter configs ONCE from the search space and write them to a
# JSON file. The search pipeline then runs exactly those configs — reproducible,
# inspectable, and stable across re-runs (re-running overwrites only with --force).
#
# Usage (the config.yaml points configs_file at configs_v2.json):
#   python snake_search/scripts/sample_configs.py --n 256 --seed 42 \
#       --out snake_search/configs_v2.json
#
# Augmenting later (256 -> 400) WITHOUT throwing away finished jobs:
#   python snake_search/scripts/sample_configs.py --n 400 --append \
#       --out snake_search/configs_v2.json
# --append keeps every existing config verbatim (same config_id, same
# hyperparameters) and only samples the additional ones, deduped against what's
# already there. So the per_config/{config_id}/result.json outputs the search has
# already computed stay valid and Snakemake reuses them — only the new config_ids
# run, then search_aggregate re-ranks over the full set. This is robust even if
# you edited SPACE in between (old configs keep their original choices; only the
# new ones explore the edited space). Re-sampling without --append is only
# reuse-safe if BOTH --seed and SPACE are unchanged.
#
# ── On batch size and learning rate ───────────────────────────────────────────
# We sample `base_lr` (the rate AT batch_size=128) and `batch_size` independently,
# then DERIVE the actual `lr` and `warmup_epochs` from a scaling rule. This keeps
# every sampled config a FAIR candidate (a large batch is paired with an
# appropriately scaled lr), so the ranking answers "is a bigger batch viable for
# GPU efficiency?" instead of penalising large batches that simply had too small
# an lr.
#
# LR_SCALING:
#   "sqrt"   lr = base_lr * sqrt(batch/128)   (DEFAULT — the right rule for Adam;
#                                              linear tends to overshoot at large
#                                              batch with adaptive optimisers)
#   "linear" lr = base_lr * (batch/128)       (the SGD+momentum rule, included
#                                              for comparison)
#   "none"   lr = base_lr                      (no scaling)
# Warmup is auto-enabled for batch>128 (a few epochs) — the standard large-batch
# stabiliser; batch==128 keeps warmup at 0 (original behaviour).

import argparse
import json
import math
import os
import random

# ── Search space (edit here) ─────────────────────────────────────────────────
SPACE = {
    "base_lr":       [1e-3, 4e-4, 1e-4],          # lr AT batch_size 128 (then scaled)
    "batch_size":    [128, 256, 512],             # GPU-efficiency axis
    "weight_decay":  [0.0, 1e-5, 1e-4, 1e-3],
    "dropout":       [0.0, 0.1, 0.2, 0.3],        # scalar, applied to all hidden layers
    "hidden_dims":   [[512, 128], [1024, 256], [1024, 512, 128], [2048, 512, 128]],
    "activation":    ["relu", "gelu"],
    "normalization": ["layernorm", "none"],       # pre-activation LayerNorm on/off
}
# Full grid size = 3*3*4*4*4*2*2 = 2304 combinations; sample a subset at random.

LR_SCALING     = "sqrt"      # "sqrt" | "linear" | "none" — see header
WARMUP_FOR_BIG = 3           # warmup epochs when batch_size > 128 (0 at 128)

# Fixed settings recorded alongside each config for provenance (not sampled).
FIXED = {
    "in_dim": 3072, "max_epochs": 25, "patience": 5,
    "seed": 42, "radius": "0.75",
    "lr_scaling": LR_SCALING, "warmup_for_big_batch": WARMUP_FOR_BIG,
}


def derive_lr_warmup(base_lr, batch_size):
    """Apply the lr-vs-batch scaling rule and pick warmup. Returns (lr, warmup)."""
    factor = batch_size / 128
    if LR_SCALING == "sqrt":
        lr = base_lr * math.sqrt(factor)
    elif LR_SCALING == "linear":
        lr = base_lr * factor
    else:                                # "none"
        lr = base_lr
    warmup = 0 if batch_size <= 128 else WARMUP_FOR_BIG
    return float(f"{lr:.6g}"), warmup    # round for clean JSON


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True, help="number of configs to sample")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    ap.add_argument("--append", action="store_true",
                    help="keep the existing configs verbatim and only ADD new ones "
                         "up to --n (the reuse-safe way to grow the search).")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing config file from scratch (discards the "
                         "old ids — finished per_config outputs no longer line up).")
    a = ap.parse_args()

    # Decide what to do with an existing file: append to it, overwrite it, or refuse.
    existing, fixed, sample_seed = [], FIXED, a.seed
    if os.path.exists(a.out):
        if a.append:
            with open(a.out) as f:
                blob = json.load(f)
            existing = blob["configs"]
            fixed, sample_seed = blob["fixed"], blob.get("sample_seed", a.seed)
            print(f"--append: loaded {len(existing)} existing configs from {a.out}")
        elif not a.force:
            raise SystemExit(
                f"{a.out} already exists. To GROW the search reuse-safely pass "
                f"--append; to rebuild it from scratch (discarding the old ids) "
                f"pass --force."
            )

    rng = random.Random(a.seed)

    # Sample WITHOUT duplicates (a config = the tuple of its sampled choices).
    # Existing configs (append mode) are kept verbatim and seed the dedup set, so
    # their ids/hyperparameters never move and new draws can't collide with them.
    seen, configs = set(), []
    for c in existing:
        seen.add(json.dumps({k: c[k] for k in SPACE}, sort_keys=True))
        configs.append(c)
    n_kept = len(configs)

    attempts = 0
    max_attempts = a.n * 200
    while len(configs) < a.n and attempts < max_attempts:
        attempts += 1
        cfg = {k: rng.choice(v) for k, v in SPACE.items()}
        key = json.dumps(cfg, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        # Derive the trained lr + warmup from the (base_lr, batch_size) pair.
        cfg["lr"], cfg["warmup_epochs"] = derive_lr_warmup(cfg["base_lr"], cfg["batch_size"])
        cfg["config_id"] = f"cfg{len(configs):03d}"
        configs.append(cfg)

    if len(configs) < a.n:
        print(f"WARNING: only {len(configs)} unique configs available "
              f"(space may be smaller than n={a.n}).")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"fixed": fixed, "sample_seed": sample_seed, "configs": configs}, f, indent=2)
    n_new = len(configs) - n_kept
    print(f"Wrote {len(configs)} configs to {a.out} "
          f"({n_kept} kept, {n_new} new; lr_scaling={LR_SCALING})")
    for i, c in enumerate(configs):
        tag = "kept" if i < n_kept else "NEW "
        print(f"  [{tag}] {c['config_id']}: hidden={c['hidden_dims']} act={c['activation']} "
              f"norm={c.get('normalization', 'layernorm')} "
              f"drop={c['dropout']} wd={c['weight_decay']} "
              f"bs={c['batch_size']} base_lr={c['base_lr']} -> lr={c['lr']} "
              f"warmup={c['warmup_epochs']}")


if __name__ == "__main__":
    main()
