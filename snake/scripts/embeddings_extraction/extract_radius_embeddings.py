#!/usr/bin/env python
"""
extract_radius_embeddings.py
────────────────────────────
Cluster driver: extract patch embeddings across a sweep of crop radii (field-of-view
distribution-shift study). radius_factor sets the crop half-width =
round(spot_diam * radius_factor): 0.5 = tight crop (spot only), 0.75 = with neighbor
context, larger = progressively more surrounding tissue downsampled into the fixed 224
model input.

The heavy loop lives in src/embedding_extraction/patch_embed_extraction.py::run_radius_sweep,
which is slide-outer: each slide's full-res .tif is read once, then every radius re-crops,
resizes and runs the ViT, before moving to the next slide. The crop changes with the radius,
so unlike salt-and-pepper the 224 cache cannot be shared across levels, but the expensive
slide read still is. Outputs land in plain `embeddings_radius_factor_{rf}_swapxy_True/`
dirs, with the same naming and .pt format ({embeddings, obs_names, n_spots}) as the clean
0.5 / 0.75 extractions. A (slide, radius) whose .pt already exists is skipped by default, so
an already-extracted radius (e.g. the 0.5 set) is reused and a re-run resumes;
`--no-skip-existing` forces the full re-extraction.

Note on 0.75: the training embeddings live in the `..._swapxy_True_2/` dir, while this sweep
writes a plain `..._swapxy_True/` 0.75. The analysis keeps `_2` as the in-distribution
baseline; the plain 0.75 is a bonus consistency point.

Usage (submitted by snake/scripts/embeddings_extraction/launch_radius_extraction.sh):
    python -u snake/scripts/embeddings_extraction/extract_radius_embeddings.py \
        --dataset 10x_TuPro --radii 0.25 0.5 0.75 1.0 1.25 1.5
    python -u snake/scripts/embeddings_extraction/extract_radius_embeddings.py \
        --dataset TLS       --radii 0.25 0.5 0.75 1.0 1.25 1.5
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))  # parent dir of src

from src.embedding_extraction.patch_embed_extraction import (
    DatasetConfig, build_transform, load_model, run_radius_sweep,
)

# ── Per-dataset config (identical to extract_saltpepper_embeddings.py) ────────────────
# TLS keeps all 8 slides incl. KC2/LC4 to
# match the clean reference; KC2/LC4 are dropped later, at preprocessing, not at extraction.
DATASETS = {
    "10x_TuPro": dict(
        parent_dir    = "/path/to/st-data/10x_TuPro/out_all_genes/data/",
        image_subdir  = "image",
        h5ad_subdir   = "h5ad",
        meta_subdir   = "meta",
        emb_base_dir  = "/path/to/project/patch_embeddings/10x_TuPro/",
        samples       = [
            "MACEGEJ-1-1", "MACEGEJ-1-2", "MACEGEJ-2-1", "MACEGEJ-2-2",
            "MAHEFOG-1-1", "MAHEFOG-1-2",
            "MAJOFIJ-1-1", "MAJOFIJ-1-2", "MAJOFIJ-2-1", "MAJOFIJ-2-2",
            "MAKYGIW-1-1", "MAKYGIW-1-2",
            "MANOFYB-1-1", "MANOFYB-1-2",
            "MELIPIT-1-1", "MELIPIT-1-2",
            "MIDEKOG-1-1", "MIDEKOG-1-2",
        ],
    ),
    "TLS": dict(
        parent_dir    = "/path/to/project/TLS_VISIUM_USZ/",
        image_subdir  = "tif_slides",
        h5ad_subdir   = "h5ad_preprocessed",
        meta_subdir   = "h5ad_preprocessed",
        emb_base_dir  = "/path/to/project/patch_embeddings/TLS/",
        samples       = ["KC1", "KC2", "KC3", "LC1", "LC2", "LC3", "LC4", "LC5"],
    ),
}

MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/"
    "models--kaiko-ai--midnight/snapshots/adc6b15679c981cce6f9b018bbad09d16eeeda9f"
)


def emb_output_dir(ds_cfg: dict, radius_factor: float) -> str:
    """Plain per-radius dir (matches the existing 0.5 set naming). The float formats as the
    dir suffix: 0.5 -> '0.5', 1.0 -> '1.0', 0.25 -> '0.25' (same as embeddings_radius_factor_0.5_swapxy_True)."""
    return (f"{ds_cfg['emb_base_dir']}embeddings_radius_factor_{radius_factor}"
            f"_swapxy_True/")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="10x_TuPro", choices=list(DATASETS),
                    help="which dataset's slides to re-crop + extract")
    ap.add_argument("--radii", type=float, nargs="+",
                    default=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5],
                    help="crop radius_factor sweep (crop half-width = round(spot_diam * rf))")
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false",
                    help="re-extract even radii whose .pt already exist (default: skip them)")
    ap.add_argument("--model-path", default=MODEL_PATH)
    args = ap.parse_args()

    ds_cfg = DATASETS[args.dataset]
    print(f"[radius] dataset={args.dataset} ({len(ds_cfg['samples'])} slides)  "
          f"radii={args.radii}  skip_existing={args.skip_existing}")

    transform = build_transform()
    model     = load_model(args.model_path)

    # cfg carries the shared fields; the sweep ignores radius_factor / output_dir, so the
    # radii and their per-radius output dirs are passed explicitly.
    cfg = DatasetConfig(
        samples       = ds_cfg["samples"],
        parent_dir    = ds_cfg["parent_dir"],
        h5ad_subdir   = ds_cfg["h5ad_subdir"],
        image_subdir  = ds_cfg["image_subdir"],
        meta_subdir   = ds_cfg["meta_subdir"],
        output_dir    = "(swept - run_radius_sweep writes per-radius dirs)",  # unused
        radius_factor = 0.75,                                                 # unused (swept)
    )
    run_radius_sweep(
        model, cfg, transform, args.radii,
        output_dir_fn=lambda rf: emb_output_dir(ds_cfg, rf),
        skip_existing=args.skip_existing,
    )

    print(f"\nAll radii extracted for {args.dataset}.")


if __name__ == "__main__":
    main()
