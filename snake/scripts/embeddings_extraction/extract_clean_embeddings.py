#!/usr/bin/env python
"""
extract_clean_embeddings.py
───────────────────────────
Cluster driver for the CLEAN patch embeddings, the ones every training run consumes.
This is step one of the pipeline: it turns each slide's full-res .tif into one embedding
vector per spot, which `src/data/load_and_preprocess.py` then pairs with the expression
matrix. Run it before anything else.

The heavy loop lives in src/embedding_extraction/patch_embed_extraction.py::run_extraction:
per slide, read the .tif and the .h5ad, crop a square patch of half-width
round(spot_diam_fullres * radius_factor) around each spot, resize to the 224 model input,
and run the ViT. No degradation is applied: blur_factor and sp_amount stay 0, which is what
separates this driver from its blur / salt-and-pepper / radius siblings.

Writes one `{sample}.pt` per slide holding {embeddings, obs_names, n_spots}, into
    {emb_base_dir}embeddings_radius_factor_{radius}_swapxy_{swap_xy}{suffix}/
`--suffix` names a second extraction of the same dataset. 10x_TuPro and TLS train on
`_2`, which is why their `datasets.{ds}.emb_subdir` in snake/config/config.yaml ends
`_swapxy_True_2/`; LUNG_CANCER trains on the unsuffixed dir. Match the config or
preprocessing will not find the embeddings.

A slide whose .pt already exists is skipped, so a re-run resumes; `--no-skip-existing`
forces a full re-extraction.

The coordinate order is inferred per slide by spot_xy_fullres, from obsm["spatial"], so
no dataset declares it. src/embedding_extraction/patch_viz.py plots a patch over the
slide if you want to see the alignment.

Usage (submitted by snake/scripts/embeddings_extraction/launch_clean_extraction.sh):
    python -u snake/scripts/embeddings_extraction/extract_clean_embeddings.py \
        --dataset TLS --radius 0.75 --suffix _2
    python -u snake/scripts/embeddings_extraction/extract_clean_embeddings.py \
        --dataset LUNG_CANCER_LUAD --radius 0.75
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[3]))  # parent dir of src

from src.embedding_extraction.patch_embed_extraction import (
    DatasetConfig, build_transform, load_model, run_extraction,
)

# ── Per-dataset config ────────────────────────────────────────────────────────────────
# TLS keeps all 8 slides here;
# KC2 and LC4 are dropped later, at preprocessing, not at extraction. LUAD and LUSC are
# extracted separately because their source trees differ, but both write into the merged
# LUNG_CANCER embedding dir, which is what the LUNG_CANCER dataset in config.yaml reads.
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
    "LUNG_CANCER_LUSC": dict(
        parent_dir    = "/path/to/st-data/LUNG_CANCER_LUSC/out_all_genes/data/",
        image_subdir  = "image",
        h5ad_subdir   = "h5ad",
        meta_subdir   = "meta",
        emb_base_dir  = "/path/to/project/patch_embeddings/LUNG_CANCER/",
        samples       = [
            "P11-B1", "P11-B2", "P11-T1", "P11-T2", "P11-T3", "P11-T4",
            "P17-B1", "P17-B2", "P17-T1", "P17-T2",
            "P19-B1", "P19-B2", "P19-T1", "P19-T2",
        ],
    ),
    "LUNG_CANCER_LUAD": dict(
        parent_dir    = "/path/to/st-data/LUNG_CANCER_LUAD/out_all_genes/data/",
        image_subdir  = "image",
        h5ad_subdir   = "h5ad",
        meta_subdir   = "meta",
        emb_base_dir  = "/path/to/project/patch_embeddings/LUNG_CANCER/",
        samples       = [
            "P10-B1", "P10-B2", "P10-T1", "P10-T2", "P10-T3", "P10-T4",
            "P15-B1", "P15-B2", "P15-T1", "P15-T2",
            "P16-B1", "P16-B2", "P16-T1", "P16-T2",
            "P24-B1", "P24-B2", "P24-T1", "P24-T2",
            "P25-B1", "P25-B2", "P25-T1", "P25-T2",
        ],
    ),
}

MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/"
    "models--kaiko-ai--midnight/snapshots/adc6b15679c981cce6f9b018bbad09d16eeeda9f"
)


def emb_output_dir(ds_cfg: dict, radius_factor: float, suffix: str) -> str:
    """The dir this extraction writes into. The float formats as the dir suffix:
    0.5 -> '0.5', 0.75 -> '0.75'. Must match datasets.{ds}.emb_subdir in config.yaml."""
    return (f"{ds_cfg['emb_base_dir']}embeddings_radius_factor_{radius_factor}"
            f"_swapxy_True{suffix}/")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="TLS", choices=list(DATASETS),
                    help="which dataset's slides to extract")
    ap.add_argument("--radius", type=float, default=0.75,
                    help="crop radius_factor (crop half-width = round(spot_diam * radius))")
    ap.add_argument("--suffix", default="",
                    help="appended to the output dir name, e.g. _2 for a second extraction. "
                         "Must match datasets.{ds}.emb_subdir in snake/config/config.yaml")
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false",
                    help="re-extract even slides whose .pt already exists (default: skip them)")
    ap.add_argument("--model-path", default=MODEL_PATH)
    args = ap.parse_args()

    ds_cfg  = DATASETS[args.dataset]
    out_dir = emb_output_dir(ds_cfg, args.radius, args.suffix)
    print(f"[clean] dataset={args.dataset} ({len(ds_cfg['samples'])} slides)  "
          f"radius={args.radius}  out={out_dir}  skip_existing={args.skip_existing}")

    # run_extraction walks cfg.samples unconditionally, so the resume is a filter on that
    # list rather than a flag: drop the slides whose .pt is already there.
    samples = ds_cfg["samples"]
    if args.skip_existing:
        done    = [s for s in samples if Path(out_dir, f"{s}.pt").exists()]
        samples = [s for s in samples if s not in done]
        if done:
            print(f"  skip (already on disk): {done}")
        if not samples:
            print("  nothing to do")
            return

    transform = build_transform()
    model     = load_model(args.model_path)

    cfg = DatasetConfig(
        samples       = samples,
        parent_dir    = ds_cfg["parent_dir"],
        h5ad_subdir   = ds_cfg["h5ad_subdir"],
        image_subdir  = ds_cfg["image_subdir"],
        meta_subdir   = ds_cfg["meta_subdir"],
        output_dir    = out_dir,
        radius_factor = args.radius,
        # blur_factor / sp_amount left at 0: this is the clean extraction.
    )

    run_extraction(model, cfg, transform)


if __name__ == "__main__":
    main()
