#!/usr/bin/env python
"""
extract_saltpepper_embeddings.py
────────────────────────────────
Cluster driver: extract patch embeddings from slides corrupted with salt-and-pepper noise
(slide-quality distribution-shift study, the counterpart of the Gaussian-blur extraction).
Extraction is slow, hence a SLURM job rather than an interactive run.

The noise is applied to the 224x224 model input (after resize/center-crop, before
Normalize), not to the full-res slide, so the corruption density is identical across
resolutions and datasets: the input is always 224x224. Whole-pixel specks, side in
[speck_min, speck_max] px (1..1 = monopixel salt-and-pepper; 2..4 = small dust blobs).
The noise is drawn per (seed, spot index), so a run is reproducible. `amount=0.0` is a
clean control with no noise.

There is no skip-existing here, unlike the blur and radius drivers: run_saltpepper_sweep
re-extracts every (slide, amount) it is given, so a re-run recomputes everything.

All the reusable logic lives in src/embedding_extraction/patch_embed_extraction.py; this
file is only the per-dataset config plus the amount/variant sweep loop.

LUNG note: the merged LUNG_CANCER dataset has no image/meta dir of its own — its slides
live in the two per-subtype source trees. So `LUNG_CANCER_LUAD` (P10/P15/P16/P24/P25, 22
slides) and `LUNG_CANCER_LUSC` (P11/P17/P19, 14 slides) are two entries here, and BOTH
write into the merged `patch_embeddings/LUNG_CANCER/...` output dir. Slide names are
disjoint, so the two jobs together fill the 36-slide merged dir directly — no symlinking
step afterwards.

Usage (submitted by snake/scripts/embeddings_extraction/launch_saltpepper_extraction.sh):
    python -u snake/scripts/embeddings_extraction/extract_saltpepper_embeddings.py \
        --dataset 10x_TuPro --amounts 0.01 0.05 0.1 0.2 --speck-min 1 --speck-max 1 --seed 0
    # TLS, dust blobs:
    python -u snake/scripts/embeddings_extraction/extract_saltpepper_embeddings.py \
        --dataset TLS --amounts 0.01 0.05 0.1 0.2 --speck-min 2 --speck-max 4 --seed 0
    # merged lung, both subtypes (same output dir):
    python -u snake/scripts/embeddings_extraction/extract_saltpepper_embeddings.py \
        --dataset LUNG_CANCER_LUAD --amounts 0.01 0.05 0.1 0.2 --speck-min 1 --speck-max 1 --seed 0
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))  # parent dir of src

from src.embedding_extraction.patch_embed_extraction import (
    DatasetConfig, build_transform, load_model, run_saltpepper_sweep,
)

# ── Per-dataset config: the same paths and geometry as the clean extraction ──────────
# Both use radius_factor 0.75. TLS keeps all 8
# slides incl. KC2/LC4 to match the clean reference and the blur experiment; those two are
# dropped later, at preprocessing, not at extraction.
DATASETS = {
    "10x_TuPro": dict(
        radius_factor = 0.75,
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
        radius_factor = 0.75,
        parent_dir    = "/path/to/project/TLS_VISIUM_USZ/",
        image_subdir  = "tif_slides",
        h5ad_subdir   = "h5ad_preprocessed",
        meta_subdir   = "h5ad_preprocessed",
        emb_base_dir  = "/path/to/project/patch_embeddings/TLS/",
        samples       = ["KC1", "KC2", "KC3", "LC1", "LC2", "LC3", "LC4", "LC5"],
    ),
    # ── Merged lung: two source trees, ONE shared output dir (see the LUNG note above) ──
    "LUNG_CANCER_LUAD": dict(
        radius_factor = 0.75,
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
    "LUNG_CANCER_LUSC": dict(
        radius_factor = 0.75,
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
}

MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/"
    "models--kaiko-ai--midnight/snapshots/adc6b15679c981cce6f9b018bbad09d16eeeda9f"
)


def emb_output_dir(ds_cfg: dict, amount: float, speck_min: int, speck_max: int) -> str:
    """Isolated output dir per (dataset, amount, speck size) — the dataset's own emb_base_dir
    + a `saltpepper_a{amount}_sp{min}-{max}` key keeps the clean training embeddings safe AND
    separates the monopixel vs dust variants (e.g. sp1-1 vs sp2-4)."""
    return (f"{ds_cfg['emb_base_dir']}embeddings_radius_factor_{ds_cfg['radius_factor']}"
            f"_swapxy_True_saltpepper_a{amount}_sp{speck_min}-{speck_max}/")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="10x_TuPro", choices=list(DATASETS),
                    help="which dataset's slides to corrupt + extract")
    ap.add_argument("--amounts", type=float, nargs="+",
                    default=[0.01, 0.05, 0.1, 0.2],
                    help="fraction of the 224x224 model-input pixels corrupted, per level "
                         "(0.0 = clean control, dropped by default — pass it to re-add)")
    ap.add_argument("--salt-ratio", type=float, default=0.5,
                    help="fraction of corrupted specks turned white (rest black)")
    ap.add_argument("--speck-min", type=int, default=1,
                    help="smallest speck side in px (1 = monopixel)")
    ap.add_argument("--speck-max", type=int, default=1,
                    help="largest speck side in px (e.g. 4 for 2..4 dust blobs)")
    ap.add_argument("--seed", type=int, default=0,
                    help="base seed; noise is drawn per (seed, spot index)")
    ap.add_argument("--model-path", default=MODEL_PATH)
    args = ap.parse_args()

    ds_cfg = DATASETS[args.dataset]
    print(f"[saltpepper] dataset={args.dataset} ({len(ds_cfg['samples'])} slides)  "
          f"amounts={args.amounts}  specks={args.speck_min}-{args.speck_max}px  "
          f"salt_ratio={args.salt_ratio}  seed={args.seed}")

    transform = build_transform()
    model     = load_model(args.model_path)

    # Slide-outer sweep: each slide is read and resized into the 224 cache once, then every
    # amount only re-runs the noise and the ViT, so no slide is read or resized twice.
    # cfg carries the shared fields; the sweep ignores sp_amount / output_dir, so the
    # amounts and their per-amount output dirs are passed explicitly.
    cfg = DatasetConfig(
        samples       = ds_cfg["samples"],
        parent_dir    = ds_cfg["parent_dir"],
        h5ad_subdir   = ds_cfg["h5ad_subdir"],
        image_subdir  = ds_cfg["image_subdir"],
        meta_subdir   = ds_cfg["meta_subdir"],
        output_dir    = "(swept - run_saltpepper_sweep writes per-amount dirs)",  # unused
        radius_factor = ds_cfg["radius_factor"],
        sp_salt_ratio = args.salt_ratio,
        sp_speck_min  = args.speck_min,
        sp_speck_max  = args.speck_max,
        sp_seed       = args.seed,
    )
    run_saltpepper_sweep(
        model, cfg, transform, args.amounts,
        output_dir_fn=lambda a: emb_output_dir(ds_cfg, a, args.speck_min, args.speck_max),
    )

    print(f"\nAll salt-and-pepper levels extracted for {args.dataset}.")


if __name__ == "__main__":
    main()
