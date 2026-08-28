#!/usr/bin/env python
"""
extract_blur_embeddings.py
──────────────────────────
Cluster driver: extract patch embeddings from slides degraded with a whole-slide Gaussian
blur (slide-quality distribution-shift study). The heavy loop is
src/embedding_extraction/patch_embed_extraction.py::run_blur_sweep, which is slide-outer:
each slide's full-res .tif is read once, then every blur level blurs a copy of it,
re-crops, resizes and runs the ViT, before moving to the next slide.

The blur is applied to the whole slide in full-res pixel space, before any cropping, with
sigma_px = blur_factor * spot_diameter_fullres (per slide), so a given blur_factor smears
the same physical extent on slides of different resolution. `blur_factor=0.0` skips the
filter entirely, a clean control identical to the clean extraction path.

Outputs land in isolated `..._blurry_bf{bf}/` dirs, so the clean training embeddings are
never touched; the naming and the .pt format ({embeddings, obs_names, n_spots}) match the
clean extractions.

LUNG note: the merged LUNG_CANCER dataset has no image/meta dir of its own — its slides live
in the two per-subtype source trees. So `LUNG_CANCER_LUAD` (P10/P15/P16/P24/P25, 22 slides)
and `LUNG_CANCER_LUSC` (P11/P17/P19, 14 slides) are two entries here, and BOTH write into the
merged `patch_embeddings/LUNG_CANCER/...` output dir. Slide names are disjoint, so the two
jobs together fill the 36-slide merged dir directly — no symlinking step afterwards.

Usage (submitted by snake/scripts/embeddings_extraction/launch_blur_extraction.sh):
    python -u snake/scripts/embeddings_extraction/extract_blur_embeddings.py \
        --dataset LUNG_CANCER_LUAD --blur-factors 0.0 0.005 0.01 0.02 0.05
    python -u snake/scripts/embeddings_extraction/extract_blur_embeddings.py \
        --dataset LUNG_CANCER_LUSC --blur-factors 0.0 0.005 0.01 0.02 0.05
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))  # parent dir of src

from src.embedding_extraction.patch_embed_extraction import (
    DatasetConfig, build_transform, load_model, run_blur_sweep,
)

# ── Per-dataset config: the same paths and geometry as the clean extraction ───────────
# All use radius_factor 0.75. TLS keeps all 8
# slides incl. KC2/LC4 to match the clean reference; those two are dropped later, at
# preprocessing, not at extraction.
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


def emb_output_dir(ds_cfg: dict, blur_factor: float) -> str:
    """Isolated output dir per blur level: the dataset's own emb_base_dir plus a
    `blurry_bf{bf}` key, which keeps the clean training embeddings safe. The float formats
    straight into the suffix: 0.0 -> '0.0', 0.005 -> '0.005'."""
    return (f"{ds_cfg['emb_base_dir']}embeddings_radius_factor_{ds_cfg['radius_factor']}"
            f"_swapxy_True_blurry_bf{blur_factor}/")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="10x_TuPro", choices=list(DATASETS),
                    help="which dataset's slides to blur + extract")
    ap.add_argument("--blur-factors", type=float, nargs="+",
                    default=[0.0, 0.005, 0.01, 0.02, 0.05],
                    help="sigma_px = blur_factor * spot_diameter_fullres, per level "
                         "(0.0 = clean control, the filter is skipped)")
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false",
                    help="re-extract even levels whose .pt already exist (default: skip them)")
    ap.add_argument("--model-path", default=MODEL_PATH)
    args = ap.parse_args()

    ds_cfg = DATASETS[args.dataset]
    print(f"[blur] dataset={args.dataset} ({len(ds_cfg['samples'])} slides)  "
          f"blur_factors={args.blur_factors}  skip_existing={args.skip_existing}")
    print(f"[blur] output base: {emb_output_dir(ds_cfg, args.blur_factors[-1])}")

    transform = build_transform()
    model     = load_model(args.model_path)

    # Slide-outer sweep: each slide is read once, then every blur level blurs a copy of it.
    # cfg carries the shared fields; the sweep ignores blur_factor / output_dir, so the
    # levels and their per-level output dirs are passed explicitly.
    cfg = DatasetConfig(
        samples       = ds_cfg["samples"],
        parent_dir    = ds_cfg["parent_dir"],
        h5ad_subdir   = ds_cfg["h5ad_subdir"],
        image_subdir  = ds_cfg["image_subdir"],
        meta_subdir   = ds_cfg["meta_subdir"],
        output_dir    = "(swept - run_blur_sweep writes per-level dirs)",  # unused
        radius_factor = ds_cfg["radius_factor"],
    )
    run_blur_sweep(
        model, cfg, transform, args.blur_factors,
        output_dir_fn=lambda bf: emb_output_dir(ds_cfg, bf),
        skip_existing=args.skip_existing,
    )

    print(f"\nAll blur levels extracted for {args.dataset}.")


if __name__ == "__main__":
    main()
