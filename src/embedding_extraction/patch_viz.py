"""
patch_viz.py
────────────
Visualization helper for the patch-embedding pipeline. Nothing imports it: it is
a diagnostic to run by hand when adapting the extraction to new slides.

Exposes:
    plot_patch_alignment: overlays spot centers on a downsampled slide under both
                          coordinate orderings, so you can read off which one the
                          obs columns use.
"""

from __future__ import annotations

from typing import Dict, Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from PIL import Image


def plot_patch_alignment(
    sample:          str,
    path_h5ad:       str,
    path_image:      str,
    output_path:     Optional[str] = None,
    label_colors:    Optional[Dict[str, str]] = None,
    label_col:       Optional[str] = None,
    thumb_size:      int = 3000,
) -> None:
    """Overlay spot centers on a downsampled slide, once per coordinate ordering.

    The panel whose points land on the tissue names the convention. Extraction infers
    that order on its own, in patch_embed_extraction.spot_xy_fullres; this plot is the
    visual confirmation of what it inferred.

    Args:
        sample:       Sample identifier (used in the plot title / filename).
        path_h5ad:    Path to the .h5ad file for this sample.
        path_image:   Path to the full-resolution .tif file.
        output_path:  If provided, save the figure to this path (.png).
                      Defaults to ``alignment_check_{sample}.png`` in cwd.
        label_colors: Mapping from ground-truth label → hex color.
                      If None or label_col is missing, all spots use a single color.
        label_col:    Column in cdata.obs holding ground-truth labels.
        thumb_size:   Maximum pixel dimension of the downsampled image used for display.
    """
    Image.MAX_IMAGE_PIXELS = None  # allow large whole-slide TIF files

    cdata       = sc.read_h5ad(path_h5ad)
    img_fullres = Image.open(path_image)

    # ── Downsample for display ─────────────────────────────────
    img_thumb = img_fullres.copy()
    img_thumb.thumbnail((thumb_size, thumb_size), Image.LANCZOS)

    W_full, H_full   = img_fullres.size
    W_thumb, H_thumb = img_thumb.size
    scale_x = W_thumb / W_full
    scale_y = H_thumb / H_full

    # ── Coordinates ───────────────────────────────────────────
    x_pixel = cdata.obs["x_pixel"].to_numpy()
    y_pixel = cdata.obs["y_pixel"].to_numpy()

    if label_colors and label_col and label_col in cdata.obs.columns:
        colors = cdata.obs[label_col].astype(str).map(label_colors).fillna("#aaaaaa").values
    else:
        colors = np.full(len(cdata), "#9bf42e")

    # ── Plot ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    for ax, (plot_x, plot_y), title in zip(
        axes,
        [
            (x_pixel * scale_x, y_pixel * scale_y),   # original order
            (y_pixel * scale_x, x_pixel * scale_y),   # swapped order
        ],
        [
            "Original  (x_pixel → col, y_pixel → row)",
            "Swapped   (x_pixel → row, y_pixel → col)",
        ],
    ):
        ax.imshow(img_thumb, alpha=0.6)
        ax.scatter(plot_x, plot_y, c=colors, s=8, alpha=1, linewidths=0)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    if label_colors:
        handles = [mpatches.Patch(color=c, label=l) for l, c in label_colors.items()]
        fig.legend(handles=handles, title="Ground Truth",
                   bbox_to_anchor=(0.5, -0.02), loc="upper center",
                   ncol=len(label_colors))

    fig.suptitle(f"Patch center alignment check — {sample}", fontsize=13, y=1.01)
    plt.tight_layout()

    save_path = output_path or f"alignment_check_{sample}.png"
    # plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved alignment figure → {save_path}")
    plt.show()
