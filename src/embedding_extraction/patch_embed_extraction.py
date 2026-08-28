"""
patch_embed_extraction.py
─────────────────────────
Extracts patch-level embeddings from Visium whole-slide images with a ViT
(kaiko-ai midnight). One embedding per spot, in adata.obs order.

Typical usage
─────────────
    from src.embedding_extraction.patch_embed_extraction import (
        DatasetConfig, build_transform, load_model, run_extraction,
    )

    cfg = DatasetConfig(
        samples        = ["KC1", "KC2"],
        parent_dir     = "/path/to/dataset/",
        h5ad_subdir    = "h5ad_preprocessed",
        image_subdir   = "tif_slides",
        meta_subdir    = "h5ad_preprocessed",
        output_dir     = "/path/to/output/",
        radius_factor  = 0.5,   # 0.5 → tight crop ; 0.75 → 50 % larger crop with neighbor context
    )

    transform = build_transform()
    model     = load_model("/path/to/model")
    run_extraction(model, cfg, transform)
"""

from __future__ import annotations

import gc
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import scanpy as sc
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from tqdm.auto import tqdm

# ──────────────────────────────────────────────────────────────
# Configuration dataclass
# ──────────────────────────────────────────────────────────────

@dataclass
class DatasetConfig:
    """All dataset-specific parameters in one place.

    Args:
        samples:        List of sample identifiers.
        parent_dir:     Root directory that holds the dataset sub-directories.
        h5ad_subdir:    Sub-directory (relative to parent_dir) containing .h5ad files.
        image_subdir:   Sub-directory containing .tif files.
        meta_subdir:    Sub-directory containing .json metadata files.
        output_dir:     Directory where .pt embedding files will be saved.
        radius_factor:  Crop half-width = round(spot_diam_fullres * radius_factor).
                        0.5 gives a tight crop (1× the spot diameter), 0.75 a loose
                        crop (1.5×) that takes in neighboring tissue.
        blur_factor:    Degradation study. If >0, the whole slide is Gaussian-blurred
                        once, in full-res pixel space and before any cropping, with
                        σ_px = blur_factor * spot_diam_fullres. σ is tied to the spot
                        diameter, so the same physical extent is smeared on slides of
                        different resolution. 0.0 skips the filter.
        sp_amount:      Degradation study. If >0, salt-and-pepper noise is applied to
                        the 224×224 model input, after resize and center-crop and
                        before Normalize, not to the full-res slide. That keeps the
                        corruption density identical across slides of different
                        resolution. A fraction `sp_amount` of whole RGB pixels is set
                        to pure white (salt) or black (pepper). 0.0 skips the noise.
        sp_salt_ratio:  Of the corrupted pixels, the fraction turned white (255); the
                        rest are black (0).
        sp_speck_min:   Smallest speck side in pixels, inclusive. Each corrupted
                        "speck" is a square whose side is drawn uniformly in
                        [sp_speck_min, sp_speck_max]. (1, 1) gives monopixel
                        salt-and-pepper, (2, 4) small dust blobs.
        sp_speck_max:   Largest speck side in pixels, inclusive. See sp_speck_min.
        sp_seed:        Base seed. Each patch draws its noise from a per-(seed, spot
                        index) generator, so the corruption is reproducible and does
                        not depend on batch order or worker count.
        batch_size:     DataLoader batch size.
        num_workers:    DataLoader workers (defaults to SLURM_CPUS_PER_TASK if set).
    """

    samples:       List[str]
    parent_dir:    str
    h5ad_subdir:   str
    image_subdir:  str
    meta_subdir:   str
    output_dir:    str
    radius_factor: float = 0.75
    blur_factor:   float = 0.0
    sp_amount:     float = 0.0
    sp_salt_ratio: float = 0.5
    sp_speck_min:  int   = 1
    sp_speck_max:  int   = 1
    sp_seed:       int   = 0
    batch_size:    int   = 128
    num_workers:   int   = field(
        default_factory=lambda: int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    )

    # ── Derived path helpers ───────────────────────────────────

    def h5ad_path(self, sample: str) -> str:
        return str(Path(self.parent_dir) / self.h5ad_subdir / f"{sample}.h5ad")

    def image_path(self, sample: str) -> str:
        return str(Path(self.parent_dir) / self.image_subdir / f"{sample}.tif")

    def meta_path(self, sample: str) -> str:
        return str(Path(self.parent_dir) / self.meta_subdir / f"{sample}.json")

    def output_path(self, sample: str) -> str:
        return str(Path(self.output_dir) / f"{sample}.pt")


# ──────────────────────────────────────────────────────────────
# Image transform
# ──────────────────────────────────────────────────────────────

def build_transform(image_size: int = 224) -> v2.Compose:
    """Standard transform for kaiko-ai midnight: resize → center-crop → normalize."""
    return v2.Compose([
        v2.Resize(image_size),
        v2.CenterCrop(image_size),
        v2.ToTensor(),
        v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])


# ──────────────────────────────────────────────────────────────
# Salt-and-pepper noise (slide-quality study)
# ──────────────────────────────────────────────────────────────

def apply_salt_pepper(
    arr:        np.ndarray,
    amount:     float,
    salt_ratio: float = 0.5,
    rng:        np.random.Generator | None = None,
    speck_min:  int = 1,
    speck_max:  int = 1,
) -> np.ndarray:
    """Classic salt-and-pepper corruption of a (H, W, 3) uint8 image, in place.

    A fraction ``amount`` of pixels is corrupted with square "specks" whose side is
    drawn uniformly in [speck_min, speck_max] pixels: (1, 1) gives monopixel
    salt-and-pepper, (2, 4) small dust blobs. Each speck is set to pure white (salt,
    255) or black (pepper, 0) on all three RGB channels at once, and ``salt_ratio`` of
    the specks are salt. An ``amount`` <= 0, or one too small to reach a single pixel,
    returns arr untouched; above that an ``rng`` (a np.random.Generator) is required
    and its absence raises.

    The speck count targets an expected corrupted area of amount·H·W. Specks may
    overlap, so the realised fraction sits slightly below the target for blobs.
    """
    if amount <= 0:
        return arr
    H, W   = arr.shape[:2]
    target = int(round(amount * H * W))                      # target corrupted pixels
    if target == 0:
        return arr
    if rng is None:
        raise ValueError("apply_salt_pepper needs an rng for a non-zero amount")

    mean_area = float(np.mean([s * s for s in range(speck_min, speck_max + 1)]))
    n_specks  = max(1, int(round(target / mean_area)))
    sizes     = rng.integers(speck_min, speck_max + 1, size=n_specks)  # side per speck
    is_salt   = rng.random(n_specks) < salt_ratio

    # Vectorise per distinct speck size (only speck_max-speck_min+1 iterations, e.g. 3
    # for 2..4 dust): place all specks of that size at once via broadcast indexing.
    for s in np.unique(sizes):
        m  = sizes == s
        n  = int(m.sum())
        y0 = rng.integers(0, H - s + 1, size=n)
        x0 = rng.integers(0, W - s + 1, size=n)
        dy, dx = np.meshgrid(np.arange(s), np.arange(s), indexing="ij")   # (s, s) block
        ys   = (y0[:, None, None] + dy).reshape(-1)          # (n·s·s,) row coords
        xs   = (x0[:, None, None] + dx).reshape(-1)          # (n·s·s,) col coords
        vals = np.where(np.repeat(is_salt[m], s * s), 255, 0).astype(np.uint8)
        arr[ys, xs] = vals[:, None]                          # write all 3 channels
    return arr


# ──────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────

def load_model(model_path: str):
    """Load the kaiko-ai midnight ViT from a local HuggingFace snapshot directory.

    The weights come back on the CPU. The extraction functions move the model to the
    device themselves.
    """
    from transformers import AutoModel
    return AutoModel.from_pretrained(model_path)


# ──────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────

def spot_xy_fullres(adata, img_w, img_h):
    """(N, 2) spot centers as (x, y) in full-res pixels: horizontal first, vertical second.

    We infer the coordinate order rather than declare it. obs carries two pixel columns
    whose names do not say which one is horizontal, while obsm["spatial"] follows the
    scanpy convention (x, y) at some fixed scale. Dividing each obs column by the first
    spatial column therefore gives a constant ratio for one of them only, and that one is
    the horizontal coordinate. Inferring matters because the wrong order crops the
    transposed position, which yields embeddings that look fine and mean nothing.
    """
    a = adata.obs[["x_pixel", "y_pixel"]].to_numpy(dtype=float)
    s = np.asarray(adata.obsm["spatial"], dtype=float)

    def same_axis(col):
        r = s[:, 0] / col
        return bool(np.allclose(r, np.median(r), rtol=1e-4))

    if   same_axis(a[:, 1]): xy = a[:, ::-1]      # y_pixel is the horizontal one
    elif same_axis(a[:, 0]): xy = a               # x_pixel is the horizontal one
    else:
        raise ValueError("neither obs pixel column is a constant multiple of "
                         "obsm['spatial'][:, 0]; cannot infer the coordinate order")

    assert xy[:, 0].max() <= img_w and xy[:, 1].max() <= img_h, (
        f"spot centers fall outside the {img_w}x{img_h} image "
        f"(max x {xy[:, 0].max():.0f}, max y {xy[:, 1].max():.0f})")
    return xy


def _crop_spot_patch(img_np, cx, cy, radius, W, H):
    """2r×2r crop centerd at (cx, cy), reflect-padded where it runs off the slide edge.
    Every extraction path crops through this function, so one spot at one radius gives
    the same pixels whichever path produced it."""
    r  = radius
    x0 = int(round(cx - r));  y0 = int(round(cy - r))
    x1 = x0 + 2 * r;          y1 = y0 + 2 * r
    pad_l = max(0, -x0);      pad_t = max(0, -y0)
    pad_r = max(0, x1 - W);   pad_b = max(0, y1 - H)
    patch = img_np[max(0, y0):min(H, y1), max(0, x0):min(W, x1)]
    if any([pad_l, pad_t, pad_r, pad_b]):
        patch = np.pad(patch, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), mode="reflect")
    return patch


class PatchDataset(Dataset):
    """PyTorch Dataset that crops square patches around each Visium spot.

    Args:
        img_np:            Full-resolution image as a (H, W, 3) NumPy array.
        xy_fullres:        (N, 2) array of spot coordinates in full-res pixels.
        spot_diam_fullres: Spot diameter in full-res pixels (from scalefactors JSON).
        transform:         Torchvision transform applied to each cropped patch.
        radius_factor:     Crop half-width = round(spot_diam * radius_factor).
        sp_*:              Salt-and-pepper knobs; see DatasetConfig for their meaning.
    """

    def __init__(
        self,
        img_np:            np.ndarray,
        xy_fullres:        np.ndarray,
        spot_diam_fullres: float,
        transform:         v2.Compose,
        radius_factor:     float = 0.75,
        sp_amount:         float = 0.0,
        sp_salt_ratio:     float = 0.5,
        sp_speck_min:      int   = 1,
        sp_speck_max:      int   = 1,
        sp_seed:           int   = 0,
    ) -> None:
        self.img_np    = img_np
        self.xy        = xy_fullres
        self.radius    = int(round(spot_diam_fullres * radius_factor))
        self.transform = transform
        self.H, self.W = img_np.shape[:2]

        # Salt-and-pepper acts on the 224 model input, so build_transform is split into
        # its geometric part (resize → center-crop, PIL to PIL) and its tensor finish
        # (ToTensor → Normalize). Both halves are the transform's own objects, so the 224
        # and 0.5 constants are never duplicated. The assert guards that structure.
        self.sp_amount     = sp_amount
        self.sp_salt_ratio = sp_salt_ratio
        self.sp_speck_min  = sp_speck_min
        self.sp_speck_max  = sp_speck_max
        self.sp_seed       = sp_seed
        if sp_amount > 0:
            steps = list(transform.transforms)
            assert isinstance(steps[-1], v2.Normalize) and isinstance(steps[-2], v2.ToTensor), \
                "salt-and-pepper expects build_transform to end in ToTensor → Normalize"
            self._resize_crop = v2.Compose(steps[:-2])   # Resize → CenterCrop : PIL → PIL 224
            self._finish      = v2.Compose(steps[-2:])   # ToTensor → Normalize : the exact clean finish

        sp_desc = (f", salt&pepper amount={sp_amount} specks={sp_speck_min}-{sp_speck_max}px"
                   if sp_amount > 0 else "")
        print(f"  PatchDataset: img={img_np.shape}, n_spots={len(xy_fullres)}, "
              f"radius={self.radius}px{sp_desc}")

    def __len__(self) -> int:
        return len(self.xy)

    def __getitem__(self, idx: int) -> torch.Tensor:
        cx, cy = self.xy[idx]       # spot_xy_fullres returns (x, y)
        patch = _crop_spot_patch(self.img_np, cx, cy, self.radius, self.W, self.H)

        if self.sp_amount <= 0:
            return self.transform(Image.fromarray(patch))

        # Resize and crop first, corrupt the exact pixels the ViT sees, then finish with
        # ToTensor → Normalize. Seeding the generator on (seed, spot index) keeps the
        # noise reproducible and independent of worker count and batch order.
        patch224 = np.asarray(self._resize_crop(Image.fromarray(patch))).copy()  # (224,224,3) uint8
        rng = np.random.default_rng((self.sp_seed, int(idx)))
        apply_salt_pepper(patch224, self.sp_amount, self.sp_salt_ratio, rng,
                          speck_min=self.sp_speck_min, speck_max=self.sp_speck_max)
        return self._finish(Image.fromarray(patch224))


# ──────────────────────────────────────────────────────────────
# Embedding extraction
# ──────────────────────────────────────────────────────────────

def extract_cls_embedding(last_hidden_state: torch.Tensor) -> torch.Tensor:
    """Combine [CLS] token with mean-pooled patch tokens.

    Args:
        last_hidden_state: ViT output of shape (N, 1 + n_patches, hidden_dim).

    Returns:
        Tensor of shape (N, 2 * hidden_dim).
    """
    cls_emb   = last_hidden_state[:, 0, :]        # (N, D)
    patch_emb = last_hidden_state[:, 1:, :].mean(1)  # (N, D)
    return torch.cat([cls_emb, patch_emb], dim=-1)   # (N, 2D)


def _infer_embeddings(model, loader, device) -> torch.Tensor:
    """Run the ViT over a loader of 224 model-input tensors, returning (N, 2D) embeddings
    on the CPU in loader order. fp16 autocast, [CLS] concatenated with the mean-pooled
    patch tokens. Every extraction path goes through here."""
    embs: list[torch.Tensor] = []
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            for batch in tqdm(loader, desc="  Embeddings", unit="batch"):
                batch  = batch.to(device, non_blocking=True)
                output = model(batch)
                embs.append(extract_cls_embedding(output.last_hidden_state).cpu())
    return torch.cat(embs, dim=0)  # (N, 2D)


# ──────────────────────────────────────────────────────────────
# Main extraction loop
# ──────────────────────────────────────────────────────────────

def run_extraction(
    model,
    cfg:       DatasetConfig,
    transform: v2.Compose,
    device:    torch.device | None = None,
) -> None:
    """Extract and save patch embeddings for every sample in cfg.samples.

    For each sample the function:
      1. Loads the .h5ad, image (.tif) and metadata (.json).
      2. Builds a PatchDataset + DataLoader.
      3. Runs inference with fp16 autocast.
      4. Saves a .pt holding ``{"embeddings": Tensor(N, 2D), "obs_names", "n_spots"}``.

    Args:
        model:     ViT model (will be moved to device and set to eval mode).
        cfg:       DatasetConfig with all paths and hyper-parameters.
        transform: Image transform pipeline (from build_transform()).
        device:    torch.device to use; defaults to CUDA if available.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # cuDNN 9.x (bundled with torch 2.7) finds no fp16/fp32 engine for the 14x14
    # patch-embedding conv on Volta/sm_70 (V100) -> "unable to find an engine".
    # The model has a single conv, so disabling cuDNN is ~free, and fp16 matmuls
    # still hit the V100 tensor cores via cuBLAS (unaffected by this flag).
    torch.backends.cudnn.enabled = False

    Image.MAX_IMAGE_PIXELS = None          # allow huge TIF files
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    model.eval()

    for i, sample in enumerate(cfg.samples):
        print(f"\nSample {sample}  ({i + 1}/{len(cfg.samples)})")

        # ── Load data ─────────────────────────────────────────
        with open(cfg.meta_path(sample), "r", encoding="utf-8") as fh:
            meta = json.load(fh)

        cdata             = sc.read_h5ad(cfg.h5ad_path(sample))
        img_fullres_np    = np.array(Image.open(cfg.image_path(sample)))
        xy_fullres        = spot_xy_fullres(cdata, img_fullres_np.shape[1], img_fullres_np.shape[0])
        spot_diam_fullres = meta["spot_diameter_fullres"]

        # ── Slide-quality degradation (optional) ───────────────
        # A blurry scan is a property of the whole acquisition, so the entire slide is
        # blurred once here, before any cropping, rather than per patch. σ scales with
        # the spot diameter, so the same physical extent is smeared on slides of
        # different resolution. blur_factor=0 skips the filter.
        # Each RGB channel is blurred separately and in place. That is bit-identical to
        # gaussian_filter(img, sigma=(σ, σ, 0)) on the whole array, but the float32
        # working copy holds one channel instead of three, which is what keeps a
        # whole-slide array within memory.
        if cfg.blur_factor > 0:
            sigma = cfg.blur_factor * spot_diam_fullres
            for c in range(img_fullres_np.shape[2]):
                ch = gaussian_filter(img_fullres_np[..., c].astype(np.float32), sigma=sigma)
                img_fullres_np[..., c] = np.clip(ch, 0, 255).astype(np.uint8)
            print(f"  blur_factor={cfg.blur_factor} → σ={sigma:.2f}px (whole-slide Gaussian, per-channel)")

        # ── Build DataLoader ───────────────────────────────────
        dataset = PatchDataset(
            img_fullres_np, xy_fullres, spot_diam_fullres, transform,
            radius_factor=cfg.radius_factor,
            sp_amount=cfg.sp_amount,
            sp_salt_ratio=cfg.sp_salt_ratio,
            sp_speck_min=cfg.sp_speck_min,
            sp_speck_max=cfg.sp_speck_max,
            sp_seed=cfg.sp_seed,
        )
        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=True,
            prefetch_factor=4 if cfg.num_workers > 0 else None,
            persistent_workers=False,
        )

        # ── Inference ──────────────────────────────────────────
        all_embeddings_t = _infer_embeddings(model, loader, device)  # (N, 2D)
        # ── Sanity check: one embedding per spot, in order ─────
        assert all_embeddings_t.shape[0] == len(xy_fullres), \
            f"{sample}: {all_embeddings_t.shape[0]} embeddings vs {len(xy_fullres)} spots"

        # ── Save ───────────────────────────────────────────────
        # obs_names lets the loader verify the embedding order against the h5ad barcodes
        # and realign, instead of trusting positional order.
        torch.save({
            "embeddings": all_embeddings_t,
            "obs_names":  cdata.obs_names.to_numpy(),  # barcodes, in obs order
            "n_spots":    len(xy_fullres),
        }, cfg.output_path(sample))
        print(f"  Embeddings: {all_embeddings_t.shape}  → {cfg.output_path(sample)}")

        del loader, dataset, img_fullres_np, cdata, all_embeddings_t
        gc.collect()
        torch.cuda.empty_cache()

    print("\nAll samples done.")


# ──────────────────────────────────────────────────────────────
# Salt-and-pepper SWEEP (slide-outer, efficient)
# ──────────────────────────────────────────────────────────────

class _NoisyCachedDataset(Dataset):
    """Applies salt-and-pepper on the fly to a precomputed (N, 224, 224, 3) uint8 cache of
    clean model-input patches. The crop and resize happen once per slide, in the cache, so
    a sweep over amounts pays only for the noise and the tensor finish. Seeding on
    (seed, idx) makes each patch identical to PatchDataset's salt-and-pepper path."""

    def __init__(self, clean224, finish, amount, salt_ratio, speck_min, speck_max, seed):
        self.clean224   = clean224
        self.finish     = finish
        self.amount     = amount
        self.salt_ratio = salt_ratio
        self.speck_min  = speck_min
        self.speck_max  = speck_max
        self.seed       = seed

    def __len__(self):
        return len(self.clean224)

    def __getitem__(self, idx):
        patch = self.clean224[idx].copy()   # (224,224,3) uint8 — never mutate the cache
        if self.amount > 0:
            rng = np.random.default_rng((self.seed, int(idx)))
            apply_salt_pepper(patch, self.amount, self.salt_ratio, rng,
                              speck_min=self.speck_min, speck_max=self.speck_max)
        return self.finish(Image.fromarray(patch))


def _build_clean224(img_np, xy, radius, resize_crop) -> np.ndarray:
    """Crop and resize every spot into a (N, 224, 224, 3) uint8 cache.

    This is the work that does not depend on the noise amount, so run_saltpepper_sweep
    builds it once per slide and reuses it for every amount."""
    H, W = img_np.shape[:2]
    out  = None
    for i in range(len(xy)):
        cx, cy = xy[i]       # spot_xy_fullres returns (x, y)
        patch = _crop_spot_patch(img_np, cx, cy, radius, W, H)
        p224  = np.asarray(resize_crop(Image.fromarray(patch)))   # (224,224,3) uint8
        if out is None:
            out = np.empty((len(xy), *p224.shape), dtype=np.uint8)
        out[i] = p224
    return out


def run_saltpepper_sweep(
    model,
    cfg:           DatasetConfig,
    transform:     v2.Compose,
    amounts:       List[float],
    output_dir_fn,
    device:        torch.device | None = None,
) -> None:
    """Salt-and-pepper sweep, slide outer.

    Each slide is cropped and resized to the 224 model input once, into a cache. Each
    amount then pays only for the noise, the ViT and the save. The ViT runs once per
    amount because each amount is a different input. The saved .pt files match what
    run_extraction writes for that amount, since the noise is seeded per (seed, spot
    index).

    cfg supplies paths, radius_factor, sp_salt_ratio, sp_speck_min/max, sp_seed,
    batch_size and num_workers. cfg.sp_amount, cfg.blur_factor and cfg.output_dir are
    unused: the sweep is driven by ``amounts`` and ``output_dir_fn(amount) -> dir``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.backends.cudnn.enabled = False       # same V100 cuDNN workaround as run_extraction
    Image.MAX_IMAGE_PIXELS = None
    model = model.to(device)
    model.eval()

    # Split build_transform: the geometric part (resize → crop) fills the cache, and the
    # tensor finish (ToTensor → Normalize) runs per amount after noising. Both halves are
    # the transform's own objects, so the 224 and 0.5 constants are never duplicated.
    steps = list(transform.transforms)
    assert isinstance(steps[-1], v2.Normalize) and isinstance(steps[-2], v2.ToTensor), \
        "run_saltpepper_sweep expects build_transform to end in ToTensor → Normalize"
    resize_crop = v2.Compose(steps[:-2])
    finish      = v2.Compose(steps[-2:])

    for i, sample in enumerate(cfg.samples):
        print(f"\nSample {sample}  ({i + 1}/{len(cfg.samples)})")

        # ── Load slide ONCE ────────────────────────────────────
        with open(cfg.meta_path(sample), "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        cdata  = sc.read_h5ad(cfg.h5ad_path(sample))
        img    = np.array(Image.open(cfg.image_path(sample)))
        xy     = spot_xy_fullres(cdata, img.shape[1], img.shape[0])
        radius = int(round(meta["spot_diameter_fullres"] * cfg.radius_factor))

        # ── Invariant work: crop + resize every patch to a 224 cache, ONCE ─────
        clean224 = _build_clean224(img, xy, radius, resize_crop)  # (N,224,224,3)
        del img
        gc.collect()
        print(f"  cached {clean224.shape[0]} patches at 224 (radius={radius}px); "
              f"sweeping {len(amounts)} amount(s)")

        # ── Per amount: noise the cache + ViT + save ───────────
        for amount in amounts:
            out_dir = output_dir_fn(amount)
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            ds = _NoisyCachedDataset(clean224, finish, amount, cfg.sp_salt_ratio,
                                     cfg.sp_speck_min, cfg.sp_speck_max, cfg.sp_seed)
            loader = DataLoader(
                ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                pin_memory=True, prefetch_factor=4 if cfg.num_workers > 0 else None,
                persistent_workers=False,
            )
            emb = _infer_embeddings(model, loader, device)   # (N, 2D)
            assert emb.shape[0] == len(xy), \
                f"{sample} amount={amount}: {emb.shape[0]} embeddings vs {len(xy)} spots"
            out_path = str(Path(out_dir) / f"{sample}.pt")
            torch.save({
                "embeddings": emb,
                "obs_names":  cdata.obs_names.to_numpy(),
                "n_spots":    len(xy),
            }, out_path)
            print(f"  amount={amount}: {tuple(emb.shape)}  → {out_path}")
            del loader, ds, emb
            gc.collect()
            torch.cuda.empty_cache()

        del clean224, cdata
        gc.collect()

    print("\nAll samples + amounts done.")


# ──────────────────────────────────────────────────────────────
# Crop-RADIUS sweep (slide-outer, efficient)
# ──────────────────────────────────────────────────────────────

def run_radius_sweep(
    model,
    cfg:           DatasetConfig,
    transform:     v2.Compose,
    radii:         List[float],
    output_dir_fn,
    skip_existing: bool = True,
    device:        torch.device | None = None,
) -> None:
    """Crop-radius (field-of-view) sweep, slide outer.

    Each slide's full-res .tif is read once, the expensive part, then every radius_factor
    re-crops, resizes, runs the ViT and saves before the next slide. The 224 cache of
    run_saltpepper_sweep cannot be reused here: the radius changes the crop itself, so the
    slide read is the only work shared across radii. Each radius is a different field of
    view, hence a different input and different embeddings. A 2r crop that spills over the
    border is reflect-padded, through the same _crop_spot_patch as run_extraction. The
    saved .pt files have run_extraction's format, {embeddings, obs_names, n_spots}.

    cfg supplies paths, batch_size and num_workers. cfg.radius_factor, cfg.sp_*,
    cfg.blur_factor and cfg.output_dir are unused: the sweep is driven by ``radii`` and
    ``output_dir_fn(radius_factor) -> dir``. With skip_existing, a (slide, radius) whose
    .pt already exists is skipped, so a re-run resumes and an already-extracted radius is
    reused rather than recomputed.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.backends.cudnn.enabled = False       # same V100 cuDNN workaround as run_extraction
    Image.MAX_IMAGE_PIXELS = None
    model = model.to(device)
    model.eval()

    for i, sample in enumerate(cfg.samples):
        print(f"\nSample {sample}  ({i + 1}/{len(cfg.samples)})")

        # Which radii still need doing for this slide? (skip already-saved .pt so the slide
        # read below is avoided entirely when nothing is left to do.)
        todo = list(radii)
        if skip_existing:
            done = [rf for rf in radii if Path(output_dir_fn(rf), f"{sample}.pt").exists()]
            todo = [rf for rf in radii if rf not in done]
            if done:
                print(f"  skip (already on disk): {done}")
        if not todo:
            print("  nothing to do -> next slide")
            continue

        # ── Load slide ONCE (the expensive read, shared by every radius) ───────
        with open(cfg.meta_path(sample), "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        cdata     = sc.read_h5ad(cfg.h5ad_path(sample))
        img       = np.array(Image.open(cfg.image_path(sample)))
        xy        = spot_xy_fullres(cdata, img.shape[1], img.shape[0])
        spot_diam = meta["spot_diameter_fullres"]

        # ── Per radius: re-crop from the cached slide + ViT + save ─────────────
        for rf in todo:
            out_dir = output_dir_fn(rf)
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            dataset = PatchDataset(               # sp_amount=0 -> the clean crop path
                img, xy, spot_diam, transform,
                radius_factor=rf,
            )
            loader = DataLoader(
                dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                pin_memory=True, prefetch_factor=4 if cfg.num_workers > 0 else None,
                persistent_workers=False,
            )
            emb = _infer_embeddings(model, loader, device)   # (N, 2D)
            assert emb.shape[0] == len(xy), \
                f"{sample} radius_factor={rf}: {emb.shape[0]} embeddings vs {len(xy)} spots"
            out_path = str(Path(out_dir) / f"{sample}.pt")
            torch.save({
                "embeddings": emb,
                "obs_names":  cdata.obs_names.to_numpy(),
                "n_spots":    len(xy),
            }, out_path)
            print(f"  radius_factor={rf}: {tuple(emb.shape)}  -> {out_path}")
            del loader, dataset, emb
            gc.collect()
            torch.cuda.empty_cache()

        del img, cdata
        gc.collect()

    print("\nAll samples + radii done.")


# ──────────────────────────────────────────────────────────────
# Gaussian-BLUR sweep (slide-outer, efficient)
# ──────────────────────────────────────────────────────────────

def run_blur_sweep(
    model,
    cfg:           DatasetConfig,
    transform:     v2.Compose,
    blur_factors:  List[float],
    output_dir_fn,
    skip_existing: bool = True,
    device:        torch.device | None = None,
) -> None:
    """Whole-slide Gaussian blur sweep, slide outer.

    The blur analog of run_radius_sweep: each slide's full-res .tif is read once, the
    expensive part, then every blur_factor gets a blurred copy (sigma = bf *
    spot_diam_fullres, per channel) that is cropped, resized and run through the ViT
    before the next slide.

    The filter always starts from a fresh copy of the pristine slide, never from an
    already-blurred one, and runs per channel, so the result matches run_extraction with
    cfg.blur_factor=bf. bf=0.0 skips the filter and uses the pristine slide.

    cfg supplies paths, radius_factor, batch_size and num_workers.
    cfg.blur_factor, cfg.sp_* and cfg.output_dir are unused: the sweep is driven by
    ``blur_factors`` and ``output_dir_fn(bf) -> dir``. With skip_existing, a (slide, bf)
    whose .pt already exists is skipped, so a re-run resumes.

    Peak memory holds the pristine slide, the blurred copy and the float32 buffers for one
    channel, so a whole-slide array is held several times over.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.backends.cudnn.enabled = False       # same V100 cuDNN workaround as run_extraction
    Image.MAX_IMAGE_PIXELS = None
    model = model.to(device)
    model.eval()

    for i, sample in enumerate(cfg.samples):
        print(f"\nSample {sample}  ({i + 1}/{len(cfg.samples)})")

        # Which levels still need doing for this slide? (skip already-saved .pt so the slide
        # read below is avoided entirely when nothing is left to do.)
        todo = list(blur_factors)
        if skip_existing:
            done = [bf for bf in blur_factors if Path(output_dir_fn(bf), f"{sample}.pt").exists()]
            todo = [bf for bf in blur_factors if bf not in done]
            if done:
                print(f"  skip (already on disk): {done}")
        if not todo:
            print("  nothing to do -> next slide")
            continue

        # ── Load slide ONCE (the expensive read, shared by every blur level) ───
        with open(cfg.meta_path(sample), "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        cdata     = sc.read_h5ad(cfg.h5ad_path(sample))
        pristine  = np.array(Image.open(cfg.image_path(sample)))
        xy        = spot_xy_fullres(cdata, pristine.shape[1], pristine.shape[0])

        # ── Per blur level: blur a COPY of the pristine slide + ViT + save ─────
        for bf in todo:
            out_dir = output_dir_fn(bf)
            Path(out_dir).mkdir(parents=True, exist_ok=True)

            if bf > 0:
                # sigma scales with the spot diameter, so the same physical extent is
                # smeared on slides of different resolution. The float32 work is done one
                # channel at a time and written into a fresh uint8 copy.
                sigma = bf * spot_diam
                img   = np.empty_like(pristine)
                for c in range(pristine.shape[2]):
                    ch = gaussian_filter(pristine[..., c].astype(np.float32), sigma=sigma)
                    img[..., c] = np.clip(ch, 0, 255).astype(np.uint8)
                del ch
                print(f"  blur_factor={bf} -> sigma={sigma:.2f}px (whole-slide Gaussian, per-channel)")
            else:
                img = pristine          # bf=0 -> filter skipped, byte-identical to clean

            dataset = PatchDataset(               # sp_amount=0 -> the clean crop path
                img, xy, spot_diam, transform,
                radius_factor=cfg.radius_factor,
            )
            loader = DataLoader(
                dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                pin_memory=True, prefetch_factor=4 if cfg.num_workers > 0 else None,
                persistent_workers=False,
            )
            emb = _infer_embeddings(model, loader, device)   # (N, 2D)
            assert emb.shape[0] == len(xy), \
                f"{sample} blur_factor={bf}: {emb.shape[0]} embeddings vs {len(xy)} spots"
            out_path = str(Path(out_dir) / f"{sample}.pt")
            torch.save({
                "embeddings": emb,
                "obs_names":  cdata.obs_names.to_numpy(),
                "n_spots":    len(xy),
            }, out_path)
            print(f"  blur_factor={bf}: {tuple(emb.shape)}  -> {out_path}")
            del loader, dataset, emb
            if img is not pristine:
                del img
            gc.collect()
            torch.cuda.empty_cache()

        del pristine, cdata
        gc.collect()

    print("\nAll samples + blur levels done.")
