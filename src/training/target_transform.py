# src/training/target_transform.py
#
# Per-gene target standardization stats, shared by both pipelines (classic
# train.py and GCM train_gcm.py) so the transform is computed identically for
# each. That is what keeps a classic-versus-GCM comparison fair when a transform
# is active.
#
# Applies only to target_source == "X" models, meaning MSE and the Gaussians. A
# count model is left untouched, because its Poisson or NB likelihood is defined
# on the count target itself and a per-gene shift would push that target negative.
# The caller gates on the spec. The (loc, scale) constants are computed on a
# fold's train rows only, then saved with the checkpoint and applied identically
# to val and test, like the embedding train_mean.

import numpy as np
import torch


def compute_gene_target_stats(tgt_train, slide_labels_train, transform):
    """Per-gene (loc, scale) for the target transform, from train rows only.

    Returns two float32 numpy arrays of shape (n_genes,), or (None, None) when
    transform == "none". Pass train rows only: the same constants are applied to val
    and test, so a val or test row reaching this statistic is leakage.

    gene_zscore:
        loc_g   = mean over train spots of gene g
        scale_g = sqrt( mean over train slides of within-slide variance of gene g ).
                  Averaging the within-slide variance keeps between-slide mean shifts
                  out of the scale, so it reflects each slide's own spread rather
                  than cross-slide batch offsets, while staying one per-gene constant
                  that can be applied to a held-out slide whose expression is unknown
                  at predict time. A slide with fewer than two train spots has no
                  variance and is skipped.
    gene_minmax:
        loc_g   = min over train spots,  scale_g = max - min  (over train spots).

    A constant gene gets scale 1, so the transform degrades to a pure shift instead
    of dividing by zero.
    """
    if transform == "none":
        return None, None
    if transform == "gene_zscore":
        loc = tgt_train.mean(dim=0)
        var_acc = torch.zeros(tgt_train.shape[1], device=tgt_train.device)
        n_slides = 0
        for s in np.unique(slide_labels_train):
            rows = np.flatnonzero(slide_labels_train == s)
            if rows.size < 2:                       # need >=2 spots for a variance
                continue
            idx = torch.as_tensor(rows, dtype=torch.long, device=tgt_train.device)
            var_acc += tgt_train[idx].var(dim=0, unbiased=True)
            n_slides += 1
        scale = torch.sqrt(var_acc / max(n_slides, 1))
    elif transform == "gene_minmax":
        loc   = tgt_train.amin(dim=0)
        scale = tgt_train.amax(dim=0) - loc
    else:
        raise ValueError(f"Unknown target_transform: {transform!r}")
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    return (loc.cpu().numpy().astype(np.float32),
            scale.cpu().numpy().astype(np.float32))
