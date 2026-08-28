# src/data/dataset_indexed.py
#
# Index-based data holders for the classic per-gene pipeline.
#
# Design:
#   - The embedding matrix and the target matrix are each materialized once as a
#     shared tensor. On the primary path that tensor lives on the GPU, so a
#     per-batch gather is a pure on-device index with no host-to-device copy. A
#     host-resident, optionally pinned, fallback covers data that does not fit on
#     the GPU.
#   - A dataset owns no copy of the data. It owns an int64 array of row indices
#     into the shared tensors. train, val and test are three datasets over the
#     same shared tensors, differing only by their index arrays, so adding a fold
#     costs no data re-materialization, just a new index array.
#   - The classic training loop and run_inference bypass the Dataset wrappers on
#     the hot path. With the data GPU-resident they shuffle and slice row indices
#     on device directly, in train_loop_indexed.py and inference_indexed.py.
#     IndexedGeneDataset and assert_split_alignment serve the alignment self-test
#     (run_loader_correctness_test below), which checks the index-to-gather
#     contract the whole pipeline relies on.

import numpy as np
import torch
from torch.utils.data import Dataset


class SharedTensors:
    """The single source of truth for a run's data.

    Holds the full embedding and target matrices plus the per-spot group (slide)
    labels. All datasets index into this object.

    With device="cuda" the matrices live on the GPU, so a per-batch gather is a
    pure on-device index: no host-to-device copy and no pinning needed. For the
    single-gene and few-gene classic runs the embedding matrix (n_spots x
    emb_dim) fits comfortably in GPU memory and the targets are tiny.

    With device="cpu" and pin=True the matrices stay in optionally page-locked
    host memory and gather() moves each batch per call. Pinning buys less here
    than it looks: a scattered advanced-index gather returns a fresh pageable
    tensor, so even from pinned memory the follow-up .to(cuda) is effectively
    synchronous. Prefer the GPU-resident path whenever the data fits.

    Parameters
    ----------
    embeddings : np.ndarray (n_spots, emb_dim)        spot patch embeddings
    targets    : np.ndarray (n_spots, n_genes)        expression (already gene-sliced)
    groups     : np.ndarray (n_spots,)                slide label per spot
    device     : str | torch.device                   where to hold the data;
                                                       "cuda" => GPU-resident
    pin        : bool                                  page-lock host memory
                                                       (fallback path only)
    depth_scale : float or None                       the run's one depth constant,
                                                       config
                                                       datasets.{ds}.count_common_depth
                                                       / 1e4, for the cp_counts and
                                                       cp_counts_nr targets. None for
                                                       the depth-blind raw_counts
                                                       target and for every X-target
                                                       (Gaussian/MSE) run. A plain
                                                       float, not a tensor and not
                                                       per-spot, because no part of the
                                                       modeling path reads it. Its only
                                                       role is the divisor applied
                                                       before log1p when reporting the
                                                       mean correlation.
    """

    def __init__(self, embeddings, targets, groups, device="cpu", pin=False,
                 depth_scale=None):
        emb = torch.as_tensor(np.ascontiguousarray(embeddings), dtype=torch.float32)
        tgt = torch.as_tensor(np.ascontiguousarray(targets),    dtype=torch.float32)
        if tgt.ndim == 1:
            tgt = tgt.unsqueeze(1)                 # (n_spots,) -> (n_spots, 1)

        target_device = torch.device(device)
        if target_device.type == "cuda" and torch.cuda.is_available():
            # Keep the whole matrix on the GPU once. Every fold's per-batch gather
            # is then an on-device index with no host-to-device copy, so the small
            # MLP never stalls waiting on data movement.
            emb = emb.to(target_device)
            tgt = tgt.to(target_device)
        elif pin and torch.cuda.is_available():
            # Fallback: page-lock host memory. See the class docstring for why
            # this only partially helps. Prefer the GPU-resident path when the
            # data fits.
            emb = emb.pin_memory()
            tgt = tgt.pin_memory()

        self.emb = emb
        self.tgt = tgt
        self.groups = np.asarray(groups)
        # The run's one depth constant, see the class docstring. A scalar float or
        # None, deliberately not a tensor and not gathered per spot, because the
        # loss, the forward pass and the predictive distribution never see it.
        self.depth_scale = None if depth_scale is None else float(depth_scale)

        self.n_spots, self.emb_dim = self.emb.shape
        self.n_genes = self.tgt.shape[1]

        assert self.tgt.shape[0] == self.n_spots, "targets/embeddings row mismatch"
        assert len(self.groups) == self.n_spots,  "groups/embeddings row mismatch"

    def gather(self, idx, device=None, non_blocking=True):
        """Index rows `idx` of (emb, tgt).

        Indexing happens on whatever device the data lives on, so a GPU-resident
        run gathers on device with no host-to-device copy. `idx` is moved to the
        data's device first; it is tiny. `device`, when given and different from
        the data's device, moves the gathered batch there. Only the host-resident
        fallback needs that, and it costs nothing when the data already lives on
        `device`.
        """
        if not torch.is_tensor(idx):
            idx = torch.as_tensor(np.asarray(idx, dtype=np.int64))
        idx = idx.to(self.emb.device, non_blocking=non_blocking)
        xb = self.emb[idx]
        yb = self.tgt[idx]
        if device is not None and torch.device(device) != self.emb.device:
            xb = xb.to(device, non_blocking=non_blocking)
            yb = yb.to(device, non_blocking=non_blocking)
        return xb, yb


class IndexedGeneDataset(Dataset):
    """A view over SharedTensors restricted to a set of spot rows, meaning one split.

    Yields a row index into the shared tensors, not the data row. The caller
    gathers the actual tensors via SharedTensors.gather. Centering is applied at
    gather time, so the fold's train_mean is never baked into the shared tensor
    and all three splits keep sharing one tensor.
    """

    def __init__(self, shared: SharedTensors, row_indices):
        self.shared = shared
        self.indices = np.asarray(row_indices, dtype=np.int64)
        # Invariant: every index is in range for the shared tensors.
        assert self.indices.min() >= 0 and self.indices.max() < shared.n_spots, (
            "row index out of range for shared tensors"
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.indices[i]            # a single int64 row index

    def groups(self):
        """Slide labels for this split's spots, in dataset order."""
        return self.shared.groups[self.indices]


# ─────────────────────────────────────────────────────────────────────────────
# Alignment invariants. They run against constructed ground truth, so they check
# the index-to-gather contract in isolation, with no trained model and no real
# data.
# ─────────────────────────────────────────────────────────────────────────────

def assert_split_alignment(shared, splits):
    """Assert that no two splits share a spot, which would be leakage.

    Also checks that each split's group labels have the same length as its index
    array. emb, tgt and groups are indexed by that one array, so a length match
    is what ties the three together.

    splits : dict[str, IndexedGeneDataset]  e.g. {"train":..., "val":..., "test":...}
    """
    seen = {}
    for name, ds in splits.items():
        idx = ds.indices
        # emb, tgt and groups are indexed by this one idx, so they describe the
        # same spots by construction.
        assert len(ds.groups()) == len(idx)
        # disjointness across splits
        s = set(idx.tolist())
        for other_name, other in seen.items():
            overlap = s & other
            assert not overlap, f"{name} and {other_name} share {len(overlap)} spots (leakage)"
        seen[name] = s


def run_loader_correctness_test():
    """Check the loader against constructed ground truth. Raises on failure.

    Known pairing: the embeddings are built as a known function of the targets,
    so a correct gather reproduces that pairing for every spot.
    Round-trip: a handful of known spots gather back to their own rows.
    """
    rng = np.random.default_rng(0)
    n_spots, emb_dim, n_genes = 50, 8, 3
    targets = rng.normal(size=(n_spots, n_genes)).astype(np.float32)
    # embedding row r = [target[r,0] repeated emb_dim times]  -> known function
    embeddings = np.repeat(targets[:, :1], emb_dim, axis=1).astype(np.float32)
    groups = np.array([f"slide{r % 4}" for r in range(n_spots)])

    shared = SharedTensors(embeddings, targets, groups, pin=False)
    idx = np.arange(n_spots)
    rng.shuffle(idx)                       # arbitrary fold order
    ds = IndexedGeneDataset(shared, idx)

    # Known pairing: for every spot in the split, emb col 0 == target col 0
    xb, yb = shared.gather(torch.as_tensor(ds.indices), device="cpu")
    assert torch.allclose(xb[:, 0], yb[:, 0]), "embedding/target pairing broken"
    # all emb columns equal (by construction) -> alignment held through indexing
    assert torch.allclose(xb, xb[:, :1].expand_as(xb)), "row identity scrambled"

    # Round-trip: 10 known spots gather to their own original rows
    probe = np.array([3, 7, 11, 0, 49, 25, 13, 8, 40, 2])
    xb, yb = shared.gather(torch.as_tensor(probe), device="cpu")
    assert torch.allclose(yb, torch.as_tensor(targets[probe])), "target round-trip failed"
    assert torch.allclose(xb, torch.as_tensor(embeddings[probe])), "embedding round-trip failed"

    # disjointness invariant on a 3-way split
    cut1, cut2 = 30, 40
    splits = {
        "train": IndexedGeneDataset(shared, idx[:cut1]),
        "val":   IndexedGeneDataset(shared, idx[cut1:cut2]),
        "test":  IndexedGeneDataset(shared, idx[cut2:]),
    }
    assert_split_alignment(shared, splits)
    print("loader correctness test: PASS")


if __name__ == "__main__":
    run_loader_correctness_test()
