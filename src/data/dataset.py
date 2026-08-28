# src/data/dataset.py
#
# Data holders for the gene-conditioned (GCM) pipeline, whose model takes a gene
# embedding alongside the spot embedding.
#
# ── File layout ───────────────────────────────────────────────────────────────
#   PairedSharedTensors is the primary holder and the gene-conditioned analog of
#   the classic SharedTensors in src/data/dataset_indexed.py. The spot
#   embeddings, the gene embeddings and the target matrix each live on the GPU as
#   a single copy, and a batch is built by indexing them on device. The full
#   (n_spots * n_genes) pair matrix is never materialized, which is what lets one
#   GPU train the paired model with no per-batch host-to-device copy.
#
#   PairedSpotGeneDataset is the lazy per-pair Dataset. It defines the row-major
#   (spot, gene) index encoding that the flat-to-2D reshape depends on.
#
#   make_gene_emb_matrix stacks gene embeddings into a matrix in a chosen gene
#   order.
#
#   The classic per-gene pipeline does not use this file. It indexes into
#   SharedTensors instead, in src/data/dataset_indexed.py.

import numpy as np
import torch
from torch.utils.data import Dataset


# ─────────────────────────────────────────────────────────────────────────────
# Gene-conditioned datasets and helpers
# ─────────────────────────────────────────────────────────────────────────────

class PairedSpotGeneDataset(Dataset):
    """
    Lazy (spot, gene) pair dataset for the gene-conditioned MLP.

    One index is one (spot, gene) pair. __getitem__ concatenates the spot
    embedding with the gene embedding, so the (n_spots x n_genes, spot_dim +
    gene_dim) matrix is never materialized.

    The index encoding is row-major over spots: idx = spot_id * n_genes +
    gene_id. A pass with shuffle=False therefore reshapes back to
    (n_spots, n_genes). With shuffle=True that reshape still succeeds and
    scrambles spots and genes.

    Row i of gene_emb_matrix (n_genes, gene_dim) is the gene whose expression
    sits in column gene_indices[i] of y_matrix (n_spots, n_all_genes). Only the
    lengths are asserted, so a wrong order raises nothing and ties one gene's
    embedding to another gene's expression.

    X_spots is (n_spots, spot_dim) and groups is (n_spots,), one sample label per
    spot. __getitem__ returns the concatenated float32 embedding, the scalar
    expression for the pair, and the spot's group label.
    """

    def __init__(self, X_spots, y_matrix, groups, gene_emb_matrix, gene_indices):
        self.X_spots  = torch.as_tensor(X_spots,        dtype=torch.float32)
        self.y_matrix = torch.as_tensor(y_matrix,       dtype=torch.float32)
        self.gene_emb = torch.as_tensor(gene_emb_matrix, dtype=torch.float32)
        self.groups   = np.asarray(groups)
        self.gene_idx = np.asarray(gene_indices, dtype=np.int64)

        self.n_spots = self.X_spots.shape[0]
        self.n_genes = self.gene_emb.shape[0]

        # Invariants -- catch shape mistakes early rather than at __getitem__ time
        assert self.y_matrix.shape[0] == self.n_spots, (
            f"y_matrix has {self.y_matrix.shape[0]} rows but X_spots has {self.n_spots}"
        )
        assert len(self.gene_idx) == self.n_genes, (
            f"gene_indices length {len(self.gene_idx)} mismatches "
            f"gene_emb rows {self.n_genes}"
        )
        assert len(self.groups) == self.n_spots

    def __len__(self):
        return self.n_spots * self.n_genes

    def __getitem__(self, idx):
        s = idx // self.n_genes
        g = idx %  self.n_genes
        x = torch.cat([self.X_spots[s], self.gene_emb[g]], dim=0)
        y = self.y_matrix[s, self.gene_idx[g]]
        return x, y, self.groups[s]


def make_gene_emb_matrix(gene_list, gene_embs):
    """
    Stack gene embeddings into a (len(gene_list), gene_dim) float32 array.

    Row order follows gene_list, and downstream code pairs those rows with
    expression columns by position. Raises KeyError if a gene has no embedding,
    and ValueError if the vectors do not all share one dimension.
    """
    return np.stack([gene_embs[g] for g in gene_list], axis=0).astype(np.float32)



class PairedSharedTensors:
    """The single data source for a GCM run, held on the GPU.

    The gene-conditioned analog of the classic SharedTensors in
    src/data/dataset_indexed.py. Three matrices live on the device, one copy each:
        spot_emb : (n_spots, spot_dim)            spot patch embeddings
        gene_emb : (n_genes_total, gene_dim)      gene embeddings, all run genes
        y        : (n_spots, n_genes_total)       expression target matrix

    Both gather methods index spot rows and gene rows with different index
    tensors, so the (n_spots * n_genes_total, spot_dim + gene_dim) pair matrix is
    never materialized and only the in-flight batch exists. That is what lets a
    single GPU train the paired model with no per-batch host-to-device copy.
    `gather` returns a flat batch of pairs. `gather_block` keeps the spot and gene
    axes separate, which is the form the batched model consumes.

    With device="cuda" all three matrices sit on the GPU and gathering is a pure
    on-device index and concat. With device="cpu" and pin=True they stay in
    optionally page-locked host memory and the gather runs on the host, returning
    host tensors the caller must move itself: neither gather method takes a target
    device. Use that only when the data does not fit in VRAM, and see
    SharedTensors for why pinning buys less there than it looks.

    Parameters
    ----------
    spot_emb : np.ndarray (n_spots, spot_dim)
    gene_emb : np.ndarray (n_genes_total, gene_dim)   row order == columns of y
    y        : np.ndarray (n_spots, n_genes_total)
    groups   : np.ndarray (n_spots,)                  slide label per spot
    device   : str | torch.device                     "cuda" => GPU-resident
    pin      : bool                                    page-lock host (fallback only)
    depth_scale : float or None                        the run's one depth constant,
               config datasets.{ds}.count_common_depth / 1e4, for the cp_counts and
               cp_counts_nr targets. None for the depth-blind raw_counts target and
               for every X-target run. A plain float, not a tensor and not per-spot,
               because no part of the modeling path reads it. Its only role is the
               divisor applied before log1p when reporting the mean correlation.
    """

    def __init__(self, spot_emb, gene_emb, y, groups, device="cpu", pin=False,
                 depth_scale=None):
        spot = torch.as_tensor(np.ascontiguousarray(spot_emb), dtype=torch.float32)
        gene = torch.as_tensor(np.ascontiguousarray(gene_emb), dtype=torch.float32)
        tgt  = torch.as_tensor(np.ascontiguousarray(y),        dtype=torch.float32)

        target_device = torch.device(device)
        if target_device.type == "cuda" and torch.cuda.is_available():
            spot = spot.to(target_device)
            gene = gene.to(target_device)
            tgt  = tgt.to(target_device)
        elif pin and torch.cuda.is_available():
            spot = spot.pin_memory()
            gene = gene.pin_memory()
            tgt  = tgt.pin_memory()

        self.spot_emb = spot
        self.gene_emb = gene
        self.y        = tgt
        self.groups   = np.asarray(groups)
        # The run's one depth constant, see the class docstring. A scalar float or
        # None, deliberately not a tensor and not gathered per spot, because the
        # loss, the forward pass and the predictive distribution never see it.
        self.depth_scale = None if depth_scale is None else float(depth_scale)

        self.n_spots,       self.spot_dim = self.spot_emb.shape
        self.n_genes_total, self.gene_dim = self.gene_emb.shape

        assert self.y.shape[0] == self.n_spots,       "y/spot_emb row mismatch"
        assert self.y.shape[1] == self.n_genes_total, "y/gene_emb column mismatch"
        assert len(self.groups) == self.n_spots,      "groups/spot_emb row mismatch"

    @property
    def device(self):
        return self.spot_emb.device

    def gather(self, pair_idx, spot_rows, gene_cols, n_sub_genes, spot_mean=None):
        """Build a (spot, gene) batch on-device from flat pair indices.

        The pair index space is this split's local virtual matrix
        (len(spot_rows) x n_sub_genes), row-major over spots and using the same
        encoding as PairedSpotGeneDataset, so an unshuffled pass reshapes back to
        (n_local_spots, n_sub_genes).

        Parameters
        ----------
        pair_idx   : LongTensor (B,)   flat indices into the local virtual matrix.
        spot_rows  : LongTensor (n_local_spots,)  global spot rows for this split.
        gene_cols  : LongTensor (n_sub_genes,)    global gene columns for this gene subset.
        n_sub_genes: int               == gene_cols.numel() (passed to avoid a sync).
        spot_mean  : Tensor (spot_dim,) or None    centering, subtracted from the
                     spot block only; the gene block is never centered.

        Returns
        -------
        xb   : (B, spot_dim + gene_dim) float tensor on device
        yb   : (B,)                     target per pair
        rows : (B,)                     global spot row of each pair (for slide lookup)
        """
        dev = self.device
        pair_idx  = pair_idx.to(dev, non_blocking=True)
        spot_rows = spot_rows.to(dev, non_blocking=True)
        gene_cols = gene_cols.to(dev, non_blocking=True)

        s_local = torch.div(pair_idx, n_sub_genes, rounding_mode="floor")
        g_local = pair_idx - s_local * n_sub_genes
        rows = spot_rows[s_local]
        cols = gene_cols[g_local]

        xb_spot = self.spot_emb[rows]
        if spot_mean is not None:
            xb_spot = xb_spot - spot_mean
        xb = torch.cat([xb_spot, self.gene_emb[cols]], dim=1)
        yb = self.y[rows, cols]
        return xb, yb, rows

    def gather_block(self, spot_rows, gene_cols, spot_mean=None):
        """Build a (spots x genes) block with the spot and gene axes kept separate.

        No pairs are formed. The spot backbone runs on `Xs`, the gene
        hypernetwork on `E`, and the model predicts the whole `(B, G)` block in
        one matmul.

        Parameters
        ----------
        spot_rows : LongTensor (B,)   global spot rows for this batch/split.
        gene_cols : LongTensor (G,)   global gene columns for this gene subset.
        spot_mean : Tensor (spot_dim,) or None   subtracted from the spot block only.

        Returns
        -------
        Xs   : (B, spot_dim)   spot embeddings, centered if spot_mean is given
        E    : (G, gene_dim)   gene embeddings for gene_cols, never centered
        Yb   : (B, G)          targets y[spot_rows][:, gene_cols]
        rows : (B,)            global spot rows, for slide lookup
        """
        dev = self.device
        rows = spot_rows.to(dev, non_blocking=True)
        cols = gene_cols.to(dev, non_blocking=True)
        Xs = self.spot_emb[rows]
        if spot_mean is not None:
            Xs = Xs - spot_mean
        E  = self.gene_emb[cols]
        Yb = self.y[rows][:, cols]
        return Xs, E, Yb, rows


# ─────────────────────────────────────────────────────────────────────────────
# Alignment self-tests for PairedSharedTensors. They check gather and
# gather_block against constructed ground truth, so they need no real data.
# ─────────────────────────────────────────────────────────────────────────────

def run_paired_loader_correctness_test():
    """Check PairedSharedTensors.gather against constructed ground truth.
    Raises on failure, prints PASS otherwise.

    The fixture encodes each row's identity in its values: spot_emb row s is
    [s]*spot_dim, gene_emb row g is [g]*gene_dim, y[s, g] is s*1000 + g. So a
    correct gather of pair (s, g) must yield a spot block all equal to s, a gene
    block all equal to g, and target s*1000 + g.
    """
    n_spots, spot_dim, n_genes, gene_dim = 12, 4, 5, 3
    spot_emb = np.tile(np.arange(n_spots)[:, None], (1, spot_dim)).astype(np.float32)
    gene_emb = np.tile(np.arange(n_genes)[:, None], (1, gene_dim)).astype(np.float32)
    y = (np.arange(n_spots)[:, None] * 1000 + np.arange(n_genes)[None, :]).astype(np.float32)
    groups = np.array([f"slide{s % 3}" for s in range(n_spots)])

    pst = PairedSharedTensors(spot_emb, gene_emb, y, groups, device="cpu")

    # A fold/split uses a subset of spots and a subset of genes.
    spot_rows = torch.tensor([2, 5, 7, 9], dtype=torch.long)   # 4 local spots
    gene_cols = torch.tensor([1, 3, 4],    dtype=torch.long)   # 3 local genes
    n_sub = gene_cols.numel()
    n_pairs = spot_rows.numel() * n_sub
    pair_idx = torch.arange(n_pairs)                           # row-major, no shuffle

    xb, yb, rows = pst.gather(pair_idx, spot_rows, gene_cols, n_sub)
    for i in range(n_pairs):
        s = int(spot_rows[i // n_sub]); g = int(gene_cols[i % n_sub])
        assert rows[i].item() == s, "spot row mis-decoded"
        assert torch.allclose(xb[i, :spot_dim], torch.full((spot_dim,), float(s))), "spot block wrong"
        assert torch.allclose(xb[i, spot_dim:], torch.full((gene_dim,), float(g))), "gene block wrong"
        assert yb[i].item() == s * 1000 + g, "target round-trip failed"

    # Row-major reshape contract (unshuffled pass -> (n_local_spots, n_sub_genes)).
    y2d = yb.numpy().reshape(spot_rows.numel(), n_sub)
    expected = (spot_rows.numpy()[:, None] * 1000 + gene_cols.numpy()[None, :])
    assert np.array_equal(y2d, expected), "row-major reshape broken"

    # Centering touches the spot block only.
    spot_mean = torch.full((spot_dim,), 2.0)
    xb2, _, _ = pst.gather(pair_idx, spot_rows, gene_cols, n_sub, spot_mean=spot_mean)
    assert torch.allclose(xb2[:, :spot_dim], xb[:, :spot_dim] - 2.0), "centering not applied to spot block"
    assert torch.allclose(xb2[:, spot_dim:], xb[:, spot_dim:]), "centering leaked into gene block"

    print("paired loader correctness test: PASS")


def run_block_gather_correctness_test():
    """Invariants for PairedSharedTensors.gather_block (batched GeneHyperMLP path)
    on the same constructed ground truth: spot_emb row s = [s]*spot_dim, gene_emb
    row g = [g]*gene_dim, y[s, g] = s*1000 + g."""
    n_spots, spot_dim, n_genes, gene_dim = 12, 4, 5, 3
    spot_emb = np.tile(np.arange(n_spots)[:, None], (1, spot_dim)).astype(np.float32)
    gene_emb = np.tile(np.arange(n_genes)[:, None], (1, gene_dim)).astype(np.float32)
    y = (np.arange(n_spots)[:, None] * 1000 + np.arange(n_genes)[None, :]).astype(np.float32)
    groups = np.array([f"slide{s % 3}" for s in range(n_spots)])

    pst = PairedSharedTensors(spot_emb, gene_emb, y, groups, device="cpu")

    spot_rows = torch.tensor([2, 5, 7, 9], dtype=torch.long)   # 4 spots
    gene_cols = torch.tensor([1, 3, 4],    dtype=torch.long)   # 3 genes
    Xs, E, Yb, rows = pst.gather_block(spot_rows, gene_cols)

    assert Xs.shape == (4, spot_dim), "spot block wrong shape"
    assert E.shape  == (3, gene_dim), "gene block wrong shape"
    assert Yb.shape == (4, 3),        "target block wrong shape"
    for i, s in enumerate(spot_rows.tolist()):
        assert rows[i].item() == s, "spot row mis-decoded"
        assert torch.allclose(Xs[i], torch.full((spot_dim,), float(s))), "spot row content wrong"
        for j, g in enumerate(gene_cols.tolist()):
            assert torch.allclose(E[j], torch.full((gene_dim,), float(g))), "gene row content wrong"
            assert Yb[i, j].item() == s * 1000 + g, "target lookup wrong"

    # Centering touches the spot block only.
    spot_mean = torch.full((spot_dim,), 2.0)
    Xs2, E2, _, _ = pst.gather_block(spot_rows, gene_cols, spot_mean=spot_mean)
    assert torch.allclose(Xs2, Xs - 2.0), "centering not applied to spot block"
    assert torch.allclose(E2, E),         "centering leaked into gene block"

    print("block gather correctness test: PASS")


if __name__ == "__main__":
    run_paired_loader_correctness_test()
    run_block_gather_correctness_test()
