# src/data/loading.py
#
# Gene-embedding loaders for the GCM (Gene-Conditioned MLP).
#
#   load_gene_embeddings(path, gene_list)
#       Load gene embeddings (SPACE network / sequence / space variants) from a
#       CSV indexed by gene name.
#   permute_gene_embeddings(gene_embs, seed)
#       Shuffle the gene -> embedding mapping. A control for whether the model
#       uses the embeddings at all.
#
# The embeddings are uploaded into PairedSharedTensors (src/data/dataset.py) and
# concatenated with the spot embedding on device.
#
# Spot embeddings and expression are not loaded here. Preprocessing lives in
# src/data/load_and_preprocess.py, which writes the single unified adata.h5ad that
# train.py and train_gcm.py read.

import numpy as np
import pandas as pd


def load_gene_embeddings(path, gene_list):
    """
    Load gene embeddings from a CSV indexed by gene name.

    Parameters
    ----------
    path : str
        Path to a CSV with the first column being the gene name (index_col=0)
        and the remaining columns being embedding dimensions.
    gene_list : list[str]
        Genes to extract. A gene missing from the CSV index raises KeyError,
        naming the first ten missing genes.

    Returns
    -------
    gene_embs : dict[str, np.ndarray (gene_dim,) float32]
        One entry per gene in gene_list.
    gene_dim : int
        Embedding dimension.
    """
    df = pd.read_csv(path, index_col=0)
    missing = [g for g in gene_list if g not in df.index]
    if missing:
        raise KeyError(
            f"{len(missing)} genes from gene_list are missing in {path}: "
            f"{missing[:10]}{'...' if len(missing) > 10 else ''}"
        )

    gene_dim = df.shape[1]
    gene_embs = {g: df.loc[g].values.astype(np.float32) for g in gene_list}
    return gene_embs, gene_dim


def permute_gene_embeddings(gene_embs, seed):
    """
    Return a new dict with the gene -> embedding mapping randomly shuffled.

    A control experiment: if the model uses the gene embeddings, shuffling them
    should cost accuracy. No drop means the embeddings contribute nothing.

    Parameters
    ----------
    gene_embs : dict[str, np.ndarray]
    seed : int

    Returns
    -------
    permuted : dict[str, np.ndarray]
        Same keys as gene_embs, but values reshuffled across keys.
    perm_info : dict
        {'seed': seed, 'permutation_map': {gene: the gene whose embedding it got}}.
        Saved with the run so the permutation can be reproduced. "donor" in the
        code below means the source gene, not a patient.
    """
    rng = np.random.default_rng(seed)
    keys = list(gene_embs.keys())
    perm_idx = rng.permutation(len(keys))
    donor_keys = [keys[i] for i in perm_idx]

    permuted = {g: gene_embs[donor_keys[i]] for i, g in enumerate(keys)}
    perm_info = {
        'seed': int(seed),
        'permutation_map': dict(zip(keys, donor_keys)),
    }
    return permuted, perm_info
