# src/data/splitting.py
#
# Group-aware data splitting for spatial transcriptomics, on two axes.
#
# Spot axis: train, val and test are split by donor (patient), never by slide.
# Two slides of one donor are near-duplicates, so a donor sitting on both sides of
# the boundary inflates the score. cv_donor_folds turns an ordered donor list into
# cross-validation folds.
#
# Gene axis, used by the GCM only: a gene list is split by whole chromosome, so a
# held-out gene shares no chromosome with any training gene. This measures
# generalization to genes the model never saw.
#
# Usage:
#   from src.data.splitting import load_donor_order, cv_donor_folds
#   donors = load_donor_order("snake/config/donor_slide_map.yaml", "10x_TuPro")
#   folds  = cv_donor_folds(donors, "lopo_next_val")
#
#   from src.data.splitting import gene_single_split, load_gene_chromosome_map
#   gene_chrom_map = load_gene_chromosome_map("data/space/gene_chromosome_map.csv")
#   train_g, val_g, test_g, info = gene_single_split(genes, gene_chrom_map, spec)

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupKFold


# The GCM holds genes out at two levels, both built by gene_single_split:
#   1. Test genes. chromosome_split carves out whole chromosomes. These genes are
#      never trained on and never used for model selection, so the cross-gene
#      headline measured on them stays clean.
#   2. Validation genes, drawn from what is left. Early stopping can select on
#      them (gcm.early_stop_criterion), which is why they cannot also serve as
#      test genes.


# ─────────────────────────────────────────────────────────────────────────────
# Spot-level splitting (per-gene pipeline and GCM)
# ─────────────────────────────────────────────────────────────────────────────



def get_cv_splitter(strategy):
    """
    Return a scikit-learn CV splitter for group-aware cross-validation.

    `strategy` is "<N>fold", e.g. "5fold". GroupKFold is deterministic, so no seed is
    needed for reproducibility.
    """
    if strategy.endswith("fold"):
        n_splits = int(strategy.replace("fold", ""))
        return GroupKFold(n_splits=n_splits)
    raise ValueError(f"Unknown cv_strategy: {strategy!r}. Expected '<N>fold'.")


# ─────────────────────────────────────────────────────────────────────────────
# Donor-level grouping (leakage-safe CV) — see snake/config/donor_slide_map.yaml
# ─────────────────────────────────────────────────────────────────────────────
# Slides of one donor (replicate serial sections, or two lesions of one patient)
# are near-duplicates in both H&E and expression. A donor whose slides straddle
# the train/test boundary therefore biases the score upward, so the split groups
# by donor. This matches the cohort owners' published cross-validation protocol,
# and the DeepSpot and AESTETIK cross-sample papers.
#
# Donor grouping applies to the split only. The metric still aggregates per slide:
# one correlation per (slide, gene), then a nanmean over slides. The call site
# keeps the two levels apart. SharedTensors.groups holds the slide label and
# drives the metric; the donor labels below drive the fold registry and the
# donor-to-spot-row lookup. Setting SharedTensors.groups to the donor label would
# silently pool a donor's slides into one correlation and shift every reported
# number.
#
# The map file is laid out as {dataset: {donor: [slide, ...]}}.

def load_slide_to_donor(map_path, dataset):
    """Return {slide: donor} for `dataset`, or None if the dataset is absent from
    the map. None means donor == slide (1:1 datasets like TLS, where each slide is
    its own patient) — handled by donor_labels() below. Inverts the
    {donor: [slides]} layout of donor_slide_map.yaml."""
    with open(map_path) as f:
        m = yaml.safe_load(f) or {}
    block = m.get(dataset)
    if not block:
        return None
    slide_to_donor = {}
    for donor, slides in block.items():
        for s in slides:
            slide_to_donor[str(s)] = str(donor)
    return slide_to_donor


def donor_labels(sample_labels, slide_to_donor):
    """Per-spot donor label from per-spot slide label. slide_to_donor=None =>
    donor == slide (identity, for 1:1 datasets). Raises if a slide is unmapped, so
    a typo / missing donor fails loudly rather than silently leaking."""
    sample_labels = np.asarray(sample_labels)
    if slide_to_donor is None:
        return sample_labels.astype(str)
    missing = sorted(set(map(str, sample_labels)) - set(slide_to_donor))
    if missing:
        raise KeyError(
            f"slides absent from the donor map: {missing}. Add them under "
            f"this dataset in snake/config/donor_slide_map.yaml."
        )
    return np.array([slide_to_donor[str(s)] for s in sample_labels])




def load_donor_order(map_path, dataset):
    """Ordered donor-id list for `dataset` as written in donor_slide_map.yaml
    (PyYAML preserves insertion order). None if the dataset is absent from the map
    (1:1 donor==slide datasets like TLS) — the caller then falls back to sorted
    unique donors. This order fixes the lopo_next_val rotation (val = next donor)."""
    with open(map_path) as f:
        m = yaml.safe_load(f) or {}
    block = m.get(dataset)
    if not block:
        return None
    return [str(d) for d in block.keys()]


def donor_ids_to_spot_idx(donor_per_spot, donor_ids):
    """Global spot row indices whose donor is in `donor_ids` (preserves adata row
    order). The driver uses this to turn a fold's donor-id lists into the spot
    index arrays the training loop consumes."""
    donor_per_spot = np.asarray(donor_per_spot).astype(str)
    wanted = list({str(d) for d in donor_ids})
    return np.where(np.isin(donor_per_spot, wanted))[0]


def _assert_fold_contract(folds, donors, *, lopo):
    seen_test = []
    for f in folds:
        test = set(map(str, f["test"]))
        seen_test.append(tuple(sorted(test)))
        for inner in f["inner_splits"]:
            tr, va = set(map(str, inner["train"])), set(map(str, inner["val"]))
            assert tr.isdisjoint(va),   f"fold {f['fold']}: train and val overlap"
            assert tr.isdisjoint(test), f"fold {f['fold']}: train and test overlap"
            assert va.isdisjoint(test), f"fold {f['fold']}: val and test overlap"
    if lopo:
        covered = set().union(*[set(map(str, f["test"])) for f in folds])
        assert covered == set(map(str, donors)), (
            "lopo: test sets must cover every donor "
            f"(covered {sorted(covered)} vs donors {sorted(map(str, donors))})")
        assert len(seen_test) == len(set(seen_test)), "lopo: a donor is test more than once"


def _lopo_next_val(donors, **params):
    """Leave-one-patient-out. For donor i: test=[i], one inner split with
    val=[next donor in the list, wrapping], train=the remaining K-2 donors."""
    K = len(donors)
    if K < 3:
        raise ValueError(
            f"lopo_next_val needs >= 3 donors (train = K-2 >= 1); got K={K}: {donors}. "
            f"Use a dataset with more donors, or the fixed_holdout_kfold strategy.")
    folds = []
    for i, d in enumerate(donors):
        val = [donors[(i + 1) % K]]
        train = [x for x in donors if x != d and x != val[0]]
        folds.append({"fold": i, "test": [d],
                      "inner_splits": [{"train": train, "val": val}]})
    return folds


def _fixed_holdout_kfold(donors, *, test_donors=None, k=5, **params):
    """Hold out a fixed test_donors set, then GroupKFold(k) over the remaining
    donors. Every fold shares that one test set, so the spread across folds is k
    models scoring a single patient, not variation across patients."""
    if not test_donors:
        raise ValueError(
            "fixed_holdout_kfold requires datasets.{ds}.test_donors (the fixed "
            "held-out patient[s]). Set it in config.yaml, or use lopo_next_val.")
    test_set = set(map(str, test_donors))
    test = [d for d in donors if str(d) in test_set]
    missing = test_set - set(map(str, donors))
    if missing:
        raise ValueError(f"test_donors absent from the donor list: {sorted(missing)}")
    dev = [d for d in donors if d not in set(test)]
    if len(dev) < k:
        raise ValueError(
            f"fixed_holdout_kfold needs >= {k} dev donors after holding out {test}; "
            f"got {len(dev)}: {dev}.")
    # GroupKFold over the donor list, so whole donors land together in a fold. It
    # balances folds by donor count, not by spot count, so fold sizes in spots can
    # come out uneven.
    dev_arr = np.array(dev)
    folds = []
    for fold, (tr, va) in enumerate(GroupKFold(n_splits=k).split(dev_arr, groups=dev_arr)):
        folds.append({"fold": fold, "test": list(test),
                      "inner_splits": [{"train": dev_arr[tr].tolist(),
                                        "val": dev_arr[va].tolist()}]})
    return folds


_CV_STRATEGIES = {
    "lopo_next_val":       _lopo_next_val,
    "fixed_holdout_kfold": _fixed_holdout_kfold,
}


def cv_donor_folds(donors, strategy, **params):
    """Build the outer CV folds for the named donor-level strategy. `donors` is the
    ordered donor-id list (see load_donor_order); that order fixes which donor lands
    in which fold. Returns the fold list described at the top of this section, and
    raises ValueError for an unknown strategy name."""
    if strategy not in _CV_STRATEGIES:
        raise ValueError(
            f"Unknown cv strategy {strategy!r}. Available: {sorted(_CV_STRATEGIES)}.")
    donors = [str(d) for d in donors]
    folds = _CV_STRATEGIES[strategy](donors, **params)
    _assert_fold_contract(folds, donors, lopo=is_lopo_strategy(strategy))
    return folds


def is_lopo_strategy(strategy):
    """True for leave-one-patient-out strategies, where each fold tests a distinct
    donor and the folds together cover every donor. When true the driver reports
    the per-donor points with inter-donor t-statistics; otherwise it reports the
    fold mean and standard deviation."""
    return strategy.startswith("lopo")


# ─────────────────────────────────────────────────────────────────────────────
# Gene-axis splitting (Gene-Conditioned MLP)
# ─────────────────────────────────────────────────────────────────────────────
#
# Split a gene list into train and heldout by whole chromosome, so a heldout gene
# shares no chromosome with any training gene. Two modes:
#   - explicit: the caller names the chromosomes (e.g. heldout_chroms=['21'])
#   - greedy:   grow the heldout set one chromosome at a time until a target
#               fraction of genes is reached.


def _chrom_sort_key(c):
    """Natural ordering for chromosome labels: 1, 2, ..., 22, X, Y, then anything else."""
    if c in ('X', 'Y'):
        return (1, c)
    try:
        return (0, int(c))
    except ValueError:
        return (2, c)


def chromosome_split(
    gene_list,
    gene_chrom_map,
    *,
    heldout_chroms=None,
    target_heldout_fraction=None,
    strategy='smallest_first',
    sex_chroms=('X', 'Y'),
):
    """
    Split genes into train and heldout by whole-chromosome assignment.

    Pass exactly one of `heldout_chroms` or `target_heldout_fraction`:

    explicit mode (heldout_chroms=['21']):
        The caller names the chromosomes.

    greedy mode (target_heldout_fraction=0.30):
        The heldout set grows one chromosome at a time until the target fraction
        of genes is reached. `strategy` controls the order:
            'smallest_first' -> chr21 first, then chr22, chr20, ...  (default)
            'largest_first'  -> chr1 first, then chr2, ...

    Sex chromosomes (X, Y) always go to train, and so do genes whose chromosome
    is unknown. Two quiet consequences: a requested heldout chromosome matching no
    gene only warns, and in greedy mode the target fraction is measured against the
    whole gene list while only autosomes are eligible, so the achieved fraction
    falls short when many genes sit on X, Y or 'unknown'.

    Parameters
    ----------
    gene_list : list[str]
        All genes to split.
    gene_chrom_map : dict[str, str]
        Maps gene name -> chromosome string (e.g. '1', 'X', '21', 'unknown').
    heldout_chroms : list[str] or None
        Explicit list of chromosomes to hold out. Mutually exclusive with
        target_heldout_fraction.
    target_heldout_fraction : float in (0, 1) or None
        Target fraction of genes to hold out. Mutually exclusive with
        heldout_chroms.
    strategy : {'smallest_first', 'largest_first'}
        Order in which chromosomes are added to the heldout set (greedy only).
    sex_chroms : tuple[str, ...]
        Chromosomes that always go to TRAIN (never holdable out).

    Returns
    -------
    train_genes : list[str]
        Genes assigned to TRAIN, in the order they appear in gene_list.
    heldout_genes : list[str]
        Genes assigned to HELDOUT, in the order they appear in gene_list.
    info : dict
        Diagnostic info: heldout_chroms, train_chroms, n_train_genes,
        n_heldout_genes, heldout_fraction.
    """
    # ── 1. bucket genes by chromosome ────────────────────────────────────────
    gene_to_chrom = {g: str(gene_chrom_map.get(g, 'unknown')) for g in gene_list}
    chrom_to_genes = {}
    for g, c in gene_to_chrom.items():
        chrom_to_genes.setdefault(c, []).append(g)

    sex_set = set(sex_chroms)
    autosomes = {
        c: gs for c, gs in chrom_to_genes.items()
        if c not in sex_set and c != 'unknown'
    }

    # ── 2. decide which chromosomes go to HELDOUT ────────────────────────────
    if (heldout_chroms is None) == (target_heldout_fraction is None):
        raise ValueError(
            "Pass exactly one of heldout_chroms / target_heldout_fraction"
        )

    if heldout_chroms is not None:
        heldout_set = {str(c) for c in heldout_chroms}
        if heldout_set & sex_set:
            raise ValueError(
                f"Sex chromosomes not allowed in heldout: {heldout_set & sex_set}"
            )
        missing = heldout_set - set(autosomes.keys())
        if missing:
            print(
                f"Warning: requested heldout chromosomes have no genes "
                f"in gene_list: {missing}"
            )
    else:
        # greedy mode -- pick chromosome order, then accumulate until target met
        if strategy == 'smallest_first':
            order = sorted(autosomes.keys(), key=lambda c: len(autosomes[c]))
        elif strategy == 'largest_first':
            order = sorted(autosomes.keys(), key=lambda c: -len(autosomes[c]))
        else:
            raise ValueError(
                f"Unknown strategy: {strategy}. "
                f"Expected 'smallest_first' or 'largest_first'."
            )

        target_n = target_heldout_fraction * len(gene_list)
        heldout_set, running = set(), 0
        for c in order:
            heldout_set.add(c)
            running += len(autosomes[c])
            if running >= target_n:
                break

    # ── 3. assign genes (preserving input order within each split) ───────────
    heldout_genes = [g for g in gene_list if gene_to_chrom[g] in heldout_set]
    heldout_set_g = set(heldout_genes)
    train_genes   = [g for g in gene_list if g not in heldout_set_g]

    # ── 4. invariants ────────────────────────────────────────────────────────
    assert set(train_genes).isdisjoint(heldout_genes)
    assert set(train_genes) | set(heldout_genes) == set(gene_list)

    info = {
        'heldout_chroms':   sorted(heldout_set, key=_chrom_sort_key),
        'train_chroms':     sorted(
            (set(autosomes) - heldout_set) | sex_set, key=_chrom_sort_key
        ),
        'n_heldout_genes':  len(heldout_genes),
        'n_train_genes':    len(train_genes),
        'heldout_fraction': len(heldout_genes) / len(gene_list),
    }
    return train_genes, heldout_genes, info




def gene_single_split(genes, gene_chrom_map, spec, *, default_seed=42,
                      sex_chroms=('X', 'Y')):
    """
    Partition `genes` once into test-heldout, val and train, on the gene axis.

    Every donor fold reuses the same partition, which keeps the cross-gene headline
    (test donor x test-heldout genes) comparable across donors. `spec` is one
    gcm.chrom_split_specs entry:

      test genes, handed straight to chromosome_split:
          mode='explicit' + heldout_chroms=[...]                       or
          mode='greedy'   + target_heldout_fraction (+ greedy_strategy)
      val genes, the spec['val'] sub-block, drawn from the remaining train genes:
          mode='random_fraction' + fraction (+ optional seed):
              a seeded random subset of the training genes, any chromosome. The
              seed falls back to `default_seed`, the config-level seed, and never
              to the per-member ensemble seed, so every ensemble member sees the
              same gene split and members differ only by init and shuffle order.
          mode='chroms' + chroms=[...]:
              whole validation chromosomes, disjoint from the heldout set.

    Returns (train_genes, val_genes, test_heldout_genes, info). Input order is
    preserved within each returned list.
    """
    # ── 1. test-heldout carve (reuse chromosome_split) ───────────────────────
    mode = spec.get('mode', 'explicit')
    if mode == 'explicit':
        cv_genes, test_heldout_genes, chrom_info = chromosome_split(
            genes, gene_chrom_map, heldout_chroms=spec['heldout_chroms'],
            sex_chroms=sex_chroms)
    elif mode == 'greedy':
        cv_genes, test_heldout_genes, chrom_info = chromosome_split(
            genes, gene_chrom_map,
            target_heldout_fraction=spec['target_heldout_fraction'],
            strategy=spec.get('greedy_strategy', 'smallest_first'),
            sex_chroms=sex_chroms)
    else:
        raise ValueError(f"gene_single_split: unknown test-split mode {mode!r} "
                         f"(expected 'explicit' or 'greedy').")

    # ── 2. val carve from the TRAIN (cv) genes ───────────────────────────────
    val_cfg = spec.get('val')
    if not val_cfg:
        raise ValueError("gene_single_split: spec is missing a 'val' sub-block "
                         "(mode: random_fraction | chroms).")
    val_mode = val_cfg.get('mode')
    chrom_of = {g: str(gene_chrom_map.get(g, 'unknown')) for g in genes}
    if val_mode == 'random_fraction':
        frac = float(val_cfg['fraction'])
        if not (0.0 < frac < 1.0):
            raise ValueError(f"val random_fraction must be in (0, 1); got {frac}.")
        seed = int(val_cfg.get('seed', default_seed))
        n_val = int(round(frac * len(cv_genes)))
        n_val = max(1, min(n_val, len(cv_genes) - 1))   # keep >=1 train and >=1 val gene
        rng = np.random.default_rng(seed)
        chosen = set(rng.choice(np.array(cv_genes), size=n_val, replace=False).tolist())
        val_info = {"mode": "random_fraction", "fraction": frac, "seed": seed}
    elif val_mode == 'chroms':
        val_chroms = {str(c) for c in val_cfg['chroms']}
        bad = val_chroms & set(map(str, chrom_info['heldout_chroms']))
        if bad:
            raise ValueError(f"val chroms overlap the test-heldout chroms: {sorted(bad)}.")
        chosen = {g for g in cv_genes if chrom_of[g] in val_chroms}
        if not chosen:
            raise ValueError(f"val chroms {sorted(val_chroms)} match no training genes.")
        val_info = {"mode": "chroms",
                    "chroms": sorted(val_chroms, key=_chrom_sort_key)}
    else:
        raise ValueError(f"gene_single_split: unknown val mode {val_mode!r} "
                         f"(expected 'random_fraction' or 'chroms').")

    val_genes   = [g for g in cv_genes if g in chosen]
    train_genes = [g for g in cv_genes if g not in chosen]

    # ── 3. invariants ────────────────────────────────────────────────────────
    assert set(train_genes).isdisjoint(val_genes)
    assert set(train_genes).isdisjoint(test_heldout_genes)
    assert set(val_genes).isdisjoint(test_heldout_genes)
    assert set(train_genes) | set(val_genes) | set(test_heldout_genes) == set(genes)

    info = {
        "test_heldout_chroms":  list(chrom_info['heldout_chroms']),
        "train_chroms":         list(chrom_info['train_chroms']),
        "val":                  val_info,
        "n_train_genes":        len(train_genes),
        "n_val_genes":          len(val_genes),
        "n_test_heldout_genes": len(test_heldout_genes),
    }
    return train_genes, val_genes, test_heldout_genes, info


def load_gene_chromosome_map(path):
    """
    Load a gene -> chromosome dict from a CSV with columns 'gene' and 'chromosome'.
    """
    df = pd.read_csv(path)
    return dict(zip(df['gene'], df['chromosome']))


# ─────────────────────────────────────────────────────────────────────────────
# Self-test: python -m src.data.splitting
# Checks the fold-registry contract and the gene single-split invariants.
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ── Donor CV registry ────────────────────────────────────────────────────
    donors = ["P10", "P11", "P15", "P16", "P19", "P24"]      # K = 6
    lopo = cv_donor_folds(donors, "lopo_next_val")
    assert len(lopo) == 6
    for i, f in enumerate(lopo):
        assert f["test"] == [donors[i]]
        inner = f["inner_splits"][0]
        assert inner["val"] == [donors[(i + 1) % 6]], inner
        assert len(inner["train"]) == 6 - 2
        assert donors[i] not in inner["train"] and inner["val"][0] not in inner["train"]
    print("lopo_next_val OK | test rotation:", [f["test"][0] for f in lopo],
          "| val rotation:", [f["inner_splits"][0]["val"][0] for f in lopo])

    fhk = cv_donor_folds(donors, "fixed_holdout_kfold", test_donors=["P19"], k=5)
    assert all(f["test"] == ["P19"] for f in fhk)
    val_union = set().union(*[set(f["inner_splits"][0]["val"]) for f in fhk])
    assert val_union == set(donors) - {"P19"}, val_union
    print("fixed_holdout_kfold OK |", len(fhk), "folds, dev covered once")

    for bad in (["A", "B"],):                                # K < 3 guard
        try:
            cv_donor_folds(bad, "lopo_next_val"); raise AssertionError("expected ValueError")
        except ValueError:
            print("K<3 guard OK")

    # ── Gene single split ────────────────────────────────────────────────────
    genes = [f"g{c}_{j}" for c in ["1", "2", "3", "21"] for j in range(5)]
    cmap = {g: g.split("_")[0][1:] for g in genes}           # "g21_0" -> "21"
    spec_rand = {"mode": "explicit", "heldout_chroms": ["21"],
                 "val": {"mode": "random_fraction", "fraction": 0.2, "seed": 0}}
    tr, va, te, info = gene_single_split(genes, cmap, spec_rand, default_seed=42)
    assert set(te) == {g for g in genes if cmap[g] == "21"}
    assert set(tr).isdisjoint(va) and set(tr).isdisjoint(te) and set(va).isdisjoint(te)
    assert set(tr) | set(va) | set(te) == set(genes)
    # frozen across the ensemble --seed: default_seed is ignored when val.seed is set
    _, va_b, _, _ = gene_single_split(genes, cmap, spec_rand, default_seed=999)
    assert va == va_b, "val draw must ignore default_seed when val.seed is pinned"
    print("gene_single_split random_fraction OK |", info["n_train_genes"], "train /",
          info["n_val_genes"], "val /", info["n_test_heldout_genes"], "test")

    spec_chr = {"mode": "explicit", "heldout_chroms": ["21"],
                "val": {"mode": "chroms", "chroms": ["3"]}}
    tr2, va2, te2, info2 = gene_single_split(genes, cmap, spec_chr)
    assert set(va2) == {g for g in genes if cmap[g] == "3"}
    print("gene_single_split chroms OK |", info2["val"])
    print("ALL SELF-TESTS PASSED")
