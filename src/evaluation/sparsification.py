# src/evaluation/sparsification.py
#
# Sparsification curves: withhold the least certain predictions and recompute the error of what
# is left. A head whose uncertainty is informative sees that error fall.
#
# Two protocols, and the difference matters. `pooled_curves` sorts every entry at once, which lets
# a head score well by knowing which GENES are hard rather than which predictions are wrong.
# `per_group_curves` removes inside each (gene, slide), which is the protocol the paper reports.
#
# Four removal orders per curve (`spars_scores`): the head's own uncertainty, an oracle that
# removes by the true error, and the predicted mean in both directions, that last one being the
# baseline a practitioner already has with no uncertainty head at all.
#
# Read by snake/scripts/canals/canal2_uncertainty_precompute.py, which builds the arrays, and by
# notebooks/reproducing_the_numbers_on_any_predictions.ipynb, which shows the same call on one slide.
import numpy as np
import pandas as pd


class SparsificationPrecisionError(RuntimeError):
    """per_group_curves cannot compute the per-(gene, slide) curves on this error array.

    Raised INSTEAD of an assert at the one check inside per_group_curves whose failure is provably
    NUMERICAL rather than logical. At f = 0 nothing has been removed, so every score's C[:, 0] is
    the same mathematical quantity -- the group's own mean error -- and the four scores can differ
    only in the ORDER their errors were summed in. A mismatch there is therefore cancellation in the
    running cumsum, never a wrong ranking, and it says the error array spans a dynamic range this
    accumulation cannot carry. It fires on heads whose predicted spread has a heavy tail, the
    lognormal posterior mean of the Laplace count heads being the case seen here.

    Caught per CASE, so such a run still writes everything that does not average raw errors: the
    rank statistics (pair_partial, spot_partial, bin_stats, axis_stats) and the pooled curves, which
    re-sum the kept tail with .mean() and never accumulate across groups.

    The other check in per_group_curves, group contiguity after the key sort, stays a bare assert:
    it can only fail on a logic error, so it must abort the run.
    """


def spars_scores(err_col):
    """The removal orders every sparsification curve is built from: name -> (column, high first?).

    `pred` and `oracle` are the two the canal always had. `mu_high` / `mu_low` are the PREDICTED
    MEAN used alone as the score, i.e. the baseline a practitioner already has without any
    uncertainty head, and the only way to make "the uncertainty is redundant with the mean"
    quantitative: an informative head has to beat mu on its own. Both directions, because which end
    of mu carries the error depends on the error -- |y - mu| grows with mu, while |log y - log mu|
    is largest where mu is smallest -- so neither direction alone is THE mu baseline.

    There is deliberately no `random` entry. Random removal is unbiased for the mean of the kept
    set, pooled and stratified alike, so its curve is exactly flat at the f = 0 value and a figure
    normalized by that value draws it as a horizontal line at 1.0. Sampling one permutation would
    only add noise to a quantity that is known in closed form."""
    return {"pred":    ("uncertainty", True),
            "oracle":  (err_col,       True),
            "mu_high": ("mu",          True),
            "mu_low":  ("mu",          False)}


def pooled_curves(err, values, scores, fracs):
    """POOLED removal: one global sort over every entry, remove the top f, mean error of the rest.

    Its flaw IS the reason to show it: the most uncertain entries are also the high-mean ones, so
    the mu-composition of the remainder drifts as f grows and part of the curve is that drift --
    which is exactly what the mu_high curve of the same panel measures.

    `values` is a {column: 1-D array} dict, not a DataFrame, and the caller puts the SAME float64
    `err` array in it under the error column's name. That is not cosmetic: np.argsort defaults to
    an unstable quicksort, so sorting the float32 column instead of the float64 error could break
    ties in a different order and move the oracle curve. Descending is likewise spelled `-v` rather
    than a reversed ascending sort, so the tie order stays the one every pooled curve of this canal
    has always had."""
    n = len(err)
    out = {}
    for name, (col, desc) in scores.items():
        v = values[col]
        e = err[np.argsort(-v if desc else v)]     # most uncertain (or worst, or largest mu) first
        out[name] = np.array([e[int(f * n):].mean() for f in fracs])
    return out


def per_group_curves(keys, values, err, scores, fracs, rm_cols, agg_code, n_agg):
    """ONE sparsification curve PER (gene, slide), removal happening inside `rm_cols`.

    `keys` is {column: array} for the grouping columns, `values` {column: array} for the scored
    ones -- same shape of argument as pooled_curves, and for the same reason: the caller puts the
    float64 `err` into `values` under the error column's name, so ties break identically.

    This is the protocol the paper reports, and it is the operational analogue of the gene-axis
    partial correlation. Two instantiations:
        rm_cols = ["gene", "slide"]          -> `gene_slide`      (no mu-bins)
        rm_cols = ["gene", "slide", "bin"]   -> `gene_slide_bin`  (inside a mu-bin)
    In BOTH the curve is produced per (gene, slide) -- `agg_code` -- so the two are directly
    comparable; only the unit the removal fraction applies to changes.

    ── REMOVAL ──────────────────────────────
    Inside a removal unit of n entries, an entry is dropped when its within-unit percentile rank
    of the score is <= f. So `gene_slide` drops floor(f*n) of the group, and `gene_slide_bin`
    drops floor(f*n_b) of EACH bin b, which is what makes the bins lose equal FRACTIONS despite
    holding different counts.

    ── THE KEY TRICK ────────────────────────────────────────────────────────────────────────
    `key = agg_code + pct` with pct in (0, 1] puts every (gene, slide) in its own unit interval:
    group g owns (g, g+1], and g+1's smallest key is strictly above g+1, so the blocks never
    overlap. Sorting once by that key therefore orders BY GROUP and, inside a group, by removal
    order -- even for `gene_slide_bin`, where pct is a WITHIN-BIN rank: the cut is `pct > f`
    whatever bin a row sits in, so one cut per group drops the top f of every one of its bins.
    Then the curve is 19 vectorized searchsorted calls, not a groupby per fraction.

    ── NORMALIZATION ────────────────────────────────────────────────────────────────────────
    At f = 0 nothing is removed, so C[:, 0] IS the group's own mean error, and every score shares
    that divisor. That is what makes `pred >= oracle` survive: within a group the oracle minimizes
    the kept sum at an identical kept count, and dividing all four scores by the same number keeps
    the inequality. Returns (curves, divisors) -- the divisors are saved so the normalization is
    auditable without a rerun (their n-weighted mean must equal mean_err).

    ── WHERE IT CAN BE UNDEFINED ────────────────────────────────────────────────────────────
    A group empties (n_kept = 0 -> NaN) only when the score is CONSTANT over the whole unit: all
    ranks then tie in the middle, so nothing is removed below f ~ 0.5 and everything is removed at
    once above it. That is the gene-homoscedastic head, whose sigma is one number per gene, and it
    happens to ALL its groups at the same fractions -- so the head goes blank wholesale and is
    caught, never silently mixed. With distinct values and n >= 60, f <= 0.9 always keeps >= 6
    entries. The caller prints the per-fraction support so a shrinking gene set is visible rather
    than hidden inside a nanmean."""
    out, divisors = {}, None
    gidx = np.arange(n_agg, dtype=np.float64)
    rm_keys = [keys[k] for k in rm_cols]
    for name, (col, desc) in scores.items():
        # Rank the array from `values`, NOT a column of the frame: the caller puts the float64 err
        # in there under the error column's name, so the oracle ranks the same numbers it sums.
        # Ranking the float32 column instead would tie errors that differ in float64, letting the
        # oracle miss the strictly largest ones -- tiny (1e-7 relative), but it is precisely the
        # `pred >= oracle` invariant this whole design rests on, so it is made exact.
        pct = pd.Series(values[col]).groupby(rm_keys, sort=False).rank(
            ascending=not desc, pct=True).to_numpy()
        key = agg_code + pct
        order = np.argsort(key)
        ks = key[order]
        # The groups must come out CONTIGUOUS in the sorted order. If they ever did not, the cut
        # below would be a global sort by the score with the group as a tiebreak -- the pooled
        # protocol wearing this one's name, which no output would reveal. Note this asserts the
        # GROUPING, not the sort: `ks` is sorted by construction, so checking it would test nothing.
        assert np.all(np.diff(agg_code[order]) >= 0), "groups not contiguous after the key sort"
        cum = np.concatenate([[0.0], np.cumsum(err[order])])
        end = np.searchsorted(ks, gidx + 1.0, side="right")     # one past each group's last row
        C = np.empty((n_agg, len(fracs)), dtype=np.float64)
        for q, f in enumerate(fracs):
            cut = np.searchsorted(ks, gidx + f, side="right")   # first KEPT row: pct > f
            n_kept = end - cut
            with np.errstate(invalid="ignore", divide="ignore"):
                C[:, q] = np.where(n_kept > 0, (cum[end] - cum[cut]) / n_kept, np.nan)
        if divisors is None:
            divisors = C[:, 0].copy()          # f = 0 removes nothing: the group's own mean error
        else:
            # Every score MUST start from the same value, since at f = 0 none of them has removed
            # anything. This is the invariant the whole "pred >= oracle after aggregation" argument
            # rests on, so it is checked rather than assumed. Not an assert: the mathematical value
            # is identical for the four scores by construction, so only the summation order can
            # differ and a failure is a precision statement about the error array -- which the
            # caller can act on, by dropping this case's per-group curves instead of the whole run.
            if not np.allclose(C[:, 0], divisors, rtol=1e-9, equal_nan=True):
                with np.errstate(invalid="ignore", divide="ignore"):
                    dev = np.nanmax(np.abs(C[:, 0] - divisors) / np.abs(divisors))
                raise SparsificationPrecisionError(
                    f"score {name!r} starts from a different f=0 value (max relative deviation "
                    f"{dev:.3g}); the scores sum the SAME errors in different orders, so this is "
                    f"cancellation in the cumsum, not a ranking error")
        # Guard the division: a group whose every error is exactly 0 would give inf and poison the
        # gene average silently. It does not happen in practice (mu > 0 for the count families,
        # and an exactly-zero Gaussian residual over a whole group is not a thing), which is
        # precisely why it must not be left to chance.
        bad = ~np.isfinite(divisors) | (divisors <= 0)
        C[bad, :] = np.nan
        C /= np.where(bad, 1.0, divisors)[:, None]
        out[name] = C
    return out, divisors
