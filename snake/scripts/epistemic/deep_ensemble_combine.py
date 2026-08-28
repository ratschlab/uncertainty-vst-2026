# snake/scripts/epistemic/deep_ensemble_combine.py
#
# Combine the M seed-members of a deep ensemble into ONE consolidated TEST file of
# per-(spot,gene) uncertainty summaries, with the same layout and layer names as
# mc_dropout_posthoc.py so the two are drop-in siblings.
#
# Reads : <model-dir>/seed_{s}/predictions/preds_fold{k}.h5ad, one per member seed s.
# Writes: <model-dir>/ensemble/test_summaries.h5ad
#
# A draw here is a seed-member: M full runs differing only by --seed (init + batch shuffle). The
# CV split is seed-invariant, so all members share the same folds, spots and genes, and each one
# already saved its own predictions. No torch, no checkpoint, no adata reload: read the M member
# files per fold and reduce over members. The family comes from metrics.json plus the spec.
#
# The per-member model mean m_s IS the saved prediction X:
#   Gaussian -> E[log1p-CP10k], the adata.X scale.
#   count    -> mu on the model's own fitted target scale: round(CP_T) for the _cp families,
#               CP_T unrounded for _cp_nr, raw counts for _raw. There is no size factor and no
#               offset; mean and dispersion share that one scale and nothing here rescales either.
# A registered spec's target_source is "X", "cp_counts", "cp_counts_nr" or "raw_counts"; anything
# else fails fast in detect_family() (see MEAN_SCALE_BY_TARGET_SOURCE below).
#
# Stored per (spot, gene), on the model's native mean scale:
#   pred        = mean_s m_s                        (X)
#   epistemic   = var_s  m_s                        (layer "epistemic")
#   y           = the observed target               (layer "y") -- from member 0's own
#                 layers["y"] when present, else rebuilt ONCE from the shared preprocessed adata,
#                 since GCM prediction files hold test rows only and carry no y (see
#                 train_gcm.py::fold_predictions_adata + target_values_for_predictions)
#   + per family, the EXTRA term needed to rebuild the aleatoric:
#       Gaussian -> aleatoric_var = mean_s sigma^2(x)        (= the aleatoric itself)
#       NB       -> mean_alpha_m2 = mean_s (alpha_s * m_s^2)  (the overdispersion term of
#                   the NB variance mu + alpha*mu^2, on the same target scale as m_s)
#       Poisson  -> nothing (aleatoric == pred on the target scale)
#
# POST-HOC RECONSTRUCTION (the job bakes in nothing), identical to mc_dropout. Everything stays
# on the model's own target scale and no constant is applied:
#   pred=X  epi=epistemic
#       aleatoric = X (Poisson) | aleatoric_var (Gaussian) | X + mean_alpha_m2 (NB)
#       total     = aleatoric + epistemic
#
# THE OPTION-A ESTIMATOR (count families only) is deliberately not a layer: it is a pure function
# of the pred layer and one scalar in uns, and this file stores sufficient statistics rather than
# derived quantities. Compute it as
#   div      = 1.0 if uns["deep_ensemble"].get("depth_scale") is None else float(...["depth_scale"])
#   option_A = np.log1p(X / div)
# which is log1p(mean_s m_s / div), NOT mean_s log1p(m_s / div): log1p is concave, so the two
# differ by a Jensen gap, and the first form is the ensemble's own mean-only prediction, exactly
# what the pred layer holds. option_A and the gstar layer are correlated against the same target
# on the same axis, so their difference isolates one thing, the effect of routing the prediction
# through the full predictive distribution. It is the ensemble analogue of the single-model
# "{metric}" vs "{metric}_logmean" pair in atoms.parquet. Gaussian and deterministic families
# have no A/B distinction: their pred is already an E[log1p-CP10k] mean, no depth_scale is
# written for them and there is no gstar layer, so the formula above does not apply to them.
#
# TEST ONLY, ONE FILE. Under LOPO each donor is the held-out TEST donor exactly once, so the
# union of all folds' test spots is every spot once. Each row of the single matrix .h5ad carries
# the summaries from the fold where ITS donor was held out; validation spots are dropped.
#
# COUNT-CANAL EXTENSION (g*, native-CDF PIT bounds, zero mass). For a count family whose members
# already carry predictions/gstar_fold{k}.h5ad (written by DE_gstar_precompute.py as a sibling of
# preds_fold{k}.h5ad: X = g* = E[log1p(Y/depth_scale)], test rows), three more layers are
# consolidated:
#   gstar             = mean_s g*_s = mean_s E_s[log1p(Y/depth_scale)]  (canal-1 g* headline)
#   pit_flo / pit_fhi = mean_s F_lo/F_hi(y; mu_c_s[, alpha_s])          (canal-1 calibration)
#   zero_prob         = mean_s P_s(Y=0)                                 (canal-3 zero detection)
# Each of the three is an expectation under member s's own predictive distribution (a mean, a CDF
# value, a PMF value at 0), so the M-member mixture value is exactly the plain average over
# members, by linearity of expectation over a finite mixture. Epistemic and aleatoric above need
# a second moment and are not plain averages. g* needs the expensive windowed truncated sum
# (count_metrics.count_expected_log1p), hence its own precompute; F_lo/F_hi and P(Y=0) are O(1)
# closed-form and run inline in the same per-member streaming loop, at no extra I/O.
# Which formula applies is resolved per member from whether that member's own alpha is available
# this iteration, not from a family flag: NB uses the native NB CDF
# (calibration_pit.nb_pit_bounds) and P_s(Y=0) = (r_s/(r_s+mu_c_s))^r_s with r_s = 1/alpha_s;
# Poisson uses the native Poisson CDF (calibration_pit.poisson_pit_bounds) and
# P_s(Y=0) = exp(-mu_c_s), needing no alpha. Gaussian heads reach the PIT branch too (want_pit is
# want_gstar OR want_var, see combine_model_dir): their bounds are Phi((y-m_s)/sigma_s) with
# F_lo == F_hi, since a continuous predictive has no probability atom for the observation to
# straddle. They reach neither g* (they already predict on the reporting axis) nor zero_prob (no
# atom at 0), and msemeanonly reaches none of the three: no predictive distribution.
# mu_c_s = m_s, because a member marked uns["count_convention"] == "v2_no_offset" already holds
# mu on the fitted count scale. An unmarked member RAISES and nothing is substituted.
#
# SCOPE GATE. A model dir opts into the extension only if at least one gstar_fold*.h5ad exists
# under some seed_*/predictions/ (one cheap glob, no h5ad read); a dir without them gets none of
# the three extra layers. A gstar file missing or misaligned for one particular (seed, fold)
# skips only THAT member's g* contribution, with a warning rather than a crash, and never touches
# m_s / pred / epistemic. Not gated on the glob: the option-A metadata
# uns["deep_ensemble"]["depth_scale"], written for every count family, because option A needs
# only pred and that scalar.
#
# Usage (repo root, on the cluster; CPU is plenty, this is pure read + numpy):
#   # one (dataset, model): point at the dir holding the seed_{s}/ subdirs
#   python -u snake/scripts/epistemic/deep_ensemble_combine.py \
#       --model-dir results/<dataset>/<run_id>/<gene_set>/r75/poisson_cp
#   # the whole sweep in one call (the shell expands the globs):
#   python -u snake/scripts/epistemic/deep_ensemble_combine.py \
#       --model-dir results/*/*/*/r75/* results_gcm/*/*/r75/*

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc

# Wall-clock instrumentation (like mc_dropout_posthoc.py): every milestone prints
# "[   s]" since start, so the log shows where time goes (per member read, per fold).
_T0 = time.time()
def _el():
    return time.time() - _T0

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)
from src.models.specs import get_spec
from src.data.load_and_preprocess import target_values_for_predictions
from src.evaluation.calibration_pit import nb_pit_bounds, poisson_pit_bounds, gaussian_pit_bounds

# target_source values that mean "this is a count-likelihood model", i.e. one whose SAVED
# prediction X is a count-scale mean mu (on its OWN target scale, see MEAN_SCALE_BY_TARGET_SOURCE)
# rather than a log1p-space mean. "cp_counts"/"cp_counts_nr" are the common-depth CP_T count
# targets, "raw_counts" is the depth-blind one. This must stay identical to the set the scoring
# path routes as counts (src/models/specs.py::COUNT_ROUTED_TARGETS): a target_source missing here
# is silently written as non-count in uns["deep_ensemble"]["count"]/["mean_scale"].
COUNT_TARGET_SOURCES = ("raw_counts", "cp_counts", "cp_counts_nr")

# The mean scale of the SAVED prediction X, keyed by spec.target_source -- the single source of
# truth for uns["deep_ensemble"]["mean_scale"]. Each count family fits its likelihood DIRECTLY on
# its own target, so X is mu on THAT target's scale and nothing rescales it here. Every
# target_source a registered spec can carry must have an entry here; an unlisted one fails fast in
# detect_family() rather than guessing.
MEAN_SCALE_BY_TARGET_SOURCE = {
    "X":                 "log1p_cp10k",           # Gaussian / MSE axis
    "raw_counts":        "mu_raw_counts",         # mu on the raw-count target, depth-blind
    "cp_counts":         "mu_cp_counts",          # mu on the round(CP_T) target (T = count_common_depth)
    "cp_counts_nr":      "mu_cp_counts",          # same CP_T target, unrounded
}


def detect_family(model_dir, seeds):
    """Read the run's own metrics.json (from the first member) + the spec — the single source of
    truth — and return everything family-specific. Mirrors mc_dropout_posthoc.py."""
    with open(os.path.join(model_dir, f"seed_{seeds[0]}", "metrics.json")) as f:
        M = json.load(f)
    model_type = M["model_type"]                 # present even in the compute_final_metrics=False STUB
    spec = get_spec(model_type)
    # target_source from the SPEC, not metrics.json: the stub (compute_final_metrics=False)
    # carries model_type but NOT target_source, and the spec is the single source of truth anyway.
    target_source = spec.target_source
    want_var  = (target_source == "X" and spec.is_probabilistic)   # Gaussian variance head
    want_disp = ("dispersion" in spec.extra_prediction_columns)    # NB dispersion
    # The per-family EXTRA layer (the aleatoric ingredient). None for Poisson / non-probabilistic,
    # whose aleatoric == pred on the model's own target scale.
    extra_layer = "aleatoric_var" if want_var else ("mean_alpha_m2" if want_disp else None)
    if target_source not in MEAN_SCALE_BY_TARGET_SOURCE:
        raise ValueError(
            f"{model_type}: unhandled target_source={target_source!r} -- add it to "
            f"MEAN_SCALE_BY_TARGET_SOURCE (and COUNT_TARGET_SOURCES if it is a count "
            f"likelihood) in deep_ensemble_combine.py before combining this model."
        )
    return dict(model_type=model_type, target_source=target_source,
                count=(target_source in COUNT_TARGET_SOURCES),
                mean_scale=MEAN_SCALE_BY_TARGET_SOURCE[target_source],
                # cp_counts_nr trains on a continuous target, so its PIT uses the smeared CDF
                # F~ (calibration_pit smear=True); a no-op for the integer-target families.
                # Derived from target_source here, as inference_indexed.py and train_loop_gcm.py
                # do, so the call site below cannot hardcode it and be silently wrong.
                smear=(target_source == "cp_counts_nr"),
                want_var=want_var, want_disp=want_disp, extra_layer=extra_layer,
                adata_path=M.get("input_adata"))


def combine_test_fold(seed_paths, fam, k):
    """Reduce the M members of ONE fold over its TEST spots, streaming one member file at a time.

    Never holds more than one member in RAM: each member's predictions .h5ad is large. The
    moments s_m, s_m2 and s_ex accumulate in float64 so V = E[m^2] - E[m]^2 stays accurate.
    Returns (obs, y, M1, V, EX, var, GSTAR, FLO, FHI, P0) on the test rows as float32, with
    M1 = mean_s m_s, V = var_s m_s (ddof=0), EX = mean_s sigma^2 (Gaussian) or
    mean_s(alpha*m^2) (NB). GSTAR is None unless fam["want_gstar"], FLO/FHI unless
    fam["want_pit"], P0 unless fam["want_zero"]; combine_model_dir sets those three.

    g*_s comes from that member's own predictions/gstar_fold{k}.h5ad, realigned to this loop's
    test rows and gene order by name: an independent script wrote it, so positional alignment is
    never assumed. A missing or misaligned file warns and skips only that member's g*
    contribution (n_gstar counts the members that did contribute). The PIT bounds and the zero
    mass are closed-form and use the m/alpha already in RAM this iteration, so they cost no extra
    read. Prints one timing line per member so the log shows progress within a fold.
    """
    M = len(seed_paths)
    s_m = s_m2 = s_ex = None
    names0 = gene_order = gene_set0 = var0 = obs0 = y0 = test = None
    want_gstar = fam.get("want_gstar", False)
    want_pit   = fam.get("want_pit", False)
    want_zero  = fam.get("want_zero", False)      # count only: no point mass for a continuous law
    s_gstar = s_Flo = s_Fhi = s_p0 = None
    n_gstar = 0                                           # members that actually contributed g*
    for i, p in enumerate(seed_paths):
        t0 = time.time()
        a = sc.read_h5ad(p)                                # ONE member in RAM
        seed_dir = os.path.basename(os.path.dirname(os.path.dirname(p)))   # seed_{s}
        if a.uns.get("count_convention") != "v2_no_offset":
            raise ValueError(
                f"{p}: artifact with no count_convention marker. "
                f"Re-run the precompute.")
        if i == 0:
            names0     = a.obs_names.values
            gene_order = a.var_names.values                 # canonical column order (member 0)
            gene_set0  = set(gene_order)
            var0       = a.var.copy()                       # var (gene_role etc.) in gene_order
            test       = (a.obs["split"].to_numpy() == "test")  # same split for every member (seed-invariant)
            obs0       = a.obs.iloc[test].copy()            # join keys (slide/obs_name/adata_row/...)
            if fam["count"]:
                # depth_scale, the divisor the option-A estimator needs, and the one
                # read_arrays requires to rebuild a cp_counts / cp_counts_nr target below.
                # Read off member 0's own uns: it is a per-run constant
                # (datasets.{ds}.count_common_depth / 1e4), so it is seed- and
                # fold-invariant. Absent => None => divisor 1.0, the right answer for the
                # depth-blind raw_counts target.
                ds_val = a.uns.get("depth_scale")
                fam["depth_scale"] = None if ds_val is None else float(ds_val)
            if "y" in a.layers:
                y0 = np.asarray(a.layers["y"], dtype=np.float32)[test]
            else:
                # GCM predictions hold test rows only and carry no `layers["y"]`
                # (train_gcm.py::fold_predictions_adata). Ground truth does not depend on
                # the model, so rebuild it once, from member 0, out of the shared
                # preprocessed adata this run trained on.
                loc   = (var0["target_loc"].to_numpy()
                        if "target_loc" in var0.columns else None)
                scale = (var0["target_scale"].to_numpy() if loc is not None else None)
                y0 = target_values_for_predictions(
                    fam["adata_path"], fam["target_source"], gene_order,
                    obs0["adata_row"].to_numpy(), depth_scale=fam.get("depth_scale"),
                    target_loc=loc, target_scale=scale,
                )
            # Defensive: the spec said this family carries these layers — fail early on a stale file.
            if fam["want_var"]:
                assert "pred_var" in a.layers, f"{fam['model_type']}: expected layer 'pred_var'"
            if fam["want_disp"]:
                assert "dispersion" in a.layers, f"{fam['model_type']}: expected layer 'dispersion'"
            n_test, G = int(test.sum()), len(gene_order)
            s_m  = np.zeros((n_test, G), dtype=np.float64)
            s_m2 = np.zeros((n_test, G), dtype=np.float64)
            s_ex = np.zeros((n_test, G), dtype=np.float64) if fam["extra_layer"] else None
            if want_gstar:
                s_gstar = np.zeros((n_test, G), dtype=np.float64)
            if want_pit:
                s_Flo = np.zeros((n_test, G), dtype=np.float64)
                s_Fhi = np.zeros((n_test, G), dtype=np.float64)
            if want_zero:
                s_p0  = np.zeros((n_test, G), dtype=np.float64)
        else:                                              # same fold/spots; genes may be REORDERED
            assert np.array_equal(a.obs_names.values, names0), "members misaligned (obs)"
            # GCM gene order is hash-randomized across member processes (ALL_GENES = list over a
            # set), so var_names differ in ORDER, not in content. Realign by gene NAME below; only
            # a genuinely different gene SET is an error.
            assert set(a.var_names) == gene_set0, "members have DIFFERENT gene sets (not just reordered)"

        col = a.var_names.get_indexer(gene_order)          # reorder member columns -> gene_order (identity for i==0)
        m = np.asarray(a.X, dtype=np.float64)[test][:, col]  # (n_test, G), genes in canonical order
        s_m  += m
        s_m2 += m * m
        alpha = sig2 = None
        if fam["want_var"]:                                # Gaussian: + sigma^2
            # Bound to a name rather than consumed inline: the Gaussian PIT below needs this
            # member's sigma.
            sig2 = np.asarray(a.layers["pred_var"], dtype=np.float64)[test][:, col]
            s_ex += sig2
        elif fam["want_disp"]:                             # NB: + alpha_s * m_s^2
            alpha = np.asarray(a.layers["dispersion"], dtype=np.float64)[test][:, col]
            s_ex += alpha * m * m

        # g* — this member's OWN precompute file (sibling of preds_fold{k}.h5ad). Aligned by
        # obs_name/gene name, NOT trusted positionally (written independently of this loop).
        if want_gstar:
            gpath = os.path.join(os.path.dirname(p), f"gstar_fold{k}.h5ad")
            if not os.path.exists(gpath):
                print(f"[{_el():7.1f}s]   fold {k} member {i + 1}/{M} ({seed_dir}): "
                      f"WARNING gstar_fold{k}.h5ad not found ({gpath}) — skipping g* for this member",
                      flush=True)
            else:
                g = sc.read_h5ad(gpath)
                row_pos = g.obs_names.get_indexer(names0[test])
                col_pos = g.var_names.get_indexer(gene_order)
                if (row_pos < 0).any() or (col_pos < 0).any():
                    print(f"[{_el():7.1f}s]   fold {k} member {i + 1}/{M} ({seed_dir}): "
                          f"WARNING {gpath} does not cover this fold's test rows/genes — "
                          f"skipping g* for this member", flush=True)
                else:
                    s_gstar += np.asarray(g.X, dtype=np.float64)[row_pos][:, col_pos]
                    n_gstar += 1
                del g

        # F_lo/F_hi + P(Y=0), O(1) closed-form, computed inline. The branch keys on whether this
        # member's alpha is set this iteration (NB: the fam["want_disp"] branch above filled it;
        # Poisson: it stays the None it was reset to at the top of the loop), not on a family
        # flag fixed once per model, so it is right for whichever count family fam is. smear
        # comes from fam and is true only for the continuous cp_counts_nr target, the one family
        # whose PIT needs the smeared CDF F~.
        # The Gaussian arm must come first and key on want_var. A Gaussian has want_disp False,
        # so alpha is None and it would otherwise fall into the count `else`, where scipy's
        # poisson.cdf accepts a float mu, floors a non-integer y, and returns finite, plausible,
        # meaningless PIT values. That failure is silent, not a crash.
        if want_pit:
            if fam["want_var"]:                                # Gaussian: continuous predictive
                # F_lo == F_hi: there is no probability atom for the observation to straddle, so
                # the two-bound form degenerates. No `smear` either (that is the correction for a
                # DISCRETE predictive fitted to a continuous target, the *_cp_nr case).
                F_lo, F_hi = gaussian_pit_bounds(m, np.sqrt(sig2), y0)
            else:
                mu_c = m                                       # X is already mu on the fitted count scale
                if alpha is not None:                          # NB
                    F_lo, F_hi = nb_pit_bounds(mu_c, alpha, y0, smear=fam["smear"])
                else:                                          # Poisson (no dispersion needed)
                    F_lo, F_hi = poisson_pit_bounds(mu_c, y0, smear=fam["smear"])
            # Note: gaussian_pit_bounds returns the SAME array object twice. Never zero out F_lo
            # on observed zeros by analogy with the count cdf(-1)=0 handling: it would mutate
            # F_hi too and collapse the PIT on every observed zero, i.e. most of the matrix.
            s_Flo += F_lo
            s_Fhi += F_hi
        if want_zero:                                          # count families only
            if alpha is not None:
                r = 1.0 / alpha
                s_p0 += (r / (r + m)) ** r
            else:
                s_p0 += np.exp(-m)

        del a, m, sig2                                     # FREE the member before the next read
        print(f"[{_el():7.1f}s]   fold {k} member {i + 1}/{M} ({seed_dir}): "
              f"read+accum in {time.time() - t0:.1f}s", flush=True)

    M1 = s_m / M
    V  = np.maximum(s_m2 / M - M1 * M1, 0.0)               # ddof=0 (mixture variance); clip rounding
    EX = (s_ex / M).astype(np.float32) if s_ex is not None else None
    GSTAR = FLO = FHI = P0 = None
    if want_gstar:
        if n_gstar == 0:
            print(f"[{_el():7.1f}s]   fold {k}: WARNING no member had a gstar_fold{k}.h5ad — "
                  f"'gstar' layer is all-NaN for this fold's rows", flush=True)
            GSTAR = np.full(s_m.shape, np.nan, dtype=np.float32)
        else:
            GSTAR = (s_gstar / n_gstar).astype(np.float32)
    if want_pit:
        FLO = (s_Flo / M).astype(np.float32)
        FHI = (s_Fhi / M).astype(np.float32)
    if want_zero:
        P0  = (s_p0 / M).astype(np.float32)
    return obs0, y0, M1.astype(np.float32), V.astype(np.float32), EX, var0, GSTAR, FLO, FHI, P0


def combine_model_dir(model_dir, seeds, out_dir=None):
    """Combine every fold's TEST spots under one model dir into ONE consolidated test_summaries.h5ad."""
    model_dir = os.path.abspath(model_dir)
    out_dir = out_dir or os.path.join(model_dir, "ensemble")

    seed0_pred = os.path.join(model_dir, f"seed_{seeds[0]}", "predictions")
    fold_files = sorted(glob.glob(os.path.join(seed0_pred, "preds_fold*.h5ad")))
    if not fold_files:
        print(f"[skip] {model_dir}  (no preds_fold*.h5ad under seed_{seeds[0]}/predictions)")
        return False
    folds = [int(os.path.basename(f).replace("preds_fold", "").replace(".h5ad", "")) for f in fold_files]

    fam = detect_family(model_dir, seeds)
    # g*, native-CDF PIT bounds and zero mass are gated on DE_gstar_precompute.py having run for
    # THIS model dir, detected with one cheap glob and no h5ad read. A dir whose members have no
    # gstar_fold*.h5ad gets none of the three extra layers.
    fam["want_gstar"] = fam["count"] and bool(
        glob.glob(os.path.join(model_dir, "seed_*", "predictions", "gstar_fold*.h5ad")))
    # PIT and zero mass are separate gates. A PIT is uniform on [0,1] whatever the target scale,
    # which is what makes a continuous Gaussian and a discrete count head comparable on one
    # calibration axis, so it applies to every family with a predictive distribution. P(Y=0) is a
    # point mass, which a continuous predictive does not have, so it stays count-only, and it is
    # keyed on want_gstar rather than fam["count"]: a count model with no precomputed g* gets
    # neither the gstar layer nor a zero_prob layer.
    # Neither is gated on want_disp: Poisson's own formulas (poisson_pit_bounds, exp(-mu_c)) need
    # no dispersion, and which formula runs is decided per member inside combine_test_fold, from
    # whether that member's alpha is set, not here.
    fam["want_pit"]  = fam["want_gstar"] or fam["want_var"]
    fam["want_zero"] = fam["want_gstar"]
    print(f"[{_el():7.1f}s] [run ] {os.path.basename(model_dir)} | model={fam['model_type']} "
          f"(count={fam['count']}, extra={fam['extra_layer']}, gstar={fam['want_gstar']}, "
          f"pit={fam['want_pit']}) | {len(folds)} folds | members={seeds}",
          flush=True)

    obs_parts, y_parts, pred_parts, epi_parts, ex_parts = [], [], [], [], []
    gstar_parts, flo_parts, fhi_parts, p0_parts = [], [], [], []
    var0 = None
    for k in folds:
        paths = [os.path.join(model_dir, f"seed_{s}", "predictions", f"preds_fold{k}.h5ad") for s in seeds]
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            print(f"[{_el():7.1f}s]   fold {k}: SKIP — {len(missing)}/{len(seeds)} member(s) missing", flush=True)
            continue
        tf = time.time()
        obs, y, M1, V, EX, var0, GSTAR, FLO, FHI, P0 = combine_test_fold(paths, fam, k)
        obs_parts.append(obs); y_parts.append(y)
        pred_parts.append(M1); epi_parts.append(V)
        if fam["extra_layer"]:
            ex_parts.append(EX)
        if fam["want_gstar"]:
            gstar_parts.append(GSTAR)
        if fam["want_pit"]:
            flo_parts.append(FLO); fhi_parts.append(FHI)
        if fam["want_zero"]:
            p0_parts.append(P0)
        print(f"[{_el():7.1f}s]   fold {k} DONE: {obs.shape[0]} test spots x {M1.shape[1]} genes "
              f"in {time.time() - tf:.1f}s", flush=True)

    if not pred_parts:
        print(f"[skip] {model_dir}  (no complete fold)")
        return False

    obs = pd.concat(obs_parts, axis=0)
    if obs.index.duplicated().any():                        # LOPO -> test sets are disjoint; warn otherwise
        print(f"  WARNING: {int(obs.index.duplicated().sum())} duplicate spots across folds "
              f"(split is not LOPO?) — keeping all rows.")
    layers = {"y": np.concatenate(y_parts, 0), "epistemic": np.concatenate(epi_parts, 0)}
    if fam["extra_layer"]:
        layers[fam["extra_layer"]] = np.concatenate(ex_parts, 0)
    if fam["want_gstar"]:
        layers["gstar"] = np.concatenate(gstar_parts, 0)
    if fam["want_pit"]:
        layers["pit_flo"] = np.concatenate(flo_parts, 0)
        layers["pit_fhi"] = np.concatenate(fhi_parts, 0)
    if fam["want_zero"]:                                   # count only: a continuous law has no atom at 0
        layers["zero_prob"] = np.concatenate(p0_parts, 0)

    out = ad.AnnData(X=np.concatenate(pred_parts, 0), obs=obs, var=var0, layers=layers)
    # Top-level convention marker, written on every output, count or not, exactly as
    # mc_dropout_posthoc.py and laplace_posthoc.py do on theirs. One marker, one rule across the
    # three epistemic summary files: a reader can assert
    # uns["count_convention"] == "v2_no_offset" on any of them.
    out.uns["count_convention"] = "v2_no_offset"
    out.uns["deep_ensemble"] = dict(
        model_type=fam["model_type"], target_source=fam["target_source"],
        n_members=len(seeds), seeds=np.asarray(seeds), count=bool(fam["count"]), split="test",
        mean_scale=fam["mean_scale"],
        layers_doc=("X = pred = mean_s m_s ; y = observed target ; epistemic = var_s m_s ; "
                    + ("aleatoric_var = mean_s sigma^2 (Gaussian)" if fam["want_var"] else
                       "mean_alpha_m2 = mean_s(alpha_s * m_s^2) (NB)" if fam["want_disp"] else
                       "no extra layer (Poisson: aleatoric == pred on the target scale)" if fam["count"] else
                       "no extra layer (deterministic mean-only: no aleatoric; total == epistemic)")
                    + (" ; gstar = mean_s g*_s = mean_s E_s[log1p(Y/depth_scale)] (canal-1 g* "
                       "headline, log1p-CP10k scale for the _cp/_cp_nr families and "
                       "log1p(raw counts) for _raw; all-NaN for a fold if no member's "
                       "gstar_fold{k}.h5ad was found)" if fam["want_gstar"] else "")
                    + (" ; pit_flo/pit_fhi = mean_s F_lo/F_hi(y; ...) (native-CDF PIT bounds, the "
                       "MIXTURE CDF = mean of the member CDFs -- NB or Poisson for a count head, "
                       "Phi((y-m_s)/sigma_s) with F_lo == F_hi for a continuous Gaussian head, "
                       "whichever this model_type is -- canal-1 calibration)"
                       if fam["want_pit"] else "")
                    + (" ; zero_prob = mean_s P_s(Y=0) (canal-3 zero-detection score: "
                       "(r_s/(r_s+mu_c_s))^r_s, r_s=1/alpha_s for NB, or exp(-mu_c_s) for "
                       "Poisson; COUNT families only -- a continuous predictive has no atom "
                       "at 0)" if fam["want_zero"] else "")),
        reconstruct=("everything is on the model's OWN target scale, no constant applies: "
                     "aleatoric = pred (Poisson) | aleatoric_var (Gaussian) | "
                     "pred + mean_alpha_m2 (NB) ; total = aleatoric + epistemic."
                     + (" gstar/pit_flo/pit_fhi/zero_prob are each an ensemble-MIXTURE average "
                        "(mean_s Q_s) of a per-member Q_s that is ITSELF an expectation under "
                        "member s's own predictive distribution (a mean / a CDF value / a PMF "
                        "value at 0) -- exact by linearity of expectation over the M-member "
                        "mixture (no approximation, unlike epistemic/aleatoric above). "
                        "mu_c_s = m_s is member s's own count-scale mean: the saved X is "
                        "ALREADY mu on the fitted target scale (uns['count_convention'] == "
                        "'v2_no_offset'), so nothing is rescaled."
                        if fam["want_gstar"] else "")
                     + (" OPTION-A (mean-only) estimator, NOT stored as a layer because it is a "
                        "pure function of pred and the scalar below: "
                        "div = 1.0 if 'depth_scale' is absent here else float(depth_scale) ; "
                        "option_A = log1p(X / div) = log1p(mean_s m_s / div) -- average the member "
                        "means FIRST, then transform (NOT mean_s log1p(m_s / div); the two differ "
                        "because log1p is concave). option_A and the gstar layer are correlated "
                        "against the SAME target on the SAME axis, so their difference isolates "
                        "the effect of routing the prediction through the full predictive "
                        "distribution -- the ensemble analogue of the single-model "
                        "'{metric}' vs '{metric}_logmean' pair in atoms.parquet."
                        if fam["count"] else "")),
    )
    if fam["count"]:
        # The option-A divisor (see reconstruct above). Written only when not None: h5ad uns
        # cannot serialize a bare Python None, so readers must use .get("depth_scale") and treat
        # a missing key as None -> divisor 1.0 (the correct answer for the depth-blind targets).
        if fam.get("depth_scale") is not None:
            out.uns["deep_ensemble"]["depth_scale"] = float(fam["depth_scale"])
        print(f"[{_el():7.1f}s]   option-A divisor: depth_scale={fam.get('depth_scale')} "
              f"-> option_A = log1p(pred / "
              f"{1.0 if fam.get('depth_scale') is None else float(fam['depth_scale'])})", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "test_summaries.h5ad")
    tw = time.time()
    out.write_h5ad(out_path)
    print(f"[{_el():7.1f}s] [ok  ] {out.n_obs} spots x {out.n_vars} genes "
          f"(write {time.time() - tw:.1f}s) -> {out_path}", flush=True)
    return True


def main():
    p = argparse.ArgumentParser(description="Combine deep-ensemble seed-members into ONE test_summaries.h5ad.")
    p.add_argument("--model-dir", nargs="+", required=True,
                   help="One or more dirs each containing seed_{s}/ subdirs (shell globs expand here).")
    p.add_argument("--seeds", default="0,1,2,3,4", help="Comma-separated member seeds (default 0,1,2,3,4).")
    p.add_argument("--out-dir", default=None,
                   help="Output dir (default: <model-dir>/ensemble). Only valid with a single --model-dir.")
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    if args.out_dir and len(args.model_dir) > 1:
        p.error("--out-dir is only valid with a single --model-dir (it would collide otherwise).")

    n_ok = sum(combine_model_dir(md, seeds, out_dir=args.out_dir) for md in args.model_dir)
    print(f"\nDone: wrote test_summaries.h5ad for {n_ok}/{len(args.model_dir)} model dir(s).")


if __name__ == "__main__":
    main()
