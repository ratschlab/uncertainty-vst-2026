# src/training/train_loop_indexed.py
#
# Single-fold training loop: h5ad-native, index-based, spec-driven.
# CV iteration stays in the Snakemake script; this trains one fold.
#
# The model_type string resolves to a ModelSpec in src/models/specs.py. The loop
# reads behavior from that spec rather than branching on an is_gaussian bool, so
# adding a model (Poisson, NB, epistemic) needs no change here.
#
#   • Data is a shared GPU-resident SharedTensors object; a fold is index sets
#     (train, val, optional test) into it. Nothing is re-tensorized per fold and,
#     on the primary path, no batch is copied host->device.
#   • The training hot loop uses no DataLoader. With the data already on the GPU
#     there is nothing to prefetch, so row indices are shuffled on-device
#     (torch.randperm) and sliced. A loader would only add Python work between
#     GPU steps.
#   • The batch loop issues no host sync; the epoch-end reduction is the only one.
#   • Centering by the fold's train_mean is subtracted from the gathered batch,
#     never baked into the shared tensor, which all splits share.
#   • Metrics are slide-aware (per-slide, then averaged); groups flow into
#     run_inference.

import numpy as np
import torch

from src.models.specs import get_spec, build_model
from src.training.losses import apply_spec_loss
from src.training.inference_indexed import run_inference


# ─────────────────────────────────────────────────────────────────────────────
# Small helpers (kept here, kept few)
# ─────────────────────────────────────────────────────────────────────────────

def _probabilistic_extras(train_result, val_result):
    """Epoch-log fields for any probabilistic model, meaning any spec with a
    pred_var. The key names become the learning-curve CSV columns and feed the
    per-epoch log line, so they are part of the interface.
    """
    extras = {
        # Native-CDF PIT calibration (see calibration_pit): ece_pit = magnitude,
        # pit_bias = ½−E[PIT] (location, NOT confidence), disp_signed = over/under-
        # confidence. train+val for ece_pit; val for the directional diagnostics.
        "train_ece_pit":        train_result["ece_pit"],
        "val_ece_pit":          val_result["ece_pit"],
        "val_pit_bias":         val_result.get("pit_bias"),
        "val_disp_signed":      val_result.get("disp_signed"),
        "train_mean_pred_var":  train_result.get("mean_pred_var"),
        "val_mean_pred_var":    val_result.get("mean_pred_var"),
    }
    if val_result.get("var_residual_corr") is not None:
        # Log the train side next to the val side, so a train/val gap in
        # uncertainty-error quality is as visible as the mean's overfit gap.
        extras["val_var_residual_corr"]           = val_result["var_residual_corr"]
        extras["val_var_residual_corr_partial"]   = val_result.get("var_residual_corr_partial")
        extras["train_var_residual_corr"]         = train_result.get("var_residual_corr")
        extras["train_var_residual_corr_partial"] = train_result.get("var_residual_corr_partial")
    return extras


def _epoch_log_line(epoch, max_epochs, e, spec, corr_metrics):
    name = spec.loss_display_name
    line = (
        f"  Epoch {epoch+1}/{max_epochs} — "
        f"train_{name}: {e['train_loss']:.4f}, val_{name}: {e['val_loss']:.4f}, "
        f"train_MSE: {e['train_mse']:.4f}, val_MSE: {e['val_mse']:.4f}"
    )
    # One train/val pair per configured correlation metric, label capitalized
    # (Pearson, Spearman, ...).
    for cm in corr_metrics:
        line += (f", train_{cm.capitalize()}: {e[f'train_{cm}']:.4f}, "
                 f"val_{cm.capitalize()}: {e[f'val_{cm}']:.4f}")
    if spec.is_probabilistic and e.get("val_mean_pred_var") is not None:
        line += (
            f", val_ECEpit: {e['val_ece_pit']:.4f}, val_disp: {e['val_disp_signed']:+.4f}, "
            f"val_var: {e['val_mean_pred_var']:.3f}"
        )
    line += f", lr: {e['lr']:.1e}"
    return line


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def train_one_fold(
    shared,                    # SharedTensors  (the one source of truth)
    train_idx, val_idx,        # np.int64 row-index arrays into `shared`
    train_mean,                # np.ndarray (emb_dim,) — fold's train-set embedding mean
    model_type,
    lam=1.0,
    n_genes=1,
    lr=1e-4,
    weight_decay=1e-4,
    batch_size=128,
    max_epochs=50,
    patience=5,
    device="cpu",
    center=True,
    train_subsample_idx=None,
    arch=None,
    corr_metrics=("pearson",),
    warmup_epochs=0,
    grad_clip_norm=None,
    test_idx=None,
    log_test_curve=False,
    model_kwargs=None,
    target_loc=None,
    target_scale=None,
    nb_per_epoch_calibration=True,
    early_stop_criterion="corr",
):
    """Train one fold; return (best_state, epoch_log, model).

    Centering: when `center` is True, the fold's `train_mean`, computed by the
    caller on train rows only so there is no val/test leakage, is subtracted from
    embeddings at gather time. The caller must save `train_mean` and `center` with
    the model: post-hoc reproduction applies the mean only if center was True.

    `train_subsample_idx`: fixed, seeded subset of train rows used only for the
    per-epoch train metric. Training itself uses all train rows.

    `warmup_epochs`: if > 0, ramp the LR linearly from 0 to `lr` over the first
    `warmup_epochs` epochs, then hold at `lr`. 0 keeps the LR constant. Warmup is
    the standard stabilizer for large-batch training.

    `grad_clip_norm`: if truthy, clip the global gradient norm to this value each
    step, a direct guard against loss spikes when scaling up the batch size. None
    or 0 disables clipping.

    `test_idx` / `log_test_curve`: opt-in diagnostic. When `log_test_curve` is
    True and `test_idx` is given, the held-out test slides are also scored each
    epoch and their metrics added to the epoch log under `test_*` keys, so the
    train/val/test generalization gap can be plotted. Diagnostic only: the test
    set never influences early stopping, which selects on val, and the test
    metrics reach the learning-curve CSV but not the per-epoch line. Off by
    default, and then it costs no extra forward pass.

    `early_stop_criterion`: which per-epoch val scalar selects the best epoch's
    weights. The classic-loop counterpart of `gcm.early_stop_criterion`:
      - "corr" (default): keep the epoch that maximizes the primary correlation
        metric `val_result[corr_metrics[0]]`.
      - "val_loss": keep the epoch that minimizes the validation loss, that is the
        model's own training objective, whatever `spec.loss_fn` computes, for
        reproducing a baseline's own recipe.
      - "mse": keep the epoch that minimizes the validation MSE on the axis the
        correlation is reported on (`mse_corr_axis`, see run_inference):
        log1p(mu/depth_scale) vs log1p(y/depth_scale) for the count specs, the
        native target otherwise. Deliberately not the `mse` key, which stays on
        each model's own target scale (CP_T, raw counts) and would make the
        criterion mean a different thing per family. This is a third,
        family-neutral axis, neither the reported correlation nor the model's own
        likelihood, so a ranking that survives it is not an artifact of either.
    """
    if early_stop_criterion not in ("corr", "val_loss", "mse"):
        raise ValueError(
            f"Unknown early_stop_criterion {early_stop_criterion!r}; "
            f"expected 'corr', 'val_loss' or 'mse'"
        )
    spec = get_spec(model_type)
    emb_dim = shared.emb_dim
    data_device = shared.emb.device          # GPU on the primary (data-on-GPU) path

    # Centering tensor on the MODEL device. Subtracted from the gathered batch
    # below (never baked into the shared tensor — all splits share one tensor).
    mean_t = (torch.as_tensor(np.asarray(train_mean), dtype=torch.float32).to(device)
              if center else None)

    # Per-gene target standardization, X-target models only. When active the batch
    # target is mapped to z-space below and the loss is computed there, which
    # rebalances genes and frees the variance head from encoding gene scale.
    # run_inference keeps metrics and predictions in z-space too, so σ is scale-free.
    # Per-(slide,gene) metrics are affine-invariant, so they stay comparable.
    # loc/scale are train-fold constants of shape (n_genes,); None is a no-op.
    loc_t = scale_t = None
    if target_loc is not None:
        loc_t   = torch.as_tensor(np.asarray(target_loc),   dtype=torch.float32).to(device)
        scale_t = torch.as_tensor(np.asarray(target_scale), dtype=torch.float32).to(device)

    # Train rows as a LongTensor on the data's device, shuffled on-device each epoch
    # with torch.randperm and then sliced. With the data already GPU-resident there
    # is nothing to prefetch, so a DataLoader would only add Python work between GPU
    # steps.
    train_idx_t = torch.as_tensor(np.asarray(train_idx), dtype=torch.long,
                                  device=data_device)
    n_train = train_idx_t.numel()

    model = build_model(model_type, in_dim=emb_dim, n_genes=n_genes,
                        device=device, arch=arch, model_kwargs=model_kwargs)
    # fused=True runs the whole Adam step in one CUDA kernel, which cuts the launch
    # count on this small, launch-bound MLP. It is asked for only when CUDA is
    # present, and a torch whose Adam has no such keyword raises TypeError, so fall
    # back to the plain step.
    try:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay,
                                     fused=torch.cuda.is_available())
    except TypeError:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    metric_train_idx = train_idx if train_subsample_idx is None else train_subsample_idx

    # The primary correlation metric is corr_metrics[0], "pearson" by default. It has
    # to be read from corr_metrics[0] rather than hardcoded: compute_metrics populates
    # result keys only for the metrics listed in corr_metrics, and reading it from
    # there keeps selection and the headline reported metric on one criterion.
    primary_corr = corr_metrics[0]

    best_val_metric = float("-inf")
    best_state = None
    patience_count = 0
    epoch_log = []

    for epoch in range(max_epochs):
        # Optional LR warmup: ramp lr linearly over the first `warmup_epochs`, then
        # hold at `lr`. warmup_epochs == 0 keeps lr constant.
        if warmup_epochs > 0:
            warm_factor = min(1.0, (epoch + 1) / warmup_epochs)
            for g in optimizer.param_groups:
                g["lr"] = lr * warm_factor

        # ── Training pass — SYNC-FREE: accumulate on-GPU, sync once at epoch end
        model.train()
        running = torch.zeros((), device=device)
        grad_norm_sum = torch.zeros((), device=device)   # mean pre-clip grad norm
        n_batches = 0
        # Shuffle row indices on-device; iterate FULL batches only (drop_last=True).
        epoch_idx = train_idx_t[torch.randperm(n_train, device=data_device)]
        n_full = (n_train // batch_size) * batch_size
        for start in range(0, n_full, batch_size):
            b = epoch_idx[start:start + batch_size]
            xb, yb = shared.gather(b, device=device)
            if center:
                xb = xb - mean_t
            if loc_t is not None:                 # train loss lives in z-space
                yb = (yb - loc_t) / scale_t
            optimizer.zero_grad()
            raw = model(xb)
            # Every model, count models included, fits its likelihood on its own
            # target: no offset, no per-spot exposure. The only thing the loss takes
            # beyond (raw, yb) is lam, for the Gaussian and NB-anchor specs.
            loss = apply_spec_loss(spec, raw, yb, lam=lam)
            loss.backward()
            # Total pre-clip gradient norm, an instability and LR diagnostic logged
            # per epoch. clip_grad_norm_ returns that norm, and max_norm=inf makes it
            # measure without scaling, so one call both clips, when grad_clip_norm is
            # set, and yields the number to log.
            gnorm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip_norm if grad_clip_norm else float("inf"))
            grad_norm_sum += gnorm.detach()
            optimizer.step()
            running += loss.detach()
            n_batches += 1
        running_train_loss = (running / max(n_batches, 1)).item()
        grad_norm_mean = (grad_norm_sum / max(n_batches, 1)).item()

        # ── Eval-mode metrics (slide-aware; groups passed through) ────────────
        # metrics_on_gpu=True computes the per-epoch correlation and calibration
        # scalars on the data device (corr_scalars_gpu / calibration_scalars_gpu)
        # instead of round-tripping every prediction to scipy. This is the early-stop
        # hot path, so it has to stay cheap as n_genes grows: the scipy path loops
        # per-(slide,gene). The final inference in train.py keeps the CPU path
        # (run_inference defaults to False), so the saved atoms.parquet and per-gene
        # arrays come from the CPU path.
        # nb_pit=False on the train-metric pass, the heaviest one: NB calibration
        # there is diagnostic only, so its NB ece_pit stays NaN and the NB CDF cost is
        # skipped. The held-out val and test passes compute NB PIT when the config
        # toggle is on. Gaussian and Poisson PIT are computed on every pass.
        train_result = run_inference(
            model, shared, metric_train_idx, train_mean, spec=spec,
            batch_size=batch_size, device=device,
            lam=lam, center=center, corr_metrics=corr_metrics, metrics_on_gpu=True,
            target_loc=target_loc, target_scale=target_scale,
            nb_pit=False,
        )
        val_result = run_inference(
            model, shared, val_idx, train_mean, spec=spec,
            batch_size=batch_size, device=device,
            lam=lam, center=center, corr_metrics=corr_metrics, metrics_on_gpu=True,
            target_loc=target_loc, target_scale=target_scale,
            nb_pit=nb_per_epoch_calibration,
        )

        # ── Epoch log row ─────────────────────────────────────────────────────
        e = {
            "epoch":              epoch + 1,
            "lr":                 optimizer.param_groups[0]["lr"],
            "train_loss":         train_result["loss"],
            "val_loss":           val_result["loss"],
            "running_train_loss": running_train_loss,
            "grad_norm":          grad_norm_mean,
            "train_mse":          train_result["mse"],
            "val_mse":            val_result["mse"],
            # MSE on the correlation's own axis (log1p(mu/div) for count specs; for
            # every other spec the same quantity as the "mse" above, to ~1e-9).
            # Always logged, so the learning curve carries the selection scalar of
            # early_stop_criterion "mse" whatever criterion this run actually used.
            "train_mse_corr_axis": train_result["mse_corr_axis"],
            "val_mse_corr_axis":   val_result["mse_corr_axis"],
        }
        # One train_/val_ column per configured correlation metric, each named after
        # the metric it holds (train_pearson, train_spearman), so a column never lies
        # about its contents. corr_metrics[0] is the primary one, used for early
        # stopping; the rest are logged alongside.
        for name in corr_metrics:
            e[f"train_{name}"] = train_result[name]
            e[f"val_{name}"]   = val_result[name]
        if spec.is_probabilistic:
            e.update(_probabilistic_extras(train_result, val_result))

        # ── Optional per-epoch TEST curve (diagnostic only) ───────────────────
        # An extra eval-mode forward pass over the held-out test slides, mirroring
        # the val_* columns: loss, MSE, each corr metric, the probabilistic extras.
        # It exists to plot the generalization gap. The early-stopping block below
        # never reads these keys and selects on val only.
        if log_test_curve and test_idx is not None:
            test_result = run_inference(
                model, shared, test_idx, train_mean, spec=spec,
                batch_size=batch_size, device=device,
                lam=lam, center=center, corr_metrics=corr_metrics, metrics_on_gpu=True,
                target_loc=target_loc, target_scale=target_scale,
                nb_pit=nb_per_epoch_calibration,
            )
            e["test_loss"] = test_result["loss"]
            e["test_mse"]  = test_result["mse"]
            for name in corr_metrics:
                e[f"test_{name}"] = test_result[name]
            if spec.is_probabilistic:
                e["test_ece_pit"]       = test_result["ece_pit"]
                e["test_pit_bias"]      = test_result.get("pit_bias")
                e["test_disp_signed"]   = test_result.get("disp_signed")
                e["test_mean_pred_var"] = test_result.get("mean_pred_var")
                if test_result.get("var_residual_corr") is not None:
                    e["test_var_residual_corr"]         = test_result["var_residual_corr"]
                    e["test_var_residual_corr_partial"] = test_result.get("var_residual_corr_partial")

        epoch_log.append(e)
        print(_epoch_log_line(epoch, max_epochs, e, spec, corr_metrics))

        # ── Early stopping / best-epoch selection ──────────────────────────────
        # "corr", the default, selects on the primary correlation metric
        # corr_metrics[0], maximized. "val_loss" selects on the validation loss.
        # "mse" selects on the validation MSE taken on the correlation's own axis,
        # which is family-neutral; see the docstring. The two minimized criteria are
        # negated so all three share the ">" and patience bookkeeping below.
        if early_stop_criterion == "val_loss":
            val_metric = -val_result["loss"]
        elif early_stop_criterion == "mse":
            val_metric = -val_result["mse_corr_axis"]
        else:
            val_metric = val_result[primary_corr]
        if val_metric > best_val_metric:
            best_val_metric = val_metric
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # The monitored metric can fail to beat its -inf init at every epoch, for instance
    # a near-constant gene whose per-slide val correlation is NaN throughout, since
    # NaN > x is always False. best_state is then still None, so fall back to the
    # final-epoch weights: the run completes and writes metrics.json with NaN
    # correlations, surfaced in nan_summary, instead of crashing on
    # load_state_dict(None).
    if best_state is None:
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return best_state, epoch_log, model
