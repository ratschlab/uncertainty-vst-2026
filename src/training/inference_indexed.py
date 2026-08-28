# src/training/inference_indexed.py
#
# Run a trained model forward over a set of spot indices. Returns the raw
# predictions, the slide-aware metrics and the eval loss from one forward pass.
#
# Everything that varies by model comes from the model's ModelSpec
# (src/models/specs.py), so this file never branches on a model class:
#   spec.extract_predictions  turns raw model output into {"means", "pred_var", ...};
#                             a spec that yields "pred_var" also gets calibration
#   spec.uses_lam             whether `lam` is passed to the loss
#   spec.target_source        routes the count specs onto their log1p reporting axis
#
# Adding a model means writing its spec, not editing this file.

import numpy as np
import torch

from src.evaluation.metrics_slide_aware import (
    compute_metrics, corr_scalars_gpu, calibration_scalars_gpu,
)
from src.models.specs import COUNT_ROUTED_TARGETS
from src.training.losses import apply_spec_loss
# The count metric code lives in src/evaluation/count_metrics.py, shared with the
# GCM pipeline so both report the same numbers.
from src.evaluation.count_metrics import (
    count_expected_log1p, count_corr_inputs, count_metrics_final,
)


def run_inference(model, shared, idx, train_mean,
                  spec, batch_size=128, device="cpu",
                  lam=1.0, center=True,
                  corr_metrics=("pearson",), metrics_on_gpu=False,
                  target_loc=None, target_scale=None,
                  nb_pit=True, compute_cpu_metrics=True):
    """Forward `model` over the spots `idx` of `shared`; return preds + metrics + loss.

    Parameters
    ----------
    model        : nn.Module (trained)
    shared       : SharedTensors — the run's single data source
    idx          : np.int64 array — spot rows to evaluate (a split/fold)
    train_mean   : np.ndarray (emb_dim,) — fold train mean (used iff center)
    spec         : ModelSpec — drives output extraction + loss + calibration
    lam          : float — passed to the loss when spec.uses_lam
    center       : bool — subtract train_mean from embeddings (must match training)
    corr_metrics : iterable of correlation-metric names to compute
                   (defaults to ("pearson",); see CORR_METRICS in metrics file).
    metrics_on_gpu : compute the slide-aware scalars the training loop needs on the
                   data device (correlations via corr_scalars_gpu, plus PIT
                   calibration and var_residual_corr via calibration_scalars_gpu for
                   probabilistic specs) and skip the matching scipy work. The
                   per-epoch / early-stop path sets this, so the hot loop never
                   copies predictions to the host. Off by default, because the final
                   inference needs the per-(slide,gene) atoms and per-gene arrays
                   that only the CPU path produces.
    target_loc, target_scale : per-gene (n_genes,) constants of the X-target
                   standardization (from the fold's train rows; saved with the
                   checkpoint). When given, the target is mapped to z-space and loss,
                   metrics and the returned predictions all live there (scale-free
                   variance); the per-(slide,gene) metrics are affine-invariant so the
                   reported numbers stay comparable to log-CP10k. None -> off.

    Returns (dict) holding groups, loss, and:
        metrics_on_gpu=True : the slide-aware scalars, one per corr metric, plus the
            calibration scalars for probabilistic specs, the diagnostic `mse` and
            `mean_pred_var`, and `mse_corr_axis`, the MSE on the correlation's own
            axis (this path only). means, y and pred_var are None, which is what
            keeps the batch loop sync-free.
        metrics_on_gpu=False: means, y, pred_var (or None) as host arrays, plus
            every compute_metrics key (`pearson`, `pearson_per_slide_gene`, ...).
    """
    idx = np.asarray(idx, dtype=np.int64)
    n = idx.size
    # idx as a LongTensor on the data's device, so the gather indexes on-device with
    # no DataLoader. Eval is forward-only and the data is GPU-resident, so a loader
    # would only add per-item collate overhead between GPU steps.
    # The loop iterates in order and never shuffles, so the concatenated outputs
    # align row-for-row with `groups_np` below.
    idx_t = torch.as_tensor(idx, dtype=torch.long, device=shared.emb.device)

    if center:
        mean_t = torch.as_tensor(np.asarray(train_mean), dtype=torch.float32).to(device)

    # Per-gene target standardization, X-target models only. loc/scale are train-fold
    # constants of shape (n_genes,). None is a no-op, which is the case for count
    # models and for runs with the transform off. The reasoning is at the in-loop
    # site below, where the target is actually mapped.
    loc_t = scale_t = None
    if target_loc is not None:
        loc_t   = torch.as_tensor(np.asarray(target_loc),   dtype=torch.float32).to(device)
        scale_t = torch.as_tensor(np.asarray(target_scale), dtype=torch.float32).to(device)

    # On the per-epoch path nothing leaves the device inside the loop, so the batch
    # iteration is sync-free apart from one .item() at the end. Only the final path
    # copies each block to the host, because the saved atoms and per-gene arrays come
    # from the scipy slide-aware metrics. That also keeps a per-batch device-to-host
    # copy of a (batch, n_genes) block off the per-epoch path at the all-genes scale.
    cpu_means, cpu_y, cpu_pred_var = [], [], []
    cpu_extra = {}             # extra prediction keys → list of cpu tensors
    gpu_means, gpu_y, gpu_pred_var = [], [], []
    # NB dispersion is kept on-device per epoch only when this pass computes NB PIT,
    # so the train pass never allocates the (N, G) alpha. Poisson has no dispersion;
    # its per-epoch PIT needs only mu and y.
    gather_disp = (metrics_on_gpu and nb_pit
                   and "dispersion" in spec.extra_prediction_columns)
    gpu_disp = []
    loss_sum = torch.zeros((), device=device)
    n_batches = 0
    has_pv = False

    model.eval()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            b = idx_t[start:start + batch_size]
            xb, yb = shared.gather(b, device=device)
            if center:
                xb = xb - mean_t

            raw = model(xb)
            outputs = spec.extract_predictions(raw)   # {means, pred_var, ...} on the target's own scale

            # Under an active transform the model predicts in the per-gene standardized
            # space, so the target moves there too and loss, metrics and saved
            # predictions all stay there. Keeping the saved predictions in that space
            # is the point of the transform: the predicted variance then carries no
            # gene magnitude. Per-(slide,gene) correlation and calibration are
            # invariant to a per-gene affine map, so they stay comparable with
            # untransformed runs. loc/scale are saved with the predictions, so
            # log-CP10k is recoverable offline as mu_y = scale*mu_z + loc.
            if loc_t is not None:
                yb = (yb - loc_t) / scale_t

            # Loss from the same forward pass, accumulated on the GPU with no
            # per-batch sync. Every model fits its likelihood directly on its own
            # target, so the only argument the loss needs beyond (raw, yb) is lam.
            # apply_spec_loss is the one site both this and the train loop call.
            loss_sum += apply_spec_loss(spec, raw, yb, lam=lam).detach()
            n_batches += 1

            m = outputs["means"]
            pv = outputs.get("pred_var")
            has_pv = has_pv or (pv is not None)
            if metrics_on_gpu:
                gpu_means.append(m); gpu_y.append(yb)
                if pv is not None:
                    gpu_pred_var.append(pv)
                if gather_disp:
                    gpu_disp.append(outputs["dispersion"])
            else:
                cpu_means.append(m.cpu()); cpu_y.append(yb.cpu())
                if pv is not None:
                    cpu_pred_var.append(pv.cpu())
                for k, v in outputs.items():
                    if k not in ("means", "pred_var"):
                        cpu_extra.setdefault(k, []).append(v.cpu())

    groups_np = shared.groups[idx]
    loss = (loss_sum / n_batches).item() if n_batches else float("nan")

    # Count specs report their correlation in log space, dividing by the run's single
    # depth constant `shared.depth_scale` before log1p: prediction vs log1p(y/div).
    # For the cp targets div = count_common_depth/1e4, so the axis is exactly
    # log1p(CP10k) == adata.X and the number is comparable to the Gaussian models.
    # For the depth-blind raw_counts target depth_scale is None, so div = 1 and the
    # axis is log1p(raw counts), comparable only to other raw runs. Loss, pred_var,
    # PIT and residuals stay on the model's own target scale and use no constant.
    #
    # The prediction side differs by path:
    #   Final inference uses g* = E[log1p(Y/div)] (count_expected_log1p), the
    #   predictive mean of log1p, which is the estimand the Gaussian models produce.
    #   log1p(mu/div) is kept beside it under the *_logmean keys.
    #   The per-epoch path uses log1p(mu/div). It is mean-only, so it needs no
    #   per-spot count sum and does not depend on a half-trained dispersion.
    # Selection therefore runs on log1p(mu/div) while the saved headline is g*. The
    # two differ only by the Jensen gap.
    #
    # The gate is keyed on target_source alone, and a missing depth constant is a hard
    # failure below. Letting a count spec with no depth_scale fall through to
    # compute_metrics would score an NB/Poisson predictive with Gaussian PIT bounds,
    # written under the very same ece_pit key, with no error raised.
    # COUNT_ROUTED_TARGETS is the same tuple the import-time loss/target guard in
    # specs.py checks, so this router and that guard cannot drift apart.
    count_metric = spec.target_source in COUNT_ROUTED_TARGETS
    if spec.target_source in ("cp_counts", "cp_counts_nr") and shared.depth_scale is None:
        raise RuntimeError(
            f"spec {spec.name!r} has target_source={spec.target_source!r} but "
            f"shared.depth_scale is None. The cp targets are defined at a specific "
            f"depth (count_common_depth/1e4), so their reported log1p axis cannot be "
            f"formed without it. Note raw_counts legitimately has depth_scale None "
            f"(it is aligned to no depth), which is why this guard is cp-only."
        )
    # cp_counts_nr trains on a continuous target, so its PIT calibration uses the
    # smeared CDF. A no-op for the integer-target count models.
    smear = (spec.target_source == "cp_counts_nr")

    # ── Per-epoch / early-stop path: slide-aware scalars on-device, no host copy ──
    # np.unique gives a contiguous 0..S-1 slide labelling. Both aggregations are
    # order-invariant over slides, so this matches the CPU scalar. The diagnostic MSE
    # and mean variance are pooled scalars, logged and never used for selection.
    # means, y and pred_var are returned as None here.
    if metrics_on_gpu:
        means_t = torch.cat(gpu_means)
        y_t     = torch.cat(gpu_y)
        pv_t    = torch.cat(gpu_pred_var) if gpu_pred_var else None
        # Single-gene models squeeze means and pred_var to (N,) while y stays (N, 1).
        # Align them to y's 2-D shape before any arithmetic, or the MSE diagnostic
        # broadcasts into an (N, N) cross-product. Multi-gene is a no-op.
        means_t = means_t.reshape(y_t.shape)
        if pv_t is not None:
            pv_t = pv_t.reshape(y_t.shape)
        # Correlation inputs: log1p over the depth constant for count specs, native
        # otherwise. depth_scale is a scalar, None for the raw targets.
        if count_metric:
            # The per-epoch count metric: predicted log1p(mu/div), observed
            # log1p(y/div), and the target-scale y for the MSE diagnostic and the
            # calibration, which both stay on the model's own scale.
            corr_means, corr_y, y_for_diag = count_corr_inputs(means_t, y_t,
                                                               shared.depth_scale)
        else:
            corr_means, corr_y = means_t, y_t
            y_for_diag = y_t
        _, slide_ids = np.unique(groups_np, return_inverse=True)
        slide_ids = slide_ids.reshape(-1)        # numpy 2.0.0 could return non-flat
        n_groups = int(slide_ids.max()) + 1
        gid = torch.as_tensor(slide_ids, dtype=torch.long, device=means_t.device)
        result = {
            "means": None, "y": None, "groups": groups_np, "pred_var": None,
            "loss": loss,
            "mse": float(((means_t - y_for_diag) ** 2).mean()),
            # MSE on the axis the correlation is reported on, from the same two
            # tensors corr_scalars_gpu consumes. For a count spec that is
            # log1p(mu/div) vs log1p(y/div), a different scale from "mse" above. For
            # every other spec corr_means and corr_y are means_t and y_t, so the two
            # agree to ~1e-9: same operands, different accumulation dtype, not
            # bit-identical.
            # A shared axis is what makes this usable as a cross-family selection
            # criterion (early_stop_criterion="mse"). "mse" above stays on each
            # model's own target scale and is therefore family-dependent.
            # Pooled, not slide-aware: MSE is not affine-invariant, so a per-slide
            # aggregation would be a different quantity, not a refinement.
            # float64 accumulation, the same rule as _pearson_cols. The operands stay
            # float32, but a selection scalar must not be at the mercy of float32
            # summation over a few million elements.
            "mse_corr_axis": float(((corr_means - corr_y) ** 2).mean(dtype=torch.float64)),
            "mean_pred_var": (float(pv_t.mean()) if pv_t is not None else None),
        }
        result.update(corr_scalars_gpu(corr_means, corr_y, gid, n_groups=n_groups,
                                       corr_metrics=corr_metrics))
        if pv_t is not None:
            # Native PIT per slice on-device. Gaussian uses erf on every pass. Count
            # specs pass count_pit_inputs = (mu, alpha_or_None, y_raw), all three on
            # the model's own fitted target scale, with no depth constant applied to
            # any of them. Poisson runs on every pass, its gammaincc being cheap. NB
            # runs only when this pass gathered the dispersion; otherwise it is
            # computed exactly at final inference. vrc and Gaussian PIT are float32.
            count_pit_inputs = None
            if count_metric:
                if gpu_disp:                                  # NB this pass
                    alpha_t = torch.cat(gpu_disp).reshape(y_t.shape)
                    count_pit_inputs = (means_t, alpha_t, y_t)
                elif "dispersion" not in spec.extra_prediction_columns:
                    count_pit_inputs = (means_t, None, y_t)   # Poisson — always
                # else: NB but nb_pit=False -> None (skip; final-inference only)
            result.update(calibration_scalars_gpu(means_t, y_for_diag, pv_t, gid,
                                                  n_groups=n_groups,
                                                  is_count_spec=count_metric,
                                                  count_pit_inputs=count_pit_inputs,
                                                  smear=smear))
        return result

    # ── Final path: host arrays + full CPU (scipy) slide-aware metrics ───────────
    means_np    = torch.cat(cpu_means).numpy()
    y_np        = torch.cat(cpu_y).numpy()
    pred_var_np = torch.cat(cpu_pred_var).numpy() if has_pv else None
    # Single-gene models squeeze means and pred_var to (N,) while y stays (N, 1).
    # Align them to y's 2-D shape so every downstream elementwise op pairs mean with
    # target row-for-row. Multi-gene is a no-op.
    means_np = means_np.reshape(y_np.shape)
    if pred_var_np is not None:
        pred_var_np = pred_var_np.reshape(y_np.shape)
    result = {
        "means": means_np, "y": y_np, "groups": groups_np,
        "pred_var": pred_var_np, "loss": loss,
    }
    # compute_cpu_metrics=False skips the per-(slide,gene) scipy and g* pass and
    # returns only the prediction matrices plus the extras below, so a run can save
    # predictions and recompute atoms and metrics offline. run_paired_inference in
    # the GCM pipeline carries the same switch.
    if not compute_cpu_metrics:
        result["pred_var"] = pred_var_np
        for k, lst in cpu_extra.items():
            result[k] = torch.cat(lst).numpy()
        return result
    if count_metric:
        # Count metrics via the shared helper, the same code the GCM's
        # run_paired_inference calls: the g* = E[log1p(Y/div)] headline, the
        # {metric}_logmean cross-check, and the calibration. means_np, y_np, alpha and
        # pred_var are all on the model's own fitted target scale, so mean and
        # dispersion always agree. depth_scale only forms the log1p reporting axis,
        # with div = 1 when it is None. alpha is the per-(spot,gene) NB dispersion
        # extra; its absence means Poisson.
        alpha_np = (torch.cat(cpu_extra["dispersion"]).numpy()
                    if "dispersion" in cpu_extra else None)
        result.update(count_metrics_final(means_np, y_np, shared.depth_scale,
                                          groups_np, corr_metrics,
                                          alpha=alpha_np, pred_var=pred_var_np,
                                          device=shared.emb.device, smear=smear))
    else:
        result.update(compute_metrics(means_np, y_np, groups=groups_np,
                                      pred_var=pred_var_np, corr_metrics=corr_metrics))
    # Both branches above overwrite the result's pred_var key. The count branch runs
    # its correlation pass with pred_var=None, so that key comes back None. Restore
    # the target-scale array so the caller still sees the variance it computed.
    result["pred_var"] = pred_var_np
    # Extra model-specific arrays (NB dispersion, epistemic variances, ...).
    for k, lst in cpu_extra.items():
        result[k] = torch.cat(lst).numpy()
    return result


# The count-metric self-test lives with count_expected_log1p:
#   python -m src.evaluation.count_metrics
