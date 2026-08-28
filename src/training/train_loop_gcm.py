# src/training/train_loop_gcm.py
#
# GCM (GeneConditionedMLP) single-fold training and inference on one GPU. The
# paired (spot, gene) counterpart of train_loop_indexed.py / inference_indexed.py.
#
#   • Data lives in a GPU-resident PairedSharedTensors (spot_emb / gene_emb / y,
#     each stored once). A batch gathers spot rows and gene columns separately, so
#     the (n_spots * n_genes) pair matrix is never materialized and no batch is
#     copied host->device.
#   • The training hot loop forms no pairs: it shuffles spot rows with
#     torch.randperm and, per step, predicts the whole gene-train block for that
#     spot batch in one matmul (see GeneHyperMLP). Full spot batches only, except
#     a split smaller than one batch, which runs whole. The batch loop issues no
#     host sync; the epoch-end reduction is the only one.
#   • Model behavior comes from the ModelSpec (loss_fn / uses_lam /
#     is_probabilistic / extract_predictions), so sweeping model types needs no
#     change here.
#   • Per-epoch metrics use the on-device slide-aware twins (corr_scalars_gpu /
#     calibration_scalars_gpu). The final inference, run by the orchestrator, uses
#     the CPU compute_metrics so the saved atoms and per-gene arrays are exact.
#
# Two-axis cross-validation:
#   The caller holds out a donor group (val spots) and a gene group (gene-val).
#   Per epoch the val spots are scored on the gene-TRAIN genes (quadrant ②: new
#   spot, seen gene) and on the gene-VAL genes (quadrant ④: new spot, new gene).
#   Early stopping uses a configurable combination of the two (see
#   early_stop_criterion); both are always logged, so the gene-generalization gap
#   (② − ④) is visible whichever criterion drives selection.

import numpy as np
import torch

from src.models.mlp import build_gene_conditioned_model
from src.models.specs import COUNT_ROUTED_TARGETS
from src.training.losses import apply_spec_loss
from src.evaluation.metrics_slide_aware import (
    compute_metrics, corr_scalars_gpu, calibration_scalars_gpu,
)
from src.evaluation.count_metrics import count_corr_inputs, count_metrics_final


# ─────────────────────────────────────────────────────────────────────────────
# Small helpers (kept here, kept few)
# ─────────────────────────────────────────────────────────────────────────────

def _select_scalar(seen, new, criterion):
    """Combine the seen-gene (②) and new-gene (④) val correlations into the
    early-stopping scalar. 'blend' averages whichever are finite, so if the
    new-gene signal is NaN (e.g. an empty/near-constant gene-val set) it
    gracefully degrades to the seen-gene number."""
    if criterion == "seen_genes":
        return seen
    if criterion == "new_genes":
        return new
    if criterion == "blend":
        vals = [v for v in (seen, new) if v is not None and np.isfinite(v)]
        return float(np.mean(vals)) if vals else float("nan")
    raise ValueError(
        f"Unknown early_stop_criterion {criterion!r}; "
        f"expected one of: blend | seen_genes | new_genes"
    )


def _epoch_line(fold, epoch, max_epochs, e, spec, primary, criterion):
    """One-line per-epoch summary (always shows the seen/new gene split + gap)."""
    seen = e[f"val_{primary}_seengenes"]
    new  = e[f"val_{primary}_newgenes"]
    gap  = e[f"val_{primary}_genegap"]
    sel  = e[f"val_{primary}_select"]
    line = (
        f"  Fold {fold} Ep {epoch + 1:2d}/{max_epochs} | "
        f"train {spec.loss_display_name}={e['train_loss']:+.4f} "
        f"{primary}={e[f'train_{primary}']:.4f} | "
        f"val {primary}: seen={seen:.4f} new={new:.4f} gap={gap:+.4f} "
        f"select({criterion})={sel:.4f}"
    )
    if spec.is_probabilistic and e.get("val_mean_pred_var") is not None:
        ece_seen = e.get("val_ece_pit_seengenes")
        line += (f" | ECEpit seen={ece_seen if ece_seen is not None else float('nan'):.4f} "
                 f"σ²={e['val_mean_pred_var']:.3f}")
    line += f" | lr={e['lr']:.1e}"
    return line


# ─────────────────────────────────────────────────────────────────────────────
# Paired inference — forward over a (spot subset × gene subset), on-device
# ─────────────────────────────────────────────────────────────────────────────

def run_paired_inference(model, pst, spot_rows, gene_cols, spec, *,
                         batch_size, device, lam, spot_mean,
                         corr_metrics=("pearson",), metrics_on_gpu=False,
                         target_loc=None, target_scale=None,
                         nb_pit=True, compute_cpu_metrics=True):
    """Forward the batched gene-conditioned `model` over (spot_rows × gene_cols) and
    return per-gene prediction matrices, slide-aware metrics and the eval loss.

    Iterates over spot batches of `batch_size` spots, scoring all gene_cols per
    batch. Each batch yields a (b_spots, n_genes) block; stacking the blocks along
    the spot axis gives the matrices the metrics consume, so the gene axis is never
    flattened away.

    metrics_on_gpu:
        True  -> compute only the slide-aware scalars the training loop needs
                 (corr_scalars_gpu, plus calibration_scalars_gpu for probabilistic
                 specs) on the data device. No scipy, no host round-trip. This is
                 the per-epoch / early-stop hot path.
        False -> compute the full CPU compute_metrics (per-(slide,gene) atoms,
                 per-gene arrays, coverage) for the saved artifacts. Used by the
                 orchestrator's final per-fold inference.

    compute_cpu_metrics (only meaningful when metrics_on_gpu=False):
        True  -> also run the per-(slide,gene) scipy metrics and g*.
        False -> return the prediction matrices only, so a run can save predictions
                 now and recompute the metrics offline.

    Returns a dict with groups (per spot), loss, and:
        metrics_on_gpu=True : the slide-aware scalars (corr per metric, plus
            ece/var_residual_corr for probabilistic specs) with `mse` and
            `mean_pred_var`, all computed on-device. means_2d/y_2d/pred_var_2d are
            None, which is what keeps the batch loop free of host syncs. The
            degenerate empty-split return below is the exception: it hands back
            zero-shaped means_2d/y_2d on either path.
        metrics_on_gpu=False: means_2d, y_2d, pred_var_2d host matrices plus the
            full compute_metrics keys, for the saved artifacts.
    """
    n_spots_local = int(spot_rows.numel())
    n_genes_local = int(gene_cols.numel())

    spot_rows = spot_rows.to(device)
    gene_cols = gene_cols.to(device)
    groups = pst.groups[spot_rows.cpu().numpy()]

    # Count specs report their correlation in log space against log1p(y/div), where
    # div is the run's single depth constant `pst.depth_scale`. This is the same
    # slide-aware count metric the classic pipeline uses, through the same
    # count_metrics helpers. For the cp targets that axis is log1p(CP10k), comparable
    # to the Gaussian models. For the depth-blind raw_counts target depth_scale is
    # None, so div = 1, the axis is log1p(raw counts), and it compares only against
    # other *_raw runs. Everything else stays on the model's own target scale.
    #
    # The gate keys on target_source alone, and a cp spec without a depth constant
    # fails loudly below. A count spec that instead fell through to compute_metrics
    # would have its NB/Poisson predictive scored with Gaussian PIT bounds under the
    # very same ece_pit key, and nothing would raise.
    # COUNT_ROUTED_TARGETS is the same tuple the loss/target guard in specs.py checks
    # at module load, so this router and that guard cannot drift apart.
    count_metric = spec.target_source in COUNT_ROUTED_TARGETS
    if spec.target_source in ("cp_counts", "cp_counts_nr") and pst.depth_scale is None:
        raise RuntimeError(
            f"spec {spec.name!r} has target_source={spec.target_source!r} but "
            f"pst.depth_scale is None. The cp targets are defined at a specific "
            f"depth (count_common_depth/1e4), so their reported log1p axis cannot be "
            f"formed without it. Note raw_counts legitimately has depth_scale None "
            f"(it is aligned to no depth), which is why this guard is cp-only."
        )
    # cp_counts_nr trains on a CONTINUOUS target -> its PIT calibration uses the smeared
    # CDF F~ (calibration_pit smear=True); a no-op for the integer-target count models.
    smear = (spec.target_source == "cp_counts_nr")

    # Degenerate split (empty gene-val set or no spots): return NaN metrics rather
    # than crashing on torch.cat([]). Callers gate on this where it matters.
    if n_spots_local == 0 or n_genes_local == 0:
        empty = np.zeros((n_spots_local, n_genes_local), dtype=np.float32)
        result = {"means_2d": empty, "y_2d": empty.copy(), "pred_var_2d": None,
                  "groups": groups, "loss": float("nan"),
                  "mse": float("nan"), "mean_pred_var": None}
        for cm in corr_metrics:
            result[cm] = float("nan")
        if spec.is_probabilistic:
            for k in ("ece_pit", "pit_bias", "disp_signed",
                      "var_residual_corr", "var_residual_corr_partial"):
                result[k] = float("nan")
        return result

    # On the per-epoch path nothing leaves the device inside the loop: blocks stay on
    # the GPU and corr, calibration and MSE are computed there, so the eval iteration
    # issues no host sync until the end. Only the final path copies each block to the
    # host, which the saved atoms, the per-gene arrays and the scipy slide-aware
    # metrics all need, and which avoids holding a second all-genes matrix on the GPU
    # next to the resident pst.y.
    cpu_means, cpu_y, cpu_pv = [], [], []
    cpu_extra = {}
    gpu_means, gpu_y, gpu_pv = [], [], []
    # NB dispersion is kept on-device per epoch only when this pass computes NB PIT,
    # so the train pass never allocates the (N, G) alpha. Poisson has no dispersion.
    gather_disp = (metrics_on_gpu and nb_pit
                   and "dispersion" in spec.extra_prediction_columns)
    gpu_disp = []
    loss_sum = torch.zeros((), device=device)
    n_batches = 0
    has_pv = False

    # X-target standardization, Gaussian and MSE models only: map the target to
    # z-space so the loss, the metrics and the saved predictions all live there, which
    # makes σ scale-free. target_loc/scale cover every gene, so subset them to this
    # call's gene_cols. None for count models and untransformed runs.
    loc_t = scale_t = None
    if target_loc is not None:
        loc_t   = torch.as_tensor(np.asarray(target_loc),   dtype=torch.float32, device=device)[gene_cols]
        scale_t = torch.as_tensor(np.asarray(target_scale), dtype=torch.float32, device=device)[gene_cols]

    model.eval()
    with torch.no_grad():
        for start in range(0, n_spots_local, batch_size):
            rows = spot_rows[start:start + batch_size]
            Xs, E, Yb, _ = pst.gather_block(rows, gene_cols, spot_mean=spot_mean)
            if loc_t is not None:                           # target -> z-space
                Yb = (Yb - loc_t) / scale_t
            raw = model(Xs, E)
            outputs = spec.extract_predictions(raw)        # {means (b,G), pred_var, ...}
            # Every model, count models included, fits its likelihood on its own
            # target, so the loss needs nothing beyond (raw, Yb) and lam.
            loss_sum += apply_spec_loss(spec, raw, Yb, lam=lam).detach()
            n_batches += 1

            m = outputs["means"]
            pv = outputs.get("pred_var")
            has_pv = has_pv or (pv is not None)
            if metrics_on_gpu:
                gpu_means.append(m); gpu_y.append(Yb)
                if pv is not None:
                    gpu_pv.append(pv)
                if gather_disp:
                    gpu_disp.append(outputs["dispersion"])
            else:
                cpu_means.append(m.cpu()); cpu_y.append(Yb.cpu())
                if pv is not None:
                    cpu_pv.append(pv.cpu())
                for k, v in outputs.items():
                    if k not in ("means", "pred_var"):
                        cpu_extra.setdefault(k, []).append(v.cpu())

    loss = (loss_sum / n_batches).item() if n_batches else float("nan")

    # ── Per-epoch / early-stop path: slide-aware SCALARS on-device, no host copy ──
    if metrics_on_gpu:
        means_t = torch.cat(gpu_means, dim=0)
        y_t     = torch.cat(gpu_y, dim=0)
        pv_t    = torch.cat(gpu_pv, dim=0) if gpu_pv else None
        # Count specs: correlate option-A log1p(mu/div) vs log1p(y/div) (shared
        # helper), div from the run's scalar depth constant (None for *_raw). The MSE
        # diag + calibration use the RAW target-scale y. Gaussian: identity.
        if count_metric:
            corr_means, corr_y, y_for_diag = count_corr_inputs(means_t, y_t,
                                                               pst.depth_scale)
        else:
            corr_means, corr_y, y_for_diag = means_t, y_t, y_t
        _, slide_ids = np.unique(groups, return_inverse=True)
        n_groups = int(slide_ids.max()) + 1
        gid = torch.as_tensor(slide_ids.reshape(-1), dtype=torch.long, device=means_t.device)
        result = {
            "means_2d": None, "y_2d": None, "pred_var_2d": None,
            "groups": groups, "loss": loss,
            "mse": float(((means_t - y_for_diag) ** 2).mean()),         # pooled, diagnostic
            "mean_pred_var": (float(pv_t.mean()) if pv_t is not None else None),
        }
        result.update(corr_scalars_gpu(corr_means, corr_y, gid, n_groups=n_groups,
                                       corr_metrics=corr_metrics))
        if pv_t is not None:
            # Native PIT per slice on-device. Gaussian uses erf on every pass. Count
            # builds it from count_pit_inputs = (mu, alpha_or_None, y_raw), all three
            # on the model's own fitted target scale, with no depth constant applied
            # to the mean. Poisson runs on every pass, its CDF being cheap; NB only
            # when this pass gathered the dispersion, so the train pass skips NB PIT
            # and the exact value comes from the final inference.
            count_pit_inputs = None
            if count_metric:
                if gpu_disp:                                  # NB this pass
                    alpha_t = torch.cat(gpu_disp, dim=0).reshape(y_t.shape)
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

    # ── Final path: host matrices + full CPU metrics for the saved artifacts ─────
    means_2d = torch.cat(cpu_means, dim=0).numpy()
    y_2d     = torch.cat(cpu_y, dim=0).numpy()
    pv_2d    = torch.cat(cpu_pv, dim=0).numpy() if has_pv else None
    result = {"means_2d": means_2d, "y_2d": y_2d, "pred_var_2d": pv_2d,
              "groups": groups, "loss": loss}
    # compute_cpu_metrics=False -> skip the per-(slide,gene) scipy / g* pass entirely
    # and return only the prediction matrices (+ extras below). Lets a fast run save
    # predictions now and recompute atoms/metrics offline.
    if compute_cpu_metrics:
        if count_metric:
            # The same shared count metrics the classic final inference uses, with both
            # log1p estimators and their full per-gene and per-(slide,gene) fan-out:
            # g* (option B, the headline, uses mu and alpha) under {metric}, and
            # log1p(mu/div) (option A, mean only) under {metric}_logmean, plus
            # calibration. Both pass straight through, so atoms_dataframe_gcm writes
            # one atoms.parquet column per estimator. means, y, alpha and pred_var are
            # all on the model's own fitted target scale; depth_scale only forms the
            # log1p reporting axis, with div = 1 when it is None.
            alpha_np = (torch.cat(cpu_extra["dispersion"], dim=0).numpy()
                        if "dispersion" in cpu_extra else None)
            result.update(count_metrics_final(means_2d, y_2d, pst.depth_scale,
                                              groups, corr_metrics,
                                              alpha=alpha_np, pred_var=pv_2d, device=pst.device,
                                              smear=smear))
        else:
            result.update(compute_metrics(means_2d, y_2d, groups=groups,
                                          pred_var=pv_2d, corr_metrics=corr_metrics))
    for k, lst in cpu_extra.items():
        result[k] = torch.cat(lst, dim=0).numpy()
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Single-fold training (two-axis CV, configurable early stopping)
# ─────────────────────────────────────────────────────────────────────────────

def train_one_fold_gcm(
    *,
    fold,
    model_type,
    spec,
    lam,
    pst,                          # PairedSharedTensors (GPU-resident)
    train_spot_rows, val_spot_rows,
    train_metric_spot_rows,       # per-slide subsample of train spots (per-epoch metric)
    gene_train_cols, gene_val_cols,   # GLOBAL gene columns for this fold's gene split
    spot_dim, gene_dim,
    arch, hyper_hidden, hyper_dropout, interaction_dim,
    lr, weight_decay, batch_size, max_epochs, patience,
    warmup_epochs, grad_clip_norm,
    center, early_stop_criterion,
    corr_metrics, seed, device,
    amp_dtype=None,               # autocast dtype for the TRAIN forward (None = fp32)
    log_alpha_init=-2.0,          # count (NB) models: initial dispersion for the GCM head
    target_loc=None, target_scale=None,   # X-target z-transform stats (full ALL_GENES; None=off)
    nb_per_epoch_calibration=True,  # compute NB per-epoch PIT on the val passes (toggle)
):
    """Train one fold of the GCM on a single GPU; return best state + epoch log.

    All row/column inputs are positional indices into `pst` (global spot rows,
    global gene columns). Centering subtracts the TRAIN-fold spot-embedding mean
    from the SPOT block only (gene embeddings are fixed external vectors); the
    mean + center flag are returned so the orchestrator can save them with the
    checkpoint and reproduce predictions.

    Early stopping maximizes `_select_scalar(seen, new, early_stop_criterion)`
    where seen = val corr on gene-TRAIN genes (②) and new = val corr on gene-VAL
    genes (④), both using the primary metric corr_metrics[0]. NaN-fallback: if the
    selection scalar is never finite, the final-epoch weights are kept.
    """
    primary = corr_metrics[0]
    dev = pst.device

    # Index arrays -> LongTensors on the data device (tiny; moved once).
    train_spot_rows        = torch.as_tensor(np.asarray(train_spot_rows),        dtype=torch.long, device=dev)
    val_spot_rows          = torch.as_tensor(np.asarray(val_spot_rows),          dtype=torch.long, device=dev)
    train_metric_spot_rows = torch.as_tensor(np.asarray(train_metric_spot_rows), dtype=torch.long, device=dev)
    gene_train_cols        = torch.as_tensor(np.asarray(gene_train_cols),        dtype=torch.long, device=dev)
    gene_val_cols          = torch.as_tensor(np.asarray(gene_val_cols),          dtype=torch.long, device=dev)
    n_gene_train = int(gene_train_cols.numel())
    n_gene_val   = int(gene_val_cols.numel())

    # Centering: spot-block mean over TRAIN spots only (no leakage). Kept as a
    # device tensor for the gather; saved as numpy for reproduction.
    spot_mean = pst.spot_emb[train_spot_rows].mean(dim=0) if center else None
    spot_mean_np = spot_mean.detach().cpu().numpy() if center else None

    # Model: seed before construction so the init is reproducible per fold.
    torch.manual_seed(seed + fold)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + fold)
    model = build_gene_conditioned_model(
        model_type, spot_dim, gene_dim, device,
        arch=arch, hyper_hidden=hyper_hidden, hyper_dropout=hyper_dropout,
        interaction_dim=interaction_dim, head=spec.gcm_head,
        log_alpha_init=log_alpha_init,
    )

    # X-target z-transform stats (full ALL_GENES arrays) -> device tensors. The train
    # pass subsets to gene_train_cols; run_paired_inference subsets to its own gene_cols.
    loc_t = scale_t = None
    if target_loc is not None:
        loc_t   = torch.as_tensor(np.asarray(target_loc),   dtype=torch.float32, device=dev)
        scale_t = torch.as_tensor(np.asarray(target_scale), dtype=torch.float32, device=dev)
    # fused=True runs the whole Adam step in one CUDA kernel, which cuts the launch
    # count on this small, launch-bound model. It is asked for only when CUDA is
    # present, and a torch whose Adam has no such keyword raises TypeError, so fall
    # back to the plain step.
    try:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay,
                                     fused=torch.cuda.is_available())
    except TypeError:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    n_train_spots = int(train_spot_rows.numel())

    best_val_metric = float("-inf")
    best_state = None
    patience_count = 0
    epoch_log = []

    # Mixed precision. The caller picks the dtype from the GPU's compute capability.
    # bf16 keeps fp32's exponent range so it needs no GradScaler; fp16 needs one.
    # amp_dtype None disables both the autocast and the scaler, leaving a pure fp32
    # loop. Only the train forward is autocast: eval and inference stay fp32, so the
    # saved metrics are full precision.
    amp_enabled = amp_dtype is not None
    scaler = torch.amp.GradScaler('cuda', enabled=(amp_dtype == torch.float16))

    for epoch in range(max_epochs):
        # Optional linear LR warmup over the first `warmup_epochs` (then hold).
        if warmup_epochs > 0:
            wf = min(1.0, (epoch + 1) / warmup_epochs)
            for g in optimizer.param_groups:
                g["lr"] = lr * wf

        # ── Training pass — SYNC-FREE, batched over SPOTS (all gene-train genes
        # per step). Each step runs the spot backbone on `batch_size` spots and the
        # gene hypernetwork on this fold's gene-train genes, then predicts the whole
        # (batch_size × n_gene_train) block in one matmul (no pairs materialized).
        model.train()
        running = torch.zeros((), device=dev)
        grad_norm_sum = torch.zeros((), device=dev)   # mean pre-clip grad norm
        nb = 0
        # Shuffle spot rows each epoch; the gene axis is taken whole per batch. The
        # permutation is one int64 per train spot, small enough to live on-device.
        perm = torch.randperm(n_train_spots, device=dev)
        n_full = (n_train_spots // batch_size) * batch_size
        n_full = n_full if n_full > 0 else n_train_spots          # tiny split -> one batch
        for start in range(0, n_full, batch_size):
            rows = train_spot_rows[perm[start:start + batch_size]]
            Xs, E, Yb, _ = pst.gather_block(rows, gene_train_cols, spot_mean=spot_mean)
            if loc_t is not None:                           # target -> z-space (train genes)
                Yb = (Yb - loc_t[gene_train_cols]) / scale_t[gene_train_cols]
            optimizer.zero_grad()
            # Only the forward matmuls are autocast; the loss is computed outside it,
            # so the count NLLs, with their exp and lgamma, never run under autocast's
            # reduced precision. The loss is on the small (B, G) output, so keeping it
            # out costs almost nothing.
            with torch.autocast(device_type="cuda",
                                dtype=(amp_dtype if amp_enabled else torch.bfloat16),
                                enabled=amp_enabled):
                raw = model(Xs, E)
            loss = apply_spec_loss(spec, raw, Yb, lam=lam)
            scaler.scale(loss).backward()
            # Unscale on every step, so the logged gradient norm is in real scale.
            # The single clip_grad_norm_ call both clips, when grad_clip_norm is set,
            # and returns the total pre-clip norm; max_norm=inf makes it measure only.
            scaler.unscale_(optimizer)
            gnorm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip_norm if grad_clip_norm else float("inf"))
            grad_norm_sum += gnorm.detach()
            scaler.step(optimizer)
            scaler.update()
            running += loss.detach()
            nb += 1
        running_train_loss = (running / max(nb, 1)).item()
        grad_norm_mean = (grad_norm_sum / max(nb, 1)).item()

        # ── Eval passes (on-device scalars) ──────────────────────────────────
        # nb_pit=False on the train-metric pass: NB calibration there is diagnostic
        # only, so its NB ece_pit stays NaN and the NB CDF cost is skipped. The val
        # passes compute NB PIT when the config toggle is on. Gaussian and Poisson PIT
        # are computed on every pass.
        train_res = run_paired_inference(
            model, pst, train_metric_spot_rows, gene_train_cols, spec,
            batch_size=batch_size, device=device, lam=lam, spot_mean=spot_mean,
            corr_metrics=corr_metrics, metrics_on_gpu=True,
            target_loc=target_loc, target_scale=target_scale, nb_pit=False,
        )
        val_seen = run_paired_inference(
            model, pst, val_spot_rows, gene_train_cols, spec,
            batch_size=batch_size, device=device, lam=lam, spot_mean=spot_mean,
            corr_metrics=corr_metrics, metrics_on_gpu=True,
            target_loc=target_loc, target_scale=target_scale,
            nb_pit=nb_per_epoch_calibration,
        )
        val_new = None
        if n_gene_val > 0:
            val_new = run_paired_inference(
                model, pst, val_spot_rows, gene_val_cols, spec,
                batch_size=batch_size, device=device, lam=lam, spot_mean=spot_mean,
                corr_metrics=corr_metrics, metrics_on_gpu=True,
                target_loc=target_loc, target_scale=target_scale,
                nb_pit=nb_per_epoch_calibration,
            )

        # ── Epoch-log row ────────────────────────────────────────────────────
        e = {
            "epoch":              epoch + 1,
            "lr":                 optimizer.param_groups[0]["lr"],
            "running_train_loss": running_train_loss,
            "grad_norm":          grad_norm_mean,
            "train_loss":         train_res["loss"],
            "val_loss_seengenes": val_seen["loss"],
            "val_loss_newgenes":  (val_new["loss"] if val_new is not None else float("nan")),
            "train_mse":          train_res["mse"],
            "val_mse_seengenes":  val_seen["mse"],
        }
        for cm in corr_metrics:
            seen_cm = val_seen[cm]
            new_cm  = val_new[cm] if val_new is not None else float("nan")
            e[f"train_{cm}"]          = train_res[cm]
            e[f"val_{cm}_seengenes"]  = seen_cm
            e[f"val_{cm}_newgenes"]   = new_cm
            e[f"val_{cm}_genegap"]    = (
                seen_cm - new_cm
                if (np.isfinite(seen_cm) and np.isfinite(new_cm)) else float("nan")
            )
            e[f"val_{cm}_select"]     = _select_scalar(seen_cm, new_cm, early_stop_criterion)
        if spec.is_probabilistic:
            # Native-CDF PIT calibration, seen-gene (②) + new-gene (④) variants for
            # ece_pit; the directional diagnostics (pit_bias/disp_signed) + the
            # uncertainty-error corrs are logged on the seen-gene quadrant.
            e["train_ece_pit"]             = train_res.get("ece_pit")
            e["val_ece_pit_seengenes"]     = val_seen.get("ece_pit")
            e["val_ece_pit_newgenes"]      = (val_new.get("ece_pit") if val_new is not None else float("nan"))
            e["val_pit_bias_seengenes"]    = val_seen.get("pit_bias")
            e["val_disp_signed_seengenes"] = val_seen.get("disp_signed")
            e["val_var_residual_corr_seengenes"]         = val_seen.get("var_residual_corr")
            e["val_var_residual_corr_partial_seengenes"] = val_seen.get("var_residual_corr_partial")
            e["train_var_residual_corr"]                 = train_res.get("var_residual_corr")
            e["train_mean_pred_var"] = train_res.get("mean_pred_var")
            e["val_mean_pred_var"]   = val_seen.get("mean_pred_var")
        epoch_log.append(e)
        print(_epoch_line(fold, epoch, max_epochs, e, spec, primary, early_stop_criterion),
              flush=True)

        # ── Early stopping on the selection scalar (primary metric) ──────────
        select = e[f"val_{primary}_select"]
        if np.isfinite(select) and select > best_val_metric:
            best_val_metric = select
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"  Fold {fold}: early stopping at epoch {epoch + 1}", flush=True)
                break

    # The selection scalar can be non-finite at every epoch, for instance a NaN
    # new-gene correlation under the new_genes criterion. Keep the final-epoch
    # weights in that case, so the run still completes.
    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    return {
        "best_state":      best_state,
        "best_val_select": float(best_val_metric),
        "epoch_log":       epoch_log,
        "spot_mean":       spot_mean_np,
        "center":          bool(center),
        "model":           model,
        "target_loc":      target_loc,      # full ALL_GENES (None if no transform)
        "target_scale":    target_scale,
    }
