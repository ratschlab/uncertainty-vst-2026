# snake/scripts/ — script index

One-line purpose for the scripts listed here, organized to match the folder layout (`canals/`,
`epistemic/`, `embeddings_extraction/`, plus the training and utility drivers at the root).
This file answers one question: what each script does, and how it is launched.

Every `launch_*.sh` submits its paired `*.py` as one or more SLURM jobs (see the script's own
header comment for the exact `Usage` examples, env-var knobs, and `DRYRUN=1` preview mode).

Terms used throughout. A **run directory** is one trained model:
`results/{dataset}/{run_id}/{gene_set}/{radius}/{model_type}[/lambda_*]`, and it is what the
`--run-dir` flag takes. **Atoms** are the rows of a run's `atoms.parquet`, one per (fold, slide,
gene); every aggregate below is built from them. **LOPO** is leave-one-patient-out, the
cross-validation used here: each patient is the test set exactly once. A **canal** is one of the
three channels the analysis is split into, described in `canals/README.md`. **g\*** is what a
count model is scored on, its predicted mean on the `log1p` axis; `snake/config/config.yaml`
defines it next to `corr_metrics`.

## Root — training entrypoints + utilities

- **`train.py`** — Trains one classic model: it reads the preprocessed adata, trains the head named
  by `model_type`, and writes the run directory.
  One run per `(run_id_classic, dataset, gene_set, radius, model_type, [lambda])`. Called by
  Snakemake, not run directly.
- **`train_gcm.py`** — Trains one gene-conditioned model, on one GPU. The paired (spot, gene)
  counterpart of `train.py`, with two independent CV axes: donors on the spot side, a fixed
  gene holdout on the other.
- **`complete_deferred_metrics.py`** — Offline, CPU-only completion of the metrics a
  `compute_final_metrics=False` run deferred (recomputes `atoms.parquet` + `metrics.json` from
  saved predictions, no GPU/model forward needed). Handles both classic (`results/`) and GCM
  (`results_gcm/`) runs.
- **`gpu_summarize.py`** — Summarizes an `nvidia-smi` CSV log (mean/median/p90/p99 utilization,
  peak memory) into a short text report. It never crashes a training job if the CSV is
  missing/malformed.

## canals/ — Canal 1/2/3 precompute (performance, uncertainty-vs-error, zero-detection)

`canals/README.md` explains what the three channels are and which launcher runs which one. The
entries below are the one-line purpose of each file.

- **`canal1_benchmark_summary_precompute.py`** — Canal 1 performance precompute: per-gene/
  per-donor correlation + calibration aggregates. TWO mutually exclusive modes: `--run-dir` (a
  single classic run's `atoms.parquet`) or `--ensemble-dir` (a deep ensemble's `test_summaries.h5ad`
  — scores `g*` against the reconstructed true `log1p(y/depth_scale)`, directly comparable to a `--run-dir`
  count model's headline pearson. The ensemble mode reports the correlation and the calibration
  only, an ensemble carrying no per-run atoms to build the rest from).
  - **`launch_canal1_benchmark_precompute.sh`** — submits the `--run-dir` sweep (every classic run).
  - **`launch_canal1_ensemble_precompute.sh`** — submits the `--ensemble-dir` sweep for the
    ensemble-combined models listed at the top of the script.
- **`canal1_marginal_calibration_baseline_precompute.py`** +
  **`launch_canal1_marginal_calibration_baseline_precompute.sh`** — Canal 1 no-H&E
  marginal calibration baseline: fits a per-gene, per-family (Gaussian/Poisson/NB) distribution
  with NO image conditioning, scored with the same native-CDF PIT as the real models. ONE job per
  dataset (model-independent: one count run is enough to recover the folds, the genes and the
  path to the adata).
- **`dataset_gene_expression_stats_precompute.py`** +
  **`launch_dataset_gene_expression_stats_precompute.sh`** — Per-gene, per-dataset,
  model-independent expression statistics (mean expression, % zeros) from the merged preprocessed
  adata. One job per dataset.
- **`canal2_uncertainty_precompute.py`** — Canal 2: is the predicted uncertainty informative about
  the log-CP10k error, beyond the predicted mean? Central metric = per-(gene,slide) partial Spearman, restricted to
  `y>0`. THREE mutually exclusive modes behind one script: `--run-dir` (a single model's own
  `pred_var`/dispersion), `--ensemble-dir` (a deep ensemble's aleatoric/epistemic/total, from
  `epistemic/deep_ensemble_combine.py`'s output), `--mc-dropout-dir` (MC-dropout's
  aleatoric/epistemic/total, from `epistemic/mc_dropout_posthoc.py`'s output).
  - **`launch_canal2_precompute.sh`** — submits the `--run-dir` sweep (every classic model run).
  - **`launch_canal2_ensemble_precompute.sh`** — submits the `--ensemble-dir` sweep
    (`UNCERTAINTY_SOURCE=total|epistemic|aleatoric`) for the ensemble-combined models listed in
    the list at the top of the script.
  - **`launch_canal2_epistemic_precompute.sh`** — submits the `--mc-dropout-dir` sweep
    (`CHANNEL=epistemic|aleatoric|total`) for the specific models that have MC-dropout summaries.
- **`canal3_zero_detection_precompute.py`** + **`launch_canal3_precompute.sh`** — Canal 3:
  zero-detection AUROC (can the model's score distinguish true zeros from positives?), per-slide,
  `--run-dir` or `--ensemble-dir` mode.

- **`launch_canal2_knn_precompute.sh`** — Submits canal 2 for the training-free kNN signals
  (`canal2_uncertainty_precompute.py --knn-dir` + `--run-dir`): the non-parametric neighborhood
  distance scored against each model's error, on canal 2's own axes.
- **`launch_canal2_laplace_precompute.sh`** — Submits canal 2 for the runs that carry last-layer
  Laplace summaries (`--laplace-dir`, reading `<run-dir>/laplace/test_summaries.h5ad`).
- **`launch_canal2_uform_precompute.sh`** — Submits canal 2 under two alternative uncertainty
  forms, for the robustness appendix: the coefficient of variation `sqrt(var)/mu`, and the NB
  dispersion read alone as `1 + alpha`.
- **`epi_ale_density_precompute.py`** + **`launch_epi_ale_density_precompute.sh`** — Two
  descriptive views of how one head's two uncertainty components sit relative to each other. It
  feeds the epistemic-to-aleatoric ratio figure, which bounds how much the full predictive would
  widen the intervals.

- **`gene_slide_nplus_regime_precompute.py`** + **`launch_gene_slide_nplus_regime_precompute.sh`**
  — Per (gene, slide), the number of spots that detect the gene and the regime label that says what
  a statistic inside that pair can afford (3, 5 or 10 bins, or none). Model-independent, one pass
  over each preprocessed adata. Must run before the kNN precompute, which reads it.

## epistemic/ — post-hoc epistemic-uncertainty methods

- **`deep_ensemble_combine.py`** — Fans in M independently-trained seed-members' predictions into
  ONE per-model `test_summaries.h5ad` (mean, cross-seed epistemic variance, aleatoric ingredient —
  and, for the count heads, `g*`, the PIT bounds and the zero mass). Called from the
  Snakefile's `combine_ensemble_classic` / `combine_ensemble_gcm` rules.
  - **`launch_deep_ensemble_combine.sh`** — ad-hoc launcher (outside the Snakefile rules) for a
    manual sweep, for instance to combine two ensembles that sit under different run ids. ONE
    SLURM job PER (dataset, model) — `combine_model_dir` has no shared state across datasets, so
    never bundle several into one job via a `--model-dir` glob of globs (that only serializes
    embarrassingly parallel work). CPU-only.
- **`DE_gstar_precompute.py`** + **`launch_DE_gstar_precompute.sh`** — Precomputes
  `g* = E[log1p(Y/depth_scale)]` for ONE (seed, fold) prediction file of a deep-ensemble count-model member
  (the expensive part, a truncated sum over the count distribution; `deep_ensemble_combine.py`
  then averages the M per-seed values). One job per (seed, fold) — also usable on a single
  non-ensemble classic/GCM run's own prediction file, see its header. GPU by default (`GPUS=0` to
  force CPU) — `count_expected_log1p` is markedly faster on CUDA.
- **`mc_dropout_posthoc.py`** — Post-hoc MC-dropout for one classic run: T stochastic forward
  passes collapsed ONLINE into per-(spot,gene) moment summaries (pred/epistemic/aleatoric
  ingredient), written as one consolidated LOPO `test_summaries.h5ad`. CPU-only.
- **`laplace_posthoc.py`** — Post-hoc last-layer Laplace approximation: closed-form (no sampling)
  Gaussian posterior on the linear head, producing the same `test_summaries.h5ad` schema as
  MC-dropout so downstream notebooks are a drop-in replacement.
- **`launch_epistemic.sh`** — Submits MC-dropout and/or Laplace jobs (`METHODS=mc laplace`) for
  classic runs — ALWAYS as independent SLURM jobs (never chained; their cost profiles differ by
  orders of magnitude).
- **`artifact_uncertainty_precompute.py`** + **`launch_artifact_uncertainty_precompute.sh`** — Dose-response
  of aleatoric + epistemic uncertainty vs image-artifact severity (blur, salt-and-pepper). The model is
  frozen; only its INPUT changes (the same test spots re-embedded from corrupted slides). MC-dropout and
  the deep ensemble share one decomposition, so one script does both — only the draw source differs
  (dropout masks vs seed_*/ members). Gaussian heads only (log1p-CP10k, no depth_scale). Writes ONE small
  aggregate parquet (row per corruption x severity x donor). One GPU job per (dataset, method).
- **`knn_uncertainty_precompute.py`** + **`launch_knn_uncertainty_precompute.sh`** — Precomputes
  kNN local-uncertainty signals (empirical local dispersion, density/coverage score) plus OOD
  corruption/cross-dataset diagnostics, for one held-out LOPO donor of a reference heteroscedastic
  run. One job per (donor, stage).

## bio/ — is the uncertainty anything more than a property of the gene or the tissue?

- **`gene_variance_precompute.py`** + **`launch_gene_variance_precompute.sh`** — Per-gene variance
  references computed from the data alone, no model involved. The reference against which the
  predicted per-gene uncertainty is compared: a gene that is simply variable needs no image to
  look uncertain.
- **`tissue_sigma_precompute.py`** + **`launch_tissue_sigma_precompute.sh`** — Does the predicted
  uncertainty vary with the tissue type, and does it still vary once the predicted mean is
  accounted for?
- **`tissue_matched_precompute.py`** + **`launch_tissue_matched_precompute.sh`** — The same
  question asked at a *matched* predicted mean, so that a tissue difference cannot be the mean
  difference in disguise.
- **`bio_axis_precompute.py`** + **`launch_bio_axis_precompute.sh`** — Three reductions of the
  same per-fold quantities along the biological axes, for one classic run dir.

## reason_failure/ — what would have to be true for the uncertainty to be informative

- **`rf_gene_meanvar_precompute.py`** + **`launch_rf_gene_meanvar_precompute.sh`** — The
  mean-variance coupling in the data itself: how much of the per-gene variance is fixed by the
  per-gene mean, before any model is involved.
- **`rf_axes_precompute.py`** + **`launch_rf_axes_precompute.sh`** — Three aggregated readings of
  one run, the counterparts of canal 2's per-entry ones: whether the predicted uncertainty ranks the
  genes the way the predicted mean does, which of the two axes carries the variation, and how large
  that variation is where it exists.

## embeddings_extraction/ — corruption / distribution-shift patch-embedding extraction

- **`extract_clean_embeddings.py`** + **`launch_clean_extraction.sh`** — The CLEAN embeddings,
  step one of the whole pipeline and the input every training run consumes: each slide's full-res
  `.tif` becomes one embedding per spot. The three scripts below corrupt that same input; this
  one is the reference they are compared against.
- **`extract_radius_embeddings.py`** + **`launch_radius_extraction.sh`** — Extracts patch
  embeddings across a sweep of crop radii (field-of-view distribution-shift study). One GPU job
  covers all slides x all radii for a dataset.
- **`extract_saltpepper_embeddings.py`** + **`launch_saltpepper_extraction.sh`** — Extracts patch
  embeddings from slides corrupted with salt-and-pepper noise (sensor dead-pixel / dust-speck
  analog). One GPU job covers all slides x all noise amounts for a dataset.
- **`extract_blur_embeddings.py`** + **`launch_blur_extraction.sh`** — Extracts patch embeddings
  from slides degraded with a whole-slide Gaussian blur (out-of-focus scan analog; sigma scales
  with the spot diameter). Slide-outer: each slide is read once, then blurred per level. One GPU
  job covers all slides x all blur levels.

  **Merged LUNG_CANCER**: it has no image/meta tree of its own, so it is launched as its two
  source subtypes, `LUNG_CANCER_LUAD` (22 slides) + `LUNG_CANCER_LUSC` (14), which BOTH write
  into `patch_embeddings/LUNG_CANCER/...`; slide names are disjoint, so the two jobs fill the
  36-slide dir directly with no symlinking. Applies to blur and salt-and-pepper alike.
