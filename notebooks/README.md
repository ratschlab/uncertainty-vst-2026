# Notebooks

Every figure and table of the paper, traced to the notebook that produces it, and the analysis
the paper is drawn from.

These notebooks read the outputs of the Snakemake sweep and of the post-hoc precomputes; what
has to be run before each of them is below. They are **not runnable as they stand**: the cohorts
are not distributed and the paths in `snake/config/config.yaml` are placeholders. They are here
so that any reported number can be traced to the code that computed it, and all of them keep
their stored outputs, so what they print can be read without running anything.

Three conventions are worth knowing before reading any of them:

- **Every figure has a printed twin.** The cell that draws a figure also prints the exact values
  it plotted, so the numbers are readable without regenerating the PDF.
- **`RUN_ID` selects the sweep** a notebook reads. `pr_classic3` is the headline run; it appears
  in the output filenames. The other run ids the notebooks name are listed, with the config
  change that produces each one, in the comments of `snake/config/config.yaml` next to
  `run_id_classic`.
- **`RESULTS`, set in the first cell, is the only path to change.** The project root itself is
  found by walking up from the kernel's working directory.

## Layout

| directory | contents |
|---|---|
| `paper_figures/` | one notebook per float of the paper |
| `three_channels_analysis/canal_1/` | channel 1: mean accuracy and calibration, per sweep and per selection rule |
| `three_channels_analysis/canal_2/` | channel 2: uncertainty against error, the two axes, sparsification, the mean confound |
| `three_channels_analysis/canal_2/unc_error_corr_vs_gene_predictability/` | is the uncertainty-error correlation concentrated on the genes predicted worst? asked of three uncertainty sources |
| `synthetic_artifact_response/` | the image-corruption dose response |
| `tissue_uncertainty/` | uncertainty by tissue type, pathway enrichment |
| `reason_for_failure/` | diagnostics for why the uncertainty is uninformative |
| `reproducing_the_numbers_on_any_predictions.ipynb` | every number of the paper rebuilt on one slide, from the functions of `src/`, plus how they are aggregated over patients |

The three channels are defined in the main README: 1 = performance and calibration, 2 =
informativeness, 3 = zero detection (computed, not reported).

## What has to be run first

Three stages, in order. A notebook opened before its stage has run finds no input and stops in its
first cell.

1. **The sweep**, `snakemake ... all_classic` (main README). It writes `results/{dataset}/{run_id}/.../`
   with `metrics.json`, `predictions/preds_fold{k}.h5ad`, `atoms.parquet` and `splits.json`.
   `all_train_ensemble_classic` adds the deep-ensemble members, and the post-hoc epistemic methods
   (`snake/scripts/epistemic/launch_epistemic.sh`) add a `test_summaries.h5ad` under `mc_dropout/`
   and `laplace/` in that same run directory.
2. **The precomputes**, each a `launch_*.sh` under `snake/scripts/` (indexed in its own README).
   They reduce the predictions into the small tables the notebooks read. The table below says
   which one each notebook needs.
3. **The notebooks**, with `RESULTS` in the first cell pointing at the tree from stage 1.

`atoms.parquet` and `preds_fold*.h5ad` come straight out of stage 1; everything else is a
precompute.

| the notebook reads | produced by |
|---|---|
| `benchmark_summary` | `canals/launch_canal1_benchmark_precompute.sh` |
| `canal2_uncertainty` | `canals/launch_canal2_precompute.sh` |
| `canal2_knn` | `canals/launch_canal2_knn_precompute.sh` (needs `canals/launch_gene_slide_nplus_regime_precompute.sh`, then `epistemic/launch_knn_uncertainty_precompute.sh`) |
| `canal2_epistemic`, `canal2_laplace` | `canals/launch_canal2_epistemic_precompute.sh`, `canals/launch_canal2_laplace_precompute.sh` |
| `canal2_uform` | `canals/launch_canal2_uform_precompute.sh` |
| `marginal_calibration` | `canals/launch_canal1_marginal_calibration_baseline_precompute.sh` |
| `dataset_gene_expression` | `canals/launch_dataset_gene_expression_stats_precompute.sh` |
| `epi_ale_density` | `canals/launch_epi_ale_density_precompute.sh` |
| `artifact_uncertainty`, `knn_uncertainty` | `epistemic/launch_artifact_uncertainty_precompute.sh`, `epistemic/launch_knn_uncertainty_precompute.sh` |
| `bio_axis`, `gene_variance`, `tissue_sigma`, `tissue_matched` | the four `bio/launch_*.sh` of the same name |
| `rf_axes`, `rf_gene_meanvar` | `reason_failure/launch_rf_axes_precompute.sh`, `reason_failure/launch_rf_gene_meanvar_precompute.sh` |

Per notebook, in the order the paper needs them:

| notebook | needs |
|---|---|
| `paper_figures/mean_performance_classic` | `benchmark_summary`, `dataset_gene_expression` |
| `paper_figures/calibration_classic`, `rounding_cost` | `benchmark_summary` |
| `paper_figures/unc_error_correlations_classic`, `unc_error_correlations_STI_vs_nb` | `benchmark_summary`, `canal2_uncertainty` |
| `paper_figures/sparsification_classic` | `canal2_uncertainty` |
| `paper_figures/sparsification_all_table` | `canal2_uncertainty`, `canal2_epistemic`, `canal2_laplace`, across four run ids: `pr_classic3` (each head's own uncertainty), `pr_classic1` (MC-dropout and Laplace), `de_6` (the deep ensemble), `sti_v1` (STimage) |
| `paper_figures/unc_error_correlation_vs_gene_predictability` | `benchmark_summary`, `canal2_uncertainty`, `atoms.parquet` |
| `paper_figures/unc_error_correlations_neighborhood_signal` | `benchmark_summary`, `canal2_knn` |
| `paper_figures/unc_error_correlations_epistemics` | `benchmark_summary`, `canal2_uncertainty`, `canal2_epistemic`, `canal2_laplace` |
| `paper_figures/two_estimators_comparison_for_counts_table` | `benchmark_summary`, `atoms.parquet` |
| `paper_figures/alternative_trials_for_robustness_table` | `benchmark_summary`, `canal2_uncertainty`, `canal2_uform` |
| `synthetic_artifact_response/` | `artifact_uncertainty`, `knn_uncertainty` (and the corrupted embeddings, `embeddings_extraction/launch_{blur,saltpepper}_extraction.sh`) |
| `tissue_uncertainty/tissue_uncertainty{,_ratio}` | `tissue_matched`, `tissue_sigma` |
| `tissue_uncertainty/pathway_enrichment` | `gene_variance`, `test_summaries` |
| `tissue_uncertainty/bio_axis_insight` | `bio_axis`, `tissue_sigma`, `atoms.parquet` |
| `reason_for_failure/` | `benchmark_summary`, `canal2_uncertainty`, `bio_axis`, `rf_axes`, `rf_gene_meanvar` |
| `three_channels_analysis/canal_1/` | `benchmark_summary`, plus `dataset_gene_expression` for the accuracy and calibration notebooks, `marginal_calibration` for `canal1_STimage_vs_DE`, `epi_ale_density` for `epistemic_over_aleatoric_distribution` |
| `three_channels_analysis/canal_2/` | `canal2_uncertainty`, plus `benchmark_summary` for all but the mean-confound, openings and lin_all notebooks, and `canal2_knn` / `canal2_epistemic` / `canal2_laplace` where the notebook's title names that source |

## The paper's floats

| float | notebook |
|---|---|
| mean performance vs expression | `paper_figures/mean_performance_classic.ipynb` |
| calibration | `paper_figures/calibration_classic.ipynb` |
| the two axes | `paper_figures/unc_error_correlations_classic.ipynb` |
| epistemic sources | `paper_figures/unc_error_correlations_epistemics.ipynb` |
| kNN neighborhood signal | `paper_figures/unc_error_correlations_neighborhood_signal.ipynb` |
| gene axis vs predictability | `paper_figures/unc_error_correlation_vs_gene_predictability.ipynb` |
| sparsification, all entries and detected entries | `paper_figures/sparsification_classic.ipynb` |
| the other uncertainty sources under sparsification (`app:othersources`) | `paper_figures/sparsification_all_table.ipynb` |
| STimage readout | `paper_figures/unc_error_correlations_STI_vs_nb.ipynb` |
| rounding cost | `paper_figures/rounding_cost.ipynb` |
| corruption dose response | `synthetic_artifact_response/synthetic_artifact_response.ipynb` |
| uncertainty by tissue, ratio | `tissue_uncertainty/tissue_uncertainty_ratio.ipynb` |
| `tab:estimator-gap` | `paper_figures/two_estimators_comparison_for_counts_table.ipynb` |
| `tab:robustness-gene`, `tab:robustness-spot` | `paper_figures/alternative_trials_for_robustness_table.ipynb` |
| `tab:sparsification-mu` | assembled by hand from the printed output of `paper_figures/sparsification_classic.ipynb` |

`figures/fig_overview.png` is a drawing, not a computed figure.

Only two notebooks emit LaTeX (`two_estimators_comparison_for_counts_table`, `alternative_trials_for_robustness_table`). Every other
table was assembled by hand from printed notebook output.

## The analysis behind them

The notebooks of `paper_figures/` are presentation code: each reads the precompute outputs
directly and draws one float. The notebooks below are where the comparisons were made. They
produce the figures of the thesis this paper is drawn from, which is why their filenames carry
run ids the paper does not use.

| notebook | what it establishes |
|---|---|
| `three_channels_analysis/canal_1/canal1_classic_aleatoric.ipynb` | mean accuracy per head, by expression bin and by top-n gene |
| `three_channels_analysis/canal_1/canal1_classic_aleatoric_calibration.ipynb` | the calibration numbers: PIT error, bias, dispersion, per head |
| `three_channels_analysis/canal_1/canal1_beta_nll_sweep.ipynb` | the beta-NLL exponent, one Gaussian head compared to itself across beta |
| `three_channels_analysis/canal_1/canal1_rounding_cost.ipynb` | what rounding the depth-corrected target to integers costs |
| `three_channels_analysis/canal_1/canal1_STimage_vs_DE.ipynb` | the STimage readout against the deep ensemble |
| `three_channels_analysis/canal_1/canal1_NLL_selection_classic.ipynb` | the accuracy battery under validation-loss early stopping |
| `three_channels_analysis/canal_1/canal1_NLL_selection_classic_calibration.ipynb` | its calibration counterpart |
| `three_channels_analysis/canal_1/canal1_MSE_selection_classic.ipynb` | the accuracy battery under MSE early stopping |
| `three_channels_analysis/canal_1/canal1_MSE_selection_classic_calibration.ipynb` | its calibration counterpart |
| `three_channels_analysis/canal_1/epistemic_over_aleatoric_distribution.ipynb` | the epistemic to aleatoric variance ratio, five heads: it bounds how much the full predictive would widen the intervals |
| `three_channels_analysis/canal_2/canal2_uncertainty_vs_error_classic.ipynb` | the two axes per donor and the mean-confounder proof, the channel's main notebook |
| `three_channels_analysis/canal_2/mean_confound.ipynb` | whether the predicted mean drives both the error and the predicted uncertainty |
| `three_channels_analysis/canal_2/canal2_view_per_tissue_and_genes.ipynb` | the same means opened group by group, by gene and by tissue |
| `three_channels_analysis/canal_2/canal2_uncertainty_vs_error_epistemic.ipynb` | the same two axes for the three post-hoc epistemic estimators |
| `three_channels_analysis/canal_2/unc_error_corr_vs_gene_predictability/classic.ipynb` | the reference reading: the head's own predicted spread against how well each gene is predicted |
| `three_channels_analysis/canal_2/unc_error_corr_vs_gene_predictability/epistemic.ipynb` | the same reading for the three post-hoc epistemic estimators |
| `three_channels_analysis/canal_2/unc_error_corr_vs_gene_predictability/knn.ipynb` | the same reading for the training-free kNN neighborhood signals |
| `tissue_uncertainty/tissue_uncertainty.ipynb` | uncertainty by annotated tissue type |
| `tissue_uncertainty/pathway_enrichment.ipynb` | pathway over-representation among the genes carrying the most uncertainty |
| `tissue_uncertainty/bio_axis_insight.ipynb` | the biological axis behind the gene ordering |
| `reason_for_failure/reason_for_failure.ipynb` | the diagnostics: what would have to be true for the uncertainty to be informative, and which of those fails |
