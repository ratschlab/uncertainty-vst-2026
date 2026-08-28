# The three channels

A run produces predictions. These scripts turn them into small tables that the notebooks read.
The work is split into three channels, each asking one question. `canal` is French for channel.

| channel | question | script |
|---|---|---|
| 1 | Is the predicted mean accurate, and is the predicted distribution calibrated? | `canal1_benchmark_summary_precompute.py` |
| 2 | Does the uncertainty rank the errors, beyond what the predicted mean already explains? | `canal2_uncertainty_precompute.py` |
| 3 | Does the model know which measurements are true zeros? | `canal3_zero_detection_precompute.py` |

Channels 1 and 2 carry the paper. Channel 3 runs by default and writes its output, but the paper
does not report it.

Each script takes one run directory and writes next to it. The tables are small, a few kilobytes,
so comparing twenty models means reading twenty small files instead of the predictions themselves.

## Running them

Snakemake already runs all three as part of `all_classic`. Nothing here has to be launched by
hand for the main sweep.

The `launch_*.sh` scripts exist for the cases Snakemake does not cover: a different uncertainty
source, or a sweep that was trained separately. Each submits one SLURM job per run directory.
Every one of them accepts `DRYRUN=1` to print what it would submit.

| launcher | runs channel | on |
|---|---|---|
| `launch_canal1_benchmark_precompute.sh` | 1 | each model's own uncertainty |
| `launch_canal1_ensemble_precompute.sh` | 1 | a deep ensemble |
| `launch_canal2_precompute.sh` | 2 | each model's own uncertainty |
| `launch_canal2_ensemble_precompute.sh` | 2 | a deep ensemble |
| `launch_canal2_epistemic_precompute.sh` | 2 | MC-dropout |
| `launch_canal2_laplace_precompute.sh` | 2 | last-layer Laplace |
| `launch_canal2_knn_precompute.sh` | 2 | the training-free neighborhood signal |
| `launch_canal2_uform_precompute.sh` | 2 | two alternative forms of the uncertainty, for the robustness appendix |
| `launch_canal3_precompute.sh` | 3 | each model's own uncertainty |

Channel 2 has the most launchers because the paper compares several sources of uncertainty on the
same axes. They all call `canal2_uncertainty_precompute.py`; only the input directory changes.

## The other scripts here

They are not channels. They sit here because the channel notebooks read them. Each has a
`launch_` script of the same name.

- `canal1_marginal_calibration_baseline_precompute.py` fits a distribution per gene with no image
  input at all, and scores it with the same calibration metric. It is the floor a model has to
  beat: a model that only learned the marginal distribution of each gene would land here.
- `dataset_gene_expression_stats_precompute.py` measures each gene in each cohort, straight from
  the data. The figures use it to order genes by how much they are expressed.
- `gene_slide_nplus_regime_precompute.py` counts, per (gene, slide), how many spots detect the
  gene, and labels the pair with what that sample size allows: too few to use, usable in 3 bins,
  in 5, or in 10. The kNN precompute reads the label to decide how to stratify, so it has to run
  before it.
- `epi_ale_density_precompute.py` compares the epistemic and the aleatoric part of one model's
  uncertainty. It feeds the ratio figure, which bounds how much wider the intervals would be if
  the epistemic part were included.
