# Experiment configs

One config per **spatial covariate mode**. Each run fine-tunes Chronos-2 on that
layout and evaluates the base ("baseline") and fine-tuned models over the same
forecast origins, so the two are directly comparable.

| file | `spatial.covariate_mode` | covariate rows per anchor | what the anchor sees |
| --- | --- | --- | --- |
| `mode_self_only.yaml` | `self_only` | 0 | nothing but its own burdens (ablation) |
| `mode_neighbours.yaml` | `neighbours` | `degree x burdens` (~18-40) | every neighbour's series individually |
| `mode_mean_all_others.yaml` | `mean_all_others` | `burdens` (2) | the mean series over all other MSOAs |
| `mode_neighbour_nonneighbour_means.yaml` | `neighbour_and_nonneighbour_means` | `2 x burdens` (4) | mean over neighbours and mean over non-neighbours |
| `mode_neighbour_mean.yaml` | `neighbour_mean` | `burdens` (2) | the mean over its neighbours only |
| `smoke_test.yaml` | `neighbour_and_nonneighbour_means` | 4 | tiny end-to-end check |

Cost scales as `sims x origins x horizon x num_samples x MSOAs`. Run
`python run_experiment.py --config <file> --estimate` first; shrink
`testing.origin_stride`, `testing.max_test_sims` or `testing.num_samples` if the
reported workload is too large. Keep those three identical across modes -- the
comparison is only meaningful on a shared evaluation grid.
