# ChronoGenie: Chronos-2 with spatial covariates

A foundation-model counterpart to the GNN forecasters in `../GENIE`. Same data,
same forecast task, same scoring rules, same output files — so results from the
two repos drop into one comparison.

---

## 1. The forecasting task

For each simulation, each forecast origin `t0`, and each MSOA, produce a
probabilistic forecast of the next `H` days (default 60) of every burden
(`daily_hospitalised`, `deaths`), using only data from before `t0`.

Chronos-2 is a 1-step-ahead model here, so the horizon is covered by an
**autoregressive rollout**: predict day `t`, sample a realisation, append it to the
context, predict day `t+1`. That mirrors what the GENIE models do.

### Sample paths

Uncertainty is carried by `S` **independent trajectories** (`testing.num_samples`).
Path `s` keeps its own rolling history: at every step it draws one inverse-CDF
realisation from the predicted quantiles and writes *that* value back into *its own*
context. The `S` paths are then the predictive sample used by every score.

This is the part that has to be right. Drawing `S` marginal samples at each step
around a *single* shared history would make the forecast spread at lead 30 look like
one-step-ahead uncertainty, and would break the cross-MSOA dependence that the energy
and variogram scores exist to measure. `test_pipeline.py` asserts both that the paths
differ and that their spread grows with lead time.

### Where the spatial information enters

Each Chronos input item is one **anchor** MSOA:

| rows | content | role |
| --- | --- | --- |
| `C` | the anchor's own burden series | **targets** (forecast jointly, so cross-burden structure is available) |
| `R` | spatial context, per covariate mode | past-only **covariates** |
| `2` | weekday sin/cos, if enabled | **known-future** covariate |

Only the anchor's target rows are read back. Covariates are never forecast — they are
context. Chronos-2's grouped attention means each anchor sees exactly the series it was
given and nothing else, which is what makes the covariate modes a genuine comparison.

---

## 2. Neighbourhoods

`spatial_context.build_neighbour_map` runs the two k-NN searches GENIE's
`graph_construction.get_graph` uses:

1. k-NN (`knn_k=9`, Euclidean) in **scaled static-feature** space,
2. k-NN (`knn_k=9`, haversine) in **raw geographic** space,

and takes the **per-node union**. On the 84-MSOA dataset that gives degree 11–18
(mean 15.4); the two sets overlap by about a third.

Neighbours are ordered nearest-first by great-circle distance, so the row order is
deterministic and identical between fine-tuning and testing.

Two knobs, both off by default:

* `spatial.symmetrise` — also add reciprocal edges. This reproduces the *undirected
  graph* the GENIE GNNs message-pass over (`to_undirected`), mean degree 20. Off,
  because as a covariate list the intended relation is "the MSOAs this one is nearest
  to", not "plus every MSOA that happens to pick this one".
* `spatial.max_neighbours` — cap the degree, nearest first. The main cost lever for
  the `neighbours` mode.

The exact map used is written to `neighbour_map.json` in every run directory.

---

## 3. Covariate modes

One config per mode, in `configs/experiments/`. All four the comparison needs, plus an
ablation:

| mode | covariate rows | what the anchor sees |
| --- | --- | --- |
| `self_only` | 0 | nothing but its own burdens (ablation floor) |
| `neighbours` | `degree x C` (~31) | every neighbour's series, individually |
| `mean_all_others` | `C` (2) | the mean series over all *other* MSOAs |
| `neighbour_mean` | `C` (2) | the mean over its neighbours |
| `neighbour_and_nonneighbour_means` | `2C` (4) | neighbour mean and non-neighbour mean, separately |

The mean modes are computed as one `[M, M] @ [M, C*L]` product per rollout step, so
they cost nothing beyond the extra rows.

---

## 4. Baseline vs fine-tuned

Every config evaluates both, on the same origins, sims and seed:

```
<output_dir>/baseline/ARW_60/TESTING/SIM_*/...    pre-trained Chronos-2
<output_dir>/finetuned/ARW_60/TESTING/SIM_*/...   after LoRA fine-tuning
```

Fine-tuning uses the **same covariate layout** as testing — the same
`SpatialCovariateBuilder` object builds both, so they cannot drift apart. Training
items are whole simulations: Chronos-2's training loader samples a random context
window from each series on every step, so one item per (sim, anchor) already covers
every window. Validation items are truncated copies, because Chronos-2 validates on
the *last* window of whatever it is handed, and a single truncation would only ever
score the epidemic tail.

Count targets are jittered (`y -> floor(y + U(0,1))`) during fine-tuning so a
continuous quantile loss can fit discrete counts; sampled counts are floored at test
time to match. Set `count_dequantisation: false` to turn both off.

---

## 5. Outputs

Identical in schema to GENIE, per window:

```
<variant>/ARW_<H>/TESTING/SIM_<id>/predictions/<origin>/predictions.feather
                                                       sample_spaghetti.feather
                                                       spatial_scores.feather
<variant>/ARW_<H>/TESTING/SIM_<id>/window_times.feather
```

* `predictions.feather` — per (timestep, MSOA): median, 50%/95% intervals, ground
  truth, and `IS_95` / `CRPS` on raw and `log1p` scales.
* `spatial_scores.feather` — per (timestep, burden): energy and variogram (p=0.5)
  scores, raw and `log1p`.
* `sample_spaghetti.feather` — every raw sample; set `save_spaghetti: false` to skip
  (it dominates disk use).

Scores come from `proper_scoring_torch.py`, which is byte-identical to GENIE's;
`test_pipeline.py` cross-checks the outputs against GENIE's own functions.

**Deliberate deviation from GENIE**: near `t=0` GENIE zero-pads the context. Zero-padding
raw counts tells a foundation model "no cases", so ChronoGenie truncates instead and
skips origins with less than `testing.min_context` days of history. Those early windows
are absent rather than wrong; every other window aligns on `window_start`.

---

## 6. Running

```bash
# What will this cost? (loads data only)
python run_experiment.py --config configs/experiments/mode_neighbours.yaml --estimate

# Same, but times real forward passes on this GPU and projects wall-clock hours
python run_experiment.py --config configs/experiments/mode_neighbours.yaml --benchmark

# Fine-tune + evaluate baseline and fine-tuned
python run_experiment.py --config configs/experiments/mode_neighbours.yaml

# All modes in sequence
python run_experiment.py --config-dir configs/experiments

# Correctness checks (add --chronos to include a real CPU forward pass)
python test_pipeline.py
```

Runs resume: a window is redone only if its feather files are missing, and fine-tuning
is skipped when `finetune/finetuned-ckpt` already exists.

### Cost

Forward-pass rows scale as `sims x origins x horizon x num_samples x MSOAs x rows_per_anchor`.
For 10 sims, `origin_stride: 10` (17 origins), horizon 60 and 20 paths:

| mode | rows/anchor | total rows |
| --- | --- | --- |
| `self_only` | 4 | 69M |
| `mean_all_others` / `neighbour_mean` | 6 | 103M |
| `neighbour_and_nonneighbour_means` | 8 | 137M |
| `neighbours` | 34.9 | 598M |

`--benchmark` converts these into hours on your hardware. Shrink `origin_stride`,
`max_test_sims` or `num_samples` if needed — but keep all three **identical across
modes**, or the comparison is between evaluation grids rather than between models.

---

## 7. Comparing

```bash
python compare_runs.py \
    --run baseline=RESULTS/.../CHRONOS2_NEIGHBOURS_*/baseline/ARW_60 \
    --run finetuned=RESULTS/.../CHRONOS2_NEIGHBOURS_*/finetuned/ARW_60 \
    --run mean_all=RESULTS/.../CHRONOS2_MEAN_ALL_OTHERS_*/finetuned/ARW_60 \
    --run genie_mlp=../GENIE/RESULTS/RESULTS_MLP/<run> \
    --reference baseline \
    --output RESULTS/comparison
```

Produces `summary.md`, `scores_long.csv`, `summary_by_run.csv`, `summary_by_lead.csv`
and `skill_vs_<reference>.csv`. Comparisons are **paired**: only
(sim, window, lead, burden) cells that every run produced are used, so a run that
stopped early cannot look good by having been scored on an easier subset.

GENIE's own plotting scripts also read these directories unchanged:

```bash
python ../GENIE/Plotting/Plot_compare_models/compare_models_es.py \
    --dir1 <...>/baseline/ARW_60 --dir2 <...>/finetuned/ARW_60 \
    --name1 "Chronos-2 base" --name2 "Chronos-2 fine-tuned"
```

---

## 8. Files

| file | role |
| --- | --- |
| `spatial_context.py` | neighbourhoods, covariate modes, Chronos input packing |
| `chronos_adapter.py` | pipeline wrapper, checkpoint resolution, inverse-CDF sampling |
| `testing_sliding_window.py` | autoregressive rollout, scoring, feather output |
| `testing.py` | per-simulation orchestration and resume |
| `run_experiment.py` | config, fine-tuning, baseline/fine-tuned variants, logging |
| `compare_runs.py` | aggregation, paired skill scores, summary tables |
| `test_pipeline.py` | correctness checks (121, no GPU needed) |
| `proper_scoring_torch.py` | scoring rules, identical to GENIE's |
