# GNN-AR Restructuring Task

## Done

- `[x]` **Neighbourhoods** — `spatial_context.build_neighbour_map` runs GENIE's two
  k-NN searches (9 in scaled static-feature space, 9 in haversine space) and takes the
  **per-node union**: degree 11-18, mean 15.4 on the 84-MSOA data. Reciprocal edges are
  *not* added (`symmetrise: false`); `symmetrise: true` reproduces GENIE's undirected
  GNN graph if ever needed. Ordered nearest-first, exported to `neighbour_map.json`.
- `[x]` **Autoregressive rollout** — `testing_sliding_window.py` rewritten: per-anchor
  1-step rollout with `prediction_length=1`, `S` independent sample trajectories each
  feeding its own inverse-CDF draw back into its own context.
- `[x]` **Anchor input layout** — anchor's burdens are the Chronos *targets*; spatial
  context enters as *past-only covariates*; weekday sin/cos as a *known-future*
  covariate. One `SpatialCovariateBuilder` serves both fine-tuning and testing, so the
  two layouts cannot drift apart.
- `[x]` **Covariate modes** — `self_only`, `neighbours`, `mean_all_others`,
  `neighbour_mean`, `neighbour_and_nonneighbour_means`; one config each.
- `[x]` **Baseline vs fine-tuned** — both evaluated per run on the same grid, into
  `baseline/ARW_<H>/` and `finetuned/ARW_<H>/`.
- `[x]` **Fine-tuning inputs** rebuilt in the neighbour-stacked layout, with subsampling
  controls and truncated validation series.
- `[x]` **Output layout** — one folder per forecast origin, GENIE-identical feather
  schemas, per-window resume.
- `[x]` **Scoring** — `proper_scoring_torch.py` unchanged and cross-checked against
  GENIE's implementations in `test_pipeline.py`.
- `[x]` **Logging** — `run.log`, `run_manifest.json`, `training_log.jsonl`,
  `training_curves.png`, per-window timings, `--estimate` / `--benchmark`.
- `[x]` **Comparison** — `compare_runs.py` (paired skill scores); GENIE's own plotting
  scripts also read the output directories unchanged.
- `[x]` **Verification** — `test_pipeline.py`, 121 checks, plus a real Chronos-2 CPU
  forward pass under `--chronos`.

## Bugs fixed along the way

- Fine-tuned checkpoints were never reloaded: the code looked for
  `adapter_config.json` in `output_dir`, but `Chronos2Pipeline.fit` writes it to
  `output_dir/finetuned-ckpt/`. Every "resumed" test run silently used the base model.
- Only sample path 0 was fed back, so at lead `k` the reported spread was one-step-ahead
  uncertainty around a single trajectory rather than a `k`-step forecast distribution.
- Forecast origins near the end of the series produced short slices that crashed the
  batch `stack`.
- Quantiles were requested at `[0.025, 0.25, 0.5, 0.75, 0.975]` and then sampled from;
  the rollout now samples from the model's native 21-level grid (no interpolation) and
  derives the reported intervals from the samples, as GENIE does.
- Contexts before day 0 were zero-padded, which tells a foundation model "no cases";
  they are now truncated, and origins with less than `min_context` days are skipped.
- Validation inputs enumerated every (day, anchor) for all 150 validation sims
  (~2.3M windows); they are now subsampled truncations.

## Open decisions for you

- **Evaluation grid.** Cost is `sims x origins x horizon x paths x MSOAs x rows/anchor`.
  Defaults are 10 sims, `origin_stride: 10`, 20 paths — run `--benchmark` on the HPC GPU
  and scale from there. Keep the grid identical across covariate modes.
- **`neighbours` mode cost.** ~35 rows/anchor vs 6-8 for the mean modes, so ~5x the
  compute. `spatial.max_neighbours: 9` halves it if needed.

See `PIPELINE.md` for the full description.
