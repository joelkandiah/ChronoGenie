#!/usr/bin/env python3
"""Autoregressive sliding-window forecasting and scoring for Chronos-2.

Formulation
-----------
For a forecast origin ``t0`` the model produces a ``H``-day rolling forecast by
repeatedly predicting one day ahead and feeding the realisation back in. Each
input item is a single *anchor* MSOA:

* **targets** -- the anchor's own burden series (all burdens jointly, so the
  model can use cross-burden structure),
* **covariates** -- the spatial context selected by the covariate mode
  (neighbour series, means over neighbours / non-neighbours / all others),
* optionally a known-future weekday pair.

Only the anchor's target rows are read back; predicted covariate rows do not
exist, because covariates are never forecast by Chronos-2.

Sample paths
------------
Uncertainty is represented by ``S`` **independent trajectories**, exactly as in
the GENIE models. Path ``s`` keeps its own rolling context: at every step it
draws one inverse-CDF realisation and writes that value back into its own
history. Scores are then computed over the ``S`` paths.

This matters: drawing ``S`` marginal samples at each step around a *single*
history would collapse the forecast spread to one-step-ahead uncertainty and
make long leads look far more certain than they are, and would also destroy the
cross-MSOA dependence that the energy and variogram scores measure.

Outputs
-------
Feather files matching the GENIE layout byte-for-byte in schema, so the
comparison scripts in ``../GENIE/Plotting`` read both without modification::

    <output_folder>/TESTING/SIM_<id>/predictions/<origin>/predictions.feather
    <output_folder>/TESTING/SIM_<id>/predictions/<origin>/sample_spaghetti.feather
    <output_folder>/TESTING/SIM_<id>/predictions/<origin>/spatial_scores.feather
    <output_folder>/TESTING/SIM_<id>/window_times.feather
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field

import numpy as np
import pyarrow as pa
import torch
from pyarrow import feather

from chronos_adapter import sample_from_quantiles
from proper_scoring_torch import (
    crps_torch,
    energy_score_torch,
    interval_score_torch,
    variogram_torch,
)
from spatial_context import SpatialCovariateBuilder, day_of_week_features

logger = logging.getLogger(__name__)

REPORT_QUANTILES = (0.5, 0.025, 0.975, 0.25, 0.75)
FALLBACK_QUANTILE_LEVELS = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]


@dataclass
class RolloutConfig:
    """Everything that controls one autoregressive evaluation run."""

    context_size: int
    horizon: int
    num_samples: int
    window_batch_size: int = 4
    predict_row_batch_size: int = 1024
    ar_step_size: int = 1
    count_rounding: str = "floor"
    save_spaghetti: bool = True
    resume: bool = True
    seed: int = 26

    def __post_init__(self) -> None:
        if self.context_size < 1:
            raise ValueError("context_size must be >= 1")
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")
        if self.num_samples < 2:
            raise ValueError("num_samples must be >= 2 for the sample-based scores to be defined")
        if self.window_batch_size < 1:
            raise ValueError("window_batch_size must be >= 1")
        if self.ar_step_size < 1:
            raise ValueError("ar_step_size must be >= 1")
        if self.count_rounding not in ("floor", "round", "none"):
            raise ValueError(f"count_rounding must be 'floor', 'round' or 'none', got '{self.count_rounding}'")
        if self.ar_step_size > 1:
            logger.warning(
                "ar_step_size=%d generates %d days per model call, which is cheaper but draws each "
                "day in the block from its own marginal quantiles. Within-block temporal dependence "
                "is lost; use ar_step_size=1 for trajectories comparable with GENIE's.",
                self.ar_step_size,
                self.ar_step_size,
            )


@dataclass
class RolloutStats:
    """Counters used for progress logging and the run manifest."""

    forward_items: int = 0
    forward_rows: int = 0
    predict_seconds: float = 0.0
    build_seconds: float = 0.0
    score_seconds: float = 0.0
    windows_written: int = 0
    windows_skipped: int = 0
    per_sim_seconds: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "forward_items": self.forward_items,
            "forward_rows": self.forward_rows,
            "predict_seconds": round(self.predict_seconds, 2),
            "build_seconds": round(self.build_seconds, 2),
            "score_seconds": round(self.score_seconds, 2),
            "windows_written": self.windows_written,
            "windows_skipped": self.windows_skipped,
            "per_sim_seconds": {k: round(v, 2) for k, v in self.per_sim_seconds.items()},
        }


# ─── Helpers ─────────────────────────────────────────────────────────────────


def resolve_origins(
    num_time_steps: int,
    min_timestep: int,
    horizon: int,
    origin_stride: int = 1,
    max_origins: int | None = None,
    min_context: int = 1,
) -> list[int]:
    """Forecast origins to evaluate.

    An origin ``t0`` forecasts days ``t0 .. t0+H-1`` using days ``< t0`` as history,
    matching GENIE's sliding windows. Origins with fewer than ``min_context`` days of
    history are dropped: a foundation model needs at least some history, and unlike
    the GENIE models we do not fabricate one by zero-padding.
    """
    if min_context < 1:
        raise ValueError("min_context must be >= 1")

    # The grid is anchored at day 0 so that runs with different min_context still
    # share window-start ids, which is what the comparison scripts join on.
    grid = list(range(0, num_time_steps, max(1, origin_stride)))
    earliest = max(min_timestep, min_context)
    origins = [t for t in grid if t >= earliest]
    if max_origins is not None:
        origins = origins[:max_origins]

    dropped = len(grid) - len([t for t in grid if t >= earliest])
    if dropped:
        logger.info(
            "Skipping %d forecast origin(s) before day %d (min_timestep=%d, min_context=%d).",
            dropped,
            earliest,
            min_timestep,
            min_context,
        )
    truncated = sum(1 for t in origins if t + horizon > num_time_steps)
    if truncated:
        logger.info(
            "%d of %d origins run past the end of the series and are scored on a shortened horizon.",
            truncated,
            len(origins),
        )
    return origins


def window_is_complete(prediction_folder: str, expect_spaghetti: bool) -> bool:
    """A window counts as done only if every file it should have is on disk."""
    required = ["predictions.feather", "spatial_scores.feather"]
    if expect_spaghetti:
        required.append("sample_spaghetti.feather")
    return all(os.path.isfile(os.path.join(prediction_folder, name)) for name in required)


def _chunks(items: list, size: int) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _start_day_of_week(dataset_directory, sim_id) -> int:
    """Weekday index of day 0 for a simulation, or 0 when the metadata is missing."""
    metadata = getattr(dataset_directory, "df_start_time_metadata", None)
    if metadata is None or "dow" not in getattr(metadata, "columns", []):
        return 0
    rows = metadata[metadata["sim"] == sim_id]
    if len(rows) == 0:
        try:
            rows = metadata[metadata["sim"] == type(metadata["sim"].iloc[0])(sim_id)]
        except (ValueError, TypeError, IndexError):
            return 0
    return int(rows["dow"].iloc[0]) if len(rows) else 0


# ─── Rollout ─────────────────────────────────────────────────────────────────


def _predict_step_quantiles(
    temporal_model,
    inputs: list[dict],
    prediction_length: int,
    quantile_levels: list[float],
    row_batch_size: int,
    stats: RolloutStats,
) -> torch.Tensor:
    """Run one batched Chronos call. Returns ``[N_items, C, prediction_length, Q]``."""
    started = time.time()
    quantiles_list, _ = temporal_model.predict_quantiles(
        inputs=inputs,
        prediction_length=prediction_length,
        quantile_levels=quantile_levels,
        batch_size=row_batch_size,
    )
    stats.predict_seconds += time.time() - started
    stats.forward_items += len(inputs)
    stats.forward_rows += sum(int(item["context"].shape[0]) for item in inputs)

    stacked = torch.stack(quantiles_list, dim=0)
    if stacked.shape[2] != prediction_length or stacked.shape[3] != len(quantile_levels):
        raise RuntimeError(
            f"Unexpected Chronos output shape {tuple(stacked.shape)}; expected "
            f"[N, C, {prediction_length}, {len(quantile_levels)}]"
        )
    return stacked


def _rollout_origin_batch(
    temporal_model,
    covariate_builder: SpatialCovariateBuilder,
    history: torch.Tensor,
    origins: list[int],
    horizons: list[int],
    config: RolloutConfig,
    quantile_levels: list[float],
    count_channels: list[int],
    start_dow: int | None,
    generator: torch.Generator,
    stats: RolloutStats,
) -> torch.Tensor:
    """Roll every (origin, path) forward one step at a time.

    Args:
        history: ``[C, M, T]`` ground-truth series for one simulation, raw units.
        origins: Forecast origins in this batch.
        horizons: Number of days to forecast for each origin.

    Returns:
        ``[W, S, H_max, M, C]`` sampled trajectories. Windows whose horizon is shorter
        than the batch maximum are zero-padded at the end; the caller scores only
        ``[:horizons[w]]``, so the padding is never read.
    """
    num_channels, num_msoas, num_time_steps = history.shape
    num_windows = len(origins)
    num_paths = config.num_samples
    max_horizon = max(horizons)

    # Timeline per (origin, path): ground truth up to the origin, sampled after it.
    rollout = history.reshape(1, 1, num_channels, num_msoas, num_time_steps).repeat(
        num_windows, num_paths, 1, 1, 1
    )

    for step in range(0, max_horizon, config.ar_step_size):
        active = [w for w in range(num_windows) if step < horizons[w]]
        if not active:
            break

        block = min(config.ar_step_size, max_horizon - step)
        build_started = time.time()

        inputs: list[dict] = []
        for w in active:
            pred_time = origins[w] + step
            context_start = max(0, pred_time - config.context_size)

            past_dow = future_dow = None
            if covariate_builder.include_day_of_week:
                past_dow = day_of_week_features(start_dow or 0, np.arange(context_start, pred_time))
                future_dow = day_of_week_features(start_dow or 0, np.arange(pred_time, pred_time + block))

            for path in range(num_paths):
                inputs.extend(
                    covariate_builder.build_inputs(
                        rollout[w, path, :, :, context_start:pred_time],
                        prediction_length=block,
                        past_dow=past_dow,
                        future_dow=future_dow,
                    )
                )
        stats.build_seconds += time.time() - build_started

        quantiles = _predict_step_quantiles(
            temporal_model,
            inputs,
            prediction_length=block,
            quantile_levels=quantile_levels,
            row_batch_size=config.predict_row_batch_size,
            stats=stats,
        )

        # One inverse-CDF draw per (window, path, msoa, channel, lead).
        levels = torch.tensor(quantile_levels, dtype=quantiles.dtype)
        drawn = sample_from_quantiles(quantiles, levels, num_samples=1, generator=generator)[0]
        drawn = drawn.reshape(len(active), num_paths, num_msoas, num_channels, block)
        drawn = drawn.permute(0, 1, 3, 2, 4).contiguous()  # [W_active, S, C, M, block]

        if count_channels and config.count_rounding != "none":
            index = torch.tensor(count_channels, dtype=torch.long)
            rounder = torch.floor if config.count_rounding == "floor" else torch.round
            drawn[:, :, index] = rounder(drawn[:, :, index].clamp_min(0.0))
        drawn = drawn.clamp_min(0.0)

        for position, w in enumerate(active):
            pred_time = origins[w] + step
            usable = min(block, horizons[w] - step, num_time_steps - pred_time)
            if usable <= 0:
                continue
            rollout[w, :, :, :, pred_time : pred_time + usable] = drawn[position, :, :, :, :usable]

    # Origins near the end of the series have a shorter horizon than the batch maximum,
    # so pad to a common length; the padding is never scored.
    trajectories = torch.zeros(
        (num_windows, num_paths, max_horizon, num_msoas, num_channels), dtype=rollout.dtype
    )
    for w, origin in enumerate(origins):
        available = min(max_horizon, num_time_steps - origin)
        window = rollout[w, :, :, :, origin : origin + available]  # [S, C, M, h]
        trajectories[w, :, :available] = window.permute(0, 3, 2, 1)
    return trajectories


# ─── Scoring & serialisation ─────────────────────────────────────────────────


def score_window(samples_win: torch.Tensor, truth_win: torch.Tensor) -> dict[str, torch.Tensor]:
    """Compute the GENIE scoring-rule set for one forecast window.

    Args:
        samples_win: ``[S, H, M, V]`` predictive sample trajectories.
        truth_win: ``[H, M, V]`` observed values.

    Returns:
        Dict of score tensors: per-node scores are ``[H, M, V]``, multivariate
        (spatial) scores are ``[H, V]``. Log-scale variants use ``log1p``.
    """
    if samples_win.ndim != 4 or truth_win.ndim != 3:
        raise ValueError("Expected samples [S, H, M, V] and truth [H, M, V]")
    if samples_win.shape[1:] != truth_win.shape:
        raise ValueError(
            f"Sample/truth shape mismatch: {tuple(samples_win.shape)} vs {tuple(truth_win.shape)}"
        )

    device = samples_win.device
    num_prediction = truth_win.shape[-1]
    quantile_probs = torch.tensor(REPORT_QUANTILES, device=device, dtype=samples_win.dtype)

    quantiles = torch.quantile(samples_win, quantile_probs, dim=0)  # [5, H, M, V]
    median, lower_95, upper_95, lower_50, upper_50 = quantiles

    interval_95 = interval_score_torch(truth_win, lower_95, upper_95)
    crps = crps_torch(truth_win, samples_win)

    horizon = truth_win.shape[0]
    energy = torch.zeros(horizon, num_prediction, device=device)
    variogram = torch.zeros(horizon, num_prediction, device=device)
    for v in range(num_prediction):
        energy[:, v] = energy_score_torch(truth_win[:, :, v], samples_win[:, :, :, v])
        variogram[:, v] = variogram_torch(truth_win[:, :, v], samples_win[:, :, :, v], p=0.5)

    truth_log = torch.log1p(truth_win)
    samples_log = torch.log1p(samples_win)
    interval_95_log = interval_score_torch(truth_log, torch.log1p(lower_95), torch.log1p(upper_95))
    crps_log = crps_torch(truth_log, samples_log)

    energy_log = torch.zeros(horizon, num_prediction, device=device)
    variogram_log = torch.zeros(horizon, num_prediction, device=device)
    for v in range(num_prediction):
        energy_log[:, v] = energy_score_torch(truth_log[:, :, v], samples_log[:, :, :, v])
        variogram_log[:, v] = variogram_torch(truth_log[:, :, v], samples_log[:, :, :, v], p=0.5)

    return {
        "median": median,
        "lower_95": lower_95,
        "upper_95": upper_95,
        "lower_50": lower_50,
        "upper_50": upper_50,
        "truth": truth_win,
        "interval_95": interval_95,
        "crps": crps,
        "interval_95_log": interval_95_log,
        "crps_log": crps_log,
        "energy": energy,
        "variogram": variogram,
        "energy_log": energy_log,
        "variogram_log": variogram_log,
    }


def write_window_outputs(
    prediction_folder: str,
    origin: int,
    scores: dict[str, torch.Tensor],
    samples_win: torch.Tensor,
    prediction_names: list[str],
    msoa_name_map: np.ndarray,
    save_spaghetti: bool,
) -> None:
    """Serialise one window to the three GENIE-compatible feather tables."""
    os.makedirs(prediction_folder, exist_ok=True)

    per_node = torch.stack(
        [
            scores["median"], scores["lower_95"], scores["upper_95"],
            scores["lower_50"], scores["upper_50"], scores["truth"],
            scores["interval_95"], scores["crps"],
            scores["interval_95_log"], scores["crps_log"],
        ],
        dim=0,
    ).cpu().numpy()  # [10, H, M, V]
    spatial = torch.stack(
        [scores["energy"], scores["variogram"], scores["energy_log"], scores["variogram_log"]], dim=0
    ).cpu().numpy()  # [4, H, V]

    horizon, num_msoas, num_prediction = per_node.shape[1:]
    num_samples = samples_win.shape[0]

    timestep_grid = np.repeat(np.arange(origin, origin + horizon), num_msoas).astype(np.int32)
    msoa_grid = np.tile(np.arange(num_msoas), horizon).astype(np.int32)

    dynamic_columns = {}
    for j, name in enumerate(prediction_names):
        dynamic_columns[f"{name}_unscaled"] = per_node[0, :, :, j].reshape(-1)
        dynamic_columns[f"{name}_unscaled_gt"] = per_node[5, :, :, j].reshape(-1)
        dynamic_columns[f"{name}_lower_95"] = per_node[1, :, :, j].reshape(-1)
        dynamic_columns[f"{name}_upper_95"] = per_node[2, :, :, j].reshape(-1)
        dynamic_columns[f"{name}_lower_50"] = per_node[3, :, :, j].reshape(-1)
        dynamic_columns[f"{name}_upper_50"] = per_node[4, :, :, j].reshape(-1)
        dynamic_columns[f"{name}_IS_95"] = per_node[6, :, :, j].reshape(-1)
        dynamic_columns[f"{name}_CRPS"] = per_node[7, :, :, j].reshape(-1)
        dynamic_columns[f"{name}_IS_95_log"] = per_node[8, :, :, j].reshape(-1)
        dynamic_columns[f"{name}_CRPS_log"] = per_node[9, :, :, j].reshape(-1)

    feather.write_feather(
        pa.table(
            {
                "timestep": timestep_grid,
                "msoa": msoa_grid,
                "is_window_start": (timestep_grid == origin).astype(np.int8),
                **dynamic_columns,
            }
        ),
        os.path.join(prediction_folder, "predictions.feather"),
        compression="lz4",
    )

    if save_spaghetti:
        values = samples_win.cpu().numpy().reshape(-1)
        total = num_samples * horizon * num_msoas * num_prediction
        sample_idx = np.repeat(np.arange(num_samples), horizon * num_msoas * num_prediction)
        step_idx = np.tile(np.repeat(np.arange(horizon), num_msoas * num_prediction), num_samples)
        msoa_idx = np.tile(np.repeat(np.arange(num_msoas), num_prediction), num_samples * horizon)
        burden_idx = np.tile(np.arange(num_prediction), num_samples * horizon * num_msoas)

        feather.write_feather(
            pa.table(
                {
                    "window_start": np.full(total, origin, dtype=np.int32),
                    "timestep": (origin + step_idx).astype(np.int32),
                    "window_step": step_idx.astype(np.int32),
                    "msoa": msoa_idx.astype(np.int32),
                    "msoa_name": pa.array(msoa_name_map[msoa_idx]),
                    "sample_idx": sample_idx.astype(np.int32),
                    "burden_type": pa.array(np.array(prediction_names)[burden_idx]),
                    "value": values,
                    "mu_pred": values,
                    "var_pred": np.zeros(total, dtype=np.float32),
                }
            ),
            os.path.join(prediction_folder, "sample_spaghetti.feather"),
            compression="lz4",
        )

    timesteps = np.arange(origin, origin + horizon)
    spatial_tables = [
        pa.table(
            {
                "timestep": timesteps,
                "burden": pa.array([name] * horizon),
                "energy_score": spatial[0, :, v],
                "variogram_score_p05": spatial[1, :, v],
                "energy_score_log": spatial[2, :, v],
                "variogram_score_p05_log": spatial[3, :, v],
            }
        )
        for v, name in enumerate(prediction_names)
    ]
    feather.write_feather(
        pa.concat_tables(spatial_tables),
        os.path.join(prediction_folder, "spatial_scores.feather"),
        compression="lz4",
    )


# ─── Entry point ─────────────────────────────────────────────────────────────


def run_sliding_window_forecasts(
    temporal_model,
    dataset_directory,
    covariate_builder: SpatialCovariateBuilder,
    config: RolloutConfig,
    origins: list[int],
    output_folder: str,
    sim_ids,
    quantile_levels: list[float] | None = None,
) -> RolloutStats:
    """Run autoregressive forecasts for every requested simulation and origin.

    Args:
        temporal_model: A :class:`chronos_adapter.ChronosTemporalAdapter`.
        dataset_directory: The loaded :class:`dataset.DatesetDirectory`.
        covariate_builder: Defines the spatial covariate layout (must match training).
        config: Rollout settings.
        origins: Forecast origins (absolute day indices).
        output_folder: Run root; ``TESTING/SIM_<id>/...`` is created beneath it.
        sim_ids: Simulations to evaluate.
        quantile_levels: Levels requested from Chronos. Defaults to the model's own
            training grid, which avoids any quantile interpolation before sampling.

    Returns:
        Counters describing the work done, for the run manifest.
    """
    prediction_names = list(dataset_directory.columns_for_prediction)
    context_names = list(dataset_directory.columns_for_context)
    if set(context_names) != set(prediction_names):
        raise ValueError(
            "Autoregressive rollout requires columns_for_context to match columns_for_prediction "
            f"(context={context_names}, prediction={prediction_names}); a context channel that is "
            "not forecast cannot be rolled forward without leaking future ground truth."
        )

    if quantile_levels is None:
        quantile_levels = getattr(temporal_model, "native_quantile_levels", None) or FALLBACK_QUANTILE_LEVELS
    quantile_levels = sorted(float(q) for q in quantile_levels)

    prediction_indices = [dataset_directory.columns_with_data.index(name) for name in prediction_names]
    distribution_types = dict(zip(prediction_names, dataset_directory.prediction_distribution_types))
    count_channels = [
        i for i, name in enumerate(prediction_names)
        if distribution_types.get(name, "nb").lower() != "lognormal"
    ]

    num_msoas = dataset_directory.num_geographies
    num_time_steps = dataset_directory.num_time_steps
    msoa_name_map = dataset_directory.df_geo_metadata["MSOA"].to_numpy()

    generator = torch.Generator().manual_seed(config.seed)
    stats = RolloutStats()

    logger.info(
        "Rollout | %d sims x %d origins | horizon=%d | paths=%d | context=%d | mode=%s | "
        "rows/anchor=%.1f | quantiles=%d",
        len(list(sim_ids)),
        len(origins),
        config.horizon,
        config.num_samples,
        config.context_size,
        covariate_builder.mode,
        covariate_builder.total_rows() / max(1, num_msoas),
        len(quantile_levels),
    )

    for sim_id in sim_ids:
        sim_started = time.time()
        sim_idx = dataset_directory.resolve_sim_idx(sim_id)
        sim_folder = os.path.join(output_folder, "TESTING", f"SIM_{sim_id}")
        os.makedirs(sim_folder, exist_ok=True)

        pending = []
        for origin in origins:
            folder = os.path.join(sim_folder, "predictions", str(origin))
            if config.resume and window_is_complete(folder, config.save_spaghetti):
                stats.windows_skipped += 1
            else:
                pending.append(origin)

        if not pending:
            logger.info("SIM_%s | all %d windows already complete, skipping", sim_id, len(origins))
            continue

        history = dataset_directory.raw_data_tensor[prediction_indices, sim_idx].to(torch.float32)  # [C, M, T]
        start_dow = _start_day_of_week(dataset_directory, sim_id) if covariate_builder.include_day_of_week else None
        window_times: dict[int, float] = {}

        for batch_number, origin_batch in enumerate(_chunks(pending, config.window_batch_size), start=1):
            horizons = [min(config.horizon, num_time_steps - origin) for origin in origin_batch]
            batch_started = time.time()

            trajectories = _rollout_origin_batch(
                temporal_model=temporal_model,
                covariate_builder=covariate_builder,
                history=history,
                origins=origin_batch,
                horizons=horizons,
                config=config,
                quantile_levels=quantile_levels,
                count_channels=count_channels,
                start_dow=start_dow,
                generator=generator,
                stats=stats,
            )
            rollout_seconds = time.time() - batch_started

            score_started = time.time()
            for w, origin in enumerate(origin_batch):
                horizon = horizons[w]
                samples_win = trajectories[w, :, :horizon]  # [S, H, M, V]
                truth_win = history[:, :, origin : origin + horizon].permute(2, 1, 0).contiguous()

                scores = score_window(samples_win, truth_win)
                write_window_outputs(
                    prediction_folder=os.path.join(sim_folder, "predictions", str(origin)),
                    origin=origin,
                    scores=scores,
                    samples_win=samples_win,
                    prediction_names=prediction_names,
                    msoa_name_map=msoa_name_map,
                    save_spaghetti=config.save_spaghetti,
                )
                window_times[origin] = rollout_seconds / len(origin_batch)
                stats.windows_written += 1
            stats.score_seconds += time.time() - score_started

            elapsed = time.time() - sim_started
            done = batch_number * config.window_batch_size
            remaining = max(0, len(pending) - done)
            logger.info(
                "SIM_%s | windows %d/%d | %.1fs/window | predict %.1fs build %.1fs | ETA %.1f min",
                sim_id,
                min(done, len(pending)),
                len(pending),
                rollout_seconds / len(origin_batch),
                stats.predict_seconds,
                stats.build_seconds,
                (elapsed / max(1, min(done, len(pending)))) * remaining / 60.0,
            )

        existing = {}
        timing_path = os.path.join(sim_folder, "window_times.feather")
        if os.path.isfile(timing_path):
            table = feather.read_table(timing_path).to_pydict()
            existing = dict(zip(table.get("window_start", []), table.get("window_times_seconds", [])))
        existing.update(window_times)
        ordered = sorted(existing.items())
        feather.write_feather(
            pa.table(
                {
                    "window_start": pa.array([int(k) for k, _ in ordered], type=pa.int32()),
                    "window_times_seconds": pa.array([float(v) for _, v in ordered]),
                }
            ),
            timing_path,
            compression="lz4",
        )

        stats.per_sim_seconds[str(sim_id)] = time.time() - sim_started
        logger.info("SIM_%s | finished in %.1f min", sim_id, stats.per_sim_seconds[str(sim_id)] / 60.0)

    logger.info("Rollout complete: %s", json.dumps(stats.as_dict())[:400])
    return stats
