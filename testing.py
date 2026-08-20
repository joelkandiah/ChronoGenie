#!/usr/bin/env python3
"""Test-time orchestration: mirrors GENIE's ``testing.test`` entry point.

Chooses which simulations still need work, lays out the output tree so the
comparison scripts in ``../GENIE/Plotting`` can read it unchanged, and hands the
rollout itself to :mod:`testing_sliding_window`.
"""

from __future__ import annotations

import logging
import os

import torch

from spatial_context import SpatialCovariateBuilder
from testing_sliding_window import (
    RolloutConfig,
    RolloutStats,
    run_sliding_window_forecasts,
    window_is_complete,
)

logger = logging.getLogger(__name__)


def select_pending_sims(
    output_folder: str,
    sim_ids,
    origins: list[int],
    expect_spaghetti: bool,
) -> list:
    """Simulations with at least one unfinished window, logged like GENIE's resume."""
    pending = []
    for sim_id in sim_ids:
        sim_folder = os.path.join(output_folder, "TESTING", f"SIM_{sim_id}")
        complete = sum(
            window_is_complete(os.path.join(sim_folder, "predictions", str(origin)), expect_spaghetti)
            for origin in origins
        )
        if complete >= len(origins):
            logger.info("[RESUME] SIM_%s -> COMPLETE (%d/%d windows)", sim_id, complete, len(origins))
        else:
            logger.info("[RESUME] SIM_%s -> PENDING (%d/%d windows)", sim_id, complete, len(origins))
            pending.append(sim_id)
    return pending


def test(
    temporal_model,
    dataset_directory,
    covariate_builder: SpatialCovariateBuilder,
    config: RolloutConfig,
    origins: list[int],
    output_folder: str,
    sim_ids=None,
    split_name: str = "TESTING",
) -> RolloutStats:
    """Evaluate ``temporal_model`` over the requested simulations and origins.

    Args:
        temporal_model: A :class:`chronos_adapter.ChronosTemporalAdapter`.
        dataset_directory: The loaded :class:`dataset.DatesetDirectory`.
        covariate_builder: Spatial covariate layout; must match the one used to fine-tune.
        config: Rollout settings.
        origins: Forecast origins (absolute day indices).
        output_folder: Variant root; ``TESTING/SIM_<id>/...`` is created beneath it.
        sim_ids: Simulations to evaluate, defaulting to the dataset's test split.
        split_name: Which split ``sim_ids`` came from, for logging only.

    Returns:
        Rollout counters for the run manifest.
    """
    temporal_model.eval()
    os.makedirs(output_folder, exist_ok=True)

    all_sims = list(dataset_directory.test_sims if sim_ids is None else sim_ids)
    pending = (
        select_pending_sims(output_folder, all_sims, origins, config.save_spaghetti)
        if config.resume
        else all_sims
    )

    logger.info(
        "[%s] sims total: %d | pending: %d | origins: %d (%s..%s) | horizon: %d",
        split_name,
        len(all_sims),
        len(pending),
        len(origins),
        origins[0] if origins else "-",
        origins[-1] if origins else "-",
        config.horizon,
    )

    if not pending:
        logger.info("[%s] nothing to do; every simulation is already complete.", split_name)
        return RolloutStats()

    with torch.no_grad():
        return run_sliding_window_forecasts(
            temporal_model=temporal_model,
            dataset_directory=dataset_directory,
            covariate_builder=covariate_builder,
            config=config,
            origins=origins,
            output_folder=output_folder,
            sim_ids=pending,
        )
