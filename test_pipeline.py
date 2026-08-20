#!/usr/bin/env python3
"""Correctness checks for the ChronoGenie spatial/autoregressive pipeline.

Run with::

    python test_pipeline.py            # fast: stub model, no downloads
    python test_pipeline.py --chronos  # additionally runs the real Chronos-2 on CPU

The stub model returns quantiles that are a deterministic function of the last
context value, which makes the autoregressive feedback, the sample-path
bookkeeping and the output schema verifiable without a GPU or a model download.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile

import numpy as np
import pyarrow.feather as feather
import torch

from chronos_adapter import dequantise_counts, resolve_model_source, sample_from_quantiles
from spatial_context import (
    COVARIATE_MODES,
    SpatialCovariateBuilder,
    build_neighbour_map,
    day_of_week_features,
    haversine_matrix,
    neighbour_map_summary,
)
from testing_sliding_window import (
    RolloutConfig,
    resolve_origins,
    run_sliding_window_forecasts,
    score_window,
    window_is_complete,
)

PASSED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise AssertionError(f"{name} FAILED {detail}")
    PASSED.append(name)
    print(f"  ok  {name}")


# ─── Fixtures ────────────────────────────────────────────────────────────────


class StubDatasetDirectory:
    """Minimal stand-in for ``dataset.DatesetDirectory`` with reproducible data."""

    def __init__(self, num_msoas=6, num_sims=2, num_time_steps=40, seed=0):
        import pandas as pd

        rng = np.random.default_rng(seed)
        self.columns_for_prediction = ["daily_hospitalised", "deaths"]
        self.columns_for_context = ["daily_hospitalised", "deaths"]
        self.columns_with_data = list(self.columns_for_prediction)
        self.prediction_distribution_types = ["nb", "nb"]
        self.num_geographies = num_msoas
        self.num_time_steps = num_time_steps
        self.min_timestep = 0

        wave = np.sin(np.linspace(0, 3 * np.pi, num_time_steps)) + 1.2
        base = rng.gamma(2.0, 3.0, size=(2, num_sims, num_msoas, 1))
        self.raw_data_tensor = torch.tensor(
            np.round(base * wave[None, None, None, :]).clip(0), dtype=torch.float32
        )

        self.df_geo_metadata = pd.DataFrame({"MSOA": [f"E{i:08d}" for i in range(num_msoas)]})
        self.df_start_time_metadata = pd.DataFrame({"sim": list(range(num_sims)), "dow": [2] * num_sims})
        self.test_sims = list(range(num_sims))
        self.train_sims = list(range(num_sims))
        self.val_sims = list(range(num_sims))
        self.sim_id_to_idx = {i: i for i in range(num_sims)}

        latitudes = 54.9 + rng.random(num_msoas) * 0.3
        longitudes = -1.6 + rng.random(num_msoas) * 0.3
        features = rng.random((num_msoas, 5))
        features[:, 0] = latitudes
        features[:, 1] = longitudes
        self.raw_static_features_tensor = torch.tensor(features, dtype=torch.float32)
        self.static_features_tensor = torch.tensor(
            (features - features.min(0)) / np.ptp(features, axis=0), dtype=torch.float32
        )

    def resolve_sim_idx(self, sim_id):
        return self.sim_id_to_idx[int(sim_id)]


class StubChronosModel:
    """Returns quantiles centred on ``last_context_value + offset`` per target row.

    Deterministic in the context, so a rollout's feedback path can be checked exactly.
    """

    def __init__(self, spread=0.0, offset=1.0):
        self.spread = spread
        self.offset = offset
        self.calls = 0
        self.seen_row_counts: list[int] = []
        self.native_quantile_levels = [0.1, 0.5, 0.9]

    def eval(self):
        return self

    def predict_quantiles(self, inputs, prediction_length, quantile_levels, **_):
        self.calls += 1
        quantiles = []
        means = []
        for item in inputs:
            self.seen_row_counts.append(int(item["context"].shape[0]))
            n_targets = int(item["n_targets"])
            last = item["context"][:n_targets, -1]  # [C]
            centre = (last + self.offset).reshape(n_targets, 1, 1)
            spread = torch.tensor(
                [(q - 0.5) * 2.0 * self.spread for q in quantile_levels], dtype=torch.float32
            ).reshape(1, 1, -1)
            quantiles.append((centre + spread).expand(n_targets, prediction_length, len(quantile_levels)).clone())
            means.append(centre.squeeze(-1).expand(n_targets, prediction_length).clone())
        return quantiles, means


# ─── Tests ───────────────────────────────────────────────────────────────────


def test_neighbour_map():
    print("\n[neighbour map]")
    data = StubDatasetDirectory()
    neighbour_map, distances = build_neighbour_map(
        data.static_features_tensor, data.raw_static_features_tensor, ["latitude", "longitude", "a", "b", "c"], knn_k=2
    )
    check("no self loops", all(i not in js for i, js in neighbour_map.items()))
    check(
        "degree is between k and 2k (per-node union of the two k-NN sets)",
        all(2 <= len(js) <= 4 for js in neighbour_map.values()),
        str({i: len(js) for i, js in neighbour_map.items()}),
    )
    check(
        "neighbours are ordered nearest-first",
        all(
            all(distances[i, js[k]] <= distances[i, js[k + 1]] + 1e-9 for k in range(len(js) - 1))
            for i, js in neighbour_map.items()
        ),
    )
    symmetric_map, _ = build_neighbour_map(
        data.static_features_tensor, data.raw_static_features_tensor, None, knn_k=2, symmetrise=True
    )
    check(
        "symmetrise=True closes the relation both ways",
        all(i in symmetric_map[j] for i, js in symmetric_map.items() for j in js),
    )
    check(
        "symmetrise=True only ever adds neighbours",
        all(set(neighbour_map[i]) <= set(symmetric_map[i]) for i in neighbour_map),
    )

    capped, _ = build_neighbour_map(
        data.static_features_tensor, data.raw_static_features_tensor, None, knn_k=2, max_neighbours=2
    )
    check("max_neighbours caps the degree", all(len(js) <= 2 for js in capped.values()))
    check(
        "max_neighbours keeps the nearest",
        all(js == neighbour_map[i][:2] for i, js in capped.items()),
    )

    features_only, _ = build_neighbour_map(
        data.static_features_tensor, data.raw_static_features_tensor, None, knn_k=2, use_k_nearest_distance=False
    )
    geo_only, _ = build_neighbour_map(
        data.static_features_tensor, data.raw_static_features_tensor, None, knn_k=2, use_k_nearest_features=False
    )
    check("feature-only k-NN gives exactly k neighbours", all(len(js) == 2 for js in features_only.values()))
    check("geo-only k-NN gives exactly k neighbours", all(len(js) == 2 for js in geo_only.values()))
    check(
        "the union is exactly the two k-NN sets combined",
        all(set(neighbour_map[i]) == set(features_only[i]) | set(geo_only[i]) for i in neighbour_map),
    )

    matrix = haversine_matrix(np.array([54.0, 55.0]), np.array([-1.0, -1.0]))
    check("haversine ~111 km per degree of latitude", abs(matrix[0, 1] - 111.19) < 0.5, f"got {matrix[0, 1]:.2f}")

    summary = neighbour_map_summary(neighbour_map)
    check(
        "summary counts directed edges",
        summary["num_directed_edges"] == sum(len(v) for v in neighbour_map.values()),
    )
    check("summary reports reciprocity", 0.0 <= summary["fraction_reciprocal"] <= 1.0)


def test_covariate_layouts():
    print("\n[covariate layouts]")
    data = StubDatasetDirectory()
    neighbour_map, _ = build_neighbour_map(
        data.static_features_tensor, data.raw_static_features_tensor, None, knn_k=2
    )
    num_msoas, num_channels, length = data.num_geographies, 2, 5
    context = torch.arange(num_channels * num_msoas * length, dtype=torch.float32).reshape(
        num_channels, num_msoas, length
    )

    for mode in COVARIATE_MODES:
        builder = SpatialCovariateBuilder(
            mode, neighbour_map, num_msoas, num_channels, ["h", "d"], include_day_of_week=True
        )
        inputs = builder.build_inputs(
            context,
            prediction_length=1,
            past_dow=day_of_week_features(0, np.arange(length)),
            future_dow=day_of_week_features(0, np.arange(length, length + 1)),
        )
        check(f"{mode}: one item per anchor", len(inputs) == num_msoas)
        check(
            f"{mode}: context and future rows agree",
            all(i["context"].shape[0] == i["future_covariates"].shape[0] for i in inputs),
        )
        check(
            f"{mode}: declared row counts are right",
            all(
                i["context"].shape[0] == i["n_targets"] + i["n_covariates"] == builder.rows_per_anchor(m)
                for m, i in enumerate(inputs)
            ),
        )
        check(
            f"{mode}: targets are the anchor's own series",
            all(torch.equal(i["context"][:num_channels], context[:, m, :]) for m, i in enumerate(inputs)),
        )
        check(
            f"{mode}: weekday rows are known into the future",
            all(
                i["n_future_covariates"] == 2 and not torch.isnan(i["future_covariates"][-2:]).any()
                for i in inputs
            ),
        )
        check(
            f"{mode}: non-weekday future rows are NaN",
            all(torch.isnan(i["future_covariates"][:-2]).all() for i in inputs),
        )
        check(f"{mode}: covariate names match covariate rows",
              all(len(builder.covariate_names(m)) == builder.num_covariates(m) for m in range(num_msoas)))

    anchor = 3
    others = [j for j in range(num_msoas) if j != anchor]
    neighbours = neighbour_map[anchor]
    non_neighbours = [j for j in others if j not in neighbours]

    builder = SpatialCovariateBuilder("mean_all_others", neighbour_map, num_msoas, num_channels, ["h", "d"])
    rows = builder.build_inputs(context, 1)[anchor]["context"]
    check(
        "mean_all_others averages every other MSOA",
        torch.allclose(rows[num_channels:], context[:, others, :].mean(1), atol=1e-5),
    )

    builder = SpatialCovariateBuilder("neighbour_mean", neighbour_map, num_msoas, num_channels, ["h", "d"])
    rows = builder.build_inputs(context, 1)[anchor]["context"]
    check(
        "neighbour_mean averages the neighbours",
        torch.allclose(rows[num_channels:], context[:, neighbours, :].mean(1), atol=1e-5),
    )

    builder = SpatialCovariateBuilder(
        "neighbour_and_nonneighbour_means", neighbour_map, num_msoas, num_channels, ["h", "d"]
    )
    rows = builder.build_inputs(context, 1)[anchor]["context"]
    check(
        "neighbour/non-neighbour means are the right two blocks",
        torch.allclose(rows[num_channels : 2 * num_channels], context[:, neighbours, :].mean(1), atol=1e-5)
        and torch.allclose(rows[2 * num_channels :], context[:, non_neighbours, :].mean(1), atol=1e-5),
    )

    builder = SpatialCovariateBuilder("neighbours", neighbour_map, num_msoas, num_channels, ["h", "d"])
    rows = builder.build_inputs(context, 1)[anchor]["context"]
    check(
        "neighbours block order follows the neighbour map",
        all(
            torch.equal(
                rows[num_channels + rank * num_channels : num_channels + (rank + 1) * num_channels],
                context[:, node, :],
            )
            for rank, node in enumerate(neighbours)
        ),
    )


def test_sampling():
    print("\n[inverse-CDF sampling]")
    levels = torch.tensor([0.1, 0.5, 0.9])
    values = torch.tensor([[0.0, 10.0, 20.0]])
    samples = sample_from_quantiles(values, levels, num_samples=40000, generator=torch.Generator().manual_seed(0))
    check("sample mean matches the quantile centre", abs(samples.mean().item() - 10.0) < 0.5)
    check("median splits the samples", abs((samples < 10.0).float().mean().item() - 0.5) < 0.02)
    check("shape is [S, ...]", tuple(samples.shape) == (40000, 1))

    monotone = sample_from_quantiles(
        torch.tensor([[1.0, 2.0, 3.0]]), levels, num_samples=1000, generator=torch.Generator().manual_seed(1)
    )
    check("samples stay within the extrapolated support", monotone.min() > 0.0 and monotone.max() < 4.0)

    jittered = dequantise_counts(
        torch.zeros(2, 4), torch.tensor([True, False]), generator=torch.Generator().manual_seed(0)
    )
    check("dequantisation keeps counts non-negative integers", bool((jittered >= 0).all() and (jittered % 1 == 0).all()))
    check("dequantisation leaves unmasked rows alone", bool((jittered[1] == 0).all()))


def test_scoring_matches_genie():
    print("\n[scoring]")
    torch.manual_seed(0)
    samples = torch.rand(8, 3, 4, 2) * 10
    truth = torch.rand(3, 4, 2) * 10
    scores = score_window(samples, truth)

    check("per-node scores are [H, M, V]", scores["crps"].shape == truth.shape)
    check("spatial scores are [H, V]", scores["energy"].shape == (3, 2))
    check("median lies inside the 95% interval", bool((scores["lower_95"] <= scores["upper_95"]).all()))
    check("50% interval is inside the 95% interval",
          bool((scores["lower_50"] >= scores["lower_95"]).all() and (scores["upper_50"] <= scores["upper_95"]).all()))

    # A perfect deterministic forecast scores zero on every rule.
    perfect = truth.unsqueeze(0).repeat(8, 1, 1, 1)
    perfect_scores = score_window(perfect, truth)
    check("CRPS of a perfect forecast is 0", bool(perfect_scores["crps"].abs().max() < 1e-5))
    check("energy score of a perfect forecast is 0", bool(perfect_scores["energy"].abs().max() < 1e-4))
    check("variogram of a perfect forecast is 0", bool(perfect_scores["variogram"].abs().max() < 1e-4))
    check("interval score of a perfect forecast is 0", bool(perfect_scores["interval_95"].abs().max() < 1e-5))

    # Compare against GENIE's own implementations on the same inputs.
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "GENIE"))
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "genie_scoring",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "GENIE", "proper_scoring_torch.py"),
        )
        genie = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(genie)
        check(
            "CRPS matches GENIE's implementation",
            torch.allclose(scores["crps"], genie.crps_torch(truth, samples), atol=1e-5),
        )
        check(
            "energy score matches GENIE's implementation",
            torch.allclose(
                scores["energy"][:, 0], genie.energy_score_torch(truth[:, :, 0], samples[:, :, :, 0]), atol=1e-5
            ),
        )
        check(
            "variogram matches GENIE's implementation",
            torch.allclose(
                scores["variogram"][:, 0],
                genie.variogram_torch(truth[:, :, 0], samples[:, :, :, 0], p=0.5),
                atol=1e-4,
            ),
        )
    except (FileNotFoundError, ImportError) as error:
        print(f"  skip  GENIE cross-check ({error})")


def test_origins():
    print("\n[forecast origins]")
    check("stride is respected", resolve_origins(40, 0, 10, origin_stride=10, min_context=7) == [10, 20, 30])
    check("min_context drops early origins", resolve_origins(12, 0, 5, origin_stride=1, min_context=7)[0] == 7)
    check("max_origins truncates", len(resolve_origins(100, 0, 5, origin_stride=1, max_origins=3)) == 3)
    check("min_timestep is respected", resolve_origins(40, 20, 5, origin_stride=1, min_context=7)[0] == 20)


def test_rollout_feedback():
    print("\n[autoregressive feedback]")
    data = StubDatasetDirectory(num_msoas=5, num_sims=1, num_time_steps=25)
    neighbour_map, _ = build_neighbour_map(
        data.static_features_tensor, data.raw_static_features_tensor, None, knn_k=2
    )
    builder = SpatialCovariateBuilder(
        "neighbour_and_nonneighbour_means", neighbour_map, data.num_geographies, 2, data.columns_for_prediction
    )
    model = StubChronosModel(spread=0.0, offset=1.0)
    config = RolloutConfig(
        context_size=5, horizon=4, num_samples=2, window_batch_size=2, save_spaghetti=True, resume=False
    )
    origins = [10, 15]

    output_dir = tempfile.mkdtemp(prefix="chronogenie_test_")
    try:
        stats = run_sliding_window_forecasts(
            temporal_model=model,
            dataset_directory=data,
            covariate_builder=builder,
            config=config,
            origins=origins,
            output_folder=output_dir,
            sim_ids=[0],
        )
        check("every window was written", stats.windows_written == len(origins))
        check(
            "one forward call per rollout step",
            model.calls == config.horizon,
            f"calls={model.calls}",
        )
        check(
            "each call covers windows x paths x anchors items",
            stats.forward_items == config.horizon * len(origins) * config.num_samples * data.num_geographies,
            f"items={stats.forward_items}",
        )

        window_dir = os.path.join(output_dir, "TESTING", "SIM_0", "predictions", "10")
        check("window is reported complete", window_is_complete(window_dir, True))

        table = feather.read_table(os.path.join(window_dir, "predictions.feather")).to_pandas()
        check("prediction rows = horizon x MSOAs", len(table) == config.horizon * data.num_geographies)
        check(
            "timesteps run from the origin",
            sorted(table["timestep"].unique().tolist()) == [10, 11, 12, 13],
        )
        for burden in data.columns_for_prediction:
            for suffix in ["unscaled", "unscaled_gt", "lower_95", "upper_95", "lower_50", "upper_50",
                           "IS_95", "CRPS", "IS_95_log", "CRPS_log"]:
                check(f"predictions.feather has {burden}_{suffix}", f"{burden}_{suffix}" in table.columns)

        truth = data.raw_data_tensor[0, 0, :, 10]
        stored = table[table["timestep"] == 10].sort_values("msoa")["daily_hospitalised_unscaled_gt"].to_numpy()
        check("ground truth column matches the data", np.allclose(stored, truth.numpy()))

        # The stub adds +1 per step to the last context value, so lead L must be truth + L.
        for lead in range(config.horizon):
            row = table[table["timestep"] == 10 + lead].sort_values("msoa")
            expected = data.raw_data_tensor[0, 0, :, 9].numpy() + lead + 1
            check(
                f"feedback compounds at lead {lead}",
                np.allclose(row["daily_hospitalised_unscaled"].to_numpy(), expected),
                f"got {row['daily_hospitalised_unscaled'].to_numpy()[:3]} want {expected[:3]}",
            )

        spaghetti = feather.read_table(os.path.join(window_dir, "sample_spaghetti.feather")).to_pandas()
        check(
            "spaghetti rows = S x H x M x V",
            len(spaghetti) == config.num_samples * config.horizon * data.num_geographies * 2,
        )
        check("spaghetti burden names are populated", set(spaghetti["burden_type"]) == set(data.columns_for_prediction))
        check("spaghetti msoa names are populated", spaghetti["msoa_name"].nunique() == data.num_geographies)

        spatial = feather.read_table(os.path.join(window_dir, "spatial_scores.feather")).to_pandas()
        check("spatial scores have one row per (lead, burden)", len(spatial) == config.horizon * 2)
        check(
            "spatial score columns match GENIE",
            list(spatial.columns)
            == ["timestep", "burden", "energy_score", "variogram_score_p05", "energy_score_log",
                "variogram_score_p05_log"],
        )

        timing = feather.read_table(os.path.join(output_dir, "TESTING", "SIM_0", "window_times.feather")).to_pandas()
        check("timings recorded per window", len(timing) == len(origins))

        # Resume: a second call must not re-run anything.
        model.calls = 0
        resumed = RolloutConfig(
            context_size=5, horizon=4, num_samples=2, window_batch_size=2, save_spaghetti=True, resume=True
        )
        again = run_sliding_window_forecasts(
            temporal_model=model,
            dataset_directory=data,
            covariate_builder=builder,
            config=resumed,
            origins=origins,
            output_folder=output_dir,
            sim_ids=[0],
        )
        check("resume skips completed windows", model.calls == 0 and again.windows_skipped == len(origins))
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def test_truncated_horizon_at_series_end():
    print("\n[origins near the end of the series]")
    data = StubDatasetDirectory(num_msoas=3, num_sims=1, num_time_steps=20)
    neighbour_map, _ = build_neighbour_map(
        data.static_features_tensor, data.raw_static_features_tensor, None, knn_k=1
    )
    builder = SpatialCovariateBuilder("self_only", neighbour_map, data.num_geographies, 2, data.columns_for_prediction)
    model = StubChronosModel(spread=1.0, offset=1.0)
    # Origin 8 has a full 8-day horizon; origin 16 only has 4 days left. Both are in
    # the same rollout batch, so the batch must cope with unequal horizons.
    config = RolloutConfig(
        context_size=4, horizon=8, num_samples=2, window_batch_size=2, save_spaghetti=True, resume=False
    )

    output_dir = tempfile.mkdtemp(prefix="chronogenie_trunc_")
    try:
        stats = run_sliding_window_forecasts(
            temporal_model=model,
            dataset_directory=data,
            covariate_builder=builder,
            config=config,
            origins=[8, 16],
            output_folder=output_dir,
            sim_ids=[0],
        )
        check("both windows were written", stats.windows_written == 2)
        for origin, expected_horizon in [(8, 8), (16, 4)]:
            table = feather.read_table(
                os.path.join(output_dir, "TESTING", "SIM_0", "predictions", str(origin), "predictions.feather")
            ).to_pandas()
            check(
                f"origin {origin} is scored on {expected_horizon} days",
                sorted(table["timestep"].unique().tolist()) == list(range(origin, origin + expected_horizon)),
            )
            check(
                f"origin {origin} has no padding leaking into the scores",
                bool(np.isfinite(table["daily_hospitalised_CRPS"]).all()),
            )
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def test_paths_are_independent():
    print("\n[sample-path independence]")
    data = StubDatasetDirectory(num_msoas=4, num_sims=1, num_time_steps=20)
    neighbour_map, _ = build_neighbour_map(
        data.static_features_tensor, data.raw_static_features_tensor, None, knn_k=2
    )
    builder = SpatialCovariateBuilder("self_only", neighbour_map, data.num_geographies, 2, data.columns_for_prediction)
    model = StubChronosModel(spread=6.0, offset=0.0)  # wide quantiles -> visibly different paths
    config = RolloutConfig(
        context_size=5, horizon=6, num_samples=8, window_batch_size=1, save_spaghetti=True, resume=False,
        count_rounding="none",
    )

    output_dir = tempfile.mkdtemp(prefix="chronogenie_paths_")
    try:
        run_sliding_window_forecasts(
            temporal_model=model,
            dataset_directory=data,
            covariate_builder=builder,
            config=config,
            origins=[10],
            output_folder=output_dir,
            sim_ids=[0],
        )
        window_dir = os.path.join(output_dir, "TESTING", "SIM_0", "predictions", "10")
        spaghetti = feather.read_table(os.path.join(window_dir, "sample_spaghetti.feather")).to_pandas()
        one_series = spaghetti[(spaghetti["msoa"] == 0) & (spaghetti["burden_type"] == "daily_hospitalised")]

        spread_by_lead = one_series.groupby("window_step")["value"].std()
        check("paths differ from each other", float(spread_by_lead.iloc[0]) > 0.0)
        check(
            "forecast spread grows with lead time",
            float(spread_by_lead.iloc[-1]) > float(spread_by_lead.iloc[0]),
            f"lead0={spread_by_lead.iloc[0]:.3f} lead{len(spread_by_lead)-1}={spread_by_lead.iloc[-1]:.3f}",
        )
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def test_checkpoint_resolution():
    print("\n[checkpoint resolution]")
    root = tempfile.mkdtemp(prefix="chronogenie_ckpt_")
    try:
        check("missing directory falls back to the base model", resolve_model_source("base", root) == "base")
        checkpoint = os.path.join(root, "finetuned-ckpt")
        os.makedirs(checkpoint)
        open(os.path.join(checkpoint, "adapter_config.json"), "w").close()
        check(
            "finetuned-ckpt subdirectory is found",
            resolve_model_source("base", root) == checkpoint,
            resolve_model_source("base", root),
        )
        check("None means the base model", resolve_model_source("base", None) == "base")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_real_chronos():
    print("\n[real Chronos-2 forward pass]")
    from chronos_adapter import ChronosTemporalAdapter

    data = StubDatasetDirectory(num_msoas=4, num_sims=1, num_time_steps=30)
    neighbour_map, _ = build_neighbour_map(
        data.static_features_tensor, data.raw_static_features_tensor, None, knn_k=2
    )
    builder = SpatialCovariateBuilder(
        "neighbour_and_nonneighbour_means", neighbour_map, data.num_geographies, 2,
        data.columns_for_prediction, include_day_of_week=True,
    )
    model = ChronosTemporalAdapter(model_id="amazon/chronos-2", device="cpu").to("cpu").eval()
    check("model exposes its native quantile grid", len(model.native_quantile_levels) > 5)

    config = RolloutConfig(
        context_size=14, horizon=3, num_samples=3, window_batch_size=1, predict_row_batch_size=256, resume=False
    )
    output_dir = tempfile.mkdtemp(prefix="chronogenie_real_")
    try:
        stats = run_sliding_window_forecasts(
            temporal_model=model,
            dataset_directory=data,
            covariate_builder=builder,
            config=config,
            origins=[20],
            output_folder=output_dir,
            sim_ids=[0],
        )
        check("real rollout wrote its window", stats.windows_written == 1)
        table = feather.read_table(
            os.path.join(output_dir, "TESTING", "SIM_0", "predictions", "20", "predictions.feather")
        ).to_pandas()
        check("real predictions are finite", bool(np.isfinite(table["daily_hospitalised_unscaled"]).all()))
        check("real predictions are non-negative", bool((table["daily_hospitalised_unscaled"] >= 0).all()))
        check("real counts are integers", bool((table["daily_hospitalised_unscaled"] % 1 == 0).all()))
        check("CRPS is non-negative", bool((table["daily_hospitalised_CRPS"] >= -1e-6).all()))
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chronos", action="store_true", help="also exercise the real Chronos-2 model on CPU")
    args = parser.parse_args()

    import logging

    logging.basicConfig(level=logging.WARNING)

    tests = [
        test_neighbour_map,
        test_covariate_layouts,
        test_sampling,
        test_scoring_matches_genie,
        test_origins,
        test_rollout_feedback,
        test_truncated_horizon_at_series_end,
        test_paths_are_independent,
        test_checkpoint_resolution,
    ]
    if args.chronos:
        tests.append(test_real_chronos)

    for test_fn in tests:
        test_fn()

    print(f"\n{len(PASSED)} checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
