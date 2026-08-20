#!/usr/bin/env python3
"""ChronoGenie runner: Chronos-2 with spatial covariates, GENIE-comparable outputs.

One config describes one *spatial covariate mode*. The runner can fine-tune
Chronos-2 on that layout and then evaluate the **base** and **fine-tuned** models
side by side, writing each into its own results tree::

    <output_dir>/run.log                     structured log for the whole run
    <output_dir>/run_manifest.json           config, graph stats, timings, checkpoints
    <output_dir>/neighbour_map.json          the exact graph used, for reproducibility
    <output_dir>/finetune/                   trainer output + finetuned-ckpt/
    <output_dir>/baseline/ARW_<H>/TESTING/   pre-trained Chronos-2 results
    <output_dir>/finetuned/ARW_<H>/TESTING/  fine-tuned results

``baseline/ARW_<H>`` and ``finetuned/ARW_<H>`` are drop-in ``--dir1``/``--dir2``
arguments for the GENIE comparison scripts.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import yaml

from chronos_adapter import ChronosTemporalAdapter, dequantise_counts, find_finetuned_checkpoint
from dataset import DatesetDirectory
from spatial_context import (
    COVARIATE_MODES,
    DEFAULT_KNN_K,
    SpatialCovariateBuilder,
    build_neighbour_map,
    day_of_week_features,
    neighbour_map_summary,
)
from testing import test
from testing_sliding_window import RolloutConfig, resolve_origins

logger = logging.getLogger("chronogenie")

VARIANT_BASELINE = "baseline"
VARIANT_FINETUNED = "finetuned"


# ─── Configuration ───────────────────────────────────────────────────────────


@dataclass
class DataSourceConfig:
    name: str
    data_csv: str
    metadata_csv: str
    static_features_csv: str
    split: List[int]
    column_with_date: str = "date"
    column_with_geography: str = "MSOA"
    description: str = ""


@dataclass
class SpatialConfig:
    """How much of the rest of the map each MSOA is allowed to see."""

    covariate_mode: str = "neighbours"
    knn_k: int = DEFAULT_KNN_K
    use_k_nearest_features: bool = True
    use_k_nearest_distance: bool = True
    symmetrise: bool = False
    max_neighbours: Optional[int] = None
    include_day_of_week: bool = True

    def __post_init__(self) -> None:
        if self.covariate_mode not in COVARIATE_MODES:
            raise ValueError(
                f"spatial.covariate_mode must be one of {COVARIATE_MODES}, got '{self.covariate_mode}'"
            )


@dataclass
class FineTuneConfig:
    mode: str = "lora"
    learning_rate: float = 1e-5
    steps: int = 300
    batch_size: int = 512
    logging_steps: int = 10
    eval_steps: int = 50
    count_dequantisation: bool = True
    count_noise_low: float = 0.0
    count_noise_high: float = 1.0
    max_train_sims: Optional[int] = None
    max_train_items: Optional[int] = 20000
    max_val_items: Optional[int] = 512
    val_truncations_per_sim: int = 4


@dataclass
class TestingConfig:
    variants: List[str] = field(default_factory=lambda: [VARIANT_BASELINE, VARIANT_FINETUNED])
    autoregressive_windows: List[int] = field(default_factory=lambda: [60])
    num_samples: int = 20
    origin_stride: int = 1
    max_origins: Optional[int] = None
    min_context: int = 7
    max_test_sims: Optional[int] = None
    window_batch_size: int = 4
    predict_row_batch_size: int = 1024
    ar_step_size: int = 1
    count_rounding: str = "floor"
    save_spaghetti: bool = True
    resume: bool = True

    def __post_init__(self) -> None:
        unknown = [v for v in self.variants if v not in (VARIANT_BASELINE, VARIANT_FINETUNED)]
        if unknown:
            raise ValueError(f"testing.variants may only contain 'baseline'/'finetuned', got {unknown}")


@dataclass
class ExperimentConfig:
    name: str
    description: str
    data_source: DataSourceConfig
    output_dir: str

    columns_for_prediction: List[str]
    columns_for_context: List[str]
    continuous_targets: List[str]

    mode: str
    device: str
    seed: int

    context_size: int
    min_timestep: int

    spatial: SpatialConfig
    finetune: FineTuneConfig
    testing: TestingConfig

    chronos_model_id: str
    transfer_from: Optional[str]
    load_checkpoints_from: Optional[str]

    def summary(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["data_source"] = asdict(self.data_source)
        return payload


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as handle:
        return yaml.safe_load(handle) or {}


def load_data_source(data_source_path: str) -> DataSourceConfig:
    if not os.path.exists(data_source_path):
        raise FileNotFoundError(f"Data source configuration not found: {data_source_path}")

    payload = load_yaml(data_source_path)
    return DataSourceConfig(
        name=payload.get("name", "unnamed"),
        description=payload.get("description", ""),
        data_csv=payload["data_csv"],
        metadata_csv=payload["metadata_csv"],
        static_features_csv=payload["static_features_csv"],
        split=payload["split"],
        column_with_date=payload.get("column_with_date", "date"),
        column_with_geography=payload.get("column_with_geography", "MSOA"),
    )


def format_output_dir(template: str, name: str) -> str:
    return template.format(name=name, timestamp=datetime.now().strftime("%d_%m_%y"))


def _filter_known(payload: Dict[str, Any], cls) -> Dict[str, Any]:
    """Keep only keys the dataclass accepts, warning about the rest."""
    known = {f.name for f in cls.__dataclass_fields__.values()}
    unknown = sorted(set(payload) - known)
    if unknown:
        logger.warning("Ignoring unknown %s keys in config: %s", cls.__name__, unknown)
    return {k: v for k, v in payload.items() if k in known}


def load_experiment_config(
    config_path: str,
    mode_override: Optional[str] = None,
    output_override: Optional[str] = None,
) -> ExperimentConfig:
    cfg = load_yaml(config_path)

    data_source = load_data_source(cfg["data_source"])
    experiment = cfg.get("experiment", {})
    output = cfg.get("output", {})
    columns = cfg.get("columns", {})
    execution = cfg.get("execution", {})
    context = cfg.get("context", {})
    model = cfg.get("model", {})
    transfer = cfg.get("transfer", {})

    return ExperimentConfig(
        name=experiment.get("name", "unnamed"),
        description=experiment.get("description", ""),
        data_source=data_source,
        output_dir=output_override
        or format_output_dir(output.get("base_dir", "./results/{name}"), experiment.get("name", "unnamed")),
        columns_for_prediction=columns.get("prediction", []),
        columns_for_context=columns.get("context", []),
        continuous_targets=columns.get("continuous_targets", []),
        mode=mode_override or execution.get("mode", "both"),
        device=execution.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
        seed=int(execution.get("seed", 26)),
        context_size=int(context.get("size", 30)),
        min_timestep=int(context.get("min_timestep", 0)),
        spatial=SpatialConfig(**_filter_known(cfg.get("spatial", {}), SpatialConfig)),
        finetune=FineTuneConfig(**_filter_known(model.get("chronos", {}), FineTuneConfig)),
        testing=TestingConfig(**_filter_known(cfg.get("testing", {}), TestingConfig)),
        chronos_model_id=model.get("chronos_model_id", "amazon/chronos-2"),
        transfer_from=transfer.get("scalers_from"),
        load_checkpoints_from=transfer.get("checkpoints_from"),
    )


# ─── Logging & provenance ────────────────────────────────────────────────────


def setup_logging(output_dir: str, verbose: bool = False) -> str:
    """Log to console and to ``<output_dir>/run.log``; returns the log path."""
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "run.log")

    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%H:%M:%S")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    )
    root.addHandler(file_handler)

    for noisy in ("matplotlib", "httpx", "httpcore", "urllib3", "filelock", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return log_path


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=os.path.dirname(os.path.abspath(__file__)), text=True
        ).strip()
    except Exception:
        return "unknown"


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RunManifest:
    """Accumulates everything needed to reproduce and interpret a run."""

    def __init__(self, cfg: ExperimentConfig):
        self.path = os.path.join(cfg.output_dir, "run_manifest.json")
        self.payload: Dict[str, Any] = {
            "experiment": cfg.name,
            "description": cfg.description,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "git_revision": git_revision(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "config": cfg.summary(),
            "stages": {},
        }

    def record(self, key: str, value: Any) -> None:
        self.payload["stages"][key] = value
        self.save()

    def save(self) -> None:
        self.payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
        with open(self.path, "w") as handle:
            json.dump(self.payload, handle, indent=2, default=str)


# ─── Dataset & spatial context ───────────────────────────────────────────────


def build_dataset_directory(cfg: ExperimentConfig) -> DatesetDirectory:
    ds = cfg.data_source
    for label, path in [
        ("data_csv", ds.data_csv),
        ("metadata_csv", ds.metadata_csv),
        ("static_features_csv", ds.static_features_csv),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} not found: {path}")

    df_data = pd.read_csv(ds.data_csv, usecols=["sim"])
    df_settings = pd.read_csv(ds.metadata_csv)
    if df_data["sim"].nunique() != df_settings["sim"].nunique():
        raise ValueError("Number of simulations in data and metadata files does not match")
    del df_data, df_settings

    continuous_prediction_columns = [c for c in cfg.columns_for_prediction if c in cfg.continuous_targets]

    temporal_scalers = None
    spatial_scaler = None
    inference_only = False

    if cfg.transfer_from is not None:
        source = load_data_source(cfg.transfer_from)
        logger.info("Transfer run: fitting scalers on %s", source.name)
        source_directory = DatesetDirectory(
            data_csv=source.data_csv,
            metadata_csv=source.metadata_csv,
            static_features_csv=source.static_features_csv,
            columns_for_prediction=cfg.columns_for_prediction,
            columns_for_context=cfg.columns_for_context,
            split=source.split,
            continuous_prediction_columns=continuous_prediction_columns,
            column_with_date=source.column_with_date,
            column_with_geography=source.column_with_geography,
            min_timestep=cfg.min_timestep,
            fit_scalers=True,
        )
        temporal_scalers = source_directory.temporal_scalers
        spatial_scaler = source_directory.spatial_scaler
        inference_only = True

    return DatesetDirectory(
        data_csv=ds.data_csv,
        metadata_csv=ds.metadata_csv,
        static_features_csv=ds.static_features_csv,
        columns_for_prediction=cfg.columns_for_prediction,
        columns_for_context=cfg.columns_for_context,
        split=ds.split,
        continuous_prediction_columns=continuous_prediction_columns,
        column_with_date=ds.column_with_date,
        column_with_geography=ds.column_with_geography,
        min_timestep=cfg.min_timestep,
        temporal_scalers=temporal_scalers,
        spatial_scaler=spatial_scaler,
        fit_scalers=(temporal_scalers is None),
        inference_only=inference_only,
        output_dir=cfg.output_dir,
    )


def build_covariate_builder(
    cfg: ExperimentConfig, dataset_directory: DatesetDirectory
) -> tuple[SpatialCovariateBuilder, dict[int, list[int]], Dict[str, float]]:
    """Build the neighbourhood and the covariate layout shared by training and testing."""
    feature_names = list(dataset_directory.xr_static_features.coords["static_features"].values)
    neighbour_map, _ = build_neighbour_map(
        scaled_static_features=dataset_directory.static_features_tensor,
        raw_static_features=dataset_directory.raw_static_features_tensor,
        static_feature_names=feature_names,
        knn_k=cfg.spatial.knn_k,
        use_k_nearest_features=cfg.spatial.use_k_nearest_features,
        use_k_nearest_distance=cfg.spatial.use_k_nearest_distance,
        symmetrise=cfg.spatial.symmetrise,
        max_neighbours=cfg.spatial.max_neighbours,
    )
    stats = neighbour_map_summary(neighbour_map)
    logger.info(
        "Neighbourhood (per-node union of %d feature-NN and %d geo-NN, symmetrise=%s): "
        "%d nodes, degree min/mean/max = %d/%.1f/%d, %d directed edges, %.0f%% reciprocal",
        cfg.spatial.knn_k,
        cfg.spatial.knn_k,
        cfg.spatial.symmetrise,
        int(stats["num_nodes"]),
        int(stats["min_degree"]),
        stats["mean_degree"],
        int(stats["max_degree"]),
        int(stats["num_directed_edges"]),
        100.0 * stats["fraction_reciprocal"],
    )

    builder = SpatialCovariateBuilder(
        mode=cfg.spatial.covariate_mode,
        neighbour_map=neighbour_map,
        num_nodes=dataset_directory.num_geographies,
        num_channels=len(cfg.columns_for_prediction),
        channel_names=cfg.columns_for_prediction,
        include_day_of_week=cfg.spatial.include_day_of_week,
    )
    logger.info(
        "Covariate mode '%s': %d target rows + %d..%d covariate rows per anchor (weekday=%s)",
        builder.mode,
        builder.num_channels,
        min(builder.num_covariates(m) for m in range(builder.num_nodes)),
        max(builder.num_covariates(m) for m in range(builder.num_nodes)),
        builder.include_day_of_week,
    )
    return builder, neighbour_map, stats


def export_neighbour_map(output_dir: str, neighbour_map: dict[int, list[int]], dataset_directory) -> None:
    """Persist the graph so a later run can be checked against it."""
    names = dataset_directory.df_geo_metadata["MSOA"].to_numpy()
    payload = {
        str(node): {"msoa": str(names[node]), "neighbours": [int(j) for j in neighbours],
                    "neighbour_msoas": [str(names[j]) for j in neighbours]}
        for node, neighbours in neighbour_map.items()
    }
    with open(os.path.join(output_dir, "neighbour_map.json"), "w") as handle:
        json.dump(payload, handle, indent=1)


# ─── Fine-tuning inputs ──────────────────────────────────────────────────────


def _count_row_mask(num_rows: int, count_channels: List[int]) -> torch.Tensor:
    """True for the anchor's own count rows.

    Only the target rows are jittered: they are what the quantile loss is computed
    against. Covariate rows are inputs, and the mean-based ones are not counts at all.
    """
    mask = torch.zeros(num_rows, dtype=torch.bool)
    for channel in count_channels:
        mask[channel] = True
    return mask


def build_finetune_inputs(
    cfg: ExperimentConfig,
    dataset_directory: DatesetDirectory,
    builder: SpatialCovariateBuilder,
    prediction_length: int,
    rng: np.random.Generator,
) -> tuple[List[dict], List[dict]]:
    """Build training and validation inputs in the *same* layout used at test time.

    Training items are whole simulations: Chronos-2's training dataset samples a
    random context window from each series on every step, so one item per
    (simulation, anchor MSOA) already covers every window.

    Validation items are truncated copies of a series -- Chronos-2 validates on the
    final window of whatever it is given, so several truncations per simulation are
    needed for the validation loss to reflect more than the epidemic tail.
    """
    prediction_indices = [dataset_directory.columns_with_data.index(c) for c in cfg.columns_for_prediction]
    distribution_types = dict(
        zip(dataset_directory.columns_for_prediction, dataset_directory.prediction_distribution_types)
    )
    count_channels = [
        i for i, name in enumerate(cfg.columns_for_prediction)
        if distribution_types.get(name, "nb").lower() != "lognormal"
    ]
    num_time_steps = dataset_directory.num_time_steps
    num_anchors = dataset_directory.num_geographies

    def series_inputs(sim_id, end_step: int) -> List[dict]:
        sim_idx = dataset_directory.resolve_sim_idx(sim_id)
        history = dataset_directory.raw_data_tensor[prediction_indices, sim_idx, :, :end_step].to(torch.float32)
        past_dow = future_dow = None
        if builder.include_day_of_week:
            start_dow = 0
            metadata = dataset_directory.df_start_time_metadata
            rows = metadata[metadata["sim"] == sim_id]
            if len(rows):
                start_dow = int(rows["dow"].iloc[0])
            past_dow = day_of_week_features(start_dow, np.arange(end_step))
            future_dow = day_of_week_features(start_dow, np.arange(end_step, end_step + prediction_length))
        return builder.build_inputs(
            history, prediction_length=prediction_length, past_dow=past_dow, future_dow=future_dow
        )

    train_sims = list(dataset_directory.train_sims)
    if cfg.finetune.max_train_sims is not None and len(train_sims) > cfg.finetune.max_train_sims:
        train_sims = [train_sims[i] for i in rng.choice(len(train_sims), cfg.finetune.max_train_sims, replace=False)]

    train_inputs: List[dict] = []
    for sim_id in train_sims:
        train_inputs.extend(series_inputs(sim_id, num_time_steps))

    if cfg.finetune.max_train_items is not None and len(train_inputs) > cfg.finetune.max_train_items:
        keep = rng.choice(len(train_inputs), cfg.finetune.max_train_items, replace=False)
        train_inputs = [train_inputs[i] for i in sorted(keep)]

    if cfg.finetune.count_dequantisation and count_channels:
        for item in train_inputs:
            mask = _count_row_mask(item["context"].shape[0], count_channels)
            item["context"] = dequantise_counts(
                item["context"], mask, cfg.finetune.count_noise_low, cfg.finetune.count_noise_high
            )

    val_inputs: List[dict] = []
    val_sims = list(getattr(dataset_directory, "val_sims", []))
    if val_sims and cfg.finetune.max_val_items:
        truncations = max(1, cfg.finetune.val_truncations_per_sim)
        # Space the truncation points across the usable part of the series.
        first = max(cfg.testing.min_context, cfg.context_size) + prediction_length
        candidates = np.linspace(first, num_time_steps, truncations + 1)[:-1].astype(int)
        candidates = sorted({int(c) for c in candidates if first <= c <= num_time_steps})

        per_sim_items = max(1, len(candidates) * num_anchors)
        sims_needed = max(1, int(np.ceil(cfg.finetune.max_val_items / per_sim_items)))
        for sim_id in val_sims[:sims_needed]:
            for end_step in candidates:
                val_inputs.extend(series_inputs(sim_id, end_step))

        if len(val_inputs) > cfg.finetune.max_val_items:
            keep = rng.choice(len(val_inputs), cfg.finetune.max_val_items, replace=False)
            val_inputs = [val_inputs[i] for i in sorted(keep)]

    logger.info(
        "Fine-tuning inputs: %d training items (%d sims) / %d validation items | rows per item %d..%d",
        len(train_inputs),
        len(train_sims),
        len(val_inputs),
        min((int(i["context"].shape[0]) for i in train_inputs), default=0),
        max((int(i["context"].shape[0]) for i in train_inputs), default=0),
    )
    return train_inputs, val_inputs


def plot_training_curves(log_history: List[Dict[str, Any]], output_dir: str) -> Optional[str]:
    """Plot loss and learning rate against training step."""
    if not log_history:
        logger.warning("No trainer logs captured; skipping the training-curve plot.")
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train = [(e["step"], e["loss"]) for e in log_history if "loss" in e and e.get("step") is not None]
    validation = [(e["step"], e["eval_loss"]) for e in log_history if "eval_loss" in e and e.get("step") is not None]
    learning_rates = [
        (e["step"], e["learning_rate"]) for e in log_history if "learning_rate" in e and e.get("step") is not None
    ]

    if not train:
        logger.warning("Trainer logs contained no loss entries; skipping the training-curve plot.")
        return None

    figure, (left, right) = plt.subplots(1, 2, figsize=(13, 4.5))
    left.plot(*zip(*train), label="training loss", color="royalblue", marker="o", markersize=3)
    if validation:
        left.plot(*zip(*validation), label="validation loss", color="crimson", ls="--", marker="s", markersize=4)
    left.set(title="Chronos-2 fine-tuning loss", xlabel="step", ylabel="quantile loss")
    left.grid(True, ls=":", alpha=0.6)
    left.legend()

    if learning_rates:
        right.plot(*zip(*learning_rates), color="forestgreen", marker="o", markersize=3)
        right.set(title="Learning rate schedule", xlabel="step", ylabel="lr")
        right.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        right.grid(True, ls=":", alpha=0.6)
    else:
        right.text(0.5, 0.5, "no learning-rate entries", ha="center", va="center")

    plt.tight_layout()
    path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close(figure)
    logger.info("Wrote training curves to %s", path)
    return path


# ─── Workload estimation ─────────────────────────────────────────────────────


def estimate_workload(
    cfg: ExperimentConfig, builder: SpatialCovariateBuilder, num_sims: int, origins: List[int], horizon: int
) -> Dict[str, float]:
    """Forward-pass counts for one variant, so the cost is visible before launching.

    Cost is linear in ``sims x origins x horizon x paths x MSOAs``; the covariate mode
    only changes the rows per item. This is the number to shrink (via ``origin_stride``,
    ``max_test_sims`` or ``num_samples``) when a configuration is too slow.
    """
    rows_per_anchor = builder.total_rows() / max(1, builder.num_nodes)
    steps = int(np.ceil(horizon / max(1, cfg.testing.ar_step_size)))
    items = num_sims * len(origins) * steps * cfg.testing.num_samples * builder.num_nodes
    rows = items * rows_per_anchor
    estimate = {
        "sims": float(num_sims),
        "origins": float(len(origins)),
        "rollout_steps": float(steps),
        "sample_paths": float(cfg.testing.num_samples),
        "anchors": float(builder.num_nodes),
        "rows_per_anchor": round(rows_per_anchor, 1),
        "forward_items": float(items),
        "forward_rows": float(rows),
        "forward_batches_at_row_batch": float(np.ceil(rows / max(1, cfg.testing.predict_row_batch_size))),
    }
    logger.info(
        "Workload per variant: %.3gM items / %.3gM rows / %.3gk forward batches "
        "(sims=%d x origins=%d x steps=%d x paths=%d x anchors=%d)",
        items / 1e6,
        rows / 1e6,
        estimate["forward_batches_at_row_batch"] / 1e3,
        num_sims,
        len(origins),
        steps,
        cfg.testing.num_samples,
        builder.num_nodes,
    )
    return estimate


def benchmark_throughput(
    cfg: ExperimentConfig,
    dataset_directory: DatesetDirectory,
    builder: SpatialCovariateBuilder,
    estimate: Dict[str, float],
    num_steps: int = 3,
) -> Dict[str, float]:
    """Time a few real forward passes and turn the workload estimate into hours.

    Uses one rollout step's worth of inputs (all anchors for a single path), which is
    the unit the rollout repeats, so the measured rows/second carries straight over.
    """
    prediction_indices = [dataset_directory.columns_with_data.index(c) for c in cfg.columns_for_prediction]
    sim_idx = dataset_directory.resolve_sim_idx(list(dataset_directory.test_sims)[0])
    end = min(dataset_directory.num_time_steps, cfg.context_size + cfg.testing.min_context)
    history = dataset_directory.raw_data_tensor[
        prediction_indices, sim_idx, :, max(0, end - cfg.context_size) : end
    ].to(torch.float32)

    past_dow = future_dow = None
    if builder.include_day_of_week:
        past_dow = day_of_week_features(0, np.arange(history.shape[-1]))
        future_dow = day_of_week_features(0, np.arange(history.shape[-1], history.shape[-1] + cfg.testing.ar_step_size))

    inputs = builder.build_inputs(
        history, prediction_length=cfg.testing.ar_step_size, past_dow=past_dow, future_dow=future_dow
    )
    rows = sum(int(i["context"].shape[0]) for i in inputs)

    model = ChronosTemporalAdapter(model_id=cfg.chronos_model_id, device=cfg.device).to(cfg.device).eval()
    with torch.no_grad():
        model.predict_quantiles(  # warm-up: excludes compilation and allocator growth
            inputs=inputs,
            prediction_length=cfg.testing.ar_step_size,
            quantile_levels=model.native_quantile_levels,
            batch_size=cfg.testing.predict_row_batch_size,
        )
        started = time.perf_counter()
        for _ in range(num_steps):
            model.predict_quantiles(
                inputs=inputs,
                prediction_length=cfg.testing.ar_step_size,
                quantile_levels=model.native_quantile_levels,
                batch_size=cfg.testing.predict_row_batch_size,
            )
        elapsed = time.perf_counter() - started
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    rows_per_second = (rows * num_steps) / max(elapsed, 1e-9)
    hours = estimate["forward_rows"] / rows_per_second / 3600.0
    result = {
        "device": cfg.device,
        "rows_per_second": round(rows_per_second, 1),
        "projected_hours_per_variant": round(hours, 2),
        "projected_hours_all_variants": round(hours * len(cfg.testing.variants), 2),
    }
    logger.info(
        "Benchmark on %s: %.0f rows/s -> %.1f h per variant, %.1f h for %d variant(s). "
        "Halve it by halving any of origins / sims / num_samples.",
        cfg.device,
        rows_per_second,
        hours,
        hours * len(cfg.testing.variants),
        len(cfg.testing.variants),
    )
    return result


# ─── Experiment ──────────────────────────────────────────────────────────────


def select_test_sims(cfg: ExperimentConfig, dataset_directory: DatesetDirectory) -> List:
    sims = list(dataset_directory.test_sims)
    if cfg.testing.max_test_sims is not None:
        sims = sims[: cfg.testing.max_test_sims]
    return sims


def run_experiment(cfg: ExperimentConfig, estimate_only: bool = False, benchmark: bool = False) -> None:
    os.makedirs(cfg.output_dir, exist_ok=True)
    setup_logging(cfg.output_dir)
    set_global_seed(cfg.seed)

    logger.info("=" * 78)
    logger.info("EXPERIMENT %s | mode=%s | device=%s", cfg.name, cfg.mode, cfg.device)
    logger.info("%s", cfg.description)
    logger.info("Output directory: %s", cfg.output_dir)
    logger.info("=" * 78)

    manifest = RunManifest(cfg)

    dataset_directory = build_dataset_directory(cfg)
    builder, neighbour_map, neighbour_stats = build_covariate_builder(cfg, dataset_directory)
    export_neighbour_map(cfg.output_dir, neighbour_map, dataset_directory)
    manifest.record("spatial", {"neighbour_stats": neighbour_stats, "covariate_mode": builder.mode})

    test_sims = select_test_sims(cfg, dataset_directory)
    horizons = cfg.testing.autoregressive_windows
    origins = resolve_origins(
        num_time_steps=dataset_directory.num_time_steps,
        min_timestep=cfg.min_timestep,
        horizon=max(horizons),
        origin_stride=cfg.testing.origin_stride,
        max_origins=cfg.testing.max_origins,
        min_context=cfg.testing.min_context,
    )
    estimate = estimate_workload(cfg, builder, len(test_sims), origins, max(horizons))
    plan = {
        "test_sims": [str(s) for s in test_sims],
        "num_origins": len(origins),
        "origins": origins[:10] + (["..."] if len(origins) > 10 else []),
        "horizons": horizons,
        "estimate": estimate,
    }
    if benchmark:
        plan["benchmark"] = benchmark_throughput(cfg, dataset_directory, builder, estimate)
    manifest.record("plan", plan)

    if estimate_only:
        logger.info("Stopping before any training or evaluation (--estimate).")
        return

    inference_only = getattr(dataset_directory, "inference_only", False) or (cfg.transfer_from is not None)
    finetune_dir = os.path.join(cfg.output_dir, "finetune")
    checkpoint = cfg.load_checkpoints_from or find_finetuned_checkpoint(finetune_dir)

    # -- fine-tuning ---------------------------------------------------------
    if cfg.mode in ("train", "both") and not inference_only:
        if checkpoint and cfg.testing.resume:
            logger.info("Fine-tuned checkpoint already present at %s; skipping training.", checkpoint)
        else:
            logger.info("### FINE-TUNING (%s, %d steps) ###", cfg.finetune.mode, cfg.finetune.steps)
            os.makedirs(finetune_dir, exist_ok=True)
            rng = np.random.default_rng(cfg.seed)
            train_inputs, val_inputs = build_finetune_inputs(
                cfg, dataset_directory, builder, prediction_length=cfg.testing.ar_step_size, rng=rng
            )

            model = ChronosTemporalAdapter(model_id=cfg.chronos_model_id, device=cfg.device).to(cfg.device)
            model.fine_tune(
                inputs=train_inputs,
                validation_inputs=val_inputs or None,
                prediction_length=cfg.testing.ar_step_size,
                finetune_mode=cfg.finetune.mode,
                learning_rate=cfg.finetune.learning_rate,
                num_steps=cfg.finetune.steps,
                batch_size=cfg.finetune.batch_size,
                context_length=cfg.context_size,
                output_dir=finetune_dir,
                disable_data_parallel=True,
                logging_steps=cfg.finetune.logging_steps,
                eval_steps=cfg.finetune.eval_steps,
                report_to="none",
            )
            model.save_training_logs(cfg.output_dir)
            plot_training_curves(model.run_logs, cfg.output_dir)
            manifest.record(
                "finetune",
                {
                    "steps": cfg.finetune.steps,
                    "train_items": len(train_inputs),
                    "val_items": len(val_inputs),
                    "checkpoint": find_finetuned_checkpoint(finetune_dir),
                    "final_logs": model.run_logs[-3:],
                },
            )
            del model, train_inputs, val_inputs
            checkpoint = find_finetuned_checkpoint(finetune_dir)

    # -- evaluation ----------------------------------------------------------
    if cfg.mode not in ("test", "both"):
        logger.info("Mode '%s' does not include testing; done.", cfg.mode)
        return

    for variant in cfg.testing.variants:
        if variant == VARIANT_FINETUNED and not checkpoint:
            logger.warning("Variant 'finetuned' requested but no checkpoint was found; skipping it.")
            continue

        model_source = checkpoint if variant == VARIANT_FINETUNED else cfg.chronos_model_id
        logger.info("### TESTING variant=%s | model=%s ###", variant, model_source)
        model = ChronosTemporalAdapter(model_id=model_source, device=cfg.device).to(cfg.device).eval()

        for horizon in horizons:
            rollout = RolloutConfig(
                context_size=cfg.context_size,
                horizon=horizon,
                num_samples=cfg.testing.num_samples,
                window_batch_size=cfg.testing.window_batch_size,
                predict_row_batch_size=cfg.testing.predict_row_batch_size,
                ar_step_size=cfg.testing.ar_step_size,
                count_rounding=cfg.testing.count_rounding,
                save_spaghetti=cfg.testing.save_spaghetti,
                resume=cfg.testing.resume,
                seed=cfg.seed,
            )
            variant_dir = os.path.join(cfg.output_dir, variant, f"ARW_{horizon}")
            stats = test(
                temporal_model=model,
                dataset_directory=dataset_directory,
                covariate_builder=builder,
                config=rollout,
                origins=origins,
                output_folder=variant_dir,
                sim_ids=test_sims,
            )
            manifest.record(
                f"test::{variant}::ARW_{horizon}",
                {"results_dir": variant_dir, "model_source": model_source, **stats.as_dict()},
            )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info("Run complete. Manifest: %s", manifest.path)
    if {VARIANT_BASELINE, VARIANT_FINETUNED}.issubset(set(cfg.testing.variants)):
        horizon = horizons[0]
        logger.info(
            "Compare with:\n  python compare_runs.py --run %s=%s --run %s=%s --output %s",
            VARIANT_BASELINE,
            os.path.join(cfg.output_dir, VARIANT_BASELINE, f"ARW_{horizon}"),
            VARIANT_FINETUNED,
            os.path.join(cfg.output_dir, VARIANT_FINETUNED, f"ARW_{horizon}"),
            os.path.join(cfg.output_dir, "comparison"),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ChronoGenie Chronos-2 experiments.")
    parser.add_argument("--config", type=str, help="Path to an experiment YAML config")
    parser.add_argument("--config-dir", type=str, help="Directory of experiment YAML configs to run in order")
    parser.add_argument("--mode", type=str, choices=["train", "test", "both"], help="Override execution mode")
    parser.add_argument("--output-dir", type=str, help="Override the output directory")
    parser.add_argument(
        "--estimate", action="store_true", help="Report the forward-pass workload and exit without running"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Time real forward passes on this device and project the run time (implies --estimate)",
    )
    args = parser.parse_args()

    if not args.config and not args.config_dir:
        parser.error("Either --config or --config-dir must be provided")

    config_paths = (
        [args.config]
        if args.config
        else sorted(
            os.path.join(args.config_dir, name)
            for name in os.listdir(args.config_dir)
            if name.endswith((".yaml", ".yml"))
        )
    )
    if not config_paths:
        raise FileNotFoundError(f"No YAML config files found in {args.config_dir}")

    for config_path in config_paths:
        print("\n" + "=" * 80 + f"\nRUNNING CONFIG: {config_path}\n" + "=" * 80)
        cfg = load_experiment_config(config_path, mode_override=args.mode, output_override=args.output_dir)
        run_experiment(cfg, estimate_only=args.estimate or args.benchmark, benchmark=args.benchmark)


if __name__ == "__main__":
    main()
