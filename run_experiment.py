#!/usr/bin/env python3
"""
ChronoGenie runner (Chronos-only branch).

This entrypoint keeps only the direct Chronos-2 workflow:
- load data and static context
- optional Chronos LoRA fine-tuning
- autoregressive testing and metric export
"""

import argparse
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
import yaml
import matplotlib.pyplot as plt

from chronos_adapter import ChronosTemporalAdapter, pack_chronos_input
from dataset import DatesetDirectory
from testing import test


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

    context_size: int
    min_timestep: int

    autoregressive_windows: List[int]
    num_samples: int
    num_steps: int
    window_batch_size: int

    chronos_model_id: str
    chronos_finetune_mode: str
    chronos_finetune_steps: int
    chronos_finetune_lr: float
    chronos_finetune_batch_size: int

    transfer_from: Optional[str]
    load_checkpoints_from: Optional[str]

    count_noise_low: float
    count_noise_high: float


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_data_source(data_source_path: str) -> DataSourceConfig:
    if not os.path.exists(data_source_path):
        raise FileNotFoundError(f"Data source configuration not found: {data_source_path}")

    ds_yaml = load_yaml(data_source_path)
    return DataSourceConfig(
        name=ds_yaml.get("name", "unnamed"),
        description=ds_yaml.get("description", ""),
        data_csv=ds_yaml["data_csv"],
        metadata_csv=ds_yaml["metadata_csv"],
        static_features_csv=ds_yaml["static_features_csv"],
        split=ds_yaml["split"],
        column_with_date=ds_yaml.get("column_with_date", "date"),
        column_with_geography=ds_yaml.get("column_with_geography", "MSOA"),
    )


def format_output_dir(template: str, name: str) -> str:
    timestamp = datetime.now().strftime("%d_%m_%y")
    return template.format(name=name, timestamp=timestamp)


def load_experiment_config(
    config_path: str,
    mode_override: Optional[str] = None,
    output_override: Optional[str] = None,
) -> ExperimentConfig:
    cfg = load_yaml(config_path)

    data_source_path = cfg["data_source"]
    data_source = load_data_source(data_source_path)

    experiment = cfg.get("experiment", {})
    output = cfg.get("output", {})
    columns = cfg.get("columns", {})
    execution = cfg.get("execution", {})
    context = cfg.get("context", {})
    testing_cfg = cfg.get("testing", {})
    model = cfg.get("model", {})
    transfer = cfg.get("transfer", {})

    chronos_cfg = model.get("chronos", {})

    output_dir = output_override or format_output_dir(
        output.get("base_dir", "./results/{name}"),
        experiment.get("name", "unnamed"),
    )

    mode = mode_override or execution.get("mode", "both")

    return ExperimentConfig(
        name=experiment.get("name", "unnamed"),
        description=experiment.get("description", ""),
        data_source=data_source,
        output_dir=output_dir,
        columns_for_prediction=columns.get("prediction", []),
        columns_for_context=columns.get("context", []),
        continuous_targets=columns.get("continuous_targets", []),
        mode=mode,
        device=execution.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
        context_size=context.get("size", 30),
        min_timestep=context.get("min_timestep", 0),
        autoregressive_windows=testing_cfg.get("autoregressive_windows", [180]),
        num_samples=testing_cfg.get("num_samples", 100),
        num_steps=testing_cfg.get("num_steps", 50),
        window_batch_size=testing_cfg.get("window_batch_size", 16),
        chronos_model_id=model.get("chronos_model_id", "amazon/chronos-2"),
        chronos_finetune_mode=chronos_cfg.get("mode", "lora"),
        chronos_finetune_steps=int(chronos_cfg.get("steps", 300)),
        chronos_finetune_lr=float(chronos_cfg.get("learning_rate", 1e-5)),
        chronos_finetune_batch_size=int(chronos_cfg.get("batch_size", 32)),
        transfer_from=transfer.get("scalers_from"),
        load_checkpoints_from=transfer.get("checkpoints_from"),
        count_noise_low=float(chronos_cfg.get("count_noise_low", 0.0)),
        count_noise_high=float(chronos_cfg.get("count_noise_high", 1.0)),
    )


def build_dataset_directory(cfg: ExperimentConfig) -> DatesetDirectory:
    ds = cfg.data_source

    for path_name, path in [
        ("data_csv", ds.data_csv),
        ("metadata_csv", ds.metadata_csv),
        ("static_features_csv", ds.static_features_csv),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path_name} not found: {path}")

    df_data = pd.read_csv(ds.data_csv)
    df_settings = pd.read_csv(ds.metadata_csv)
    if df_data["sim"].nunique() != df_settings["sim"].nunique():
        raise ValueError("Number of simulations in data and metadata files does not match")

    continuous_prediction_columns = [
        c for c in cfg.columns_for_prediction if c in cfg.continuous_targets
    ]

    temporal_scalers = None
    spatial_scaler = None
    inference_only = False

    if cfg.transfer_from is not None:
        source_ds = load_data_source(cfg.transfer_from)
        source_dataset_directory = DatesetDirectory(
            data_csv=source_ds.data_csv,
            metadata_csv=source_ds.metadata_csv,
            static_features_csv=source_ds.static_features_csv,
            columns_for_prediction=cfg.columns_for_prediction,
            columns_for_context=cfg.columns_for_context,
            split=source_ds.split,
            continuous_prediction_columns=continuous_prediction_columns,
            column_with_date=source_ds.column_with_date,
            column_with_geography=source_ds.column_with_geography,
            min_timestep=cfg.min_timestep,
            fit_scalers=True,
        )
        temporal_scalers = source_dataset_directory.temporal_scalers
        spatial_scaler = source_dataset_directory.spatial_scaler
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


def build_chronos_training_inputs(cfg: ExperimentConfig, dataset_directory: DatesetDirectory) -> List[torch.Tensor]:
    ctx_col_indices = [dataset_directory.columns_with_data.index(c) for c in cfg.columns_for_context]
    series = []
    for sim_id in dataset_directory.train_sims:
        sim_idx = dataset_directory.resolve_sim_idx(sim_id)
        history_tensors = [dataset_directory.raw_data_tensor[idx, sim_idx] for idx in ctx_col_indices]
        packed = pack_chronos_input(history_tensors, dataset_directory.static_features_tensor)
        series.append(packed)
    return series


def build_chronos_validation_inputs(cfg: ExperimentConfig, dataset_directory: DatesetDirectory) -> List[torch.Tensor]:
    ctx_col_indices = [dataset_directory.columns_with_data.index(c) for c in cfg.columns_for_context]
    series = []
    val_sims = getattr(dataset_directory, "val_sims", [])
    for sim_id in val_sims:
        sim_idx = dataset_directory.resolve_sim_idx(sim_id)
        history_tensors = [dataset_directory.raw_data_tensor[idx, sim_idx] for idx in ctx_col_indices]
        packed = pack_chronos_input(history_tensors, dataset_directory.static_features_tensor)
        series.append(packed)
    return series


def plot_training_metrics(log_history: List[Dict[str, Any]], output_dir: str) -> None:
    """Parses Hugging Face Trainer log history and creates a Loss & LR plot."""
    if not log_history:
        print("--> Warning: No log history found. Skipping metrics plotting.")
        return

    epochs = []
    steps = []
    train_loss = []
    learning_rates = []
    
    val_epochs = []
    val_steps = []
    val_loss = []

    for log in log_history:
        # Check for training metric logs
        if "loss" in log:
            steps.append(log.get("step", 0))
            epochs.append(log.get("epoch", 0.0))
            train_loss.append(log["loss"])
            if "learning_rate" in log:
                learning_rates.append(log["learning_rate"])
        # Check for validation metric logs
        elif "eval_loss" in log:
            val_steps.append(log.get("step", 0))
            val_epochs.append(log.get("epoch", 0.0))
            val_loss.append(log["eval_loss"])

    if not steps:
        print("--> Warning: Log history did not contain continuous training logs. Skipping plot.")
        return

    # Initialize a 2-panel plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Loss Curve
    ax1.plot(steps, train_loss, label="Training Loss", color="royalblue", lw=2)
    if val_loss:
        ax1.plot(val_steps, val_loss, label="Validation Loss", color="crimson", linestyle="--", marker="o", lw=1.5)
    ax1.set_title("Fine-tuning Loss over Steps")
    ax1.set_xlabel("Training Steps")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    # Panel 2: Learning Rate Schedule
    if learning_rates:
        ax2.plot(steps, learning_rates, label="Learning Rate", color="forestgreen", lw=2)
        ax2.set_title("Learning Rate Schedule")
        ax2.set_xlabel("Training Steps")
        ax2.set_ylabel("LR")
        ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, "Learning Rate data not recorded", ha='center', va='center')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "loss_lr_plot.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"--> Successfully generated metrics plot saved to: {plot_path}")


def run_experiment(cfg: ExperimentConfig) -> None:
    print("#" * 80)
    print(f"EXPERIMENT: {cfg.name}")
    print(f"Description: {cfg.description}")
    print(f"Execution Mode: {cfg.mode}")
    print("#" * 80)

    os.makedirs(cfg.output_dir, exist_ok=True)

    dataset_directory = build_dataset_directory(cfg)

    # Check if a fine-tuned adapter exists in the output directory from a previous run
    if cfg.load_checkpoints_from:
        chronos_model_source = cfg.load_checkpoints_from
    elif os.path.exists(os.path.join(cfg.output_dir, "adapter_config.json")):
        print(f"--> Found existing fine-tuned checkpoint in {cfg.output_dir}. Resuming/Testing from this directory.")
        chronos_model_source = cfg.output_dir
    else:
        print(f"--> No checkpoint found. Loading base model: {cfg.chronos_model_id}")
        chronos_model_source = cfg.chronos_model_id

    temporal_model = ChronosTemporalAdapter(model_id=chronos_model_source, device=cfg.device).to(cfg.device)
    # Check if inference_only exists on the object
    if not hasattr(dataset_directory, "inference_only"):
        import warnings
        warnings.warn(
            "'inference_only' attribute was not found on the DatesetDirectory object! "
            "Falling back to False unless this is a transfer/inference-only run.",
            UserWarning
        )
    # Use getattr with a default value of False if the attribute doesn't exist
    inference_only = getattr(dataset_directory, "inference_only", False) or (cfg.transfer_from is not None)

    if cfg.mode in ["train", "both"] and not inference_only:
        print("###############")
        print("# FINE-TUNING #")
        print("###############")

        train_inputs = build_chronos_training_inputs(cfg, dataset_directory)
        val_inputs = build_chronos_validation_inputs(cfg, dataset_directory)
        pred_type_by_name = dict(zip(dataset_directory.columns_for_prediction, dataset_directory.prediction_distribution_types))
        count_context_count = sum(
            1 for c in cfg.columns_for_context if pred_type_by_name.get(c, "lognormal").lower() != "lognormal"
        )

        # Catch logs returned by the adapter
        temporal_model.fine_tune(
            inputs=train_inputs,
            validation_inputs=val_inputs if len(val_inputs) > 0 else None,
            prediction_length=max(cfg.autoregressive_windows),
            finetune_mode=cfg.chronos_finetune_mode,
            learning_rate=cfg.chronos_finetune_lr,
            num_steps=cfg.chronos_finetune_steps,
            batch_size=cfg.chronos_finetune_batch_size,
            context_length=cfg.context_size,
            output_dir=cfg.output_dir,
            disable_data_parallel=True,
            count_row_count=dataset_directory.num_geographies * count_context_count,
            count_noise_low=cfg.count_noise_low,
            count_noise_high=cfg.count_noise_high,
            logging_steps=10,
            eval_steps=50,
            report_to="none",
            disable_tqdm=False,
        )

        # Plot the logs instantly
        logs = getattr(temporal_model, "run_logs", None)
        plot_training_metrics(logs, cfg.output_dir)

    if cfg.mode in ["test", "both"]:
        print("###########")
        print("# TESTING #")
        print("###########")

        for autoregressive_window in cfg.autoregressive_windows:
            print(f"\nEvaluating with autoregressive window: {autoregressive_window}")
            test(
                temporal_model=temporal_model,
                local_profile_encoder=None,
                interaction_encoder=None,
                graph_data=[None, None],
                dataset_directory=dataset_directory,
                context_size=cfg.context_size,
                autoregressive_window_size=autoregressive_window,
                device=cfg.device,
                output_folder=cfg.output_dir,
                num_samples=cfg.num_samples,
                spatial_ablation=True,
                include_latest_context=False,
                spatial_encoder_type="mlp",
                num_steps=cfg.num_steps,
                window_batch_size=cfg.window_batch_size,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ChronoGenie Chronos-only experiments.")
    parser.add_argument("--config", type=str, help="Path to experiment YAML config file")
    parser.add_argument("--config-dir", type=str, help="Directory containing experiment YAML files")
    parser.add_argument("--mode", type=str, choices=["train", "test", "both"], help="Override mode")
    parser.add_argument("--output-dir", type=str, help="Override output directory")

    args = parser.parse_args()

    if not args.config and not args.config_dir:
        parser.error("Either --config or --config-dir must be provided")

    if args.config:
        cfg = load_experiment_config(
            config_path=args.config,
            mode_override=args.mode,
            output_override=args.output_dir,
        )
        run_experiment(cfg)
        return

    config_files = sorted(
        [
            os.path.join(args.config_dir, f)
            for f in os.listdir(args.config_dir)
            if f.endswith(".yaml") or f.endswith(".yml")
        ]
    )

    if len(config_files) == 0:
        raise FileNotFoundError(f"No YAML config files found in {args.config_dir}")

    for config_path in config_files:
        print("\n" + "=" * 80)
        print(f"RUNNING CONFIG: {config_path}")
        print("=" * 80)
        cfg = load_experiment_config(
            config_path=config_path,
            mode_override=args.mode,
            output_override=args.output_dir,
        )
        run_experiment(cfg)


if __name__ == "__main__":
    main()