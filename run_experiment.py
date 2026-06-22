#!/usr/bin/env python3
"""
###########################
# GENIE Experiment Runner #
###########################

This is a script for running GENIE experiments using YAML configuration files.

Usage:
    # Run a single experiment (train + test)
    python run_experiment.py --config configs/experiments/gnn_final_1k.yaml
    
    # Run all experiments in a directory
    python run_experiment.py --config-dir configs/experiments/
    
    # Run only training
    python run_experiment.py --config configs/experiments/gnn_final_1k.yaml --mode train
    
    # Run only testing
    python run_experiment.py --config configs/experiments/gnn_final_1k.yaml --mode test
    
    # Cross-geography inference: load trained model and test on different geography
    python run_experiment.py --config configs/experiments/darlington_inference.yaml --mode test
    
    # Override output directory
    python run_experiment.py --config configs/experiments/gnn_final_1k.yaml --output-dir /custom/path
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import copy

import yaml
import torch
import pandas as pd

from graph_construction import get_graph
from models import GNNEncoder, MLP, MLPEncoder
from diagnostics.param_counter import write_model_features
from training import train
from testing import test
from train_test_helpers import cache_graph
from lr_finder_runner import run_lr_finder


##############################
# Configuration Data Classes #
##############################

@dataclass
class DataSourceConfig:
    """Configuration for data source paths and settings."""
    name: str
    data_csv: str
    metadata_csv: str
    static_features_csv: str
    split: List[int]
    column_with_date: str = "date"
    column_with_geography: str = "MSOA"
    description: str = ""
    max_sims: Optional[int] = None


@dataclass
class LRFinderConfig:
    """Configuration for a learning-rate sweep run."""
    start_lr: float = 1e-7
    end_lr: float = 10.0
    num_iter: int = 100
    beta: float = 0.98
    diverge_th: float = 5.0
    skip_start: int = 10
    skip_end: int = 5
    output_subdir: str = "lr_finder"
    csv_filename: str = "lr_finder_curve.csv"
    plot_filename: str = "lr_finder_curve.png"


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    # Identification
    name: str
    description: str
    
    # Data source
    data_source: DataSourceConfig
    
    # Output
    output_dir: str
    
    # Columns
    columns_for_prediction: List[str]
    columns_for_context: List[str]
    continuous_targets: List[str]
    
    # Execution
    mode: str  # "train", "test", "both", or "lr_finder"
    resume_training: bool
    device: str  # "cuda" or "cpu"
    
    # Training
    num_epochs: int
    learning_rate: float
    num_batches: int
    batch_size: int
    validation_batch_size: Optional[int]
    use_early_stopping: bool
    ema_alpha: float
    use_compile: bool
    validation_thinning_factor: int
    
    # Context
    context_size: int
    include_latest_context: bool
    min_timestep: int  # Minimum timestep to start predictions (e.g., 30 for R_t)
    
    # Multistep Prediction
    multistep_k: int
    multistep_weights: List[float]
    
    # Embeddings
    embedding_size_spatial: int
    embedding_size_temporal: int
    
    # Model architecture
    model_type: str  # "mlp", "flow_matching", "diffusion", or "normalising_flow"
    temporal_mlp_hidden: List[int]
    spatial_encoder_type: str  # "gnn" or "mlp"
    
    # Flow Matching
    flow_matching_hidden: List[int]
    flow_matching_sigma: float
    flow_matching_dropout: float

    # Diffusion
    diffusion_hidden: List[int]
    diffusion_time_embed_dim: int
    diffusion_num_train_timesteps: int
    diffusion_beta_start: float
    diffusion_beta_end: float
    diffusion_dropout: float
    
    # Variational Flow Matching
    vfm_hidden: List[int]
    vfm_sigma: float
    vfm_dropout: float
    vfm_parameterization: str
    vfm_use_skip_connection: bool
    vfm_debug: bool
    vfm_latent_dim: int
    vfm_kl_weight: float
    vfm_kl_warmup_steps: int
    vfm_encoder_hidden: List[int]
    
    # NSF Flow
    normalising_flow_nsf_hidden: List[int]
    normalising_flow_nsf_num_transforms: int
    normalising_flow_nsf_sigma: float
    normalising_flow_nsf_dropout: float
    normalising_flow_nsf_tail_bound: float
    
    # MAF Flow
    normalising_flow_maf_hidden: List[int]
    normalising_flow_maf_num_transforms: int
    normalising_flow_maf_sigma: float
    normalising_flow_maf_dropout: float

    # CNF Flow
    normalising_flow_cnf_hidden: List[int]
    normalising_flow_cnf_sigma: float
    normalising_flow_cnf_dropout: float


    gnn_dropout: float
    gnn_self_loops: bool
    gnn_residual: bool
    gnn_attention_heads: List[int]
    gnn_attention_concat: List[bool]

    mlp_hidden_layers: List[int] 
    mlp_dropout: float
    
    # Graph
    edge_threshold_spatial: Optional[float]
    edge_threshold_temporal: Optional[float]
    use_edge_attr: bool
    edge_attr_dim: int
    use_k_nearest_features: bool
    use_k_nearest_distance: bool
    knn_k: int
    
    # Ablation 
    spatial_ablation: bool
    
    # Testing
    autoregressive_windows: List[int]
    num_samples: int
    num_steps: int = 50
    window_batch_size: int = 16
    testing_checkpoint_source: str = "best"
    
    # Cross-geography inference (optional)
    # If set, load scalers from this trained experiment instead of fitting new ones
    transfer_from: Optional[str] = None
    # Path to load model checkpoints from (if different from output_dir)

    # LR finder
    lr_finder: LRFinderConfig = field(default_factory=LRFinderConfig)
    load_checkpoints_from: Optional[str] = None

    poisson_noise_context: bool = False  # Apply Poisson noise to NB context columns during training only
    augmentation_thinning_factor: int = 5  # Use augmented batches once every N training batches
    optimiser_type: str = "adamw"
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    muon_weight_decay: float = 0.0
    adamw_lr: float = 3e-4
    adamw_weight_decay: float = 1e-2
    gradient_accumulation_steps: int = 1
    use_bfloat16: bool = False
    scheduler_type: str = "onecycle"
    cosine_eta_min: float = 0.0
    cosine_warmup_fraction: float = 0.05
    max_grad_norm: Optional[float] = 1.0
    gradient_clip_value: Optional[float] = 1.0


#########################
# Configuration Loading #
#########################

def load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML file and return its contents as a dictionary."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_data_source(data_source_path: str) -> DataSourceConfig:
    """
    Load data source configuration from an explicit path.
    
    Args:
        data_source_path: Full path to the data source YAML file
        
    Returns:
        DataSourceConfig object
        
    Raises:
        FileNotFoundError: If the specified path does not exist
    """
    if not os.path.exists(data_source_path):
        raise FileNotFoundError(
            f"Data source configuration not found: {data_source_path}\n"
            f"Please provide the full path to the data source YAML file."
        )
    
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
        max_sims=ds_yaml.get("max_sims"),
    )


def format_output_dir(template: str, name: str) -> str:
    """Format output directory template with placeholders."""
    timestamp = datetime.now().strftime("%d_%m_%y")
    return template.format(name=name, timestamp=timestamp)


def load_experiment_config(config_path: str, mode_override: Optional[str] = None,
                           output_override: Optional[str] = None,
                           testing_checkpoint_source_override: Optional[str] = None) -> ExperimentConfig:
    """
    Load an experiment configuration from a YAML file.
    
    Args:
        config_path: Path to the experiment YAML file
        mode_override: Override the execution mode ("train", "test", or "both")
        output_override: Override the output directory
    """
    cfg = load_yaml(config_path)
    
    # Load data source from explicitly specified path
    data_source_path = cfg["data_source"]
    data_source = load_data_source(data_source_path)
    
    # Extract nested configurations with defaults
    experiment = cfg.get("experiment", {})
    output = cfg.get("output", {})
    columns = cfg.get("columns", {})
    execution = cfg.get("execution", {})
    training = cfg.get("training", {})
    context = cfg.get("context", {})
    embeddings = cfg.get("embeddings", {})
    model = cfg.get("model", {})
    graph = cfg.get("graph", {})
    ablations = cfg.get("ablations", {})
    testing = cfg.get("testing", {})
    transfer = cfg.get("transfer", {})  # NEW: cross-geography transfer settings
    lr_finder = cfg.get("lr_finder", {})
    
    # Model sub-configs
    gnn = model.get("gnn", {})
    mlp_encoder = model.get("mlp_encoder", {})
    temporal_mlp = model.get("temporal_mlp", {})
    diffusion = model.get("diffusion", {})
    
    # NEW: extract VFM config from top-level or model-level
    vfm = cfg.get("variational_flow_matching", model.get("variational_flow_matching", {}))
    
    # Resolve output directory
    output_dir = output_override or format_output_dir(
        output.get("base_dir", "./results/{name}"),
        experiment.get("name", "unnamed")
    )
    
    # Determine execution mode
    mode = mode_override or execution.get("mode", "both")
    
    return ExperimentConfig(
        # Identification
        name=experiment.get("name", "unnamed"),
        description=experiment.get("description", ""),
        
        # Data source
        data_source=data_source,
        
        # Output
        output_dir=output_dir,
        
        # Columns
        columns_for_prediction=columns.get("prediction", []),
        columns_for_context=columns.get("context", []),
        continuous_targets=columns.get("continuous_targets", []),
        
        # Execution
        mode=mode,
        resume_training=execution.get("resume_training", False),
        device=execution.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
        
        # Training
        num_epochs=training.get("num_epochs", training.get("num_iterations", 300)),
        learning_rate=training.get("learning_rate", 1e-3),
        num_batches=training.get("num_batches", 10),
        batch_size=training.get("batch_size", 64),
        validation_batch_size=training.get("validation_batch_size", None),
        use_early_stopping=training.get("use_early_stopping", False),
        ema_alpha=training.get("ema_alpha", 0.1),
        use_compile=execution.get("use_compile", False),
        validation_thinning_factor=training.get("validation_thinning_factor", 5),
        optimiser_type=training.get("optimiser", training.get("optimiser", "adamw")),
        muon_lr=float(training.get("muon", {}).get("lr", 0.02)),
        muon_momentum=float(training.get("muon", {}).get("momentum", 0.95)),
        muon_weight_decay=float(training.get("muon", {}).get("weight_decay", 0.0)),
        adamw_lr=float(training.get("adamw", {}).get("lr", training.get("learning_rate", 3e-4))),
        adamw_weight_decay=float(training.get("adamw", {}).get("weight_decay", 1e-2)),
        gradient_accumulation_steps=training.get("gradient_accumulation_steps", 1),
        use_bfloat16=execution.get("use_bfloat16", False),
        
        # Context
        context_size=context.get("size", 30),
        include_latest_context=context.get("include_latest", True),
        min_timestep=context.get("min_timestep", 0),
        
        # Multistep
        multistep_k=execution.get("multistep_k", 1),
        multistep_weights=execution.get("multistep_weights", [1.0]),
        
        # Embeddings
        embedding_size_spatial=embeddings.get("spatial", 8),
        embedding_size_temporal=embeddings.get("temporal", 8),
        
        # Model architecture
        model_type=model.get("type", "mlp"),
        temporal_mlp_hidden=temporal_mlp.get("hidden_layers", [256, 256, 128, 128]),
        flow_matching_hidden=model.get("flow_matching", {}).get("hidden_layers", [128, 128]),
        flow_matching_sigma=model.get("flow_matching", {}).get("sigma", 0.1),
        flow_matching_dropout=model.get("flow_matching", {}).get("dropout", 0.2),
        diffusion_hidden=diffusion.get("hidden_layers", [128, 128]),
        diffusion_time_embed_dim=diffusion.get("time_embed_dim", 128),
        diffusion_num_train_timesteps=diffusion.get("num_train_timesteps", 1000),
        diffusion_beta_start=diffusion.get("beta_start", 1e-4),
        diffusion_beta_end=diffusion.get("beta_end", 2e-2),
        diffusion_dropout=diffusion.get("dropout", 0.2),
        vfm_hidden=vfm.get("hidden_layers", [128, 128]),
        vfm_sigma=vfm.get("sigma", 0.5),
        vfm_dropout=vfm.get("dropout", 0.1),
        vfm_parameterization=vfm.get("parameterization", "mean"),
        vfm_use_skip_connection=vfm.get("use_skip_connection", vfm.get("vfm_skip_connection", False)),
        vfm_debug=vfm.get("debug", vfm.get("vfm_debug", False)),
        vfm_latent_dim=vfm.get("latent_dim", 8),
        vfm_kl_weight=vfm.get("kl_weight", 1.0),
        vfm_kl_warmup_steps=vfm.get("kl_warmup_steps", 2000),
        vfm_encoder_hidden=vfm.get("encoder_hidden", [128, 128]),
        normalising_flow_nsf_hidden=model.get("normalising_flow_nsf", {}).get("hidden_layers", [128, 128]),
        normalising_flow_nsf_num_transforms=model.get("normalising_flow_nsf", {}).get("num_transforms", 3),
        normalising_flow_nsf_sigma=model.get("normalising_flow_nsf", {}).get("sigma", 0.1),
        normalising_flow_nsf_dropout=model.get("normalising_flow_nsf", {}).get("dropout", 0.2),
        normalising_flow_nsf_tail_bound=model.get("normalising_flow_nsf", {}).get("tail_bound", 5.0),
        normalising_flow_maf_hidden=model.get("normalising_flow_maf", {}).get("hidden_layers", [128, 128]),
        normalising_flow_maf_num_transforms=model.get("normalising_flow_maf", {}).get("num_transforms", 3),
        normalising_flow_maf_sigma=model.get("normalising_flow_maf", {}).get("sigma", 0.1),
        normalising_flow_maf_dropout=model.get("normalising_flow_maf", {}).get("dropout", 0.2),
        normalising_flow_cnf_hidden=model.get("normalising_flow_cnf", {}).get("hidden_layers", [128, 128]),
        normalising_flow_cnf_sigma=model.get("normalising_flow_cnf", {}).get("sigma", 0.1),
        normalising_flow_cnf_dropout=model.get("normalising_flow_cnf", {}).get("dropout", 0.2),
        spatial_encoder_type=model.get("spatial_encoder_type", "gnn"),
        gnn_dropout=gnn.get("dropout", 0.1),
        gnn_self_loops=gnn.get("self_loops", True),
        gnn_residual=gnn.get("residual_connections", True),
        gnn_attention_heads=gnn.get("attention_heads", [2, 2]),
        gnn_attention_concat=gnn.get("attention_head_concat", [True, False]),
        mlp_hidden_layers=mlp_encoder.get("hidden_layers", [128, 64, 32]),
        mlp_dropout=mlp_encoder.get("dropout", 0.2),
        
        # Graph
        edge_threshold_spatial=graph.get("edge_threshold_spatial"),
        edge_threshold_temporal=graph.get("edge_threshold_temporal"),
        use_edge_attr=graph.get("use_edge_attr", True),
        edge_attr_dim=graph.get("edge_attr_dim", 1),
        use_k_nearest_features=graph.get("use_k_nearest_features", True),
        use_k_nearest_distance=graph.get("use_k_nearest_distance", True),
        knn_k=graph.get("knn_k", 9),
        
        # Ablations
        spatial_ablation=ablations.get("spatial", False),
        
        # Testing
        autoregressive_windows=testing.get("autoregressive_windows", [180]),
        num_samples=testing.get("num_samples", 100),
        num_steps=testing.get("num_steps", 50),
        window_batch_size=testing.get("window_batch_size", 16),
        testing_checkpoint_source=testing_checkpoint_source_override or testing.get("checkpoint_source", "best"),
        
        # Cross-geography transfer
        transfer_from=transfer.get("scalers_from"),
        load_checkpoints_from=transfer.get("checkpoints_from"),

        # LR finder
        lr_finder=LRFinderConfig(
            start_lr=float(lr_finder.get("start_lr", 1e-7)),
            end_lr=float(lr_finder.get("end_lr", 10.0)),
            num_iter=int(lr_finder.get("num_iter", 100)),
            beta=float(lr_finder.get("beta", 0.98)),
            diverge_th=float(lr_finder.get("diverge_th", 5.0)),
            skip_start=int(lr_finder.get("skip_start", 10)),
            skip_end=int(lr_finder.get("skip_end", 5)),
            output_subdir=lr_finder.get("output_subdir", "lr_finder"),
            csv_filename=lr_finder.get("csv_filename", "lr_finder_curve.csv"),
            plot_filename=lr_finder.get("plot_filename", "lr_finder_curve.png"),
        ),

        # Data augmentation
        poisson_noise_context=training.get("augmentation", {"poisson_noise_context": False}).get("poisson_noise_context", False),
        augmentation_thinning_factor=training.get("augmentation", {}).get("thinning_factor", 5),
        scheduler_type=training.get("scheduler_type", "onecycle"),
        cosine_eta_min=float(training.get("cosine_eta_min", 0.0)),
        cosine_warmup_fraction=float(training.get("cosine_warmup_fraction", 0.05)),
        max_grad_norm=float(training.get("max_grad_norm", 1.0)) if training.get("max_grad_norm") is not None else None,
        gradient_clip_value=float(training.get("gradient_clip_value", 1.0)) if training.get("gradient_clip_value") is not None else None,
    )


def config_to_parameters_dict(cfg: ExperimentConfig) -> Dict[str, Any]:
    """Convert ExperimentConfig to the legacy parameters dictionary format for logging."""
    return {
        # Training parameters
        "device": cfg.device,
        "num_epochs": cfg.num_epochs,
        "context_size": cfg.context_size,
        "min_timestep": cfg.min_timestep,
        "lr": cfg.learning_rate,
        "num_batches": cfg.num_batches,
        "batch_size": cfg.batch_size,
        "validation_batch_size": cfg.validation_batch_size,
        "autoregressive_windows": cfg.autoregressive_windows,
        "use_early_stopping": cfg.use_early_stopping,
        "ema_alpha": cfg.ema_alpha,
        "use_compile": cfg.use_compile,
        "validation_thinning_factor": cfg.validation_thinning_factor,
        "optimiser_type": cfg.optimiser_type,
        "muon_lr": cfg.muon_lr,
        "muon_momentum": cfg.muon_momentum,
        "muon_weight_decay": cfg.muon_weight_decay,
        "adamw_lr": cfg.adamw_lr,
        "adamw_weight_decay": cfg.adamw_weight_decay,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "use_bfloat16": cfg.use_bfloat16,
        "scheduler_type": cfg.scheduler_type,
        "cosine_eta_min": cfg.cosine_eta_min,
        "cosine_warmup_fraction": cfg.cosine_warmup_fraction,
        "max_grad_norm": cfg.max_grad_norm,
        "gradient_clip_value": cfg.gradient_clip_value,
        "multistep_k": cfg.multistep_k,
        "multistep_weights": cfg.multistep_weights,
        
        # Embedding sizes
        "embedding_size_spatial": cfg.embedding_size_spatial,
        "embedding_size_temporal": cfg.embedding_size_temporal,
        
        # Model parameters
        "model_type": cfg.model_type,
        "temporal_module_hidden_dim": cfg.temporal_mlp_hidden,
        "flow_matching_hidden": cfg.flow_matching_hidden,
        "flow_matching_sigma": cfg.flow_matching_sigma,
        "flow_matching_dropout": cfg.flow_matching_dropout,
        "vfm_hidden": cfg.vfm_hidden,
        "vfm_sigma": cfg.vfm_sigma,
        "vfm_dropout": cfg.vfm_dropout,
        "vfm_parameterization": cfg.vfm_parameterization,
        "vfm_use_skip_connection": cfg.vfm_use_skip_connection,
        "vfm_debug": cfg.vfm_debug,
        "normalising_flow_nsf_hidden": cfg.normalising_flow_nsf_hidden,
        "normalising_flow_nsf_num_transforms": cfg.normalising_flow_nsf_num_transforms,
        "normalising_flow_nsf_sigma": cfg.normalising_flow_nsf_sigma,
        "normalising_flow_nsf_dropout": cfg.normalising_flow_nsf_dropout,
        "normalising_flow_nsf_tail_bound": cfg.normalising_flow_nsf_tail_bound,
        "normalising_flow_maf_hidden": cfg.normalising_flow_maf_hidden,
        "normalising_flow_maf_num_transforms": cfg.normalising_flow_maf_num_transforms,
        "normalising_flow_maf_sigma": cfg.normalising_flow_maf_sigma,
        "normalising_flow_maf_dropout": cfg.normalising_flow_maf_dropout,
        "normalising_flow_cnf_hidden": cfg.normalising_flow_cnf_hidden,
        "normalising_flow_cnf_sigma": cfg.normalising_flow_cnf_sigma,
        "normalising_flow_cnf_dropout": cfg.normalising_flow_cnf_dropout,
        "graph_dropout": cfg.gnn_dropout,
        "self_loops": cfg.gnn_self_loops,
        "residual_connections": cfg.gnn_residual,
        "attention_heads": cfg.gnn_attention_heads,
        "attention_head_concat": cfg.gnn_attention_concat,
        "spatial_encoder_type": cfg.spatial_encoder_type,
        "spatial_mlp_hidden_dim": cfg.mlp_hidden_layers,
        
        "spatial_ablation": cfg.spatial_ablation,
        "include_latest_context": cfg.include_latest_context,
        
        # Edge parameters
        "edge_threshold_spatial": cfg.edge_threshold_spatial,
        "edge_threshold_temporal": cfg.edge_threshold_temporal,
        "use_edge_attr": cfg.use_edge_attr,
        "edge_attr_dim": cfg.edge_attr_dim,
        "use_k_nearest_features": cfg.use_k_nearest_features,
        "use_k_nearest_distance": cfg.use_k_nearest_distance,
        "knn_k": cfg.knn_k,
        
        # Testing parameters
        "num_samples": cfg.num_samples,
        "window_batch_size": cfg.window_batch_size,
        "testing_checkpoint_source": cfg.testing_checkpoint_source,
        
        # Transfer settings
        "transfer_from": cfg.transfer_from,
        "load_checkpoints_from": cfg.load_checkpoints_from,
        "mode": cfg.mode,

        # LR finder settings
        "lr_finder_start_lr": cfg.lr_finder.start_lr,
        "lr_finder_end_lr": cfg.lr_finder.end_lr,
        "lr_finder_num_iter": cfg.lr_finder.num_iter,
        "lr_finder_beta": cfg.lr_finder.beta,
        "lr_finder_diverge_th": cfg.lr_finder.diverge_th,
        "lr_finder_skip_start": cfg.lr_finder.skip_start,
        "lr_finder_skip_end": cfg.lr_finder.skip_end,

        # Data augmentation
        "poisson_noise_context": cfg.poisson_noise_context,
        "augmentation_thinning_factor": cfg.augmentation_thinning_factor,
    }


#######################
# Experiment Runner   #
#######################

def run_experiment(cfg: ExperimentConfig) -> None:
    """
    Run a complete experiment based on the configuration.
    
    Supports two modes:
    1. Standard: Train and/or test on the same geography
    2. Cross-geography transfer: Load scalers from a trained experiment and 
       test on a different geography (inference_only mode)
    
    Args:
        cfg: ExperimentConfig object with all settings
    """
    print("#" * 80)
    print(f"EXPERIMENT: {cfg.name}")
    print(f"Description: {cfg.description}")
    print(f"Execution Mode: {cfg.mode}")
    print("#" * 80)
    
    # Check CUDA availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
    
    # Create output directory
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    # Determine continuous prediction columns
    continuous_prediction_columns = [
        c for c in cfg.columns_for_prediction 
        if c in cfg.continuous_targets
    ]
    
    # Validate data files exist
    ds = cfg.data_source
    max_sims = ds.max_sims
    for path_name, path in [("data_csv", ds.data_csv), 
                             ("metadata_csv", ds.metadata_csv),
                             ("static_features_csv", ds.static_features_csv)]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path_name} not found: {path}")
    
    # Validate simulation counts match
    df_data = pd.read_csv(ds.data_csv)
    df_settings = pd.read_csv(ds.metadata_csv)
    n_data_sims = df_data['sim'].nunique()
    n_settings_sims = df_settings['sim'].nunique()
    
    print(f"Sims in data file: {n_data_sims}")
    print(f"Sims in settings file: {n_settings_sims}")
    
    if n_data_sims != n_settings_sims:
        raise ValueError("Number of sims in data file does not match settings file.")
    
    # Import the dataset module
    from dataset import DatesetDirectory
    
    ##############################################
    # CROSS-GEOGRAPHY TRANSFER: Load scalers     #
    ##############################################
    # If transfer_from is set, we load scalers from that experiment
    # and apply them to the new geography (inference_only mode)
    
    temporal_scalers = None
    spatial_scaler = None
    inference_only = False
    
    if cfg.transfer_from is not None:
        print("\n" + "=" * 60)
        print("CROSS-GEOGRAPHY TRANSFER MODE")
        print(f"Loading scalers from: {cfg.transfer_from}")
        print("=" * 60)
        
        # Load the source experiment's data source config
        source_ds_path = cfg.transfer_from
        source_ds = load_data_source(source_ds_path)
        
        # Create source dataset directory to get fitted scalers
        print("Creating source dataset directory to extract scalers...")
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
            fit_scalers=True,  # Fit scalers on source data
            max_sims=max_sims,
        )
        
        # Extract the fitted scalers
        temporal_scalers = source_dataset_directory.temporal_scalers
        spatial_scaler = source_dataset_directory.spatial_scaler
        inference_only = True
        
        print(f"Loaded {len(temporal_scalers)} temporal scalers and spatial scaler from source")
    
    parameters = config_to_parameters_dict(cfg)
    parameters["inference_only"] = inference_only
    
    ##############################################
    # Create dataset directory for this experiment #
    ##############################################
    dataset_directory = DatesetDirectory(
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
        # Cross-geography transfer: use pre-fitted scalers
        temporal_scalers=temporal_scalers,
        spatial_scaler=spatial_scaler,
        fit_scalers=(temporal_scalers is None),  # Only fit if not transferred
        inference_only=inference_only,
        output_dir=cfg.output_dir,
        max_sims=max_sims
    )
    
    # Store simulation splits in parameters for logging
    # parameters = config_to_parameters_dict(cfg)
    parameters["testing_sims"] = dataset_directory.test_sims
    parameters["training_sims"] = dataset_directory.train_sims
    parameters["validation_sims"] = dataset_directory.val_sims
    parameters["n_train_sims"] = len(dataset_directory.train_sims)
    parameters["n_test_sims"] = len(dataset_directory.test_sims)
    parameters["n_val_sims"] = len(dataset_directory.val_sims)
    
    # Build graph data
    print("\nBuilding graph structures...")
    graph_data_temporal = get_graph(
        dataset_directory.xr_static_features_scaled,
        dataset_directory.xr_static_features,
        edge_haversine_threshold=cfg.edge_threshold_spatial,
        use_k_nearest_distance=cfg.use_k_nearest_distance,
        use_k_nearest_features=cfg.use_k_nearest_features,
        knn_k=cfg.knn_k,
        msoa_lookup=dataset_directory.df_geo_metadata['MSOA'],
        save_path=cfg.output_dir
    )
    
    if not cfg.spatial_ablation and cfg.spatial_encoder_type == "gnn":
        graph_data_spatial = get_graph(
            dataset_directory.xr_static_features_scaled,
            dataset_directory.xr_static_features,
            edge_haversine_threshold=cfg.edge_threshold_spatial,
            use_k_nearest_distance=cfg.use_k_nearest_distance,
            use_k_nearest_features=cfg.use_k_nearest_features,
            knn_k=cfg.knn_k,
            msoa_lookup=dataset_directory.df_geo_metadata['MSOA'],
            save_path=cfg.output_dir
        )
    else:
        graph_data_spatial = None
    
    graph_data = [graph_data_temporal, graph_data_spatial]
    graph_data[0] = cache_graph(graph_data[0], cfg.device)
    if graph_data[1] is not None:
        graph_data[1] = cache_graph(graph_data[1], cfg.device)
    
    # Build encoders
    print("\nBuilding model architecture...")
    
    if cfg.spatial_ablation:
        local_profile_encoder = None
    else:
        if cfg.spatial_encoder_type == "gnn":
            local_profile_encoder = GNNEncoder(
                num_features=dataset_directory.num_static_features,
                layer_specs=[
                    (128, cfg.gnn_attention_heads[0], cfg.gnn_attention_concat[0]),
                    (cfg.embedding_size_spatial, cfg.gnn_attention_heads[1], cfg.gnn_attention_concat[1])
                ],
                dropout=cfg.gnn_dropout,
                self_loop=cfg.gnn_self_loops,
                use_edge_attr=cfg.use_edge_attr,
                edge_attr_dim=cfg.edge_attr_dim,
                residual_connection=cfg.gnn_residual
            )
        else:
            local_profile_encoder = MLPEncoder(
                input_dim=dataset_directory.num_static_features,
                hidden_layers=cfg.mlp_hidden_layers,
                output_dim=cfg.embedding_size_spatial,
                dropout=cfg.mlp_dropout
            )
    
    interaction_encoder = GNNEncoder(
        num_features=cfg.context_size * len(cfg.columns_for_context),
        layer_specs=[
            (128, cfg.gnn_attention_heads[0], cfg.gnn_attention_concat[0]),
            (cfg.embedding_size_temporal, cfg.gnn_attention_heads[1], cfg.gnn_attention_concat[1])
        ],
        dropout=cfg.gnn_dropout,
        self_loop=cfg.gnn_self_loops,
        use_edge_attr=cfg.use_edge_attr,
        edge_attr_dim=cfg.edge_attr_dim,
        residual_connection=cfg.gnn_residual
    )
    
    # Calculate input dimension for temporal model
    D = 7  # Day of week encoding
    has_spatial = local_profile_encoder is not None
    
    if has_spatial:
        base_dim = cfg.embedding_size_spatial + cfg.embedding_size_temporal
    else:
        base_dim = cfg.embedding_size_temporal
    
    input_dim = (
        (len(cfg.columns_for_context) if cfg.include_latest_context else 0)
        + base_dim
        + D
    )
    
    print(f"Input dimension to prediction module: {input_dim}")
    
    if cfg.model_type == "flow_matching":
        from flow_matching_model import FlowMatchingModel
        temporal_model = FlowMatchingModel(
            hidden_layers=cfg.flow_matching_hidden,
            input_dim=input_dim,
            output_dim=len(cfg.columns_for_prediction),
            prediction_distribution_types=dataset_directory.prediction_distribution_types,
            scalers=dataset_directory.scalers_for_prediction,
            sigma=cfg.flow_matching_sigma,
            dropout=cfg.flow_matching_dropout
        )
    elif cfg.model_type == "diffusion":
        from diffusion_model import DiffusionModel
        temporal_model = DiffusionModel(
            hidden_layers=cfg.diffusion_hidden,
            input_dim=input_dim,
            output_dim=len(cfg.columns_for_prediction),
            prediction_distribution_types=dataset_directory.prediction_distribution_types,
            scalers=dataset_directory.scalers_for_prediction,
            time_embed_dim=cfg.diffusion_time_embed_dim,
            num_train_timesteps=cfg.diffusion_num_train_timesteps,
            beta_start=cfg.diffusion_beta_start,
            beta_end=cfg.diffusion_beta_end,
            dropout=cfg.diffusion_dropout
        )
    elif cfg.model_type == "variational_flow_matching":
        # Calculate log-space scalers for VFM state evolution
        # Only use training simulations for the fit
        train_sim_indices = [dataset_directory.sim_id_to_idx[sid] for sid in dataset_directory.train_sims]
        
        # Transform the entire data_tensor to log-space for consistency across all context and GNN inputs
        # We use log1p(raw) and then Z-score the result
        print("Transforming dataset to log-space for VFM...")
        raw_train_data = dataset_directory.raw_data_tensor[:, train_sim_indices]
        log_train_data = torch.log1p(raw_train_data)
        # raw_data_tensor shape: [num_cols, num_sims, num_geos, num_timesteps]
        # Get indices of prediction columns in the data tensor
        pred_col_indices = [dataset_directory.columns_with_data.index(col) for col in cfg.columns_for_prediction]
        log_raw_data = torch.log1p(dataset_directory.raw_data_tensor[pred_col_indices][:, train_sim_indices])
        
        # Calculate mean/std across all dimensions except columns
        vfm_log_means = log_raw_data.mean(dim=(1, 2, 3), keepdim=True)
        vfm_log_stds = log_raw_data.std(dim=(1, 2, 3), keepdim=True).clamp(min=1e-6)
        
        # Update dataset_directory.data_tensor to be log-scaled
        # This affects all contexts fetched by ProcessedData
        dataset_directory.data_tensor = (torch.log1p(dataset_directory.raw_data_tensor) - vfm_log_means) / vfm_log_stds
        
        # We also need these for the model's velocity field
        pred_log_means = vfm_log_means.squeeze().tolist()
        pred_log_stds = vfm_log_stds.squeeze().tolist()
        # Ensure they are lists if single output
        if not isinstance(pred_log_means, list): pred_log_means = [pred_log_means]
        if not isinstance(pred_log_stds, list): pred_log_stds = [pred_log_stds]

        from variational_flow_matching_model import VFMFlowMatchingModel
        temporal_model = VFMFlowMatchingModel(
            hidden_layers=cfg.vfm_hidden,
            input_dim=input_dim,
            output_dim=len(cfg.columns_for_prediction),
            prediction_distribution_types=dataset_directory.prediction_distribution_types,
            raw_scalers=dataset_directory.scalers_for_prediction,
            log_means=pred_log_means,
            log_stds=pred_log_stds,
            sigma=cfg.vfm_sigma,
            dropout=cfg.vfm_dropout,
            parameterization=cfg.vfm_parameterization,
            latent_dim=cfg.vfm_latent_dim,
            kl_weight=cfg.vfm_kl_weight,
            kl_warmup_steps=cfg.vfm_kl_warmup_steps,
            encoder_hidden=cfg.vfm_encoder_hidden,
            debug=cfg.vfm_debug
        )
    elif cfg.model_type == "normalising_flow_nsf":
        from normalising_flow_model import NormalisingFlowModel
        temporal_model = NormalisingFlowModel(
            hidden_layers=cfg.normalising_flow_nsf_hidden,
            input_dim=input_dim,
            output_dim=len(cfg.columns_for_prediction),
            prediction_distribution_types=dataset_directory.prediction_distribution_types,
            scalers=dataset_directory.scalers_for_prediction,
            num_transforms=cfg.normalising_flow_nsf_num_transforms,
            sigma=cfg.normalising_flow_nsf_sigma,
            dropout=cfg.normalising_flow_nsf_dropout,
            tail_bound=cfg.normalising_flow_nsf_tail_bound,
        )
    elif cfg.model_type == "normalising_flow_maf":
        from maf_model import MAFFlowModel
        temporal_model = MAFFlowModel(
            hidden_layers=cfg.normalising_flow_maf_hidden,
            input_dim=input_dim,
            output_dim=len(cfg.columns_for_prediction),
            prediction_distribution_types=dataset_directory.prediction_distribution_types,
            scalers=dataset_directory.scalers_for_prediction,
            num_transforms=cfg.normalising_flow_maf_num_transforms,
            sigma=cfg.normalising_flow_maf_sigma,
            dropout=cfg.normalising_flow_maf_dropout
        )
    elif cfg.model_type == "normalising_flow_cnf":
        from cnf_model import CNFFlowModel
        temporal_model = CNFFlowModel(
            hidden_layers=cfg.normalising_flow_cnf_hidden,
            input_dim=input_dim,
            output_dim=len(cfg.columns_for_prediction),
            prediction_distribution_types=dataset_directory.prediction_distribution_types,
            scalers=dataset_directory.scalers_for_prediction,
            sigma=cfg.normalising_flow_cnf_sigma,
            dropout=cfg.normalising_flow_cnf_dropout
        )
    else:
        temporal_model = MLP(
            hidden_layers=cfg.temporal_mlp_hidden,
            input_dim=input_dim,
            num_outputs=len(cfg.columns_for_prediction),
            prediction_distribution_types=dataset_directory.prediction_distribution_types,
            scalers=dataset_directory.scalers_for_prediction
        )

    # Enable multi-step prediction by instantiating k independent identical predictive heads
    if cfg.multistep_k > 1:
        temporal_model = torch.nn.ModuleList([copy.deepcopy(temporal_model) for _ in range(cfg.multistep_k)])
    else:
        # Wrap it in a ModuleList for consistency in training.py
        temporal_model = torch.nn.ModuleList([temporal_model])

    if cfg.mode == "lr_finder":
        run_lr_finder(
            cfg=cfg,
            temporal_model=temporal_model,
            local_profile_encoder=local_profile_encoder,
            interaction_encoder=interaction_encoder,
            graph_data=graph_data,
            dataset_directory=dataset_directory,
            poisson_noise_context=cfg.poisson_noise_context,
            augmentation_thinning_factor=cfg.augmentation_thinning_factor,
        )
        return

    write_model_features(
        output_dir=cfg.output_dir,
        temporal_model=temporal_model,
        local_profile_encoder=local_profile_encoder,
        interaction_encoder=interaction_encoder,
        model_type=cfg.model_type,
    )
    
    # Save configuration
    config_save_path = os.path.join(cfg.output_dir, "config.txt")
    with open(config_save_path, "w") as f:
        f.write(f"# Experiment: {cfg.name}\n")
        f.write(f"# Description: {cfg.description}\n")
        f.write(f"# Data source: {cfg.data_source.name}\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n\n")
        for key, value in parameters.items():
            f.write(f"{key} = {value}\n")
    
    # Also save full YAML config for reproducibility
    full_config_path = os.path.join(cfg.output_dir, "experiment_config.yaml")
    with open(full_config_path, "w") as f:
        yaml.dump({
            "experiment": {"name": cfg.name, "description": cfg.description},
            "data_source": {
                "name": cfg.data_source.name,
                "data_csv": cfg.data_source.data_csv,
                "metadata_csv": cfg.data_source.metadata_csv,
                "static_features_csv": cfg.data_source.static_features_csv,
                "split": cfg.data_source.split,
            },
            "columns": {
                "prediction": cfg.columns_for_prediction,
                "context": cfg.columns_for_context,
                "continuous_targets": cfg.continuous_targets,
            },
            "context": {
                "size": cfg.context_size,
                "min_timestep": cfg.min_timestep,
                "include_latest": cfg.include_latest_context,
            },
            "testing": {
                "autoregressive_windows": cfg.autoregressive_windows,
                "num_samples": cfg.num_samples,
                "num_steps": cfg.num_steps,
                "window_batch_size": cfg.window_batch_size,
                "checkpoint_source": cfg.testing_checkpoint_source,
            },
            "transfer": {
                "scalers_from": cfg.transfer_from,
                "checkpoints_from": cfg.load_checkpoints_from,
            },
            "parameters": parameters,
        }, f, default_flow_style=False)
    
    ##################
    # MODEL TRAINING #
    ##################
    if cfg.mode in ["train", "both"]:
        # Cannot train in inference_only mode
        if inference_only:
            print("\n[WARNING] Skipping training - inference_only mode is enabled (transfer_from is set)")
        else:
            print("############")
            print("# TRAINING #")
            print("############")
            
            # Check for resume checkpoint
            resume_path = None
            if cfg.resume_training:
                latest_ckpt = os.path.join(cfg.output_dir, "trained_models", "checkpoint_latest.pth")
                if os.path.exists(latest_ckpt):
                    resume_path = latest_ckpt
                    print(f"Resuming from checkpoint: {latest_ckpt}")
            
            train(
                temporal_model=temporal_model,
                local_profile_encoder=local_profile_encoder,
                interaction_encoder=interaction_encoder,
                graph_data=graph_data,
                dataset_directory=dataset_directory,
                num_epochs=cfg.num_epochs,
                lr=cfg.learning_rate,
                num_batches=cfg.num_batches,
                batch_size=cfg.batch_size,
                context_size=cfg.context_size,
                device=cfg.device,
                output_folder=cfg.output_dir,
                lpe_ablation=cfg.spatial_ablation,
                include_latest_context=cfg.include_latest_context,
                spatial_encoder_type=cfg.spatial_encoder_type,
                resume_from_checkpoint=resume_path,
                model_type=cfg.model_type,
                ema_alpha=cfg.ema_alpha,
                use_compile=cfg.use_compile,
                multistep_k=cfg.multistep_k,
                multistep_weights=cfg.multistep_weights,
                validation_thinning_factor=cfg.validation_thinning_factor,
                validation_batch_size=cfg.validation_batch_size,
                poisson_noise_context=cfg.poisson_noise_context,
                augmentation_thinning_factor=cfg.augmentation_thinning_factor,
                optimiser_type=cfg.optimiser_type,
                muon_lr=cfg.muon_lr,
                muon_momentum=cfg.muon_momentum,
                muon_weight_decay=cfg.muon_weight_decay,
                adamw_lr=cfg.adamw_lr,
                adamw_weight_decay=cfg.adamw_weight_decay,
                gradient_accumulation_steps=cfg.gradient_accumulation_steps,
                use_bfloat16=cfg.use_bfloat16,
                scheduler_type=cfg.scheduler_type,
                cosine_eta_min=cfg.cosine_eta_min,
                cosine_warmup_fraction=cfg.cosine_warmup_fraction,
                max_grad_norm=cfg.max_grad_norm,
                gradient_clip_value=cfg.gradient_clip_value,
            )
    
    ##################
    # MODEL TESTING  #
    ##################
    if cfg.mode in ["test", "both"]:
        print("###########")
        print("# TESTING #")
        print("###########")
        
        # Determine where to load checkpoints from
        checkpoint_dir = cfg.load_checkpoints_from or cfg.output_dir
        print(f"Loading models from: {checkpoint_dir}")

        checkpoint_source = cfg.testing_checkpoint_source.lower()
        if checkpoint_source not in {"best", "final"}:
            raise ValueError(f"Invalid testing checkpoint source: {cfg.testing_checkpoint_source}. Expected 'best' or 'final'.")

        print(f"Loading {checkpoint_source} models from checkpoints...")

        checkpoint_prefix = f"{checkpoint_source}_"

        checkpoint_path = os.path.join(checkpoint_dir, "trained_models", f"{checkpoint_prefix}temporal_model.pth")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Temporal model checkpoint not found: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location="cpu")
        temporal_model[0].load_state_dict(state)

        if not cfg.spatial_ablation:
            checkpoint_path = os.path.join(checkpoint_dir, "trained_models", f"{checkpoint_prefix}location_specific_encoder.pth")
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Spatial encoder checkpoint not found: {checkpoint_path}")
            state = torch.load(checkpoint_path, map_location="cpu")
            local_profile_encoder.load_state_dict(state)

        checkpoint_path = os.path.join(checkpoint_dir, "trained_models", f"{checkpoint_prefix}interaction_encoder.pth")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Interaction encoder checkpoint not found: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location="cpu")
        interaction_encoder.load_state_dict(state)
        
        # Move to device
        temporal_model.to(cfg.device)
        interaction_encoder.to(cfg.device)
        if local_profile_encoder is not None:
            local_profile_encoder.to(cfg.device)
        
        # Run testing for each autoregressive window
        test_start = time.time()
        for autoregressive_window in cfg.autoregressive_windows:
            print(f"\nEvaluating with autoregressive window: {autoregressive_window}")
            test(
                temporal_model=temporal_model[0],
                local_profile_encoder=local_profile_encoder,
                interaction_encoder=interaction_encoder,
                graph_data=graph_data,
                dataset_directory=dataset_directory,
                context_size=cfg.context_size,
                autoregressive_window_size=autoregressive_window,
                device=cfg.device,
                output_folder=cfg.output_dir,
                num_samples=cfg.num_samples,
                spatial_ablation=cfg.spatial_ablation,
                include_latest_context=cfg.include_latest_context,
                spatial_encoder_type=cfg.spatial_encoder_type,
                num_steps=cfg.num_steps,
                window_batch_size=cfg.window_batch_size
            )
        
        test_duration = time.time() - test_start
        print(f"\nTesting completed in {test_duration/60:.2f} minutes")
    
    print("\n" + "#" * 80)
    print(f"EXPERIMENT COMPLETE: {cfg.name}")
    print(f"Results saved to: {cfg.output_dir}")
    print("#" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Run GENIE experiments using YAML configuration files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Configuration source (mutually exclusive)
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument(
        "--config", "-c",
        type=str,
        help="Path to a single experiment YAML configuration file"
    )
    config_group.add_argument(
        "--config-dir", "-d",
        type=str,
        help="Path to directory containing multiple experiment YAML files"
    )
    
    # Overrides
    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["train", "test", "both", "lr_finder"],
        help="Override execution mode (train, test, both, or lr_finder)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        help="Override output directory"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        help="Override device (cuda or cpu)"
    )
    parser.add_argument(
        "--test-checkpoint-source",
        type=str,
        choices=["best", "final"],
        help="Override which saved checkpoint set to use for testing"
    )
    
    args = parser.parse_args()
    
    # Collect configuration files to process
    config_files = []
    
    if args.config:
        config_files.append(args.config)
    else:
        # Load all YAML files from directory
        config_dir = Path(args.config_dir)
        if not config_dir.exists():
            print(f"Error: Config directory not found: {config_dir}")
            sys.exit(1)
        
        config_files = sorted(config_dir.glob("*.yaml")) + sorted(config_dir.glob("*.yml"))
        
        if not config_files:
            print(f"Error: No YAML files found in {config_dir}")
            sys.exit(1)
        
        print(f"Found {len(config_files)} experiment configurations:")
        for cf in config_files:
            print(f"  - {cf.name}")
        print()
    
    # Run each experiment
    for config_file in config_files:
        try:
            print(f"\nLoading configuration: {config_file}")
            cfg = load_experiment_config(
                str(config_file),
                mode_override=args.mode,
                output_override=args.output_dir,
                testing_checkpoint_source_override=args.test_checkpoint_source
            )
            
            # Apply device override if specified
            if args.device:
                cfg.device = args.device
            
            run_experiment(cfg)
            
        except Exception as e:
            print(f"\nError running experiment from {config_file}:")
            print(f"  {type(e).__name__}: {e}")
            if len(config_files) > 1:
                print("  Continuing with next experiment...")
            else:
                raise


if __name__ == "__main__":
    main()