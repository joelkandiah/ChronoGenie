# GENIE: Generative Neural Inference for Epidemics

A spatio-temporal machine learning framework for probabilistic epidemic forecasting at the MSOA (Middle Layer Super Output Area) level.

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Project Structure](#project-structure)
5. [Configuration Guide](#configuration-guide)
   - [Data Source Configuration](#data-source-configuration)
   - [Experiment Configuration](#experiment-configuration)
6. [Running Experiments](#running-experiments)
7. [Understanding Outputs](#understanding-outputs)
8. [Advanced Experimental Setups](#advanced-experimental-setups)
   - [Geography Transfer](#i-geography-transfer)
   - [Spatial Ablation](#ii-spatial-ablation)
   - [Unconnected GNNs](#iii-unconnected-gnns)
   - [Predicting R_t](#iv-predicting-rt)

---

## Overview

GENIE predicts infectious disease burden eg: hospitalisations, deaths, R_t and infections across geographic regions (MSOAs) using:

- **Graph Neural Networks (GATv2)** to capture dependencies between MSOAs
- **Probabilistic outputs** via Negative Binomial (count data) and LogNormal (continuous data) distributions
- **Autoregressive forecasting** with Monte Carlo uncertainty quantification
- **Proper scoring rules** (CRPS, Energy Score, Variogram) for evaluation

### Key Features

- YAML-based experiment configuration
- Multi-Token Prediction (MTP) for stable independent forecasting horizons
- Support for mixed distribution types (counts + continuous)
- Cross-geography transfer learning
- Checkpoint resumption for long training runs
- Evaluation metrics

---

## Installation

### Using the Environment File

The project includes a `environment.yaml` file for setup:

```bash
# Create the conda environment
conda env create -f environment.yaml

# Activate the environment
conda activate GENIE
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch_geometric; print(f'PyG: {torch_geometric.__version__}')"
```

---
## Quick Start

### 1. Prepare Your Data

Ensure you have three CSV files:
- **Simulation data**: Time-series with columns `sim`, `date`, `MSOA`, and target variables
- **Simulation metadata**: Settings for each simulation
- **Static features**: MSOA-level geographic/demographic features
#### Preparing Static Features for a New Geography

If you're working with a new geographic region, use `data_processing.py` to prepare the static features:

```python
import pandas as pd
from data_processing import preprocess_feature_data

# Define the MSOAs in your region of interest
my_region_msoas = ['E02001234', 'E02001235', 'E02001236', ...]

# Load raw MSOA features (from a master file containing all MSOAs)
all_features = pd.read_csv("DATA/msoa_feature_data.csv")

# Filter to your region
region_features = all_features[all_features['msoa'].isin(my_region_msoas)]
region_features.to_csv("DATA/static_features_MY_REGION.csv", index=False)

# Preprocess: encode categorical columns, drop identifiers
processed = preprocess_feature_data(
    pd.read_csv("DATA/static_features_MY_REGION.csv"),
    dont_drop=["msoa", "lat_long"]  # Keep these columns
)
processed.to_csv("DATA/processed_msoa_features_MY_REGION.csv", index=False)

# Ensure columns match the training data format
training_data = pd.read_csv("DATA/static_features_12_10_2025.csv")
columns_to_keep = ['msoa'] + [col for col in training_data.columns if col != 'msoa']
final_features = processed[columns_to_keep]
final_features.to_csv("DATA/static_features_MY_REGION_filtered.csv", index=False)
```

### 2. Create a Data Source Configuration

```yaml
# configs/data/my_data.yaml
name: "MY_DATA"
description: "My simulation dataset"

data_csv: "DATA/simulation_data.csv"
metadata_csv: "DATA/simulation_metadata.csv"
static_features_csv: "DATA/msoa_features.csv"

split: [700, 150, 150]  # train, test, val counts

column_with_date: "date"
column_with_geography: "MSOA"
```

### 3. Create an Experiment Configuration

```yaml
# configs/experiments/my_experiment.yaml
experiment:
  name: "MY_EXPERIMENT"
  description: "My first GENIE experiment"

data_source: "configs/data/my_data.yaml"

output:
  base_dir: "results/{name}"

columns:
  prediction: ["daily_hospitalised", "deaths"]
  context: ["daily_hospitalised", "deaths"]
  continuous_targets: []

execution:
  mode: "both"
  device: "cuda"

training:
  num_iterations: 300
  learning_rate: 1.0e-3
  batch_size: 64

context:
  size: 30
  min_timestep: 0

# See full configuration reference below for all options
```

### 4. Run the Experiment

```bash
python run_experiment.py --config configs/experiments/my_experiment.yaml
```

---

## Project Structure

```
genie/
├── run_experiment.py            # Main entry point - coordinates everything
├── dataset.py                   # Data loading and preprocessing
├── graph_construction.py        # Graph building and static features
├── models.py                    # Neural network architectures (MLP, GNN)
├── training.py                  # Training loop with coordination
├── testing.py                   # Testing orchestration
├── testing_sliding_window.py    # Autoregressive prediction engine
├── flow_matching_model.py       # Generative flow matching components (Velocity-based)
├── variational_flow_matching_model.py # Variational Flow Matching (Distribution-based)
├── normalising_flow_model.py    # Generative normalising flow (NSF - Neural Spline Flows)
├── maf_model.py                 # Masked Autoencoder Flow (MAF) model
├── cnf_model.py                 # Continuous Normalising Flow (CNF) model
├── distribution_construction.py # Loss functions and sampling
├── proper_scoring.py            # Evaluation metrics (CRPS, Energy Score)
├── proper_scoring_torch.py      # PyTorch implementations of scoring rules
├── train_test_helpers.py        # Graph caching utilities
├── data_processing.py           # Static feature preprocessing for new geographies
├── environment.yaml             # Conda environment specification
├── configs/
│   ├── data/                    # Data source configurations
│   │   ├── 1K_june.yaml
│   │   └── 1K_june_DARLINGTON.yaml
│   └── experiments/             # Experiment configurations
│       ├── gnn_final_1k.yaml
│       └── gnn_final_1k_DARLINGTON.yaml
└── Plotting/
    ├── Plot_single_simulations/  # Plotting scripts for individual simulation results
    │   ├── plot_mlp_results.py
    │   ├── plot_normalising_flow_results.py
    │   ├── plot_variational_flow_matching_results.py
    │   └── ...
    └── Plot_compare_models/      # Plotting scripts for model comparison
        └── ...
```

---

## Configuration Guide

GENIE uses a two-level YAML configuration system:
1. **Data Source configs** - Define paths to data files
2. **Experiment configs** - Define model architecture and training settings

### Data Source Configuration

Data sources define where your input files are located and how to split the data.

```yaml
# configs/data/1K_june.yaml

# Identifier
name: "1K_DATA"
description: "1000 simulation dataset"

# Required file paths (relative to project root or absolute)
data_csv: "DATA/simulation_data.csv"           # Time-series data
metadata_csv: "DATA/simulation_settings.csv"   # Simulation metadata
static_features_csv: "DATA/msoa_features.csv"  # Geographic features

# Train/Test/Validation split
# Values are treated as proportions and scaled to match dataset size
# Examples: [700, 150, 150] or [0.7, 0.15, 0.15] are equivalent
split: [700, 150, 150]

# Column naming conventions (these are the defaults)
column_with_date: "date"
column_with_geography: "MSOA"
```

#### Required CSV Formats

**data_csv** (time-series simulation data):
```csv
sim,date,MSOA,daily_hospitalised,deaths,R_t,...
0,2020-03-01,E02001234,5,0,1.2
0,2020-03-01,E02001235,3,1,1.1
0,2020-03-02,E02001234,7,1,1.3
...
```

**metadata_csv** (simulation settings):
```csv
sim,parameter1,parameter2
0,0.5,100
1,0.6,150
...
```

**static_features_csv** (MSOA-level features):
```csv
msoa,lat_long,iomd_centile,population_density,num_residents_in_msoa,...
E02001234,"[53.4, -2.3]",45,1234.5,8500,...
E02001235,"[53.5, -2.2]",67,2345.6,9200,...
```

### Experiment Configuration

Full reference for experiment YAML files:

```yaml
#######################
# EXPERIMENT IDENTITY #
#######################
experiment:
  name: "GNN_FINAL_1K"                                    # Used in output path
  description: "GNN model predicting hospitalisations"    # For documentation

################
# DATA SOURCE  #
################
data_source: "configs/data/1K_june.yaml"  # Path to data config

##########
# OUTPUT #
##########
output:
  # Supports placeholders: {name}, {timestamp}
  base_dir: "RESULTS/{name}"

###########
# COLUMNS #
###########
columns:
  # What to predict (output variables)
  prediction: ["daily_hospitalised", "deaths"]
  
  # What to use as context features (input variables)
  context: ["daily_hospitalised", "deaths"]
  
  # Which prediction columns are continuous (use LogNormal distribution)
  # All others use Negative Binomial for count data
  continuous_targets: []  # e.g., ["R_t"] for R_t prediction

#############
# EXECUTION #
#############
execution:
  mode: "both"           # "train", "test", or "both"
  resume_training: false # Resume from checkpoint_latest.pth if available
  device: "cuda"         # "cuda" or "cpu"
  
  # Multi-Token Prediction parameters
  multistep_k: 3         # Number of independent prediction horizons to target
  multistep_weights: [1.0, 0.5, 0.25] # Decay scheduling weighting for sequential heads

############
# TRAINING #
############
training:
  num_iterations: 300      # Total training iterations
  learning_rate: 1.0e-3    # Initial learning rate (uses OneCycleLR)
  num_batches: 10          # Batches per iteration
  batch_size: 64           # Samples per batch
  use_early_stopping: false

###########
# CONTEXT #
###########
context:
  size: 30              # Number of past timesteps as context window
  include_latest: true  # Include t-1 value as explicit feature
  min_timestep: 0       # Skip first N timesteps (e.g., 30 for R_t)

##############
# EMBEDDINGS #
##############
embeddings:
  spatial: 8   # Output dimension of spatial encoder
  temporal: 8  # Output dimension of temporal encoder

#########
# MODEL #
#########
model:
  type: "mlp"  # "mlp" (default), "flow_matching", "variational_flow_matching", or "normalising_flow"
  
  # Settings for type: "mlp"
  temporal_mlp:
    hidden_layers: [256, 256, 128, 128]  # MLP hidden layer sizes
  
  # Settings for type: "flow_matching"
  flow_matching:
    sigma: 0.1                 # Noise level for the flow matcher
    dropout: 0.1               # Dropout probability for velocity field
  
  # Settings for type: "variational_flow_matching"
  variational_flow_matching:
    hidden_layers: [128, 128]  # MLP hidden layer sizes
    sigma: 0.5                 # Noise level (higher is better for VFM, e.g., 0.5-1.0)
    dropout: 0.1               # Dropout probability
    parameterization: "mean"   # "mean" (recommended/direct) or "natural" (mu, var)
  
  # Settings for type: "normalising_flow"
  normalising_flow:
    hidden_layers: [128, 128]  # MLP hidden layer sizes for coupling networks
    num_transforms: 3          # Number of coupling layers (NSF)
    sigma: 0.1                 # Noise level for dequantisation
    dropout: 0.1               # Dropout probability (reserved)
  
  spatial_encoder_type: "gnn"  # "gnn" or "mlp"
  
  # GNN-specific settings (used when spatial_encoder_type: "gnn")
  gnn:
    dropout: 0.1
    self_loops: true
    residual_connections: true
    attention_heads: [2, 2]           # Heads per GATv2 layer
    attention_head_concat: [true, false]  # Concatenate or average heads

  # MLP encoder settings (used when spatial_encoder_type: "mlp")
  mlp_encoder:
    hidden_layers: [128, 64, 32]
    dropout: 0.2

#########
# GRAPH #
#########
graph:
  # Distance threshold for edge creation (null = don't use)
  edge_threshold_spatial: null
  edge_threshold_temporal: null
  
  # Edge attributes (distance-weighted attention)
  use_edge_attr: true
  edge_attr_dim: 1
  
  # K-nearest neighbours settings
  use_k_nearest_features: true   # k-NN in static feature space (Euclidean)
  use_k_nearest_distance: true   # k-NN by geographic distance (Haversine)
  knn_k: 9                       # Number of neighbours

############
# ABLATIONS #
############
ablations:
  spatial: false  # If true, disable spatial encoder entirely

###########
# TESTING #
###########
testing:
  autoregressive_windows: [180]  # Prediction horizon (use [180])
  num_samples: 100               # Monte Carlo samples for uncertainty
  num_steps: 50                  # ODE solver steps in sampling (Flow Matching)

############
# TRANSFER #
############
# OPTIONAL: For cross-geography inference (see Advanced section)
transfer:
  scalers_from: null       # Path to source data config
  checkpoints_from: null   # Path to trained model directory
```

---

## Running Experiments

### Basic Commands

```bash
# Train and test (default)
python run_experiment.py --config configs/experiments/gnn_final_1k.yaml

# Train only
python run_experiment.py --config configs/experiments/gnn_final_1k.yaml --mode train

# Test only (requires trained model)
python run_experiment.py --config configs/experiments/gnn_final_1k.yaml --mode test

# Override output directory
python run_experiment.py --config configs/experiments/gnn_final_1k.yaml --output-dir /custom/path

# Force CPU execution
python run_experiment.py --config configs/experiments/gnn_final_1k.yaml --device cpu

# Run all experiments in a directory
python run_experiment.py --config-dir configs/experiments/
```

### Resume Training

If training was interrupted, set `resume_training: true` in your config and run the same command:

```yaml
execution:
  resume_training: true
```

The system will automatically load `checkpoint_latest.pth` and continue from where it stopped.

---

## Understanding Outputs

### Directory Structure

After running an experiment, outputs are organised as follows:

```
RESULTS/GNN_FINAL_1K/
├── config.txt                     # Human-readable parameters
├── experiment_config.yaml         # Full config for reproducibility
├── geo_metadata.csv               # GID to MSOA mapping
├── graph_msoa_pairs.csv           # Graph edges as MSOA pairs
├── knn_neighbours_features.gz     # k-NN analysis data
├── trained_models/
│   ├── checkpoint_latest.pth      # Resumable checkpoint (full state)
│   ├── best_temporal_model.pth    # Best MLP weights
│   ├── best_location_specific_encoder.pth  # Best spatial encoder
│   ├── best_interaction_encoder.pth        # Best temporal encoder
│   ├── loss_plot.png              # Training/validation curves
│   └── nll_losses.csv             # Per-iteration loss log
└── TESTING/
    ├── SIM_1/
    │   ├── embeddings/
    │   │   ├── 0/
    │   │   │   ├── spatial_embeddings.feather
    │   │   │   └── temporal_embeddings.feather
    │   │   ├── 1/
    │   │   │   └── ...
    │   │   └── 179/
    │   │       └── ...
    │   ├── predictions/
    │   │   ├── 0/
    │   │   │   ├── predictions.feather
    │   │   │   ├── sample_spaghetti.feather
    │   │   │   └── spatial_scores.feather
    │   │   ├── 1/
    │   │   │   └── ...
    │   │   └── 179/
    │   │       └── ...
    │   └── window_times.feather
    ├── SIM_2/
    │   └── ...
    └── ...
```

### Output File Contents

#### Embeddings

**spatial_embeddings.feather** - Learned spatial representations per MSOA at each timestep:

| Column | Type | Description |
|--------|------|-------------|
| `timestep` | int32 | Absolute timestep |
| `msoa` | int32 | MSOA index (gid) |
| `emb_0`, `emb_1`, ... | float32 | Spatial embedding dimensions (mean across Monte Carlo samples) |

**temporal_embeddings.feather** - Learned temporal/context representations per MSOA:

| Column | Type | Description |
|--------|------|-------------|
| `timestep` | int32 | Absolute timestep |
| `msoa` | int32 | MSOA index (gid) |
| `emb_0`, `emb_1`, ... | float32 | Temporal embedding dimensions (mean across Monte Carlo samples) |

#### Predictions

**predictions.feather** - Probabilistic forecasts with prediction intervals:

| Column | Type | Description |
|--------|------|-------------|
| `timestep` | int32 | Absolute timestep |
| `msoa` | int32 | MSOA index (gid) |
| `is_window_start` | int8 | 1 if this is the first timestep of the prediction window, 0 otherwise |
| `{burden}_unscaled` | float32 | Median (50th percentile) prediction for each burden |
| `{burden}_unscaled_gt` | float32 | Ground truth value for each burden |
| `{burden}_lower_95` | float32 | 2.5th percentile (95% CI lower bound) |
| `{burden}_upper_95` | float32 | 97.5th percentile (95% CI upper bound) |
| `{burden}_lower_50` | float32 | 25th percentile (50% CI lower bound) |
| `{burden}_upper_50` | float32 | 75th percentile (50% CI upper bound) |
| `{burden}_IS_95` | float32 | Interval Score for 95% prediction interval |
| `{burden}_CRPS` | float32 | Continuous Ranked Probability Score |

*Note: `{burden}` is replaced with each prediction column name, e.g., `daily_hospitalised_unscaled`, `deaths_lower_95`*

**sample_spaghetti.feather** - Individual Monte Carlo samples for detailed uncertainty analysis:

| Column | Type | Description |
|--------|------|-------------|
| `window_start` | int32 | Starting timestep of prediction window |
| `timestep` | int32 | Absolute timestep |
| `window_step` | int32 | Step within prediction window (0, 1, 2, ...) |
| `msoa` | int32 | MSOA index (gid) |
| `msoa_name` | string | MSOA code (e.g., "E02001234") |
| `sample_idx` | int32 | Monte Carlo sample index (0 to num_samples-1) |
| `burden_type` | string | Name of prediction target (e.g., "daily_hospitalised") |
| `value` | float32 | Sampled prediction value (unscaled) |
| `mu_pred` | float32 | Raw μ parameter from model output |
| `var_pred` | float32 | Raw variance/dispersion parameter from model output |

*This file enables "spaghetti plots" showing all Monte Carlo trajectories and analysis of the predicted distribution parameters.*

**spatial_scores.feather** - Multivariate spatial evaluation metrics per timestep:

| Column | Type | Description |
|--------|------|-------------|
| `timestep` | int32 | Absolute timestep |
| `burden` | string | Name of prediction target |
| `energy_score` | float32 | Energy Score (multivariate probabilistic accuracy across all MSOAs) |
| `variogram_score_p05` | float32 | Variogram Score with p=0.5 (spatial correlation structure) |

#### Timing

**window_times.feather** - Computation time for each prediction window:

| Column | Type | Description |
|--------|------|-------------|
| `window_times_seconds` | float64 | Time in seconds to compute each window |

## Advanced Experimental Setups

### i) Geography Transfer

Geography transfer allows you to apply a model trained on one geographic region to a completely different region without retraining.

#### How It Works

1. **Scalers from source**: The temporal and spatial scalers fitted on the training geography are reused to ensure consistent data transformation
2. **Model from source**: The trained neural network weights are loaded
3. **Data from target**: The new geography's data is processed using the source scalers
4. **Testing only**: Training is automatically skipped

#### Configuration

**Step 1**: Create a data source config for the target geography:

```yaml
# configs/data/1K_june_DARLINGTON.yaml
name: "1K_DATA_DARLINGTON"
description: "Darlington geography for transfer testing"

data_csv: "DATA/data_20251128_Darlington.csv"
metadata_csv: "DATA/settings_20251128_Darlington_sims.csv"
static_features_csv: "DATA/static_features_DARLINGTON_filtered.csv"

# IMPORTANT: Set split to [0, 0, 0] - all data goes to test set
split: [0, 0, 0]

column_with_date: "date"
column_with_geography: "MSOA"
```

**Step 2**: Create a transfer experiment config:

```yaml
# configs/experiments/gnn_final_1k_DARLINGTON.yaml
experiment:
  name: "GNN_FINAL_1K_DARLINGTON"
  description: "Transfer model to Darlington geography"

# Target geography data
data_source: "configs/data/1K_june_DARLINGTON.yaml"

output:
  base_dir: "RESULTS/{name}"

columns:
  # MUST match the trained model exactly
  prediction: ["daily_hospitalised", "deaths"]
  context: ["daily_hospitalised", "deaths"]
  continuous_targets: []

execution:
  mode: "test"  # Only testing - training is skipped automatically
  device: "cuda"

#############################
# TRANSFER CONFIGURATION    #
#############################
transfer:
  # Path to the SOURCE data config (used to recreate fitted scalers)
  scalers_from: "configs/data/1K_june.yaml"
  
  # Path to the trained model directory
  checkpoints_from: "RESULTS/GNN_FINAL_1K"

# Model architecture MUST match the trained model exactly
model:
  spatial_encoder_type: "gnn"
  temporal_mlp:
    hidden_layers: [256, 256, 128, 128]
  gnn:
    dropout: 0.1
    self_loops: true
    residual_connections: true
    attention_heads: [2, 2]
    attention_head_concat: [true, false]

# ... rest of config must match training config
```

**Step 3**: Run the transfer experiment:

```bash
python run_experiment.py --config configs/experiments/gnn_final_1k_DARLINGTON.yaml
```

### ii) Spatial Ablation

Spatial ablation disables the spatial encoder entirely, creating a temporal-only baseline model.

#### Configuration

```yaml
# configs/experiments/gnn_spatial_ablation.yaml
experiment:
  name: "GNN_SPATIAL_ABLATION"
  description: "Temporal-only baseline (no spatial encoder)"

# ... standard config ...

ablations:
  spatial: true  # This disables the spatial encoder

# Note: The graph is still constructed for the temporal encoder,
# but no local profile embeddings are used
```

#### What Changes
- The spatial encoder (`local_profile_encoder`) is set to `None`
- Input dimension to the prediction MLP decreases by `embedding_size_spatial`
- Only temporal context embeddings are used for prediction

### iii) Unconnected GNNs

To test whether the graph structure matters, you can create graphs with no edges between MSOAs (only self-loops remain if enabled).

#### Configuration

```yaml
# configs/experiments/gnn_unconnected.yaml
experiment:
  name: "GNN_UNCONNECTED"
  description: "GNN with no inter-MSOA connections"

# ... standard config ...

graph:
  # Disable all edge creation methods
  use_k_nearest_features: false
  use_k_nearest_distance: false
  edge_threshold_spatial: null
  
  # Keep self-loops so each node can still attend to itself
  # (This is controlled in model.gnn.self_loops)

model:
  gnn:
    self_loops: true  # Each node attends only to itself
```

#### Interpretation
- If performance degrades significantly, the spatial graph structure provides valuable information
- If performance is similar, the model may be learning primarily from temporal patterns

### iv) Predicting R_t

R_t (effective reproduction number) is a continuous positive value requiring different handling than count data.

#### Key Considerations
1. **Distribution**: Use LogNormal instead of Negative Binomial
2. **min_timestep**: R_t estimation requires historical data, so skip early timesteps
3. **Context**: Include R_t in both prediction and context columns

#### Configuration

```yaml
# configs/experiments/gnn_with_rt.yaml
experiment:
  name: "GNN_WITH_RT"
  description: "Predict hospitalisations, deaths, and R_t"

columns:
  # Include R_t in predictions
  prediction: ["daily_hospitalised", "deaths", "R_t"]
  
  # Include R_t in context for autoregressive feedback
  context: ["daily_hospitalised", "deaths", "R_t"]
  
  # Mark R_t as continuous (uses LogNormal distribution)
  continuous_targets: ["R_t"]

context:
  size: 30
  include_latest: true
  
  # CRITICAL: Skip first 30 days
  # R_t estimation algorithms need sufficient historical data
  # Zero-padded context at early timesteps would be meaningless for R_t
  min_timestep: 30
```

#### How min_timestep Works

| Scenario | min_timestep | Training Range | Dataset Size |
|----------|-------------|----------------|--------------|
| Count data only | 0 | t=0 to t=179 | num_sims × 180 |
| With R_t | 30 | t=30 to t=179 | num_sims × 150 |

The context window still uses zero-padding if `current_t - context_size < 0`, but prediction starts from `min_timestep`.
