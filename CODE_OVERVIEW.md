# GENIE Codebase Overview

This document provides a high-level overview of each Python file in the GENIE project, describing what each script and function does. The actual code contains detailed inline comments for implementation specifics.

---

## Table of Contents

1. [run_experiment.py](#run_experimentpy) - Main Entry Point
2. [dataset.py](#datasetpy) - Data Loading & Preprocessing
3. [graph_construction.py](#graph_constructionpy) - Graph Building
4. [models.py](#modelspy) - Neural Network Architectures
5. [training.py](#trainingpy) - Training Loop
6. [testing.py](#testingpy) - Testing Coordination
7. [testing_sliding_window.py](#testing_sliding_windowpy) - Autoregressive Inference
8. [flow_matching_model.py](#flow_matching_modelpy) - Generative Flow Matching (Velocity-based)
9. [variational_flow_matching_model.py](#variational_flow_matching_modelpy) - Variational Flow Matching (Distribution-based)
10. [normalising_flow_model.py](#normalising_flow_modelpy) - Generative Normalising Flows (NSF)
11. [maf_model.py](#maf_modelpy) - Masked Autoencoder Flow (MAF)
12. [cnf_model.py](#cnf_modelpy) - Continuous Normalising Flow (CNF)
13. [distribution_construction.py](#distribution_constructionpy) - Loss Functions & Sampling
14. [proper_scoring.py](#proper_scoringpy) - Evaluation Metrics
15. [proper_scoring_torch.py](#proper_scoring_torchpy) - PyTorch Evaluation Metrics
16. [train_test_helpers.py](#train_test_helperspy) - Utility Functions
17. [data_processing.py](#data_processingpy) - Static Feature Preparation
18. [File Dependency Graph](#file-dependency-graph)

---

## run_experiment.py

**Purpose**: Primary script that coordinates the experiment pipeline from configuration loading to training and testing.

### Data Classes

| Class | Description |
|-------|-------------|
| `DataSourceConfig` | Dataclass holding paths to data files, train, validation and testing split ratios, and column naming conventions |
| `ExperimentConfig` | Dataclass containing experiment configuration including model architecture details, training parameters|

### Functions

| Function | Description |
|----------|-------------|
| `load_yaml(path)` | Load and parse a YAML file, returning a dictionary |
| `load_data_source(path)` | Load a data source configuration from a YAML file path, returning a `DataSourceConfig` object |
| `format_output_dir(template, name)` | Format an output directory string, replacing `{name}` and `{timestamp}` placeholders |
| `load_experiment_config(config_path, mode_override, output_override)` | Load a complete experiment configuration from YAML, applying any CLI overrides. Returns an `ExperimentConfig` object |
| `config_to_parameters_dict(cfg)` | Convert an `ExperimentConfig` object to a flat dictionary for logging purposes |
| `run_experiment(cfg)` | **Main function**: loads data, builds models, runs training and/or testing based on configuration |
| `main()` | CLI argument parsing and experiment execution entry point |

### Key Responsibilities

- Parse command-line arguments (`--config`, `--mode`, `--device`, etc.)
- Load and validate YAML configurations
- Handle cross-geography transfer by loading source scalers
- Instantiate dataset, graph, and model components
- Construct MTP (Multi-Token Prediction) `nn.ModuleList` array wraps if predicting forward `k` targets
- Coordinate training and testing phases
- Save configuration and results for reproducibility

---

## dataset.py

**Purpose**: Script which handles all data loading, scaling, splitting, and PyTorch Dataset creation.

### Classes

| Class | Description |
|-------|-------------|
| `DatesetDirectory` | Loads CSV files, creates geographic IDs (gids), applies scalers, and manages train/test/val splits |
| `ProcessedData` | PyTorch `Dataset` subclass that creates individual training/testing examples with context windows |

### DatesetDirectory Methods/Attributes

| Attribute/Method | Description |
|------------------|-------------|
| `__init__(...)` | Load all data, create gid mappings, fit/apply scalers, create splits |
| `df_data` | Main DataFrame with simulation data indexed by (sim, gid, tid) |
| `df_geo_metadata` | DataFrame mapping gid to MSOA codes |
| `xr_static_features` | xarray DataArray of raw static features |
| `xr_static_features_scaled` | xarray DataArray of scaled static features |
| `temporal_scalers` | List of StandardScaler objects for each temporal column |
| `spatial_scaler` | MinMaxScaler for static features |
| `train_sims`, `test_sims`, `val_sims` | Arrays of simulation IDs for each split |
| `prediction_distribution_types` | List of "nb" or "lognormal" per prediction column |
| `get_raw_data_split()` | Return raw data with split labels |
| `get_column_as_xarray(column, type)` | Get a specific column as xarray for a given split type |

### ProcessedData Methods

| Method | Description |
|--------|-------------|
| `__init__(data_directory, past_context_size, type, loss_type)` | Initialise dataset for "training", "testing", or "validation" |
| `__len__()` | Return total number of (sim, timestep) pairs |
| `__getitem__(idx)` | Return a tuple of (static_features, current_t, current_dow, *contexts, *targets) |

### Standalone Functions

| Function | Description |
|----------|-------------|
| `get_testing_data(testing_dataset, dataset_directory)` | Extract ground truth arrays for scoring. Returns scaled and unscaled ground truth data tensors |

### Key Concepts

- **GID (Geographic ID)**: Internal node index (0, 1, 2, ...) assigned to each MSOA in sorted alphabetical order
- **TID (Time ID)**: Days since the start of each simulation
- **min_timestep**: Allows skipping early timesteps (e.g., for R_t prediction)
- **multistep_k**: Defines the number of multi-step sequential prediction heads to spawn, returning bounds `[current_t, current_t + multistep_k - 1]` per temporal extraction

---

## graph_construction.py

**Purpose**: Build PyTorch Geometric graph structures and process static features.

### Functions

| Function | Description |
|----------|-------------|
| `save_msoa_pairs(data, output_name)` | Save the unique undirected edges from a PyG graph as MSOA pairs to CSV |
| `load_static_features(feature_data, geography_metadata, save_path)` | Load MSOA-level static features from CSV, align to gid ordering, and return as xarray DataArray |
| `haversine(lat1, lon1, lat2, lon2)` | Compute haversine (great-circle) distance between two points in kilometres |
| `get_graph(scaled_features, unscaled_features, ...)` | **Main function**: Build a PyTorch Geometric graph with configurable edge construction |

### Graph Construction Modes

The `get_graph` function supports three edge construction modes (can be combined):

| Mode | Parameter | Description |
|------|-----------|-------------|
| Feature-space k-NN | `use_k_nearest_features=True` | Connect k-nearest neighbours using Euclidean distance in scaled static feature space |
| Geographic k-NN | `use_k_nearest_distance=True` | Connect k-nearest neighbours using Haversine distance |
| Distance threshold | `edge_haversine_threshold=X` | Connect all pairs within X kilometres |

### Graph Output Structure

The returned `torch_geometric.data.Data` object contains:

| Attribute | Shape | Description |
|-----------|-------|-------------|
| `edge_index` | [2, E] | Undirected edges (both directions explicit) |
| `edge_attr` | [E, 1] | Normalised inverse distances (closer = higher weight) |
| `num_nodes` | scalar | Number of MSOAs |
| `pos` | [N, 2] | Longitude, latitude coordinates |
| `gid` | [N] | Node-to-gid mapping |
| `msoa_names` | list | Node-to-MSOA name mapping |
| `msoa_to_nid` | dict | MSOA name to node index mapping |

---

## models.py

**Purpose**: Define neural network architectures for encoding and prediction.

### Classes

| Class | Description |
|-------|-------------|
| `MLP` | Prediction head with dual outputs (mu_head, disp_head) for distribution parameters |
| `GNNEncoder` | GATv2Conv-based graph neural network encoder |
| `MLPEncoder` | Feedforward MLP alternative to GNN for spatial encoding |

### MLP Class

| Method | Description |
|--------|-------------|
| `__init__(hidden_layers, input_dim, num_outputs, prediction_distribution_types, scalers)` | Create MLP with scaling buffers |
| `scale(x_raw)` | Scales raw data using stored means/stds buffers on device |
| `unscale(x_scaled)` | Unscales data using stored means/stds buffers on device |
| `forward(data)` | Return (mu, var) tensors for distribution parameters |

### GNNEncoder Class

| Method | Description |
|--------|-------------|
| `__init__(num_features, layer_specs, dropout, self_loop, use_edge_attr, edge_attr_dim, residual_connection)` | Create GNN with specified layers. `layer_specs` is list of (out_dim, num_heads, concat) tuples |
| `forward(data, return_attention)` | Process graph data, optionally returning attention weights for interpretability |

### MLPEncoder Class

| Method | Description |
|--------|-------------|
| `__init__(input_dim, hidden_layers, output_dim, dropout)` | Create feedforward encoder |
| `forward(x)` | Process input of shape [M, F] or [B, M, F] |

---

## flow_matching_model.py

**Purpose**: Define the neural velocity field and flow matching components for generative modelling.

### Classes

| Class | Description |
|-------|-------------|
| `FlowMatchingModel` | Wrapper handling flow matching training, sampling via ODE solver, and scaling |
| `VelocityField` | Neural network predicting the vector field $v(t, x_t, \text{context})$ |

---

## variational_flow_matching_model.py

**Purpose**: Define the Variational Flow Matching components, combining flow dynamics with explicit distribution modeling.

### Classes

| Class | Description |
|-------|-------------|
| `VFMPredictionHead` | Neural network predicting parameters for the variational distribution $q_t(x_1 | x_t)$ |
| `VFMFlowMatchingModel` | Wrapper handling VFM training (NLL), sampling via optimal transport velocity field, and scaling |

### Key Features

- **Parameterization Switch**: Supports `mean` (direct prediction) and `natural` (distribution parameters) modes.
- **Velocity field**: Derived directly from the expected value of $x_1$ under the predicted distribution.
- **Mixed Likelihood**: Leverages `MixedLoss` to handle both discrete count and continuous positive data.

---

## normalising_flow_model.py

**Purpose**: Define the normalising flow components for generative modelling using the Zuko library.

### Classes

| Class | Description |
|-------|-------------|
| `NormalisingFlowModel` | Wrapper handling normalizing flow training (NLL), sampling, and scaling. Uses Neural Spline Flows (NSF) with coupling transforms. |

### Key Concepts

- **Coupling Transforms**: By setting `passes=2` in Zuko's NSF, the model uses coupling layers (RealNVP-style) which allow for efficient parallel computation.
- **Lazy Jacobian**: The interface masks the fact that the Jacobian determinant is computed "lazily" during the transformation pass, ensuring efficient training.
- **Consistency**: Uses the same dequantization and scaling logic as the Flow Matching model.

---

## maf_model.py

**Purpose**: Define the Masked Autoencoder Flow (MAF) components for generative modelling using the Zuko library.

### Classes

| Class | Description |
|-------|-------------|
| `MAFModel` | Wrapper handling MAF training (NLL), sampling, and scaling. Uses Masked Autoencoder Flow transforms for flexible density estimation. |

### Key Concepts

- **Masked Autoencoding**: MAF uses autoregressive masking to compute Jacobian determinants efficiently without coupling layer constraints.

---

## cnf_model.py

**Purpose**: Define the Continuous Normalising Flow (CNF) components for generative modelling using neural ODE techniques.

### Classes

| Class | Description |
|-------|-------------|
| `CNFModel` | Wrapper handling CNF training (NLL), sampling via ODE solvers, and scaling. Uses continuous transformations for smooth density estimation. |

### Key Concepts

- **Neural ODE**: CNF uses continuous transformations parameterized by neural networks solving ODEs.
- **Variable Trajectory Length**: ODE solver adaptively chooses trajectory length for accuracy-speed trade-off.

---

## training.py

**Purpose**: Implement the training loop with validation, checkpointing, and logging.

### Functions

| Function | Description |
|----------|-------------|
| `train(temporal_model, local_profile_encoder, interaction_encoder, graph_data, dataset_directory, ...)` | Main training loop |

### Training Loop Features

| Feature | Description |
|---------|-------------|
| **Optimiser** | AdamW with OneCycleLR scheduler |
| **Batching** | RandomSampler for training data |
| **Validation** | Evaluated every iteration on full validation set |
| **Checkpointing** | `checkpoint_latest.pth` (resumable) and `best_*.pth` (best weights only) |
| **Logging** | Per-iteration loss to `nll_losses.csv`, per-burden NLL tracking |
| **Resume** | Load from `resume_from_checkpoint` path if provided |

### Checkpoint Contents

`checkpoint_latest.pth` contains:
- Model state dicts (temporal, spatial encoder, temporal encoder)
- Optimiser and scheduler states
- Current iteration and best validation loss
- Training/validation loss history

---

## testing.py

**Purpose**: Orchestrate testing across multiple simulations with resume support.

### Functions

| Function | Description |
|----------|-------------|
| `test(temporal_model, local_profile_encoder, interaction_encoder, graph_data, dataset_directory, ..., num_steps=50)` | Main testing orchestration function |

### Testing Features

| Feature | Description |
|---------|-------------|
| **Resume Logic** | Checks for `predictions/179/predictions.feather` and `spatial_scores.feather` to determine if a simulation is complete |
| **Per-simulation** | Creates `ProcessedData` for each test simulation individually |
| **Sliding Window** | Generates predictions for all timesteps using `get_sliding_window_predictions` |

---

## testing_sliding_window.py

**Purpose**: Implement autoregressive prediction with Monte Carlo uncertainty quantification.

### Functions

| Function | Description |
|----------|-------------|
| `get_sliding_window_predictions(..., num_steps=50)` | Core autoregressive prediction loop |

### Autoregressive Process

```
For each starting timestep t:
    1. Initialize context from ground truth [t-context_size : t-1]
    2. For timesteps t to max_timestep:
        a. Encode spatial features (if not ablated)
        b. Encode temporal context through GNN
        c. Build input features (embeddings + day-of-week + latest context)
        d. Predict distribution parameters (mu, var)
        e. Sample S times from mixed distribution
        f. Inverse transform samples to original scale
        g. Roll context window: drop oldest, append new scaled prediction
    3. Compute proper scoring rules against ground truth
    4. Save results to feather files
```

### Output Files

| File | Contents |
|------|----------|
| `predictions.feather` | Median, 2.5th, 97.5th percentiles per MSOA per timestep |
| `spatial_scores.feather` | CRPS and Interval Score per MSOA per timestep |
| `window_times.feather` | Energy Score and Variogram per starting timestep |

---

## distribution_construction.py

**Purpose**: Define loss functions and sampling procedures for mixed Negative Binomial + LogNormal distributions.

### Functions

| Function | Description |
|----------|-------------|
| `MixedLoss(mu_pred, var_pred, y_true, types, parameterization)` | Compute mixed NLL loss for training. Supports `natural` and `mean` parameterization modes. |
| `sample_mixed(mu_pred, var_pred, types, parameterization)` | Draw samples from mixed distribution for inference or internal validation |

### Distribution Parameterisation

| Distribution | When Used | mu Transform | var Transform |
|--------------|-----------|--------------|---------------|
| Negative Binomial | Count data (`"nb"`) | `softplus(mu) + 1e-3` | `mu + softplus(var) + 1.0` (natural) or `1/disp` (mean) |
| LogNormal | Continuous data (`"lognormal"`) | Raw (loc parameter) | `softplus(var) + 1e-6` (natural) or `1e-6` (mean) |

### Key Guarantee

**`MixedLoss` and `sample_mixed` use identical transformations** to ensure consistency between training and inference.

---

## proper_scoring.py

**Purpose**: Implement proper scoring rules for probabilistic forecast evaluation.

### Functions

| Function | Description |
|----------|-------------|
| `energy_score(y_true, samples)` | Multivariate extension of CRPS |
| `variogram(y_true, samples, p=0.5)` | Evaluates whether spatial correlations are captured |
| `interval_score(y, lower, upper, alpha=0.05)` | Rewards narrow intervals, penalises observations outside |
| `crps(y, samples)` | Generalisation of MAE to probabilistic forecasts |

### Score Interpretation

**All scores are negatively oriented**: Lower is better

---

## proper_scoring_torch.py

**Purpose**: PyTorch implementations of proper scoring rules for GPU-accelerated evaluation during inference.

### Functions

| Function | Description |
|----------|-------------|
| `energy_score_torch(y_true, samples)` | GPU-accelerated multivariate Energy Score |
| `variogram_torch(y_true, samples, p=0.5)` | GPU-accelerated Variogram Score |
| `interval_score_torch(y, lower, upper, alpha=0.05)` | GPU-accelerated Interval Score |
| `crps_torch(y, samples)` | GPU-accelerated CRPS computation |

### Key Features

- **Batch Processing**: Efficiently computes scores for multiple samples and timesteps in parallel
- **GPU Acceleration**: All operations use PyTorch tensor operations for CUDA compatibility
- **Consistency**: Matches NumPy implementations from `proper_scoring.py`

---

## train_test_helpers.py

**Purpose**: Provide utility functions for graph caching and plotting.

### Functions

| Function | Description |
|----------|-------------|
| `cache_graph(graph_data, device)` | Move graph topology (edge_index, edge_attr) to device once and track cached device |
| `cache_graph_topology(graph_data, S, device)` | Pre-build batched graph topology for S Monte Carlo samples |
| `get_batched_graph(node_features, graph_data, device)` | Build PyG Data object for 2D [N,F] or 3D [S,N,F] node features |
| `plot_train_val_loss(validation_loss, training_loss, path)` | Save training/validation loss curves as PNG |

### Why Graph Caching Matters

- Graph topology is constant during training and inference
- Avoids repeated CPU→GPU tensor transfers
- `cache_graph_topology` enables efficient Monte Carlo by pre-computing the batched edge structure

---

## data_processing.py

**Purpose**: Standalone utility script for preparing static feature CSVs when working with new regions. This script is used **before** running experiments to ensure features are correctly processed.

### Functions

| Function | Description |
|----------|-------------|
| `preprocess_feature_data(msoa_features, dont_drop=[])` | Preprocess MSOA-level features by dropping identifier columns and encoding categorical variables |

### preprocess_feature_data Details

**Inputs:**
- `msoa_features`: pandas DataFrame with one row per MSOA containing raw feature data
- `dont_drop`: List of column names to preserve even if they're in the default drop list

**Processing Steps:**
1. Drops default identifier columns: `msoa`, `lat_long`, `areas_in_msoa`, `associated_city`
2. Identifies non-numeric (object) columns
3. Applies `LabelEncoder` to convert categorical strings to integer codes
4. Specifically handles hospital codes, university UKPRNs, and school URNs

**Output:** Processed DataFrame ready for use with GENIE

### Typical Workflow for New Geographies

The script includes example code demonstrating the standard workflow:

```python
# 1. Define MSOAs for your region
sunderland_msoas = ['E02001791', 'E02001792', ...]

# 2. Extract those MSOAs from master feature file
static_features_data = pd.read_csv("DATA/msoa_feature_data.csv")
region_features = static_features_data[static_features_data['msoa'].isin(sunderland_msoas)]
region_features.to_csv("DATA/static_features_SUNDERLAND.csv", index=False)

# 3. Preprocess (encode categoricals, drop identifiers)
processed = preprocess_feature_data(
    pd.read_csv("DATA/static_features_SUNDERLAND.csv"),
    dont_drop=["msoa", "lat_long"]
)
processed.to_csv("DATA/processed_msoa_features_SUNDERLAND.csv", index=False)

# 4. Filter to match training data columns (for transfer learning)
training_cols = pd.read_csv("DATA/static_features_12_10_2025.csv").columns
final = processed[training_cols]
final.to_csv("DATA/static_features_SUNDERLAND_filtered.csv", index=False)
```

### When to Use This Script
- **New geography**: Before running experiments for a new region
- **New geography transfer**: Before running transfer experiments on a new region
---

## File Dependency Graph

```
data_processing.py (standalone preprocessing utility)
    │
    └── Used BEFORE experiments to prepare static features for new geographies
        └── Output: static_features_*.csv files

run_experiment.py (entry point)
    │
    ├── dataset.py
    │       └── graph_construction.py (load_static_features)
    │
    ├── graph_construction.py (get_graph)
    │
    ├── models.py (MLP, GNNEncoder, MLPEncoder)
    │
    ├── train_test_helpers.py (cache_graph)
    │
    ├── training.py
    │       ├── dataset.py (ProcessedData)
    │       ├── distribution_construction.py (MixedLoss)
    │       └── train_test_helpers.py (get_batched_graph, plot_train_val_loss)
    │
    └── testing.py
            ├── dataset.py (ProcessedData, get_testing_data)
            │
            └── testing_sliding_window.py
                    ├── distribution_construction.py (sample_mixed)
                    ├── flow_matching_model.py
                    ├── variational_flow_matching_model.py
                    ├── normalising_flow_model.py
                    ├── maf_model.py
                    ├── cnf_model.py
                    ├── proper_scoring.py (crps, interval_score, energy_score, variogram)
                    ├── proper_scoring_torch.py (torch-accelerated versions)
                    └── train_test_helpers.py (get_batched_graph)
```

---

---

## Data Flow Summary

```
┌───────────────────────────────────────────────────────────────────────┐
│                    DATA PREPARATION PHASE                             │
│                    (data_processing.py)                               │
│                                                                       │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐         │
│  │ Raw MSOA     │ ──── │ Filter to    │ ──── │ Encode cats, │         │
│  │ Features     │      │ Region       │      │ Drop IDs     │         │
│  └──────────────┘      └──────────────┘      └──────────────┘         │
│                                                      │                │
│                                              ┌───────▼─────────┐      │
│                                              │ Filtered CSV    │      │
│                                              │ (column-aligned)│      │
│                                              └───────┬─────────┘      │
└──────────────────────────────────────────────────────┼────────────────┘
                                                       │
┌──────────────────────────────────────────────────────▼──────────┐
│                    EXPERIMENT PIPELINE                          │
│                                                                 │
│  ┌─────────────────┐                                            │
│  │  CSV Files      │                                            │
│  │  (data, meta,   │                                            │
│  │   static)       │                                            │
│  └────────┬────────┘                                            │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                            │
│  │ DatesetDirectory│  ← Loads, scales, splits data              │
│  │                 │  ← Creates gid mappings                    │
│  └────────┬────────┘                                            │
│           │                                                     │
│      ┌────┴────┐                                                │
│      │         │                                                │
│      ▼         ▼                                                │
│  ┌───────┐ ┌─────────┐                                          │
│  │ Graph │ │Processed│                                          │
│  │ (PyG) │ │  Data   │                                          │
│  └───┬───┘ └────┬────┘                                          │
│      │          │                                               │
│      └────┬─────┘                                               │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                            │
│  │    Encoders     │  ← GNNEncoder (spatial)                    │
│  │   (GNN/MLP)     │  ← GNNEncoder (temporal)                   │
│  └────────┬────────┘                                            │
│           │                                                     │
│           ▼                                                     │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  MODEL SELECTION (choose one):                           │ │
│  │                                                            │ │
│  │  • MLP: Predicts distribution parameters                 │ │
│  │  • Flow Matching: Predicts velocity field (ODE)          │ │
│  │  • Variational Flow Matching: Distribution-based flow    │ │
│  │  • NSF (Normalising Flow): Neural Spline Flows           │ │
│  │  • MAF: Masked Autoencoder Flow                          │ │
│  │  • CNF: Continuous Normalising Flow                      │ │
│  └────────┬───────────────────────────────────────────────────┘ │
│           │                                                     │
│      ┌────┴─────────────────────────────────┬──────────────┐    │
│      │                                      │              │    │
│      ▼                                      ▼              ▼    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│  │MLP Head  │  │Flow Match│  │VFM Head  │  │NSF/MAF   │  │CNF       │ │
│  │Params    │  │Velocity  │  │Params    │  │Direct    │  │Direct    │ │
│  └─────┬────┘  └─────┬────┘  └─────┬────┘  └─────┬────┘  └─────┬────┘ │
│        │             │             │             │             │      │
│        ▼             ▼             ▼             ▼             ▼      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│  │Sampling  │  │ODE Solve │  │Sampling  │  │Sampling  │  │Sampling  │ │
│  │from Dist │  │Integrate │  │from Dist │  │from Flow │  │from Flow │ │
│  └─────┬────┘  └─────┬────┘  └─────┬────┘  └─────┬────┘  └─────┬────┘ │
│        │             │             │             │             │      │
│        └─────────────┴─────────────┴─────────────┴─────────────┘      │
│                                    │                                   │
│                                    ▼                                   │
│  ┌─────────────────────────────────────────────────────┐              │
│  │ Proper Scoring & Evaluation                        │              │
│  │ ← CRPS, Energy Score, Interval Score, Variogram   │              │
│  │ ← proper_scoring.py or proper_scoring_torch.py    │              │
│  └─────────────────────────────────────────────────────┘              │
│                                                                 │
└──────────────────────────────────────────────────────────────────┘
```