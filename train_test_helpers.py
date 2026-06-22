import random
import os
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Batch, Data

random.seed(26)

def cache_graph(graph_data, device):
    """
    This script moves reusable graph tensors (edge_index / edge_attr) onto a target device once,
    and remember which device they are cached on to avoid repeated .to(device) calls.

    Inputs:
        graph_data:
            torch_geometric.data.Data graph object. Must contain:
            - graph_data.edge_index
            Optionally may contain:
            - graph_data.edge_attr

        device:
            Target torch device (e.g. "cuda" or "cpu").

    Output:
        Returns the same graph_data object, with edge_index/edge_attr moved onto `device`.
        Adds/updates graph_data.cached_device so we can skip re-copying next time.
    """
    # If the graph has been moved to device already return immediately.
    if getattr(graph_data, "cached_device", None) == device:
        return graph_data

    ##########################################################
    # Cache graph topology tensors on the target device once #
    ##########################################################
    # Move edge_index to the target device
    # non_blocking is used to helpfully speed up the transfer speed, here is the documentation for ref https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html
    graph_data.edge_index = graph_data.edge_index.to(device, non_blocking=True)

    # If edge_attr exists, also move it to the target device.
    if hasattr(graph_data, 'edge_attr') and graph_data.edge_attr is not None:
        graph_data.edge_attr = graph_data.edge_attr.to(device, non_blocking=True) # Again using non_blocking

    # Store which device this graph is currently cached on so we can skip future copies.
    graph_data.cached_device = device

    # Return the (now cached) graph object.
    return graph_data

def build_batched_topology(graph_data, S: int, device):
    """
    Build the batched graph topology for S repeated copies of the same graph.

    During Monte Carlo / multi-sample inference, node features are shaped [S, N, F]
    (S samples, N nodes, F features).  PyG expects a single "big" batched graph where:
        - edge_index describes edges for all S graphs concatenated
        - batch vector assigns each node to its graph id (0..S-1)

    Because the graph structure (edges) does not change across samples, this function
    builds that representation once.  It contains no caching logic — see
    cache_graph_topology if you want a memoised version.

    Inputs:
        graph_data:
            A torch_geometric.data.Data graph containing:
            - graph_data.edge_index : [2, E]
            - graph_data.edge_attr  : optional [E, ...]
            - graph_data.num_nodes  : N

        S:
            Number of repeated graph copies (i.e. number of Monte Carlo samples).

        device:
            Torch device to place the output tensors on.

    Output:
        Returns a tuple:
            (edge_index_batched, edge_attr_batched, batch_vec)
        where:
            - edge_index_batched : [2, S*E]
            - edge_attr_batched  : [S*E, ...] or None
            - batch_vec          : [S*N]
    """
    ####################################
    # Build the batched graph topology #
    ####################################
    N = graph_data.num_nodes  # Number of nodes in the original graph
    dummy_x = torch.zeros(N, 1, device=device)  # Dummy node features just to create Data objects
    edge_index = graph_data.edge_index  # [2, E]
    edge_attr = getattr(graph_data, 'edge_attr', None)  # [E, ...] or None

    # Create S identical Data objects and batch them
    shells = [Data(x=dummy_x, edge_index=edge_index, edge_attr=edge_attr) for _ in range(S)]
    batched = Batch.from_data_list(shells)  # Batched graph with S copies

    # Extract the batched topology tensors:
    # - edge_index_batched: edges for all S graphs (with node indices offset appropriately)
    # - batch_vec: assigns each node to which graph it came from (0..S-1)
    edge_index_batched = batched.edge_index  # [2, S*E]
    batch_vec = batched.batch  # [S*N]
    edge_attr_batched = getattr(batched, 'edge_attr', None)  # [S*E, ...] or None

    return edge_index_batched, edge_attr_batched, batch_vec


def cache_graph_topology(graph_data, S: int, device):
    """
    Return the batched graph topology for S repeated copies of graph_data,
    building and storing it on the first call for each value of S.

    This is a pure caching wrapper around build_batched_topology.  The topology
    is stored in graph_data.topo_cache[S] so subsequent calls with the same S
    skip reconstruction entirely.

    Inputs:
        graph_data:
            A torch_geometric.data.Data graph containing:
            - graph_data.edge_index : [2, E]
            - graph_data.edge_attr  : optional [E, ...]
            - graph_data.num_nodes  : N

        S:
            Number of repeated graph copies (i.e. number of Monte Carlo samples).

        device:
            Torch device to place cached tensors on.

    Output:
        Returns a cached tuple:
            (edge_index_batched, edge_attr_batched, batch_vec)
        where:
            - edge_index_batched : [2, S*E]
            - edge_attr_batched  : [S*E, ...] or None
            - batch_vec          : [S*N]
    """
    # Initialise the topology cache dict on first use.
    if not hasattr(graph_data, "topo_cache"):
        graph_data.topo_cache = {}

    # Return the cached topology if we have already built it for this S.
    if S in graph_data.topo_cache:
        return graph_data.topo_cache[S]

    # Build topology and store it for future calls.
    graph_data.topo_cache[S] = build_batched_topology(graph_data, S, device)
    return graph_data.topo_cache[S]

def build_batched_graph_data(x, ei_b, ea_b, batch_vec):
    """
    Build a PyG Data object for a batch of S graphs from pre-built topology tensors
    and flattened node features.

    This function contains no caching logic — see cache_batched_data if you want a
    memoised version that reuses the Data shell across calls.

    Inputs:
        x        : [S*N, F]  flattened node features (already on the target device)
        ei_b     : [2, S*E]  batched edge_index
        ea_b     : [S*E, ...] or None  batched edge_attr
        batch_vec: [S*N]     graph assignment vector

    Output:
        A new torch_geometric.data.Data object.
    """
    return Data(x=x, edge_index=ei_b, edge_attr=ea_b, batch=batch_vec)


def cache_batched_graph_data(graph_data, S: int, x, ei_b, ea_b, batch_vec):
    """
    Return a PyG Data object for a batch of S graphs, reusing a cached shell when
    one already exists for this value of S.

    On the first call for a given S the Data object is built via build_batched_graph_data
    and stored in graph_data._data_cache[S].  On subsequent calls only the node
    features (.x) are updated — avoiding the cost of allocating a new object.

    Inputs:
        graph_data: the graph whose _data_cache dict is used for storage
        S        : number of graph copies (cache key)
        x        : [S*N, F]  flattened node features (already on the target device)
        ei_b     : [2, S*E]  batched edge_index
        ea_b     : [S*E, ...] or None  batched edge_attr
        batch_vec: [S*N]     graph assignment vector

    Output:
        A torch_geometric.data.Data object (potentially shared / mutated in place).
    """
    # Initialise the data cache dict on first use.
    if not hasattr(graph_data, '_data_cache'):
        graph_data._data_cache = {}

    cached = graph_data._data_cache.get(S)
    if cached is not None:
        # Fast path: just swap the node features on the existing Data shell.
        cached.x = x
        return cached

    # Slow path (first call for this S): build and store the Data shell.
    data_obj = build_batched_graph_data(x, ei_b, ea_b, batch_vec)
    graph_data._data_cache[S] = data_obj
    return data_obj


def get_batched_graph(node_features, graph_data, device, use_data_cache=True, use_topology_cache=True):
    """
    Function to build a PyTorch Geometric `Data` object for either:
      - a single graph with node features [N, F], or
      - a batch of S identical graphs (same topology) with node features [S, N, F].
            Handling the batching of the graph data, link: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Batch.html
            To quote the documentation:
            "A data object describing a batch of graphs as one big (disconnected) graph... 
            Single graphs can be identified via the assignment vector batch, which maps each node to its respective graph identifier.
      
    Why this exists:
        PyG represents a batch of graphs as one big (disconnected) graph.
        Nodes are concatenated, edges are offset accordingly, and "batch" tells you
        which original graph each node belongs to.

    Performance note:
        For 3D (batched) inputs, topology is built via build_batched_topology /
        cache_graph_topology, and the Data shell is built via build_batched_graph_data /
        cache_batched_graph_data.  Both pairs follow the same pattern: build_* constructs
        fresh each time; cache_* stores on the graph_data object and reuses on
        repeat calls for the same S.  Features already on `device` skip .to().
    """
    
    # Graph topology tensors (edge_index / edge_attr) should already be cached on the device via cache_graph(...).
    edge_index = graph_data.edge_index # Get the existing edge_index
    edge_attr = getattr(graph_data, 'edge_attr', None) # Get existing edge_attr if it exists

    ######################################
    # Case 1: single graph node features #
    ######################################
    if node_features.dim() == 2:
        # Single sample so the dimensions are [N, F] where N is num_nodes, F is num_features
        # Skip .to(device) if features are already on the target device
        x = node_features if node_features.device == torch.device(device) else node_features.to(device, non_blocking=True)
        # batch_vec has shape [N] and is all zeros, meaning "all nodes belong to graph 0".
        batch_vec = torch.zeros(x.size(0), dtype=torch.long, device=device)
        # Return a Data object representing one graph.
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch_vec)

    ######################################
    # Case 2: batched graphs (S samples) #
    ######################################
    elif node_features.dim() == 3:
        # Multi-sample: node_features has shape [S, N, F]
        S, N, F = node_features.size()

        # Build (or reuse) the batched topology for S copies of the same graph.
        # Returns:
        # - ei_b: [2, S*E]
        # - ea_b: [S*E, ...] or None
        # - batch_vec: [S*N]
        # Build (or reuse from cache) the batched topology for S copies of the same graph.
        if use_topology_cache:
            ei_b, ea_b, batch_vec = cache_graph_topology(graph_data, S, device)
        else:
            ei_b, ea_b, batch_vec = build_batched_topology(graph_data, S, device)

        # Flatten features so PyG sees one big graph with S*N nodes.
        x = node_features.reshape(S * N, F)
        # Skip .to(device) if features are already on the target device.
        if x.device != torch.device(device):
            x = x.to(device, non_blocking=True)

        # Build (or reuse from cache) the Data object for this batch.
        if use_data_cache:
            return cache_batched_graph_data(graph_data, S, x, ei_b, ea_b, batch_vec)

        return build_batched_graph_data(x, ei_b, ea_b, batch_vec)

    ##########################
    # Unexpected input shape #
    ##########################
    else:
        raise ValueError(
            f"node_features must be 2D [N,F] or 3D [S,N,F], got shape {tuple(node_features.shape)}"
        )
        
def apply_poisson_noise_to_contexts(
    contexts,
    columns_for_context,
    columns_with_data,
    temporal_scalers,
    prediction_distribution_types,
    columns_for_prediction,
    device,
    model_type="mlp",
):
    """
    Apply Poisson noise to context history tensors for discrete/count (NB) variables.

    During training, the model receives a history window of past observations as context.
    This function perturbs those context values by sampling from a Poisson distribution
    whose rate equals the reconstructed raw count, i.e.:

        x_noisy ~ Poisson(lambda = x_raw)

    The idea is to make the model robust to small observation-level noise in the history
    (a principled data augmentation strategy for count data).

    IMPORTANT CONTRACT:
    - Applied ONLY inside the training loop — never during validation or testing.
    - Targets (y_t_truth) are NOT modified; the model is still evaluated against
      the original, clean simulated values.
    - Only columns typed as 'nb' (Negative Binomial / discrete counts) are perturbed.
      Continuous columns ('lognormal') are returned unchanged.

    VFM COMPATIBILITY NOTE:
    The variational_flow_matching model rewrites dataset_directory.data_tensor to a
    log-scaled representation (log1p + Z-score), so the raw counts can no longer be
    recovered via the StandardScaler inverse transform. Attempting to apply Poisson
    noise in this case would corrupt the inputs. A NotImplementedError is raised
    so the user is explicitly informed.

    Args:
        contexts              : list of [B, M, context_size] tensors (scaled), one per context column.
        columns_for_context   : list of context column name strings.
        columns_with_data     : list of all data column name strings (scalers are aligned to this).
        temporal_scalers      : list of fitted StandardScaler objects, aligned to columns_with_data.
        prediction_distribution_types : list of 'nb' or 'lognormal', aligned to columns_for_prediction.
        columns_for_prediction: list of prediction column name strings.
        device                : torch device string (e.g. 'cuda' or 'cpu').
        model_type            : model type string from config. VFM raises NotImplementedError.

    Returns:
        List of [B, M, context_size] tensors with Poisson noise applied to NB columns.
    """
    # VFM rewrites data_tensor to log-space, so StandardScaler inverse transform no longer
    # recovers raw counts. Raise early with a clear message.
    if model_type == "variational_flow_matching":
        raise NotImplementedError(
            "poisson_noise_context is not currently compatible with model_type='variational_flow_matching'. "
            "VFM overwrites dataset_directory.data_tensor with log-scaled values, so the context tensors "
            "cannot be inverse-transformed to raw counts via the StandardScaler. "
            "To use Poisson noise with VFM, the augmentation would need to be applied to the raw counts "
            "before the log transform is computed in run_experiment.py. "
            "Please either set poisson_noise_context: false in your config, or implement pre-log augmentation."
        )

    # Build a quick lookup: context column name -> 'nb' or 'lognormal' (or None if not a prediction col)
    pred_type_map = {
        col: prediction_distribution_types[i]
        for i, col in enumerate(columns_for_prediction)
    }

    augmented = []
    for i, ctx_tensor in enumerate(contexts):  # ctx_tensor: [B, M, context_size]
        col_name = columns_for_context[i]

        # Only perturb columns that appear in columns_for_prediction and are typed as 'nb'.
        # Context-only columns (not in columns_for_prediction) are left unchanged.
        is_nb = pred_type_map.get(col_name, "lognormal") == "nb"

        if not is_nb:
            augmented.append(ctx_tensor)
            continue

        # Locate the matching StandardScaler for this column.
        col_idx = columns_with_data.index(col_name)
        scaler = temporal_scalers[col_idx]

        # Cache scaler parameters as scalar tensors on the target device so that the
        # inverse/forward transforms are fully in-graph without per-batch CPU round-trips.
        mean = torch.tensor(
            float(scaler.mean_[0]), device=device, dtype=ctx_tensor.dtype
        )
        std = torch.tensor(
            float(scaler.scale_[0]), device=device, dtype=ctx_tensor.dtype
        )

        # Inverse-transform: recover approximate raw counts from scaled values.
        raw = ctx_tensor * std + mean

        # Clamp to zero — Poisson rate must be non-negative.
        # Slight negative values can arise from numerical noise or zero-padding.
        raw = raw.clamp(min=0.0)

        # Draw Poisson samples. torch.poisson() is a native CUDA op with no overhead.
        noisy_raw = torch.poisson(raw)

        # Re-apply the same StandardScaler forward transform.
        noisy_scaled = (noisy_raw - mean) / std

        augmented.append(noisy_scaled)

    return augmented


def plot_train_val_loss(training_loss, validation_loss_1step, validation_loss_multistep, trained_models_subfolder):

    plt.figure()
    plt.plot(training_loss, label='Training Loss')
    
    # Filter out NaN/None values from validation_loss_1step and align with x-axis epochs
    val_epochs_1 = [i for i, v in enumerate(validation_loss_1step) if v is not None and not (isinstance(v, float) and v != v)]
    val_losses_1 = [validation_loss_1step[i] for i in val_epochs_1]
    
    if len(val_losses_1) > 0:
        plt.plot(val_epochs_1, val_losses_1, label='Validation Loss (1-Step Ahead)', marker='o')

    if validation_loss_multistep is not None:
        # Filter out NaN/None values from validation_loss_multistep and align with x-axis epochs
        val_epochs_multi = [i for i, v in enumerate(validation_loss_multistep) if v is not None and not (isinstance(v, float) and v != v)]
        val_losses_multi = [validation_loss_multistep[i] for i in val_epochs_multi]

        if len(val_losses_multi) > 0:
            plt.plot(val_epochs_multi, val_losses_multi, label='Validation Loss (Multistep Combined)', marker='x')
        
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.ylim(bottom=0)
    plt.legend()
    plot_path = os.path.join(trained_models_subfolder, "loss_plot.png")
    plt.savefig(plot_path)
    plt.close()


def plot_lr_vs_iteration(lr_history_muon, lr_history_adamw, trained_models_subfolder):

    plt.figure()

    has_muon = any(value is not None and not (isinstance(value, float) and value != value) for value in lr_history_muon)
    has_adamw = any(value is not None and not (isinstance(value, float) and value != value) for value in lr_history_adamw)

    if has_muon:
        muon_epochs = [i for i, value in enumerate(lr_history_muon) if value is not None and not (isinstance(value, float) and value != value)]
        muon_lrs = [lr_history_muon[i] for i in muon_epochs]
        plt.plot(muon_epochs, muon_lrs, label='Muon LR', marker='o', markersize=2)

    if has_adamw:
        adamw_epochs = [i for i, value in enumerate(lr_history_adamw) if value is not None and not (isinstance(value, float) and value != value)]
        adamw_lrs = [lr_history_adamw[i] for i in adamw_epochs]
        plt.plot(adamw_epochs, adamw_lrs, label='AdamW LR', marker='x', markersize=2)

    plt.xlabel('Iteration')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(trained_models_subfolder, "lr_plot.png")
    plt.savefig(plot_path)
    plt.close()