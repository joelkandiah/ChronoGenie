import os
import random
import time

import numpy as np
import torch
import pyarrow as pa
from pyarrow import feather

from distribution_construction import sample_mixed
from train_test_helpers import get_batched_graph, cache_graph_topology
from proper_scoring import interval_score, crps, energy_score, variogram
from proper_scoring_torch import (
    interval_score_torch, crps_torch, energy_score_torch, variogram_torch
)
from flow_matching_model import FlowMatchingModel
from diffusion_model import DiffusionModel
from normalising_flow_model import NormalisingFlowModel
from variational_flow_matching_model import VFMFlowMatchingModel
from maf_model import MAFFlowModel
from cnf_model import CNFFlowModel

torch.manual_seed(26)
random.seed(26) 
np.random.seed(26)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(26)

def get_sliding_window_predictions(
        dataset_directory, loader, scalers, num_msoas, device,
        temporal_model, spatial_encoder, temporal_encoder,
        graph_data, max_timestep, output_folder, 
        truth_unscaled, num_samples = 100, 
        spatial_ablation=False, include_latest_context = False,
        spatial_encoder_type = "mlp", num_steps = 50,
        window_batch_size = 16):

    temp_graph_data, spatial_graph_data = graph_data[0], graph_data[1]

    # Creating folder for storing the embeddings
    save_path_embeddings = f"{output_folder}/embeddings"
    os.makedirs(save_path_embeddings, exist_ok=True)

    # Ensure truth_unscaled is a Torch tensor on the GPU for faster scoring
    if isinstance(truth_unscaled, np.ndarray):
        truth_unscaled_gpu = torch.from_numpy(truth_unscaled).float().to(device)
    else:
        truth_unscaled_gpu = truth_unscaled.to(device)

    # Disabling gradient calculations during inference. 
    # We don't call backward here so it will reduce memory consumption
    with torch.no_grad(): 
        for testing_trajectory in loader:
            # location_specific_info: [B, M, F_static]
            # timesteps: [B]
            location_specific_info, timesteps, current_dow_batch, *ctx_and_targets = testing_trajectory

            prediction_names = list(dataset_directory.columns_for_prediction) 
            context_names = list(dataset_directory.columns_for_context) 
            num_prediction  = len(prediction_names)
            num_context   = len(context_names)

            # context_list: list of [B, M, L]
            context_list = [c.to(device) for c in ctx_and_targets[:num_context]]
            location_specific_info = location_specific_info.to(device)
            timesteps = timesteps.to(device)
            current_dow_batch = current_dow_batch.to(device)

            num_autoreg_windows = context_list[0].size(0)
            print(f"num_autoreg_windows {num_autoreg_windows}")

            # Precompute context-to-prediction index mapping once.
            # Avoids repeated list.index() lookups inside the inner loop.
            ctx_to_pred_idx = {}
            for c_idx, c_name in enumerate(context_names):
                if c_name in prediction_names:
                    ctx_to_pred_idx[c_idx] = prediction_names.index(c_name)

            # Cache model type check once — avoids isinstance() dispatch every step.
            is_generative_model = isinstance(temporal_model, (
                FlowMatchingModel, DiffusionModel, NormalisingFlowModel, VFMFlowMatchingModel,
                MAFFlowModel, CNFFlowModel
            ))

            # These arrays are the same for every window — build once.
            burden_map = np.array(prediction_names)
            msoa_name_map = dataset_directory.df_geo_metadata["MSOA"].to_numpy()

            window_times_seconds = []

            # Iterate over batches of sliding windows
            for batch_start in range(0, num_autoreg_windows, window_batch_size):
                batch_end = min(batch_start + window_batch_size, num_autoreg_windows)
                W = batch_end - batch_start
                window_indices = torch.arange(batch_start, batch_end, device=device)
                
                window_start_time = time.time()
                print(f"Processing windows {batch_start+1}-{batch_end} of {num_autoreg_windows}")
                
                starting_timesteps = timesteps[window_indices] # [W]
                starting_dows = current_dow_batch[window_indices] # [W]
                
                # Vectorised max-horizon calculation — avoids a Python loop
                # with per-element .item() GPU syncs.
                horizons = torch.clamp(starting_timesteps + 60, max=max_timestep) - starting_timesteps
                batch_max_horizon = horizons.max().item()  # single sync point
                
                # current_contexts_batch: list of [W, M, L]
                # contexts_multi_sample: list of [W, S, M, L]
                # .contiguous() instead of .clone(): both make the expanded view
                # writable, but .contiguous() avoids a redundant deep copy when
                # the memory layout is already contiguous after expand().
                contexts_multi_sample = [
                    c[window_indices].unsqueeze(1).expand(-1, num_samples, -1, -1).contiguous()
                    for c in context_list
                ]

                # Pre-allocate tensor for all samples in this batch: [W, S, T, M, V]
                # T is batch_max_horizon
                batch_samples_tensor = torch.zeros(
                    (W, num_samples, batch_max_horizon, num_msoas, num_prediction),
                    device=device, dtype=torch.float32
                )
                
                # Pre-allocate for distributions if applicable: [W, S, T, M, V]
                mu_preds_tensor = torch.zeros_like(batch_samples_tensor)
                var_preds_tensor = torch.zeros_like(batch_samples_tensor)

                # Precompute spatial embeddings for the whole batch: [W, M, d_s]
                h_spatial_batch = None
                if spatial_encoder is not None:
                    lsi_batch = location_specific_info[window_indices] # [W, M, F_static]
                    use_gnn_spatial = (spatial_encoder_type or "gnn").lower() == "gnn"
                    if use_gnn_spatial:
                        # Batch the graphs for all windows in the batch
                        spatial_encoder_batch = get_batched_graph(lsi_batch, spatial_graph_data, device)
                        out = spatial_encoder(spatial_encoder_batch, return_attention=False)
                        h_spatial_batch = out[0] if isinstance(out, (tuple, list)) else out.squeeze(0)
                        h_spatial_batch = h_spatial_batch.view(W, num_msoas, -1) # [W, M, d_s]
                    else:
                        h_spatial_batch = spatial_encoder(lsi_batch) # [W, M, d_s]

                # Pre-warm topology caches before the autoregressive loop so
                # the first step does not pay the one-time construction cost.
                # Temporal graph: batched for W * num_samples copies.
                cache_graph_topology(temp_graph_data, W * num_samples, device)

                # Precompute shape constants used repeatedly inside the loop.
                WS  = W * num_samples                   # num graphs in temporal batch
                WSM = WS * num_msoas                     # total nodes across all graphs
                WSNV = (W, num_samples, num_msoas, num_prediction)  # reshape target for model output

                # lsi_expanded is constant across steps — hoist outside the loop.
                if spatial_encoder is None and not spatial_ablation:
                    lsi_expanded = location_specific_info[window_indices].unsqueeze(1).expand(-1, num_samples, -1, -1)

                # h_spatial_expanded is also constant across steps — hoist outside.
                if spatial_encoder is not None:
                    h_spatial_expanded = h_spatial_batch.unsqueeze(1).expand(-1, num_samples, -1, -1)
                
                # Autoregressive loop vectorized over W windows and S samples
                for step in range(batch_max_horizon):
                    # Combine context: [W, S, M, num_ctx * L]
                    current_context_combined = torch.cat(contexts_multi_sample, dim=3)
                    
                    # Current absolute timesteps for each window: [W]
                    current_timesteps = starting_timesteps + step
                    
                    if spatial_encoder is not None:
                        # h_spatial_expanded already computed outside the loop.
                        
                        # temporal_encoder_batch: needs [WS, M, F]
                        ctx_flat = current_context_combined.view(WS, num_msoas, -1)
                        temporal_encoder_batch = get_batched_graph(ctx_flat, temp_graph_data, device)
                        h_temporal_flat = temporal_encoder(temporal_encoder_batch, return_attention=False)
                        h_temporal = h_temporal_flat.view(W, num_samples, num_msoas, -1) # [W, S, M, d_t]
                    else:
                        if spatial_ablation:
                            combined_node_features = current_context_combined.view(WS, num_msoas, -1)
                        else:
                            combined_node_features = torch.cat([lsi_expanded, current_context_combined], dim=3)
                            combined_node_features = combined_node_features.view(WS, num_msoas, -1)

                        combined_batch = get_batched_graph(combined_node_features, temp_graph_data, device)
                        h_combined_flat = temporal_encoder(combined_batch, return_attention=False)
                        h_combined = h_combined_flat.view(W, num_samples, num_msoas, -1)

                    # Day of week vectorized: [W, 1, 1] expand to [W, S, M, 7]
                    dow_indices = (starting_dows + step) % 7
                    dow_oh = torch.nn.functional.one_hot(dow_indices, num_classes=7).float()
                    day_of_week = dow_oh.view(W, 1, 1, 7).expand(-1, num_samples, num_msoas, -1)

                    # Create feature list for model input
                    feature_list = []
                    if include_latest_context:
                        # latest_vals: [W, S, M, num_ctx]
                        latest_vals = torch.cat([c[:, :, :, -1:].contiguous() for c in contexts_multi_sample], dim=3)
                        feature_list.append(latest_vals)

                    if spatial_encoder is not None:
                        feature_list.extend([h_spatial_expanded, h_temporal, day_of_week])
                    else:
                        feature_list.extend([h_combined, day_of_week])
                    
                    y_aug = torch.cat(feature_list, dim=-1) # [W, S, M, F_aug]
                    
                    # Flatten for model: [WSM, F_aug]
                    y_aug_flat = y_aug.view(WSM, -1)

                    if is_generative_model:
                        samples_flat = temporal_model(y_aug_flat, num_steps=num_steps)
                        samples = samples_flat.view(WSNV)
                        mu_pred = samples
                        var_pred = torch.zeros_like(samples)
                    else:
                        mu_pred_flat, var_pred_flat = temporal_model(y_aug_flat)
                        # Reshape to 3D [B*S, M, V] for sample_mixed which expects 3 dimensions
                        mu_pred = mu_pred_flat.view(WS, num_msoas, num_prediction)
                        var_pred = var_pred_flat.view(WS, num_msoas, num_prediction)
                        
                        types = getattr(dataset_directory, 'prediction_distribution_types', ['nb'] * num_prediction)
                        samples_3d = sample_mixed(
                            mu_pred=mu_pred, var_pred=var_pred, types=types,
                            parameterization=getattr(temporal_model, 'parameterization', 'natural'),
                            transform=getattr(temporal_model, 'transform_type', 'softplus')
                        ) # [WS, M, V]
                        
                        # Reshape back to 4D [W, S, M, V] for storage
                        samples = samples_3d.view(WSNV)
                        mu_pred = mu_pred.view(WSNV)
                        var_pred = var_pred.view(WSNV)

                    # Store results (masking by validity if needed, but here we just fill)
                    batch_samples_tensor[:, :, step] = samples
                    mu_preds_tensor[:, :, step] = mu_pred
                    var_preds_tensor[:, :, step] = var_pred

                    # Update context for next step using precomputed mapping.
                    for c_idx, v_idx in ctx_to_pred_idx.items():
                        p_raw = samples[:, :, :, v_idx:v_idx+1] # [W, S, M, 1]
                        
                        if hasattr(temporal_model, 'scale'):
                            p_scaled = temporal_model.scale(p_raw, indices=[v_idx])
                        else:
                            # Fallback to GPU-based z-score scaling if scalers are available in model
                            means = temporal_model.means if hasattr(temporal_model, 'means') else 0.0
                            stds = temporal_model.stds if hasattr(temporal_model, 'stds') else 1.0
                            p_scaled = (p_raw - means[v_idx]) / stds[v_idx]

                        # Cycle context: [W, S, M, L] 
                        contexts_multi_sample[c_idx] = torch.cat(
                            [contexts_multi_sample[c_idx][:, :, :, 1:], p_scaled], dim=3
                        )

                # Quantile levels — constant, so allocate once outside the per-window loop.
                q = torch.tensor([0.5, 0.025, 0.975, 0.25, 0.75], device=device)

                # Processing outputs for each window in the batch
                for i in range(W):
                    window_idx_global = batch_start + i
                    starting_timestep = starting_timesteps[i].item()
                    end_t = min(starting_timestep + 60, max_timestep)
                    num_steps_to_predict = end_t - starting_timestep
                    
                    # Truncate to actual horizon for this window: [S, T, M, V]
                    samples_win = batch_samples_tensor[i, :, :num_steps_to_predict] 
                    mu_win = mu_preds_tensor[i, :, :num_steps_to_predict]
                    var_win = var_preds_tensor[i, :, :num_steps_to_predict]
                    
                    # GPU-native Quantiles: [5, T, M, V]
                    quantiles = torch.quantile(samples_win, q, dim=0)
                    preds_unscaled = quantiles[0]
                    preds_lower_95 = quantiles[1]
                    preds_upper_95 = quantiles[2]
                    preds_lower_50 = quantiles[3]
                    preds_upper_50 = quantiles[4]

                    # Ground Truth slice for this window: [T, M, V]
                    truth_win = truth_unscaled_gpu[starting_timestep : starting_timestep + num_steps_to_predict, :num_msoas, :]
                    
                    # Torch-Native Scoring (Vectorized across T, M, V)
                    is_scores_win = interval_score_torch(truth_win, preds_lower_95, preds_upper_95)
                    crps_scores_win = crps_torch(truth_win, samples_win)
                    
                    # ES and Variogram: Vectorized across T for each prediction variable
                    es_scores_win = torch.zeros(num_steps_to_predict, num_prediction, device=device)
                    vario_scores_win = torch.zeros(num_steps_to_predict, num_prediction, device=device)
                    for v in range(num_prediction):
                        es_scores_win[:, v] = energy_score_torch(truth_win[:, :, v], samples_win[:, :, :, v])
                        vario_scores_win[:, v] = variogram_torch(truth_win[:, :, v], samples_win[:, :, :, v], p=0.5)

                    # Log-transformed evaluation
                    truth_win_log = torch.log1p(truth_win)
                    preds_lower_95_log = torch.log1p(preds_lower_95)
                    preds_upper_95_log = torch.log1p(preds_upper_95)
                    samples_win_log = torch.log1p(samples_win)

                    is_scores_win_log = interval_score_torch(truth_win_log, preds_lower_95_log, preds_upper_95_log)
                    crps_scores_win_log = crps_torch(truth_win_log, samples_win_log)

                    es_scores_win_log = torch.zeros(num_steps_to_predict, num_prediction, device=device)
                    vario_scores_win_log = torch.zeros(num_steps_to_predict, num_prediction, device=device)
                    for v in range(num_prediction):
                        es_scores_win_log[:, v] = energy_score_torch(truth_win_log[:, :, v], samples_win_log[:, :, :, v])
                        vario_scores_win_log[:, v] = variogram_torch(truth_win_log[:, :, v], samples_win_log[:, :, :, v], p=0.5)

                    ###################################################################
                    # Batched GPU → CPU transfer.                                     #
                    # Group tensors by shape, stack, do one .cpu().numpy() per group.  #
                    # This reduces 13 individual CUDA sync points down to 3.           #
                    ###################################################################

                    # Group 1: shape [S, T, M, V] — sample-level tensors (3 tensors → 1 transfer)
                    stv_stack = torch.stack([samples_win, mu_win, var_win], dim=0)  # [3, S, T, M, V]
                    stv_np = stv_stack.cpu().numpy()
                    samples_win_np = stv_np[0]
                    mu_win_np      = stv_np[1]
                    var_win_np     = stv_np[2]

                    # Group 2: shape [T, M, V] — per-node prediction & scoring tensors (10 tensors → 1 transfer)
                    tmv_stack = torch.stack([
                        preds_unscaled, preds_lower_95, preds_upper_95,
                        preds_lower_50, preds_upper_50, truth_win,
                        is_scores_win, crps_scores_win,
                        is_scores_win_log, crps_scores_win_log
                    ], dim=0)  # [10, T, M, V]
                    tmv_np = tmv_stack.cpu().numpy()
                    preds_unscaled_np = tmv_np[0]
                    preds_lower_95_np = tmv_np[1]
                    preds_upper_95_np = tmv_np[2]
                    preds_lower_50_np = tmv_np[3]
                    preds_upper_50_np = tmv_np[4]
                    truth_win_np      = tmv_np[5]
                    is_scores_np      = tmv_np[6]
                    crps_scores_np    = tmv_np[7]
                    is_scores_log_np  = tmv_np[8]
                    crps_scores_log_np = tmv_np[9]

                    # Group 3: shape [T, V] — spatial scoring tensors (4 tensors → 1 transfer)
                    tv_stack = torch.stack([es_scores_win, vario_scores_win, es_scores_win_log, vario_scores_win_log], dim=0)  # [4, T, V]
                    tv_np = tv_stack.cpu().numpy()
                    es_scores_np        = tv_np[0]
                    vario_scores_np     = tv_np[1]
                    es_scores_log_np    = tv_np[2]
                    vario_scores_log_np = tv_np[3]

                    # Prepare and save Feather tables
                    T_win = num_steps_to_predict
                    S_win = num_samples
                    M_win = num_msoas
                    V_win = num_prediction
                    
                    # Prediction Table
                    t_grid = np.repeat(np.arange(starting_timestep, starting_timestep + T_win), M_win)
                    m_grid = np.tile(np.arange(M_win), T_win)
                    is_window_start = (t_grid == starting_timestep).astype(np.int8)
                    
                    dynamic_cols = {}
                    for j, name in enumerate(prediction_names):
                        dynamic_cols[f"{name}_unscaled"] = preds_unscaled_np[:, :, j].reshape(-1)
                        dynamic_cols[f"{name}_unscaled_gt"] = truth_win_np[:, :, j].reshape(-1)
                        dynamic_cols[f"{name}_lower_95"] = preds_lower_95_np[:, :, j].reshape(-1)
                        dynamic_cols[f"{name}_upper_95"] = preds_upper_95_np[:, :, j].reshape(-1)
                        dynamic_cols[f"{name}_lower_50"] = preds_lower_50_np[:, :, j].reshape(-1)
                        dynamic_cols[f"{name}_upper_50"] = preds_upper_50_np[:, :, j].reshape(-1)
                        dynamic_cols[f"{name}_IS_95"] = is_scores_np[:, :, j].reshape(-1)
                        dynamic_cols[f"{name}_CRPS"] = crps_scores_np[:, :, j].reshape(-1)
                        dynamic_cols[f"{name}_IS_95_log"] = is_scores_log_np[:, :, j].reshape(-1)
                        dynamic_cols[f"{name}_CRPS_log"] = crps_scores_log_np[:, :, j].reshape(-1)

                    pred_tbl = pa.table({"timestep": t_grid, "msoa": m_grid, "is_window_start": is_window_start, **dynamic_cols})
                    
                    prediction_folder = f"{output_folder}/predictions/{starting_timestep}"
                    os.makedirs(prediction_folder, exist_ok=True)
                    feather.write_feather(pred_tbl, f"{prediction_folder}/predictions.feather", compression="lz4")

                    # Sample Spaghetti Table
                    vals = samples_win_np.reshape(-1)
                    mu_vals = mu_win_np.reshape(-1)
                    var_vals = var_win_np.reshape(-1)
                    
                    s_idx_col = np.repeat(np.arange(S_win), T_win * M_win * V_win)
                    t_idx_col = np.tile(np.repeat(np.arange(T_win), M_win * V_win), S_win)
                    m_idx_col = np.tile(np.repeat(np.arange(M_win), V_win), S_win * T_win)
                    v_idx_col = np.tile(np.arange(V_win), S_win * T_win * M_win)
                    
                    sp_tbl = pa.table({
                        "window_start": np.full(S_win*T_win*M_win*V_win, starting_timestep, dtype=np.int32),
                        "timestep": (starting_timestep + t_idx_col).astype(np.int32),
                        "window_step": t_idx_col.astype(np.int32),
                        "msoa": m_idx_col.astype(np.int32),
                        "msoa_name": pa.array(msoa_name_map[m_idx_col]),
                        "sample_idx": s_idx_col.astype(np.int32),
                        "burden_type": pa.array(burden_map[v_idx_col]),
                        "value": vals,
                        "mu_pred": mu_vals,
                        "var_pred": var_vals,
                    })
                    feather.write_feather(sp_tbl, f"{prediction_folder}/sample_spaghetti.feather", compression="lz4")

                    # Spatial Scores Table
                    es_tbls = []
                    ts_vec = np.arange(starting_timestep, starting_timestep + T_win)
                    for v, bname in enumerate(prediction_names):
                        es_tbls.append(pa.table({
                            "timestep": ts_vec,
                            "burden": pa.array([bname] * T_win),
                            "energy_score": es_scores_np[:, v],
                            "variogram_score_p05": vario_scores_np[:, v],
                            "energy_score_log": es_scores_log_np[:, v],
                            "variogram_score_p05_log": vario_scores_log_np[:, v],
                        }))
                    feather.write_feather(pa.concat_tables(es_tbls), f"{prediction_folder}/spatial_scores.feather", compression="lz4")

                    window_time = time.time() - window_start_time
                    window_times_seconds.append(window_time)

            timing_table = pa.table({"window_times_seconds": pa.array(window_times_seconds)})
            feather.write_feather(timing_table, os.path.join(output_folder, "window_times.feather"), compression="lz4")

