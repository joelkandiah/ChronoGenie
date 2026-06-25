#!/usr/bin/env python3
"""
Chronos-only sliding-window autoregressive testing.
"""

import os
import time
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import pyarrow as pa
from pyarrow import feather

from chronos_adapter import pack_chronos_input, sample_from_quantiles, unpack_prediction_blocks
from proper_scoring_torch import (
    interval_score_torch, crps_torch, energy_score_torch, variogram_torch
)

def get_sliding_window_predictions(
    temporal_model,
    interaction_encoder,
    graph_data,
    dataset_directory,
    context_size,
    autoregressive_window_size,
    predictions,
    ground_truths,
    num_samples,
    from_index,
    include_latest_context,
    num_steps,
    windows,
    output_folder,
    spatial_ablation,
    spatial_encoder_type,
    window_batch_size,
):
    del interaction_encoder, graph_data, include_latest_context, spatial_ablation, spatial_encoder_type, num_steps, windows

    device = predictions.device if hasattr(predictions, "device") else "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Identify context and prediction channel mappings
    prediction_names = list(dataset_directory.columns_for_prediction)
    context_names = list(dataset_directory.columns_for_context)
    num_prediction = len(prediction_names)
    num_msoas = dataset_directory.num_geographies
    
    # FIX: Extract geographic identifiers via df_geo_metadata index structure
    msoa_name_map = dataset_directory.df_geo_metadata["MSOA"].to_numpy()

    prediction_indices = [i for i, col in enumerate(dataset_directory.columns_with_data) if col in prediction_names]
    context_indices = [i for i, col in enumerate(dataset_directory.columns_with_data) if col in context_names]

    if len(prediction_indices) == 0 or len(context_indices) == 0:
        raise ValueError("Missing context or prediction columns in dataset metadata.")

    pred_type_by_name = dict(zip(dataset_directory.columns_for_prediction, dataset_directory.prediction_distribution_types))
    count_channel_positions = [i for i, pred_idx in enumerate(prediction_indices) if pred_type_by_name.get(dataset_directory.columns_with_data[pred_idx], "lognormal").lower() != "lognormal"]

    # Tracking simulation-level metrics across steps
    sim_data_tracker = defaultdict(lambda: defaultdict(dict))
    window_times_seconds = []

    # 2. Autoregressive Sliding Window Loop
    for current_test_window in range(autoregressive_window_size):
        window_start_time = time.time()
        starting_timestep = from_index + current_test_window
        print(f"[TESTING] Processing window step {current_test_window + 1}/{autoregressive_window_size} (Global Timestep: {starting_timestep})")

        for sim_id in dataset_directory.test_sims:
            sim_idx = dataset_directory.resolve_sim_idx(sim_id)

            # Gather historical contexts
            context_tensors = []
            for idx in context_indices:
                context_start = from_index + current_test_window
                context_end = context_start + context_size
                context_tensors.append(predictions[idx, sim_idx, :, context_start:context_end])

            # Pack context and run inference
            packed_context = pack_chronos_input(context_tensors, dataset_directory.static_features_tensor)
            quantile_levels = [0.025, 0.25, 0.5, 0.75, 0.975]  # Matches required interval scopes (95% and 50%)
            
            with torch.no_grad():
                quantiles_list, _ = temporal_model.predict_quantiles(
                    inputs=[packed_context],
                    prediction_length=1,
                    quantile_levels=quantile_levels,
                )

                quantiles = quantiles_list[0]
                prediction_blocks = unpack_prediction_blocks(
                    quantile_tensor=quantiles,
                    context_names=dataset_directory.columns_for_context,
                    prediction_names=dataset_directory.columns_for_prediction,
                    num_nodes=num_msoas,
                )

                # Sample tracking
                sampled = sample_from_quantiles(
                    quantile_values=prediction_blocks,
                    quantile_levels=torch.tensor(quantile_levels, device=prediction_blocks.device),
                    num_samples=num_samples,
                )
                sampled = sampled[:, 0].permute(2, 0, 1)  # [V, S, M]

                if len(count_channel_positions) > 0:
                    sampled[count_channel_positions] = torch.floor(torch.clamp(sampled[count_channel_positions], min=0.0))

                # Update running context prediction tensor with the execution mean
                means = sampled.mean(dim=1)
                pred_time_idx = from_index + context_size + current_test_window
                predictions[prediction_indices, sim_idx, :, pred_time_idx] = means

                # Extract ground truth target
                gt = ground_truths[prediction_indices, sim_idx, :, pred_time_idx]
                if len(count_channel_positions) > 0:
                    gt[count_channel_positions] = torch.floor(torch.clamp(gt[count_channel_positions], min=0.0))

            # Store for metrics mapping out to Feather files
            sim_data_tracker[sim_id][pred_time_idx] = {
                "samples": sampled.permute(1, 2, 0), # [S, M, V]
                "gt": gt.permute(1, 0),              # [M, V]
            }

        window_times_seconds.append(time.time() - window_start_time)

    # 3. Post-Process and Save Predictions into standard Feather Templates per Simulation
    print(f"\n[TESTING] Exporting results to Feather format under {output_folder}...")
    q_indices = torch.tensor([0.5, 0.025, 0.975, 0.25, 0.75], device=device)

    for sim_id in dataset_directory.test_sims:
        sim_root_folder = os.path.join(output_folder, "TESTING", f"SIM_{sim_id}")
        
        for pred_time_idx, data in sim_data_tracker[sim_id].items():
            prediction_folder = os.path.join(sim_root_folder, "predictions", str(from_index + context_size))
            os.makedirs(prediction_folder, exist_ok=True)

            samples_win = data["samples"].unsqueeze(0)  # [1, S, M, V]
            truth_win = data["gt"].unsqueeze(0)        # [1, M, V]

            # Native Scoring metrics
            quantiles = torch.quantile(samples_win, q_indices, dim=1) # [5, 1, M, V]
            preds_unscaled = quantiles[0][0]
            preds_lower_95 = quantiles[1][0]
            preds_upper_95 = quantiles[2][0]
            preds_lower_50 = quantiles[3][0]
            preds_upper_50 = quantiles[4][0]
            
            truth_win_flat = truth_win[0]
            samples_win_flat = samples_win[0]

            print(f"DEBUG: truth_win shape: {truth_win.shape}")
            print(f"DEBUG: samples_win shape: {samples_win.shape}")

            is_scores_win = interval_score_torch(truth_win_flat, preds_lower_95, preds_upper_95)
            crps_scores_win = crps_torch(truth_win_flat, samples_win_flat)

            temporal_dim = truth_win.shape[0]

            # Spatial scoring structures
            es_scores_win = torch.zeros(temporal_dim, num_prediction, device=device)
            vario_scores_win = torch.zeros(temporal_dim, num_prediction, device=device)
            for v in range(num_prediction):
                # truth_win[:, :, v] is [Time, MSOA]
                # samples_win[:, :, :, v] is [Time, Samples, MSOA]
                target = truth_win[:, :, v].squeeze(0)          # Result: [MSOA]
                prediction = samples_win[:, :, :, v].squeeze(0) # Result: [Sample, MSOA]
                
                es_scores_win[:, v] = energy_score_torch(target, prediction)
                vario_scores_win[:, v] = variogram_torch(target, prediction, p=0.5)

            # Log-scale Transformations
            truth_win_log = torch.log1p(truth_win_flat)
            preds_lower_95_log = torch.log1p(preds_lower_95)
            preds_upper_95_log = torch.log1p(preds_upper_95)
            samples_win_log = torch.log1p(samples_win_flat)

            is_scores_win_log = interval_score_torch(truth_win_log, preds_lower_95_log, preds_upper_95_log)
            crps_scores_win_log = crps_torch(truth_win_log, samples_win_log)

            es_scores_win_log = torch.zeros(temporal_dim, num_prediction, device=device)
            vario_scores_win_log = torch.zeros(temporal_dim, num_prediction, device=device)
            for v in range(num_prediction):
                # truth_win[:, :, v] is [Time, MSOA]
                # samples_win[:, :, :, v] is [Time, Samples, MSOA]
                target = truth_win[:, :, v].squeeze(0)          # Result: [MSOA]
                prediction = samples_win[:, :, :, v].squeeze(0) # Result: [Sample, MSOA]
                es_scores_win_log[:, v] = energy_score_torch(torch.log1p(target), torch.log1p(prediction))
                vario_scores_win_log[:, v] = variogram_torch(torch.log1p(target), torch.log1p(prediction), p=0.5)

            # Unified stack arrays
            tmv_stack = torch.stack([
                preds_unscaled, preds_lower_95, preds_upper_95, preds_lower_50, preds_upper_50, truth_win_flat,
                is_scores_win, crps_scores_win, is_scores_win_log, crps_scores_win_log
            ], dim=0).cpu().numpy()

            tv_stack = torch.stack([es_scores_win, vario_scores_win, es_scores_win_log, vario_scores_win_log], dim=0).cpu().numpy()

            # --- A. Save predictions.feather ---
            t_grid = np.full(num_msoas, pred_time_idx, dtype=np.int32)
            m_grid = np.arange(num_msoas, dtype=np.int32)
            is_window_start = np.where(t_grid == (from_index + context_size), 1, 0).astype(np.int8)

            dynamic_cols = {}
            for j, name in enumerate(prediction_names):
                dynamic_cols[f"{name}_unscaled"] = tmv_stack[0, :, j]
                dynamic_cols[f"{name}_unscaled_gt"] = tmv_stack[5, :, j]
                dynamic_cols[f"{name}_lower_95"] = tmv_stack[1, :, j]
                dynamic_cols[f"{name}_upper_95"] = tmv_stack[2, :, j]
                dynamic_cols[f"{name}_lower_50"] = tmv_stack[3, :, j]
                dynamic_cols[f"{name}_upper_50"] = tmv_stack[4, :, j]
                dynamic_cols[f"{name}_IS_95"] = tmv_stack[6, :, j]
                dynamic_cols[f"{name}_CRPS"] = tmv_stack[7, :, j]
                dynamic_cols[f"{name}_IS_95_log"] = tmv_stack[8, :, j]
                dynamic_cols[f"{name}_CRPS_log"] = tmv_stack[9, :, j]

            pred_tbl = pa.table({"timestep": t_grid, "msoa": m_grid, "is_window_start": is_window_start, **dynamic_cols})
            feather.write_feather(pred_tbl, os.path.join(prediction_folder, "predictions.feather"), compression="lz4")

            # --- B. Save sample_spaghetti.feather ---
            samples_np = samples_win_flat.cpu().numpy()
            s_idx_col = np.repeat(np.arange(num_samples), num_msoas * num_prediction)
            t_idx_col = np.full(num_samples * num_msoas * num_prediction, pred_time_idx - (from_index + context_size), dtype=np.int32)
            m_idx_col = np.tile(np.repeat(np.arange(num_msoas), num_prediction), num_samples)
            v_idx_col = np.tile(np.arange(num_prediction), num_samples * num_msoas)

            sp_tbl = pa.table({
                "window_start": np.full(num_samples * num_msoas * num_prediction, from_index + context_size, dtype=np.int32),
                "timestep": np.full(num_samples * num_msoas * num_prediction, pred_time_idx, dtype=np.int32),
                "window_step": t_idx_col,
                "msoa": m_idx_col,
                "msoa_name": pa.array(msoa_name_map[m_idx_col]),
                "sample_idx": s_idx_col.astype(np.int32),
                "burden_type": pa.array(np.array(prediction_names)[v_idx_col]),
                "value": samples_np.reshape(-1),
                "mu_pred": samples_np.reshape(-1),
                "var_pred": np.zeros_like(samples_np).reshape(-1),
            })
            feather.write_feather(sp_tbl, os.path.join(prediction_folder, "sample_spaghetti.feather"), compression="lz4")

            # --- C. Save spatial_scores.feather ---
            es_tbls = []
            for v, bname in enumerate(prediction_names):
                es_tbls.append(pa.table({
                    "timestep": np.array([pred_time_idx], dtype=np.int32),
                    "burden": pa.array([bname]),
                    "energy_score": tv_stack[0, :, v],
                    "variogram_score_p05": tv_stack[1, :, v],
                    "energy_score_log": tv_stack[2, :, v],
                    "variogram_score_p05_log": tv_stack[3, :, v],
                }))
            feather.write_feather(pa.concat_tables(es_tbls), os.path.join(prediction_folder, "spatial_scores.feather"), compression="lz4")

        # Save timing summary 
        timing_table = pa.table({"window_times_seconds": pa.array(window_times_seconds)})
        feather.write_feather(timing_table, os.path.join(sim_root_folder, "window_times.feather"), compression="lz4")

    print("[TESTING] Finished compiling and tracking predictions.")
    return predictions