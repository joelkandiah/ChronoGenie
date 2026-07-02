#!/usr/bin/env python3
"""
Chronos-only sliding-window autoregressive testing with spatial neighbour context.

Each anchor MSOA is evaluated from a per-step frozen context snapshot. Inputs are
packed as self + unioned neighbour sequences, with variable row counts allowed per
anchor because the neighbour sets are the union of feature-space and geographic
9-NN sets.

MSOAs are processed in batches of *window_batch_size* per predict_quantiles call.
After each 1-step prediction the sampled self-block values are written back into
the rolling predictions tensor for the next autoregressive step.
"""

import os
import time
from collections import defaultdict

import numpy as np
import torch
import pyarrow as pa
from pyarrow import feather

from chronos_adapter import (
    build_anchor_spatial_inputs,
    build_neighbor_map,
    sample_from_quantiles,
    NUM_SPATIAL_NEIGHBORS,
)
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
    """Run the autoregressive sliding-window inference loop.

    For each of *autoregressive_window_size* daily steps the function:
      1. Processes all MSOAs in batches of *window_batch_size* (isolated attention).
      2. For each MSOA batch, stacks self+9-neighbor sequences per MSOA and calls
         temporal_model.predict_quantiles(prediction_length=1).
      3. Draws *num_samples* ICDF trajectories; injects path-0 into the rolling
         context for the next step (true stochastic AR feedback).
      4. Accumulates samples and ground truth, then serialises feather outputs.
    """
    del interaction_encoder, graph_data, include_latest_context
    del spatial_ablation, spatial_encoder_type, num_steps, windows

    device = (
        predictions.device
        if hasattr(predictions, "device")
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # 1. Column/index bookkeeping
    prediction_names = list(dataset_directory.columns_for_prediction)
    context_names    = list(dataset_directory.columns_for_context)
    num_prediction   = len(prediction_names)
    num_msoas        = dataset_directory.num_geographies

    msoa_name_map = dataset_directory.df_geo_metadata["MSOA"].to_numpy()

    prediction_indices = [
        i for i, col in enumerate(dataset_directory.columns_with_data)
        if col in prediction_names
    ]
    context_indices = [
        i for i, col in enumerate(dataset_directory.columns_with_data)
        if col in context_names
    ]

    if not prediction_indices or not context_indices:
        raise ValueError("Missing context or prediction columns in dataset metadata.")

    # Row positions inside the self-block that correspond to prediction columns
    pred_pos_in_context = [context_names.index(p) for p in prediction_names if p in context_names]

    pred_type_by_name = dict(
        zip(dataset_directory.columns_for_prediction, dataset_directory.prediction_distribution_types)
    )
    count_channel_positions = [
        i for i, name in enumerate(prediction_names)
        if pred_type_by_name.get(name, "lognormal").lower() != "lognormal"
    ]

    quantile_levels   = [0.025, 0.25, 0.5, 0.75, 0.975]
    quantile_levels_t = torch.tensor(quantile_levels, device=device)

    # 2. Build spatial neighbour map (once, deterministic union KNN)
    print(
        f"[TESTING] Building neighbour map "
        f"(k={NUM_SPATIAL_NEIGHBORS} per space, unioned feature+geo)..."
    )
    neighbor_map = build_neighbor_map(
        dataset_directory.static_features_tensor,
        raw_static_features_tensor=dataset_directory.raw_static_features_tensor,
        n_neighbors=NUM_SPATIAL_NEIGHBORS,
    )
    print(f"[TESTING] Neighbour map ready. MSOA-0 neighbours: {neighbor_map[0]}")

    sim_data_tracker   = defaultdict(lambda: defaultdict(dict))
    window_times_seconds = []

    # 3. Autoregressive loop -- one 1-step prediction per iteration
    for current_test_window in range(autoregressive_window_size):
        window_start_time = time.time()

        context_start = from_index + current_test_window
        context_end   = context_start + context_size
        pred_time_idx = from_index + context_size + current_test_window

        print(
            f"[TESTING] AR step {current_test_window + 1}/{autoregressive_window_size}  "
            f"ctx=[{context_start}, {context_end})  pred_t={pred_time_idx}"
        )

        for sim_id in dataset_directory.test_sims:
            sim_idx = dataset_directory.resolve_sim_idx(sim_id)
            sim_context = predictions[:, sim_idx].clone()

            # Accumulators: one entry per MSOA, filled across MSOA batches
            all_samples = [None] * num_msoas  # each [S, V]
            all_gt      = [None] * num_msoas  # each [V]
            step_feedback = torch.empty((num_msoas, num_prediction), device=device)

            # MSOA batching -- isolates Chronos attention per anchor MSOA
            for batch_start_m in range(0, num_msoas, window_batch_size):
                batch_end_m = min(batch_start_m + window_batch_size, num_msoas)
                msoa_batch  = list(range(batch_start_m, batch_end_m))

                # One variable-width [rows, context_size] tensor per MSOA in this batch
                series_tensors = [sim_context[idx, :, context_start:context_end] for idx in context_indices]
                inputs = build_anchor_spatial_inputs(
                    series_tensors=series_tensors,
                    anchor_ids=msoa_batch,
                    neighbor_map=neighbor_map,
                )

                with torch.no_grad():
                    quantiles_list, _ = temporal_model.predict_quantiles(
                        inputs=inputs,
                        prediction_length=1,
                        quantile_levels=quantile_levels,
                    )
                # quantiles_list[b] shape: [rows, pred_len=1, num_quantiles]

                for b, m in enumerate(msoa_batch):
                    q_tensor = quantiles_list[b]  # [10*V, 1, Q]

                    # Extract self-block prediction rows: [num_pred, 1, Q]
                    self_pred_q = q_tensor[pred_pos_in_context]

                    # Reshape to [T=1, M=1, V=num_pred, Q] for sample_from_quantiles
                    self_pred_4d = self_pred_q.permute(1, 0, 2).unsqueeze(1)

                    # Draw num_samples ICDF paths: [S, T=1, M=1, V]
                    sampled = sample_from_quantiles(
                        quantile_values=self_pred_4d,
                        quantile_levels=quantile_levels_t,
                        num_samples=num_samples,
                    )
                    sampled_vals = sampled[:, 0, 0, :]  # [S, V]

                    # Clamp / floor count channels
                    if count_channel_positions:
                        sampled_vals[:, count_channel_positions] = torch.floor(
                            torch.clamp(sampled_vals[:, count_channel_positions], min=0.0)
                        )

                    # AR feedback: inject path-0 (first ICDF sample) into rolling context
                    icdf_feedback = sampled_vals[0]  # [V]
                    step_feedback[m] = icdf_feedback

                    # Ground truth for this MSOA at this timestep
                    gt_vals = ground_truths[prediction_indices, sim_idx, m, pred_time_idx].clone()
                    if count_channel_positions:
                        gt_vals[count_channel_positions] = torch.floor(
                            torch.clamp(gt_vals[count_channel_positions], min=0.0)
                        )

                    all_samples[m] = sampled_vals  # [S, V]
                    all_gt[m]      = gt_vals        # [V]

            for m in range(num_msoas):
                for v_i, col_idx in enumerate(prediction_indices):
                    predictions[col_idx, sim_idx, m, pred_time_idx] = step_feedback[m, v_i]

            # Stack across all MSOAs
            samples_smv = torch.stack(all_samples, dim=1)  # [S, M, V]
            gt_mv       = torch.stack(all_gt,      dim=0)  # [M, V]

            sim_data_tracker[sim_id][pred_time_idx] = {
                "samples": samples_smv,  # [S, M, V]
                "gt":      gt_mv,        # [M, V]
            }

        window_times_seconds.append(time.time() - window_start_time)

    # 4. Post-process and serialise feather outputs
    print(f"\n[TESTING] Exporting results to Feather under {output_folder}...")
    q_indices = torch.tensor([0.5, 0.025, 0.975, 0.25, 0.75], device=device)

    for sim_id in dataset_directory.test_sims:
        sim_root_folder = os.path.join(output_folder, "TESTING", f"SIM_{sim_id}")

        for pred_time_idx, data in sim_data_tracker[sim_id].items():
            prediction_folder = os.path.join(sim_root_folder, "predictions", str(pred_time_idx))
            os.makedirs(prediction_folder, exist_ok=True)

            samples_win = data["samples"].unsqueeze(0)  # [1, S, M, V]
            truth_win   = data["gt"].unsqueeze(0)       # [1, M, V]

            # Quantile summaries
            quantiles_q    = torch.quantile(samples_win, q_indices, dim=1)  # [5, 1, M, V]
            preds_unscaled = quantiles_q[0][0]   # [M, V]
            preds_lower_95 = quantiles_q[1][0]
            preds_upper_95 = quantiles_q[2][0]
            preds_lower_50 = quantiles_q[3][0]
            preds_upper_50 = quantiles_q[4][0]

            truth_win_flat   = truth_win[0]    # [M, V]
            samples_win_flat = samples_win[0]  # [S, M, V]

            # Native-scale scoring
            is_scores_win   = interval_score_torch(truth_win_flat, preds_lower_95, preds_upper_95)
            crps_scores_win = crps_torch(truth_win_flat, samples_win_flat)

            temporal_dim     = truth_win.shape[0]   # 1
            es_scores_win    = torch.zeros(temporal_dim, num_prediction, device=device)
            vario_scores_win = torch.zeros(temporal_dim, num_prediction, device=device)
            for v in range(num_prediction):
                target      = truth_win[:, :, v]                        # [1, M]
                prediction  = samples_win[:, :, :, v].permute(1, 0, 2)  # [S, 1, M]
                es_scores_win[:, v]    = energy_score_torch(target, prediction)
                vario_scores_win[:, v] = variogram_torch(target, prediction, p=0.5)

            # Log-scale scoring
            truth_win_log   = torch.log1p(truth_win)
            samples_win_log = torch.log1p(samples_win)

            quantiles_log      = torch.quantile(samples_win_log, q_indices, dim=1)
            preds_lower_95_log = quantiles_log[1][0]
            preds_upper_95_log = quantiles_log[2][0]

            is_scores_win_log   = interval_score_torch(
                torch.log1p(truth_win_flat), preds_lower_95_log, preds_upper_95_log
            )
            crps_scores_win_log = crps_torch(
                torch.log1p(truth_win_flat), torch.log1p(samples_win_flat)
            )

            es_scores_win_log    = torch.zeros(temporal_dim, num_prediction, device=device)
            vario_scores_win_log = torch.zeros(temporal_dim, num_prediction, device=device)
            for v in range(num_prediction):
                target_log     = truth_win_log[:, :, v]
                prediction_log = samples_win_log[:, :, :, v].permute(1, 0, 2)
                es_scores_win_log[:, v]    = energy_score_torch(target_log, prediction_log)
                vario_scores_win_log[:, v] = variogram_torch(target_log, prediction_log, p=0.5)

            # Unified CPU arrays: [10, M, V] and [4, 1, V]
            tmv_stack = torch.stack([
                preds_unscaled, preds_lower_95, preds_upper_95,
                preds_lower_50, preds_upper_50, truth_win_flat,
                is_scores_win,  crps_scores_win,
                is_scores_win_log, crps_scores_win_log,
            ], dim=0).cpu().numpy()  # [10, M, V]

            tv_stack = torch.stack([
                es_scores_win, vario_scores_win,
                es_scores_win_log, vario_scores_win_log,
            ], dim=0).cpu().numpy()  # [4, 1, V]

            # A. predictions.feather
            t_grid          = np.full(num_msoas, pred_time_idx, dtype=np.int32)
            m_grid          = np.arange(num_msoas, dtype=np.int32)
            is_window_start = np.ones(num_msoas, dtype=np.int8)

            dynamic_cols = {}
            for j, name in enumerate(prediction_names):
                dynamic_cols[f"{name}_unscaled"]    = tmv_stack[0, :, j]
                dynamic_cols[f"{name}_unscaled_gt"] = tmv_stack[5, :, j]
                dynamic_cols[f"{name}_lower_95"]    = tmv_stack[1, :, j]
                dynamic_cols[f"{name}_upper_95"]    = tmv_stack[2, :, j]
                dynamic_cols[f"{name}_lower_50"]    = tmv_stack[3, :, j]
                dynamic_cols[f"{name}_upper_50"]    = tmv_stack[4, :, j]
                dynamic_cols[f"{name}_IS_95"]       = tmv_stack[6, :, j]
                dynamic_cols[f"{name}_CRPS"]        = tmv_stack[7, :, j]
                dynamic_cols[f"{name}_IS_95_log"]   = tmv_stack[8, :, j]
                dynamic_cols[f"{name}_CRPS_log"]    = tmv_stack[9, :, j]

            pred_tbl = pa.table({
                "timestep": t_grid, "msoa": m_grid,
                "is_window_start": is_window_start, **dynamic_cols,
            })
            feather.write_feather(
                pred_tbl,
                os.path.join(prediction_folder, "predictions.feather"),
                compression="lz4",
            )

            # B. sample_spaghetti.feather
            samples_np = samples_win_flat.cpu().numpy()  # [S, M, V]
            total_rows = num_samples * num_msoas * num_prediction

            s_idx_col = np.repeat(np.arange(num_samples), num_msoas * num_prediction)
            t_idx_col = np.zeros(total_rows, dtype=np.int32)
            m_idx_col = np.tile(np.repeat(np.arange(num_msoas), num_prediction), num_samples)
            v_idx_col = np.tile(np.arange(num_prediction), num_samples * num_msoas)

            sp_tbl = pa.table({
                "window_start": np.full(total_rows, pred_time_idx, dtype=np.int32),
                "timestep":     np.full(total_rows, pred_time_idx, dtype=np.int32),
                "window_step":  t_idx_col,
                "msoa":         m_idx_col,
                "msoa_name":    pa.array(msoa_name_map[m_idx_col]),
                "sample_idx":   s_idx_col.astype(np.int32),
                "burden_type":  pa.array(np.array(prediction_names)[v_idx_col]),
                "value":        samples_np.reshape(-1),
                "mu_pred":      samples_np.reshape(-1),
                "var_pred":     np.zeros(total_rows, dtype=np.float32),
            })
            feather.write_feather(
                sp_tbl,
                os.path.join(prediction_folder, "sample_spaghetti.feather"),
                compression="lz4",
            )

            # C. spatial_scores.feather
            es_tbls = []
            for v, bname in enumerate(prediction_names):
                es_tbls.append(pa.table({
                    "timestep":                np.array([pred_time_idx], dtype=np.int32),
                    "burden":                  pa.array([bname]),
                    "energy_score":            tv_stack[0, :, v],
                    "variogram_score_p05":     tv_stack[1, :, v],
                    "energy_score_log":        tv_stack[2, :, v],
                    "variogram_score_p05_log": tv_stack[3, :, v],
                }))
            feather.write_feather(
                pa.concat_tables(es_tbls),
                os.path.join(prediction_folder, "spatial_scores.feather"),
                compression="lz4",
            )

        # Timing summary for this sim
        timing_table = pa.table({"window_times_seconds": pa.array(window_times_seconds)})
        feather.write_feather(
            timing_table,
            os.path.join(sim_root_folder, "window_times.feather"),
            compression="lz4",
        )

    print("[TESTING] Finished compiling and tracking predictions.")
    return predictions
