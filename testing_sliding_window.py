#!/usr/bin/env python3
"""
Chronos-only sliding-window autoregressive testing.
"""

import os
from collections import defaultdict

import numpy as np
import torch

from chronos_adapter import pack_chronos_input, sample_from_quantiles, unpack_prediction_blocks
from proper_scoring_torch import compute_scores_torch


def _save_chronos_window_outputs(
    output_folder,
    dataset_directory,
    prediction_indices,
    current_test_window,
    global_predictions,
    global_ground_truth,
    actual_dates,
    from_index,
):
    os.makedirs(output_folder, exist_ok=True)

    for local_horizon in range(global_predictions.shape[2]):
        absolute_horizon = current_test_window + local_horizon

        all_data = []
        for global_sim_idx, sim_id in enumerate(dataset_directory.test_sims):
            for col_pos, pred_idx in enumerate(prediction_indices):
                col_name = dataset_directory.columns_with_data[pred_idx]
                for geo_idx in range(dataset_directory.num_geographies):
                    all_data.append(
                        {
                            "date": actual_dates[global_sim_idx],
                            "sim": sim_id,
                            "horizon": absolute_horizon,
                            "column": col_name,
                            "msoa": dataset_directory.geographies[geo_idx],
                            "ground_truth": float(
                                global_ground_truth[global_sim_idx, col_pos, local_horizon, geo_idx]
                            ),
                            "prediction": float(
                                global_predictions[global_sim_idx, col_pos, local_horizon, geo_idx]
                            ),
                        }
                    )

        output_path = os.path.join(
            output_folder,
            f"predictions_timestep_{from_index + absolute_horizon}.csv",
        )

        import pandas as pd

        pd.DataFrame(all_data).to_csv(output_path, index=False)


def create_test_chunks(test_sims, chunk_size):
    test_sims_sorted = sorted(test_sims)
    chunks = []
    for i in range(0, len(test_sims_sorted), chunk_size):
        chunks.append(test_sims_sorted[i : i + chunk_size])
    return chunks


def testing_sliding_window(
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

    spatial_scores = defaultdict(list)

    prediction_indices = [
        i
        for i, col in enumerate(dataset_directory.columns_with_data)
        if col in dataset_directory.columns_for_prediction
    ]
    context_indices = [
        i
        for i, col in enumerate(dataset_directory.columns_with_data)
        if col in dataset_directory.columns_for_context
    ]

    if len(prediction_indices) == 0:
        raise ValueError("No prediction columns found")
    if len(context_indices) == 0:
        raise ValueError("No context columns found")

    pred_type_by_name = dict(
        zip(dataset_directory.columns_for_prediction, dataset_directory.prediction_distribution_types)
    )
    count_channel_positions = [
        i
        for i, pred_idx in enumerate(prediction_indices)
        if pred_type_by_name.get(dataset_directory.columns_with_data[pred_idx], "lognormal").lower() != "lognormal"
    ]

    test_chunks = create_test_chunks(dataset_directory.test_sims, max(1, window_batch_size))

    all_dates = sorted(dataset_directory.dataframe["date"].unique())

    for current_test_window in range(autoregressive_window_size):
        print(f"Testing timestep {from_index + current_test_window}")

        chunk_predictions = []
        chunk_ground_truths = []
        chunk_sim_ids = []

        for test_chunk in test_chunks:
            batch_predictions = []
            batch_ground_truths = []

            for sim_id in test_chunk:
                sim_idx = dataset_directory.sim_id_to_idx[sim_id]

                context_tensors = []
                for idx in context_indices:
                    context_start = from_index + current_test_window
                    context_end = context_start + context_size
                    context_tensors.append(
                        predictions[idx, sim_idx, context_start:context_end]
                    )

                packed_context = pack_chronos_input(
                    context_tensors,
                    dataset_directory.static_features_tensor,
                )

                quantile_levels = [0.1, 0.5, 0.9]
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
                    num_nodes=dataset_directory.num_geographies,
                )

                sampled = sample_from_quantiles(
                    quantile_values=prediction_blocks,
                    quantile_levels=torch.tensor(quantile_levels, device=prediction_blocks.device),
                    num_samples=num_samples,
                )
                sampled = sampled[:, 0].permute(2, 0, 1)

                if len(count_channel_positions) > 0:
                    sampled[count_channel_positions] = torch.floor(torch.clamp(sampled[count_channel_positions], min=0.0))

                means = sampled.mean(dim=1)
                pred_time_idx = from_index + context_size + current_test_window
                predictions[prediction_indices, sim_idx, pred_time_idx] = means

                gt = ground_truths[
                    prediction_indices,
                    sim_idx,
                    pred_time_idx,
                ]

                batch_predictions.append(sampled)
                batch_ground_truths.append(gt)

            if len(batch_predictions) == 0:
                continue

            pred_tensor = torch.stack(batch_predictions, dim=0)
            gt_tensor = torch.stack(batch_ground_truths, dim=0)

            if len(count_channel_positions) > 0:
                gt_tensor[:, count_channel_positions] = torch.floor(torch.clamp(gt_tensor[:, count_channel_positions], min=0.0))

            scores = compute_scores_torch(
                forecast_sample=pred_tensor,
                y=gt_tensor,
                alpha=0.05,
                score_type="all",
                weighted=False,
            )

            spatial_scores["interval_score"].append(scores[0].detach().cpu().numpy())
            spatial_scores["crps"].append(scores[1].detach().cpu().numpy())
            spatial_scores["energy_score"].append(scores[2].detach().cpu().numpy())
            spatial_scores["variogram_score"].append(scores[3].detach().cpu().numpy())

            chunk_predictions.append(pred_tensor.mean(dim=2).detach().cpu().numpy())
            chunk_ground_truths.append(gt_tensor.detach().cpu().numpy())
            chunk_sim_ids.extend(test_chunk)

        if len(chunk_predictions) == 0:
            continue

        pred_np = np.concatenate(chunk_predictions, axis=0)
        gt_np = np.concatenate(chunk_ground_truths, axis=0)

        sort_idx = np.argsort(np.array(chunk_sim_ids))
        pred_np = pred_np[sort_idx]
        gt_np = gt_np[sort_idx]

        sim_to_date = {}
        for sim_id in dataset_directory.test_sims:
            sim_idx = dataset_directory.sim_id_to_idx[sim_id]
            t = from_index + context_size + current_test_window
            if t >= dataset_directory.full_data_shape[1]:
                continue
            date_idx = int(dataset_directory.raw_data_tensor[0, sim_idx, t].item())
            if 0 <= date_idx < len(all_dates):
                sim_to_date[sim_id] = all_dates[date_idx]
            else:
                sim_to_date[sim_id] = None

        ordered_dates = [sim_to_date.get(sim_id) for sim_id in sorted(dataset_directory.test_sims)]

        _save_chronos_window_outputs(
            output_folder=output_folder,
            dataset_directory=dataset_directory,
            prediction_indices=prediction_indices,
            current_test_window=current_test_window,
            global_predictions=pred_np[:, :, None, :],
            global_ground_truth=gt_np[:, :, None, :],
            actual_dates=ordered_dates,
            from_index=from_index + context_size,
        )

    for metric_name, values in spatial_scores.items():
        if len(values) == 0:
            continue
        metric_array = np.stack(values, axis=0)
        np.save(os.path.join(output_folder, f"spatial_scores_{metric_name}.npy"), metric_array)

    return predictions
