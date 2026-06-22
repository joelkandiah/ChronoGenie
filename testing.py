import os
import random

import torch

from testing_sliding_window import get_sliding_window_predictions

random.seed(26)

def test(temporal_model, local_profile_encoder, interaction_encoder,
         graph_data,
         dataset_directory,
         context_size,
         autoregressive_window_size=60,
         device="cuda",
         output_folder="MODEL_OUTPUT",
         num_samples = 100,
         spatial_ablation=False,
         include_latest_context = False,
         spatial_encoder_type = "mlp",
         num_steps = 50,
         window_batch_size = 16):

    # Set models to evaluation mode
    temporal_model.eval()
    if interaction_encoder is not None:
        interaction_encoder.eval()

    if local_profile_encoder is not None:
        local_profile_encoder.eval()

    del local_profile_encoder, graph_data, device

    output_folder = os.path.join(output_folder, f"ARW_{autoregressive_window_size}")
    os.makedirs(output_folder, exist_ok=True)

    predictions = dataset_directory.raw_data_tensor.clone()
    ground_truths = dataset_directory.raw_data_tensor

    with torch.no_grad():
        get_sliding_window_predictions(
            temporal_model=temporal_model,
            interaction_encoder=interaction_encoder,
            graph_data=None,
            dataset_directory=dataset_directory,
            context_size=context_size,
            autoregressive_window_size=autoregressive_window_size,
            predictions=predictions,
            ground_truths=ground_truths,
            num_samples=num_samples,
            from_index=getattr(dataset_directory, "min_timestep", 0),
            include_latest_context=include_latest_context,
            num_steps=num_steps,
            windows=list(range(autoregressive_window_size)),
            output_folder=output_folder,
            spatial_ablation=spatial_ablation,
            spatial_encoder_type=spatial_encoder_type,
            window_batch_size=window_batch_size,
        )
