import os
import random
import torch

from torch.utils.data import DataLoader, Subset

from dataset import ProcessedData, get_testing_data
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
    
    all_test_sims = list(dataset_directory.test_sims)
    
    # Set models to evaluation mode
    temporal_model.eval()
    if interaction_encoder is not None:
        interaction_encoder.eval()

    if local_profile_encoder is not None:
        local_profile_encoder.eval()
    
    # Number of nodes (MSOAs)
    num_msoas = graph_data[0].num_nodes if graph_data[0] is not None else dataset_directory.num_geographies

    base_output_folder = output_folder

    # RESUME LOGIC (strict): a SIM is COMPLETE iff predictions/179 has both required files
    testing_root = os.path.join(base_output_folder, "TESTING")
    os.makedirs(testing_root, exist_ok=True)

    def _sim_is_complete_179(sim_dir: str) -> bool:
        req_dir = os.path.join(sim_dir, "predictions", "179")
        pred_fp = os.path.join(req_dir, "predictions.feather")
        score_fp = os.path.join(req_dir, "spatial_scores.feather")
        pred_ok = os.path.isfile(pred_fp)
        score_ok = os.path.isfile(score_fp)
        status = "COMPLETE" if (pred_ok and score_ok) else "PENDING"
        print(f"[RESUME] {os.path.basename(sim_dir)} -> {status} "
              f"(179/pred:{'Y' if pred_ok else 'N'}, 179/score:{'Y' if score_ok else 'N'})")
        return pred_ok and score_ok

    completed = []
    for sim in all_test_sims:
        sim_dir = os.path.join(testing_root, f"SIM_{sim}")
        if _sim_is_complete_179(sim_dir):
            completed.append(sim)

    pending_sims = [s for s in all_test_sims if s not in set(completed)]

    total = len(all_test_sims)
    done = len(completed)
    left = len(pending_sims)
    print(f"[TESTING] Sims total: {total} | done: {done} | left: {left}")

    if left == 0:
        print("[TESTING] All test simulations are already complete. Exiting.")
        return

    print(f"[TESTING] Will run {left} pending SIMs.")

    for sim in pending_sims:
        print(f"[TESTING] Computing predictions for SIM {sim}")
        # Loading the testing dataset with the specified context size
        testing_dataset = ProcessedData(data_directory=dataset_directory, past_context_size=context_size, type="testing")
        testing_dataset.sims = [sim]
        num_time_steps = testing_dataset.num_time_steps

        # Create the output folder for saving results
        sim_output_folder_sliding = os.path.join(base_output_folder, "TESTING", f"SIM_{sim}")

        window_starts = []
        current_start = 0

        while current_start + autoregressive_window_size <= num_time_steps:
            window_starts.append(current_start)
            current_start += autoregressive_window_size
        # Add the final window to cover remaining timesteps
        if current_start < num_time_steps:
            window_starts.append(current_start)

        # De-duplicate and sort
        window_starts = sorted(set([ws for ws in window_starts if ws < num_time_steps]))
        print(f"Adjusted window starts: {window_starts}")

        # Get ground truth data
        _, truth_unscaled = get_testing_data(testing_dataset, dataset_directory)
        
        sliding_window_starts = []
        sliding_start = 0
        while sliding_start + 1 <= num_time_steps: # and sliding_start < 60:
            sliding_window_starts.append(sliding_start)
            sliding_start += 1

        sliding_indices = [t for t in sliding_window_starts]

        test_subset = Subset(testing_dataset, sliding_indices)
        test_loader_sliding_window = DataLoader(test_subset, batch_size=len(sliding_indices), shuffle=False)
        with torch.no_grad():
            get_sliding_window_predictions(
            dataset_directory = dataset_directory,
            loader=test_loader_sliding_window,
            scalers=dataset_directory.scalers_for_prediction,
            num_msoas=num_msoas,
            device=device,
            temporal_model=temporal_model,
            spatial_encoder=local_profile_encoder, 
            temporal_encoder=interaction_encoder,    
            graph_data=graph_data,                  
            max_timestep=testing_dataset.num_time_steps,
            output_folder=sim_output_folder_sliding,
            truth_unscaled=truth_unscaled,
            num_samples=num_samples,
            spatial_ablation=spatial_ablation,
            include_latest_context = include_latest_context,
            spatial_encoder_type = spatial_encoder_type,
            num_steps = num_steps,
            window_batch_size = window_batch_size
        )
