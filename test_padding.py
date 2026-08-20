import sys
import os
import torch

# Ensure the script can see dataset.py and run_experiment.py
sys.path.append(os.getcwd())

from run_experiment import load_experiment_config, build_dataset_directory
from dataset import ProcessedData

def inspect_real_dataset(config_path):
    print(f"Loading experiment configuration from: {config_path}...")
    
    # 1. Force min_timestep to 0 so we can test t=0 with padding
    cfg = load_experiment_config(config_path)
    cfg.min_timestep = 0 
    
    print(f"Building DatasetDirectory for: {cfg.name}...")
    dataset_directory = build_dataset_directory(cfg)
    
    # 2. Instantiate the actual PyTorch ProcessedData for validation
    # Use your chosen past_context_size (e.g., 30)
    context_size = cfg.context_size if hasattr(cfg, 'context_size') else 30
    val_dataset = ProcessedData(
        data_directory=dataset_directory, 
        past_context_size=context_size, 
        type="validation"
    )
    
    print("\n" + "="*50)
    print("SUCCESSFULLY INITIALIZED REAL DATASET")
    print("="*50)
    print(f"Number of simulations in Validation split: {len(val_dataset.sims)}")
    print(f"Total extractable items: {len(val_dataset)}")
    
    # 3. Pull the very first item (Index 0 maps to Sim #1 at t=0)
    print(f"\n---> Fetching item at index 0 (Simulation Start, t=0) <---")
    sample = val_dataset[0]
    
    # Unpack based on your exact dataset.py return signature:
    # (static_features, current_t, current_dow, *contexts, *targets)
    static_features = sample[0]
    current_t = sample[1]
    current_dow = sample[2]
    
    # Contexts and targets are appended dynamically at the end
    num_ctx_vars = len(val_dataset.ctx_col_indices)
    num_pred_vars = len(val_dataset.pred_col_indices)
    
    contexts = sample[3 : 3 + num_ctx_vars]
    targets = sample[3 + num_ctx_vars : 3 + num_ctx_vars + num_pred_vars]
    
    print(f"Extracted Timepoint `current_t`: {current_t}  (Should be 0)")
    print(f"Day of Week `current_dow`: {current_dow}")
    print(f"Static Features Tensor Shape: {list(static_features.shape)}")
    
    print(f"\n--- Context Vectors for t=0 (Context Size: {context_size}) ---")
    for idx, ctx in enumerate(contexts):
        col_name = val_dataset.columns_for_context[idx]
        print(f"Variable '{col_name}' Context Shape: {list(ctx.shape)}")
        print(f"Is context entirely zeros? {torch.all(ctx == 0).item()}")
        
    print(f"\n--- Target Vectors at t=0 ---")
    for idx, tgt in enumerate(targets):
        col_name = val_dataset.columns_for_prediction[idx]
        print(f"Variable '{col_name}' Target Shape: {list(tgt.shape)}")
        print(f"Sample values (first 5 geographies): {tgt[:5].tolist()}")
        
    return dataset_directory, val_dataset

if __name__ == "__main__":
    # Point this to whichever experiment config triggers your data config
    # Example: "configs/experiments/your_experiment.yaml"
    target_config = "configs/chronos2_test.yaml" 
    
    if not os.path.exists(target_config):
        print(f"\n[Error] Could not find {target_config}.")
        print("Please modify the `target_config` path in this script to point to an experiment yaml.")
    else:
        inspect_real_dataset(target_config)