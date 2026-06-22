import os
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import pyarrow.feather as feather

def aggregate_spatial_scores(results_dir, output_csv, raw_csv=None):
    """
    Traverse the results directory, load all spatial_scores.feather files,
    and output an aggregated CSV table.
    """
    results_path = Path(results_dir)
    testing_path = results_path / "TESTING"
    
    if not testing_path.exists():
        # Fallback to checking if results_dir itself is the TESTING dir or contains SIM_ dirs
        if any(d.name.startswith("SIM_") for d in results_path.iterdir() if d.is_dir()):
            testing_path = results_path
        else:
            print(f"Error: Could not find SIM_ directories in {results_dir}")
            return
        
    all_data = []
    
    # Use os.scandir for faster directory traversal
    with os.scandir(testing_path) as it:
        sim_dirs = [entry for entry in it if entry.is_dir() and entry.name.startswith("SIM_")]
        sim_dirs.sort(key=lambda x: x.name)
        
        print(f"Processing {len(sim_dirs)} simulations in {results_dir}...")
        for sim_entry in tqdm(sim_dirs, desc="Loading Feather files"):
            sim_id = sim_entry.name
            pred_root = os.path.join(sim_entry.path, "predictions")
            
            if not os.path.exists(pred_root):
                continue
                
            with os.scandir(pred_root) as pred_it:
                for w_entry in pred_it:
                    if not w_entry.is_dir():
                        continue
                    
                    try:
                        window_start = int(w_entry.name)
                    except ValueError:
                        continue
                        
                    file_path = Path(w_entry.path) / "spatial_scores.feather"
                    if not file_path.exists():
                        continue
                    
                    try:
                        df = feather.read_feather(file_path)
                        df['sim_id'] = sim_id
                        df['window_start'] = window_start
                        all_data.append(df)
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
                        continue
    
    if not all_data:
        print("No spatial scores found.")
        return
        
    print("Concatenating data...")
    full_df = pd.concat(all_data, ignore_index=True)
    
    # Save raw data if requested
    if raw_csv:
        full_df.to_csv(raw_csv, index=False)
        print(f"Saved raw data to {raw_csv}")
        
    # Aggregate per simulation and burden
    # Columns typically include: timestep, burden, energy_score, variogram_score_p05, sim_id, window_start
    print("Calculating summary statistics...")
    agg_cols = {
        'energy_score': 'mean',
        'variogram_score_p05': 'mean'
    }
    rename_cols = {
        'energy_score': 'energy_score_mean',
        'variogram_score_p05': 'variogram_score_mean'
    }
    
    if 'energy_score_log' in full_df.columns:
        agg_cols['energy_score_log'] = 'mean'
        rename_cols['energy_score_log'] = 'energy_score_log_mean'
    if 'variogram_score_p05_log' in full_df.columns:
        agg_cols['variogram_score_p05_log'] = 'mean'
        rename_cols['variogram_score_p05_log'] = 'variogram_score_log_mean'
        
    summary_df = full_df.groupby(['sim_id', 'burden']).agg(agg_cols).reset_index()
    summary_df.rename(columns=rename_cols, inplace=True)
    
    summary_df.to_csv(output_csv, index=False)
    print(f"Saved summary table to {output_csv}")

def main():
    parser = argparse.ArgumentParser(description="Aggregate spatial scores from a results directory.")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory containing TESTING/SIM_*/predictions/*")
    parser.add_argument("--output", type=str, default="spatial_scores_summary.csv", help="Output summary CSV filename")
    parser.add_argument("--raw_output", type=str, default=None, help="Optional: Output raw joined data CSV filename")
    
    args = parser.parse_args()
    aggregate_spatial_scores(args.results_dir, args.output, args.raw_output)

if __name__ == "__main__":
    main()
