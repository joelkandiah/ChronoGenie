import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import pyarrow.feather as feather

"""
Typical Burden Names:
- daily_hospitalised
- deaths
- daily_infected
- Rt2 (or R_t)

Output Options:
- <output_prefix>_energy.png: Comparison based on Energy Score (spatial score)
- <output_prefix>_crps.png: Comparison based on CRPS (probabilistic score)
- <output_prefix>_variogram.png: Comparison based on Variogram Score (spatial correlation score)
"""

def ensure_parent_dir(file_path):
    """Create parent directory for a file path if needed."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

def resolve_output_base(output_prefix):
    """Resolve an output root directory and filename stem from a prefix or directory path."""
    prefix_path = Path(output_prefix)
    raw_prefix = str(output_prefix)

    if prefix_path.suffix:
        output_root = prefix_path.parent / prefix_path.stem
        output_stem = prefix_path.stem
    elif os.sep in raw_prefix or (os.altsep and os.altsep in raw_prefix):
        output_root = prefix_path
        output_stem = prefix_path.name
    else:
        output_root = Path(".")
        output_stem = prefix_path.name

    output_root.mkdir(parents=True, exist_ok=True)
    return output_root, output_stem

def load_all_scores(results_dir, score_type='energy', burden='daily_hospitalised', use_log=False):
    """
    Load scores from all simulations and windows in a results directory efficiently using os.scandir.
    """
    results_path = Path(results_dir)
    testing_path = results_path / "TESTING"
    
    if not testing_path.exists():
        print(f"Warning: TESTING directory not found in {results_dir}")
        return pd.DataFrame()
        
    all_data = []
    
    # Use os.scandir for faster directory traversal
    with os.scandir(testing_path) as it:
        sim_dirs = [entry for entry in it if entry.is_dir() and entry.name.startswith("SIM_")]
        sim_dirs.sort(key=lambda x: x.name)
        
        print(f"Processing {len(sim_dirs)} simulations in {results_dir}...")
        for sim_entry in tqdm(sim_dirs, desc=f"Loading {score_type} (use_log={use_log})"):
            sim_id = sim_entry.name
            pred_root = os.path.join(sim_entry.path, "predictions")
            
            if not os.path.exists(pred_root):
                continue
                
            # Iterate over window start directories
            with os.scandir(pred_root) as pred_it:
                for w_entry in pred_it:
                    if not w_entry.is_dir():
                        continue
                    
                    try:
                        window_start = int(w_entry.name)
                    except ValueError:
                        continue
                        
                    w_dir = Path(w_entry.path)
                        
                    if score_type == 'energy':
                        file_path = w_dir / "spatial_scores.feather"
                        if not file_path.exists():
                            continue
                        
                        col = 'energy_score_log' if use_log else 'energy_score'
                        try:
                            table = feather.read_table(file_path, columns=['burden', col])
                            df = table.to_pandas()
                            burden_df = df[df['burden'] == burden]
                            if not burden_df.empty:
                                score = burden_df[col].mean()
                                all_data.append({
                                    'sim_id': sim_id,
                                    'window_start': window_start,
                                    'score': score
                                })
                        except Exception:
                            continue

                    elif score_type == 'crps':
                        file_path = w_dir / "predictions.feather"
                        if not file_path.exists():
                            continue
                        
                        col_name = f"{burden}_CRPS_log" if use_log else f"{burden}_CRPS"
                        try:
                            table = feather.read_table(file_path, columns=['msoa', col_name])
                            df = table.to_pandas()
                            msoa_scores = df.groupby('msoa')[col_name].mean().reset_index()
                            for _, row in msoa_scores.iterrows():
                                all_data.append({
                                    'sim_id': sim_id,
                                    'msoa': row['msoa'],
                                    'window_start': window_start,
                                    'score': row[col_name]
                                })
                        except Exception:
                            continue

                    elif score_type == 'variogram':
                        file_path = w_dir / "spatial_scores.feather"
                        if not file_path.exists():
                            continue

                        col = 'variogram_score_p05_log' if use_log else 'variogram_score_p05'
                        try:
                            table = feather.read_table(file_path, columns=['burden', col])
                            df = table.to_pandas()
                            burden_df = df[df['burden'] == burden]
                            if not burden_df.empty:
                                score = burden_df[col].mean()
                                all_data.append({
                                    'sim_id': sim_id,
                                    'window_start': window_start,
                                    'score': score
                                })
                        except Exception:
                            continue
    
    return pd.DataFrame(all_data)

def calculate_win_rates(df1, df2, score_type='energy'):
    """
    Calculate the percentage of times model1 performed better than model2.
    """
    if df1.empty or df2.empty:
        return pd.DataFrame()
        
    print(f"Merging data for {score_type} comparison...")
    if score_type in ['energy', 'variogram']:
        merged = pd.merge(df1, df2, on=['sim_id', 'window_start'], suffixes=('_1', '_2'))
    else: # crps
        merged = pd.merge(df1, df2, on=['sim_id', 'msoa', 'window_start'], suffixes=('_1', '_2'))
        
    if merged.empty:
        print("No matching simulations/windows found between the two directories.")
        return pd.DataFrame()
        
    merged['model1_wins'] = merged['score_1'] < merged['score_2']
    merged['model2_wins'] = merged['score_2'] < merged['score_1']
    
    win_rates = merged.groupby('window_start').agg({
        'model1_wins': 'mean',
        'model2_wins': 'mean'
    }).reset_index()
    
    win_rates['model1_wins'] *= 100
    win_rates['model2_wins'] *= 100
    
    return win_rates

def plot_comparison(win_rates, title, xlabel, model1_name, model2_name, output_path):
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid", {'axes.grid': True, 'grid.linestyle': '--'})
    
    # Sort by window_start to ensure smooth lines
    win_rates = win_rates.sort_values('window_start')
    
    plt.plot(win_rates['window_start'], win_rates['model1_wins'], label=model1_name, color='black', linewidth=2)
    plt.plot(win_rates['window_start'], win_rates['model2_wins'], label=model2_name, color='black', linestyle='--', linewidth=1.5)
    
    plt.ylim(0, 100)
    plt.axhline(50, color='gray', linestyle=':', alpha=0.5)
    
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("% of cases where model performed best", fontsize=12)
    plt.legend(frameon=True, loc='upper right')
    
    plt.tight_layout()
    ensure_parent_dir(output_path)
    plt.savefig(output_path, dpi=300)
    print(f"Saved plot to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Compare two GENIE models using win-rate plots.")
    parser.add_argument("--dir1", type=str, required=True, help="Directory for model 1 results")
    parser.add_argument("--dir2", type=str, required=True, help="Directory for model 2 results")
    parser.add_argument("--name1", type=str, default="Model 1", help="Name for model 1")
    parser.add_argument("--name2", type=str, default="Model 2", help="Name for model 2")
    parser.add_argument("--burden", type=str, default="daily_hospitalised", help="Burden to evaluate (e.g. daily_hospitalised, deaths)")
    parser.add_argument("--output_prefix", type=str, default="comparison", help="Prefix or directory for output plots")
    
    args = parser.parse_args()

    output_root, output_stem = resolve_output_base(args.output_prefix)
    
    for use_log in [False, True]:
        log_prefix = "_log" if use_log else ""
        log_title = " (Log Data)" if use_log else ""
        
        # Energy Score Comparison
        print(f"\n--- Energy Score Comparison (use_log={use_log}) ---")
        e1 = load_all_scores(args.dir1, 'energy', args.burden, use_log=use_log)
        e2 = load_all_scores(args.dir2, 'energy', args.burden, use_log=use_log)
        
        if not e1.empty and not e2.empty:
            e_wins = calculate_win_rates(e1, e2, 'energy')
            if not e_wins.empty:
                plot_comparison(
                    e_wins, 
                    f"Energy Score Win Rate{log_title} ({args.burden})", 
                    "Forecast Initiation Day (window start)", 
                    args.name1, args.name2, 
                    str(output_root / f"{output_stem}_energy{log_prefix}.png")
                )
        
        # CRPS Comparison
        print(f"\n--- CRPS Comparison (use_log={use_log}) ---")
        c1 = load_all_scores(args.dir1, 'crps', args.burden, use_log=use_log)
        c2 = load_all_scores(args.dir2, 'crps', args.burden, use_log=use_log)
        
        if not c1.empty and not c2.empty:
            c_wins = calculate_win_rates(c1, c2, 'crps')
            if not c_wins.empty:
                plot_comparison(
                    c_wins, 
                    f"CRPS Win Rate{log_title} ({args.burden})", 
                    "Forecast Initiation Day (window start)", 
                    args.name1, args.name2, 
                    str(output_root / f"{output_stem}_crps{log_prefix}.png")
                )

        # Variogram Score Comparison
        print(f"\n--- Variogram Score Comparison (use_log={use_log}) ---")
        v1 = load_all_scores(args.dir1, 'variogram', args.burden, use_log=use_log)
        v2 = load_all_scores(args.dir2, 'variogram', args.burden, use_log=use_log)

        if not v1.empty and not v2.empty:
            v_wins = calculate_win_rates(v1, v2, 'variogram')
            if not v_wins.empty:
                plot_comparison(
                    v_wins, 
                    f"Variogram Score Win Rate{log_title} ({args.burden})", 
                    "Forecast Initiation Day (window start)", 
                    args.name1, args.name2, 
                    str(output_root / f"{output_stem}_variogram{log_prefix}.png")
                )

if __name__ == "__main__":
    main()
