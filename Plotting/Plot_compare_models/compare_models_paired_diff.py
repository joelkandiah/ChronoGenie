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
Script to compare two models using paired differences.
Requested plots:
1. Paired difference distribution (NSF - MLP)
2. Rolling paired difference median
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
    
    with os.scandir(testing_path) as it:
        sim_dirs = [entry for entry in it if entry.is_dir() and entry.name.startswith("SIM_")]
        sim_dirs.sort(key=lambda x: x.name)
        
        print(f"Processing {len(sim_dirs)} simulations in {results_dir}...")
        for sim_entry in tqdm(sim_dirs, desc=f"Loading {score_type} (use_log={use_log})"):
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

def plot_stat_distribution(merged, stat_col, score_type, model1_name, model2_name, output_path, label="Difference"):
    """
    Plot the distribution of a statistic (e.g. Difference or Skill Score).
    """
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    data = merged[stat_col]
    
    sns.histplot(data, kde=True, color='blue', alpha=0.6)
    plt.axvline(0, color='red', linestyle='--')
    
    mean_val = data.mean()
    median_val = data.median()
    
    plt.axvline(mean_val, color='green', label=f'Mean {label}: {mean_val:.4f}')
    plt.axvline(median_val, color='orange', label=f'Median {label}: {median_val:.4f}')
    
    plt.title(f"{label} Distribution: {model1_name} vs {model2_name} ({score_type})", fontsize=14)
    plt.xlabel(f"{label} (Score 1: {model1_name}, Score 2: {model2_name})", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.legend()
    
    plt.tight_layout()
    ensure_parent_dir(output_path)
    plt.savefig(output_path, dpi=300)
    print(f"Saved distribution plot to {output_path}")

def plot_rolling_stat_over_time(merged, stat_col, score_type, model1_name, model2_name, output_path, rolling_window=7, label="Difference"):
    """
    Plot the median and 95% CI of a statistic over forecast windows.
    """
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    
    # Calculate median, 2.5th, and 97.5th percentiles per window_start
    window_stats = merged.groupby('window_start')[stat_col].agg([
        ('median', 'median'),
        ('q025', lambda x: np.percentile(x, 2.5)),
        ('q975', lambda x: np.percentile(x, 97.5))
    ]).reset_index().sort_values('window_start')
    
    if rolling_window > 1:
        # 1. Rolling Median of raw stats (Original implementation)
        window_stats['median_roll'] = window_stats['median'].rolling(window=rolling_window, center=True).median()
        window_stats['q025_roll'] = window_stats['q025'].rolling(window=rolling_window, center=True).median()
        window_stats['q975_roll'] = window_stats['q975'].rolling(window=rolling_window, center=True).median()
        
        plt.fill_between(window_stats['window_start'], window_stats['q025_roll'], window_stats['q975_roll'], color='red', alpha=0.1, label='95% CI (Rolling Median)')
        plt.plot(window_stats['window_start'], window_stats['median_roll'], color='red', linewidth=2, label=f'Rolling Median (window={rolling_window})')
        plt.plot(window_stats['window_start'], window_stats['median'], marker='o', alpha=0.3, color='blue', label='Median per Window')

        # 2. Rolling Mean taken over the trajectories before evaluating median and confidence interval
        group_cols = ['sim_id']
        if 'msoa' in merged.columns:
            group_cols.append('msoa')
            
        merged_sorted = merged.sort_values(group_cols + ['window_start']).copy()
        merged_sorted['stat_roll_mean'] = merged_sorted.groupby(group_cols)[stat_col].transform(
            lambda x: x.rolling(window=rolling_window, center=True, min_periods=1).mean()
        )
        
        roll_mean_stats = merged_sorted.groupby('window_start')['stat_roll_mean'].agg([
            ('median', 'median'),
            ('q025', lambda x: np.percentile(x, 2.5)),
            ('q975', lambda x: np.percentile(x, 97.5))
        ]).reset_index().sort_values('window_start')
        
        plt.fill_between(roll_mean_stats['window_start'], roll_mean_stats['q025'], roll_mean_stats['q975'], color='teal', alpha=0.15, label='95% CI (Rolling Mean)')
        plt.plot(roll_mean_stats['window_start'], roll_mean_stats['median'], color='teal', linestyle='--', linewidth=2, label=f'Rolling Mean (window={rolling_window})')
    else:
        plt.fill_between(window_stats['window_start'], window_stats['q025'], window_stats['q975'], color='red', alpha=0.1, label='95% CI')
        plt.plot(window_stats['window_start'], window_stats['median'], marker='o', color='red', linewidth=2, label=f'Median {label}')
        
    plt.axhline(0, color='black', linestyle='--')
    
    plt.title(f"Median {label} over Time: {model1_name} vs {model2_name} ({score_type})", fontsize=14)
    plt.xlabel("Forecast Initiation Day (window start)", fontsize=12)
    plt.ylabel(f"{label} ({model1_name}, {model2_name})", fontsize=12)
    plt.legend()
    
    plt.tight_layout()
    ensure_parent_dir(output_path)
    plt.savefig(output_path, dpi=300)
    print(f"Saved rolling plot to {output_path}")

def save_outperforming_samples(merged, score_type, model1_name, model2_name, output_prefix, top_n=20):
    """
    Identify and save cases where one model significantly outperforms the other.
    """
    # Model 1 outperforms Model 2 (Lowest diff, most negative)
    m1_better = merged.nsmallest(top_n, 'diff').copy()
    m1_better['outperformer'] = model1_name
    
    # Model 2 outperforms Model 1 (Highest diff, most positive)
    m2_better = merged.nlargest(top_n, 'diff').copy()
    m2_better['outperformer'] = model2_name
    
    combined = pd.concat([m1_better, m2_better])
    
    cols = ['sim_id', 'window_start', 'score_1', 'score_2', 'diff', 'skill_score', 'outperformer']
    if 'msoa' in merged.columns:
        cols.insert(1, 'msoa')
        
    output_path = f"{output_prefix}_{score_type}_outperformers.csv"
    ensure_parent_dir(output_path)
    combined[cols].to_csv(output_path, index=False)
    print(f"Saved outperforming samples to {output_path}")

def save_consistency_analysis(merged, score_type, model1_name, model2_name, output_prefix):
    """
    Analyze consistency of performance across windows for each sim/msoa.
    """
    group_cols = ['sim_id']
    if 'msoa' in merged.columns:
        group_cols.append('msoa')
        
    # Calculate win rate and mean diff/skill score per group
    # Note: win_rate is for model1_wins (diff < 0)
    consistency = merged.groupby(group_cols).agg(
        mean_diff=('diff', 'mean'),
        mean_skill_score=('skill_score', 'mean'),
        win_rate=('diff', lambda x: (x < 0).mean()),
        num_windows=('window_start', 'count')
    ).reset_index()
    
    # Sort by win_rate (1.0 means Model 1 always wins, 0.0 means Model 2 always wins)
    consistency = consistency.sort_values(['win_rate', 'mean_skill_score'], ascending=[False, False])
    
    output_path = f"{output_prefix}_{score_type}_consistency.csv"
    ensure_parent_dir(output_path)
    consistency.to_csv(output_path, index=False)
    print(f"Saved consistency analysis to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate paired difference and skill score plots for model comparison.")
    parser.add_argument("--dir1", type=str, required=True, help="Directory for model 1 results (e.g. NSF)")
    parser.add_argument("--dir2", type=str, required=True, help="Directory for model 2 results (e.g. MLP)")
    parser.add_argument("--name1", type=str, default="NSF", help="Name for model 1")
    parser.add_argument("--name2", type=str, default="MLP", help="Name for model 2")
    parser.add_argument("--burden", type=str, default="daily_hospitalised", help="Burden to evaluate")
    parser.add_argument("--output_prefix", type=str, default="paired_diff", help="Prefix or directory for output plots")
    parser.add_argument("--rolling", type=int, default=1, help="Rolling window size for the median plot")
    parser.add_argument("--top_n", type=int, default=20, help="Number of top outperforming cases to save")
    
    args = parser.parse_args()

    output_root, output_stem = resolve_output_base(args.output_prefix)
    
    for score_type in ['energy', 'crps', 'variogram']:
        for use_log in [False, True]:
            log_prefix = "_log" if use_log else ""
            log_title = " (Log Data)" if use_log else " (Natural Data)"
            
            print(f"\n--- Processing {score_type} (use_log={use_log}) ---")
            df1 = load_all_scores(args.dir1, score_type, args.burden, use_log=use_log)
            if not df1.empty:
                csv1 = str(output_root / f"{output_stem}_{score_type}{log_prefix}_{args.name1.replace(' ', '_')}.csv")
                ensure_parent_dir(csv1)
                df1.to_csv(csv1, index=False)
                print(f"Saved {args.name1} scores to {csv1}")

            df2 = load_all_scores(args.dir2, score_type, args.burden, use_log=use_log)
            if not df2.empty:
                csv2 = str(output_root / f"{output_stem}_{score_type}{log_prefix}_{args.name2.replace(' ', '_')}.csv")
                ensure_parent_dir(csv2)
                df2.to_csv(csv2, index=False)
                print(f"Saved {args.name2} scores to {csv2}")
            
            if df1.empty or df2.empty:
                print(f"Skipping {score_type} (use_log={use_log}) due to missing data.")
                continue
                
            on_cols = ['sim_id', 'window_start']
            if score_type == 'crps':
                on_cols.append('msoa')
                
            merged = pd.merge(df1, df2, on=on_cols, suffixes=('_1', '_2'))
            
            if merged.empty:
                print(f"No matching data found for {score_type} (use_log={use_log}).")
                continue
                
            # Calculate Difference and Skill Score
            # Diff = S1 - S2 (Negative is better if S1 is new model)
            merged['diff'] = merged['score_1'] - merged['score_2']
            # Skill Score = 1 - S1/S2 (Positive is better if S1 is new model)
            # Avoid division by zero
            merged = merged[merged['score_2'] != 0].copy()
            merged['skill_score'] = 1 - (merged['score_1'] / merged['score_2'])
            
            # 1. Difference Distribution Plot
            plot_stat_distribution(
                merged, 'diff', score_type + log_title, args.name1, args.name2, 
                str(output_root / f"{output_stem}_{score_type}{log_prefix}_diff_dist.png"),
                label="Difference"
            )
            
            # 2. Difference Rolling Median Plot
            plot_rolling_stat_over_time(
                merged, 'diff', score_type + log_title, args.name1, args.name2, 
                str(output_root / f"{output_stem}_{score_type}{log_prefix}_diff_rolling.png"),
                rolling_window=args.rolling,
                label="Difference"
            )
            
            # 3. Skill Score Distribution Plot
            plot_stat_distribution(
                merged, 'skill_score', score_type + log_title, args.name1, args.name2, 
                str(output_root / f"{output_stem}_{score_type}{log_prefix}_ss_dist.png"),
                label="Skill Score"
            )
            
            # 4. Skill Score Rolling Median Plot
            plot_rolling_stat_over_time(
                merged, 'skill_score', score_type + log_title, args.name1, args.name2, 
                str(output_root / f"{output_stem}_{score_type}{log_prefix}_ss_rolling.png"),
                rolling_window=args.rolling,
                label="Skill Score"
            )
            
            # 5. Identify and save outperforming cases (per window)
            save_outperforming_samples(
                merged, score_type + log_prefix, args.name1, args.name2, 
                str(output_root / output_stem), top_n=args.top_n
            )
            
            # 6. Consistency Analysis (across all windows)
            save_consistency_analysis(
                merged, score_type + log_prefix, args.name1, args.name2, 
                str(output_root / output_stem)
            )

if __name__ == "__main__":
    main()
