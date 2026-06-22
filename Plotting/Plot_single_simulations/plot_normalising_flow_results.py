import os
import argparse
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyarrow import feather

def load_predictions(results_dir, sim_id=None):
    """Load predictions.feather files efficiently using os.scandir."""
    testing_path = os.path.join(results_dir, "TESTING")
    if not os.path.exists(testing_path):
        return pd.DataFrame()
    
    dfs = []
    
    # Iterate over SIM directories
    with os.scandir(testing_path) as it:
        for entry in it:
            if not entry.is_dir() or not entry.name.startswith("SIM_"):
                continue
            
            # If a specific sim_id is requested, skip others
            if sim_id and entry.name != sim_id:
                continue
            
            sim_name = entry.name
            pred_root = os.path.join(entry.path, "predictions")
            if not os.path.exists(pred_root):
                continue
                
            # Iterate over window directories
            with os.scandir(pred_root) as pred_it:
                for w_entry in pred_it:
                    if not w_entry.is_dir():
                        continue
                    
                    try:
                        window_start = int(w_entry.name)
                    except ValueError:
                        continue
                        
                    f = os.path.join(w_entry.path, "predictions.feather")
                    if os.path.exists(f):
                        try:
                            df = feather.read_feather(f)
                            df['window_start_day'] = window_start
                            df['sim_id'] = sim_name
                            dfs.append(df)
                        except Exception as e:
                            print(f"Error reading {f}: {e}")
    
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)

def random_subset_samples(df, num_samples):
    """Pick a random subset of sample IDs."""
    all_s = df['sample_idx'].unique()
    if len(all_s) <= num_samples:
        return all_s
    return np.random.choice(all_s, size=num_samples, replace=False)

def plot_forecasts(results_dir, df, msoa_id, output_path, window_freq=None, sim_id=None, start_day=0, max_prediction_days=60, show_samples=False, num_samples_to_plot=20):
    """Replicate the reference plot style for Normalising Flow."""
    # Filter for selected simulation if provided
    if sim_id:
        df = df[df['sim_id'] == sim_id]
        if df.empty:
            print(f"No results found for {sim_id}. Available: {list(df['sim_id'].unique()) if not df.empty else 'none'}")
            return

    # Filter for selected MSOA
    df_msoa = df[df['msoa'] == msoa_id].copy()
    
    # Identify unique windows and burden types
    all_windows = sorted(df_msoa['window_start_day'].unique())
    all_windows = [w for w in all_windows if w >= start_day]
    
    if not all_windows:
        print(f"No windows found starting at or after day {start_day}.")
        return

    if window_freq:
        # Filter windows to keep only those that are multiples of freq (or starting from the first)
        start_w = all_windows[0]
        windows = [w for w in all_windows if (w - start_w) % window_freq == 0]
    else:
        windows = all_windows

    # Prediction names: we look for columns ending in _unscaled
    all_cols = df_msoa.columns
    prediction_names = [c.replace("_unscaled", "") for c in all_cols if c.endswith("_unscaled") and not c.endswith("_unscaled_gt")]
    
    # Sort burdens to match: deaths, daily_infected, hospitalisations, Rt (last)
    sort_order = {
        "deaths": 0, 
        "daily_infected": 1, 
        "daily_hospitalised": 2, 
        "hospitalisations": 2, 
        "hospitalised": 2,
        "R_t": 10, "Rt": 10, "Rt2": 10
    }
    prediction_names.sort(key=lambda x: sort_order.get(x, 5))
    
    num_burdens = len(prediction_names)
    fig, axes = plt.subplots(num_burdens, 1, figsize=(15, 4 * num_burdens), sharex=True)
    if num_burdens == 1:
        axes = [axes]
    
    colors = sns.color_palette("viridis", n_colors=len(windows)) # Different palette to distinguish from PM/FM
    
    for i, burden in enumerate(prediction_names):
        ax = axes[i]
        
        # Ground truth
        gt_df = df_msoa[['timestep', f'{burden}_unscaled_gt']].drop_duplicates().sort_values('timestep')
        # Ensure we show all available ground truth by not clipping to the max prediction timestep
        
        ax.scatter(gt_df['timestep'], gt_df[f'{burden}_unscaled_gt'], color='black', s=10, label='ground truth' if i == 0 else "")
        
        for w_idx, w_start in enumerate(windows):
            w_df = df_msoa[df_msoa['window_start_day'] == w_start].sort_values('timestep')
            
            # Cap the prediction days
            w_df = w_df[w_df['timestep'] <= w_start + max_prediction_days]
            
            if w_df.empty:
                continue

            color = colors[w_idx]
            
            # Vertical line for window start
            ax.axvline(x=w_start, color=color, linestyle='-', alpha=0.8, linewidth=1)
            
            # Median
            ax.plot(w_df['timestep'], w_df[f'{burden}_unscaled'], color=color, linewidth=2)
            
            if not show_samples:
                # 50% CI
                ax.fill_between(w_df['timestep'], 
                                w_df[f'{burden}_lower_50'], 
                                w_df[f'{burden}_upper_50'], 
                                color=color, alpha=0.4)
                
                # 95% CI
                ax.fill_between(w_df['timestep'], 
                                w_df[f'{burden}_lower_95'], 
                                w_df[f'{burden}_upper_95'], 
                                color=color, alpha=0.2)
            
            # --- OVERLAY SPAGHETTI (SAMPLES) ---
            if show_samples:
                spaghetti_path = os.path.join(results_dir, "TESTING", sim_id, "predictions", str(w_start), "sample_spaghetti.feather")
                if os.path.exists(spaghetti_path):
                    s_df = feather.read_feather(spaghetti_path)
                    s_df = s_df[(s_df['msoa'] == msoa_id) & (s_df['burden_type'] == burden)]
                    s_df = s_df[s_df['timestep'] <= w_start + max_prediction_days]
                    
                    if not s_df.empty:
                        s_ids = random_subset_samples(s_df, num_samples_to_plot)
                        for s_id in s_ids:
                            sample_traj = s_df[s_df['sample_idx'] == s_id].sort_values('timestep')
                            ax.plot(sample_traj['timestep'], sample_traj['value'], color=color, alpha=0.1, linewidth=0.5)
        
        ax.set_ylabel(f"{burden}\n(in MSOA {msoa_id})")
        ax.grid(True, linestyle='--', alpha=0.5)
        
        if burden.lower() in ['rt', 'r_t']:
            ax.axhline(y=1.0, color='red', linestyle='-.', alpha=0.6)

    axes[-1].set_xlabel("day")
    
    # Custom legend for CI
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='black', lw=2, label='median'),
        Patch(facecolor='gray', alpha=0.6, label='50% CI'),
        Patch(facecolor='gray', alpha=0.3, label='95% CI'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=5, label='ground truth'),
    ]
    axes[0].legend(handles=legend_elements, loc='upper right')
    
    plt.suptitle(f"Normalising Flow Forecasts - MSOA {msoa_id}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300)
    print(f"Saved plot to {output_path}")

def plot_spaghetti(results_dir, msoa_id, window_start, sim_id, output_path, start_day=0, max_prediction_days=60):
    """Plot individual sample trajectories for a specific Normalising Flow window."""
    spaghetti_path = os.path.join(results_dir, "TESTING", sim_id, "predictions", str(window_start), "sample_spaghetti.feather")
    if not os.path.exists(spaghetti_path):
        print(f"Spaghetti file not found: {spaghetti_path}")
        return
    
    df = feather.read_feather(spaghetti_path)
    df = df[df['msoa'] == msoa_id]
    
    # Load predictions from the same folder for ground truth
    pred_path = os.path.join(os.path.dirname(spaghetti_path), "predictions.feather")
    df_gt = feather.read_feather(pred_path)
    df_gt = df_gt[df_gt['msoa'] == msoa_id]
    
    burdens = df['burden_type'].unique()
    num_burdens = len(burdens)
    
    fig, axes = plt.subplots(num_burdens, 1, figsize=(15, 4 * num_burdens), sharex=True)
    if num_burdens == 1:
        axes = [axes]
        
    for i, burden in enumerate(burdens):
        ax = axes[i]
        b_df = df[df['burden_type'] == burden]
        
        # Filter by start day and max prediction days
        b_df = b_df[(b_df['timestep'] >= start_day) & (b_df['timestep'] <= window_start + max_prediction_days)]
        
        if b_df.empty:
            continue

        # Plot individual samples
        for s_idx in b_df['sample_idx'].unique():
            s_df = b_df[b_df['sample_idx'] == s_idx].sort_values('timestep')
            ax.plot(s_df['timestep'], s_df['value'], color='gray', alpha=0.15, linewidth=0.5)
        
        # Plot ground truth dots
        gt_col = f"{burden}_unscaled_gt"
        if gt_col in df_gt.columns:
            gt_df = df_gt[['timestep', gt_col]].drop_duplicates().sort_values('timestep')
            ax.scatter(gt_df['timestep'], gt_df[gt_col], color='black', s=10, label='ground truth' if i == 0 else "")

        # Plot mean/median of samples for reference
        agg_df = b_df.groupby('timestep')['value'].agg(['mean', 'median']).reset_index()
        ax.plot(agg_df['timestep'], agg_df['median'], color='red', linewidth=2, label='sample median')
        
        ax.set_ylabel(f"{burden}\n(MSOA {msoa_id}, Window {window_start})")
        ax.grid(True, linestyle='--', alpha=0.5)
        
    axes[-1].set_xlabel("day")
    axes[0].legend()
    plt.suptitle(f"Normalising Flow Spaghetti - MSOA {msoa_id}, Window {window_start}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300)
    print(f"Saved spaghetti plot to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot results from Normalising Flow experiments.")
    parser.add_argument(
        "--results_dir",
        type=str,
        default=".",
        help="Root directory containing TESTING/SIM_* results",
    )
    parser.add_argument("--msoa", type=int, default=0)
    parser.add_argument("--sim", type=str, default="SIM_1", help="Filter by simulation ID")
    parser.add_argument("--freq", type=int, default=None, help="Plot windows every N days")
    parser.add_argument("--window", type=int, default=None, help="Specific window start to plot spaghetti for")
    parser.add_argument("--output", type=str, default="normalising_flow_forecast.png")
    parser.add_argument("--spaghetti_output", type=str, default="normalising_flow_spaghetti.png")
    parser.add_argument("--start_day", type=int, default=0, help="Start plotting from this day")
    parser.add_argument("--max_prediction_days", type=int, default=60, help="Max days to plot for each prediction window")
    parser.add_argument("--show_samples", action="store_true", help="Overlay spaghetti lines in forecast plots")
    parser.add_argument("--num_samples_to_plot", type=int, default=20, help="Number of samples to plot per window")
    args = parser.parse_args()
    
    if args.window is not None:
        print(f"Plotting Normalising Flow spaghetti for MSOA {args.msoa}, Sim {args.sim}, Window {args.window}...")
        plot_spaghetti(args.results_dir, args.msoa, args.window, args.sim, args.spaghetti_output, 
                       start_day=args.start_day, max_prediction_days=args.max_prediction_days)
    else:
        print(f"Loading results from {args.results_dir} (Filter: {args.sim})...")
        df = load_predictions(args.results_dir, sim_id=args.sim)
        
        if df.empty:
            print(f"No predictions found in {args.results_dir}!")
            return
        
        print(f"Plotting for MSOA {args.msoa} (Sim: {args.sim}, freq: {args.freq})...")
        plot_forecasts(args.results_dir, df, args.msoa, args.output, window_freq=args.freq, sim_id=args.sim, 
                       start_day=args.start_day, max_prediction_days=args.max_prediction_days,
                       show_samples=args.show_samples, num_samples_to_plot=args.num_samples_to_plot)

if __name__ == "__main__":
    main()
