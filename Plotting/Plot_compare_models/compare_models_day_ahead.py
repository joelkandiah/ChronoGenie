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
Day-ahead model comparison plots.

Typical Burden Names:
- daily_hospitalised
- deaths
- daily_infected
- Rt2 (or R_t)

Output Options:
- <output_prefix>_energy_day_ahead.png: Comparison based on Energy Score by forecast horizon
- <output_prefix>_crps_day_ahead.png: Comparison based on CRPS by forecast horizon
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

    The returned frame is aligned to forecast horizon via:
        day_ahead = timestep - window_start
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
                            table = feather.read_table(file_path, columns=['timestep', 'burden', col])
                            df = table.to_pandas()
                            burden_df = df[df['burden'] == burden].copy()
                            if burden_df.empty:
                                continue

                            burden_df['day_ahead'] = burden_df['timestep'] - window_start
                            burden_df = burden_df[burden_df['day_ahead'] >= 1]

                            for _, row in burden_df.iterrows():
                                all_data.append({
                                    'sim_id': sim_id,
                                    'window_start': window_start,
                                    'day_ahead': int(row['day_ahead']),
                                    'score': row[col]
                                })
                        except Exception:
                            continue

                    elif score_type == 'crps':
                        file_path = w_dir / "predictions.feather"
                        if not file_path.exists():
                            continue

                        col_name = f"{burden}_CRPS_log" if use_log else f"{burden}_CRPS"
                        try:
                            table = feather.read_table(file_path, columns=['timestep', 'msoa', col_name])
                            df = table.to_pandas()
                            df['day_ahead'] = df['timestep'] - window_start
                            df = df[df['day_ahead'] >= 1]
                            if df.empty:
                                continue

                            horizon_scores = df.groupby('day_ahead')[col_name].mean().reset_index()
                            for _, row in horizon_scores.iterrows():
                                all_data.append({
                                    'sim_id': sim_id,
                                    'window_start': window_start,
                                    'day_ahead': int(row['day_ahead']),
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
                            table = feather.read_table(file_path, columns=['timestep', 'burden', col])
                            df = table.to_pandas()
                            burden_df = df[df['burden'] == burden].copy()
                            if burden_df.empty:
                                continue

                            burden_df['day_ahead'] = burden_df['timestep'] - window_start
                            burden_df = burden_df[burden_df['day_ahead'] >= 1]

                            for _, row in burden_df.iterrows():
                                all_data.append({
                                    'sim_id': sim_id,
                                    'window_start': window_start,
                                    'day_ahead': int(row['day_ahead']),
                                    'score': row[col]
                                })
                        except Exception:
                            continue

    return pd.DataFrame(all_data)


def calculate_win_rates(df1, df2, df3=None, score_type='energy'):
    """
    Calculate the percentage of times each model performed best at each forecast horizon.
    """
    if df1.empty or df2.empty:
        return pd.DataFrame()

    print(f"Merging data for {score_type} comparison...")
    on_cols = ['sim_id', 'window_start', 'day_ahead']

    merged = pd.merge(df1, df2, on=on_cols, suffixes=('_1', '_2'))

    if df3 is not None and not df3.empty:
        merged = pd.merge(merged, df3.rename(columns={'score': 'score_3'}), on=on_cols)
        score_cols = ['score_1', 'score_2', 'score_3']
        num_models = 3
    else:
        score_cols = ['score_1', 'score_2']
        num_models = 2

    if merged.empty:
        print("No matching simulations/windows found between the directories.")
        return pd.DataFrame()

    scores = merged[score_cols].values
    best_idx = np.argmin(scores, axis=1)

    for i in range(num_models):
        merged[f'model{i+1}_wins'] = (best_idx == i)

    agg_dict = {f'model{i+1}_wins': 'mean' for i in range(num_models)}
    win_rates = merged.groupby('day_ahead').agg(agg_dict).reset_index()

    for i in range(num_models):
        win_rates[f'model{i+1}_wins'] *= 100

    return win_rates


def plot_comparison(win_rates, title, xlabel, model_names, output_path):
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid", {'axes.grid': True, 'grid.linestyle': '--'})

    win_rates = win_rates.sort_values('day_ahead')

    styles = [
        {'color': 'black', 'linestyle': '-', 'linewidth': 2},
        {'color': 'black', 'linestyle': '--', 'linewidth': 1.5},
        {'color': '#555555', 'linestyle': ':', 'linewidth': 2}
    ]

    for i, name in enumerate(model_names):
        col = f'model{i+1}_wins'
        if col in win_rates.columns:
            style = styles[i % len(styles)]
            plt.plot(win_rates['day_ahead'], win_rates[col], label=name, **style)

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

    parser = argparse.ArgumentParser(description="Compare two or three GENIE models using day-ahead win-rate plots.")
    parser.add_argument("--dir1", type=str, required=True, help="Directory for model 1 results")
    parser.add_argument("--dir2", type=str, required=True, help="Directory for model 2 results")
    parser.add_argument("--dir3", type=str, default=None, help="Optional directory for model 3 results")
    parser.add_argument("--name1", type=str, default="Model 1", help="Name for model 1")
    parser.add_argument("--name2", type=str, default="Model 2", help="Name for model 2")
    parser.add_argument("--name3", type=str, default="Model 3", help="Name for model 3")
    parser.add_argument("--burden", type=str, default="daily_hospitalised", help="Burden to evaluate (e.g. daily_hospitalised, deaths)")
    parser.add_argument("--output_prefix", type=str, default="comparison", help="Prefix or directory for output plots")

    args = parser.parse_args()

    output_root, output_stem = resolve_output_base(args.output_prefix)

    model_dirs = [args.dir1, args.dir2]
    model_names = [args.name1, args.name2]

    if args.dir3:
        model_dirs.append(args.dir3)
        model_names.append(args.name3)

    for use_log in [False, True]:
        log_prefix = "_log" if use_log else ""
        log_title = " (Log Data)" if use_log else ""

        print(f"\n--- Energy Score Comparison (use_log={use_log}) ---")
        valid_e_scores = []
        valid_e_names = []
        for d, name in zip(model_dirs, model_names):
            df = load_all_scores(d, 'energy', args.burden, use_log=use_log)
            if not df.empty:
                csv_name = str(output_root / f"{output_stem}_energy_day_ahead{log_prefix}_{name.replace(' ', '_')}.csv")
                ensure_parent_dir(csv_name)
                df.to_csv(csv_name, index=False)
                print(f"Saved scores to {csv_name}")
                valid_e_scores.append(df)
                valid_e_names.append(name)

        if len(valid_e_scores) >= 2:
            df3 = valid_e_scores[2] if len(valid_e_scores) > 2 else None
            e_wins = calculate_win_rates(valid_e_scores[0], valid_e_scores[1], df3, 'energy')

            if not e_wins.empty:
                current_names = valid_e_names[:3] if len(valid_e_names) >= 3 else valid_e_names[:2]
                plot_comparison(
                    e_wins,
                    f"Energy Score Win Rate by Day Ahead{log_title} ({args.burden})",
                    "Predicted Day Ahead (days)",
                    current_names,
                    str(output_root / f"{output_stem}_energy_day_ahead{log_prefix}.png")
                )

        print(f"\n--- CRPS Comparison (use_log={use_log}) ---")
        valid_c_scores = []
        valid_c_names = []
        for d, name in zip(model_dirs, model_names):
            df = load_all_scores(d, 'crps', args.burden, use_log=use_log)
            if not df.empty:
                csv_name = str(output_root / f"{output_stem}_crps_day_ahead{log_prefix}_{name.replace(' ', '_')}.csv")
                ensure_parent_dir(csv_name)
                df.to_csv(csv_name, index=False)
                print(f"Saved scores to {csv_name}")
                valid_c_scores.append(df)
                valid_c_names.append(name)

        if len(valid_c_scores) >= 2:
            df3 = valid_c_scores[2] if len(valid_c_scores) > 2 else None
            c_wins = calculate_win_rates(valid_c_scores[0], valid_c_scores[1], df3, 'crps')

            if not c_wins.empty:
                current_names = valid_c_names[:3] if len(valid_c_names) >= 3 else valid_c_names[:2]
                plot_comparison(
                    c_wins,
                    f"CRPS Win Rate by Day Ahead{log_title} ({args.burden})",
                    "Predicted Day Ahead (days)",
                    current_names,
                    str(output_root / f"{output_stem}_crps_day_ahead{log_prefix}.png")
                )

        print(f"\n--- Variogram Score Comparison (use_log={use_log}) ---")
        valid_v_scores = []
        valid_v_names = []
        for d, name in zip(model_dirs, model_names):
            df = load_all_scores(d, 'variogram', args.burden, use_log=use_log)
            if not df.empty:
                csv_name = str(output_root / f"{output_stem}_variogram_day_ahead{log_prefix}_{name.replace(' ', '_')}.csv")
                ensure_parent_dir(csv_name)
                df.to_csv(csv_name, index=False)
                print(f"Saved scores to {csv_name}")
                valid_v_scores.append(df)
                valid_v_names.append(name)

        if len(valid_v_scores) >= 2:
            df3 = valid_v_scores[2] if len(valid_v_scores) > 2 else None
            v_wins = calculate_win_rates(valid_v_scores[0], valid_v_scores[1], df3, 'variogram')

            if not v_wins.empty:
                current_names = valid_v_names[:3] if len(valid_v_names) >= 3 else valid_v_names[:2]
                plot_comparison(
                    v_wins,
                    f"Variogram Score Win Rate by Day Ahead{log_title} ({args.burden})",
                    "Predicted Day Ahead (days)",
                    current_names,
                    str(output_root / f"{output_stem}_variogram_day_ahead{log_prefix}.png")
                )


if __name__ == "__main__":
    main()
