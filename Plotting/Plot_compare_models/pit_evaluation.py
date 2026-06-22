import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.feather as feather
import seaborn as sns
from tqdm import tqdm
import argparse

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None


def _sanitize_name(name: str) -> str:
    return name.replace(" ", "_")


def _ensure_parent_dir(file_path):
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def _resolve_device(device: Optional[str]) -> str:
    if not device or device == "cpu":
        return "cpu"
    if device == "auto":
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if device == "cuda" and (torch is None or not torch.cuda.is_available()):
        print("CUDA was requested for PIT evaluation, but it is not available. Falling back to CPU.")
        return "cpu"
    return device


def _plot_pit_histogram(output_path: str, title: str, u_grid: np.ndarray, counts: np.ndarray, num_bins: int):
    plt.figure(figsize=(8, 5))
    sns.set_style("whitegrid", {'axes.grid': True, 'grid.linestyle': '--'})

    bin_centers = (u_grid[:-1] + u_grid[1:]) / 2
    width = 1.0 / num_bins
    plt.bar(bin_centers, counts, width=width, color='blue', edgecolor='black')

    total = counts.sum()
    plt.axhline(total / num_bins, color='gray', linestyle='--', linewidth=2, label='Ideal Uniform Density')

    plt.title(title, fontsize=14)
    plt.xlabel("Binned cumulative probability", fontsize=12)
    plt.ylabel("Frequency of occurrences", fontsize=12)
    plt.legend()
    plt.tight_layout()

    _ensure_parent_dir(output_path)
    plt.savefig(output_path, dpi=300)
    plt.close()


def _plot_stacked_pit_histogram(output_path: str, title: str, u_grid: np.ndarray, bucket_counts: Dict[str, np.ndarray], num_bins: int):
    plt.figure(figsize=(8, 5))
    sns.set_style("whitegrid", {'axes.grid': True, 'grid.linestyle': '--'})

    bin_centers = (u_grid[:-1] + u_grid[1:]) / 2
    width = 1.0 / num_bins

    def get_bucket_start(label: str) -> int:
        try:
            return int(label.split('-')[0])
        except Exception:
            return 9999

    sorted_buckets = sorted(bucket_counts.keys(), key=get_bucket_start)
    bottoms = np.zeros(num_bins, dtype=np.float32)
    colors = sns.color_palette("muted", len(sorted_buckets))

    for idx, bucket in enumerate(sorted_buckets):
        counts = bucket_counts[bucket]
        plt.bar(bin_centers, counts, width=width, bottom=bottoms, color=colors[idx], edgecolor='black', label=bucket)
        bottoms += counts

    total = bottoms.sum()
    plt.axhline(total / num_bins, color='gray', linestyle='--', linewidth=2, label='Ideal Uniform Density')

    plt.title(title, fontsize=14)
    plt.xlabel("Binned cumulative probability", fontsize=12)
    plt.ylabel("Frequency of occurrences", fontsize=12)
    plt.legend()
    plt.tight_layout()

    _ensure_parent_dir(output_path)
    plt.savefig(output_path, dpi=300)
    plt.close()


def _pit_group_contribution_numpy(samples: np.ndarray, y_true: np.ndarray, is_continuous: bool, u_grid: np.ndarray) -> np.ndarray:
    sample_count = samples.shape[1]

    if is_continuous:
        f_y = np.sum(samples <= y_true[:, None], axis=1) / sample_count
        g = np.zeros((len(y_true), len(u_grid)), dtype=np.float32)
        for i, u in enumerate(u_grid):
            g[:, i] = (f_y <= u).astype(np.float32)
        return np.sum(g, axis=0)

    f_y = np.sum(samples <= y_true[:, None], axis=1) / sample_count
    f_y_minus_1 = np.sum(samples <= (y_true - 1)[:, None], axis=1) / sample_count
    g = np.zeros((len(y_true), len(u_grid)), dtype=np.float32)

    for i, u in enumerate(u_grid):
        cond1 = u <= f_y_minus_1
        cond2 = (f_y_minus_1 < u) & (u <= f_y)
        cond3 = u > f_y

        g[cond1, i] = 0.0
        denom = f_y - f_y_minus_1
        valid = denom > 0
        mask = cond2 & valid
        g[mask, i] = (u - f_y_minus_1[mask]) / denom[mask]
        g[cond3, i] = 1.0

    return np.sum(g, axis=0)


def _pit_group_contribution_torch(samples: np.ndarray, y_true: np.ndarray, is_continuous: bool, u_grid: np.ndarray, device: str) -> np.ndarray:
    if torch is None:
        return _pit_group_contribution_numpy(samples, y_true, is_continuous, u_grid)

    samples_t = torch.as_tensor(samples, device=device, dtype=torch.float32)
    y_true_t = torch.as_tensor(y_true, device=device, dtype=torch.float32)
    u_t = torch.as_tensor(u_grid, device=device, dtype=torch.float32)
    sample_count = samples_t.shape[1]

    if is_continuous:
        f_y = torch.sum(samples_t <= y_true_t.unsqueeze(1), dim=1) / sample_count
        g = torch.zeros((y_true_t.shape[0], u_t.shape[0]), device=device, dtype=torch.float32)
        for i, u in enumerate(u_t):
            g[:, i] = (f_y <= u).to(torch.float32)
        return torch.sum(g, dim=0).detach().cpu().numpy()

    f_y = torch.sum(samples_t <= y_true_t.unsqueeze(1), dim=1) / sample_count
    f_y_minus_1 = torch.sum(samples_t <= (y_true_t - 1).unsqueeze(1), dim=1) / sample_count
    g = torch.zeros((y_true_t.shape[0], u_t.shape[0]), device=device, dtype=torch.float32)

    for i, u in enumerate(u_t):
        cond1 = u <= f_y_minus_1
        cond2 = (f_y_minus_1 < u) & (u <= f_y)
        cond3 = u > f_y

        g[cond1, i] = 0.0
        denom = f_y - f_y_minus_1
        valid = denom > 0
        mask = cond2 & valid
        g[mask, i] = (u - f_y_minus_1[mask]) / denom[mask]
        g[cond3, i] = 1.0

    return torch.sum(g, dim=0).detach().cpu().numpy()


def _pit_group_contribution(samples: np.ndarray, y_true: np.ndarray, is_continuous: bool, u_grid: np.ndarray, device: str) -> np.ndarray:
    if device != "cpu" and torch is not None and torch.cuda.is_available():
        return _pit_group_contribution_torch(samples, y_true, is_continuous, u_grid, device)
    return _pit_group_contribution_numpy(samples, y_true, is_continuous, u_grid)


def _load_window_tables(window_dir: Path, burden: str):
    pred_file = window_dir / "predictions.feather"
    spag_file = window_dir / "sample_spaghetti.feather"

    if not pred_file.exists() or not spag_file.exists():
        return None, None

    truth_col = f"{burden}_unscaled_gt"
    try:
        pred_df = feather.read_table(pred_file, columns=['timestep', 'msoa', truth_col]).to_pandas()
    except Exception:
        return None, None

    if truth_col not in pred_df.columns:
        return None, None

    try:
        spag_df = feather.read_table(
            spag_file,
            columns=['timestep', 'msoa', 'sample_idx', 'burden_type', 'value']
        ).to_pandas()
    except Exception:
        return None, None

    spag_df = spag_df[spag_df['burden_type'] == burden]
    if spag_df.empty:
        return None, None

    return spag_df, pred_df


def _accumulate_day_ahead_pit(spag_df: pd.DataFrame, pred_df: pd.DataFrame, burden: str, window_start: int, is_continuous: bool, u_grid: np.ndarray, device: str, by_window_start: Dict[int, np.ndarray], by_window_counts: Dict[int, int], by_day_ahead: Dict[Tuple[int, int], np.ndarray], by_day_counts: Dict[Tuple[int, int], int]):
    truth_col = f"{burden}_unscaled_gt"
    merged = spag_df.merge(pred_df, on=['timestep', 'msoa'], how='left')
    merged['day_ahead'] = merged['timestep'] - window_start
    merged = merged[merged['day_ahead'] >= 1]

    if merged.empty:
        return

    for day_ahead, group in merged.groupby('day_ahead'):
        pivot = group.pivot(index=['timestep', 'msoa', truth_col], columns='sample_idx', values='value').reset_index()
        y_true = pivot[truth_col].to_numpy()
        samples = pivot.drop(columns=['timestep', 'msoa', truth_col]).to_numpy()

        contribution = _pit_group_contribution(samples, y_true, is_continuous, u_grid, device)
        num_rows = len(y_true)

        if window_start not in by_window_start:
            by_window_start[window_start] = np.zeros_like(u_grid, dtype=np.float32)
            by_window_counts[window_start] = 0
        by_window_start[window_start] += contribution
        by_window_counts[window_start] += num_rows

        day_key = (window_start, int(day_ahead))
        if day_key not in by_day_ahead:
            by_day_ahead[day_key] = np.zeros_like(u_grid, dtype=np.float32)
            by_day_counts[day_key] = 0
        by_day_ahead[day_key] += contribution
        by_day_counts[day_key] += num_rows


def run_pit_evaluations(
    model_dir,
    model_name,
    burden,
    output_root,
    *,
    include_window_start=False,
    include_day_ahead=True,
    num_bins=20,
    device="cpu",
):
    print("\n--- Probability Integral Transform (PIT) ---")
    is_continuous = burden.startswith('Rt')
    device = _resolve_device(device)

    output_root = Path(output_root)
    pit_dir = output_root / "PIT_Histograms"
    pit_dir.mkdir(parents=True, exist_ok=True)

    if include_day_ahead:
        horizon_pit_dir = pit_dir / "By_Horizon"
        horizon_pit_dir.mkdir(parents=True, exist_ok=True)
    else:
        horizon_pit_dir = None

    u_grid = np.linspace(0, 1, num_bins + 1)

    print(f"Processing PIT for {model_name}...")
    testing_path = Path(model_dir) / "TESTING"
    if not testing_path.exists():
        print(f"Warning: TESTING directory not found at {testing_path}")
        return

    ws_G_sums = {}
    ws_counts = {}
    ws_da_G_sums = {}
    ws_da_counts = {}

    sim_dirs = [entry for entry in os.scandir(testing_path) if entry.is_dir() and entry.name.startswith("SIM_")]
    for sim_entry in tqdm(sim_dirs, desc=f"Loading samples for {model_name}"):
        pred_root = os.path.join(sim_entry.path, "predictions")
        if not os.path.exists(pred_root):
            continue

        for w_entry in os.scandir(pred_root):
            if not w_entry.is_dir():
                continue

            try:
                window_start = int(w_entry.name)
            except ValueError:
                continue

            spag_df, pred_df = _load_window_tables(Path(w_entry.path), burden)
            if spag_df is None or pred_df is None:
                continue

            _accumulate_day_ahead_pit(
                spag_df,
                pred_df,
                burden,
                window_start,
                is_continuous,
                u_grid,
                device,
                ws_G_sums,
                ws_counts,
                ws_da_G_sums,
                ws_da_counts,
            )

    if include_window_start:
        for ws in sorted(ws_G_sums.keys()):
            G_bar = ws_G_sums[ws] / ws_counts[ws]
            hist_counts = ws_counts[ws] * np.diff(G_bar)
            out_name = pit_dir / f"pit_histogram_ws_{ws}_{_sanitize_name(model_name)}.png"
            _plot_pit_histogram(
                str(out_name),
                f"PIT Histogram - {model_name} (Window Start: {ws})",
                u_grid,
                hist_counts,
                num_bins,
            )

            # Generate a stacked copy of the results
            bucket_counts = defaultdict(lambda: np.zeros(num_bins, dtype=np.float32))
            for (curr_ws, day_ahead), da_G in ws_da_G_sums.items():
                if curr_ws == ws:
                    da_counts = np.diff(da_G)
                    start_day = (day_ahead // 5) * 5
                    end_day = start_day + 4
                    bucket_label = f"{start_day}-{end_day} days"
                    bucket_counts[bucket_label] += da_counts

            if bucket_counts:
                stacked_out_name = pit_dir / f"pit_histogram_ws_{ws}_stacked_{_sanitize_name(model_name)}.png"
                _plot_stacked_pit_histogram(
                    str(stacked_out_name),
                    f"Stacked PIT Histogram - {model_name} (Window Start: {ws})",
                    u_grid,
                    dict(bucket_counts),
                    num_bins,
                )

    if include_day_ahead and horizon_pit_dir is not None:
        for ws, day_ahead in sorted(ws_da_G_sums.keys()):
            G_bar = ws_da_G_sums[(ws, day_ahead)] / ws_da_counts[(ws, day_ahead)]
            hist_counts = ws_da_counts[(ws, day_ahead)] * np.diff(G_bar)
            out_name = horizon_pit_dir / f"pit_histogram_ws_{ws}_da_{day_ahead:02d}_{_sanitize_name(model_name)}.png"
            _plot_pit_histogram(
                str(out_name),
                f"PIT - {model_name} (Window Start: {ws}, Day Ahead: {day_ahead})",
                u_grid,
                hist_counts,
                num_bins,
            )


def main():
    parser = argparse.ArgumentParser(description="Run PIT evaluation and save PIT histogram outputs.")
    parser.add_argument("--dir", type=str, required=True, help="Directory for model results")
    parser.add_argument("--name", type=str, default="Model", help="Name for model")
    parser.add_argument("--burden", type=str, default="daily_hospitalised", help="Burden to evaluate")
    parser.add_argument("--output_prefix", type=str, default="comparison", help="Prefix or directory for PIT outputs")
    parser.add_argument("--num_bins", type=int, default=20, help="Number of PIT histogram bins")
    parser.add_argument("--device", type=str, default="cpu", help="PIT evaluation device: cpu, cuda, or auto")
    parser.add_argument("--include_window_start", action="store_true", help="Write window-start PIT histograms")
    parser.add_argument("--include_day_ahead", action="store_true", help="Write day-ahead PIT histograms")

    args = parser.parse_args()

    output_root = Path(args.output_prefix)
    output_root.mkdir(parents=True, exist_ok=True)

    include_window_start = args.include_window_start
    include_day_ahead = args.include_day_ahead
    if not include_window_start and not include_day_ahead:
        include_window_start = True
        include_day_ahead = True

    run_pit_evaluations(
        args.dir,
        args.name,
        args.burden,
        str(output_root),
        include_window_start=include_window_start,
        include_day_ahead=include_day_ahead,
        num_bins=args.num_bins,
        device=args.device,
    )


if __name__ == "__main__":
    main()