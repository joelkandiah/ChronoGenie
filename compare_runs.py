#!/usr/bin/env python3
"""Aggregate and compare ChronoGenie / GENIE result trees.

Every run -- a ChronoGenie covariate mode, its baseline or fine-tuned variant, or a
GENIE model directory -- writes the same three feather tables per forecast window,
so they can all be pooled and compared with the same code.

Usage::

    python compare_runs.py \
        --run baseline=RESULTS/.../baseline/ARW_60 \
        --run finetuned=RESULTS/.../finetuned/ARW_60 \
        --run genie_mlp=../GENIE/RESULTS/RESULTS_MLP/RUN \
        --reference baseline \
        --output RESULTS/comparison

Each ``--run`` value is a directory containing ``TESTING/SIM_*/predictions/<window>``.

Outputs
-------
``scores_long.csv``
    One row per (run, sim, window, lead, burden) with every scoring rule.
``summary_by_run.csv`` / ``summary_by_lead.csv``
    Means over the pooled evaluation grid.
``skill_vs_<reference>.csv``
    ``1 - score/reference_score`` per run, burden and score -- positive is better --
    computed on the windows all runs share, so the comparison is paired.
``summary.md``
    The same numbers as a readable table.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather

# Univariate scores live in predictions.feather as "<burden>_<score>".
NODE_SCORES = ["CRPS", "CRPS_log", "IS_95", "IS_95_log"]
# Multivariate (across-MSOA) scores live in spatial_scores.feather.
SPATIAL_SCORES = ["energy_score", "energy_score_log", "variogram_score_p05", "variogram_score_p05_log"]
ALL_SCORES = NODE_SCORES + SPATIAL_SCORES


def discover_windows(results_dir: str | Path) -> list[tuple[str, int, Path]]:
    """Find every ``(sim_id, window_start, directory)`` under a results tree."""
    testing = Path(results_dir) / "TESTING"
    if not testing.is_dir():
        raise FileNotFoundError(f"No TESTING directory under {results_dir}")

    windows = []
    for sim_dir in sorted(testing.glob("SIM_*")):
        predictions = sim_dir / "predictions"
        if not predictions.is_dir():
            continue
        for window_dir in sorted(predictions.iterdir()):
            if not window_dir.is_dir():
                continue
            try:
                window_start = int(window_dir.name)
            except ValueError:
                continue
            windows.append((sim_dir.name.removeprefix("SIM_"), window_start, window_dir))
    return windows


def load_run(name: str, results_dir: str | Path, burdens: list[str] | None = None) -> pd.DataFrame:
    """Load one run into long form: (run, sim, window_start, lead, burden, score...).

    Univariate scores are averaged over MSOAs, which is how GENIE's comparison
    scripts summarise them; the multivariate scores are already per (lead, burden).
    """
    frames = []
    for sim_id, window_start, window_dir in discover_windows(results_dir):
        prediction_file = window_dir / "predictions.feather"
        spatial_file = window_dir / "spatial_scores.feather"
        if not prediction_file.is_file() or not spatial_file.is_file():
            continue

        table = feather.read_table(prediction_file).to_pandas()
        available = burdens or sorted(
            column[: -len("_CRPS")] for column in table.columns if column.endswith("_CRPS")
        )

        per_burden = []
        for burden in available:
            columns = {f"{burden}_{score}": score for score in NODE_SCORES if f"{burden}_{score}" in table.columns}
            if not columns:
                continue
            node = table.groupby("timestep")[list(columns)].mean().rename(columns=columns).reset_index()
            node["burden"] = burden
            per_burden.append(node)
        if not per_burden:
            continue
        node_scores = pd.concat(per_burden, ignore_index=True)

        spatial = feather.read_table(spatial_file).to_pandas()
        spatial = spatial[spatial["burden"].isin(available)]
        keep = ["timestep", "burden"] + [c for c in SPATIAL_SCORES if c in spatial.columns]
        merged = node_scores.merge(spatial[keep], on=["timestep", "burden"], how="outer")

        merged.insert(0, "run", name)
        merged.insert(1, "sim_id", sim_id)
        merged.insert(2, "window_start", window_start)
        merged["lead"] = merged["timestep"] - window_start + 1
        frames.append(merged)

    if not frames:
        raise ValueError(f"No usable windows found for run '{name}' under {results_dir}")
    return pd.concat(frames, ignore_index=True)


def restrict_to_shared_grid(long_df: pd.DataFrame) -> pd.DataFrame:
    """Keep only (sim, window, lead, burden) cells that *every* run produced.

    Without this, a run that stopped early would be compared on an easier subset of
    the epidemic and look better than it is.
    """
    keys = ["sim_id", "window_start", "lead", "burden"]
    runs = long_df["run"].nunique()
    counts = long_df.groupby(keys)["run"].nunique()
    shared = counts[counts == runs].index
    if len(shared) == 0:
        raise ValueError("The runs share no common (sim, window, lead, burden) cells; cannot compare them.")

    filtered = long_df.set_index(keys).loc[shared].reset_index()
    dropped = len(long_df) - len(filtered)
    if dropped:
        print(f"[compare] Restricted to the shared evaluation grid: dropped {dropped:,} of {len(long_df):,} rows.")
    return filtered


def skill_scores(long_df: pd.DataFrame, reference: str) -> pd.DataFrame:
    """``1 - score / reference`` per (run, burden, score). Positive means better."""
    if reference not in set(long_df["run"]):
        raise ValueError(f"Reference run '{reference}' is not among {sorted(set(long_df['run']))}")

    scores = [c for c in ALL_SCORES if c in long_df.columns]
    means = long_df.groupby(["run", "burden"])[scores].mean()
    baseline = means.xs(reference, level="run")

    rows = []
    for (run, burden), values in means.iterrows():
        for score in scores:
            reference_value = baseline.loc[burden, score]
            rows.append(
                {
                    "run": run,
                    "burden": burden,
                    "score": score,
                    "value": values[score],
                    "reference_value": reference_value,
                    "skill_vs_reference": (
                        np.nan if not np.isfinite(reference_value) or reference_value == 0
                        else 1.0 - values[score] / reference_value
                    ),
                }
            )
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, float_format: str = "{:.4f}") -> str:
    """Render a DataFrame as a markdown table without pulling in ``tabulate``."""
    def cell(value) -> str:
        if isinstance(value, float) and np.isfinite(value):
            return float_format.format(value)
        return "" if value is None or (isinstance(value, float) and not np.isfinite(value)) else str(value)

    headers = [str(c) for c in frame.columns]
    rows = [[cell(v) for v in record] for record in frame.itertuples(index=False, name=None)]
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) if rows else len(headers[i])
              for i in range(len(headers))]

    def line(values: list[str]) -> str:
        return "| " + " | ".join(v.ljust(widths[i]) for i, v in enumerate(values)) + " |"

    return "\n".join([line(headers), "|" + "|".join("-" * (w + 2) for w in widths) + "|", *(line(r) for r in rows)])


def write_markdown(output_dir: Path, by_run: pd.DataFrame, skill: pd.DataFrame, reference: str) -> None:
    pivot = (
        skill.pivot_table(index=["burden", "score"], columns="run", values="skill_vs_reference")
        .reset_index()
    )
    lines = [
        "# Run comparison",
        "",
        "Lower is better for every score. Skill is `1 - score / reference`, so **positive means better "
        f"than `{reference}`**.",
        "",
        "## Mean scores (shared evaluation grid)",
        "",
        markdown_table(by_run),
        "",
        f"## Skill against `{reference}`",
        "",
        markdown_table(pivot, "{:+.4f}"),
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--run", action="append", required=True, metavar="NAME=DIR",
        help="A results tree to include; repeat for each run. DIR must contain TESTING/.",
    )
    parser.add_argument("--reference", type=str, help="Run to compute skill against (default: the first --run)")
    parser.add_argument("--burdens", type=str, nargs="*", help="Restrict to these burden columns")
    parser.add_argument("--output", type=str, required=True, help="Directory for the comparison outputs")
    parser.add_argument(
        "--no-shared-grid", action="store_true",
        help="Compare on each run's own windows instead of the intersection (not paired)",
    )
    args = parser.parse_args()

    runs = {}
    for entry in args.run:
        if "=" not in entry:
            parser.error(f"--run expects NAME=DIR, got '{entry}'")
        name, path = entry.split("=", 1)
        runs[name] = path

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for name, path in runs.items():
        print(f"[compare] Loading {name} from {path}")
        frame = load_run(name, path, args.burdens)
        print(f"[compare]   {len(frame):,} rows | {frame['sim_id'].nunique()} sims | "
              f"{frame['window_start'].nunique()} windows | leads 1..{int(frame['lead'].max())}")
        frames.append(frame)

    long_df = pd.concat(frames, ignore_index=True)
    if not args.no_shared_grid and len(runs) > 1:
        long_df = restrict_to_shared_grid(long_df)
    long_df.to_csv(output_dir / "scores_long.csv", index=False)

    scores = [c for c in ALL_SCORES if c in long_df.columns]
    by_run = long_df.groupby(["run", "burden"])[scores].mean().reset_index()
    by_run.to_csv(output_dir / "summary_by_run.csv", index=False)
    long_df.groupby(["run", "burden", "lead"])[scores].mean().reset_index().to_csv(
        output_dir / "summary_by_lead.csv", index=False
    )

    reference = args.reference or next(iter(runs))
    skill = skill_scores(long_df, reference)
    skill.to_csv(output_dir / f"skill_vs_{reference}.csv", index=False)
    write_markdown(output_dir, by_run, skill, reference)

    print(f"\n[compare] Wrote {output_dir}/summary.md and 4 CSVs")
    print(by_run.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
