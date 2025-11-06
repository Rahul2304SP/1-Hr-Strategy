#!/usr/bin/env python
"""
Visualize outcome distributions for the 1h strategy trades.

Reads the per-trade metric CSVs produced by analyze_trades.py and creates
overlayed histograms (buy vs short) for the requested metrics.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd


def load_metrics(analysis_dir: Path) -> pd.DataFrame:
    """Load and merge buy/short metrics from the analysis directory."""
    frames: List[pd.DataFrame] = []
    for side in ("buy", "short"):
        path = analysis_dir / f"{side}_metrics.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if not df.empty and "direction" not in df.columns:
            df["direction"] = side
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No metrics CSVs found in {analysis_dir}. Run analyze_trades.py first."
        )

    return pd.concat(frames, ignore_index=True)


def plot_distribution(
    df: pd.DataFrame,
    column: str,
    bins: int,
    output_path: Path,
) -> None:
    """Create an overlapped histogram for the specified metric column."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not present in metrics dataframe.")

    plt.figure(figsize=(10, 6))
    for direction, subset in df.groupby("direction"):
        plt.hist(
            subset[column],
            bins=bins,
            alpha=0.6,
            label=f"{direction.capitalize()} (n={len(subset)})",
        )
        mean_val = subset[column].mean()
        plt.axvline(
            mean_val,
            color="tab:blue" if direction == "buy" else "tab:orange",
            linestyle="--",
            linewidth=1.2,
        )

    plt.title(f"Distribution of {column.replace('_', ' ').title()}")
    plt.xlabel(column.replace("_", " ").title())
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize the distribution of trade outcomes.",
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "analysis",
        help="Directory containing buy_metrics.csv and short_metrics.csv (default: 1hrStrategy/analysis).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "analysis" / "plots",
        help="Folder to store generated plots (default: analysis/plots).",
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        default=["final_return", "final_return_pct"],
        help="Metric columns to visualize (default: final_return final_return_pct).",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=40,
        help="Number of histogram bins (default: 40).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_metrics(args.analysis_dir)
    for column in args.columns:
        output_path = args.output_dir / f"{column}_distribution.png"
        plot_distribution(df, column, bins=args.bins, output_path=output_path)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
