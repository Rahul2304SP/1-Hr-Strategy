#!/usr/bin/env python
"""
Aggregate statistics for the generated 1h strategy trade windows.

Reads every CSV under `buy/` and `short/`, computes max return, max drawdown,
and closing return relative to the 11:00 entry price, and writes both the
per-trade metrics and per-side summaries to `analysis/`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import pandas as pd


@dataclass(frozen=True)
class TradeMetrics:
    trade_day: str
    signal_day: str
    direction: str
    entry_price: float
    exit_price: float
    final_return: float
    final_return_pct: float
    max_return: float
    max_return_pct: float
    max_drawdown: float
    max_drawdown_pct: float
    hours_captured: int


def load_trade_csv(path: Path) -> pd.DataFrame:
    """Load a single trade CSV and validate expected columns."""
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Trade CSV {path} is empty.")
    required = {
        "open",
        "high",
        "low",
        "close",
        "direction",
        "entry_price",
        "trade_day",
        "signal_day",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing} in {path}.")
    return df


def compute_buy_metrics(df: pd.DataFrame) -> TradeMetrics:
    """Compute trade metrics for a long window."""
    entry_price = float(df.iloc[0]["entry_price"])
    exit_price = float(df.iloc[-1]["close"])
    max_price = float(max(df["close"].max(), df["high"].max()))
    min_price = float(df["low"].min())

    max_return = max(0.0, max_price - entry_price)
    max_drawdown = max(0.0, entry_price - min_price)
    final_return = exit_price - entry_price

    trade_day = str(df.iloc[0]["trade_day"])
    signal_day = str(df.iloc[0]["signal_day"])

    return TradeMetrics(
        trade_day=trade_day,
        signal_day=signal_day,
        direction="buy",
        entry_price=entry_price,
        exit_price=exit_price,
        final_return=final_return,
        final_return_pct=(final_return / entry_price) * 100.0,
        max_return=max_return,
        max_return_pct=(max_return / entry_price) * 100.0,
        max_drawdown=max_drawdown,
        max_drawdown_pct=(max_drawdown / entry_price) * 100.0,
        hours_captured=int(df.shape[0]),
    )


def compute_short_metrics(df: pd.DataFrame) -> TradeMetrics:
    """Compute trade metrics for a short window."""
    entry_price = float(df.iloc[0]["entry_price"])
    exit_price = float(df.iloc[-1]["close"])
    min_price = float(min(df["close"].min(), df["low"].min()))
    max_price = float(max(df["close"].max(), df["high"].max()))

    max_return = max(0.0, entry_price - min_price)
    max_drawdown = max(0.0, max_price - entry_price)
    final_return = entry_price - exit_price

    trade_day = str(df.iloc[0]["trade_day"])
    signal_day = str(df.iloc[0]["signal_day"])

    return TradeMetrics(
        trade_day=trade_day,
        signal_day=signal_day,
        direction="short",
        entry_price=entry_price,
        exit_price=exit_price,
        final_return=final_return,
        final_return_pct=(final_return / entry_price) * 100.0,
        max_return=max_return,
        max_return_pct=(max_return / entry_price) * 100.0,
        max_drawdown=max_drawdown,
        max_drawdown_pct=(max_drawdown / entry_price) * 100.0,
        hours_captured=int(df.shape[0]),
    )


def gather_trade_metrics(paths: Iterable[Path], direction: str) -> List[TradeMetrics]:
    """Load each path and compute the associated metrics."""
    metrics: List[TradeMetrics] = []
    compute_fn = compute_buy_metrics if direction == "buy" else compute_short_metrics
    for path in sorted(paths):
        df = load_trade_csv(path)
        metrics.append(compute_fn(df))
    return metrics


def write_metrics_csv(rows: List[TradeMetrics], path: Path) -> None:
    """Write the per-trade metrics to disk."""
    if not rows:
        # Still write an empty CSV with headers for consistency.
        columns = [
            "trade_day",
            "signal_day",
            "direction",
            "entry_price",
            "exit_price",
            "final_return",
            "final_return_pct",
            "max_return",
            "max_return_pct",
            "max_drawdown",
            "max_drawdown_pct",
            "hours_captured",
        ]
        pd.DataFrame(columns=columns).to_csv(path, index=False)
        return

    df = pd.DataFrame(
        [
            {
                "trade_day": row.trade_day,
                "signal_day": row.signal_day,
                "direction": row.direction,
                "entry_price": row.entry_price,
                "exit_price": row.exit_price,
                "final_return": row.final_return,
                "final_return_pct": row.final_return_pct,
                "max_return": row.max_return,
                "max_return_pct": row.max_return_pct,
                "max_drawdown": row.max_drawdown,
                "max_drawdown_pct": row.max_drawdown_pct,
                "hours_captured": row.hours_captured,
            }
            for row in rows
        ]
    )
    df.sort_values("trade_day").to_csv(path, index=False)


def summarize_metrics(rows: List[TradeMetrics], direction: str) -> dict:
    """Compute aggregate statistics for a set of trades."""
    count = len(rows)
    sum_final = float(sum(row.final_return for row in rows))
    sum_max_return = float(sum(row.max_return for row in rows))
    sum_max_drawdown = float(sum(row.max_drawdown for row in rows))
    avg_final = sum_final / count if count else 0.0
    avg_max_return = sum_max_return / count if count else 0.0
    avg_max_drawdown = sum_max_drawdown / count if count else 0.0

    return {
        "direction": direction,
        "trade_count": count,
        "sum_final_return": sum_final,
        "sum_max_return": sum_max_return,
        "sum_max_drawdown": sum_max_drawdown,
        "avg_final_return": avg_final,
        "avg_max_return": avg_max_return,
        "avg_max_drawdown": avg_max_drawdown,
    }


def gather_csv_paths(folder: Path) -> List[Path]:
    """Return all CSV files within the supplied folder."""
    if not folder.exists():
        return []
    return sorted(p for p in folder.glob("*.csv") if p.is_file())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate buy/short metrics for the 1h strategy.",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory that contains the buy/ and short/ subfolders (default: script directory).",
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=(Path(__file__).resolve().parent / "analysis"),
        help="Destination for the per-trade and summary CSVs (default: base_dir/analysis).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    buy_paths = gather_csv_paths(args.base_dir / "buy")
    short_paths = gather_csv_paths(args.base_dir / "short")

    buy_metrics = gather_trade_metrics(buy_paths, "buy")
    short_metrics = gather_trade_metrics(short_paths, "short")

    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    write_metrics_csv(buy_metrics, args.analysis_dir / "buy_metrics.csv")
    write_metrics_csv(short_metrics, args.analysis_dir / "short_metrics.csv")

    summary_rows = [
        summarize_metrics(buy_metrics, "buy"),
        summarize_metrics(short_metrics, "short"),
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(args.analysis_dir / "summary.csv", index=False)

    print(f"Wrote analysis outputs to {args.analysis_dir}")


if __name__ == "__main__":
    main()
