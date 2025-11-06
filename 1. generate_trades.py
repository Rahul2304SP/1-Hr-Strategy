#!/usr/bin/env python
"""
Generate 1h strategy trade windows.

For every trading day in the source CSV, inspect the 23:00 candle. If that
hour's close is below its open we plan to go long the following day; otherwise,
if the close is above the open we plan to go short. The resulting trade is
entered at a configurable intraday hour (default 11:00) on the following day
and monitored until 21:00.

Each trade window (entry hour through 21:00) is written to a dedicated CSV
under `buy/` or `short/` depending on the direction.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

# Default path requested by the user.
DEFAULT_DATA_PATH = Path(r"E:/Data/US30 1h Data/BLACKBULL_US30, 60.csv")


@dataclass(frozen=True)
class TradeWindow:
    signal_day: pd.Timestamp
    trade_day: pd.Timestamp
    direction: str  # "buy" or "short"
    entry_price: float
    data: pd.DataFrame


def load_price_data(csv_path: Path) -> pd.DataFrame:
    """Load and normalize the US30 hourly price data."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"time", "open", "high", "low", "close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing} in {csv_path}")

    df["timestamp"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["date"] = df["timestamp"].dt.tz_convert(None).dt.date
    df["hour"] = df["timestamp"].dt.hour
    return df


def determine_signal(day_df: pd.DataFrame) -> Optional[str]:
    """Return 'buy', 'short', or None for the provided day's 23:00 candle."""
    hour_23 = day_df.loc[day_df["hour"] == 23]
    if hour_23.empty:
        return None

    row = hour_23.iloc[0]
    if row["close"] < row["open"]:
        return "buy"
    if row["close"] > row["open"]:
        return "short"
    return None


def build_trade_window(
    trade_df: pd.DataFrame,
    signal_day: pd.Timestamp,
    direction: str,
    entry_hour: int,
    exit_hour: int,
) -> Optional[TradeWindow]:
    """Slice the next day's data between entry and exit hours (inclusive)."""
    entry_rows = trade_df.loc[trade_df["hour"] == entry_hour]
    exit_rows = trade_df.loc[trade_df["hour"] == exit_hour]
    if entry_rows.empty or exit_rows.empty:
        return None

    mask = (trade_df["hour"] >= entry_hour) & (trade_df["hour"] <= exit_hour)
    window = trade_df.loc[mask].copy()
    if window.empty:
        return None

    entry_price = float(entry_rows.iloc[0]["open"])
    trade_day = pd.Timestamp(window.iloc[0]["date"])

    window["signal_day"] = signal_day
    window["trade_day"] = trade_day.date()
    window["direction"] = direction
    window["entry_price"] = entry_price
    window["hours_from_entry"] = window["hour"] - entry_hour

    if direction == "buy":
        price_delta = window["close"] - entry_price
    else:
        price_delta = entry_price - window["close"]
    window["price_delta_from_entry"] = price_delta
    window["return_pct_from_entry"] = (price_delta / entry_price) * 100.0

    return TradeWindow(
        signal_day=pd.Timestamp(signal_day),
        trade_day=pd.Timestamp(trade_day),
        direction=direction,
        entry_price=entry_price,
        data=window.reset_index(drop=True),
    )


def iter_daily_data(df: pd.DataFrame) -> Iterable[pd.DataFrame]:
    """Yield daily dataframes in chronological order."""
    for _, day_df in df.groupby("date", sort=True):
        yield day_df.reset_index(drop=True)


def generate_trade_windows(
    df: pd.DataFrame,
    entry_hour: int,
    exit_hour: int,
) -> List[TradeWindow]:
    """Generate trade windows according to the provided configuration."""
    days = list(iter_daily_data(df))
    trade_windows: List[TradeWindow] = []

    for idx in range(len(days) - 1):
        signal_day_df = days[idx]
        trade_day_df = days[idx + 1]
        signal_day = pd.Timestamp(signal_day_df["date"].iloc[0])

        direction = determine_signal(signal_day_df)
        if direction is None:
            continue

        trade_window = build_trade_window(
            trade_df=trade_day_df,
            signal_day=signal_day,
            direction=direction,
            entry_hour=entry_hour,
            exit_hour=exit_hour,
        )
        if trade_window:
            trade_windows.append(trade_window)

    return trade_windows


def write_trade_windows(windows: Iterable[TradeWindow], output_dir: Path) -> None:
    """Persist each trade window into the buy/short folders."""
    buy_dir = output_dir / "buy"
    short_dir = output_dir / "short"
    buy_dir.mkdir(parents=True, exist_ok=True)
    short_dir.mkdir(parents=True, exist_ok=True)

    for window in windows:
        target_dir = buy_dir if window.direction == "buy" else short_dir
        signal_str = window.signal_day.date().isoformat()
        trade_str = window.trade_day.date().isoformat()
        file_name = f"trade_{trade_str}_signal_{signal_str}_{window.direction}.csv"
        window.data.to_csv(target_dir / file_name, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 1h strategy trade CSVs from hourly price data.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the US30 hourly CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Folder that will receive the buy/short CSVs (default: script directory).",
    )
    parser.add_argument(
        "--entry-hour",
        type=int,
        default=11,
        help="Hour (0-23) used for entering trades on the signal day + 1 (default: 12).",
    )
    parser.add_argument(
        "--exit-hour",
        type=int,
        default=21,
        help="Hour (0-23) used for ending the recorded window (default: 21).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (0 <= args.entry_hour <= 23):
        raise ValueError("entry-hour must be between 0 and 23.")
    if not (0 <= args.exit_hour <= 23):
        raise ValueError("exit-hour must be between 0 and 23.")
    if args.exit_hour < args.entry_hour:
        raise ValueError("exit-hour must be >= entry-hour.")

    df = load_price_data(args.data_path)
    windows = generate_trade_windows(df, entry_hour=args.entry_hour, exit_hour=args.exit_hour)
    write_trade_windows(windows, output_dir=args.output_dir)
    print(f"Wrote {len(windows)} trade CSVs to {args.output_dir}")


if __name__ == "__main__":
    main()
