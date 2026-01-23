import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest MES tick data with integrity checks.")
    parser.add_argument("--data-dir", default="datasets/MES/tick_data", help="Directory of raw CSV files.")
    parser.add_argument("--start-date", default="2025-01-01", help="Inclusive start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", default="2025-03-31", help="Inclusive end date (YYYY-MM-DD).")
    parser.add_argument("--contract", default=None, help="Optional contract filter (exact match).")
    parser.add_argument("--tick-size", type=float, default=0.25, help="Tick size for price grid checks.")
    parser.add_argument("--max-spread-ticks", type=int, default=10, help="Max spread in ticks.")
    parser.add_argument("--chunksize", type=int, default=1_000_000, help="CSV read chunksize.")
    parser.add_argument("--out", default="data/sprint_data.parquet", help="Output file path.")
    return parser.parse_args()


def off_grid_mask(series: pd.Series, tick_size: float, eps: float = 1e-8) -> pd.Series:
    if series.empty:
        return series.astype(bool)
    rounded = (series / tick_size).round() * tick_size
    return (series - rounded).abs() > eps


def update_contract_stats(
    stats: dict, contract_series: pd.Series, timestamp_series: pd.Series
) -> None:
    if contract_series.empty:
        return
    counts = contract_series.value_counts(dropna=False)
    stats["contract_counts"].update(counts.to_dict())
    for contract, count in counts.items():
        subset = timestamp_series[contract_series == contract]
        if subset.empty:
            continue
        min_ts = subset.min()
        max_ts = subset.max()
        current_min, current_max = stats["contract_ranges"].get(contract, (None, None))
        stats["contract_ranges"][contract] = (
            min_ts if current_min is None else min(current_min, min_ts),
            max_ts if current_max is None else max(current_max, max_ts),
        )


def apply_integrity_rules(df: pd.DataFrame, cfg: argparse.Namespace, stats: dict) -> pd.DataFrame:
    stats["rows_total"] += len(df)

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    bad_ts = df["timestamp"].isna()
    stats["drop_bad_timestamp"] += int(bad_ts.sum())
    df = df.loc[~bad_ts].copy()

    start_ts = pd.Timestamp(cfg.start_date, tz="UTC")
    end_ts = pd.Timestamp(cfg.end_date, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    in_range = (df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)
    stats["drop_out_of_range"] += int((~in_range).sum())
    df = df.loc[in_range].copy()

    if cfg.contract:
        match_contract = df["contract"] == cfg.contract
        stats["drop_contract"] += int((~match_contract).sum())
        df = df.loc[match_contract].copy()

    missing_bid_ask = df["bid"].isna() | df["ask"].isna()
    stats["drop_missing_bid_ask"] += int(missing_bid_ask.sum())
    df = df.loc[~missing_bid_ask].copy()

    non_positive = (df["bid"] <= 0) | (df["ask"] <= 0) | (df["last"] <= 0) | (df["volume"] <= 0)
    stats["drop_non_positive"] += int(non_positive.sum())
    df = df.loc[~non_positive].copy()

    crossed = df["bid"] >= df["ask"]
    stats["drop_crossed"] += int(crossed.sum())
    df = df.loc[~crossed].copy()

    df["spread"] = df["ask"] - df["bid"]
    spread_too_wide = df["spread"] > (cfg.max_spread_ticks * cfg.tick_size)
    stats["drop_spread_too_wide"] += int(spread_too_wide.sum())
    df = df.loc[~spread_too_wide].copy()

    off_grid = off_grid_mask(df["bid"], cfg.tick_size) | off_grid_mask(df["ask"], cfg.tick_size) | off_grid_mask(
        df["last"], cfg.tick_size
    )
    stats["drop_off_grid"] += int(off_grid.sum())
    df = df.loc[~off_grid].copy()

    df["mid_price"] = (df["bid"] + df["ask"]) / 2.0

    update_contract_stats(stats, df["contract"], df["timestamp"])

    return df


def main() -> int:
    cfg = parse_args()
    data_dir = Path(cfg.data_dir)
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return 1

    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in: {data_dir}")
        return 1

    stats = {
        "rows_total": 0,
        "rows_kept": 0,
        "drop_bad_timestamp": 0,
        "drop_out_of_range": 0,
        "drop_contract": 0,
        "drop_missing_bid_ask": 0,
        "drop_non_positive": 0,
        "drop_crossed": 0,
        "drop_spread_too_wide": 0,
        "drop_off_grid": 0,
        "drop_exact_duplicates": 0,
        "contract_counts": Counter(),
        "contract_ranges": {},
        "out_of_order": False,
    }

    chunks = []
    last_ts = None
    for path in csv_files:
        for chunk in pd.read_csv(path, chunksize=cfg.chunksize):
            filtered = apply_integrity_rules(chunk, cfg, stats)
            if filtered.empty:
                continue
            if last_ts is not None and (filtered["timestamp"].min() < last_ts):
                stats["out_of_order"] = True
            last_ts = filtered["timestamp"].max()
            chunks.append(filtered)

    if not chunks:
        print("No rows left after filtering.")
        return 0

    df = pd.concat(chunks, ignore_index=True)
    before_dupes = len(df)
    df = df.drop_duplicates()
    stats["drop_exact_duplicates"] += before_dupes - len(df)

    if stats["out_of_order"]:
        df = df.sort_values("timestamp").reset_index(drop=True)

    stats["rows_kept"] = len(df)

    out_path = Path(cfg.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".parquet":
        try:
            df.to_parquet(out_path, index=False)
        except Exception as exc:
            print(f"Parquet write failed ({exc}); falling back to CSV.")
            out_path = out_path.with_suffix(".csv")
            df.to_csv(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)

    print("Ingest complete.")
    print(f"Rows total: {stats['rows_total']}")
    print(f"Rows kept: {stats['rows_kept']}")
    print("Drops:")
    print(f"  bad_timestamp: {stats['drop_bad_timestamp']}")
    print(f"  out_of_range: {stats['drop_out_of_range']}")
    print(f"  contract_filter: {stats['drop_contract']}")
    print(f"  missing_bid_ask: {stats['drop_missing_bid_ask']}")
    print(f"  non_positive: {stats['drop_non_positive']}")
    print(f"  crossed: {stats['drop_crossed']}")
    print(f"  spread_too_wide: {stats['drop_spread_too_wide']}")
    print(f"  off_grid: {stats['drop_off_grid']}")
    print(f"  exact_duplicates: {stats['drop_exact_duplicates']}")
    if stats["out_of_order"]:
        print("Note: timestamps were out of order; output sorted by timestamp.")

    if stats["contract_counts"]:
        print("Contracts observed:")
        for contract, count in stats["contract_counts"].most_common():
            min_ts, max_ts = stats["contract_ranges"].get(contract, (None, None))
            print(f"  {contract}: {count} rows, {min_ts} to {max_ts}")

    print(f"Output: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
