"""
Order Flow Imbalance Signal Generator

Computes rolling buy/sell volume imbalance and generates signals
when imbalance reaches extreme levels.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Order Flow Imbalance signal generation.")
    parser.add_argument("--input", default="data/sprint_data.parquet", help="Input data path.")
    parser.add_argument("--output", default="data/sprint_with_ofi.parquet", help="Output data path.")
    parser.add_argument("--window", type=int, default=1000, help="Rolling window size (ticks).")
    parser.add_argument("--thresholds", default="0.3,0.4,0.5", help="Imbalance thresholds (comma-separated).")
    parser.add_argument("--buy-side", default="A", help="Side value indicating buyer-initiated.")
    parser.add_argument("--sell-side", default="B", help="Side value indicating seller-initiated.")
    return parser.parse_args()


def compute_ofi(
    df: pd.DataFrame,
    window: int,
    buy_side: str = "A",
    sell_side: str = "B",
) -> pd.Series:
    """
    Compute Order Flow Imbalance as rolling (buy_vol - sell_vol) / (buy_vol + sell_vol).

    Returns values in [-1, 1] where:
    - Positive = buyer-dominated
    - Negative = seller-dominated
    """
    # Create buy/sell volume columns
    buy_vol = np.where(df["side"] == buy_side, df["volume"], 0).astype(float)
    sell_vol = np.where(df["side"] == sell_side, df["volume"], 0).astype(float)

    # Rolling sums
    roll_buy = pd.Series(buy_vol).rolling(window, min_periods=window).sum()
    roll_sell = pd.Series(sell_vol).rolling(window, min_periods=window).sum()

    # Imbalance ratio
    total = roll_buy + roll_sell
    imbalance = (roll_buy - roll_sell) / total
    imbalance = imbalance.replace([np.inf, -np.inf], np.nan)

    return imbalance


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Input not found: {input_path}")
        return 1

    print(f"Loading {input_path}...")
    if input_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path, parse_dates=["timestamp"])

    print(f"Rows: {len(df):,}")
    print(f"Computing OFI with window={args.window}...")

    # Compute order flow imbalance
    df["ofi"] = compute_ofi(df, args.window, args.buy_side, args.sell_side)

    valid = df["ofi"].notna()
    print(f"Valid OFI ticks: {valid.sum():,}")

    # Generate signals for each threshold
    thresholds = [float(t.strip()) for t in args.thresholds.split(",") if t.strip()]

    for thresh in thresholds:
        col = f"signal_ofi_{str(thresh).replace('.', '_')}"
        signal = np.zeros(len(df), dtype=np.int8)

        # Long when buyers dominate (OFI > threshold)
        # Short when sellers dominate (OFI < -threshold)
        # Lagged by 1 tick to avoid look-ahead
        ofi_lagged = df["ofi"].shift(1)

        long_mask = valid & (ofi_lagged > thresh)
        short_mask = valid & (ofi_lagged < -thresh)

        signal[long_mask.to_numpy()] = 1
        signal[short_mask.to_numpy()] = -1
        df[col] = signal

        counts = pd.Series(signal).value_counts().to_dict()
        print(f"{col}: {counts}")

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".parquet":
        try:
            df.to_parquet(output_path, index=False)
        except Exception as exc:
            output_path = output_path.with_suffix(".csv")
            df.to_csv(output_path, index=False)
            print(f"Parquet write failed ({exc}); wrote CSV.")
    else:
        df.to_csv(output_path, index=False)

    # Print OFI distribution
    ofi_valid = df.loc[valid, "ofi"]
    print(f"\nOFI distribution:")
    print(f"  min:    {ofi_valid.min():.4f}")
    print(f"  25%:    {ofi_valid.quantile(0.25):.4f}")
    print(f"  median: {ofi_valid.median():.4f}")
    print(f"  75%:    {ofi_valid.quantile(0.75):.4f}")
    print(f"  max:    {ofi_valid.max():.4f}")

    print(f"\nOutput: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
