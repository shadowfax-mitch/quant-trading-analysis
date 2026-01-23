"""
Contrarian Order Flow Imbalance Signal

Fade extreme flow: go short when buyers dominate (exhaustion),
go long when sellers dominate (capitulation).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Contrarian OFI signal generation.")
    parser.add_argument("--input", default="data/sprint_data.parquet")
    parser.add_argument("--output", default="data/sprint_with_ofi_contrarian.parquet")
    parser.add_argument("--window", type=int, default=1000)
    parser.add_argument("--thresholds", default="0.15,0.20,0.25")
    parser.add_argument("--buy-side", default="A")
    parser.add_argument("--sell-side", default="B")
    return parser.parse_args()


def compute_ofi(df: pd.DataFrame, window: int, buy_side: str, sell_side: str) -> pd.Series:
    buy_vol = np.where(df["side"] == buy_side, df["volume"], 0).astype(float)
    sell_vol = np.where(df["side"] == sell_side, df["volume"], 0).astype(float)
    roll_buy = pd.Series(buy_vol).rolling(window, min_periods=window).sum()
    roll_sell = pd.Series(sell_vol).rolling(window, min_periods=window).sum()
    total = roll_buy + roll_sell
    imbalance = (roll_buy - roll_sell) / total
    return imbalance.replace([np.inf, -np.inf], np.nan)


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    print(f"Loading {input_path}...")
    df = pd.read_parquet(input_path) if input_path.suffix == ".parquet" else pd.read_csv(input_path)
    print(f"Rows: {len(df):,}")

    df["ofi"] = compute_ofi(df, args.window, args.buy_side, args.sell_side)
    valid = df["ofi"].notna()

    thresholds = [float(t.strip()) for t in args.thresholds.split(",") if t.strip()]

    for thresh in thresholds:
        col = f"signal_ofi_ctr_{str(thresh).replace('.', '_')}"
        signal = np.zeros(len(df), dtype=np.int8)
        ofi_lagged = df["ofi"].shift(1)

        # CONTRARIAN: short when OFI high (buyer exhaustion), long when OFI low (seller exhaustion)
        short_mask = valid & (ofi_lagged > thresh)  # buyers dominated -> fade -> short
        long_mask = valid & (ofi_lagged < -thresh)  # sellers dominated -> fade -> long

        signal[long_mask.to_numpy()] = 1
        signal[short_mask.to_numpy()] = -1
        df[col] = signal

        counts = pd.Series(signal).value_counts().to_dict()
        print(f"{col}: {counts}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
