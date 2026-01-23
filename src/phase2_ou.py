import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ou_model import rolling_ou_params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2: OU estimation + signal generation.")
    parser.add_argument("--input", default="data/sprint_data.parquet", help="Input data path.")
    parser.add_argument("--output", default="data/sprint_with_ou.parquet", help="Output data path.")
    parser.add_argument("--window", type=int, default=10_000, help="Rolling window size (pair count).")
    parser.add_argument("--lag", type=int, default=1, help="Lag parameters by this many ticks.")
    parser.add_argument("--step", type=int, default=1, help="Compute parameters every N ticks.")
    parser.add_argument("--forward-fill", action="store_true", help="Forward fill params between steps.")
    parser.add_argument("--n-values", default="1.5,2.0,2.5", help="Comma-separated N values.")
    return parser.parse_args()


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def parse_n_values(raw: str) -> list[float]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    return values


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    df = load_data(input_path)
    if "mid_price" not in df.columns:
        if {"bid", "ask"} <= set(df.columns):
            df["mid_price"] = (df["bid"] + df["ask"]) / 2.0
        else:
            raise ValueError("Input must include mid_price or bid/ask columns.")

    mu, theta, sigma = rolling_ou_params(
        df["mid_price"].to_numpy(dtype=float),
        window=args.window,
        dt=1.0,
        lag=args.lag,
        step=args.step,
        forward_fill=args.forward_fill,
    )

    df["mu"] = mu
    df["theta"] = theta
    df["sigma"] = sigma

    valid = (df["theta"] > 0) & (df["sigma"] > 0)
    stationary_std = np.full(len(df), np.nan, dtype=float)
    stationary_std[valid.to_numpy()] = df.loc[valid, "sigma"] / np.sqrt(2.0 * df.loc[valid, "theta"])
    df["stationary_std"] = stationary_std
    n_values = parse_n_values(args.n_values)
    for n in n_values:
        col = f"signal_{str(n).replace('.', '_')}"
        signal = np.zeros(len(df), dtype=np.int8)
        long_mask = valid & (df["mid_price"] < (df["mu"] - n * df["stationary_std"]))
        short_mask = valid & (df["mid_price"] > (df["mu"] + n * df["stationary_std"]))
        signal[long_mask.to_numpy()] = 1
        signal[short_mask.to_numpy()] = -1
        df[col] = signal

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".parquet":
        try:
            df.to_parquet(output_path, index=False)
        except Exception as exc:
            output_path = output_path.with_suffix(".csv")
            df.to_csv(output_path, index=False)
            print(f"Parquet write failed ({exc}); wrote CSV instead.")
    else:
        df.to_csv(output_path, index=False)

    valid_count = int(valid.sum())
    print(f"Valid parameter ticks: {valid_count}")
    for n in n_values:
        col = f"signal_{str(n).replace('.', '_')}"
        counts = df[col].value_counts().to_dict()
        print(f"{col} counts: {counts}")
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
