"""
Order Flow Imbalance Backtest

Exit logic: Exit when OFI crosses zero (flow exhaustion).
"""
import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
    commission_per_side: float = 0.85
    tick_size: float = 0.25
    tick_value: float = 1.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OFI Backtest.")
    parser.add_argument("--input", default="data/sprint_with_ofi.parquet", help="Input data path.")
    parser.add_argument("--output", default="results/backtest_ofi_results.csv", help="Trade log output.")
    parser.add_argument("--thresholds", default="0.3,0.4,0.5", help="OFI thresholds to backtest.")
    parser.add_argument("--commission", type=float, default=0.85, help="Commission per side.")
    parser.add_argument("--tick-size", type=float, default=0.25)
    parser.add_argument("--tick-value", type=float, default=1.25)
    parser.add_argument("--train-end", default="2025-02-28")
    parser.add_argument("--test-start", default="2025-03-01")
    parser.add_argument("--max-hold", type=int, default=10000, help="Max hold time in ticks (safety).")
    return parser.parse_args()


def run_backtest(df: pd.DataFrame, signal_col: str, cfg: BacktestConfig, max_hold: int) -> list[dict]:
    """
    Backtest OFI signals with flow-reversal exit.

    Entry: Signal at t fills at t+1 (long at ask, short at bid)
    Exit: When OFI crosses zero from entry direction, or max hold reached
    """
    trades = []
    n = len(df)
    if n < 2:
        return trades

    signal = df[signal_col].to_numpy()
    ofi = df["ofi"].to_numpy()
    bid = df["bid"].to_numpy()
    ask = df["ask"].to_numpy()
    timestamps = df["timestamp"].to_numpy()

    position = 0
    entry_idx = 0
    entry_price = 0.0

    for i in range(1, n):
        if position == 0:
            prev_signal = signal[i - 1]
            if prev_signal == 1:
                position = 1
                entry_idx = i
                entry_price = ask[i]
            elif prev_signal == -1:
                position = -1
                entry_idx = i
                entry_price = bid[i]
        else:
            hold_time = i - entry_idx
            current_ofi = ofi[i]

            should_exit = False
            # Exit on flow reversal or max hold
            if position == 1:
                # Long: exit when OFI goes negative (sellers take over)
                if current_ofi <= 0 or hold_time >= max_hold:
                    should_exit = True
                    exit_price = bid[i]
            else:
                # Short: exit when OFI goes positive (buyers take over)
                if current_ofi >= 0 or hold_time >= max_hold:
                    should_exit = True
                    exit_price = ask[i]

            if should_exit:
                if position == 1:
                    gross_pnl = (exit_price - entry_price) / cfg.tick_size * cfg.tick_value
                else:
                    gross_pnl = (entry_price - exit_price) / cfg.tick_size * cfg.tick_value

                net_pnl = gross_pnl - 2 * cfg.commission_per_side
                ticks = gross_pnl / cfg.tick_value

                trades.append({
                    "entry_idx": entry_idx,
                    "exit_idx": i,
                    "direction": "long" if position == 1 else "short",
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "entry_time": pd.Timestamp(timestamps[entry_idx]),
                    "exit_time": pd.Timestamp(timestamps[i]),
                    "hold_ticks": hold_time,
                    "gross_pnl": gross_pnl,
                    "net_pnl": net_pnl,
                    "ticks": ticks,
                })
                position = 0

    return trades


def compute_metrics(trades: list[dict]) -> dict:
    if not trades:
        return {
            "num_trades": 0,
            "total_net_pnl": 0.0,
            "win_rate": 0.0,
            "avg_trade_net": 0.0,
            "avg_trade_ticks": 0.0,
            "profit_factor": 0.0,
            "avg_hold_ticks": 0.0,
        }

    net_pnls = [t["net_pnl"] for t in trades]
    gross_pnls = [t["gross_pnl"] for t in trades]
    ticks = [t["ticks"] for t in trades]
    holds = [t["hold_ticks"] for t in trades]

    wins = [p for p in net_pnls if p > 0]
    gross_profit = sum(p for p in gross_pnls if p > 0)
    gross_loss = abs(sum(p for p in gross_pnls if p < 0))

    return {
        "num_trades": len(trades),
        "total_net_pnl": sum(net_pnls),
        "win_rate": len(wins) / len(trades) * 100 if trades else 0.0,
        "avg_trade_net": np.mean(net_pnls),
        "avg_trade_ticks": np.mean(ticks),
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else float("inf"),
        "avg_hold_ticks": np.mean(holds),
    }


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    cfg = BacktestConfig(
        commission_per_side=args.commission,
        tick_size=args.tick_size,
        tick_value=args.tick_value,
    )

    if not input_path.exists():
        print(f"Input not found: {input_path}")
        return 1

    print(f"Loading {input_path}...")
    if input_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path, parse_dates=["timestamp"])

    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Split train/test
    train_end = pd.Timestamp(args.train_end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    test_start = pd.Timestamp(args.test_start, tz="UTC")

    train_df = df[df["timestamp"] <= train_end].copy()
    test_df = df[df["timestamp"] >= test_start].copy()

    print(f"Train rows: {len(train_df):,}")
    print(f"Test rows: {len(test_df):,}")
    print()

    thresholds = [float(t.strip()) for t in args.thresholds.split(",") if t.strip()]
    all_results = []

    for thresh in thresholds:
        signal_col = f"signal_ofi_{str(thresh).replace('.', '_')}"
        if signal_col not in df.columns:
            print(f"Warning: {signal_col} not found, skipping.")
            continue

        print(f"=== OFI Threshold = {thresh} ===")

        # Train
        train_trades = run_backtest(train_df, signal_col, cfg, args.max_hold)
        train_m = compute_metrics(train_trades)
        print(f"TRAIN: {train_m['num_trades']} trades, "
              f"Net P&L: ${train_m['total_net_pnl']:.2f}, "
              f"Win Rate: {train_m['win_rate']:.1f}%, "
              f"PF: {train_m['profit_factor']:.2f}, "
              f"Avg Ticks: {train_m['avg_trade_ticks']:.2f}, "
              f"Avg Hold: {train_m['avg_hold_ticks']:.0f}")

        # Test
        test_trades = run_backtest(test_df, signal_col, cfg, args.max_hold)
        test_m = compute_metrics(test_trades)
        print(f"TEST:  {test_m['num_trades']} trades, "
              f"Net P&L: ${test_m['total_net_pnl']:.2f}, "
              f"Win Rate: {test_m['win_rate']:.1f}%, "
              f"PF: {test_m['profit_factor']:.2f}, "
              f"Avg Ticks: {test_m['avg_trade_ticks']:.2f}, "
              f"Avg Hold: {test_m['avg_hold_ticks']:.0f}")

        # Go/No-Go
        gates_passed = (
            test_m["total_net_pnl"] > 0 and
            test_m["profit_factor"] >= 1.1 and
            test_m["avg_trade_ticks"] >= 1.0 and
            test_m["num_trades"] >= 30
        )
        print(f"GO/NO-GO: {'GO' if gates_passed else 'NO-GO'}")
        print()

        for t in test_trades:
            t["threshold"] = thresh
            t["period"] = "test"
            all_results.append(t)

    if all_results:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_results).to_csv(output_path, index=False)
        print(f"Trade log saved: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
