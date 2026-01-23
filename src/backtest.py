import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
    commission_per_side: float = 0.85
    tick_size: float = 0.25
    tick_value: float = 1.25  # MES: $1.25 per tick (0.25 * $5)


@dataclass
class TradeResult:
    entry_idx: int
    exit_idx: int
    direction: int  # 1 = long, -1 = short
    entry_price: float
    exit_price: float
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    gross_pnl: float
    net_pnl: float
    ticks: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3: Vectorized tick-level backtest.")
    parser.add_argument("--input", default="data/sprint_with_ou.parquet", help="Input data path.")
    parser.add_argument("--output", default="results/backtest_results.csv", help="Trade log output path.")
    parser.add_argument("--n-values", default="1.5,2.0,2.5", help="Comma-separated N values to backtest.")
    parser.add_argument("--commission", type=float, default=0.85, help="Commission per side per contract.")
    parser.add_argument("--tick-size", type=float, default=0.25, help="Tick size.")
    parser.add_argument("--tick-value", type=float, default=1.25, help="Dollar value per tick.")
    parser.add_argument("--train-end", default="2025-02-28", help="End of train period (inclusive).")
    parser.add_argument("--test-start", default="2025-03-01", help="Start of test period (inclusive).")
    return parser.parse_args()


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, parse_dates=["timestamp"])


def parse_n_values(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def run_backtest(
    df: pd.DataFrame,
    signal_col: str,
    cfg: BacktestConfig,
) -> list[TradeResult]:
    """
    Vectorized backtest with proper entry/exit logic.

    Entry rules:
    - Signal at tick t fills at tick t+1
    - Long enters at ask, short enters at bid

    Exit rules:
    - Exit when mid_price crosses mu (using lagged mu)
    - Long exits at bid, short exits at ask
    """
    trades = []

    n = len(df)
    if n < 2:
        return trades

    signal = df[signal_col].to_numpy()
    mid = df["mid_price"].to_numpy()
    mu = df["mu"].to_numpy()
    bid = df["bid"].to_numpy()
    ask = df["ask"].to_numpy()
    timestamps = df["timestamp"].to_numpy()

    position = 0  # 0 = flat, 1 = long, -1 = short
    entry_idx = 0
    entry_price = 0.0

    for i in range(1, n):
        if position == 0:
            # Check for entry signal at previous tick
            prev_signal = signal[i - 1]
            if prev_signal == 1:
                # Enter long at ask
                position = 1
                entry_idx = i
                entry_price = ask[i]
            elif prev_signal == -1:
                # Enter short at bid
                position = -1
                entry_idx = i
                entry_price = bid[i]
        else:
            # Check for exit: mid crosses mu
            prev_mu = mu[i - 1]
            if np.isnan(prev_mu):
                continue

            should_exit = False
            if position == 1:
                # Long: exit when mid >= mu (crossed above)
                if mid[i] >= prev_mu:
                    should_exit = True
                    exit_price = bid[i]
            else:
                # Short: exit when mid <= mu (crossed below)
                if mid[i] <= prev_mu:
                    should_exit = True
                    exit_price = ask[i]

            if should_exit:
                # Calculate P&L
                if position == 1:
                    gross_pnl = (exit_price - entry_price) / cfg.tick_size * cfg.tick_value
                else:
                    gross_pnl = (entry_price - exit_price) / cfg.tick_size * cfg.tick_value

                net_pnl = gross_pnl - 2 * cfg.commission_per_side
                ticks = gross_pnl / cfg.tick_value

                trades.append(TradeResult(
                    entry_idx=entry_idx,
                    exit_idx=i,
                    direction=position,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    entry_time=pd.Timestamp(timestamps[entry_idx]),
                    exit_time=pd.Timestamp(timestamps[i]),
                    gross_pnl=gross_pnl,
                    net_pnl=net_pnl,
                    ticks=ticks,
                ))

                position = 0

    return trades


def compute_metrics(trades: list[TradeResult]) -> dict:
    if not trades:
        return {
            "num_trades": 0,
            "total_gross_pnl": 0.0,
            "total_net_pnl": 0.0,
            "win_rate": 0.0,
            "avg_trade_net": 0.0,
            "avg_trade_ticks": 0.0,
            "profit_factor": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
        }

    net_pnls = [t.net_pnl for t in trades]
    gross_pnls = [t.gross_pnl for t in trades]
    ticks = [t.ticks for t in trades]

    wins = [p for p in net_pnls if p > 0]
    losses = [p for p in net_pnls if p <= 0]

    gross_profit = sum(p for p in gross_pnls if p > 0)
    gross_loss = abs(sum(p for p in gross_pnls if p < 0))

    return {
        "num_trades": len(trades),
        "total_gross_pnl": sum(gross_pnls),
        "total_net_pnl": sum(net_pnls),
        "win_rate": len(wins) / len(trades) * 100 if trades else 0.0,
        "avg_trade_net": np.mean(net_pnls) if net_pnls else 0.0,
        "avg_trade_ticks": np.mean(ticks) if ticks else 0.0,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else float("inf"),
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
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

    df = load_data(input_path)

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Split into train/test
    train_end = pd.Timestamp(args.train_end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    test_start = pd.Timestamp(args.test_start, tz="UTC")

    train_df = df[df["timestamp"] <= train_end].copy()
    test_df = df[df["timestamp"] >= test_start].copy()

    print(f"Train rows: {len(train_df):,}")
    print(f"Test rows: {len(test_df):,}")
    print()

    n_values = parse_n_values(args.n_values)
    all_results = []

    for n in n_values:
        signal_col = f"signal_{str(n).replace('.', '_')}"
        if signal_col not in df.columns:
            print(f"Warning: {signal_col} not found, skipping.")
            continue

        print(f"=== N = {n} ===")

        # Train backtest
        train_trades = run_backtest(train_df, signal_col, cfg)
        train_metrics = compute_metrics(train_trades)

        print(f"TRAIN: {train_metrics['num_trades']} trades, "
              f"Net P&L: ${train_metrics['total_net_pnl']:.2f}, "
              f"Win Rate: {train_metrics['win_rate']:.1f}%, "
              f"PF: {train_metrics['profit_factor']:.2f}, "
              f"Avg Ticks: {train_metrics['avg_trade_ticks']:.2f}")

        # Test backtest
        test_trades = run_backtest(test_df, signal_col, cfg)
        test_metrics = compute_metrics(test_trades)

        print(f"TEST:  {test_metrics['num_trades']} trades, "
              f"Net P&L: ${test_metrics['total_net_pnl']:.2f}, "
              f"Win Rate: {test_metrics['win_rate']:.1f}%, "
              f"PF: {test_metrics['profit_factor']:.2f}, "
              f"Avg Ticks: {test_metrics['avg_trade_ticks']:.2f}")

        # Check Go/No-Go gates for test
        gates_passed = (
            test_metrics["total_net_pnl"] > 0 and
            test_metrics["profit_factor"] >= 1.1 and
            test_metrics["avg_trade_ticks"] >= 1.0 and
            test_metrics["num_trades"] >= 30
        )
        print(f"GO/NO-GO: {'GO' if gates_passed else 'NO-GO'}")
        print()

        # Collect trade logs
        for t in test_trades:
            all_results.append({
                "n_value": n,
                "period": "test",
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "direction": "long" if t.direction == 1 else "short",
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "gross_pnl": t.gross_pnl,
                "net_pnl": t.net_pnl,
                "ticks": t.ticks,
            })

    # Save trade log
    if all_results:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(output_path, index=False)
        print(f"Trade log saved: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
