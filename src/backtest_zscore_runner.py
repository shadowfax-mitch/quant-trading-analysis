"""
Z-Score Runner Strategy Backtest

TREND-FOLLOWING strategy that enters early when detecting a potential
extreme Z-score move, and rides the momentum.

Based on precursor analysis findings:
- Extreme moves (|Z| > 3.5) have high Z-velocity at start
- Average duration: ~14 bars to reach peak
- Z-velocity at start: mean ~1.27

Entry Logic:
1. Z-Score crosses entry threshold (1.0-2.0) in either direction
2. Z-Velocity is high (confirming momentum)
3. Optionally: Volume above average

Exit Logic:
- Target: Z reaches 3.5+ (extreme)
- Profit taking: Z starts decelerating
- Stop: Z reverts to 0 (failed runner)
- Time stop: Max bars if no progress
"""
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple
from itertools import product
import gc


@dataclass
class BacktestConfig:
    commission_per_side: float = 0.85
    tick_size: float = 0.25
    tick_value: float = 5.0  # MNQ


@dataclass
class RunnerParams:
    """Parameters for Z-Score Runner strategy."""
    # EMA/Z-Score params
    ema_period: int = 21
    zscore_lookback: int = 21

    # Entry conditions
    entry_z_threshold: float = 1.5  # Enter when |Z| crosses this level
    min_z_velocity: float = 0.5  # Minimum Z velocity to enter
    min_vol_ratio: float = 1.0  # Minimum volume ratio (1.0 = no filter)

    # Exit conditions
    target_z: float = 3.5  # Take profit when Z reaches this
    stop_z: float = 0.0  # Stop loss when Z reverts to this
    max_hold_bars: int = 30  # Max bars to hold

    # Trading hours
    rth_only: bool = True
    rth_start: int = 9
    rth_end: int = 16


def load_cached_bars(start_date: str, end_date: str) -> pd.DataFrame:
    """Load cached 5-min bars."""
    cache_file = Path(f'data/mnq_5min_ofi_{start_date}_{end_date}.parquet')
    if cache_file.exists():
        return pd.read_parquet(cache_file)
    return pd.DataFrame()


def compute_features(df: pd.DataFrame, params: RunnerParams) -> pd.DataFrame:
    """Compute Z-score and velocity features."""
    df = df.copy()

    # Z-Score
    df['ema'] = df['close'].ewm(span=params.ema_period, adjust=False).mean()
    df['dist'] = (df['close'] - df['ema']) / df['ema']
    df['dist_std'] = df['dist'].rolling(params.zscore_lookback).std()
    df['zscore'] = df['dist'] / df['dist_std']

    # Z-Score velocity
    df['z_velocity'] = df['zscore'].diff()

    # Volume ratio
    df['vol_avg'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_avg']

    return df


def is_rth(timestamp, start: int, end: int) -> bool:
    """Check if timestamp is during RTH."""
    hour = timestamp.hour
    return start <= hour < end


def run_runner_backtest(
    df: pd.DataFrame,
    params: RunnerParams,
    cfg: BacktestConfig
) -> List[Dict]:
    """Run Z-Score Runner backtest."""
    trades = []
    n = len(df)

    if n < 50:
        return trades

    zscore = df['zscore'].values
    z_velocity = df['z_velocity'].values
    vol_ratio = df['vol_ratio'].values
    open_prices = df['open'].values
    timestamps = df['timestamp'].values

    position = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    entry_bar = 0
    entry_dir = ""
    entry_z = 0.0

    for i in range(1, n):
        z = zscore[i-1]  # Use previous bar's signal
        zv = z_velocity[i-1]
        vr = vol_ratio[i-1]

        if np.isnan(z) or np.isnan(zv):
            continue

        current_ts = pd.Timestamp(timestamps[i])

        # RTH filter
        in_rth = is_rth(current_ts, params.rth_start, params.rth_end)

        if params.rth_only and not in_rth:
            # Close position at end of RTH
            if position != 0:
                if position == 1:
                    exit_price = open_prices[i] - cfg.tick_size
                    gross_pnl = (exit_price - entry_price) / cfg.tick_size * cfg.tick_value
                else:
                    exit_price = open_prices[i] + cfg.tick_size
                    gross_pnl = (entry_price - exit_price) / cfg.tick_size * cfg.tick_value

                net_pnl = gross_pnl - 2 * cfg.commission_per_side
                trades.append({
                    "entry_time": pd.Timestamp(timestamps[entry_bar]),
                    "exit_time": current_ts,
                    "direction": entry_dir,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "entry_z": entry_z,
                    "exit_z": z,
                    "net_pnl": net_pnl,
                    "ticks": gross_pnl / cfg.tick_value,
                    "exit_reason": "RTH_CLOSE",
                    "bars_held": i - entry_bar,
                    "reached_target": False
                })
                position = 0
            continue

        # Entry logic
        if position == 0:
            # Check for potential UP runner
            if (z > params.entry_z_threshold and
                zv > params.min_z_velocity and
                vr >= params.min_vol_ratio):
                position = 1
                entry_price = open_prices[i] + cfg.tick_size
                entry_bar = i
                entry_dir = "LONG"
                entry_z = z

            # Check for potential DOWN runner
            elif (z < -params.entry_z_threshold and
                  zv < -params.min_z_velocity and
                  vr >= params.min_vol_ratio):
                position = -1
                entry_price = open_prices[i] - cfg.tick_size
                entry_bar = i
                entry_dir = "SHORT"
                entry_z = z

        # Exit logic
        else:
            hold_time = i - entry_bar
            should_exit = False
            exit_reason = ""
            reached_target = False

            # Max hold time
            if hold_time >= params.max_hold_bars:
                should_exit = True
                exit_reason = "MAX_HOLD"

            # Long exits
            elif position == 1:
                # Target reached (Z hit extreme)
                if z >= params.target_z:
                    should_exit = True
                    exit_reason = f"TARGET(Z={z:.2f})"
                    reached_target = True
                # Stop loss (Z reverted to zero)
                elif z <= params.stop_z:
                    should_exit = True
                    exit_reason = f"STOP(Z={z:.2f})"
                # Momentum fade (velocity reversed)
                elif zv < -0.3 and z < entry_z:
                    should_exit = True
                    exit_reason = f"MOMENTUM_FADE"

            # Short exits
            elif position == -1:
                # Target reached
                if z <= -params.target_z:
                    should_exit = True
                    exit_reason = f"TARGET(Z={z:.2f})"
                    reached_target = True
                # Stop loss
                elif z >= -params.stop_z:
                    should_exit = True
                    exit_reason = f"STOP(Z={z:.2f})"
                # Momentum fade
                elif zv > 0.3 and z > entry_z:
                    should_exit = True
                    exit_reason = f"MOMENTUM_FADE"

            if should_exit:
                if position == 1:
                    exit_price = open_prices[i] - cfg.tick_size
                    gross_pnl = (exit_price - entry_price) / cfg.tick_size * cfg.tick_value
                else:
                    exit_price = open_prices[i] + cfg.tick_size
                    gross_pnl = (entry_price - exit_price) / cfg.tick_size * cfg.tick_value

                net_pnl = gross_pnl - 2 * cfg.commission_per_side
                trades.append({
                    "entry_time": pd.Timestamp(timestamps[entry_bar]),
                    "exit_time": current_ts,
                    "direction": entry_dir,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "entry_z": entry_z,
                    "exit_z": z,
                    "net_pnl": net_pnl,
                    "ticks": gross_pnl / cfg.tick_value,
                    "exit_reason": exit_reason,
                    "bars_held": hold_time,
                    "reached_target": reached_target
                })
                position = 0

    return trades


def compute_metrics(trades: List[Dict]) -> Dict:
    """Compute performance metrics."""
    if not trades:
        return {"n": 0, "pnl": 0, "wr": 0, "pf": 0, "avg_trade": 0, "target_rate": 0, "tpd": 0}

    net_pnls = [t["net_pnl"] for t in trades]
    wins = sum(1 for p in net_pnls if p > 0)
    targets_hit = sum(1 for t in trades if t["reached_target"])
    net_profit = sum(p for p in net_pnls if p > 0)
    net_loss = abs(sum(p for p in net_pnls if p < 0))

    # Trades per day
    if len(trades) > 1:
        first = trades[0]["entry_time"]
        last = trades[-1]["entry_time"]
        days = max((last - first).days, 1)
        tpd = len(trades) / days
    else:
        tpd = 0

    return {
        "n": len(trades),
        "pnl": sum(net_pnls),
        "wr": wins / len(trades) * 100,
        "pf": net_profit / net_loss if net_loss > 0 else float("inf"),
        "avg_trade": sum(net_pnls) / len(trades),
        "target_rate": targets_hit / len(trades) * 100,
        "tpd": tpd
    }


def run_grid_search(bars_by_period: Dict[str, pd.DataFrame], cfg: BacktestConfig) -> pd.DataFrame:
    """Run parameter grid search."""
    param_grid = {
        'entry_z': [1.0, 1.5, 2.0, 2.5],
        'min_velocity': [0.3, 0.5, 0.8, 1.0],
        'target_z': [3.0, 3.5, 4.0],
        'max_hold': [20, 30, 40],
    }

    results = []
    total = (len(param_grid['entry_z']) * len(param_grid['min_velocity']) *
             len(param_grid['target_z']) * len(param_grid['max_hold']))

    print(f"\nTesting {total} parameter combinations...")

    idx = 0
    for entry_z, min_vel, target_z, max_hold in product(
        param_grid['entry_z'],
        param_grid['min_velocity'],
        param_grid['target_z'],
        param_grid['max_hold']
    ):
        idx += 1
        if idx % 30 == 0:
            print(f"   Progress: {idx}/{total}")

        params = RunnerParams(
            entry_z_threshold=entry_z,
            min_z_velocity=min_vel,
            target_z=target_z,
            max_hold_bars=max_hold
        )

        row = {"params": f"Entry={entry_z}, Vel={min_vel}, Target={target_z}, Hold={max_hold}"}
        all_trades = []
        oos_pnl = 0
        oos_trades = 0

        for period_name, bars in bars_by_period.items():
            if len(bars) == 0:
                continue

            df = compute_features(bars, params)
            trades = run_runner_backtest(df, params, cfg)
            metrics = compute_metrics(trades)

            row[f"{period_name}_n"] = metrics["n"]
            row[f"{period_name}_pnl"] = metrics["pnl"]
            row[f"{period_name}_wr"] = metrics["wr"]
            row[f"{period_name}_target_rate"] = metrics["target_rate"]

            all_trades.extend(trades)

            if "OOS" in period_name:
                oos_pnl += metrics["pnl"]
                oos_trades += metrics["n"]

        total_metrics = compute_metrics(all_trades)
        row["total_n"] = total_metrics["n"]
        row["total_pnl"] = total_metrics["pnl"]
        row["total_wr"] = total_metrics["wr"]
        row["total_pf"] = total_metrics["pf"]
        row["total_tpd"] = total_metrics["tpd"]
        row["total_target_rate"] = total_metrics["target_rate"]
        row["oos_pnl"] = oos_pnl
        row["oos_trades"] = oos_trades

        results.append(row)

    return pd.DataFrame(results)


def main():
    print("=" * 100)
    print("Z-SCORE RUNNER STRATEGY BACKTEST")
    print("=" * 100)
    print()
    print("Goal: Trend-follow into extreme Z-scores")
    print("      Enter early when momentum detected, exit at Z=3.5+ target")
    print()

    cfg = BacktestConfig()

    # Load cached data
    periods = [
        ("Train", "2025-01-01", "2025-02-28"),
        ("OOS1", "2025-03-01", "2025-03-31"),
    ]

    bars_by_period = {}
    for name, start, end in periods:
        print(f"Loading {name}: {start} to {end}...")
        bars = load_cached_bars(start, end)
        if len(bars) > 0:
            print(f"   {len(bars):,} bars")
            bars_by_period[name] = bars
        else:
            print(f"   No data found")

    if not bars_by_period:
        print("No data available!")
        return 1

    # Run grid search
    print("\nRunning parameter grid search...")
    results = run_grid_search(bars_by_period, cfg)

    # Save results
    results_file = Path('results/zscore_runner_grid.csv')
    results_file.parent.mkdir(exist_ok=True)
    results.to_csv(results_file, index=False)
    print(f"\nResults saved to {results_file}")

    # Find best configs
    print("\n" + "=" * 100)
    print("TOP 10 CONFIGURATIONS (by OOS P&L)")
    print("=" * 100)

    # Filter for viable configs
    viable = results[
        (results['oos_trades'] >= 10) &
        (results['oos_pnl'] > 0)
    ].copy()

    if len(viable) > 0:
        viable = viable.sort_values('oos_pnl', ascending=False)

        print(f"\n{'Config':<55} {'OOS P&L':>10} {'OOS #':>8} {'Target%':>10} {'TPD':>8}")
        print("-" * 95)

        for _, row in viable.head(10).iterrows():
            print(f"{row['params']:<55} ${row['oos_pnl']:>9.2f} {row['oos_trades']:>8} {row['total_target_rate']:>9.1f}% {row['total_tpd']:>7.2f}")

        # Detailed best config
        best = viable.iloc[0]
        print(f"\n{'='*100}")
        print("BEST CONFIGURATION")
        print(f"{'='*100}")
        print(f"\n{best['params']}")
        print(f"\nPeriod breakdown:")
        for name, _, _ in periods:
            n = best.get(f'{name}_n', 0)
            pnl = best.get(f'{name}_pnl', 0)
            wr = best.get(f'{name}_wr', 0)
            tr = best.get(f'{name}_target_rate', 0)
            print(f"   {name:<10} {n:>5} trades, ${pnl:>8.2f}, WR={wr:.1f}%, Target={tr:.1f}%")

    else:
        print("\nNo viable configurations found with positive OOS P&L!")
        print("\nTop 5 by total P&L:")
        top5 = results.sort_values('total_pnl', ascending=False).head(5)
        for _, row in top5.iterrows():
            print(f"   {row['params']}: ${row['total_pnl']:.2f} total, ${row['oos_pnl']:.2f} OOS, {row['oos_trades']} trades")

    # Compare to baseline
    print(f"\n{'='*100}")
    print("COMPARISON TO BASELINE (Mean Reversion)")
    print(f"{'='*100}")
    print("""
    Z=5.0 Mean Reversion (wait for extreme, then revert):
    - OOS P&L: +$1,178
    - Trades: 21
    - ~0.1 trades/day

    Z-Score Runner (catch the move early):
    - Higher trade frequency
    - Trend-following approach
    - Target: reach Z=3.5+ from Z=1.5 entry
    """)

    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
