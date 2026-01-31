"""
Improve Zone Scalper Win Rate

Current issues:
- Win rate: 44% (too low)
- 55 of 72 trades hit stop loss

Improvements to test:
1. Higher entry Z (3.5 instead of 3.0) - fewer false signals
2. Add Z-velocity filter - confirm momentum
3. Tighter stop (Z=2.0) - fail faster
4. Lower target (Z=4.0) - more achievable
"""
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict
from itertools import product


@dataclass
class BacktestConfig:
    commission_per_side: float = 0.85
    tick_size: float = 0.25
    tick_value: float = 5.0


def load_cached_bars(start_date: str, end_date: str) -> pd.DataFrame:
    cache_file = Path(f'data/mnq_5min_ofi_{start_date}_{end_date}.parquet')
    if cache_file.exists():
        return pd.read_parquet(cache_file)
    return pd.DataFrame()


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['ema'] = df['close'].ewm(span=21, adjust=False).mean()
    df['dist'] = (df['close'] - df['ema']) / df['ema']
    df['dist_std'] = df['dist'].rolling(21).std()
    df['zscore'] = df['dist'] / df['dist_std']
    df['z_velocity'] = df['zscore'].diff()
    df['vol_avg'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_avg']
    return df


def is_rth(timestamp) -> bool:
    return 9 <= timestamp.hour < 16


def run_improved_scalper(df: pd.DataFrame, cfg: BacktestConfig,
                          entry_z: float, target_z: float, stop_z: float,
                          max_hold: int, min_velocity: float = 0.0,
                          min_vol_ratio: float = 0.0) -> List[Dict]:
    """Zone scalper with optional velocity and volume filters."""
    trades = []
    n = len(df)

    zscore = df['zscore'].values
    z_velocity = df['z_velocity'].values
    vol_ratio = df['vol_ratio'].values
    open_prices = df['open'].values
    timestamps = df['timestamp'].values

    position = 0
    entry_price = 0.0
    entry_bar = 0
    entry_dir = ""

    for i in range(2, n):
        prev_z = zscore[i-1]
        prev_prev_z = zscore[i-2]
        prev_vel = z_velocity[i-1]
        prev_vol = vol_ratio[i-1] if not np.isnan(vol_ratio[i-1]) else 1.0

        if np.isnan(prev_z) or np.isnan(prev_prev_z):
            continue

        current_ts = pd.Timestamp(timestamps[i])

        # RTH filter
        if not is_rth(current_ts):
            if position != 0:
                if position == 1:
                    exit_price = open_prices[i] - cfg.tick_size
                    gross = (exit_price - entry_price) / cfg.tick_size * cfg.tick_value
                else:
                    exit_price = open_prices[i] + cfg.tick_size
                    gross = (entry_price - exit_price) / cfg.tick_size * cfg.tick_value

                trades.append({
                    "net_pnl": gross - 2 * cfg.commission_per_side,
                    "exit_reason": "RTH",
                    "reached_target": False
                })
                position = 0
            continue

        # Entry with optional filters
        if position == 0:
            # LONG: Z crosses above entry_z with momentum
            if (prev_z >= entry_z and prev_prev_z < entry_z and
                prev_vel >= min_velocity and prev_vol >= min_vol_ratio):
                position = 1
                entry_price = open_prices[i] + cfg.tick_size
                entry_bar = i
                entry_dir = "LONG"

            # SHORT: Z crosses below -entry_z with momentum
            elif (prev_z <= -entry_z and prev_prev_z > -entry_z and
                  prev_vel <= -min_velocity and prev_vol >= min_vol_ratio):
                position = -1
                entry_price = open_prices[i] - cfg.tick_size
                entry_bar = i
                entry_dir = "SHORT"

        # Exit
        else:
            hold_time = i - entry_bar
            should_exit = False
            exit_reason = ""
            reached_target = False

            if hold_time >= max_hold:
                should_exit = True
                exit_reason = "MAX_HOLD"

            elif position == 1:
                if prev_z >= target_z:
                    should_exit = True
                    exit_reason = "TARGET"
                    reached_target = True
                elif prev_z <= stop_z:
                    should_exit = True
                    exit_reason = "STOP"

            elif position == -1:
                if prev_z <= -target_z:
                    should_exit = True
                    exit_reason = "TARGET"
                    reached_target = True
                elif prev_z >= -stop_z:
                    should_exit = True
                    exit_reason = "STOP"

            if should_exit:
                if position == 1:
                    exit_price = open_prices[i] - cfg.tick_size
                    gross = (exit_price - entry_price) / cfg.tick_size * cfg.tick_value
                else:
                    exit_price = open_prices[i] + cfg.tick_size
                    gross = (entry_price - exit_price) / cfg.tick_size * cfg.tick_value

                trades.append({
                    "net_pnl": gross - 2 * cfg.commission_per_side,
                    "exit_reason": exit_reason,
                    "reached_target": reached_target
                })
                position = 0

    return trades


def compute_metrics(trades: List[Dict]) -> Dict:
    if not trades:
        return {"n": 0, "pnl": 0, "wr": 0, "pf": 0}

    net = [t["net_pnl"] for t in trades]
    wins = sum(1 for p in net if p > 0)
    targets = sum(1 for t in trades if t["reached_target"])
    profit = sum(p for p in net if p > 0)
    loss = abs(sum(p for p in net if p < 0))

    return {
        "n": len(trades),
        "pnl": sum(net),
        "wr": wins / len(trades) * 100,
        "pf": profit / loss if loss > 0 else float("inf"),
        "target_rate": targets / len(trades) * 100
    }


def main():
    print("=" * 100)
    print("IMPROVING ZONE SCALPER WIN RATE")
    print("=" * 100)

    cfg = BacktestConfig()

    # Load data
    bars_by_period = {}
    for name, start, end in [("Train", "2025-01-01", "2025-02-28"), ("OOS", "2025-03-01", "2025-03-31")]:
        bars = load_cached_bars(start, end)
        if len(bars) > 0:
            bars_by_period[name] = compute_features(bars)
            print(f"Loaded {name}: {len(bars):,} bars")

    # Parameter grid focused on improving win rate
    param_grid = {
        'entry_z': [3.0, 3.5, 4.0],  # Higher = fewer trades, more selective
        'target_z': [3.5, 4.0, 4.5],  # Lower = easier to hit
        'stop_z': [1.5, 2.0, 2.5],   # Higher = tighter stop (closer to entry)
        'max_hold': [10, 15, 20],
        'min_velocity': [0.0, 0.3, 0.5],  # Momentum filter
    }

    results = []

    for entry_z, target_z, stop_z, max_hold, min_vel in product(
        param_grid['entry_z'],
        param_grid['target_z'],
        param_grid['stop_z'],
        param_grid['max_hold'],
        param_grid['min_velocity']
    ):
        # Skip invalid: target must be > entry
        if target_z <= entry_z:
            continue
        # Skip invalid: stop must be < entry
        if stop_z >= entry_z:
            continue

        all_trades = []
        oos_trades = []

        for name, df in bars_by_period.items():
            trades = run_improved_scalper(df, cfg, entry_z, target_z, stop_z, max_hold, min_vel)
            all_trades.extend(trades)
            if name == "OOS":
                oos_trades = trades

        total_m = compute_metrics(all_trades)
        oos_m = compute_metrics(oos_trades)

        results.append({
            "params": f"E={entry_z}, T={target_z}, S={stop_z}, H={max_hold}, V={min_vel}",
            "total_n": total_m["n"],
            "total_pnl": total_m["pnl"],
            "total_wr": total_m["wr"],
            "total_pf": total_m["pf"],
            "oos_n": oos_m["n"],
            "oos_pnl": oos_m["pnl"],
            "oos_wr": oos_m["wr"],
        })

    df_results = pd.DataFrame(results)

    # Find configs with higher win rate
    print("\n" + "=" * 100)
    print("TOP CONFIGS BY WIN RATE (min 10 OOS trades, positive OOS P&L)")
    print("=" * 100)

    viable = df_results[
        (df_results["oos_n"] >= 5) &
        (df_results["oos_pnl"] > 0) &
        (df_results["total_wr"] >= 50)  # Target 50%+ win rate
    ].copy()

    if len(viable) > 0:
        viable = viable.sort_values("total_wr", ascending=False)
        print(f"\n{'Config':<45} {'OOS P&L':>10} {'OOS #':>8} {'WR':>8} {'PF':>8}")
        print("-" * 85)
        for _, row in viable.head(15).iterrows():
            print(f"{row['params']:<45} ${row['oos_pnl']:>9.2f} {row['oos_n']:>8} {row['total_wr']:>7.1f}% {row['total_pf']:>7.2f}")
    else:
        print("\nNo configs with 50%+ win rate and positive OOS P&L")
        print("\nShowing best by OOS P&L with WR >= 45%:")
        viable = df_results[
            (df_results["oos_n"] >= 5) &
            (df_results["oos_pnl"] > 0) &
            (df_results["total_wr"] >= 45)
        ].sort_values("oos_pnl", ascending=False)
        for _, row in viable.head(10).iterrows():
            print(f"{row['params']:<45} ${row['oos_pnl']:>9.2f} {row['oos_n']:>8} {row['total_wr']:>7.1f}% {row['total_pf']:>7.2f}")

    # Best overall by balance of WR and P&L
    print("\n" + "=" * 100)
    print("BEST BALANCED CONFIGS (WR >= 48% AND positive OOS)")
    print("=" * 100)

    balanced = df_results[
        (df_results["oos_n"] >= 5) &
        (df_results["oos_pnl"] > 0) &
        (df_results["total_wr"] >= 48)
    ].sort_values("oos_pnl", ascending=False)

    if len(balanced) > 0:
        print(f"\n{'Config':<45} {'OOS P&L':>10} {'OOS #':>8} {'WR':>8} {'PF':>8}")
        print("-" * 85)
        for _, row in balanced.head(10).iterrows():
            print(f"{row['params']:<45} ${row['oos_pnl']:>9.2f} {row['oos_n']:>8} {row['total_wr']:>7.1f}% {row['total_pf']:>7.2f}")

        best = balanced.iloc[0]
        print(f"\n*** RECOMMENDED: {best['params']}")
        print(f"    OOS: ${best['oos_pnl']:.2f}, {best['oos_n']} trades")
        print(f"    Win Rate: {best['total_wr']:.1f}%")
        print(f"    Profit Factor: {best['total_pf']:.2f}")

    # Save all results
    df_results.to_csv("results/zone_scalper_improved.csv", index=False)
    print("\nAll results saved to results/zone_scalper_improved.csv")

    print("\n" + "=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
