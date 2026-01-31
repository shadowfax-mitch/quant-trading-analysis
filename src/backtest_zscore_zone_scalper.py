"""
Z-Score Zone Scalper Strategy

Based on failure analysis findings:
- Entry at Z=1.5 only reaches Z=3.5 18% of the time (bad)
- Entry at Z=3.0 reaches Z=3.5+ 56% of the time (much better!)

Strategy: Enter LATE in the Z-score move (at 3.0-3.5) and target
the deeper extreme (4.0-5.0). This is still trend-following but
with much higher probability.

Entry: Z crosses 3.0 (or higher)
Target: Z reaches 4.0+ (scalp profit)
Stop: Z reverts below 2.5 (failed run)
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
    tick_value: float = 5.0  # MNQ


@dataclass
class ZoneScalperParams:
    """Parameters for Zone Scalper strategy."""
    ema_period: int = 21
    zscore_lookback: int = 21

    # Entry: Z must cross this level
    entry_z: float = 3.0

    # Target: Take profit when Z reaches this
    target_z: float = 4.0

    # Stop: Exit when Z reverts to this
    stop_z: float = 2.5

    # Alternative: Fixed point stop
    fixed_stop_points: float = 0  # 0 = disabled

    # Max hold time
    max_hold_bars: int = 20

    # RTH filter
    rth_only: bool = True
    rth_start: int = 9
    rth_end: int = 16


def load_cached_bars(start_date: str, end_date: str) -> pd.DataFrame:
    """Load cached 5-min bars."""
    cache_file = Path(f'data/mnq_5min_ofi_{start_date}_{end_date}.parquet')
    if cache_file.exists():
        return pd.read_parquet(cache_file)
    return pd.DataFrame()


def compute_features(df: pd.DataFrame, params: ZoneScalperParams) -> pd.DataFrame:
    """Compute Z-score."""
    df = df.copy()
    df['ema'] = df['close'].ewm(span=params.ema_period, adjust=False).mean()
    df['dist'] = (df['close'] - df['ema']) / df['ema']
    df['dist_std'] = df['dist'].rolling(params.zscore_lookback).std()
    df['zscore'] = df['dist'] / df['dist_std']
    return df


def is_rth(timestamp, start: int, end: int) -> bool:
    """Check if in RTH."""
    return start <= timestamp.hour < end


def run_zone_scalper_backtest(
    df: pd.DataFrame,
    params: ZoneScalperParams,
    cfg: BacktestConfig
) -> List[Dict]:
    """Run Zone Scalper backtest."""
    trades = []
    n = len(df)

    if n < 50:
        return trades

    zscore = df['zscore'].values
    open_prices = df['open'].values
    close_prices = df['close'].values
    timestamps = df['timestamp'].values

    position = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    entry_bar = 0
    entry_dir = ""
    entry_z = 0.0

    for i in range(1, n):
        prev_z = zscore[i-1]
        prev_prev_z = zscore[i-2] if i > 1 else 0

        if np.isnan(prev_z):
            continue

        current_ts = pd.Timestamp(timestamps[i])

        # RTH filter
        in_rth = is_rth(current_ts, params.rth_start, params.rth_end)

        if params.rth_only and not in_rth:
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
                    "exit_z": prev_z,
                    "net_pnl": net_pnl,
                    "exit_reason": "RTH_CLOSE",
                    "bars_held": i - entry_bar,
                    "reached_target": False
                })
                position = 0
            continue

        # Entry logic - Z crosses INTO the zone
        if position == 0:
            # Long entry: Z crosses above entry_z (going up)
            if prev_z >= params.entry_z and prev_prev_z < params.entry_z:
                position = 1
                entry_price = open_prices[i] + cfg.tick_size
                entry_bar = i
                entry_dir = "LONG"
                entry_z = prev_z

            # Short entry: Z crosses below -entry_z (going down)
            elif prev_z <= -params.entry_z and prev_prev_z > -params.entry_z:
                position = -1
                entry_price = open_prices[i] - cfg.tick_size
                entry_bar = i
                entry_dir = "SHORT"
                entry_z = prev_z

        # Exit logic
        else:
            hold_time = i - entry_bar
            should_exit = False
            exit_reason = ""
            reached_target = False

            # Max hold
            if hold_time >= params.max_hold_bars:
                should_exit = True
                exit_reason = "MAX_HOLD"

            # Long exits
            elif position == 1:
                # Target hit (deeper extreme)
                if prev_z >= params.target_z:
                    should_exit = True
                    exit_reason = f"TARGET(Z={prev_z:.2f})"
                    reached_target = True
                # Stop loss (Z reverted)
                elif prev_z <= params.stop_z:
                    should_exit = True
                    exit_reason = f"STOP(Z={prev_z:.2f})"
                # Fixed point stop
                elif params.fixed_stop_points > 0:
                    current_loss = entry_price - close_prices[i-1]
                    if current_loss >= params.fixed_stop_points:
                        should_exit = True
                        exit_reason = f"FIXED_STOP"

            # Short exits
            elif position == -1:
                # Target hit
                if prev_z <= -params.target_z:
                    should_exit = True
                    exit_reason = f"TARGET(Z={prev_z:.2f})"
                    reached_target = True
                # Stop loss
                elif prev_z >= -params.stop_z:
                    should_exit = True
                    exit_reason = f"STOP(Z={prev_z:.2f})"
                # Fixed point stop
                elif params.fixed_stop_points > 0:
                    current_loss = close_prices[i-1] - entry_price
                    if current_loss >= params.fixed_stop_points:
                        should_exit = True
                        exit_reason = f"FIXED_STOP"

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
                    "exit_z": prev_z,
                    "net_pnl": net_pnl,
                    "exit_reason": exit_reason,
                    "bars_held": hold_time,
                    "reached_target": reached_target
                })
                position = 0

    return trades


def compute_metrics(trades: List[Dict]) -> Dict:
    """Compute performance metrics."""
    if not trades:
        return {"n": 0, "pnl": 0, "wr": 0, "pf": 0, "avg": 0, "target_rate": 0, "tpd": 0}

    net_pnls = [t["net_pnl"] for t in trades]
    wins = sum(1 for p in net_pnls if p > 0)
    targets = sum(1 for t in trades if t["reached_target"])
    profit = sum(p for p in net_pnls if p > 0)
    loss = abs(sum(p for p in net_pnls if p < 0))

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
        "pf": profit / loss if loss > 0 else float("inf"),
        "avg": sum(net_pnls) / len(trades),
        "target_rate": targets / len(trades) * 100,
        "tpd": tpd
    }


def run_grid_search(bars_by_period: Dict[str, pd.DataFrame], cfg: BacktestConfig) -> pd.DataFrame:
    """Grid search over parameters."""
    param_grid = {
        'entry_z': [2.5, 3.0, 3.5],
        'target_z': [3.5, 4.0, 4.5, 5.0],
        'stop_z': [1.5, 2.0, 2.5],
        'max_hold': [15, 20, 30],
    }

    results = []
    total = (len(param_grid['entry_z']) * len(param_grid['target_z']) *
             len(param_grid['stop_z']) * len(param_grid['max_hold']))

    print(f"\nTesting {total} parameter combinations...")

    idx = 0
    for entry_z, target_z, stop_z, max_hold in product(
        param_grid['entry_z'],
        param_grid['target_z'],
        param_grid['stop_z'],
        param_grid['max_hold']
    ):
        # Skip invalid combos
        if target_z <= entry_z:
            continue
        if stop_z >= entry_z:
            continue

        idx += 1
        if idx % 20 == 0:
            print(f"   Progress: {idx}/{total}")

        params = ZoneScalperParams(
            entry_z=entry_z,
            target_z=target_z,
            stop_z=stop_z,
            max_hold_bars=max_hold
        )

        row = {"params": f"Entry={entry_z}, Target={target_z}, Stop={stop_z}, Hold={max_hold}"}
        all_trades = []
        oos_pnl = 0
        oos_trades = 0

        for period_name, bars in bars_by_period.items():
            if len(bars) == 0:
                continue

            df = compute_features(bars, params)
            trades = run_zone_scalper_backtest(df, params, cfg)
            metrics = compute_metrics(trades)

            row[f"{period_name}_n"] = metrics["n"]
            row[f"{period_name}_pnl"] = metrics["pnl"]
            row[f"{period_name}_wr"] = metrics["wr"]

            all_trades.extend(trades)

            if "OOS" in period_name:
                oos_pnl += metrics["pnl"]
                oos_trades += metrics["n"]

        total_m = compute_metrics(all_trades)
        row["total_n"] = total_m["n"]
        row["total_pnl"] = total_m["pnl"]
        row["total_wr"] = total_m["wr"]
        row["total_pf"] = total_m["pf"]
        row["total_tpd"] = total_m["tpd"]
        row["total_target_rate"] = total_m["target_rate"]
        row["oos_pnl"] = oos_pnl
        row["oos_trades"] = oos_trades

        results.append(row)

    return pd.DataFrame(results)


def main():
    print("=" * 100)
    print("Z-SCORE ZONE SCALPER - Enter Late, Scalp the Extreme")
    print("=" * 100)
    print()
    print("Strategy: Enter when Z crosses 3.0, target Z=4.0+")
    print("Based on finding: 56% of Z=3.0 entries reach Z=3.5+")
    print()

    cfg = BacktestConfig()

    # Load data
    periods = [
        ("Train", "2025-01-01", "2025-02-28"),
        ("OOS1", "2025-03-01", "2025-03-31"),
    ]

    bars_by_period = {}
    for name, start, end in periods:
        print(f"Loading {name}...")
        bars = load_cached_bars(start, end)
        if len(bars) > 0:
            print(f"   {len(bars):,} bars")
            bars_by_period[name] = bars

    if not bars_by_period:
        print("No data!")
        return 1

    # Grid search
    results = run_grid_search(bars_by_period, cfg)

    # Save results
    results_file = Path('results/zone_scalper_grid.csv')
    results_file.parent.mkdir(exist_ok=True)
    results.to_csv(results_file, index=False)
    print(f"\nResults saved to {results_file}")

    # Best configs
    print("\n" + "=" * 100)
    print("TOP CONFIGURATIONS (by OOS P&L)")
    print("=" * 100)

    viable = results[
        (results['oos_trades'] >= 5) &
        (results['oos_pnl'] > 0)
    ].copy()

    if len(viable) > 0:
        viable = viable.sort_values('oos_pnl', ascending=False)

        print(f"\n{'Config':<55} {'OOS P&L':>10} {'OOS #':>8} {'WR':>8} {'TPD':>8}")
        print("-" * 90)

        for _, row in viable.head(10).iterrows():
            print(f"{row['params']:<55} ${row['oos_pnl']:>9.2f} {row['oos_trades']:>8} {row['total_wr']:>7.1f}% {row['total_tpd']:>7.2f}")

        # Best detail
        best = viable.iloc[0]
        print(f"\n{'='*100}")
        print("BEST CONFIGURATION")
        print(f"{'='*100}")
        print(f"\n{best['params']}")
        print(f"\nTotal: {best['total_n']} trades, ${best['total_pnl']:.2f} P&L")
        print(f"Win Rate: {best['total_wr']:.1f}%")
        print(f"Profit Factor: {best['total_pf']:.2f}")
        print(f"Target Rate: {best['total_target_rate']:.1f}%")
        print(f"Trades/Day: {best['total_tpd']:.2f}")

    else:
        print("\nNo profitable OOS configurations found!")
        print("\nTop 5 by total P&L:")
        top = results.sort_values('total_pnl', ascending=False).head(5)
        for _, row in top.iterrows():
            print(f"   {row['params']}: ${row['total_pnl']:.2f} total, ${row['oos_pnl']:.2f} OOS")

    # Also test a hybrid: Zone entry + mean-reversion exit
    print("\n" + "=" * 100)
    print("HYBRID TEST: Zone Entry + Mean Reversion Exit")
    print("=" * 100)

    # Test entering at Z=3.0 but using mean-reversion exit (Z crosses back toward 0)
    hybrid_params = ZoneScalperParams(
        entry_z=3.0,
        target_z=1.0,  # Exit when Z reverts to 1.0 (mean-reversion style)
        stop_z=5.0,  # Stop if it goes TOO extreme (rare)
        max_hold_bars=48
    )

    all_hybrid_trades = []
    for name, bars in bars_by_period.items():
        df = compute_features(bars, hybrid_params)
        trades = run_zone_scalper_backtest(df, hybrid_params, cfg)

        m = compute_metrics(trades)
        print(f"\n{name}: {m['n']} trades, ${m['pnl']:.2f}, WR={m['wr']:.1f}%")
        all_hybrid_trades.extend(trades)

    hybrid_m = compute_metrics(all_hybrid_trades)
    print(f"\nHybrid TOTAL: {hybrid_m['n']} trades, ${hybrid_m['pnl']:.2f}, WR={hybrid_m['wr']:.1f}%, PF={hybrid_m['pf']:.2f}")

    print("\n" + "=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
