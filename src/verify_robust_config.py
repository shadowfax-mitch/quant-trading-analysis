"""
Verify the Robust EMA Z-Score Configuration on MNQ Data

Robust Config Parameters:
- EMA Period: 21
- Z-Score Lookback: 21
- Entry Threshold: 5.0 (extreme only)
- Exit Threshold: 1.0
- Max Hold: 48 bars
- Trading Hours: RTH only (9 AM - 4 PM)
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BacktestConfig:
    commission_per_side: float = 0.85
    tick_size: float = 0.25
    tick_value: float = 0.50  # MNQ


def load_bars():
    """Load cached MNQ 5-min bars."""
    cache_file = Path('data/mnq_5min_bars_full.parquet')
    return pd.read_parquet(cache_file)


def is_rth(timestamp) -> bool:
    """Check if timestamp is during Regular Trading Hours (9 AM - 4 PM)."""
    hour = timestamp.hour
    return 9 <= hour < 16


def run_backtest_robust(df: pd.DataFrame, zscore: np.ndarray,
                        entry_thresh: float, exit_thresh: float,
                        cfg: BacktestConfig, max_bars: int,
                        rth_only: bool = True) -> list:
    """Mean reversion with RTH filter."""
    trades = []
    n = len(df)
    open_prices = df['open'].values
    timestamps = df['timestamp'].values
    position = 0
    entry_price = 0.0
    entry_bar = 0
    entry_dir = ""

    for i in range(1, n):
        if np.isnan(zscore[i-1]):
            continue

        prev_z = zscore[i - 1]
        current_ts = pd.Timestamp(timestamps[i])
        in_rth = is_rth(current_ts)

        # If RTH filter is on and we're outside RTH
        if rth_only and not in_rth:
            # Force close any open position at end of RTH
            if position != 0:
                if position == 1:
                    exit_price = open_prices[i] - cfg.tick_size
                    gross_pnl = (exit_price - entry_price) / cfg.tick_size * cfg.tick_value
                else:
                    exit_price = open_prices[i] + cfg.tick_size
                    gross_pnl = (entry_price - exit_price) / cfg.tick_size * cfg.tick_value

                net_pnl = gross_pnl - 2 * cfg.commission_per_side
                trades.append({
                    "net_pnl": net_pnl,
                    "ticks": gross_pnl / cfg.tick_value,
                    "direction": entry_dir,
                    "exit_reason": "RTH_CLOSE"
                })
                position = 0
            continue

        if position == 0:
            # Entry conditions - only during RTH
            if prev_z < -entry_thresh:
                position = 1
                entry_price = open_prices[i] + cfg.tick_size
                entry_bar = i
                entry_dir = "LONG"
            elif prev_z > entry_thresh:
                position = -1
                entry_price = open_prices[i] - cfg.tick_size
                entry_bar = i
                entry_dir = "SHORT"
        else:
            hold_time = i - entry_bar
            should_exit = hold_time >= max_bars

            if position == 1 and (prev_z > -exit_thresh or prev_z > 0):
                should_exit = True
            elif position == -1 and (prev_z < exit_thresh or prev_z < 0):
                should_exit = True

            if should_exit:
                if position == 1:
                    exit_price = open_prices[i] - cfg.tick_size
                    gross_pnl = (exit_price - entry_price) / cfg.tick_size * cfg.tick_value
                else:
                    exit_price = open_prices[i] + cfg.tick_size
                    gross_pnl = (entry_price - exit_price) / cfg.tick_size * cfg.tick_value

                net_pnl = gross_pnl - 2 * cfg.commission_per_side
                trades.append({
                    "net_pnl": net_pnl,
                    "ticks": gross_pnl / cfg.tick_value,
                    "direction": entry_dir,
                    "exit_reason": "SIGNAL" if hold_time < max_bars else "MAX_HOLD"
                })
                position = 0

    # Close any remaining position
    if position != 0:
        if position == 1:
            exit_price = open_prices[-1] - cfg.tick_size
            gross_pnl = (exit_price - entry_price) / cfg.tick_size * cfg.tick_value
        else:
            exit_price = open_prices[-1] + cfg.tick_size
            gross_pnl = (entry_price - exit_price) / cfg.tick_size * cfg.tick_value
        net_pnl = gross_pnl - 2 * cfg.commission_per_side
        trades.append({
            "net_pnl": net_pnl,
            "ticks": gross_pnl / cfg.tick_value,
            "direction": entry_dir,
            "exit_reason": "EOD"
        })

    return trades


def compute_metrics(trades: list) -> dict:
    if not trades:
        return {"n": 0, "pnl": 0, "wr": 0, "pf_net": 0, "avg_trade": 0}

    net_pnls = [t["net_pnl"] for t in trades]
    wins = sum(1 for p in net_pnls if p > 0)
    net_profit = sum(p for p in net_pnls if p > 0)
    net_loss = abs(sum(p for p in net_pnls if p < 0))

    return {
        "n": len(trades),
        "pnl": sum(net_pnls),
        "wr": wins / len(trades) * 100 if trades else 0,
        "pf_net": net_profit / net_loss if net_loss > 0 else float("inf"),
        "avg_trade": sum(net_pnls) / len(trades) if trades else 0,
    }


def main():
    print("=" * 100)
    print("VERIFYING ROBUST EMA Z-SCORE CONFIG ON MNQ")
    print("=" * 100)
    print()

    print("ROBUST CONFIG PARAMETERS:")
    print("  - EMA Period: 21")
    print("  - Z-Score Lookback: 21")
    print("  - Entry Threshold: 5.0")
    print("  - Exit Threshold: 1.0")
    print("  - Max Hold: 48 bars")
    print("  - Trading Hours: RTH only (9 AM - 4 PM)")
    print()

    cfg = BacktestConfig()

    # Load data
    print("1. Loading MNQ 5-minute bars...")
    all_bars = load_bars()
    print(f"   Total bars: {len(all_bars):,}")

    # Define periods
    periods = [
        ("Train (Jan-Feb 25)", "2025-01-01", "2025-02-28 23:59:59"),
        ("OOS1 (Mar 25)", "2025-03-01", "2025-03-31 23:59:59"),
        ("OOS2 (Apr-Jun 25)", "2025-04-01", "2025-06-30 23:59:59"),
        ("OOS3 (Jul-Sep 25)", "2025-07-01", "2025-09-30 23:59:59"),
        ("OOS4 (Oct-Jan 26)", "2025-10-01", "2026-01-14 23:59:59"),
    ]

    # Robust config parameters
    ema = 21
    z_lb = 21
    entry_thresh = 5.0
    exit_thresh = 1.0
    max_bars = 48

    # Also test original config for comparison
    configs = [
        ("ORIGINAL (Z=3.5, Exit=0, 24/7)", 3.5, 0.0, 36, False),
        ("ROBUST (Z=5.0, Exit=1.0, RTH)", 5.0, 1.0, 48, True),
    ]

    print("\n2. Testing configurations...")
    print()

    for config_name, entry_z, exit_z, max_b, rth_only in configs:
        print("=" * 100)
        print(f"CONFIG: {config_name}")
        print("=" * 100)

        all_trades = []
        total_oos_pnl = 0
        total_oos_trades = 0

        print(f"\n{'Period':<25} {'Trades':>8} {'P&L':>12} {'WR':>8} {'PF':>8} {'Avg Trade':>12}")
        print("-" * 80)

        for name, start, end in periods:
            start_dt = pd.Timestamp(start, tz='UTC')
            end_dt = pd.Timestamp(end, tz='UTC')
            bars = all_bars[(all_bars['timestamp'] >= start_dt) & (all_bars['timestamp'] <= end_dt)].copy()

            # Compute indicators
            bars['ema'] = bars['close'].ewm(span=ema, adjust=False).mean()
            bars['dist'] = (bars['close'] - bars['ema']) / bars['ema']
            bars['dist_std'] = bars['dist'].rolling(z_lb).std()
            bars['zscore'] = bars['dist'] / bars['dist_std']

            # Run backtest
            trades = run_backtest_robust(
                bars, bars['zscore'].values, entry_z, exit_z, cfg, max_b, rth_only
            )
            m = compute_metrics(trades)

            status = "+" if m['pnl'] > 0 else "-"
            print(f"{name:<25} {m['n']:>8} ${m['pnl']:>10.2f} {m['wr']:>7.1f}% {m['pf_net']:>7.2f} ${m['avg_trade']:>10.2f} {status}")

            if "OOS" in name:
                total_oos_pnl += m['pnl']
                total_oos_trades += m['n']

            all_trades.extend(trades)

        # Summary
        print("-" * 80)
        total_m = compute_metrics(all_trades)
        print(f"{'TOTAL':<25} {total_m['n']:>8} ${total_m['pnl']:>10.2f} {total_m['wr']:>7.1f}% {total_m['pf_net']:>7.2f} ${total_m['avg_trade']:>10.2f}")
        print(f"{'OOS TOTAL':<25} {total_oos_trades:>8} ${total_oos_pnl:>10.2f}")

        # Direction breakdown
        if all_trades:
            longs = [t for t in all_trades if t['direction'] == 'LONG']
            shorts = [t for t in all_trades if t['direction'] == 'SHORT']
            long_pnl = sum(t['net_pnl'] for t in longs)
            short_pnl = sum(t['net_pnl'] for t in shorts)
            print(f"\n  Direction breakdown:")
            print(f"    Longs:  {len(longs):>4} trades, ${long_pnl:>8.2f} P&L")
            print(f"    Shorts: {len(shorts):>4} trades, ${short_pnl:>8.2f} P&L")

        print()

    # Final comparison
    print("\n" + "=" * 100)
    print("VERIFICATION SUMMARY")
    print("=" * 100)

    print("""
    Testing the robust configuration from EMA_ZSCORE_ROBUST_SPECIFICATION.md:

    Key differences:
    - Entry Z: 3.5 -> 5.0 (only extreme conditions)
    - Exit Z: 0.0 -> 1.0 (exit earlier)
    - Hours: 24/7 -> RTH only (9 AM - 4 PM)
    - Max Hold: 36 -> 48 bars

    The robust config trades much less frequently but should be more selective.
    """)

    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
