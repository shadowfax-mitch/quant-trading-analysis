"""
Verify the Robust EMA Z-Score Configuration on MES Data

MES data: Jan 2025 - Jul 2025 (6 months)

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
import gc


@dataclass
class BacktestConfig:
    commission_per_side: float = 0.85
    tick_size: float = 0.25
    tick_value: float = 1.25  # MES (not MNQ)


def load_and_cache_mes_bars():
    """Load MES tick data and cache as 5-min bars."""
    cache_file = Path('data/mes_5min_bars_full.parquet')

    if cache_file.exists():
        print("   Loading cached 5-min bars...")
        return pd.read_parquet(cache_file)

    print("   Building 5-min bars from tick data (this will be cached)...")
    data_dir = Path('datasets/MES/tick_data')

    all_bars = []

    for i in range(1, 105):
        file_path = data_dir / f'mes_ticks_part{i:04d}.csv'
        if not file_path.exists():
            continue

        if i % 20 == 0:
            print(f"      Processing file {i}/104...")

        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

        # Aggregate to 5-min bars immediately
        df['bar'] = df['timestamp'].dt.floor('5min')
        bars = df.groupby('bar').agg({
            'last': ['first', 'max', 'min', 'last'],
        }).reset_index()
        bars.columns = ['timestamp', 'open', 'high', 'low', 'close']

        all_bars.append(bars)
        del df
        gc.collect()

    result = pd.concat(all_bars, ignore_index=True)
    result = result.sort_values('timestamp').drop_duplicates(subset=['timestamp']).reset_index(drop=True)

    # Cache for future use
    result.to_parquet(cache_file)
    print(f"   Cached {len(result):,} bars to {cache_file}")

    return result


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
    print("VERIFYING ROBUST EMA Z-SCORE CONFIG ON MES")
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
    print("1. Loading MES 5-minute bars...")
    all_bars = load_and_cache_mes_bars()
    print(f"   Total bars: {len(all_bars):,}")
    print(f"   Date range: {all_bars['timestamp'].min()} to {all_bars['timestamp'].max()}")

    # Define periods (MES has Jan-Jul 2025)
    periods = [
        ("Train (Jan-Feb 25)", "2025-01-01", "2025-02-28 23:59:59"),
        ("OOS1 (Mar 25)", "2025-03-01", "2025-03-31 23:59:59"),
        ("OOS2 (Apr-Jun 25)", "2025-04-01", "2025-06-30 23:59:59"),
        ("OOS3 (Jul 25)", "2025-07-01", "2025-07-31 23:59:59"),
    ]

    # Parameters
    ema = 21
    z_lb = 21

    # Test both configs
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

            if len(bars) == 0:
                print(f"{name:<25} {'N/A':>8} {'N/A':>12} {'N/A':>8} {'N/A':>8} {'N/A':>12}")
                continue

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

            status = "+" if m['pnl'] > 0 else "-" if m['n'] > 0 else " "
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

    # Cross-instrument comparison
    print("\n" + "=" * 100)
    print("CROSS-INSTRUMENT SUMMARY (Robust Config)")
    print("=" * 100)
    print("""
    MNQ Results (verified earlier):
    - OOS Total: +$1,177.80 (21 trades)
    - PF: 3.23
    - All 4 OOS periods profitable

    MES Results (above):
    - Compare OOS totals to verify cross-instrument robustness
    """)

    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
