"""
EMA Z-Score Mean Reversion - Extended OOS Testing on MNQ

Test strategy across multiple out-of-sample periods using the full MNQ dataset.

Periods:
- Train: Jan 1 - Feb 28, 2025
- OOS1: Mar 1 - Mar 31, 2025 (original test)
- OOS2: Apr 1 - Jun 30, 2025
- OOS3: Jul 1 - Sep 30, 2025
- OOS4: Oct 1, 2025 - Jan 14, 2026
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
    tick_value: float = 0.50  # MNQ


def load_and_cache_mnq_bars():
    """Load all MNQ tick data and cache as 5-min bars."""
    cache_file = Path('data/mnq_5min_bars_full.parquet')

    if cache_file.exists():
        print("   Loading cached 5-min bars...")
        return pd.read_parquet(cache_file)

    print("   Building 5-min bars from tick data (this will be cached)...")
    data_dir = Path('datasets/MNQ/tick_data')

    all_bars = []

    # Only read files 49-124 which contain 2025 data
    for i in range(49, 125):
        file_path = data_dir / f'mnq_ticks_part{i:04d}.csv'
        if not file_path.exists():
            continue

        if i % 20 == 0:
            print(f"      Processing file {i}/124...")

        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

        # Filter to 2025 and later only (we don't need 2024 data)
        df = df[df['timestamp'] >= '2025-01-01']

        if len(df) == 0:
            del df
            gc.collect()
            continue

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
    cache_file.parent.mkdir(exist_ok=True)
    result.to_parquet(cache_file)
    print(f"   Cached {len(result):,} bars to {cache_file}")

    return result


def run_backtest(df: pd.DataFrame, zscore: np.ndarray, entry_thresh: float,
                 exit_thresh: float, cfg: BacktestConfig, max_bars: int) -> list:
    """Mean reversion on Z-score."""
    trades = []
    n = len(df)
    open_prices = df['open'].values
    position = 0
    entry_price = 0.0
    entry_bar = 0

    for i in range(1, n):
        if np.isnan(zscore[i-1]):
            continue

        prev_z = zscore[i - 1]

        if position == 0:
            if prev_z < -entry_thresh:
                position = 1
                entry_price = open_prices[i] + cfg.tick_size
                entry_bar = i
            elif prev_z > entry_thresh:
                position = -1
                entry_price = open_prices[i] - cfg.tick_size
                entry_bar = i
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
                trades.append({"net_pnl": net_pnl, "ticks": gross_pnl / cfg.tick_value})
                position = 0

    if position != 0:
        if position == 1:
            exit_price = open_prices[-1] - cfg.tick_size
            gross_pnl = (exit_price - entry_price) / cfg.tick_size * cfg.tick_value
        else:
            exit_price = open_prices[-1] + cfg.tick_size
            gross_pnl = (entry_price - exit_price) / cfg.tick_size * cfg.tick_value
        net_pnl = gross_pnl - 2 * cfg.commission_per_side
        trades.append({"net_pnl": net_pnl, "ticks": gross_pnl / cfg.tick_value})

    return trades


def compute_metrics(trades: list) -> dict:
    if not trades:
        return {"n": 0, "pnl": 0, "wr": 0, "ticks": 0, "pf_net": 0}

    net_pnls = [t["net_pnl"] for t in trades]
    ticks_arr = [t["ticks"] for t in trades]

    wins = sum(1 for p in net_pnls if p > 0)
    net_profit = sum(p for p in net_pnls if p > 0)
    net_loss = abs(sum(p for p in net_pnls if p < 0))

    return {
        "n": len(trades),
        "pnl": sum(net_pnls),
        "wr": wins / len(trades) * 100 if trades else 0,
        "ticks": np.mean(ticks_arr) if ticks_arr else 0,
        "pf_net": net_profit / net_loss if net_loss > 0 else float("inf"),
    }


def check_gates(m: dict, min_trades: int = 20) -> bool:
    return m["pnl"] > 0 and m["pf_net"] >= 1.1 and m["n"] >= min_trades


def main():
    print("=" * 100)
    print("EMA Z-SCORE EXTENDED OOS TESTING - MNQ")
    print("=" * 100)
    print()

    cfg = BacktestConfig()

    # Load all data
    print("1. Loading MNQ 5-minute bars...")
    all_bars = load_and_cache_mnq_bars()
    print(f"   Total bars: {len(all_bars):,}")
    print(f"   Date range: {all_bars['timestamp'].min()} to {all_bars['timestamp'].max()}")
    print()

    # Define periods
    periods = [
        ("Train", "2025-01-01", "2025-02-28 23:59:59"),
        ("OOS1 (Mar)", "2025-03-01", "2025-03-31 23:59:59"),
        ("OOS2 (Apr-Jun)", "2025-04-01", "2025-06-30 23:59:59"),
        ("OOS3 (Jul-Sep)", "2025-07-01", "2025-09-30 23:59:59"),
        ("OOS4 (Oct-Jan)", "2025-10-01", "2026-01-14 23:59:59"),
    ]

    # Extract period data
    print("2. Extracting period data...")
    period_data = {}
    for name, start, end in periods:
        start_dt = pd.Timestamp(start, tz='UTC')
        end_dt = pd.Timestamp(end, tz='UTC')
        bars = all_bars[(all_bars['timestamp'] >= start_dt) & (all_bars['timestamp'] <= end_dt)].copy()
        period_data[name] = bars
        print(f"   {name}: {len(bars):,} bars")
    print()

    # Best configs from MNQ testing (BOTH GO configs)
    configs = [
        (21, 21, 3.5, 0.0, 36),  # Best MNQ config
        (21, 21, 3.5, 0.5, 36),
        (21, 34, 3.5, 0.0, 24),
        (21, 21, 3.5, 0.5, 24),
        (13, 21, 3.5, 0.0, 24),
        (13, 21, 3.5, 0.0, 36),
    ]

    print("3. Testing configs across all periods...")
    print()

    # Results table header
    print(f"{'Config':<25} | ", end="")
    for name, _, _ in periods:
        print(f"{name:^20} | ", end="")
    print("Status")
    print("-" * 150)

    all_results = []

    for ema, z_lb, entry, exit_z, max_b in configs:
        config_str = f"E{ema} Z{z_lb} {entry}/{exit_z} M{max_b}"

        period_results = {}

        for name, _, _ in periods:
            bars = period_data[name].copy()

            if len(bars) < z_lb + 10:
                period_results[name] = {"n": 0, "pnl": 0, "pf_net": 0, "wr": 0}
                continue

            # Compute Z-score
            bars['ema'] = bars['close'].ewm(span=ema, adjust=False).mean()
            bars['dist'] = (bars['close'] - bars['ema']) / bars['ema']
            bars['dist_std'] = bars['dist'].rolling(z_lb).std()
            bars['zscore'] = bars['dist'] / bars['dist_std']

            zscore = bars['zscore'].values

            # Backtest
            trades = run_backtest(bars, zscore, entry, exit_z, cfg, max_b)
            metrics = compute_metrics(trades)
            period_results[name] = metrics

        # Print row
        print(f"{config_str:<25} | ", end="")

        all_profitable = True
        all_go = True

        for name, _, _ in periods:
            m = period_results[name]
            pnl_str = f"${m['pnl']:>7.0f}" if m['n'] > 0 else "N/A"
            n_str = f"n={m['n']:<3}"
            pf_str = f"PF={m['pf_net']:.2f}" if m['pf_net'] < 100 else "PF=Inf"

            if m['pnl'] <= 0:
                all_profitable = False
            if not check_gates(m, 15 if "OOS" in name else 30):
                all_go = False

            print(f"{pnl_str} {n_str} {pf_str:>8} | ", end="")

        if all_profitable and all_go:
            status = "ALL GO!"
        elif all_profitable:
            status = "ALL PROFIT"
        else:
            # Count profitable periods
            profitable = sum(1 for m in period_results.values() if m['pnl'] > 0)
            status = f"{profitable}/5 profit"

        print(status)

        all_results.append({
            "config": config_str,
            "ema": ema, "z_lb": z_lb, "entry": entry, "exit": exit_z, "max": max_b,
            "periods": period_results,
            "all_profitable": all_profitable,
            "all_go": all_go,
        })

    # Summary
    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)

    all_profit_configs = [r for r in all_results if r['all_profitable']]
    all_go_configs = [r for r in all_results if r['all_go']]

    print(f"\nConfigs tested: {len(all_results)}")
    print(f"All periods profitable: {len(all_profit_configs)}")
    print(f"All periods GO: {len(all_go_configs)}")

    # Period-by-period summary
    print("\n" + "-" * 60)
    print("PERIOD-BY-PERIOD SUMMARY (Best Config: E21 Z21 3.5/0.0 M36)")
    print("-" * 60)

    best_config = all_results[0]  # First config is best MNQ config
    total_pnl = 0
    total_trades = 0

    for name, _, _ in periods:
        m = best_config['periods'][name]
        print(f"{name:20} | Trades: {m['n']:>4} | P&L: ${m['pnl']:>8.2f} | PF: {m['pf_net']:>5.2f} | WR: {m['wr']:>5.1f}%")
        if "OOS" in name:
            total_pnl += m['pnl']
            total_trades += m['n']

    print("-" * 60)
    print(f"{'TOTAL OOS (4 periods)':<20} | Trades: {total_trades:>4} | P&L: ${total_pnl:>8.2f}")

    # Calculate cumulative stats
    print("\n" + "=" * 100)
    print("EXTENDED OOS VALIDATION RESULT")
    print("=" * 100)

    if all_profit_configs:
        print("\nSTRATEGY SURVIVES EXTENDED OOS TESTING!")
        print(f"- {len(all_profit_configs)}/{len(all_results)} configs profitable across ALL 5 periods")

        # Show best config cumulative
        best = all_profit_configs[0]
        oos_total = sum(best['periods'][name]['pnl'] for name, _, _ in periods if 'OOS' in name)
        print(f"- Best config OOS total: ${oos_total:.2f}")
        print(f"- Time span: Jan 2025 - Jan 2026 (12+ months)")
    else:
        print("\nSTRATEGY FAILS EXTENDED OOS TESTING")
        print("- No config profitable across all periods")

        # Show which periods failed
        for r in all_results[:3]:
            failed = [name for name, _, _ in periods if r['periods'][name]['pnl'] <= 0]
            if failed:
                print(f"- {r['config']}: Failed in {', '.join(failed)}")

    print("\n" + "=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
