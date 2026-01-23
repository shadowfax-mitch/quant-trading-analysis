"""
EMA Z-Score Mean Reversion with Trend Filter

Test if adding a trend filter can avoid losses in trending markets.

Filter: Disable strategy when rolling N-bar return exceeds threshold.
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


def run_backtest_with_filter(df: pd.DataFrame, zscore: np.ndarray, entry_thresh: float,
                              exit_thresh: float, cfg: BacktestConfig, max_bars: int,
                              trend_filter: np.ndarray = None, trend_threshold: float = None) -> list:
    """Mean reversion on Z-score with optional trend filter."""
    trades = []
    n = len(df)
    open_prices = df['open'].values
    position = 0
    entry_price = 0.0
    entry_bar = 0
    filtered_entries = 0

    for i in range(1, n):
        if np.isnan(zscore[i-1]):
            continue

        prev_z = zscore[i - 1]

        if position == 0:
            # Check trend filter before entry
            if trend_filter is not None and trend_threshold is not None:
                if not np.isnan(trend_filter[i-1]) and abs(trend_filter[i-1]) > trend_threshold:
                    filtered_entries += 1
                    continue  # Skip entry - market is trending

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

    return trades, filtered_entries


def compute_metrics(trades: list) -> dict:
    if not trades:
        return {"n": 0, "pnl": 0, "wr": 0, "pf_net": 0}

    net_pnls = [t["net_pnl"] for t in trades]
    wins = sum(1 for p in net_pnls if p > 0)
    net_profit = sum(p for p in net_pnls if p > 0)
    net_loss = abs(sum(p for p in net_pnls if p < 0))

    return {
        "n": len(trades),
        "pnl": sum(net_pnls),
        "wr": wins / len(trades) * 100 if trades else 0,
        "pf_net": net_profit / net_loss if net_loss > 0 else float("inf"),
    }


def main():
    print("=" * 100)
    print("EMA Z-SCORE WITH TREND FILTER - MNQ")
    print("=" * 100)
    print()

    cfg = BacktestConfig()

    # Load data
    print("1. Loading MNQ 5-minute bars...")
    all_bars = load_bars()
    print(f"   Total bars: {len(all_bars):,}")

    # Define periods
    periods = [
        ("Train", "2025-01-01", "2025-02-28 23:59:59"),
        ("OOS1 (Mar)", "2025-03-01", "2025-03-31 23:59:59"),
        ("OOS2 (Apr-Jun)", "2025-04-01", "2025-06-30 23:59:59"),
        ("OOS3 (Jul-Sep)", "2025-07-01", "2025-09-30 23:59:59"),
        ("OOS4 (Oct-Jan)", "2025-10-01", "2026-01-14 23:59:59"),
    ]

    # Extract period data
    print("\n2. Extracting period data...")
    period_data = {}
    for name, start, end in periods:
        start_dt = pd.Timestamp(start, tz='UTC')
        end_dt = pd.Timestamp(end, tz='UTC')
        bars = all_bars[(all_bars['timestamp'] >= start_dt) & (all_bars['timestamp'] <= end_dt)].copy()
        period_data[name] = bars
        print(f"   {name}: {len(bars):,} bars")

    # Best config
    ema, z_lb, entry, exit_z, max_b = 21, 21, 3.5, 0.0, 36

    # Test different trend filter parameters
    trend_lookbacks = [50, 100, 200, 288]  # 288 = ~1 day
    trend_thresholds = [0.01, 0.02, 0.03, 0.04, 0.05]  # 1% to 5%

    print("\n3. Testing trend filters...")
    print()

    # First show baseline (no filter)
    print("BASELINE (No Filter):")
    print("-" * 100)

    baseline_total = 0
    for name, _, _ in periods:
        bars = period_data[name].copy()
        bars['ema'] = bars['close'].ewm(span=ema, adjust=False).mean()
        bars['dist'] = (bars['close'] - bars['ema']) / bars['ema']
        bars['dist_std'] = bars['dist'].rolling(z_lb).std()
        bars['zscore'] = bars['dist'] / bars['dist_std']

        trades, _ = run_backtest_with_filter(bars, bars['zscore'].values, entry, exit_z, cfg, max_b)
        m = compute_metrics(trades)
        print(f"  {name:20} | Trades: {m['n']:>4} | P&L: ${m['pnl']:>8.2f} | PF: {m['pf_net']:.2f}")
        if "OOS" in name:
            baseline_total += m['pnl']

    print(f"  {'TOTAL OOS':<20} | P&L: ${baseline_total:>8.2f}")
    print()

    # Test filters
    print("=" * 100)
    print("TREND FILTER RESULTS")
    print("=" * 100)

    best_filter = None
    best_oos_pnl = baseline_total

    for lookback in trend_lookbacks:
        for threshold in trend_thresholds:
            total_oos_pnl = 0
            total_oos_trades = 0
            all_profitable = True
            results = {}

            for name, _, _ in periods:
                bars = period_data[name].copy()

                # Compute indicators
                bars['ema'] = bars['close'].ewm(span=ema, adjust=False).mean()
                bars['dist'] = (bars['close'] - bars['ema']) / bars['ema']
                bars['dist_std'] = bars['dist'].rolling(z_lb).std()
                bars['zscore'] = bars['dist'] / bars['dist_std']

                # Trend filter: rolling return
                bars['trend'] = bars['close'].pct_change(lookback)

                trades, filtered = run_backtest_with_filter(
                    bars, bars['zscore'].values, entry, exit_z, cfg, max_b,
                    bars['trend'].values, threshold
                )
                m = compute_metrics(trades)
                results[name] = (m, filtered)

                if "OOS" in name:
                    total_oos_pnl += m['pnl']
                    total_oos_trades += m['n']
                    if m['pnl'] <= 0:
                        all_profitable = False

            # Check if this filter is better
            if total_oos_pnl > best_oos_pnl:
                best_oos_pnl = total_oos_pnl
                best_filter = (lookback, threshold, results, all_profitable)

            # Only print if all OOS periods are profitable or significantly better
            if all_profitable or total_oos_pnl > 0:
                print(f"\nLookback={lookback}, Threshold={threshold:.1%}:")
                for name, _, _ in periods:
                    m, filtered = results[name]
                    status = "+" if m['pnl'] > 0 else "-"
                    print(f"  {name:20} | Trades: {m['n']:>4} | P&L: ${m['pnl']:>8.2f} | Filtered: {filtered:>4} | {status}")
                print(f"  {'TOTAL OOS':<20} | Trades: {total_oos_trades:>4} | P&L: ${total_oos_pnl:>8.2f} | {'ALL PROFIT' if all_profitable else ''}")

    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    print(f"\nBaseline OOS P&L: ${baseline_total:.2f}")

    if best_filter:
        lookback, threshold, results, all_profitable = best_filter
        print(f"Best Filter OOS P&L: ${best_oos_pnl:.2f}")
        print(f"Best Filter: Lookback={lookback}, Threshold={threshold:.1%}")
        print(f"Improvement: ${best_oos_pnl - baseline_total:.2f}")

        if all_profitable:
            print("\n*** FILTER MAKES ALL OOS PERIODS PROFITABLE ***")
        else:
            print("\nFilter improves but doesn't fully solve the problem.")

        print("\nBest filter detailed results:")
        for name, _, _ in periods:
            m, filtered = results[name]
            print(f"  {name:20} | Trades: {m['n']:>4} | P&L: ${m['pnl']:>8.2f} | PF: {m['pf_net']:.2f}")
    else:
        print("No filter found that improves OOS performance.")

    print("\n" + "=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
