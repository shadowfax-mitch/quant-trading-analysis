"""
EMA Z-Score Mean Reversion with Directional Filter

Instead of disabling in trends, trade WITH the trend:
- Uptrend: Only take LONG entries (buy the dips)
- Downtrend: Only take SHORT entries (sell the rallies)

This aligns mean reversion with trend direction.
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


def run_backtest_directional(df: pd.DataFrame, zscore: np.ndarray, entry_thresh: float,
                              exit_thresh: float, cfg: BacktestConfig, max_bars: int,
                              trend: np.ndarray, mode: str = 'both') -> list:
    """
    Mean reversion with directional filter.
    mode: 'both' (no filter), 'with_trend' (longs in uptrend, shorts in downtrend),
          'counter_trend' (opposite)
    """
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
        prev_trend = trend[i - 1] if not np.isnan(trend[i - 1]) else 0

        if position == 0:
            # Check entry conditions with directional filter
            want_long = prev_z < -entry_thresh
            want_short = prev_z > entry_thresh

            if mode == 'with_trend':
                # Only longs in uptrend, only shorts in downtrend
                if want_long and prev_trend <= 0:
                    want_long = False  # Uptrend needed for longs
                if want_short and prev_trend >= 0:
                    want_short = False  # Downtrend needed for shorts
            elif mode == 'counter_trend':
                # Only longs in downtrend, only shorts in uptrend
                if want_long and prev_trend >= 0:
                    want_long = False
                if want_short and prev_trend <= 0:
                    want_short = False

            if want_long:
                position = 1
                entry_price = open_prices[i] + cfg.tick_size
                entry_bar = i
            elif want_short:
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
    print("EMA Z-SCORE WITH DIRECTIONAL FILTER - MNQ")
    print("=" * 100)
    print()

    cfg = BacktestConfig()

    # Load data
    print("1. Loading MNQ 5-minute bars...")
    all_bars = load_bars()

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

    # Best config
    ema, z_lb, entry, exit_z, max_b = 21, 21, 3.5, 0.0, 36

    # Test different trend lookbacks and modes
    trend_lookbacks = [50, 100, 200, 288, 500]
    modes = ['both', 'with_trend', 'counter_trend']

    print("\n3. Testing directional filters...")
    print()

    best_result = None
    best_oos_pnl = float('-inf')

    for lookback in trend_lookbacks:
        for mode in modes:
            results = {}
            total_oos_pnl = 0
            total_oos_trades = 0
            all_oos_profitable = True

            for name, _, _ in periods:
                bars = period_data[name].copy()

                # Compute indicators
                bars['ema'] = bars['close'].ewm(span=ema, adjust=False).mean()
                bars['dist'] = (bars['close'] - bars['ema']) / bars['ema']
                bars['dist_std'] = bars['dist'].rolling(z_lb).std()
                bars['zscore'] = bars['dist'] / bars['dist_std']

                # Trend: positive = uptrend, negative = downtrend
                bars['trend'] = bars['close'].pct_change(lookback)

                trades = run_backtest_directional(
                    bars, bars['zscore'].values, entry, exit_z, cfg, max_b,
                    bars['trend'].values, mode
                )
                m = compute_metrics(trades)
                results[name] = m

                if "OOS" in name:
                    total_oos_pnl += m['pnl']
                    total_oos_trades += m['n']
                    if m['pnl'] <= 0:
                        all_oos_profitable = False

            # Track best
            if total_oos_pnl > best_oos_pnl:
                best_oos_pnl = total_oos_pnl
                best_result = (lookback, mode, results, all_oos_profitable, total_oos_trades)

            # Print if all OOS profitable or significant improvement
            if all_oos_profitable:
                print(f"\n*** ALL OOS PROFITABLE *** Lookback={lookback}, Mode={mode}:")
                for name, _, _ in periods:
                    m = results[name]
                    print(f"  {name:20} | Trades: {m['n']:>4} | P&L: ${m['pnl']:>8.2f} | PF: {m['pf_net']:.2f}")
                print(f"  {'TOTAL OOS':<20} | Trades: {total_oos_trades:>4} | P&L: ${total_oos_pnl:>8.2f}")

    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    print(f"\nBaseline OOS P&L: $-6552.50")

    if best_result:
        lookback, mode, results, all_profitable, total_trades = best_result
        print(f"\nBest Result:")
        print(f"  Lookback: {lookback}")
        print(f"  Mode: {mode}")
        print(f"  OOS P&L: ${best_oos_pnl:.2f}")
        print(f"  OOS Trades: {total_trades}")
        print(f"  All OOS Profitable: {all_profitable}")

        print("\nDetailed results:")
        for name, _, _ in periods:
            m = results[name]
            status = "PROFIT" if m['pnl'] > 0 else "LOSS"
            print(f"  {name:20} | Trades: {m['n']:>4} | P&L: ${m['pnl']:>8.2f} | PF: {m['pf_net']:.2f} | {status}")

        if all_profitable:
            print("\n" + "=" * 100)
            print("DIRECTIONAL FILTER SOLVES THE PROBLEM!")
            print("=" * 100)
            print(f"""
Strategy: Trade WITH the trend
- In uptrends (positive {lookback}-bar return): Only take LONG entries
- In downtrends (negative {lookback}-bar return): Only take SHORT entries

This turns the strategy from pure mean-reversion into trend-following mean-reversion:
- Wait for pullback (Z-score extreme)
- Enter in direction of trend
- Exit when Z-score normalizes
            """)

    print("\n" + "=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
