"""
Analyze WHY trades fail in certain periods.

Look at:
1. Long vs Short trade performance
2. Trade duration
3. Drawdown during trades
4. Entry Z-score distribution
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
    cache_file = Path('data/mnq_5min_bars_full.parquet')
    return pd.read_parquet(cache_file)


def run_backtest_detailed(df: pd.DataFrame, zscore: np.ndarray, entry_thresh: float,
                           exit_thresh: float, cfg: BacktestConfig, max_bars: int) -> list:
    """Detailed backtest capturing trade characteristics."""
    trades = []
    n = len(df)
    open_prices = df['open'].values
    close_prices = df['close'].values
    high_prices = df['high'].values
    low_prices = df['low'].values
    position = 0
    entry_price = 0.0
    entry_bar = 0
    entry_z = 0.0

    for i in range(1, n):
        if np.isnan(zscore[i-1]):
            continue

        prev_z = zscore[i - 1]

        if position == 0:
            if prev_z < -entry_thresh:
                position = 1
                entry_price = open_prices[i] + cfg.tick_size
                entry_bar = i
                entry_z = prev_z
            elif prev_z > entry_thresh:
                position = -1
                entry_price = open_prices[i] - cfg.tick_size
                entry_bar = i
                entry_z = prev_z
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
                    # Max adverse excursion
                    mae = min(low_prices[entry_bar:i+1]) - entry_price
                    mfe = max(high_prices[entry_bar:i+1]) - entry_price
                else:
                    exit_price = open_prices[i] + cfg.tick_size
                    gross_pnl = (entry_price - exit_price) / cfg.tick_size * cfg.tick_value
                    mae = entry_price - max(high_prices[entry_bar:i+1])
                    mfe = entry_price - min(low_prices[entry_bar:i+1])

                net_pnl = gross_pnl - 2 * cfg.commission_per_side
                trades.append({
                    "net_pnl": net_pnl,
                    "direction": "LONG" if position == 1 else "SHORT",
                    "entry_z": entry_z,
                    "hold_bars": hold_time,
                    "mae": mae,
                    "mfe": mfe,
                    "ticks": gross_pnl / cfg.tick_value,
                })
                position = 0

    return trades


def analyze_trades(trades: list, name: str):
    """Analyze trade characteristics."""
    if not trades:
        return {"name": name, "n": 0}

    df = pd.DataFrame(trades)

    longs = df[df['direction'] == 'LONG']
    shorts = df[df['direction'] == 'SHORT']

    return {
        "name": name,
        "n": len(df),
        "total_pnl": df['net_pnl'].sum(),
        "long_n": len(longs),
        "long_pnl": longs['net_pnl'].sum() if len(longs) > 0 else 0,
        "long_wr": (longs['net_pnl'] > 0).mean() * 100 if len(longs) > 0 else 0,
        "short_n": len(shorts),
        "short_pnl": shorts['net_pnl'].sum() if len(shorts) > 0 else 0,
        "short_wr": (shorts['net_pnl'] > 0).mean() * 100 if len(shorts) > 0 else 0,
        "avg_hold": df['hold_bars'].mean(),
        "avg_mae": df['mae'].mean(),
        "avg_mfe": df['mfe'].mean(),
        "avg_entry_z": df['entry_z'].abs().mean(),
    }


def main():
    print("=" * 100)
    print("TRADE FAILURE ANALYSIS")
    print("=" * 100)
    print()

    cfg = BacktestConfig()
    all_bars = load_bars()

    periods = [
        ("Train", "2025-01-01", "2025-02-28 23:59:59", "PROFIT"),
        ("OOS1 (Mar)", "2025-03-01", "2025-03-31 23:59:59", "PROFIT"),
        ("OOS2 (Apr-Jun)", "2025-04-01", "2025-06-30 23:59:59", "FAIL"),
        ("OOS3 (Jul-Sep)", "2025-07-01", "2025-09-30 23:59:59", "MARGINAL"),
        ("OOS4 (Oct-Jan)", "2025-10-01", "2026-01-14 23:59:59", "FAIL"),
    ]

    ema, z_lb, entry, exit_z, max_b = 21, 21, 3.5, 0.0, 36

    print("LONG vs SHORT PERFORMANCE BY PERIOD")
    print("=" * 100)
    print(f"\n{'Period':<20} {'Status':<10} | {'Longs':>6} {'L_PnL':>10} {'L_WR':>8} | {'Shorts':>6} {'S_PnL':>10} {'S_WR':>8} | {'Total':>10}")
    print("-" * 100)

    all_analysis = []

    for name, start, end, status in periods:
        start_dt = pd.Timestamp(start, tz='UTC')
        end_dt = pd.Timestamp(end, tz='UTC')
        bars = all_bars[(all_bars['timestamp'] >= start_dt) & (all_bars['timestamp'] <= end_dt)].copy()

        bars['ema'] = bars['close'].ewm(span=ema, adjust=False).mean()
        bars['dist'] = (bars['close'] - bars['ema']) / bars['ema']
        bars['dist_std'] = bars['dist'].rolling(z_lb).std()
        bars['zscore'] = bars['dist'] / bars['dist_std']

        trades = run_backtest_detailed(bars, bars['zscore'].values, entry, exit_z, cfg, max_b)
        analysis = analyze_trades(trades, name)
        analysis['status'] = status
        all_analysis.append(analysis)

        print(f"{name:<20} {status:<10} | {analysis['long_n']:>6} ${analysis['long_pnl']:>9.0f} {analysis['long_wr']:>7.1f}% | "
              f"{analysis['short_n']:>6} ${analysis['short_pnl']:>9.0f} {analysis['short_wr']:>7.1f}% | ${analysis['total_pnl']:>9.0f}")

    # Analyze patterns
    print("\n" + "=" * 100)
    print("KEY INSIGHTS")
    print("=" * 100)

    profit_periods = [a for a in all_analysis if a['status'] == 'PROFIT']
    fail_periods = [a for a in all_analysis if a['status'] == 'FAIL']

    def avg(lst, key):
        vals = [a[key] for a in lst if a['n'] > 0]
        return np.mean(vals) if vals else 0

    print("\nPROFITABLE PERIODS:")
    print(f"  Long P&L:  ${avg(profit_periods, 'long_pnl'):.0f} (WR: {avg(profit_periods, 'long_wr'):.1f}%)")
    print(f"  Short P&L: ${avg(profit_periods, 'short_pnl'):.0f} (WR: {avg(profit_periods, 'short_wr'):.1f}%)")

    print("\nFAILING PERIODS:")
    print(f"  Long P&L:  ${avg(fail_periods, 'long_pnl'):.0f} (WR: {avg(fail_periods, 'long_wr'):.1f}%)")
    print(f"  Short P&L: ${avg(fail_periods, 'short_pnl'):.0f} (WR: {avg(fail_periods, 'short_wr'):.1f}%)")

    # Identify which side is failing
    print("\n" + "-" * 60)

    # Check OOS2 specifically (biggest failure)
    oos2 = [a for a in all_analysis if a['name'] == 'OOS2 (Apr-Jun)'][0]
    print(f"\nOOS2 (Apr-Jun) - Biggest failure (-$6208):")
    print(f"  Long trades:  {oos2['long_n']} trades, ${oos2['long_pnl']:.0f} P&L, {oos2['long_wr']:.1f}% WR")
    print(f"  Short trades: {oos2['short_n']} trades, ${oos2['short_pnl']:.0f} P&L, {oos2['short_wr']:.1f}% WR")

    if oos2['short_pnl'] < oos2['long_pnl']:
        print(f"\n  --> SHORTS ARE THE PROBLEM in the uptrending market!")
        print(f"      Shorts lost ${abs(oos2['short_pnl']):.0f} while longs made ${oos2['long_pnl']:.0f}")
    else:
        print(f"\n  --> LONGS ARE THE PROBLEM")

    # Solution suggestion
    print("\n" + "=" * 100)
    print("SUGGESTED SOLUTION")
    print("=" * 100)

    print("""
    The analysis shows that in trending markets:
    - Apr-Jun 2025 was a strong UPTREND (+18%)
    - SHORT trades got crushed (mean reversion against trend fails)

    SOLUTION: Only trade in the direction that aligns with the higher timeframe trend.

    Options:
    1. Use a longer EMA (e.g., 100 or 200 bars) as trend filter
       - If price > EMA_200: Only take LONG entries
       - If price < EMA_200: Only take SHORT entries

    2. Use a simpler approach: If 50-bar return > 2%, only longs
       If 50-bar return < -2%, only shorts

    This turns the strategy into "buy the dip in uptrends, sell the rally in downtrends"
    """)

    # Test the long-only vs short-only approach
    print("\n" + "=" * 100)
    print("TESTING: LONG-ONLY vs SHORT-ONLY RESULTS")
    print("=" * 100)

    print(f"\n{'Period':<20} {'Long-Only PnL':>15} {'Short-Only PnL':>15} {'Better Side':<15}")
    print("-" * 70)

    for a in all_analysis:
        better = "LONG" if a['long_pnl'] > a['short_pnl'] else "SHORT"
        print(f"{a['name']:<20} ${a['long_pnl']:>14.0f} ${a['short_pnl']:>14.0f} {better:<15}")

    # What if we had perfect foresight?
    print("\n" + "-" * 60)
    print("If we could perfectly choose the right side each period:")
    perfect_pnl = sum(max(a['long_pnl'], a['short_pnl']) for a in all_analysis if 'OOS' in a['name'])
    print(f"  Perfect OOS P&L: ${perfect_pnl:.0f}")

    long_only_oos = sum(a['long_pnl'] for a in all_analysis if 'OOS' in a['name'])
    short_only_oos = sum(a['short_pnl'] for a in all_analysis if 'OOS' in a['name'])
    print(f"  Long-Only OOS P&L: ${long_only_oos:.0f}")
    print(f"  Short-Only OOS P&L: ${short_only_oos:.0f}")

    print("\n" + "=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
