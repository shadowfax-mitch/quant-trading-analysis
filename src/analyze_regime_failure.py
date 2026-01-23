"""
Analyze regime characteristics to find a filter that predicts strategy failure.

Compare profitable periods (Train, OOS1, OOS3) vs failing periods (OOS2, OOS4)
to identify detectable conditions.
"""
import numpy as np
import pandas as pd
from pathlib import Path


def load_bars():
    """Load cached MNQ 5-min bars."""
    cache_file = Path('data/mnq_5min_bars_full.parquet')
    return pd.read_parquet(cache_file)


def compute_period_characteristics(bars: pd.DataFrame, name: str) -> dict:
    """Compute various market characteristics for a period."""

    # Basic stats
    returns = bars['close'].pct_change().dropna()

    # Volatility measures
    volatility = returns.std() * np.sqrt(288)  # Annualized (288 5-min bars per day)
    realized_vol_20 = returns.rolling(20).std().mean() * np.sqrt(288)

    # Trend measures
    total_return = (bars['close'].iloc[-1] / bars['close'].iloc[0] - 1) * 100

    # Price range
    high = bars['high'].max()
    low = bars['low'].min()
    range_pct = (high - low) / bars['close'].iloc[0] * 100

    # Directional bias
    up_bars = (bars['close'] > bars['open']).sum()
    down_bars = (bars['close'] < bars['open']).sum()
    up_ratio = up_bars / len(bars)

    # Mean reversion potential - how often price returns to mean
    ema_21 = bars['close'].ewm(span=21, adjust=False).mean()
    dist = (bars['close'] - ema_21) / ema_21
    dist_std = dist.rolling(21).std()
    zscore = dist / dist_std

    # Z-score characteristics
    zscore_mean = zscore.mean()
    zscore_std = zscore.std()
    extreme_long = (zscore < -3.5).sum()  # Entry opportunities
    extreme_short = (zscore > 3.5).sum()

    # Trend strength - how much time spent away from EMA
    above_ema = (bars['close'] > ema_21).mean()

    # Autocorrelation of returns (mean reversion indicator)
    autocorr_1 = returns.autocorr(lag=1)
    autocorr_5 = returns.autocorr(lag=5)

    # Average true range
    tr = pd.concat([
        bars['high'] - bars['low'],
        (bars['high'] - bars['close'].shift()).abs(),
        (bars['low'] - bars['close'].shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().mean()
    atr_pct = atr / bars['close'].mean() * 100

    return {
        'name': name,
        'bars': len(bars),
        'volatility': volatility,
        'realized_vol_20': realized_vol_20,
        'total_return': total_return,
        'range_pct': range_pct,
        'up_ratio': up_ratio,
        'zscore_mean': zscore_mean,
        'zscore_std': zscore_std,
        'extreme_long': extreme_long,
        'extreme_short': extreme_short,
        'above_ema_pct': above_ema * 100,
        'autocorr_1': autocorr_1,
        'autocorr_5': autocorr_5,
        'atr_pct': atr_pct,
    }


def main():
    print("=" * 100)
    print("REGIME ANALYSIS - IDENTIFYING FAILURE CONDITIONS")
    print("=" * 100)
    print()

    # Load data
    all_bars = load_bars()

    # Define periods with their P&L results
    periods = [
        ("Train (Jan-Feb)", "2025-01-01", "2025-02-28 23:59:59", 1593, "PROFIT"),
        ("OOS1 (Mar)", "2025-03-01", "2025-03-31 23:59:59", 468, "PROFIT"),
        ("OOS2 (Apr-Jun)", "2025-04-01", "2025-06-30 23:59:59", -6208, "FAIL"),
        ("OOS3 (Jul-Sep)", "2025-07-01", "2025-09-30 23:59:59", 207, "MARGINAL"),
        ("OOS4 (Oct-Jan)", "2025-10-01", "2026-01-14 23:59:59", -1019, "FAIL"),
    ]

    # Compute characteristics for each period
    print("1. Computing period characteristics...")
    results = []

    for name, start, end, pnl, status in periods:
        start_dt = pd.Timestamp(start, tz='UTC')
        end_dt = pd.Timestamp(end, tz='UTC')
        bars = all_bars[(all_bars['timestamp'] >= start_dt) & (all_bars['timestamp'] <= end_dt)].copy()

        chars = compute_period_characteristics(bars, name)
        chars['pnl'] = pnl
        chars['status'] = status
        results.append(chars)

    df = pd.DataFrame(results)

    # Display comparison table
    print("\n2. Period Characteristics Comparison")
    print("=" * 100)

    print(f"\n{'Period':<20} {'Status':<10} {'P&L':>10} {'Vol':>8} {'Return':>8} {'Range':>8} {'UpRatio':>8}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<20} {r['status']:<10} ${r['pnl']:>8} {r['volatility']:>7.1%} {r['total_return']:>7.1f}% {r['range_pct']:>7.1f}% {r['up_ratio']:>7.1%}")

    print(f"\n{'Period':<20} {'Z-Mean':>8} {'Z-Std':>8} {'ExtLong':>8} {'ExtShort':>8} {'AboveEMA':>10}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<20} {r['zscore_mean']:>8.3f} {r['zscore_std']:>8.2f} {r['extreme_long']:>8} {r['extreme_short']:>8} {r['above_ema_pct']:>9.1f}%")

    print(f"\n{'Period':<20} {'AutoCorr1':>10} {'AutoCorr5':>10} {'ATR%':>8}")
    print("-" * 60)
    for r in results:
        print(f"{r['name']:<20} {r['autocorr_1']:>10.4f} {r['autocorr_5']:>10.4f} {r['atr_pct']:>7.3f}%")

    # Analyze differences between profitable and failing periods
    print("\n" + "=" * 100)
    print("3. PROFITABLE vs FAILING PERIODS COMPARISON")
    print("=" * 100)

    profit_periods = [r for r in results if r['status'] == 'PROFIT']
    fail_periods = [r for r in results if r['status'] == 'FAIL']

    def avg(lst, key):
        return np.mean([r[key] for r in lst])

    metrics = ['volatility', 'total_return', 'range_pct', 'up_ratio', 'zscore_std',
               'extreme_long', 'extreme_short', 'above_ema_pct', 'autocorr_1', 'atr_pct']

    print(f"\n{'Metric':<20} {'Profitable':>15} {'Failing':>15} {'Difference':>15} {'Potential Filter?'}")
    print("-" * 80)

    for metric in metrics:
        profit_avg = avg(profit_periods, metric)
        fail_avg = avg(fail_periods, metric)
        diff = fail_avg - profit_avg

        # Check if this could be a filter
        if metric == 'volatility':
            pct_diff = (fail_avg / profit_avg - 1) * 100
            filter_note = "YES - higher vol in failures" if pct_diff > 20 else "Maybe"
        elif metric == 'total_return':
            filter_note = "YES - strong trends in failures" if abs(fail_avg) > abs(profit_avg) * 1.5 else "Maybe"
        elif metric == 'autocorr_1':
            filter_note = "YES - more trending" if fail_avg > profit_avg + 0.02 else "Maybe"
        elif metric == 'above_ema_pct':
            filter_note = "YES - directional bias" if abs(fail_avg - 50) > abs(profit_avg - 50) else "No"
        else:
            filter_note = ""

        if metric in ['volatility', 'up_ratio']:
            print(f"{metric:<20} {profit_avg:>14.1%} {fail_avg:>14.1%} {diff:>+14.1%} {filter_note}")
        elif metric in ['extreme_long', 'extreme_short']:
            print(f"{metric:<20} {profit_avg:>15.0f} {fail_avg:>15.0f} {diff:>+15.0f} {filter_note}")
        elif metric in ['autocorr_1']:
            print(f"{metric:<20} {profit_avg:>15.4f} {fail_avg:>15.4f} {diff:>+15.4f} {filter_note}")
        else:
            print(f"{metric:<20} {profit_avg:>15.2f} {fail_avg:>15.2f} {diff:>+15.2f} {filter_note}")

    # Key findings
    print("\n" + "=" * 100)
    print("4. KEY FINDINGS")
    print("=" * 100)

    # Trend analysis
    profit_trend = avg(profit_periods, 'total_return')
    fail_trend = avg(fail_periods, 'total_return')

    print(f"\nTREND DIRECTION:")
    print(f"  - Profitable periods avg return: {profit_trend:+.1f}%")
    print(f"  - Failing periods avg return: {fail_trend:+.1f}%")

    if abs(fail_trend) > abs(profit_trend) * 1.5:
        print(f"  --> FINDING: Failing periods have STRONGER TRENDS ({abs(fail_trend):.1f}% vs {abs(profit_trend):.1f}%)")
        print(f"  --> Mean reversion fails when market is trending!")

    # Volatility analysis
    profit_vol = avg(profit_periods, 'volatility')
    fail_vol = avg(fail_periods, 'volatility')

    print(f"\nVOLATILITY:")
    print(f"  - Profitable periods: {profit_vol:.1%}")
    print(f"  - Failing periods: {fail_vol:.1%}")

    # Autocorrelation (trending indicator)
    profit_ac = avg(profit_periods, 'autocorr_1')
    fail_ac = avg(fail_periods, 'autocorr_1')

    print(f"\nAUTOCORRELATION (trend persistence):")
    print(f"  - Profitable periods: {profit_ac:.4f}")
    print(f"  - Failing periods: {fail_ac:.4f}")

    if fail_ac > profit_ac:
        print(f"  --> FINDING: Higher autocorrelation in failing periods = more trending")

    # Directional bias
    profit_bias = avg(profit_periods, 'above_ema_pct')
    fail_bias = avg(fail_periods, 'above_ema_pct')

    print(f"\nDIRECTIONAL BIAS (% time above EMA):")
    print(f"  - Profitable periods: {profit_bias:.1f}%")
    print(f"  - Failing periods: {fail_bias:.1f}%")

    if abs(fail_bias - 50) > abs(profit_bias - 50):
        print(f"  --> FINDING: Failing periods have stronger directional bias away from 50%")

    # Suggest filters
    print("\n" + "=" * 100)
    print("5. SUGGESTED FILTERS TO TEST")
    print("=" * 100)

    print("""
    Based on the analysis, the following filters might help:

    1. TREND STRENGTH FILTER
       - Calculate rolling 20-bar return
       - If |return| > threshold, disable strategy
       - Rationale: Mean reversion fails in strong trends

    2. DIRECTIONAL BIAS FILTER
       - Calculate % of bars above EMA over rolling window
       - If < 40% or > 60%, disable strategy
       - Rationale: Strategy needs balanced mean-reverting behavior

    3. AUTOCORRELATION FILTER
       - Calculate rolling autocorrelation of returns
       - If autocorr > threshold, disable strategy
       - Rationale: High autocorr = trending market

    4. VOLATILITY REGIME FILTER
       - Track rolling volatility
       - May need to adjust thresholds based on vol regime
    """)

    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
