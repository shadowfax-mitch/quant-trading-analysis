"""
Analyze What Made March Different from Jan-Feb

Compare market characteristics, OFI behavior, and trade outcomes
between the losing (Jan-Feb) and winning (March) periods.
"""
import numpy as np
import pandas as pd
from scipy import stats


def main():
    print("=" * 80)
    print("REGIME ANALYSIS: WHY DID MARCH WORK?")
    print("=" * 80)

    # Load data
    print("\n1. Loading data...")
    df = pd.read_parquet('data/sprint_with_ofi.parquet')
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    print(f"   Total ticks: {len(df):,}")

    # Aggregate to 1-min bars
    df['bar'] = df['timestamp'].dt.floor('1min')
    df['buy_vol'] = np.where(df['side'] == 'A', df['volume'], 0)
    df['sell_vol'] = np.where(df['side'] == 'B', df['volume'], 0)

    bars = df.groupby('bar').agg({
        'last': ['first', 'max', 'min', 'last'],
        'bid': 'last',
        'ask': 'last',
        'buy_vol': 'sum',
        'sell_vol': 'sum',
        'volume': 'sum',
    }).reset_index()
    bars.columns = ['timestamp', 'open', 'high', 'low', 'close', 'bid', 'ask', 'buy_vol', 'sell_vol', 'volume']

    # Compute features
    bars['range'] = bars['high'] - bars['low']
    bars['range_pct'] = bars['range'] / bars['close'] * 100
    bars['return'] = bars['close'].pct_change()
    bars['abs_return'] = bars['return'].abs()
    bars['volatility'] = bars['return'].rolling(20).std()
    bars['spread'] = bars['ask'] - bars['bid']

    # OFI with window=10 (best config)
    window = 10
    roll_buy = bars['buy_vol'].rolling(window, min_periods=window).sum()
    roll_sell = bars['sell_vol'].rolling(window, min_periods=window).sum()
    bars['ofi'] = (roll_buy - roll_sell) / (roll_buy + roll_sell)
    bars['ofi'] = bars['ofi'].replace([np.inf, -np.inf], np.nan)
    bars['abs_ofi'] = bars['ofi'].abs()

    # Split periods
    bars['month'] = bars['timestamp'].dt.month
    bars['date'] = bars['timestamp'].dt.date

    jan_feb = bars[bars['month'].isin([1, 2])].copy()
    march = bars[bars['month'] == 3].copy()

    print(f"   Jan-Feb bars: {len(jan_feb):,}")
    print(f"   March bars: {len(march):,}")

    # =========================================================================
    # PRICE ACTION COMPARISON
    # =========================================================================
    print("\n" + "=" * 80)
    print("PRICE ACTION COMPARISON")
    print("=" * 80)

    def describe_period(data, name):
        valid = data.dropna()
        print(f"\n{name}:")
        print(f"  Bar Range (ticks):     Mean={data['range'].mean()/0.25:.1f}, Median={data['range'].median()/0.25:.1f}")
        print(f"  Volatility (20-bar):   Mean={data['volatility'].mean()*100:.4f}%, Max={data['volatility'].max()*100:.4f}%")
        print(f"  Abs Return per bar:    Mean={data['abs_return'].mean()*100:.4f}%")
        print(f"  Spread (ticks):        Mean={data['spread'].mean()/0.25:.2f}")
        print(f"  Volume per bar:        Mean={data['volume'].mean():.0f}")

    describe_period(jan_feb, "JAN-FEB (Losing Period)")
    describe_period(march, "MARCH (Winning Period)")

    # =========================================================================
    # OFI DISTRIBUTION COMPARISON
    # =========================================================================
    print("\n" + "=" * 80)
    print("OFI DISTRIBUTION COMPARISON")
    print("=" * 80)

    def ofi_stats(data, name):
        ofi = data['ofi'].dropna()
        abs_ofi = data['abs_ofi'].dropna()
        print(f"\n{name}:")
        print(f"  OFI Mean:              {ofi.mean():.4f}")
        print(f"  OFI Std:               {ofi.std():.4f}")
        print(f"  OFI Skew:              {ofi.skew():.4f}")
        print(f"  |OFI| Mean:            {abs_ofi.mean():.4f}")
        print(f"  |OFI| > 0.20:          {(abs_ofi > 0.20).sum():,} ({(abs_ofi > 0.20).mean()*100:.1f}%)")
        print(f"  |OFI| > 0.30:          {(abs_ofi > 0.30).sum():,} ({(abs_ofi > 0.30).mean()*100:.1f}%)")

    ofi_stats(jan_feb, "JAN-FEB")
    ofi_stats(march, "MARCH")

    # =========================================================================
    # SIGNAL QUALITY ANALYSIS
    # =========================================================================
    print("\n" + "=" * 80)
    print("SIGNAL QUALITY: WHAT HAPPENS AFTER EXTREME OFI?")
    print("=" * 80)

    def analyze_signal_quality(data, name, thresh=0.20, forward_bars=30):
        """Analyze price movement after OFI crosses threshold."""
        ofi = data['ofi'].values
        close = data['close'].values
        n = len(data)

        long_signals = []
        short_signals = []

        for i in range(window, n - forward_bars):
            if np.isnan(ofi[i]):
                continue

            if ofi[i] > thresh:
                # Long signal - measure forward return
                entry = close[i + 1] if i + 1 < n else close[i]
                exit = close[i + forward_bars] if i + forward_bars < n else close[-1]
                ret = (exit - entry) / entry * 100
                long_signals.append(ret)
            elif ofi[i] < -thresh:
                # Short signal - measure forward return (inverted)
                entry = close[i + 1] if i + 1 < n else close[i]
                exit = close[i + forward_bars] if i + forward_bars < n else close[-1]
                ret = (entry - exit) / entry * 100  # Profit if price goes down
                short_signals.append(ret)

        print(f"\n{name} (threshold={thresh}, hold={forward_bars} bars):")

        if long_signals:
            long_arr = np.array(long_signals)
            print(f"  LONG signals: {len(long_signals)}")
            print(f"    Mean return: {long_arr.mean():.4f}%")
            print(f"    Win rate:    {(long_arr > 0).mean()*100:.1f}%")
            print(f"    Best:        {long_arr.max():.4f}%")
            print(f"    Worst:       {long_arr.min():.4f}%")
        else:
            print(f"  LONG signals: 0")

        if short_signals:
            short_arr = np.array(short_signals)
            print(f"  SHORT signals: {len(short_signals)}")
            print(f"    Mean return: {short_arr.mean():.4f}%")
            print(f"    Win rate:    {(short_arr > 0).mean()*100:.1f}%")
            print(f"    Best:        {short_arr.max():.4f}%")
            print(f"    Worst:       {short_arr.min():.4f}%")
        else:
            print(f"  SHORT signals: 0")

        all_signals = long_signals + short_signals
        if all_signals:
            all_arr = np.array(all_signals)
            print(f"  COMBINED:")
            print(f"    Total signals: {len(all_signals)}")
            print(f"    Mean return:   {all_arr.mean():.4f}%")
            print(f"    Win rate:      {(all_arr > 0).mean()*100:.1f}%")

        return long_signals, short_signals

    jf_long, jf_short = analyze_signal_quality(jan_feb, "JAN-FEB")
    mar_long, mar_short = analyze_signal_quality(march, "MARCH")

    # =========================================================================
    # TREND vs MEAN-REVERSION ANALYSIS
    # =========================================================================
    print("\n" + "=" * 80)
    print("TREND VS MEAN-REVERSION BEHAVIOR")
    print("=" * 80)

    def trend_analysis(data, name):
        """Check if market was trending or mean-reverting."""
        returns = data['return'].dropna()

        # Autocorrelation of returns
        if len(returns) > 10:
            autocorr_1 = returns.autocorr(lag=1)
            autocorr_5 = returns.autocorr(lag=5)
        else:
            autocorr_1 = autocorr_5 = np.nan

        # Hurst exponent approximation (simplified)
        # H > 0.5 = trending, H < 0.5 = mean-reverting, H = 0.5 = random walk
        prices = data['close'].dropna().values
        if len(prices) > 100:
            lags = range(2, min(100, len(prices)//4))
            tau = []
            for lag in lags:
                tau.append(np.std(np.subtract(prices[lag:], prices[:-lag])))
            if len(tau) > 1:
                reg = np.polyfit(np.log(list(lags)), np.log(tau), 1)
                hurst = reg[0]
            else:
                hurst = np.nan
        else:
            hurst = np.nan

        # Daily range vs daily close-to-close
        data_copy = data.copy()
        data_copy['date'] = data_copy['timestamp'].dt.date
        daily = data_copy.groupby('date').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
        })
        daily['range'] = daily['high'] - daily['low']
        daily['move'] = (daily['close'] - daily['open']).abs()
        daily['efficiency'] = daily['move'] / daily['range']  # 1 = perfect trend, 0 = choppy

        print(f"\n{name}:")
        print(f"  Return autocorr (lag 1): {autocorr_1:.4f}")
        print(f"  Return autocorr (lag 5): {autocorr_5:.4f}")
        print(f"  Hurst exponent:          {hurst:.3f} ({'Trending' if hurst > 0.5 else 'Mean-Reverting' if hurst < 0.5 else 'Random'})")
        print(f"  Daily efficiency:        {daily['efficiency'].mean():.3f} (1=trend, 0=chop)")
        print(f"  Trading days:            {len(daily)}")

        return hurst, daily['efficiency'].mean()

    jf_hurst, jf_eff = trend_analysis(jan_feb, "JAN-FEB")
    mar_hurst, mar_eff = trend_analysis(march, "MARCH")

    # =========================================================================
    # TIME-OF-DAY ANALYSIS
    # =========================================================================
    print("\n" + "=" * 80)
    print("TIME-OF-DAY ANALYSIS")
    print("=" * 80)

    def time_analysis(data, name, thresh=0.20):
        """When do extreme OFI signals occur?"""
        data_copy = data.copy()
        data_copy['hour'] = data_copy['timestamp'].dt.hour
        data_copy['extreme_ofi'] = data_copy['abs_ofi'] > thresh

        hourly = data_copy.groupby('hour').agg({
            'extreme_ofi': ['sum', 'mean'],
            'abs_ofi': 'mean',
        })
        hourly.columns = ['signal_count', 'signal_pct', 'avg_abs_ofi']

        print(f"\n{name} - Signals by Hour (OFI > {thresh}):")
        top_hours = hourly.nlargest(5, 'signal_count')
        for hour, row in top_hours.iterrows():
            print(f"  {hour:02d}:00 - {int(row['signal_count']):>4} signals ({row['signal_pct']*100:.1f}%), Avg |OFI|={row['avg_abs_ofi']:.3f}")

    time_analysis(jan_feb, "JAN-FEB")
    time_analysis(march, "MARCH")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY: KEY DIFFERENCES")
    print("=" * 80)

    print("""
METRIC                          JAN-FEB         MARCH           IMPLICATION
--------------------------------------------------------------------------------""")

    # Compute key metrics for comparison
    jf_vol = jan_feb['volatility'].mean() * 100
    mar_vol = march['volatility'].mean() * 100
    jf_abs_ofi = jan_feb['abs_ofi'].mean()
    mar_abs_ofi = march['abs_ofi'].mean()
    jf_extreme = (jan_feb['abs_ofi'] > 0.20).mean() * 100
    mar_extreme = (march['abs_ofi'] > 0.20).mean() * 100

    jf_all = jf_long + jf_short
    mar_all = mar_long + mar_short
    jf_wr = np.mean([r > 0 for r in jf_all]) * 100 if jf_all else 0
    mar_wr = np.mean([r > 0 for r in mar_all]) * 100 if mar_all else 0

    print(f"Volatility (20-bar)         {jf_vol:.4f}%        {mar_vol:.4f}%        {'March higher' if mar_vol > jf_vol else 'Jan-Feb higher'}")
    print(f"Mean |OFI|                  {jf_abs_ofi:.4f}          {mar_abs_ofi:.4f}          {'March more extreme' if mar_abs_ofi > jf_abs_ofi else 'Similar'}")
    print(f"Extreme OFI freq (>0.20)    {jf_extreme:.1f}%           {mar_extreme:.1f}%           {'March more signals' if mar_extreme > jf_extreme else 'Jan-Feb more'}")
    print(f"Signal win rate             {jf_wr:.1f}%           {mar_wr:.1f}%           {'March much better' if mar_wr > jf_wr + 10 else 'Similar'}")
    print(f"Hurst exponent              {jf_hurst:.3f}           {mar_hurst:.3f}           {'March more trending' if mar_hurst > jf_hurst else 'March more MR'}")
    print(f"Daily efficiency            {jf_eff:.3f}           {mar_eff:.3f}           {'March cleaner trends' if mar_eff > jf_eff else 'Similar'}")

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    if mar_wr > jf_wr + 15:
        print("""
The key difference is SIGNAL QUALITY, not signal frequency.

In March, when OFI exceeded the threshold, the market followed through in the
predicted direction much more reliably. This could be due to:

1. HIGHER CONVICTION FLOW: March had more "real" institutional flow vs noise
2. LESS CHOPPY CONDITIONS: Daily efficiency was higher, meaning trends persisted
3. REGIME SHIFT: Market behavior fundamentally changed in March

RECOMMENDATION: Add a regime filter based on:
- Daily efficiency ratio (trade only when > threshold)
- Recent signal win rate (adaptive threshold)
- Volatility regime (may need different thresholds per regime)
""")
    else:
        print("""
The difference may be due to random variation or subtle regime differences.
More data is needed to determine if March was anomalous or represents a
recurring market state.
""")

    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
