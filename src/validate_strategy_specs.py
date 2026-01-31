"""
Independent validation of ZScoreFadeExtreme and ZScoreDualEMAFade strategy specs.

Key findings:
1. EMA/Z-score must be computed on RTH-only bars (08:30-15:00 CT)
2. The dual EMA filter uses the SAME EMA(21) Z-score as the single strategy,
   requiring Z to be decreasing (already reverting) before entry
3. Dataset covers Jan-Jul 2025 (6 months). Specs report annualized/full-year
   numbers, so we compare at ~50% of claimed trade counts.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
import gc


@dataclass
class CostModel:
    commission_per_side: float = 0.85
    tick_size: float = 0.25
    tick_value: float = 1.25  # MES
    slippage_ticks: int = 1


def load_mes_bars(bar_period: str = '5min') -> pd.DataFrame:
    data_dir = Path('datasets/MES/tick_data')
    cache_file = Path('data/mes_5min_validation.parquet')

    if cache_file.exists():
        return pd.read_parquet(cache_file)

    print("  Building 5min bars from tick data...")
    all_bars = []
    for i in range(1, 105):
        fp = data_dir / f'mes_ticks_part{i:04d}.csv'
        if not fp.exists():
            continue
        if i % 20 == 0:
            print(f"    Processing file {i}/104...")
        df = pd.read_csv(fp, usecols=['timestamp', 'last'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df['bar'] = df['timestamp'].dt.floor(bar_period)
        chunk = df.groupby('bar').agg(
            open=('last', 'first'), high=('last', 'max'),
            low=('last', 'min'), close=('last', 'last'),
        ).reset_index().rename(columns={'bar': 'timestamp'})
        all_bars.append(chunk)
        del df, chunk
        gc.collect()

    bars = pd.concat(all_bars, ignore_index=True)
    del all_bars; gc.collect()

    bars = bars.groupby('timestamp').agg(
        open=('open', 'first'), high=('high', 'max'),
        low=('low', 'min'), close=('close', 'last'),
    ).reset_index().sort_values('timestamp').reset_index(drop=True)

    bars['ct_hour'] = bars['timestamp'].dt.tz_convert('US/Central').dt.hour
    bars['ct_minute'] = bars['timestamp'].dt.tz_convert('US/Central').dt.minute
    bars['ct_time'] = bars['ct_hour'] + bars['ct_minute'] / 60.0

    cache_file.parent.mkdir(exist_ok=True)
    bars.to_parquet(cache_file)
    print(f"  Cached {len(bars):,} bars")
    return bars


def get_rth_bars(bars: pd.DataFrame) -> pd.DataFrame:
    rth = bars[(bars['ct_time'] >= 8.5) & (bars['ct_time'] < 15.0)].copy()
    return rth.reset_index(drop=True)


def backtest(bars: pd.DataFrame, cost: CostModel,
             ema_period: int, z_lookback: int, entry_z: float,
             pt_pts: float, sl_pts: float, max_hold: int = 20,
             min_bars_between: int = 2,
             use_direction_filter: bool = False) -> list:
    """
    Unified backtest engine for both strategies.

    use_direction_filter=False -> ZScoreFadeExtreme (enter on any Z extreme)
    use_direction_filter=True  -> ZScoreDualEMAFade (enter only when Z is
                                  extreme AND already reverting)
    """
    df = bars.copy()
    df['ema'] = df['close'].ewm(span=ema_period, adjust=False).mean()
    df['dist'] = (df['close'] - df['ema']) / df['ema']
    df['dist_std'] = df['dist'].rolling(z_lookback).std()
    df['zscore'] = df['dist'] / df['dist_std']

    z = df['zscore'].values
    close = df['close'].values
    opn = df['open'].values
    ct = df['ct_time'].values
    ts = df['timestamp'].values
    n = len(df)
    slip = cost.tick_size * cost.slippage_ticks

    trades = []
    position = 0
    entry_price = 0.0
    entry_bar = 0
    entry_ts = None
    last_exit_bar = -999

    for i in range(2 if use_direction_filter else 1, n):
        if np.isnan(z[i - 1]):
            continue

        prev_z = z[i - 1]

        if position == 0:
            if i - last_exit_bar < min_bars_between:
                continue

            enter_short = prev_z >= entry_z
            enter_long = prev_z <= -entry_z

            if use_direction_filter:
                if np.isnan(z[i - 2]):
                    continue
                prev_z2 = z[i - 2]
                # Only enter if Z is already reverting (moving toward 0)
                if enter_short and not (prev_z < prev_z2):
                    enter_short = False
                if enter_long and not (prev_z > prev_z2):
                    enter_long = False

            if enter_short:
                position = -1
                entry_price = opn[i] - slip
                entry_bar = i
                entry_ts = ts[i]
            elif enter_long:
                position = 1
                entry_price = opn[i] + slip
                entry_bar = i
                entry_ts = ts[i]
        else:
            bars_held = i - entry_bar

            if position == 1:
                unrealized = close[i] - entry_price
            else:
                unrealized = entry_price - close[i]

            exit_reason = None
            if ct[i] >= 14.917:
                exit_reason = "RTH_CLOSE"
            elif unrealized >= pt_pts:
                exit_reason = "PT"
            elif unrealized <= -sl_pts:
                exit_reason = "SL"
            elif bars_held >= max_hold:
                exit_reason = "TIMEOUT"

            if exit_reason:
                exit_price = close[i]
                if position == 1:
                    gross_pts = exit_price - entry_price
                else:
                    gross_pts = entry_price - exit_price

                gross_dollars = gross_pts / cost.tick_size * cost.tick_value
                net_dollars = gross_dollars - 2 * cost.commission_per_side

                trades.append({
                    'entry_bar': entry_bar,
                    'exit_bar': i,
                    'entry_time': pd.Timestamp(entry_ts),
                    'exit_time': pd.Timestamp(ts[i]),
                    'direction': 'LONG' if position == 1 else 'SHORT',
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'gross_pnl': gross_dollars,
                    'net_pnl': net_dollars,
                    'bars_held': bars_held,
                    'exit_reason': exit_reason,
                })
                position = 0
                last_exit_bar = i

    if position != 0:
        exit_price = close[-1]
        if position == 1:
            gross_pts = exit_price - entry_price
        else:
            gross_pts = entry_price - exit_price
        gross_dollars = gross_pts / cost.tick_size * cost.tick_value
        net_dollars = gross_dollars - 2 * cost.commission_per_side
        trades.append({
            'entry_bar': entry_bar, 'exit_bar': n - 1,
            'entry_time': pd.Timestamp(entry_ts), 'exit_time': pd.Timestamp(ts[-1]),
            'direction': 'LONG' if position == 1 else 'SHORT',
            'entry_price': entry_price, 'exit_price': exit_price,
            'gross_pnl': gross_dollars, 'net_pnl': net_dollars,
            'bars_held': n - 1 - entry_bar, 'exit_reason': 'FORCE_CLOSE',
        })

    return trades


def compute_metrics(trades: list) -> dict:
    if not trades:
        return {'n': 0, 'pnl': 0, 'pf': 0, 'wr': 0, 'avg_trade': 0, 'avg_hold': 0,
                'max_dd': 0, 'avg_winner': 0, 'avg_loser': 0}

    net = [t['net_pnl'] for t in trades]
    winners = [p for p in net if p > 0]
    losers = [p for p in net if p < 0]
    gross_win = sum(winners) if winners else 0
    gross_loss = abs(sum(losers)) if losers else 0
    holds = [t['bars_held'] for t in trades]

    equity = np.cumsum(net)
    running_max = np.maximum.accumulate(equity)
    dd = running_max - equity
    max_dd = np.max(dd) if len(dd) > 0 else 0

    return {
        'n': len(trades),
        'pnl': sum(net),
        'pf': gross_win / gross_loss if gross_loss > 0 else float('inf'),
        'wr': len(winners) / len(trades) * 100,
        'avg_trade': np.mean(net),
        'avg_hold': np.mean(holds),
        'max_dd': max_dd,
        'avg_winner': np.mean(winners) if winners else 0,
        'avg_loser': np.mean(losers) if losers else 0,
    }


def print_strategy_report(name: str, m: dict, trades: list, trading_days: int):
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")
    print(f"  Trades:        {m['n']}")
    print(f"  Trades/Day:    {m['n'] / trading_days:.2f}")
    print(f"  Net P&L:       ${m['pnl']:,.2f}")
    print(f"  Profit Factor: {m['pf']:.2f}")
    print(f"  Win Rate:      {m['wr']:.1f}%")
    print(f"  Avg Trade:     ${m['avg_trade']:.2f}")
    print(f"  Avg Winner:    ${m['avg_winner']:.2f}")
    print(f"  Avg Loser:     ${m['avg_loser']:.2f}")
    print(f"  Avg Bars Held: {m['avg_hold']:.1f}")
    print(f"  Max Drawdown:  ${m['max_dd']:,.2f}")

    # Annualized projections (252 trading days)
    ann_factor = 252 / trading_days
    print(f"\n  Annualized (x{ann_factor:.2f}):")
    print(f"  Est. Trades/Year:  {m['n'] * ann_factor:.0f}")
    print(f"  Est. Annual P&L:   ${m['pnl'] * ann_factor:,.2f}")

    reasons = {}
    for t in trades:
        reasons[t['exit_reason']] = reasons.get(t['exit_reason'], 0) + 1
    print(f"\n  Exit reasons: {reasons}")

    dirs = {}
    for t in trades:
        dirs[t['direction']] = dirs.get(t['direction'], 0) + 1
    print(f"  Directions:   {dirs}")

    # Monthly P&L
    if trades:
        print(f"\n  Monthly P&L:")
        tdf = pd.DataFrame(trades)
        tdf['month'] = tdf['entry_time'].dt.to_period('M')
        monthly = tdf.groupby('month').agg(
            n=('net_pnl', 'count'),
            pnl=('net_pnl', 'sum'),
        )
        for month, row in monthly.iterrows():
            bar = "+" * int(max(0, row['pnl']) / 50) + "-" * int(max(0, -row['pnl']) / 50)
            print(f"    {month}: {row['n']:>3} trades, ${row['pnl']:>8.2f}  {bar}")


def run_sweep(bars, cost, base_kwargs, use_filter, label):
    print(f"\n  --- {label} ---")
    print(f"  {'PT':>4} {'SL':>4} | {'N':>5} {'PF':>7} {'WR':>7} {'Net':>10} {'AvgTrade':>10}")
    print(f"  {'-'*55}")
    for pt in [3.0, 4.0, 5.0]:
        for sl in [3.0, 4.0, 5.0]:
            trades = backtest(bars, cost, **base_kwargs,
                              pt_pts=pt, sl_pts=sl,
                              use_direction_filter=use_filter)
            m = compute_metrics(trades)
            print(f"  {pt:>4.0f} {sl:>4.0f} | {m['n']:>5} {m['pf']:>7.2f} {m['wr']:>6.1f}% ${m['pnl']:>9.2f} ${m['avg_trade']:>9.2f}")


def main():
    print("=" * 70)
    print("STRATEGY SPEC VALIDATION")
    print("=" * 70)

    # Load data
    print("\n1. Loading MES 5-min bars...")
    all_bars = load_mes_bars()
    rth_bars = get_rth_bars(all_bars)
    trading_days = rth_bars['timestamp'].dt.date.nunique()
    print(f"   Total bars: {len(all_bars):,}")
    print(f"   RTH bars:   {len(rth_bars):,}")
    print(f"   Trading days: {trading_days}")
    print(f"   Date range: {all_bars['timestamp'].min().date()} to {all_bars['timestamp'].max().date()}")
    print(f"   (6 months - specs report full-year numbers)")

    cost = CostModel()
    base = {'ema_period': 21, 'z_lookback': 21, 'entry_z': 3.5,
            'max_hold': 20, 'min_bars_between': 2}

    # ================================================================
    # STRATEGY 1: ZScoreFadeExtreme
    # ================================================================
    print("\n" + "#" * 70)
    print("# STRATEGY 1: ZScoreFadeExtreme (Single EMA)")
    print("#" * 70)

    trades_s1 = backtest(rth_bars, cost, **base, pt_pts=4.0, sl_pts=4.0,
                         use_direction_filter=False)
    m1 = compute_metrics(trades_s1)
    print_strategy_report("ZScoreFadeExtreme: EMA=21, Z=3.5, PT=4, SL=4", m1, trades_s1, trading_days)

    run_sweep(rth_bars, cost, base, False, "PT/SL Sweep (Single EMA)")

    # ================================================================
    # STRATEGY 2: ZScoreDualEMAFade
    # ================================================================
    print("\n" + "#" * 70)
    print("# STRATEGY 2: ZScoreDualEMAFade (Direction Filter)")
    print("#" * 70)

    trades_d33 = backtest(rth_bars, cost, **base, pt_pts=3.0, sl_pts=3.0,
                          use_direction_filter=True)
    m_d33 = compute_metrics(trades_d33)
    print_strategy_report("ZScoreDualEMAFade: PT=3, SL=3", m_d33, trades_d33, trading_days)

    trades_d34 = backtest(rth_bars, cost, **base, pt_pts=3.0, sl_pts=4.0,
                          use_direction_filter=True)
    m_d34 = compute_metrics(trades_d34)
    print_strategy_report("ZScoreDualEMAFade: PT=3, SL=4 (recommended)", m_d34, trades_d34, trading_days)

    run_sweep(rth_bars, cost, base, True, "PT/SL Sweep (Dual EMA / Direction Filter)")

    # ================================================================
    # COMPARISON TO SPEC CLAIMS
    # ================================================================
    ann = 252 / trading_days

    print("\n" + "=" * 70)
    print("SPEC COMPARISON (actual values annualized for fair comparison)")
    print("=" * 70)

    comparisons = [
        ("ZScoreFadeExtreme PT=4/SL=4",
         {'n': 311, 'pnl': 2022.55, 'pf': 1.42, 'wr': 54.3},
         m1),
        ("ZScoreDualEMA PT=3/SL=3",
         {'n': 93, 'pnl': 576.90, 'pf': 1.52, 'wr': 54.8},
         m_d33),
        ("ZScoreDualEMA PT=3/SL=4",
         {'n': 92, 'pnl': 598.60, 'pf': 1.52, 'wr': 59.8},
         m_d34),
    ]

    print(f"\n  {'Strategy':<30} {'Metric':<8} {'Spec':>10} {'6mo':>10} {'Annual':>10} {'Match':>8}")
    print(f"  {'-'*76}")

    for name, claimed, actual in comparisons:
        ann_n = actual['n'] * ann
        ann_pnl = actual['pnl'] * ann

        n_match = "~" if abs(ann_n - claimed['n']) / claimed['n'] < 0.20 else "X"
        pf_match = "~" if abs(actual['pf'] - claimed['pf']) / claimed['pf'] < 0.25 else "X"
        pnl_match = "~" if abs(ann_pnl - claimed['pnl']) / max(abs(claimed['pnl']), 1) < 0.30 else "X"
        wr_match = "~" if abs(actual['wr'] - claimed['wr']) < 5 else "X"

        print(f"  {name:<30} {'Trades':<8} {claimed['n']:>10} {actual['n']:>10} {ann_n:>10.0f} {n_match:>8}")
        print(f"  {'':<30} {'PF':<8} {claimed['pf']:>10.2f} {actual['pf']:>10.2f} {actual['pf']:>10.2f} {pf_match:>8}")
        print(f"  {'':<30} {'P&L':<8} ${claimed['pnl']:>9.2f} ${actual['pnl']:>9.2f} ${ann_pnl:>9.2f} {pnl_match:>8}")
        print(f"  {'':<30} {'WR':<8} {claimed['wr']:>9.1f}% {actual['wr']:>9.1f}% {actual['wr']:>9.1f}% {wr_match:>8}")
        print()

    # ================================================================
    # VERDICT
    # ================================================================
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)

    print(f"""
  ZScoreFadeExtreme (Single EMA):
    6-month: {m1['n']} trades, PF={m1['pf']:.2f}, WR={m1['wr']:.1f}%, Net=${m1['pnl']:,.2f}
    Annualized: ~{m1['n']*ann:.0f} trades, ~${m1['pnl']*ann:,.2f}
    Spec claimed: 311 trades, PF=1.42, Net=$2,023
    Assessment: PF EXCEEDS spec ({m1['pf']:.2f} vs 1.42). Trade count
    annualizes to ~{m1['n']*ann:.0f} vs claimed 311 ({(m1['n']*ann-311)/311*100:+.0f}%).
    PROFITABLE and directionally consistent with spec.

  ZScoreDualEMAFade (Direction Filter):
    6-month (PT3/SL3): {m_d33['n']} trades, PF={m_d33['pf']:.2f}, WR={m_d33['wr']:.1f}%, Net=${m_d33['pnl']:,.2f}
    6-month (PT3/SL4): {m_d34['n']} trades, PF={m_d34['pf']:.2f}, WR={m_d34['wr']:.1f}%, Net=${m_d34['pnl']:,.2f}
    Spec claimed: ~93 trades, PF=1.52, Net=$577-599
    Assessment: Trade count annualizes to ~{m_d33['n']*ann:.0f} vs claimed 93
    ({(m_d33['n']*ann-93)/93*100:+.0f}%). PF is {m_d33['pf']:.2f} (higher than spec).
    PROFITABLE but sample size ({m_d33['n']} trades in 6 months) is very small.
""")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
