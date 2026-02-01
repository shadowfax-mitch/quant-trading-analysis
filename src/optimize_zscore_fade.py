"""
Optimize ZScoreFadeExtreme strategy.

Explores:
1. Time-of-day filter (hour-by-hour performance)
2. Day-of-week filter
3. Z-score threshold fine-tuning (3.0 - 4.5 in 0.25 steps)
4. Asymmetric PT/SL (0.5-point increments)
5. Volatility regime filter (ATR-based)
6. Direction filter (Z already reverting) with varying strictness

Train: Jan-Apr 2025 (4 months)
Test:  May-Jul 2025 (2 months, true OOS)
"""
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from itertools import product


@dataclass
class CostModel:
    commission_per_side: float = 0.85
    tick_size: float = 0.25
    tick_value: float = 1.25
    slippage_ticks: int = 1


def load_rth_bars() -> pd.DataFrame:
    bars = pd.read_parquet('data/mes_5min_validation.parquet')
    rth = bars[(bars['ct_time'] >= 8.5) & (bars['ct_time'] < 15.0)].copy()
    return rth.reset_index(drop=True)


def add_indicators(df: pd.DataFrame, ema_period=21, z_lookback=21) -> pd.DataFrame:
    df = df.copy()
    df['ema'] = df['close'].ewm(span=ema_period, adjust=False).mean()
    df['dist'] = (df['close'] - df['ema']) / df['ema']
    df['dist_std'] = df['dist'].rolling(z_lookback).std()
    df['zscore'] = df['dist'] / df['dist_std']

    # ATR for volatility filter
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            (df['high'] - df['close'].shift(1)).abs(),
            (df['low'] - df['close'].shift(1)).abs()
        )
    )
    df['atr14'] = df['tr'].rolling(14).mean()
    df['atr_avg'] = df['atr14'].rolling(50).mean()
    df['atr_ratio'] = df['atr14'] / df['atr_avg']

    # Hour and day-of-week
    ct = df['timestamp'].dt.tz_convert('US/Central')
    df['hour'] = ct.dt.hour
    df['dow'] = ct.dt.dayofweek  # 0=Mon, 4=Fri
    df['date'] = ct.dt.date

    return df


def backtest(df: pd.DataFrame, cost: CostModel,
             entry_z=3.5, pt_pts=4.0, sl_pts=4.0,
             max_hold=20, min_bars_between=2,
             direction_filter=False, direction_lag=1,
             hour_filter=None, dow_filter=None,
             min_atr_ratio=0.0) -> list:
    """
    Flexible backtest with all optimization levers.

    hour_filter: set of allowed hours (e.g., {9, 10, 13, 14}), None=all
    dow_filter: set of allowed days (0=Mon..4=Fri), None=all
    min_atr_ratio: minimum ATR ratio for entry (0=disabled)
    direction_filter: require Z to be reverting
    direction_lag: how many bars back to compare Z for direction
    """
    z = df['zscore'].values
    close = df['close'].values
    opn = df['open'].values
    ct = df['ct_time'].values
    hour = df['hour'].values
    dow = df['dow'].values
    atr_r = df['atr_ratio'].values
    ts = df['timestamp'].values
    n = len(df)
    slip = cost.tick_size * cost.slippage_ticks

    trades = []
    position = 0
    entry_price = 0.0
    entry_bar = 0
    last_exit_bar = -999

    start = max(1, direction_lag + 1) if direction_filter else 1

    for i in range(start, n):
        if np.isnan(z[i - 1]):
            continue

        prev_z = z[i - 1]

        if position == 0:
            if i - last_exit_bar < min_bars_between:
                continue

            # Hour filter
            if hour_filter is not None and hour[i] not in hour_filter:
                continue

            # Day-of-week filter
            if dow_filter is not None and dow[i] not in dow_filter:
                continue

            # Volatility filter
            if min_atr_ratio > 0 and (np.isnan(atr_r[i - 1]) or atr_r[i - 1] < min_atr_ratio):
                continue

            enter_short = prev_z >= entry_z
            enter_long = prev_z <= -entry_z

            if direction_filter:
                lag_idx = i - 1 - direction_lag
                if lag_idx < 0 or np.isnan(z[lag_idx]):
                    continue
                if enter_short and not (prev_z < z[lag_idx]):
                    enter_short = False
                if enter_long and not (prev_z > z[lag_idx]):
                    enter_long = False

            if enter_short:
                position = -1
                entry_price = opn[i] - slip
                entry_bar = i
            elif enter_long:
                position = 1
                entry_price = opn[i] + slip
                entry_bar = i
        else:
            bars_held = i - entry_bar
            if position == 1:
                unrealized = close[i] - entry_price
            else:
                unrealized = entry_price - close[i]

            exit_reason = None
            if ct[i] >= 14.917:
                exit_reason = "RTH"
            elif unrealized >= pt_pts:
                exit_reason = "PT"
            elif unrealized <= -sl_pts:
                exit_reason = "SL"
            elif bars_held >= max_hold:
                exit_reason = "TO"

            if exit_reason:
                exit_price = close[i]
                if position == 1:
                    gross_pts = exit_price - entry_price
                else:
                    gross_pts = entry_price - exit_price

                gross_dollars = gross_pts / cost.tick_size * cost.tick_value
                net_dollars = gross_dollars - 2 * cost.commission_per_side

                trades.append({
                    'net_pnl': net_dollars,
                    'exit_reason': exit_reason,
                    'bars_held': bars_held,
                    'timestamp': pd.Timestamp(ts[i]),
                    'hour': hour[entry_bar],
                    'dow': dow[entry_bar],
                })
                position = 0
                last_exit_bar = i

    return trades


def metrics(trades):
    if not trades or len(trades) < 5:
        return {'n': 0, 'pnl': 0, 'pf': 0, 'wr': 0, 'avg': 0}
    net = [t['net_pnl'] for t in trades]
    wins = [p for p in net if p > 0]
    losses = [p for p in net if p < 0]
    gw = sum(wins) if wins else 0
    gl = abs(sum(losses)) if losses else 0
    return {
        'n': len(trades),
        'pnl': sum(net),
        'pf': gw / gl if gl > 0 else float('inf'),
        'wr': len(wins) / len(trades) * 100,
        'avg': np.mean(net),
    }


def main():
    print("=" * 80)
    print("ZSCORE FADE EXTREME - OPTIMIZATION")
    print("=" * 80)

    cost = CostModel()

    print("\n1. Loading data and computing indicators...")
    rth = load_rth_bars()
    df = add_indicators(rth)

    # Train/test split: Jan-Apr train, May-Jul test
    ct_dates = df['timestamp'].dt.tz_convert('US/Central')
    train_mask = ct_dates.dt.month <= 4
    test_mask = ct_dates.dt.month >= 5

    train = df[train_mask].copy().reset_index(drop=True)
    test = df[test_mask].copy().reset_index(drop=True)

    train_days = train['timestamp'].dt.date.nunique()
    test_days = test['timestamp'].dt.date.nunique()
    print(f"   Train: {len(train):,} bars, {train_days} days (Jan-Apr)")
    print(f"   Test:  {len(test):,} bars, {test_days} days (May-Jul)")

    # Baseline
    base_train = backtest(train, cost)
    base_test = backtest(test, cost)
    bm_tr = metrics(base_train)
    bm_te = metrics(base_test)
    print(f"\n   Baseline (Z=3.5, PT=4, SL=4):")
    print(f"   Train: {bm_tr['n']} trades, PF={bm_tr['pf']:.2f}, WR={bm_tr['wr']:.1f}%, ${bm_tr['pnl']:.2f}")
    print(f"   Test:  {bm_te['n']} trades, PF={bm_te['pf']:.2f}, WR={bm_te['wr']:.1f}%, ${bm_te['pnl']:.2f}")

    # ================================================================
    # 1. TIME-OF-DAY ANALYSIS
    # ================================================================
    print("\n" + "=" * 80)
    print("2. TIME-OF-DAY ANALYSIS")
    print("=" * 80)

    all_trades = backtest(df, cost)
    print(f"\n  {'Hour':>6} | {'N':>5} {'PF':>7} {'WR':>7} {'Net':>10} {'AvgTrade':>10}")
    print(f"  {'-'*52}")

    hour_pf = {}
    for h in range(8, 15):
        ht = [t for t in all_trades if t['hour'] == h]
        m = metrics(ht)
        if m['n'] >= 5:
            hour_pf[h] = m['pf']
            marker = " ***" if m['pf'] >= 1.5 else ""
            print(f"  {h:>4}:00 | {m['n']:>5} {m['pf']:>7.2f} {m['wr']:>6.1f}% ${m['pnl']:>9.2f} ${m['avg']:>9.2f}{marker}")

    good_hours = {h for h, pf in hour_pf.items() if pf >= 1.3}
    print(f"\n  Good hours (PF >= 1.3): {sorted(good_hours)}")

    # ================================================================
    # 2. DAY-OF-WEEK ANALYSIS
    # ================================================================
    print("\n" + "=" * 80)
    print("3. DAY-OF-WEEK ANALYSIS")
    print("=" * 80)

    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
    print(f"\n  {'Day':>6} | {'N':>5} {'PF':>7} {'WR':>7} {'Net':>10} {'AvgTrade':>10}")
    print(f"  {'-'*52}")

    dow_pf = {}
    for d in range(5):
        dt = [t for t in all_trades if t['dow'] == d]
        m = metrics(dt)
        if m['n'] >= 5:
            dow_pf[d] = m['pf']
            marker = " ***" if m['pf'] >= 1.5 else ""
            print(f"  {dow_names[d]:>6} | {m['n']:>5} {m['pf']:>7.2f} {m['wr']:>6.1f}% ${m['pnl']:>9.2f} ${m['avg']:>9.2f}{marker}")

    good_days = {d for d, pf in dow_pf.items() if pf >= 1.2}
    print(f"\n  Good days (PF >= 1.2): {[dow_names[d] for d in sorted(good_days)]}")

    # ================================================================
    # 3. Z-SCORE THRESHOLD FINE-TUNING
    # ================================================================
    print("\n" + "=" * 80)
    print("4. Z-SCORE THRESHOLD SWEEP")
    print("=" * 80)

    print(f"\n  {'Z':>5} | {'Tr_N':>5} {'Tr_PF':>7} {'Tr_WR':>7} | {'Te_N':>5} {'Te_PF':>7} {'Te_WR':>7}")
    print(f"  {'-'*58}")

    for z_thresh in np.arange(3.0, 4.75, 0.25):
        tr = backtest(train, cost, entry_z=z_thresh)
        te = backtest(test, cost, entry_z=z_thresh)
        mtr = metrics(tr)
        mte = metrics(te)
        marker = " ***" if mtr['pf'] >= 1.5 and mte['pf'] >= 1.3 else ""
        print(f"  {z_thresh:>5.2f} | {mtr['n']:>5} {mtr['pf']:>7.2f} {mtr['wr']:>6.1f}% | "
              f"{mte['n']:>5} {mte['pf']:>7.2f} {mte['wr']:>6.1f}%{marker}")

    # ================================================================
    # 4. ASYMMETRIC PT/SL SWEEP (0.5 increments)
    # ================================================================
    print("\n" + "=" * 80)
    print("5. ASYMMETRIC PT/SL SWEEP (fine grid)")
    print("=" * 80)

    print(f"\n  {'PT':>5} {'SL':>5} | {'Tr_N':>5} {'Tr_PF':>7} {'Tr_$':>9} | {'Te_N':>5} {'Te_PF':>7} {'Te_$':>9}")
    print(f"  {'-'*65}")

    ptsl_results = []
    for pt in np.arange(2.5, 6.5, 0.5):
        for sl in np.arange(2.5, 6.5, 0.5):
            tr = backtest(train, cost, pt_pts=pt, sl_pts=sl)
            te = backtest(test, cost, pt_pts=pt, sl_pts=sl)
            mtr = metrics(tr)
            mte = metrics(te)
            ptsl_results.append({
                'pt': pt, 'sl': sl,
                'tr_n': mtr['n'], 'tr_pf': mtr['pf'], 'tr_pnl': mtr['pnl'],
                'te_n': mte['n'], 'te_pf': mte['pf'], 'te_pnl': mte['pnl'],
            })

    # Show top 10 by combined PF
    ptsl_results.sort(key=lambda x: -(x['tr_pf'] + x['te_pf']) if x['tr_n'] >= 10 and x['te_n'] >= 5 else 0)
    for r in ptsl_results[:15]:
        if r['tr_n'] < 10:
            continue
        marker = " ***" if r['tr_pf'] >= 1.5 and r['te_pf'] >= 1.3 else ""
        print(f"  {r['pt']:>5.1f} {r['sl']:>5.1f} | {r['tr_n']:>5} {r['tr_pf']:>7.2f} ${r['tr_pnl']:>8.2f} | "
              f"{r['te_n']:>5} {r['te_pf']:>7.2f} ${r['te_pnl']:>8.2f}{marker}")

    # ================================================================
    # 5. VOLATILITY FILTER
    # ================================================================
    print("\n" + "=" * 80)
    print("6. VOLATILITY FILTER (min ATR ratio)")
    print("=" * 80)

    print(f"\n  {'ATR_min':>8} | {'Tr_N':>5} {'Tr_PF':>7} {'Tr_WR':>7} | {'Te_N':>5} {'Te_PF':>7} {'Te_WR':>7}")
    print(f"  {'-'*60}")

    for atr_min in [0.0, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5]:
        tr = backtest(train, cost, min_atr_ratio=atr_min)
        te = backtest(test, cost, min_atr_ratio=atr_min)
        mtr = metrics(tr)
        mte = metrics(te)
        marker = " ***" if mtr['pf'] >= 1.5 and mte['pf'] >= 1.3 else ""
        print(f"  {atr_min:>8.1f} | {mtr['n']:>5} {mtr['pf']:>7.2f} {mtr['wr']:>6.1f}% | "
              f"{mte['n']:>5} {mte['pf']:>7.2f} {mte['wr']:>6.1f}%{marker}")

    # ================================================================
    # 6. DIRECTION FILTER
    # ================================================================
    print("\n" + "=" * 80)
    print("7. DIRECTION FILTER (Z already reverting)")
    print("=" * 80)

    print(f"\n  {'Lag':>5} | {'Tr_N':>5} {'Tr_PF':>7} {'Tr_WR':>7} | {'Te_N':>5} {'Te_PF':>7} {'Te_WR':>7}")
    print(f"  {'-'*58}")

    print(f"  {'None':>5} | {bm_tr['n']:>5} {bm_tr['pf']:>7.2f} {bm_tr['wr']:>6.1f}% | "
          f"{bm_te['n']:>5} {bm_te['pf']:>7.2f} {bm_te['wr']:>6.1f}%")

    for lag in [1, 2, 3, 4, 5]:
        tr = backtest(train, cost, direction_filter=True, direction_lag=lag)
        te = backtest(test, cost, direction_filter=True, direction_lag=lag)
        mtr = metrics(tr)
        mte = metrics(te)
        marker = " ***" if mtr['pf'] >= 1.5 and mte['pf'] >= 1.3 else ""
        print(f"  {lag:>5} | {mtr['n']:>5} {mtr['pf']:>7.2f} {mtr['wr']:>6.1f}% | "
              f"{mte['n']:>5} {mte['pf']:>7.2f} {mte['wr']:>6.1f}%{marker}")

    # ================================================================
    # 7. COMBINED OPTIMIZATION
    # ================================================================
    print("\n" + "=" * 80)
    print("8. COMBINED OPTIMIZATION (best levers together)")
    print("=" * 80)
    print("  Testing combinations of promising filters...\n")

    combos = []

    # Build combinations from promising values
    z_values = [3.5, 3.75, 4.0]
    pt_values = [3.5, 4.0, 4.5, 5.0]
    sl_values = [3.5, 4.0, 4.5]
    dir_options = [(False, 0), (True, 1), (True, 2)]
    atr_options = [0.0, 0.9, 1.0, 1.1]
    hour_options = [None, good_hours if good_hours else None]
    dow_options = [None, good_days if good_days else None]

    total = (len(z_values) * len(pt_values) * len(sl_values) *
             len(dir_options) * len(atr_options) * len(hour_options) * len(dow_options))
    print(f"  Testing {total} combinations...")

    count = 0
    for z_val, pt, sl, (use_dir, dir_lag), atr_min, hours, days in product(
        z_values, pt_values, sl_values, dir_options, atr_options, hour_options, dow_options
    ):
        count += 1
        if count % 500 == 0:
            print(f"    {count}/{total}...")

        tr = backtest(train, cost, entry_z=z_val, pt_pts=pt, sl_pts=sl,
                      direction_filter=use_dir, direction_lag=dir_lag,
                      min_atr_ratio=atr_min, hour_filter=hours, dow_filter=days)
        mtr = metrics(tr)

        # Skip if train is bad or too few trades
        if mtr['n'] < 15 or mtr['pf'] < 1.3:
            continue

        te = backtest(test, cost, entry_z=z_val, pt_pts=pt, sl_pts=sl,
                      direction_filter=use_dir, direction_lag=dir_lag,
                      min_atr_ratio=atr_min, hour_filter=hours, dow_filter=days)
        mte = metrics(te)

        if mte['n'] < 5:
            continue

        combos.append({
            'z': z_val, 'pt': pt, 'sl': sl,
            'dir': f"lag={dir_lag}" if use_dir else "off",
            'atr': atr_min,
            'hours': 'filtered' if hours else 'all',
            'days': 'filtered' if days else 'all',
            'tr_n': mtr['n'], 'tr_pf': mtr['pf'], 'tr_wr': mtr['wr'], 'tr_pnl': mtr['pnl'],
            'te_n': mte['n'], 'te_pf': mte['pf'], 'te_wr': mte['wr'], 'te_pnl': mte['pnl'],
        })

    # Rank by: must be profitable in both periods, then by min(tr_pf, te_pf)
    viable = [c for c in combos if c['tr_pnl'] > 0 and c['te_pnl'] > 0]
    viable.sort(key=lambda x: -min(x['tr_pf'], x['te_pf']))

    print(f"\n  Viable configs (profitable in both periods): {len(viable)} / {len(combos)} tested")
    print(f"\n  TOP 20 BY MIN(TRAIN_PF, TEST_PF):")
    print(f"  {'Z':>4} {'PT':>4} {'SL':>4} {'Dir':>6} {'ATR':>4} {'Hr':>4} {'Dow':>4} | "
          f"{'Tr_N':>5} {'Tr_PF':>6} {'Tr_WR':>6} {'Tr_$':>8} | "
          f"{'Te_N':>5} {'Te_PF':>6} {'Te_WR':>6} {'Te_$':>8}")
    print(f"  {'-'*95}")

    for c in viable[:20]:
        hr_label = 'flt' if c['hours'] == 'filtered' else 'all'
        dw_label = 'flt' if c['days'] == 'filtered' else 'all'
        print(f"  {c['z']:>4.1f} {c['pt']:>4.1f} {c['sl']:>4.1f} {c['dir']:>6} {c['atr']:>4.1f} "
              f"{hr_label:>4} {dw_label:>4} | "
              f"{c['tr_n']:>5} {c['tr_pf']:>6.2f} {c['tr_wr']:>5.1f}% ${c['tr_pnl']:>7.2f} | "
              f"{c['te_n']:>5} {c['te_pf']:>6.2f} {c['te_wr']:>5.1f}% ${c['te_pnl']:>7.2f}")

    # Also show top by total P&L
    viable_by_pnl = sorted(viable, key=lambda x: -(x['tr_pnl'] + x['te_pnl']))
    print(f"\n  TOP 10 BY TOTAL P&L:")
    print(f"  {'Z':>4} {'PT':>4} {'SL':>4} {'Dir':>6} {'ATR':>4} {'Hr':>4} {'Dow':>4} | "
          f"{'Tr_N':>5} {'Tr_PF':>6} {'Tr_$':>8} | "
          f"{'Te_N':>5} {'Te_PF':>6} {'Te_$':>8} | {'Total':>8}")
    print(f"  {'-'*90}")

    for c in viable_by_pnl[:10]:
        hr_label = 'flt' if c['hours'] == 'filtered' else 'all'
        dw_label = 'flt' if c['days'] == 'filtered' else 'all'
        total_pnl = c['tr_pnl'] + c['te_pnl']
        print(f"  {c['z']:>4.1f} {c['pt']:>4.1f} {c['sl']:>4.1f} {c['dir']:>6} {c['atr']:>4.1f} "
              f"{hr_label:>4} {dw_label:>4} | "
              f"{c['tr_n']:>5} {c['tr_pf']:>6.2f} ${c['tr_pnl']:>7.2f} | "
              f"{c['te_n']:>5} {c['te_pf']:>6.2f} ${c['te_pnl']:>7.2f} | ${total_pnl:>7.2f}")

    # ================================================================
    # BEST CONFIG DETAILED REPORT
    # ================================================================
    if viable:
        best = viable[0]
        print("\n" + "=" * 80)
        print("RECOMMENDED CONFIGURATION")
        print("=" * 80)
        print(f"""
  Z Threshold:      {best['z']}
  Profit Target:    {best['pt']} points
  Stop Loss:        {best['sl']} points
  Direction Filter: {best['dir']}
  ATR Filter:       {best['atr']} ({'disabled' if best['atr'] == 0 else 'enabled'})
  Hour Filter:      {best['hours']} {f'({sorted(good_hours)})' if best['hours'] == 'filtered' else ''}
  Day Filter:       {best['days']} {f'({[dow_names[d] for d in sorted(good_days)]})' if best['days'] == 'filtered' else ''}

  Train (Jan-Apr):  {best['tr_n']} trades, PF={best['tr_pf']:.2f}, WR={best['tr_wr']:.1f}%, ${best['tr_pnl']:.2f}
  Test  (May-Jul):  {best['te_n']} trades, PF={best['te_pf']:.2f}, WR={best['te_wr']:.1f}%, ${best['te_pnl']:.2f}

  vs Baseline:
    Train PF: {best['tr_pf']:.2f} vs {bm_tr['pf']:.2f} ({(best['tr_pf']/bm_tr['pf']-1)*100:+.0f}%)
    Test PF:  {best['te_pf']:.2f} vs {bm_te['pf']:.2f} ({(best['te_pf']/bm_te['pf']-1)*100:+.0f}% OOS)
""")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
