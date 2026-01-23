"""
Walk-Forward Validation for OFI Momentum Strategy

Tests the verified GO configurations on training data (Jan-Feb 2025)
to check for consistency and potential overfitting.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class BacktestConfig:
    commission_per_side: float = 0.85
    tick_size: float = 0.25
    tick_value: float = 1.25


def run_backtest(df: pd.DataFrame, ofi: np.ndarray, thresh: float,
                 cfg: BacktestConfig, exit_bars: int) -> list:
    """OFI Momentum with conservative execution model."""
    trades = []
    n = len(df)

    open_prices = df['open'].values
    position = 0
    entry_price = 0.0
    entry_bar = 0

    for i in range(1, n):
        if np.isnan(ofi[i-1]):
            continue

        prev_ofi = ofi[i - 1]

        if position == 0:
            if prev_ofi > thresh:
                position = 1
                entry_price = open_prices[i] + cfg.tick_size
                entry_bar = i
            elif prev_ofi < -thresh:
                position = -1
                entry_price = open_prices[i] - cfg.tick_size
                entry_bar = i
        else:
            hold_time = i - entry_bar
            should_exit = hold_time >= exit_bars

            if position == 1 and prev_ofi < -thresh:
                should_exit = True
            elif position == -1 and prev_ofi > thresh:
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
                    "gross_pnl": gross_pnl,
                    "net_pnl": net_pnl,
                    "ticks": gross_pnl / cfg.tick_value,
                    "bar": i,
                })
                position = 0

    # Force close
    if position != 0:
        if position == 1:
            exit_price = open_prices[-1] - cfg.tick_size
            gross_pnl = (exit_price - entry_price) / cfg.tick_size * cfg.tick_value
        else:
            exit_price = open_prices[-1] + cfg.tick_size
            gross_pnl = (entry_price - exit_price) / cfg.tick_size * cfg.tick_value

        net_pnl = gross_pnl - 2 * cfg.commission_per_side
        trades.append({"gross_pnl": gross_pnl, "net_pnl": net_pnl, "ticks": gross_pnl / cfg.tick_value, "bar": n-1})

    return trades


def compute_metrics(trades: list) -> dict:
    if not trades:
        return {"n": 0, "pnl": 0, "wr": 0, "ticks": 0, "pf_net": 0, "max_dd": 0}

    net_pnls = [t["net_pnl"] for t in trades]
    ticks_arr = [t["ticks"] for t in trades]

    wins = sum(1 for p in net_pnls if p > 0)
    net_profit = sum(p for p in net_pnls if p > 0)
    net_loss = abs(sum(p for p in net_pnls if p < 0))

    # Compute max drawdown
    cumulative = np.cumsum(net_pnls)
    peak = np.maximum.accumulate(cumulative)
    drawdown = peak - cumulative
    max_dd = np.max(drawdown) if len(drawdown) > 0 else 0

    return {
        "n": len(trades),
        "pnl": sum(net_pnls),
        "wr": wins / len(trades) * 100 if trades else 0,
        "ticks": np.mean(ticks_arr) if ticks_arr else 0,
        "pf_net": net_profit / net_loss if net_loss > 0 else float("inf"),
        "max_dd": max_dd,
    }


def check_gates(m: dict) -> bool:
    return m["pnl"] > 0 and m["pf_net"] >= 1.1 and m["ticks"] >= 1.0 and m["n"] >= 30


def main():
    print("=" * 85)
    print("WALK-FORWARD VALIDATION: OFI MOMENTUM")
    print("=" * 85)

    # Load data
    print("\n1. Loading tick data...")
    df = pd.read_parquet('data/sprint_with_ofi.parquet')
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    print(f"   Total ticks: {len(df):,}")

    # Aggregate to 1-min bars
    print("\n2. Aggregating to 1-minute bars...")
    df['bar'] = df['timestamp'].dt.floor('1min')
    df['buy_vol'] = np.where(df['side'] == 'A', df['volume'], 0)
    df['sell_vol'] = np.where(df['side'] == 'B', df['volume'], 0)

    bars = df.groupby('bar').agg({
        'last': ['first', 'last'],
        'bid': 'last',
        'ask': 'last',
        'buy_vol': 'sum',
        'sell_vol': 'sum',
    }).reset_index()
    bars.columns = ['timestamp', 'open', 'close', 'bid', 'ask', 'buy_vol', 'sell_vol']

    # Define periods - use end of day for proper split
    train_end = pd.Timestamp("2025-02-28 23:59:59", tz="UTC")
    test_start = pd.Timestamp("2025-03-01 00:00:00", tz="UTC")

    train_bars = bars[bars['timestamp'] <= train_end].copy()
    test_bars = bars[bars['timestamp'] >= test_start].copy()

    # Verify no gap
    total_check = len(train_bars) + len(test_bars)
    if total_check != len(bars):
        print(f"   WARNING: {len(bars) - total_check} bars in gap between periods")

    print(f"   Train bars: {len(train_bars):,} (Jan-Feb 2025)")
    print(f"   Test bars:  {len(test_bars):,} (Mar 2025)")

    cfg = BacktestConfig()

    # GO configurations to validate
    go_configs = [
        (10, 0.20, 30, "Primary"),
        (5, 0.30, 10, "Alternative"),
        (10, 0.20, 20, "GO #3"),
        (10, 0.20, 10, "GO #4"),
        (10, 0.20, 40, "GO #5"),
    ]

    print("\n" + "=" * 85)
    print("WALK-FORWARD RESULTS")
    print("=" * 85)

    results = []

    for window, thresh, exit_bars, label in go_configs:
        print(f"\n--- {label}: Window={window}, Thresh={thresh}, Exit={exit_bars} ---")

        # Compute OFI for full dataset
        roll_buy = bars['buy_vol'].rolling(window, min_periods=window).sum()
        roll_sell = bars['sell_vol'].rolling(window, min_periods=window).sum()
        total = roll_buy + roll_sell
        ofi_full = ((roll_buy - roll_sell) / total).replace([np.inf, -np.inf], np.nan).values

        # Train period - use first len(train_bars) OFI values
        train_ofi = ofi_full[:len(train_bars)]
        train_trades = run_backtest(train_bars, train_ofi, thresh, cfg, exit_bars)
        train_m = compute_metrics(train_trades)
        train_go = check_gates(train_m)

        # Test period - use LAST len(test_bars) OFI values (matches V2 approach)
        test_ofi = ofi_full[-len(test_bars):]
        test_trades = run_backtest(test_bars, test_ofi, thresh, cfg, exit_bars)
        test_m = compute_metrics(test_trades)
        test_go = check_gates(test_m)

        print(f"{'Period':<8} {'Trades':>7} {'Net P&L':>10} {'WR':>7} {'Ticks':>8} {'PF_Net':>8} {'MaxDD':>8} {'GO?':>5}")
        print("-" * 70)
        print(f"{'TRAIN':<8} {train_m['n']:>7} ${train_m['pnl']:>9.2f} {train_m['wr']:>6.1f}% "
              f"{train_m['ticks']:>8.2f} {train_m['pf_net']:>8.2f} ${train_m['max_dd']:>7.2f} "
              f"{'GO' if train_go else '':>5}")
        print(f"{'TEST':<8} {test_m['n']:>7} ${test_m['pnl']:>9.2f} {test_m['wr']:>6.1f}% "
              f"{test_m['ticks']:>8.2f} {test_m['pf_net']:>8.2f} ${test_m['max_dd']:>7.2f} "
              f"{'GO' if test_go else '':>5}")

        # Consistency check
        consistent = train_go and test_go
        if train_m['pnl'] > 0 and test_m['pnl'] > 0:
            consistency = "CONSISTENT (both profitable)"
        elif train_m['pnl'] < 0 and test_m['pnl'] > 0:
            consistency = "IMPROVED (train loss, test profit)"
        elif train_m['pnl'] > 0 and test_m['pnl'] < 0:
            consistency = "DEGRADED (train profit, test loss)"
        else:
            consistency = "BOTH LOSING"

        print(f"Status: {consistency}")

        results.append({
            "label": label,
            "window": window,
            "thresh": thresh,
            "exit": exit_bars,
            "train_pnl": train_m['pnl'],
            "test_pnl": test_m['pnl'],
            "train_go": train_go,
            "test_go": test_go,
            "train_n": train_m['n'],
            "test_n": test_m['n'],
            "train_pf": train_m['pf_net'],
            "test_pf": test_m['pf_net'],
        })

    # Monthly breakdown for best config
    print("\n" + "=" * 85)
    print("MONTHLY BREAKDOWN: Primary Config (W=10, T=0.20, E=30)")
    print("=" * 85)

    window, thresh, exit_bars = 10, 0.20, 30
    roll_buy = bars['buy_vol'].rolling(window, min_periods=window).sum()
    roll_sell = bars['sell_vol'].rolling(window, min_periods=window).sum()
    total = roll_buy + roll_sell
    ofi_full = ((roll_buy - roll_sell) / total).replace([np.inf, -np.inf], np.nan).values

    bars_copy = bars.copy()
    bars_copy['ofi'] = ofi_full
    bars_copy['month'] = bars_copy['timestamp'].dt.to_period('M')

    print(f"\n{'Month':<10} {'Trades':>7} {'Net P&L':>10} {'WR':>7} {'Ticks':>8} {'PF_Net':>8} {'GO?':>5}")
    print("-" * 60)

    monthly_results = []
    for month in bars_copy['month'].unique():
        month_bars = bars_copy[bars_copy['month'] == month].copy()
        if len(month_bars) < window:
            continue

        month_ofi = month_bars['ofi'].values
        trades = run_backtest(month_bars, month_ofi, thresh, cfg, exit_bars)
        m = compute_metrics(trades)
        go = check_gates(m)

        print(f"{str(month):<10} {m['n']:>7} ${m['pnl']:>9.2f} {m['wr']:>6.1f}% "
              f"{m['ticks']:>8.2f} {m['pf_net']:>8.2f} {'GO' if go else '':>5}")

        monthly_results.append({"month": str(month), **m, "go": go})

    # Summary
    print("\n" + "=" * 85)
    print("VALIDATION SUMMARY")
    print("=" * 85)

    print("\n| Config | Train P&L | Test P&L | Train GO | Test GO | Consistent |")
    print("|--------|-----------|----------|----------|---------|------------|")
    for r in results:
        consistent = "Yes" if (r['train_pnl'] > 0 and r['test_pnl'] > 0) else "No"
        print(f"| {r['label']:<14} | ${r['train_pnl']:>8.2f} | ${r['test_pnl']:>7.2f} | "
              f"{'GO' if r['train_go'] else 'NO':>8} | {'GO' if r['test_go'] else 'NO':>7} | {consistent:>10} |")

    # Final assessment
    print("\n" + "=" * 85)
    print("CONCLUSION")
    print("=" * 85)

    consistent_configs = [r for r in results if r['train_pnl'] > 0 and r['test_pnl'] > 0]
    both_go = [r for r in results if r['train_go'] and r['test_go']]

    print(f"\nConfigs profitable in BOTH periods: {len(consistent_configs)}/{len(results)}")
    print(f"Configs achieving GO in BOTH periods: {len(both_go)}/{len(results)}")

    if len(consistent_configs) >= 3:
        print("\nSTATUS: VALIDATED - Strategy shows consistency across train/test periods")
    elif len(consistent_configs) >= 1:
        print("\nSTATUS: PARTIALLY VALIDATED - Some configs consistent, proceed with caution")
    else:
        print("\nSTATUS: NOT VALIDATED - Strategy may be overfit to test period")

    print("=" * 85)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
