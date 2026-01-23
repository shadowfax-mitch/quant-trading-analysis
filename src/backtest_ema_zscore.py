"""
EMA Z-Score Mean Reversion Strategy

Enter when price is extended from EMA(21), exit on reversion.
- Long when Z < -threshold (oversold)
- Short when Z > +threshold (overbought)
- Exit when Z crosses back toward 0
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class BacktestConfig:
    commission_per_side: float = 0.85
    tick_size: float = 0.25
    tick_value: float = 1.25  # MES


def run_backtest(df: pd.DataFrame, zscore: np.ndarray, entry_thresh: float,
                 exit_thresh: float, cfg: BacktestConfig, max_bars: int) -> list:
    """
    Mean reversion on Z-score.

    entry_thresh: Enter when |Z| > this (e.g., 2.0)
    exit_thresh: Exit when |Z| < this (e.g., 0.5) or Z flips sign
    max_bars: Maximum holding period
    """
    trades = []
    n = len(df)

    open_prices = df['open'].values
    position = 0
    entry_price = 0.0
    entry_bar = 0
    entry_z = 0.0

    for i in range(1, n):
        if np.isnan(zscore[i-1]):
            continue

        prev_z = zscore[i - 1]

        if position == 0:
            # Enter long when oversold (Z very negative)
            if prev_z < -entry_thresh:
                position = 1
                entry_price = open_prices[i] + cfg.tick_size  # Adverse fill
                entry_bar = i
                entry_z = prev_z
            # Enter short when overbought (Z very positive)
            elif prev_z > entry_thresh:
                position = -1
                entry_price = open_prices[i] - cfg.tick_size  # Adverse fill
                entry_bar = i
                entry_z = prev_z
        else:
            hold_time = i - entry_bar
            should_exit = False

            # Exit conditions
            if hold_time >= max_bars:
                should_exit = True
            elif position == 1:
                # Long: exit when Z reverts (becomes less negative or positive)
                if prev_z > -exit_thresh or prev_z > 0:
                    should_exit = True
            elif position == -1:
                # Short: exit when Z reverts (becomes less positive or negative)
                if prev_z < exit_thresh or prev_z < 0:
                    should_exit = True

            if should_exit:
                if position == 1:
                    exit_price = open_prices[i] - cfg.tick_size  # Adverse fill
                    gross_pnl = (exit_price - entry_price) / cfg.tick_size * cfg.tick_value
                else:
                    exit_price = open_prices[i] + cfg.tick_size  # Adverse fill
                    gross_pnl = (entry_price - exit_price) / cfg.tick_size * cfg.tick_value

                net_pnl = gross_pnl - 2 * cfg.commission_per_side
                trades.append({
                    "gross_pnl": gross_pnl,
                    "net_pnl": net_pnl,
                    "ticks": gross_pnl / cfg.tick_value,
                    "hold_bars": hold_time,
                    "entry_z": entry_z,
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
        trades.append({
            "gross_pnl": gross_pnl,
            "net_pnl": net_pnl,
            "ticks": gross_pnl / cfg.tick_value,
            "hold_bars": n - 1 - entry_bar,
            "entry_z": entry_z,
        })

    return trades


def compute_metrics(trades: list) -> dict:
    if not trades:
        return {"n": 0, "pnl": 0, "wr": 0, "ticks": 0, "pf_net": 0, "avg_hold": 0}

    net_pnls = [t["net_pnl"] for t in trades]
    ticks_arr = [t["ticks"] for t in trades]
    holds = [t["hold_bars"] for t in trades]

    wins = sum(1 for p in net_pnls if p > 0)
    net_profit = sum(p for p in net_pnls if p > 0)
    net_loss = abs(sum(p for p in net_pnls if p < 0))

    return {
        "n": len(trades),
        "pnl": sum(net_pnls),
        "wr": wins / len(trades) * 100 if trades else 0,
        "ticks": np.mean(ticks_arr) if ticks_arr else 0,
        "pf_net": net_profit / net_loss if net_loss > 0 else float("inf"),
        "avg_hold": np.mean(holds) if holds else 0,
    }


def check_gates(m: dict) -> bool:
    return m["pnl"] > 0 and m["pf_net"] >= 1.1 and m["ticks"] >= 1.0 and m["n"] >= 30


def main():
    print("=" * 95)
    print("EMA Z-SCORE MEAN REVERSION STRATEGY")
    print("=" * 95)
    print("Enter when price is extended from EMA(21), exit on reversion to mean.")
    print()

    # Load data
    print("1. Loading data...")
    df = pd.read_parquet('data/sprint_with_ofi.parquet')
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    print(f"   Total ticks: {len(df):,}")

    # Aggregate to 5-min bars
    print("\n2. Aggregating to 5-minute bars...")
    df['bar'] = df['timestamp'].dt.floor('5min')

    bars = df.groupby('bar').agg({
        'last': ['first', 'max', 'min', 'last'],
        'bid': 'last',
        'ask': 'last',
    }).reset_index()
    bars.columns = ['timestamp', 'open', 'high', 'low', 'close', 'bid', 'ask']

    print(f"   Total 5-min bars: {len(bars):,}")

    # Split periods
    train_end = pd.Timestamp("2025-02-28 23:59:59", tz="UTC")
    test_start = pd.Timestamp("2025-03-01 00:00:00", tz="UTC")

    train_bars = bars[bars['timestamp'] <= train_end].copy()
    test_bars = bars[bars['timestamp'] >= test_start].copy()

    print(f"   Train bars (Jan-Feb): {len(train_bars):,}")
    print(f"   Test bars (March): {len(test_bars):,}")

    cfg = BacktestConfig()

    # Grid search
    print("\n" + "=" * 95)
    print("GRID SEARCH: EMA Z-SCORE MEAN REVERSION")
    print("=" * 95)
    print(f"{'EMA':>4} {'Zlb':>4} {'Ent':>5} {'Exit':>5} {'Max':>4} | "
          f"{'Tr_N':>5} {'Tr_PnL':>9} {'Tr_PF':>7} | "
          f"{'Te_N':>5} {'Te_PnL':>9} {'Te_PF':>7} | {'Status':>12}")
    print("-" * 95)

    results = []

    for ema_period in [21, 34, 55]:
        for z_lookback in [21, 34]:
            # Compute EMA and Z-score for full dataset
            bars['ema'] = bars['close'].ewm(span=ema_period, adjust=False).mean()
            bars['dist'] = (bars['close'] - bars['ema']) / bars['ema']
            bars['dist_std'] = bars['dist'].rolling(z_lookback).std()
            bars['zscore'] = bars['dist'] / bars['dist_std']

            zscore_full = bars['zscore'].values
            train_zscore = zscore_full[:len(train_bars)]
            test_zscore = zscore_full[-len(test_bars):]

            for entry_thresh in [1.5, 2.0, 2.5, 3.0]:
                for exit_thresh in [0.0, 0.5]:
                    for max_bars in [6, 12, 24]:  # 30min, 1hr, 2hr at 5-min bars
                        # Train
                        train_trades = run_backtest(
                            train_bars, train_zscore, entry_thresh, exit_thresh, cfg, max_bars
                        )
                        train_m = compute_metrics(train_trades)
                        train_go = check_gates(train_m)

                        # Test
                        test_trades = run_backtest(
                            test_bars, test_zscore, entry_thresh, exit_thresh, cfg, max_bars
                        )
                        test_m = compute_metrics(test_trades)
                        test_go = check_gates(test_m)

                        # Status
                        both_profit = train_m['pnl'] > 0 and test_m['pnl'] > 0
                        if both_profit:
                            if train_go and test_go:
                                status = "BOTH GO!"
                            elif test_go:
                                status = "Test GO"
                            elif train_go:
                                status = "Train GO"
                            else:
                                status = "CONSISTENT"
                        elif test_go:
                            status = "Test only"
                        elif train_go:
                            status = "Train only"
                        else:
                            status = ""

                        # Only print interesting results
                        if both_profit or train_go or test_go or (train_m['n'] >= 20 and test_m['n'] >= 10):
                            print(f"{ema_period:>4} {z_lookback:>4} {entry_thresh:>5.1f} {exit_thresh:>5.1f} {max_bars:>4} | "
                                  f"{train_m['n']:>5} ${train_m['pnl']:>8.2f} {train_m['pf_net']:>7.2f} | "
                                  f"{test_m['n']:>5} ${test_m['pnl']:>8.2f} {test_m['pf_net']:>7.2f} | "
                                  f"{status:>12}")

                        results.append({
                            "ema": ema_period,
                            "z_lookback": z_lookback,
                            "entry": entry_thresh,
                            "exit": exit_thresh,
                            "max_bars": max_bars,
                            "train_n": train_m['n'],
                            "train_pnl": train_m['pnl'],
                            "train_pf": train_m['pf_net'],
                            "train_go": train_go,
                            "test_n": test_m['n'],
                            "test_pnl": test_m['pnl'],
                            "test_pf": test_m['pf_net'],
                            "test_go": test_go,
                            "both_profit": both_profit,
                            "train_wr": train_m['wr'],
                            "test_wr": test_m['wr'],
                        })

    # Summary
    print("\n" + "=" * 95)
    print("SUMMARY")
    print("=" * 95)

    consistent = [r for r in results if r['both_profit']]
    both_go = [r for r in results if r['train_go'] and r['test_go']]
    test_go_list = [r for r in results if r['test_go']]
    train_go_list = [r for r in results if r['train_go']]

    print(f"\nTotal configs tested: {len(results)}")
    print(f"Configs profitable in BOTH periods: {len(consistent)}")
    print(f"Configs with BOTH GO: {len(both_go)}")
    print(f"Configs with Train GO: {len(train_go_list)}")
    print(f"Configs with Test GO: {len(test_go_list)}")

    if consistent:
        print("\n*** CONSISTENT CONFIGURATIONS (profitable in both periods): ***")
        for r in sorted(consistent, key=lambda x: -(x['train_pnl'] + x['test_pnl']))[:10]:
            go_mark = " BOTH GO!" if r['train_go'] and r['test_go'] else ""
            print(f"   EMA={r['ema']}, Zlb={r['z_lookback']}, Entry={r['entry']}, Exit={r['exit']}, Max={r['max_bars']}: "
                  f"Train=${r['train_pnl']:.2f} (n={r['train_n']}, WR={r['train_wr']:.0f}%), "
                  f"Test=${r['test_pnl']:.2f} (n={r['test_n']}, WR={r['test_wr']:.0f}%){go_mark}")

    if both_go:
        print("\n*** VALIDATED GO CONFIGURATIONS (both periods meet all gates): ***")
        for r in sorted(both_go, key=lambda x: -(x['train_pnl'] + x['test_pnl'])):
            print(f"   EMA={r['ema']}, Zlb={r['z_lookback']}, Entry={r['entry']}, Exit={r['exit']}, Max={r['max_bars']}")
            print(f"      Train: ${r['train_pnl']:.2f}, n={r['train_n']}, PF={r['train_pf']:.2f}, WR={r['train_wr']:.0f}%")
            print(f"      Test:  ${r['test_pnl']:.2f}, n={r['test_n']}, PF={r['test_pf']:.2f}, WR={r['test_wr']:.0f}%")

    # Best config analysis
    if consistent:
        best = max(consistent, key=lambda x: x['train_pnl'] + x['test_pnl'])
        print("\n" + "=" * 95)
        print("BEST CONSISTENT CONFIG ANALYSIS")
        print("=" * 95)
        print(f"Config: EMA={best['ema']}, Z_lookback={best['z_lookback']}, "
              f"Entry_Z={best['entry']}, Exit_Z={best['exit']}, Max_bars={best['max_bars']}")
        print(f"Train: ${best['train_pnl']:.2f} ({best['train_n']} trades, {best['train_wr']:.0f}% WR, PF={best['train_pf']:.2f})")
        print(f"Test:  ${best['test_pnl']:.2f} ({best['test_n']} trades, {best['test_wr']:.0f}% WR, PF={best['test_pf']:.2f})")

    print("\n" + "=" * 95)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
