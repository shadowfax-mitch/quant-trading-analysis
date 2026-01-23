"""
EMA Z-Score Mean Reversion - Tick Bar Analysis

Test if tick-based bars outperform time-based bars:
- 1000 tick bars
- 1500 tick bars
- 2000 tick bars
- Compare to 5-minute time bars baseline
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class BacktestConfig:
    commission_per_side: float = 0.85
    tick_size: float = 0.25
    tick_value: float = 1.25  # MES


def aggregate_to_tick_bars(df: pd.DataFrame, ticks_per_bar: int) -> pd.DataFrame:
    """Aggregate tick data to OHLC bars with fixed tick count."""
    df = df.copy().reset_index(drop=True)

    # Assign bar number based on tick count
    df['bar_num'] = df.index // ticks_per_bar

    bars = df.groupby('bar_num').agg({
        'timestamp': 'first',
        'last': ['first', 'max', 'min', 'last'],
    }).reset_index(drop=True)
    bars.columns = ['timestamp', 'open', 'high', 'low', 'close']

    return bars


def aggregate_to_time_bars(df: pd.DataFrame, bar_size: str) -> pd.DataFrame:
    """Aggregate tick data to OHLC bars of specified time size."""
    df = df.copy()
    df['bar'] = df['timestamp'].dt.floor(bar_size)

    bars = df.groupby('bar').agg({
        'last': ['first', 'max', 'min', 'last'],
    }).reset_index()
    bars.columns = ['timestamp', 'open', 'high', 'low', 'close']

    return bars


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


def check_gates(m: dict, min_trades: int = 30) -> bool:
    return m["pnl"] > 0 and m["pf_net"] >= 1.1 and m["n"] >= min_trades


def main():
    print("=" * 100)
    print("EMA Z-SCORE TICK BAR ANALYSIS")
    print("=" * 100)
    print()

    # Load data
    print("1. Loading tick data...")
    df = pd.read_parquet('data/sprint_with_ofi.parquet')
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    print(f"   Total ticks: {len(df):,}")

    cfg = BacktestConfig()

    # Split points
    train_end = pd.Timestamp("2025-02-28 23:59:59", tz="UTC")
    test_start = pd.Timestamp("2025-03-01 00:00:00", tz="UTC")

    # Filter to train/test periods in tick data
    train_ticks = df[df['timestamp'] <= train_end].copy()
    test_ticks = df[df['timestamp'] >= test_start].copy()

    print(f"   Train ticks: {len(train_ticks):,}")
    print(f"   Test ticks: {len(test_ticks):,}")

    # Bar configurations to test
    # Tick bars + time bar baseline
    bar_configs = [
        ('tick', 1000, '1000-tick', 30),
        ('tick', 1500, '1500-tick', 25),
        ('tick', 2000, '2000-tick', 20),
        ('tick', 2500, '2500-tick', 20),
        ('tick', 3000, '3000-tick', 20),
        ('time', '5min', '5-min (baseline)', 20),
    ]

    # Best parameters from 5-min analysis
    param_sets = [
        (21, 21, 3.5, 1.0, 36),
        (21, 21, 3.5, 0.5, 36),
        (34, 21, 3.5, 1.0, 36),
        (34, 21, 3.5, 0.0, 36),
        (13, 21, 3.0, 0.5, 24),
        (13, 21, 3.0, 0.5, 36),
    ]

    all_results = []

    for bar_type, bar_param, bar_label, min_trades in bar_configs:
        print(f"\n{'='*100}")
        print(f"{bar_label.upper()} BARS")
        print(f"{'='*100}")

        # Aggregate to bars
        if bar_type == 'tick':
            train_bars = aggregate_to_tick_bars(train_ticks, bar_param)
            test_bars = aggregate_to_tick_bars(test_ticks, bar_param)
        else:
            train_bars = aggregate_to_time_bars(train_ticks, bar_param)
            test_bars = aggregate_to_time_bars(test_ticks, bar_param)

        print(f"   Train bars: {len(train_bars):,}")
        print(f"   Test bars: {len(test_bars):,}")

        if bar_type == 'tick':
            # Calculate average time per bar
            if len(train_bars) > 1:
                time_span = (train_bars['timestamp'].iloc[-1] - train_bars['timestamp'].iloc[0]).total_seconds()
                avg_seconds = time_span / len(train_bars)
                print(f"   Avg bar duration: {avg_seconds:.1f} seconds ({avg_seconds/60:.2f} minutes)")

        print(f"\n   {'EMA':>4} {'Zlb':>4} {'Ent':>5} {'Exit':>5} {'Max':>4} | "
              f"{'Tr_N':>5} {'Tr_PnL':>9} {'Tr_PF':>7} | "
              f"{'Te_N':>5} {'Te_PnL':>9} {'Te_PF':>7} | {'Status':>12}")
        print("   " + "-" * 90)

        bar_results = []

        for ema, z_lb, entry, exit_z, max_b in param_sets:
            # Compute Z-score on train bars
            train_bars['ema'] = train_bars['close'].ewm(span=ema, adjust=False).mean()
            train_bars['dist'] = (train_bars['close'] - train_bars['ema']) / train_bars['ema']
            train_bars['dist_std'] = train_bars['dist'].rolling(z_lb).std()
            train_bars['zscore'] = train_bars['dist'] / train_bars['dist_std']

            # Compute Z-score on test bars
            test_bars['ema'] = test_bars['close'].ewm(span=ema, adjust=False).mean()
            test_bars['dist'] = (test_bars['close'] - test_bars['ema']) / test_bars['ema']
            test_bars['dist_std'] = test_bars['dist'].rolling(z_lb).std()
            test_bars['zscore'] = test_bars['dist'] / test_bars['dist_std']

            train_z = train_bars['zscore'].values
            test_z = test_bars['zscore'].values

            # Backtest
            train_trades = run_backtest(train_bars, train_z, entry, exit_z, cfg, max_b)
            test_trades = run_backtest(test_bars, test_z, entry, exit_z, cfg, max_b)

            train_m = compute_metrics(train_trades)
            test_m = compute_metrics(test_trades)

            train_go = check_gates(train_m, min_trades)
            test_go = check_gates(test_m, min_trades // 2)
            both_profit = train_m['pnl'] > 0 and test_m['pnl'] > 0

            if both_profit:
                if train_go and test_go:
                    status = "BOTH GO!"
                elif train_go:
                    status = "Train GO"
                elif test_go:
                    status = "Test GO"
                else:
                    status = "CONSISTENT"
            elif train_go:
                status = "Train only"
            elif test_go:
                status = "Test only"
            else:
                status = ""

            # Print all results for comparison
            print(f"   {ema:>4} {z_lb:>4} {entry:>5.1f} {exit_z:>5.1f} {max_b:>4} | "
                  f"{train_m['n']:>5} ${train_m['pnl']:>8.2f} {train_m['pf_net']:>7.2f} | "
                  f"{test_m['n']:>5} ${test_m['pnl']:>8.2f} {test_m['pf_net']:>7.2f} | "
                  f"{status:>12}")

            bar_results.append({
                "bar_type": bar_label,
                "bar_param": bar_param,
                "ema": ema, "z_lb": z_lb, "entry": entry, "exit": exit_z, "max": max_b,
                "train_pnl": train_m['pnl'], "test_pnl": test_m['pnl'],
                "train_pf": train_m['pf_net'], "test_pf": test_m['pf_net'],
                "train_n": train_m['n'], "test_n": test_m['n'],
                "both_profit": both_profit, "train_go": train_go, "test_go": test_go,
            })

        # Summary for this bar type
        consistent = [r for r in bar_results if r['both_profit']]
        both_go = [r for r in bar_results if r['train_go'] and r['test_go']]

        print(f"\n   {bar_label} Summary:")
        print(f"   - Configs tested: {len(bar_results)}")
        print(f"   - Consistent (both profitable): {len(consistent)}")
        print(f"   - Both GO: {len(both_go)}")

        if consistent:
            best = max(consistent, key=lambda x: x['train_pnl'] + x['test_pnl'])
            print(f"   - Best combined P&L: ${best['train_pnl'] + best['test_pnl']:.2f}")

        all_results.extend(bar_results)

    # Overall comparison
    print("\n" + "=" * 100)
    print("BAR TYPE COMPARISON SUMMARY")
    print("=" * 100)

    print(f"\n{'Bar Type':<20} {'Consistent':>12} {'Both GO':>10} {'Best Combined P&L':>20}")
    print("-" * 65)

    for bar_type, bar_param, bar_label, _ in bar_configs:
        bar_results = [r for r in all_results if r['bar_type'] == bar_label]
        consistent = [r for r in bar_results if r['both_profit']]
        both_go = [r for r in bar_results if r['train_go'] and r['test_go']]

        if consistent:
            best = max(consistent, key=lambda x: x['train_pnl'] + x['test_pnl'])
            best_pnl = f"${best['train_pnl'] + best['test_pnl']:.2f}"
        else:
            best_pnl = "N/A"

        marker = " ***" if bar_label == '5-min (baseline)' else ""
        print(f"{bar_label:<20} {len(consistent):>12} {len(both_go):>10} {best_pnl:>20}{marker}")

    # Best configs comparison
    print("\n" + "=" * 100)
    print("BEST CONFIGURATION BY BAR TYPE")
    print("=" * 100)

    for bar_type, bar_param, bar_label, _ in bar_configs:
        bar_results = [r for r in all_results if r['bar_type'] == bar_label]
        both_go = [r for r in bar_results if r['train_go'] and r['test_go']]
        consistent = [r for r in bar_results if r['both_profit']]

        if both_go:
            best = max(both_go, key=lambda x: x['train_pnl'] + x['test_pnl'])
            status = "BOTH GO"
        elif consistent:
            best = max(consistent, key=lambda x: x['train_pnl'] + x['test_pnl'])
            status = "Consistent"
        else:
            print(f"\n{bar_label}: No profitable configs")
            continue

        print(f"\n{bar_label} ({status}):")
        print(f"   EMA={best['ema']}, Zlb={best['z_lb']}, Entry={best['entry']}, Exit={best['exit']}, Max={best['max']}")
        print(f"   Train: ${best['train_pnl']:.2f} (n={best['train_n']}, PF={best['train_pf']:.2f})")
        print(f"   Test:  ${best['test_pnl']:.2f} (n={best['test_n']}, PF={best['test_pf']:.2f})")
        print(f"   Combined: ${best['train_pnl'] + best['test_pnl']:.2f}")

    print("\n" + "=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
