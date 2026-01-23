"""
EMA Z-Score Mean Reversion - Timeframe Analysis

Test if the strategy works across different bar sizes:
- 1-minute bars
- 5-minute bars (baseline)
- 15-minute bars
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class BacktestConfig:
    commission_per_side: float = 0.85
    tick_size: float = 0.25
    tick_value: float = 1.25  # MES


def aggregate_to_bars(df: pd.DataFrame, bar_size: str) -> pd.DataFrame:
    """Aggregate tick data to OHLC bars of specified size."""
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
    print("EMA Z-SCORE TIMEFRAME ANALYSIS")
    print("=" * 100)
    print()

    # Load data
    print("1. Loading tick data...")
    df = pd.read_parquet('data/sprint_with_ofi.parquet')
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    print(f"   Total ticks: {len(df):,}")

    cfg = BacktestConfig()

    # Test different timeframes
    timeframes = [
        ('1min', '1-minute', 30),   # min_trades adjusted for frequency
        ('5min', '5-minute', 20),
        ('15min', '15-minute', 15),
    ]

    # Adjust parameters for each timeframe to maintain similar real-time windows
    # 5-min baseline: EMA=21 (105 min), Z_lb=21 (105 min), max=36 (180 min)
    param_sets = {
        '1min': [
            # Scale up by 5x for 1-min bars to match real-time window
            (105, 105, 3.0, 0.5, 180),  # ~same real-time as 5-min baseline
            (105, 105, 3.5, 1.0, 180),
            (55, 55, 3.0, 0.5, 90),     # shorter window
            (55, 55, 3.5, 1.0, 90),
            # Also test original 5-min params on 1-min (will be faster signals)
            (21, 21, 3.0, 0.5, 36),
            (21, 21, 3.5, 1.0, 36),
            (34, 21, 3.5, 1.0, 36),
        ],
        '5min': [
            # Original best configs
            (21, 21, 3.5, 1.0, 36),
            (21, 21, 3.5, 0.5, 36),
            (21, 34, 3.0, 0.5, 36),
            (34, 21, 3.5, 0.0, 36),
            (34, 21, 3.5, 1.0, 36),
            (13, 21, 3.0, 0.5, 24),
        ],
        '15min': [
            # Scale down by 3x for 15-min bars
            (7, 7, 3.0, 0.5, 12),
            (7, 7, 3.5, 1.0, 12),
            (11, 7, 3.5, 1.0, 12),
            # Also try original params (will be slower signals)
            (21, 21, 3.0, 0.5, 24),
            (21, 21, 3.5, 1.0, 24),
            (34, 21, 3.5, 1.0, 24),
        ],
    }

    all_results = []

    for bar_size, bar_label, min_trades in timeframes:
        print(f"\n{'='*100}")
        print(f"{bar_label.upper()} BARS")
        print(f"{'='*100}")

        # Aggregate to bars
        bars = aggregate_to_bars(df, bar_size)
        print(f"   Total bars: {len(bars):,}")

        # Split periods
        train_end = pd.Timestamp("2025-02-28 23:59:59", tz="UTC")
        test_start = pd.Timestamp("2025-03-01 00:00:00", tz="UTC")

        train_bars = bars[bars['timestamp'] <= train_end].copy()
        test_bars = bars[bars['timestamp'] >= test_start].copy()

        print(f"   Train bars: {len(train_bars):,}")
        print(f"   Test bars: {len(test_bars):,}")

        print(f"\n   {'EMA':>4} {'Zlb':>4} {'Ent':>5} {'Exit':>5} {'Max':>4} | "
              f"{'Tr_N':>5} {'Tr_PnL':>9} {'Tr_PF':>7} | "
              f"{'Te_N':>5} {'Te_PnL':>9} {'Te_PF':>7} | {'Status':>12}")
        print("   " + "-" * 90)

        timeframe_results = []

        for ema, z_lb, entry, exit_z, max_b in param_sets[bar_size]:
            # Compute Z-score
            bars['ema'] = bars['close'].ewm(span=ema, adjust=False).mean()
            bars['dist'] = (bars['close'] - bars['ema']) / bars['ema']
            bars['dist_std'] = bars['dist'].rolling(z_lb).std()
            bars['zscore'] = bars['dist'] / bars['dist_std']

            train_z = bars['zscore'].values[:len(train_bars)]
            test_z = bars['zscore'].values[-len(test_bars):]

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

            # Only print interesting results
            if both_profit or train_go or test_go:
                print(f"   {ema:>4} {z_lb:>4} {entry:>5.1f} {exit_z:>5.1f} {max_b:>4} | "
                      f"{train_m['n']:>5} ${train_m['pnl']:>8.2f} {train_m['pf_net']:>7.2f} | "
                      f"{test_m['n']:>5} ${test_m['pnl']:>8.2f} {test_m['pf_net']:>7.2f} | "
                      f"{status:>12}")

            timeframe_results.append({
                "timeframe": bar_label,
                "ema": ema, "z_lb": z_lb, "entry": entry, "exit": exit_z, "max": max_b,
                "train_pnl": train_m['pnl'], "test_pnl": test_m['pnl'],
                "train_pf": train_m['pf_net'], "test_pf": test_m['pf_net'],
                "train_n": train_m['n'], "test_n": test_m['n'],
                "both_profit": both_profit, "train_go": train_go, "test_go": test_go,
            })

        # Summary for this timeframe
        consistent = [r for r in timeframe_results if r['both_profit']]
        both_go = [r for r in timeframe_results if r['train_go'] and r['test_go']]

        print(f"\n   {bar_label} Summary:")
        print(f"   - Configs tested: {len(timeframe_results)}")
        print(f"   - Consistent (both profitable): {len(consistent)}")
        print(f"   - Both GO: {len(both_go)}")

        all_results.extend(timeframe_results)

    # Overall comparison
    print("\n" + "=" * 100)
    print("TIMEFRAME COMPARISON SUMMARY")
    print("=" * 100)

    print(f"\n{'Timeframe':<15} {'Configs':>10} {'Consistent':>12} {'Both GO':>10} {'Best Combined P&L':>20}")
    print("-" * 70)

    for bar_size, bar_label, _ in timeframes:
        tf_results = [r for r in all_results if r['timeframe'] == bar_label]
        consistent = [r for r in tf_results if r['both_profit']]
        both_go = [r for r in tf_results if r['train_go'] and r['test_go']]

        if consistent:
            best = max(consistent, key=lambda x: x['train_pnl'] + x['test_pnl'])
            best_pnl = f"${best['train_pnl'] + best['test_pnl']:.2f}"
        else:
            best_pnl = "N/A"

        print(f"{bar_label:<15} {len(tf_results):>10} {len(consistent):>12} {len(both_go):>10} {best_pnl:>20}")

    # Best configs per timeframe
    print("\n" + "=" * 100)
    print("BEST CONFIGURATIONS BY TIMEFRAME")
    print("=" * 100)

    for bar_size, bar_label, _ in timeframes:
        tf_results = [r for r in all_results if r['timeframe'] == bar_label]
        both_go = [r for r in tf_results if r['train_go'] and r['test_go']]

        if both_go:
            print(f"\n{bar_label} - BOTH GO Configs:")
            for r in sorted(both_go, key=lambda x: -(x['train_pnl'] + x['test_pnl']))[:3]:
                print(f"   EMA={r['ema']}, Zlb={r['z_lb']}, Entry={r['entry']}, Exit={r['exit']}, Max={r['max']}")
                print(f"      Train: ${r['train_pnl']:.2f} (n={r['train_n']}, PF={r['train_pf']:.2f})")
                print(f"      Test:  ${r['test_pnl']:.2f} (n={r['test_n']}, PF={r['test_pf']:.2f})")
        else:
            consistent = [r for r in tf_results if r['both_profit']]
            if consistent:
                print(f"\n{bar_label} - Best Consistent (no BOTH GO):")
                best = max(consistent, key=lambda x: x['train_pnl'] + x['test_pnl'])
                print(f"   EMA={best['ema']}, Zlb={best['z_lb']}, Entry={best['entry']}, Exit={best['exit']}, Max={best['max']}")
                print(f"      Train: ${best['train_pnl']:.2f} (n={best['train_n']}, PF={best['train_pf']:.2f})")
                print(f"      Test:  ${best['test_pnl']:.2f} (n={best['test_n']}, PF={best['test_pf']:.2f})")
            else:
                print(f"\n{bar_label} - No consistent configurations found")

    print("\n" + "=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
