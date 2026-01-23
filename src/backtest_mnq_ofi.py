"""
OFI Momentum on MNQ Data

Test if OFI momentum shows consistency on MNQ (Micro Nasdaq futures)
where MES showed regime dependency.

Memory-efficient version: processes files one at a time.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
import gc


@dataclass
class BacktestConfig:
    commission_per_side: float = 0.85
    tick_size: float = 0.25  # MNQ tick size
    tick_value: float = 0.50  # MNQ tick value ($0.50 per tick)


def load_mnq_bars(file_range: tuple[int, int], start_date: str, end_date: str) -> pd.DataFrame:
    """Load MNQ tick data and aggregate to 1-min bars, file by file."""
    data_dir = Path("datasets/MNQ/tick_data")
    all_bars = []

    for i in range(file_range[0], file_range[1] + 1):
        file_path = data_dir / f"mnq_ticks_part{i:04d}.csv"
        if not file_path.exists():
            continue

        # Load file
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

        # Filter to date range
        df = df[(df['timestamp'] >= start_date) & (df['timestamp'] < end_date)]

        if len(df) == 0:
            print(f"   Skipped {file_path.name}: no data in range")
            continue

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
        }).reset_index()
        bars.columns = ['timestamp', 'open', 'high', 'low', 'close', 'bid', 'ask', 'buy_vol', 'sell_vol']

        all_bars.append(bars)
        print(f"   Processed {file_path.name}: {len(df):,} ticks -> {len(bars):,} bars")

        # Free memory
        del df
        gc.collect()

    # Combine all bars
    combined = pd.concat(all_bars, ignore_index=True)

    # Re-aggregate in case bars span files
    combined = combined.groupby('timestamp').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'bid': 'last',
        'ask': 'last',
        'buy_vol': 'sum',
        'sell_vol': 'sum',
    }).reset_index()

    combined = combined.sort_values('timestamp').reset_index(drop=True)
    return combined


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
    print("=" * 90)
    print("OFI MOMENTUM ON MNQ DATA")
    print("=" * 90)
    print("Testing if MNQ shows consistency where MES showed regime dependency.")
    print()

    # Load MNQ data (files 49-69 cover Dec 2024 - Apr 2025)
    print("1. Loading MNQ tick data and aggregating to 1-min bars...")
    bars = load_mnq_bars((49, 69), "2025-01-01", "2025-04-01")
    print(f"\n   Total bars: {len(bars):,}")

    # Define periods
    train_end = pd.Timestamp("2025-02-28 23:59:59", tz="UTC")
    test_start = pd.Timestamp("2025-03-01 00:00:00", tz="UTC")

    train_bars = bars[bars['timestamp'] <= train_end].copy()
    test_bars = bars[bars['timestamp'] >= test_start].copy()

    print(f"   Train bars (Jan-Feb): {len(train_bars):,}")
    print(f"   Test bars (March): {len(test_bars):,}")

    cfg = BacktestConfig()

    # Grid search
    print("\n" + "=" * 90)
    print("GRID SEARCH: OFI MOMENTUM ON MNQ")
    print("=" * 90)
    print(f"{'Win':>4} {'Thr':>5} {'Exit':>5} | {'Tr_N':>5} {'Tr_PnL':>9} {'Tr_PF':>7} | "
          f"{'Te_N':>5} {'Te_PnL':>9} {'Te_PF':>7} | {'Status':>12}")
    print("-" * 90)

    results = []

    for window in [5, 10, 15]:
        # Compute OFI for full dataset
        roll_buy = bars['buy_vol'].rolling(window, min_periods=window).sum()
        roll_sell = bars['sell_vol'].rolling(window, min_periods=window).sum()
        total = roll_buy + roll_sell
        ofi_full = ((roll_buy - roll_sell) / total).replace([np.inf, -np.inf], np.nan).values

        train_ofi = ofi_full[:len(train_bars)]
        test_ofi = ofi_full[-len(test_bars):]

        for thresh in [0.15, 0.20, 0.25, 0.30]:
            for exit_bars in [10, 20, 30, 40]:
                # Train
                train_trades = run_backtest(train_bars, train_ofi, thresh, cfg, exit_bars)
                train_m = compute_metrics(train_trades)
                train_go = check_gates(train_m)

                # Test
                test_trades = run_backtest(test_bars, test_ofi, thresh, cfg, exit_bars)
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

                print(f"{window:>4} {thresh:>5.2f} {exit_bars:>5} | "
                      f"{train_m['n']:>5} ${train_m['pnl']:>8.2f} {train_m['pf_net']:>7.2f} | "
                      f"{test_m['n']:>5} ${test_m['pnl']:>8.2f} {test_m['pf_net']:>7.2f} | "
                      f"{status:>12}")

                results.append({
                    "window": window,
                    "thresh": thresh,
                    "exit": exit_bars,
                    "train_n": train_m['n'],
                    "train_pnl": train_m['pnl'],
                    "train_pf": train_m['pf_net'],
                    "train_go": train_go,
                    "test_n": test_m['n'],
                    "test_pnl": test_m['pnl'],
                    "test_pf": test_m['pf_net'],
                    "test_go": test_go,
                    "both_profit": both_profit,
                })

    # Summary
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

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
        for r in sorted(consistent, key=lambda x: -(x['train_pnl'] + x['test_pnl'])):
            go_mark = " BOTH GO!" if r['train_go'] and r['test_go'] else ""
            print(f"   W={r['window']}, T={r['thresh']}, E={r['exit']}: "
                  f"Train=${r['train_pnl']:.2f} (n={r['train_n']}, PF={r['train_pf']:.2f}), "
                  f"Test=${r['test_pnl']:.2f} (n={r['test_n']}, PF={r['test_pf']:.2f}){go_mark}")

    if both_go:
        print("\n*** VALIDATED GO CONFIGURATIONS (both periods meet all gates): ***")
        for r in both_go:
            print(f"   W={r['window']}, T={r['thresh']}, E={r['exit']}: "
                  f"Train=${r['train_pnl']:.2f} (PF={r['train_pf']:.2f}), "
                  f"Test=${r['test_pnl']:.2f} (PF={r['test_pf']:.2f})")

    # Compare with MES
    print("\n" + "=" * 90)
    print("COMPARISON: MNQ vs MES")
    print("=" * 90)
    print(f"{'Metric':<35} {'MES':>15} {'MNQ':>15}")
    print("-" * 65)
    print(f"{'Configs tested':<35} {'~530':>15} {len(results):>15}")
    print(f"{'Consistent (both profitable)':<35} {'0':>15} {len(consistent):>15}")
    print(f"{'Both GO':<35} {'0':>15} {len(both_go):>15}")

    # Conclusion
    print("\n" + "=" * 90)
    print("CONCLUSION")
    print("=" * 90)

    if len(both_go) > 0:
        print(f"""
SUCCESS! MNQ shows {len(both_go)} configurations that achieve GO in BOTH periods.

This suggests OFI momentum MAY be more robust on MNQ than MES. The Nasdaq
has different microstructure characteristics that may provide more consistent
order flow signals.

RECOMMENDED NEXT STEPS:
1. Monthly breakdown analysis on best MNQ config
2. Extend test to additional months (April/May 2025)
3. Compare MNQ vs MES microstructure differences
""")
    elif len(consistent) > 0:
        print(f"""
PARTIAL SUCCESS: {len(consistent)} configs profitable in both periods, but none meet full GO gates.

The strategy shows some consistency on MNQ, but metrics don't meet minimum thresholds.
Consider:
- Adjusting gate thresholds for MNQ's smaller tick value
- Testing longer holding periods
- Combining with regime filter
""")
    else:
        print("""
NO IMPROVEMENT: MNQ shows same regime dependency as MES.

OFI momentum is not a robust strategy on either instrument with current
configuration space. Consider abandoning this approach or waiting for more data.
""")

    print("=" * 90)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
