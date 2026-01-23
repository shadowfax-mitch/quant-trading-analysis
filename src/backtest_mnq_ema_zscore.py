"""
EMA Z-Score Mean Reversion on MNQ Data

Test if the strategy that works on MES also works on MNQ.
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
    """Load MNQ tick data and aggregate to 5-min bars, file by file."""
    data_dir = Path("datasets/MNQ/tick_data")
    all_bars = []

    for i in range(file_range[0], file_range[1] + 1):
        file_path = data_dir / f"mnq_ticks_part{i:04d}.csv"
        if not file_path.exists():
            continue

        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df = df[(df['timestamp'] >= start_date) & (df['timestamp'] < end_date)]

        if len(df) == 0:
            continue

        # Aggregate to 5-min bars
        df['bar'] = df['timestamp'].dt.floor('5min')

        bars = df.groupby('bar').agg({
            'last': ['first', 'max', 'min', 'last'],
            'bid': 'last',
            'ask': 'last',
        }).reset_index()
        bars.columns = ['timestamp', 'open', 'high', 'low', 'close', 'bid', 'ask']

        all_bars.append(bars)
        print(f"   Processed {file_path.name}: {len(bars):,} bars")

        del df
        gc.collect()

    combined = pd.concat(all_bars, ignore_index=True)
    combined = combined.groupby('timestamp').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'bid': 'last',
        'ask': 'last',
    }).reset_index()

    combined = combined.sort_values('timestamp').reset_index(drop=True)
    return combined


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


def check_gates(m: dict, min_trades: int = 20) -> bool:
    return m["pnl"] > 0 and m["pf_net"] >= 1.1 and m["n"] >= min_trades


def main():
    print("=" * 95)
    print("EMA Z-SCORE MEAN REVERSION ON MNQ")
    print("=" * 95)
    print("Testing if the MES edge transfers to MNQ (Nasdaq futures).")
    print()

    # Load MNQ data
    print("1. Loading MNQ tick data and aggregating to 5-min bars...")
    bars = load_mnq_bars((49, 69), "2025-01-01", "2025-04-01")
    print(f"\n   Total 5-min bars: {len(bars):,}")

    # Split periods
    train_end = pd.Timestamp("2025-02-28 23:59:59", tz="UTC")
    test_start = pd.Timestamp("2025-03-01 00:00:00", tz="UTC")

    train_bars = bars[bars['timestamp'] <= train_end].copy()
    test_bars = bars[bars['timestamp'] >= test_start].copy()

    print(f"   Train bars (Jan-Feb): {len(train_bars):,}")
    print(f"   Test bars (March): {len(test_bars):,}")

    cfg = BacktestConfig()

    # Test the GO configurations from MES
    print("\n" + "=" * 95)
    print("TESTING MES GO CONFIGURATIONS ON MNQ")
    print("=" * 95)

    # Best configs from MES
    mes_go_configs = [
        (34, 21, 3.5, 1.0, 36, "Best combined"),
        (34, 21, 3.5, 0.0, 36, "Best test PF"),
        (13, 21, 3.0, 0.5, 24, "Most trades"),
        (13, 21, 3.0, 0.5, 36, "Rank 4"),
        (13, 34, 3.0, 0.0, 36, "Rank 5"),
        (21, 34, 3.0, 0.5, 36, "Rank 6"),
        (21, 21, 3.5, 1.0, 12, "Rank 7"),
    ]

    print(f"\n{'Config':<30} | {'Tr_N':>5} {'Tr_PnL':>9} {'Tr_PF':>7} | "
          f"{'Te_N':>5} {'Te_PnL':>9} {'Te_PF':>7} | {'MNQ Status':>12}")
    print("-" * 95)

    mnq_results = []

    for ema, z_lb, entry, exit_z, max_b, label in mes_go_configs:
        # Compute indicators
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

        train_go = check_gates(train_m, min_trades=30)
        test_go = check_gates(test_m, min_trades=15)
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
            status = "FAILED"

        config_str = f"EMA={ema}, Z={entry}, E={exit_z}, M={max_b}"
        print(f"{config_str:<30} | "
              f"{train_m['n']:>5} ${train_m['pnl']:>8.2f} {train_m['pf_net']:>7.2f} | "
              f"{test_m['n']:>5} ${test_m['pnl']:>8.2f} {test_m['pf_net']:>7.2f} | "
              f"{status:>12}")

        mnq_results.append({
            "config": config_str,
            "label": label,
            "train_pnl": train_m['pnl'],
            "test_pnl": test_m['pnl'],
            "train_pf": train_m['pf_net'],
            "test_pf": test_m['pf_net'],
            "train_n": train_m['n'],
            "test_n": test_m['n'],
            "both_profit": both_profit,
            "train_go": train_go,
            "test_go": test_go,
        })

    # Expanded search on MNQ
    print("\n" + "=" * 95)
    print("EXPANDED GRID SEARCH ON MNQ")
    print("=" * 95)
    print(f"{'EMA':>4} {'Zlb':>4} {'Ent':>5} {'Exit':>5} {'Max':>4} | "
          f"{'Tr_N':>5} {'Tr_PnL':>9} {'Tr_PF':>7} | "
          f"{'Te_N':>5} {'Te_PnL':>9} {'Te_PF':>7} | {'Status':>12}")
    print("-" * 95)

    all_results = []

    for ema in [13, 21, 34]:
        for z_lb in [21, 34]:
            bars['ema'] = bars['close'].ewm(span=ema, adjust=False).mean()
            bars['dist'] = (bars['close'] - bars['ema']) / bars['ema']
            bars['dist_std'] = bars['dist'].rolling(z_lb).std()
            bars['zscore'] = bars['dist'] / bars['dist_std']

            train_z = bars['zscore'].values[:len(train_bars)]
            test_z = bars['zscore'].values[-len(test_bars):]

            for entry in [2.5, 3.0, 3.5]:
                for exit_z in [0.0, 0.5, 1.0]:
                    for max_b in [12, 24, 36]:
                        train_trades = run_backtest(train_bars, train_z, entry, exit_z, cfg, max_b)
                        test_trades = run_backtest(test_bars, test_z, entry, exit_z, cfg, max_b)

                        train_m = compute_metrics(train_trades)
                        test_m = compute_metrics(test_trades)

                        train_go = check_gates(train_m, min_trades=30)
                        test_go = check_gates(test_m, min_trades=15)
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

                            print(f"{ema:>4} {z_lb:>4} {entry:>5.1f} {exit_z:>5.1f} {max_b:>4} | "
                                  f"{train_m['n']:>5} ${train_m['pnl']:>8.2f} {train_m['pf_net']:>7.2f} | "
                                  f"{test_m['n']:>5} ${test_m['pnl']:>8.2f} {test_m['pf_net']:>7.2f} | "
                                  f"{status:>12}")

                        all_results.append({
                            "ema": ema, "z_lb": z_lb, "entry": entry, "exit": exit_z, "max": max_b,
                            "train_pnl": train_m['pnl'], "test_pnl": test_m['pnl'],
                            "train_pf": train_m['pf_net'], "test_pf": test_m['pf_net'],
                            "train_n": train_m['n'], "test_n": test_m['n'],
                            "both_profit": both_profit, "train_go": train_go, "test_go": test_go,
                        })

    # Summary
    print("\n" + "=" * 95)
    print("MNQ SUMMARY")
    print("=" * 95)

    consistent = [r for r in all_results if r['both_profit']]
    both_go = [r for r in all_results if r['train_go'] and r['test_go']]

    print(f"\nTotal configs tested: {len(all_results)}")
    print(f"Configs profitable in BOTH periods: {len(consistent)}")
    print(f"Configs with BOTH GO: {len(both_go)}")

    # Compare MES vs MNQ
    print("\n" + "=" * 95)
    print("COMPARISON: MES vs MNQ")
    print("=" * 95)
    print(f"{'Metric':<35} {'MES':>15} {'MNQ':>15}")
    print("-" * 65)
    print(f"{'Consistent (both profitable)':<35} {'57':>15} {len(consistent):>15}")
    print(f"{'Both GO':<35} {'7':>15} {len(both_go):>15}")

    if both_go:
        print("\n*** MNQ VALIDATED GO CONFIGURATIONS ***")
        for r in sorted(both_go, key=lambda x: -(x['train_pnl'] + x['test_pnl'])):
            print(f"   EMA={r['ema']}, Zlb={r['z_lb']}, Entry={r['entry']}, Exit={r['exit']}, Max={r['max']}")
            print(f"      Train: ${r['train_pnl']:.2f} (n={r['train_n']}, PF={r['train_pf']:.2f})")
            print(f"      Test:  ${r['test_pnl']:.2f} (n={r['test_n']}, PF={r['test_pf']:.2f})")

    # Conclusion
    print("\n" + "=" * 95)
    print("CONCLUSION")
    print("=" * 95)

    if len(both_go) >= 3:
        print(f"""
STRONG CONFIRMATION: MNQ shows {len(both_go)} BOTH GO configurations.

The EMA Z-Score mean reversion edge appears to be robust across both
MES (S&P 500) and MNQ (Nasdaq) futures. This significantly reduces
the likelihood of curve fitting.
""")
    elif len(consistent) >= 10:
        print(f"""
MODERATE CONFIRMATION: MNQ shows {len(consistent)} consistent configs.

The strategy shows some consistency on MNQ, supporting the MES findings.
However, fewer configs meet the full GO gates.
""")
    else:
        print(f"""
WEAK/NO CONFIRMATION: MNQ shows limited consistency.

The strategy may be more specific to MES characteristics.
Consider this a cautionary finding for robustness.
""")

    print("=" * 95)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
