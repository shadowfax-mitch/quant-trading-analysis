"""
Adaptive Threshold OFI Momentum

Instead of a fixed OFI threshold, use a percentile-based threshold
that adapts to recent OFI distribution. This should produce consistent
signal frequency across different regimes.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class BacktestConfig:
    commission_per_side: float = 0.85
    tick_size: float = 0.25
    tick_value: float = 1.25


def run_backtest_adaptive(df: pd.DataFrame, ofi: np.ndarray, percentile: float,
                          lookback: int, cfg: BacktestConfig, exit_bars: int) -> list:
    """
    Adaptive threshold: use rolling percentile of |OFI| to set threshold.

    percentile: e.g., 99 means only enter when |OFI| is in top 1%
    lookback: bars to use for percentile calculation
    """
    trades = []
    n = len(df)

    open_prices = df['open'].values
    position = 0
    entry_price = 0.0
    entry_bar = 0

    # Precompute rolling thresholds
    abs_ofi = np.abs(ofi)
    rolling_thresh = np.full(n, np.nan)

    for i in range(lookback, n):
        window_data = abs_ofi[i-lookback:i]
        valid_data = window_data[~np.isnan(window_data)]
        if len(valid_data) >= lookback // 2:
            rolling_thresh[i] = np.percentile(valid_data, percentile)

    for i in range(lookback + 1, n):
        if np.isnan(ofi[i-1]) or np.isnan(rolling_thresh[i-1]):
            continue

        prev_ofi = ofi[i - 1]
        thresh = rolling_thresh[i - 1]

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
        trades.append({"gross_pnl": gross_pnl, "net_pnl": net_pnl, "ticks": gross_pnl / cfg.tick_value})

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


def check_gates(m: dict) -> bool:
    return m["pnl"] > 0 and m["pf_net"] >= 1.1 and m["ticks"] >= 1.0 and m["n"] >= 30


def main():
    print("=" * 85)
    print("ADAPTIVE THRESHOLD OFI MOMENTUM")
    print("=" * 85)
    print("Uses rolling percentile of |OFI| to set threshold, adapting to regime changes.")
    print()

    # Load data
    print("1. Loading data...")
    df = pd.read_parquet('data/sprint_with_ofi.parquet')
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

    # Aggregate to 1-min bars
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

    # Compute OFI
    window = 10
    roll_buy = bars['buy_vol'].rolling(window, min_periods=window).sum()
    roll_sell = bars['sell_vol'].rolling(window, min_periods=window).sum()
    ofi = ((roll_buy - roll_sell) / (roll_buy + roll_sell)).replace([np.inf, -np.inf], np.nan).values

    # Split periods
    train_end = pd.Timestamp("2025-02-28 23:59:59", tz="UTC")
    test_start = pd.Timestamp("2025-03-01 00:00:00", tz="UTC")

    train_bars = bars[bars['timestamp'] <= train_end].copy()
    test_bars = bars[bars['timestamp'] >= test_start].copy()

    train_ofi = ofi[:len(train_bars)]
    test_ofi = ofi[-len(test_bars):]

    print(f"   Train bars: {len(train_bars):,}")
    print(f"   Test bars: {len(test_bars):,}")

    cfg = BacktestConfig()

    # Grid search
    print("\n" + "=" * 85)
    print("GRID SEARCH: ADAPTIVE THRESHOLD")
    print("=" * 85)
    print(f"{'Pctl':>6} {'Look':>6} {'Exit':>6} {'Train_N':>8} {'Train_PnL':>10} {'Test_N':>7} {'Test_PnL':>10} {'Test_PF':>8} {'GO?':>5}")
    print("-" * 85)

    results = []

    for percentile in [95, 97, 98, 99]:
        for lookback in [500, 1000, 2000]:
            for exit_bars in [10, 20, 30]:
                # Train
                train_trades = run_backtest_adaptive(train_bars, train_ofi, percentile, lookback, cfg, exit_bars)
                train_m = compute_metrics(train_trades)

                # Test
                test_trades = run_backtest_adaptive(test_bars, test_ofi, percentile, lookback, cfg, exit_bars)
                test_m = compute_metrics(test_trades)

                test_go = check_gates(test_m)
                both_profitable = train_m['pnl'] > 0 and test_m['pnl'] > 0

                status = "GO" if test_go else ""
                if both_profitable:
                    status += "*"

                print(f"{percentile:>6} {lookback:>6} {exit_bars:>6} {train_m['n']:>8} ${train_m['pnl']:>9.2f} "
                      f"{test_m['n']:>7} ${test_m['pnl']:>9.2f} {test_m['pf_net']:>8.2f} {status:>5}")

                results.append({
                    "percentile": percentile,
                    "lookback": lookback,
                    "exit": exit_bars,
                    "train_n": train_m['n'],
                    "train_pnl": train_m['pnl'],
                    "test_n": test_m['n'],
                    "test_pnl": test_m['pnl'],
                    "test_pf": test_m['pf_net'],
                    "test_go": test_go,
                    "both_profitable": both_profitable,
                })

    # Summary
    print("\n" + "=" * 85)
    print("SUMMARY")
    print("=" * 85)

    consistent = [r for r in results if r['both_profitable']]
    go_results = [r for r in results if r['test_go']]
    both_go = [r for r in results if r['both_profitable'] and r['test_go']]

    print(f"\nConfigs with GO on test: {len(go_results)}")
    print(f"Configs profitable in BOTH periods: {len(consistent)}")
    print(f"Configs with BOTH profitable AND test GO: {len(both_go)}")

    if consistent:
        print("\n*** CONSISTENT CONFIGURATIONS (profitable in both periods): ***")
        for r in sorted(consistent, key=lambda x: -x['test_pnl']):
            go_mark = " GO" if r['test_go'] else ""
            print(f"   Pctl={r['percentile']}, Look={r['lookback']}, Exit={r['exit']}: "
                  f"Train=${r['train_pnl']:.2f} ({r['train_n']} trades), "
                  f"Test=${r['test_pnl']:.2f} ({r['test_n']} trades), PF={r['test_pf']:.2f}{go_mark}")

    if both_go:
        print("\n*** VALIDATED GO CONFIGURATIONS: ***")
        for r in both_go:
            print(f"   Pctl={r['percentile']}, Look={r['lookback']}, Exit={r['exit']}")

    print("\n" + "=" * 85)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
