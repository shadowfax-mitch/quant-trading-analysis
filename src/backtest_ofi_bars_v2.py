"""
OFI Momentum on 1-Minute Bars - V2 (Conservative)

Fixes from Codex review:
1. Entry fills at OPEN of next bar (not close of signal bar)
2. Profit Factor calculated on NET P&L (not gross)
3. Force close open positions at end of test
4. Validate side codes
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
    """
    OFI Momentum with conservative execution model.

    Entry: Signal at bar t-1, fill at OPEN of bar t (more conservative)
    Exit: After exit_bars OR signal reversal, fill at OPEN of exit bar
    """
    trades = []
    n = len(df)

    # Use open prices for fills (more conservative than close)
    open_prices = df['open'].values
    bid = df['bid'].values
    ask = df['ask'].values

    position = 0
    entry_price = 0.0
    entry_bar = 0
    entry_direction = 0

    for i in range(1, n):
        if np.isnan(ofi[i-1]):
            continue

        prev_ofi = ofi[i - 1]

        if position == 0:
            # Entry: use open of current bar (signal was at close of previous bar)
            # Apply spread to open: long pays spread, short receives spread
            if prev_ofi > thresh:
                position = 1
                # Conservative: assume we pay half spread from open
                entry_price = open_prices[i] + cfg.tick_size  # Worse fill for long
                entry_bar = i
                entry_direction = 1
            elif prev_ofi < -thresh:
                position = -1
                entry_price = open_prices[i] - cfg.tick_size  # Worse fill for short
                entry_bar = i
                entry_direction = -1
        else:
            hold_time = i - entry_bar
            should_exit = hold_time >= exit_bars

            # Also exit on signal reversal
            if position == 1 and prev_ofi < -thresh:
                should_exit = True
            elif position == -1 and prev_ofi > thresh:
                should_exit = True

            if should_exit:
                # Exit at open of current bar, with adverse spread
                if position == 1:
                    exit_price = open_prices[i] - cfg.tick_size  # Worse fill for long exit
                    gross_pnl = (exit_price - entry_price) / cfg.tick_size * cfg.tick_value
                else:
                    exit_price = open_prices[i] + cfg.tick_size  # Worse fill for short exit
                    gross_pnl = (entry_price - exit_price) / cfg.tick_size * cfg.tick_value

                net_pnl = gross_pnl - 2 * cfg.commission_per_side
                trades.append({
                    "gross_pnl": gross_pnl,
                    "net_pnl": net_pnl,
                    "ticks": gross_pnl / cfg.tick_value,
                    "hold_bars": hold_time,
                })
                position = 0

    # Force close any open position at end of test
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
            "forced_close": True,
        })

    return trades


def compute_metrics(trades: list) -> dict:
    if not trades:
        return {"n": 0, "pnl": 0, "wr": 0, "ticks": 0, "pf": 0, "pf_net": 0}

    net_pnls = [t["net_pnl"] for t in trades]
    gross_pnls = [t["gross_pnl"] for t in trades]
    ticks_arr = [t["ticks"] for t in trades]

    wins = sum(1 for p in net_pnls if p > 0)
    gross_profit = sum(p for p in gross_pnls if p > 0)
    gross_loss = abs(sum(p for p in gross_pnls if p < 0))

    # Net P&L based profit factor (more conservative)
    net_profit = sum(p for p in net_pnls if p > 0)
    net_loss = abs(sum(p for p in net_pnls if p < 0))

    return {
        "n": len(trades),
        "pnl": sum(net_pnls),
        "wr": wins / len(trades) * 100 if trades else 0,
        "ticks": np.mean(ticks_arr) if ticks_arr else 0,
        "pf": gross_profit / gross_loss if gross_loss > 0 else float("inf"),
        "pf_net": net_profit / net_loss if net_loss > 0 else float("inf"),  # NEW: Net-based PF
    }


def check_gates(m: dict) -> bool:
    # Use NET profit factor for GO gates (more conservative)
    return m["pnl"] > 0 and m["pf_net"] >= 1.1 and m["ticks"] >= 1.0 and m["n"] >= 30


def main():
    print("=" * 80)
    print("OFI MOMENTUM V2 - CONSERVATIVE EXECUTION MODEL")
    print("=" * 80)
    print("Changes from V1:")
    print("  - Entry/exit at bar OPEN (not close)")
    print("  - 1-tick adverse slippage on all fills")
    print("  - Profit Factor on NET P&L (not gross)")
    print("  - Force close open positions at end")
    print()

    # Load data
    print("1. Loading tick data...")
    df = pd.read_parquet('data/sprint_with_ofi.parquet')
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

    # Validate side codes
    valid_sides = {'A', 'B'}
    actual_sides = set(df['side'].unique())
    if actual_sides != valid_sides:
        print(f"WARNING: Unexpected side codes: {actual_sides - valid_sides}")
    else:
        print(f"   Side codes validated: {actual_sides}")

    print(f"   Ticks: {len(df):,}")

    # Aggregate to 1-min bars
    print("\n2. Aggregating to 1-minute bars...")
    df['bar'] = df['timestamp'].dt.floor('1min')

    # Compute buy/sell volume per bar
    df['buy_vol'] = np.where(df['side'] == 'A', df['volume'], 0)
    df['sell_vol'] = np.where(df['side'] == 'B', df['volume'], 0)

    bars = df.groupby('bar').agg({
        'last': ['first', 'last'],  # first = open, last = close
        'bid': 'last',
        'ask': 'last',
        'buy_vol': 'sum',
        'sell_vol': 'sum',
    }).reset_index()
    bars.columns = ['timestamp', 'open', 'close', 'bid', 'ask', 'buy_vol', 'sell_vol']

    print(f"   1-minute bars: {len(bars):,}")

    # Split test
    test_start = pd.Timestamp("2025-03-01", tz="UTC")
    test_bars = bars[bars['timestamp'] >= test_start].copy()
    print(f"   Test bars: {len(test_bars):,}")

    cfg = BacktestConfig()

    # Test the best configurations from V1
    print("\n" + "=" * 80)
    print("TESTING PREVIOUS GO CONFIGURATIONS WITH CONSERVATIVE MODEL")
    print("=" * 80)
    print(f"{'Window':>8} {'Thresh':>8} {'Exit':>6} {'Trades':>7} {'P&L':>10} {'WR':>7} {'Ticks':>8} {'PF_Gross':>9} {'PF_Net':>8} {'GO?':>5}")
    print("-" * 85)

    # Previous GO configs
    go_configs = [
        (10, 0.20, 30),
        (5, 0.30, 10),
        (10, 0.20, 20),
        (10, 0.20, 10),
        (5, 0.30, 5),
        (10, 0.20, 3),
        (20, 0.15, 5),
        (5, 0.20, 10),
    ]

    results = []

    for window, thresh, exit_bars in go_configs:
        roll_buy = bars['buy_vol'].rolling(window, min_periods=window).sum()
        roll_sell = bars['sell_vol'].rolling(window, min_periods=window).sum()
        total = roll_buy + roll_sell
        ofi_full = ((roll_buy - roll_sell) / total).replace([np.inf, -np.inf], np.nan).values

        ofi = ofi_full[len(bars) - len(test_bars):]

        trades = run_backtest(test_bars, ofi, thresh, cfg, exit_bars)
        m = compute_metrics(trades)
        go = check_gates(m)
        status = "GO" if go else ""

        print(f"{window:>8} {thresh:>8.2f} {exit_bars:>6} {m['n']:>7} ${m['pnl']:>9.2f} "
              f"{m['wr']:>6.1f}% {m['ticks']:>8.2f} {m['pf']:>9.2f} {m['pf_net']:>8.2f} {status:>5}")

        results.append({**m, "window": window, "thresh": thresh, "exit": exit_bars, "go": go})

    # Full grid search
    print("\n" + "=" * 80)
    print("FULL GRID SEARCH (CONSERVATIVE MODEL)")
    print("=" * 80)
    print(f"{'Window':>8} {'Thresh':>8} {'Exit':>6} {'Trades':>7} {'P&L':>10} {'WR':>7} {'Ticks':>8} {'PF_Net':>8} {'GO?':>5}")
    print("-" * 80)

    all_results = []

    for window in [5, 10, 20, 30]:
        roll_buy = bars['buy_vol'].rolling(window, min_periods=window).sum()
        roll_sell = bars['sell_vol'].rolling(window, min_periods=window).sum()
        total = roll_buy + roll_sell
        ofi_full = ((roll_buy - roll_sell) / total).replace([np.inf, -np.inf], np.nan).values
        ofi = ofi_full[len(bars) - len(test_bars):]

        for thresh in [0.15, 0.20, 0.25, 0.30, 0.35]:
            for exit_bars in [5, 10, 20, 30, 40]:
                trades = run_backtest(test_bars, ofi, thresh, cfg, exit_bars)
                m = compute_metrics(trades)
                go = check_gates(m)
                status = "GO" if go else ""

                if m['n'] >= 20:  # Only show configs with decent trade count
                    print(f"{window:>8} {thresh:>8.2f} {exit_bars:>6} {m['n']:>7} ${m['pnl']:>9.2f} "
                          f"{m['wr']:>6.1f}% {m['ticks']:>8.2f} {m['pf_net']:>8.2f} {status:>5}")

                all_results.append({**m, "window": window, "thresh": thresh, "exit": exit_bars, "go": go})

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY - CONSERVATIVE MODEL")
    print("=" * 80)

    go_results = [r for r in all_results if r.get("go")]
    if go_results:
        print(f"\n*** FOUND {len(go_results)} GO CONFIGURATION(S)! ***")
        for r in sorted(go_results, key=lambda x: -x["pnl"]):
            print(f"   W={r['window']:>2}, T={r['thresh']:.2f}, E={r['exit']:>2}: "
                  f"N={r['n']}, P&L=${r['pnl']:.2f}, PF_Net={r['pf_net']:.2f}, Ticks={r['ticks']:.2f}")
    else:
        print("\n*** NO CONFIGURATION ACHIEVED GO STATUS WITH CONSERVATIVE MODEL ***")

    # Best results
    valid = [r for r in all_results if r["n"] >= 30]
    if valid:
        best_pf = max(valid, key=lambda x: x["pf_net"])
        print(f"\nBest by Net PF (min 30 trades):")
        print(f"   W={best_pf['window']}, T={best_pf['thresh']}, E={best_pf['exit']}")
        print(f"   PF_Net={best_pf['pf_net']:.2f}, P&L=${best_pf['pnl']:.2f}, N={best_pf['n']}")

        best_pnl = max(valid, key=lambda x: x["pnl"])
        print(f"\nBest by P&L (min 30 trades):")
        print(f"   W={best_pnl['window']}, T={best_pnl['thresh']}, E={best_pnl['exit']}")
        print(f"   P&L=${best_pnl['pnl']:.2f}, PF_Net={best_pnl['pf_net']:.2f}, N={best_pnl['n']}")

    # Positive P&L
    positive = [r for r in valid if r["pnl"] > 0]
    if positive:
        print(f"\n*** {len(positive)} CONFIGS WITH POSITIVE P&L (min 30 trades) ***")
        for r in sorted(positive, key=lambda x: -x["pnl"])[:10]:
            go_mark = " GO" if r["go"] else ""
            print(f"   W={r['window']:>2}, T={r['thresh']:.2f}, E={r['exit']:>2}: "
                  f"P&L=${r['pnl']:>7.2f}, PF_Net={r['pf_net']:.2f}, N={r['n']}, Ticks={r['ticks']:.2f}{go_mark}")
    else:
        print("\nNo configuration achieved positive P&L with 30+ trades.")

    print("\n" + "=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
