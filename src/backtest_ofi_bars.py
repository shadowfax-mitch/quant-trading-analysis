"""
OFI Momentum on 1-Minute Bars (Fast Version)
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
    """OFI Momentum: Long when OFI > thresh, Short when OFI < -thresh"""
    trades = []
    n = len(df)

    bid = df['bid'].values
    ask = df['ask'].values

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
                entry_price = ask[i]
                entry_bar = i
            elif prev_ofi < -thresh:
                position = -1
                entry_price = bid[i]
                entry_bar = i
        else:
            hold_time = i - entry_bar
            should_exit = hold_time >= exit_bars

            # Also exit on signal reversal
            if position == 1 and prev_ofi < -thresh:
                should_exit = True
            elif position == -1 and prev_ofi > thresh:
                should_exit = True

            if should_exit:
                if position == 1:
                    exit_price = bid[i]
                    gross_pnl = (exit_price - entry_price) / cfg.tick_size * cfg.tick_value
                else:
                    exit_price = ask[i]
                    gross_pnl = (entry_price - exit_price) / cfg.tick_size * cfg.tick_value

                net_pnl = gross_pnl - 2 * cfg.commission_per_side
                trades.append({"gross_pnl": gross_pnl, "net_pnl": net_pnl, "ticks": gross_pnl / cfg.tick_value})
                position = 0

    return trades


def compute_metrics(trades: list) -> dict:
    if not trades:
        return {"n": 0, "pnl": 0, "wr": 0, "ticks": 0, "pf": 0}

    net_pnls = [t["net_pnl"] for t in trades]
    gross_pnls = [t["gross_pnl"] for t in trades]
    ticks_arr = [t["ticks"] for t in trades]

    wins = sum(1 for p in net_pnls if p > 0)
    gross_profit = sum(p for p in gross_pnls if p > 0)
    gross_loss = abs(sum(p for p in gross_pnls if p < 0))

    return {
        "n": len(trades),
        "pnl": sum(net_pnls),
        "wr": wins / len(trades) * 100 if trades else 0,
        "ticks": np.mean(ticks_arr) if ticks_arr else 0,
        "pf": gross_profit / gross_loss if gross_loss > 0 else float("inf"),
    }


def check_gates(m: dict) -> bool:
    return m["pnl"] > 0 and m["pf"] >= 1.1 and m["ticks"] >= 1.0 and m["n"] >= 30


def main():
    print("=" * 70)
    print("OFI MOMENTUM ON 1-MINUTE BARS")
    print("=" * 70)

    # Load data
    print("\n1. Loading tick data...")
    df = pd.read_parquet('data/sprint_with_ofi.parquet')
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    print(f"   Ticks: {len(df):,}")

    # Aggregate to 1-min bars
    print("\n2. Aggregating to 1-minute bars...")
    df['bar'] = df['timestamp'].dt.floor('1min')

    # Compute buy/sell volume per bar
    df['buy_vol'] = np.where(df['side'] == 'A', df['volume'], 0)
    df['sell_vol'] = np.where(df['side'] == 'B', df['volume'], 0)

    bars = df.groupby('bar').agg({
        'last': 'last',
        'bid': 'last',
        'ask': 'last',
        'buy_vol': 'sum',
        'sell_vol': 'sum',
    }).reset_index()
    bars.columns = ['timestamp', 'close', 'bid', 'ask', 'buy_vol', 'sell_vol']

    print(f"   1-minute bars: {len(bars):,}")

    # Split test
    test_start = pd.Timestamp("2025-03-01", tz="UTC")
    test_bars = bars[bars['timestamp'] >= test_start].copy()
    print(f"   Test bars: {len(test_bars):,}")

    cfg = BacktestConfig()

    # Grid search
    print("\n" + "=" * 70)
    print("GRID SEARCH")
    print("=" * 70)
    print(f"{'Window':>8} {'Thresh':>8} {'Exit':>6} {'Trades':>7} {'P&L':>10} {'WR':>7} {'Ticks':>8} {'PF':>6} {'GO?':>5}")
    print("-" * 70)

    results = []

    for window in [5, 10, 20, 30, 60, 120]:
        # Compute rolling OFI
        roll_buy = bars['buy_vol'].rolling(window, min_periods=window).sum()
        roll_sell = bars['sell_vol'].rolling(window, min_periods=window).sum()
        total = roll_buy + roll_sell
        ofi_full = ((roll_buy - roll_sell) / total).replace([np.inf, -np.inf], np.nan).values

        # Get test portion
        ofi = ofi_full[len(bars) - len(test_bars):]

        for thresh in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
            for exit_bars in [3, 5, 10, 20, 30]:
                trades = run_backtest(test_bars, ofi, thresh, cfg, exit_bars)
                m = compute_metrics(trades)
                go = check_gates(m)
                status = "GO" if go else ""

                print(f"{window:>8} {thresh:>8.2f} {exit_bars:>6} {m['n']:>7} ${m['pnl']:>9.2f} "
                      f"{m['wr']:>6.1f}% {m['ticks']:>8.2f} {m['pf']:>6.2f} {status:>5}")

                results.append({**m, "window": window, "thresh": thresh, "exit": exit_bars, "go": go})

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    go_results = [r for r in results if r.get("go")]
    if go_results:
        print(f"\n*** FOUND {len(go_results)} GO CONFIGURATION(S)! ***")
        for r in sorted(go_results, key=lambda x: -x["pnl"]):
            print(f"   Window={r['window']}, Thresh={r['thresh']}, Exit={r['exit']}: "
                  f"Trades={r['n']}, P&L=${r['pnl']:.2f}, PF={r['pf']:.2f}, Ticks={r['ticks']:.2f}")
    else:
        print("\n*** NO CONFIGURATION ACHIEVED GO STATUS ***")

    valid = [r for r in results if r["n"] >= 30]
    if valid:
        best_pf = max(valid, key=lambda x: x["pf"])
        print(f"\nBest by PF: Window={best_pf['window']}, Thresh={best_pf['thresh']}, Exit={best_pf['exit']}")
        print(f"   PF={best_pf['pf']:.2f}, P&L=${best_pf['pnl']:.2f}, Trades={best_pf['n']}")

        best_pnl = max(valid, key=lambda x: x["pnl"])
        print(f"\nBest by P&L: Window={best_pnl['window']}, Thresh={best_pnl['thresh']}, Exit={best_pnl['exit']}")
        print(f"   P&L=${best_pnl['pnl']:.2f}, PF={best_pnl['pf']:.2f}, Trades={best_pnl['n']}")

    positive = [r for r in valid if r["pnl"] > 0]
    if positive:
        print(f"\n*** {len(positive)} CONFIGURATIONS WITH POSITIVE P&L ***")
        for r in sorted(positive, key=lambda x: -x["pnl"])[:15]:
            print(f"   W={r['window']:>3}, T={r['thresh']:.2f}, E={r['exit']:>2}: "
                  f"P&L=${r['pnl']:>7.2f}, PF={r['pf']:.2f}, N={r['n']}, Ticks={r['ticks']:.2f}")

    print("\n" + "=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
