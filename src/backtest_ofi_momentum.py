"""
OFI Momentum Strategy Backtest

Uses Order Flow Imbalance as a momentum signal:
- Go long when OFI is strongly positive (buyers in control)
- Go short when OFI is strongly negative (sellers in control)

This is the OPPOSITE of the contrarian OFI strategy we tried before.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List


@dataclass
class BacktestConfig:
    commission_per_side: float = 0.85
    tick_size: float = 0.25
    tick_value: float = 1.25


def compute_ofi(df: pd.DataFrame, window: int) -> np.ndarray:
    """Compute Order Flow Imbalance: (buy_vol - sell_vol) / total_vol"""
    buy_vol = np.where(df["side"] == "A", df["volume"], 0).astype(float)
    sell_vol = np.where(df["side"] == "B", df["volume"], 0).astype(float)

    roll_buy = pd.Series(buy_vol).rolling(window, min_periods=window).sum()
    roll_sell = pd.Series(sell_vol).rolling(window, min_periods=window).sum()

    total = roll_buy + roll_sell
    ofi = (roll_buy - roll_sell) / total
    return ofi.replace([np.inf, -np.inf], np.nan).values


def run_ofi_momentum_backtest(df: pd.DataFrame, ofi: np.ndarray, thresh: float,
                               cfg: BacktestConfig, exit_bars: int = None,
                               exit_on_zero: bool = True) -> List[dict]:
    """
    OFI Momentum: Follow the flow.
    - Long when OFI > thresh (buyers dominating)
    - Short when OFI < -thresh (sellers dominating)
    - Exit when OFI crosses zero OR after exit_bars
    """
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
            # Entry: follow the flow
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
            should_exit = False

            # Exit conditions
            if exit_bars and hold_time >= exit_bars:
                should_exit = True
            elif exit_on_zero:
                # Exit when OFI crosses zero (flow exhausted)
                if position == 1 and prev_ofi <= 0:
                    should_exit = True
                elif position == -1 and prev_ofi >= 0:
                    should_exit = True

            if should_exit:
                if position == 1:
                    exit_price = bid[i]
                    gross_pnl = (exit_price - entry_price) / cfg.tick_size * cfg.tick_value
                else:
                    exit_price = ask[i]
                    gross_pnl = (entry_price - exit_price) / cfg.tick_size * cfg.tick_value

                net_pnl = gross_pnl - 2 * cfg.commission_per_side

                trades.append({
                    "gross_pnl": gross_pnl,
                    "net_pnl": net_pnl,
                    "ticks": gross_pnl / cfg.tick_value,
                    "hold_bars": hold_time,
                })
                position = 0

    return trades


def compute_metrics(trades: list) -> dict:
    if not trades:
        return {"n": 0, "pnl": 0, "wr": 0, "ticks": 0, "pf": 0, "avg_hold": 0}

    net_pnls = [t["net_pnl"] for t in trades]
    gross_pnls = [t["gross_pnl"] for t in trades]
    ticks = [t["ticks"] for t in trades]
    holds = [t["hold_bars"] for t in trades]

    wins = sum(1 for p in net_pnls if p > 0)
    gross_profit = sum(p for p in gross_pnls if p > 0)
    gross_loss = abs(sum(p for p in gross_pnls if p < 0))

    return {
        "n": len(trades),
        "pnl": sum(net_pnls),
        "wr": wins / len(trades) * 100 if trades else 0,
        "ticks": np.mean(ticks) if ticks else 0,
        "pf": gross_profit / gross_loss if gross_loss > 0 else float("inf"),
        "avg_hold": np.mean(holds) if holds else 0,
    }


def check_gates(m: dict) -> bool:
    return (
        m["pnl"] > 0 and
        m["pf"] >= 1.1 and
        m["ticks"] >= 1.0 and
        m["n"] >= 30
    )


def main():
    print("=" * 70)
    print("OFI MOMENTUM STRATEGY BACKTEST")
    print("=" * 70)

    # Load tick data
    print("\n1. Loading tick data...")
    df = pd.read_parquet('data/sprint_with_ofi.parquet')
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    print(f"   Ticks: {len(df):,}")

    # Split test set
    test_start = pd.Timestamp("2025-03-01", tz="UTC")
    test_df = df[df['timestamp'] >= test_start].copy()
    print(f"   Test ticks: {len(test_df):,}")

    cfg = BacktestConfig()

    # =========================================================================
    # TEST AT TICK LEVEL
    # =========================================================================
    print("\n" + "=" * 70)
    print("TICK-LEVEL OFI MOMENTUM")
    print("=" * 70)
    print(f"{'Window':>8} {'Thresh':>8} {'Exit':>6} {'Trades':>7} {'P&L':>10} {'WR':>7} {'Ticks':>8} {'PF':>6} {'Hold':>6} {'GO?':>5}")
    print("-" * 80)

    tick_results = []

    for window in [500, 1000, 2000, 5000]:
        ofi = compute_ofi(test_df, window)

        for thresh in [0.1, 0.2, 0.3, 0.4, 0.5]:
            # Exit on zero crossing
            trades = run_ofi_momentum_backtest(test_df, ofi, thresh, cfg, exit_bars=None, exit_on_zero=True)
            m = compute_metrics(trades)
            go = check_gates(m)
            status = "GO" if go else ""

            print(f"{window:>8} {thresh:>8.2f} {'zero':>6} {m['n']:>7} ${m['pnl']:>9.2f} "
                  f"{m['wr']:>6.1f}% {m['ticks']:>8.2f} {m['pf']:>6.2f} {m['avg_hold']:>6.0f} {status:>5}")

            tick_results.append({**m, "window": window, "thresh": thresh, "exit": "zero", "go": go})

            # Fixed exit bars
            for exit_bars in [100, 500, 1000]:
                trades = run_ofi_momentum_backtest(test_df, ofi, thresh, cfg, exit_bars=exit_bars, exit_on_zero=False)
                m = compute_metrics(trades)
                go = check_gates(m)
                status = "GO" if go else ""

                print(f"{window:>8} {thresh:>8.2f} {exit_bars:>6} {m['n']:>7} ${m['pnl']:>9.2f} "
                      f"{m['wr']:>6.1f}% {m['ticks']:>8.2f} {m['pf']:>6.2f} {m['avg_hold']:>6.0f} {status:>5}")

                tick_results.append({**m, "window": window, "thresh": thresh, "exit": exit_bars, "go": go})

    # =========================================================================
    # TEST AT 1-MINUTE BAR LEVEL
    # =========================================================================
    print("\n" + "=" * 70)
    print("1-MINUTE BAR OFI MOMENTUM")
    print("=" * 70)

    # Aggregate to 1-min bars with OFI computed on bars
    df['bar'] = df['timestamp'].dt.floor('1min')
    bars = df.groupby('bar').agg({
        'last': 'last',
        'bid': 'last',
        'ask': 'last',
        'volume': 'sum',
        'side': lambda x: (x == 'A').sum() - (x == 'B').sum(),  # net buy ticks
    }).reset_index()
    bars.columns = ['timestamp', 'close', 'bid', 'ask', 'volume', 'net_buy']

    # Compute bar-level OFI
    bars['ofi'] = bars['net_buy'].rolling(20).sum() / bars['volume'].rolling(20).sum()
    bars['ofi'] = bars['ofi'].replace([np.inf, -np.inf], np.nan)

    test_bars = bars[bars['timestamp'] >= test_start].copy()
    print(f"   Test bars: {len(test_bars):,}")

    print(f"\n{'Window':>8} {'Thresh':>8} {'Exit':>6} {'Trades':>7} {'P&L':>10} {'WR':>7} {'Ticks':>8} {'PF':>6} {'Hold':>6} {'GO?':>5}")
    print("-" * 80)

    bar_results = []

    for window in [10, 20, 30, 60]:
        # Recompute OFI with different windows
        ofi = bars['net_buy'].rolling(window).sum() / bars['volume'].rolling(window).sum()
        ofi = ofi.replace([np.inf, -np.inf], np.nan).values

        # Get test portion
        test_ofi = ofi[len(bars) - len(test_bars):]

        for thresh in [0.05, 0.1, 0.15, 0.2]:
            for exit_bars in [5, 10, 20]:
                trades = run_ofi_momentum_backtest(test_bars, test_ofi, thresh, cfg,
                                                    exit_bars=exit_bars, exit_on_zero=False)
                m = compute_metrics(trades)
                go = check_gates(m)
                status = "GO" if go else ""

                print(f"{window:>8} {thresh:>8.2f} {exit_bars:>6} {m['n']:>7} ${m['pnl']:>9.2f} "
                      f"{m['wr']:>6.1f}% {m['ticks']:>8.2f} {m['pf']:>6.2f} {m['avg_hold']:>6.0f} {status:>5}")

                bar_results.append({**m, "window": window, "thresh": thresh, "exit": exit_bars, "go": go})

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_results = tick_results + bar_results
    go_results = [r for r in all_results if r.get("go")]

    if go_results:
        print(f"\n*** FOUND {len(go_results)} GO CONFIGURATION(S)! ***")
        for r in go_results:
            print(f"   Window={r['window']}, Thresh={r['thresh']}, Exit={r['exit']}: "
                  f"Trades={r['n']}, P&L=${r['pnl']:.2f}, PF={r['pf']:.2f}")
    else:
        print("\n*** NO CONFIGURATION ACHIEVED GO STATUS ***")

    # Best results
    valid_results = [r for r in all_results if r["n"] >= 30]
    if valid_results:
        best_pf = max(valid_results, key=lambda x: x["pf"])
        print(f"\nBest by Profit Factor (min 30 trades):")
        print(f"   Window={best_pf.get('window')}, Thresh={best_pf.get('thresh')}, Exit={best_pf.get('exit')}")
        print(f"   PF={best_pf['pf']:.2f}, Trades={best_pf['n']}, P&L=${best_pf['pnl']:.2f}")

        best_pnl = max(valid_results, key=lambda x: x["pnl"])
        print(f"\nBest by P&L (min 30 trades):")
        print(f"   Window={best_pnl.get('window')}, Thresh={best_pnl.get('thresh')}, Exit={best_pnl.get('exit')}")
        print(f"   P&L=${best_pnl['pnl']:.2f}, Trades={best_pnl['n']}, PF={best_pnl['pf']:.2f}")

    # Positive P&L
    positive = [r for r in valid_results if r["pnl"] > 0]
    if positive:
        print(f"\n*** CONFIGURATIONS WITH POSITIVE P&L: ***")
        for r in sorted(positive, key=lambda x: -x["pnl"])[:10]:
            print(f"   Window={r.get('window')}, Thresh={r.get('thresh')}, Exit={r.get('exit')}: "
                  f"P&L=${r['pnl']:.2f}, PF={r['pf']:.2f}, Trades={r['n']}")

    print("\n" + "=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
