"""
Momentum Strategy Backtest

Tests momentum/trend-following signals instead of mean-reversion:
1. Rate of Change (ROC) - price momentum
2. Moving Average Crossover (fast/slow)
3. Breakout (price breaks N-bar high/low)
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class BacktestConfig:
    commission_per_side: float = 0.85
    tick_size: float = 0.25
    tick_value: float = 1.25


def compute_roc(prices: np.ndarray, period: int) -> np.ndarray:
    """Rate of Change: (price - price_n_bars_ago) / price_n_bars_ago"""
    roc = np.full(len(prices), np.nan)
    for i in range(period, len(prices)):
        if prices[i - period] != 0:
            roc[i] = (prices[i] - prices[i - period]) / prices[i - period]
    return roc


def compute_ma_crossover(prices: np.ndarray, fast: int, slow: int) -> np.ndarray:
    """Returns 1 when fast MA > slow MA, -1 when fast MA < slow MA"""
    fast_ma = pd.Series(prices).rolling(fast).mean().values
    slow_ma = pd.Series(prices).rolling(slow).mean().values

    signal = np.zeros(len(prices))
    valid = ~np.isnan(fast_ma) & ~np.isnan(slow_ma)
    signal[valid & (fast_ma > slow_ma)] = 1
    signal[valid & (fast_ma < slow_ma)] = -1
    return signal


def compute_breakout(prices: np.ndarray, period: int) -> Tuple[np.ndarray, np.ndarray]:
    """Returns rolling high and low over period"""
    high = pd.Series(prices).rolling(period).max().values
    low = pd.Series(prices).rolling(period).min().values
    return high, low


def run_momentum_backtest(df: pd.DataFrame, signal: np.ndarray, cfg: BacktestConfig,
                          exit_bars: int = 10) -> List[dict]:
    """
    Run momentum backtest with time-based exit.

    Entry: Signal at bar t, fill at bar t+1
    Exit: After exit_bars bars OR signal reverses
    """
    trades = []
    n = len(df)

    bid = df['bid'].values
    ask = df['ask'].values

    position = 0
    entry_price = 0.0
    entry_bar = 0

    for i in range(1, n):
        prev_signal = signal[i - 1]

        if position == 0:
            if prev_signal == 1:
                position = 1
                entry_price = ask[i]
                entry_bar = i
            elif prev_signal == -1:
                position = -1
                entry_price = bid[i]
                entry_bar = i
        else:
            hold_time = i - entry_bar
            should_exit = False

            # Exit on time or signal reversal
            if hold_time >= exit_bars:
                should_exit = True
            elif position == 1 and prev_signal == -1:
                should_exit = True
            elif position == -1 and prev_signal == 1:
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


def run_breakout_backtest(df: pd.DataFrame, period: int, cfg: BacktestConfig,
                          exit_bars: int = 10) -> List[dict]:
    """
    Breakout strategy: Buy when price breaks above N-bar high,
    sell when price breaks below N-bar low.
    """
    trades = []
    n = len(df)

    close = df['close'].values
    bid = df['bid'].values
    ask = df['ask'].values

    # Compute rolling high/low (excluding current bar)
    roll_high = np.full(n, np.nan)
    roll_low = np.full(n, np.nan)

    for i in range(period, n):
        roll_high[i] = np.max(close[i-period:i])
        roll_low[i] = np.min(close[i-period:i])

    position = 0
    entry_price = 0.0
    entry_bar = 0

    for i in range(period + 1, n):
        if np.isnan(roll_high[i-1]) or np.isnan(roll_low[i-1]):
            continue

        prev_close = close[i - 1]
        prev_high = roll_high[i - 1]
        prev_low = roll_low[i - 1]

        if position == 0:
            # Long on breakout above high
            if prev_close > prev_high:
                position = 1
                entry_price = ask[i]
                entry_bar = i
            # Short on breakdown below low
            elif prev_close < prev_low:
                position = -1
                entry_price = bid[i]
                entry_bar = i
        else:
            hold_time = i - entry_bar
            should_exit = False

            # Exit on time or opposite breakout
            if hold_time >= exit_bars:
                should_exit = True
            elif position == 1 and prev_close < prev_low:
                should_exit = True
            elif position == -1 and prev_close > prev_high:
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
        return {"n": 0, "pnl": 0, "wr": 0, "ticks": 0, "pf": 0}

    net_pnls = [t["net_pnl"] for t in trades]
    gross_pnls = [t["gross_pnl"] for t in trades]
    ticks = [t["ticks"] for t in trades]

    wins = sum(1 for p in net_pnls if p > 0)
    gross_profit = sum(p for p in gross_pnls if p > 0)
    gross_loss = abs(sum(p for p in gross_pnls if p < 0))

    return {
        "n": len(trades),
        "pnl": sum(net_pnls),
        "wr": wins / len(trades) * 100 if trades else 0,
        "ticks": np.mean(ticks) if ticks else 0,
        "pf": gross_profit / gross_loss if gross_loss > 0 else float("inf"),
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
    print("MOMENTUM STRATEGY BACKTEST")
    print("=" * 70)

    # Load data
    print("\n1. Loading tick data...")
    df = pd.read_parquet('data/sprint_with_ofi.parquet')
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    print(f"   Ticks: {len(df):,}")

    # Aggregate to 1-minute bars (best timeframe from OU tests)
    print("\n2. Aggregating to 1-minute bars...")
    df['bar'] = df['timestamp'].dt.floor('1min')
    bars = df.groupby('bar').agg({
        'last': ['first', 'max', 'min', 'last'],
        'bid': 'last',
        'ask': 'last',
        'volume': 'sum',
    }).reset_index()
    bars.columns = ['timestamp', 'open', 'high', 'low', 'close', 'bid', 'ask', 'volume']

    print(f"   1-minute bars: {len(bars):,}")

    # Split train/test
    test_start = pd.Timestamp("2025-03-01", tz="UTC")
    test_df = bars[bars['timestamp'] >= test_start].copy()
    print(f"   Test bars: {len(test_df):,}")

    cfg = BacktestConfig()

    # =========================================================================
    # TEST 1: Rate of Change (ROC) Momentum
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 1: RATE OF CHANGE (ROC) MOMENTUM")
    print("=" * 70)
    print(f"{'Period':>8} {'Exit':>6} {'Thresh':>8} {'Trades':>7} {'P&L':>10} {'WR':>7} {'Ticks':>8} {'PF':>6} {'GO?':>5}")
    print("-" * 70)

    roc_results = []
    for period in [5, 10, 20, 30, 60]:
        roc = compute_roc(test_df['close'].values, period)

        for thresh in [0.0005, 0.001, 0.002]:
            # Generate signal: long when ROC > thresh, short when ROC < -thresh
            signal = np.zeros(len(test_df))
            valid = ~np.isnan(roc)
            signal[valid & (roc > thresh)] = 1
            signal[valid & (roc < -thresh)] = -1

            for exit_bars in [5, 10, 20]:
                trades = run_momentum_backtest(test_df, signal, cfg, exit_bars)
                m = compute_metrics(trades)
                go = check_gates(m)

                status = "GO" if go else ""
                print(f"{period:>8} {exit_bars:>6} {thresh:>8.4f} {m['n']:>7} ${m['pnl']:>9.2f} "
                      f"{m['wr']:>6.1f}% {m['ticks']:>8.2f} {m['pf']:>6.2f} {status:>5}")

                roc_results.append({**m, "period": period, "exit": exit_bars, "thresh": thresh, "go": go})

    # =========================================================================
    # TEST 2: Moving Average Crossover
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 2: MOVING AVERAGE CROSSOVER")
    print("=" * 70)
    print(f"{'Fast':>6} {'Slow':>6} {'Exit':>6} {'Trades':>7} {'P&L':>10} {'WR':>7} {'Ticks':>8} {'PF':>6} {'GO?':>5}")
    print("-" * 70)

    ma_results = []
    for fast, slow in [(5, 20), (10, 30), (10, 50), (20, 60), (5, 10), (10, 20)]:
        signal = compute_ma_crossover(test_df['close'].values, fast, slow)

        for exit_bars in [10, 20, 30]:
            trades = run_momentum_backtest(test_df, signal, cfg, exit_bars)
            m = compute_metrics(trades)
            go = check_gates(m)

            status = "GO" if go else ""
            print(f"{fast:>6} {slow:>6} {exit_bars:>6} {m['n']:>7} ${m['pnl']:>9.2f} "
                  f"{m['wr']:>6.1f}% {m['ticks']:>8.2f} {m['pf']:>6.2f} {status:>5}")

            ma_results.append({**m, "fast": fast, "slow": slow, "exit": exit_bars, "go": go})

    # =========================================================================
    # TEST 3: Breakout Strategy
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 3: BREAKOUT (N-BAR HIGH/LOW)")
    print("=" * 70)
    print(f"{'Period':>8} {'Exit':>6} {'Trades':>7} {'P&L':>10} {'WR':>7} {'Ticks':>8} {'PF':>6} {'GO?':>5}")
    print("-" * 70)

    breakout_results = []
    for period in [5, 10, 20, 30, 60, 120]:
        for exit_bars in [5, 10, 20, 30]:
            trades = run_breakout_backtest(test_df, period, cfg, exit_bars)
            m = compute_metrics(trades)
            go = check_gates(m)

            status = "GO" if go else ""
            print(f"{period:>8} {exit_bars:>6} {m['n']:>7} ${m['pnl']:>9.2f} "
                  f"{m['wr']:>6.1f}% {m['ticks']:>8.2f} {m['pf']:>6.2f} {status:>5}")

            breakout_results.append({**m, "period": period, "exit": exit_bars, "go": go})

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_results = roc_results + ma_results + breakout_results
    go_results = [r for r in all_results if r.get("go")]

    if go_results:
        print(f"\n*** FOUND {len(go_results)} GO CONFIGURATION(S)! ***")
        for r in go_results:
            print(f"   {r}")
    else:
        print("\n*** NO CONFIGURATION ACHIEVED GO STATUS ***")

    # Best by PF
    valid_results = [r for r in all_results if r["n"] >= 30]
    if valid_results:
        best_pf = max(valid_results, key=lambda x: x["pf"])
        print(f"\nBest by Profit Factor (min 30 trades):")
        print(f"   PF={best_pf['pf']:.2f}, Trades={best_pf['n']}, P&L=${best_pf['pnl']:.2f}")

        # Best by P&L
        best_pnl = max(valid_results, key=lambda x: x["pnl"])
        print(f"\nBest by P&L (min 30 trades):")
        print(f"   P&L=${best_pnl['pnl']:.2f}, Trades={best_pnl['n']}, PF={best_pnl['pf']:.2f}")

    # Positive P&L configurations
    positive = [r for r in valid_results if r["pnl"] > 0]
    if positive:
        print(f"\nConfigurations with POSITIVE P&L:")
        for r in sorted(positive, key=lambda x: -x["pnl"])[:10]:
            print(f"   P&L=${r['pnl']:.2f}, PF={r['pf']:.2f}, Trades={r['n']}")

    print("\n" + "=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
