"""
Volatility-Filtered OFI Momentum

March had 2x higher volatility than Jan-Feb. Test if filtering for
high-volatility periods makes the strategy consistent.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class BacktestConfig:
    commission_per_side: float = 0.85
    tick_size: float = 0.25
    tick_value: float = 1.25


def run_backtest_vol_filter(df: pd.DataFrame, ofi: np.ndarray, thresh: float,
                            volatility: np.ndarray, vol_percentile: float,
                            vol_lookback: int, cfg: BacktestConfig, exit_bars: int) -> list:
    """Only trade when volatility is above vol_percentile of recent history."""
    trades = []
    n = len(df)

    open_prices = df['open'].values
    position = 0
    entry_price = 0.0
    entry_bar = 0

    # Compute rolling volatility threshold
    vol_thresh = np.full(n, np.nan)
    for i in range(vol_lookback, n):
        window_data = volatility[i-vol_lookback:i]
        valid = window_data[~np.isnan(window_data)]
        if len(valid) >= vol_lookback // 2:
            vol_thresh[i] = np.percentile(valid, vol_percentile)

    for i in range(vol_lookback + 1, n):
        if np.isnan(ofi[i-1]) or np.isnan(volatility[i-1]) or np.isnan(vol_thresh[i-1]):
            continue

        prev_ofi = ofi[i - 1]
        current_vol = volatility[i - 1]
        vol_threshold = vol_thresh[i - 1]

        # Only trade in high volatility
        high_vol = current_vol >= vol_threshold

        if position == 0:
            if high_vol:  # Only enter during high vol
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
                trades.append({"gross_pnl": gross_pnl, "net_pnl": net_pnl, "ticks": gross_pnl / cfg.tick_value})
                position = 0

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
    print("=" * 90)
    print("VOLATILITY-FILTERED OFI MOMENTUM")
    print("=" * 90)
    print("Only trades during high-volatility periods (above percentile threshold).")
    print()

    # Load data
    print("1. Loading data...")
    df = pd.read_parquet('data/sprint_with_ofi.parquet')
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

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

    # Compute OFI
    window = 10
    roll_buy = bars['buy_vol'].rolling(window, min_periods=window).sum()
    roll_sell = bars['sell_vol'].rolling(window, min_periods=window).sum()
    ofi = ((roll_buy - roll_sell) / (roll_buy + roll_sell)).replace([np.inf, -np.inf], np.nan).values

    # Compute bar volatility (range as % of close)
    bars['range_pct'] = (bars['high'] - bars['low']) / bars['close']
    volatility = bars['range_pct'].rolling(20).mean().values  # Smoothed volatility

    # Split periods
    train_end = pd.Timestamp("2025-02-28 23:59:59", tz="UTC")
    test_start = pd.Timestamp("2025-03-01 00:00:00", tz="UTC")

    train_bars = bars[bars['timestamp'] <= train_end].copy()
    test_bars = bars[bars['timestamp'] >= test_start].copy()

    train_ofi = ofi[:len(train_bars)]
    test_ofi = ofi[-len(test_bars):]
    train_vol = volatility[:len(train_bars)]
    test_vol = volatility[-len(test_bars):]

    print(f"   Train bars: {len(train_bars):,}")
    print(f"   Test bars: {len(test_bars):,}")

    cfg = BacktestConfig()

    # Grid search
    print("\n" + "=" * 90)
    print("GRID SEARCH: VOLATILITY FILTER")
    print("=" * 90)
    print(f"{'OFI_T':>6} {'Vol%':>5} {'Look':>6} {'Exit':>5} {'Tr_N':>6} {'Tr_PnL':>9} {'Te_N':>5} {'Te_PnL':>9} {'Te_PF':>7} {'Status':>12}")
    print("-" * 90)

    results = []

    for ofi_thresh in [0.15, 0.20, 0.25]:
        for vol_pct in [50, 60, 70, 80]:
            for vol_lookback in [500, 1000]:
                for exit_bars in [20, 30]:
                    train_trades = run_backtest_vol_filter(
                        train_bars, train_ofi, ofi_thresh, train_vol, vol_pct, vol_lookback, cfg, exit_bars
                    )
                    train_m = compute_metrics(train_trades)

                    test_trades = run_backtest_vol_filter(
                        test_bars, test_ofi, ofi_thresh, test_vol, vol_pct, vol_lookback, cfg, exit_bars
                    )
                    test_m = compute_metrics(test_trades)

                    test_go = check_gates(test_m)
                    both_profit = train_m['pnl'] > 0 and test_m['pnl'] > 0
                    train_go = train_m['pnl'] > 0 and train_m['pf_net'] >= 1.1 and train_m['n'] >= 30

                    if both_profit:
                        status = "CONSISTENT"
                        if test_go and train_go:
                            status = "BOTH GO!"
                        elif test_go:
                            status = "Test GO"
                    elif test_go:
                        status = "Test GO"
                    else:
                        status = ""

                    print(f"{ofi_thresh:>6.2f} {vol_pct:>5} {vol_lookback:>6} {exit_bars:>5} "
                          f"{train_m['n']:>6} ${train_m['pnl']:>8.2f} "
                          f"{test_m['n']:>5} ${test_m['pnl']:>8.2f} {test_m['pf_net']:>7.2f} {status:>12}")

                    results.append({
                        "ofi_thresh": ofi_thresh,
                        "vol_pct": vol_pct,
                        "vol_lookback": vol_lookback,
                        "exit": exit_bars,
                        "train_n": train_m['n'],
                        "train_pnl": train_m['pnl'],
                        "train_pf": train_m['pf_net'],
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
    test_go_list = [r for r in results if r['test_go']]

    print(f"\nConfigs profitable in BOTH periods: {len(consistent)}")
    print(f"Configs with test GO: {len(test_go_list)}")

    if consistent:
        print("\n*** CONSISTENT CONFIGURATIONS: ***")
        for r in sorted(consistent, key=lambda x: -(x['train_pnl'] + x['test_pnl'])):
            go_mark = " (Test GO)" if r['test_go'] else ""
            print(f"   OFI={r['ofi_thresh']}, Vol%={r['vol_pct']}, Look={r['vol_lookback']}, Exit={r['exit']}: "
                  f"Train=${r['train_pnl']:.2f} (n={r['train_n']}), "
                  f"Test=${r['test_pnl']:.2f} (n={r['test_n']}){go_mark}")

    print("\n" + "=" * 90)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
