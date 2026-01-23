"""
Regime-Gated OFI Momentum

Trades OFI momentum only when a pre-trade regime gate is on. The gate uses
rolling volatility and OFI tail-rate features computed from prior bars.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class BacktestConfig:
    commission_per_side: float = 0.85
    tick_size: float = 0.25
    tick_value: float = 1.25


def rolling_percentile(values: np.ndarray, lookback: int, pct: float) -> np.ndarray:
    thresh = np.full(len(values), np.nan)
    for i in range(lookback, len(values)):
        window = values[i - lookback:i]
        valid = window[~np.isnan(window)]
        if len(valid) >= lookback // 2:
            thresh[i] = np.percentile(valid, pct)
    return thresh


def compute_gate(
    bars: pd.DataFrame,
    ofi: np.ndarray,
    gate_lookback: int,
    vol_pct: float,
    tail_lookback: int,
    tail_pct: float,
    tail_mode: str,
    tail_ofi_thresh: float,
) -> np.ndarray:
    # Volatility feature (range as % of close)
    range_pct = (bars["high"].values - bars["low"].values) / bars["close"].values
    range_pct = np.where(np.isfinite(range_pct), range_pct, np.nan)

    # OFI tail-rate feature (fraction of bars with extreme OFI)
    tail_flag = np.abs(ofi) >= tail_ofi_thresh
    tail_rate = (
        pd.Series(tail_flag)
        .rolling(tail_lookback, min_periods=max(5, tail_lookback // 2))
        .mean()
        .to_numpy()
    )

    vol_thresh = rolling_percentile(range_pct, gate_lookback, vol_pct)
    tail_thresh = rolling_percentile(tail_rate, gate_lookback, tail_pct)

    if tail_mode not in {"low", "high"}:
        raise ValueError("tail_mode must be 'low' or 'high'")

    tail_ok = tail_rate <= tail_thresh if tail_mode == "low" else tail_rate >= tail_thresh

    valid = (
        ~np.isnan(range_pct)
        & ~np.isnan(vol_thresh)
        & ~np.isnan(tail_rate)
        & ~np.isnan(tail_thresh)
    )

    gate_on = np.zeros(len(ofi), dtype=bool)
    gate_on[valid] = (range_pct[valid] >= vol_thresh[valid]) & tail_ok[valid]
    return gate_on


def run_backtest(
    df: pd.DataFrame,
    ofi: np.ndarray,
    gate_on: np.ndarray,
    thresh: float,
    cfg: BacktestConfig,
    exit_bars: int,
) -> list:
    trades = []
    n = len(df)

    open_prices = df["open"].values
    position = 0
    entry_price = 0.0
    entry_bar = 0

    for i in range(1, n):
        if np.isnan(ofi[i - 1]):
            continue

        prev_ofi = ofi[i - 1]

        if position == 0:
            if gate_on[i - 1]:
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


def main() -> int:
    print("=" * 95)
    print("REGIME-GATED OFI MOMENTUM")
    print("=" * 95)
    print("Gate = high volatility + OFI tail-rate condition (low or high).")
    print()

    print("1. Loading data...")
    df = pd.read_parquet("data/sprint_with_ofi.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    valid_sides = {"A", "B"}
    actual_sides = set(df["side"].unique())
    if actual_sides != valid_sides:
        print(f"WARNING: Unexpected side codes: {actual_sides - valid_sides}")
    else:
        print(f"   Side codes validated: {actual_sides}")

    df["bar"] = df["timestamp"].dt.floor("1min")
    df["buy_vol"] = np.where(df["side"] == "A", df["volume"], 0)
    df["sell_vol"] = np.where(df["side"] == "B", df["volume"], 0)

    bars = df.groupby("bar").agg({
        "last": ["first", "max", "min", "last"],
        "bid": "last",
        "ask": "last",
        "buy_vol": "sum",
        "sell_vol": "sum",
    }).reset_index()
    bars.columns = ["timestamp", "open", "high", "low", "close", "bid", "ask", "buy_vol", "sell_vol"]

    train_end = pd.Timestamp("2025-02-28 23:59:59", tz="UTC")
    test_start = pd.Timestamp("2025-03-01 00:00:00", tz="UTC")

    train_bars = bars[bars["timestamp"] <= train_end].copy()
    test_bars = bars[bars["timestamp"] >= test_start].copy()

    print(f"   Train bars: {len(train_bars):,}")
    print(f"   Test bars: {len(test_bars):,}")

    cfg = BacktestConfig()

    ofi_configs = [
        (10, 0.20, 30),
        (5, 0.30, 10),
    ]

    vol_pcts = [60, 70, 80]
    gate_lookbacks = [500, 1000]
    tail_lookbacks = [200, 500]
    tail_modes = ["low", "high"]
    tail_pcts = {"low": [20, 30], "high": [70, 80]}

    results = []

    print("\n" + "=" * 95)
    print("GRID SEARCH: REGIME GATE")
    print("=" * 95)
    print(
        f"{'W':>3} {'T':>5} {'E':>3} {'Vol%':>5} {'Look':>6} {'Tail%':>6} {'TailL':>6} "
        f"{'Mode':>5} {'TrN':>5} {'TrPnL':>9} {'TeN':>5} {'TePnL':>9} {'TePF':>6} {'Status':>10}"
    )
    print("-" * 95)

    for window, thresh, exit_bars in ofi_configs:
        roll_buy = bars["buy_vol"].rolling(window, min_periods=window).sum()
        roll_sell = bars["sell_vol"].rolling(window, min_periods=window).sum()
        total = roll_buy + roll_sell
        ofi_full = ((roll_buy - roll_sell) / total).replace([np.inf, -np.inf], np.nan).to_numpy()

        train_ofi = ofi_full[:len(train_bars)]
        test_ofi = ofi_full[-len(test_bars):]

        for vol_pct in vol_pcts:
            for gate_lookback in gate_lookbacks:
                for tail_lookback in tail_lookbacks:
                    for tail_mode in tail_modes:
                        for tail_pct in tail_pcts[tail_mode]:
                            train_gate = compute_gate(
                                train_bars, train_ofi, gate_lookback, vol_pct,
                                tail_lookback, tail_pct, tail_mode, thresh
                            )
                            test_gate = compute_gate(
                                test_bars, test_ofi, gate_lookback, vol_pct,
                                tail_lookback, tail_pct, tail_mode, thresh
                            )

                            train_trades = run_backtest(train_bars, train_ofi, train_gate, thresh, cfg, exit_bars)
                            test_trades = run_backtest(test_bars, test_ofi, test_gate, thresh, cfg, exit_bars)

                            train_m = compute_metrics(train_trades)
                            test_m = compute_metrics(test_trades)

                            test_go = check_gates(test_m)
                            train_go = check_gates(train_m)
                            both_profit = train_m["pnl"] > 0 and test_m["pnl"] > 0

                            status = ""
                            if both_profit:
                                status = "CONSISTENT"
                            if test_go:
                                status = "Test GO"
                            if train_go and test_go:
                                status = "BOTH GO"

                            print(
                                f"{window:>3} {thresh:>5.2f} {exit_bars:>3} {vol_pct:>5} {gate_lookback:>6} "
                                f"{tail_pct:>6} {tail_lookback:>6} {tail_mode:>5} "
                                f"{train_m['n']:>5} ${train_m['pnl']:>8.2f} "
                                f"{test_m['n']:>5} ${test_m['pnl']:>8.2f} {test_m['pf_net']:>6.2f} {status:>10}"
                            )

                            results.append({
                                "window": window,
                                "thresh": thresh,
                                "exit": exit_bars,
                                "vol_pct": vol_pct,
                                "gate_lookback": gate_lookback,
                                "tail_pct": tail_pct,
                                "tail_lookback": tail_lookback,
                                "tail_mode": tail_mode,
                                "train_n": train_m["n"],
                                "train_pnl": train_m["pnl"],
                                "train_pf": train_m["pf_net"],
                                "test_n": test_m["n"],
                                "test_pnl": test_m["pnl"],
                                "test_pf": test_m["pf_net"],
                                "test_go": test_go,
                                "train_go": train_go,
                                "both_profit": both_profit,
                            })

    print("\n" + "=" * 95)
    print("SUMMARY")
    print("=" * 95)

    consistent = [r for r in results if r["both_profit"]]
    both_go = [r for r in results if r["train_go"] and r["test_go"]]
    test_go_list = [r for r in results if r["test_go"]]

    print(f"\nConfigs profitable in BOTH periods: {len(consistent)}")
    print(f"Configs with test GO: {len(test_go_list)}")
    print(f"Configs with BOTH GO: {len(both_go)}")

    if consistent:
        best = max(consistent, key=lambda x: x["train_pnl"] + x["test_pnl"])
        print("\nBest consistent by total P&L:")
        print(
            f"   W={best['window']}, T={best['thresh']}, E={best['exit']}, "
            f"Vol%={best['vol_pct']}, Look={best['gate_lookback']}, "
            f"Tail%={best['tail_pct']}({best['tail_mode']}), TailL={best['tail_lookback']}"
        )
        print(
            f"   Train=${best['train_pnl']:.2f} (n={best['train_n']}), "
            f"Test=${best['test_pnl']:.2f} (n={best['test_n']})"
        )

    print("\n" + "=" * 95)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
