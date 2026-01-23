"""
1-Second Bar Aggregation + OU Backtest

Aggregates tick data to 1-second OHLC bars and runs OU mean-reversion backtest.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BacktestConfig:
    commission_per_side: float = 0.85
    tick_size: float = 0.25
    tick_value: float = 1.25


def rolling_ou_params(prices, window=300):
    """Rolling OU estimation on 1s bars."""
    n = len(prices)
    mu = np.full(n, np.nan)
    theta = np.full(n, np.nan)
    sigma = np.full(n, np.nan)

    values = prices.values

    for i in range(window, n):
        x = values[i-window:i-1]
        y = values[i-window+1:i]

        n_obs = len(x)
        sum_x = x.sum()
        sum_y = y.sum()
        sum_xx = (x * x).sum()
        sum_xy = (x * y).sum()

        denom = n_obs * sum_xx - sum_x * sum_x
        if denom <= 0:
            continue

        b = (n_obs * sum_xy - sum_x * sum_y) / denom
        if not (0 < b < 1):
            continue

        a = (sum_y - b * sum_x) / n_obs

        resid = y - (a + b * x)
        sigma_eps2 = (resid ** 2).sum() / (n_obs - 2)
        if sigma_eps2 <= 0:
            continue

        theta_val = -np.log(b)
        if theta_val <= 0:
            continue

        denom_sigma = 1 - b * b
        if denom_sigma <= 0:
            continue

        sigma_val = np.sqrt(sigma_eps2 * 2 * theta_val / denom_sigma)
        mu_val = a / (1 - b)

        mu[i] = mu_val
        theta[i] = theta_val
        sigma[i] = sigma_val

    return mu, theta, sigma


def run_backtest(df, signal_col, cfg):
    trades = []
    n = len(df)

    signal = df[signal_col].values
    mid = df['close'].values
    mu = df['mu'].values
    bid = df['bid'].values
    ask = df['ask'].values
    timestamps = df['timestamp'].values

    position = 0
    entry_idx = 0
    entry_price = 0.0

    for i in range(1, n):
        if position == 0:
            prev_signal = signal[i - 1]
            if prev_signal == 1:
                position = 1
                entry_idx = i
                entry_price = ask[i]
            elif prev_signal == -1:
                position = -1
                entry_idx = i
                entry_price = bid[i]
        else:
            prev_mu = mu[i - 1]
            if np.isnan(prev_mu):
                continue

            should_exit = False
            if position == 1:
                if mid[i] >= prev_mu:
                    should_exit = True
                    exit_price = bid[i]
            else:
                if mid[i] <= prev_mu:
                    should_exit = True
                    exit_price = ask[i]

            if should_exit:
                if position == 1:
                    gross_pnl = (exit_price - entry_price) / cfg.tick_size * cfg.tick_value
                else:
                    gross_pnl = (entry_price - exit_price) / cfg.tick_size * cfg.tick_value

                net_pnl = gross_pnl - 2 * cfg.commission_per_side
                ticks = gross_pnl / cfg.tick_value

                trades.append({
                    "entry_time": pd.Timestamp(timestamps[entry_idx]),
                    "exit_time": pd.Timestamp(timestamps[i]),
                    "direction": position,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "gross_pnl": gross_pnl,
                    "net_pnl": net_pnl,
                    "ticks": ticks,
                })
                position = 0

    return trades


def compute_metrics(trades):
    if not trades:
        return {"num_trades": 0, "total_net_pnl": 0, "win_rate": 0, "avg_ticks": 0, "pf": 0}

    net_pnls = [t["net_pnl"] for t in trades]
    gross_pnls = [t["gross_pnl"] for t in trades]
    ticks = [t["ticks"] for t in trades]

    wins = sum(1 for p in net_pnls if p > 0)
    gross_profit = sum(p for p in gross_pnls if p > 0)
    gross_loss = abs(sum(p for p in gross_pnls if p < 0))

    return {
        "num_trades": len(trades),
        "total_net_pnl": sum(net_pnls),
        "win_rate": wins / len(trades) * 100,
        "avg_ticks": np.mean(ticks),
        "pf": gross_profit / gross_loss if gross_loss > 0 else float("inf"),
    }


def main():
    print("=" * 60)
    print("1-SECOND BAR AGGREGATION + OU BACKTEST")
    print("=" * 60)

    # Load tick data
    print("\n1. Loading tick data...")
    df = pd.read_parquet('data/sprint_with_ofi.parquet')
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    print(f"   Ticks: {len(df):,}")

    # Aggregate to 1-second bars
    print("\n2. Aggregating to 1-second bars...")
    df['second'] = df['timestamp'].dt.floor('1s')

    bars = df.groupby('second').agg({
        'last': ['first', 'max', 'min', 'last'],
        'bid': 'last',
        'ask': 'last',
        'volume': 'sum',
    }).reset_index()

    bars.columns = ['timestamp', 'open', 'high', 'low', 'close', 'bid', 'ask', 'volume']
    bars['mid_price'] = (bars['bid'] + bars['ask']) / 2

    print(f"   1-second bars: {len(bars):,}")
    print(f"   Date range: {bars['timestamp'].min()} to {bars['timestamp'].max()}")

    # OU parameter estimation
    print("\n3. Estimating OU parameters (rolling 300-bar window)...")
    mu, theta, sigma = rolling_ou_params(bars['close'], window=300)
    bars['mu'] = mu
    bars['theta'] = theta
    bars['sigma'] = sigma

    valid_ou = bars['mu'].notna().sum()
    print(f"   Valid OU estimates: {valid_ou:,}")

    # Generate signals
    print("\n4. Generating signals...")
    bars['z_score'] = (bars['close'] - bars['mu']) / bars['sigma']

    for n in [1.5, 2.0, 2.5]:
        col = f'signal_{str(n).replace(".", "_")}'
        signal = np.zeros(len(bars), dtype=np.int8)

        z_lagged = bars['z_score'].shift(1)
        valid = bars['mu'].notna() & bars['sigma'].notna()

        long_mask = valid & (z_lagged < -n)
        short_mask = valid & (z_lagged > n)

        signal[long_mask.values] = 1
        signal[short_mask.values] = -1
        bars[col] = signal

        counts = pd.Series(signal).value_counts().to_dict()
        print(f"   {col}: {counts}")

    # Split test set
    print("\n5. Running backtest...")
    test_start = pd.Timestamp("2025-03-01", tz="UTC")
    test_df = bars[bars['timestamp'] >= test_start].copy()
    print(f"   Test bars: {len(test_df):,}")

    cfg = BacktestConfig()

    print("\n" + "=" * 60)
    print("RESULTS: 1-SECOND BARS OU MEAN-REVERSION")
    print("=" * 60)

    all_trades = []

    for n in [1.5, 2.0, 2.5]:
        signal_col = f'signal_{str(n).replace(".", "_")}'
        trades = run_backtest(test_df, signal_col, cfg)
        m = compute_metrics(trades)

        gates = (
            m["total_net_pnl"] > 0 and
            m["pf"] >= 1.1 and
            m["avg_ticks"] >= 1.0 and
            m["num_trades"] >= 30
        )

        print(f"\nN = {n}:")
        print(f"  Trades:        {m['num_trades']}")
        print(f"  Net P&L:       ${m['total_net_pnl']:.2f}")
        print(f"  Win Rate:      {m['win_rate']:.1f}%")
        print(f"  Avg Ticks:     {m['avg_ticks']:.2f}")
        print(f"  Profit Factor: {m['pf']:.2f}")
        print(f"  GO/NO-GO:      {'GO ✓' if gates else 'NO-GO'}")

        for t in trades:
            t['n_value'] = n
            all_trades.append(t)

    # Save results
    if all_trades:
        results_df = pd.DataFrame(all_trades)
        results_df.to_csv('results/backtest_1s_bars_results.csv', index=False)
        print(f"\nSaved {len(all_trades)} trades to results/backtest_1s_bars_results.csv")

    print("\n" + "=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
