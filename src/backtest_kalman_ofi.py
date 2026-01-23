"""
Kalman-Filtered OFI Momentum Strategy

Uses a Kalman filter to:
1. De-noise the raw OFI signal
2. Extract OFI velocity (rate of change)
3. Generate cleaner trading signals

State vector: [ofi, ofi_velocity]
Observation: raw_ofi
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple


@dataclass
class BacktestConfig:
    commission_per_side: float = 0.85
    tick_size: float = 0.25
    tick_value: float = 1.25


class KalmanFilterOFI:
    """
    Kalman filter for OFI signal with position and velocity state.

    State: [ofi, ofi_velocity]
    - ofi: filtered OFI value
    - ofi_velocity: rate of change of OFI
    """

    def __init__(self, process_noise: float = 0.001, obs_noise: float = 0.01):
        """
        Args:
            process_noise: Q - how much we expect OFI to change randomly
            obs_noise: R - how noisy we think the raw OFI measurement is
        """
        # State vector [ofi, velocity]
        self.x = np.array([0.0, 0.0])

        # State covariance
        self.P = np.eye(2) * 0.1

        # State transition matrix (constant velocity model)
        # ofi_new = ofi_old + velocity * dt (dt=1 bar)
        self.F = np.array([
            [1.0, 1.0],  # ofi = ofi + velocity
            [0.0, 1.0]   # velocity = velocity (constant)
        ])

        # Process noise covariance
        self.Q = np.array([
            [process_noise, 0],
            [0, process_noise * 0.1]  # velocity changes slower
        ])

        # Observation matrix (we only observe ofi, not velocity)
        self.H = np.array([[1.0, 0.0]])

        # Observation noise covariance
        self.R = np.array([[obs_noise]])

        self.initialized = False

    def reset(self):
        self.x = np.array([0.0, 0.0])
        self.P = np.eye(2) * 0.1
        self.initialized = False

    def update(self, raw_ofi: float) -> Tuple[float, float]:
        """
        Process one observation and return filtered OFI and velocity.

        Returns:
            (filtered_ofi, ofi_velocity)
        """
        if np.isnan(raw_ofi):
            return np.nan, np.nan

        if not self.initialized:
            self.x[0] = raw_ofi
            self.x[1] = 0.0
            self.initialized = True
            return raw_ofi, 0.0

        # Predict
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # Update
        z = np.array([raw_ofi])
        y = z - self.H @ x_pred  # Innovation
        S = self.H @ P_pred @ self.H.T + self.R  # Innovation covariance
        K = P_pred @ self.H.T @ np.linalg.inv(S)  # Kalman gain

        self.x = x_pred + K @ y
        self.P = (np.eye(2) - K @ self.H) @ P_pred

        return self.x[0], self.x[1]


def run_backtest_kalman(df: pd.DataFrame, raw_ofi: np.ndarray,
                        process_noise: float, obs_noise: float,
                        ofi_thresh: float, vel_thresh: float,
                        cfg: BacktestConfig, exit_bars: int,
                        use_velocity: bool = True) -> Tuple[list, np.ndarray, np.ndarray]:
    """
    Kalman-filtered OFI momentum backtest.

    Entry signals:
    - LONG: filtered_ofi > ofi_thresh AND (velocity > vel_thresh if use_velocity)
    - SHORT: filtered_ofi < -ofi_thresh AND (velocity < -vel_thresh if use_velocity)
    """
    trades = []
    n = len(df)

    open_prices = df['open'].values

    # Run Kalman filter
    kf = KalmanFilterOFI(process_noise=process_noise, obs_noise=obs_noise)
    filtered_ofi = np.full(n, np.nan)
    ofi_velocity = np.full(n, np.nan)

    for i in range(n):
        if not np.isnan(raw_ofi[i]):
            filtered_ofi[i], ofi_velocity[i] = kf.update(raw_ofi[i])

    position = 0
    entry_price = 0.0
    entry_bar = 0

    for i in range(1, n):
        if np.isnan(filtered_ofi[i-1]):
            continue

        prev_ofi = filtered_ofi[i - 1]
        prev_vel = ofi_velocity[i - 1] if use_velocity else 0

        if position == 0:
            # Entry conditions
            long_signal = prev_ofi > ofi_thresh
            short_signal = prev_ofi < -ofi_thresh

            if use_velocity:
                long_signal = long_signal and prev_vel > vel_thresh
                short_signal = short_signal and prev_vel < -vel_thresh

            if long_signal:
                position = 1
                entry_price = open_prices[i] + cfg.tick_size
                entry_bar = i
            elif short_signal:
                position = -1
                entry_price = open_prices[i] - cfg.tick_size
                entry_bar = i
        else:
            hold_time = i - entry_bar
            should_exit = hold_time >= exit_bars

            # Exit on signal reversal
            if position == 1 and prev_ofi < -ofi_thresh:
                should_exit = True
            elif position == -1 and prev_ofi > ofi_thresh:
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

    return trades, filtered_ofi, ofi_velocity


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
    print("=" * 95)
    print("KALMAN-FILTERED OFI MOMENTUM STRATEGY")
    print("=" * 95)
    print("Uses Kalman filter to de-noise OFI and extract velocity.")
    print()

    # Load data
    print("1. Loading data...")
    df = pd.read_parquet('data/sprint_with_ofi.parquet')
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

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

    # Compute raw OFI
    window = 10
    roll_buy = bars['buy_vol'].rolling(window, min_periods=window).sum()
    roll_sell = bars['sell_vol'].rolling(window, min_periods=window).sum()
    raw_ofi = ((roll_buy - roll_sell) / (roll_buy + roll_sell)).replace([np.inf, -np.inf], np.nan).values

    # Split periods
    train_end = pd.Timestamp("2025-02-28 23:59:59", tz="UTC")
    test_start = pd.Timestamp("2025-03-01 00:00:00", tz="UTC")

    train_bars = bars[bars['timestamp'] <= train_end].copy()
    test_bars = bars[bars['timestamp'] >= test_start].copy()

    train_ofi = raw_ofi[:len(train_bars)]
    test_ofi = raw_ofi[-len(test_bars):]

    print(f"   Train bars: {len(train_bars):,}")
    print(f"   Test bars: {len(test_bars):,}")

    cfg = BacktestConfig()

    # ==========================================================================
    # Test 1: Kalman-filtered OFI only (no velocity requirement)
    # ==========================================================================
    print("\n" + "=" * 95)
    print("TEST 1: KALMAN-FILTERED OFI (no velocity requirement)")
    print("=" * 95)
    print(f"{'P_noise':>8} {'O_noise':>8} {'Thresh':>7} {'Exit':>5} {'Tr_N':>6} {'Tr_PnL':>10} {'Te_N':>6} {'Te_PnL':>10} {'Te_PF':>7} {'Status':>10}")
    print("-" * 95)

    results1 = []

    for process_noise in [0.0001, 0.001, 0.01]:
        for obs_noise in [0.01, 0.05, 0.1]:
            for ofi_thresh in [0.10, 0.15, 0.20]:
                for exit_bars in [20, 30]:
                    # Train
                    train_trades, _, _ = run_backtest_kalman(
                        train_bars, train_ofi, process_noise, obs_noise,
                        ofi_thresh, 0, cfg, exit_bars, use_velocity=False
                    )
                    train_m = compute_metrics(train_trades)

                    # Test
                    test_trades, _, _ = run_backtest_kalman(
                        test_bars, test_ofi, process_noise, obs_noise,
                        ofi_thresh, 0, cfg, exit_bars, use_velocity=False
                    )
                    test_m = compute_metrics(test_trades)

                    test_go = check_gates(test_m)
                    both_profit = train_m['pnl'] > 0 and test_m['pnl'] > 0

                    status = ""
                    if both_profit and test_go:
                        status = "BOTH GO!"
                    elif both_profit:
                        status = "CONSISTENT"
                    elif test_go:
                        status = "Test GO"

                    if train_m['n'] >= 20 or test_m['n'] >= 10:
                        print(f"{process_noise:>8.4f} {obs_noise:>8.2f} {ofi_thresh:>7.2f} {exit_bars:>5} "
                              f"{train_m['n']:>6} ${train_m['pnl']:>9.2f} "
                              f"{test_m['n']:>6} ${test_m['pnl']:>9.2f} {test_m['pf_net']:>7.2f} {status:>10}")

                    results1.append({
                        "p_noise": process_noise, "o_noise": obs_noise,
                        "ofi_thresh": ofi_thresh, "exit": exit_bars,
                        "train_n": train_m['n'], "train_pnl": train_m['pnl'],
                        "test_n": test_m['n'], "test_pnl": test_m['pnl'],
                        "test_pf": test_m['pf_net'], "test_go": test_go,
                        "both_profit": both_profit, "use_vel": False
                    })

    # ==========================================================================
    # Test 2: Kalman-filtered OFI + Velocity requirement
    # ==========================================================================
    print("\n" + "=" * 95)
    print("TEST 2: KALMAN-FILTERED OFI + VELOCITY CONFIRMATION")
    print("=" * 95)
    print(f"{'P_noise':>8} {'O_noise':>8} {'OFI_T':>6} {'Vel_T':>6} {'Exit':>5} {'Tr_N':>6} {'Tr_PnL':>10} {'Te_N':>6} {'Te_PnL':>10} {'Te_PF':>7} {'Status':>10}")
    print("-" * 95)

    results2 = []

    for process_noise in [0.001, 0.005, 0.01]:
        for obs_noise in [0.02, 0.05, 0.1]:
            for ofi_thresh in [0.10, 0.15]:
                for vel_thresh in [0.001, 0.005, 0.01]:
                    for exit_bars in [20, 30]:
                        # Train
                        train_trades, _, _ = run_backtest_kalman(
                            train_bars, train_ofi, process_noise, obs_noise,
                            ofi_thresh, vel_thresh, cfg, exit_bars, use_velocity=True
                        )
                        train_m = compute_metrics(train_trades)

                        # Test
                        test_trades, _, _ = run_backtest_kalman(
                            test_bars, test_ofi, process_noise, obs_noise,
                            ofi_thresh, vel_thresh, cfg, exit_bars, use_velocity=True
                        )
                        test_m = compute_metrics(test_trades)

                        test_go = check_gates(test_m)
                        both_profit = train_m['pnl'] > 0 and test_m['pnl'] > 0

                        status = ""
                        if both_profit and test_go:
                            status = "BOTH GO!"
                        elif both_profit:
                            status = "CONSISTENT"
                        elif test_go:
                            status = "Test GO"

                        if train_m['n'] >= 20 or test_m['n'] >= 10:
                            print(f"{process_noise:>8.4f} {obs_noise:>8.2f} {ofi_thresh:>6.2f} {vel_thresh:>6.3f} {exit_bars:>5} "
                                  f"{train_m['n']:>6} ${train_m['pnl']:>9.2f} "
                                  f"{test_m['n']:>6} ${test_m['pnl']:>9.2f} {test_m['pf_net']:>7.2f} {status:>10}")

                        results2.append({
                            "p_noise": process_noise, "o_noise": obs_noise,
                            "ofi_thresh": ofi_thresh, "vel_thresh": vel_thresh,
                            "exit": exit_bars,
                            "train_n": train_m['n'], "train_pnl": train_m['pnl'],
                            "test_n": test_m['n'], "test_pnl": test_m['pnl'],
                            "test_pf": test_m['pf_net'], "test_go": test_go,
                            "both_profit": both_profit, "use_vel": True
                        })

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 95)
    print("SUMMARY")
    print("=" * 95)

    all_results = results1 + results2

    consistent = [r for r in all_results if r['both_profit']]
    test_go_list = [r for r in all_results if r['test_go']]
    both_go = [r for r in all_results if r['both_profit'] and r['test_go']]

    print(f"\nTotal configurations tested: {len(all_results)}")
    print(f"Configs profitable in BOTH periods: {len(consistent)}")
    print(f"Configs with test GO: {len(test_go_list)}")
    print(f"Configs with BOTH profitable AND test GO: {len(both_go)}")

    if both_go:
        print("\n*** VALIDATED GO CONFIGURATIONS (profitable in both + test GO): ***")
        for r in sorted(both_go, key=lambda x: -x['test_pnl']):
            vel_str = f", Vel={r['vel_thresh']}" if r['use_vel'] else ""
            print(f"   P={r['p_noise']}, O={r['o_noise']}, OFI={r['ofi_thresh']}{vel_str}, Exit={r['exit']}: "
                  f"Train=${r['train_pnl']:.2f} (n={r['train_n']}), "
                  f"Test=${r['test_pnl']:.2f} (n={r['test_n']}), PF={r['test_pf']:.2f}")

    if consistent and not both_go:
        print("\n*** CONSISTENT CONFIGURATIONS (profitable in both, but test not GO): ***")
        for r in sorted(consistent, key=lambda x: -(x['train_pnl'] + x['test_pnl']))[:10]:
            vel_str = f", Vel={r['vel_thresh']}" if r['use_vel'] else ""
            print(f"   P={r['p_noise']}, O={r['o_noise']}, OFI={r['ofi_thresh']}{vel_str}, Exit={r['exit']}: "
                  f"Train=${r['train_pnl']:.2f}, Test=${r['test_pnl']:.2f}, PF={r['test_pf']:.2f}")

    print("\n" + "=" * 95)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
