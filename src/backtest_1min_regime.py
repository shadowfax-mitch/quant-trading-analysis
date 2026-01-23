"""
1-Minute Bar OU Backtest with HMM Regime Filter

Uses MarkovRegime class (from Quant-Guild Lecture 74) to filter out
HIGH volatility regimes where mean-reversion fails.
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


class MarkovRegime:
    """
    3-state Hidden Markov Model for volatility regime classification.

    States: 0=LOW, 1=MED, 2=HIGH volatility
    Uses Bayesian filtering with transition probabilities and Gaussian emissions.

    Based on Quant-Guild Lecture 74.
    """

    def __init__(self):
        self.n_states = 3
        self.current_state = 0
        self.state_names = ['LOW', 'MED', 'HIGH']

        # State probability vector
        self.state_probs = np.array([1/3, 1/3, 1/3])

        # Transition matrix: sticky regimes
        self.transition_matrix = np.array([
            [0.90, 0.08, 0.02],  # LOW stays LOW
            [0.10, 0.80, 0.10],  # MED
            [0.02, 0.08, 0.90]   # HIGH stays HIGH
        ])

        # Emission parameters (will be calibrated)
        self.emission_means = np.array([0.001, 0.003, 0.008])
        self.emission_stds = np.array([0.0005, 0.001, 0.003])

    def calibrate(self, volatilities: np.ndarray):
        """
        Calibrate emission parameters from historical volatility data.
        Uses percentile-based regime assignment.
        """
        vols = volatilities[~np.isnan(volatilities)]
        vols = vols[vols > 0]

        if len(vols) < 100:
            print("Warning: Not enough data for calibration")
            return

        # Assign regimes using percentile thresholds
        p33 = np.percentile(vols, 33)
        p67 = np.percentile(vols, 67)

        regime_assignments = np.zeros(len(vols), dtype=int)
        regime_assignments[vols >= p33] = 1
        regime_assignments[vols >= p67] = 2

        # Estimate emission parameters
        for regime in range(self.n_states):
            regime_vols = vols[regime_assignments == regime]
            if len(regime_vols) >= 10:
                self.emission_means[regime] = np.mean(regime_vols)
                self.emission_stds[regime] = max(np.std(regime_vols), 1e-6)

        # Ensure ordering
        sorted_idx = np.argsort(self.emission_means)
        self.emission_means = self.emission_means[sorted_idx]
        self.emission_stds = self.emission_stds[sorted_idx]

        # Estimate transition matrix
        transition_counts = np.zeros((self.n_states, self.n_states))
        for t in range(1, len(regime_assignments)):
            prev = regime_assignments[t-1]
            curr = regime_assignments[t]
            transition_counts[prev, curr] += 1

        # Normalize with Laplace smoothing
        for i in range(self.n_states):
            row_sum = transition_counts[i].sum()
            if row_sum > 0:
                self.transition_matrix[i] = (transition_counts[i] + 0.1) / (row_sum + 0.3)

        # Reset state probabilities
        self.state_probs = np.array([1/3, 1/3, 1/3])

        print(f"Calibrated emission means: {self.emission_means}")
        print(f"Calibrated emission stds: {self.emission_stds}")
        print(f"Percentile thresholds: p33={p33:.6f}, p67={p67:.6f}")

    def _gaussian_likelihood(self, vol: float, regime: int) -> float:
        """Compute P(vol | regime) using Gaussian PDF."""
        mean = self.emission_means[regime]
        std = self.emission_stds[regime]
        coeff = 1 / (std * np.sqrt(2 * np.pi))
        exponent = -0.5 * ((vol - mean) / std) ** 2
        return coeff * np.exp(exponent)

    def update(self, vol: float) -> int:
        """
        Update regime estimate using Bayesian filtering.
        Returns: regime (0=LOW, 1=MED, 2=HIGH)
        """
        if vol <= 0 or np.isnan(vol):
            return self.current_state

        # Prediction step
        prior_probs = self.transition_matrix.T @ self.state_probs

        # Likelihood step
        likelihoods = np.array([
            self._gaussian_likelihood(vol, i) for i in range(self.n_states)
        ])

        # Bayesian update
        posterior_probs = prior_probs * likelihoods
        prob_sum = posterior_probs.sum()

        if prob_sum > 0:
            posterior_probs = posterior_probs / prob_sum
        else:
            posterior_probs = prior_probs

        self.state_probs = posterior_probs
        self.current_state = int(np.argmax(posterior_probs))

        return self.current_state

    def reset(self):
        """Reset state for new sequence."""
        self.state_probs = np.array([1/3, 1/3, 1/3])
        self.current_state = 0


def rolling_ou_params(prices: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate rolling OU mu and sigma."""
    n = len(prices)
    mu = np.full(n, np.nan)
    sigma = np.full(n, np.nan)

    for i in range(window, n):
        x = prices[i-window:i-1]
        y = prices[i-window+1:i]
        n_obs = len(x)

        sum_x, sum_y = x.sum(), y.sum()
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

        mu[i] = a / (1 - b)
        sigma[i] = np.sqrt(sigma_eps2 * 2 * theta_val / denom_sigma)

    return mu, sigma


def run_backtest(df: pd.DataFrame, n_thresh: float, cfg: BacktestConfig,
                 use_regime_filter: bool = False, regime_model: MarkovRegime = None) -> list:
    """Run backtest with optional regime filter."""
    trades = []
    n = len(df)

    close = df['close'].values
    mu_arr = df['mu'].values
    sigma_arr = df['sigma'].values
    bid = df['bid'].values
    ask = df['ask'].values
    vol_arr = df['volatility'].values

    position = 0
    entry_price = 0.0
    entry_regime = 0

    if regime_model:
        regime_model.reset()

    for i in range(1, n):
        # Update regime if using filter
        current_regime = 0
        if use_regime_filter and regime_model:
            current_regime = regime_model.update(vol_arr[i-1])

        # Check for valid OU params
        if np.isnan(mu_arr[i-1]) or np.isnan(sigma_arr[i-1]) or sigma_arr[i-1] <= 0:
            continue

        z = (close[i-1] - mu_arr[i-1]) / sigma_arr[i-1]

        if position == 0:
            # Only enter if not in HIGH regime (when filter is on)
            can_enter = True
            if use_regime_filter and current_regime == 2:  # HIGH volatility
                can_enter = False

            if can_enter:
                if z < -n_thresh:
                    position = 1
                    entry_price = ask[i]
                    entry_regime = current_regime
                elif z > n_thresh:
                    position = -1
                    entry_price = bid[i]
                    entry_regime = current_regime
        else:
            # Check exit
            prev_mu = mu_arr[i-1]
            should_exit = False

            if position == 1 and close[i] >= prev_mu:
                should_exit = True
                exit_price = bid[i]
            elif position == -1 and close[i] <= prev_mu:
                should_exit = True
                exit_price = ask[i]

            if should_exit:
                if position == 1:
                    gross_pnl = (exit_price - entry_price) / cfg.tick_size * cfg.tick_value
                else:
                    gross_pnl = (entry_price - exit_price) / cfg.tick_size * cfg.tick_value

                net_pnl = gross_pnl - 2 * cfg.commission_per_side

                trades.append({
                    "gross_pnl": gross_pnl,
                    "net_pnl": net_pnl,
                    "ticks": gross_pnl / cfg.tick_value,
                    "entry_regime": entry_regime,
                    "exit_regime": current_regime,
                })
                position = 0

    return trades


def compute_metrics(trades: list) -> dict:
    """Compute backtest metrics."""
    if not trades:
        return {
            "num_trades": 0,
            "total_net_pnl": 0,
            "win_rate": 0,
            "avg_ticks": 0,
            "pf": 0,
        }

    net_pnls = [t["net_pnl"] for t in trades]
    gross_pnls = [t["gross_pnl"] for t in trades]
    ticks = [t["ticks"] for t in trades]

    wins = sum(1 for p in net_pnls if p > 0)
    gross_profit = sum(p for p in gross_pnls if p > 0)
    gross_loss = abs(sum(p for p in gross_pnls if p < 0))

    return {
        "num_trades": len(trades),
        "total_net_pnl": sum(net_pnls),
        "win_rate": wins / len(trades) * 100 if trades else 0,
        "avg_ticks": np.mean(ticks) if ticks else 0,
        "pf": gross_profit / gross_loss if gross_loss > 0 else float("inf"),
    }


def run_backtest_regime_subset(df: pd.DataFrame, n_thresh: float, cfg: BacktestConfig,
                                allowed_regimes: set) -> list:
    """Run backtest only on bars where regime is in allowed_regimes."""
    trades = []
    n = len(df)

    close = df['close'].values
    mu_arr = df['mu'].values
    sigma_arr = df['sigma'].values
    bid = df['bid'].values
    ask = df['ask'].values
    regime_arr = df['regime'].values

    position = 0
    entry_price = 0.0
    entry_regime = 0

    for i in range(1, n):
        current_regime = regime_arr[i-1]

        if np.isnan(mu_arr[i-1]) or np.isnan(sigma_arr[i-1]) or sigma_arr[i-1] <= 0:
            continue

        z = (close[i-1] - mu_arr[i-1]) / sigma_arr[i-1]

        if position == 0:
            can_enter = current_regime in allowed_regimes
            if can_enter:
                if z < -n_thresh:
                    position = 1
                    entry_price = ask[i]
                    entry_regime = current_regime
                elif z > n_thresh:
                    position = -1
                    entry_price = bid[i]
                    entry_regime = current_regime
        else:
            prev_mu = mu_arr[i-1]
            should_exit = False

            if position == 1 and close[i] >= prev_mu:
                should_exit = True
                exit_price = bid[i]
            elif position == -1 and close[i] <= prev_mu:
                should_exit = True
                exit_price = ask[i]

            if should_exit:
                if position == 1:
                    gross_pnl = (exit_price - entry_price) / cfg.tick_size * cfg.tick_value
                else:
                    gross_pnl = (entry_price - exit_price) / cfg.tick_size * cfg.tick_value

                net_pnl = gross_pnl - 2 * cfg.commission_per_side

                trades.append({
                    "gross_pnl": gross_pnl,
                    "net_pnl": net_pnl,
                    "ticks": gross_pnl / cfg.tick_value,
                    "entry_regime": entry_regime,
                })
                position = 0

    return trades


def main():
    print("=" * 70)
    print("15-MINUTE BARS + OU: COMPREHENSIVE GRID SEARCH")
    print("=" * 70)

    # Load data
    print("\n1. Loading tick data...")
    df = pd.read_parquet('data/sprint_with_ofi.parquet')
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    print(f"   Ticks: {len(df):,}")

    # Aggregate to 15-minute bars
    print("\n2. Aggregating to 15-minute bars...")
    df['bar'] = df['timestamp'].dt.floor('15min')
    bars = df.groupby('bar').agg({
        'last': ['first', 'max', 'min', 'last'],
        'bid': 'last',
        'ask': 'last',
        'volume': 'sum',
    }).reset_index()
    bars.columns = ['timestamp', 'open', 'high', 'low', 'close', 'bid', 'ask', 'volume']
    bars['volatility'] = (bars['high'] - bars['low']) / bars['close']

    print(f"   15-minute bars: {len(bars):,}")

    # Estimate OU parameters (20-bar window = 5 hours for 15-min bars)
    print("\n3. Estimating OU parameters (20-bar window)...")
    mu, sigma = rolling_ou_params(bars['close'].values, window=20)
    bars['mu'] = mu
    bars['sigma'] = sigma

    # Split train/test
    train_end = pd.Timestamp("2025-02-28", tz="UTC")
    test_start = pd.Timestamp("2025-03-01", tz="UTC")

    train_df = bars[bars['timestamp'] <= train_end].copy()
    test_df = bars[bars['timestamp'] >= test_start].copy()

    print(f"   Train bars: {len(train_df):,}")
    print(f"   Test bars: {len(test_df):,}")

    # Calibrate regime model
    print("\n4. Calibrating HMM regime model...")
    regime_model = MarkovRegime()
    regime_model.calibrate(train_df['volatility'].values)

    # Compute regimes for test set
    regime_model.reset()
    test_regimes = []
    for vol in test_df['volatility'].values:
        regime = regime_model.update(vol)
        test_regimes.append(regime)

    test_df = test_df.copy()
    test_df['regime'] = test_regimes

    regime_counts = pd.Series(test_regimes).value_counts().sort_index()
    print(f"\n   Test regime distribution:")
    print(f"   LOW (0):  {regime_counts.get(0, 0):,} bars ({regime_counts.get(0, 0)/len(test_regimes)*100:.1f}%)")
    print(f"   MED (1):  {regime_counts.get(1, 0):,} bars ({regime_counts.get(1, 0)/len(test_regimes)*100:.1f}%)")
    print(f"   HIGH (2): {regime_counts.get(2, 0):,} bars ({regime_counts.get(2, 0)/len(test_regimes)*100:.1f}%)")

    # Grid search
    cfg = BacktestConfig()
    n_values = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    regime_configs = [
        ("ALL", {0, 1, 2}),
        ("LOW", {0}),
        ("MED", {1}),
        ("HIGH", {2}),
        ("LOW+MED", {0, 1}),
        ("MED+HIGH", {1, 2}),
    ]

    print("\n" + "=" * 70)
    print("GRID SEARCH RESULTS")
    print("=" * 70)
    print(f"{'Regimes':<12} {'N':>5} {'Trades':>7} {'Net P&L':>10} {'WinRate':>8} {'AvgTicks':>9} {'PF':>6} {'GO?':>6}")
    print("-" * 70)

    results = []
    best_result = None

    for regime_name, allowed_regimes in regime_configs:
        for n_thresh in n_values:
            trades = run_backtest_regime_subset(test_df, n_thresh, cfg, allowed_regimes)
            m = compute_metrics(trades)

            gates_passed = (
                m["total_net_pnl"] > 0 and
                m["pf"] >= 1.1 and
                m["avg_ticks"] >= 1.0 and
                m["num_trades"] >= 30
            )

            status = "GO" if gates_passed else ""

            print(f"{regime_name:<12} {n_thresh:>5.1f} {m['num_trades']:>7} ${m['total_net_pnl']:>9.2f} "
                  f"{m['win_rate']:>7.1f}% {m['avg_ticks']:>9.2f} {m['pf']:>6.2f} {status:>6}")

            result = {
                "regime": regime_name,
                "n": n_thresh,
                "num_trades": m["num_trades"],
                "net_pnl": m["total_net_pnl"],
                "win_rate": m["win_rate"],
                "avg_ticks": m["avg_ticks"],
                "pf": m["pf"],
                "go": gates_passed,
            }
            results.append(result)

            # Track best by PF (if enough trades)
            if m["num_trades"] >= 30:
                if best_result is None or m["pf"] > best_result["pf"]:
                    best_result = result

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    go_results = [r for r in results if r["go"]]
    if go_results:
        print(f"\n*** FOUND {len(go_results)} GO CONFIGURATION(S)! ***")
        for r in go_results:
            print(f"   {r['regime']} N={r['n']}: {r['num_trades']} trades, "
                  f"P&L=${r['net_pnl']:.2f}, PF={r['pf']:.2f}")
    else:
        print("\n*** NO CONFIGURATION ACHIEVED GO STATUS ***")

    if best_result:
        print(f"\nBest by Profit Factor (min 30 trades):")
        print(f"   {best_result['regime']} N={best_result['n']}: "
              f"{best_result['num_trades']} trades, P&L=${best_result['net_pnl']:.2f}, PF={best_result['pf']:.2f}")

    # Check if ANY positive P&L exists
    positive_pnl = [r for r in results if r["net_pnl"] > 0 and r["num_trades"] >= 30]
    if positive_pnl:
        print(f"\nConfigurations with positive P&L (min 30 trades):")
        for r in sorted(positive_pnl, key=lambda x: -x["net_pnl"]):
            print(f"   {r['regime']} N={r['n']}: {r['num_trades']} trades, "
                  f"P&L=${r['net_pnl']:.2f}, PF={r['pf']:.2f}")
    else:
        print("\nNo configuration achieved positive P&L with 30+ trades.")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    if go_results:
        print("The OU mean-reversion strategy PASSES the GO gate.")
    else:
        print("The OU mean-reversion strategy does NOT pass GO gates.")
        print("Potential next steps:")
        print("  1. Try longer timeframes (5-min, 15-min bars)")
        print("  2. Consider alternative strategies (momentum, breakout)")
        print("  3. Accept that mean-reversion may not work on this instrument/period")

    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
