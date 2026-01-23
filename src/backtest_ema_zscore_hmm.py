"""
EMA Z-Score Mean Reversion with HMM Regime Filter

Test if adding Hidden Markov Model regime filtering improves
the EMA Z-Score strategy consistency.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from hmmlearn import hmm
import warnings
warnings.filterwarnings('ignore')


@dataclass
class BacktestConfig:
    commission_per_side: float = 0.85
    tick_size: float = 0.25
    tick_value: float = 1.25  # MES


def fit_hmm(bars: pd.DataFrame, n_states: int = 2, features: list = None) -> tuple:
    """
    Fit HMM to market data and return state predictions.

    Returns:
        states: Array of state labels for each bar
        model: Fitted HMM model
    """
    if features is None:
        features = ['returns', 'volatility', 'range_pct']

    # Calculate features
    bars = bars.copy()
    bars['returns'] = bars['close'].pct_change()
    bars['volatility'] = bars['returns'].rolling(20).std()
    bars['range_pct'] = (bars['high'] - bars['low']) / bars['close']
    bars['volume_ma'] = bars.get('volume', bars['close']).rolling(20).mean()

    # Prepare feature matrix
    feature_cols = []
    if 'returns' in features:
        feature_cols.append('returns')
    if 'volatility' in features:
        feature_cols.append('volatility')
    if 'range_pct' in features:
        feature_cols.append('range_pct')

    # Drop NaN rows
    valid_idx = bars.dropna(subset=feature_cols).index
    X = bars.loc[valid_idx, feature_cols].values

    # Fit HMM
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=100,
        random_state=42
    )
    model.fit(X)

    # Predict states for all data
    states = np.full(len(bars), -1)
    states[valid_idx] = model.predict(X)

    return states, model


def analyze_regimes(bars: pd.DataFrame, states: np.ndarray) -> dict:
    """Analyze characteristics of each regime."""
    bars = bars.copy()
    bars['state'] = states
    bars['returns'] = bars['close'].pct_change()
    bars['volatility'] = bars['returns'].rolling(20).std()

    regime_stats = {}
    for state in sorted(bars['state'].unique()):
        if state == -1:
            continue
        state_data = bars[bars['state'] == state]
        regime_stats[state] = {
            'count': len(state_data),
            'pct': len(state_data) / len(bars) * 100,
            'avg_vol': state_data['volatility'].mean(),
            'avg_return': state_data['returns'].mean(),
            'return_std': state_data['returns'].std(),
        }

    return regime_stats


def run_backtest_with_regime(df: pd.DataFrame, zscore: np.ndarray, states: np.ndarray,
                              entry_thresh: float, exit_thresh: float,
                              cfg: BacktestConfig, max_bars: int,
                              allowed_states: list = None) -> list:
    """
    EMA Z-Score backtest with optional regime filter.

    allowed_states: List of state numbers where trading is allowed.
                   If None, trade in all states.
    """
    trades = []
    n = len(df)
    open_prices = df['open'].values
    position = 0
    entry_price = 0.0
    entry_bar = 0

    for i in range(1, n):
        if np.isnan(zscore[i-1]):
            continue

        prev_z = zscore[i - 1]
        current_state = states[i - 1] if i - 1 < len(states) else -1

        # Check if we're in an allowed regime
        in_allowed_regime = (allowed_states is None) or (current_state in allowed_states)

        if position == 0:
            # Only enter if in allowed regime
            if in_allowed_regime:
                if prev_z < -entry_thresh:
                    position = 1
                    entry_price = open_prices[i] + cfg.tick_size
                    entry_bar = i
                elif prev_z > entry_thresh:
                    position = -1
                    entry_price = open_prices[i] - cfg.tick_size
                    entry_bar = i
        else:
            hold_time = i - entry_bar
            should_exit = hold_time >= max_bars

            if position == 1 and (prev_z > -exit_thresh or prev_z > 0):
                should_exit = True
            elif position == -1 and (prev_z < exit_thresh or prev_z < 0):
                should_exit = True

            if should_exit:
                if position == 1:
                    exit_price = open_prices[i] - cfg.tick_size
                    gross_pnl = (exit_price - entry_price) / cfg.tick_size * cfg.tick_value
                else:
                    exit_price = open_prices[i] + cfg.tick_size
                    gross_pnl = (entry_price - exit_price) / cfg.tick_size * cfg.tick_value

                net_pnl = gross_pnl - 2 * cfg.commission_per_side
                trades.append({"net_pnl": net_pnl, "ticks": gross_pnl / cfg.tick_value, "state": current_state})
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
        trades.append({"net_pnl": net_pnl, "ticks": gross_pnl / cfg.tick_value, "state": -1})

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


def check_gates(m: dict, min_trades: int = 20) -> bool:
    return m["pnl"] > 0 and m["pf_net"] >= 1.1 and m["n"] >= min_trades


def main():
    print("=" * 100)
    print("EMA Z-SCORE WITH HMM REGIME FILTER")
    print("=" * 100)
    print()

    # Load data
    print("1. Loading data...")
    df = pd.read_parquet('data/sprint_with_ofi.parquet')
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

    # Aggregate to 5-min bars
    print("2. Aggregating to 5-minute bars...")
    df['bar'] = df['timestamp'].dt.floor('5min')
    df['buy_vol'] = np.where(df['side'] == 'A', df['volume'], 0)
    df['sell_vol'] = np.where(df['side'] == 'B', df['volume'], 0)

    bars = df.groupby('bar').agg({
        'last': ['first', 'max', 'min', 'last'],
        'volume': 'sum',
    }).reset_index()
    bars.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

    print(f"   Total bars: {len(bars):,}")

    # Split periods
    train_end = pd.Timestamp("2025-02-28 23:59:59", tz="UTC")
    test_start = pd.Timestamp("2025-03-01 00:00:00", tz="UTC")

    train_bars = bars[bars['timestamp'] <= train_end].copy()
    test_bars = bars[bars['timestamp'] >= test_start].copy()

    print(f"   Train bars: {len(train_bars):,}")
    print(f"   Test bars: {len(test_bars):,}")

    # Fit HMM on training data
    print("\n3. Fitting HMM on training data...")

    for n_states in [2, 3]:
        print(f"\n{'='*100}")
        print(f"HMM WITH {n_states} STATES")
        print(f"{'='*100}")

        # Fit on train data
        train_states, hmm_model = fit_hmm(train_bars, n_states=n_states)

        # Predict on test data (using trained model)
        test_bars_copy = test_bars.copy()
        test_bars_copy['returns'] = test_bars_copy['close'].pct_change()
        test_bars_copy['volatility'] = test_bars_copy['returns'].rolling(20).std()
        test_bars_copy['range_pct'] = (test_bars_copy['high'] - test_bars_copy['low']) / test_bars_copy['close']

        valid_idx = test_bars_copy.dropna(subset=['returns', 'volatility', 'range_pct']).index
        X_test = test_bars_copy.loc[valid_idx, ['returns', 'volatility', 'range_pct']].values

        test_states = np.full(len(test_bars), -1)
        if len(X_test) > 0:
            test_states[valid_idx - test_bars.index[0]] = hmm_model.predict(X_test)

        # Analyze regimes
        print("\n   REGIME ANALYSIS (Training Data):")
        train_stats = analyze_regimes(train_bars, train_states)
        for state, stats in train_stats.items():
            vol_label = "HIGH VOL" if stats['avg_vol'] > np.mean([s['avg_vol'] for s in train_stats.values()]) else "LOW VOL"
            print(f"   State {state}: {stats['count']:,} bars ({stats['pct']:.1f}%), "
                  f"Avg Vol={stats['avg_vol']*100:.4f}% [{vol_label}]")

        print("\n   REGIME ANALYSIS (Test Data):")
        test_stats = analyze_regimes(test_bars, test_states)
        for state, stats in test_stats.items():
            print(f"   State {state}: {stats['count']:,} bars ({stats['pct']:.1f}%), "
                  f"Avg Vol={stats['avg_vol']*100:.4f}%")

        cfg = BacktestConfig()

        # Test best configs from previous analysis
        best_configs = [
            (34, 21, 3.5, 1.0, 36, "MES Best #1"),
            (34, 21, 3.5, 0.0, 36, "MES Best #2"),
            (21, 21, 3.5, 1.0, 12, "MES Best #7"),
        ]

        print(f"\n   COMPARING FILTERED vs UNFILTERED:")
        print(f"   {'Config':<20} | {'Filter':<12} | {'Tr_N':>5} {'Tr_PnL':>9} {'Tr_PF':>7} | "
              f"{'Te_N':>5} {'Te_PnL':>9} {'Te_PF':>7} | {'Status':>12}")
        print("   " + "-" * 95)

        for ema, z_lb, entry, exit_z, max_b, label in best_configs:
            # Compute Z-score
            bars['ema'] = bars['close'].ewm(span=ema, adjust=False).mean()
            bars['dist'] = (bars['close'] - bars['ema']) / bars['ema']
            bars['dist_std'] = bars['dist'].rolling(z_lb).std()
            bars['zscore'] = bars['dist'] / bars['dist_std']

            train_z = bars['zscore'].values[:len(train_bars)]
            test_z = bars['zscore'].values[-len(test_bars):]

            # Test unfiltered (baseline)
            train_trades_base = run_backtest_with_regime(
                train_bars, train_z, train_states, entry, exit_z, cfg, max_b, allowed_states=None
            )
            test_trades_base = run_backtest_with_regime(
                test_bars, test_z, test_states, entry, exit_z, cfg, max_b, allowed_states=None
            )
            train_m_base = compute_metrics(train_trades_base)
            test_m_base = compute_metrics(test_trades_base)

            both_profit_base = train_m_base['pnl'] > 0 and test_m_base['pnl'] > 0
            status_base = "CONSISTENT" if both_profit_base else ""
            if check_gates(train_m_base, 30) and check_gates(test_m_base, 15):
                status_base = "BOTH GO!"

            print(f"   {label:<20} | {'None':<12} | "
                  f"{train_m_base['n']:>5} ${train_m_base['pnl']:>8.2f} {train_m_base['pf_net']:>7.2f} | "
                  f"{test_m_base['n']:>5} ${test_m_base['pnl']:>8.2f} {test_m_base['pf_net']:>7.2f} | "
                  f"{status_base:>12}")

            # Test each state as filter
            for allowed in range(n_states):
                train_trades_filt = run_backtest_with_regime(
                    train_bars, train_z, train_states, entry, exit_z, cfg, max_b, allowed_states=[allowed]
                )
                test_trades_filt = run_backtest_with_regime(
                    test_bars, test_z, test_states, entry, exit_z, cfg, max_b, allowed_states=[allowed]
                )
                train_m_filt = compute_metrics(train_trades_filt)
                test_m_filt = compute_metrics(test_trades_filt)

                both_profit_filt = train_m_filt['pnl'] > 0 and test_m_filt['pnl'] > 0
                status_filt = "CONSISTENT" if both_profit_filt else ""
                if check_gates(train_m_filt, 20) and check_gates(test_m_filt, 10):
                    status_filt = "BOTH GO!"

                # Mark improvement
                improved = ""
                if both_profit_filt and not both_profit_base:
                    improved = " **NEW**"
                elif both_profit_filt and train_m_filt['pf_net'] > train_m_base['pf_net'] * 1.1:
                    improved = " *BETTER*"

                print(f"   {'':<20} | {'State '+str(allowed):<12} | "
                      f"{train_m_filt['n']:>5} ${train_m_filt['pnl']:>8.2f} {train_m_filt['pf_net']:>7.2f} | "
                      f"{test_m_filt['n']:>5} ${test_m_filt['pnl']:>8.2f} {test_m_filt['pf_net']:>7.2f} | "
                      f"{status_filt:>12}{improved}")

    # Grid search with HMM filter
    print("\n" + "=" * 100)
    print("GRID SEARCH: EMA Z-SCORE WITH HMM FILTER (2 states)")
    print("=" * 100)

    # Use 2-state HMM
    train_states, hmm_model = fit_hmm(train_bars, n_states=2)

    # Predict test states
    test_bars_copy = test_bars.copy()
    test_bars_copy['returns'] = test_bars_copy['close'].pct_change()
    test_bars_copy['volatility'] = test_bars_copy['returns'].rolling(20).std()
    test_bars_copy['range_pct'] = (test_bars_copy['high'] - test_bars_copy['low']) / test_bars_copy['close']
    valid_idx = test_bars_copy.dropna(subset=['returns', 'volatility', 'range_pct']).index
    X_test = test_bars_copy.loc[valid_idx, ['returns', 'volatility', 'range_pct']].values
    test_states = np.full(len(test_bars), -1)
    test_states[valid_idx - test_bars.index[0]] = hmm_model.predict(X_test)

    # Identify which state is "favorable" based on training performance
    # We'll test both and see which works better

    print(f"\n{'EMA':>4} {'Zlb':>4} {'Ent':>5} {'Exit':>5} {'Max':>4} {'State':>6} | "
          f"{'Tr_N':>5} {'Tr_PnL':>9} {'Tr_PF':>7} | "
          f"{'Te_N':>5} {'Te_PnL':>9} {'Te_PF':>7} | {'Status':>12}")
    print("-" * 100)

    results = []

    for ema in [21, 34]:
        for z_lb in [21, 34]:
            bars['ema'] = bars['close'].ewm(span=ema, adjust=False).mean()
            bars['dist'] = (bars['close'] - bars['ema']) / bars['ema']
            bars['dist_std'] = bars['dist'].rolling(z_lb).std()
            bars['zscore'] = bars['dist'] / bars['dist_std']

            train_z = bars['zscore'].values[:len(train_bars)]
            test_z = bars['zscore'].values[-len(test_bars):]

            for entry in [3.0, 3.5]:
                for exit_z in [0.0, 0.5, 1.0]:
                    for max_b in [24, 36]:
                        for allowed in [None, [0], [1]]:
                            train_trades = run_backtest_with_regime(
                                train_bars, train_z, train_states, entry, exit_z, cfg, max_b, allowed
                            )
                            test_trades = run_backtest_with_regime(
                                test_bars, test_z, test_states, entry, exit_z, cfg, max_b, allowed
                            )

                            train_m = compute_metrics(train_trades)
                            test_m = compute_metrics(test_trades)

                            both_profit = train_m['pnl'] > 0 and test_m['pnl'] > 0
                            train_go = check_gates(train_m, 20 if allowed else 30)
                            test_go = check_gates(test_m, 10 if allowed else 15)

                            if both_profit:
                                if train_go and test_go:
                                    status = "BOTH GO!"
                                elif train_go:
                                    status = "Train GO"
                                elif test_go:
                                    status = "Test GO"
                                else:
                                    status = "CONSISTENT"

                                state_str = "All" if allowed is None else f"S{allowed[0]}"
                                print(f"{ema:>4} {z_lb:>4} {entry:>5.1f} {exit_z:>5.1f} {max_b:>4} {state_str:>6} | "
                                      f"{train_m['n']:>5} ${train_m['pnl']:>8.2f} {train_m['pf_net']:>7.2f} | "
                                      f"{test_m['n']:>5} ${test_m['pnl']:>8.2f} {test_m['pf_net']:>7.2f} | "
                                      f"{status:>12}")

                            results.append({
                                "ema": ema, "z_lb": z_lb, "entry": entry, "exit": exit_z,
                                "max": max_b, "state_filter": "All" if allowed is None else f"S{allowed[0]}",
                                "train_pnl": train_m['pnl'], "test_pnl": test_m['pnl'],
                                "train_pf": train_m['pf_net'], "test_pf": test_m['pf_net'],
                                "train_n": train_m['n'], "test_n": test_m['n'],
                                "both_profit": both_profit, "train_go": train_go, "test_go": test_go,
                            })

    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY: HMM FILTER IMPACT")
    print("=" * 100)

    no_filter = [r for r in results if r['state_filter'] == 'All']
    with_filter = [r for r in results if r['state_filter'] != 'All']

    no_filter_consistent = len([r for r in no_filter if r['both_profit']])
    no_filter_both_go = len([r for r in no_filter if r['train_go'] and r['test_go']])

    with_filter_consistent = len([r for r in with_filter if r['both_profit']])
    with_filter_both_go = len([r for r in with_filter if r['train_go'] and r['test_go']])

    print(f"\n{'Metric':<35} {'No Filter':>15} {'With HMM Filter':>15}")
    print("-" * 65)
    print(f"{'Configs tested':<35} {len(no_filter):>15} {len(with_filter):>15}")
    print(f"{'Consistent (both profitable)':<35} {no_filter_consistent:>15} {with_filter_consistent:>15}")
    print(f"{'Both GO':<35} {no_filter_both_go:>15} {with_filter_both_go:>15}")

    # Best filtered configs
    both_go_filtered = [r for r in with_filter if r['train_go'] and r['test_go']]
    if both_go_filtered:
        print("\n*** BEST HMM-FILTERED CONFIGURATIONS ***")
        for r in sorted(both_go_filtered, key=lambda x: -(x['train_pnl'] + x['test_pnl']))[:5]:
            print(f"   EMA={r['ema']}, Zlb={r['z_lb']}, Entry={r['entry']}, Exit={r['exit']}, "
                  f"Max={r['max']}, Filter={r['state_filter']}")
            print(f"      Train: ${r['train_pnl']:.2f} (n={r['train_n']}, PF={r['train_pf']:.2f})")
            print(f"      Test:  ${r['test_pnl']:.2f} (n={r['test_n']}, PF={r['test_pf']:.2f})")

    print("\n" + "=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
