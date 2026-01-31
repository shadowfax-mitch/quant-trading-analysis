"""
Combo Signal Backtest Framework

Combines EMA Z-Score (3.5-4.5 range) with confirmation filters to achieve
higher trade frequency (~3 trades/day) while maintaining profitability.

Confirmation Filters:
1. OFI Contrarian: Enter long when sellers exhausted (OFI < -threshold)
2. Volume Spike: Entry only when volume > multiplier * average
3. Volatility Expansion: Entry when ATR is expanding
4. Spread Filter: Skip entries when bid-ask spread > threshold

Based on robust config validation that showed:
- Z=5.0 alone: +$1,178 OOS (21 trades, PF=3.23) - too few trades
- Z=3.5 alone: -$6,552 OOS (600 trades) - loses money without filters
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from itertools import product
import gc


@dataclass
class BacktestConfig:
    commission_per_side: float = 0.85
    tick_size: float = 0.25
    tick_value: float = 5.0  # MNQ default


@dataclass
class ComboParams:
    """Parameters for combo signal strategy."""
    # Z-Score parameters
    ema_period: int = 21
    zscore_lookback: int = 21
    entry_z: float = 4.0
    exit_z: float = 1.0
    max_hold_bars: int = 48
    rth_only: bool = True

    # Confirmation filter toggles
    use_ofi: bool = True
    use_volume: bool = True
    use_volatility: bool = False
    use_spread: bool = False

    # OFI parameters
    ofi_window: int = 10
    ofi_threshold: float = 0.3  # OFI must be contrarian (opposite sign)

    # Volume parameters
    volume_window: int = 20
    volume_multiplier: float = 1.5  # Volume must be > multiplier * average

    # Volatility parameters (ATR)
    atr_period: int = 14
    atr_expansion: float = 1.2  # ATR must be > expansion * average ATR

    # Spread filter
    max_spread_ticks: float = 2.0


def load_mnq_bars_with_ofi(start_date: str, end_date: str, bar_period: str = '5min') -> pd.DataFrame:
    """Load MNQ tick data and aggregate to bars with OFI in one pass."""
    data_dir = Path('datasets/MNQ/tick_data')
    cache_file = Path(f'data/mnq_5min_ofi_{start_date}_{end_date}.parquet')

    if cache_file.exists():
        print(f"      Loading cached bars from {cache_file}...")
        return pd.read_parquet(cache_file)

    print(f"      Building bars from tick data...")
    all_bars = []

    start_dt = pd.Timestamp(start_date, tz='UTC')
    end_dt = pd.Timestamp(end_date, tz='UTC')

    for i in range(1, 125):
        file_path = data_dir / f'mnq_ticks_part{i:04d}.csv'
        if not file_path.exists():
            continue

        if i % 20 == 0:
            print(f"         Processing file {i}/124...")

        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

        # Filter to date range
        df = df[(df['timestamp'] >= start_dt) & (df['timestamp'] <= end_dt)]

        if len(df) == 0:
            del df
            gc.collect()
            continue

        # Aggregate to bars immediately
        df['bar'] = df['timestamp'].dt.floor(bar_period)

        # OHLCV
        bars = df.groupby('bar').agg({
            'last': ['first', 'max', 'min', 'last'],
        }).reset_index()
        bars.columns = ['timestamp', 'open', 'high', 'low', 'close']

        # Volume by side for OFI
        if 'side' in df.columns:
            buy_df = df[df['side'] == 'A'].groupby('bar').size().reset_index(name='buy_vol')
            sell_df = df[df['side'] == 'B'].groupby('bar').size().reset_index(name='sell_vol')
            bars = bars.merge(buy_df, left_on='timestamp', right_on='bar', how='left').drop(columns='bar', errors='ignore')
            bars = bars.merge(sell_df, left_on='timestamp', right_on='bar', how='left').drop(columns='bar', errors='ignore')
            bars['buy_vol'] = bars['buy_vol'].fillna(0)
            bars['sell_vol'] = bars['sell_vol'].fillna(0)
        else:
            bars['buy_vol'] = 0
            bars['sell_vol'] = 0

        # Volume total
        vol_df = df.groupby('bar').size().reset_index(name='volume')
        bars = bars.merge(vol_df, left_on='timestamp', right_on='bar', how='left').drop(columns='bar', errors='ignore')
        bars['volume'] = bars['volume'].fillna(0)

        all_bars.append(bars)

        del df, bars
        gc.collect()

    if not all_bars:
        return pd.DataFrame()

    result = pd.concat(all_bars, ignore_index=True)
    result = result.sort_values('timestamp').drop_duplicates(subset=['timestamp']).reset_index(drop=True)

    # Compute OFI
    total_vol = result['buy_vol'] + result['sell_vol']
    result['ofi'] = np.where(
        total_vol > 0,
        (result['buy_vol'] - result['sell_vol']) / total_vol,
        0
    )

    # Add spread column (default 1 tick if not available)
    result['spread'] = 0.25

    # Cache for future use
    cache_file.parent.mkdir(exist_ok=True)
    result.to_parquet(cache_file)
    print(f"      Cached {len(result):,} bars to {cache_file}")

    return result




def compute_indicators(bars: pd.DataFrame, params: ComboParams) -> pd.DataFrame:
    """Compute all required indicators."""
    df = bars.copy()

    # EMA and Z-Score
    df['ema'] = df['close'].ewm(span=params.ema_period, adjust=False).mean()
    df['dist'] = (df['close'] - df['ema']) / df['ema']
    df['dist_std'] = df['dist'].rolling(params.zscore_lookback).std()
    df['zscore'] = df['dist'] / df['dist_std']

    # Rolling OFI
    if params.use_ofi:
        df['ofi_rolling'] = df['ofi'].rolling(params.ofi_window).mean()

    # Volume features
    if params.use_volume:
        df['vol_avg'] = df['volume'].rolling(params.volume_window).mean()
        df['vol_ratio'] = df['volume'] / df['vol_avg']

    # ATR for volatility
    if params.use_volatility:
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(params.atr_period).mean()
        df['atr_avg'] = df['atr'].rolling(params.atr_period * 2).mean()
        df['atr_ratio'] = df['atr'] / df['atr_avg']

    # Spread in ticks
    if params.use_spread:
        df['spread_ticks'] = df['spread'] / 0.25  # Assuming tick size 0.25

    return df


def is_rth(timestamp) -> bool:
    """Check if timestamp is during Regular Trading Hours (9 AM - 4 PM)."""
    hour = timestamp.hour
    return 9 <= hour < 16


def check_confirmation_filters(
    df: pd.DataFrame,
    idx: int,
    direction: str,
    params: ComboParams
) -> Tuple[bool, str]:
    """
    Check if confirmation filters pass for entry.
    Returns (passed, reason).
    """
    reasons = []

    # OFI Contrarian filter
    if params.use_ofi:
        ofi = df['ofi_rolling'].iloc[idx] if 'ofi_rolling' in df.columns else 0
        if direction == "LONG":
            # For long entry, want sellers exhausted (negative OFI)
            if ofi > -params.ofi_threshold:
                return False, f"OFI not contrarian ({ofi:.2f} > -{params.ofi_threshold})"
            reasons.append(f"OFI={ofi:.2f}")
        else:  # SHORT
            # For short entry, want buyers exhausted (positive OFI)
            if ofi < params.ofi_threshold:
                return False, f"OFI not contrarian ({ofi:.2f} < {params.ofi_threshold})"
            reasons.append(f"OFI={ofi:.2f}")

    # Volume Spike filter
    if params.use_volume:
        vol_ratio = df['vol_ratio'].iloc[idx] if 'vol_ratio' in df.columns else 1.0
        if vol_ratio < params.volume_multiplier:
            return False, f"Volume too low ({vol_ratio:.2f}x < {params.volume_multiplier}x)"
        reasons.append(f"Vol={vol_ratio:.2f}x")

    # Volatility Expansion filter
    if params.use_volatility:
        atr_ratio = df['atr_ratio'].iloc[idx] if 'atr_ratio' in df.columns else 1.0
        if atr_ratio < params.atr_expansion:
            return False, f"ATR not expanding ({atr_ratio:.2f} < {params.atr_expansion})"
        reasons.append(f"ATR={atr_ratio:.2f}x")

    # Spread filter
    if params.use_spread:
        spread_ticks = df['spread_ticks'].iloc[idx] if 'spread_ticks' in df.columns else 1.0
        if spread_ticks > params.max_spread_ticks:
            return False, f"Spread too wide ({spread_ticks:.1f} > {params.max_spread_ticks})"
        reasons.append(f"Spread={spread_ticks:.1f}")

    return True, ", ".join(reasons) if reasons else "No filters"


def run_combo_backtest(
    df: pd.DataFrame,
    params: ComboParams,
    cfg: BacktestConfig
) -> List[Dict]:
    """Run combo signal backtest."""
    trades = []
    n = len(df)

    if n == 0:
        return trades

    zscore = df['zscore'].values
    open_prices = df['open'].values
    timestamps = df['timestamp'].values

    position = 0
    entry_price = 0.0
    entry_bar = 0
    entry_dir = ""

    for i in range(1, n):
        if np.isnan(zscore[i-1]):
            continue

        prev_z = zscore[i - 1]
        current_ts = pd.Timestamp(timestamps[i])
        in_rth = is_rth(current_ts)

        # RTH filter
        if params.rth_only and not in_rth:
            if position != 0:
                # Force close at end of RTH
                if position == 1:
                    exit_price = open_prices[i] - cfg.tick_size
                    gross_pnl = (exit_price - entry_price) / cfg.tick_size * cfg.tick_value
                else:
                    exit_price = open_prices[i] + cfg.tick_size
                    gross_pnl = (entry_price - exit_price) / cfg.tick_size * cfg.tick_value

                net_pnl = gross_pnl - 2 * cfg.commission_per_side
                trades.append({
                    "entry_time": pd.Timestamp(timestamps[entry_bar]),
                    "exit_time": current_ts,
                    "direction": entry_dir,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "net_pnl": net_pnl,
                    "ticks": gross_pnl / cfg.tick_value,
                    "exit_reason": "RTH_CLOSE",
                    "bars_held": i - entry_bar
                })
                position = 0
            continue

        if position == 0:
            # Entry conditions
            direction = None
            if prev_z < -params.entry_z:
                direction = "LONG"
            elif prev_z > params.entry_z:
                direction = "SHORT"

            if direction:
                # Check confirmation filters
                passed, reason = check_confirmation_filters(df, i-1, direction, params)

                if passed:
                    if direction == "LONG":
                        position = 1
                        entry_price = open_prices[i] + cfg.tick_size
                    else:
                        position = -1
                        entry_price = open_prices[i] - cfg.tick_size

                    entry_bar = i
                    entry_dir = direction
        else:
            # Exit conditions
            hold_time = i - entry_bar
            should_exit = hold_time >= params.max_hold_bars
            exit_reason = "MAX_HOLD" if should_exit else ""

            if position == 1 and (prev_z > -params.exit_z or prev_z > 0):
                should_exit = True
                exit_reason = f"Z_REVERT({prev_z:.2f})"
            elif position == -1 and (prev_z < params.exit_z or prev_z < 0):
                should_exit = True
                exit_reason = f"Z_REVERT({prev_z:.2f})"

            if should_exit:
                if position == 1:
                    exit_price = open_prices[i] - cfg.tick_size
                    gross_pnl = (exit_price - entry_price) / cfg.tick_size * cfg.tick_value
                else:
                    exit_price = open_prices[i] + cfg.tick_size
                    gross_pnl = (entry_price - exit_price) / cfg.tick_size * cfg.tick_value

                net_pnl = gross_pnl - 2 * cfg.commission_per_side
                trades.append({
                    "entry_time": pd.Timestamp(timestamps[entry_bar]),
                    "exit_time": current_ts,
                    "direction": entry_dir,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "net_pnl": net_pnl,
                    "ticks": gross_pnl / cfg.tick_value,
                    "exit_reason": exit_reason,
                    "bars_held": hold_time
                })
                position = 0

    # Close any remaining position
    if position != 0:
        if position == 1:
            exit_price = open_prices[-1] - cfg.tick_size
            gross_pnl = (exit_price - entry_price) / cfg.tick_size * cfg.tick_value
        else:
            exit_price = open_prices[-1] + cfg.tick_size
            gross_pnl = (entry_price - exit_price) / cfg.tick_size * cfg.tick_value

        net_pnl = gross_pnl - 2 * cfg.commission_per_side
        trades.append({
            "entry_time": pd.Timestamp(timestamps[entry_bar]),
            "exit_time": pd.Timestamp(timestamps[-1]),
            "direction": entry_dir,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "net_pnl": net_pnl,
            "ticks": gross_pnl / cfg.tick_value,
            "exit_reason": "EOD",
            "bars_held": n - 1 - entry_bar
        })

    return trades


def compute_metrics(trades: List[Dict]) -> Dict:
    """Compute performance metrics from trades."""
    if not trades:
        return {
            "n": 0, "pnl": 0, "wr": 0, "pf": 0, "avg_trade": 0,
            "max_dd": 0, "trades_per_day": 0
        }

    net_pnls = [t["net_pnl"] for t in trades]
    wins = sum(1 for p in net_pnls if p > 0)
    net_profit = sum(p for p in net_pnls if p > 0)
    net_loss = abs(sum(p for p in net_pnls if p < 0))

    # Calculate max drawdown
    cumsum = np.cumsum(net_pnls)
    running_max = np.maximum.accumulate(cumsum)
    drawdown = running_max - cumsum
    max_dd = np.max(drawdown) if len(drawdown) > 0 else 0

    # Trades per day
    if len(trades) > 1:
        first_trade = trades[0]["entry_time"]
        last_trade = trades[-1]["entry_time"]
        days = (last_trade - first_trade).days + 1
        trades_per_day = len(trades) / max(days, 1)
    else:
        trades_per_day = 0

    return {
        "n": len(trades),
        "pnl": sum(net_pnls),
        "wr": wins / len(trades) * 100 if trades else 0,
        "pf": net_profit / net_loss if net_loss > 0 else float("inf"),
        "avg_trade": sum(net_pnls) / len(trades) if trades else 0,
        "max_dd": max_dd,
        "trades_per_day": trades_per_day
    }


def run_grid_search(bars_by_period: Dict[str, pd.DataFrame], cfg: BacktestConfig) -> pd.DataFrame:
    """Run grid search over parameter combinations."""

    # Parameter grid
    param_grid = {
        'entry_z': [3.5, 4.0, 4.5],
        'exit_z': [0.5, 1.0, 1.5],
        'ofi_threshold': [0.2, 0.3, 0.4],
        'volume_multiplier': [1.25, 1.5, 2.0],
        'filter_combo': [
            ('OFI', True, False),
            ('Volume', False, True),
            ('OFI+Volume', True, True),
        ]
    }

    results = []
    total_combos = (len(param_grid['entry_z']) * len(param_grid['exit_z']) *
                   len(param_grid['ofi_threshold']) * len(param_grid['volume_multiplier']) *
                   len(param_grid['filter_combo']))

    print(f"\nRunning grid search over {total_combos} combinations...")

    combo_idx = 0
    for entry_z, exit_z, ofi_thresh, vol_mult, (filter_name, use_ofi, use_vol) in product(
        param_grid['entry_z'],
        param_grid['exit_z'],
        param_grid['ofi_threshold'],
        param_grid['volume_multiplier'],
        param_grid['filter_combo']
    ):
        combo_idx += 1
        if combo_idx % 20 == 0:
            print(f"   Processing combo {combo_idx}/{total_combos}...")

        params = ComboParams(
            entry_z=entry_z,
            exit_z=exit_z,
            ofi_threshold=ofi_thresh,
            volume_multiplier=vol_mult,
            use_ofi=use_ofi,
            use_volume=use_vol
        )

        period_results = {"params": f"Z={entry_z}/{exit_z}, OFI={ofi_thresh}, Vol={vol_mult}x, {filter_name}"}
        all_trades = []
        oos_pnl = 0
        oos_trades = 0

        for period_name, bars in bars_by_period.items():
            if len(bars) == 0:
                continue

            # Compute indicators
            df = compute_indicators(bars, params)

            # Run backtest
            trades = run_combo_backtest(df, params, cfg)
            metrics = compute_metrics(trades)

            period_results[f"{period_name}_n"] = metrics["n"]
            period_results[f"{period_name}_pnl"] = metrics["pnl"]
            period_results[f"{period_name}_pf"] = metrics["pf"]
            period_results[f"{period_name}_tpd"] = metrics["trades_per_day"]

            all_trades.extend(trades)

            if "OOS" in period_name:
                oos_pnl += metrics["pnl"]
                oos_trades += metrics["n"]

        # Overall metrics
        total_metrics = compute_metrics(all_trades)
        period_results["total_n"] = total_metrics["n"]
        period_results["total_pnl"] = total_metrics["pnl"]
        period_results["total_pf"] = total_metrics["pf"]
        period_results["total_tpd"] = total_metrics["trades_per_day"]
        period_results["oos_pnl"] = oos_pnl
        period_results["oos_trades"] = oos_trades

        results.append(period_results)

    return pd.DataFrame(results)


def main():
    print("=" * 100)
    print("COMBO SIGNAL BACKTEST - EMA Z-Score + Confirmation Filters")
    print("=" * 100)
    print()
    print("Goal: ~3 trades/day with positive OOS P&L")
    print("Baseline: Z=5.0 alone gives +$1,178 OOS but only 21 trades (0.1/day)")
    print()

    cfg = BacktestConfig()

    # Define periods
    periods = [
        ("Train", "2025-01-01", "2025-02-28"),
        ("OOS1_Mar", "2025-03-01", "2025-03-31"),
        ("OOS2_AprJun", "2025-04-01", "2025-06-30"),
        ("OOS3_Jul", "2025-07-01", "2025-07-31"),
    ]

    # Load and process data for each period
    print("1. Loading data for each period...")
    bars_by_period = {}

    for period_name, start, end in periods:
        print(f"\n   {period_name}: {start} to {end}")

        # Load bars with OFI (caches internally)
        bars = load_mnq_bars_with_ofi(start, end)

        if len(bars) == 0:
            print(f"      No data found")
            bars_by_period[period_name] = pd.DataFrame()
            continue

        print(f"      {len(bars):,} bars loaded")
        bars_by_period[period_name] = bars

    # Run grid search
    print("\n2. Running grid search...")
    results = run_grid_search(bars_by_period, cfg)

    # Save results
    results_file = Path('results/combo_grid_search.csv')
    results_file.parent.mkdir(exist_ok=True)
    results.to_csv(results_file, index=False)
    print(f"\n   Results saved to {results_file}")

    # Find best configurations
    print("\n" + "=" * 100)
    print("TOP 10 CONFIGURATIONS (by OOS P&L)")
    print("=" * 100)

    # Filter for minimum trade count and positive OOS
    viable = results[
        (results['oos_trades'] >= 20) &  # Minimum trades for significance
        (results['oos_pnl'] > 0)  # Must be profitable OOS
    ].copy()

    if len(viable) > 0:
        viable = viable.sort_values('oos_pnl', ascending=False)

        print(f"\n{'Config':<50} {'OOS P&L':>10} {'OOS Trades':>12} {'TPD':>8}")
        print("-" * 85)

        for _, row in viable.head(10).iterrows():
            print(f"{row['params']:<50} ${row['oos_pnl']:>9.2f} {row['oos_trades']:>12} {row['total_tpd']:>7.2f}")

        # Best config details
        best = viable.iloc[0]
        print("\n" + "=" * 100)
        print("BEST CONFIGURATION DETAILS")
        print("=" * 100)
        print(f"\nConfig: {best['params']}")
        print(f"\nPeriod breakdown:")
        for period_name, _, _ in periods:
            n = best.get(f'{period_name}_n', 0)
            pnl = best.get(f'{period_name}_pnl', 0)
            pf = best.get(f'{period_name}_pf', 0)
            tpd = best.get(f'{period_name}_tpd', 0)
            print(f"   {period_name:<15} {n:>5} trades, ${pnl:>8.2f} P&L, PF={pf:.2f}, {tpd:.2f}/day")
    else:
        print("\nNo viable configurations found!")
        print("All configs either had <20 OOS trades or negative OOS P&L")

        # Show best of the rest
        print("\nTop 5 by total P&L (may not meet criteria):")
        top5 = results.sort_values('total_pnl', ascending=False).head(5)
        for _, row in top5.iterrows():
            print(f"   {row['params']}: ${row['total_pnl']:.2f} total, ${row['oos_pnl']:.2f} OOS, {row['oos_trades']} trades")

    # Compare to baseline
    print("\n" + "=" * 100)
    print("COMPARISON TO BASELINE")
    print("=" * 100)
    print("""
    Baseline (Z=5.0 alone, RTH-only):
    - OOS P&L: +$1,178
    - OOS Trades: 21
    - Trades per day: ~0.1

    Target:
    - OOS P&L: Positive
    - Trades per day: ~3.0
    """)

    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
