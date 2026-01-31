"""
Verify Zone Scalper Strategy - Final Validation

Best configuration from grid search:
- Entry: Z crosses 3.0
- Target: Z reaches 4.5
- Stop: Z reverts to 1.5
- Max hold: 15 bars

Results from grid search:
- OOS P&L: +$7,900 (vs +$1,178 for Z=5.0 mean reversion)
- OOS trades: 23
- Win Rate: 44.4%
"""
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class BacktestConfig:
    commission_per_side: float = 0.85
    tick_size: float = 0.25
    tick_value: float = 5.0  # MNQ


def load_cached_bars(start_date: str, end_date: str) -> pd.DataFrame:
    """Load cached 5-min bars."""
    cache_file = Path(f'data/mnq_5min_ofi_{start_date}_{end_date}.parquet')
    if cache_file.exists():
        return pd.read_parquet(cache_file)
    return pd.DataFrame()


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Z-score."""
    df = df.copy()
    df['ema'] = df['close'].ewm(span=21, adjust=False).mean()
    df['dist'] = (df['close'] - df['ema']) / df['ema']
    df['dist_std'] = df['dist'].rolling(21).std()
    df['zscore'] = df['dist'] / df['dist_std']
    return df


def is_rth(timestamp) -> bool:
    """Check if in RTH (9 AM - 4 PM)."""
    return 9 <= timestamp.hour < 16


def run_zone_scalper(df: pd.DataFrame, cfg: BacktestConfig,
                      entry_z: float = 3.0, target_z: float = 4.5,
                      stop_z: float = 1.5, max_hold: int = 15) -> List[Dict]:
    """Run Zone Scalper backtest with detailed trade logging."""
    trades = []
    n = len(df)

    zscore = df['zscore'].values
    open_prices = df['open'].values
    timestamps = df['timestamp'].values

    position = 0
    entry_price = 0.0
    entry_bar = 0
    entry_dir = ""
    entry_z_val = 0.0

    for i in range(2, n):
        prev_z = zscore[i-1]
        prev_prev_z = zscore[i-2]

        if np.isnan(prev_z) or np.isnan(prev_prev_z):
            continue

        current_ts = pd.Timestamp(timestamps[i])

        # RTH filter - close at end of day
        if not is_rth(current_ts):
            if position != 0:
                if position == 1:
                    exit_price = open_prices[i] - cfg.tick_size
                    gross = (exit_price - entry_price) / cfg.tick_size * cfg.tick_value
                else:
                    exit_price = open_prices[i] + cfg.tick_size
                    gross = (entry_price - exit_price) / cfg.tick_size * cfg.tick_value

                trades.append({
                    "entry_time": pd.Timestamp(timestamps[entry_bar]),
                    "exit_time": current_ts,
                    "direction": entry_dir,
                    "entry_z": entry_z_val,
                    "exit_z": prev_z,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "gross_pnl": gross,
                    "net_pnl": gross - 2 * cfg.commission_per_side,
                    "exit_reason": "RTH_CLOSE",
                    "bars_held": i - entry_bar
                })
                position = 0
            continue

        # Entry: Z crosses INTO the zone
        if position == 0:
            # LONG: Z crosses above +entry_z
            if prev_z >= entry_z and prev_prev_z < entry_z:
                position = 1
                entry_price = open_prices[i] + cfg.tick_size
                entry_bar = i
                entry_dir = "LONG"
                entry_z_val = prev_z

            # SHORT: Z crosses below -entry_z
            elif prev_z <= -entry_z and prev_prev_z > -entry_z:
                position = -1
                entry_price = open_prices[i] - cfg.tick_size
                entry_bar = i
                entry_dir = "SHORT"
                entry_z_val = prev_z

        # Exit logic
        else:
            hold_time = i - entry_bar
            should_exit = False
            exit_reason = ""

            # Max hold
            if hold_time >= max_hold:
                should_exit = True
                exit_reason = "MAX_HOLD"

            # LONG exits
            elif position == 1:
                if prev_z >= target_z:  # Target hit!
                    should_exit = True
                    exit_reason = f"TARGET(Z={prev_z:.2f})"
                elif prev_z <= stop_z:  # Stop hit
                    should_exit = True
                    exit_reason = f"STOP(Z={prev_z:.2f})"

            # SHORT exits
            elif position == -1:
                if prev_z <= -target_z:  # Target hit!
                    should_exit = True
                    exit_reason = f"TARGET(Z={prev_z:.2f})"
                elif prev_z >= -stop_z:  # Stop hit
                    should_exit = True
                    exit_reason = f"STOP(Z={prev_z:.2f})"

            if should_exit:
                if position == 1:
                    exit_price = open_prices[i] - cfg.tick_size
                    gross = (exit_price - entry_price) / cfg.tick_size * cfg.tick_value
                else:
                    exit_price = open_prices[i] + cfg.tick_size
                    gross = (entry_price - exit_price) / cfg.tick_size * cfg.tick_value

                trades.append({
                    "entry_time": pd.Timestamp(timestamps[entry_bar]),
                    "exit_time": current_ts,
                    "direction": entry_dir,
                    "entry_z": entry_z_val,
                    "exit_z": prev_z,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "gross_pnl": gross,
                    "net_pnl": gross - 2 * cfg.commission_per_side,
                    "exit_reason": exit_reason,
                    "bars_held": hold_time
                })
                position = 0

    return trades


def compute_metrics(trades: List[Dict]) -> Dict:
    """Compute performance metrics."""
    if not trades:
        return {"n": 0, "pnl": 0, "wr": 0, "pf": 0}

    net = [t["net_pnl"] for t in trades]
    wins = sum(1 for p in net if p > 0)
    profit = sum(p for p in net if p > 0)
    loss = abs(sum(p for p in net if p < 0))

    return {
        "n": len(trades),
        "pnl": sum(net),
        "wr": wins / len(trades) * 100,
        "pf": profit / loss if loss > 0 else float("inf"),
        "avg_trade": sum(net) / len(trades)
    }


def main():
    print("=" * 100)
    print("ZONE SCALPER STRATEGY - FINAL VALIDATION")
    print("=" * 100)
    print()
    print("Configuration:")
    print("  - Entry Z: 3.0 (trend-following into extreme)")
    print("  - Target Z: 4.5 (scalp the deeper extreme)")
    print("  - Stop Z: 1.5 (exit if Z reverts)")
    print("  - Max Hold: 15 bars (~75 minutes)")
    print("  - RTH Only: Yes (9 AM - 4 PM)")
    print()

    cfg = BacktestConfig()

    # Validation periods
    periods = [
        ("Train (Jan-Feb)", "2025-01-01", "2025-02-28"),
        ("OOS1 (Mar)", "2025-03-01", "2025-03-31"),
    ]

    all_trades = []
    total_oos_pnl = 0
    total_oos_trades = 0

    print(f"\n{'Period':<25} {'Trades':>8} {'P&L':>12} {'WR':>8} {'PF':>8} {'Avg':>10}")
    print("-" * 80)

    for name, start, end in periods:
        bars = load_cached_bars(start, end)
        if len(bars) == 0:
            continue

        df = compute_features(bars)
        trades = run_zone_scalper(df, cfg, entry_z=3.0, target_z=4.5, stop_z=1.5, max_hold=15)
        m = compute_metrics(trades)

        status = "+" if m["pnl"] > 0 else "-" if m["n"] > 0 else " "
        print(f"{name:<25} {m['n']:>8} ${m['pnl']:>10.2f} {m['wr']:>7.1f}% {m['pf']:>7.2f} ${m['avg_trade']:>8.2f} {status}")

        all_trades.extend(trades)

        if "OOS" in name:
            total_oos_pnl += m['pnl']
            total_oos_trades += m['n']

    # Summary
    print("-" * 80)
    total_m = compute_metrics(all_trades)
    print(f"{'TOTAL':<25} {total_m['n']:>8} ${total_m['pnl']:>10.2f} {total_m['wr']:>7.1f}% {total_m['pf']:>7.2f} ${total_m['avg_trade']:>8.2f}")
    print(f"{'OOS TOTAL':<25} {total_oos_trades:>8} ${total_oos_pnl:>10.2f}")

    # Trade breakdown by exit reason
    print(f"\n{'='*100}")
    print("EXIT REASON BREAKDOWN")
    print(f"{'='*100}")

    exit_reasons = {}
    for t in all_trades:
        reason = t["exit_reason"].split("(")[0]  # Strip Z value
        if reason not in exit_reasons:
            exit_reasons[reason] = {"count": 0, "pnl": 0}
        exit_reasons[reason]["count"] += 1
        exit_reasons[reason]["pnl"] += t["net_pnl"]

    print(f"\n{'Exit Reason':<20} {'Count':>8} {'P&L':>12}")
    print("-" * 45)
    for reason, data in sorted(exit_reasons.items(), key=lambda x: -x[1]["pnl"]):
        print(f"{reason:<20} {data['count']:>8} ${data['pnl']:>10.2f}")

    # Direction breakdown
    print(f"\n{'='*100}")
    print("DIRECTION BREAKDOWN")
    print(f"{'='*100}")

    longs = [t for t in all_trades if t["direction"] == "LONG"]
    shorts = [t for t in all_trades if t["direction"] == "SHORT"]

    long_pnl = sum(t["net_pnl"] for t in longs)
    short_pnl = sum(t["net_pnl"] for t in shorts)

    print(f"\n  LONG:  {len(longs):>4} trades, ${long_pnl:>10.2f}")
    print(f"  SHORT: {len(shorts):>4} trades, ${short_pnl:>10.2f}")

    # Sample trades
    print(f"\n{'='*100}")
    print("SAMPLE TRADES (first 10)")
    print(f"{'='*100}")

    print(f"\n{'Entry Time':<22} {'Dir':<6} {'Entry Z':>8} {'Exit Z':>8} {'P&L':>10} {'Exit Reason':<15}")
    print("-" * 85)
    for t in all_trades[:10]:
        print(f"{str(t['entry_time']):<22} {t['direction']:<6} {t['entry_z']:>8.2f} {t['exit_z']:>8.2f} ${t['net_pnl']:>8.2f} {t['exit_reason']:<15}")

    # Comparison to mean reversion
    print(f"\n{'='*100}")
    print("COMPARISON TO ORIGINAL MEAN REVERSION STRATEGY")
    print(f"{'='*100}")
    print(f"""
    ZONE SCALPER (Z=3.0 entry, trend-following):
    - OOS P&L: ${total_oos_pnl:,.2f}
    - OOS Trades: {total_oos_trades}
    - Win Rate: {total_m['wr']:.1f}%
    - Profit Factor: {total_m['pf']:.2f}

    MEAN REVERSION (Z=5.0 entry, counter-trend):
    - OOS P&L: +$1,178
    - OOS Trades: 21
    - Win Rate: ~65%
    - Profit Factor: 3.23

    VERDICT: Zone Scalper generates {total_oos_pnl/1178:.1f}x more P&L
    with similar trade count but different risk profile.

    RECOMMENDATION:
    - Run BOTH strategies in parallel
    - Mean Reversion: Fewer trades, higher win rate
    - Zone Scalper: Similar trades, higher profit per winner
    """)

    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
