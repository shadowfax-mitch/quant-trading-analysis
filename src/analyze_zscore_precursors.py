"""
Z-Score Runner Analysis

Goal: Identify precursor patterns that predict extreme Z-score events (|Z| > 3.5).
Instead of waiting for Z=5.0 and mean-reverting, we want to catch the trend EARLY
and ride it to the extreme.

Approach:
1. Find all extreme Z-score events (|Z| >= 3.5)
2. Look back to find the START of each move (when Z crossed a threshold)
3. Analyze indicators at the start: volume, velocity, momentum
4. Build rules to identify "Z-score runners" early
"""
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple
import gc


@dataclass
class ExtremeMoveEvent:
    """Represents an extreme Z-score move."""
    peak_time: pd.Timestamp
    peak_z: float
    direction: str  # "UP" or "DOWN"
    start_time: pd.Timestamp
    start_idx: int
    peak_idx: int
    duration_bars: int
    # Precursor features at start
    start_z: float
    start_volume: float
    start_vol_ratio: float  # vs average
    z_velocity: float  # rate of change at start
    z_acceleration: float  # 2nd derivative
    price_velocity: float  # price change rate
    start_ofi: float  # order flow imbalance


def load_cached_bars(start_date: str, end_date: str) -> pd.DataFrame:
    """Load cached 5-min bars. Returns empty DataFrame if not cached."""
    cache_file = Path(f'data/mnq_5min_ofi_{start_date}_{end_date}.parquet')

    if cache_file.exists():
        return pd.read_parquet(cache_file)

    # If not cached, return empty (don't try to build - takes too long)
    print(f"   No cached data found for {start_date} to {end_date}")
    return pd.DataFrame()


def compute_zscore_features(df: pd.DataFrame, ema_period: int = 21, z_lookback: int = 21) -> pd.DataFrame:
    """Compute Z-score and additional velocity/momentum features."""
    df = df.copy()

    # Basic Z-score
    df['ema'] = df['close'].ewm(span=ema_period, adjust=False).mean()
    df['dist'] = (df['close'] - df['ema']) / df['ema']
    df['dist_std'] = df['dist'].rolling(z_lookback).std()
    df['zscore'] = df['dist'] / df['dist_std']

    # Z-score velocity (rate of change)
    df['z_velocity'] = df['zscore'].diff()
    df['z_velocity_5'] = df['zscore'].diff(5) / 5  # Smoother velocity

    # Z-score acceleration
    df['z_acceleration'] = df['z_velocity'].diff()

    # Price velocity
    df['price_velocity'] = df['close'].diff()
    df['price_velocity_5'] = df['close'].diff(5) / 5

    # Volume features
    df['vol_avg_20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_avg_20']
    df['vol_spike'] = df['vol_ratio'] > 1.5

    # OFI rolling average
    if 'ofi' in df.columns:
        df['ofi_rolling'] = df['ofi'].rolling(5).mean()
    else:
        df['ofi_rolling'] = 0

    return df


def find_extreme_events(df: pd.DataFrame, min_peak_z: float = 3.5, start_threshold: float = 0.5) -> List[ExtremeMoveEvent]:
    """
    Find all extreme Z-score events and extract precursor features.

    Args:
        df: DataFrame with Z-score and features
        min_peak_z: Minimum absolute Z-score to qualify as extreme
        start_threshold: Z-score threshold to identify move start
    """
    events = []
    n = len(df)
    zscore = df['zscore'].values

    # Track if we're in an extreme move
    in_move = False
    move_start_idx = 0
    peak_z = 0
    peak_idx = 0
    direction = ""

    for i in range(50, n):  # Skip warmup period
        z = zscore[i]

        if np.isnan(z):
            continue

        # Detect new extreme move starting
        if not in_move:
            # Check if Z just crossed into significant territory
            if z > start_threshold and (i == 0 or zscore[i-1] <= start_threshold):
                in_move = True
                move_start_idx = i
                peak_z = z
                peak_idx = i
                direction = "UP"
            elif z < -start_threshold and (i == 0 or zscore[i-1] >= -start_threshold):
                in_move = True
                move_start_idx = i
                peak_z = z
                peak_idx = i
                direction = "DOWN"
        else:
            # Track peak
            if direction == "UP" and z > peak_z:
                peak_z = z
                peak_idx = i
            elif direction == "DOWN" and z < peak_z:
                peak_z = z
                peak_idx = i

            # Check for move end (reversion toward zero)
            move_ended = False
            if direction == "UP" and z < start_threshold:
                move_ended = True
            elif direction == "DOWN" and z > -start_threshold:
                move_ended = True

            if move_ended:
                # Only record if peak was extreme enough
                if abs(peak_z) >= min_peak_z:
                    # Extract precursor features at move start
                    start_row = df.iloc[move_start_idx]

                    event = ExtremeMoveEvent(
                        peak_time=pd.Timestamp(df['timestamp'].iloc[peak_idx]),
                        peak_z=peak_z,
                        direction=direction,
                        start_time=pd.Timestamp(df['timestamp'].iloc[move_start_idx]),
                        start_idx=move_start_idx,
                        peak_idx=peak_idx,
                        duration_bars=peak_idx - move_start_idx,
                        start_z=start_row['zscore'],
                        start_volume=start_row['volume'],
                        start_vol_ratio=start_row['vol_ratio'] if not np.isnan(start_row['vol_ratio']) else 1.0,
                        z_velocity=start_row['z_velocity'] if not np.isnan(start_row['z_velocity']) else 0,
                        z_acceleration=start_row['z_acceleration'] if not np.isnan(start_row['z_acceleration']) else 0,
                        price_velocity=start_row['price_velocity'] if not np.isnan(start_row['price_velocity']) else 0,
                        start_ofi=start_row['ofi_rolling'] if not np.isnan(start_row['ofi_rolling']) else 0,
                    )
                    events.append(event)

                in_move = False
                peak_z = 0

    return events


def analyze_precursor_patterns(events: List[ExtremeMoveEvent]) -> Dict:
    """Analyze common patterns in precursors to extreme moves."""
    if not events:
        return {}

    up_events = [e for e in events if e.direction == "UP"]
    down_events = [e for e in events if e.direction == "DOWN"]

    def stats(values):
        arr = np.array(values)
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            return {"mean": 0, "std": 0, "min": 0, "max": 0, "p25": 0, "p75": 0}
        return {
            "mean": np.mean(arr),
            "std": np.std(arr),
            "min": np.min(arr),
            "max": np.max(arr),
            "p25": np.percentile(arr, 25),
            "p75": np.percentile(arr, 75),
        }

    analysis = {
        "total_events": len(events),
        "up_events": len(up_events),
        "down_events": len(down_events),

        "all": {
            "peak_z": stats([abs(e.peak_z) for e in events]),
            "duration_bars": stats([e.duration_bars for e in events]),
            "start_vol_ratio": stats([e.start_vol_ratio for e in events]),
            "z_velocity": stats([abs(e.z_velocity) for e in events]),
            "z_acceleration": stats([abs(e.z_acceleration) for e in events]),
            "start_ofi": stats([abs(e.start_ofi) for e in events]),
        },

        "up": {
            "peak_z": stats([e.peak_z for e in up_events]),
            "start_vol_ratio": stats([e.start_vol_ratio for e in up_events]),
            "z_velocity": stats([e.z_velocity for e in up_events]),
            "start_ofi": stats([e.start_ofi for e in up_events]),  # Should be positive for up
        },

        "down": {
            "peak_z": stats([e.peak_z for e in down_events]),
            "start_vol_ratio": stats([e.start_vol_ratio for e in down_events]),
            "z_velocity": stats([e.z_velocity for e in down_events]),
            "start_ofi": stats([e.start_ofi for e in down_events]),  # Should be negative for down
        }
    }

    return analysis


def find_high_probability_patterns(events: List[ExtremeMoveEvent], all_starts: pd.DataFrame) -> Dict:
    """
    Compare extreme events to ALL move starts to find discriminating features.
    This helps identify which factors predict that a move will become extreme.
    """
    # Features of extreme events
    extreme_vol_ratios = [e.start_vol_ratio for e in events if not np.isnan(e.start_vol_ratio)]
    extreme_z_velocities = [abs(e.z_velocity) for e in events if not np.isnan(e.z_velocity)]
    extreme_ofi = [abs(e.start_ofi) for e in events if not np.isnan(e.start_ofi)]

    # Define thresholds based on extreme event statistics
    if extreme_vol_ratios:
        vol_threshold = np.percentile(extreme_vol_ratios, 25)  # 25th percentile of extremes
    else:
        vol_threshold = 1.5

    if extreme_z_velocities:
        vel_threshold = np.percentile(extreme_z_velocities, 25)
    else:
        vel_threshold = 0.1

    if extreme_ofi:
        ofi_threshold = np.percentile(extreme_ofi, 25)
    else:
        ofi_threshold = 0.1

    # Count how many extreme events meet each criteria
    vol_hits = sum(1 for e in events if e.start_vol_ratio >= vol_threshold)
    vel_hits = sum(1 for e in events if abs(e.z_velocity) >= vel_threshold)
    ofi_hits = sum(1 for e in events if abs(e.start_ofi) >= ofi_threshold)
    combined_hits = sum(1 for e in events
                       if e.start_vol_ratio >= vol_threshold
                       and abs(e.z_velocity) >= vel_threshold)

    return {
        "thresholds": {
            "volume_ratio": vol_threshold,
            "z_velocity": vel_threshold,
            "ofi": ofi_threshold,
        },
        "hit_rates": {
            "volume_alone": vol_hits / len(events) * 100 if events else 0,
            "velocity_alone": vel_hits / len(events) * 100 if events else 0,
            "ofi_alone": ofi_hits / len(events) * 100 if events else 0,
            "volume_and_velocity": combined_hits / len(events) * 100 if events else 0,
        }
    }


def main():
    print("=" * 100)
    print("Z-SCORE RUNNER ANALYSIS - Finding Precursor Patterns")
    print("=" * 100)
    print()
    print("Goal: Find indicators that predict extreme Z-score moves EARLY")
    print("      so we can ride the trend rather than wait for extremes.")
    print()

    # Load data from cached parquet files only (already built)
    periods = [
        ("2025-01-01", "2025-02-28"),  # Train
        ("2025-03-01", "2025-03-31"),  # OOS1
        # Skip Apr-Jun and Jul - no cached data yet
    ]

    all_events = []

    for start, end in periods:
        print(f"\n1. Loading data for {start} to {end}...")
        bars = load_cached_bars(start, end)

        if len(bars) == 0:
            print("   No data found, skipping...")
            continue

        print(f"   {len(bars):,} bars loaded")

        print("2. Computing Z-score and features...")
        df = compute_zscore_features(bars)

        print("3. Finding extreme Z-score events...")
        events = find_extreme_events(df, min_peak_z=3.5, start_threshold=0.5)
        print(f"   Found {len(events)} extreme events")

        all_events.extend(events)

    print(f"\n{'='*100}")
    print(f"TOTAL EXTREME EVENTS: {len(all_events)}")
    print(f"{'='*100}")

    if not all_events:
        print("\nNo extreme events found. Try lowering min_peak_z threshold.")
        return 1

    # Analyze precursor patterns
    print("\n4. Analyzing precursor patterns...")
    analysis = analyze_precursor_patterns(all_events)

    print(f"\n{'='*100}")
    print("PRECURSOR PATTERN ANALYSIS")
    print(f"{'='*100}")

    print(f"\nTotal extreme events: {analysis['total_events']}")
    print(f"  - UP moves (Z > 3.5):   {analysis['up_events']}")
    print(f"  - DOWN moves (Z < -3.5): {analysis['down_events']}")

    print(f"\n--- ALL EVENTS (combined) ---")
    print(f"\nPeak Z-Score reached:")
    s = analysis['all']['peak_z']
    print(f"  Mean: {s['mean']:.2f}, Std: {s['std']:.2f}")
    print(f"  Range: {s['min']:.2f} to {s['max']:.2f}")

    print(f"\nDuration (bars from start to peak):")
    s = analysis['all']['duration_bars']
    print(f"  Mean: {s['mean']:.1f} bars (~{s['mean']*5:.0f} min)")
    print(f"  Range: {s['min']:.0f} to {s['max']:.0f} bars")

    print(f"\nVolume Ratio at move START:")
    s = analysis['all']['start_vol_ratio']
    print(f"  Mean: {s['mean']:.2f}x average")
    print(f"  25th percentile: {s['p25']:.2f}x")
    print(f"  75th percentile: {s['p75']:.2f}x")

    print(f"\nZ-Score Velocity at START (|dZ/dt|):")
    s = analysis['all']['z_velocity']
    print(f"  Mean: {s['mean']:.4f}")
    print(f"  25th percentile: {s['p25']:.4f}")
    print(f"  75th percentile: {s['p75']:.4f}")

    print(f"\nOFI at START (|order flow imbalance|):")
    s = analysis['all']['start_ofi']
    print(f"  Mean: {s['mean']:.4f}")
    print(f"  25th percentile: {s['p25']:.4f}")
    print(f"  75th percentile: {s['p75']:.4f}")

    # Pattern thresholds
    print(f"\n{'='*100}")
    print("PROPOSED ENTRY CRITERIA FOR 'Z-SCORE RUNNER'")
    print(f"{'='*100}")

    patterns = find_high_probability_patterns(all_events, None)

    print(f"\nThresholds (25th percentile of extreme events):")
    print(f"  Volume Ratio >= {patterns['thresholds']['volume_ratio']:.2f}x")
    print(f"  |Z Velocity| >= {patterns['thresholds']['z_velocity']:.4f}")
    print(f"  |OFI|        >= {patterns['thresholds']['ofi']:.4f}")

    print(f"\nHit rates (% of extreme events that met criteria at start):")
    print(f"  Volume alone:          {patterns['hit_rates']['volume_alone']:.1f}%")
    print(f"  Velocity alone:        {patterns['hit_rates']['velocity_alone']:.1f}%")
    print(f"  OFI alone:             {patterns['hit_rates']['ofi_alone']:.1f}%")
    print(f"  Volume AND Velocity:   {patterns['hit_rates']['volume_and_velocity']:.1f}%")

    # Direction-specific patterns
    up_events = [e for e in all_events if e.direction == "UP"]
    down_events = [e for e in all_events if e.direction == "DOWN"]

    print(f"\n{'='*100}")
    print("DIRECTION-SPECIFIC PATTERNS")
    print(f"{'='*100}")

    if up_events:
        print(f"\n--- UP MOVES (to Z > 3.5) ---")
        print(f"  Count: {len(up_events)}")
        s = analysis['up']['z_velocity']
        print(f"  Z Velocity: Mean={s['mean']:.4f} (should be positive)")
        s = analysis['up']['start_ofi']
        print(f"  OFI: Mean={s['mean']:.4f} (positive = more buyers)")

    if down_events:
        print(f"\n--- DOWN MOVES (to Z < -3.5) ---")
        print(f"  Count: {len(down_events)}")
        s = analysis['down']['z_velocity']
        print(f"  Z Velocity: Mean={s['mean']:.4f} (should be negative)")
        s = analysis['down']['start_ofi']
        print(f"  OFI: Mean={s['mean']:.4f} (negative = more sellers)")

    # Example events
    print(f"\n{'='*100}")
    print("SAMPLE EXTREME EVENTS")
    print(f"{'='*100}")

    print(f"\n{'Start Time':<22} {'Dir':<5} {'Peak Z':>8} {'Duration':>10} {'Vol Ratio':>10} {'Z Vel':>10}")
    print("-" * 75)
    for e in all_events[:15]:
        print(f"{str(e.start_time):<22} {e.direction:<5} {e.peak_z:>8.2f} {e.duration_bars:>7} bars {e.start_vol_ratio:>9.2f}x {e.z_velocity:>10.4f}")

    # Suggested strategy rules
    print(f"\n{'='*100}")
    print("SUGGESTED Z-SCORE RUNNER STRATEGY RULES")
    print(f"{'='*100}")
    print("""
    ENTRY (Trend-Following into Extreme):
    1. Z-Score crosses above +0.5 (potential up runner)
       - AND Z velocity > 0.05 (accelerating upward)
       - AND Volume ratio > 1.5x (high interest)
       - THEN: Enter LONG expecting Z to reach 3.5+

    2. Z-Score crosses below -0.5 (potential down runner)
       - AND Z velocity < -0.05 (accelerating downward)
       - AND Volume ratio > 1.5x (high interest)
       - THEN: Enter SHORT expecting Z to reach -3.5+

    EXIT:
    - Target: Z reaches +/- 3.5 (or higher)
    - Stop: Z reverts to 0 (failed runner)
    - Time stop: 24 bars (~2 hours) if no extreme reached

    This is TREND-FOLLOWING, not mean-reversion!
    """)

    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
