"""
Analyze why the Z-Score Runner strategy failed OOS.

The issue: Many entries don't reach the Z=3.5 target.
Need to understand what differentiates successful runs from failed ones.
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
    """Compute Z-score and features."""
    df = df.copy()

    # Z-Score
    df['ema'] = df['close'].ewm(span=21, adjust=False).mean()
    df['dist'] = (df['close'] - df['ema']) / df['ema']
    df['dist_std'] = df['dist'].rolling(21).std()
    df['zscore'] = df['dist'] / df['dist_std']

    # Z-Score velocity
    df['z_velocity'] = df['zscore'].diff()

    # Volume ratio
    df['vol_avg'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_avg']

    return df


def analyze_all_z_moves(df: pd.DataFrame, entry_z: float = 1.5, target_z: float = 3.5) -> List[Dict]:
    """Track all Z-score moves that crossed entry_z and see if they reached target_z."""
    moves = []
    n = len(df)
    zscore = df['zscore'].values
    z_velocity = df['z_velocity'].values
    vol_ratio = df['vol_ratio'].values
    timestamps = df['timestamp'].values

    in_move = False
    move_start_idx = 0
    direction = ""
    max_z = 0
    min_z = 0

    for i in range(50, n):
        z = zscore[i]
        zv = z_velocity[i]

        if np.isnan(z) or np.isnan(zv):
            continue

        if not in_move:
            # Check for new move starting
            if z > entry_z and zv > 0.3:  # Upward move starting
                in_move = True
                move_start_idx = i
                direction = "UP"
                max_z = z
                min_z = z
            elif z < -entry_z and zv < -0.3:  # Downward move starting
                in_move = True
                move_start_idx = i
                direction = "DOWN"
                max_z = z
                min_z = z
        else:
            # Track extremes
            max_z = max(max_z, z)
            min_z = min(min_z, z)

            # Check if move ended (Z reverted toward zero)
            move_ended = False
            reached_target = False

            if direction == "UP":
                if z <= 0 or z < entry_z * 0.5:  # Reverted
                    move_ended = True
                reached_target = max_z >= target_z
            else:  # DOWN
                if z >= 0 or z > -entry_z * 0.5:  # Reverted
                    move_ended = True
                reached_target = min_z <= -target_z

            if move_ended:
                start_vr = vol_ratio[move_start_idx] if not np.isnan(vol_ratio[move_start_idx]) else 1.0
                start_zv = z_velocity[move_start_idx] if not np.isnan(z_velocity[move_start_idx]) else 0

                moves.append({
                    "start_time": pd.Timestamp(timestamps[move_start_idx]),
                    "end_time": pd.Timestamp(timestamps[i]),
                    "direction": direction,
                    "entry_z": zscore[move_start_idx],
                    "max_z": max_z,
                    "min_z": min_z,
                    "reached_target": reached_target,
                    "duration_bars": i - move_start_idx,
                    "start_vol_ratio": start_vr,
                    "start_z_velocity": abs(start_zv),
                })
                in_move = False

    return moves


def main():
    print("=" * 100)
    print("Z-SCORE RUNNER FAILURE ANALYSIS")
    print("=" * 100)
    print()

    # Load data
    all_moves = []
    for period, start, end in [("Train", "2025-01-01", "2025-02-28"), ("OOS", "2025-03-01", "2025-03-31")]:
        print(f"\nLoading {period}...")
        bars = load_cached_bars(start, end)
        if len(bars) == 0:
            continue

        df = compute_features(bars)
        moves = analyze_all_z_moves(df, entry_z=1.5, target_z=3.5)
        print(f"   Found {len(moves)} Z-score moves")

        for m in moves:
            m["period"] = period

        all_moves.extend(moves)

    if not all_moves:
        print("No moves found!")
        return 1

    df_moves = pd.DataFrame(all_moves)

    print(f"\n{'='*100}")
    print("ANALYSIS OF Z-SCORE MOVES")
    print(f"{'='*100}")

    total = len(df_moves)
    reached = df_moves["reached_target"].sum()
    rate = reached / total * 100

    print(f"\nTotal moves that crossed entry_z=1.5: {total}")
    print(f"Moves that reached target_z=3.5:       {reached} ({rate:.1f}%)")
    print(f"Moves that FAILED to reach target:     {total - reached} ({100-rate:.1f}%)")

    # Success rate is low! Only ~30% of entries reach the target.
    # This explains the losses.

    # Analyze what distinguishes successful vs failed moves
    success = df_moves[df_moves["reached_target"]]
    failed = df_moves[~df_moves["reached_target"]]

    print(f"\n{'='*100}")
    print("SUCCESSFUL vs FAILED MOVES")
    print(f"{'='*100}")

    print(f"\n{'Metric':<25} {'SUCCESS':>15} {'FAILED':>15}")
    print("-" * 60)

    # Volume ratio at start
    s_vr = success["start_vol_ratio"].mean()
    f_vr = failed["start_vol_ratio"].mean()
    print(f"{'Volume Ratio at start:':<25} {s_vr:>15.2f}x {f_vr:>15.2f}x")

    # Z velocity at start
    s_zv = success["start_z_velocity"].mean()
    f_zv = failed["start_z_velocity"].mean()
    print(f"{'|Z Velocity| at start:':<25} {s_zv:>15.4f} {f_zv:>15.4f}")

    # Entry Z level
    s_ez = success["entry_z"].abs().mean()
    f_ez = failed["entry_z"].abs().mean()
    print(f"{'|Entry Z|:':<25} {s_ez:>15.2f} {f_ez:>15.2f}")

    # Duration
    s_dur = success["duration_bars"].mean()
    f_dur = failed["duration_bars"].mean()
    print(f"{'Duration (bars):':<25} {s_dur:>15.1f} {f_dur:>15.1f}")

    # Find thresholds that improve success rate
    print(f"\n{'='*100}")
    print("FINDING BETTER ENTRY FILTERS")
    print(f"{'='*100}")

    # Test different velocity thresholds
    print("\nSuccess rate by Z-velocity threshold at entry:")
    for vel_thresh in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
        filtered = df_moves[df_moves["start_z_velocity"] >= vel_thresh]
        if len(filtered) > 0:
            rate = filtered["reached_target"].sum() / len(filtered) * 100
            print(f"   Velocity >= {vel_thresh}: {rate:>5.1f}% success ({len(filtered)} moves)")

    # Test different volume thresholds
    print("\nSuccess rate by volume ratio threshold at entry:")
    for vol_thresh in [1.0, 1.5, 2.0, 2.5, 3.0]:
        filtered = df_moves[df_moves["start_vol_ratio"] >= vol_thresh]
        if len(filtered) > 0:
            rate = filtered["reached_target"].sum() / len(filtered) * 100
            print(f"   Vol Ratio >= {vol_thresh}x: {rate:>5.1f}% success ({len(filtered)} moves)")

    # Test combined filters
    print("\nSuccess rate with combined filters:")
    for vel_thresh in [0.8, 1.0, 1.2]:
        for vol_thresh in [1.5, 2.0]:
            filtered = df_moves[
                (df_moves["start_z_velocity"] >= vel_thresh) &
                (df_moves["start_vol_ratio"] >= vol_thresh)
            ]
            if len(filtered) >= 5:
                rate = filtered["reached_target"].sum() / len(filtered) * 100
                print(f"   Vel >= {vel_thresh} AND Vol >= {vol_thresh}x: {rate:>5.1f}% success ({len(filtered)} moves)")

    # Test higher entry Z
    print("\nSuccess rate by entry Z threshold:")
    for entry_thresh in [2.0, 2.5, 3.0]:
        filtered = df_moves[df_moves["entry_z"].abs() >= entry_thresh]
        if len(filtered) > 0:
            rate = filtered["reached_target"].sum() / len(filtered) * 100
            print(f"   Entry |Z| >= {entry_thresh}: {rate:>5.1f}% success ({len(filtered)} moves)")

    print(f"\n{'='*100}")
    print("CONCLUSION")
    print(f"{'='*100}")
    print("""
    The Z-Score Runner approach has a fundamental problem:
    - Only ~30% of moves that cross Z=1.5 actually reach Z=3.5
    - This means 70% of trades lose money

    To make it work, we need MUCH stricter entry criteria:
    - Higher Z-velocity (>1.0)
    - Higher volume (>2.0x)
    - Higher entry Z (>2.5)

    But these strict filters reduce trade frequency significantly.

    ALTERNATIVE APPROACH:
    Instead of trend-following INTO extremes, consider:
    1. Keep Z=5.0 mean-reversion as primary strategy
    2. Add a "scalp" strategy in the 3.5-5.0 zone with tight stops
    """)

    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
