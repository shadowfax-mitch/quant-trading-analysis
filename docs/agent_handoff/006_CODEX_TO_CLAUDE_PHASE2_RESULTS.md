# Handoff: Phase 2 OU Results (Codex -> Claude)

## Command run
```
python src/phase2_ou.py --input data/sprint_data.parquet --output data/sprint_with_ou.parquet --window 10000 --lag 1 --step 1
```

## Output
- Output file: `data/sprint_with_ou.parquet`
- Valid parameter ticks: 17,385,744
- Signal counts:
  - `signal_1_5`: {1: 8,490,060, -1: 8,464,235, 0: 5,176,913}
  - `signal_2_0`: {1: 8,417,244, -1: 8,393,835, 0: 5,320,129}
  - `signal_2_5`: {1: 8,345,112, -1: 8,323,121, 0: 5,462,975}

## Notes
- Rolling OU parameters computed with window=10,000 ticks, lag=1, step=1, dt=1.
- Signals are symmetric around mu with N values 1.5, 2.0, 2.5.

## Request
- Please review whether the signal balance looks reasonable and if any parameter sanity checks are missing before Phase 3 backtest.
