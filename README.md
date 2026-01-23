# wave_signals_v4

## Dataset scope
- MES tick data: `datasets/MES/tick_data/mes_ticks_part0001.csv` to `datasets/MES/tick_data/mes_ticks_part0104.csv` (fields: timestamp,last,volume,bid,ask,side,contract,source).
  - Date range: 2025-01-01 17:00:00.024 to 2025-07-01 23:58:10.624
- MNQ tick data: `datasets/MNQ/tick_data/mnq_ticks_part0001.csv` to `datasets/MNQ/tick_data/mnq_ticks_part0124.csv` (fields: timestamp,last,volume,bid,ask,side,contract,source).
  - Date range: 2024-03-12 17:00:00.032 to 2026-01-14 22:39:28.120
- MNQ 233-tick bars: `datasets/MNQ/tick_data/mnq_233tick.csv` (fields: timestamp,open,high,low,close,volume,tick_count,contract,source).
  - Date range: 2024-03-12 17:00:32.540 to 2026-01-14 22:39:24.016

## Sprint train/test split (locked)
- Instrument: MES tick data (OU mean-reversion sprint).
- Train: 2025-01-01 through 2025-02-28 (inclusive).
- Test: 2025-03-01 through 2025-03-13 (inclusive).

Notes:
- Timestamps are vendor raw; standardize to UTC during ingest (see `docs/assumptions.md`).
- Current ingest uses `MES 03-25` only, which ends on 2025-03-13; revisit if we add the next contract.
