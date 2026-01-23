# Handoff: Ingest Results + Decision (Codex -> Claude)

## Ingest run summary (MES 03-25 only)
- Rows total: 103,420,245
- Rows kept: 22,131,208
- Date range: 2025-01-01 17:00 to 2025-03-13 23:59 UTC
- Drops:
  - out_of_range: 67,367,001
  - exact_duplicates: 13,797,077
  - contract_filter: 93,081
  - spread_too_wide: 31,815
  - crossed: 63
  - bad_timestamp / missing_bid_ask / non_positive / off_grid: 0
- Output: `data/sprint_data.parquet`
- Note: high duplicate count suggests overlapping vendor rows; dropped as exact duplicates.

## Decision
- Proceed to Phase 2 on the current dataset and revisit the split later (no changes to split docs yet).

## Next action requested
- Please review the ingest stats for any red flags and confirm whether we should:
  1) keep March truncated to 2025-03-13 for MES 03-25 only, or
  2) add MES 06-25 data for 2025-03-17 onward to preserve a full March test window.
