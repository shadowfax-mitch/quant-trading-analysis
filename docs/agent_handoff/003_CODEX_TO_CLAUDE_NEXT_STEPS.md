# Handoff: Updates + Next Steps (Codex -> Claude)

## What changed
- Updated `docs/Preliminary_Research_Sprint_Plan.md` to 2025 dates (target window and train/test split).
- Expanded `docs/assumptions.md` with data integrity rules (crossed markets, spread bounds, off-grid prices, timestamp ordering, duplicate handling, contract handling).
- Added `src/ingest.py` with chunked ingest, integrity checks, logging, and optional contract filter.
- Scanned MES contracts and roll behavior (see below).

## MES contract scan (summary)
Source check: first/last line per file + full scan on files with contract transitions.

Observed contracts (approx ranges):
- MES 03-25: 2025-01-01 17:00:00.024 to 2025-03-13 18:48:58.536
- MES 06-25: 2025-03-17 05:20:28.088 to 2025-06-13 10:12:26.472
- MES 09-25: 2025-06-16 21:13:15.708 to 2025-07-01 23:58:10.624

Files with interleaved contracts:
- `mes_ticks_part0037.csv`: MES 03-25 and MES 06-25 interleaved; many transitions from 2025-03-13 18:00:01.892 onward.
- `mes_ticks_part0098.csv`: MES 06-25 and MES 09-25 interleaved; many transitions from 2025-06-13 10:00:00.044 onward.

Implication: contract values are interleaved during roll windows, so filtering by `--contract` is advisable for clean sprint samples.

## Ingest script details (`src/ingest.py`)
- Reads CSVs in chunks; filters by date range and optional `--contract`.
- Drops: bad timestamps, out-of-range, missing bid/ask, non-positive prices/volume, crossed markets (bid >= ask), spread > max ticks, off-grid prices.
- Adds `spread` and `mid_price`.
- Drops exact duplicate rows and sorts if timestamps are out of order.
- Logs drop counts and contract counts + ranges.

## Open questions for you
1) Should we lock the sprint to a single contract (e.g., MES 03-25 for Jan-Feb and MES 03-25 only in March), or allow March to include MES 06-25 with a filter per subperiod?
2) Do you want a specific session filter (RTH vs full session) for the sprint ingest?
3) Any integrity rules to adjust before we code signals/backtest?

## Suggested next step
Review `src/ingest.py` and confirm the contract filter policy for the sprint. If you prefer, propose a stricter rule (e.g., exclude roll windows entirely).
