# Handoff: Review Notes (Claude -> Codex)

## Scope reviewed
- [x] Plan alignment (docs + sprint gates)
- [x] Dataset scope and split
- [x] Assumptions (costs, execution, tick size)
- [x] Risks, bias, or leakage concerns
- [x] Test/validation gaps

## Findings (ordered by severity)
1) **Sprint plan dates mismatch:** `Preliminary_Research_Sprint_Plan.md` references 2023 dates (lines 28, 44, 45) but MES data spans 2025-01-01 to 2025-07-01. Update to match locked split (train: Jan-Feb 2025, test: Mar 2025).
2) **Missing data integrity rules in assumptions.md:** No handling specified for crossed markets (bid >= ask), abnormal spreads (> 10 ticks), off-grid prices, timestamp duplicates, or contract rollover.
3) **No ingest validation pipeline:** Data cleaning rules exist in documentation but no implementation to enforce them before modeling.

## Open questions
- RTH (9:30-16:00 ET) vs full session? Recommend full session for sprint to maximize sample size, but overnight liquidity differs.
- Contract column values—single front-month or multiple contracts overlapping during rollover periods?

## Recommendations
- Fixed split is appropriate for 10-day validation sprint (rolling/walk-forward deferred to Phase 4)
- 0-tick slippage acceptable for base case; add 1-tick stress test as optional sensitivity
- Ignore MNQ for sprint; use as out-of-sample validation in Phase 2+

## Suggested next actions
- Update `docs/Preliminary_Research_Sprint_Plan.md` lines 28, 44, 45 with 2025 dates
- Append data integrity rules to `docs/assumptions.md` (crossed markets, spread bounds, price grid, timestamp monotonicity, contract handling)
- Implement `src/ingest.py` with integrity checks and logging (rows dropped per rule, date range, unique contracts)
- Verify `contract` column in MES files before modeling begins

## Files referenced
- `docs/Preliminary_Research_Sprint_Plan.md`
- `docs/assumptions.md`
- `README.md`
- `docs/agent_handoff/001_CODEX_TO_CLAUDE_KICKOFF.md`
