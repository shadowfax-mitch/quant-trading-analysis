# Handoff: Sprint Kickoff (Codex -> Claude)

## Scope and goals
- Kick off the OU mean-reversion sprint on MES tick data and establish a shared baseline.
- Align on dataset scope, train/test split, and execution/cost assumptions.
- Define the first concrete research steps and any clarifications needed before coding.

## Key docs to read first
- `AGENTS.md` (collaboration playbook overview; roles, review loop).
- `COLLABORATION_PLAYBOOK.md` (same content as AGENTS, kept as a standalone reference).
- `docs/Preliminary_Research_Sprint_Plan.md` (full sprint plan, model, and Go/No-Go gates).
- `docs/assumptions.md` (cost model, tick size, execution rules).
- `docs/decision_log.md` (template for Go/No-Go decisions).
- `README.md` (dataset scope and locked split).

## Dataset scope (current inventory)
- MES tick data: `datasets/MES/tick_data/mes_ticks_part0001.csv` to `mes_ticks_part0104.csv`.
  - Date range observed: 2025-01-01 17:00:00.024 to 2025-07-01 23:58:10.624.
  - Columns: timestamp,last,volume,bid,ask,side,contract,source.
- MNQ tick data: `datasets/MNQ/tick_data/mnq_ticks_part0001.csv` to `mnq_ticks_part0124.csv`.
  - Date range observed: 2024-03-12 17:00:00.032 to 2026-01-14 22:39:28.120.
  - Columns: timestamp,last,volume,bid,ask,side,contract,source.
- MNQ 233-tick bars: `datasets/MNQ/tick_data/mnq_233tick.csv`.
  - Date range observed: 2024-03-12 17:00:32.540 to 2026-01-14 22:39:24.016.
  - Columns: timestamp,open,high,low,close,volume,tick_count,contract,source.

## Sprint train/test split (locked)
- Instrument: MES tick data only for the OU sprint.
- Train: 2025-01-01 through 2025-02-28 (inclusive).
- Test: 2025-03-01 through 2025-03-31 (inclusive).
- Rationale: We do not have the 2023 window mentioned in the plan; this uses earliest contiguous MES data.

## Modeling and execution assumptions (current defaults)
- Spread crossing: entries/exits at bid/ask, not mid.
- Commission/fees: $0.85 per side per contract (default for MES/MNQ).
- Slippage: 0 ticks beyond quoted bid/ask (adjust for stress tests).
- Tick sizes: MES 0.25, MNQ 0.25.
- Fill rules: signal at tick t fills at tick t+1; long at ask, short at bid.
- Exit: mid crosses rolling mean; long exits at bid, short exits at ask.
- 1 contract per trade; no scaling/pyramids; full fills assumed.
- Timestamps standardized to UTC in preprocessing; raw files unchanged.

## Plan highlights (OU sprint)
- Model: OU process estimated via rolling window (AR(1)/OLS approach).
- Signals: long if mid < mu - N*sigma; short if mid > mu + N*sigma.
- N values: 1.5, 2.0, 2.5.
- Exit: mid crosses rolling mu (lagged).
- Backtest: vectorized tick-level, bid/ask fills, commissions.
- Go/No-Go gate: positive P&L after costs for at least two N values and:
  - Profit Factor >= 1.1
  - Average trade >= 1.0 tick
  - min trades >= 30 (test set)

## Files created/updated by Codex
- `README.md` (dataset scope + split).
- `docs/assumptions.md` (costs/tick size/execution).
- `docs/decision_log.md` (Go/No-Go log template).
- `docs/agent_handoff/.gitkeep`.
- `COLLABORATION_PLAYBOOK.md`.
- `scripts/run_sprint.ps1` (optional run wrapper; expects `src/run_sprint.py`).

## Open questions / clarifications needed
- Do you agree with the locked MES split (Jan-Feb train, Mar test), or should we adjust?
- Should we restrict to specific sessions (RTH vs ETH), or use full session for sprint?
- Is the 0-tick slippage assumption acceptable for the first pass, or should we add a stress case?
- Should we ignore MNQ data for now (plan focuses on MES), or parallel-check MNQ if time allows?

## Explicit asks for Claude
1) Review `docs/Preliminary_Research_Sprint_Plan.md` and confirm if any steps need edits based on the actual 2025 MES data window.
2) Audit `docs/assumptions.md` for missing realism items (e.g., bid/ask anomalies, timestamp monotonicity checks).
3) Propose a minimal data integrity checklist we should run before modeling (timestamp order, missing bid/ask, spread sanity).
4) Suggest whether the locked split should be rolling instead of fixed (for this sprint only).

## Notes
- No code changes beyond documentation and structure yet.
- We should record any Go/No-Go call in `docs/decision_log.md`.
