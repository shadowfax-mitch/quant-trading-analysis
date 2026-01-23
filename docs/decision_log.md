# Decision Log

Use this file to record Go/No-Go calls and why.

## Entries
| Date (UTC) | Decision | Scope | Evidence | Rationale | Next Steps |
| --- | --- | --- | --- | --- | --- |
| 2026-01-21 | Proceed to Phase 2 | OU sprint ingest | Ingest summary: 103,420,245 rows total; 22,131,208 kept; date range 2025-01-01 17:00 to 2025-03-13 23:59 UTC; drops: out_of_range 67,367,001, exact_duplicates 13,797,077, contract_filter 93,081, spread_too_wide 31,815, crossed 63; output data/sprint_data.parquet | Proceed with current dataset and revisit split after Phase 2 | Run OU estimation and signal generation; revisit split if needed |
| 2026-01-21 | NO-GO | Phase 3 backtest (OU mean-reversion) | Train/Test results: N=1.5/2.0/2.5 negative P&L across train/test; PF 0.64-0.85; avg trade -1.13 to -2.07 ticks; test trades 517-1,546; all gates failed | Strategy does not overcome transaction costs; per sprint plan, stop and review thesis | Strategic review before additional infra; consider alternative alpha sources |
