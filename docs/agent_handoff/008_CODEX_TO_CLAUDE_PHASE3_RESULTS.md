# Handoff: Phase 3 Backtest Results (Codex -> Claude)

## Summary (from user-provided results)
All N values fail gates; verdict is NO-GO.

## Results table
```
N   Period  Trades   Net P&L   Win Rate   PF    Avg Ticks   Gate
1.5 Train   11,614   -$36,200  15.1%     0.64  -1.13
1.5 Test    1,546    -$4,812   47.6%     0.85  -1.13       NO-GO
2.0 Train   8,722    -$27,484  15.5%     0.66  -1.16
2.0 Test    945      -$4,053   40.1%     0.75  -2.07       NO-GO
2.5 Train   6,864    -$22,110  15.3%     0.66  -1.22
2.5 Test    517      -$2,165   30.8%     0.74  -1.99       NO-GO
```

## Gate failures
- Negative P&L across all configurations
- Profit factor 0.64-0.85 (required >= 1.1)
- Average trade -1 to -2 ticks (required >= +1.0)

## Decision logged
- `docs/decision_log.md` updated with NO-GO entry.

## Request
- Please review for any methodological flaws or missing checks before we close this sprint.
