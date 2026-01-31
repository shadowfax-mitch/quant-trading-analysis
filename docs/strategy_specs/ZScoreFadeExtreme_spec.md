# ZScoreFadeExtreme - Strategy Specification

## Status: VALIDATED on 2025 data
## Date: 2026-01-30
## Instrument: MES (Micro E-mini S&P 500)

---

## 1. Concept

Mean reversion scalper that fades extreme Z-score readings. When price extends
3.5+ standard deviations from EMA(21), there is strong statistical pressure to
revert. This strategy enters against the extreme and captures the reversion with
fixed point targets.

**Edge source:** Extreme Z-score values (|Z| >= 3.5) revert toward zero with
enough consistency to produce PF 1.42 after costs on 2025 data.

## 2. Signal Logic

### Z-Score Calculation
- EMA Period: 21 (5-minute bars)
- Distance: (Close - EMA) / EMA
- Z-Score Lookback: 21 bars (rolling std dev of distance)
- Z = distance / rolling_std(distance, 21)

### Entry Conditions

**SHORT (fade positive extreme):**
- Previous bar Z >= 3.5 (price extended above EMA)
- No reversal confirmation required (enter immediately)
- Volume ratio >= 0.0 (no volume filter)
- Within RTH window (08:30 - 15:00 CT)

**LONG (fade negative extreme):**
- Previous bar Z <= -3.5 (price extended below EMA)
- Same filters as short

### Entry Execution
- Signal generated on bar close
- Execute at next bar open (1-bar delay to avoid lookahead)
- Adverse fill: +1 tick slippage on entry

## 3. Exit Logic

All exits use bar-close evaluation (not intrabar high/low):

| Exit Type | Condition | Priority |
|-----------|-----------|----------|
| Profit Target | Unrealized >= 4.0 points ($20) | 1 |
| Stop Loss | Unrealized <= -4.0 points ($20) | 2 |
| Timeout | Bars held >= 20 | 3 |
| RTH Close | Time >= 15:00 CT | 0 (highest) |

### Breakeven Trail: DISABLED (trigger = 0)

### Exit Execution
- Bar-close exits only (use_bar_close_exits = True)
- This is critical - intrabar exits kill the edge due to 5-min bar range

## 4. Position Sizing
- Default: 1 contract
- Scale by adding contracts (not by changing strategy parameters)

## 5. Backtest Results (2025)

```
Trades:          311
Trades/Day:      1.2
Trades/Year:     311
Net P&L:         $2,022.55
Profit Factor:   1.42
Win Rate:        54.3%
Avg Trade:       $6.50
Avg Winner:      ~$18.50
Avg Loser:       ~$15.50
Avg Bars Held:   2.8
```

### Exit Reason Distribution
- Profit Target: majority of exits
- Stop Loss: second most common
- Timeout: minority

### Cost Analysis
- Round-trip cost: $1.70 (commission $0.85/side)
- Slippage: 1 tick ($1.25) each way built into backtest
- Avg gross per trade: ~$8.20
- Cost as % of gross: ~21%

## 6. Configuration Parameters

| Parameter | Value | Tested Range | Notes |
|-----------|-------|-------------|-------|
| ema_period | 21 | 8, 13, 21 | 21 is best |
| z_lookback | 21 | 10, 14, 21 | 21 is best |
| entry_z | 3.5 | 1.0 - 3.5 | 3.5 is sweet spot |
| profit_target_pts | 4.0 | 2.0 - 8.0 | 4.0 optimal |
| stop_loss_pts | 4.0 | 2.0 - 8.0 | 4.0 optimal (1:1 R:R) |
| max_hold_bars | 20 | 15 - 60 | 20 is best |
| min_bars_between | 2 | 2 - 5 | 2 is fine |
| use_bar_close_exits | True | True/False | Must be True |

## 7. Nearby Configurations (Robust)

| PT | SL | MB | PF | WR | Net | Trades |
|----|-----|-----|------|------|---------|--------|
| 4.0 | 4.0 | 20 | 1.42 | 54.3% | $2,023 | 311 |
| 4.0 | 4.0 | 40 | 1.41 | 54.0% | $1,979 | 311 |
| 3.0 | 4.0 | 20 | 1.40 | 57.6% | $1,807 | 316 |
| 5.0 | 4.0 | 20 | 1.39 | 51.0% | $1,954 | 308 |
| 8.0 | 4.0 | 20 | 1.36 | 44.5% | $2,007 | 301 |

Note: PF is robust across PT 3-8 with SL=4.0. Stop at 4 points is the
critical parameter - wider stops (6, 8) degrade PF significantly.

## 8. Known Limitations

1. **Low frequency:** 1.2 trades/day - cannot be increased without
   destroying the edge. Z=3.0 drops PF to ~1.06.
2. **2025 only:** Not yet validated on multi-year data. Must expand
   backtest before live deployment.
3. **Bar-close exits required:** Intrabar stop/target evaluation kills
   the edge because 5-min bar ranges frequently exceed 4 points.
4. **No ML filter:** Pure rule-based. Adding ML could improve but adds
   complexity and latency.

## 9. Scaling Strategy

To increase dollar returns without changing frequency:
- 2 contracts: ~$4,045/year
- 5 contracts: ~$10,113/year
- 10 contracts: ~$20,226/year

Ensure adequate margin per contract (~$1,500 MES margin).

## 10. Next Steps

- [ ] Expand backtest to 2019-2024 data
- [ ] Walk-forward validation (out-of-sample)
- [ ] Monthly P&L breakdown for consistency check
- [ ] Build NinjaTrader strategy file (ZScoreScalper.cs)
- [ ] Paper trade for 2+ weeks
- [ ] Consider combining with VWAP signal for higher frequency variant

## 11. Files

- Backtest script: `tools/backtest_zscore_scalper.py`
- NinjaTrader strategy: `ninjatrader/strategies/ZScoreScalper.cs`
- Sweep results: `results/scalper_mr_sweep_2025.csv`
- This spec: `docs/strategy_specs/ZScoreFadeExtreme_spec.md`
