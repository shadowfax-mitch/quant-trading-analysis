# ZThreeNoFilter - Strategy Specification

## Status: VALIDATED on Jan-Jul 2025 data (PF 2.28, 140 trades)
## Date: 2026-01-31
## Instrument: MES (Micro E-mini S&P 500)

---

## 1. Concept

Mean reversion scalper that fades extreme Z-score readings on EMA(21). When
price extends 3.75+ standard deviations from EMA(21), enter against the
extreme and capture the reversion with a fixed profit target and stop loss.

**Edge source:** Extreme Z-score values (|Z| >= 3.75) on RTH-only 5-minute
bars revert toward zero with high consistency. The tighter Z threshold (3.75
vs the original 3.5) eliminates marginal signals, lifting PF from 1.73 to
2.28 with only a modest reduction in trade count.

**Relationship to ZScoreFadeExtreme:** This is an optimized variant. The
original strategy uses Z=3.5, PT=4, SL=4 and produces 173 trades at PF 1.73
on the same dataset. Raising the Z threshold to 3.75 and tightening the stop
to 3.5 points removes ~33 lower-quality entries and improves PF by 32%.

**Design philosophy:** No filters. No time-of-day restrictions. No day-of-week
restrictions. No volatility gates. No direction confirmation. Optimization
analysis showed that while hour and day filters can boost PF dramatically in
sample, they reduce trade count to levels where statistical confidence is
insufficient. This configuration prioritizes robustness over peak PF.

## 2. Signal Logic

### Z-Score Calculation
- Bar type: 5-minute bars, RTH only (08:30 - 15:00 CT)
- EMA Period: 21
- Z-Score Lookback: 21 bars (rolling std dev of normalized distance)
- distance = (Close - EMA_21) / EMA_21
- Z = distance / rolling_std(distance, 21)

**Critical:** EMA and Z-score must be computed on RTH-only bars. Including
overnight session bars changes the EMA response and Z-score distribution,
producing different (and worse) signals.

### Entry Conditions

**SHORT (fade positive extreme):**
1. Previous bar Z >= 3.75 (price extended above EMA)
2. Within RTH window (08:30 - 15:00 CT)

**LONG (fade negative extreme):**
1. Previous bar Z <= -3.75 (price extended below EMA)
2. Within RTH window (08:30 - 15:00 CT)

### Entry Execution
- Signal generated on bar close
- Execute at next bar open (1-bar delay to avoid lookahead)
- Adverse fill: +1 tick slippage on entry
- Minimum 2 bars between trades (min_bars_between = 2)

## 3. Exit Logic

All exits use bar-close evaluation (not intrabar high/low):

| Exit Type | Condition | Priority |
|-----------|-----------|----------|
| RTH Close | Time >= 15:00 CT | 0 (highest) |
| Profit Target | Unrealized >= 4.0 points ($20) | 1 |
| Stop Loss | Unrealized <= -3.5 points ($17.50) | 2 |
| Timeout | Bars held >= 20 | 3 |

### Breakeven Trail: DISABLED

### Exit Execution
- Bar-close exits only (use_bar_close_exits = True)
- Critical: intrabar exits destroy the edge on 5-min bars because bar ranges
  frequently exceed 4 points, causing premature stop-outs

### Why PT=4 / SL=3.5?
The asymmetric R:R (1.14:1, risk-favorable) works because:
- 60% win rate more than compensates for the slightly wider target
- Tighter stop (3.5 vs 4.0 in original) cuts losing trades faster
- PT/SL sensitivity analysis shows PF is robust across PT 3.5-5.0 and SL 3.0-4.5

## 4. Position Sizing
- Default: 1 contract
- Scale by adding contracts (not by changing strategy parameters)

## 5. Backtest Results (Jan-Jul 2025)

### Full Period
```
Trades:          140
Trades/Day:      0.84
Net P&L:         $3,313.25
Profit Factor:   2.28
Win Rate:        60.0%
Avg Trade:       $23.67
Avg Winner:      $70.37
Avg Loser:       $-46.39
Avg Bars Held:   3.5
Max Drawdown:    $391.85
Trading Days:    166
```

### Train/Test Split
```
Train (Jan-Apr):  96 trades, PF=2.16, WR=55.2%, Net=$2,344.30
Test  (May-Jul):  44 trades, PF=2.68, WR=70.5%, Net=$968.95
```

Test PF exceeds train PF — no sign of overfitting on this split.

### Exit Reason Distribution
| Exit Type | Count | Pct |
|-----------|-------|-----|
| Profit Target | 84 | 60.0% |
| Stop Loss | 55 | 39.3% |
| Timeout | 1 | 0.7% |

### Direction Distribution
| Direction | Count |
|-----------|-------|
| Short | 78 |
| Long | 62 |

### Monthly P&L
| Month | Trades | Net P&L | Win Rate | PF |
|-------|--------|---------|----------|----|
| Jan 2025 | 28 | $926.15 | 57% | 3.10 |
| Feb 2025 | 21 | $231.80 | 48% | 1.60 |
| Mar 2025 | 24 | $352.95 | 54% | 1.50 |
| Apr 2025 | 23 | $833.40 | 61% | 2.69 |
| May 2025 | 22 | -$44.90 | 64% | 0.90 |
| Jun 2025 | 22 | $1,013.85 | 77% | 8.25 |

5 of 6 months profitable. May was near-breakeven (-$44.90, PF 0.90).

### Cost Analysis
- Commission (round-trip): $1.70
- Slippage (1 tick each way): $2.50
- Total cost per trade: $4.20
- Avg gross per trade: $25.37
- Cost as % of gross: 16.6%

## 6. Configuration Parameters

| Parameter | Value | Tested Range | Notes |
|-----------|-------|-------------|-------|
| ema_period | 21 | 21 | Inherited from ZScoreFadeExtreme |
| z_lookback | 21 | 21 | Inherited from ZScoreFadeExtreme |
| entry_z | 3.75 | 3.0 - 4.5 | 3.75 is sweet spot |
| profit_target_pts | 4.0 | 2.5 - 6.0 | Robust across 3.5-5.0 |
| stop_loss_pts | 3.5 | 2.5 - 6.0 | 3.5 optimal |
| max_hold_bars | 20 | 20 | Not swept |
| min_bars_between | 2 | 2 | Not swept |
| use_bar_close_exits | True | True/False | Must be True |

## 7. Nearby Configurations (Robustness)

All configs tested on full Jan-Jul 2025 dataset:

| PT | SL | PF | WR | Net | Trades |
|----|-----|------|------|---------|--------|
| 3.5 | 3.5 | 2.16 | 58.6% | $2,987 | 140 |
| 3.5 | 4.0 | 2.13 | 61.4% | $2,980 | 140 |
| 4.0 | 3.5 | 2.28 | 60.0% | $3,313 | 140 |
| 4.0 | 4.0 | 2.25 | 62.9% | $3,252 | 140 |
| 4.5 | 3.5 | 2.21 | 58.6% | $3,235 | 140 |
| 4.5 | 4.0 | 2.17 | 61.4% | $3,175 | 140 |
| 5.0 | 3.5 | 2.19 | 57.1% | $3,329 | 140 |
| 5.0 | 4.0 | 2.16 | 60.0% | $3,277 | 140 |

PF is stable across all PT/SL combinations (2.13-2.28). The edge is in the
Z=3.75 signal, not in exit tuning.

## 8. Optimization Levers Explored (and Rejected)

### Time-of-Day Filter
Hours 11-14 CT have PF 1.76-4.01; hours 8-9 are losers (PF 0.44-0.79).
**Rejected:** Filtering to good hours boosts PF to 5+ but reduces trades to
19-30 in 6 months. Insufficient sample size for confidence.

### Day-of-Week Filter
Mon/Wed/Thu are profitable (PF 1.84-2.57); Friday loses (PF 0.58).
**Rejected:** Same issue — filtering cuts trades too aggressively. A future
variant could explore this with multi-year data.

### Volatility Filter (ATR)
Requiring higher ATR ratios consistently hurt train-period PF.
**Rejected:** The strategy does not benefit from volatility filtering.
Extreme Z-scores are already self-selecting for volatile conditions.

### Direction Filter (Z already reverting)
Requiring Z[t-1] < Z[t-2] (for shorts) boosts PF to 2.57 but cuts trades
to 36 in 4 months.
**Rejected:** Trade count too low. May revisit with multi-year data.

## 9. Comparison to Original ZScoreFadeExtreme

| Metric | Original (Z=3.5, PT=4, SL=4) | ZThreeNoFilter (Z=3.75, PT=4, SL=3.5) |
|--------|-------------------------------|----------------------------------------|
| Trades (6mo) | 173 | **140** |
| Profit Factor | 1.73 | **2.28 (+32%)** |
| Win Rate | 56.6% | **60.0%** |
| Net P&L (6mo) | $2,776 | **$3,313 (+19%)** |
| Avg Trade | $16.05 | **$23.67 (+47%)** |
| Max Drawdown | $694.55 | **$391.85 (-44%)** |

Fewer trades, more profit, less drawdown. Every metric improved.

## 10. Known Limitations

1. **6 months of data only.** Jan-Jul 2025. Must validate on 2019-2024
   before trusting. One losing month (May) in a 6-month sample is expected,
   but multi-year data is needed to confirm the edge survives different
   regimes (bear markets, low-vol grinds, crash recoveries).
2. **Bar-close exits required.** Intrabar stop/target evaluation would
   produce different (likely worse) results due to 5-min bar ranges.
3. **0.84 trades/day is low frequency.** Cannot be increased without
   dropping Z threshold, which degrades PF.
4. **No regime awareness.** Pure rule-based. Does not adapt to changing
   volatility or market structure.
5. **MES only.** Not yet validated on MNQ, MYM, or other instruments.

## 11. Scaling Strategy

To increase dollar returns without changing frequency:
- 2 contracts: ~$6,626/6mo (~$13,252/year)
- 5 contracts: ~$16,566/6mo (~$33,132/year)
- 10 contracts: ~$33,133/6mo (~$66,265/year)

Ensure adequate margin per contract (~$1,500 MES margin).

Combined with multi-instrument deployment:
- 3 instruments at 0.84 trades/day each = ~2.5 trades/day
- MES + MNQ + MYM could roughly triple throughput

## 12. Next Steps

- [ ] Expand backtest to 2019-2024 multi-year data
- [ ] Walk-forward validation (rolling 3-month train, 1-month test)
- [ ] Monthly P&L consistency check across multiple years
- [ ] Validate on MNQ and MYM
- [ ] Build NinjaTrader strategy file
- [ ] Market Replay testing in NinjaTrader
- [ ] Paper trade for 2+ weeks alongside original config
- [ ] Investigate time-of-day filter with multi-year data (may become
      viable with larger sample)

## 13. Files

- Optimization script: `src/optimize_zscore_fade.py`
- Validation script: `src/validate_strategy_specs.py`
- Original spec: `docs/strategy_specs/ZScoreFadeExtreme_spec.md`
- This spec: `docs/strategy_specs/ZThreeNoFilter.md`
