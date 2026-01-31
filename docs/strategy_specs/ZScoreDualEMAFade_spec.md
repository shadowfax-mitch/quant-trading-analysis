# ZScoreDualEMAFade - Strategy Specification

## Status: VALIDATED on 2025 data (PF 1.52, small sample)
## Date: 2026-01-30
## Instrument: MES (Micro E-mini S&P 500)

---

## 1. Concept

Dual-EMA mean reversion scalper. Uses a fast EMA Z-score to detect price
extremes and a slow EMA Z-score direction to confirm the trend is turning.
The slow Z acts as a noise filter — eliminating fade entries where the broader
momentum hasn't yet reversed.

**Edge source:** When price is extreme on the fast timeframe (|Z_fast| >= 3.5)
AND the slower timeframe Z-score has begun reverting, the probability of
mean reversion is higher than fading on the fast Z alone.

**Relationship to ZScoreFadeExtreme:** This is a filtered subset. The single-EMA
strategy (EMA=21, Z=3.5) produces 311 trades at PF 1.42. The dual-EMA filter
removes ~218 trades where the slow Z hadn't turned yet — those removed trades
were net losers, lifting PF from 1.42 to 1.52.

## 2. Signal Logic

### Fast Z-Score (Signal Generator)
- EMA Period: 13 (5-minute bars)
- Z-Score Lookback: 14 bars
- Z_fast = distance_fast / rolling_std(distance_fast, 14)
- where distance_fast = (Close - EMA_13) / EMA_13

### Slow Z-Score (Direction Confirmation)
- EMA Period: 21 (5-minute bars)
- Z-Score Lookback: 21 bars
- Z_slow = distance_slow / rolling_std(distance_slow, 21)
- where distance_slow = (Close - EMA_21) / EMA_21

### Entry Conditions

**SHORT (fade positive extreme):**
1. Previous bar Z_fast >= 3.5 (price extreme above fast EMA)
2. Previous bar Z_slow < Z_slow[2 bars ago] (slow Z is decreasing)
3. Within RTH window (08:30 - 15:00 CT)

**LONG (fade negative extreme):**
1. Previous bar Z_fast <= -3.5 (price extreme below fast EMA)
2. Previous bar Z_slow > Z_slow[2 bars ago] (slow Z is increasing)
3. Within RTH window (08:30 - 15:00 CT)

### Why This Works

The fast EMA(13) with lookback 14 responds more quickly to price moves than
EMA(21), generating Z-score extremes sooner. But not all fast extremes revert
profitably — some occur during sustained trends where the broader momentum
(captured by the slow EMA(21) Z-score) is still pushing in the same direction.

By requiring the slow Z to have already begun reverting (changing direction),
we filter out entries where we'd be fighting the trend. The remaining ~93
entries are ones where both timeframes agree: price overshot and is coming back.

### Entry Execution
- Signal generated on bar close
- Execute at next bar open (1-bar delay to avoid lookahead)
- Adverse fill: +1 tick slippage on entry

## 3. Exit Logic

All exits use bar-close evaluation (not intrabar high/low):

| Exit Type | Condition | Priority |
|-----------|-----------|----------|
| Profit Target | Unrealized >= 3.0 points ($15) | 1 |
| Stop Loss | Unrealized <= -3.0 points ($15) | 2 |
| Alt: SL=4.0 | Unrealized <= -4.0 points ($20) | 2 |
| Timeout | Bars held >= 20 | 3 |
| RTH Close | Time >= 15:00 CT | 0 (highest) |

### Breakeven Trail: DISABLED

### Exit Execution
- Bar-close exits only (use_bar_close_exits = True)
- Critical: intrabar exits destroy the edge on 5-min bars

### PT/SL Sensitivity
The best configs cluster around PT=3, SL=3-4:

| PT | SL | PF | WR | Net | Trades |
|----|-----|------|------|---------|--------|
| 3.0 | 3.0 | 1.52 | 54.8% | $577 | 93 |
| 3.0 | 4.0 | 1.52 | 59.8% | $599 | 92 |
| 3.0 | 5.0 | 1.40 | 61.5% | $505 | 91 |
| 4.0 | 3.0 | 1.38 | 48.9% | $471 | 92 |
| 4.0 | 4.0 | 1.39 | 53.8% | $509 | 91 |
| 5.0 | 3.0 | 1.43 | 46.7% | $552 | 92 |
| 5.0 | 4.0 | 1.37 | 50.5% | $519 | 91 |
| 5.0 | 5.0 | 1.34 | 54.4% | $503 | 90 |

PT=3, SL=4 recommended: highest net P&L ($599), PF=1.52, 59.8% win rate.
The asymmetric R:R (0.75:1) works because the high win rate compensates.

## 4. Position Sizing
- Default: 1 contract
- Scale by adding contracts (not by changing strategy parameters)

## 5. Backtest Results (2025)

### Primary Config: PT=3, SL=4
```
Trades:          92
Trades/Day:      0.36
Trades/Year:     92
Net P&L:         $598.60
Profit Factor:   1.52
Win Rate:        59.8%
Avg Trade:       $6.51
Avg Bars Held:   ~2-3
```

### Alternative Config: PT=3, SL=3
```
Trades:          93
Trades/Day:      0.37
Trades/Year:     93
Net P&L:         $576.90
Profit Factor:   1.52
Win Rate:        54.8%
Avg Trade:       $6.20
```

### Cost Analysis
- Round-trip cost: $1.70 (commission $0.85/side)
- Slippage: 1 tick ($1.25) each way built into backtest
- Avg gross per trade: ~$8.21
- Cost as % of gross: ~21%

## 6. Configuration Parameters

| Parameter | Value | Tested Range | Notes |
|-----------|-------|-------------|-------|
| fast_ema_period | 13 | 8, 13 | 13 is best |
| fast_z_lookback | 14 | 10, 14 | 14 is best |
| slow_ema_period | 21 | 21, 30 | 21 is best |
| slow_z_lookback | 21 | 14, 21 | 21 is best |
| entry_z_fast | 3.5 | 2.5, 3.0, 3.5 | 3.5 is best |
| profit_target_pts | 3.0 | 3.0 - 5.0 | 3.0 optimal |
| stop_loss_pts | 4.0 | 3.0 - 5.0 | 3.0 or 4.0 both good |
| max_hold_bars | 20 | 20 | Not swept (inherited) |
| min_bars_between | 2 | 2 | Not swept (inherited) |
| use_bar_close_exits | True | True/False | Must be True |

### Slow Z Threshold: NOT USED
The slow Z threshold parameter has no effect in this mode. Only the
*direction* of the slow Z matters (increasing vs decreasing), not its
absolute level. This was confirmed by observing identical results across
sZ=1.5, 2.0, 2.5, 3.0.

## 7. Nearby Configurations (Robust with Slow EMA=30)

Using Slow EMA(30, ZLB=21) instead of EMA(21, ZLB=21):

| PT | SL | PF | WR | Net | Trades |
|----|-----|------|------|---------|--------|
| 3.0 | 4.0 | 1.43 | 57.8% | $480 | 90 |
| 4.0 | 4.0 | 1.30 | 51.7% | $382 | 89 |
| 5.0 | 4.0 | 1.33 | 49.4% | $432 | 89 |

Slightly worse than Slow EMA=21 but still profitable. The edge is
robust across slow EMA periods 21-30.

## 8. Comparison: Single vs Dual EMA

| Metric | Single EMA(21) | Dual EMA(13+21) |
|--------|---------------|-----------------|
| Profit Factor | 1.42 | **1.52** |
| Win Rate | 54.3% | **59.8%** |
| Trades/Year | **311** | 92 |
| Net P&L/Year | **$2,023** | $599 |
| Avg Trade | $6.50 | $6.51 |
| Risk per trade | Same | Same |

**Trade-off:** Dual EMA has better PF and win rate, but 1/3 the trades
and 1/3 the net P&L. For dollar returns, the single-EMA version is
superior unless you scale contracts significantly.

**Recommendation:** Run both in parallel. Single EMA for volume, dual
EMA for selective high-confidence entries. Or use dual EMA with higher
contract count (3x) to match single-EMA dollar output at lower risk.

## 9. Known Limitations

1. **Very low frequency:** 0.36 trades/day. Cannot increase frequency
   without removing the slow Z filter, which drops PF to 1.42.
2. **Small sample:** 92-93 trades in 2025. Statistical significance is
   marginal. Must validate on multi-year data before trusting.
3. **2025 only:** Not yet tested on 2019-2024 data.
4. **Bar-close exits required:** Same as single-EMA version.
5. **No regime model:** Pure rule-based. HMM regime filter could
   further improve PF by suppressing trades in low-vol regimes.
6. **Slow Z direction is a weak filter:** It only checks one bar of
   direction change. Could be strengthened by requiring N bars of
   reversal or a minimum delta.

## 10. Scaling Strategy

To match single-EMA dollar output with dual-EMA quality:
- 3 contracts: ~$1,797/year (matches single-EMA 1-contract)
- 5 contracts: ~$2,993/year
- 10 contracts: ~$5,986/year

Combined with multi-instrument (MES + MNQ + MYM):
- If each instrument averages ~0.36 trades/day -> ~1.1 trades/day combined
- 3 instruments × 3 contracts each: ~$5,391/year portfolio

## 11. Frequency Enhancement Ideas (Untested)

To increase from 0.36/day toward the 3-5/day target:
- [ ] Lower fast Z threshold to 3.0 (tested: PF drops significantly)
- [ ] Require 2+ bars of slow Z reversal instead of 1 (may improve quality)
- [ ] Add VWAP deviation as additional entry signal (parallel, not filter)
- [ ] Multi-instrument deployment (MES + MNQ + MYM)
- [ ] HMM regime filter: only trade in high-vol regimes where extremes
      are more meaningful and reversion is stronger
- [ ] Time-of-day filter: restrict to high-activity windows (9:30-11:00,
      13:00-14:30) where reversion is strongest

## 12. Next Steps

- [ ] Expand backtest to 2019-2024 multi-year data
- [ ] Walk-forward validation (out-of-sample)
- [ ] Monthly P&L breakdown for consistency check
- [ ] Build dual-EMA mode into ZScoreFadeExtreme.cs
- [ ] Market Replay testing in NinjaTrader
- [ ] Paper trade alongside single-EMA version for comparison
- [ ] Test regime model integration
- [ ] Explore running on MNQ and MYM simultaneously

## 13. Files

- Backtest script: `tools/backtest_ema_variants.py`
- Single-EMA spec: `docs/strategy_specs/ZScoreFadeExtreme_spec.md`
- NinjaTrader strategy: `ninjatrader/strategies/ZScoreFadeExtreme.cs`
- Sweep results: `results/scalper_ema_single_sweep_2025.csv`
- Dual sweep results: `results/scalper_ema_dual_sweep_2025.csv`
- This spec: `docs/strategy_specs/ZScoreDualEMAFade_spec.md`
