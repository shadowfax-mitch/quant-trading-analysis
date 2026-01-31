# Z-Score Runner Research Summary

## Goal
Find a way to increase trading frequency from ~3 trades/month (Z=5.0 mean reversion) to ~3 trades/day while maintaining profitability.

## Research Approach
Based on user insight: "Since achieving Z=5.0 is rare, are there indicators near the beginning of those extreme moves that could predict them?"

## Key Findings

### 1. Precursor Analysis
Analyzed 142 extreme Z-score events (|Z| >= 3.5):
- **Mean peak Z**: 4.31
- **Mean duration**: 13.8 bars (~69 min) from start to peak
- **Z-Velocity at start**: Mean 1.27 (high momentum indicator)
- **Volume ratio at start**: Mean 1.5x average

### 2. Trend-Following Approach (Failed)
Tried entering early at Z=1.5 to ride momentum to Z=3.5:
- **Result**: Negative OOS P&L
- **Problem**: Only 18% of Z=1.5 entries reach Z=3.5
- Too many false signals

### 3. "Zone Scalper" Approach (Mixed Results)
Entering later at Z=3.0, targeting Z=4.5:
- **OOS P&L**: +$7,900 (excellent)
- **Training P&L**: -$1,843 (concerning)
- **Win Rate**: 44%
- **Trades**: 23 in OOS

#### Exit Breakdown
| Exit Type | Count | P&L |
|-----------|-------|-----|
| TARGET | 7 | +$6,363 |
| MAX_HOLD | 6 | +$10,644 |
| STOP | 55 | -$14,773 |
| RTH_CLOSE | 4 | +$3,823 |

The winners (TARGET, MAX_HOLD) are much larger than losers (STOP).

### 4. Why Training vs OOS Divergence?
- Training (Jan-Feb): -$1,843
- OOS (March): +$7,900

Possible explanations:
1. March 2025 had favorable conditions for trend-following
2. Sample size too small (23 OOS trades)
3. Strategy may be curve-fit to OOS period

## Comparison to Original Strategy

| Metric | Z=5.0 Mean Reversion | Zone Scalper (Z=3.0) |
|--------|---------------------|----------------------|
| OOS P&L | +$1,178 | +$7,900 |
| Training P&L | +$417 | -$1,843 |
| OOS Trades | 21 | 23 |
| Win Rate | ~65% | 44% |
| Profit Factor | 3.23 | 1.19 |
| Consistency | All periods positive | Negative in training |

## Recommendation

### Primary Strategy: Z=5.0 Mean Reversion (Conservative)
- Proven profitable across all periods
- Higher win rate and profit factor
- Lower risk profile
- ~3 trades/month

### Experimental Strategy: Zone Scalper (High Risk)
- Needs more OOS validation
- Negative training performance is concerning
- Consider paper trading for 3+ months before live
- Higher potential return but higher variance

### Alternative Idea: Hybrid Approach
Combine both strategies:
1. **Primary**: Mean reversion at Z=5.0
2. **Add-on**: Zone scalp when Z hits 3.5 with tight stops

This way you get:
- Core profits from mean reversion
- Bonus scalps from zone trading
- Diversified approach

## Files Created
- `src/analyze_zscore_precursors.py` - Precursor pattern analysis
- `src/backtest_zscore_runner.py` - Trend-following backtest
- `src/analyze_runner_failures.py` - Failure analysis
- `src/backtest_zscore_zone_scalper.py` - Zone scalper grid search
- `src/verify_zone_scalper.py` - Final validation

## Next Steps
1. Build more cached data for Apr-Jul 2025 periods
2. Run Zone Scalper on extended OOS data
3. If still profitable, create NinjaTrader implementation
4. Paper trade both strategies in parallel
