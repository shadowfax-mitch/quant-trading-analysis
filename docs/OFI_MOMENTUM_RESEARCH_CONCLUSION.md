# OFI Momentum Strategy - Research Conclusion

**Date:** 2025-03-31
**Status:** NO-GO (Regime Dependent)
**Total Configurations Tested:** ~580+ (MES + MNQ)

## Executive Summary

OFI (Order Flow Imbalance) momentum on 1-minute bars achieved GO status on March 2025 test data but failed walk-forward validation on January-February 2025 training data. Extensive attempts to fix the regime dependency were unsuccessful.

## Testing Summary

### MES (Micro E-mini S&P 500)

| Approach | Configs Tested | Test GO | Both Periods Profitable |
|----------|---------------|---------|------------------------|
| Base OFI Momentum | 48 | 5 | 0 |
| Adaptive Percentile Threshold | 36 | 0 | 0 |
| Volatility Filter | 48 | 0 | 0 |
| Kalman-Filtered OFI | 162 | 9 | 0 |
| **MES Total** | **~530+** | **14** | **0** |

### MNQ (Micro E-mini Nasdaq)

| Approach | Configs Tested | Test GO | Both Periods Profitable |
|----------|---------------|---------|------------------------|
| Base OFI Momentum | 48 | 4 | 0 |
| **MNQ Total** | **48** | **4** | **0** |

### Combined Total: ~580 configurations, 0 consistent

## Best Configuration (Test Period Only)

- **Parameters:** Window=10, Threshold=0.20, Exit=30 bars
- **Test Results (March):** 57 trades, $333 net profit, PF=1.29
- **Train Results (Jan-Feb):** 261 trades, -$2,490 net profit

## Root Cause: Regime Difference

Regime analysis identified fundamental differences between periods:

| Metric | Jan-Feb | March | Implication |
|--------|---------|-------|-------------|
| Volatility (20-bar std) | 0.0186% | 0.0356% | March 2x more volatile |
| Extreme OFI frequency | 6.2% | 0.9% | March signals 7x rarer |
| Long signal win rate | 49.9% | 65.1% | March signals more reliable |
| Daily efficiency | Similar | Similar | Not a trend vs chop issue |

**Conclusion:** In March, when extreme OFI occurred, it represented genuine institutional flow with high follow-through. In Jan-Feb, the same threshold captured noise.

## Approaches Attempted

### 1. Adaptive Percentile Threshold
- **Theory:** Use rolling percentile of |OFI| to adapt threshold to regime
- **Result:** Failed - adaptive thresholds couldn't rescue Jan-Feb performance

### 2. Volatility Filter
- **Theory:** Only trade during high-volatility periods (like March)
- **Result:** Failed - Jan-Feb high-vol periods still had poor signal quality

### 3. Kalman Filter
- **Theory:** De-noise OFI, extract velocity for confirmation
- **Parameters tested:**
  - Process noise: 0.001, 0.005, 0.01, 0.05
  - Observation noise: 0.01, 0.05, 0.1
  - Velocity confirmation: enabled/disabled
- **Result:** Failed - filtering cannot create edge where none exists

## MNQ Cross-Validation

To verify whether the regime dependency was specific to MES (S&P 500) or fundamental to OFI momentum, we tested on MNQ (Nasdaq futures) with the same parameters.

**MNQ Results:**
- Train period (Jan-Feb): All 48 configs lost money
- Test period (March): 4 configs achieved GO, but all lost money in training
- Consistent configs: 0

**Conclusion:** The regime dependency is NOT instrument-specific. Both S&P 500 and Nasdaq show the same pattern - OFI momentum fails in Jan-Feb regardless of instrument.

## Key Insight

The issue is not signal processing - it's market microstructure. March 2025 had:
1. Higher conviction institutional flow
2. Less noise in the order book
3. Better price follow-through after OFI extremes

No amount of filtering, smoothing, or threshold adjustment can fix this. The edge simply doesn't exist in Jan-Feb 2025 data.

## Files Created

```
src/backtest_ofi_bars_v2.py        - Conservative execution model (MES)
src/walk_forward_validation.py     - Walk-forward testing
src/analyze_regime_difference.py   - Regime analysis
src/backtest_adaptive_threshold.py - Adaptive threshold attempt
src/backtest_vol_filter.py         - Volatility filter attempt
src/backtest_kalman_ofi.py         - Kalman filter attempt
src/backtest_mnq_ofi.py            - MNQ cross-validation
```

## Recommendations

### Short Term
1. **Archive OFI momentum** as a regime-dependent pattern
2. **Do not deploy** without regime detection
3. **Collect more data** - need to see if March-like conditions recur

### Medium Term
1. **Build regime classifier** that detects March-like conditions:
   - Volatility level (above historical 80th percentile)
   - OFI extreme rarity (below 2% frequency)
   - Signal win rate tracking (requires warm-up period)

2. **Research alternative approaches:**
   - Pure mean-reversion (OU) with tighter stops
   - Microstructure features (order book depth, trade clustering)
   - Multi-timeframe momentum (longer holding periods)

### Long Term
1. Deploy OFI momentum only when regime classifier confirms favorable conditions
2. Size positions according to regime confidence
3. Maintain kill switch for adverse regimes

## Lessons Learned

1. **Walk-forward validation is essential** - Test GO is meaningless without training validation
2. **Regime dependency is the primary risk** - even "good" signals fail in wrong conditions
3. **More data needed** - 3 months is insufficient for regime-dependent strategies
4. **Signal processing has limits** - cannot manufacture edge from noise

## Data Used

- **Instrument:** MES (Micro E-mini S&P 500)
- **Training:** January 1 - February 28, 2025 (~52,000 1-min bars)
- **Testing:** March 1 - March 31, 2025 (~23,000 1-min bars)
- **Total ticks:** 7.3 million

## Cost Model

- **Entry:** Bar open + 1 tick adverse slippage
- **Exit:** Bar open - 1 tick adverse slippage
- **Commission:** $0.85 per side ($1.70 round trip)
- **Tick value:** $1.25 per tick

---

*This research was conducted following the project's Go/No-Go framework with explicit gates and walk-forward validation.*
