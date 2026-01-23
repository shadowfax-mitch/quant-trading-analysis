# Agent Handoff: OFI Momentum Strategy

**From:** Claude
**To:** Codex
**Date:** 2025-03-31
**Status:** NO-GO (Walk-Forward Validation Failed)

---

## Summary

OFI Momentum achieved GO status on March 2025 test data but **failed walk-forward validation**. The strategy lost money on January-February training data, indicating regime dependency rather than a robust edge.

## What Was Tested (NO-GO)

| Approach | Configs | Best Result | Status |
|----------|---------|-------------|--------|
| OU Mean-Reversion | 5 timeframes | PF=1.06 | NO-GO |
| OU + HMM Regime Filter | 42 | All failed | NO-GO |
| Price Momentum (ROC) | 45 | PF=1.00 | NO-GO |
| MA Crossover | 18 | PF=0.92 | NO-GO |
| Breakout | 24 | PF=0.99 | NO-GO |
| OFI Contrarian | All | Negative P&L | NO-GO |
| **OFI Momentum** | **48** | **See below** | **NO-GO** |
| Adaptive Threshold | 36 | 0 consistent | NO-GO |
| Volatility Filter | 48 | 0 consistent | NO-GO |
| Kalman-Filtered OFI | 162 | 0 consistent | NO-GO |

**Total configurations tested:** ~530+

## OFI Momentum Results

### Test Period (March 2025) - Appeared Promising

| Config | Trades | Net P&L | PF (Net) | Avg Ticks |
|--------|--------|---------|----------|-----------|
| W=10, T=0.20, E=30 | 57 | $333 | 1.29 | 4.66 |
| W=5, T=0.30, E=10 | 30 | $248 | 2.68 | 7.97 |

### Walk-Forward Validation (Jan-Feb 2025) - Failed

| Config | Trades | Net P&L | Status |
|--------|--------|---------|--------|
| W=10, T=0.20, E=30 | 261 | **-$2,490** | LOSS |
| W=5, T=0.30, E=10 | 66 | **-$191** | LOSS |

**All 5 GO configs lost money in training period.**

## Root Cause Analysis

Regime analysis identified fundamental market differences:

| Metric | Jan-Feb | March | Ratio |
|--------|---------|-------|-------|
| Volatility | 0.0186% | 0.0356% | 1.9x |
| Extreme OFI (>0.20) | 6.2% | 0.9% | 0.14x |
| Long Win Rate | 49.9% | 65.1% | 1.3x |

**Conclusion:** March had uniquely favorable conditions. Extreme OFI was rare but meaningful. In Jan-Feb, the same threshold captured noise.

## Codex Concerns (Addressed)

1. **Side codes:** Validated - only 'A' (bid) and 'B' (ask) exist
2. **Execution model:** Changed to bar OPEN + 1-tick adverse slippage
3. **PF calculation:** Now uses NET P&L (not gross)
4. **Open positions:** Force-closed at end of period

## Attempted Fixes (All Failed)

### 1. Adaptive Percentile Threshold
- Use rolling percentile of |OFI| instead of fixed threshold
- **Result:** 0 configs profitable in both periods

### 2. Volatility Filter
- Only trade during high-volatility periods
- **Result:** 0 configs profitable in both periods

### 3. Kalman Filter
- De-noise OFI, use velocity confirmation
- **Result:** 162 configs tested, 0 profitable in both periods

## Phase 0 Status: NO-GO

The OFI momentum strategy is regime-dependent and cannot be validated with current data.

## Recommendations

1. **Archive this research** - Document learnings for future reference
2. **Collect more data** - Wait for April/May 2025 to see if March conditions recur
3. **Research alternatives:**
   - Multi-timeframe approaches
   - Microstructure features (order book depth)
   - Different instruments (MNQ may have different characteristics)

## Files Created

```
src/backtest_ofi_bars_v2.py        # Conservative execution model
src/walk_forward_validation.py     # Walk-forward testing
src/analyze_regime_difference.py   # Regime analysis
src/backtest_adaptive_threshold.py # Adaptive threshold attempt
src/backtest_vol_filter.py         # Volatility filter attempt
src/backtest_kalman_ofi.py         # Kalman filter attempt
docs/OFI_MOMENTUM_RESEARCH_CONCLUSION.md  # Full research summary
```

## Key Lesson

Walk-forward validation is essential. A strategy that achieves GO on test data is meaningless without training period validation. The March 2025 results were likely due to favorable market conditions, not a robust edge.

---

*Handoff complete. No action required from Codex unless new data becomes available.*
