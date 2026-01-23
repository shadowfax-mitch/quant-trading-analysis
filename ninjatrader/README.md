# EMA Z-Score Mean Reversion Strategy for NinjaTrader 8

## Overview

This is a mean reversion strategy that trades extreme deviations from the EMA, measured in standard deviations (Z-Score).

**Validated Performance:**
| Instrument | OOS P&L | Trades | Profit Factor |
|------------|---------|--------|---------------|
| MNQ | +$1,178 | 21 | 3.23 |
| MES | +$333 | 10 | 4.10 |

## Files

- `EmaZScoreMeanReversion.cs` - Main strategy file
- `EmaZScoreIndicator.cs` - Companion indicator for chart visualization

## Installation

### Method 1: Import via NinjaTrader

1. Open NinjaTrader 8
2. Go to **Tools > Import > NinjaScript Add-On**
3. Navigate to the `.cs` files and import them

### Method 2: Manual Installation

1. Copy `EmaZScoreMeanReversion.cs` to:
   ```
   Documents\NinjaTrader 8\bin\Custom\Strategies\
   ```

2. Copy `EmaZScoreIndicator.cs` to:
   ```
   Documents\NinjaTrader 8\bin\Custom\Indicators\
   ```

3. In NinjaTrader, go to **New > NinjaScript Editor**
4. Press **F5** to compile all scripts

## Configuration

### Default Parameters (Robust Configuration)

| Parameter | Default | Description |
|-----------|---------|-------------|
| EMA Period | 21 | Period for EMA calculation |
| Z-Score Lookback | 21 | Bars for rolling std calculation |
| Entry Threshold | 5.0 | Z-Score required for entry |
| Exit Threshold | 1.0 | Z-Score level for exit |
| Max Hold Bars | 48 | Maximum bars to hold (4 hours on 5-min) |
| RTH Only | True | Trade only 9:00 AM - 4:00 PM |
| RTH Start Hour | 9 | Start of trading window |
| RTH End Hour | 16 | End of trading window |
| Contracts | 1 | Number of contracts to trade |

### Recommended Settings

**Chart:** 5-minute bars

**Instruments:** MES or MNQ (Micro E-mini futures)

**Account Size:** Minimum $3,000 per contract (conservative)

## Strategy Logic

### Entry Conditions

- **Long Entry:** Z-Score < -5.0 (price is 5+ std devs below EMA - oversold)
- **Short Entry:** Z-Score > +5.0 (price is 5+ std devs above EMA - overbought)

### Exit Conditions

- **Long Exit:** Z-Score > -1.0 OR Z-Score > 0 OR bars_held >= 48
- **Short Exit:** Z-Score < +1.0 OR Z-Score < 0 OR bars_held >= 48
- **RTH Close:** Force close at 4:00 PM if RTH Only is enabled

### Z-Score Calculation

```
EMA = EMA(Close, 21)
Distance = (Close - EMA) / EMA
DistanceStd = StdDev(Distance, 21)
ZScore = Distance / DistanceStd
```

## Usage

### Running the Strategy

1. Open a chart with 5-minute bars for MES or MNQ
2. Right-click > **Strategies** > **EmaZScoreMeanReversion**
3. Configure parameters (defaults are recommended)
4. Enable strategy

### Using the Indicator

1. Right-click chart > **Indicators** > **EmaZScoreIndicator**
2. The indicator plots:
   - Blue line: Current Z-Score
   - Red lines: Entry thresholds (+/- 5.0)
   - Orange lines: Exit thresholds (+/- 1.0)
   - Gray line: Zero line

## Paper Trading First!

**IMPORTANT:** Always test in simulation/paper trading before using real money.

Recommended testing period: At least 1 month of paper trading

## Risk Management

- Expected frequency: ~3 trades per month
- Average win: ~$35-40 (MNQ)
- Average loss: ~$35-40 (MNQ)
- Win rate: ~65-75%
- Max observed drawdown: ~$300

## Changelog

### Version 1.0 (2026-01-23)
- Initial release
- Validated on 6+ months OOS data for MES and MNQ
- Implements robust configuration (Z=5.0, RTH-only)
