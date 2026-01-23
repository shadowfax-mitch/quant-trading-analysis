# Assumptions

## Cost model
- Spread crossing: entries and exits occur at bid or ask, not midpoint.
- Commission and fees: $0.85 per side per contract (default for MES/MNQ).
- Slippage: 0 ticks beyond quoted bid or ask (adjust if stress testing).

## Tick size
- MES tick size: 0.25
- MNQ tick size: 0.25

## Execution rules
- Signals generated on tick t are filled on tick t+1.
- Long entry fills at ask(t+1); short entry fills at bid(t+1).
- Exit when mid crosses rolling mean; long exits at bid, short exits at ask.
- 1 contract per trade; no scaling or pyramiding.
- All fills assumed complete; no partial fills, queue priority ignored.
- Use full timestamp range unless a session filter is specified.
- Timestamps are standardized to UTC during preprocessing; raw files remain unchanged.

## Data integrity rules
- Drop rows with missing bid/ask or non-positive prices/volume.
- Drop crossed markets where bid >= ask; drop zero/negative spreads.
- Drop rows with spread > 10 ticks (configurable); log counts.
- Drop rows with off-grid prices (bid/ask/last not aligned to tick size within epsilon).
- Enforce non-decreasing timestamps in output; sort by timestamp if out-of-order data is detected.
- Drop exact duplicate rows; keep same-timestamp rows if price/size differ.
- Contract handling: do not stitch or roll-adjust by default; log unique contracts and counts, and optionally filter to a specified contract for sprint runs.
