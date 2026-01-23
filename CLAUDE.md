# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**wave_signals_v4** is a quantitative research project for building an institutional-grade tick-level trading system on MES/MNQ futures. The project follows a phased, hypothesis-driven approach with explicit Go/No-Go gates and emphasizes capital protection as a first principle.

**Current Phase:** Early-stage (Phase 0: OU Mean-Reversion Validation Sprint)

## Development Commands

```bash
# Run the sprint (PowerShell)
.\scripts\run_sprint.ps1 -Python python -Runner src/run_sprint.py -OutRoot runs

# Expected Python workflow (once implemented)
python src/run_sprint.py
pytest tests/
```

## Architecture

### Multi-Phase Research Workflow

1. **Phase 0:** Tick Value Validation - prove tick data adds edge over coarser bars
2. **Preliminary Sprint:** OU Mean-Reversion Validation (10 business days)
3. **Phase 1:** Tick Data Infrastructure & Core Libraries
4. **Phase 2:** State-Space Model R&D (Fractional Differentiation, Kalman Filter, OU, GARCH)
5. **Phase 3:** Predictive Modeling (XGBoost, Bayesian Inference)
6. **Phase 4:** High-Frequency Backtesting & Integration
7. **Phase 5+:** Production & Live Trading (NinjaTrader C# integration)

### Core Modeling Components

- **Ornstein-Uhlenbeck (OU) Process:** Mean-reverting price dynamics
- **Fractional Differentiation:** Stationarity while preserving memory
- **Kalman Filter:** De-noises price, extracts velocity
- **GARCH(1,1):** Volatility forecasting for dynamic stops
- **XGBoost Classifier:** Short-term direction prediction
- **Kelly Criterion:** Mathematically optimal position sizing

### Two-Agent Collaboration Model

- **Builder (Codex):** Implements, documents, proposes tests
- **Reviewer (Claude):** Audits correctness, risks, missing tests
- **Handoff protocol:** `docs/agent_handoff/NNN_AGENT_TO_AGENT_TOPIC.md`
- No step is "done" without at least one review pass

## Data

### Dataset Location
- MES tick data: `datasets/MES/tick_data/` (104 files, 2025-01-01 to 2025-07-01)
- MNQ tick data: `datasets/MNQ/tick_data/` (124 files + 233-tick bars)

### Train/Test Split (Locked)
- **Train:** 2025-01-01 through 2025-02-28
- **Test:** 2025-03-01 through 2025-03-31

### Schema
`timestamp, last, volume, bid, ask, side, contract, source`

## Cost Model Assumptions

- **Spread crossing:** Entries at ask (long) / bid (short); exits at bid (long) / ask (short)
- **Commission:** $0.85 per side per contract
- **Slippage:** 0 ticks beyond bid/ask (base case)
- **Tick sizes:** MES 0.25, MNQ 0.25
- **Execution:** Signals at tick t filled at tick t+1; 1 contract per trade; full fills assumed

## Success Criteria (Phase 0 Gate)

- Positive P&L after costs for at least 2 of 3 N values (1.5, 2.0, 2.5)
- Profit Factor >= 1.1
- Average trade >= 1.0 tick (after costs)
- Minimum 30 trades in test set

## Review Checklist

Every change must pass:
1. **Bias controls:** No look-ahead bias, no leakage, split locked early
2. **Cost realism:** Slippage, commission, spread crossing modeled
3. **Data integrity:** Monotonic timestamps, schema validation
4. **Regime robustness:** At least one stress test or regime split
5. **Metrics:** CAGR, Sharpe/Sortino, max DD, hit rate, turnover
6. **Trade count:** Minimum sample size for statistical decisions

## Coding Conventions

- One class per file (NinjaTrader/C# standard carries to Python)
- Stable script names and namespaces
- Parameterize inputs with safe defaults
- Deterministic seeds; log key decisions
