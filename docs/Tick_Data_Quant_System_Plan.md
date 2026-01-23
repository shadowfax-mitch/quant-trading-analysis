# State-Space Tick Data Quant System - Implementation Plan

## Executive Summary

This document outlines the end-to-end implementation of an institutional-grade quantitative trading system for MES futures, built on tick-level data. The architecture transitions from standard pattern recognition to a first-principles model of price dynamics and risk geometry.

**Core Architectural Pillars:**
1.  **Fractional Differentiation (The Data Fix):** To achieve data stationarity while preserving the "memory" essential for predictive modeling.
2.  **Ornstein-Uhlenbeck Process (The Physics Fix):** To model price movement as a mean-reverting process, capturing the "pull-to-value" dynamics crucial for scalping, rather than a simple random walk.
3.  **GARCH Model (The Risk Fix):** To forecast volatility dynamically, allowing stop-loss barriers to adapt to changing market conditions.
4.  **Kelly Criterion (The Sizing Fix):** To employ mathematically optimal position sizing based on a Bayesian-updated win probability, maximizing geometric growth while managing risk of ruin.

**System Workflow:**
- **Transform:** Raw tick data is made stationary via Fractional Differentiation.
- **De-Noise:** A Kalman Filter extracts the true underlying state (price, velocity) from noisy observations.
- **Predict:** An XGBoost classifier predicts short-term direction using fractionally differenced prices, Kalman-filtered velocity, and order flow features.
- **Verify:** A Bayesian Inference engine updates the win probability (`P(win)`) based on real-world performance.
- **Size & Limit:** The Kelly Criterion calculates optimal position size, and a GARCH model sets dynamic stop-loss levels.

---

## Codex Annotations (Scope and Risk Controls)

1. Add a Phase 0 to prove tick data adds value versus coarser bars before committing to full tick-native modeling.
2. Prefer midprice/microprice and event-time bars (volume/imbalance) as the base series to reduce microstructure noise.
3. Cap Kelly sizing (fractional Kelly + hard risk limits) to respect capital protection rules.
4. Define a NinjaTrader transfer plan early (feature parity, model export or simplified model).

---

## Phase 0: Tick Value Validation and Bar Builder (Weeks 0-2)

**Dependencies:** None
**Goal:** Prove tick-level data adds edge over OHLCV bars before full system build.

- [ ] 0.A.1: Load a 3-6 month tick sample (MES) and validate schema/timestamps.
- [ ] 0.A.2: Build a bar builder that can aggregate ticks into 5/15/30-min OHLCV.
- [ ] 0.A.3: Recompute existing factors on 15/30-min bars and compare WFA to 5-min baseline.
- [ ] 0.A.4: Build event-time bars (volume/imbalance) and compare signal quality versus time bars.
- [ ] 0.A.5: Decide go/no-go for full tick-native pipeline based on measurable improvement.

**Phase 0 Success Checkpoint:**
- [ ] Clear evidence that tick-derived bars (time or event-time) improve robustness versus 5-min baseline.
- [ ] If no improvement, pause full tick-native build and focus on bar-based strategy refinement.

**Phase 0 Go/No-Go Decision Checklist:**
- [ ] OOS Sharpe improves by at least +0.3 over the 5-min baseline on WFA.
- [ ] Max drawdown does not worsen by more than 10% relative to baseline.
- [ ] Minimum trades per fold >= 30 (avoid unstable Sharpe from low counts).
- [ ] Execution cost model shows net edge after spread/commission assumptions.
- [ ] Feasibility: tick storage + processing meets performance targets on sample set.

---

## Phase 1: Tick Data Infrastructure & Foundational Engineering (Weeks 1-4)

**Dependencies:** None - can start immediately
**Critical Path:** Track A (Tick data processing) and Track B (Core library development)

### Parallel Track A: High-Frequency Data Pipeline
**Agent:** Claude | **Skills:** Data engineering, Pandas, high-frequency data formats

- [ ] 1.A.1: Acquire 5-year MES tick dataset (e.g., from TickData, FirstRateData) and verify schema (Timestamp, Price, Size, Bid/Ask).
- [ ] 1.A.2: Create `src/data/tick_ingestion.py` for loading and parsing raw tick files.
- [ ] 1.A.3: Implement robust timestamp parsing (nanosecond precision) and UTC standardization.
- [ ] 1.A.4: Implement tick data cleaning: remove zero/negative prices/sizes, correct exchange timestamp errors.
- [ ] 1.A.5: Choose and implement a high-performance storage solution. Options:
    - **Arctic (MongoDB-based):** Excellent for versioning and time-series queries.
    - **TimescaleDB (PostgreSQL extension):** Powerful for complex SQL queries on time-series.
    - **Partitioned Parquet/Feather:** Simpler, file-based approach.
- [ ] 1.A.6: Write ingestion script to process and store all 5 years of data in the chosen format.
- [ ] 1.A.7: Create data quality report: ticks per day, trade frequency distribution, bid-ask spread analysis.
- [ ] 1.A.8: Implement a production bar builder (time bars + event-time bars) for later comparison and backtests.

### Parallel Track B: Core Mathematical Libraries
**Agent:** Codex | **Skills:** Numerical computing, Python, statistical libraries

- [ ] 1.B.1: Create `src/math/fractional_diff.py`. Implement the Fixed-Window Fractional Differentiation algorithm.
- [ ] 1.B.2: Create `src/math/ornstein_uhlenbeck.py`. Implement OU parameter estimation via Maximum Likelihood Estimation (MLE).
- [ ] 1.B.3: Create `src/math/garch.py`. Implement a GARCH(1,1) model wrapper using the `arch` library for forecasting volatility.
- [ ] 1.B.4: Create `src/math/kelly.py`. Implement a capped Half-Kelly Criterion (with hard max size and drawdown caps).
- [ ] 1.B.5: For each module, create comprehensive unit tests with known inputs and expected outputs. Validate `fracdiff` against a reference R implementation.

### Parallel Track C: Order Flow Feature Engineering
**Agent:** Claude | **Skills:** Market microstructure, data manipulation

- [ ] 1.C.1: Create `src/features/order_flow.py` for processing raw ticks into meaningful features.
- [ ] 1.C.2: Implement Tick Imbalance Bars: sample data not by time, but by order flow imbalance.
- [ ] 1.C.3: Implement Volume Order Imbalance (VOI) calculation on a rolling tick window.
- [ ] 1.C.4: Implement VWAP (Volume-Weighted Average Price) calculation and deviation features.
- [ ] 1.C.5: Develop a feature pipeline to compute these features and join them with the tick data stream.
- [ ] 1.C.6: Create unit tests to validate the correctness of each order flow feature calculation.

### Parallel Track D: Project & Testing Infrastructure
**Agent:** Codex | **Skills:** DevOps, Pytest, Python packaging

- [ ] 1.D.1: Set up project structure, Git repository, and `pyproject.toml` (similar to previous plan).
- [ ] 1.D.2: Initialize Pytest framework with fixtures for loading sample tick data.
- [ ] 1.D.3: Set up structured logging, configuration management (`config.yaml`), and a `README.md`.
- [ ] 1.D.4: Create unit tests for the data pipeline (Track A) and feature pipeline (Track C).

**Phase 1 Success Checkpoint:**
- ✅ 5 years of tick data cleaned, stored, and queryable with millisecond-level performance.
- ✅ Core mathematical libraries (FracDiff, OU, GARCH, Kelly) implemented and unit-tested.
- ✅ Order flow feature pipeline generates at least 3 features (VOI, Tick Imbalance, VWAP-dev).
- ✅ Unit test coverage for all new modules > 85%.

---

## Phase 2: State-Space Model R&D (Weeks 5-9)

**Dependencies:** Phase 1 complete.
**Critical Path:** Track A (Fractional Differentiation) followed by Track B (Kalman Filter).

### Parallel Track A: Fractional Differentiation Pipeline (Step 1: Transform)
**Agent:** Claude | **Skills:** Time series analysis, `fracdiff` library

- [ ] 2.A.1: Create `notebooks/01_fractional_differentiation.ipynb` for research.
- [ ] 2.A.2: Implement a process to find the optimal differentiation parameter `d` (e.g., iterate `d` from 0.1 to 0.9 and find the minimum `d` that results in a stationary series via ADF test).
- [ ] 2.A.3: Apply the optimal `fracdiff` transformation to the 5-year tick price series.
- [ ] 2.A.4: Validate that the resulting series is stationary (ADF test p-value < 0.05).
- [ ] 2.A.5: Analyze the memory preservation by comparing the autocorrelation of the `fracdiff` series vs. a standard first-difference series.
- [ ] 2.A.6: Create `src/pipelines/transformation_pipeline.py` to productionize this step.

### Parallel Track B: Kalman Filter De-Noising (Step 2: De-Noise)
**Agent:** Codex | **Skills:** State-space models, signal processing, `pykalman`

- [ ] 2.B.1: Design the state-space model for the Kalman Filter.
    - State Vector: `[price, velocity]`
    - Observation: Fractionally differenced midprice/microprice (preferred over last trade).
- [ ] 2.B.2: Implement the Kalman Filter using a library like `pykalman`.
- [ ] 2.B.3: Tune the filter's noise parameters (`observation_covariance`, `transition_covariance`) on a sample of data.
- [ ] 2.B.4: Apply the filter to the full `fracdiff` series to extract the smoothed "true" price and velocity.
- [ ] 2.B.5: Visualize the output: plot the noisy `fracdiff` series against the smoothed Kalman-filtered price.

### Parallel Track C: Ornstein-Uhlenbeck Process Analysis (Physics Model)
**Agent:** Claude | **Skills:** Econometrics, statistical modeling

- [ ] 2.C.1: Using the de-noised price series from the Kalman Filter (Track B), estimate the OU parameters:
    - `theta` (speed of reversion)
    - `mu` (long-term mean level)
    - `sigma` (volatility of the process)
- [ ] 2.C.2: Validate the OU model fit. The model is only valid if `theta > 0`.
- [ ] 2.C.3: Calculate the characteristic time scale of mean reversion (1 / `theta`). This gives an objective measure of how long deviations from the mean are expected to last.
- [ ] 2.C.4: Define "Overbought/Oversold" quantitatively as a function of standard deviations from the OU mean `mu`.

### Parallel Track D: GARCH Volatility Forecasting (Risk Model)
**Agent:** Codex | **Skills:** Volatility modeling, `arch` library

- [ ] 2.D.1: Using the residuals from the Kalman Filter (observation - filtered_state), fit a GARCH(1,1) model.
- [ ] 2.D.2: Validate the GARCH model fit (ARCH effects, parameter significance).
- [ ] 2.D.3: Create a function to produce a one-step-ahead volatility forecast at any point in time.
- [ ] 2.D.4: Backtest the GARCH forecasts against realized volatility over the next N ticks to ensure predictive power.

**Phase 2 Success Checkpoint:**
- ✅ Optimal fractional differentiation parameter `d` found, producing a stationary series.
- ✅ Kalman Filter successfully de-noises the price series, yielding a smooth underlying price and velocity.
- ✅ OU process parameters estimated with `theta > 0`, confirming mean-reverting dynamics.
- ✅ GARCH model successfully fits the data and produces reasonable short-term volatility forecasts.

---

## Phase 3: Predictive Modeling & Strategy Formulation (Weeks 10-14)

**Dependencies:** Phase 2 complete.
**Critical Path:** Track A (XGBoost Classifier Development).

### Track A: XGBoost Classifier Development (Step 3: Predict)
**Agent:** Codex | **Skills:** Machine learning, XGBoost, feature engineering

- [ ] 3.A.1: Define the prediction target (labeling). A common choice for HFT is ternary classification:
    - `+1`: Price will cross `mid + N ticks` before `mid - N ticks`.
    - `-1`: Price will cross `mid - N ticks` before `mid + N ticks`.
    - `0`: Neither (timeout). This is the "meta-labeling" approach.
- [ ] 3.A.2: Assemble the feature set for the XGBoost model:
    - **Primary:** FracDiff Price, Kalman Velocity.
    - **Order Flow:** VOI, Tick Imbalance, VWAP-dev.
    - **OU-based:** Current deviation from the OU mean `mu`.
- [ ] 3.A.3: Implement time-series cross-validation (e.g., Purged K-Fold) to prevent lookahead bias in model validation.
- [ ] 3.A.4: Tune XGBoost hyperparameters using Bayesian optimization on a validation set.
- [ ] 3.A.5: Train the final XGBoost model on the full training set.
- [ ] 3.A.6: Evaluate model performance: Accuracy, F1-score, and importantly, the probability calibration (is a 70% prediction correct 70% of the time?).
- [ ] 3.A.7: Save the trained model and feature pipeline to `models/`.

### Track B: Bayesian Inference Framework (Step 4: Verify)
**Agent:** Claude | **Skills:** Bayesian statistics, probability theory

- [ ] 3.B.1: Design the Bayesian updating framework.
    - **Prior:** Start with a prior belief about the strategy's win rate `P(win)` (e.g., a Beta distribution centered at 55%).
    - **Likelihood:** The outcome of each trade (win/loss).
- [ ] 3.B.2: Implement the update rule: `Posterior = Prior * Likelihood`. For a Beta distribution, this is a simple update of its alpha and beta parameters.
- [ ] 3.B.3: The output of this module will be a constantly updated `P(win)` estimate, which will be fed into the Kelly Criterion. For backtesting, this can be simulated as a rolling update.

### Track C: Strategy Logic Formulation
**Agent:** Claude | **Skills:** Quantitative strategy design

- [ ] 3.C.1: Define the precise entry conditions in `src/strategies/state_space_scalper.py`:
    - `Entry Condition`: XGBoost prediction probability > `threshold` (e.g., 0.65) AND the OU model confirms a mean-reverting setup (e.g., price is `> 1.5 std dev` from `mu` for a short signal).
- [ ] 3.C.2: Define the exit conditions:
    - `Take Profit`: Price crosses the OU mean `mu`, or a fixed tick target.
    - `Stop Loss`: To be determined dynamically by the GARCH model (Phase 4).

### Track D: NinjaTrader Transfer Plan (C# Parity)
**Agent:** Codex | **Skills:** C# transfer, model packaging

- [ ] 3.D.1: Define a minimal feature set that can be computed in NinjaTrader at tick frequency.
- [ ] 3.D.2: Decide model deployment path: ONNX export or simplified linear model for C#.
- [ ] 3.D.3: Specify latency and data requirements for live NinjaTrader integration.

**Phase 3 Success Checkpoint:**
- ✅ XGBoost model achieves an F1-score significantly better than random guessing on out-of-sample data.
- ✅ The model's probability outputs are well-calibrated.
- ✅ A clear, implementable set of entry/exit rules is defined, combining the predictive model with the physics model.

---

## Phase 4: High-Frequency Backtesting & Integration (Weeks 15-20)

**Dependencies:** Phase 3 complete.
**Critical Path:** Track A (HF Backtester) and Track C (Full Integration).

### Track A: High-Frequency Event-Driven Backtester
**Agent:** Codex | **Skills:** System architecture, low-latency programming, Cython/Numba (optional)

- [ ] 4.A.1: Design a backtester capable of processing tick-by-tick data. This is a major build.
- [ ] 4.A.2: Accurately model the bid-ask spread and the order book queue. A simple "cross the spread" cost model is a minimum requirement.
- [ ] 4.A.3: Implement `MarketOrder` simulation (filled at ask for buy, bid for sell) and `LimitOrder` simulation (filled when market price touches the limit).
- [ ] 4.A.4: Handle realistic latency simulation (e.g., a 5-10ms delay from signal to order).
- [ ] 4.A.5: Ensure the backtester logs every event: signal, order submission, fill, cancellation.
- [ ] 4.A.6: Optimize the backtesting loop for performance, potentially using Numba or Cython for the hottest loops.

### Track B: Strategy Backtesting & Walk-Forward Analysis
**Agent:** Claude | **Skills:** Performance analysis, statistics

- [ ] 4.B.1: Integrate the `StateSpaceScalper` strategy from Phase 3 into the new backtester.
- [ ] 4.B.2: Implement the same Walk-Forward Analysis (WFA) methodology as the previous plan, but on a much shorter time scale (e.g., train on 3 months, test on 1 month, rolling forward).
- [ ] 4.B.3: During WFA, all models (XGBoost, GARCH, OU, etc.) must be re-estimated on each new training fold.
- [ ] 4.B.4: Aggregate the out-of-sample performance from all WFA folds.

### Track C: Sizing & Risk Integration (Step 5: Size & Limit)
**Agent:** Codex | **Skills:** Risk management, system integration

- [ ] 4.C.1: Integrate the Bayesian `P(win)` updater into the backtest loop. The `P(win)` should update after each simulated trade.
- [ ] 4.C.2: Integrate the capped Half-Kelly module. Apply hard caps on max position size, max drawdown, and per-trade risk (capital protection first).
- [ ] 4.C.3: Integrate the GARCH volatility forecast. The stop-loss for each trade should be set dynamically (e.g., `Entry Price - 3 * GARCH_forecast`).
- [ ] 4.C.4: Run the full, integrated backtest through the WFA process.

### Track D: Performance & Transaction Cost Analysis
**Agent:** Claude | **Skills:** Financial metrics, TCA

- [ ] 4.D.1: Analyze the final portfolio performance from the WFA backtest.
- [ ] 4.D.2: **Key Metrics:** Sharpe Ratio, Calmar Ratio, Sortino Ratio, Profit Factor, Max Drawdown. Given the high frequency, metrics like "P&L per trade" and "average hold time" are also critical.
- [ ] 4.D.3: Conduct a thorough Transaction Cost Analysis (TCA). How much of the gross P&L was consumed by crossing the spread and commissions? A successful scalping strategy must be profitable *after* these costs.
- [ ] 4.D.4: Generate final backtest reports with visualizations.

**Phase 4 Success Checkpoint:**
- ✅ HF backtester can process 5 years of tick data and produces realistic fill simulations.
- ✅ The fully integrated strategy is profitable (Sharpe > 1.5) after transaction costs in the aggregated WFA results.
- ✅ Dynamic position sizing via Kelly leads to better risk-adjusted returns than fixed-size betting.
- ✅ GARCH-based stops are shown to control risk effectively during volatile periods.

---

## Phase 5 & 6: Production, Live Trading & Iteration (Weeks 21+)

These phases mirror the structure of the previous plan but are adapted for a high-frequency, model-driven system.

- **Infrastructure:** Focus on a low-latency setup. Co-location with the exchange's servers might be considered for a fully mature system. A VPS physically close to the exchange is a good start.
- **Code Optimization:** The live trading bot code must be highly optimized. Python's `asyncio` might be used, or critical parts rewritten in C++ or Rust.
- **Live Monitoring:** The dashboard must monitor not just P&L, but also the health of all models in real-time.
    - Is the current market data consistent with the OU model's assumptions?
    - Is the GARCH forecast tracking realized volatility?
    - Are the XGBoost prediction probabilities drifting?
- **Model Retraining:** A robust, automated pipeline for daily or weekly retraining of all models is not optional; it is a core requirement.
- **Bayesian Loop:** The mechanism to feed live trade outcomes back to the Bayesian `P(win)` updater must be robust and automated. This closes the loop on the entire system.

This plan is significantly more complex from an engineering and mathematical perspective, but it directly addresses the core challenges of building a robust trading system on unstable, noisy, high-frequency data.
