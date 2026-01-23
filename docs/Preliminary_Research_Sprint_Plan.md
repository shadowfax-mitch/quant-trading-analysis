# Preliminary Research Sprint: Validating Mean-Reversion in MES Tick Data

## 1. Executive Summary

### Objective
To determine if a statistically significant and tradable mean-reversion "edge" exists in MES tick data. This sprint is strictly time-boxed to **10 business days**.

### Methodology
We will use a simplified implementation of the **Ornstein-Uhlenbeck (OU) process** to model mean-reversion dynamics. Signals will be generated when the price deviates significantly from the model's estimated mean. These signals will be evaluated using a basic, vector-based tick-level backtest that realistically accounts for transaction costs by **crossing the bid-ask spread**.

### Primary Success Criterion (Go/No-Go Decision)
The sprint is a "Go" if, and only if, the simplified strategy yields a **positive total Profit & Loss (P&L)** after transaction costs **and** clears basic tradability guards. A positive result validates the core thesis and greenlights the full State-Space project. A negative result indicates the raw mean-reversion edge is too weak to overcome costs, prompting a strategic review before further investment.

---

## 2. Sprint Plan & Task Breakdown

### **Phase 1: Accelerated Data Prep & Environment Setup (Days 1-2)**
**Goal:** To prepare a manageable, clean dataset for rapid research. Perfection is the enemy of speed here.

- [ ] **1.1. Minimal Environment Setup:**
    - Create a dedicated project folder (e.g., `ou_sprint/`).
    - Initialize a Git repository.
    - Set up a Python virtual environment and install core dependencies: `pandas`, `numpy`, `statsmodels`, `matplotlib`, `jupyter`.

- [ ] **1.2. Ingest a Data Subset:**
    - Select a representative **3-month period** of MES tick data. Avoid major event periods like the COVID crash for this initial test to get a "typical" market sample.
    - **Target window:** `2025-01-01` to `2025-03-13` (adjust only if those dates are unavailable in the tick archive).
    - Ensure the subset includes **best bid/ask** (and sizes if available). If only trades exist, pause and source NBBO data before proceeding.
    - Write a simple script (`src/ingest.py`) to load the raw CSV/Parquet files for this period into a Pandas DataFrame.

- [ ] **1.3. Perform "Good Enough" Cleaning:**
    - Standardize all timestamps to UTC.
    - Remove obvious outliers (e.g., trades with zero or negative price/size).
    - Create two essential columns for the backtest:
        - `mid_price`: `(best_bid + best_ask) / 2`
        - `spread`: `best_ask - best_bid`
    - If size data is available, add `microprice`: `(best_bid * ask_size + best_ask * bid_size) / (bid_size + ask_size)`.
    - Drop rows with non-positive spreads or missing bid/ask.
    - Save the cleaned subset to a single, fast-loading Feather or Parquet file (`data/sprint_data.parquet`).

- [ ] **1.4. Lock the Train/Test Split:**
    - Define a split before modeling.
    - **Train:** `2025-01-01` to `2025-02-28`
    - **Test:** `2025-03-01` to `2025-03-13`
    - Use train only for sanity checks; report metrics on test for the Go/No-Go decision.

### **Phase 2: OU Model Estimation & Signal Generation (Days 3-5)**
**Goal:** To implement the core "physics" model and generate trading signals.

- [ ] **2.1. Implement OU Parameter Estimation:**
    - In a new script (`src/ou_model.py`), create a function to estimate OU parameters (`mu`, `theta`, `sigma`) from a price series.
    - Use a simple, fast method like the one derived from the discrete-time AR(1) representation of the OU process. This can be done via a simple Ordinary Least Squares (OLS) regression.

- [ ] **2.2. Generate Rolling OU Parameters:**
    - Apply the estimation function on a **rolling window** over the `mid_price` series. A window of **10,000 ticks** is a reasonable starting point.
    - **Lag the parameters by one tick** (or one window) to avoid look-ahead bias.
    - If `theta <= 0` or `sigma <= 0`, mark the window invalid and suppress signals until parameters are valid again.
    - This will produce a time series of `mu(t)`, `theta(t)`, and `sigma(t)`, allowing the model to adapt to changing market conditions.

- [ ] **2.3. Implement Signal Generation Logic:**
    - In a Jupyter Notebook (`notebooks/01_Signal_Generation.ipynb`), define the entry rule based on the rolling OU parameters.
    - Create a `signal` column in your DataFrame, initialized to `0`.
    - **Parameter to Test `N`:** We will test `N` values of `[1.5, 2.0, 2.5]`.
    - For each `N`:
        - **Long Signal (+1):** Set `signal` to `1` where `mid_price < (mu - N * sigma)`.
        - **Short Signal (-1):** Set `signal` to `-1` where `mid_price > (mu + N * sigma)`.

- [ ] **2.4. Define Exit Condition:**
    - The exit condition is simple and symmetrical: **Exit any trade when the `mid_price` crosses its corresponding rolling mean `mu`**.
    - Use the lagged `mu` for exits to avoid reacting to same-tick information.

### **Phase 3: Simplified Tick-Level Backtest (Days 6-8)**
**Goal:** To simulate the generated signals against historical tick data, including realistic transaction costs.

- [ ] **3.1. Create a Vectorized Backtesting Script:**
    - In `src/backtest.py`, create a script that takes the DataFrame with signals as input. This will not be an event-driven backtester; a fast, vectorized approach using Pandas shifts and boolean indexing is sufficient for this sprint.

- [ ] **3.2. Simulate Trade Execution & Costs:**
    - Create columns for `position` and `pnl`.
    - When a `signal` of `+1` is triggered at tick `t`, assume a long position is entered on the **next tick (`t+1`)**. The entry price is the `ask` price at `t+1`.
    - When a `signal` of `-1` is triggered, the entry price is the `bid` price at `t+1`.
    - Hold the position until the exit condition is met (price crosses `mu`). The exit price for a long trade is the `bid` at the exit tick, and for a short trade, it's the `ask`.
    - This `bid/ask` logic automatically accounts for the cost of crossing the spread on every trade.
    - Add a fixed **commission/fees per side** (parameterized, default to **$0.85 per side** for MES/MNQ) and deduct on both entry and exit.

- [ ] **3.3. Log Trades:**
    - Iterate through the backtest and generate a list of all closed trades. Each entry should include: `entry_timestamp`, `exit_timestamp`, `direction`, `entry_price`, `exit_price`, and `trade_pnl`.
    - Store this trade log in a new DataFrame.

### **Phase 4: Analysis & Go/No-Go Decision (Days 9-10)**
**Goal:** To analyze the backtest results objectively and make the final strategic decision.

- [ ] **4.1. Calculate Key Performance Metrics:**
    - In a new Jupyter Notebook (`notebooks/02_Results_Analysis.ipynb`), load the trade log.
    - Calculate the essential metrics for each tested `N` value:
        - **Total P&L (after costs):** The primary success metric.
        - **Number of Trades:** Is the strategy active enough?
        - **Win Rate (%):** What percentage of trades were profitable?
        - **Average P&L per Trade:** Is the average trade profitable after costs?
        - **Profit Factor:** `Gross Profits / Gross Losses`.
        - **Average Trade in Ticks:** Must exceed a minimum threshold after costs (use MES tick size = 0.25).

- [ ] **4.2. Perform Sensitivity Analysis:**
    - Create a summary table comparing the performance metrics across the different `N` values (`1.5`, `2.0`, `2.5`).
    - This analysis shows how sensitive the strategy is to the entry threshold.

- [ ] **4.3. Create a Final Report and Make the Decision:**
    - Write a brief, one-page summary of the findings, including the summary table and a P&L equity curve plot.
    - **Present the Go/No-Go Decision:**
        - **GO:** The strategy shows **positive Total P&L** for at least two of the three `N` values tested **and** clears basic guards: **Profit Factor >= 1.1**, **Average Trade >= 1.0 tick**, and **min trades >= 30** in the test set. This confirms a raw edge exists. **Proceed with the full State-Space project plan.**
        - **NO-GO:** The strategy fails any of the above in the test set. This indicates the simple mean-reversion edge is not strong enough to overcome transaction costs. **Do not proceed. Re-evaluate the core thesis and research alternative alpha sources before building complex infrastructure.**

This focused plan provides the quickest path to the most important answer: **Is there a real, tradable edge here?** Answering this question first will ensure that any future engineering effort is built on a solid foundation of proven market dynamics.
