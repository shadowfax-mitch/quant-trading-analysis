# Collaboration Playbook (Codex + Claude)

This document mirrors an AGENTS.md-style workflow for collaborative quant research and engineering. Copy it into new sprint projects to keep the same review cadence and rigor.

## 1. Mission and Ground Rules
- Primary goal: find a tradeable edge; protect capital first.
- Research in Python must be transferable to C# / NinjaTrader.
- All changes are hypothesis-driven, measurable, and logged.
- Prefer robustness over brilliance; test across regimes.
- Data is immutable; append-only; preserve raw vendor formats.

## 2. Roles and Review Loop
- **Builder (Codex)**: implements, documents, and proposes tests.
- **Reviewer (Claude)**: audits correctness, risks, and missing tests.
- **Swap roles** when useful (e.g., Claude implements, Codex reviews).
- No step is "done" without at least one review pass.

## 3. Handoff Protocol
- Use a shared `docs/agent_handoff/` folder.
- Name files with order and direction:
  - `NNN_CLAUDE_TO_CODEX_TOPIC.md`
  - `NNN_CODEX_TO_CLAUDE_TOPIC.md`
- Each handoff should include:
  - Scope and goals
  - Files touched or to be created
  - Assumptions and open questions
  - Explicit asks for the other agent

## 4. Work Phases (Suggested)
1) **Data & Feature Prep**
2) **Model / Signal Research**
3) **Backtest / WFA**
4) **Analytics & Validation**
5) **NinjaTrader Transfer Plan**

## 5. Review Checklist (Minimum)
- **Bias controls:** no look-ahead, no leakage, split locked early.
- **Cost realism:** slippage, commission, and spread crossing.
- **Data integrity:** monotonic timestamps, missing bars/ticks, schema validation.
- **Regime robustness:** at least one stress or regime split.
- **Metrics:** CAGR, Sharpe/Sortino, max DD, hit rate, turnover.
- **Trade count:** minimum sample size for decisions.

## 6. Coding Conventions
- One class per file (when in NinjaTrader / C#).
- Stable script names and namespaces to avoid workspace breakage.
- Parameterize user inputs with safe defaults.
- Use deterministic seeds and log key decisions.
- Add minimal comments only when logic is non-obvious.

## 7. Validation Gates (Go / No-Go)
- Positive P&L after all costs is necessary but not sufficient.
- Require minimum trade count and basic profitability thresholds.
- If all parameter combos fail risk constraints, flag as **no valid params**.
- If a model adds complexity, demand measurable edge improvement.

## 8. Communication Expectations
- Write short, precise handoffs; avoid large dumps.
- If unexpected changes are found, pause and ask how to proceed.
- Prefer explicit questions to assumptions when blocked.
- Document each decision in the relevant handoff.

## 9. Deliverables Snapshot
- `docs/agent_handoff/` for coordination and review notes.
- `docs/` for plans, assumptions, and validation outcomes.
- `src/` for research code, backtests, and analytics.
- `tests/` for unit and integration checks.

## 10. Done Definition
- Implementation + review notes
- Tests or validation run notes
- Risks and limitations documented
- Clear next steps and decision gate
