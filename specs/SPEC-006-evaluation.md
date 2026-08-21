# SPEC-006 — Baselines, Walk-Forward Evaluation, Metrics, Gates

_Status: draft v1 — 2026-08-21_

## 1. Purpose

Baselines are built **before** any learning: they are the yardstick, the reward references (SPEC-001 §4), the Q-table prior (SPEC-005 §4), and the counterfactual continuation values (SPEC-005 §2). If the learned policy cannot reliably beat Baseline 3 out-of-sample, the learning layer is not justified — that is this project's falsifiable claim.

## 2. Baselines (all run through the SPEC-003 simulator, identical friction)

| # | Name | Policy |
|---|---|---|
| B1 | Fixed Wheel | Always PUT_BALANCED-equivalent at fixed 20-delta target, 30 DTE; calls 20-delta. **Also the cash-state reward reference.** |
| B2 | Conservative Wheel | Fixed 10–15 delta puts and calls |
| B3 | Adaptive Rules | regime-keyed: BULL_LOW_VOL + ATTRACTIVE → PUT_AGGRESSIVE; BULL_* + FAIR → PUT_BALANCED; SIDEWAYS → PUT_CONSERVATIVE; BEAR_STRESS → WAIT unless vol_comp=ATTRACTIVE → PUT_DEFENSIVE; calls mirrored (bearish/expensive → CALL_AGGRESSIVE). Full 36-state → action table checked in as a fixture (it doubles as the Q prior). |
| B4 | Buy-and-hold | 100 shares at episode start; **the stock-state reward reference.** |
| B5 | Grid-search Wheel | Re-implementation of the sibling's `optimize_wheel_mult` (per-ticker ATR-multiplier grid, optimized in-sample on the training window only) — an "optimized static" comparator |

## 3. Metric suite (per run, per ticker, and pooled)

Returns: CAGR, total return, annualized vol, Sharpe, Sortino, max drawdown, CVaR(5%) of monthly returns. Wheel-specific: premium income and yield, assignment rate, call-away rate, average |delta| and DTE at open, capital utilization, % time in cash / stock / short-option. Frictions: turnover, total costs. Learning-specific: differential return vs B3 (the headline), per-state action-agreement rate with B3.

**Segment every metric by regime** (the 4 regime values, plus crash/recovery sub-windows: 2008-09→2009-06, 2020-02→2020-08, 2022-01→2022-12 where data covers them). The regime table matters more than headline CAGR. Every report states the DATA-GAP-5 survivorship scope disclaimer.

## 4. Walk-forward protocol

Never randomly split time series. Expanding-window folds (dates configurable, defaults):

| Fold | Train | Validate | Test |
|---|---|---|---|
| F1 | 2010–2017 | 2018–2019 | 2020–2021 |
| F2 | 2010–2019 | 2020–2021 | 2022–2023 |
| F3 | 2010–2021 | 2022–2023 | 2024–present |

- Sweeps and estimation use Train only; promotion tests use Validate; Test is touched exactly once per fold by the final promoted table.
- Every feature (SMAs, VIX percentile, regimes, valuation, calibration scalar) is computed as-of (REQ-2.1); the G1 `iv_uplift` scalar is fitted before F1's train start where possible and never refit inside folds.

## 5. GATE G2 — the falsifiable claim

Promoted policy beats B3 on **test** windows, pooled across folds:

- Pooled differential annualized return vs B3 > 0, and > 0 in at least 2 of 3 folds
- Max drawdown ≤ 1.1 × B3's on each test window
- No regime segment worse than B3 by > 2 pts annualized
- Result must hold under the A/B split halves (SPEC-005 §3)

Failing G2 is a legitimate documented outcome: the deliverable is then the rule-driven assistant (B3 + selector + risk engine + explanations), and MVP 3+ is re-scoped before any further learning work.

## 6. Reports

`rlbot/evaluation/report.py` renders one markdown + one CSV per evaluation run into `data_local/reports/`: metric tables (pooled, per-ticker, per-regime), equity curves, the G1/G2 verdicts, config manifest, and disclaimers. The report is the artifact reviewed at each gate.

## 7. Requirements

- **REQ-6.1** All baselines implement the same policy interface as the learned policy (swap-in, no simulator changes).
- **REQ-6.2** B3's rule table fixture and the Q-prior derive from a single source file (no drift).
- **REQ-6.3** Fold boundaries enforced in code: a training call with a date range overlapping validate/test raises.
- **REQ-6.4** Metric functions unit-tested against hand-computed values on a synthetic NAV series.
- **REQ-6.5** Every evaluation report embeds the data-snapshot version, config hash, and table_version it evaluated.

## 8. Acceptance criteria

- **AC-1** All five baselines run end-to-end over the full universe/history without risk-engine violations; results persisted with manifests.
- **AC-2** Metric golden tests pass (REQ-6.4).
- **AC-3** Fold-leak test: overlapping-range training call raises (REQ-6.3).
- **AC-4** A full G2 evaluation runs from one command and emits the pass/fail verdict with all four criteria itemized.
- **AC-5** B1 reward-reference consistency: evaluating B1 against itself reports 0 differential everywhere (ties SPEC-003 AC-3).
