# SPEC-009 — Valuation Gates: Wheel-FV Boundaries, Action Masks, NO_CALL

_Status: implemented v1 — 2026-08-27 (companion to fair-value-discount SPEC-002)_

## 0. Motivation and division of labor

VMI's post-mortem stands: valuation has little stock-ranking alpha, but it
can prevent economically irrational assignment and call-away decisions. This
spec uses valuation as a **constraint layer**, never a return predictor.

The sibling repo `../fair-value-discount` (SPEC-002) computes, per ticker,
a reliability-weighted **Wheel_FV** (intrinsic ensemble blended with a
haircut analyst target), `IV_Reliability`, `put_required_mos`, and analyst
sentiment, published daily as `fv_outputs/fair_value_ensemble_<date>.csv`.
That repo knows nothing about option chains or positions. **This spec is the
option-side consumption**: economic boundaries recomputed against live spot,
hard risk-engine rules, selector pre-filters, action-space masks, and the
daily-brief surface. wheel-strategy (the old sibling) gets nothing — it
remains archive-only per SPEC-008 §1c.

Asymmetry principle: **PUT = maximum acceptable acquisition price; CALL =
minimum acceptable exit price.** Both tested on the *net* basis (premium
included), not the raw strike:

```
put:   strike − premium ≤ PutBasisCeiling = min( FV·(1−MOS), spot·(1−D) )
call:  strike + premium ≥ ExitFloor      = max( f·FV, basis·(1+min_gain) )
```

The spot-discount leg of the ceiling is load-bearing: deep undervaluation
must never authorize a near-the-money put.

## 1. Requirements

* `VREQ-1` **Ingestion** (`rlbot/data/fv_ensemble.py`): load the newest
  `fair_value_ensemble_*.csv` from `DataConfig.fv_ensemble_dir` (default
  `../fair-value-discount/fv_outputs`) into `{ticker: WheelValuation}`
  (wheel_fv, reliability + tier, put_required_mos, sentiment, coverage).
  Rows without a `wheel_fv` are skipped. A file older than
  `ValuationGateConfig.max_age_days` (7) or a missing directory yields an
  empty map plus a warning — **gates degrade to no-op, mirroring the
  FAIR-unknown philosophy of SPEC-001 §3.1.** No look-ahead: only files
  dated ≤ the run date are eligible (PIT discipline for any future replay).
* `VREQ-2` **Boundaries recomputed live** (`rlbot/risk/valuation.py`):
  regime, ceiling, and floor are pure functions of
  `(WheelValuation, live spot, cost basis, config)` — intraday spot moves
  can't stale them. Regime bands on `spot/wheel_fv`: <0.80
  DEEP_UNDERVALUED, <0.95 UNDERVALUED, ≤1.05 FAIR_VALUED, ≤1.20
  EXPENSIVE, >1.20 VERY_EXPENSIVE.
* `VREQ-3` **Action masks** (the RL contribution — valuation constrains
  the *legal action space*, SPEC-001 identities untouched):

  | Regime | CASH (puts) allowed up to | STOCK (calls) allowed up to |
  |---|---|---|
  | DEEP_UNDERVALUED | VERY_AGGRESSIVE | DEFENSIVE only (protect upside) |
  | UNDERVALUED | AGGRESSIVE | CONSERVATIVE |
  | FAIR_VALUED | BALANCED | BALANCED |
  | EXPENSIVE | CONSERVATIVE | AGGRESSIVE |
  | VERY_EXPENSIVE | **WAIT only (NO_PUT)** | AGGRESSIVE |

  WAIT is always legal. `clamp_action` maps a blocked tier to the highest
  allowed tier at/below it (WAIT if none). Unknown/stale valuation → no
  masking. The frozen 3-state `ValuationState` in the Q-state is unchanged;
  masks are a risk-layer construct usable by the env later (training use
  remains blocked on PIT accrual — SPEC-002 non-goal).
* `VREQ-4` **Hard rules** (`rlbot/risk/engine.py`, keyword-only additions;
  every existing call site unchanged):
  `VAL-1:net_basis_above_ceiling` (puts), `VAL-2:very_expensive_no_put`,
  `VAL-3:below_exit_floor` (calls). RL proposes, risk engine disposes.
* `VREQ-5` **Selector pre-filter** (`rlbot/options/selector.py`, optional
  `valuation=` kwarg): puts must satisfy `strike − mid ≤ ceiling`; calls
  `strike + mid ≥ exit floor`. An emptied candidate list returns `None` —
  **NO_CALL / NO_PUT emerge naturally as WAIT** instead of "always sell
  something".
* `VREQ-6` **Daily brief** (`rlbot/assistant/daily.py`): openings table
  gains Wheel FV / Regime / Ceiling / **Prem req** (`max(0, strike −
  ceiling)` — the minimum live premium that makes the strike acceptable;
  verify against the broker quote since model premiums are synthetic).
  Blocked openings show `WAIT` with a `valuation gate:` reason. Position
  guidance flags an open CSP whose `strike − premium_fill` exceeds the
  ceiling and an open CC whose `strike + premium_fill` sits below the
  fundamental exit floor. Legend documents all of it.
* `VREQ-7` **Degradation**: no CSV, stale CSV, missing ticker, or
  `wheel_fv` empty → identical behavior to today (no gates), plus a brief
  warning. All 162 pre-existing tests must stay green.

## 2. Acceptance criteria

* `VAC-1`: loader parses a real ensemble CSV; stale file → `({}, warning)`;
  files dated after the run date are ignored.
* `VAC-2`: ceiling = min(115·0.85, 105·0.95) = 97.75 for FV 115 / spot 105 /
  MOS 15%; the spot leg binds at FV 200 / spot 150 (ceiling 142.50); exit
  floor = max(0.95·200, 160·1.10) = 190 for basis 160.
* `VAC-3`: VERY_EXPENSIVE masks every put to WAIT; DEEP_UNDERVALUED masks
  calls to {WAIT, CALL_DEFENSIVE}; `clamp_action(PUT_AGGRESSIVE, FAIR)` →
  PUT_BALANCED.
* `VAC-4`: `validate_open` emits VAL-1/2/3 exactly when the boundary is
  crossed and stays silent with `valuation=None`.
* `VAC-5`: selector drops put candidates whose net basis exceeds the
  ceiling and returns `None` when none survive; call side respects the
  exit floor together with the existing `strike ≥ cost_basis` rule.
* `VAC-6`: brief renders the new columns; a gated ticker reads
  `WAIT — valuation gate: ...`; full suite green.

## 3. Out of scope

* Feeding masks into training/backtests (PIT accrual not yet sufficient).
* Changing SPEC-001 frozen enums, delta bands, or the reward.
* FV computation of any kind (SPEC-002 owns it).
