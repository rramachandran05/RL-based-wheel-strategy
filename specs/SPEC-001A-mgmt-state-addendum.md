# SPEC-001A — Management-State Addendum (SPEC-001 v2)

_Status: draft v1 — 2026-08-22. Extends the frozen SPEC-001 by version bump, never mutation: opening-decision records remain `trajectory_v1` and are untouched; management-decision records use `trajectory_v2` (additive)._

## 1. Scope

Defines what SPEC-001 §3.3 reserved: the management state, the mechanics of the management actions from §2.3, management decision epochs, and the management reward. This is the interface for MVP-3 Track A (SPEC-007).

## 2. Management Q-state — `MgmtStateV1`

Same 36-cell discipline as the opening state, for the same effective-sample-size reasons (SPEC-001 §3.1 preamble). The QStateV1 valuation and vol-comp axes are **logged but not conditioned on** here — regime is kept because defensive management is regime-driven; the position's own geometry replaces the rest.

```python
MgmtStateV1 = (
    market_regime,      # int 0-3, same enum as QStateV1
    moneyness_bucket,   # int 0-2
    dte_bucket,         # int 0-2
)                       # 4 × 3 × 3 = 36 cells per management policy
```

**moneyness_bucket** — short put: `m = (spot − strike) / strike`; covered call: `m = (strike − spot) / spot` (positive = safe side in both):

| id | Name | Rule |
|---|---|---|
| 0 | BREACHED | m < 0 (option in the money) |
| 1 | NEAR | 0 ≤ m ≤ 0.05 |
| 2 | SAFE | m > 0.05 |

**dte_bucket:**

| id | Name | Rule |
|---|---|---|
| 0 | EXPIRY_WEEK | dte ≤ 7 |
| 1 | MID | 8 ≤ dte ≤ 21 |
| 2 | EARLY | dte ≥ 22 |

Logged-only per management record (mandatory, promotion via ablation only): `valuation_state`, `vol_compensation`, `premium_captured_bucket` (0: <50%, 1: 50–85%, 2: >85% of `premium_fill` decayed, i.e. `1 − mark/premium_fill`), current delta, unrealized P&L.

## 3. Management action mechanics

Enums are unchanged from SPEC-001 §2.3. Mechanics, previously unspecified:

| Action | Mechanics |
|---|---|
| HOLD | No trade; next management epoch per §4. |
| CLOSE | Buy to close at `mark × (1 + slippage_pct)` + commission → CASH (put) / LONG_STOCK (call). |
| ROLL_SAME_RISK | CLOSE, then immediately open a new contract via the selector at the tier whose delta band contains the *original* open's |delta| target, new 25–45 DTE expiry. One combined decision; both legs charged friction. |
| ROLL_LOWER_RISK | As above, one tier lower (further OTM). |
| ROLL_HIGHER_RISK | As above, one tier higher. Legal, but excluded from the MVP-3 sweep (rarely defensible; saves 1/6 of branch budget) — documented deviation. |
| ACCEPT_ASSIGNMENT / ALLOW_CALL_AWAY | Commit: no further management epochs this cycle; settle at expiration per SPEC-003 §5. |

Rolls pass through the risk engine like any opening trade (SPEC-004 §2); a rejected roll leg degrades to CLOSE.

## 4. Management decision epochs (extends SPEC-001 §5)

While an option is open, a management epoch fires on any of:

| # | Trigger |
|---|---|
| M1 | `mgmt_cadence` (default 5 trading days) since last management decision |
| M2 | Moneyness bucket changed (strike breach or recovery) |
| M3 | \|delta\| of the open option ≥ 0.40 (challenge threshold) |
| M4 | Market regime value changed |
| M5 | dte enters EXPIRY_WEEK (once) |

E2 (expiration) remains a forced resolution, not a decision.

## 5. Management reward — `diff_v2`

For a management decision at t with next management epoch t+Δt:

```
r = ΔNAV%(action branch) − ΔNAV%(HOLD branch)      over [t, t+Δt]
```

The reference is **HOLD on the same contract over the same window** — the incumbent MVP-2 rule. This directly prices the deviation ("was intervening better than doing nothing?"), is computable in every counterfactual sweep at zero extra cost (HOLD is always a branch), and makes G3's claim exact: the learned management policy must beat hold-to-expiry. `reward_version: "diff_v2"` on all management records.

## 6. `trajectory_v2` (additive)

`trajectory_v2` = `trajectory_v1` plus:

- `"mgmt_state"`: nullable `[int, int, int]` (MgmtStateV1) — required non-null when `position_state` ∈ {SHORT_PUT, COVERED_CALL}, null otherwise.
- `"features.raw"` gains: `open_strike`, `open_delta_now`, `open_dte`, `premium_captured`, `mark`.
- `schema_version: "trajectory_v2"`; `reward_version` may be `diff_v1` (opening) or `diff_v2` (management).

v1 consumers must reject v2 records (per SPEC-001 §7 rules); the learner reads both. Opening-only pipelines may keep emitting v1.

## 7. Acceptance criteria

- **AC-1** JSON Schema for trajectory_v2 checked in; a v2 management fixture validates; the same record fails the v1 schema (version discrimination works).
- **AC-2** Bucket golden tests: ≥ 9 hand-built (spot, strike, dte) cases covering every moneyness/dte bucket boundary (m = 0, m = 0.05, dte = 7/8/21/22).
- **AC-3** Roll mechanics test: ROLL_LOWER_RISK on a fixture produces close-fill + open-fill cash effects to the cent, and the new contract's band is one tier lower.
- **AC-4** HOLD branch reward ≡ 0 by construction in every sweep record (the reference is itself).
- **AC-5** Committed ACCEPT_ASSIGNMENT suppresses all subsequent management epochs in that cycle.
