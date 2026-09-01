# SPEC-011 — MCB Gates: Maximum-Comfortable-Basis Feed Replaces Wheel-FV

_Status: implemented v1 — 2026-08-31 (consumer side of `../mcb-wheel`'s
contract; supersedes SPEC-009 as the live valuation-gate input)._

## 0. Motivation

SPEC-009's Wheel-FV gates asked "is the stock cheap relative to a fair-value
estimate?" — a valuation question. The MCB report asks the wheel's actual
question directly: **what is the highest net cost basis (strike − premium) I
would still be comfortable owning?** The producer (`../mcb-wheel`) publishes
that number per ticker in three descending zones and resolves its own
behavioral guardrail into a minimum eligible tier. This repo consumes it as a
hard constraint layer — valuation remains a constraint, never a return
predictor (VMI post-mortem, unchanged).

Asymmetry principle preserved: **PUT = maximum acceptable acquisition price**
(now literally the MCB ceiling). The CALL side has no MCB analogue — MCB is
an acquisition-side construct; call discipline reduces to the standing
cost-basis rule (never write a call below basis, selector-enforced).

## 1. Producer interface (what mcb-wheel publishes)

`../mcb-wheel/outputs/mcb_<date>.csv` (universe = the FV sheet's sections),
consumed columns per ticker:

| Field | Meaning |
|---|---|
| `mcb_fair / mcb_attractive / mcb_excellent` | Descending net-basis ceilings (FAIR > ATTRACTIVE > EXCELLENT) |
| `min_eligible_tier` | Guardrail-resolved: NORMAL→FAIR, CAUTION→ATTRACTIVE, SEVERE→EXCELLENT |
| `guardrail_status` | NORMAL / CAUTION / SEVERE (NaN for ETFs) |
| `layer_a` | OWN / MONITOR_ONLY / HALT — ownership eligibility |
| `reachability` | NORMAL / PATIENCE / UNREACHABLE (advisory) |
| `conf` | Producer confidence; all-NaN zones ⇒ constraint absent |

ETFs get drawdown-derived zones (`etf_subtype`, dd50/75/90), so — unlike the
Wheel-FV era — **ETFs are gated too**, softening SPEC-010 §3's "constraint
absent" default (G6 remains open for technical anchors on top).

## 2. Consumer contract (as enforced here)

1. **HARD net-basis rule** — `strike − premium ≤ MCB(required tier)`, where
   `required tier = deeper of (report min_eligible_tier, regime posture)`;
   regime posture: BULL_LOW_VOL→FAIR, every other regime→at least ATTRACTIVE.
   A tier missing from the row falls back to the nearest *shallower*
   available tier (never loosen past what exists).
2. **Never trade** `layer_a ∈ {MONITOR_ONLY, HALT}` — masked to WAIT before
   the policy's tier even matters.
3. **Reachability is informational** (user decision 2026-08-31, diverging
   from the producer's recommended reading): the strike scan always runs;
   PATIENCE/UNREACHABLE ride along in the brief's `MCB flags` column and the
   JSON `mcb.advisory` field. In practice the hard ceiling blocks the same
   names — the signals agree by construction.
4. **Momentum decoupling** (producer contract rule): momentum inputs never
   touch put timing or the MCB constraint. Momentum selects *names only*
   (SPEC-010 Feed B); MCB covers candidates only if they appear on the FV
   sheet — otherwise the candidate trades constraint-absent with a standing
   not-validated note.
5. **Staleness** — a report older than 5 trading sessions is expired: gates
   no-op entirely with a warning (constraint absent, never a stale ceiling).

## 3. Implementation map

| Piece | Module | Notes |
|---|---|---|
| Loader | `rlbot/data/mcb_feed.py` | PIT-safe (files ≤ as_of only); NaN rows → absent + warning; `DataConfig.mcb_dir` |
| Tier/ceiling/masks | `rlbot/risk/mcb_gates.py` | `required_tier`, `mcb_ceiling`, `tradeable`, `reachability_advice`, `net_basis_flag` (MCB-1), `premium_required` |
| Selector pre-filter | `rlbot/options/selector.py` | `net_basis_ceiling=` kwarg, puts only; empty list → None → WAIT |
| Daily surface | `rlbot/assistant/daily.py` | Openings/candidates columns: `MCB tier / Ceiling / Prem req / MCB flags`; open-CSP flag when filled net basis > ceiling; `--download` runs `run_mcb.py` (sheet-driven) instead of `run_fv.py` |
| WAIT attribution | `rlbot/assistant/daily.py` | An empty scan blames the MCB ceiling **only if** rescanning without the ceiling finds a contract (else "tier unimplementable" — chain/liquidity); MCB reasons show the ceiling and the premium the best band strike would need |

`Prem req` = `max(0, strike − ceiling)`: the minimum live premium making the
shown strike acceptable — always verify against the broker quote.

## 4. Relationship to SPEC-009

SPEC-009's modules (`rlbot/risk/valuation.py`, `rlbot/data/fv_ensemble.py`,
engine VAL-1/2/3 flags, selector `valuation=` path) are **retained but no
longer feed the daily brief**. They remain for gate-history tooling and the
G7 retrospective (whose proxy-FV method is unaffected). The 5-band
WheelRegime action masks (VREQ-3) are superseded live by MCB tier deepening;
the frozen 3-state `ValuationState` in the Q-state is untouched, as always.

## 5. Acceptance (shipped, tests/test_mcb_gates.py)

- Loader: parse real-schema CSV; PIT; staleness expiry; NaN row → absent.
- Tiers: deeper-of logic incl. defensive-regime floor and SEVERE dominance;
  missing-zone conservative fallback.
- Masks: MONITOR_ONLY/HALT always WAIT; reachability advisory recorded,
  never the WAIT reason.
- Selector: harsh ceiling → None; survivors honor the bound.
- Daily: annotation (tier/ceiling/prem-req), honest WAIT attribution, CSP
  position flag, CC exempt; brief renders MCB columns + legend.
- Full suite green (204 at merge).

## 6. Out of scope

- MCB computation of any kind (`../mcb-wheel` owns it).
- Feeding MCB into training/backtests (no point-in-time MCB history exists).
- Call-side floors (cost-basis rule already covers the exit).
