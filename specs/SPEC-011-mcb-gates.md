# SPEC-011 — MCB Gates: Maximum-Comfortable-Basis Feed Replaces Wheel-FV

_Status: implemented v1 — 2026-08-31 (consumer side of `../mcb-wheel`'s
contract; supersedes SPEC-009 as the live valuation-gate input).
§6 opportunity scan added 2026-09-01 — **specced, not yet implemented**._

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

## 6. MCB opportunity scan — advisory pathway (v2, 2026-09-01)

_Design review conclusion: the WAIT behavior when no in-band strike clears
the MCB ceiling is logically correct, but the selector was too rigid in how
it defined "conservative." **Delta describes risk; economics determine
whether the trade is worthwhile.** The MCB ceiling stays hard; the
DEFENSIVE delta floor (0.05) stays the executable boundary; what was
missing is visibility into MCB-compliant strikes *below* the bands._

### 6.1 Placement in the architecture

```
RL action → normal delta-band selector → MCB gate
                     │ no candidate
                     ▼
        MCB_OPPORTUNITY_SCAN (advisory, outside the RL action space)
                     │
        economics + liquidity ──→ unattractive        → WAIT (with the data)
                              └─→ potentially attractive → ⚠ HUMAN REVIEW
```

The scan is **not** the RL policy choosing another action — the frozen
SPEC-001 action semantics are untouched (DEFENSIVE still means Δ 0.05–0.10
for the executable scan; letting the selector wander to Δ 0.003 while still
calling it DEFENSIVE would change the meaning of an RL action without
retraining). The scan is the system saying: *"your requested action couldn't
produce an acceptable trade — here is whether anything farther OTM is worth
human consideration."*

### 6.2 Algorithm

Runs only after the normal tier scan returns no MCB-compliant candidate,
and only when an MCB ceiling exists for the name:

1. Examine strikes **below** the tier band, progressively farther OTM —
   including below the 0.05 DEFENSIVE floor — within the normal DTE window
   (25–45).
2. Require `strike − premium ≤ MCB(required tier)` (the hard rule, unchanged).
3. Apply the normal RISK-6 liquidity rules (min OI, spread caps).
4. Compute annualized return on the escrowed capital:

   `ROC_ann = premium / (strike − premium) × 365 / DTE`

   (the question is not "can I buy AAPL at $200?" but "is someone paying me
   enough to reserve $20K while I wait for that possibility?").
5. Surface the best candidate **only as an advisory**, classified by a
   minimum worthwhile-return threshold (`min_opportunity_roc`, default
   **10%/yr**, config).

### 6.3 Advisory content

The advisory must show enough for the human to judge, not just a strike:

| Metric | Why |
|---|---|
| Strike | Acquisition price |
| Premium (bid/mid where real) | Actual compensation |
| Net basis | Must satisfy MCB |
| Delta | How far outside the normal RL band |
| DTE | Holding period |
| Annualized ROC on escrow | Opportunity cost |
| OI / spread | Executability |
| MCB headroom | How comfortably below the ceiling |
| Assessment | **attractive → HUMAN REVIEW** / unattractive → WAIT |

### 6.4 Two states that were previously collapsed

The scan outcome refines what "unreachable" means in the brief:

* **MCB geometrically unreachable** — no strike/premium combination in the
  chain satisfies the ownership ceiling at all.
* **MCB reachable but economically unattractive** — a compliant strike
  exists, but premium/liquidity/ROC is too poor to justify tying up the
  cash ("MCB-compliant strikes exist, but are not economically tradeable").

Both remain WAIT; the reason string and JSON must distinguish them. In a
volatility event the same scan flips naturally: the compliant strike's
premium fattens, ROC clears the threshold, and the row escalates to
⚠ HUMAN REVIEW — which is exactly when reachability turns and deep-OTM
selling becomes genuinely interesting.

### 6.5 Promotion path (explicitly deferred)

The scan ships advisory-only. It may later become a human-approved
*executable* pathway **only if** a retrospective backtest on real chains
demonstrates that below-0.05-delta, MCB-compliant opportunities have
worthwhile ROC and outcomes (gate **G8**, to be specced alongside G7's
proxy-ceiling machinery). Until that verdict exists, the advisory never
feeds `decisions.jsonl` as a chosen action.

### 6.6 Acceptance (planned)

* AC-6.1 Harsh-ceiling fixture on a healthy chain → advisory row with all
  §6.3 metrics; thin premium → "economically unattractive — WAIT".
* AC-6.2 Elevated-IV fixture (compliant strike pays ≥ threshold) →
  "below-band opportunity — HUMAN REVIEW" and the brief renders it under
  the human-review section.
* AC-6.3 No compliant strike in the whole chain → "geometrically
  unreachable"; the two reason strings are distinct in JSON and brief.
* AC-6.4 The RL decision record is unchanged in all cases (WAIT, action 0);
  the advisory lives outside the trajectory action.

## 7. Out of scope

- MCB computation of any kind (`../mcb-wheel` owns it).
- Feeding MCB into training/backtests (no point-in-time MCB history exists).
- Call-side floors (cost-basis rule already covers the exit).
