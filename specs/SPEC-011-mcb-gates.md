# SPEC-011 — MCB Gates: Maximum-Comfortable-Basis Feed Replaces Wheel-FV

_Status: v1 implemented 2026-08-31; §6 opportunity scan implemented
2026-09-01; **v2 aggressiveness-pivot gate (this revision, 2026-09-09) —
specced here, code follow-up tracked in §8.2, not yet shipped.** §§1–2
below describe the v2 contract as the target; until §8.2 lands, the code
still enforces the v1 universal-hard-ceiling behavior, which is strictly
more conservative (safe, but wrongly WAITs on income-mode names)._

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
| `min_eligible_tier` | **v2: data-quality only** (low confidence, leveraged-ETF) — no longer guardrail-resolved |
| `shadow_min_tier` | v2: the old guardrail-resolved tier, recorded for the producer's own calibration backtest — **not enforced by this repo** |
| `wheel_entry` | **v2, the intended ceiling** `= MCB(min_eligible_tier)`; replaces the old deeper-of(min_eligible_tier, regime posture) derivation |
| `delta_posture` | **v2, the aggressiveness pivot** — CONSERVATIVE / MODERATE / HIGHER; computed by the producer from valuation + drawdown distance (§2 rule 1) |
| `dd_now` | v2: current drawdown from trailing high (not just the historical dd50/75/90 percentiles) |
| `drop_needed` | v2: `(spot − wheel_entry) / spot` — how far a correction would need to go to reach entry |
| `action` | v2 advisory label: Check CSPs now / Add to core list / Watch only / Too far away |
| `guardrail_status` | NORMAL / CAUTION / SEVERE (NaN for ETFs) |
| `layer_a` | OWN / MONITOR_ONLY / HALT — ownership eligibility |
| `reachability` | NORMAL / PATIENCE / UNREACHABLE (advisory, distinct from and unchanged by `delta_posture`) |
| `conf` | Producer confidence; all-NaN zones ⇒ constraint absent |

Producer cadence is now **daily** (was weekly-ish); the 5-trading-session
staleness rule (§2 rule 5) is unchanged. The producer's universe is driven
solely by the FV sheet (the momentum-monitor relationship was removed on
their side) — SPEC-010's Feed B remains a name-source for this repo's
brief only and never reaches the MCB constraint (rule 4, unchanged).

ETFs get drawdown-derived zones (`etf_subtype`, dd50/75/90), so — unlike the
Wheel-FV era — **ETFs are gated too**, softening SPEC-010 §3's "constraint
absent" default (G6 remains open for technical anchors on top).

## 2. Consumer contract (as enforced here)

1. **Net-basis rule, bindingness scaled to assignment intent (v2, revises
   the v1 universal-hard rule)** — `strike − premium ≤ wheel_entry` is
   **not** applied uniformly. A trending, compounding stock can be safely
   sold at conservative delta far above its comfortable acquisition price:
   assignment there is a low-probability tail event, not the trade's
   purpose. The rule now keys off the producer's `delta_posture`:

   | `delta_posture` | Ceiling treatment |
   |---|---|
   | CONSERVATIVE | **Advisory only — never blocks.** Conservative-delta puts with net basis above `wheel_entry` are legitimate: assignment is neither sought nor likely. |
   | MODERATE | **Hard for our BALANCED-and-up tiers** (assignment-seeking); **advisory for WAIT/DEFENSIVE/CONSERVATIVE tiers** (income-only) — the one case the producer's contract leaves to us; resolved here by our own delta-tier split (2026-09-09), reusing the 0.10–0.18Δ boundary already frozen by SPEC-001. |
   | HIGHER | **Hard, always** — assignment is welcome and actively sought; this is where the ceiling's protection is the entire point. |

   **Trend override — ours, not the producer's:** the producer computes no
   trend signal. A confirmed downtrend (`classify_structure` ∈
   {Pullback in Uptrend, Breakdown}, §5 of APPROACH.md — already computed
   per-day, previously logged-only) forces conservative-or-wait regardless
   of `delta_posture` — "don't catch a falling knife." This applies even
   when `delta_posture = HIGHER`, since the producer's posture reflects
   valuation/drawdown distance, not price trend, and a name can be both
   "at a comfortable basis" and "actively falling further."

   `wheel_entry` missing (row absent or `conf` too low) → constraint
   absent, never zero, same as v1.
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
| Loader | `rlbot/data/mcb_feed.py` | PIT-safe (files ≤ as_of only); NaN rows → absent + warning; `DataConfig.mcb_dir`; **v2**: parses `wheel_entry`, `delta_posture`, `dd_now`, `drop_needed`, `action`, `shadow_min_tier` |
| Posture/ceiling/masks | `rlbot/risk/mcb_gates.py` | **v2**: `mcb_binding(row, action, trend_structure)` → (`ceiling`, `hard: bool`) replaces the old unconditional `required_tier`/`mcb_ceiling`; `tradeable`, `reachability_advice`, `net_basis_flag` (MCB-1), `premium_required` unchanged |
| Trend overlay | `rlbot/assistant/daily.py` (reads `underlying.structure`, already computed) | Pullback-in-Uptrend / Breakdown forces conservative-or-wait regardless of `delta_posture` |
| Selector pre-filter | `rlbot/options/selector.py` | `net_basis_ceiling=` kwarg now optional per-tier (`None` when the gate is advisory for the chosen tier); puts only; empty list → None → WAIT |
| Daily surface | `rlbot/assistant/daily.py` | Openings/candidates columns gain `Posture` alongside `MCB tier / Ceiling / Prem req / MCB flags`; open-CSP flag when filled net basis > ceiling *and* the position's posture was/is non-CONSERVATIVE; `--download` runs `run_mcb.py` (sheet-driven) instead of `run_fv.py` |
| WAIT attribution | `rlbot/assistant/daily.py` | An empty scan blames the ceiling **only if** it was actually binding (hard for this tier+posture) *and* rescanning without it finds a contract; else "tier unimplementable" — chain/liquidity |

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
        every compliant candidate surfaced with its economics —
        the USER judges opportunity cost; ROC < 7%/yr carries a
        LOW YIELD flag (decision support, never a verdict or blocker)
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
5. Surface the best candidate **always, as an advisory** — the system
   renders the economics and makes no accept/reject judgment (user decision
   2026-09-01: the threshold is not a blocker; opportunity cost depends on
   the situation and belongs to the human). Candidates with ROC below
   `low_yield_roc` (default **7%/yr**, config / `--low-yield-roc`) carry a
   **LOW YIELD** flag as decision support.

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
| LOW YIELD flag (ROC < 7%/yr) | Decision support — never a verdict; the user weighs opportunity cost |

### 6.4 Two states that were previously collapsed

The scan outcome refines what "unreachable" means in the brief:

* **MCB geometrically unreachable** — no strike/premium combination in the
  chain satisfies the ownership ceiling at all.
* **MCB reachable, candidate surfaced** — a compliant strike exists; its
  premium, ROC, and liquidity render in the advisory for the user to judge,
  with a LOW YIELD flag when ROC < 7%/yr.

Both remain WAIT on the executable side; the reason string and JSON must
distinguish them. In a volatility event the same scan responds naturally:
the compliant strike's premium fattens, the LOW YIELD flag drops away, and
the advisory reads as genuinely interesting — exactly when reachability
turns and deep-OTM selling starts paying.

### 6.5 Promotion path (explicitly deferred)

The scan ships advisory-only. It may later become a human-approved
*executable* pathway **only if** a retrospective backtest on real chains
demonstrates that below-0.05-delta, MCB-compliant opportunities have
worthwhile ROC and outcomes (gate **G8**, to be specced alongside G7's
proxy-ceiling machinery). Until that verdict exists, the advisory never
feeds `decisions.jsonl` as a chosen action.

### 6.6 Acceptance (shipped, tests/test_mcb_gates.py AC-6.1..6.4)

* AC-6.1 Harsh-ceiling fixture on a healthy chain → advisory row with all
  §6.3 metrics; thin premium → surfaced with the LOW YIELD flag, no verdict.
* AC-6.2 Elevated-IV fixture (compliant strike pays ≥ 7%/yr) → surfaced
  without the LOW YIELD flag; the brief renders it in the opportunity-scan
  section marked as worth a look.
* AC-6.3 No compliant strike in the whole chain → "geometrically
  unreachable"; the two reason strings are distinct in JSON and brief.
* AC-6.4 The RL decision record is unchanged in all cases (WAIT, action 0);
  the advisory lives outside the trajectory action.

## 7. Out of scope

- MCB computation of any kind (`../mcb-wheel` owns it).
- Feeding MCB into training/backtests (no point-in-time MCB history exists).
- Call-side floors (cost-basis rule already covers the exit).

## 8. Producer contract v2 — changelog (2026-09-08/09)

_§§1–2 above now state the v2 contract directly; this section is the
record of what changed and why, kept for provenance._

The producer revised its contract (mcb-wheel SPEC-000 v2 + logic spec
§11a), confirmed live in `mcb_2026-09-09.csv` and later. Columns are
backward-compatible (v1 fields unchanged), so nothing broke; three
semantics changed:

1. **Guardrail became advisory**, not gate-determining — `min_eligible_tier`
   is data-quality-only now; the old guardrail mapping moved to
   `shadow_min_tier` (producer calibration only, never enforced here).
2. **MCB became an aggressiveness pivot** via `delta_posture` — §2 rule 1.
3. **New advisory columns** (`drop_needed`, `action`, `dd_now`, wider
   `a4_low/a4_high`) and daily cadence (was weekly-ish).

The trend override in §2 rule 1 is this repo's own addition — the producer
computes no trend signal, so a downtrend override was never part of their
contract and had to be built here (closes the "falling knife" gap
identified when this rule was first designed, 2026-09-08).

**Code status:** specced in §§1–2/§3; implementation tracked and, once
shipped, this line updates to name the commit. Until it lands, the code
enforces v1's universal-hard ceiling — strictly more conservative than v2
(safe, but wrongly WAITs on legitimate income-mode trades); the §6
opportunity scan partially compensates for that gap in the meantime.
