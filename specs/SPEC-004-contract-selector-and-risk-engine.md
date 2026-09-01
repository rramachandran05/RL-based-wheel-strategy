# SPEC-004 — Deterministic Contract Selector and Hard Risk Engine

_Status: draft v1 — 2026-08-21_

## 1. Contract selector

Input: an opening action (risk tier → delta band + DTE window per SPEC-001 §2), a chain snapshot (real or synthetic), spot, FV levels (nullable), config. Output: one contract or `None` (→ WAIT), plus the scored candidate list (logged in trajectory `candidates_considered` and assistant payload).

### 1.1 Filter
- Right type (put/call); |delta| within the tier band; DTE within [25, 45].
- Real chains only: `volume ≥ min_volume (10)`, `oi ≥ min_oi (100)`, `spread_pct = (ask−bid)/mid ≤ max_spread_pct (0.10)`. Synthetic chains skip liquidity filters (no meaning) — the live assistant on real quotes must not.
- Covered calls: strike ≥ cost basis (never write below basis — adopted from sibling).

### 1.2 Score

```
Score = w1·PremiumYield + w2·VolPremium − w3·SpreadCost − w4·DownsideRisk − w5·AssignmentPenalty
```

| Term | Definition (v1) | Default w |
|---|---|---|
| PremiumYield | premium / strike, annualized ×365/DTE | 1.0 |
| VolPremium | (iv − realized_vol_30) · vega_proxy, 0 on synthetic track (identically zero — no info) | 0.5 |
| SpreadCost | spread_pct | 0.5 |
| DownsideRisk | `expected_put_payout`/strike (vendored fn), annualized; calls: assignment_prob · max(0, spot−strike)/spot | 1.0 |
| AssignmentPenalty | assignment_prob · valuation_multiplier | 1.0 |

`valuation_multiplier`: ATTRACTIVE → 0.25, FAIR/unknown → 1.0, EXPENSIVE → 2.0 (config). This is where valuation shapes behavior without being a hard constraint — an attractive stock makes assignment cheap; an expensive one makes it costly.

Ties: highest premium wins; then nearest DTE to 30. Weights are config, logged into every run's manifest; they are engineering parameters, not learned.

### 1.3 Degradation
If the filter leaves zero candidates in-band: widen delta band by ±0.02 once; if still empty, return `None` (the tier is unimplementable today → WAIT, logged with reason).

## 2. Risk engine — RL proposes, risk engine disposes

_(v2 — 2026-08-31: two-tier disposition, potential-exposure RISK-3,
assignment-stress RISK-8, human-review RISK-7/9. Supersedes the v1 table.)_

Validation runs **after** trade selection and **before** execution.

The risk engine is never learnable and cannot be bypassed. The simulator and live execution path share the same implementation; there is no direct path from the RL policy to execution that skips risk validation.

Risk controls are divided into:

* **Hard blocks** — the trade cannot proceed.
* **Warnings requiring human review** — the risk is surfaced with supporting information, and a human explicitly approves or rejects the trade.

Capital-based limits use **exposure / NAV**, not position counts. Position counts are misleading when contract sizes differ materially. The only count-based limit retained is the number of distinct underlyings, which is an operational attention constraint.

Book-level inputs come from `rlbot/risk/book.py` (`BookState`) and are reconstructed from synchronized positions on each daily run.

### 2.1 Risk rules

| ID | Rule | Default | Action |
|---|---|---:|---|
| **RISK-1** | **Cash-secured puts only.** Before `SELL_PUT`, sufficient cash must be available to secure `strike × 100 × contracts`. | — | **Hard block** |
| **RISK-2** | **No naked calls.** Before `SELL_CALL`, owned shares must be at least `100 × contracts`. | — | **Hard block** |
| **RISK-3** | **Maximum potential exposure per underlying, existing + new.** For a proposed put, calculate potential underlying exposure as **current share market value + assignment value of all existing short puts + assignment value of the proposed put**. The total may not exceed the configured percentage of NAV. Covered calls do not add new underlying exposure and therefore do not increase this calculation. | **15% of NAV** | **Hard block** |
| **RISK-4** | **Maximum distinct active underlyings.** Limit the number of different ticker names being actively managed across the Wheel book. A new trade on a ticker already represented in the portfolio does not add another name. This is primarily an operational/attention cap rather than a capital-risk measure. | **12** | **Hard block** |
| **RISK-5** | **Maximum put exposure expiring in one ISO week, existing + new.** Total assignment value of short puts expiring in any single ISO week may not exceed the configured percentage of NAV. This limits how much capital can convert into stock during one scheduled expiration event. Covered calls are excluded because call assignment is the intended stock-exit mechanism. | **15% of NAV** | **Hard block** |
| **RISK-6** | **Option liquidity / execution-quality floor.** Proposed contracts must satisfy the liquidity requirements defined in §1.1, including minimum open interest and maximum allowable bid/ask spread. | Per §1.1 | **Hard block** |
| **RISK-7** | **Earnings-risk warning.** If the expected life of a proposed opening trade overlaps a known or estimated earnings-risk window, flag the trade for human review. Show the earnings date, whether it is **confirmed or estimated**, the option expiration date, and the overlap. Estimated dates must be clearly identified as estimates rather than treated as authoritative dates. | **On** | **Warning + human decision** |
| **RISK-8** | **Assignment-stress liquidity reserve.** The portfolio must retain sufficient liquidity to manage clustered assignments. Stress the open short-put book assuming **100% assignment of the nearest expiry week + 50% assignment of the following expiry week + 100% assignment of later puts already sufficiently ITM**. After the stressed assignments, **unencumbered cash must remain at least 15% of NAV**. | **15% of NAV reserve** | **Hard block** |
| **RISK-9** | **Correlation / concentration warning.** If a proposed new put is highly correlated with existing Wheel exposures, flag it for human review. The warning must identify the correlated underlyings, trailing correlations, their current/potential NAV exposures, the proposed position's exposure, and the resulting combined related exposure. Correlation should inform the decision rather than automatically veto the trade. | Corr ≥ **0.80** over trailing **120d** | **Warning + human decision** |

### 2.2 RISK-3 — potential underlying exposure

RISK-3 protects against accumulating excessive exposure to the same company across both assigned shares and open puts.

For ticker `T`:

`potential_exposure(T) = current_share_market_value(T) + existing_short_put_assignment_value(T) + proposed_short_put_assignment_value(T)`

Require:

`potential_exposure(T) / NAV <= 15%`

Example with $500,000 NAV:

* Existing META shares: $45,000
* Existing META short-put assignment value: $10,000
* Proposed META put assignment value: $25,000
* Potential META exposure: $80,000
* Exposure / NAV: 16%

**Result: reject.**

This prevents the system from treating already-assigned stock and new puts as unrelated exposures.

Covered calls do not add additional stock exposure and therefore are not added to the RISK-3 exposure calculation.

_Implementation note: the positions feed tracks CSPs and CCs; share holdings are inferred from covered calls (100 × contracts) and marked at the latest close. Uncovered long stock is invisible to the book until it appears on the monitor sheet._

### 2.3 RISK-4 — active-underlying attention cap

The maximum-underlyings rule remains count-based because its purpose is different from the capital rules.

It answers:

> How many separate companies can reasonably be monitored and managed at the same time?

The active-underlying count should include tickers represented by:

* assigned shares,
* open short puts,
* open covered calls.

Multiple positions in the same ticker count as one active underlying.

Default:

`max_underlyings = 12`

### 2.4 RISK-5 — expiration concentration

RISK-5 limits scheduled assignment concentration.

For every ISO expiry week:

`put_assignment_value_for_week / NAV <= 15%`

Example with $500,000 NAV:

* Week 1 put exposure: $70,000 → 14% ✓
* Week 2 put exposure: $65,000 → 13% ✓
* Proposed Week 2 put: $20,000
* New Week 2 exposure: $85,000 → 17%

**Result: reject.**

This rule allows substantial overall capital deployment while preventing too much of the book from becoming stock during one normal expiration cycle.

### 2.5 RISK-7 — earnings warning

Earnings exposure is not automatically prohibited.

Instead, if a proposed position may remain open through an earnings event, generate a warning such as:

> **EARNINGS RISK:** META earnings are estimated for October 28. Proposed put expires November 6. The position may remain open through earnings. Earnings date source: estimated. Human approval required.

The reviewer chooses:

* **APPROVE** — permit execution.
* **REJECT** — do not execute.

Where possible, the warning should distinguish:

* **Confirmed earnings date**
* **Estimated earnings date**

A fallback estimate such as Alpha Vantage `reportedDate + ~91 days` must be labeled **estimated** and retain an uncertainty window.

The purpose is to prevent the system from unknowingly carrying event risk while still allowing the human to intentionally accept that risk when the premium or setup justifies it.

### 2.6 RISK-8 — assignment-stress liquidity reserve

RISK-8 does **not** impose a blanket limit such as:

`total short-put escrow <= 40% NAV`

That would unnecessarily constrain capital utilization in a cash-secured Wheel.

Instead, it asks:

> If assignments cluster more heavily than the normal expiry schedule suggests, will enough liquidity remain to manage the rest of the portfolio?

The default stress scenario is:

`stressed_assignment =`

* `100% × nearest-expiry-week put exposure`
* `+ 50% × following-expiry-week put exposure`
* `+ 100% × later puts classified as sufficiently ITM`

After applying this stress:

`remaining_unencumbered_cash / NAV >= 15%`

Example with $500,000 NAV:

* Nearest week: $70,000
* Following week: $70,000
* 50% of following week: $35,000
* Later deeply ITM put: $20,000

Stressed assignment:

`$70,000 + $35,000 + $20,000 = $125,000`

The system then checks whether at least:

`15% × $500,000 = $75,000`

of unencumbered liquidity remains after the modeled assignments.

If not, the proposed trade is rejected.

This permits the Wheel to deploy a high proportion of available capital while retaining a consistent 15% management reserve.

_Implementation notes: "nearest / following expiry week" = the first and second distinct ISO weeks holding open put expiries (the proposed put joins its own week's bucket); "sufficiently ITM" = strike ≥ latest close (a later put with no available close is not counted); "unencumbered cash" = NAV-proxy cash minus the stressed assignment value._

### 2.7 RISK-9 — correlation / concentration warning

Correlation is treated as decision support, not an automatic veto.

When a proposed put has trailing 120-day return correlation of at least `0.80` with relevant existing Wheel holdings, generate a human-review warning.

The warning should include:

* proposed ticker;
* correlated existing tickers;
* pairwise correlations;
* existing potential exposure for each ticker as % NAV;
* proposed trade exposure as % NAV;
* combined exposure represented by the correlated group.

Example:

> **CORRELATION WARNING:** Proposed MSFT put has 120-day correlation of 0.84 with META and 0.82 with GOOGL.
>
> META potential exposure: 11% NAV
> GOOGL potential exposure: 9% NAV
> Proposed MSFT exposure: 8% NAV
> Combined related exposure after trade: approximately 28% NAV.
>
> Human approval required.

The reviewer chooses:

* **APPROVE** — correlated concentration is acceptable.
* **REJECT** — do not add the exposure.

This avoids two problems with a simple correlation-count rule:

1. rejecting small positions that pose little portfolio risk;
2. accepting large correlated positions simply because the correlation falls slightly below an arbitrary cutoff.

### 2.8 Risk-engine summary

The resulting hierarchy is:

**Structural and capital safeguards — hard blocks**

* **RISK-1:** puts must be cash secured.
* **RISK-2:** calls must be covered.
* **RISK-3:** ≤15% potential exposure per underlying.
* **RISK-4:** ≤12 active underlyings.
* **RISK-5:** ≤15% put assignment exposure per expiry week.
* **RISK-6:** minimum liquidity/execution quality.
* **RISK-8:** ≥15% NAV liquidity remaining under assignment stress.

**Context-dependent risks — human review**

* **RISK-7:** earnings-event exposure.
* **RISK-9:** correlated/concentrated exposure.

The design principle is:

> **Hard blocks protect against structural or portfolio-capacity failures. Human-review warnings surface risks whose acceptability depends on market context, valuation, premium, and investor judgment.**

In the recommendations-only daily assistant, "human approval" is operational
reality: a hard block renders as WAIT; a warning-carrying recommendation
renders with a **⚠ REVIEW** marker plus the full warning text, and the human
executes (or not) at the broker.

### 2.9 Live overrides

Live operation exposes configurable overrides:

* `--max-underlyings`
* `--max-week-pct`
* `--max-exposure-pct`
* `--min-stress-reserve-pct`

NAV should come from the live account/book state rather than assuming that cash alone represents NAV (current approximation: `--cash <NAV>`, floored at book escrow with a loud warning).

The `single_ticker()` simulation preset may relax portfolio-wide book constraints to the simulation sleeve where required, but it must preserve the intended semantics of the individual risk rule being tested.

## 3. Requirements

- **REQ-4.1** Selector is a pure function of (action, chain, spot, fv, config) — property-tested for determinism.
- **REQ-4.2** For every tier and 100 random synthetic chains, the selected contract's |delta| lies within the (possibly once-widened) band, or result is None.
- **REQ-4.3** Valuation monotonicity: same chain, ATTRACTIVE vs EXPENSIVE state ⇒ selected strike (put) under EXPENSIVE is ≤ (never nearer the money than) under ATTRACTIVE.
- **REQ-4.4** Each hard-block rule has a dedicated violating fixture that is rejected and a boundary fixture that passes; each human-review rule has a fixture that emits its warning (with the required supporting detail) while the decision still passes.
- **REQ-4.5** Simulator and (future) live assistant call the identical `validate()`; enforced by module structure, verified by import-graph test.

## 4. Acceptance criteria

- **AC-1** Golden selection tests: fixed synthetic chain fixture → expected strike per each of the 5 put tiers and 4 call tiers.
- **AC-2** REQ-4.2 property test passes (1000 draws, seeded).
- **AC-3** REQ-4.3 monotonicity test passes.
- **AC-4** RISK-1..9 fixtures behave per the §2.1 Action column: hard blocks reject, RISK-7/9 warn without rejecting.
- **AC-5** A full baseline run's trajectories contain zero executed contracts with any failed risk check.
