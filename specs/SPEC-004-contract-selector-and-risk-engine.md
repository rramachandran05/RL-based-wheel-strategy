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

Validation runs **after** selection, **before** execution. Never learnable, never bypassable (the simulator and live path share one implementation; there is no code path from policy to execution that skips it).

### 2.1 Hard rules (v1; RISK-3/4/5 normalized 2026-08-31)

Position **counts** mislead when sizes vary 10x (one TQQQ put escrows ~$5K,
one META put ~$62K), so every capital rule reads **escrow / NAV**. The one
surviving count is distinct underlyings — an attention cap, which genuinely
doesn't scale with dollars. Book-level inputs come from `rlbot/risk/book.py`
(`BookState`: per-ticker escrow, per-expiry-week escrow, distinct tickers),
built from the synced positions each daily run (2026-08-30 review fix).

| ID | Rule | Default |
|---|---|---|
| RISK-1 | Cash-secured puts only: escrow strike×100×contracts available | — |
| RISK-2 | No naked calls: shares ≥ 100×contracts before SELL_CALL | — |
| RISK-3 | Max put escrow per underlying, **existing + new** (book-aware; covered calls exempt — the share exposure exists regardless) | 15% of NAV |
| RISK-4 | Max **distinct underlyings** (was: 9 positions) — a trade on a ticker already held adds no name | 12 |
| RISK-5 | Max put escrow expiring in one ISO-week, existing + new (was: 3 positions/week) — bounds what a single expiry Friday can force onto the book; put-side only, CC assignment is the intended exit | 15% of NAV |
| RISK-6 | Liquidity floor (real chains): min_oi, max_spread_pct as §1.1 | — |
| RISK-7 | Earnings blackout: no opening trade whose window contains the estimated next earnings date (AV `reportedDate` + ~91d, ±5d tolerance; live-era only) | on |
| RISK-8 | **Synchronized-assignment cap:** Σ over open short puts of (strike×100×contracts) ≤ `max_assignment_at_once` × NAV. Portfolio-level; per-ticker Q-tables cannot see this tail risk, so it lives here permanently | 40% of NAV |
| RISK-9 | Correlation-cluster cap: no new put on a ticker whose 120d return corr ≥ 0.80 with ≥ 2 existing short-put underlyings (vendored `portfolio_risk` machinery, structured wrapper) | on |

Live overrides: `--max-underlyings / --max-week-pct / --max-escrow-pct`
(daily assistant CLI); NAV comes from `--cash`. The `single_ticker()` preset
(simulation episodes) relaxes all book caps to the whole sleeve by design.

### 2.2 Disposition
On violation: step the action down one risk tier and re-select (once); if still violating → WAIT. Every rejection/downgrade is logged in `risk_checks.flags` with rule IDs. E5 epochs (SPEC-001 §5): if a *held* position violates a rule that applies to held positions (RISK-8 after NAV drop), a management epoch is forced.

## 3. Requirements

- **REQ-4.1** Selector is a pure function of (action, chain, spot, fv, config) — property-tested for determinism.
- **REQ-4.2** For every tier and 100 random synthetic chains, the selected contract's |delta| lies within the (possibly once-widened) band, or result is None.
- **REQ-4.3** Valuation monotonicity: same chain, ATTRACTIVE vs EXPENSIVE state ⇒ selected strike (put) under EXPENSIVE is ≤ (never nearer the money than) under ATTRACTIVE.
- **REQ-4.4** Each RISK rule has a dedicated violating fixture that is rejected, and a boundary fixture that passes.
- **REQ-4.5** Simulator and (future) live assistant call the identical `validate()`; enforced by module structure, verified by import-graph test.

## 4. Acceptance criteria

- **AC-1** Golden selection tests: fixed synthetic chain fixture → expected strike per each of the 5 put tiers and 4 call tiers.
- **AC-2** REQ-4.2 property test passes (1000 draws, seeded).
- **AC-3** REQ-4.3 monotonicity test passes.
- **AC-4** RISK-1..9 fixture pairs pass/reject as specified; downgrade-then-WAIT cascade verified for a double violation.
- **AC-5** A full baseline run's trajectories contain zero executed contracts with any failed risk check.
