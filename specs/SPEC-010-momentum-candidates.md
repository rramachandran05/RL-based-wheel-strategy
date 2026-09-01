# SPEC-010 — Momentum Candidates Feed + Pipeline Integration

_Status: v1 — 2026-08-27; **appraiser swapped 2026-08-31**: the valuation
leg is now `../mcb-wheel` (SPEC-011 MCB gates), not `fair-value-discount`.
Completes the three-codebase integration: `momentum-monitor` (scout) →
`mcb-wheel` (appraiser) → `wheel-strategy-rlbot` (underwriter + monitor).
This spec covers the candidate feed (Feed B), the ETF anchor policy, and the
outstanding experiments. Feed A (valuation) lives in
`SPEC-011-mcb-gates.md` (formerly SPEC-009)._

## 1. Architecture (as built, 2026-08-31)

```
momentum-monitor (Sat 07:00)            mcb-wheel (run_mcb.py, sheet-driven)
  monitor_rankings @ vmi.sqlite           outputs/mcb_<date>.csv
        │ top-decile candidates                  │ MCB zones, guardrail, layer A
        ▼                                        ▼
        rlbot daily 06:00 ── SPEC-011 MCB gates (risk layer)
   Core recs · CANDIDATE recs · position monitoring — one brief
```

Design principle preserved — and now a producer contract term: momentum
selects **names**, never actions or timing (the trend-axis ablation failed;
mcb-wheel's contract rule 3 keeps momentum out of the MCB constraint).
Valuation acts as a **constraint layer**, never a return predictor
(SPEC-011; VMI post-mortem). The two feeds are fully decoupled: they meet
only in the daily brief, where MCB covers a candidate **only if the FV sheet
lists it** — an unlisted candidate trades constraint-absent under the
standing not-validated note.

## 2. Feed B — candidates (`rlbot/data/candidates.py`)

- **Contract:** read-only SQLite (`monitor_rankings` in
  `../vmi-stock-search/data/vmi.sqlite`, override `MONITOR_DB_PATH`):
  latest `run_date`, `in_top_decile = 1`, ordered by `percentile`; top
  `CANDIDATE_TOP_N` (10) minus tickers already in the assistant universe;
  4-week percentile change computed from the monitor's own history.
- **Treatment (reasoned defaults, not gate-validated):** capped at
  PUT_CONSERVATIVE (`cap_candidate_action`); max 2 open candidate positions;
  SPEC-011 MCB gates apply **only where the FV sheet covers the name**
  (2026-08-31 — mcb-wheel's universe is the sheet, so an unlisted candidate
  has no ceiling; add it to the sheet's "Watching for Pullback" section to
  gate it); leveraged/inverse products are not expected from the monitor's
  universe filters.
- **Onboarding** (inside `--download`): bars fetched on first sight; the
  MCB refresh (`run_mcb.py`, sheet-driven) does **not** auto-include
  candidates — the deliberate cost of the momentum/valuation decoupling;
  chains accrue daily thereafter (synthetic quotes until the first chain
  snapshot exists).
- **Brief:** separate *Candidates* section (momentum percentile, 4-week
  change, full state/valuation columns) with a standing not-validated note.
- **Promotion** to Core is manual: Rahul edits the config universe
  (optionally adds a sheet FV row). Candidates never auto-promote.
- **Degradation (REQ):** missing/unreadable monitor store, or per-candidate
  onboarding failure → warning line, never a broken brief.

## 3. ETF anchor policy (2026-08-27; revised 2026-08-31)

The Wheel-FV era left ETFs constraint-absent (no fundamentals). **MCB closes
that gap**: mcb-wheel derives drawdown-based zones for ETFs (`etf_subtype`,
dd50/75/90), so TQQQ/SPXL/CHPS/SPYI/MCHI now carry real net-basis ceilings
like any stock. Standing layers on top:
- Leveraged (TQQQ, SPXL): capped rule table (max BALANCED, WAIT-in-stress)
  unchanged; MCB ceiling applies in addition.
- G6 (technical anchors) remains open but is now about whether volume-profile
  or 200-SMA adds anything **beyond** the MCB drawdown zones, not about
  filling an empty slot.

## 4. Outstanding experiments

| ID | Question | Method | Effect on defaults |
|---|---|---|---|
| **G6** | Do technical anchors help leveraged-ETF puts? | 4-arm paired A/B on real TQQQ chains 2012→present: (A) caps only, (B) VP-support soft nudge, (C) VP-support hard ceiling, (D) 200-SMA hard ceiling; judged on diff, dd-ratio, and the 2020/2022 crash windows | Winner ships; A is the incumbent |
| **G7** | Do hard net-basis gates help historically? (specced against SPEC-009; verdict now informs the SPEC-011 MCB ceiling, same mechanism) | Retrospective paired A/B on real chains 2013→present with proxy ceilings (EPS-percentile pseudo-FV — no point-in-time FV or MCB history exists): B3+gates vs B3 | Fail ⇒ demote net-basis pre-filter to advisory |
| — | Candidate caps (CONSERVATIVE, max 2) | Not testable yet (no candidate history); revisit after ~6 months of live candidate logs | Reasoned defaults stand |

## 5. Acceptance

- **AC-1** Fixture rankings DB → correct top-N, exclusion, 4-week change,
  graceful missing-store handling. ✔ (tests/test_candidates.py)
- **AC-2** Live end-to-end daily run renders the Candidates section with
  MCB columns (tier/ceiling/flags, "—" when the sheet doesn't cover the
  name); a run with no monitor store still produces the full Core brief.
- **AC-3** G6/G7 verdict JSONs committed with reports, whatever they say.
