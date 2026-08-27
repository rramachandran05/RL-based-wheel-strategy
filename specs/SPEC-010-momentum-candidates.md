# SPEC-010 — Momentum Candidates Feed + Pipeline Integration

_Status: v1 — 2026-08-27. Completes the three-codebase integration:
`momentum-monitor` (scout) → `fair-value-discount` (appraiser, integrated as
the **SPEC-009 valuation gates**) → `wheel-strategy-rlbot` (underwriter +
monitor). This spec covers the candidate feed (Feed B), the ETF anchor
policy, and the outstanding experiments. Feed A (ensemble valuation) is
specified and shipped in `SPEC-009-valuation-gates.md`._

## 1. Architecture (as built)

```
momentum-monitor (Sat 07:00)            fair-value-discount (run_fv.py)
  monitor_rankings @ vmi.sqlite           fair_value_ensemble_<date>.csv
        │ top-decile candidates                  │ wheel_fv, MOS, reliability
        ▼                                        ▼
        rlbot daily 06:00 ── SPEC-009 valuation gates (risk layer)
   Core recs · CANDIDATE recs · position monitoring — one brief
```

Design principle preserved: momentum selects **names**, never actions
(the trend-axis ablation failed); valuation acts as a **constraint layer**,
never a return predictor (SPEC-009; VMI post-mortem).

## 2. Feed B — candidates (`rlbot/data/candidates.py`)

- **Contract:** read-only SQLite (`monitor_rankings` in
  `../vmi-stock-search/data/vmi.sqlite`, override `MONITOR_DB_PATH`):
  latest `run_date`, `in_top_decile = 1`, ordered by `percentile`; top
  `CANDIDATE_TOP_N` (10) minus tickers already in the assistant universe;
  4-week percentile change computed from the monitor's own history.
- **Treatment (reasoned defaults, not gate-validated):** capped at
  PUT_CONSERVATIVE (`cap_candidate_action`); max 2 open candidate positions;
  SPEC-009 valuation gates apply on top (a VERY_EXPENSIVE candidate is
  masked to WAIT like any core name); leveraged/inverse products are not
  expected from the monitor's universe filters.
- **Onboarding** (inside `--download`): bars fetched on first sight; the
  ensemble run (`run_fv.py --tickers`) includes candidates so they arrive
  with real Wheel FV; chains accrue daily thereafter (synthetic quotes
  until the first chain snapshot exists).
- **Brief:** separate *Candidates* section (momentum percentile, 4-week
  change, full state/valuation columns) with a standing not-validated note.
- **Promotion** to Core is manual: Rahul edits the config universe
  (optionally adds a sheet FV row). Candidates never auto-promote.
- **Degradation (REQ):** missing/unreadable monitor store, or per-candidate
  onboarding failure → warning line, never a broken brief.

## 3. ETF anchor policy (per 2026-08-27 discussion)

Ensemble FV cannot exist for ETFs (no fundamentals → `status=failed`,
constraint absent — never zero). Defaults:
- Leveraged (TQQQ, SPXL): **regime-asymmetry only** (tier caps +
  WAIT-in-stress); no manufactured value anchor — contested, hence G6.
- Unleveraged (CHPS, SPYI, MCHI): volume-profile support/resistance may act
  as a *soft, labeled* technical anchor — only if G6 supports it.

## 4. Outstanding experiments

| ID | Question | Method | Effect on defaults |
|---|---|---|---|
| **G6** | Do technical anchors help leveraged-ETF puts? | 4-arm paired A/B on real TQQQ chains 2012→present: (A) caps only, (B) VP-support soft nudge, (C) VP-support hard ceiling, (D) 200-SMA hard ceiling; judged on diff, dd-ratio, and the 2020/2022 crash windows | Winner ships; A is the incumbent |
| **G7** | Do the (already-live) SPEC-009 hard gates help historically? | Retrospective paired A/B on real chains 2013→present with proxy ceilings (EPS-percentile pseudo-FV — point-in-time ensemble FV doesn't exist): B3+gates vs B3 | Fail ⇒ demote net-basis pre-filter to advisory; masks re-reviewed |
| — | Candidate caps (CONSERVATIVE, max 2) | Not testable yet (no candidate history); revisit after ~6 months of live candidate logs | Reasoned defaults stand |

## 5. Acceptance

- **AC-1** Fixture rankings DB → correct top-N, exclusion, 4-week change,
  graceful missing-store handling. ✔ (tests/test_candidates.py)
- **AC-2** Live end-to-end daily run renders the Candidates section with
  valuation-gate columns; a run with no monitor store still produces the
  full Core brief.
- **AC-3** G6/G7 verdict JSONs committed with reports, whatever they say.
