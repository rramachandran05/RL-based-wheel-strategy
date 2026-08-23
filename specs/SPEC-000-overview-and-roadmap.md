# SPEC-000 — Overview, Roadmap, and Spec Index

_Status: draft v1 — 2026-08-21_
_Parent doc: [APPROACH.md](../APPROACH.md)_

## 1. What this project is

An event-driven, hierarchical RL system for the options wheel strategy. An RL policy chooses a **risk posture** (assignment-risk / call-away-risk tier); a deterministic contract selector converts that posture into an actual option; a hard risk engine validates it. Learning is offline-first via **counterfactual sweeps** over historical paths, with a **differential reward** against reference policies.

**Falsifiable claim:** the learned policy must beat a simple rule-based adaptive wheel (Baseline 3) out-of-sample, or the learning layer is not justified. Steps before that gate produce a useful rule-driven assistant regardless.

## 2. Relationship to the existing `wheel-strategy` project

The sibling repo `../wheel-strategy` (Python 3.8, Tiingo OHLCV, synthetic Black-Scholes premiums, rule-based daily brief) is the seed. SPEC-002 §4 has the full reuse map. Headlines:

- **Reuse as-is (vendored):** `technicals.py` (indicators, structure/momentum classifiers), `options_engine.py` (BS pricing, assignment probability, EV — becomes the synthetic `PremiumSource`), `data_access.py` (Tiingo + CSV cache), `portfolio_risk.py` (concentration/correlation/clustering checks).
- **Reuse as pattern/seed:** `wheel_backtest.simulate_wheel` (seed of the simulator step loop), `optimize_wheel_mult` (an extra baseline), `daily_brief.analyze_ticker` (observation-vector assembly), `fv_levels.py` anchor math (valuation feature), `config.py` dataclass pattern.
- **Known gaps the old repo cannot fill:** no historical option chains/IV, no historical Fear & Greed, no historical earnings dates, no historical valuation series, ~5y daily bars × 21 tickers, latest-snapshot-only feature computation. Resolutions in SPEC-002 §5.

This project uses its **own venv (Python ≥ 3.11)**. Vendored 3.8-era modules are forward-compatible (pure numpy/pandas/scipy).

## 3. Spec index

| Spec | Title | Layer | Frozen? |
|---|---|---|---|
| SPEC-000 | This document — overview, roadmap, gates | — | no |
| SPEC-001 | MDP contract: states, actions, reward, epochs, trajectory schema | interface | **yes — frozen after v1 sign-off** |
| SPEC-002 | Data layer: adapters, canonical tables, reuse map, gap resolutions | data | no |
| SPEC-003 | Simulator: environment, execution, assignment, MTM, calibration gate | simulator | no |
| SPEC-004 | Contract selector and risk engine | options/risk | no |
| SPEC-005 | Learning: counterfactual sweep, estimators, pessimism, promotion | learning | no |
| SPEC-006 | Baselines, walk-forward evaluation, metrics, gates | evaluation | no |
| SPEC-001A | Management-state addendum (SPEC-001 v2: MgmtStateV1, roll mechanics, trajectory_v2) | interface | **yes — extends frozen SPEC-001 by version bump** |
| SPEC-007 | MVP-3 rescope: management-policy learning (G3) + AV valuation proxy (G2-rerun) | learning | no |

SPEC-001 is the immutable interface (frozen-manifest pattern): simulator, policy, and assistant layers all program against it, so the learning method can be swapped (Q-table → bandit → ICRL) without touching the simulator, and every trajectory logged from day one stays usable.

## 4. Roadmap and gates

> **Status 2026-08-21 — MVP 1 and MVP 2 executed.**
> **G1 PASSED**: PUT-index replication, fitted `iv_uplift=0.10`; validation-half monthly-return corr 0.959, ann-return diff −1.6 pt, vol ratio 1.15.
> **G2 FAILED (legitimate outcome per SPEC-006 §5)**: pooled test differential vs Baseline 3 = −0.21%/yr (F1 −0.54%, F2 +0.12%); drawdown, regime-segment, and A/B criteria all passed — the learned policy is indistinguishable from the rules, not riskier. Root cause: with valuation inert historically (DATA-GAP-3) only 10 of 36 states are populated, and regime+vol-comp alone don't separate the tiers. **Deliverable stands as the rule-driven assistant (B3 + selector + risk engine).** MVP 3 must be re-scoped before further learning: leading candidates are closing DATA-GAP-3 (historical valuation), richer vol-comp once per-ticker IV exists, and management-policy learning where the rule baseline is weakest.

**MVP 1 — Rule-driven simulator (no learning).** Zero research risk.
1. Data adapters + feature series (SPEC-002)
2. Simulator + assignment mechanics + daily MTM (SPEC-003)
3. Contract selector + risk engine (SPEC-004)
4. The four baselines (SPEC-006)
5. **GATE G1: simulator calibration vs. CBOE PUT index (SPEC-003 §7). Do not proceed to learning until it passes.**
6. Trajectory logging live for baseline runs (SPEC-001 §7)

**MVP 2 — Learned cash policy.**
7. Counterfactual sweep engine (SPEC-005 §2)
8. 36-state cash Q-table: double estimator + LCB + rule-baseline prior (SPEC-005 §3–4)
9. Walk-forward evaluation vs. Baseline 3 (SPEC-006)
10. **GATE G2: learned policy beats Baseline 3 out-of-sample (SPEC-006 §6). No expansion before this.**

> **Status 2026-08-22 — MVP 3 (SPEC-007) executed. Both gates FAILED — honestly and informatively.**
> **G2-RERUN FAILED**: with the AV EPS proxy the valuation axis populated 30/36 states (AC-4 pass), yet the learned opening policy still matches Baseline 3 to noise (pooled test −0.04%/yr, F1 drawdown ratio 1.49 where it deviated). B3 conditions on valuation too, so the axis helped both arms; opening decisions are now settled twice in favor of the rule table.
> **G3 FAILED on the no-added-drawdown criterion**: F1's learned management table deviated from HOLD in 0 of 70 states (≡ incumbent); F2 found exactly one credible deviation — SHORT_PUT / BULL_HIGH_VOL / SAFE / EXPIRY_WEEK → ROLL_SAME_RISK (the practitioner "roll early when safe" rule, +1.5bp/cycle, n_eff 71) — worth +0.08%/yr on test but with drawdown ratio 1.094 > 1.00. Both return criteria passed; the strict risk criterion did not.
> **Bonus finding: M-B2 (the mechanical ±3% MOS roll rule) is actively harmful** — −1.2%/yr (F1) and −2.3%/yr (F2) vs hold-to-expiry. Do not adopt the sibling project's roll rule mechanically.
> **Structural caveat**: the synthetic-BS track prices every option at fair vol + one global uplift, so management alpha driven by IV dynamics (vol spikes, skew shifts) is invisible by construction. Management learning may be under-powered here; a real-IV data source is the prerequisite for a fair retest (blocked by the Tiingo/AV-only constraint).
> **Standing deliverable: the rule-driven assistant — B3 openings + hold-to-expiry + selector + risk engine — is now the twice-validated production policy.**

**MVP 3 — RESCOPED 2026-08-22 (see SPEC-007; original scope retired after the G2 post-mortem).**
- Track A: learned management policies (roll/close/accept) vs. the hold-to-expiry incumbent — **GATE G3** (interface: SPEC-001A)
- Track B: Alpha Vantage EPS-based valuation proxy to revive the dead valuation axis — **GATE G2-rerun**
- Deferred indefinitely: HMM, GARCH, adaptive Q-blending, trend/momentum promotion, ICRL prep

> **Ablation 2026-08-22 (trend replaces valuation, SPEC-001 §3.2 gate): FAILED, worst of the three axis variants.** State = regime × trend(3) × vol-comp, momentum-following rule prior, otherwise byte-identical G2 pipeline. Pooled test −0.74%/yr (F1 −0.62%, F2 −0.86%; only 11–23% of episodes positive; sideways segment lagging >2pts in F2) vs −0.04%/yr for the EPS-proxy valuation axis and −0.21%/yr original. Trend deviations from the rule prior look strong in-sample (trend clusters within regimes inflate apparent n_eff) and generalize worst out-of-sample. Conclusion: the trend axis is *worse* than the valuation axis for the learner; the opening-decision question stays closed and trend stays a logged-only feature. Verdict: `reports/ablation_trend_verdict.json`.

> **Real-chain era (2026-08-23, AV Premium; DATA-GAP-1 closed).** 34M chain rows 2012→present, 11 tickers, 0 null IVs, FB/GOOG rename eras merged; spread-aware fills; BS fallback used on ≤0.1% of marks.
> **B3 real-premium absolutes** (`reports/b3_performance_historical.json`): full 2013–2026 **+30.2% CAGR / −42.6% maxDD** ($100K→$3.64M) — synthetic had undersold by ~6pts/yr; Test-1 +11.0%/−14.2%, Test-2 +12.5%/−12.1%. New finding: WAIT-in-stress cost ~7pts/yr vs B1 in the 2022 bear (bought 4.7pts less DD) because the market-level vol-comp proxy missed the real per-ticker IV richness → **next candidate experiment: per-ticker IV-percentile vol-comp** (now computable).
> **G3 RETEST on real chains: FAILED — and conclusively.** The learned management table deviated from HOLD in **0 of 67–69 states in both folds** (diff ≡ 0.0): even with real IV-spike economics, no roll/close deviation clears the pessimistic bar — the synthetic run's one marginal edge did not survive real spread costs. M-B2 (mechanical ±3% MOS roll) harmful again on real premiums (−0.9, −1.5 %/yr). **Hold-to-expiry management is now validated on both premium tracks; the management question is closed.**

**MVP 4 — ICRL.** Only reconsidered after the G3 and G2-rerun verdicts exist.

## 5. Non-goals (v1)

- No live order execution — output is recommendations only.
- No multi-ticker portfolio *policy* (portfolio effects live in the risk engine; per-ticker policies only).
- No early-exercise/dividend assignment modeling (expiration-only; config hook exists).
- No neural policies before MVP 4.
- No second valuation model — ingest the existing FV signal.

## 6. Repo layout (target)

```
wheel-strategy-rlbot/
├── APPROACH.md
├── specs/
├── rlbot/
│   ├── config.py
│   ├── vendor/            # copied unchanged from ../wheel-strategy (provenance header added)
│   │   ├── technicals.py
│   │   ├── options_engine.py
│   │   ├── data_access.py
│   │   └── portfolio_risk.py
│   ├── data/              # adapters + canonical table builders   (SPEC-002)
│   ├── features/          # per-day feature/state series           (SPEC-002)
│   ├── state/             # state machine + encoder                (SPEC-001)
│   ├── options/           # premium sources, selector              (SPEC-004)
│   ├── risk/              # hard constraints                       (SPEC-004)
│   ├── simulator/         # environment, execution, portfolio      (SPEC-003)
│   ├── learning/          # sweep, estimators, promotion           (SPEC-005)
│   ├── benchmarks/        # baselines 1–4 (+ grid-search)          (SPEC-006)
│   ├── evaluation/        # walk-forward, metrics, reports         (SPEC-006)
│   └── assistant/         # recommendation object, explanation
├── tests/                 # pytest; every spec's ACs map to tests
└── data_local/            # gitignored: bars, external CSVs, snapshots, trajectories
```

## 7. Conventions

- Python ≥ 3.11, pytest, type hints. Every module has tests (workspace rule: verify before done).
- All randomness seeded; all backtests reproducible from config + data snapshot.
- Trajectory records and state encodings carry `schema_version`; breaking a frozen schema requires a new version, never mutation.
- Secrets via `.env` (`TIINGO_API_KEY`, optional `FMP_API_KEY`, `ALPHA_API_KEY`); never committed.
