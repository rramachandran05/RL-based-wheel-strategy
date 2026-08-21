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

SPEC-001 is the immutable interface (frozen-manifest pattern): simulator, policy, and assistant layers all program against it, so the learning method can be swapped (Q-table → bandit → ICRL) without touching the simulator, and every trajectory logged from day one stays usable.

## 4. Roadmap and gates

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

**MVP 3 — Regime-aware adaptive policy.** Position-management Q-tables (roll/close/assign), ablation-gated state additions (trend, momentum), optional HMM/GARCH, adaptive Q blending.

**MVP 4 — ICRL.** Sequence model over recent + comparable historical trajectories; only after the trajectory DB is mature.

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
