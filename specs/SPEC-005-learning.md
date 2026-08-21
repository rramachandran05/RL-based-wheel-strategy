# SPEC-005 — Learning: Counterfactual Sweep, Estimators, Pessimism, Promotion

_Status: draft v1 — 2026-08-21. Blocked on GATE G1 (SPEC-003 §7)._

## 1. Core insight

The environment is exogenous — our actions never move the underlying or option prices. So offline training does not explore: at every decision epoch we simulate **every** legal action against the same historical path and record each action's realized outcome. Full-information feedback; no epsilon-greedy; sample efficiency ×|A|; visitation counts become true per-action counts.

## 2. Counterfactual sweep engine

For each (ticker, date) decision epoch drawn from the walk-forward **training window only**:

```
for a in legal_actions(position_state):
    contract = selector(a, chain(d))               # None for WAIT; risk engine applied
    branch   = simulate_to_next_epoch(d, contract) # SPEC-003 loop, one inter-epoch window
    target[a] = branch.differential_return + continuation(branch.end_state)
```

- **Branch closure:** branches end at the branch's own next epoch (E1–E5). Portfolio-state divergence (assigned vs not) is closed with a **baseline continuation value**: the differential return of Baseline 3 (rule policy) run from `branch.end_state` to a fixed horizon `continuation_days` (default 63 trading days), computed once per (state, date) and cached. No tree expansion.
- Each sweep emits one `trajectory_v1` record per epoch with `action_source="sweep"`, `chosen_action` = the rule-policy action (for lineage), and the full `counterfactuals` map.
- Epoch inventory expectation: ~10 tickers × ~14 years × ~8–12 epochs/yr ≈ **1,100–1,700 epochs**, each yielding all-action targets → dense coverage of 36 states is plausible; coverage report is a required artifact (§6).

## 3. Estimator — per-state regression, not TD

With full action feedback, learning collapses to estimating `Q(s,a) = E[target | s, a]`:

- Maintain per (s, a): running mean, M2 (variance), raw count `n`, and **cluster labels** for effective sample size.
- **Effective N:** targets are correlated across overlapping windows and co-moving tickers. Cluster key = `(regime_episode_id, calendar_quarter)` where `regime_episode_id` increments each time `market_regime` changes value in the market table. `n_eff(s,a)` = number of distinct clusters contributing. All confidence math uses `n_eff`, never raw `n`.
- **Double estimation:** epochs are split A/B by hash of `episode_id`. Mean_A is evaluated against selection done on B and vice versa wherever a max/argmax is taken (bias check in evaluation); the deployed value is the pooled mean, but promotion tests (§5) must pass on both halves.
- **Bootstrapping:** none in MVP 2 (γ_epoch = 1, finite horizon, continuation value plays the role of the tail). The semi-MDP TD update `Q(s,a) ← Q(s,a) + α[R + γ^Δt max Q(s',·) − Q(s,a)]` is reserved for MVP 3 management policies where sweeps get expensive; if used, Double Q-learning is mandatory.

## 4. Pessimism and prior

- **Prior:** initialize every (s, a) from Baseline 3's implied preference: `Q0(s,a) = mean target of the rule policy's chosen tier in s` (or 0 where the rule never visits), with pseudo-count `n0 = 5` clusters. Learned values are shrunk: `Q_shrunk = (n_eff·Q̂ + n0·Q0) / (n_eff + n0)`.
- **Act on the lower confidence bound:** `LCB(s,a) = Q_shrunk − z·σ̂(s,a)/√n_eff`, default `z = 1.0` (config). Deployed policy: `argmax_a LCB(s,a)` over legal actions. Honest framing: the learned policy is a data-driven perturbation of the rule table.
- Ties → the more conservative tier (lower assignment risk).

## 5. Production vs. learning tables — promotion protocol

- Two artifacts: `qtable_learning.json` (updated by every training run) and `qtable_production.json` (what evaluation/live uses). Both carry `table_version`, training-window manifest, and full per-cell stats (Q, n, n_eff, mean, var, last_updated).
- Promotion requires, on the walk-forward **validation** split (SPEC-006 §4): beats Baseline 3 on differential return; max drawdown ≤ 1.1× Baseline 3's; no single regime segment with differential return < −2 pts annualized; both A/B halves individually non-negative vs Baseline 3; all risk-engine invariants clean.
- Live-era experience appends to learning only; promotion is always an explicit, logged, human-triggered step.

## 6. Required artifacts per training run

`run_manifest.json` (config hash, data snapshot, window), coverage report (per-state × action n_eff heatmap; cells with n_eff < 3 flagged UNTRUSTED and excluded from LCB argmax — rule prior acts there), A/B agreement report, and the promoted/rejected verdict.

## 7. Requirements

- **REQ-5.1** Sweep uses the SPEC-003 simulator verbatim (no reimplementation) — import-graph test.
- **REQ-5.2** Sweep targets for WAIT on a flat path = 0 exactly (ties to SPEC-001 AC-5).
- **REQ-5.3** Estimator math (running mean/M2, shrinkage, LCB) unit-tested against numpy reference on synthetic data.
- **REQ-5.4** `n_eff` provably ≤ raw n; clustering verified on a fixture with known episode boundaries.
- **REQ-5.5** Policy loader refuses a `qtable_production.json` whose manifest doesn't match the current schema/enums version.
- **REQ-5.6** With `z` large (→ ∞), deployed policy converges to the rule prior everywhere (sanity limit test).

## 8. Acceptance criteria

- **AC-1** End-to-end sweep on a 2-ticker, 3-year fixture produces trajectory records validating against schema, with counterfactuals for all legal actions at every epoch.
- **AC-2** Coverage report generates; UNTRUSTED cells fall back to prior (verified by constructing a starved cell).
- **AC-3** Promotion protocol rejects a deliberately overfit table (trained and "validated" on the same window is refused by manifest check).
- **AC-4** REQ-5.6 limit test passes.
- **AC-5** Reproducibility: same manifest ⇒ identical `qtable_learning.json` bytes.
