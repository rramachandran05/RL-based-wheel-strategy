# SPEC-003 — Simulator: Environment, Execution, Assignment, MTM, Calibration Gate

_Status: draft v1 — 2026-08-21_

## 1. Scope

Event-driven wheel simulator over historical daily bars. Consumes: canonical tables (SPEC-002), a `PremiumSource`, a policy callable, the contract selector + risk engine (SPEC-004). Produces: daily NAV series, `trajectory_v1` records, episode summaries. It is the single implementation used by baselines, counterfactual sweeps, learned-policy evaluation, and (replayed) live recommendations — no forked logic.

Seed: `../wheel-strategy/wheel_backtest.simulate_wheel` (fixed-policy, touch-assignment, premium-at-open-only). This spec generalizes it: per-epoch policy decisions, daily mark-to-market, expiration-settled assignment, differential-reward bookkeeping.

## 2. Episode loop

```
init: cash = config.starting_cash (default 100_000), shares = 0, no option
for each trading day d in [start, start + episode_days]:
    update marks (§3)
    if d is a decision epoch (SPEC-001 §5):
        features = as_of_features(d, ticker)          # SPEC-002
        q_state  = encode_q_state(features)            # SPEC-001
        action   = policy(position_state, q_state, features)
        if action opens a position:
            contract = selector(action, chain(d))      # SPEC-004
            contract = risk_engine.validate(contract)  # may downgrade tier or force WAIT
            execute (§4)
        emit trajectory record at next epoch boundary (rewards need the window)
    if d is an expiration date for the open option: settle (§5)
advance; at horizon: liquidate marks (no forced sales), mark terminal record
```

Reference legs (SPEC-001 §4) run in lockstep inside the same loop harness with identical friction, so `reward` is computable at each epoch boundary.

## 3. Mark-to-market

`NAV_d = cash + shares·close_d − option_liability_d`

- Synthetic track: `option_liability_d` = BS repricing of the open contract at day-d spot, remaining DTE, and day-d realized-vol proxy.
- Historical-chain track: mid of the day-d chain row (fallback BS if row missing, flagged).
- A deteriorating short put therefore shows losses before assignment.

## 4. Execution model

- Sell to open: `fill = mid − k·(ask − bid)`, buy to close: `fill = mid + k·(ask − bid)`, default `k = 0.5` (config).
- Synthetic track has no quoted spread: use `slippage_pct_of_premium = 0.03` haircut (adopted from sibling backtest) in place of the spread term, i.e. `fill = bs_price·(1 − 0.03)` on sells.
- Commission `$0.65/contract` per leg. All costs charged to both policy and reference legs.
- Cash-secured: opening a put escrows `strike × 100 × contracts` (enforced by risk engine, mirrored here).

## 5. Expiration and assignment (MVP: expiration-settled)

- Put: if `close_T < strike`: cash −= strike·100, shares += 100, `cost_basis = strike − premiums_received_this_cycle`; SHORT_PUT → LONG_STOCK. Else SHORT_PUT → CASH.
- Call: if `close_T > strike`: shares −= 100, cash += strike·100; realized stock P&L vs cost basis recorded; COVERED_CALL → CASH. Else COVERED_CALL → LONG_STOCK.
- Config hook `assignment_model ∈ {expiration, touch}`: `touch` reproduces the sibling's intra-cycle high/low American approximation, kept for comparison runs; **default = expiration**. Early exercise, ex-dividend, corporate actions: out of scope v1 (splits/dividends already handled by adjusted bars).

## 6. Determinism and replay

- No wall-clock, no RNG in the simulator itself (policy exploration, if any, passes its own seeded RNG).
- `run_id = hash(config, data snapshot version, policy version)`; identical inputs ⇒ byte-identical trajectory output (AC-4).

## 7. GATE G1 — Calibration vs. CBOE PUT index

The simulator (and especially the synthetic premium source) must reproduce a known external ground truth before any learning runs.

**Procedure:**
1. Replicate PUT methodology inside the simulator: SPY underlying, sell one ATM (nearest-strike) put at each monthly expiration, hold to expiration, fully cash-collateralized, premiums reinvested.
2. Window: max overlap of SPY bars and the PUT series, minimum 2010-01-01 → present.
3. Compare monthly return series vs `put_index` (SPEC-002 §3.8).

**Pass thresholds (config, defaults):**
- Monthly-return correlation ≥ 0.85
- |annualized return difference| ≤ 3.0 pts
- Annualized volatility ratio within [0.75, 1.30]

**Calibration knob:** the synthetic source may apply a single global premium scalar `iv_uplift` (BS vol input = `realized_vol_30 · (1 + iv_uplift)`, default 0) fitted **once** on the first half of the window and validated on the second half — never refit per-ticker or per-regime. Rationale: realized vol systematically underprices IV; one scalar captures the average VRP without giving the synthetic track free parameters. Fitted value and both-half metrics are recorded in `data_local/calibration/put_gate.json`.

**Failing the gate blocks MVP 2** (enforced: the learning CLI refuses to run without a passing `put_gate.json` newer than the current data snapshot).

## 8. Requirements

- **REQ-3.1** One simulator implementation serves baselines, sweeps, evaluation, and live replay (no code forks).
- **REQ-3.2** Daily NAV identity holds: `NAV_d − NAV_{d-1}` decomposes exactly into stock P&L + option MTM + premiums + costs (accounting test, per day, to the cent).
- **REQ-3.3** Assignment mechanics unit-tested on constructed paths (ITM/OTM put and call, exact boundary `close_T == strike` → no assignment).
- **REQ-3.4** Reference legs run with identical friction and data; a policy identical to its reference produces reward ≡ 0 every epoch.
- **REQ-3.5** Gate G1 procedure is a single command (`python -m rlbot.evaluation.put_gate`) producing the JSON verdict + a comparison plot.

## 9. Acceptance criteria

- **AC-1** Constructed 3-cycle path (premium kept → assigned → called away) reproduces hand-computed cash/shares/cost-basis/NAV at every step.
- **AC-2** NAV decomposition test (REQ-3.2) passes over a full historical episode for 3 tickers.
- **AC-3** Baseline-1-as-its-own-reference yields zero reward at every epoch (REQ-3.4).
- **AC-4** Two runs with identical config + snapshot produce byte-identical JSONL (REQ-3.6/determinism).
- **AC-5** `put_gate.json` exists with pass=true on the current snapshot before any SPEC-005 artifact is produced (enforced, not honor-system).
- **AC-6** `assignment_model=touch` vs `expiration` comparison run completes and reports assignment-rate deltas (documentation artifact, no threshold).
