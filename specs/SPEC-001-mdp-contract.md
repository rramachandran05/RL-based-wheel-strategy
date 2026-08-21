# SPEC-001 — MDP Contract (Frozen Interface)

_Status: draft v1 — 2026-08-21. Becomes **frozen** at v1 sign-off; changes after that require `schema_version` bump, never mutation._

This spec defines the immutable contract between simulator, policy, learning, and assistant layers: position state machine, state encoding, action spaces, reward equation, decision epochs, trajectory record schema, and transition/terminal rules.

## 1. Position state machine

```
CASH ──sell put──▶ SHORT_PUT ──expires/closed──▶ CASH
                       └──────assigned─────────▶ LONG_STOCK
LONG_STOCK ──sell call──▶ COVERED_CALL ──expires/closed──▶ LONG_STOCK
                              └────────called away───────▶ CASH
```

`PositionState = {CASH, SHORT_PUT, LONG_STOCK, COVERED_CALL}`. The state machine is the single authority on legal actions; the policy layer can never emit an action outside the current state's set (REQ-1.1).

## 2. Action spaces

### 2.1 Cash policy (learned in MVP 2)

| Action id | Name | Target put delta (abs) |
|---|---|---|
| 0 | WAIT | — |
| 1 | PUT_DEFENSIVE | 0.05–0.10 |
| 2 | PUT_CONSERVATIVE | 0.10–0.18 |
| 3 | PUT_BALANCED | 0.18–0.25 |
| 4 | PUT_AGGRESSIVE | 0.25–0.35 |
| 5 | PUT_VERY_AGGRESSIVE | 0.35–0.45 |

### 2.2 Stock policy (learned in MVP 2)

| Action id | Name | Target call delta |
|---|---|---|
| 0 | WAIT (no call) | — |
| 1 | CALL_DEFENSIVE | 0.05–0.10 |
| 2 | CALL_CONSERVATIVE | 0.10–0.18 |
| 3 | CALL_BALANCED | 0.18–0.25 |
| 4 | CALL_AGGRESSIVE | 0.25–0.35 |

### 2.3 Management policies (fixed rules in MVP 1–2; learned in MVP 3)

- SHORT_PUT: `{HOLD, CLOSE, ROLL_SAME_RISK, ROLL_LOWER_RISK, ROLL_HIGHER_RISK, ACCEPT_ASSIGNMENT}`
- COVERED_CALL: `{HOLD, CLOSE, ROLL_OUT, ROLL_UP_AND_OUT, ALLOW_CALL_AWAY}`

MVP fixed rule: HOLD to expiration; assignment per SPEC-003 §5. The action enums are defined now so trajectory records logged in MVP 1–2 remain valid when MVP 3 learns these decisions.

Target DTE window for all opening actions: **25–45 days**, preference 30 (config: `target_dte=(25, 45, 30)`).

Delta bands are engineering parameters (`config.delta_bands`), not part of the frozen schema — the frozen part is the action *identity* (tier ordering and names).

## 3. State encoding contract

### 3.1 Q-state (v1) — what the policy conditions on

```python
QStateV1 = (
    market_regime,     # int 0-3
    valuation_state,   # int 0-2
    vol_compensation,  # int 0-2
)                      # 4 × 3 × 3 = 36 cells per position-state policy
```

**market_regime** (rule-based; from SPY bars + VIX; exact rules in SPEC-002 §3.3):

| id | Name | Definition (defaults; thresholds in config) |
|---|---|---|
| 0 | BULL_LOW_VOL | SPY > SMA200, drawdown > −10%, VIX 5y-percentile < 60 |
| 1 | BULL_HIGH_VOL | SPY > SMA200, drawdown > −10%, VIX 5y-percentile ≥ 60 |
| 2 | SIDEWAYS | SPY > SMA200 fails the bull tests but drawdown > −15% and VIX pct < 85; or |SPY/SMA200 − 1| ≤ 2% |
| 3 | BEAR_STRESS | SPY < SMA200 with drawdown ≤ −10%, or drawdown ≤ −15%, or VIX pct ≥ 85 |

Precedence: BEAR_STRESS ▸ BULL_HIGH_VOL ▸ BULL_LOW_VOL ▸ SIDEWAYS (first match wins, evaluated in that order after computing all conditions).

**valuation_state** (from FV anchors, SPEC-002 §3.4): `d = (price − fv_buy) / fv_buy`

| id | Name | Rule |
|---|---|---|
| 0 | ATTRACTIVE | d < −0.05 |
| 1 | FAIR | −0.05 ≤ d ≤ +0.05, **or FV unknown** |
| 2 | EXPENSIVE | d > +0.05 |

FV-unknown → FAIR is the deliberate degradation path for the historical era where no valuation series exists (SPEC-002 §5); valuation still acts through the selector's assignment penalty whenever known.

**vol_compensation** (market-level VRP proxy until per-ticker IV exists; SPEC-002 §3.5): with `vrp = VIX − realized_vol_20d(SPY)` (both in vol points) and `vix_pct` = VIX 5y rolling percentile:

| id | Name | Rule |
|---|---|---|
| 0 | POOR | vrp < 0 or vix_pct < 20 |
| 1 | NORMAL | otherwise |
| 2 | ATTRACTIVE | vrp ≥ 2.0 and vix_pct ≥ 60 |

When historical chains arrive, per-ticker IV percentile and IV−RV replace the proxy behind the same 3-level enum — the encoding contract does not change.

### 3.2 Logged-but-not-conditioned features (mandatory in every trajectory record)

`stock_trend` (5-level, from `technicals.classify_structure` mapping), `momentum` (5-level, from `classify_momentum`), `event_risk` (bool: earnings inside candidate cycle window), `concentration` (2-level), `drawdown_bucket`, plus the raw floats behind every bucket. These are candidates for ablation-gated promotion into the Q-state (MVP 3). Promotion = new `schema_version`, new table; old trajectories remain trainable because raw features were logged.

### 3.3 Management state extension (MVP 3, reserved now)

`MgmtStateV1 = QStateV1 + (dte_bucket, moneyness_bucket, premium_captured_bucket)` — bucket definitions to be fixed in a SPEC-001 v2 addendum before MVP 3; field names reserved in the trajectory schema now.

## 4. Reward (differential)

For a decision at epoch `t` with next epoch `t+Δt`:

```
r = [NAV_policy(t+Δt) − NAV_policy(t)] / NAV_policy(t)
  − [NAV_ref(t+Δt)   − NAV_ref(t)]   / NAV_ref(t)
```

- Reference for CASH-state decisions: **Baseline 1, fixed 20-delta wheel** (SPEC-006 §2), simulated over the same window, same ticker, same starting cash.
- Reference for LONG_STOCK-state decisions: **buy-and-hold** of the same shares.
- Both legs are charged identical friction (SPEC-003 §4). Both legs are marked to market daily; NAV per SPEC-003 §3.
- **γ = 1 within a cycle** (no per-day discounting). Across decision epochs, discounting per SPEC-005 §3 (default γ_epoch = 1.0 with finite-horizon episodes).
- No drawdown/tail penalty terms in v1. Adding them later = reward version bump (`reward_version` field), never silent change.

## 5. Decision epochs (semi-MDP)

A policy call happens only at decision events:

| # | Trigger | Applies to |
|---|---|---|
| E1 | Flat (CASH, or LONG_STOCK with no call) and `epoch_cadence` (default 5 trading days) elapsed since last decision | opening decisions |
| E2 | Open option reaches expiration | management (MVP: forced resolution) |
| E3 | Market regime value changed since last epoch | all |
| E4 | Earnings enters the open position's remaining window | management |
| E5 | Risk engine invalidates the held position (SPEC-004 §4) | management |

A WAIT decision holds until the next E1/E3 trigger. Rewards accumulate over the full inter-epoch window and attach to the decision that opened it.

## 6. Transitions and terminals

- Assignment/call-away mechanics: SPEC-003 §5. Cost basis after put assignment = strike − total premiums received this cycle.
- Episode: one (ticker, start_date) pair, horizon `episode_days` (default 252 trading days) or end of data. Terminal state contributes no bootstrap value (finite-horizon).
- After call-away → CASH, the next epoch is a fresh E1 decision (no automatic re-entry).

## 7. Trajectory record schema (frozen) — `trajectory_v1`

One JSON object per decision epoch, JSONL files under `data_local/trajectories/`. All prices/NAVs in USD floats; dates ISO `YYYY-MM-DD`.

```json
{
  "schema_version": "trajectory_v1",
  "reward_version": "diff_v1",
  "run_id": "…", "episode_id": "…", "decision_id": "…",

  "date": "2024-03-15", "ticker": "AAPL",
  "position_state": "CASH",

  "q_state": [1, 0, 2],
  "features": {
    "market_regime": 1, "valuation_state": 0, "vol_compensation": 2,
    "stock_trend": 4, "momentum": 2, "event_risk": false, "concentration": 0,
    "raw": {"spot": 172.4, "vix": 19.8, "vix_pct": 0.63, "vrp": 4.1,
             "fv_buy": 185.0, "fv_dist": -0.068, "rsi14": 57.2, "adx14": 22.0,
             "sma50": 169.1, "sma200": 175.9, "atr20": 3.4,
             "realized_vol_30": 0.24, "drawdown": -0.03}
  },

  "available_actions": [0,1,2,3,4,5],
  "chosen_action": 4,
  "action_source": "policy|baseline|sweep",
  "policy_meta": {"q_values": {"0": 0.0, "...": 0.0}, "q_lcb": {}, "n_eff": {}, "table_version": "…"},

  "contract": {"type": "PUT", "strike": 165.0, "expiration": "2024-04-19",
                "dte": 35, "delta": -0.29, "premium_fill": 2.84,
                "premium_source": "synthetic_bs|historical_chain",
                "candidates_considered": 7},
  "risk_checks": {"passed": true, "flags": []},

  "portfolio_before": {"cash": 100000.0, "shares": 0, "cost_basis": null, "nav": 100000.0},
  "next_epoch_date": "2024-04-19",
  "portfolio_after":  {"cash": 100284.0, "shares": 0, "cost_basis": null, "nav": 100284.0},

  "reward": 0.0041,
  "reference_nav_change": 0.0102,
  "next_q_state": [1, 1, 1],
  "next_position_state": "CASH",
  "terminal": false,

  "counterfactuals": {"0": 0.0, "1": 0.0012, "2": 0.0019, "3": 0.0031, "5": -0.0080}
}
```

Rules:
- `contract` is `null` for WAIT/HOLD. `counterfactuals` present only for sweep-generated records (SPEC-005 §2); keys are action ids **other than** `chosen_action`.
- Every field above is required (nullable where shown). Producers may add keys under `features.raw` freely; nowhere else.
- Consumers must ignore unknown `features.raw` keys and must reject records whose `schema_version` they don't support.

## 8. Requirements

- **REQ-1.1** The state machine module exposes `legal_actions(position_state) -> list[Action]`; simulator and policy both consult it; an illegal action raises.
- **REQ-1.2** `encode_q_state(features) -> tuple[int,int,int]` is a pure function; identical inputs give identical outputs across simulator/live paths (single implementation, no duplicates).
- **REQ-1.3** Every decision epoch in any run (baseline, sweep, learned, live) emits a `trajectory_v1` record that validates against a checked-in JSON Schema.
- **REQ-1.4** Reward computation is implemented once, takes two NAV series (policy leg, reference leg), and is unit-tested against hand-computed cases.
- **REQ-1.5** All enum ↔ int mappings live in one module (`rlbot/state/enums.py`) with exhaustive round-trip tests.

## 9. Acceptance criteria

- **AC-1** JSON Schema file exists; ≥ 3 fixture records (CASH open, LONG_STOCK open, expiry management) validate; a record with a missing required field fails validation.
- **AC-2** Property test: for every `PositionState`, `legal_actions` is non-empty and WAIT/HOLD ∈ set.
- **AC-3** State-encoder golden tests: ≥ 12 hand-constructed feature dicts (covering every enum value and each regime-precedence tie-break) map to expected `q_state` tuples.
- **AC-4** Reward test: flat stock path + kept premium → positive differential reward vs 20-delta reference on a window where reference also kept premium ⇒ r reflects only the delta-tier difference; capped-rally covered-call case yields negative differential vs buy-and-hold.
- **AC-5** WAIT counterfactual on a flat path returns exactly 0 differential vs a WAIT reference leg (`test_wait_action_earns_zero_when_flat`).
