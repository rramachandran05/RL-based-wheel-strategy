# Wheel-Strategy RL Bot — Overall Approach

_Revision 2 — 2026-08-21. Incorporates the review-round changes: counterfactual sweep training, differential reward, hard state-space reduction with pessimistic offline learning, data reality checks, and a risk-reordered MVP. Supersedes the v1 approach text._

---

## 1. Philosophy

The strategy seeks to continuously monetize option premium when sufficiently compensated for risk. It adjusts strike selection dynamically according to the desirability and probability of assignment rather than restricting put sales to preferred acquisition prices. Favorable conditions permit more aggressive strikes and greater assignment exposure; unfavorable conditions progressively reduce assignment exposure by moving strikes farther out of the money. No-trade (WAIT) remains an available action when even conservative strikes do not provide adequate compensation for risk.

**The central design choice: the policy decides a risk budget, not a contract.** Instead of asking the model for a strike, the RL policy answers "how much assignment risk am I willing to take right now?" A deterministic option-selection engine then converts that risk budget into delta → strike → DTE → premium, and a hard risk engine validates the result. RL proposes; the risk engine disposes.

```
Keep the Wheel working when the market is paying enough for the risk,
but dynamically change how much assignment or call-away risk you accept.
Better conditions → more aggressive strikes, more premium.
Worse conditions → progressively safer strikes, less premium.
Extreme conditions → wait.
```

---

## 2. The Wheel as an Explicit State Machine

Built before any RL. The portfolio exists in four primary states, each with its own permitted actions — this prevents the agent from ever proposing a mechanically invalid action.

```
CASH
  ↓ sell put
SHORT_PUT
  ├── expires/closed → CASH
  └── assigned       → LONG_STOCK

LONG_STOCK
  ↓ sell call
COVERED_CALL
  ├── expires/closed → LONG_STOCK
  └── called away    → CASH
```

| Position state | Permitted actions |
|---|---|
| CASH | WAIT, SELL_PUT (at a chosen risk tier) |
| SHORT_PUT | HOLD, CLOSE, ROLL_SAME_RISK, ROLL_LOWER_RISK, ROLL_HIGHER_RISK, ACCEPT_ASSIGNMENT |
| LONG_STOCK | WAIT, SELL_CALL (at a chosen risk tier) |
| COVERED_CALL | HOLD, CLOSE, ROLL_OUT, ROLL_UP_AND_OUT, ALLOW_CALL_AWAY |

---

## 3. Sequential Decision Logic

At each **decision epoch** (see §8 — not every day):

1. **Portfolio state.** Cash, shares, open short put, or open covered call — determines available actions.
2. **Suitability gate.** Underlying still a company the investor is willing to own; option liquidity acceptable; assignment would not create unacceptable concentration. If not suitable, stop initiating new Wheel trades.
3. **Market regime.** Bull/low-vol, bull/high-vol, sideways, bear/stress — rule-based for MVP; HMM deferred (see §12).
4. **Stock technical condition.** Trend, momentum, drawdown, realized volatility, position vs. 50/100/200-day SMAs. (Deferred from the MVP state — see §5 — but computed and logged from day one.)
5. **Valuation condition.** Very attractive → very expensive, relative to fair-value and preferred-buy ranges. Valuation is not a hard constraint; it calibrates comfort with assignment.
6. **Options environment.** IV, IV percentile, IV vs. realized vol (volatility risk premium), skew, term structure, liquidity, available premium — how well is the market paying for risk?
7. **Near-term risk check.** Earnings, major events, extreme volatility, poor liquidity, gap risk.
8. **Risk posture.** Combine into: very favorable / favorable / neutral / unfavorable / extremely unfavorable for taking assignment risk.
9. **Cash side:** default to collecting premium when compensation is reasonable; WAIT when even conservative strikes don't pay. Choose an assignment-risk tier (§6), translate to a contract (§7). Never pick purely on probability of profit — compare premium, assignment probability, effective acquisition price, downside exposure, expected loss, liquidity, portfolio impact.
10. **Open short put:** continuously reevaluate hold/close/roll/accept-assignment. Never auto-roll a challenged put — a roll must improve risk/reward, not delay loss recognition.
11. **Assignment:** record effective cost basis (strike − premiums received); reassess the stock under the same framework.
12. **Stock side:** decide whether a covered call is worthwhile; choose a call-away-risk tier. Strongly bullish / undervalued → farther OTM or no call (preserve upside). Neutral → balanced. Bearish / overvalued / weakening → closer strikes (premium + exposure reduction becomes attractive). Opportunity cost counts: a capped rally is a real cost of the decision (the differential reward in §9 makes this automatic).
13. **Open covered call:** reevaluate hold/close/roll/allow-assignment against current valuation, trend, volatility, remaining premium, and whether selling at the strike is still desirable.
14. **Called away:** return to CASH; restart from the top — no automatic next put.
15. **Record everything** as a trajectory (§13); measure the full economic outcome; learn (§10).

---

## 4. Architecture

```
Data → Features → Market/Stock State → Wheel Position State
     → RL Policy → Assignment/Call-Away Risk Budget
     → Deterministic Contract Selector → Risk Engine (hard constraints)
     → Recommendation (+ LLM explanation) → Outcome → Learning
```

Separate Q-functions per position state — never one giant table:

- `Q_cash(state, action)` — cash policy
- `Q_put(state, action)` — short-put management
- `Q_stock(state, action)` — stock policy
- `Q_call(state, action)` — covered-call management

The MVP trains only `Q_cash` and `Q_stock`; put/call management uses fixed rules until Phase 2.

---

## 5. State Design — Small on Purpose

**Effective sample size is the binding constraint.** Overlapping episodes resampled from the same market path are heavily correlated; the real unit of independent evidence is the *regime episode*, and in the era where historical option chains are obtainable (~2012+ affordably) there are only three to four independent bear/stress episodes. Every conclusion about defensive actions rests on a handful of events. The state space must be sized to that reality, not to the feature wishlist.

**MVP cash-policy state: 36 cells.**

```
state = (market_regime, valuation_state, volatility_compensation)
         4 levels      × 3 levels       × 3 levels            = 36
```

- **market_regime:** BULL_LOW_VOL, BULL_HIGH_VOL, SIDEWAYS, BEAR_STRESS (rule-based from SPY vs. SMAs, SMA slopes, VIX level/percentile, drawdown)
- **valuation_state:** ATTRACTIVE, FAIR, EXPENSIVE (percentage distance from fair value; thresholds to be tested, e.g. <−5% / −5..+5% / >+5% initially collapsed from the 5-level scheme)
- **volatility_compensation:** POOR, NORMAL, ATTRACTIVE (IV percentile and IV − expected realized vol)

Trend, momentum, event risk, and concentration are **computed and logged in every trajectory record from day one** but excluded from the MVP Q-state. They earn their way in only via ablations showing they add value. Event risk and concentration act through the risk engine (§11) instead, where hard rules are more appropriate than learned behavior anyway.

Management states (Phase 2) add position fields: DTE, current delta, distance to strike, % premium captured, unrealized P&L.

Categorical encodings, not raw numbers: never feed $103.72 fair value or a raw SMA into a Q-table.

---

## 6. Action Space — Risk Budgets, Not Contracts

Actions are assignment-risk tiers (cash side) and call-away-risk tiers (stock side). The action space stays fixed while actual strikes change daily.

| Cash action | Approx. put delta |
|---|---|
| WAIT | 0 |
| PUT_DEFENSIVE | 0.05–0.10 |
| PUT_CONSERVATIVE | 0.10–0.18 |
| PUT_BALANCED | 0.18–0.25 |
| PUT_AGGRESSIVE | 0.25–0.35 |
| PUT_VERY_AGGRESSIVE | 0.35–0.45 |

| Stock action | Posture |
|---|---|
| WAIT (no call) | preserve full upside |
| CALL_DEFENSIVE | very far OTM |
| CALL_CONSERVATIVE | far OTM |
| CALL_BALANCED | moderate OTM |
| CALL_AGGRESSIVE | closer strike — premium + exposure reduction |

Delta ranges are initial engineering parameters, not strategy rules — the policy learns which tier fits which conditions.

---

## 7. Deterministic Contract Selector

Given a risk tier (e.g., PUT_CONSERVATIVE → delta 0.10–0.18, DTE 25–45), filter the chain (right type, delta in range, DTE in range, volume, open interest, bid/ask spread), then score candidates:

```
Score = w1·PremiumYield + w2·VolatilityPremium
      − w3·SpreadCost − w4·DownsideRisk − w5·AssignmentPenalty
```

The assignment penalty scales inversely with valuation attractiveness: an attractive stock makes assignment cheap to accept; an expensive one makes it costly. This is where valuation shapes behavior without being a hard constraint.

---

## 8. Decision Epochs — a Semi-MDP

No pointless daily trades. The policy is called at **decision events**:

- no option currently open
- option reaches an expiry threshold or profit target
- delta changes materially / underlying crosses strike
- regime or volatility-regime change
- earnings approaches

Between epochs the position simply rides. Rewards are aggregated over the inter-decision window. **Within a cycle, γ = 1** — with 20–45-day horizons, per-day discounting buys nothing and adds a tuning knob; keep it simple.

---

## 9. Reward — Differential, Not Raw NAV

Raw NAV-change reward in LONG_STOCK is dominated by the stock's drift: the table would learn "stocks go up in bull regimes" — true and useless. **The reward is the policy's NAV change minus a reference policy's NAV change over the same window:**

```
r = ΔNAV(policy) − ΔNAV(reference)     over the same inter-decision window
```

- **Reference in cash states:** a fixed 20-delta wheel.
- **Reference in stock states:** buy-and-hold.

This isolates the decision's contribution to wealth, and covered-call opportunity cost falls out automatically — a capped rally shows up as negative differential reward with no special accounting. The evaluation baselines (§14) serve double duty as reward references.

Costs (commissions, slippage) are charged inside each leg's NAV path. Drawdown/tail-risk penalty terms are deferred until the simple differential reward is demonstrably insufficient — complex rewards are easy to optimize incorrectly.

---

## 10. Learning — Counterfactual Sweep First, Q-Learning Second

**The environment is exogenous: our actions never move option prices or the underlying.** So at every decision epoch we do not have to choose one action and observe one outcome — we can simulate *all* actions against the same historical path and observe each one's realized outcome. This converts bandit feedback into full-information feedback, multiplies sample efficiency by roughly the action-space size, and eliminates the exploration/exploitation problem for the offline phase entirely. No epsilon-greedy in offline training.

```python
def counterfactual_sweep(epoch, chain, actions, simulate_to_next_epoch, baseline_value):
    """Evaluate every action against the same historical path."""
    results = {}
    for a in actions:
        contract = select_contract(chain, a)              # None for WAIT
        branch = simulate_to_next_epoch(epoch, contract)  # premium, MTM path, assignment
        results[a] = branch.cycle_return + baseline_value(branch.end_state)
    return results   # supervised targets for ALL actions at this state
```

Because different actions branch portfolio state (assigned vs. not), each branch is simulated **only to the next decision epoch**, then closed with a baseline continuation value (fixed-rule wheel from the branch's end state) instead of expanding a full tree. With full action feedback, learning largely collapses into **per-state regression over observed action returns** — simpler and more stable than TD bootstrapping, and visitation counts become real per-action counts.

Guards against offline-RL failure modes:

- **Double Q-learning** wherever a max operator appears (tabular — trivial to add): max + noisy financial rewards otherwise guarantees optimistic bias.
- **Act on a lower confidence bound** of Q, not the point estimate: optimistic offline RL reliably exploits estimation noise; pessimism is the standard fix.
- **Initialize from and shrink toward the rule baseline** (Baseline 3, §14), in proportion to correlation-adjusted effective sample size. Honest framing: the learned policy is a data-driven perturbation of the rule table, not a from-scratch discovery.

Per Q-entry bookkeeping: Q value, observation count, average return, return variance, last-updated — with the caveat that raw N overstates evidence (correlated episodes); the LCB uses effective sample size.

**Semi-MDP update** (when bootstrapping is used at all): `Q(s,a) ← Q(s,a) + α[R + γ^Δt · max_a' Q(s',a') − Q(s,a)]` with R the accumulated inter-decision reward and Δt the days to the next decision.

**Production vs. learning policy:** live experience updates a learning table only; promotion to the production table requires passing performance, drawdown, regime, and risk-limit tests in backtest. One unusual month must not rewrite the strategy.

---

## 11. Hard Risk Engine — Not Learnable, Not Bypassable

Some decisions never depend on Q-learning:

- No naked calls; cash-secured puts only (cash sufficient for assignment)
- Max position size; max ticker concentration; max simultaneous positions
- Min option liquidity; max bid/ask spread
- Configurable event policies (e.g., earnings blackout)
- **Portfolio-level synchronized-assignment constraint:** the wheel's true tail risk is every short put assigning in the same crash week. That is inherently invisible to per-ticker Q-tables and lives permanently here as a portfolio-level cap on aggregate assignment-at-once exposure — it is never something to learn.

```
RL recommends → Risk engine validates → allowed: contract recommendation
                                      → rejected: step down risk tier, or WAIT
```

---

## 12. Data Plan and Reality Checks

Canonical daily historical snapshots sufficient to recreate what would have been known on each date: market table (SPY, SMAs, VIX, breadth, realized vol), underlying table (OHLCV, returns, drawdown, SMAs, RSI, momentum, realized vol), valuation table (ingest the existing wheel-strategy fair-value signal — do not build a second valuation model), **historical options-chain table** (the critical dataset: per snapshot date × expiration × strike: bid/ask/mid, IV, greeks, OI, volume), events table, portfolio-state table.

**Code-review finding (2026-08-21):** the existing `wheel-strategy` repo contains **no option-chain data at all** — every premium it produces, live and backtested, is Black-Scholes synthetic on a 30-day realized-vol proxy with r=0. The RL project therefore defines a `PremiumSource` interface with two implementations: **synthetic-BS** (available day one, reusing `options_engine.py`, known conservative bias) and **historical-chain** (when data is acquired). The simulator, policy, and trajectory schema are identical across both; only premium/greeks provenance changes. The PUT-index calibration gate is what keeps the synthetic track honest.

| Item | Issue | Decision |
|---|---|---|
| Historical option chains | The critical path and main cost | Theta Data or ORATS (affordable tiers); OptionMetrics is academic-grade but pricey. Start with SPY + 5–10 liquid mega-caps. Until purchased: synthetic-BS track |
| Fear & Greed history | No long official CNN history | Build a reproducible proxy composite (VIX percentile, put/call ratio, breadth) usable in walk-forward |
| Survivorship bias | Training only on NOW/MSFT/META — survivors — overstates aggressive-put value | Include delisted/cratered names where feasible; otherwise explicitly scope the policy as conditional on the "willing to own" screen and state so in every evaluation |
| Simulator correctness | How do we know the simulator itself is right? | **Calibration gate:** run a fixed-delta SPY wheel through the simulator and compare against the CBOE PUT index — free external ground truth. Required test before any learning begins |
| HMM regimes (later phase) | Refitting per walk-forward step causes regime-label switching, silently breaking Q-state identity | Rule-based regimes for MVP. If HMM later: train only through time t (no look-ahead), enforce label ordering by volatility |
| GARCH (later phase) | Complexity before it's earned | Defer; MVP uses IV percentile + realized-vol spread for volatility compensation |

---

## 13. Simulator

Event-driven backtesting environment. Episode: start with cash (e.g., $100,000), no stock, no option; run the wheel 6–12 months.

- **Execution friction:** never assume mid fills. `Fill = Mid − k·(Ask−Bid)` when selling; reverse when buying to close. Plus commissions and fees.
- **Daily mark-to-market:** `NAV = cash + shares·price − short_option_liability`. A deteriorating short put shows as a loss before assignment.
- **Expiration/assignment mechanics:** put ITM at expiry → assign (cash −= strike×100, shares += 100, cost basis = strike − premium received, SHORT_PUT → LONG_STOCK); call ITM → called away (COVERED_CALL → CASH). Expiration-only assignment for MVP; early exercise, ex-dividend effects, and corporate actions later.
- **Walk-forward only — never randomly split time series.** Every computed quantity (valuation, IV percentile, SMAs, regimes, earnings knowledge) uses only information available at that historical date.

---

## 14. Baselines and Evaluation

Baselines built **before** any learning — they are both the yardstick and the reward references:

1. **Fixed Wheel** — 30–45 DTE, 20-delta puts and calls
2. **Conservative Wheel** — 10–15 delta
3. **Simple adaptive rules** — Bull+Cheap → 30Δ; Neutral → 20Δ; Bear → 10Δ; Extreme bear → WAIT (also the Q-table prior)
4. **Buy-and-hold**

**If RL cannot reliably beat Baseline 3, the complexity isn't justified.** That is the project's falsifiable claim.

Metrics beyond win rate: CAGR, volatility, Sharpe, Sortino, max drawdown, CVaR; premium income and yield; assignment and call-away rates; average delta/DTE; capital utilization, time in cash/stock; turnover and costs. **Segment all results by regime** (bull, bear, crash, recovery, sideways, high/low vol) — the regime breakdown matters more than headline CAGR.

---

## 15. Assistant Layer

The LLM never calculates the trade. Quantitative engines produce a structured recommendation object; the LLM renders it human-readable and answers comparative questions ("why not the 15-delta put?") from the quantitative outputs.

```json
{
  "position_state": "CASH",
  "market_regime": "BULL_HIGH_VOL",
  "valuation": "ATTRACTIVE",
  "volatility_compensation": "ATTRACTIVE",
  "policy_action": "PUT_AGGRESSIVE",
  "target_delta": [0.25, 0.35],
  "target_dte": [25, 45],
  "selected_contract": {"strike": 180, "dte": 32, "delta": 0.29, "premium": 3.10},
  "q_value": 0.024,
  "q_lcb": 0.011,
  "effective_n": 212,
  "risk_checks": "PASS"
}
```

Every recommendation is logged as a trajectory record — `(s, a, r, s')` plus contract candidates, portfolio before/after, Q values, and the rejected alternatives. This dataset is the long-term asset and the substrate for later phases.

---

## 16. Roadmap — Ordered by Risk, With Gates

Steps 1–4 are pure engineering with zero research risk and produce a useful rule-driven assistant even if learning never beats the baseline — the project's floor stays high.

**MVP 1 — Rule-driven simulator (no learning)**
1. Data adapters (reusing wheel-strategy code where it fits) + simulator + assignment mechanics
2. The four baselines
3. **Gate: simulator calibration vs. CBOE PUT index — do not proceed until this passes**
4. Trajectory logging schema (frozen interface)

**MVP 2 — Learned cash policy**
5. Counterfactual sweep engine
6. 36-state cash Q-table with double-Q + LCB + baseline-rule prior
7. Walk-forward evaluation vs. Baseline 3
   **Gate: learned policy must beat Baseline 3 out-of-sample before any expansion**

**MVP 3 — Regime-aware adaptive policy**
- Position-management Q-tables (put/call roll-close-assign decisions)
- Ablation-gated state additions (trend, momentum)
- Optional HMM regimes, GARCH-based VRP; adaptive Q blending (long-term + recent + regime-conditional, weighted by regime confidence)

**MVP 4 — In-context adaptation (ICRL)**
- Only after the trajectory database is mature: sequence model over recent decisions + comparable historical episodes → next risk posture, adapting from context without per-trade weight updates.

---

## 17. Module Layout

```
wheel-strategy-rlbot/
├── data/          market, options, valuation, event adapters
├── features/      technicals, volatility, valuation, regime
├── state/         state_encoder, wheel_state (state machine)
├── policy/        cash_qtable, stock_qtable (put/call tables in Phase 2)
├── options/       contract_filter, contract_ranker, greeks
├── risk/          portfolio_limits, event_risk, trade_validator
├── simulator/     environment, execution, assignment, portfolio
├── learning/      counterfactual_sweep, q_learning, trajectories, policy_evaluation
├── benchmarks/    fixed_wheel, conservative_wheel, adaptive_rules, buy_and_hold
├── assistant/     recommendation, explanation, api
└── specs/         frozen interface + implementation specs
```

---

## 18. Frozen Interfaces

The formal MDP spec is the contract between simulator, policy, and assistant layers. Two schemas are **immutable interfaces** (frozen-manifest pattern): the **trajectory record schema** and the **state-encoding contract**. The learning method can then be swapped (Q-table → bandit → ICRL) without touching the simulator, and every logged trajectory from day one remains usable forever. The specs in `specs/` define these precisely: state bins, action definitions, contract-selection rules, differential-reward equation, transition rules, terminal conditions, and the trajectory schema.
