# Wheel-Strategy RL Bot — Overall Approach

_Revision 2 — 2026-08-21. Incorporates the review-round changes: counterfactual sweep training, differential reward, hard state-space reduction with pessimistic offline learning, data reality checks, and a risk-reordered MVP. Supersedes the v1 approach text._
_Revision 3 — 2026-09-09. Plain-language pass; MCB v2 (delta_posture aggressiveness pivot) replaces the FV-sheet valuation signal in the selector's score; current-status honesty note added._

**A wheel-trading bot that, given market conditions, learns how much risk to take while maximizing premium — while fixed rules handle contract selection and safety.**

> **Current status (see SPEC-000 for the full record):** every learning gate run to date (G2, G2-rerun, the trend-axis ablation, G3, G3-retest) has failed to beat the rule-based baseline out-of-sample. The system described below as the target design is **not what's running in production today** — the live daily assistant is 100% rule-based (Baseline 3 + deterministic selector + hard risk engine), with zero learned Q-tables deployed. Learning is paused, not abandoned: MVP-4 stays open pending a materially new signal (real-IV dynamics, a validated state axis, etc.), and every trajectory is still logged so a future retry starts from real data. Treat §§8–10 below as the design this repo is built to support, not a status report.

---

## 1. Philosophy

**The bot asks: "Is the premium worth the risk, and how much risk should I accept?"** When conditions look favorable, it can sell options with a greater chance of assignment in exchange for more premium. When conditions look worse, it becomes more conservative — or does nothing.

The key separation:
- **The learning model chooses a risk level.**
- **A fixed algorithm chooses the actual option contract.**
- **A risk engine decides whether the trade is allowed.**

The model cannot freely invent trades or override limits.

```
Keep the Wheel working when the market is paying enough for the risk,
but dynamically change how much assignment or call-away risk you accept.
Better conditions → more aggressive strikes, more premium.
Worse conditions → progressively safer strikes, less premium.
Extreme conditions → wait.
```

---

## 2. The Wheel as an Explicit State Machine

Built before any RL. The wheel moves through four stages:

1. **Cash** — you have money available and can sell a cash-secured put.
2. **Short put** — you have promised to buy shares at the strike if assigned.
3. **Long stock** — assignment has left you owning shares; you can sell a covered call.
4. **Covered call** — you own shares and have promised to sell them at the strike if assigned.

Each state has its own permitted actions — this prevents the agent from ever proposing a mechanically invalid move.

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

At each **decision epoch** (§8 — not every day), the bot first checks what it owns and whether the stock is still suitable, then evaluates the market, valuation, option premiums, and upcoming risks. Those checks lead to a risk posture and a possible trade.

Four ideas matter most:
- **A high chance of profit is insufficient** — a small premium can still come with a large potential loss.
- **Rolling is not automatic** — replacing an existing option with another must improve the situation, not merely postpone recognizing a loss.
- **Covered calls have an opportunity cost** — collecting premium can mean missing a large stock rally.
- **Each completed cycle requires a fresh decision** — selling the shares does not automatically trigger another put.

The full ordered checklist:

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
Data → assessment of conditions → risk choice → contract selection
     → safety checks → explanation → outcome tracking
```

A **Q-function** is a scorecard estimating how useful an action is in a particular situation. The design uses separate scorecards for cash, open puts, owned stock, and open calls, because those situations involve different decisions:

- `Q_cash(state, action)` — cash policy
- `Q_put(state, action)` — short-put management
- `Q_stock(state, action)` — stock policy
- `Q_call(state, action)` — covered-call management

The MVP trains only `Q_cash` and `Q_stock`; put/call management uses fixed rules until Phase 2. **As of this revision, none of these tables are live in production** (see the status note at the top) — the rule-based B3 policy plays every role.

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

**Candidate axis — run and FAILED (G9, 2026-09-09; SPEC-007 §3C):** `valuation_state` is inert for nearly all of training history (no historical FV series exists; see DATA-GAP-3, §12), defaulting to FAIR almost everywhere. A drawdown-percentile proxy is a candidate replacement that has zero historical gap: `classify_drawdown_series`, thresholds set by the *rolling percentile of the ticker's own drawdown distribution* (not a fixed %, since a stock that routinely swings ±30% needs different bands than one that rarely moves) — mirroring the EPS-proxy's percentile approach rather than the fixed 5% band. To be tested via the same ablation protocol as G2-rerun and the trend-axis ablation: byte-identical B3 pipeline, same test windows, only the valuation axis swapped. **Result: it did not clear the gate** — pooled test −0.26%/yr vs the rule table, and the pre-registered value-trap check fired in the 2022–23 bear fold (deep-drawdown ATTRACTIVE entries were the worst bucket, 70% losers) while the same bucket was positive in 2024–26. "Cheaper than it was" is not "cheaper than it is worth." The axis stays logged-only; four valuation-axis variants have now lost to the rule table (EPS proxy −0.04% > FV −0.21% > drawdown −0.26% > trend −0.74%).

Management states (Phase 2) add position fields: DTE, current delta, distance to strike, % premium captured, unrealized P&L.

Categorical encodings, not raw numbers: never feed $103.72 fair value or a raw SMA into a Q-table.

---

## 6. Action Space — Risk Budgets, Not Contracts

Instead of choosing among thousands of options, the model chooses a category: **Wait → defensive → conservative → balanced → aggressive.**

For puts, those categories map to delta ranges — delta acts as a rough indicator of exposure and assignment likelihood, not an exact assignment probability. The put ranges use delta's magnitude:

| Cash action | Approx. put delta |
|---|---|
| WAIT | 0 |
| PUT_DEFENSIVE | 0.05–0.10 |
| PUT_CONSERVATIVE | 0.10–0.18 |
| PUT_BALANCED | 0.18–0.25 |
| PUT_AGGRESSIVE | 0.25–0.35 |
| PUT_VERY_AGGRESSIVE | 0.35–0.45 |

For covered calls, the categories determine how close the strike is to the current stock price — farther away generally preserves more upside; closer generally collects more premium but makes selling the shares more likely. "Defensive" on the call side mainly means less call-away risk; it does not mean strong protection against a stock-price decline.

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

Once the model chooses a risk category, a conventional algorithm finds the best qualifying option: filter for expiration, delta, liquidity, and trading costs, then score the remaining contracts. In live operation the filter is also **book-aware**: expiries whose ISO week is already at the portfolio's weekly assignment cap are skipped up front (the risk engine would reject them anyway), so a capped week never hides a viable sibling expiry.

```
Score = w1·PremiumYield + w2·VolatilityPremium
      − w3·SpreadCost − w4·DownsideRisk − w5·AssignmentPenalty
```

The scoring weights are design choices that still need to be specified and tested.

**Valuation input to the score (revised 2026-09-09 — the Google Sheet FV anchor is retired from this role): the assignment penalty is driven by MCB position and drawdown severity, not the sheet's fair-value distance.** If the stock sits near or below its Maximum Comfortable Basis (`wheel_entry`), assignment receives a smaller penalty — you're being paid to acquire something you already consider fairly priced. If it sits far above `wheel_entry` with no correction underway, the same assignment exposure receives a larger penalty. The exact mapping (how `drop_needed` / `dd_now` combine into a multiplier) is a design detail still to be specified and tested — same status as the scoring weights above.

**MCB acquisition ceiling — an aggressiveness pivot, not a uniform rejection boundary (mcb-wheel producer contract v2, 2026-09-09).** The old rule ("no put may ever be sold above the MCB ceiling, regardless of tier") is **wrong for a trending, compounding stock** — a stock can be safely sold at conservative delta far above its comfortable acquisition price, because assignment there is a low-probability tail event, not the trade's purpose. The producer now publishes `delta_posture` per ticker (CONSERVATIVE / MODERATE / HIGHER), computed from the stock's own valuation and drawdown distance:

| `delta_posture` | Situation | MCB ceiling treatment |
|---|---|---|
| CONSERVATIVE | Far above entry, rising or sideways | Advisory only — never blocks. Conservative-delta puts with net basis above the ceiling are legitimate here: assignment is neither sought nor likely. |
| MODERATE | Entry reachable through a typical correction | Hard for our BALANCED-and-up tiers (assignment-seeking); advisory for WAIT/DEFENSIVE/CONSERVATIVE tiers (income-only) — the one case the producer leaves to us to resolve. |
| HIGHER | At/near entry, or at its typical correction level, stable or rising | Hard, always — assignment is welcome and actively sought here. |

**One override belongs to us, not the producer: a strong downtrend forces conservative-or-wait regardless of posture** ("don't catch a falling knife"). The producer computes no trend signal; this repo overlays its own (`classify_structure` — Bull Trend / Recovery / Base / Pullback / Breakdown, already computed and logged per §5, previously unused for this purpose).

---

## 8. Decision Epochs — a Semi-MDP

No pointless daily trades. The bot makes decisions at meaningful events rather than trading simply because another day has passed:

- no option currently open
- **an option approaches expiration, or a profit target is reached** — two distinct triggers, not one
- delta changes materially / underlying crosses strike
- regime or volatility-regime change
- earnings approaches

"Semi-MDP" means the time between decisions can vary. Between epochs the position simply rides; rewards are aggregated over the inter-decision window. **Within a cycle, later rewards get the same weight as earlier ones (γ = 1)** — with 20–45-day horizons, per-day discounting buys nothing and adds a tuning knob to avoid.

---

## 9. Reward — Differential, Not Raw NAV

**NAV** means the portfolio's total net value. Raw NAV-change reward in LONG_STOCK is dominated by the stock's drift: the table would learn "stocks go up in bull regimes" — true and useless. The bot is instead rewarded for doing better than a named reference strategy over the same window:

```
Reward = bot's change in value − reference strategy's change in value
```

- **Reference in cash states: a fixed 20-delta wheel.**
- **Reference in stock states: buy-and-hold.**

Examples:
- Bot gains $800; reference gains $500 → reward is +$300.
- Bot gains $800; reference gains $1,200 → reward is −$400.

This isolates the decision's contribution to wealth, and covered-call opportunity cost falls out automatically — a capped rally shows up as negative differential reward with no special accounting. The evaluation baselines (§14) serve double duty as reward references. Costs (commissions, slippage) are charged inside each leg's NAV path. Drawdown/tail-risk penalty terms are deferred until the simple differential reward is demonstrably insufficient — complex rewards are easy to optimize incorrectly.

---

## 10. Learning — Counterfactual Sweep First, Q-Learning Second

**The central idea: the environment is exogenous — our actions never move option prices or the underlying — so at every historical decision point we can simulate every available action against the same path and compare the outcomes,** instead of picking one and observing only its result. This converts bandit feedback into full-information feedback and eliminates the offline exploration/exploitation problem entirely.

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

Because different actions branch portfolio state (assigned vs. not), each branch is simulated **only to the next decision epoch**, then closed with a baseline continuation value (fixed-rule wheel from the branch's end state) instead of expanding a full tree. With full action feedback, learning largely collapses into per-state regression over observed action returns — simpler and more stable than TD bootstrapping.

Guards against offline-RL failure modes:

- **Double Q-learning** wherever a max operator appears (tabular — trivial to add): max + noisy financial rewards otherwise guarantees optimistic bias.
- **Act on a lower confidence bound** of Q, not the point estimate: optimistic offline RL reliably exploits estimation noise; pessimism is the standard fix.
- **Initialize from and shrink toward the rule baseline** (Baseline 3, §14), in proportion to correlation-adjusted effective sample size. Honest framing: the learned policy is a data-driven perturbation of the rule table, not a from-scratch discovery.

Per Q-entry bookkeeping: Q value, observation count, average return, return variance, last-updated — with the caveat that raw N overstates evidence (correlated episodes); the LCB uses effective sample size.

**Semi-MDP update** (when bootstrapping is used at all): `Q(s,a) ← Q(s,a) + α[R + γ^Δt · max_a' Q(s',a') − Q(s,a)]` with R the accumulated inter-decision reward and Δt the days to the next decision.

**Production vs. learning policy:** live experience updates a learning table only; promotion to the production table requires passing performance, drawdown, regime, and risk-limit tests in backtest. One unusual month must not rewrite the strategy.

---

## 11. Hard Risk Engine — Not Learnable, Not Bypassable

These are rules the model cannot override: sufficient cash for put assignment, owning shares before selling calls, position limits, acceptable liquidity, and event restrictions.

- **Spread exposure across tickers, strikes, and expiration dates** — no more than **15% of NAV** in potential exposure to a single underlying (shares plus open puts), no more than 12 distinct active names, no more than 15% of NAV in put escrow expiring in any one ISO week.
- **MCB acquisition ceiling remains hard for acquisition-intent situations** (producer `delta_posture = HIGHER`, or our own BALANCED-and-up tiers within MODERATE) — a non-bypassable limit alongside the exposure caps above. It is deliberately **not** universal; see §7 for the full aggressiveness-pivot split and why a uniform ceiling was wrong.

```
RL recommends → Risk engine validates → allowed: contract recommendation
                                      → rejected: step down risk tier, or WAIT
```

---

## 12. Data Plan and Reality Checks

Canonical daily historical snapshots sufficient to recreate what would have been known on each date: market table (SPY, SMAs, VIX, breadth, realized vol), underlying table (OHLCV, returns, drawdown, SMAs, RSI, momentum, realized vol), valuation table, historical options-chain table (per snapshot date × expiration × strike: bid/ask/mid, IV, greeks, OI, volume), events table, portfolio-state table.

**Alpha Vantage premium is the primary data vendor** (bars, historical + daily option chains, quarterly EPS), with Tiingo as fallback — closing what was originally the project's single biggest data gap.

| Item | Issue | Status |
|---|---|---|
| Historical option chains | The critical path and main cost | **Closed 2026-08-22:** Alpha Vantage Premium purchased; 34M rows backfilled 2012→present, 11 tickers + leveraged-ETF era |
| Fear & Greed history | No long official CNN history | Reproducible proxy composite (VIX percentile, put/call ratio, breadth) |
| No historical valuation series | Analyst-consensus FV only exists from 2026-07-26 forward | `valuation_state = FAIR` historically (declared limitation, §5); drawdown-percentile proxy queued as a candidate replacement |
| No historical earnings dates | — | Estimated live-only (last reported + ~91d, ±5d), surfaced as a human-review warning, never a blackout |
| Survivorship bias | Training only on survivors overstates aggressive-put value | Scoped explicitly as conditional on the "willing to own" screen; stated in every evaluation report |
| Simulator correctness | How do we know the simulator itself is right? | **Calibration gate:** fixed-delta SPY wheel vs. the CBOE PUT index — required before any learning begins |
| HMM / GARCH regimes | Complexity before it's earned | Deferred; rule-based regimes and IV-percentile/realized-vol spread cover the MVP |

---

## 13. Simulator

The simulator recreates trading over historical periods. It must include:

- **Realistic execution costs** — never assume mid fills: `Fill = Mid − k·(Ask−Bid)` when selling, reversed when buying to close, plus commissions and fees.
- **Daily mark-to-market** — `NAV = cash + shares·price − short_option_liability`. A losing put must show up as a loss while it is open, rather than disappearing from the results until assignment.
- **Assignment and expiration mechanics** — put ITM at expiry → assign (cash −= strike×100, shares += 100, cost basis = strike − premium received, SHORT_PUT → LONG_STOCK); call ITM → called away (COVERED_CALL → CASH). Expiration-only assignment for the first version; early exercise, ex-dividend effects, and corporate actions come later.
- **Cash and share accounting**, start to finish.

Testing moves forward chronologically only — walk-forward, never a random split. Every computed quantity (valuation, IV percentile, SMAs, regimes, earnings knowledge) uses only information available at that historical date.

---

## 14. Baselines and Evaluation

The learned strategy must compete against four simpler alternatives:

1. A fixed wheel.
2. A conservative wheel.
3. A wheel that adjusts risk using simple market rules.
4. Buy-and-hold.

The key test is: **if learning cannot reliably improve on simple adaptive rules, it is not worth the complexity.**

Evaluation includes growth, volatility, drawdowns, severe losses, trading costs, and capital usage — not just win rate. Results are also separated by market conditions so a strong bull-market result cannot hide poor crash performance.

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
