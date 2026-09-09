# SPEC-007 — MVP-3 Rescope: Management-Policy Learning + Valuation Proxy

_Status: draft v1 — 2026-08-22. Replaces the original MVP-3 scope (HMM, GARCH, adaptive Q-blending, ICRL prep) — all deferred indefinitely per the G2 post-mortem._

## 1. Why the rescope

G2 failed (SPEC-000 §4 status): the learned *opening* policy conditioned on the same two live axes (regime, vol-comp) the rule baseline already uses — valuation was inert historically (DATA-GAP-3) — so it had no informational edge and matched the rules to within noise. The original MVP-3 additions would have increased model sophistication without adding information. This spec attacks the two root causes instead:

- **Track A (primary): management-policy learning.** The MVP holds every option to expiration — the weakest rule in the system. Management decisions have a rich, well-populated state (position geometry), a clean incumbent to beat (HOLD), and need **no new data**.
- **Track B (parallel): historical valuation proxy.** Revive the dead valuation axis within the Tiingo + Alpha Vantage constraint, then re-run the MVP-2 opening-policy pipeline against G2 once the axis carries signal.

## 2. Track A — learned management policies

Interface: SPEC-001A (MgmtStateV1, action mechanics, M-epochs, `diff_v2` reward, `trajectory_v2`).

### 2.1 Simulator extension
- `WheelEnv` gains management epochs (SPEC-001A §4) and executes CLOSE/ROLL/commit actions (SPEC-001A §3). Opening behavior, G1 calibration, and all `trajectory_v1` outputs are unchanged (regression-tested).
- Buy-to-close friction mirrors sell-to-open: `fill = mark × (1 + slippage_pct)` + commission.

### 2.2 Counterfactual sweep
Same engine as SPEC-005 with management branches: at each M-epoch in the training window, branch every legal management action (minus ROLL_HIGHER_RISK, SPEC-001A §3) to the branch's own next M-epoch or resolution; target = `diff_v2` reward + rule-continuation value. Opening decisions along the path follow **Baseline 3** (frozen — Track A learns management only, so credit assignment is clean).

### 2.3 Estimation and policy
Identical machinery to SPEC-005 §3–5: per-(mgmt_state, action) regression, cluster-based `n_eff`, A/B halves, shrinkage toward the rule prior, LCB action selection, learning/production table split.

**Rule prior / incumbent baseline (M-B1):** HOLD always (the MVP-2 behavior). **Comparison baseline (M-B2):** the sibling project's mechanical roll rule — roll when MOS crosses ±3% (`position_monitor.should_roll` logic, vendored reference) — included so the learned policy is judged against both "do nothing" and "the obvious rule."

### 2.4 GATE G3 — falsifiable claim of Track A
On walk-forward test windows (same folds as SPEC-006 §4), the learned management policy (with B3 openings) vs. hold-to-expiry (with B3 openings):

- Pooled differential annualized return > 0, and > 0 in ≥ half the folds
- Max drawdown ≤ 1.0× the incumbent's on each test window (management's job is risk control — it must not add drawdown)
- CVaR(5%) of monthly returns no worse than the incumbent's
- A/B halves individually non-negative

Failing G3 is again a legitimate outcome: it would say hold-to-expiry is adequate and management adds nothing — worth knowing, and cheap.

## 3. Track B — valuation proxy from Alpha Vantage fundamentals

### 3.1 Construction (DATA-GAP-3 partial closure)
- Source: Alpha Vantage `EARNINGS` endpoint (quarterly EPS history, free tier; ~10 requests total, well under the daily cap). Snapshot to `data_local/external/eps/{ticker}.csv`.
- Per ticker, per day (causal): `eps_ttm(t)` = sum of the last 4 quarterly EPS **reported** on or before t (use `reportedDate`, not fiscal quarter end — walk-forward integrity); `pe(t) = close / eps_ttm`; `pe_pct(t)` = rolling 5-year percentile of `pe` (right-inclusive, min 252 obs).
- Mapping to the existing enum (no schema change): `pe_pct < 0.20 → ATTRACTIVE`, `> 0.80 → EXPENSIVE`, else `FAIR`. Negative/zero `eps_ttm` → FAIR (unknown).
- This is a *relative-to-own-history* proxy, not intrinsic value: documented limitation. Live-era sheet-based FV snapshots override the proxy when present (fresher, better).

### 3.2 GATE G2-rerun
Re-run the unmodified SPEC-005/006 opening-policy pipeline with the proxy-populated valuation axis. Pass criteria identical to G2. Prediction to test: state coverage rises from 10 toward ~30 of 36 cells.

## 3C. Track C — drawdown-percentile valuation proxy (added 2026-09-09; APPROACH.md §5)

### 3C.1 Motivation
Track B (§3) narrowed DATA-GAP-3 but still leaves the valuation axis
dependent on fundamentals that are absent or unreliable for ETFs and
partially covered for stocks. Per-ticker drawdown from the running high
(`underlying.drawdown`, SPEC-002 §2) exists for **every ticker on every day
back to 2013 — zero historical gap** — and is the price-position signal
the trend-axis ablation only approximated through SMA-crossover structure.
The trend ablation was the worst of the three axis variants (−0.74%/yr
pooled), so a related price-position idea is **not assumed to succeed**;
it earns its way in through the same gate or it does not ship.

### 3C.2 Construction
- Input: `drawdown = close / cummax(close) − 1` (causal, already materialized).
- `dd_pct(t)` = right-inclusive rolling 5-year percentile (1260 bars, min
  252 obs) of the ticker's **own** drawdown series — not a fixed % band,
  since a stock that routinely swings ±30% needs different bands than one
  that rarely moves. Mirrors Track B's percentile approach exactly.
- Mapping to the frozen enum (no SPEC-001 change): `dd_pct < 0.20 →
  ATTRACTIVE` (drawdown in the worst fifth of its own history — a deep
  correction), `dd_pct > 0.80 → EXPENSIVE` (at/near highs), else `FAIR`.
  Warmup (< 252 obs) → NA → state undefined, as today.
- Implementation: `rlbot/features/valuation.py::classify_drawdown_series`;
  ablation harness `rlbot/evaluation/ablation_drawdown.py` (a FrameStore
  wrapper overwriting `valuation_state`, byte-identical to
  `ablation_trend.py` in every other respect).

### 3C.3 GATE G9 — drawdown axis
Byte-identical B3 pipeline, same folds, same estimator/LCB/promotion
criteria as G2-rerun and the trend ablation; only the valuation axis is
swapped. Verdict JSON: `data_local/reports/ablation_drawdown_verdict.json`.
Pass criteria identical to G2. **Pre-registered failure mode to inspect
regardless of verdict:** does the learned table buy into names that keep
falling (deep-drawdown ATTRACTIVE → aggressive tier → further decline)?
Segment test-period losses by whether the entry state was
drawdown-ATTRACTIVE. Fail ⇒ the axis stays logged-only; pass ⇒ it becomes
a candidate for the live Q-state only after a separate review of the
value-trap risk, since "cheaper than it was" is not "cheaper than it is worth".

**G9 VERDICT (run 2026-09-09): FAILED.** Pooled test differential vs B3
**−0.26%/yr** (F1 −0.39%, F2 −0.13%); both return criteria missed; all
three risk criteria passed (drawdown ratio ≤ 1.1×, no regime segment
lagging 2 pts, A/B halves non-negative) — the same "indistinguishable to
slightly worse, not riskier" shape as every prior axis. Coverage 28/36 (F1)
and 29/36 (F2): zero historical gap in the *signal*, but the empty cells
are rare regime × vol-comp combinations, so the axis does not populate more
of the table than the EPS proxy did (30/36). Ranking of the four axis
variants by pooled test differential: EPS proxy −0.04% > original FV axis
−0.21% > **drawdown −0.26%** > trend −0.74%.

**Pre-registered value-trap check: the failure mode appeared exactly where
predicted.** In the F1 test window (2022–23, the bear), drawdown-ATTRACTIVE
entries were the *worst* bucket — n=277, mean differential reward
−0.0022/decision, 70% of decisions lost — versus FAIR −0.00013 and
EXPENSIVE −0.00004. In F2 (2024–26) the same bucket was positive (+0.0019,
n=53). Read: buying the deep correction works in a recovery and buys falling
knives in a bear. "Cheaper than it was" is not "cheaper than it is worth."
**Disposition: the axis stays logged-only; it is not a live Q-state
candidate.** The opening-decision question is now settled four times in
favor of the rule table. (Note for SPEC-011 rule 6, which also reads
drawdown: there it is a *score* input bounded by the producer's own dd50/dd90
history, and its relief notch is already withheld in BEAR_STRESS — the fold
where this check failed. Unaffected by this verdict, but the same caution
applies.)

## 4. Explicitly deferred (from original MVP-3)
HMM regime model, GARCH volatility, adaptive Q-blending, trend/momentum state promotion, ICRL. Reconsider only after G3 and G2-rerun verdicts exist.

## 5. Requirements

- **REQ-7.1** Management extension changes no `trajectory_v1` output: byte-identical baseline-run regression test vs. the MVP-2 pipeline.
- **REQ-7.2** Management sweep HOLD branch reward ≡ 0 (SPEC-001A AC-4) enforced in the sweep engine, not just tested.
- **REQ-7.3** Track B builder is causal (truncation-equivalence test on `pe_pct`) and uses `reportedDate` for availability.
- **REQ-7.4** G3 and G2-rerun each produce a verdict JSON + markdown report via the SPEC-006 §6 renderer.
- **REQ-7.5** The learning CLI for Track A refuses to run without a passing, current `put_gate.json` (G1 remains the foundation gate).

## 6. Acceptance criteria

- **AC-1** SPEC-001A AC-1..5 all pass.
- **AC-2** Regression: MVP-2 walk-forward reproduces its committed G2 verdict bytes on the same data snapshot after the simulator extension lands.
- **AC-3** M-B2 (MOS-roll rule) runs end-to-end and its assignment rate differs measurably from M-B1 (sanity that management actions actually engage).
- **AC-4** Track B: EPS snapshots exist for all 10 tickers; valuation axis populates ≥ 25 of 36 states in the F2 training window.
- **AC-5** G3 verdict and G2-rerun verdict committed with reports, whatever their outcome.

## 7. Order of work

1. SPEC-001A mechanics in the simulator + trajectory_v2 (REQ-7.1 regression first)
2. M-epoch sweep + M-B1/M-B2 baselines
3. G3 walk-forward run → verdict
4. Track B EPS snapshots + proxy builder (can proceed in parallel after step 1)
5. G2-rerun → verdict
6. SPEC-000 status update with both verdicts
