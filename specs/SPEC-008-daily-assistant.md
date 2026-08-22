# SPEC-008 — Daily Assistant and Live Trajectory Logging (MVP-4)

_Status: draft v1 — 2026-08-22. Productionizes the twice-validated rule policy (SPEC-000 status): B3 openings + hold-to-expiry + selector + risk engine._

## 1. Scope

One command turns the research system into a daily decision aid:

```
python -m rlbot.assistant.daily [--download] [--cash 100000] \
       [--positions data_local/positions.csv]
```

1. `--download`: refresh Tiingo bars + FRED VIX, rebuild canonical tables (full re-pull; incremental update is a wish-list item).
2. Build frames with `use_valuation_proxy=True` (sheet FV snapshots still win where present, SPEC-007 §3.1).
3. **Opening recommendations** per universe ticker from the latest row: Q-state → B3 action → synthetic chain (G1-fitted `iv_uplift`) → selector → risk engine → recommendation object (SPEC-006 §6 shape).
4. **Position guidance** for open positions listed in `positions.csv` (columns: `ticker,type,strike,expiration,premium_fill` with type ∈ {CSP, CC}): management state, current model delta and premium-captured, and guidance = **HOLD** (the validated incumbent), with attention flags for breach, delta ≥ 0.40, and expiry week. The brief carries the standing G3 finding: do not roll mechanically on MOS.
5. Outputs: `data_local/live/brief_<date>.md` (human) + `recommendations_<date>.json` (machine), and every decision appended to `data_local/live/decisions.jsonl` as a `trajectory_v2` record with null outcome fields (outcomes attach when a future run or retest closes them).

## 1a. ETF coverage (added 2026-08-22)

The assistant universe is `tickers + etfs` (config: TQQQ, SPXL, CHPS, SPYI). ETFs are **assistant-only**: excluded from training, walk-forward, and the EPS proxy (no EPS → valuation defaults to FAIR). Unleveraged ETFs (CHPS, SPYI) use the standard B3 rules.

**Leveraged ETFs (TQQQ, SPXL) use a capped rule table** (`leveraged_cash_action`):

| Regime | Cash action | Stock action |
|---|---|---|
| BULL_LOW_VOL | PUT_BALANCED (cap) | CALL_BALANCED |
| BULL_HIGH_VOL / SIDEWAYS | PUT_CONSERVATIVE | CALL_AGGRESSIVE |
| BEAR_STRESS | **WAIT — no exceptions** (B3's attractive-vol-comp defensive put is removed) | CALL_AGGRESSIVE |

Rationale: assignment on a 3x fund is 3x market exposure into whatever regime caused it; the realized-vol premium proxy lags hardest exactly during vol spikes; and the high nominal premium the user targets is preserved anyway, because delta-targeting converts 3x vol into proportionally wider strikes. The brief marks these rows "(3x)" and carries a reduced-size note. **Untested caveat: these caps are reasoned, not gate-validated** — leveraged ETFs were excluded from all walk-forwards (path-dependence + survivorship hazards), so no G-series verdict covers them.

## 2. Operational caveats (printed in every brief)

- Premiums and deltas are **model values** (synthetic BS, calibrated to the PUT index at the index level). Always compare against live broker quotes before trading; if the live premium is materially below model, the compensation gate that justified the trade may not hold.
- Recommendations only — the system never executes. Not investment advice.
- Portfolio cash/NAV come from `--cash` (single-sleeve assumption); risk caps are per-ticker.
- Bars must be fresh: the brief warns if the latest bar is > 3 trading days old.

## 3. Requirements & acceptance

- **REQ-8.1** The assistant reuses the selector/risk/state modules verbatim (no reimplementation; import-graph is the proof).
- **REQ-8.2** Every emitted decision record validates against trajectory_v2.
- **REQ-8.3** Missing/malformed positions file degrades to an openings-only brief with a warning, never a crash.
- **REQ-8.4** Stale-data warning when the latest bar is old (test with a truncated frame).
- **AC-1** Smoke test: canned frame → recommendation object with contract in the action's delta band and risk_checks passed.
- **AC-2** Positions fixture (one safe CSP, one breached CC) → HOLD guidance with correct flags.
- **AC-3** A real run produces the three artifacts and `decisions.jsonl` lines that validate.
