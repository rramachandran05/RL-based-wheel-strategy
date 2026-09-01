# SPEC-002 — Data Layer: Adapters, Canonical Tables, Reuse Map, Gap Resolutions

_Status: draft v1 — 2026-08-21_

## 1. Principles

- Every quantity must be computable **as-of a historical date** using only information available then (walk-forward integrity).
- All external pulls land in `data_local/` as versioned snapshots; training never hits a live API.
- Reused `wheel-strategy` modules are **vendored** (copied into `rlbot/vendor/` with a provenance header naming source path + commit/date), not imported across repos — the sibling is a separate git repo pinned to Python 3.8.

## 2. Canonical tables (parquet in `data_local/canonical/`)

| Table | Grain | Columns |
|---|---|---|
| `market` | date | spy_close, spy_ret_1d, spy_sma50, spy_sma100, spy_sma200, spy_sma50_slope, spy_drawdown, spy_realized_vol_20, vix_close, vix_pct_5y, vrp, breadth_proxy (nullable), market_regime |
| `underlying` | date × ticker | open, high, low, close, volume, ret_1d/5d/20d, drawdown, sma50/100/200, rsi14, adx14, di_plus, di_minus, atr20, realized_vol_30, structure, momentum, trend_bucket, momentum_bucket |
| `valuation` | date × ticker | fv_buy, fv_sell, fmp_median, source, confidence, fv_dist, valuation_state |
| `options_chain` | snapshot_date × ticker × expiration × strike × cp | bid, ask, mid, iv, delta, gamma, theta, vega, oi, volume — **empty until historical chains acquired; schema fixed now** |
| `events` | ticker × date | event_type ∈ {earnings, dividend, ex_dividend, split, other} |
| `external` | date | put_index_close (CBOE PUT), any other benchmark series |

## 3. Adapters and feature builders

### 3.1 Bars — reuse `vendor/data_access.py` as-is
Tiingo daily-adjusted OHLCV → `stockData/`-style CSV cache → canonical `underlying`. Extend lookback: `years=20` for SPY, `years=15` for the ticker universe (Tiingo supports it; the sibling's cache only holds 5y — re-download, don't copy).

### 3.2 Per-day feature series — new module `rlbot/features/technicals_series.py`
The vendored `technicals.compute_context` returns **latest-snapshot only**; RL training needs per-day values. This module calls the vendored indicator functions (`sma`, `atr`, `rsi`, `adx` — they already return full Series) and materializes the `underlying` table for all dates. `classify_structure`/`classify_momentum` are re-expressed vectorized with **golden tests asserting exact agreement with the vendored scalar versions** on ≥ 200 sampled (ticker, date) points.

Trend bucket mapping (logged feature): Bull Trend→4, Recovery→3, Base→2, Pullback in Uptrend→1, Breakdown→0. Momentum: Overextended→4, Extended/Building→3, Neutral/Mixed→2, Weakening→1 (5th level reserved).

### 3.3 Market regime — new module `rlbot/features/regime.py`
Implements SPEC-001 §3.1 exactly. Inputs: SPY bars (Tiingo) + VIX daily closes. **VIX source: FRED `VIXCLS` series (free CSV, no key)**, snapshotted to `data_local/external/vixcls.csv`. `vix_pct_5y` = rolling 1260-trading-day percentile, right-inclusive, min 252 observations. The sibling's `market_regime.py` (manual Fear & Greed input) is **not** reused for training; its 5-band posture mapping may inform the live assistant later.

### 3.4 Valuation — adapter over the sibling's FV signal

> **2026-08-31 note:** everything below concerns the Q-state's frozen
> 3-level `valuation_state` axis (sheet anchors + EPS proxy) — still
> current. The separate **valuation-gate feed** has moved twice: sheet FV →
> fair-value-discount ensemble (SPEC-009) → `../mcb-wheel` MCB report
> (SPEC-011, live). MCB ingestion: `rlbot/data/mcb_feed.py` reading
> `DataConfig.mcb_dir` (`../mcb-wheel/outputs/mcb_<date>.csv`), PIT-safe,
> 5-trading-session staleness. Bars/chains/EPS are Alpha Vantage premium
> primary with Tiingo fallback (2026-08-29), superseding the Tiingo-primary
> wording elsewhere in this spec.
- **Live/recent:** ingest the dated `fair_value_*.csv` snapshots (`Ticker, Price, FV_Buy, FV_Sell, …, Source, Confidence`) produced by `../wheel-strategy` (13 snapshots exist, 2026-07-26 →). New snapshots keep accruing from the sibling's daily run; an import job copies them into `canonical.valuation`.
- **Historical era:** no valuation series exists. Per SPEC-001 §3.1, `valuation_state = FAIR` when FV unknown. This is a declared limitation: in the training era the valuation dimension is inert, and the Q-table's valuation axis only differentiates once live-era data accrues (or a historical FV series is acquired — e.g. FMP historical price-target consensus, a paid-tier item, tracked as **DATA-GAP-3**).
- The pure anchor math (`fv_anchors`, `buy_sell_levels`, `etf_fv_proxy` from `fv_levels.py`) is vendored for the live path; the Google-Sheet fetch is not part of this project.

### 3.5 Vol compensation — `rlbot/features/vol_comp.py`
Market-level proxy per SPEC-001 §3.1: `vrp = vix_close − 100·spy_realized_vol_20` (both as vol points), plus `vix_pct_5y`. When `options_chain` is populated, per-ticker `iv_pctile` and `iv − rv30` replace the proxy behind the same enum (config switch `vol_comp_source`).

### 3.6 Events — Alpha Vantage forward calendar (vendored `earnings_calendar.py` pattern) for live use. **Historical earnings dates are absent (DATA-GAP-2)**; MVP resolution: `event_risk=false` in the historical era, logged as such, and the risk engine's earnings blackout applies live-only until a historical source is added. Note: the sibling's cached AV pulls are currently returning empty — verify the feed before relying on it.

### 3.7 Options chains — `PremiumSource` implementations (consumed by SPEC-003/004)
- `SyntheticBSPremiumSource` (day one): wraps vendored `options_engine` — `bs_put_price`/`bs_call_price` on `realized_vol_30`, r=0; BS analytic delta (from d1) so delta-tier targeting works without chain data; synthetic "chain" = strike grid at $0.5/$1/$2.5/$5 increments (per spot magnitude) × listed monthly expirations approximated as every Friday in the 25–45 DTE window. Spread model for friction: `spread = max(0.02, spread_pct · premium)` with `spread_pct` calibrated in SPEC-003 §7.
- `HistoricalChainPremiumSource` (when acquired): reads `canonical.options_chain`. Vendor decision (Theta Data vs ORATS) is **DATA-GAP-1** — a purchase decision for Rahul; schema is already fixed so ingestion is mechanical.

### 3.8 PUT index — manual CSV download from CBOE's website into `data_local/external/put_index.csv` (date, close). Loader validates coverage of the calibration window (SPEC-003 §7).

## 4. Reuse map (from the 2026-08-21 code review of `../wheel-strategy`)

| Sibling module | Verdict | Use here |
|---|---|---|
| `technicals.py` | vendor as-is | indicator functions + scalar classifiers (golden reference for §3.2) |
| `options_engine.py` | vendor as-is | synthetic premium source: BS prices, `put/call_assignment_probability`, `expected_put_payout`, `realized_volatility_proxy` |
| `data_access.py` | vendor as-is | Tiingo download + CSV cache (`TIINGO_API_KEY`) |
| `portfolio_risk.py` | vendor + structured-output wrapper | risk engine inputs (SPEC-004 §4): concentration, correlation clusters, expiry-week clustering — wrapper returns numbers, not prose flags |
| `wheel_backtest.py` | pattern seed, not vendored | `simulate_wheel` informs SPEC-003 step loop; friction constants (3% premium slippage, $0.65/contract) adopted as defaults; `optimize_wheel_mult` re-implemented as Baseline 5 |
| `fv_levels.py` | vendor pure functions only | anchor math for live valuation path |
| `daily_brief.analyze_ticker` | pattern seed | observation assembly informs `features/` + assistant payload; rendering not reused |
| `market_regime.py` | not reused for training | manual F&G input; posture mapping noted for assistant |
| `position_monitor.py`, `sheet_data.py` | not reused v1 | personal-sheet workflow; `compute_mos`/`should_roll` noted as MVP-3 fixed-rule candidates |
| `config.py` | pattern only | own `rlbot/config.py` dataclass tree; adopt ticker universe + thresholds as defaults |

Default universe (from sibling config): stocks AAPL, AMZN, BRK-B, GOOGL, TSM, MA, META, MSFT, NOW, NVDA, UNH, V, WMT, CHPS; ETFs MCHI, QTUM, SMH, TQQQ, SPXL, SPYI, QQQI. **Training universe v1 = SPY + the 10 most liquid stocks** (AAPL, MSFT, NVDA, AMZN, GOOGL, META, V, MA, WMT, UNH); leveraged ETFs excluded from training (survivorship/path-dependence hazards), permitted live behind the suitability gate.

## 5. Declared data gaps

| ID | Gap | MVP resolution | Real resolution |
|---|---|---|---|
| DATA-GAP-1 | No historical option chains/IV | Synthetic-BS track + PUT-index calibration gate | **CLOSED 2026-08-22: Alpha Vantage Premium purchased. `HISTORICAL_OPTIONS` (chains + IV + greeks, coverage to 2008) backfilling 2012→present for the 10 training tickers + TQQQ via `rlbot/data/options_ingest.py` (parallel, rate-adaptive, resumable per ticker-year) into `data_local/chains/`. `HistoricalChainPremiumSource` built; selector liquidity floors active on real quotes. Unblocks: real-premium B3 absolutes, per-ticker G1 recalibration, G3 retest with real IV dynamics.** |
| DATA-GAP-2 | No historical earnings dates | event_risk=false historically; blackout live-only | Historical earnings source (e.g. FMP/AV historical endpoints) |
| DATA-GAP-3 | No historical valuation series | valuation_state=FAIR historically (inert axis) | FMP historical price-target consensus or manual Morningstar backfill |
| DATA-GAP-4 | No historical Fear & Greed | Not used; regime is computed from SPY+VIX (reproducible) | n/a — proxy is permanent by design |
| DATA-GAP-5 | Survivorship: universe is all survivors | Scope claim: policy is conditional on the "willing to own" screen; stated in every evaluation report | Add delisted/cratered names when chain data purchased |

## 6. Requirements

- **REQ-2.1** Every canonical table builder takes an `as_of` date and provably uses no later rows (tested by truncation-equivalence: building with data truncated at `as_of` equals rows ≤ `as_of` of the full build).
- **REQ-2.2** Vendored files are byte-identical to source except an added provenance header; a test hashes them against recorded checksums.
- **REQ-2.3** Vectorized structure/momentum classifiers agree exactly with vendored scalar versions (golden test, ≥ 200 samples).
- **REQ-2.4** All adapters degrade gracefully (missing file/API → explicit `DataUnavailable`, never silent NaN propagation).
- **REQ-2.5** A single `build_all(config)` entry point materializes every canonical table deterministically from `data_local/` snapshots.

## 7. Acceptance criteria

- **AC-1** `market` table spans ≥ 2008-01-01 → present with no gaps > 5 trading days; regime column takes all 4 values over the span; regime series for 2008-Q4, 2020-03, 2022-H1 lands in BEAR_STRESS (sanity anchors).
- **AC-2** Truncation-equivalence test passes for `market`, `underlying`, `valuation` builders (REQ-2.1).
- **AC-3** Golden agreement test (REQ-2.3) passes.
- **AC-4** `options_chain` schema round-trips an empty and a synthetic fixture parquet.
- **AC-5** Checksum test (REQ-2.2) passes for all vendored modules.
