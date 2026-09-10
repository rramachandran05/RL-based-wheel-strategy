# Baseline 3 on Real Chains — Rebaseline and Return Decomposition

_Report date: 2026-09-10. Data: Alpha Vantage historical option chains 2012→present (34M rows), split/dividend-adjusted bars, 10-ticker training universe. Scripts: `rlbot/evaluation/b3_performance.py --historical` (commit d67c072 rerun), `rlbot/evaluation/b3_decompose.py` (commit 2ba52a0). Machine outputs: `data_local/reports/b3_performance_historical.json`, `data_local/reports/b3_decomposition_historical.json`._

_Recommendations only — not investment advice. All figures are simulation outputs on a survivor-biased universe (see §7)._

---

## 1. Executive summary

- **The headline B3 figure was stale, not wrong-by-drift.** A correct selector fix on 2026-08-30 (VolPremium/SpreadCost terms "actually wired" on real quotes, commit cd7a3cd) was never re-run against the real-chain backtest. Every "+29.0% CAGR / $100K→$3.19M" figure quoted between 08-30 and 09-09 was computed with the pre-fix, incomplete scoring formula. Rerun: **+10.5% CAGR / −26.5% max drawdown, $100K→$389K over 2013–2026.**
- **Nothing in the code broke.** Buy-and-hold is bit-for-bit identical between the old and new runs ($8,697,447 to the dollar in every window), proving price data, chain data and date windows are untouched; the shift is isolated to the wheel's contract-scoring formula. None of the MCB v2 / risk-engine v2 work reaches a single-ticker backtest.
- **The 10.5% is total return, roughly three-quarters premium and one-quarter stock gains.** Full window: net premium +$2.14M (74% of ΔNAV), realized stock P&L at call-away +$0.82M (28%), unrealized on still-held shares −$62K (−2%), residual −$12K. The identity reconciles to the headline within 0.1%.
- **The stock leg is where all the risk lives.** Premium alone cannot produce a 26% drawdown; that number is entirely assigned shares held through corrections. In the 2022–23 bear the stock leg roughly washes out and premium is ~97% of the return (~6%/yr simple).
- **B3's adaptive tiers did not beat a fixed 20-delta wheel (B1) over the full window** (+10.5% vs +13.2% CAGR; drawdown −26.5% vs −24.6%). In the two shorter test windows B3 holds a slightly better drawdown at slightly lower return — a wash. This is a legitimate, spec-anticipated outcome, not a failure of the software.

---

## 2. Part A — the rebaseline

### 2.1 What happened

| Date | Event |
|---|---|
| 2026-08-23 | Last real-chain B3 run written to `b3_performance_historical.json` (post dynamic-sizing fix): +29.0% / −42.8% / $3.19M |
| 2026-08-30 | Commit cd7a3cd (external-review fixes) makes `w_vol_premium` and `w_spread_cost` actually apply in `score_quote` for real quotes (`q.oi is not None`). Both weights had been declared in SPEC-004 §1.2 since MVP-1 but were dead code. Synthetic-track behavior unchanged, so all G-series verdicts stand. |
| 08-30 → 09-09 | Report never rerun; stale figures quoted in SPEC-000, memory, and conversation |
| 2026-09-09 | Rerun on request. Figures below. SPEC-000 status blocks annotated ⚠ Superseded; new standing rule: rerun after any real-quote scoring/selection change |

### 2.2 Why the numbers moved the way they did

The fix adds a **reward for IV above the realized-vol proxy** and a **penalty for quoted spread**. Two consequences, both visible in the data:

- **Drawdown improved in every window for both B3 and B1** — the spread-cost penalty steers away from illiquid, wide-spread contracts; the vol-premium reward favors strikes with genuinely elevated IV rather than the highest raw premium yield. Both reduce tail outcomes.
- **CAGR fell** — the same terms turn down some of the highest-yielding (and worst-executed) contracts. A coherent risk/return shift, not a bug.

### 2.3 Corrected figures (pooled = equal-weight average of ten $100K single-ticker sleeves, dynamic contract sizing)

| Window | Policy | CAGR | Max DD | Ann. vol | Sharpe | $100K → |
|---|---|---|---|---|---|---|
| **Full 2013–2026** (13.6y) | **B3** (adaptive rules) | **+10.5%** | **−26.5%** | 11.7% | 0.91 | **$388,695** |
| | B1 (fixed 20Δ wheel) | +13.2% | −24.6% | 12.8% | 1.03 | $537,944 |
| | Buy & hold | +38.8% | −49.5% | 30.4% | 1.23 | $8,697,447 |
| **Test-1 2022–23** (2.0y, bear) | **B3** | **+6.0%** | **−16.1%** | 13.6% | 0.50 | $112,263 |
| | B1 | +9.8% | −17.4% | 16.3% | 0.65 | $120,316 |
| | Buy & hold | +6.5% | −29.8% | 23.9% | 0.38 | $113,202 |
| **Test-2 2024–26** (2.6y) | **B3** | **+9.9%** | **−9.4%** | 8.8% | 1.12 | $128,040 |
| | B1 | +10.3% | −9.5% | 8.8% | 1.16 | $129,381 |
| | Buy & hold | +27.5% | −21.8% | 19.6% | 1.34 | $189,606 |

Superseded (pre-fix) B3 for reference: Full +29.0%/−42.8%; Test-1 +9.2%/−17.2%; Test-2 +15.5%/−13.6%.

---

## 3. Part B — where the return comes from

### 3.1 Method

The simulator's NAV is `cash + shares × spot − short_option_liability`, marked daily (`rlbot/simulator/portfolio.py`). Assignment moves `strike × 100 × n` from cash to shares; call-away moves it back at the call strike. CAGR is computed from that NAV curve, so it is **total return** — premiums plus every dollar of stock gain or loss between assignment and sale.

The decomposition (`b3_decompose.py`) splits ΔNAV per sleeve by an exact identity:

```
ΔNAV = net premium (fills − commissions)
     + realized stock P&L      (call-away strike − put strike) × shares
     + unrealized stock P&L    (final close − put strike) × shares still held
     + residual                (open option liability at window end, rounding)
```

**Accounting choice:** all premium is counted as income, and the stock leg is measured strike-to-strike. The alternative (netting the assigned put's premium into cost basis, as the simulator itself records `cost_basis = strike − premium_fill`) moves dollars from the premium row to the stock row without changing the total. The choice here answers "how much came from selling options?" without double-counting.

**Reconciliation:** Σ sleeve finals / 10 = $389,020 vs headline $388,695 (0.08%, from ffill alignment of sleeve end dates in the pooled curve). Residuals are ≤ 0.5% of capital in every window.

### 3.2 Pooled decomposition

**Full 2013–2026 (13.63 years, $1M across 10 sleeves)**

| Component | Dollars | Share of ΔNAV | Simple %/yr |
|---|---|---|---|
| Net premium income | **+$2,143,676** | **74.2%** | +15.7% |
| Realized stock P&L (called away vs put strike) | **+$820,515** | **28.4%** | +6.0% |
| Unrealized stock P&L (held at end) | −$62,285 | −2.2% | −0.5% |
| Residual (open liability, rounding) | −$11,701 | −0.4% | −0.1% |
| **ΔNAV** | **+$2,890,204** | 100% | +21.2% simple ≡ **+10.5% CAGR** |

Cycles: **1,037 puts sold → 86 assigned (8.3%) → 78 called away (91% of assignments)**; 332 calls sold. **8 of 10 sleeves end the window long stock.**

**Test-1 2022–2023 (2.0 years, bear market)**

| Component | Dollars | Share | Simple %/yr |
|---|---|---|---|
| Net premium | +$118,753 | **96.6%** | +5.95% |
| Realized stock | +$17,323 | 14.1% | +0.87% |
| Unrealized stock | −$9,453 | −7.7% | −0.47% |
| Residual | −$3,650 | −3.0% | −0.18% |
| **ΔNAV** | **+$122,972** | | +6.2% ≡ **+6.0% CAGR** |

Cycles: 127 puts → 11 assigned → 9 called away; 36 calls; 2/10 sleeves end long stock.

**Test-2 2024–2026 (2.64 years)**

| Component | Dollars | Share | Simple %/yr |
|---|---|---|---|
| Net premium | +$248,198 | **88.4%** | +9.41% |
| Realized stock | +$59,293 | 21.1% | +2.25% |
| Unrealized stock | −$22,197 | −7.9% | −0.84% |
| Residual | −$4,592 | −1.6% | −0.17% |
| **ΔNAV** | **+$280,702** | | +10.7% ≡ **+9.9% CAGR** |

Cycles: 187 puts → 23 assigned → 15 called away; 88 calls; 8/10 sleeves end long stock.

_Note on "simple %/yr": the decomposition is additive in dollars, so its annualized column is arithmetic (total ÷ years), not compounded. $2.89M on $1M over 13.6 years is 289% total, which compounds to the 10.5% CAGR in §2. The **shares** are the meaningful comparison._

### 3.3 Per-ticker decomposition — Full 2013–2026 ($100K sleeve each)

| Ticker | Net premium | Realized stock | Unrealized | ΔNAV | Puts / assigned / called away | End shares | B3 CAGR | B3 max DD | B&H CAGR |
|---|---|---|---|---|---|---|---|---|---|
| AAPL | $218,410 | $119,619 | $0 | $336,055 | 101 / 16 / 16 | 0 | 11.4% | −23.2% | 24.0% |
| MSFT | $127,196 | $40,007 | $15,558 | $182,331 | 117 / 8 / 7 | 500 | 7.9% | −27.3% | 25.4% |
| NVDA | $145,898 | $43,118 | $0 | $191,295 | 121 / 4 / 4 | 0 | 8.3% | −54.5% | 62.2% |
| AMZN | $285,404 | $150,200 | $17,260 | $445,615 | 64 / 9 / 8 | 2,000 | 13.3% | −47.6% | 24.6% |
| GOOGL | $180,954 | $98,504 | $29 | $279,457 | 96 / 10 / 9 | 1,000 | 10.3% | −38.5% | 24.2% |
| META | $692,096 | $209,368 | −$24,821 | $876,571 | 76 / 12 / 11 | 1,700 | 18.2% | −72.2% | 24.5% |
| V | $137,991 | $50,182 | $17,083 | $203,164 | 117 / 6 / 5 | 700 | 8.5% | −29.1% | 18.9% |
| MA | $121,897 | $13,847 | $13,327 | $146,839 | 130 / 5 / 4 | 400 | 6.9% | −27.6% | 20.2% |
| WMT | $91,411 | $20,807 | −$27,224 | $85,048 | 112 / 8 / 7 | 1,700 | 4.6% | −27.5% | 13.8% |
| UNH | $142,419 | $74,863 | −$73,497 | $143,830 | 103 / 8 / 7 | 500 | 6.8% | −48.1% | 17.4% |

Observations:
- **META is the single largest contributor** ($877K of the $2.89M) and also carries the worst sleeve drawdown (−72%): assigned ahead of the 2022 collapse and held through it. The stock leg giveth and taketh.
- **NVDA sold 121 puts and was assigned only 4 times** — premium-dominated (8.3% CAGR) against a 62% buy-and-hold CAGR. The largest opportunity cost in the set, and the clearest illustration that a wheel on a compounder forfeits the compounding.
- **UNH and WMT end the window with meaningful unrealized losses** (−$73K, −$27K) — shares assigned above where they trade today. These are the sleeves currently "stuck in stock."
- Premium share by sleeve ranges from ~60% (META) to ~100% (NVDA, V — few assignments).

### 3.4 Per-ticker — Test-1 2022–2023 (bear)

| Ticker | Net premium | Realized | Unrealized | ΔNAV | Puts / asg / away | End shares | B3 CAGR | B3 max DD | B&H CAGR |
|---|---|---|---|---|---|---|---|---|---|
| AAPL | $19,156 | $8,973 | $0 | $27,838 | 17 / 3 / 3 | 0 | 13.2% | −14.6% | 3.5% |
| MSFT | $9,933 | −$384 | $0 | $9,330 | 9 / 1 / 1 | 0 | 4.6% | −25.9% | 7.0% |
| NVDA | $18,247 | $89 | $0 | $17,106 | 11 / 1 / 1 | 0 | 8.3% | −52.0% | 28.6% |
| AMZN | $4,763 | $0 | −$7,836 | −$3,180 | 1 / 1 / 0 | 600 | −1.6% | −46.8% | −5.6% |
| GOOGL | $12,286 | −$693 | $0 | $10,420 | 5 / 1 / 1 | 0 | 5.1% | −28.5% | −1.9% |
| META | $10,258 | −$1,487 | $0 | $8,598 | 6 / 1 / 1 | 0 | 4.2% | −66.6% | 2.3% |
| V | $12,882 | $0 | $0 | $12,683 | 20 / 0 / 0 | 0 | 6.2% | −3.2% | 9.4% |
| MA | $11,298 | $3,065 | $0 | $14,178 | 18 / 1 / 1 | 0 | 6.9% | −5.4% | 7.9% |
| WMT | $8,606 | $0 | $0 | $8,535 | 22 / 0 / 0 | 0 | 4.2% | −1.3% | 6.1% |
| UNH | $11,323 | $7,759 | −$1,617 | $17,465 | 18 / 2 / 1 | 200 | 8.4% | −7.7% | 3.8% |

Observations: **V and WMT were pure premium sleeves** (20 and 22 puts, zero assignments, drawdowns of −3.2% and −1.3%) and returned 4–6%/yr in a year the market fell — the wheel working as designed. **AMZN sold one put in two years, was assigned on it, and held the shares under water** (−$7.8K unrealized) — the wheel *not* working: a single assignment in a bear market with no recovery inside the window. The regime-adaptive WAIT rule kept the whole book's assignment count to 11.

### 3.5 Per-ticker — Test-2 2024–2026

| Ticker | Net premium | Realized | Unrealized | ΔNAV | Puts / asg / away | End shares | B3 CAGR | B3 max DD | B&H CAGR |
|---|---|---|---|---|---|---|---|---|---|
| AAPL | $20,538 | $8,785 | $0 | $28,794 | 21 / 4 / 4 | 0 | 10.1% | −21.5% | 22.0% |
| MSFT | $13,691 | $9,019 | $6,223 | $28,749 | 18 / 3 / 2 | 200 | 10.1% | −16.8% | 11.5% |
| NVDA | $61,127 | $0 | $0 | $60,970 | 29 / 0 / 0 | 0 | 19.8% | −7.9% | 76.5% |
| AMZN | $34,740 | $2,500 | $4,315 | $39,749 | 13 / 3 / 2 | 500 | 13.6% | −24.5% | 23.0% |
| GOOGL | $29,271 | $18,299 | $12 | $47,629 | 18 / 3 / 2 | 400 | 16.0% | −20.6% | 42.0% |
| META | $29,525 | $7,079 | −$2,920 | $33,679 | 21 / 3 / 2 | 200 | 11.7% | −10.5% | 19.6% |
| V | $14,520 | $10,652 | $7,321 | $31,590 | 10 / 2 / 1 | 300 | 11.0% | −12.1% | 15.5% |
| MA | $16,762 | $0 | $6,663 | $22,304 | 23 / 1 / 0 | 200 | 7.9% | −13.9% | 13.6% |
| WMT | $18,110 | $266 | −$14,413 | $4,004 | 27 / 2 / 1 | 900 | 1.5% | −10.6% | 30.3% |
| UNH | $9,912 | $2,694 | −$29,399 | −$16,767 | 7 / 2 / 1 | 200 | −6.7% | −54.2% | −9.8% |

Observations: **NVDA is the ideal case** — 29 puts, zero assignments, 19.8% CAGR from premium alone with a −7.9% drawdown (but 76.5% buy-and-hold: the opportunity cost of not owning it). **UNH is the cautionary case** — assigned into the 2025 collapse, −$29K unrealized, −54% sleeve drawdown, the one negative sleeve. WMT's −$14K unrealized is the same pattern in miniature.

---

## 4. Reading the results

1. **This is an income strategy that occasionally becomes a stock strategy.** Roughly 74% of the long-run return is premium; in a bear year premium is essentially all of it (~6%/yr simple). The premium engine is steady across regimes; the stock leg is episodic and is the entire source of drawdown.
2. **The stock leg was net positive over 13 years — because of what these ten stocks did.** 78 of 86 assignments exited higher than the put strike. That is a property of ten survivor mega-caps in the strongest large-cap bull market on record (DATA-GAP-5), not evidence the wheel times entries well. The UNH/WMT/AMZN sleeves show what an assignment looks like when the name does not recover inside the window.
3. **B3's adaptive tiers did not earn their complexity against B1 on this history.** Full window: B1 +13.2%/−24.6% vs B3 +10.5%/−26.5%. B3's regime-aware WAIT reduced assignments in 2022 (11 in two years) but cost return; the fixed 20Δ wheel simply collected more premium. Shorter windows are a wash. Simplifying to fixed delta is a defensible outcome under the project's own philosophy — but see §6: the live system's differentiator is not the tier logic.
4. **Versus buy-and-hold the wheel roughly halves the drawdown and forfeits most of the compounding** (−26.5% vs −49.5%; +10.5% vs +38.8%). Any income strategy on this basket looks poor against holding it. That is the benchmark, not the strategy.

---

## 5. What the backtest cannot see

The historical run measures the **engine**, not the **car as driven today**. Absent from every number above, by necessity (no historical data) or by design:

- **MCB acquisition ceilings and `delta_posture`** (SPEC-011) — the live system's actual differentiator: *where* it is willing to be assigned. Unmeasured historically. **G7** (retrospective proxy-ceiling A/B) is the experiment that would measure it and is now the highest-value open item.
- **Book-level risk** (RISK-3/4/5/8/9, SPEC-004 §2) — single-ticker sleeves use `RiskConfig.single_ticker()`, which relaxes every portfolio cap. Real portfolio behavior with 20+ positions is not represented.
- **Human-review warnings, the opportunity scan, the downtrend override** — live-only.
- **The score-side MCB valuation** (SPEC-011 rule 6) — the backtest still scores on the FV/EPS axis.

---

## 6. Implications for operating the live system

- **Plan around ~+10%/yr with a ~−26% worst case on this universe, ±1–2 points for microstructure** — not the superseded $3.19M figure.
- **Expect the return to be mostly premium, and expect the pain to come from assigned stock.** Position sizing, the stress reserve (RISK-8) and per-name exposure caps (RISK-3) are doing the real work; they bound the only leg that can hurt.
- **The MCB discipline governs exactly the leg the backtest shows to be dangerous** (UNH, WMT, AMZN-2022). Whether it *improves* outcomes is untested — run G7 before treating it as validated.
- **Consider whether the adaptive tiers should stay.** A paired B1-vs-B3 comparison segmented by regime (SPEC-006 machinery exists) would settle it; if B1 ≈ B3, a fixed-delta wheel with MCB gating is simpler and equally supported by evidence.

---

## 7. Caveats and limitations

- **Survivorship (DATA-GAP-5):** the universe is the ten mega-caps that survived and compounded. The stock leg's positive contribution is not transferable to a universe with failures.
- **Dividends are embedded via adjusted prices, not paid as cash.** Both the wheel and the buy-and-hold benchmark use the same adjusted series, so the comparison is fair; absolute levels are slightly conservative for high payers.
- **Single-ticker sleeves, $100K each, dynamic contract sizing, equal-weighted at the pooled level.** Not a portfolio simulation.
- **Expiration-only assignment; no early exercise; BS fallback on ≤0.1% of marks.** Fills = mid − k·(ask−bid) plus $0.65/contract.
- **Microstructure sensitivity ±1–2 pts/yr** (from the earlier fill/liquidity calibration work).
- **Accounting choice** (§3.1): premium counted fully as income; strike-to-strike stock leg.
- **Simple vs compounded**: decomposition columns are arithmetic; headline CAGR is geometric.

---

## 8. Reproduce

```bash
cd /Users/rahul/Documents/MyFiles/Areas/Claude-Workspace/Projects/wheel-strategy-rlbot
```

```bash
/Users/rahul/opt/anaconda3/envs/wheel-rlbot/bin/python -m rlbot.evaluation.b3_performance --historical
```

```bash
/Users/rahul/opt/anaconda3/envs/wheel-rlbot/bin/python -m rlbot.evaluation.b3_decompose
```

Outputs land in `data_local/reports/` (gitignored). Rerun both after any change to real-quote scoring, selection, or execution logic (standing rule, SPEC-000 §4).

## 9. Sources

- `rlbot/simulator/portfolio.py` — NAV, assignment and call-away mechanics
- `rlbot/simulator/environment.py` — `WheelEnv.run`, decision and cycle records
- `rlbot/options/selector.py` — `score_quote` (commit cd7a3cd diff for the scoring fix)
- `rlbot/evaluation/b3_performance.py`, `rlbot/evaluation/b3_decompose.py`
- `specs/SPEC-000-overview-and-roadmap.md` §4 (rebaseline and decomposition status blocks), `specs/SPEC-004`, `specs/SPEC-006`, `specs/SPEC-011`
