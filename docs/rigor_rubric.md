# Rigor rubric — the strategy fact-checker

The rigor checker takes a **pasted Pine strategy** and answers one question: *is this
backtest trustworthy?* It does **not** judge profitability — it scores where the backtest
is fooling its author. Because the checks are computed against a fixed framework, scores
are comparable across strategies, which is what makes it a benchmark.

```bash
python -m trading_hypothesis_lab.rigor strategy.pine --instrument SPY --timeframe D
python -m trading_hypothesis_lab.rigor strategy.pine -i BINANCE:BTCUSDT -t 240 --instruments ETHUSD
python -m trading_hypothesis_lab.rigor - -i SPY -t D --no-tv   # stdin; skip TradingView entirely
```

Inputs beyond the script: **instrument(s)**, **timeframe**, and (for generalization) extra
`--instruments`. The script alone doesn't fix what it's tested on.

The empirical battery runs **without TradingView**: the script is mapped to a known strategy
archetype and re-implemented in a small lookahead-free Python engine, then backtested on free
OHLCV (Yahoo Finance). This is a *faithful model of the strategy logic*, not the literal Pine
on TradingView — the scorecard says so. TradingView is optional and used only to verify the
Pine compiles (`--no-tv` skips it).

---

## How a grade is decided

Rigor is **context-dependent** — 50 trades is plenty for a swing system and thin for a
scalper; acceptable drawdown and cost-sensitivity differ by style. So scoring is split:

1. **Objective disqualifiers (deterministic).** A backtest with look-ahead bias or
   repainting is *mechanically invalid* — no judgment needed. If one is present the grade
   is **Untrustworthy**, full stop, regardless of returns.
2. **Subjective grade (Sonnet 4.6).** Everything else is graded by an LLM that first infers
   the strategy *type*, then judges what rigor standard fits it, then weighs the evidence —
   rather than applying blanket thresholds. The grade comes with a written rationale.
3. **Deterministic fallback.** When no LLM backend is available, a transparent threshold
   sum is used instead, clearly labelled.

Grades: **Robust** · **Promising but fragile** · **Likely overfit** · **Untrustworthy** ·
**Does not compile**.

---

## Tier 1 — gates

| Gate | Type | What it catches |
|------|------|-----------------|
| **G1 Look-ahead bias** | static, **objective disqualifier** | `barmerge.lookahead_on`, future refs, unguarded MTF `request.security()` |
| **G2 Repainting** | static, **objective disqualifier** | `calc_on_every_tick=true` (orders fill intrabar) |
| **G3 Sufficient sample** | empirical | too few trades to conclude (feeds the subjective judgment, contextual by style) |
| **G4 Friction illusion** | empirical | edge that vanishes once realistic commission/slippage is added |

## Tier 2 — scored dimensions (empirical)

| Dim | What it catches |
|-----|-----------------|
| **D1 Out-of-sample** | curve-fitting — does the edge survive a held-out window |
| **D2 Overfitting surface** | tunable inputs vs trade count (degrees of freedom) |
| **D3 Multi-timeframe** | timeframe cherry-pick |
| **D4 Multi-instrument** | symbol cherry-pick |
| **D5 Inverse-edge / luck** | reversing entries also "wins" → noise, not a directional edge |
| **D6 Risk profile** | return vs max drawdown |
| **D7 Cost survival** | how much edge survives realistic costs |

## Deep review — Sonnet 4.6 (advisory, not scored)

A capable LLM reads the source for issues the regex gates miss — and crucially it is
**TradingView-independent** (needs no chart, no subscription): L1 look-ahead/repaint,
L2 overfitting, L3 cherry-picking, L4 cost & fill realism, L5 position sizing / risk,
L6 logic soundness. It cites the specific Pine construct. Advisory so the numeric grade
stays reproducible.

---

## How the empirical battery runs

The empirical checks (G3, G4 and D1/D3/D4/D5/D6/D7) run on a deterministic, offline Python
backtest engine — **not** the TradingView strategy tester. The pipeline:

1. **Infer the archetype.** Sonnet 4.6 maps the pasted Pine to one of: `rsi_mean_reversion`,
   `ma_cross`, `macd_cross`, `supertrend`, `bollinger_mean_reversion`, `donchian_breakout`,
   `orb` — or `custom` if it fits none. (A regex fallback runs when no LLM is available.)
2. **Fetch free OHLCV.** Yahoo Finance, mapped from the TradingView-style symbol
   (`BINANCE:BTCUSDT` → `BTC-USD`, `NSE:NIFTY` → `^NSEI`, `FX:EURUSD` → `EURUSD=X`, …) and
   timeframe (intraday history caps are respected: 1m≈7d, 5–30m≈60d, 1h≈730d, D/W = full).
3. **Backtest, lookahead-free.** Every signal for bar *t* is computed only from data through
   *t*'s close and **filled at bar t+1's open**. From this one engine we get the things the
   capped TradingView trade feed never could: a real out-of-sample **date** split (D1), a
   real **inverse-edge** run (D5, every long becomes a short), real **multi-timeframe** /
   **multi-instrument** re-runs (D3/D4), and a real **cost-stress** (G4/D7).

Why a model and not the literal Pine: driving the live TradingView strategy tester headlessly
proved unreliable (the compiled study would not attach to the chart, so the tester stayed
empty). Modelling the archetype trades that fragility for determinism and full control. The
trade-off is stated on every scorecard, and the inference confidence is reported.

### What's always TV-independent
Static gates G1/G2, the D2 input-count proxy, the Sonnet 4.6 deep review, the empirical
battery, and the subjective grade all run with no TradingView at all. TradingView (via the
MCP, when configured) is used **only** to verify the Pine compiles — a `Does not compile`
verdict requires that verification; `--no-tv` skips it and the scorecard notes the syntax
was not verified.

### When a check can't run
If the strategy maps to `custom` (no modellable archetype), or no free data is available, or
the strategy fires too few trades to conclude, the affected dimensions return `not_assessed`
— which **caps the grade below Robust** (you can't be rated robust on a check that never ran).
The tool never crashes and never invents a result.
