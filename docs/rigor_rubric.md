# Rigor rubric — the strategy fact-checker

The rigor checker takes a **pasted Pine strategy** and answers one question: *is this
backtest trustworthy?* It does **not** judge profitability — it scores where the backtest
is fooling its author. Because the checks are computed against a fixed framework, scores
are comparable across strategies, which is what makes it a benchmark.

```bash
python -m trading_hypothesis_lab.rigor strategy.pine --instrument SPY --timeframe 5
python -m trading_hypothesis_lab.rigor - --instrument BINANCE:BTCUSDT -t 60   # stdin
```

Inputs beyond the script: **instrument(s)**, **timeframe**, and (for generalization) extra
`--instruments`. The script alone doesn't fix what it's tested on.

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

## What runs without TradingView vs. what needs it

- **TV-independent (always runs):** static gates G1/G2, the D2 input-count proxy, the
  Sonnet 4.6 deep review, and the subjective grade. This is the reliable backbone.
- **Needs a live TradingView strategy tester:** G3, G4, and D1/D3/D4/D5/D6/D7. These compile
  the script and read the strategy tester via the MCP. Any MCP failure degrades the check to
  `not_assessed` — which **caps the grade below Robust** (you can't be rated robust on a
  check that never ran). The tool never crashes and never invents a result.

### Known limitation (open)
The empirical battery requires the compiled strategy to actually be **added to the chart**
as a study. On free-tier / current TradingView Desktop the indicator cap and a missing
"Add to chart" affordance can leave `study_added=false`, so the tester stays empty and the
empirical dimensions return `not_assessed`. Clearing existing studies now works
(`chart_manage_indicator` needs `entity_id` **and** the indicator name). Reliable
add-to-chart automation — or a TradingView-independent backtest (free OHLCV + a Python
engine, which requires accepting strategies in a portable form, not just Pine) — is the
planned upgrade for the empirical layer.
