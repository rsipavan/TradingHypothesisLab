# Rigor example — Supertrend trend-follower on BTC

**What this is:** a rigor-checker run (not a video-pipeline run). The input is a pasted Pine
strategy — a textbook always-in-market Supertrend reversal (`atrPeriod=10`, `factor=3.0`, no
stops). The question the checker answers is *is this backtest trustworthy?* — not *is it
profitable?*

**Command:**
```bash
python -m trading_hypothesis_lab.rigor examples/rigor_supertrend_btc_overfit/strategy.pine \
    --instrument BINANCE:BTCUSDT --timeframe D --instruments NASDAQ:QQQ --no-tv
```

**Bundle:**
- `strategy.pine` — the pasted strategy under test
- `scorecard.md` — the rigor scorecard (human-readable)
- `scorecard.json` — the structured scorecard

**Why this example:** it shows the check that the live TradingView strategy tester could
never give — a real **out-of-sample date split**. On the full history the strategy looks
fine (profit factor ~1.8), but D1 splits the bars chronologically and the edge **inverts**:
in-sample PF 2.54 collapses to **OOS PF 0.66** — it actively loses on unseen data. That is
the textbook signature of curve-fitting, so the grade is **Likely overfit (~31/100)**, even
though it passes the multi-instrument (QQQ), multi-timeframe, and friction checks. The
Sonnet 4.6 review independently flags the zero-cost assumption (L4) and the absent stop (L5).

**Generated:** 2026-05-23, on 3652 daily BTC-USD bars (Yahoo). The engine models the
strategy's *inferred* Supertrend logic, not the literal Pine on TradingView — the scorecard
states this and reports inference confidence. Numbers will drift as more price history
accrues and the LLM re-grades; the verdict (overfit, caught by OOS) is the stable point.
