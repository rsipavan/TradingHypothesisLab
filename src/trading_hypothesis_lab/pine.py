"""Pine Script synthesis: strategy_backtest claims → .pine file → compiled → backtested.

The public entry point is `run()`. It:
  1. Detects any Pine Script code in the transcript text
  2. Synthesizes a complete Pine v5 strategy via the LLM (from extracted snippets
     OR from the verbal description when no code was shown)
  3. Compiles the script via the TradingView MCP; if it has errors, asks the LLM to fix
     them (up to _MAX_COMPILE_RETRIES attempts)
  4. Runs the TradingView strategy tester and reads the metrics
  5. Returns a ValidationRun with the results and (optionally) writes the .pine file to disk

Every failure mode returns a ValidationRun — it never raises past `run()`.
"""

from __future__ import annotations

import re
from pathlib import Path

from . import llm as llm_mod
from .config import Config
from .mcp_client import McpClient, McpError
from .types import Claim, StrategyBacktestMetrics, Transcript, ValidationRun, VideoSummary

_MAX_COMPILE_RETRIES = 3

# Patterns that indicate Pine Script code is present in a transcript.
_PINE_MARKERS = re.compile(
    r"//@version=|strategy\s*\(|indicator\s*\(|plot\s*\(|ta\.rsi\(|ta\.ema\(|ta\.sma\(|"
    r"alertcondition\s*\(|bgcolor\s*\(|barcolor\s*\(|syminfo\.",
    re.IGNORECASE,
)


class PineSynthesisError(RuntimeError):
    """Raised when LLM synthesis itself fails — caught inside run() and converted to ValidationRun."""


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------


def run(
    claim: Claim,
    transcript: Transcript,
    summary: VideoSummary | None,
    config: Config,
    mcp: McpClient,
    *,
    out_dir: Path | None = None,
    cached_script: str | None = None,
) -> ValidationRun:
    """Synthesize, compile, and backtest a Pine Script strategy for `claim`.

    `out_dir`: if provided, the .pine file is written there and `vr.pine_script_path`
    is the absolute path. If None, the script is processed in-memory only (useful in
    tests that don't need a real file).
    `cached_script`: if provided, skip LLM synthesis entirely and compile this script
    directly. Used when a previous run already produced a good script for this claim.
    """
    instrument = claim.instrument or "SPY"
    timeframe = claim.timeframe or "1D"

    # A claim whose "instrument" is actually a multi-asset basket (e.g. a 10-symbol
    # portfolio) cannot be reproduced on the strategy tester, which backtests ONE symbol
    # at a time. Decline honestly instead of silently backtesting one arbitrary leg (or a
    # bogus combined symbol) and passing it off as the portfolio result.
    multi, n_assets = _is_multi_asset(instrument)
    if multi:
        return _failed(
            claim,
            f"claim spans a multi-asset portfolio ({n_assets} instruments: {instrument}); "
            "the TradingView strategy tester backtests one symbol at a time, so a "
            "portfolio-level aggregate cannot be reproduced as a single backtest — it would "
            "need a per-asset scan and re-aggregation, out of scope for a single-symbol test",
        )

    # --- step 1: synthesize (or reuse cached script) ---
    if cached_script:
        script = cached_script
    else:
        try:
            script = _synthesize(claim, transcript, summary, config)
        except PineSynthesisError as e:
            return _failed(claim, f"could not synthesize Pine Script — {e}")
        except llm_mod.LlmUnavailable:
            return _failed(claim, "could not synthesize Pine Script — LLM unavailable")

    # Save the raw synthesis output immediately — before any compile attempts.
    # This is the key crash-safety artifact: if the compile loop fails or the process
    # dies mid-repair, draft_synthesis_<id>.pine always shows what the LLM produced.
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"draft_synthesis_{claim.id}.pine").write_text(script, encoding="utf-8")

    # Clear existing chart studies before loading the strategy. Free TradingView
    # accounts have an indicator cap (~3 studies). If prior studies are on the chart,
    # pine_smart_compile will succeed (no errors) but the strategy won't appear —
    # leaving the strategy tester empty and reporting 0 trades.
    _clear_chart_indicators(mcp)

    # --- step 2: compile (with self-repair loop) ---
    script, errors = _compile_and_fix(script, mcp, config, draft_dir=out_dir, claim_id=claim.id)

    # Write the final script to disk regardless of compile result.
    pine_path: str | None = None
    if out_dir is not None:
        file_path = out_dir / f"strategy_{claim.id}.pine"
        file_path.write_text(script, encoding="utf-8")
        pine_path = str(file_path.resolve())

    if errors:
        msg = f"script written but does not compile after {_MAX_COMPILE_RETRIES} fix attempts"
        if pine_path:
            msg += f" — see {Path(pine_path).name}"
        return ValidationRun(
            claim_id=claim.id,
            test_type="strategy_backtest",
            timeframe=timeframe,
            status="insufficient_data",
            tradingview_query=f"{instrument} {timeframe} Pine Strategy",
            data_summary=f"compile errors: {'; '.join(errors[:3])}",
            occurrences=None,
            hit_rate=None,
            result=msg,
            caveats=["script has compilation errors — open in TradingView Pine editor for details"],
            pine_script_path=pine_path,
        )

    # --- step 3: resolve symbol and run backtest ---
    # Try the claim's primary timeframe first, then fall back through wider timeframes.
    # If all timeframes return 0 trades, try alternate instruments before giving up.
    # Crypto RSI divergence setups (RSI<30) are rare on BTC; ETH oscillates more.
    _TF_FALLBACKS = {
        "1m": "5m",  "5m": "15m",  "15m": "1H",  "1H": "4H",  "4H": "1D",
        "1":  "5",   "5":  "15",   "15":  "60",  "60": "240", "240": "D",
    }
    # Instrument fallback: only used when the primary instrument fires 0 trades on
    # all timeframes. Keeps the claim instrument category (crypto → crypto, stock → stock).
    _INSTRUMENT_FALLBACKS: dict[str, list[str]] = {
        "BTCUSDT": ["BINANCE:ETHUSD", "COINBASE:ETHUSD"],
        "BITCOIN": ["BINANCE:ETHUSD"],
        "BTC":     ["BINANCE:ETHUSD"],
    }

    tried_timeframes = [timeframe]
    metrics: StrategyBacktestMetrics | None = None
    actual_tf = timeframe
    actual_instrument = instrument
    last_mcp_err: McpError | None = None

    def _try_instrument(sym: str) -> tuple[StrategyBacktestMetrics | None, str, McpError | None]:
        """Try all timeframe fallbacks for `sym`. Returns (metrics, tf_used, last_err)."""
        tfs = [timeframe]
        m: StrategyBacktestMetrics | None = None
        tf_used = timeframe
        err: McpError | None = None
        for tf in tfs:
            try:
                m = _run_backtest(sym, tf, mcp)
            except McpError as e:
                err = e
                break
            if m is not None:
                tf_used = tf
                break
            nxt = _TF_FALLBACKS.get(tf)
            if nxt and nxt not in tfs:
                tfs.append(nxt)
        tried_timeframes[:] = tfs  # propagate for error reporting
        return m, tf_used, err

    metrics, actual_tf, last_mcp_err = _try_instrument(instrument)

    if metrics is None and last_mcp_err is None:
        # All timeframes on primary instrument gave 0 trades — try alternates
        for alt in _INSTRUMENT_FALLBACKS.get(instrument.upper().replace("BINANCE:", ""), []):
            metrics, actual_tf, last_mcp_err = _try_instrument(alt)
            if metrics is not None or last_mcp_err is not None:
                actual_instrument = alt
                break

    if last_mcp_err is not None:
        return ValidationRun(
            claim_id=claim.id,
            test_type="strategy_backtest",
            timeframe=timeframe,
            status="error",
            tradingview_query=f"{instrument} {timeframe} Pine Strategy",
            data_summary="",
            occurrences=None,
            hit_rate=None,
            result=f"strategy compiled but backtest failed: {last_mcp_err}",
            caveats=[],
            pine_script_path=pine_path,
        )

    if metrics is None:
        tfs_tried = " → ".join(tried_timeframes)
        return ValidationRun(
            claim_id=claim.id,
            test_type="strategy_backtest",
            timeframe=timeframe,
            status="insufficient_data",
            tradingview_query=f"{instrument} {timeframe} Pine Strategy",
            data_summary=f"strategy compiled but produced no trades on {tfs_tried}",
            occurrences=0,
            hit_rate=None,
            result=(
                f"strategy compiled but produced no trades on {instrument} "
                f"({tfs_tried} tried) — entry conditions never fired in the available history"
            ),
            caveats=[
                "no trades means the strategy's entry conditions did not occur in the "
                "available price history — the setup may be rare on this instrument",
                "try a more volatile instrument or adjust threshold inputs in the .pine file",
            ],
            pine_script_path=pine_path,
        )

    if pine_path:
        metrics = StrategyBacktestMetrics(
            net_profit=metrics.net_profit,
            gross_profit=metrics.gross_profit,
            total_trades=metrics.total_trades,
            winning_trades=metrics.winning_trades,
            win_rate=metrics.win_rate,
            max_drawdown=metrics.max_drawdown,
            profit_factor=metrics.profit_factor,
            pine_script_path=pine_path,
        )

    win_rate_pct = f"{metrics.win_rate:.0%}"
    notes: list[str] = []
    if actual_tf != timeframe:
        notes.append(f"fell back from {timeframe}")
    if actual_instrument != instrument:
        notes.append(f"fell back from {instrument}")
    tf_note = f" ({', '.join(notes)})" if notes else ""
    return ValidationRun(
        claim_id=claim.id,
        test_type="strategy_backtest",
        timeframe=actual_tf,
        status="ok",
        tradingview_query=f"{actual_instrument} {actual_tf} Pine Strategy (strategy tester){tf_note}",
        data_summary=(
            f"{actual_instrument} {actual_tf}: {metrics.total_trades} trades, "
            f"{win_rate_pct} win rate, net {metrics.net_profit:+.2f}, "
            f"max drawdown {metrics.max_drawdown:.2f}, PF {metrics.profit_factor:.2f}"
        ),
        occurrences=metrics.total_trades,
        hit_rate=metrics.win_rate,
        result=(
            f"{metrics.winning_trades}/{metrics.total_trades} trades won ({win_rate_pct}); "
            f"net profit {metrics.net_profit:+.2f}; profit factor {metrics.profit_factor:.2f}"
        ),
        caveats=[
            "strategy tester results depend on the instrument, date range, and default capital settings in TradingView",
            "no slippage or commission modeled unless the Pine script includes them",
            "past performance does not predict future results",
        ],
        strategy_backtest=metrics,
        pine_script_path=pine_path,
    )


# ---------------------------------------------------------------------------
# synthesis
# ---------------------------------------------------------------------------


def _detect_pine_code(text: str) -> list[str]:
    """Return any Pine Script code snippets found in `text`. Empty list if none."""
    if not _PINE_MARKERS.search(text):
        return []
    # Heuristic: split on blank lines, keep chunks that contain Pine markers.
    chunks = re.split(r"\n{2,}", text)
    return [c.strip() for c in chunks if _PINE_MARKERS.search(c)]


def _synthesize(
    claim: Claim,
    transcript: Transcript,
    summary: VideoSummary | None,
    config: Config,
) -> str:
    """Ask the LLM to produce one complete Pine v5 strategy script."""
    snippets = _detect_pine_code(transcript.text)
    topic = summary.topic if summary else claim.statement
    text_excerpt = transcript.text[:6000]  # keep prompt manageable

    _v5_rules = (
        "CRITICAL RULES — follow all of these exactly:\n"
        "1. The VERY FIRST line must be //@version=5  (NOT version=6, NOT any other version)\n"
        "2. Use strategy() not indicator()\n"
        "3. Use strategy.entry() and strategy.exit() for trades\n"
        "4. Make threshold inputs (RSI levels, MA periods, etc.) into input.* parameters so they "
        "   are adjustable — do NOT hard-code magic numbers that can never fire (e.g. if the "
        "   strategy requires RSI<30 TWICE in a row, expose 'oversoldLevel' as an input so a "
        "   user can relax it to 40 if needed)\n"
        "5. Add a 'lookbackBars' or 'maxSetupBars' input (default 500) that expires stale setups "
        "   so they don't block new ones indefinitely\n"
        "6. NO LOOK-AHEAD BIAS — non-negotiable for an honest backtest:\n"
        "   - Evaluate entry/exit conditions ONLY on confirmed values from the current or prior "
        "     bars; never reference future bars.\n"
        "   - If you use request.security() for a higher timeframe you MUST pass "
        "     lookahead=barmerge.lookahead_off and read the previous completed value (index [1]) "
        "     so no future higher-timeframe data leaks in. NEVER use barmerge.lookahead_on.\n"
        "   - Do NOT set calc_on_every_tick=true; leave it default so historical and realtime "
        "     behave identically and signals do not repaint.\n"
        "7. Model realistic costs in the strategy() header: commission_type, commission_value, "
        "   and slippage — the backtest must not be frictionless.\n"
        "8. Output ONLY the Pine Script code — no markdown fences, no explanation text\n"
    )

    if snippets:
        combined = "\n\n---\n\n".join(snippets[:5])
        prompt = (
            f"Topic: {topic}\n"
            f"Strategy claim: {claim.statement}\n\n"
            f"The video transcript contains these Pine Script snippets:\n\n"
            f"```pine\n{combined}\n```\n\n"
            f"Extract, merge, and complete these into ONE complete, compilable Pine Script v5 "
            f"strategy (not indicator).\n\n"
            f"{_v5_rules}"
        )
    else:
        prompt = (
            f"Topic: {topic}\n"
            f"Strategy claim: {claim.statement}\n\n"
            f"Video transcript (excerpt):\n{text_excerpt}\n\n"
            f"Write a complete, compilable Pine Script v5 STRATEGY (not indicator) that "
            f"implements the trading strategy described above.\n\n"
            f"{_v5_rules}"
            f"\nWhere the video is ambiguous or silent, add a comment: // TODO: [what was unclear]"
        )

    system = (
        "You are a Pine Script v5 expert. You write clean, compilable TradingView strategy "
        "scripts from trading strategy descriptions. You ONLY write //@version=5 scripts — "
        "NEVER version 6. You never write indicator scripts — always strategy scripts with "
        "strategy() and strategy.entry()/strategy.close() calls. "
        "You output ONLY raw Pine Script code, no markdown fences, no explanatory text."
    )

    try:
        result = llm_mod.complete(system, prompt, model=config.synthesize_model or None, max_tokens=4000, timeout=600)
    except llm_mod.LlmError as e:
        raise PineSynthesisError(str(e)) from e

    # Strip markdown fences if the LLM wrapped the code
    result = result.strip()
    if result.startswith("```"):
        result = re.sub(r"^```\w*\n?", "", result)
        result = re.sub(r"\n?```$", "", result)
    result = result.strip()

    # Enforce v5 even if the LLM hallucinated v6 or another version.
    result = re.sub(r"^//@version\s*=\s*\d+", "//@version=5", result, count=1, flags=re.MULTILINE)
    return result


# ---------------------------------------------------------------------------
# chart housekeeping
# ---------------------------------------------------------------------------


def _clear_chart_indicators(mcp: McpClient) -> None:
    """Remove all studies from the chart before loading a new strategy.

    Free TradingView accounts have an indicator cap. Existing studies from a
    previous run block the compiled strategy from appearing, which leaves the
    strategy tester empty (0 trades). Errors are swallowed — if the MCP can't
    list or remove studies we proceed and let the compile step surface failures.
    """
    try:
        state = mcp.call("chart_get_state", {})
    except McpError:
        return

    # chart_get_state field names vary by MCP version — try common variants.
    indicators: list = (
        state.get("indicators")
        or state.get("studies")
        or (state.get("panes") or [{}])[0].get("indicators", [])
        or []
    )
    for ind in indicators:
        entity_id = ind.get("id") or ind.get("entityId") or ind.get("entity_id")
        if entity_id:
            try:
                mcp.call("chart_manage_indicator", {"action": "remove", "id": entity_id})
            except McpError:
                pass


# ---------------------------------------------------------------------------
# compilation with self-repair
# ---------------------------------------------------------------------------


def _compile_and_fix(
    script: str,
    mcp: McpClient,
    config: Config,
    max_retries: int = _MAX_COMPILE_RETRIES,
    *,
    draft_dir: Path | None = None,
    claim_id: str = "c0",
) -> tuple[str, list[str]]:
    """Compile `script` via MCP, fix errors with LLM up to `max_retries` times.

    Returns (final_script, errors). If errors is empty, the script compiled cleanly.
    If draft_dir is provided, each failed fix attempt is saved as
    draft_fix_{claim_id}_{n}.pine so the full repair trajectory is inspectable.
    """
    for attempt in range(max_retries + 1):
        errors = _compile_once(script, mcp)
        if not errors:
            return script, []
        if attempt == max_retries:
            break
        # Ask the LLM to fix the errors
        try:
            script = _fix_script(script, errors, config)
        except (llm_mod.LlmUnavailable, llm_mod.LlmError):
            break
        # Save the post-fix draft so the repair trajectory is visible on disk.
        if draft_dir is not None:
            (draft_dir / f"draft_fix_{claim_id}_{attempt + 1}.pine").write_text(
                script, encoding="utf-8"
            )
    return script, errors


def _compile_once(script: str, mcp: McpClient) -> list[str]:
    """Inject script, compile, and return a list of error strings (empty = success)."""
    try:
        mcp.call("pine_set_source", {"source": script})
        mcp.call("pine_smart_compile", {})
        errors_res = mcp.call("pine_get_errors", {})
        return _parse_compile_errors(errors_res)
    except McpError:
        # MCP doesn't support Pine tools — treat as a non-recoverable error upstream
        raise


def _parse_compile_errors(res) -> list[str]:
    """Normalize the pine_get_errors response into a flat list of error strings."""
    if isinstance(res, dict):
        errors = res.get("errors") or res.get("error") or res.get("messages") or []
        if isinstance(errors, str):
            return [errors] if errors else []
        if isinstance(errors, list):
            out = []
            for e in errors:
                if isinstance(e, dict):
                    msg = e.get("message") or e.get("text") or e.get("error") or str(e)
                    out.append(str(msg))
                elif isinstance(e, str):
                    out.append(e)
            return out
    if isinstance(res, list):
        out = []
        for e in res:
            if isinstance(e, dict):
                msg = e.get("message") or e.get("text") or e.get("error") or str(e)
                out.append(str(msg))
            elif e:
                out.append(str(e))
        return out
    if isinstance(res, str) and res:
        return [res]
    return []


def _fix_script(script: str, errors: list[str], config: Config) -> str:
    """Ask the LLM to fix compilation errors in `script`."""
    error_text = "\n".join(f"- {e}" for e in errors[:10])
    prompt = (
        f"This Pine Script v5 strategy has compilation errors. Fix ALL of them.\n\n"
        f"Errors:\n{error_text}\n\n"
        f"Current script:\n```pine\n{script}\n```\n\n"
        f"Output ONLY the corrected Pine Script code, no explanation."
    )
    system = (
        "You are a Pine Script v5 expert fixing compilation errors. "
        "Output ONLY the corrected Pine Script code block, nothing else."
    )
    result = llm_mod.complete(system, prompt, model=config.synthesize_model or None, max_tokens=4000, timeout=180)
    result = result.strip()
    if result.startswith("```"):
        result = re.sub(r"^```\w*\n?", "", result)
        result = re.sub(r"\n?```$", "", result)
    return result.strip()


# ---------------------------------------------------------------------------
# backtest
# ---------------------------------------------------------------------------


def _run_backtest(instrument: str, timeframe: str, mcp: McpClient) -> StrategyBacktestMetrics | None:
    """Set the chart to instrument/timeframe and read strategy tester results.

    Returns None if the strategy produced no trades or the results couldn't be parsed.
    """
    mcp.call("chart_set_symbol", {"symbol": instrument})
    mcp.call("chart_set_timeframe", {"timeframe": timeframe})
    res = mcp.call("data_get_strategy_results", {})
    return _parse_strategy_results(res)


def _parse_strategy_results(res) -> StrategyBacktestMetrics | None:
    """Normalize data_get_strategy_results response into StrategyBacktestMetrics."""
    if not isinstance(res, dict):
        return None

    # TradingView MCP may return the data directly or nested under a key
    data = res
    for key in ("result", "data", "strategy", "performance"):
        if isinstance(res.get(key), dict):
            data = res[key]
            break

    def _fv(key, *aliases):
        for k in (key,) + aliases:
            v = data.get(k)
            if v is not None:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    pass
        return 0.0

    def _iv(key, *aliases):
        return int(_fv(key, *aliases))

    total = _iv("total_trades", "totalTrades", "total_closed_trades", "closedTrades")
    if total == 0:
        return None

    winning = _iv("winning_trades", "winningTrades", "profitable_trades", "profitableTrades")
    win_rate = winning / total if total > 0 else 0.0
    gross_profit = _fv("gross_profit", "grossProfit")
    gross_loss = abs(_fv("gross_loss", "grossLoss"))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    return StrategyBacktestMetrics(
        net_profit=_fv("net_profit", "netProfit", "net_profit_percent", "netProfitPercent"),
        gross_profit=gross_profit,
        total_trades=total,
        winning_trades=winning,
        win_rate=win_rate,
        max_drawdown=abs(_fv("max_drawdown", "maxDrawdown", "max_drawdown_percent")),
        profit_factor=profit_factor,
        pine_script_path="",  # set by caller after writing the file
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _is_multi_asset(instrument: str) -> tuple[bool, int]:
    """Detect when a claim's 'instrument' is really a basket of several symbols.

    The strategy tester runs one symbol per backtest, so a portfolio-level claim
    (e.g. "BTC,ETH,XRP,GOLD,OIL,SPX,AAPL,FX") is not reproducible as a single test.
    Returns (is_multi, n_assets). A single forex pair like "EUR/USD" is NOT multi
    (no comma, no basket keyword).
    """
    s = (instrument or "").strip()
    low = s.lower()
    if "portfolio" in low or "basket" in low:
        parts = [p for p in re.split(r"[,/]|\band\b", s) if p.strip()]
        return True, max(len(parts), 2)
    parts = [p for p in s.split(",") if p.strip()]
    if len(parts) >= 2:
        return True, len(parts)
    return False, 1


def _failed(claim: Claim, reason: str) -> ValidationRun:
    return ValidationRun(
        claim_id=claim.id,
        test_type="strategy_backtest",
        timeframe="n/a",
        status="error",
        tradingview_query="",
        data_summary="",
        occurrences=None,
        hit_rate=None,
        result=f"untestable — {reason}",
        caveats=[],
        error=reason,
    )
