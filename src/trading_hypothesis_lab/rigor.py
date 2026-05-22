"""rigor.py — the strategy fact-checker / rigor benchmark.

Input: a pasted Pine v5/v6 STRATEGY script + a test universe (instrument, timeframe).
Output: a RigorReport scorecard that says whether the backtest is TRUSTWORTHY — NOT
whether the strategy is profitable. It scores where the strategy is fooling you.

Tier 1 — hard gates (fail any -> Untrustworthy, returns not even scored):
  G1 look-ahead bias, G2 repainting, G3 insufficient sample, G4 friction illusion.
Tier 2 — scored dimensions (the anti-overfitting axes):
  D1 out-of-sample, D2 overfitting surface, D3 multi-timeframe, D4 multi-instrument,
  D5 inverse-edge / luck, D6 risk profile, D7 cost survival.

Static checks read the source. Empirical checks compile the script and drive the
TradingView strategy tester through the MCP. Any MCP failure degrades a check to
`not_assessed` — the tool never crashes and never invents a result. A dimension that
is not_assessed caps the maximum grade: you can't be rated "robust" on a check that
never ran.

See docs/rigor_rubric.md.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from . import llm as llm_mod
from .config import Config, load_config
from .mcp_client import McpClient, McpError
from .pine import (
    _clear_chart_indicators,
    _compile_once,
    _run_backtest,
)
from .types import RigorCheck, RigorReport, StrategyBacktestMetrics

# --- static-analysis patterns -------------------------------------------------
_RE_LOOKAHEAD_ON = re.compile(r"barmerge\.lookahead_on", re.IGNORECASE)
_RE_SECURITY = re.compile(r"(request\.)?security\s*\(", re.IGNORECASE)
_RE_CALC_EVERY_TICK = re.compile(r"calc_on_every_tick\s*=\s*true", re.IGNORECASE)
_RE_INPUT = re.compile(r"\binput\.\w+\s*\(", re.IGNORECASE)
_RE_COMMISSION = re.compile(r"commission_(type|value)\s*=", re.IGNORECASE)
_RE_SLIPPAGE = re.compile(r"\bslippage\s*=", re.IGNORECASE)

# timeframe ladder for neighbour selection (TradingView resolution strings)
_TF_LADDER = ["1", "5", "15", "30", "60", "120", "240", "D", "W"]
_TF_ALIASES = {
    "1m": "1", "5m": "5", "15m": "15", "30m": "30",
    "1h": "60", "60m": "60", "2h": "120", "4h": "240",
    "1d": "D", "d": "D", "1w": "W", "w": "W",
}

# default realistic costs injected when a script is frictionless
_COST_PARAMS = ("commission_type=strategy.commission.percent, "
                "commission_value=0.05, slippage=1")

_GRADE_ROBUST = "Robust"
_GRADE_FRAGILE = "Promising but fragile"
_GRADE_OVERFIT = "Likely overfit"
_GRADE_UNTRUSTWORTHY = "Untrustworthy"
_GRADE_NO_COMPILE = "Does not compile"


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------


def check(
    script: str,
    instrument: str,
    timeframe: str,
    config: Config,
    mcp: McpClient,
    *,
    extra_instruments: list[str] | None = None,
    script_path: str | None = None,
) -> RigorReport:
    """Run the full rigor battery on a pasted Pine strategy. Never raises."""
    instrument = instrument or "SPY"
    timeframe = _normalize_tf(timeframe or "D")
    extra_instruments = extra_instruments or []
    holds_pf = config.strategy_holds_profit_factor
    fails_pf = config.strategy_fails_profit_factor
    min_trades = config.strategy_min_trades

    # === static gates (no MCP needed) ===
    gates: list[RigorCheck] = []
    gates.append(_gate_lookahead(script))
    gates.append(_gate_repaint(script))

    # === compile (no auto-fix: the user's script is the artifact under test) ===
    compiled = False
    compile_errors: list[str] = []
    try:
        _clear_chart_indicators(mcp)
        compile_errors = _compile_once(script, mcp)
        compiled = not compile_errors
    except McpError as e:
        compile_errors = [f"MCP unavailable: {e}"]

    baseline: StrategyBacktestMetrics | None = None
    if compiled:
        baseline = _safe_backtest(instrument, timeframe, mcp)

    # === remaining gates (empirical) ===
    gates.append(_gate_sample(baseline, compiled, min_trades))
    cost_variant, had_costs = _cost_variant_metrics(script, instrument, timeframe, mcp, compiled)
    gates.append(_gate_friction(baseline, cost_variant, had_costs, fails_pf))

    # === scored dimensions ===
    dims: list[RigorCheck] = []
    dims.append(_dim_oos(mcp, compiled, baseline, fails_pf))
    dims.append(_dim_overfit_surface(script, baseline))
    dims.append(_dim_multi_tf(script, instrument, timeframe, mcp, compiled, baseline, fails_pf))
    dims.append(_dim_multi_instrument(instrument, timeframe, mcp, compiled, baseline,
                                      extra_instruments, fails_pf))
    dims.append(_dim_inverse_edge(baseline, holds_pf))
    dims.append(_dim_risk(baseline))
    dims.append(_dim_cost_survival(baseline, cost_variant, had_costs, fails_pf))

    # Sonnet 4.6 deep read of the source — advisory, TV-independent, never blocks.
    reviews = _llm_review(script, config)

    score, max_score, grade, verdict = _score(
        gates, dims, reviews, baseline, script, config, compiled, compile_errors)
    report = RigorReport(
        instrument=instrument, timeframe=timeframe, compiled=compiled,
        gates=gates, dimensions=dims, reviews=reviews, score=score, max_score=max_score,
        grade=grade, verdict=verdict, script_path=script_path, markdown="", json={},
    )
    report.markdown = _to_markdown(report, compile_errors)
    report.json = _to_json(report)
    return report


# ---------------------------------------------------------------------------
# Tier 1 — hard gates
# ---------------------------------------------------------------------------


def _gate_lookahead(script: str) -> RigorCheck:
    if _RE_LOOKAHEAD_ON.search(script):
        return RigorCheck("G1", "Look-ahead bias", "gate", "fail",
                          "uses barmerge.lookahead_on — the strategy can see future "
                          "higher-timeframe data; the backtest is invalid.",
                          "barmerge.lookahead_on")
    if _RE_SECURITY.search(script) and "[1]" not in script:
        return RigorCheck("G1", "Look-ahead bias", "gate", "warn",
                          "uses request.security() without a confirmed-bar offset ([1]); "
                          "verify it reads only completed higher-timeframe bars.",
                          "security() without [1]")
    return RigorCheck("G1", "Look-ahead bias", "gate", "pass",
                      "no future-data access detected (no lookahead_on, no unguarded MTF read).")


def _gate_repaint(script: str) -> RigorCheck:
    if _RE_CALC_EVERY_TICK.search(script):
        return RigorCheck("G2", "Repainting", "gate", "fail",
                          "calc_on_every_tick=true — orders evaluate intrabar, so the "
                          "historical fill can use the eventual bar high/low. Backtest is optimistic.",
                          "calc_on_every_tick=true")
    return RigorCheck("G2", "Repainting", "gate", "pass",
                      "evaluates on confirmed bars (calc_on_every_tick default false).")


def _gate_sample(baseline: StrategyBacktestMetrics | None, compiled: bool, min_trades: int) -> RigorCheck:
    if not compiled or baseline is None:
        return RigorCheck("G3", "Sufficient sample", "gate", "not_assessed",
                          "no backtest result (script did not compile or produced no trades).")
    if baseline.total_trades < min_trades:
        return RigorCheck("G3", "Sufficient sample", "gate", "fail",
                          f"only {baseline.total_trades} trades — below the {min_trades}-trade floor; "
                          "the result is anecdote, not evidence.",
                          f"{baseline.total_trades} trades")
    return RigorCheck("G3", "Sufficient sample", "gate", "pass",
                      f"{baseline.total_trades} trades — enough to draw a conclusion.",
                      f"{baseline.total_trades} trades")


def _gate_friction(baseline, cost_variant, had_costs, fails_pf) -> RigorCheck:
    if baseline is None:
        return RigorCheck("G4", "Friction illusion", "gate", "not_assessed",
                          "no baseline backtest to test against costs.")
    if had_costs:
        return RigorCheck("G4", "Friction illusion", "gate", "pass",
                          "script already models commission/slippage.", "costs modeled")
    if cost_variant is None:
        return RigorCheck("G4", "Friction illusion", "gate", "not_assessed",
                          "could not re-run with costs added.")
    if baseline.profit_factor >= 1.0 and cost_variant.profit_factor < fails_pf:
        return RigorCheck("G4", "Friction illusion", "gate", "fail",
                          f"edge vanishes once realistic costs are added: PF {baseline.profit_factor:.2f} "
                          f"-> {cost_variant.profit_factor:.2f}. The 'edge' is a frictionless artifact.",
                          f"PF {baseline.profit_factor:.2f}->{cost_variant.profit_factor:.2f}")
    return RigorCheck("G4", "Friction illusion", "gate", "pass",
                      f"edge survives realistic costs (PF {baseline.profit_factor:.2f} "
                      f"-> {cost_variant.profit_factor:.2f}).",
                      f"PF {baseline.profit_factor:.2f}->{cost_variant.profit_factor:.2f}")


# ---------------------------------------------------------------------------
# Tier 2 — scored dimensions
# ---------------------------------------------------------------------------


def _dim_oos(mcp, compiled, baseline, fails_pf) -> RigorCheck:
    if not compiled or baseline is None:
        return RigorCheck("D1", "Out-of-sample", "dimension", "not_assessed",
                          "no backtest to split.")
    split = _oos_via_trades(mcp)
    if split is None:
        return RigorCheck("D1", "Out-of-sample", "dimension", "not_assessed",
                          "could not obtain a per-trade list large enough to split in/out-of-sample "
                          "(the MCP trade feed is capped); a deep-backtest date split is the planned upgrade.")
    is_pf, oos_pf, n_oos = split
    if oos_pf >= fails_pf and oos_pf >= 0.6 * is_pf:
        return RigorCheck("D1", "Out-of-sample", "dimension", "pass",
                          f"edge holds out-of-sample: in-sample PF {is_pf:.2f}, OOS PF {oos_pf:.2f} "
                          f"over {n_oos} OOS trades.",
                          f"IS {is_pf:.2f} / OOS {oos_pf:.2f}")
    return RigorCheck("D1", "Out-of-sample", "dimension", "fail",
                      f"edge degrades out-of-sample: in-sample PF {is_pf:.2f} -> OOS PF {oos_pf:.2f} "
                      f"({n_oos} OOS trades) — sign of curve-fitting.",
                      f"IS {is_pf:.2f} / OOS {oos_pf:.2f}")


def _dim_overfit_surface(script: str, baseline: StrategyBacktestMetrics | None) -> RigorCheck:
    n_inputs = len(_RE_INPUT.findall(script))
    if baseline is None:
        return RigorCheck("D2", "Overfitting surface", "dimension", "not_assessed",
                          f"{n_inputs} tunable inputs, but no trade count to weigh degrees of freedom against.")
    if n_inputs == 0:
        return RigorCheck("D2", "Overfitting surface", "dimension", "pass",
                          "no tunable inputs — nothing to curve-fit.", "0 inputs")
    ratio = baseline.total_trades / n_inputs
    if ratio < 10:
        return RigorCheck("D2", "Overfitting surface", "dimension", "fail",
                          f"{n_inputs} tunable inputs vs {baseline.total_trades} trades "
                          f"({ratio:.0f} trades/param) — too many knobs for the data; high overfit risk.",
                          f"{n_inputs} inputs / {baseline.total_trades} trades")
    if ratio < 25:
        return RigorCheck("D2", "Overfitting surface", "dimension", "warn",
                          f"{n_inputs} tunable inputs vs {baseline.total_trades} trades "
                          f"({ratio:.0f} trades/param) — moderate degrees of freedom.",
                          f"{n_inputs} inputs / {baseline.total_trades} trades")
    return RigorCheck("D2", "Overfitting surface", "dimension", "pass",
                      f"{n_inputs} tunable inputs vs {baseline.total_trades} trades "
                      f"({ratio:.0f} trades/param) — low degrees of freedom.",
                      f"{n_inputs} inputs / {baseline.total_trades} trades")


def _dim_multi_tf(script, instrument, timeframe, mcp, compiled, baseline, fails_pf) -> RigorCheck:
    if not compiled or baseline is None:
        return RigorCheck("D3", "Multi-timeframe", "dimension", "not_assessed",
                          "no baseline to compare neighbouring timeframes against.")
    neighbours = _tf_neighbours(timeframe)
    results: list[str] = []
    survived = 0
    assessed = 0
    for tf in neighbours:
        m = _safe_backtest(instrument, tf, mcp)
        if m is None:
            continue
        assessed += 1
        results.append(f"{tf}: PF {m.profit_factor:.2f}")
        if m.profit_factor >= fails_pf:
            survived += 1
    if assessed == 0:
        return RigorCheck("D3", "Multi-timeframe", "dimension", "not_assessed",
                          "neighbouring timeframes produced no comparable result.")
    metric = ", ".join(results)
    if survived >= 1:
        return RigorCheck("D3", "Multi-timeframe", "dimension", "pass",
                          f"edge persists on a neighbouring timeframe ({metric}) — not a single-TF cherry-pick.",
                          metric)
    return RigorCheck("D3", "Multi-timeframe", "dimension", "fail",
                      f"edge exists only on {timeframe}; neighbours collapse ({metric}) — timeframe cherry-pick.",
                      metric)


def _dim_multi_instrument(instrument, timeframe, mcp, compiled, baseline,
                          extra_instruments, fails_pf) -> RigorCheck:
    if not compiled or baseline is None:
        return RigorCheck("D4", "Multi-instrument", "dimension", "not_assessed",
                          "no baseline to generalize.")
    if not extra_instruments:
        return RigorCheck("D4", "Multi-instrument", "dimension", "not_assessed",
                          "no comparable instruments supplied — pass --instruments to test generalization.")
    results: list[str] = []
    survived = 0
    assessed = 0
    for sym in extra_instruments:
        m = _safe_backtest(sym, timeframe, mcp)
        if m is None:
            continue
        assessed += 1
        results.append(f"{sym}: PF {m.profit_factor:.2f}")
        if m.profit_factor >= fails_pf:
            survived += 1
    if assessed == 0:
        return RigorCheck("D4", "Multi-instrument", "dimension", "not_assessed",
                          "supplied instruments produced no comparable result.")
    metric = ", ".join(results)
    if survived >= 1:
        return RigorCheck("D4", "Multi-instrument", "dimension", "pass",
                          f"edge generalizes to another instrument ({metric}).", metric)
    return RigorCheck("D4", "Multi-instrument", "dimension", "fail",
                      f"edge is single-instrument only ({metric}) — symbol cherry-pick.", metric)


def _dim_inverse_edge(baseline, holds_pf) -> RigorCheck:
    if baseline is None:
        return RigorCheck("D5", "Inverse-edge / luck", "dimension", "not_assessed",
                          "no baseline to invert.")
    pf = baseline.profit_factor
    if pf <= 0:
        return RigorCheck("D5", "Inverse-edge / luck", "dimension", "not_assessed",
                          "profit factor not computable.")
    inverse_pf = 1.0 / pf if pf > 0 else float("inf")
    if inverse_pf >= holds_pf:
        return RigorCheck("D5", "Inverse-edge / luck", "dimension", "fail",
                          f"reversing every entry/exit would also clear the bar (est. inverse PF "
                          f"{inverse_pf:.2f}) — the signal looks like noise, not a directional edge.",
                          f"PF {pf:.2f} / inverse {inverse_pf:.2f}")
    return RigorCheck("D5", "Inverse-edge / luck", "dimension", "pass",
                      f"the inverse strategy loses (est. inverse PF {inverse_pf:.2f}) — the directional "
                      f"edge is real, not coin-flip.",
                      f"PF {pf:.2f} / inverse {inverse_pf:.2f}")


def _dim_risk(baseline) -> RigorCheck:
    if baseline is None:
        return RigorCheck("D6", "Risk profile", "dimension", "not_assessed",
                          "no baseline drawdown to assess.")
    dd = baseline.max_drawdown
    net = baseline.net_profit
    if net <= 0:
        return RigorCheck("D6", "Risk profile", "dimension", "fail",
                          f"net result is non-positive ({net:+,.2f}) — no return to weigh against "
                          f"{dd:,.2f} drawdown.", f"net {net:+,.2f} / DD {dd:,.2f}")
    if dd <= 0:
        return RigorCheck("D6", "Risk profile", "dimension", "warn",
                          "drawdown reported as zero — likely too few trades to be meaningful.",
                          f"net {net:+,.2f} / DD {dd:,.2f}")
    ratio = net / dd
    if ratio >= 2:
        return RigorCheck("D6", "Risk profile", "dimension", "pass",
                          f"return/drawdown {ratio:.1f} (net {net:+,.2f} vs max DD {dd:,.2f}).",
                          f"return/DD {ratio:.1f}")
    if ratio >= 1:
        return RigorCheck("D6", "Risk profile", "dimension", "warn",
                          f"return/drawdown {ratio:.1f} — thin reward for the risk taken.",
                          f"return/DD {ratio:.1f}")
    return RigorCheck("D6", "Risk profile", "dimension", "fail",
                      f"return/drawdown {ratio:.1f} — net profit is dwarfed by the drawdown.",
                      f"return/DD {ratio:.1f}")


def _dim_cost_survival(baseline, cost_variant, had_costs, fails_pf) -> RigorCheck:
    if baseline is None:
        return RigorCheck("D7", "Cost survival", "dimension", "not_assessed",
                          "no baseline to cost-stress.")
    if had_costs:
        return RigorCheck("D7", "Cost survival", "dimension", "pass",
                          "commission/slippage already modeled in the script.", "costs modeled")
    if cost_variant is None:
        return RigorCheck("D7", "Cost survival", "dimension", "not_assessed",
                          "could not re-run with costs added.")
    base_pf, cost_pf = baseline.profit_factor, cost_variant.profit_factor
    if cost_pf >= fails_pf:
        give_back = base_pf - cost_pf
        return RigorCheck("D7", "Cost survival", "dimension", "pass",
                          f"edge mostly survives realistic costs (PF {base_pf:.2f} -> {cost_pf:.2f}, "
                          f"gave back {give_back:.2f}).", f"PF {base_pf:.2f}->{cost_pf:.2f}")
    return RigorCheck("D7", "Cost survival", "dimension", "fail",
                      f"most of the edge is eaten by costs (PF {base_pf:.2f} -> {cost_pf:.2f}).",
                      f"PF {base_pf:.2f}->{cost_pf:.2f}")


# ---------------------------------------------------------------------------
# Sonnet 4.6 deep review (advisory — TV-independent, never scored, never blocks)
# ---------------------------------------------------------------------------

_REVIEW_SYSTEM = (
    "You are a skeptical quantitative researcher reviewing a TradingView Pine strategy "
    "for BACKTEST RIGOR — the ways a backtest fools its author. You do NOT judge "
    "profitability; you judge trustworthiness. Be precise and cite the specific Pine "
    "construct. Output ONLY JSON, no prose."
)


def _llm_review(script: str, config: Config) -> list[RigorCheck]:
    """Sonnet 4.6 deep read of the Pine source for subtler rigor issues the regex gates
    miss. Advisory only (not scored). Returns [] if no LLM backend is available."""
    prompt = (
        "Review this Pine strategy for rigor / self-deception risks. Return JSON: "
        '{"checks":[{"id":"L1","name":"...","status":"pass|warn|fail",'
        '"detail":"one sentence; cite the Pine construct"}]}. '
        "Assess exactly these, in order:\n"
        "L1 Look-ahead / repaint — security() misuse, future references, "
        "calc_on_every_tick, request.*() with lookahead\n"
        "L2 Overfitting — curve-fit constants, many tunable inputs, oddly specific thresholds\n"
        "L3 Cherry-picking — hardcoded dates/sessions/symbols that bias the test window\n"
        "L4 Cost & fill realism — commission/slippage modeled? are limit/stop fills plausible?\n"
        "L5 Position sizing / risk — fixed oversized qty, no stop, martingale / averaging-down\n"
        "L6 Logic soundness — does the code implement a coherent edge or an incidental one?\n\n"
        f"```pine\n{script[:8000]}\n```"
    )
    try:
        raw = llm_mod.complete(_REVIEW_SYSTEM, prompt,
                               model=config.synthesize_model or None,
                               max_tokens=1500, timeout=180)
    except (llm_mod.LlmUnavailable, llm_mod.LlmError):
        return []
    return _parse_review(raw)


def _parse_review(raw: str) -> list[RigorCheck]:
    m = re.search(r"\{.*\}", raw or "", re.DOTALL)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return []
    out: list[RigorCheck] = []
    for c in data.get("checks", []):
        if not isinstance(c, dict):
            continue
        status = str(c.get("status", "")).lower()
        if status not in ("pass", "warn", "fail"):
            status = "warn"
        out.append(RigorCheck(
            id=str(c.get("id", f"L{len(out) + 1}")),
            name=str(c.get("name", "review"))[:60],
            tier="review",
            status=status,
            detail=str(c.get("detail", ""))[:300],
        ))
    return out


# ---------------------------------------------------------------------------
# empirical helpers
# ---------------------------------------------------------------------------


def _safe_backtest(instrument: str, timeframe: str, mcp: McpClient) -> StrategyBacktestMetrics | None:
    """Backtest the already-compiled chart strategy on (instrument, tf). None on any failure."""
    try:
        return _run_backtest(instrument, timeframe, mcp)
    except McpError:
        return None


def _cost_variant_metrics(script, instrument, timeframe, mcp, compiled):
    """Returns (cost_variant_metrics_or_None, had_costs). If the script already models
    costs, had_costs=True and we don't re-run. Otherwise inject costs, recompile, backtest."""
    had_costs = bool(_RE_COMMISSION.search(script) and _RE_SLIPPAGE.search(script))
    if had_costs or not compiled:
        return None, had_costs
    variant = _inject_costs(script)
    if variant is None:
        return None, had_costs
    try:
        _clear_chart_indicators(mcp)
        if _compile_once(variant, mcp):
            return None, had_costs
        m = _run_backtest(instrument, timeframe, mcp)
    except McpError:
        return None, had_costs
    # restore the original (uncosted) script on the chart for downstream TF/instrument runs
    try:
        _clear_chart_indicators(mcp)
        _compile_once(script, mcp)
    except McpError:
        pass
    return m, had_costs


def _inject_costs(script: str) -> str | None:
    """Insert realistic commission/slippage into the strategy() header. None if no header."""
    m = re.search(r"strategy\s*\(", script)
    if not m:
        return None
    i = m.end()  # just after the '('
    depth = 1
    while i < len(script) and depth > 0:
        c = script[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                break
        i += 1
    if depth != 0:
        return None
    return script[:i] + ", " + _COST_PARAMS + script[i:]


def _oos_via_trades(mcp: McpClient, oos_fraction: float = 0.3):
    """Pull the closed-trade list, split chronologically, return (is_pf, oos_pf, n_oos).
    None if the feed is too small / unusable to split reliably."""
    try:
        res = mcp.call("data_get_trades", {})
    except McpError:
        return None
    trades = _normalize_trades(res)
    if len(trades) < 20:  # too few (and the MCP feed is capped) to split meaningfully
        return None
    cut = int(len(trades) * (1 - oos_fraction))
    if cut < 5 or len(trades) - cut < 5:
        return None
    return _pf(trades[:cut]), _pf(trades[cut:]), len(trades) - cut


def _normalize_trades(res) -> list[float]:
    """Extract a chronological list of per-trade P&L from a data_get_trades response."""
    rows = None
    if isinstance(res, dict):
        for k in ("trades", "result", "data", "rows"):
            if isinstance(res.get(k), list):
                rows = res[k]
                break
    elif isinstance(res, list):
        rows = res
    if not rows:
        return []
    pnls: list[float] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        for k in ("profit", "pnl", "net_profit", "netProfit", "profit_abs", "realized_pnl"):
            if r.get(k) is not None:
                try:
                    pnls.append(float(r[k]))
                    break
                except (TypeError, ValueError):
                    pass
    return pnls


def _pf(pnls: list[float]) -> float:
    gross_win = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    if gross_loss == 0:
        return float("inf") if gross_win > 0 else 0.0
    return gross_win / gross_loss


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


def _score(gates, dims, reviews, baseline, script, config, compiled, compile_errors):
    """Return (score_0_100, max_score_0_100, grade, verdict).

    Objective disqualifiers (does-not-compile, look-ahead, repaint) are deterministic —
    a tainted backtest is mechanically invalid, no judgment needed. Everything else is
    graded SUBJECTIVELY by Sonnet 4.6, in light of the strategy's actual type, rather
    than blanket thresholds. Falls back to the deterministic sum when no LLM is present.
    """
    if not compiled:
        first = compile_errors[0] if compile_errors else "unknown compile error"
        return 0, 100, _GRADE_NO_COMPILE, f"script did not compile — {first}"

    objective_fail = [g for g in gates if g.status == "fail" and g.id in ("G1", "G2")]
    if objective_fail:
        names = ", ".join(f"{g.id} {g.name}" for g in objective_fail)
        return (0, 100, _GRADE_UNTRUSTWORTHY,
                f"objective disqualifier ({names}) — the backtest is mechanically invalid "
                f"regardless of returns")

    judged = _llm_judge(script, gates, dims, reviews, baseline, config)
    if judged is not None:
        score, grade, verdict = judged
        return score, 100, grade, verdict
    return _score_deterministic(gates, dims)


def _llm_judge(script, gates, dims, reviews, baseline, config):
    """Sonnet 4.6 synthesizes all evidence into a CONTEXTUAL rigor grade — weighing it
    against the strategy's type, not fixed thresholds. Returns (score, grade, verdict)
    or None if no LLM backend is available."""
    def fmt(checks):
        out = []
        for c in checks:
            metric = f" [{c.metric}]" if c.metric else ""
            out.append(f"- {c.id} {c.name}: {c.status.upper()} — {c.detail}{metric}")
        return "\n".join(out) or "(none)"

    if baseline is not None:
        base = (f"{baseline.total_trades} trades, win {baseline.win_rate:.0%}, "
                f"net {baseline.net_profit:+.2f}, PF {baseline.profit_factor:.2f}, "
                f"max drawdown {baseline.max_drawdown:.2f}")
    else:
        base = "no successful backtest (empirical dimensions not assessed)"

    evidence = (
        f"BASELINE BACKTEST: {base}\n\n"
        f"GATES:\n{fmt(gates)}\n\n"
        f"DIMENSIONS:\n{fmt(dims)}\n\n"
        f"SOURCE REVIEW:\n{fmt(reviews)}\n\n"
        f"STRATEGY SOURCE (excerpt):\n```pine\n{script[:3000]}\n```"
    )
    system = (
        "You grade the RIGOR (trustworthiness, NOT profitability) of a trading-strategy "
        "backtest. Judge SUBJECTIVELY and IN CONTEXT: a scalping strategy and a swing/"
        "position strategy have different acceptable trade counts, cost sensitivity, and "
        "drawdowns — do NOT apply blanket thresholds. Weigh the evidence holistically and "
        "explain what was decisive. Output ONLY JSON."
    )
    prompt = (
        "Grade this strategy's backtest rigor. First infer the strategy TYPE, then judge "
        "what rigor standard is appropriate for it, then weigh the evidence. Return JSON: "
        '{"grade":"Robust|Promising but fragile|Likely overfit|Untrustworthy",'
        '"score":<0-100 integer>,"verdict":"2-3 sentences naming the strategy type and the '
        'decisive factors"}.\n\n' + evidence
    )
    try:
        raw = llm_mod.complete(system, prompt, model=config.synthesize_model or None,
                               max_tokens=700, timeout=180)
    except (llm_mod.LlmUnavailable, llm_mod.LlmError):
        return None
    m = re.search(r"\{.*\}", raw or "", re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return None
    grade = str(data.get("grade", "")).strip() or _GRADE_FRAGILE
    try:
        score = int(round(float(data.get("score", 0))))
    except (TypeError, ValueError):
        score = 0
    score = max(0, min(100, score))
    verdict = str(data.get("verdict", "")).strip()[:600] or "graded by contextual review"
    return score, grade, verdict


def _score_deterministic(gates, dims):
    """Threshold fallback used only when no LLM backend is available."""
    failed_gates = [g for g in gates if g.status == "fail"]
    if failed_gates:
        names = ", ".join(f"{g.id} {g.name}" for g in failed_gates)
        return 0, 100, _GRADE_UNTRUSTWORTHY, f"hard gate failed ({names})"
    points = {"pass": 2, "warn": 1, "fail": 0}
    assessed = [d for d in dims if d.status in points]
    not_assessed = [d for d in dims if d.status == "not_assessed"]
    if not assessed:
        return 0, 0, _GRADE_OVERFIT, "no rigor dimension could be assessed (no LLM judge available)"
    pct = sum(points[d.status] for d in assessed) / (2 * len(assessed))
    score = round(pct * 100)
    if pct >= 0.8 and not not_assessed:
        grade = _GRADE_ROBUST
    elif pct >= 0.5:
        grade = _GRADE_FRAGILE
    else:
        grade = _GRADE_OVERFIT
    if not_assessed and grade == _GRADE_ROBUST:
        grade = _GRADE_FRAGILE
    return score, 100, grade, f"{score}/100 (deterministic fallback; no LLM judge)"


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

_STATUS_MARK = {"pass": "PASS", "fail": "FAIL", "warn": "WARN", "not_assessed": "n/a"}


def _to_markdown(report: RigorReport, compile_errors: list[str]) -> str:
    lines: list[str] = []
    lines.append(f"# Rigor scorecard — {report.instrument} {report.timeframe}")
    lines.append("")
    lines.append(f"**Grade: {report.grade}** ({report.score}/100) — {report.verdict}")
    lines.append("")
    lines.append("> Scores how *trustworthy* the backtest is, not whether the strategy is profitable.")
    lines.append("")

    if not report.compiled:
        lines.append("## Did not compile")
        lines.append("")
        for e in compile_errors[:5]:
            lines.append(f"- {e}")
        lines.append("")

    lines.append("## Tier 1 — hard gates")
    lines.append("")
    lines.append("| Gate | Result | Finding |")
    lines.append("|------|--------|---------|")
    for g in report.gates:
        lines.append(f"| {g.id} {g.name} | {_STATUS_MARK[g.status]} | {_md(g.detail)} |")
    lines.append("")

    lines.append("## Tier 2 — scored dimensions")
    lines.append("")
    lines.append("| Dimension | Result | Finding |")
    lines.append("|-----------|--------|---------|")
    for d in report.dimensions:
        lines.append(f"| {d.id} {d.name} | {_STATUS_MARK[d.status]} | {_md(d.detail)} |")
    lines.append("")

    if report.reviews:
        lines.append("## Deep review (Sonnet 4.6 — advisory, not scored)")
        lines.append("")
        lines.append("| Check | Result | Finding |")
        lines.append("|-------|--------|---------|")
        for r in report.reviews:
            lines.append(f"| {r.id} {r.name} | {_STATUS_MARK[r.status]} | {_md(r.detail)} |")
        lines.append("")

    not_assessed = [d for d in report.dimensions if d.status == "not_assessed"]
    if not_assessed:
        lines.append("## Not assessed")
        lines.append("")
        lines.append("These checks couldn't run, so they cap the grade below *Robust* — "
                     "you can't be rated robust on a check that never ran:")
        for d in not_assessed:
            lines.append(f"- **{d.id} {d.name}** — {d.detail}")
        lines.append("")

    lines.append("---")
    lines.append("*Generated by the TradingHypothesisLab rigor checker. Gates are pass/fail "
                 "disqualifiers; dimensions are scored. See docs/rigor_rubric.md.*")
    return "\n".join(lines) + "\n"


def _md(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ").strip()


def _to_json(report: RigorReport) -> dict:
    def _c(c: RigorCheck) -> dict:
        return {"id": c.id, "name": c.name, "tier": c.tier,
                "status": c.status, "detail": c.detail, "metric": c.metric}
    return {
        "instrument": report.instrument,
        "timeframe": report.timeframe,
        "compiled": report.compiled,
        "grade": report.grade,
        "score": report.score,
        "max_score": report.max_score,
        "verdict": report.verdict,
        "script_path": report.script_path,
        "gates": [_c(g) for g in report.gates],
        "dimensions": [_c(d) for d in report.dimensions],
        "reviews": [_c(r) for r in report.reviews],
    }


# ---------------------------------------------------------------------------
# timeframe helpers
# ---------------------------------------------------------------------------


def _normalize_tf(tf: str) -> str:
    t = str(tf).strip()
    return _TF_ALIASES.get(t.lower(), t.upper() if t.lower() in ("d", "w") else t)


def _tf_neighbours(tf: str) -> list[str]:
    tf = _normalize_tf(tf)
    if tf not in _TF_LADDER:
        return []
    i = _TF_LADDER.index(tf)
    out = []
    if i > 0:
        out.append(_TF_LADDER[i - 1])
    if i < len(_TF_LADDER) - 1:
        out.append(_TF_LADDER[i + 1])
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m trading_hypothesis_lab.rigor",
        description="Fact-check a pasted Pine strategy: is the backtest trustworthy?",
    )
    ap.add_argument("script", help="path to a .pine file, or '-' to read from stdin")
    ap.add_argument("--instrument", "-i", default="SPY", help="symbol to test on (e.g. SPY, BINANCE:BTCUSDT)")
    ap.add_argument("--timeframe", "-t", default="D", help="timeframe (e.g. 5, 60, 240, D, W)")
    ap.add_argument("--instruments", default="", help="comma-separated extra symbols for the multi-instrument check")
    ap.add_argument("--out", default="", help="optional path to write the scorecard markdown")
    args = ap.parse_args(argv)

    if args.script == "-":
        script = sys.stdin.read()
        script_path = None
    else:
        p = Path(args.script)
        script = p.read_text(encoding="utf-8")
        script_path = str(p.resolve())

    extra = [s.strip() for s in args.instruments.split(",") if s.strip()]
    config = load_config()
    with McpClient(config) as mcp:
        report = check(script, args.instrument, args.timeframe, config, mcp,
                       extra_instruments=extra, script_path=script_path)

    print(report.markdown)
    if args.out:
        Path(args.out).write_text(report.markdown, encoding="utf-8")
        Path(args.out).with_suffix(".json").write_text(
            json.dumps(report.json, indent=2), encoding="utf-8")
        print(f"\n(saved: {args.out})")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
