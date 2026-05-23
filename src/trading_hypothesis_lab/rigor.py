"""rigor.py — the strategy fact-checker / rigor benchmark.

Input: a pasted Pine v5/v6 STRATEGY script + a test universe (instrument, timeframe).
Output: a RigorReport scorecard that says whether the backtest is TRUSTWORTHY — NOT
whether the strategy is profitable. It scores where the strategy is fooling you.

Tier 1 — hard gates (fail any -> Untrustworthy):
  G1 look-ahead bias, G2 repainting, G3 insufficient sample, G4 friction illusion.
Tier 2 — scored dimensions (the anti-overfitting axes):
  D1 out-of-sample, D2 overfitting surface, D3 multi-timeframe, D4 multi-instrument,
  D5 inverse-edge / luck, D6 risk profile, D7 cost survival.

How the empirical checks run: the script is mapped to a known strategy archetype and
re-implemented in a small lookahead-free Python engine, then backtested on free OHLCV
(see backtest.py / empirical.py / freedata.py). This is a *faithful model of the strategy
logic*, not the literal Pine on TradingView — the report says so. It buys us deterministic,
offline checks that the capped TradingView trade feed could never give: a real out-of-sample
date split, a real inverse-edge run, real multi-timeframe / multi-instrument re-runs.

TradingView is now OPTIONAL: if an MCP client is supplied we use it only to verify the Pine
actually compiles. If the strategy maps to no known archetype, or no free data is available,
the empirical dimensions report `not_assessed` — which caps the grade below Robust. The
static gates (G1/G2), the Sonnet 4.6 source review, and the subjective grade always run.

See docs/rigor_rubric.md.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from . import llm as llm_mod
from .backtest import infer_spec
from .config import Config, load_config
from .empirical import EmpiricalResults, run_python_empirical
from .freedata import normalize_timeframe
from .mcp_client import McpClient, McpError
from .pine import _clear_chart_indicators, _compile_once
from .types import RigorCheck, RigorReport, StrategyBacktestMetrics

# --- static-analysis patterns -------------------------------------------------
_RE_LOOKAHEAD_ON = re.compile(r"barmerge\.lookahead_on", re.IGNORECASE)
_RE_SECURITY = re.compile(r"(request\.)?security\s*\(", re.IGNORECASE)
_RE_CALC_EVERY_TICK = re.compile(r"calc_on_every_tick\s*=\s*true", re.IGNORECASE)
_RE_INPUT = re.compile(r"\binput\.\w+\s*\(", re.IGNORECASE)

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
    mcp: McpClient | None = None,
    *,
    extra_instruments: list[str] | None = None,
    script_path: str | None = None,
) -> RigorReport:
    """Run the full rigor battery on a pasted Pine strategy. Never raises.

    `mcp` is optional and used only to verify the Pine compiles; the empirical battery runs
    on the TradingView-independent Python engine regardless.
    """
    instrument = instrument or "SPY"
    timeframe = normalize_timeframe(timeframe or "D")
    extra_instruments = extra_instruments or []
    holds_pf = config.strategy_holds_profit_factor
    fails_pf = config.strategy_fails_profit_factor
    min_trades = config.strategy_min_trades

    # === static gates (no data needed) ===
    gates: list[RigorCheck] = [_gate_lookahead(script), _gate_repaint(script)]

    # === optional Pine compile verification (TradingView) ===
    compiled, compile_errors, compile_checked = _verify_compile(script, mcp)

    # === empirical battery on the Python engine (TradingView-independent) ===
    spec = infer_spec(script, config)
    emp = run_python_empirical(spec, instrument, timeframe, extra_instruments, config)
    baseline = emp.baseline

    gates.append(_gate_sample(emp, min_trades))
    gates.append(_gate_friction(emp, fails_pf))

    # === scored dimensions ===
    dims: list[RigorCheck] = [
        _dim_oos(emp, fails_pf),
        _dim_overfit_surface(script, baseline),
        _dim_multi_tf(emp, fails_pf),
        _dim_multi_instrument(emp, extra_instruments, fails_pf),
        _dim_inverse_edge(emp, holds_pf),
        _dim_risk(baseline),
        _dim_cost_survival(emp, fails_pf),
    ]

    # Sonnet 4.6 deep read of the source — advisory, TV-independent, never blocks.
    reviews = _llm_review(script, config)

    score, max_score, grade, verdict = _score(
        gates, dims, reviews, baseline, script, config, compiled, compile_errors, compile_checked)

    engine_note = _engine_note(emp, compile_checked, compiled)
    report = RigorReport(
        instrument=instrument, timeframe=timeframe, compiled=compiled,
        gates=gates, dimensions=dims, reviews=reviews, score=score, max_score=max_score,
        grade=grade, verdict=verdict, engine_note=engine_note,
        script_path=script_path, markdown="", json={},
    )
    report.markdown = _to_markdown(report, compile_errors, compile_checked)
    report.json = _to_json(report, emp, compile_checked)
    return report


def _verify_compile(script: str, mcp: McpClient | None) -> tuple[bool, list[str], bool]:
    """Return (compiled, errors, checked). When no MCP is available we can't verify Pine
    syntax, so checked=False and we don't penalise — the engine still models the logic."""
    if mcp is None:
        return True, [], False
    try:
        _clear_chart_indicators(mcp)
        errors = _compile_once(script, mcp)
    except McpError as e:
        return True, [f"compile not verified (TradingView MCP error: {e})"], False
    return (not errors), errors, True


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


def _gate_sample(emp: EmpiricalResults, min_trades: int) -> RigorCheck:
    if not emp.available or emp.baseline is None:
        return RigorCheck("G3", "Sufficient sample", "gate", "not_assessed",
                          emp.note)
    n = emp.baseline.total_trades
    if n < min_trades:
        return RigorCheck("G3", "Sufficient sample", "gate", "fail",
                          f"only {n} trades over the tested history — below the {min_trades}-trade "
                          f"floor; the result is anecdote, not evidence.", f"{n} trades")
    return RigorCheck("G3", "Sufficient sample", "gate", "pass",
                      f"{n} trades — enough to draw a conclusion.", f"{n} trades")


def _gate_friction(emp: EmpiricalResults, fails_pf: float) -> RigorCheck:
    if not emp.available or emp.baseline is None:
        return RigorCheck("G4", "Friction illusion", "gate", "not_assessed",
                          "no baseline backtest to stress with costs.")
    if emp.had_costs:
        return RigorCheck("G4", "Friction illusion", "gate", "pass",
                          "script already models commission/slippage.", "costs modeled")
    if emp.cost_variant is None:
        return RigorCheck("G4", "Friction illusion", "gate", "not_assessed",
                          "could not re-run with costs added.")
    base, cost = emp.baseline.profit_factor, emp.cost_variant.profit_factor
    if base >= 1.0 and cost < fails_pf:
        return RigorCheck("G4", "Friction illusion", "gate", "fail",
                          f"edge vanishes once realistic costs are added: PF {base:.2f} -> {cost:.2f}. "
                          f"The 'edge' is a frictionless artifact.", f"PF {base:.2f}->{cost:.2f}")
    return RigorCheck("G4", "Friction illusion", "gate", "pass",
                      f"edge survives realistic costs (PF {base:.2f} -> {cost:.2f}).",
                      f"PF {base:.2f}->{cost:.2f}")


# ---------------------------------------------------------------------------
# Tier 2 — scored dimensions
# ---------------------------------------------------------------------------


def _dim_oos(emp: EmpiricalResults, fails_pf: float) -> RigorCheck:
    if not emp.available or emp.oos is None:
        return RigorCheck("D1", "Out-of-sample", "dimension", "not_assessed",
                          "not enough history to hold out a meaningful out-of-sample window."
                          if emp.available else "no backtest to split.")
    is_pf, oos_pf, n_oos = emp.oos
    if oos_pf >= fails_pf and oos_pf >= 0.6 * is_pf:
        return RigorCheck("D1", "Out-of-sample", "dimension", "pass",
                          f"edge holds out-of-sample: in-sample PF {is_pf:.2f}, OOS PF {oos_pf:.2f} "
                          f"over {n_oos} OOS trades.", f"IS {is_pf:.2f} / OOS {oos_pf:.2f}")
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


def _dim_multi_tf(emp: EmpiricalResults, fails_pf: float) -> RigorCheck:
    if not emp.available or not emp.multi_tf:
        return RigorCheck("D3", "Multi-timeframe", "dimension", "not_assessed",
                          "neighbouring timeframes produced no comparable result (data window or no trades).")
    results = [f"{tf}: PF {m.profit_factor:.2f}" for tf, m in emp.multi_tf]
    survived = sum(1 for _, m in emp.multi_tf if m.profit_factor >= fails_pf)
    metric = ", ".join(results)
    if survived >= 1:
        return RigorCheck("D3", "Multi-timeframe", "dimension", "pass",
                          f"edge persists on a neighbouring timeframe ({metric}) — not a single-TF cherry-pick.",
                          metric)
    return RigorCheck("D3", "Multi-timeframe", "dimension", "fail",
                      f"edge exists only on {emp.timeframe}; neighbours collapse ({metric}) — timeframe cherry-pick.",
                      metric)


def _dim_multi_instrument(emp: EmpiricalResults, extra_instruments, fails_pf: float) -> RigorCheck:
    if not emp.available:
        return RigorCheck("D4", "Multi-instrument", "dimension", "not_assessed", "no baseline to generalize.")
    if not emp.multi_instrument:
        msg = ("no comparable instruments supplied — pass --instruments to test generalization."
               if not extra_instruments
               else "supplied instruments produced no comparable result (no data or no trades).")
        return RigorCheck("D4", "Multi-instrument", "dimension", "not_assessed", msg)
    results = [f"{sym}: PF {m.profit_factor:.2f}" for sym, m in emp.multi_instrument]
    survived = sum(1 for _, m in emp.multi_instrument if m.profit_factor >= fails_pf)
    metric = ", ".join(results)
    if survived >= 1:
        return RigorCheck("D4", "Multi-instrument", "dimension", "pass",
                          f"edge generalizes to another instrument ({metric}).", metric)
    return RigorCheck("D4", "Multi-instrument", "dimension", "fail",
                      f"edge is single-instrument only ({metric}) — symbol cherry-pick.", metric)


def _dim_inverse_edge(emp: EmpiricalResults, holds_pf: float) -> RigorCheck:
    if not emp.available or emp.baseline is None or emp.inverse is None:
        return RigorCheck("D5", "Inverse-edge / luck", "dimension", "not_assessed",
                          "could not run the inverse strategy.")
    base_pf = emp.baseline.profit_factor
    inv_pf = emp.inverse.profit_factor
    if inv_pf >= holds_pf:
        return RigorCheck("D5", "Inverse-edge / luck", "dimension", "fail",
                          f"reversing every entry/exit ALSO clears the bar (inverse PF {inv_pf:.2f} vs "
                          f"strategy PF {base_pf:.2f}) — the signal looks like noise, not a directional edge.",
                          f"PF {base_pf:.2f} / inverse {inv_pf:.2f}")
    return RigorCheck("D5", "Inverse-edge / luck", "dimension", "pass",
                      f"the inverse strategy loses (inverse PF {inv_pf:.2f} vs strategy PF {base_pf:.2f}) — "
                      f"the directional edge is real, not coin-flip.",
                      f"PF {base_pf:.2f} / inverse {inv_pf:.2f}")


def _dim_risk(baseline: StrategyBacktestMetrics | None) -> RigorCheck:
    if baseline is None:
        return RigorCheck("D6", "Risk profile", "dimension", "not_assessed",
                          "no baseline drawdown to assess.")
    dd = baseline.max_drawdown
    net = baseline.net_profit
    if net <= 0:
        return RigorCheck("D6", "Risk profile", "dimension", "fail",
                          f"net result is non-positive ({net:+,.2f}%) — no return to weigh against "
                          f"{dd:,.2f}% drawdown.", f"net {net:+,.2f}% / DD {dd:,.2f}%")
    if dd <= 0:
        return RigorCheck("D6", "Risk profile", "dimension", "warn",
                          "drawdown reported as zero — likely too few trades to be meaningful.",
                          f"net {net:+,.2f}% / DD {dd:,.2f}%")
    ratio = net / dd
    if ratio >= 2:
        return RigorCheck("D6", "Risk profile", "dimension", "pass",
                          f"return/drawdown {ratio:.1f} (net {net:+,.2f}% vs max DD {dd:,.2f}%).",
                          f"return/DD {ratio:.1f}")
    if ratio >= 1:
        return RigorCheck("D6", "Risk profile", "dimension", "warn",
                          f"return/drawdown {ratio:.1f} — thin reward for the risk taken.",
                          f"return/DD {ratio:.1f}")
    return RigorCheck("D6", "Risk profile", "dimension", "fail",
                      f"return/drawdown {ratio:.1f} — net profit is dwarfed by the drawdown.",
                      f"return/DD {ratio:.1f}")


def _dim_cost_survival(emp: EmpiricalResults, fails_pf: float) -> RigorCheck:
    if not emp.available or emp.baseline is None:
        return RigorCheck("D7", "Cost survival", "dimension", "not_assessed", "no baseline to cost-stress.")
    if emp.had_costs:
        return RigorCheck("D7", "Cost survival", "dimension", "pass",
                          "commission/slippage already modeled in the script.", "costs modeled")
    if emp.cost_variant is None:
        return RigorCheck("D7", "Cost survival", "dimension", "not_assessed",
                          "could not re-run with costs added.")
    base_pf, cost_pf = emp.baseline.profit_factor, emp.cost_variant.profit_factor
    if cost_pf >= fails_pf:
        return RigorCheck("D7", "Cost survival", "dimension", "pass",
                          f"edge mostly survives realistic costs (PF {base_pf:.2f} -> {cost_pf:.2f}, "
                          f"gave back {base_pf - cost_pf:.2f}).", f"PF {base_pf:.2f}->{cost_pf:.2f}")
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
# scoring
# ---------------------------------------------------------------------------


def _score(gates, dims, reviews, baseline, script, config, compiled, compile_errors, compile_checked):
    """Return (score_0_100, max_score_0_100, grade, verdict).

    Objective disqualifiers (verified does-not-compile, look-ahead, repaint) are
    deterministic — a tainted backtest is mechanically invalid, no judgment needed.
    Everything else is graded SUBJECTIVELY by Sonnet 4.6, in light of the strategy's actual
    type, rather than blanket thresholds. Falls back to the deterministic sum with no LLM.
    """
    if compile_checked and not compiled:
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
                f"net {baseline.net_profit:+.2f}%, PF {baseline.profit_factor:.2f}, "
                f"max drawdown {baseline.max_drawdown:.2f}%")
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
        "drawdowns — do NOT apply blanket thresholds. A check that is 'not_assessed' is "
        "absence of evidence, not evidence of rigor — it should cap the grade below Robust. "
        "Weigh the evidence holistically and explain what was decisive. Output ONLY JSON."
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
        return (0, 100, _GRADE_UNTRUSTWORTHY,
                "no empirical dimension could be assessed (no archetype/data and no LLM judge) — "
                "rigor undetermined; treat as untrustworthy until it can be tested")
    # not_assessed dimensions count as absence of evidence: 0 in the numerator but still in the
    # denominator, so a check that never ran lowers the score (you can't earn rigor you didn't show).
    pct = sum(points[d.status] for d in assessed) / (2 * len(dims))
    score = round(pct * 100)
    if pct >= 0.8 and not not_assessed:
        grade = _GRADE_ROBUST
    elif pct >= 0.5:
        grade = _GRADE_FRAGILE
    else:
        grade = _GRADE_OVERFIT
    return score, 100, grade, f"{score}/100 (deterministic fallback; no LLM judge)"


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

_STATUS_MARK = {"pass": "PASS", "fail": "FAIL", "warn": "WARN", "not_assessed": "n/a"}


def _engine_note(emp: EmpiricalResults, compile_checked: bool, compiled: bool) -> str:
    parts = [emp.note]
    if emp.spec is not None and emp.spec.supported:
        parts.append(f"archetype inferred via {emp.spec.source} (confidence {emp.spec.confidence:.0%}).")
    if not compile_checked:
        parts.append("Pine syntax not verified (no TradingView connected); rigor reflects the strategy logic.")
    elif compiled:
        parts.append("Pine verified to compile on TradingView.")
    return " ".join(p for p in parts if p)


def _to_markdown(report: RigorReport, compile_errors: list[str], compile_checked: bool) -> str:
    lines: list[str] = []
    lines.append(f"# Rigor scorecard — {report.instrument} {report.timeframe}")
    lines.append("")
    lines.append(f"**Grade: {report.grade}** ({report.score}/100) — {report.verdict}")
    lines.append("")
    lines.append("> Scores how *trustworthy* the backtest is, not whether the strategy is profitable.")
    lines.append("")
    if report.engine_note:
        lines.append(f"*Engine: {report.engine_note}*")
        lines.append("")

    if compile_checked and not report.compiled:
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
                 "disqualifiers; dimensions are scored. Empirical checks run a faithful Python "
                 "re-implementation of the strategy on free OHLCV. See docs/rigor_rubric.md.*")
    return "\n".join(lines) + "\n"


def _md(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ").strip()


def _to_json(report: RigorReport, emp: EmpiricalResults, compile_checked: bool) -> dict:
    def _c(c: RigorCheck) -> dict:
        return {"id": c.id, "name": c.name, "tier": c.tier,
                "status": c.status, "detail": c.detail, "metric": c.metric}
    baseline = None
    if emp.baseline is not None:
        b = emp.baseline
        baseline = {"total_trades": b.total_trades, "win_rate": round(b.win_rate, 3),
                    "net_profit_pct": round(b.net_profit, 2), "profit_factor": round(b.profit_factor, 2),
                    "max_drawdown_pct": round(b.max_drawdown, 2)}
    return {
        "instrument": report.instrument,
        "timeframe": report.timeframe,
        "engine": emp.engine,
        "engine_note": report.engine_note,
        "archetype": emp.spec.archetype if emp.spec else None,
        "archetype_confidence": round(emp.spec.confidence, 2) if emp.spec else None,
        "pine_compile_verified": compile_checked,
        "compiled": report.compiled,
        "baseline": baseline,
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
# CLI
# ---------------------------------------------------------------------------


def _emit(text: str) -> None:
    """Write to stdout as UTF-8, bypassing a narrow console codec (e.g. Windows cp1252)
    so an LLM verdict containing a '->' arrow or smart quote can't crash the CLI."""
    data = (text + "\n").encode("utf-8", "replace")
    buf = getattr(sys.stdout, "buffer", None)
    if buf is not None:
        buf.write(data)
        buf.flush()
    else:  # pragma: no cover - unusual stdout replacement
        sys.stdout.write(text + "\n")


def _open_mcp(config: Config):
    """Open a TradingView MCP client if one is configured AND reachable; else None.
    The rigor checker runs fully without it (compile-verification is the only TV use)."""
    if not (config.tradingview_mcp_url or config.tradingview_mcp_cmd):
        return None
    try:
        client = McpClient(config)
        client.__enter__()
        return client
    except Exception:  # noqa: BLE001 — TV is optional; never let it block the check
        return None


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m trading_hypothesis_lab.rigor",
        description="Fact-check a pasted Pine strategy: is the backtest trustworthy?",
    )
    ap.add_argument("script", help="path to a .pine file, or '-' to read from stdin")
    ap.add_argument("--instrument", "-i", default="SPY", help="symbol to test on (e.g. SPY, BINANCE:BTCUSDT)")
    ap.add_argument("--timeframe", "-t", default="D", help="timeframe (e.g. 5, 60, 240, D, W)")
    ap.add_argument("--instruments", default="", help="comma-separated extra symbols for the multi-instrument check")
    ap.add_argument("--no-tv", action="store_true", help="skip TradingView compile-verification entirely")
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
    mcp = None if args.no_tv else _open_mcp(config)
    try:
        report = check(script, args.instrument, args.timeframe, config, mcp,
                       extra_instruments=extra, script_path=script_path)
    finally:
        if mcp is not None:
            try:
                mcp.__exit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass

    _emit(report.markdown)
    if args.out:
        Path(args.out).write_text(report.markdown, encoding="utf-8")
        Path(args.out).with_suffix(".json").write_text(
            json.dumps(report.json, indent=2), encoding="utf-8")
        _emit(f"\n(saved: {args.out})")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
