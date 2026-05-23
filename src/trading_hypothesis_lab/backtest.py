"""backtest.py — a small, lookahead-free backtest engine for common strategy archetypes.

The rigor checker needs to *run* a strategy to judge whether its backtest is trustworthy.
Driving the live TradingView strategy tester proved unreliable, so instead we:

  1. infer which well-known archetype a pasted Pine strategy implements (RSI mean-reversion,
     MA cross, MACD cross, Supertrend, Bollinger reversion, Donchian breakout, opening-range
     breakout) plus its parameters — via an LLM, with a regex fallback;
  2. re-implement that archetype here and backtest it on free OHLCV.

This is a *faithful model* of the strategy's core logic, not the literal Pine on TradingView.
The rigor report says so plainly. What it buys us: a deterministic, offline, costless engine
where we can do the things that actually matter for rigor — a real out-of-sample date split,
a real inverse-edge run, real multi-timeframe / multi-instrument re-runs, and a real
friction-stress — none of which the capped TradingView trade feed could give us.

No look-ahead, by construction: every signal for bar *t* is computed only from data up to and
including bar *t*'s close, and is then **executed at bar t+1's open**. Stops/targets are checked
against subsequent bars' highs/lows only after entry.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import llm as llm_mod
from .config import Config
from .types import StrategyBacktestMetrics

# archetypes the engine can model
ARCHETYPES = (
    "rsi_mean_reversion",
    "ma_cross",
    "macd_cross",
    "supertrend",
    "bollinger_mean_reversion",
    "donchian_breakout",
    "orb",
)

# default parameters per archetype (merged with whatever inference extracts)
_DEFAULTS: dict[str, dict] = {
    "rsi_mean_reversion": {"length": 14, "oversold": 30, "overbought": 70},
    "ma_cross": {"fast": 10, "slow": 30, "ma_type": "ema"},
    "macd_cross": {"fast": 12, "slow": 26, "signal": 9},
    "supertrend": {"atr_period": 10, "factor": 3.0},
    "bollinger_mean_reversion": {"length": 20, "mult": 2.0},
    "donchian_breakout": {"length": 20},
    "orb": {"or_minutes": 15},
}
_DEFAULT_SIDE = {
    "rsi_mean_reversion": "long",
    "ma_cross": "both",
    "macd_cross": "both",
    "supertrend": "both",
    "bollinger_mean_reversion": "long",
    "donchian_breakout": "both",
    "orb": "both",
}

# realistic friction used for the cost-stress runs (percent of price, per side)
DEFAULT_COMMISSION_PCT = 0.04
DEFAULT_SLIPPAGE_PCT = 0.02


@dataclass
class StrategySpec:
    """A normalised description of the strategy, sufficient to backtest it."""

    archetype: str
    side: str                       # "long" | "short" | "both"
    params: dict = field(default_factory=dict)
    sl_pct: float | None = None     # stop-loss, percent of entry (e.g. 2.0)
    tp_pct: float | None = None     # take-profit, percent of entry
    has_costs: bool = False         # the Pine already models commission/slippage
    confidence: float = 0.5         # how confident the inference is (0..1)
    source: str = "regex"           # "llm" | "regex"
    note: str = ""

    @property
    def supported(self) -> bool:
        return self.archetype in ARCHETYPES


# ---------------------------------------------------------------------------
# indicators (all backward-looking — no look-ahead)
# ---------------------------------------------------------------------------


def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(int(n)).mean()


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=int(n), adjust=False).mean()


def _rsi(close: pd.Series, n: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / int(n), adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / int(n), adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return rsi.fillna(50.0)


def _atr(df: pd.DataFrame, n: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / int(n), adjust=False).mean()


def _supertrend_dir(df: pd.DataFrame, atr_period: int, factor: float) -> np.ndarray:
    """Return the Supertrend direction series: +1 (uptrend) / -1 (downtrend)."""
    atr = _atr(df, atr_period).to_numpy()
    hl2 = ((df["high"] + df["low"]) / 2.0).to_numpy()
    close = df["close"].to_numpy()
    n = len(close)
    upper = hl2 + factor * atr
    lower = hl2 - factor * atr
    dir_ = np.ones(n, dtype=int)
    final_upper = np.full(n, np.nan)
    final_lower = np.full(n, np.nan)
    for i in range(n):
        if i == 0 or np.isnan(atr[i]):
            final_upper[i] = upper[i]
            final_lower[i] = lower[i]
            dir_[i] = 1
            continue
        final_upper[i] = (
            upper[i] if (upper[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1])
            else final_upper[i - 1]
        )
        final_lower[i] = (
            lower[i] if (lower[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1])
            else final_lower[i - 1]
        )
        if close[i] > final_upper[i - 1]:
            dir_[i] = 1
        elif close[i] < final_lower[i - 1]:
            dir_[i] = -1
        else:
            dir_[i] = dir_[i - 1]
    return dir_


def _crossover(a: pd.Series, b: pd.Series) -> np.ndarray:
    """True where a crosses above b (a was <= b, now > b)."""
    a_prev, b_prev = a.shift(1), b.shift(1)
    out = (a_prev <= b_prev) & (a > b)
    return out.fillna(False).to_numpy()


def _crossunder(a: pd.Series, b: pd.Series) -> np.ndarray:
    a_prev, b_prev = a.shift(1), b.shift(1)
    out = (a_prev >= b_prev) & (a < b)
    return out.fillna(False).to_numpy()


def _const(value: float, like: pd.Series) -> pd.Series:
    return pd.Series(value, index=like.index)


# ---------------------------------------------------------------------------
# archetype signals -> (enter_long, exit_long, enter_short, exit_short)
# each is a numpy bool array; the decision at bar t uses only data through t
# ---------------------------------------------------------------------------


def _sig_rsi(df, p, _bm):
    rsi = _rsi(df["close"], p["length"])
    os_, ob = _const(p["oversold"], rsi), _const(p["overbought"], rsi)
    el = _crossover(rsi, os_)      # rsi recovers up through oversold -> buy the dip
    xl = _crossover(rsi, ob)       # rsi reaches overbought -> take profit
    es = _crossunder(rsi, ob)
    xs = _crossunder(rsi, os_)
    return el, xl, es, xs


def _sig_ma_cross(df, p, _bm):
    f = _ema(df["close"], p["fast"]) if p.get("ma_type", "ema") == "ema" else _sma(df["close"], p["fast"])
    s = _ema(df["close"], p["slow"]) if p.get("ma_type", "ema") == "ema" else _sma(df["close"], p["slow"])
    up, dn = _crossover(f, s), _crossunder(f, s)
    return up, dn, dn, up


def _sig_macd(df, p, _bm):
    macd = _ema(df["close"], p["fast"]) - _ema(df["close"], p["slow"])
    signal = _ema(macd, p["signal"])
    up, dn = _crossover(macd, signal), _crossunder(macd, signal)
    return up, dn, dn, up


def _sig_supertrend(df, p, _bm):
    d = _supertrend_dir(df, p["atr_period"], p["factor"])
    flip_up = np.zeros(len(d), bool)
    flip_dn = np.zeros(len(d), bool)
    flip_up[1:] = (d[1:] == 1) & (d[:-1] == -1)
    flip_dn[1:] = (d[1:] == -1) & (d[:-1] == 1)
    return flip_up, flip_dn, flip_dn, flip_up


def _sig_bollinger(df, p, _bm):
    basis = _sma(df["close"], p["length"])
    dev = p["mult"] * df["close"].rolling(int(p["length"])).std()
    upper, lower = basis + dev, basis - dev
    el = _crossunder(df["close"], lower)    # poke below lower band -> long the reversion
    xl = _crossover(df["close"], basis)     # back to the mean -> exit
    es = _crossover(df["close"], upper)
    xs = _crossunder(df["close"], basis)
    return el, xl, es, xs


def _sig_donchian(df, p, _bm):
    n = int(p["length"])
    upper = df["high"].rolling(n).max().shift(1)   # prior-N high (shifted -> no look-ahead)
    lower = df["low"].rolling(n).min().shift(1)
    el = _crossover(df["close"], upper)
    xl = _crossunder(df["close"], lower)
    es = _crossunder(df["close"], lower)
    xs = _crossover(df["close"], upper)
    return el, xl, es, xs


def _sig_orb(df, p, bar_minutes):
    """Opening-range breakout. Needs intraday bars; on daily/weekly data it yields no signals."""
    n = len(df)
    el = np.zeros(n, bool)
    xl = np.zeros(n, bool)
    es = np.zeros(n, bool)
    xs = np.zeros(n, bool)
    if not bar_minutes or bar_minutes >= 1440:
        return el, xl, es, xs  # not an intraday series -> ORB undefined
    or_bars = max(1, int(round(p.get("or_minutes", 15) / bar_minutes)))
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    dates = pd.Index(df.index).normalize()
    for _, idx in pd.Series(range(n), index=dates).groupby(level=0):
        rows = idx.to_numpy()
        if len(rows) <= or_bars:
            continue
        or_hi = high[rows[:or_bars]].max()
        or_lo = low[rows[:or_bars]].min()
        for j, r in enumerate(rows):
            if j < or_bars:
                continue
            if close[r] > or_hi:
                el[r] = True
            elif close[r] < or_lo:
                es[r] = True
        last = rows[-1]              # flatten at the session close
        xl[last] = True
        xs[last] = True
    return el, xl, es, xs


_SIGNALS = {
    "rsi_mean_reversion": _sig_rsi,
    "ma_cross": _sig_ma_cross,
    "macd_cross": _sig_macd,
    "supertrend": _sig_supertrend,
    "bollinger_mean_reversion": _sig_bollinger,
    "donchian_breakout": _sig_donchian,
    "orb": _sig_orb,
}


# ---------------------------------------------------------------------------
# position state machine + simulator
# ---------------------------------------------------------------------------


def _desired_position(el, xl, es, xs, allow_long, allow_short) -> np.ndarray:
    """Walk the signal events into a target-position series d[t] ∈ {-1,0,1}, decided at
    the close of bar t. The simulator then acts on d[t] at bar t+1's open."""
    n = len(el)
    d = np.zeros(n, dtype=int)
    pos = 0
    for t in range(n):
        if pos == 0:
            if allow_long and el[t]:
                pos = 1
            elif allow_short and es[t]:
                pos = -1
        elif pos == 1:
            if allow_short and es[t]:
                pos = -1
            elif xl[t]:
                pos = 0
        elif pos == -1:
            if allow_long and el[t]:
                pos = 1
            elif xs[t]:
                pos = 0
        d[t] = pos
    return d


def _simulate(df, d, *, commission_pct, slippage_pct, sl_pct, tp_pct) -> tuple[list[float], np.ndarray]:
    """Execute target positions d at the NEXT bar's open. Returns (net per-trade returns
    as fractions, equity curve as cumulative fraction)."""
    open_ = df["open"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    n = len(open_)
    cost = (commission_pct + slippage_pct) * 2.0 / 100.0  # round-trip, as a fraction
    trades: list[float] = []
    side = 0
    entry = 0.0

    def close_trade(exit_price):
        nonlocal side, entry
        gross = side * (exit_price / entry - 1.0)
        trades.append(gross - cost)
        side = 0

    for k in range(1, n):
        target = d[k - 1]  # decision made at close of k-1, acted on at open[k]
        # 1) act on the signal at this bar's open (flip = close then re-open)
        if side != 0 and target != side:
            close_trade(open_[k])
        if side == 0 and target != 0:
            side = target
            entry = open_[k]
        # 2) intrabar stop / target on the bar we now hold
        if side != 0 and (sl_pct or tp_pct):
            if side == 1:
                stop = entry * (1 - sl_pct / 100.0) if sl_pct else None
                tgt = entry * (1 + tp_pct / 100.0) if tp_pct else None
                if stop is not None and low[k] <= stop:      # conservative: stop checked first
                    close_trade(stop)
                elif tgt is not None and high[k] >= tgt:
                    close_trade(tgt)
            else:
                stop = entry * (1 + sl_pct / 100.0) if sl_pct else None
                tgt = entry * (1 - tp_pct / 100.0) if tp_pct else None
                if stop is not None and high[k] >= stop:
                    close_trade(stop)
                elif tgt is not None and low[k] <= tgt:
                    close_trade(tgt)
    if side != 0:                                   # mark-to-market the open position at the last close
        close_trade(df["close"].to_numpy()[-1])

    equity = np.cumsum(trades) if trades else np.array([])
    return trades, equity


def _metrics(trades: list[float], equity: np.ndarray) -> StrategyBacktestMetrics | None:
    if not trades:
        return None
    arr = np.array(trades)
    wins = arr[arr > 0]
    losses = arr[arr < 0]
    gross_profit = float(wins.sum()) * 100.0
    gross_loss = float(-losses.sum()) * 100.0
    pf = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    if equity.size:
        peak = np.maximum.accumulate(equity)
        max_dd = float((peak - equity).max()) * 100.0
    else:
        max_dd = 0.0
    return StrategyBacktestMetrics(
        net_profit=float(arr.sum()) * 100.0,
        gross_profit=gross_profit,
        total_trades=len(trades),
        winning_trades=int((arr > 0).sum()),
        win_rate=float((arr > 0).mean()),
        max_drawdown=max_dd,
        profit_factor=pf,
        pine_script_path="",
    )


def run_backtest(
    spec: StrategySpec,
    df: pd.DataFrame,
    *,
    commission_pct: float = 0.0,
    slippage_pct: float = 0.0,
    bar_minutes: int | None = None,
    invert: bool = False,
) -> StrategyBacktestMetrics | None:
    """Backtest `spec` on `df`. Returns StrategyBacktestMetrics, or None if the archetype
    is unsupported / produced no trades. `invert=True` runs the genuine inverse strategy
    (every long becomes a short and vice-versa) for the inverse-edge rigor check."""
    if df is None or len(df) < 30 or spec.archetype not in _SIGNALS:
        return None
    params = {**_DEFAULTS.get(spec.archetype, {}), **(spec.params or {})}
    try:
        el, xl, es, xs = _SIGNALS[spec.archetype](df, params, bar_minutes)
    except Exception:  # noqa: BLE001 — a malformed param set must not crash the battery
        return None

    allow_long = spec.side in ("long", "both")
    allow_short = spec.side in ("short", "both")
    if invert:
        el, xl, es, xs = es, xs, el, xl
        allow_long, allow_short = allow_short, allow_long
    if not (allow_long or allow_short):
        return None

    d = _desired_position(el, xl, es, xs, allow_long, allow_short)
    trades, equity = _simulate(
        df, d, commission_pct=commission_pct, slippage_pct=slippage_pct,
        sl_pct=spec.sl_pct, tp_pct=spec.tp_pct,
    )
    return _metrics(trades, equity)


# ---------------------------------------------------------------------------
# Pine -> StrategySpec inference (LLM first, regex fallback)
# ---------------------------------------------------------------------------

_RE_COMMISSION = re.compile(r"commission_(type|value)\s*=", re.IGNORECASE)
_RE_SLIPPAGE = re.compile(r"\bslippage\s*=", re.IGNORECASE)

_INFER_SYSTEM = (
    "You map a TradingView Pine strategy to ONE well-known archetype so it can be "
    "re-implemented and backtested. Be faithful to the code's actual entry/exit logic. "
    "Output ONLY JSON."
)


def infer_spec(script: str, config: Config) -> StrategySpec:
    """Infer a StrategySpec from a pasted Pine strategy. Tries the LLM; on any failure
    falls back to a regex heuristic. Always returns a spec — `archetype='custom'` means
    the engine can't model it (the empirical battery will then report not_assessed)."""
    has_costs = bool(_RE_COMMISSION.search(script) and _RE_SLIPPAGE.search(script))
    spec = _infer_via_llm(script, config)
    if spec is not None:
        spec.has_costs = spec.has_costs or has_costs
        return spec
    return _infer_via_regex(script, has_costs)


def _infer_via_llm(script: str, config: Config) -> StrategySpec | None:
    prompt = (
        "Identify which archetype this Pine strategy implements and extract its parameters.\n"
        f"Allowed archetypes: {', '.join(ARCHETYPES)}, or \"custom\" if none fits.\n\n"
        "Return JSON exactly:\n"
        '{"archetype":"<one of the allowed>","side":"long|short|both",'
        '"params":{...archetype params...},"sl_pct":<number or null>,'
        '"tp_pct":<number or null>,"confidence":<0..1>,'
        '"note":"one sentence on how faithfully this maps"}\n\n'
        "Param keys by archetype: rsi_mean_reversion{length,oversold,overbought}; "
        "ma_cross{fast,slow,ma_type}; macd_cross{fast,slow,signal}; "
        "supertrend{atr_period,factor}; bollinger_mean_reversion{length,mult}; "
        "donchian_breakout{length}; orb{or_minutes}.\n"
        "sl_pct/tp_pct are stop/target as PERCENT of entry (null if the code has none).\n\n"
        f"```pine\n{script[:8000]}\n```"
    )
    try:
        raw = llm_mod.complete(_INFER_SYSTEM, prompt, model=config.synthesize_model or None,
                               max_tokens=600, timeout=180)
    except (llm_mod.LlmUnavailable, llm_mod.LlmError):
        return None
    m = re.search(r"\{.*\}", raw or "", re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return None
    arch = str(data.get("archetype", "custom")).strip()
    if arch not in ARCHETYPES:
        return StrategySpec(archetype="custom", side="both", source="llm",
                            confidence=float(data.get("confidence", 0.3) or 0.3),
                            note=str(data.get("note", ""))[:200])
    side = str(data.get("side", _DEFAULT_SIDE.get(arch, "both"))).lower()
    if side not in ("long", "short", "both"):
        side = _DEFAULT_SIDE.get(arch, "both")
    return StrategySpec(
        archetype=arch,
        side=side,
        params=_clean_params(data.get("params", {})),
        sl_pct=_num_or_none(data.get("sl_pct")),
        tp_pct=_num_or_none(data.get("tp_pct")),
        confidence=float(data.get("confidence", 0.7) or 0.7),
        source="llm",
        note=str(data.get("note", ""))[:200],
    )


def _infer_via_regex(script: str, has_costs: bool) -> StrategySpec:
    """A best-effort archetype guess that works with no LLM at all."""
    s = script.lower()

    def first_int(*pats, default=None):
        for pat in pats:
            m = re.search(pat, s)
            if m:
                try:
                    return int(float(m.group(1)))
                except (TypeError, ValueError):
                    pass
        return default

    if "ta.rsi" in s or re.search(r"\brsi\b", s):
        return StrategySpec("rsi_mean_reversion", "long", has_costs=has_costs, source="regex",
                            confidence=0.4, note="regex guess from ta.rsi",
                            params={"length": first_int(r"ta\.rsi\([^,]*,\s*(\d+)", default=14) or 14})
    if "ta.macd" in s:
        return StrategySpec("macd_cross", "both", has_costs=has_costs, source="regex",
                            confidence=0.4, note="regex guess from ta.macd")
    if "ta.supertrend" in s or "supertrend" in s:
        return StrategySpec("supertrend", "both", has_costs=has_costs, source="regex",
                            confidence=0.4, note="regex guess from supertrend")
    if "ta.bb(" in s or "bollinger" in s:
        return StrategySpec("bollinger_mean_reversion", "long", has_costs=has_costs, source="regex",
                            confidence=0.4, note="regex guess from Bollinger Bands")
    if "ta.highest" in s and "ta.lowest" in s:
        return StrategySpec("donchian_breakout", "both", has_costs=has_costs, source="regex",
                            confidence=0.4, note="regex guess from highest/lowest breakout")
    if ("ta.crossover" in s or "ta.crossunder" in s) and ("ta.sma" in s or "ta.ema" in s):
        return StrategySpec("ma_cross", "both", has_costs=has_costs, source="regex",
                            confidence=0.4, note="regex guess from MA crossover")
    if "opening range" in s or "session" in s and "breakout" in s:
        return StrategySpec("orb", "both", has_costs=has_costs, source="regex",
                            confidence=0.3, note="regex guess from opening-range/session breakout")
    return StrategySpec("custom", "both", has_costs=has_costs, source="regex",
                        confidence=0.2, note="no known archetype matched")


def _clean_params(params) -> dict:
    if not isinstance(params, dict):
        return {}
    out = {}
    for k, v in params.items():
        if isinstance(v, bool):
            out[str(k)] = v
        elif isinstance(v, (int, float)):
            out[str(k)] = v
        elif isinstance(v, str) and v.strip():
            out[str(k)] = v.strip()
    return out


def _num_or_none(v):
    if v is None:
        return None
    try:
        f = float(v)
        return f if f > 0 else None
    except (TypeError, ValueError):
        return None


def bar_minutes_for(timeframe: str) -> int | None:
    """Minutes per bar for a TradingView resolution string (None if not intraday-mappable)."""
    from .freedata import normalize_timeframe
    tf = normalize_timeframe(timeframe)
    table = {"1": 1, "3": 3, "5": 5, "15": 15, "30": 30, "45": 45,
             "60": 60, "120": 120, "240": 240, "D": 1440, "W": 1440 * 7, "M": 1440 * 30}
    return table.get(tf)
