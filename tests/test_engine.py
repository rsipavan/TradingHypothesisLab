"""Tests for the TradingView-independent rigor engine: freedata mapping, the lookahead-free
backtest engine, archetype signals, spec inference (regex path), and an offline end-to-end
rigor.check() smoke test. No network and no LLM are used — data and the LLM backend are stubbed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_hypothesis_lab import backtest as bt
from trading_hypothesis_lab import freedata
from trading_hypothesis_lab.backtest import StrategySpec, run_backtest


# --------------------------------------------------------------------------- freedata mapping


@pytest.mark.parametrize("instrument,expected", [
    ("SPY", "SPY"),
    ("QQQ", "QQQ"),
    ("BINANCE:BTCUSDT", "BTC-USD"),
    ("ETHUSD", "ETH-USD"),
    ("COINBASE:SOLUSD", "SOL-USD"),
    ("NSE:NIFTY", "^NSEI"),
    ("NIFTY", "^NSEI"),
    ("BANKNIFTY", "^NSEBANK"),
    ("SPX", "^GSPC"),
    ("US100", "^NDX"),
    ("FX:EURUSD", "EURUSD=X"),
    ("EURUSD", "EURUSD=X"),
    ("NSE:RELIANCE", "RELIANCE.NS"),
    ("BSE:TCS", "TCS.BO"),
    ("^GSPC", "^GSPC"),
])
def test_resolve_ticker(instrument, expected):
    assert freedata.resolve_ticker(instrument) == expected


@pytest.mark.parametrize("tf,expected", [
    ("4h", "240"), ("1h", "60"), ("1d", "D"), ("d", "D"), ("15m", "15"),
    ("60", "60"), ("D", "D"), ("1w", "W"),
])
def test_normalize_timeframe(tf, expected):
    assert freedata.normalize_timeframe(tf) == expected


def test_tf_neighbours():
    assert freedata.tf_neighbours("60") == ["30", "240"]
    assert freedata.tf_neighbours("D") == ["240", "W"]
    assert freedata.tf_neighbours("1") == ["5"]
    assert freedata.tf_neighbours("W") == ["D"]


def test_crypto_base():
    assert freedata._crypto_base("BTCUSDT") == "BTC"
    assert freedata._crypto_base("ETH-USD") == "ETH"
    assert freedata._crypto_base("AAPL") is None


# --------------------------------------------------------------------------- synthetic data


def _trend_df(n=400, seed=3, freq="D"):
    """An up-trending series with intrabar high/low consistent with open/close."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    price = 100 + 0.05 * t + np.cumsum(rng.normal(0, 0.5, n))
    idx = pd.date_range("2021-01-01", periods=n, freq=freq)
    close = pd.Series(np.maximum(price, 5), index=idx)
    open_ = close.shift(1).fillna(close.iloc[0])
    hi = np.maximum(open_, close) * 1.002
    lo = np.minimum(open_, close) * 0.998
    return pd.DataFrame({"open": open_, "high": hi, "low": lo, "close": close, "volume": 1000.0}, index=idx)


# --------------------------------------------------------------------------- indicators / crosses


def test_rsi_bounds():
    df = _trend_df()
    r = bt._rsi(df["close"], 14)
    assert r.between(0, 100).all()


def test_crossover_crossunder():
    a = pd.Series([1, 2, 3, 2, 1, 2], dtype=float)
    b = pd.Series([2, 2, 2, 2, 2, 2], dtype=float)
    up = bt._crossover(a, b)
    dn = bt._crossunder(a, b)
    assert up[2] and not up[1]      # 2->3 crosses above 2 at index 2
    assert dn[4] and not dn[3]      # 2->1 crosses below 2 at index 4


# --------------------------------------------------------------------------- the no-lookahead property


def test_execution_is_next_bar_open():
    """A signal decided at the close of bar t must be FILLED at bar t+1's open — never at
    the bar that produced the signal. This is the core no-lookahead guarantee."""
    idx = pd.date_range("2022-01-01", periods=4, freq="D")
    df = pd.DataFrame({
        "open":  [10.0, 11.0, 12.0, 13.0],
        "high":  [10.5, 11.5, 12.5, 13.5],
        "low":   [9.5, 10.5, 11.5, 12.5],
        "close": [10.2, 11.2, 12.2, 13.2],
        "volume": [1.0, 1.0, 1.0, 1.0],
    }, index=idx)
    d = np.array([0, 1, 1, 0])  # long decided at close of bar 1 -> fill at open of bar 2
    trades, _ = bt._simulate(df, d, commission_pct=0.0, slippage_pct=0.0, sl_pct=None, tp_pct=None)
    assert len(trades) == 1
    # entry at open[2]=12.0, marked out at final close[3]=13.2 -> +10%
    assert trades[0] == pytest.approx(13.2 / 12.0 - 1.0, rel=1e-9)


def test_no_lookahead_signal_truncation_invariance():
    """Truncating the future cannot change a past signal. Computing signals on bars[:k]
    must give exactly the same decisions as computing them on the full series — otherwise
    something is peeking ahead."""
    df = _trend_df(n=600, seed=11)
    p = dict(bt._DEFAULTS["macd_cross"])
    d_full = bt._desired_position(*bt._sig_macd(df, p, None), True, True)
    k = 400
    d_trunc = bt._desired_position(*bt._sig_macd(df.iloc[:k], p, None), True, True)
    assert np.array_equal(d_full[:k], d_trunc)


def test_no_lookahead_simulator_is_causal():
    """Detonating the FINAL bar must not change any earlier trade. A non-causal engine that
    peeked at future bars would leak the shock backwards."""
    df = _trend_df(n=300, seed=4)
    d = bt._desired_position(*bt._sig_ma_cross(df, dict(bt._DEFAULTS["ma_cross"]), None), True, True)
    kw = dict(commission_pct=0.0, slippage_pct=0.0, sl_pct=None, tp_pct=None)
    t1, _ = bt._simulate(df, d, **kw)
    df2 = df.copy()
    df2.iloc[-1] = df2.iloc[-1] * 10.0          # blow up the last bar
    t2, _ = bt._simulate(df2, d, **kw)
    n = min(len(t1), len(t2))
    assert n >= 2
    for a, b in zip(t1[:n - 1], t2[:n - 1]):    # every trade except the last is unchanged
        assert a == pytest.approx(b, rel=1e-12)


def test_costs_reduce_return():
    df = _trend_df()
    spec = StrategySpec("ma_cross", "both")
    free = run_backtest(spec, df, bar_minutes=1440)
    costed = run_backtest(spec, df, commission_pct=0.05, slippage_pct=0.02, bar_minutes=1440)
    assert free is not None and costed is not None
    assert costed.net_profit < free.net_profit


# --------------------------------------------------------------------------- archetypes produce trades


@pytest.mark.parametrize("arch,side", [
    ("rsi_mean_reversion", "long"),
    ("ma_cross", "both"),
    ("macd_cross", "both"),
    ("supertrend", "both"),
    ("bollinger_mean_reversion", "long"),
    ("donchian_breakout", "both"),
])
def test_archetype_runs(arch, side):
    df = _trend_df(n=500)
    m = run_backtest(StrategySpec(arch, side), df, bar_minutes=1440)
    # ma/macd/supertrend/donchian trade on a trend; mean-reversion archetypes may be sparse
    # but should still return a metrics object (or None if genuinely no signal fired)
    if m is not None:
        assert m.total_trades >= 1
        assert 0.0 <= m.win_rate <= 1.0


def test_inverse_loses_on_trend_follower():
    df = _trend_df(n=500)
    spec = StrategySpec("ma_cross", "both")
    base = run_backtest(spec, df, bar_minutes=1440)
    inv = run_backtest(spec, df, bar_minutes=1440, invert=True)
    assert base is not None and inv is not None
    # an MA-cross trend follower beats its mirror image on a trending series
    assert base.profit_factor > inv.profit_factor


def test_orb_needs_intraday():
    df = _trend_df(n=200)  # daily bars
    m = run_backtest(StrategySpec("orb", "both"), df, bar_minutes=1440)
    assert m is None  # ORB is undefined on daily data -> no trades


def test_unsupported_archetype_returns_none():
    df = _trend_df()
    assert run_backtest(StrategySpec("custom", "both"), df, bar_minutes=1440) is None


# --------------------------------------------------------------------------- spec inference (regex, no LLM)


def test_infer_spec_regex(monkeypatch):
    monkeypatch.setattr("trading_hypothesis_lab.llm.available_backend", lambda: None)
    from trading_hypothesis_lab.config import load_config
    cfg = load_config()
    assert bt.infer_spec("r = ta.rsi(close, 14)", cfg).archetype == "rsi_mean_reversion"
    assert bt.infer_spec("[m,s,h] = ta.macd(close,12,26,9)", cfg).archetype == "macd_cross"
    assert bt.infer_spec("st = ta.supertrend(3, 10)", cfg).archetype == "supertrend"
    assert bt.infer_spec("b = ta.bb(close, 20, 2)", cfg).archetype == "bollinger_mean_reversion"
    assert bt.infer_spec("plot(close)", cfg).archetype == "custom"


def test_infer_spec_detects_costs(monkeypatch):
    monkeypatch.setattr("trading_hypothesis_lab.llm.available_backend", lambda: None)
    from trading_hypothesis_lab.config import load_config
    script = ("strategy('x', commission_type=strategy.commission.percent, "
              "commission_value=0.05, slippage=2)\nr=ta.rsi(close,14)")
    assert bt.infer_spec(script, load_config()).has_costs is True


# --------------------------------------------------------------------------- offline end-to-end rigor.check


_RSI_PINE = """//@version=5
strategy("RSI Reversion", overlay=false)
len = input.int(14, "RSI Length")
os  = input.int(30, "Oversold")
ob  = input.int(70, "Overbought")
r = ta.rsi(close, len)
if ta.crossover(r, os)
    strategy.entry("L", strategy.long)
if ta.crossover(r, ob)
    strategy.close("L")
"""

_LOOKAHEAD_PINE = """//@version=5
strategy("Peeker", overlay=true, calc_on_every_tick=true)
htf = request.security(syminfo.tickerid, "D", close, lookahead=barmerge.lookahead_on)
if close > htf
    strategy.entry("L", strategy.long)
"""


def _osc_df(n=900, seed=5):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    price = 100 + np.cumsum(rng.normal(0, 1, n)) + 10 * np.sin(t / 20.0)
    idx = pd.date_range("2021-01-01", periods=n, freq="D")
    close = pd.Series(np.maximum(price, 5), index=idx)
    open_ = close.shift(1).fillna(close.iloc[0])
    return pd.DataFrame({"open": open_, "high": np.maximum(open_, close) * 1.003,
                         "low": np.minimum(open_, close) * 0.997, "close": close,
                         "volume": 1000.0}, index=idx)


def test_check_offline_smoke(monkeypatch):
    monkeypatch.setattr("trading_hypothesis_lab.llm.available_backend", lambda: None)
    df = _osc_df()
    monkeypatch.setattr("trading_hypothesis_lab.freedata.fetch_ohlcv", lambda i, tf, **k: df.copy())
    from trading_hypothesis_lab import rigor
    from trading_hypothesis_lab.config import load_config

    rep = rigor.check(_RSI_PINE, "SPY", "D", load_config(), None, extra_instruments=["QQQ"])
    assert rep.json["engine"] == "python"
    assert rep.json["archetype"] == "rsi_mean_reversion"
    assert rep.compiled is True                       # not verified, so not penalised
    assert rep.json["pine_compile_verified"] is False
    assert rep.markdown and "Rigor scorecard" in rep.markdown
    # static gates and the trend-following dimensions all resolve (not error)
    ids = {c.id: c.status for c in rep.gates + rep.dimensions}
    assert ids["G1"] == "pass" and ids["G2"] == "pass"
    assert ids["D5"] in ("pass", "fail")              # inverse-edge actually ran


def test_check_no_data_degrades(monkeypatch):
    """No free data must degrade the empirical dimensions to not_assessed, never crash."""
    monkeypatch.setattr("trading_hypothesis_lab.llm.available_backend", lambda: None)
    monkeypatch.setattr("trading_hypothesis_lab.freedata.fetch_ohlcv", lambda i, tf, **k: None)
    from trading_hypothesis_lab import rigor
    from trading_hypothesis_lab.config import load_config

    rep = rigor.check(_RSI_PINE, "OBSCURE:THING", "D", load_config(), None)
    assert rep.json["engine"] == "none"
    empirical = [c for c in rep.gates + rep.dimensions if c.id in ("G3", "G4", "D1", "D3", "D4", "D5", "D6", "D7")]
    assert all(c.status == "not_assessed" for c in empirical)
    assert rep.grade != "Robust"               # can't be robust on checks that never ran


def test_check_custom_archetype_not_assessed(monkeypatch):
    """A script that maps to no known archetype: empirical battery skipped honestly."""
    monkeypatch.setattr("trading_hypothesis_lab.llm.available_backend", lambda: None)
    monkeypatch.setattr("trading_hypothesis_lab.freedata.fetch_ohlcv", lambda i, tf, **k: _osc_df().copy())
    from trading_hypothesis_lab import rigor
    from trading_hypothesis_lab.config import load_config

    script = "//@version=5\nstrategy('Mystery')\nplot(close + volume)\n"
    rep = rigor.check(script, "SPY", "D", load_config(), None)
    assert rep.json["archetype"] == "custom"
    assert rep.json["engine"] == "none"
    assert next(d for d in rep.dimensions if d.id == "D1").status == "not_assessed"


def test_check_catches_lookahead_offline(monkeypatch):
    monkeypatch.setattr("trading_hypothesis_lab.llm.available_backend", lambda: None)
    monkeypatch.setattr("trading_hypothesis_lab.freedata.fetch_ohlcv", lambda i, tf, **k: _osc_df().copy())
    from trading_hypothesis_lab import rigor
    from trading_hypothesis_lab.config import load_config

    rep = rigor.check(_LOOKAHEAD_PINE, "SPY", "D", load_config(), None)
    g1 = next(g for g in rep.gates if g.id == "G1")
    g2 = next(g for g in rep.gates if g.id == "G2")
    assert g1.status == "fail"                        # barmerge.lookahead_on
    assert g2.status == "fail"                        # calc_on_every_tick=true
    assert rep.grade == "Untrustworthy" and rep.score == 0
