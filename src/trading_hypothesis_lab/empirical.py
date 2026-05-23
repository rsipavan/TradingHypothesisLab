"""empirical.py — run the rigor empirical battery on the Python backtest engine.

Given an inferred StrategySpec and a test universe, this fetches free OHLCV and produces
every empirical input the rigor dimensions need, bundled into one EmpiricalResults struct:

  - baseline backtest (frictionless, unless the Pine itself models costs)
  - a cost-stressed variant (realistic commission + slippage)
  - a real out-of-sample split (first 70% of bars vs last 30%)
  - a real inverse-edge run (every long becomes a short)
  - neighbouring-timeframe re-runs
  - extra-instrument re-runs

Everything degrades gracefully: no data, an unsupported archetype, or too few bars all just
leave fields None, which the dimensions render as `not_assessed`. Nothing here raises.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import freedata
from .backtest import (
    DEFAULT_COMMISSION_PCT,
    DEFAULT_SLIPPAGE_PCT,
    StrategySpec,
    bar_minutes_for,
    run_backtest,
)
from .config import Config
from .types import StrategyBacktestMetrics


@dataclass
class EmpiricalResults:
    engine: str                 # "python" | "none"
    available: bool             # did a baseline backtest actually run
    note: str                   # human-readable status (data source, or why unavailable)
    spec: StrategySpec | None
    instrument: str
    timeframe: str
    baseline: StrategyBacktestMetrics | None = None
    cost_variant: StrategyBacktestMetrics | None = None
    had_costs: bool = False
    oos: tuple[float, float, int] | None = None     # (in_sample_pf, oos_pf, n_oos_trades)
    multi_tf: list[tuple[str, StrategyBacktestMetrics]] = field(default_factory=list)
    multi_instrument: list[tuple[str, StrategyBacktestMetrics]] = field(default_factory=list)
    inverse: StrategyBacktestMetrics | None = None


def _none(spec, instrument, timeframe, note) -> EmpiricalResults:
    return EmpiricalResults(engine="none", available=False, note=note, spec=spec,
                            instrument=instrument, timeframe=timeframe)


def run_python_empirical(
    spec: StrategySpec | None,
    instrument: str,
    timeframe: str,
    extra_instruments: list[str],
    config: Config,
) -> EmpiricalResults:
    """Run the full empirical battery via the Python backtest engine. Never raises."""
    if spec is None or not spec.supported:
        arch = spec.archetype if spec else "unknown"
        return _none(spec, instrument, timeframe,
                     f"strategy maps to '{arch}' — not a modellable archetype, so the empirical "
                     f"battery was skipped (static gates + source review still apply).")

    bm = bar_minutes_for(timeframe)
    df = freedata.fetch_ohlcv(instrument, timeframe)
    if df is None or len(df) < 60:
        ticker = freedata.resolve_ticker(instrument)
        return _none(spec, instrument, timeframe,
                     f"no usable free OHLCV for {instrument} ({ticker}) at timeframe {timeframe}.")

    # Costs: if the Pine models its own friction we run the baseline WITH realistic costs and
    # call it "already costed"; otherwise the baseline is frictionless and we add a cost variant.
    had_costs = spec.has_costs
    base_comm = DEFAULT_COMMISSION_PCT if had_costs else 0.0
    base_slip = DEFAULT_SLIPPAGE_PCT if had_costs else 0.0

    def bt(frame, *, comm, slip, invert=False):
        return run_backtest(spec, frame, commission_pct=comm, slippage_pct=slip,
                            bar_minutes=bm, invert=invert)

    baseline = bt(df, comm=base_comm, slip=base_slip)
    if baseline is None:
        return _none(spec, instrument, timeframe,
                     f"strategy produced no trades on {instrument} {timeframe} over the available "
                     f"history — nothing to assess.")

    res = EmpiricalResults(
        engine="python", available=True, spec=spec, instrument=instrument, timeframe=timeframe,
        baseline=baseline, had_costs=had_costs,
        note=(f"modelled '{spec.archetype}' on free OHLCV ({freedata.resolve_ticker(instrument)}, "
              f"{len(df)} bars, {timeframe}); a faithful re-implementation of the logic, not the "
              f"literal Pine on TradingView."),
    )

    # cost-stress variant (only meaningful if the script wasn't already costed)
    if not had_costs:
        res.cost_variant = bt(df, comm=DEFAULT_COMMISSION_PCT, slip=DEFAULT_SLIPPAGE_PCT)

    # real out-of-sample split on the time axis
    res.oos = _oos_split(df, lambda frame: bt(frame, comm=base_comm, slip=base_slip))

    # real inverse-edge run
    res.inverse = bt(df, comm=base_comm, slip=base_slip, invert=True)

    # neighbouring timeframes
    for tf in freedata.tf_neighbours(timeframe):
        ndf = freedata.fetch_ohlcv(instrument, tf)
        if ndf is None or len(ndf) < 60:
            continue
        m = run_backtest(spec, ndf, commission_pct=base_comm, slippage_pct=base_slip,
                         bar_minutes=bar_minutes_for(tf))
        if m is not None:
            res.multi_tf.append((tf, m))

    # extra instruments
    for sym in extra_instruments:
        idf = freedata.fetch_ohlcv(sym, timeframe)
        if idf is None or len(idf) < 60:
            continue
        m = bt(idf, comm=base_comm, slip=base_slip)
        if m is not None:
            res.multi_instrument.append((sym, m))

    return res


def _oos_split(df: pd.DataFrame, runner, oos_fraction: float = 0.3):
    """Split the bars chronologically, backtest each half, return (is_pf, oos_pf, n_oos_trades).
    None when either half is too thin to mean anything."""
    cut = int(len(df) * (1 - oos_fraction))
    if cut < 60 or len(df) - cut < 60:
        return None
    is_m = runner(df.iloc[:cut])
    oos_m = runner(df.iloc[cut:])
    if is_m is None or oos_m is None or oos_m.total_trades < 5:
        return None
    return is_m.profit_factor, oos_m.profit_factor, oos_m.total_trades
