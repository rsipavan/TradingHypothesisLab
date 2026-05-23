"""freedata.py — TradingView-independent OHLCV access.

The rigor checker's empirical battery needs price history, but the live TradingView
strategy tester proved unreliable to drive headlessly. This module fetches free OHLCV
from Yahoo Finance (`yfinance`, no API key) so the backtest engine can run fully offline.

Two responsibilities:
  - map a TradingView-style instrument string (``SPY``, ``BINANCE:BTCUSDT``, ``NSE:NIFTY``,
    ``FX:EURUSD``) to a Yahoo ticker;
  - map a TradingView timeframe (``1``/``5``/``15``/``60``/``240``/``D``/``W``) to a Yahoo
    interval + period, honouring Yahoo's intraday-history caps (1m≈7d, 5–30m≈60d, 1h≈730d).

Everything degrades to ``None`` on failure — a missing data feed must never crash the
rigor checker; it just makes the empirical dimensions ``not_assessed``.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import pandas as pd

from .config import repo_root

_CACHE_DIR = repo_root() / ".cache" / "ohlcv"
# Re-use a cached pull this long before going back to Yahoo. Intraday moves fast but the
# rigor verdict doesn't need the last candle; a few hours old is fine and keeps it snappy.
_CACHE_TTL_SECONDS = 6 * 3600

# --- known index symbols (TradingView-ish name -> Yahoo) ----------------------
_INDEX_MAP = {
    "SPX": "^GSPC", "SPX500": "^GSPC", "US500": "^GSPC", "SP500": "^GSPC", "ES": "^GSPC",
    "NDX": "^NDX", "US100": "^NDX", "NAS100": "^NDX", "NQ": "^NDX",
    "DJI": "^DJI", "US30": "^DJI", "DOW": "^DJI",
    "RUT": "^RUT", "US2000": "^RUT",
    "VIX": "^VIX",
    "NIFTY": "^NSEI", "NIFTY50": "^NSEI", "CNX": "^NSEI",
    "BANKNIFTY": "^NSEBANK", "NIFTYBANK": "^NSEBANK",
    "SENSEX": "^BSESN", "BSE": "^BSESN",
    "NKY": "^N225", "NI225": "^N225", "JP225": "^N225",
    "DAX": "^GDAXI", "DE40": "^GDAXI", "GER40": "^GDAXI",
    "FTSE": "^FTSE", "UK100": "^FTSE",
    "HSI": "^HSI", "HK50": "^HSI",
    "CAC": "^FCHI", "FR40": "^FCHI",
}

# crypto bases we recognise from a bare "<BASE>USDT"/"<BASE>USD" symbol
_CRYPTO_BASES = {
    "BTC", "XBT", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX", "DOT", "MATIC",
    "LTC", "LINK", "TRX", "ATOM", "UNI", "BCH", "ETC", "XLM", "NEAR", "APT", "ARB",
    "OP", "FIL", "ICP", "INJ", "SUI", "SHIB", "PEPE",
}
_CRYPTO_EXCHANGES = {"BINANCE", "COINBASE", "BITSTAMP", "KRAKEN", "BYBIT", "OKX",
                     "KUCOIN", "GATEIO", "HUOBI", "BITFINEX", "CRYPTO"}
_FX_EXCHANGES = {"FX", "FX_IDC", "OANDA", "FOREXCOM", "FXCM", "SAXO", "ICMARKETS"}
_CRYPTO_QUOTES = ("USDT", "USDC", "BUSD", "USD")

# TradingView timeframe -> (yahoo interval, yahoo period, resample rule or None).
# Yahoo has no native 2h/4h, so we pull 1h and resample.
_TF_MAP = {
    "1": ("1m", "7d", None),
    "3": ("1m", "7d", "3min"),
    "5": ("5m", "60d", None),
    "15": ("15m", "60d", None),
    "30": ("30m", "60d", None),
    "45": ("15m", "60d", "45min"),
    "60": ("60m", "730d", None),
    "120": ("60m", "730d", "2h"),
    "240": ("60m", "730d", "4h"),
    "D": ("1d", "10y", None),
    "W": ("1wk", "max", None),
    "M": ("1mo", "max", None),
}

_TF_ALIASES = {
    "1m": "1", "3m": "3", "5m": "5", "15m": "15", "30m": "30", "45m": "45",
    "1h": "60", "60m": "60", "2h": "120", "4h": "240",
    "1d": "D", "d": "D", "daily": "D", "1w": "W", "w": "W", "weekly": "W",
    "1mo": "M", "mo": "M", "month": "M",
}


def normalize_timeframe(tf: str) -> str:
    """TradingView resolution string, canonicalised (e.g. '4h' -> '240', 'd' -> 'D')."""
    t = str(tf or "").strip()
    if t.lower() in _TF_ALIASES:
        return _TF_ALIASES[t.lower()]
    if t.lower() in ("d", "w", "m"):
        return t.upper()
    return t


# coarse ladder used to pick neighbouring timeframes for the multi-TF rigor check
_TF_NEIGHBOUR_LADDER = ["1", "5", "15", "30", "60", "240", "D", "W"]


def tf_neighbours(tf: str) -> list[str]:
    """The one-step-lower and one-step-higher timeframes on the coarse ladder."""
    t = normalize_timeframe(tf)
    if t not in _TF_NEIGHBOUR_LADDER:
        return []
    i = _TF_NEIGHBOUR_LADDER.index(t)
    out = []
    if i > 0:
        out.append(_TF_NEIGHBOUR_LADDER[i - 1])
    if i < len(_TF_NEIGHBOUR_LADDER) - 1:
        out.append(_TF_NEIGHBOUR_LADDER[i + 1])
    return out


def resolve_ticker(instrument: str) -> str:
    """Map a TradingView-style instrument string to a Yahoo Finance ticker.

    Precedence: explicit exchange prefix → known index → crypto → FX → exchange-suffixed
    equity → pass-through (assume a US equity/ETF that Yahoo knows by the same symbol).
    """
    raw = str(instrument or "SPY").strip()
    if not raw:
        return "SPY"

    exchange, _, rest = raw.partition(":")
    if rest:
        exchange = exchange.upper().strip()
        sym = rest.strip()
    else:
        exchange = ""
        sym = raw

    # already a Yahoo-shaped ticker — trust it
    if sym.startswith("^") or sym.endswith("=X") or sym.endswith("-USD") or "." in sym:
        return sym

    sym_u = sym.upper()

    # known index name
    if sym_u in _INDEX_MAP:
        return _INDEX_MAP[sym_u]

    # crypto: either a crypto exchange, or a bare "<base><quote>" pair
    if exchange in _CRYPTO_EXCHANGES:
        base = _crypto_base(sym_u)
        if base:
            return f"{base}-USD"
    if not exchange:
        base = _crypto_base(sym_u)
        if base:
            return f"{base}-USD"

    # FX: explicit FX exchange, or a bare 6-letter pair like EURUSD
    if exchange in _FX_EXCHANGES or re.fullmatch(r"[A-Z]{6}", sym_u):
        if re.fullmatch(r"[A-Z]{6}", sym_u):
            return f"{sym_u}=X"

    # Indian exchange equities
    if exchange == "NSE":
        return f"{sym_u}.NS"
    if exchange == "BSE":
        return f"{sym_u}.BO"

    # default: assume Yahoo knows it by this symbol (SPY, QQQ, AAPL, TSLA, ...)
    return sym_u


def _crypto_base(sym_u: str) -> str | None:
    """If `sym_u` looks like a crypto pair (BTCUSDT, ETH-USD, SOLUSD), return the base."""
    s = sym_u.replace("-", "").replace("/", "").replace("_", "")
    for q in _CRYPTO_QUOTES:
        if s.endswith(q) and len(s) > len(q):
            base = s[: -len(q)]
            if base in _CRYPTO_BASES:
                return base
    if s in _CRYPTO_BASES:
        return s
    return None


def fetch_ohlcv(
    instrument: str,
    timeframe: str,
    *,
    use_cache: bool = True,
) -> pd.DataFrame | None:
    """Fetch OHLCV for (instrument, timeframe). Returns a DataFrame indexed by datetime
    with lowercase columns [open, high, low, close, volume], or None on any failure.

    Never raises — a data-feed problem must only degrade the rigor check, not crash it.
    """
    tf = normalize_timeframe(timeframe)
    spec = _TF_MAP.get(tf) or _TF_MAP["D"]
    interval, period, resample = spec
    ticker = resolve_ticker(instrument)

    df = _load_cache(ticker, interval, period) if use_cache else None
    if df is None:
        df = _download(ticker, interval, period)
        if df is not None and use_cache:
            _save_cache(ticker, interval, period, df)
    if df is None or df.empty:
        return None

    if resample:
        df = _resample(df, resample)
    return df if df is not None and not df.empty else None


def _download(ticker: str, interval: str, period: str) -> pd.DataFrame | None:
    try:
        import yfinance as yf
    except ImportError:
        return None
    try:
        raw = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
    except Exception:  # noqa: BLE001 — any network/parse failure → no data
        return None
    return _normalize(raw)


def _normalize(raw) -> pd.DataFrame | None:
    if raw is None or getattr(raw, "empty", True):
        return None
    df = raw.copy()
    # flatten a possible MultiIndex column header
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df.columns = [str(c).strip().lower() for c in df.columns]
    need = ["open", "high", "low", "close"]
    if not all(c in df.columns for c in need):
        return None
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df = df[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=need)
    return df if not df.empty else None


def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame | None:
    try:
        out = df.resample(rule).agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        )
    except Exception:  # noqa: BLE001
        return df
    return out.dropna(subset=["open", "high", "low", "close"])


# --- cache --------------------------------------------------------------------


def _cache_path(ticker: str, interval: str, period: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{ticker}_{interval}_{period}")
    return _CACHE_DIR / f"{safe}.pkl"


def _load_cache(ticker: str, interval: str, period: str) -> pd.DataFrame | None:
    p = _cache_path(ticker, interval, period)
    try:
        if not p.exists() or (time.time() - p.stat().st_mtime) > _CACHE_TTL_SECONDS:
            return None
        return pd.read_pickle(p)
    except Exception:  # noqa: BLE001 — a bad cache file is just a cache miss
        return None


def _save_cache(ticker: str, interval: str, period: str, df: pd.DataFrame) -> None:
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_pickle(_cache_path(ticker, interval, period))
    except Exception:  # noqa: BLE001
        pass
