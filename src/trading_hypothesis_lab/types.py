"""Data contracts between pipeline modules.

These are plain dataclasses. Modules return them; the orchestrator passes them
along and logs summaries. Nothing here knows about tracing, Telegram, or the LLM.

See docs/architecture.md for the contract table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal


# ---------------------------------------------------------------------------
# transcript.py output
# ---------------------------------------------------------------------------


@dataclass
class Transcript:
    """A fetched YouTube transcript. `text` is empty if no captions were available
    (the orchestrator short-circuits to a minimal report in that case)."""

    video_id: str
    url: str
    title: str | None
    channel: str | None
    text: str
    fetched_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def is_empty(self) -> bool:
        # "near-empty" counts as empty — see docs/failure_handling.md
        return self.word_count < 50


# ---------------------------------------------------------------------------
# summarize.py output — the FIRST thing the pipeline does after fetching the
# transcript: establish what kind of video this is, before any claim extraction.
# ---------------------------------------------------------------------------

ContentType = Literal[
    "strategy_or_claim",     # presents a trading strategy / makes checkable claims about market behavior
    "educational",           # explains a concept (what RSI is, how order blocks work) — may contain weak claims
    "market_commentary",     # opinion/prediction on current markets ("I think Q3 is bullish")
    "mindset_psychology",    # trader psychology, discipline, habits — no checkable market claims
    "vlog_or_journey",       # the creator's trading story / lifestyle / day-in-the-life
    "promotion",             # course / signals / prop-firm / community sales pitch
    "mixed",                 # a real blend (e.g. strategy + psychology + a bit of promo)
    "other",                 # none of the above (off-topic, etc.)
]

# Content types where there is, by nature, nothing to validate — the pipeline
# short-circuits to a summary-only report rather than running claim extraction.
NON_CLAIM_CONTENT_TYPES = frozenset({"mindset_psychology", "vlog_or_journey", "promotion", "other"})


@dataclass
class VideoSummary:
    """What the video is about — produced before claim extraction. The report leads
    with this; the extractor (when it runs) gets it as context."""

    video_id: str
    content_type: ContentType
    topic: str       # short — what the video is about, e.g. "RSI + Bollinger Bands mean-reversion backtest"
    summary: str     # 2-4 sentences — what's actually in it
    has_checkable_claims: bool  # the model's read on whether there's anything testable at all

    @property
    def skip_extraction(self) -> bool:
        return self.content_type in NON_CLAIM_CONTENT_TYPES or not self.has_checkable_claims


# ---------------------------------------------------------------------------
# thesis.py output
# ---------------------------------------------------------------------------

Testability = Literal["yes", "partial", "no"]
TestType = Literal[
    "indicator_value_over_range",
    "level_zone_hit_rate",
    "strategy_backtest",
    "none",  # no test type maps to this claim
]


@dataclass
class Claim:
    """A single candidate claim extracted from a transcript, classified for testability.

    See docs/decision_logic.md for what `testable` and `test_type` mean and how
    they're decided.
    """

    id: str  # "c1", "c2", ...
    statement: str  # the claim, restated cleanly
    instrument: str | None  # ticker / symbol named or implied; None if absent
    timeframe: str | None  # e.g. "1D", "4H", "15m"; None if absent
    test_type: TestType
    testable: Testability
    reason_if_not: str | None  # populated when testable in {"partial", "no"}
    confidence: float  # 0..1, the extractor's confidence this is a real, correctly-parsed claim
    test_type_justification: str = ""  # one line, why this test type — goes into the trace


@dataclass
class ThesisSet:
    video_id: str
    claims: list[Claim] = field(default_factory=list)

    @property
    def testable_claims(self) -> list[Claim]:
        return [c for c in self.claims if c.testable in ("yes", "partial")]

    @property
    def has_any_testable(self) -> bool:
        return len(self.testable_claims) > 0


# ---------------------------------------------------------------------------
# validate.py output
# ---------------------------------------------------------------------------

ValidationStatus = Literal["ok", "error", "insufficient_data"]


@dataclass
class StrategyBacktestMetrics:
    """Metrics from a TradingView strategy tester run on a synthesized Pine Script.

    Produced by pine.py when a strategy_backtest claim is successfully compiled and
    tested. Attached to the ValidationRun so report.py can render it.
    """

    net_profit: float
    gross_profit: float
    total_trades: int
    winning_trades: int
    win_rate: float        # winning_trades / total_trades
    max_drawdown: float
    profit_factor: float   # gross_profit / |gross_loss|
    pine_script_path: str  # absolute path to the .pine file on disk


@dataclass
class ValidationRun:
    """The result of running one test against real market data via the TradingView MCP.

    One ValidationRun = one (claim, timeframe) test. A single claim is typically
    validated across several timeframes and produces several ValidationRuns; the
    per-claim verdict aggregates over them (see report.py).

    The validator reports the *rate* (e.g. "bounced 14 of 22 occurrences"); it does
    NOT assign a verdict. report.py turns this into holds/partial/fails/untestable
    by explicit rules — see docs/validation_logic.md.
    """

    claim_id: str
    test_type: TestType
    timeframe: str  # which timeframe this run tested (e.g. "1D", "4H", "1H")
    status: ValidationStatus
    tradingview_query: str  # human-readable description of what was queried (symbol, timeframe, indicator, range)
    data_summary: str  # what the data showed, in plain language
    occurrences: int | None  # n — how many times the trigger condition occurred; None if N/A
    hit_rate: float | None  # r — fraction of occurrences where the claimed outcome happened; None if N/A
    result: str  # one-line summary of the finding
    caveats: list[str] = field(default_factory=list)
    error: str | None = None  # populated when status == "error"
    strategy_backtest: StrategyBacktestMetrics | None = None  # set for strategy_backtest claims
    pine_script_path: str | None = None  # absolute path to .pine file; set even on compile failures


# ---------------------------------------------------------------------------
# report.py output
# ---------------------------------------------------------------------------

Verdict = Literal["holds", "partial", "fails", "untestable"]


@dataclass
class ClaimFinding:
    claim: Claim
    validations: list[ValidationRun]  # one per timeframe tested; empty if claim was never validated
    verdict: Verdict  # aggregate across timeframes
    verdict_reason: str


@dataclass
class Report:
    video_id: str
    video_title: str | None
    video_url: str
    channel: str | None
    video_summary: VideoSummary | None  # what kind of video this is — leads the report
    findings: list[ClaimFinding]
    verdict_overall: Verdict  # the "headline" — see report.py for how it's aggregated
    overall_reason: str
    markdown: str  # the human-readable report (this is what gets sent to Telegram / committed to examples)
    json: dict  # the structured form

    @property
    def run_id(self) -> str:
        # set by the orchestrator; stored on the json for convenience
        return self.json.get("run_id", "")


# ---------------------------------------------------------------------------
# tracing (used by orchestrate.py)
# ---------------------------------------------------------------------------


@dataclass
class TraceEvent:
    step: str
    ok: bool
    summary: str
    ms: int
    extra: dict = field(default_factory=dict)

    def to_jsonl(self) -> dict:
        d = {"step": self.step, "ok": self.ok, "detail": self.summary, "ms": self.ms}
        d.update(self.extra)
        return d


# ---------------------------------------------------------------------------
# rigor.py output — the strategy fact-checker / rigor benchmark
# ---------------------------------------------------------------------------

RigorTier = Literal["gate", "dimension", "review"]
RigorStatus = Literal["pass", "fail", "warn", "not_assessed"]


@dataclass
class RigorCheck:
    """One rigor check — a Tier-1 hard gate or a Tier-2 scored dimension.

    `status`: pass / fail / warn / not_assessed. A gate that fails makes the whole
    backtest untrustworthy. A dimension that is `not_assessed` caps the max grade —
    you can't be rated "robust" on a check that never ran.
    """

    id: str            # "G1", "D1", ...
    name: str
    tier: RigorTier
    status: RigorStatus
    detail: str        # plain-language finding
    metric: str = ""   # optional numeric evidence (e.g. "PF 0.91 over 66 trades")


@dataclass
class RigorReport:
    """The rigor scorecard for one pasted Pine strategy on one (instrument, timeframe)."""

    instrument: str
    timeframe: str
    compiled: bool
    gates: list[RigorCheck]
    dimensions: list[RigorCheck]
    reviews: list[RigorCheck]   # advisory Sonnet 4.6 deep-review findings (not scored)
    score: int          # 0..100 (0 when a hard gate fails)
    max_score: int      # 100, reduced when dimensions are not_assessed
    grade: str          # Untrustworthy | Likely overfit | Promising but fragile | Robust
    verdict: str        # one-line headline
    script_path: str | None
    markdown: str
    json: dict
