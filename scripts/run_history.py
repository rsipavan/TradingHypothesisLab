#!/usr/bin/env python3
"""Compute the aggregate run-history table from the committed example corpus.

Reads every examples/*/report.json, tallies the per-video verdict distribution and
the reasons claims came back untestable, and prints the markdown that fills the
"Run history" section of the README. The numbers are derived from the reports, not
asserted — re-run after adding an example to keep the README honest.

    python scripts/run_history.py
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"

# verdict_overall value -> the row label used in the README table
VERDICT_ROWS = [
    ("holds", "pass", "claim holds against data"),
    ("fails", "fail", "claim contradicted by data"),
    ("partial", "partial", "mixed evidence"),
    ("untestable", "untestable", "claim well-formed but unverifiable"),
    ("error", "error", "pipeline failure, recoverable next run"),
]


PRE_PINE = "strategy claim from an example that predates the Pine engine (pending re-run)"


def categorize_untestable(reason: str) -> str:
    r = (reason or "").lower()
    if "backtest engine" in r or "not in v1" in r:
        return PRE_PINE
    if "instrument" in r:
        return "strategy/claim described without a specific instrument"
    if any(k in r for k in ("opinion", "marketing", "assertion", "unfalsifiable",
                            "falsifiable", "always", "undefined", "no threshold")):
        return "directional opinion / assertion (no falsifiable threshold)"
    if "checkable claim" in r:
        return "point made in passing, not a checkable claim"
    if any(k in r for k in ("timeframe", "missing", "required input")):
        return "required input missing (timeframe, stop rule, threshold)"
    return "other / unspecified"


def main() -> None:
    reports = sorted(EXAMPLES.glob("*/report.json"))
    total = len(reports)
    verdicts: Counter[str] = Counter()
    testable_videos = 0
    untestable_reasons: Counter[str] = Counter()

    for rp in reports:
        data = json.loads(rp.read_text(encoding="utf-8"))
        verdicts[data.get("verdict_overall", "error")] += 1
        findings = data.get("findings", [])
        if any(str(f.get("testable", "")).lower() == "yes" for f in findings):
            testable_videos += 1
        untestable_findings = [f for f in findings
                               if str(f.get("verdict", "")).lower() == "untestable"]
        if data.get("verdict_overall") == "untestable" and not untestable_findings:
            # summary-only stop: no claims were extracted at all
            untestable_reasons["no checkable claim extracted (promo / mindset / educational)"] += 1
        for f in untestable_findings:
            untestable_reasons[categorize_untestable(f.get("verdict_reason", ""))] += 1

    def pct(n: int) -> str:
        return f"{round(100 * n / total)}%" if total else "0%"

    print(f"Corpus: {total} videos in examples/\n")
    print("| Metric | Count | % |")
    print("|--------|-------|---|")
    print(f"| Total videos processed | {total} | 100% |")
    print(f"| Videos producing a testable claim | {testable_videos} | {pct(testable_videos)} |")
    for key, label, _desc in VERDICT_ROWS:
        n = verdicts.get(key, 0)
        print(f"| Verdict: `{label}` | {n} | {pct(n)} |")

    print("\nTop reasons for `untestable`:")
    for reason, n in untestable_reasons.most_common():
        print(f"- {n} — {reason}")


if __name__ == "__main__":
    main()
