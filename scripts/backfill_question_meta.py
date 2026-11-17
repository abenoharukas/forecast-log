"""Add `question_meta` to an existing agent-eval report, without re-running any model.

The report only stored question ids, so the page could show an aggregate but not the
question behind each score. This script rebuilds the question from the id and the price
series, then verifies the rebuilt threshold reproduces the stored outcome. If any id
disagrees, it stops instead of writing a plausible-looking wrong label.

    python3 scripts/backfill_question_meta.py artifacts/agent_eval_summary.json
"""

from __future__ import annotations

import json
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forecast_log.bars import load_series  # noqa: E402

WINDOW = 504
QUANTILE = 0.6
ID = re.compile(r"^(?P<symbol>[A-Z0-9.]+)-(?P<day>\d{4}-\d{2}-\d{2})-h(?P<horizon>\d+)(?:-(?P<family>\w+))?$")


def meta_for(question_id: str, outcome: int | None) -> dict:
    match = ID.match(question_id)
    if not match:
        raise SystemExit(f"cannot parse question id: {question_id}")
    symbol = "^" + match["symbol"]
    as_of = date.fromisoformat(match["day"])
    horizon = int(match["horizon"])
    family = match["family"] or "level"

    series = load_series(ROOT, symbol)
    bars = series.bars
    index = next((i for i, bar in enumerate(bars) if bar.date == as_of), None)
    if index is None:
        raise SystemExit(f"{question_id}: {as_of} is not a session in {symbol}")
    if index + horizon >= len(bars):
        raise SystemExit(f"{question_id}: no bar {horizon} sessions later")
    if index - WINDOW + 1 < 0:
        raise SystemExit(f"{question_id}: not enough history for a {WINDOW}-session window")

    as_of_close = float(bars[index].close)
    trailing = np.array([float(bar.close) for bar in bars[index - WINDOW + 1 : index + 1]])
    if family == "level":
        threshold = round(as_of_close, 2)
    elif family == "quantile":
        threshold = round(float(np.quantile(trailing, QUANTILE)), 2)
    elif family == "band_up":
        threshold = round(as_of_close * 1.05, 2)
    elif family == "band_down":
        threshold = round(as_of_close * 0.95, 2)
    else:
        raise SystemExit(f"{question_id}: unknown family {family!r}")

    target = bars[index + horizon]
    rebuilt = 1 if float(target.close) > threshold else 0
    if outcome is not None and rebuilt != outcome:
        raise SystemExit(
            f"{question_id}: rebuilt outcome {rebuilt} != stored {outcome} "
            f"(threshold {threshold}, target close {target.close})"
        )

    return {
        "text": (
            f"Will {symbol} close strictly above {threshold} on "
            f"{target.date.isoformat()}, the session {horizon} sessions after {as_of.isoformat()}?"
        ),
        "symbol": symbol,
        "as_of": as_of.isoformat(),
        "as_of_close": round(as_of_close, 2),
        "resolve_session": target.date.isoformat(),
        "horizon": horizon,
        "threshold": threshold,
        "family": family,
        "outcome": rebuilt,
    }


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "artifacts" / "agent_eval_summary.json"
    report = json.loads(path.read_text())
    outcomes = report["outcomes"]
    meta = {
        qid: meta_for(qid, int(outcomes[qid]) if qid in outcomes else None)
        for qid in report["bench"]
    }
    report["question_meta"] = meta
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"{path}: {len(meta)} questions labelled")


if __name__ == "__main__":
    main()
