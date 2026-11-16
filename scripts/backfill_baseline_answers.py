"""Fill in the deterministic forecasters for questions the eval never asked them.

The first run scored the baselines on the 84-question shared set only, so the per-question
table has gaps that look like the constant predictor "failed" on 12 questions. It did not:
it was never called. This script answers those questions with the same three deterministic
predictors and stores the result in `answers`, leaving `pairs` (the scored set) alone.

    python3 scripts/backfill_baseline_answers.py artifacts/agent_eval_summary.json
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forecast_log.agent.eval import (  # noqa: E402
    AnaloguePredictor,
    BaseRatePredictor,
    ConstantPredictor,
    LabeledQuestion,
)
from forecast_log.agent.loop import QuestionView  # noqa: E402

BASELINES = {
    "constant 0.5": lambda: ConstantPredictor(),
    "base rate": lambda: BaseRatePredictor(ROOT),
    "analogue model": lambda: AnaloguePredictor(ROOT),
}


def row_from_meta(question_id: str, meta: dict) -> LabeledQuestion:
    return LabeledQuestion(
        question=QuestionView(
            id=question_id,
            text=meta["text"],
            symbol=meta["symbol"],
            threshold=float(meta["threshold"]),
            resolve_session=date.fromisoformat(meta["resolve_session"]),
            horizon_sessions=int(meta["horizon"]),
            kind="synthetic",
        ),
        outcome=int(meta["outcome"]),
        as_of=date.fromisoformat(meta["as_of"]),
        as_of_close=float(meta["as_of_close"]),
        family=meta["family"],
    )


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "artifacts" / "agent_eval_summary.json"
    report = json.loads(path.read_text())
    meta = report["question_meta"]
    for name, make in BASELINES.items():
        row = report["predictors"][name]
        answers = dict(row.get("answers") or row.get("pairs") or {})
        missing = [qid for qid in report["bench"] if qid not in answers]
        if missing:
            predictor = make()
            for qid in missing:
                answers[qid] = round(predictor.predict(row_from_meta(qid, meta[qid])).probability, 4)
        row["answers"] = {qid: answers[qid] for qid in report["bench"] if qid in answers}
        print(f"{name}: {len(row['answers'])}/{len(report['bench'])} answers ({len(missing)} backfilled)")
    for name, row in report["predictors"].items():
        if "answers" not in row:
            row["answers"] = dict(row.get("pairs") or {})
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"{path}: wrote answers")


if __name__ == "__main__":
    main()
