from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass(frozen=True)
class Question:
    id: str
    resolve_session: date
    symbol: str
    threshold: float
    yes_when: str


def load_questions(path: Path) -> list[Question]:
    payload = json.loads(path.read_text())
    questions = []
    for row in payload["questions"]:
        if row["comparison"] != "gt":
            raise ValueError(f"{row['id']} must use comparison gt")
        questions.append(
            Question(
                id=row["id"],
                resolve_session=date.fromisoformat(row["resolve_session"]),
                symbol=row["symbol"],
                threshold=float(row["threshold"]),
                yes_when=row["yes_when"],
            )
        )
    return questions


def reference_closes(path: Path) -> dict:
    payload = json.loads(path.read_text())
    block = payload["reference_closes"]
    return {
        "session_date": date.fromisoformat(block["session_date"]),
        "vix": float(block["vix"]),
        "ndx": float(block["ndx"]),
    }
