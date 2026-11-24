#!/usr/bin/env python3
"""Export the local `data/forecast.db` into D1-friendly SQL chunks.

`wrangler d1 execute --file` uploads one file per call and D1 rejects any single
statement longer than 100 KB. This script therefore packs rows into multi-row
INSERTs (about 60 KB each) and groups those statements into files of a few MB.

    python3 scripts/export_d1_seed.py            # -> deploy/seed/*.sql
    python3 scripts/export_d1_seed.py --check    # only report byte sizes
"""

from __future__ import annotations

import argparse
import math
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "forecast.db"
OUT = ROOT / "deploy" / "seed"

# Insert order does not matter (no foreign keys) but keep it stable.
TABLES = (
    ("bars", ("symbol", "date")),
    ("signals", ("market", "date")),
    ("news", ("as_of", "doc_id")),
    ("agent_eval", ("dataset", "question_id", "predictor")),
    ("chat_messages", ("thread_id", "seq")),
)

MAX_STATEMENT_BYTES = 60_000
MAX_FILE_BYTES = 3 * 1024 * 1024


def literal(value) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return "NULL"
        return repr(value)
    return "'" + str(value).replace("'", "''") + "'"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DB)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if not args.db.is_file():
        raise SystemExit(f"missing database: {args.db}")

    if not args.check:
        args.out.mkdir(parents=True, exist_ok=True)
        for old in args.out.glob("*.sql"):
            old.unlink()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    try:
        file_no = 0
        buffer: list[str] = []
        buffer_bytes = 0
        totals: dict[str, int] = {}
        written_files = 0
        written_bytes = 0

        def flush_file() -> None:
            nonlocal file_no, buffer, buffer_bytes, written_files, written_bytes
            if not buffer:
                return
            file_no += 1
            text = "\n".join(buffer) + "\n"
            written_files += 1
            written_bytes += len(text.encode("utf-8"))
            if not args.check:
                (args.out / f"{file_no:04d}.sql").write_text(text, encoding="utf-8")
            buffer = []
            buffer_bytes = 0

        for table, order_by in TABLES:
            columns = [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]
            select_cols = ",".join(f'"{c}"' for c in columns)
            order = ",".join(f'"{c}"' for c in order_by)
            rows = conn.execute(f'SELECT {select_cols} FROM "{table}" ORDER BY {order}')
            total = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            totals[table] = total

            prefix = f'INSERT OR REPLACE INTO "{table}" ({select_cols}) VALUES '
            statement_rows: list[str] = []
            statement_bytes = len(prefix.encode("utf-8"))

            def flush_statement() -> None:
                nonlocal statement_rows, statement_bytes, buffer, buffer_bytes
                if not statement_rows:
                    return
                statement = prefix + ",".join(statement_rows) + ";"
                size = len(statement.encode("utf-8"))
                if buffer and buffer_bytes + size > MAX_FILE_BYTES:
                    flush_file()
                buffer.append(statement)
                buffer_bytes += size
                statement_rows = []
                statement_bytes = len(prefix.encode("utf-8"))

            for row in rows:
                values = "(" + ",".join(literal(row[column]) for column in columns) + ")"
                row_bytes = len(values.encode("utf-8")) + 1
                if statement_rows and statement_bytes + row_bytes > MAX_STATEMENT_BYTES:
                    flush_statement()
                statement_rows.append(values)
                statement_bytes += row_bytes
            flush_statement()

            print(f"{table}: {total} rows", flush=True)

        flush_file()
    finally:
        conn.close()

    print(
        f"{'would write' if args.check else 'wrote'} {written_files} files, "
        f"{written_bytes / 1048576:.2f} MB"
        + ("" if args.check else f" under {args.out}"),
        flush=True,
    )


if __name__ == "__main__":
    main()
