"""Read a `.env` file, because exporting a key before every command is a chore.

Deliberately dumb: `KEY=value` per line, `#` comments, optional surrounding quotes, no
`export`, no interpolation, no `.env.local`/`.env.production` cascade. Anything cleverer
than this belongs in the shell, and anything cleverer than this is also harder to audit —
the whole point is knowing exactly which key a run used.

The file is git-ignored. `os.environ` wins over it, so a one-off `DEEPSEEK_API_KEY=... `
on the command line still overrides what is on disk.
"""

from __future__ import annotations

import os
from pathlib import Path


def read_env_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for number, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            raise ValueError(f"{path.name} line {number}: expected KEY=value, got {raw!r}")
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"{path.name} line {number}: empty key")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        values[key] = value
    return values


def merged_env(root: Path) -> dict[str, str]:
    """`.env` in the repo root, then the real environment on top."""
    return {**read_env_file(root / ".env"), **os.environ}
