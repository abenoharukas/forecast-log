from pathlib import Path


def repo_root() -> Path:
    cwd = Path.cwd()
    candidates = [cwd, *cwd.parents, *Path(__file__).resolve().parents]
    for path in candidates:
        if (path / "questions.json").is_file():
            return path
    raise FileNotFoundError("questions.json not found from the working directory")


def series_path(root: Path, symbol: str) -> Path:
    """^VIX3M -> data/raw/vix3m.json. Any symbol works; the settlement symbols are in questions.json."""
    slug = symbol.lstrip("^").lower().replace("/", "_")
    if not slug:
        raise ValueError("symbol must not be empty")
    return root / "data" / "raw" / f"{slug}.json"
