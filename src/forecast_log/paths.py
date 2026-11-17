from pathlib import Path


def repo_root() -> Path:
    cwd = Path.cwd()
    candidates = [cwd, *cwd.parents, *Path(__file__).resolve().parents]
    for path in candidates:
        if (path / "questions.json").is_file():
            return path
    raise FileNotFoundError("questions.json not found from the working directory")
