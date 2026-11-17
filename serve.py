"""兼容旧写法的入口：直接 `python3 serve.py 8731` 也能跑。

真正实现在 `src/forecast_log/server.py`；推荐用 `forecast-log serve` 或 `make dev`。
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from forecast_log.server import main  # noqa: E402

if __name__ == "__main__":
    main(sys.argv[1:])
