"""Static + live-refresh server that never caches, so a reload always shows the current build.

    python3 serve.py 8731

Exposes `GET/POST /api/refresh` to fetch the latest Yahoo daily bars (including today's
A-share open/intraday bar on Beijing time) and regenerate `artifacts/cross*.json`.
"""

from __future__ import annotations

import http.server
import json
import sys
import threading
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
SHANGHAI = ZoneInfo("Asia/Shanghai")
_REFRESH_LOCK = threading.Lock()


def refresh_cross_data() -> dict:
    """Fetch latest daily bars for cross-market symbols in parallel and rebuild cross*.json."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from forecast_log.bars import save_series
    from forecast_log.china import CHINA_SYMBOLS
    from forecast_log.commands import COMPARE_SYMBOLS, cross_command
    from forecast_log.paths import series_path
    from forecast_log.yahoo import fetch_daily

    symbols = ("^NDX", "^TNX", "DX-Y.NYB", "HYG", "TLT", "^VIX", *COMPARE_SYMBOLS, *CHINA_SYMBOLS)
    updated: dict[str, str] = {}
    errors: dict[str, str] = {}

    def _fetch_one(sym: str) -> tuple[str, object]:
        return sym, fetch_daily(sym)

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_fetch_one, sym): sym for sym in dict.fromkeys(symbols)}
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                _, series = fut.result()
                save_series(series_path(ROOT, sym), series)
                updated[sym] = series.bars[-1].date.isoformat()
            except Exception as exc:
                errors[sym] = str(exc)

    cross_command(ROOT)

    sessions = {}
    for key, fname in (("ndx", "cross.json"), ("qqq", "cross_qqq.json"), ("ssec", "cross_ssec.json")):
        path = ROOT / "artifacts" / fname
        if path.is_file():
            doc = json.loads(path.read_text())
            sessions[key] = doc.get("last_session")

    return {
        "ok": True,
        "refreshed_at": datetime.now(SHANGHAI).isoformat(timespec="seconds"),
        "beijing_date": datetime.now(SHANGHAI).date().isoformat(),
        "sessions": sessions,
        "updated": updated,
        "errors": errors,
    }


def _fetched_beijing_date(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        doc = json.loads(path.read_text())
        raw = doc.get("fetched_at")
        if not raw:
            return None
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=SHANGHAI)
        return dt.astimezone(SHANGHAI).date().isoformat()
    except Exception:
        return None


def _needs_refresh() -> bool:
    today_bj = datetime.now(SHANGHAI).date().isoformat()
    for fname in ("cross.json", "cross_qqq.json", "cross_ssec.json"):
        path = ROOT / "artifacts" / fname
        fetched_day = _fetched_beijing_date(path)
        if fetched_day is None or fetched_day < today_bj:
            return True
    return False


def _cached_status() -> dict:
    sessions = {}
    for key, fname in (("ndx", "cross.json"), ("qqq", "cross_qqq.json"), ("ssec", "cross_ssec.json")):
        path = ROOT / "artifacts" / fname
        if path.is_file():
            doc = json.loads(path.read_text())
            sessions[key] = doc.get("last_session")
    return {
        "ok": True,
        "cached": True,
        "beijing_date": datetime.now(SHANGHAI).date().isoformat(),
        "sessions": sessions,
    }


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def _handle_refresh(self, force: bool = False) -> None:
        with _REFRESH_LOCK:
            try:
                if not force and not _needs_refresh():
                    payload = _cached_status()
                else:
                    payload = refresh_cross_data()
                status = 200
            except Exception as exc:
                payload = {"ok": False, "error": str(exc)}
                status = 500
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path, _, query = self.path.partition("?")
        if path == "/api/refresh":
            self._handle_refresh(force="force=1" in query)
            return
        super().do_GET()

    def do_POST(self) -> None:
        path, _, query = self.path.partition("?")
        if path == "/api/refresh":
            self._handle_refresh(force="force=1" in query)
            return
        self.send_error(405, "Method Not Allowed")

    def log_message(self, *args) -> None:
        pass


def _startup_refresh() -> None:
    if not _needs_refresh():
        return
    print("检测到今日尚未拉取最新行情，后台自动更新中…", flush=True)
    with _REFRESH_LOCK:
        if not _needs_refresh():
            return
        try:
            res = refresh_cross_data()
            print(f"行情更新完成：{res.get('sessions')}", flush=True)
        except Exception as exc:
            print(f"启动自动更新失败：{exc}", flush=True)


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8731
    threading.Thread(target=_startup_refresh, daemon=True).start()
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"http://127.0.0.1:{port}/web/points.html", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
