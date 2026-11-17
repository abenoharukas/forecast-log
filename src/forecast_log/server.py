"""Static + live-refresh server that never caches, so a reload always shows the current build.

    forecast-log serve 8731        # 推荐，然后开 http://127.0.0.1:8731/
    python3 serve.py 8731          # 老写法，等价

Exposes `GET/POST /api/refresh` to fetch the latest Yahoo daily bars (including today's
A-share open/intraday bar on Beijing time) and regenerate `artifacts/cross*.json`.
"""

from __future__ import annotations

import argparse
import http.server
import json
import sys
import threading
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
SHANGHAI = ZoneInfo("Asia/Shanghai")
_REFRESH_LOCK = threading.Lock()
DEFAULT_PORT = 8731

# 干净的 URL → 实际文件。地址里不带 .html，也不带 /web/ 前缀。
PAGES = {
    "/": "web/points.html",
    "/signals": "web/points.html",
    "/ndx": "web/points.html",
    "/qqq": "web/points.html",
    "/a-shares": "web/points.html",
    "/ssec": "web/points.html",
    "/volatility": "web/index.html",
    "/vxn": "web/index.html",
    "/agent": "web/agent.html",
}

# 路径本身携带市场，前端据此选中标签页
MARKET_PATHS = {
    "/": "ndx",
    "/signals": "ndx",
    "/ndx": "ndx",
    "/qqq": "qqq",
    "/a-shares": "ssec",
    "/ssec": "ssec",
}

LEGACY_MARKET = {"ndx": "/ndx", "qqq": "/qqq", "ssec": "/a-shares"}

# 旧地址（带 .html / 带 /web/）永久跳转到新地址
LEGACY = {
    "/web": "/volatility",
    "/web/index.html": "/volatility",
    "/web/index": "/volatility",
    "/index.html": "/volatility",
    "/web/points": "/ndx",
    "/points": "/ndx",
    "/web/qqq.html": "/qqq",
    "/web/qqq": "/qqq",
    "/qqq.html": "/qqq",
    "/web/ssec.html": "/a-shares",
    "/web/ssec": "/a-shares",
    "/ssec.html": "/a-shares",
    "/web/agent.html": "/agent",
    "/agent.html": "/agent",
    "/web/agent": "/agent",
}

# 顶层静态资源：/theme.css → /web/theme.css
ASSET_FILES = {"/theme.css", "/theme.js", "/ctxbar.js"}
ASSET_DIRS = ("/fonts/", "/vendor/")


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
    """Serves the repo root (so /web/*.html, /artifacts/*.json and /data/* all resolve)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, directory=str(ROOT), **kwargs)

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

    def _send_redirect(self, location: str) -> None:
        self.send_response(301)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _serve_file(self, relative: str, query: str) -> None:
        self.path = "/" + relative.lstrip("/") + (f"?{query}" if query else "")
        super().do_GET()

    def do_GET(self) -> None:
        path, _, query = self.path.partition("?")
        if path == "/api/refresh":
            self._handle_refresh(force="force=1" in query)
            return

        clean = path.rstrip("/") or "/"

        # 旧 /web/points.html?market=qqq 也要落到 /qqq
        if clean in ("/web/points.html", "/points.html"):
            market = dict(p.split("=", 1) for p in query.split("&") if "=" in p).get("market")
            self._send_redirect(LEGACY_MARKET.get(market or "", "/ndx"))
            return

        if clean in LEGACY:
            self._send_redirect(LEGACY[clean])
            return

        # 顶层静态资源，省掉 /web/ 前缀
        if clean in ASSET_FILES or clean.startswith(ASSET_DIRS):
            self._serve_file("web" + clean, query)
            return

        if clean in PAGES:
            self._serve_file(PAGES[clean], query)
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="forecast-log serve", description="本地看板服务")
    parser.add_argument("port", nargs="?", type=int, default=DEFAULT_PORT, help=f"端口，默认 {DEFAULT_PORT}")
    parser.add_argument("--no-refresh", action="store_true", help="启动时不自动拉行情")
    args = parser.parse_args(argv)

    if not args.no_refresh:
        threading.Thread(target=_startup_refresh, daemon=True).start()

    try:
        server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError as exc:
        print(f"端口 {args.port} 起不来（{exc}）。多半是已经跑着一个了：", file=sys.stderr)
        print(f"  lsof -nP -iTCP:{args.port} -sTCP:LISTEN", file=sys.stderr)
        print(f"  kill $(lsof -tiTCP:{args.port} -sTCP:LISTEN)", file=sys.stderr)
        print(f"或者换端口：forecast-log serve {args.port + 1}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"看板已启动： http://127.0.0.1:{args.port}/  （信号页 /ndx · /qqq · /a-shares；波动率区间 /volatility；评测 /agent）", flush=True)
    print("（Ctrl+C 停止）", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
