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
import re
import sys
import threading
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
SHANGHAI = ZoneInfo("Asia/Shanghai")
_REFRESH_LOCK = threading.Lock()
_TIMELINE_LOCK = threading.Lock()
INTRADAY_COOLDOWN_SECONDS = 300
DEFAULT_PORT = 8731

# 干净的 URL → 实际文件。地址里不带 .html，也不带 /web/ 前缀。
PAGES = {
    "/": "web/points.html",
    "/signals": "web/points.html",
    "/ndx": "web/points.html",
    "/qqq": "web/points.html",
    "/a-shares": "web/points.html",
    "/ssec": "web/points.html",
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
    "/volatility": "/vxn",
    "/web": "/vxn",
    "/web/index.html": "/vxn",
    "/web/index": "/vxn",
    "/index.html": "/vxn",
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


def all_refresh_symbols(root: Path = ROOT) -> list[str]:
    """汇总项目中所有行情品种（含信号因子、对照 ETF、A 股、VXN 波动率族及库内全部已有品种）。"""
    from forecast_log.china import CHINA_SYMBOLS
    from forecast_log.commands import COMPARE_SYMBOLS, SYMBOLS
    from forecast_log.cross import CROSS_SYMBOLS
    from forecast_log.db import list_db_symbols
    from forecast_log.features import FEATURE_SYMBOLS

    return list(
        dict.fromkeys(
            (
                *SYMBOLS,
                *FEATURE_SYMBOLS,
                *CROSS_SYMBOLS,
                *COMPARE_SYMBOLS,
                *CHINA_SYMBOLS,
                *list_db_symbols(root),
            )
        )
    )


def _timeline_needs_update(root: Path = ROOT) -> bool:
    """检查 bars 表中的 ^VXN / ^NDX 最新日期或收盘价是否领先于 timeline_close 表。"""
    from forecast_log.db import connect, db_path

    path = db_path(root)
    if not path.is_file():
        return False
    conn = connect(path)
    try:
        for db_sym, tl_sym in (("^VXN", "vxn"), ("^NDX", "ndx")):
            bar_row = conn.execute(
                "SELECT date, close FROM bars WHERE symbol = ? ORDER BY date DESC LIMIT 1",
                (db_sym,),
            ).fetchone()
            tl_row = conn.execute(
                "SELECT date, close FROM timeline_close WHERE symbol = ? ORDER BY idx DESC LIMIT 1",
                (tl_sym,),
            ).fetchone()
            if bar_row is None:
                continue
            if tl_row is None or str(tl_row["date"]) != str(bar_row["date"]):
                return True
            if abs(float(tl_row["close"]) - float(bar_row["close"])) > 1e-4:
                return True
        return False
    finally:
        conn.close()


def _refresh_timeline_bg(root: Path = ROOT) -> None:
    if not _TIMELINE_LOCK.acquire(blocking=False):
        return
    try:
        if _timeline_needs_update(root):
            from forecast_log.commands import backtest_command

            backtest_command(root)
    except Exception as exc:
        print(f"后台更新 VXN timeline 失败：{exc}", flush=True)
    finally:
        _TIMELINE_LOCK.release()


def refresh_cross_data(sync_timeline: bool = False) -> dict:
    """从 Yahoo 并发增量拉取全库所有品种的未收盘（is_closed=0）及新增 K 线，更新单表 bars 与时间轴。"""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from forecast_log.artifacts_db import (
        cross_market_status,
        cross_sessions_summary,
    )
    from forecast_log.bars import save_series
    from forecast_log.commands import cross_command
    from forecast_log.db import get_series, unclosed_bars_summary
    from forecast_log.yahoo import fetch_incremental

    pre_unclosed_bars = unclosed_bars_summary(ROOT)

    symbols = all_refresh_symbols(ROOT)
    updated: dict[str, str] = {}
    symbol_closed: dict[str, bool] = {}
    errors: dict[str, str] = {}

    def _fetch_one(sym: str) -> tuple[str, object]:
        existing = get_series(ROOT, sym)
        return sym, fetch_incremental(sym, existing)

    with ThreadPoolExecutor(max_workers=min(12, max(1, len(symbols)))) as pool:
        futures = {pool.submit(_fetch_one, sym): sym for sym in symbols}
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                _, series = fut.result()
                save_series(ROOT, series)
                last_bar = series.bars[-1]
                updated[sym] = last_bar.date.isoformat()
                symbol_closed[sym] = bool(last_bar.is_closed)
            except Exception as exc:
                errors[sym] = str(exc)

    cross_command(ROOT)

    timeline_updating = _timeline_needs_update(ROOT)
    if timeline_updating:
        if sync_timeline:
            _refresh_timeline_bg(ROOT)
            timeline_updating = False
        else:
            threading.Thread(target=_refresh_timeline_bg, args=(ROOT,), daemon=True).start()

    post_status = cross_market_status(ROOT)
    post_unclosed_bars = unclosed_bars_summary(ROOT)

    newly_closed_bars: dict[str, list[str]] = {}
    for sym, prev_dates in pre_unclosed_bars.items():
        still = set(post_unclosed_bars.get(sym) or [])
        settled = [d for d in prev_dates if d not in still]
        if settled:
            newly_closed_bars[sym] = settled

    newly_closed = {
        mkt: newly_closed_bars[info["instrument"]]
        for mkt, info in post_status.items()
        if info.get("instrument") in newly_closed_bars
    }
    still_open = {
        mkt: info["unclosed_dates"]
        for mkt, info in post_status.items()
        if info.get("unclosed_dates")
    }

    return {
        "ok": True,
        "refreshed_at": datetime.now(SHANGHAI).isoformat(timespec="seconds"),
        "beijing_date": datetime.now(SHANGHAI).date().isoformat(),
        "sessions": cross_sessions_summary(ROOT),
        "market_status": post_status,
        "symbols_count": len(symbols),
        "updated": updated,
        "symbol_closed": symbol_closed,
        "pre_unclosed_bars": pre_unclosed_bars,
        "newly_closed": newly_closed,
        "newly_closed_bars": newly_closed_bars,
        "still_open": still_open,
        "still_open_bars": post_unclosed_bars,
        "timeline_updating": timeline_updating,
        "errors": errors,
    }


def _parse_fetched_dt(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo("America/New_York"))
        return dt
    except Exception:
        return None


def _needs_refresh(intraday_cooldown_sec: int = INTRADAY_COOLDOWN_SECONDS) -> bool:
    from forecast_log.artifacts_db import cross_fetched_at
    from forecast_log.db import refresh_check_summary

    # 1. 检查全库未收盘行：
    #    - 已过收盘时间但仍是 is_closed=0 的残缺行 -> 无视冷却立即重抓！
    #    - 仍在盘中交易的 is_closed=0 行 -> 超过冷却时间（默认 5 分钟）才自动重抓，避免频繁刷页白跑。
    check = refresh_check_summary(ROOT, intraday_cooldown_sec=intraday_cooldown_sec)
    if check["needs_refresh"]:
        return True

    ny_tz = ZoneInfo("America/New_York")
    now_sh = datetime.now(SHANGHAI)
    now_ny = datetime.now(ny_tz)

    # 2. 按各市场所在时区的开盘/收盘时间点检查是否已有新交易时段开启或刚跨过收盘点
    for market in ("ndx", "qqq", "ssec"):
        fetched_dt = _parse_fetched_dt(cross_fetched_at(ROOT, market))
        if fetched_dt is None:
            return True
        if market == "ssec":
            local_now = now_sh
            local_fetched = fetched_dt.astimezone(SHANGHAI)
            open_hm, close_hm = (9, 25), (15, 5)
        else:
            local_now = now_ny
            local_fetched = fetched_dt.astimezone(ny_tz)
            open_hm, close_hm = (9, 30), (16, 15)

        if local_fetched.date() < local_now.date():
            if (local_fetched.hour, local_fetched.minute) < close_hm:
                return True
            if local_now.weekday() < 5 and (local_now.hour, local_now.minute) >= open_hm:
                return True
        elif local_fetched.date() == local_now.date():
            if (local_fetched.hour, local_fetched.minute) < close_hm <= (local_now.hour, local_now.minute):
                return True

    return False


def _cached_status() -> dict:
    from forecast_log.artifacts_db import (
        cross_market_status,
        cross_sessions_summary,
    )
    from forecast_log.db import unclosed_bars_summary

    status = cross_market_status(ROOT)
    return {
        "ok": True,
        "cached": True,
        "beijing_date": datetime.now(SHANGHAI).date().isoformat(),
        "sessions": cross_sessions_summary(ROOT),
        "market_status": status,
        "newly_closed": {},
        "still_open": {mkt: info["unclosed_dates"] for mkt, info in status.items() if info.get("unclosed_dates")},
        "still_open_bars": unclosed_bars_summary(ROOT),
    }


class Handler(http.server.SimpleHTTPRequestHandler):
    """Serves the repo root (so /web/*.html and /api/* all resolve)."""

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

    def _send_json(self, doc: dict | None, missing_msg: str) -> None:
        if doc is None:
            body = json.dumps({"ok": False, "error": missing_msg}, ensure_ascii=False).encode("utf-8")
            self.send_response(404)
        else:
            body = json.dumps(doc, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path, _, query = self.path.partition("?")
        if path in ("/api/refresh", "/api/fetch"):
            self._handle_refresh(force="force=1" in query)
            return

        api = re.fullmatch(r"/api/cross/([a-z-]+)", path.rstrip("/"))
        if api:
            from forecast_log.artifacts_db import load_cross_market

            slug = api.group(1)
            self._send_json(load_cross_market(ROOT, slug), f"no data for {slug}")
            return

        if path.rstrip("/") == "/api/timeline":
            from forecast_log.artifacts_db import load_timeline

            self._send_json(load_timeline(ROOT), "no timeline data in database")
            return

        agent_api = re.fullmatch(r"/api/agent/([a-z0-9_-]+)", path.rstrip("/"))
        if agent_api:
            from forecast_log.artifacts_db import load_agent_eval

            dataset = agent_api.group(1)
            self._send_json(load_agent_eval(ROOT, dataset), f"no agent data for {dataset}")
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
        if path in ("/api/refresh", "/api/fetch"):
            self._handle_refresh(force="force=1" in query)
            return
        self.send_error(405, "Method Not Allowed")

    def log_message(self, *args) -> None:
        pass


def _startup_refresh() -> None:
    """服务启动后始终自动调用一次全品种增量刷新函数，补齐所有未收盘及最新数据。"""
    print("启动自动刷新：正在增量拉取全品种最新行情（含未收盘数据重抓）…", flush=True)
    with _REFRESH_LOCK:
        try:
            res = refresh_cross_data()
            print(
                f"行情更新完成：{res.get('sessions')}（共更新 {len(res.get('updated', {}))} 个品种）",
                flush=True,
            )
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

    print(f"看板已启动： http://127.0.0.1:{args.port}/  （信号页 /ndx · /qqq · /a-shares；VXN /vxn；评测 /agent）", flush=True)
    print("（Ctrl+C 停止）", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
