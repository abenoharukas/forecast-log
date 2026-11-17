#!/usr/bin/env python3
"""把静态部分打包到 deploy/dist/，供 Cloudflare Workers Static Assets 使用。

只搬页面真正会在浏览器里请求的东西：`web/` 整个目录（含字体和图表库）加上
6 个 `artifacts/*.json`。`data/forecast.db`（81 MB）和 `data/raw/` 不进包——
它们是 Python 侧算信号用的，浏览器从不下载。

    python3 scripts/build_static.py          # 生成 deploy/dist
    python3 scripts/build_static.py --check  # 只报体积，不写文件
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deploy" / "dist"

# 页面会在运行时 fetch 的产物。少一个，对应页面就是空白。
ARTIFACTS = (
    "cross.json",  # /ndx
    "cross_qqq.json",  # /qqq
    "cross_ssec.json",  # /a-shares
    "timeline.json",  # /volatility
    "agent_eval_summary.json",  # /agent（DeepSeek 标签）
    "agent_eval_qwen_summary.json",  # /agent（Qwen 标签）
)


def build() -> dict[str, int]:
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "artifacts").mkdir(parents=True)

    shutil.copytree(ROOT / "web", OUT / "web")

    missing = []
    for name in ARTIFACTS:
        src = ROOT / "artifacts" / name
        if not src.is_file():
            missing.append(name)
            continue
        shutil.copy2(src, OUT / "artifacts" / name)
    if missing:
        raise SystemExit(f"缺少产物：{', '.join(missing)}（先跑 forecast-log cross / backtest）")

    files = [p for p in OUT.rglob("*") if p.is_file()]
    return {"files": len(files), "bytes": sum(p.stat().st_size for p in files)}


def main() -> None:
    if "--check" in sys.argv:
        total = 0
        count = 0
        for path in [*(ROOT / "web").rglob("*"), *(ROOT / "artifacts" / n for n in ARTIFACTS)]:
            if path.is_file():
                total += path.stat().st_size
                count += 1
        print(f"{count} 个文件，{total / 1048576:.2f} MB（未写入）")
        return
    stats = build()
    print(f"deploy/dist：{stats['files']} 个文件，{stats['bytes'] / 1048576:.2f} MB")


if __name__ == "__main__":
    main()
