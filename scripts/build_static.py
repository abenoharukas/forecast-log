#!/usr/bin/env python3
"""把静态部分打包到 deploy/dist/，供 Cloudflare Workers Static Assets 使用。

只搬页面和静态资源：`web/` 整个目录（含字体和图表库）。
全部看板数据（cross、timeline、agent）都已放进 D1，不再打包任何 `artifacts/*.json`。

    python3 scripts/build_static.py          # 生成 deploy/dist
    python3 scripts/build_static.py --check  # 只报体积，不写文件
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deploy" / "dist"


def build() -> dict[str, int]:
    if OUT.exists():
        shutil.rmtree(OUT)
    shutil.copytree(ROOT / "web", OUT / "web")
    files = [p for p in OUT.rglob("*") if p.is_file()]
    return {"files": len(files), "bytes": sum(p.stat().st_size for p in files)}


def main() -> None:
    if "--check" in sys.argv:
        total = 0
        count = 0
        for path in (ROOT / "web").rglob("*"):
            if path.is_file():
                total += path.stat().st_size
                count += 1
        print(f"{count} 个文件，{total / 1048576:.2f} MB（未写入）")
        return
    stats = build()
    print(f"deploy/dist：{stats['files']} 个文件，{stats['bytes'] / 1048576:.2f} MB")


if __name__ == "__main__":
    main()
