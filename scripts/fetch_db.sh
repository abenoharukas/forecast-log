#!/usr/bin/env bash
# 下载「当前最新」那一份 data/forecast.db。
#
# 为什么不是 git 跟踪：SQLite 是二进制文件，每次改一行字节就全变，git/LFS 只能
# 按内容哈希存新副本、且永不覆盖 —— 仓库历史会以每个版本 ~90MB 的速度膨胀。
# 所以数据库不进版本历史，改为发布到 GitHub Release 的固定资产名上（同名覆盖，
# 只留最新一份）；clone 之后跑一次本脚本就拿到当前数据。
#
#   bash scripts/fetch_db.sh              # 本地没有才下载
#   FORCE=1 bash scripts/fetch_db.sh      # 强制重新下载最新
set -euo pipefail

REPO=${REPO:-lumixraku/forecast-log}
TAG=${TAG:-data-latest}
ASSET=${ASSET:-forecast.db}
DEST=${DEST:-data/forecast.db}
FORCE=${FORCE:-0}

root=$(cd "$(dirname "$0")/.." && pwd)
dest_path="$root/$DEST"

if [ -f "$dest_path" ] && [ "$FORCE" != "1" ]; then
  printf 'data/forecast.db 已存在（%s），跳过下载。要更新到最新：make data-refresh\n' "$(du -h "$dest_path" | cut -f1)"
  exit 0
fi

tmp="${dest_path}.download"
mkdir -p "$(dirname "$dest_path")"

printf '从 GitHub Release 下载 %s（tag: %s）...\n' "$ASSET" "$TAG"
if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
  gh release download "$TAG" --repo "$REPO" --pattern "$ASSET" --clobber --output "$tmp"
else
  curl -fL --progress-bar \
    "https://github.com/$REPO/releases/download/$TAG/$ASSET" -o "$tmp"
fi

# 完整性校验：SQLite 文件头必须是 "SQLite format 3"（防止拿到 404 页面或半截文件）
if ! head -c 15 "$tmp" | grep -q 'SQLite format 3'; then
  rm -f "$tmp"
  echo '下载下来的不是有效的 SQLite 数据库，已丢弃' >&2
  exit 1
fi

mv "$tmp" "$dest_path"
printf '已就位 %s（%s）\n' "$DEST" "$(du -h "$dest_path" | cut -f1)"
