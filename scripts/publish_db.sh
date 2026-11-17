#!/usr/bin/env bash
# 把本地当前的 data/forecast.db 覆盖发布到 GitHub Release 的固定资产名上。
#
# 数据库不进 git 历史（见 scripts/fetch_db.sh 的说明），所以「推上去」= 覆盖
# Release 资产，而不是 git commit。同名覆盖，远端永远只有一份，不累积。
#
#   make data-publish          # 或 bash scripts/publish_db.sh
set -euo pipefail

REPO=${REPO:-lumixraku/forecast-log}
TAG=${TAG:-data-latest}
ASSET=${ASSET:-forecast.db}

root=$(cd "$(dirname "$0")/.." && pwd)
db="$root/data/forecast.db"

[ -f "$db" ] || { echo 'data/forecast.db 不存在，先跑 make data' >&2; exit 1; }

# sqlite3 是硬依赖：既要用它合并 WAL，也要用它做完整性检查。缺了就不发——
# 没 checkpoint 的快照可能缺最新写入，没校验的快照可能是坏库。
command -v sqlite3 >/dev/null 2>&1 || {
  echo '需要 sqlite3（合并 WAL + 完整性检查），当前没有，拒绝发布' >&2
  exit 1
}

# 1. 把 WAL 里的最新改动合并回主文件，否则上传的是半旧的快照
sqlite3 "$db" 'PRAGMA wal_checkpoint(TRUNCATE);' >/dev/null
echo 'WAL 已合并回 forecast.db'

# 2. 完整性检查，坏库不往外发
status=$(sqlite3 "$db" 'PRAGMA quick_check;' 2>&1 | head -1)
if [ "$status" != "ok" ]; then
  echo "数据库完整性检查失败：$status" >&2
  exit 1
fi

local_size=$(wc -c < "$db" | tr -d ' ')
printf '上传 forecast.db（%s 字节）到 %s@%s ...\n' "$local_size" "$TAG" "$REPO"

# 3. 覆盖同名资产；release 不存在就先建
if ! gh release view "$TAG" --repo "$REPO" >/dev/null 2>&1; then
  gh release create "$TAG" --repo "$REPO" --target main \
    --title 'Data snapshot (always the latest)' \
    --notes 'Current `data/forecast.db`. Overwritten in place by `make data-publish` — one asset, no history.' \
    --latest=false
fi
gh release upload "$TAG" "$db" --repo "$REPO" --clobber

# 4. 核对远端大小，确认真的换掉了
remote_size=$(gh release view "$TAG" --repo "$REPO" --json assets \
  -q ".assets[] | select(.name==\"$ASSET\") | .size")
if [ "$remote_size" = "$local_size" ]; then
  printf '已更新：远端 %s = %s 字节\n' "$ASSET" "$remote_size"
else
  printf '警告：远端 %s 是 %s 字节，本地是 %s 字节\n' "$ASSET" "$remote_size" "$local_size" >&2
  exit 1
fi
