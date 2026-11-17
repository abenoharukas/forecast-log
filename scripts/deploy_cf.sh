#!/usr/bin/env bash
# 一条命令部署到 Cloudflare：D1 建库 → 建表 → 上传静态资源 → 推送信号数据。
#
#   INGEST_TOKEN=一串随机字符 make cf
#
# 前置：能连网，且 `wrangler login` 已登录。缺哪一步会直接告诉你，不会瞎猜。
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
NAME=forecast-log

say() { printf "\n\033[1m%s\033[0m\n" "$1"; }

say "1/8 检查网络"
if ! curl -sf -o /dev/null --max-time 15 https://api.cloudflare.com/client/v4/; then
  echo "✘ 连不上 api.cloudflare.com。先开代理/联网，再重跑。"
  exit 1
fi

say "2/8 检查登录"
if ! wrangler whoami 2>/dev/null | grep -qi "account"; then
  echo "✘ 没登录。先跑： wrangler login"
  exit 1
fi

say "3/8 打包静态资源"
$PY scripts/build_static.py

say "4/8 D1 数据库"
if wrangler d1 list --json 2>/dev/null | grep -q "\"$NAME\""; then
  echo "已存在，复用"
else
  wrangler d1 create "$NAME"
fi
DB_ID=$(wrangler d1 list --json | $PY -c "import json,sys; print(next(d['uuid'] for d in json.load(sys.stdin) if d['name']=='$NAME'))")
echo "database_id: $DB_ID"
$PY - "$DB_ID" <<'EOF'
import re, sys
from pathlib import Path
path = Path("wrangler.jsonc")
text = path.read_text()
updated = re.sub(r'"database_id":\s*"[^"]*"', f'"database_id": "{sys.argv[1]}"', text)
path.write_text(updated)
EOF

say "5/8 建表（幂等）"
wrangler d1 execute "$NAME" --remote --file=deploy/schema.sql --yes

say "6/8 写入 INGEST_TOKEN"
if [ -n "${INGEST_TOKEN:-}" ]; then
  printf '%s' "$INGEST_TOKEN" | wrangler secret put INGEST_TOKEN --name "$NAME"
else
  echo "未设置 INGEST_TOKEN，跳过（之后单独跑 wrangler secret put INGEST_TOKEN）"
fi

say "7/8 部署 Worker"
DEPLOY_LOG=$(wrangler deploy 2>&1 | tee /dev/stderr)
URL=$(printf '%s' "$DEPLOY_LOG" | grep -oE 'https://[A-Za-z0-9._-]+\.workers\.dev' | head -1 || true)
if [ -z "$URL" ]; then
  echo "✘ 没从部署输出里找到 workers.dev 地址，看看上面的日志"
  exit 1
fi

say "8/8 把信号数据推进 D1"
if [ -n "${INGEST_TOKEN:-}" ]; then
  $PY scripts/publish_d1.py --url "$URL"
else
  echo "跳过（没有 INGEST_TOKEN）。之后跑： INGEST_TOKEN=... $PY scripts/publish_d1.py --url $URL"
fi

say "完成"
echo "信号页   $URL/"
echo "QQQ      $URL/qqq"
echo "A 股     $URL/a-shares"
echo "区间图   $URL/volatility"
echo "评测台   $URL/agent"
