#!/usr/bin/env bash
# 一条命令部署到 Cloudflare：D1 建库 → 建表 → 上传静态资源 → 推送数据。
#
#   make cf
#
# 全程零交互：Worker 和 D1 统一命名为 forecast，INGEST_TOKEN 未传时自动生成。
set -euo pipefail
cd "$(dirname "$0")/.."

export PATH="/opt/homebrew/bin:$PATH"
export CI=1

PY=${PY:-.venv/bin/python}
NAME=forecast
INGEST_TOKEN=${INGEST_TOKEN:-$(openssl rand -hex 24)}

say() { printf "\n\033[1m%s\033[0m\n" "$1"; }

say "1/7 检查登录"
if ! wrangler whoami </dev/null 2>/dev/null | grep -qi "account"; then
  echo "✘ 没登录。先跑： wrangler login"
  exit 1
fi

say "2/7 打包静态资源"
$PY scripts/build_static.py

say "3/7 D1 数据库 ($NAME)"
if wrangler d1 list --json </dev/null 2>/dev/null | grep -q "\"$NAME\""; then
  echo "已存在，复用"
else
  wrangler d1 create "$NAME" --no-update-config </dev/null
fi
DB_ID=$(wrangler d1 list --json </dev/null | $PY -c "import json,sys; print(next(d['uuid'] for d in json.load(sys.stdin) if d['name']=='$NAME'))")
echo "database_id: $DB_ID"
$PY - "$DB_ID" "$NAME" <<'EOF'
import re, sys
from pathlib import Path
path = Path("wrangler.jsonc")
text = path.read_text()
text = re.sub(r'"name":\s*"[^"]*"', f'"name": "{sys.argv[2]}"', text, count=1)
text = re.sub(r'"database_name":\s*"[^"]*"', f'"database_name": "{sys.argv[2]}"', text)
text = re.sub(r'"database_id":\s*"[^"]*"', f'"database_id": "{sys.argv[1]}"', text)
path.write_text(text)
EOF

say "4/7 建表（幂等）"
wrangler d1 execute "$NAME" --remote --file=deploy/schema.sql --yes </dev/null

say "5/7 写入 INGEST_TOKEN"
printf '%s' "$INGEST_TOKEN" | wrangler secret put INGEST_TOKEN --name "$NAME"

say "6/7 部署 Worker"
DEPLOY_LOG=$(wrangler deploy </dev/null 2>&1 | tee /dev/stderr)
# 主地址：wrangler.jsonc 里 routes 的第一条自定义域名（workers.dev 已关）
URL=$(${PY} -c "import json,re;t=open('wrangler.jsonc').read();t=re.sub(r'^\s*//.*$','',t,flags=re.M);r=json.loads(t).get('routes') or [];print('https://'+r[0]['pattern'] if r and r[0].get('custom_domain') else '')")
if [ -z "$URL" ]; then
  URL=$(printf '%s' "$DEPLOY_LOG" | grep -oE 'https://[A-Za-z0-9._-]+\.workers\.dev' | head -1 || true)
fi
if [ -z "$URL" ]; then
  echo "✘ 没从部署输出里找到地址，看看上面的日志"
  exit 1
fi
echo "部署地址: $URL"

say "7/7 把数据推进 D1"
$PY scripts/publish_d1.py --url "$URL" --token "$INGEST_TOKEN"

say "完成"
echo "信号页   $URL/"
echo "QQQ      $URL/qqq"
echo "A 股     $URL/a-shares"
echo "VXN      $URL/vxn"
echo "评测台   $URL/agent"
