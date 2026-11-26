#!/usr/bin/env bash
# Deploy forecast-log to Cloudflare: D1 schema/seed → static assets → Worker → incremental sync.
#
#   make cf-d1-deploy       # normal deploy; creates the DB if missing, seeds only when empty
#   make cf-d1-reseed       # drop the D1 database and reload it from local data/forecast.db
#
# The D1 schema mirrors the local SQLite exactly (bars / signals / news / agent_eval /
# chat_messages). Derived payloads that cannot be recomputed cheaply at the edge
# (`_meta/timeline.json`, `_meta/cross-*.json`) are uploaded with the static assets.
set -euo pipefail
cd "$(dirname "$0")/.."

export PATH="/opt/homebrew/bin:$PATH"
export CI=1

PY=${PY:-.venv/bin/python}
NAME=forecast
SEED_DIR=deploy/seed
if [ -z "${INGEST_TOKEN:-}" ] && [ -f .env ]; then
  INGEST_TOKEN=$(
    $PY -c '
from pathlib import Path
for line in Path(".env").read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, _, v = line.partition("=")
    if k.strip() == "INGEST_TOKEN":
        print(v.strip().strip("\"\x27"))
        break
' 2>/dev/null || true
  )
fi
INGEST_TOKEN=${INGEST_TOKEN:-$(openssl rand -hex 24)}
RESEED=${RESEED:-0}

say() { printf "\n\033[1m%s\033[0m\n" "$1"; }

say "1/8 检查登录"
if ! wrangler whoami </dev/null 2>/dev/null | grep -qi "account"; then
  echo "✘ 没登录。先跑： wrangler login"
  exit 1
fi

say "2/8 打包静态资源与派生数据"
$PY scripts/build_static.py
$PY scripts/build_d1_payloads.py

say "3/8 D1 数据库 ($NAME)"
EXISTS=$(wrangler d1 list --json </dev/null 2>/dev/null | $PY -c "import json,sys; print(any(d['name']=='$NAME' for d in json.load(sys.stdin)))")
if [ "$RESEED" = "1" ] && [ "$EXISTS" = "True" ]; then
  echo "RESEED=1：删除旧库 $NAME"
  wrangler d1 delete "$NAME" --skip-confirmation </dev/null
  EXISTS=False
fi
if [ "$EXISTS" != "True" ]; then
  echo "创建新库（名称固定为 ${NAME}）"
  wrangler d1 create "$NAME" --no-update-config </dev/null
else
  echo "已存在，复用"
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

say "4/8 建表（幂等）"
wrangler d1 execute "$NAME" --remote --file=deploy/schema.sql --yes </dev/null

say "5/8 全量种子数据"
ROWS=$(wrangler d1 execute "$NAME" --remote --json --command "SELECT COUNT(*) AS n FROM bars" </dev/null 2>/dev/null \
  | $PY -c "import json,sys; print(json.load(sys.stdin)[0]['results'][0]['n'])" 2>/dev/null || echo 0)
if [ "$ROWS" = "0" ] || [ ! -d "$SEED_DIR" ] || [ -z "$(ls -A "$SEED_DIR" 2>/dev/null)" ]; then
  echo "生成 SQL 种子（本地 data/forecast.db → ${SEED_DIR}）"
  $PY scripts/export_d1_seed.py
  total=$(ls "$SEED_DIR"/*.sql | wc -l | tr -d ' ')
  index=0
  for file in "$SEED_DIR"/*.sql; do
    index=$((index + 1))
    printf '  %s/%s %s\r' "$index" "$total" "$(basename "$file")"
    wrangler d1 execute "$NAME" --remote --file="$file" --yes </dev/null >/dev/null
  done
  printf '\r  导入完成：%s 个文件\n' "$total"
else
  echo "D1 已有 $ROWS 行 bars，跳过全量种子（需要重灌请用 make cf-d1-reseed）"
fi

say "6/8 写入 INGEST_TOKEN"
printf '%s' "$INGEST_TOKEN" | wrangler secret put INGEST_TOKEN --name "$NAME"

say "7/8 部署 Worker"
DEPLOY_LOG=$(wrangler deploy </dev/null 2>&1 | tee /dev/stderr)
URL=$(${PY} -c "import json,re;t=open('wrangler.jsonc').read();t=re.sub(r'^\s*//.*$','',t,flags=re.M);r=json.loads(t).get('routes') or [];print('https://'+r[0]['pattern'] if r and r[0].get('custom_domain') else '')")
if [ -z "$URL" ]; then
  URL=$(printf '%s' "$DEPLOY_LOG" | grep -oE 'https://[A-Za-z0-9._-]+\.workers\.dev' | head -1 || true)
fi
if [ -z "$URL" ]; then
  echo "✘ 没从部署输出里找到地址，看看上面的日志"
  exit 1
fi
echo "部署地址: $URL"

say "8/8 增量同步（种子已灌过则基本是空转）"
$PY scripts/publish_d1.py --url "$URL" --token "$INGEST_TOKEN"

say "完成"
echo "信号页   $URL/"
echo "QQQ      $URL/qqq"
echo "A 股     $URL/a-shares"
echo "VXN      $URL/vxn"
echo "评测台   $URL/agent"
