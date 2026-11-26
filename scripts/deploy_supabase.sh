#!/usr/bin/env bash
# Deploy forecast-log to Cloudflare Worker + Static Assets backed by Supabase PostgreSQL.
#
#   make cf-supabase-deploy               # build assets → incremental sync Supabase → bind Hyperdrive → deploy CF Worker
#   make cf-supabase-reseed               # truncate & full reload Supabase from local data/forecast.db, then deploy CF Worker
#   USE_HYPERDRIVE=0 bash scripts/deploy_supabase.sh  # connect Worker via Wrangler secret DATABASE_URL instead of Hyperdrive
#
# Required in .env or environment (use Session Pooler port 5432 with Hyperdrive):
#   SUPABASE_DB_URL=postgresql://postgres.<ref>:<pass>@aws-0-<region>.pooler.supabase.com:5432/postgres
# Optional:
#   SUPABASE_DIRECT_URL=...       # Direct/Session URL for psql DDL + COPY (defaults to SUPABASE_DB_URL)
#   SUPABASE_REGION=us-east-1     # AWS region of your Supabase project (sets Worker placement hint)
set -euo pipefail
cd "$(dirname "$0")/.."

export PATH="/opt/homebrew/bin:$PATH"
export CI=1

PY=${PY:-.venv/bin/python}
NAME=forecast
HYPERDRIVE_NAME=${HYPERDRIVE_NAME:-forecast-supabase}
USE_HYPERDRIVE=${USE_HYPERDRIVE:-1}
RESEED=${RESEED:-0}

read_env() {
  local key="$1"
  $PY - "$key" <<'EOF'
import sys
from pathlib import Path
key = sys.argv[1]
for fname in (".env", ".dev.vars"):
    p = Path(fname)
    if not p.is_file():
        continue
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        if k.strip() == key:
            print(v.strip().strip("\"'"))
            sys.exit(0)
EOF
}

SUPABASE_DB_URL=${SUPABASE_DB_URL:-$(read_env SUPABASE_DB_URL)}
SUPABASE_DIRECT_URL=${SUPABASE_DIRECT_URL:-$(read_env SUPABASE_DIRECT_URL)}
SUPABASE_REGION=${SUPABASE_REGION:-$(read_env SUPABASE_REGION)}
INGEST_TOKEN=${INGEST_TOKEN:-$(read_env INGEST_TOKEN)}
INGEST_TOKEN=${INGEST_TOKEN:-$(openssl rand -hex 24)}

say() { printf "\n\033[1m%s\033[0m\n" "$1"; }

say "1/7 检查登录与配置"
if [ -z "$SUPABASE_DB_URL" ]; then
  echo "✘ 缺少 SUPABASE_DB_URL。请在 .env 里配置 Supabase Pooler 连接串："
  echo "  SUPABASE_DB_URL=postgresql://postgres.<ref>:<password>@aws-0-<region>.pooler.supabase.com:6543/postgres"
  exit 1
fi
if ! wrangler whoami </dev/null 2>/dev/null | grep -qi "account"; then
  echo "✘ 未登录 Cloudflare。先跑：wrangler login"
  exit 1
fi

say "2/7 安装 Worker 依赖 (pg)"
if [ ! -d node_modules/pg ]; then
  npm install
else
  echo "node_modules/pg 已就绪"
fi

say "3/7 打包静态资源与派生数据"
$PY scripts/build_static.py
$PY scripts/build_d1_payloads.py

say "4/7 初始化并同步 Supabase PostgreSQL"
SYNC_URL="${SUPABASE_DIRECT_URL:-$SUPABASE_DB_URL}"
if [ "$RESEED" = "1" ]; then
  $PY scripts/sync_supabase.py --url "$SYNC_URL" --schema --truncate
else
  $PY scripts/sync_supabase.py --url "$SYNC_URL" --schema
fi

say "5/7 配置 Worker 数据库连接"
if [ "$USE_HYPERDRIVE" = "1" ]; then
  find_hd_id() {
    wrangler hyperdrive list </dev/null 2>/dev/null | $PY -c "
import re, sys
name = '$HYPERDRIVE_NAME'
for line in sys.stdin:
    if name in line:
        m = re.search(r'\b([0-9a-f]{32})\b', line)
        if m:
            print(m.group(1))
            sys.exit(0)
" 2>/dev/null || true
  }
  HD_ID=$(find_hd_id)
  if [ -z "$HD_ID" ]; then
    echo "创建 Hyperdrive 配置: $HYPERDRIVE_NAME"
    HD_OUT=$(printf 'n\n' | wrangler hyperdrive create "$HYPERDRIVE_NAME" --connection-string="$SUPABASE_DB_URL" --caching-disabled=true 2>&1)
    echo "$HD_OUT"
    HD_ID=$(printf '%s\n' "$HD_OUT" | $PY -c "import re,sys; m=re.search(r'\b([0-9a-f]{32})\b', sys.stdin.read()); print(m.group(1) if m else '')")
    if [ -z "$HD_ID" ]; then
      HD_ID=$(find_hd_id)
    fi
  else
    echo "复用已有 Hyperdrive ($HYPERDRIVE_NAME): $HD_ID（更新连接串并关闭查询缓存）"
    wrangler hyperdrive update "$HD_ID" --connection-string="$SUPABASE_DB_URL" --caching-disabled=true </dev/null >/dev/null
  fi
  if [ -z "$HD_ID" ]; then
    echo "✘ 未能获取 Hyperdrive ID，请检查上方 wrangler 输出"
    exit 1
  fi
  $PY - "$HD_ID" "$SUPABASE_REGION" <<'EOF'
import json, re, sys
from pathlib import Path

hd_id, region = sys.argv[1], sys.argv[2].strip()
path = Path("wrangler.jsonc")
raw = path.read_text(encoding="utf-8")
# Strip line comments to parse JSON cleanly, then preserve structure
clean = re.sub(r'^\s*//.*$', '', raw, flags=re.M)
cfg = json.loads(clean)
cfg["hyperdrive"] = [{"binding": "HYPERDRIVE", "id": hd_id}]
if region:
    hint = region if ":" in region else f"aws:{region}"
    cfg["placement"] = {"region": hint}
path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
EOF
  echo "已将 HYPERDRIVE ($HD_ID) 写入 wrangler.jsonc"
else
  echo "USE_HYPERDRIVE=0：写入 Wrangler secret DATABASE_URL"
  printf '%s' "$SUPABASE_DB_URL" | wrangler secret put DATABASE_URL --name "$NAME"
fi

say "6/7 写入 INGEST_TOKEN"
printf '%s' "$INGEST_TOKEN" | wrangler secret put INGEST_TOKEN --name "$NAME"

say "7/7 部署 Worker"
DEPLOY_LOG=$(wrangler deploy </dev/null 2>&1 | tee /dev/stderr)
URL=$($PY -c "import json,re;t=open('wrangler.jsonc').read();t=re.sub(r'^\s*//.*$','',t,flags=re.M);r=json.loads(t).get('routes') or [];print('https://'+r[0]['pattern'] if r and r[0].get('custom_domain') else '')")
if [ -z "$URL" ]; then
  URL=$(printf '%s' "$DEPLOY_LOG" | grep -oE 'https://[A-Za-z0-9._-]+\.workers\.dev' | head -1 || true)
fi

say "完成"
echo "部署地址: $URL"
echo "检查状态: curl -s $URL/api/state | python3 -m json.tool"
