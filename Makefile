# 命名规则：
#   - 线上操作一律带平台前缀：supabase-* / cf-worker-* / cf-supabase-* / cf-d1-* / gh-db-*
#   - 明确区分增量与全量：*-sync / *-deploy 为增量，*-reseed 为全量清空重灌
#   - 运行 `make help` 查看分组清单；`make` 等价于 `make dev`

PORT ?= 8731
VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
FL := $(VENV)/bin/forecast-log

.DEFAULT_GOAL := dev
.PHONY: dev setup stop status refresh test help \
	gh-db-pull gh-db-pull-force gh-db-push \
	supabase-schema supabase-sync supabase-reseed \
	cf-build cf-worker-dev cf-worker-deploy \
	cf-supabase-deploy cf-supabase-reseed \
	cf-d1-deploy cf-d1-reseed cf-d1-seed cf-d1-local-schema cf-d1-local-sync

##@ 本地开发（不碰线上）
dev: $(PY) gh-db-pull ## 起本地看板（默认 8731 端口，缺库时自动拉取）
	@$(FL) serve $(PORT)

setup: $(PY) ## 只准备本地 .venv 环境，不启动服务

stop: ## 停掉占着本地 PORT 端口的进程
	@pid=$$(lsof -tiTCP:$(PORT) -sTCP:LISTEN); \
	if [ -n "$$pid" ]; then kill $$pid && printf '已停止 PID %s（端口 $(PORT)）\n' "$$pid"; \
	else echo "端口 $(PORT) 上没有进程"; fi

status: ## 查看本地 PORT 端口占用情况
	@lsof -nP -iTCP:$(PORT) -sTCP:LISTEN || echo "端口 $(PORT) 空闲"

refresh: $(PY) ## 拉最新 Yahoo 行情并重算信号，写入本地 data/forecast.db（需本地服务运行中）
	@curl -s --noproxy '*' "http://127.0.0.1:$(PORT)/api/refresh?force=1" | $(PY) -m json.tool

test: $(PY) ## 跑本地离线测试（pytest）
	@$(PY) -m pytest -q

help: ## 分组列出所有命令
	@awk 'BEGIN {FS = ":.*## "} \
		/^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } \
		/^[a-z0-9-]+:.*## / { printf "  make %-20s %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

$(PY): pyproject.toml
	@test -d $(VENV) || python3 -m venv $(VENV)
	@$(PIP) install -q -e ".[dev]"
	@touch $(PY)

##@ 一键线上发布（Cloudflare Worker + Supabase Postgres）
cf-supabase-deploy: ## [增量] 打包页面/派生数据 + 增量同步 Supabase + 更新 Hyperdrive + 部署 CF Worker
	@bash scripts/deploy_supabase.sh

cf-supabase-reseed: ## [全量重灌] 清空重灌 Supabase + 打包页面/派生数据 + 更新 Hyperdrive + 部署 CF Worker
	@RESEED=1 bash scripts/deploy_supabase.sh

##@ 只操作 Supabase 数据库（不动 Cloudflare）
supabase-sync: $(PY) ## [增量] 把本地 data/forecast.db 增量同步到线上 Supabase Postgres
	@$(PY) scripts/sync_supabase.py --schema

supabase-reseed: $(PY) ## [全量重灌] TRUNCATE 后把本地 data/forecast.db 全量重灌到线上 Supabase Postgres
	@$(PY) scripts/sync_supabase.py --schema --truncate

supabase-schema: $(PY) ## [仅建表] 在线上 Supabase Postgres 应用 deploy/schema.pg.sql
	@$(PY) scripts/sync_supabase.py --schema-only

##@ 只操作 Cloudflare Worker（不动数据库）
cf-worker-deploy: cf-build ## [只发代码/页面] 打包静态页与派生数据并部署线上 Cloudflare Worker
	@wrangler deploy

cf-worker-dev: cf-build ## 本地 workerd 调试 Cloudflare Worker（端口 8787）
	@wrangler dev --port 8787

cf-build: $(PY) ## 仅本地构建 deploy/dist（静态页面 + _meta 派生数据）
	@$(PY) scripts/build_static.py
	@$(PY) scripts/build_d1_payloads.py

##@ GitHub Release 本地库快照（data-latest）
gh-db-pull: ## 从 GitHub Release 下载 data/forecast.db（本地已存在则跳过）
	@bash scripts/fetch_db.sh

gh-db-pull-force: ## [全量覆盖] 从 GitHub Release 强制重新下载最新 data/forecast.db
	@FORCE=1 bash scripts/fetch_db.sh

gh-db-push: ## [全量覆盖] 把本地 data/forecast.db 上传覆盖到 GitHub Release data-latest
	@bash scripts/publish_db.sh

##@ 旧版 Cloudflare D1（仅作回滚 / 兼容备用）
cf-d1-deploy: ## [增量] 打包 + 部署 CF Worker + 增量同步到线上 Cloudflare D1
	@bash scripts/deploy_cf.sh

cf-d1-reseed: ## [全量重灌] 删除并重建线上 Cloudflare D1 全量导入 + 部署 CF Worker
	@RESEED=1 bash scripts/deploy_cf.sh

cf-d1-seed: $(PY) ## 仅本地从 data/forecast.db 生成 D1 全量种子 SQL（deploy/seed/）
	@$(PY) scripts/export_d1_seed.py

cf-d1-local-schema: ## 仅在本地 wrangler dev 的 D1 建表
	@wrangler d1 execute forecast --local --file=deploy/schema.sql

cf-d1-local-sync: $(PY) ## [增量] 把本地库增量推送到本地 wrangler dev 的 D1（8787）
	@$(PY) scripts/publish_d1.py --url http://127.0.0.1:8787
