# 一个命令就能跑起来。`make` 等价于 `make dev`。
#
#   make dev       起本地看板（首次会自动建 .venv、装依赖、拉最新数据库）
#   make data      拉最新一份 data/forecast.db（数据库不进 git 历史）
#   make data-publish  改完数据库后把它覆盖发布到 data-latest
#   make dev PORT=9000
#   make stop      停掉占着端口的那个进程
#   make status    看看端口上是谁
#   make refresh   强制拉一次最新行情并重算 artifacts/cross*.json
#   make test      跑测试
#   make help      列出所有命令

PORT ?= 8731
VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
FL := $(VENV)/bin/forecast-log

.DEFAULT_GOAL := dev
.PHONY: dev setup data data-refresh stop status refresh static worker deploy cf d1-schema publish test help

dev: $(PY) data ## 起本地看板（默认 8731 端口）
	@$(FL) serve $(PORT)

setup: $(PY) ## 只准备环境，不启动

data: ## 拉最新一份 data/forecast.db（已存在则跳过）
	@bash scripts/fetch_db.sh

data-refresh: ## 强制重新下载最新的 data/forecast.db
	@FORCE=1 bash scripts/fetch_db.sh

data-publish: ## 把当前 data/forecast.db 覆盖发布到 data-latest（同名覆盖，需要 gh 登录）
	@gh release upload data-latest data/forecast.db --clobber

$(PY): pyproject.toml
	@test -d $(VENV) || python3 -m venv $(VENV)
	@$(PIP) install -q -e ".[dev]"
	@touch $(PY)

stop: ## 停掉占着端口的进程
	@pid=$$(lsof -tiTCP:$(PORT) -sTCP:LISTEN); \
	if [ -n "$$pid" ]; then kill $$pid && printf '已停止 PID %s（端口 $(PORT)）\n' "$$pid"; \
	else echo "端口 $(PORT) 上没有进程"; fi

status: ## 看端口上是谁
	@lsof -nP -iTCP:$(PORT) -sTCP:LISTEN || echo "端口 $(PORT) 空闲"

static: $(PY) ## 打包静态站点到 deploy/dist（Cloudflare 用）
	@$(PY) scripts/build_static.py

worker: static ## 用 workerd 本地跑一遍 Cloudflare Worker
	@wrangler dev --port 8787

deploy: static ## 只发静态资源（旧方式，不碰 D1）
	@wrangler deploy

cf: ## 一条命令部署到 Cloudflare，数据进 D1（需先 wrangler login）
	@bash scripts/deploy_cf.sh

d1-schema: ## 只在本地 D1 建表（wrangler dev 用）
	@wrangler d1 execute forecast-log --local --file=deploy/schema.sql

publish: ## 把 artifacts 里的信号推进本地 wrangler dev 的 D1
	@.venv/bin/python scripts/publish_d1.py --url http://127.0.0.1:8787

refresh: $(PY) ## 强制刷新行情（服务需在跑）
	@curl -s --noproxy '*' "http://127.0.0.1:$(PORT)/api/refresh?force=1" | $(PY) -m json.tool

test: $(PY) ## 跑测试
	@$(PY) -m pytest -q

help: ## 列出所有命令
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-12s %s\n", $$1, $$2}'
