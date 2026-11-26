# 架构与用法（Cloudflare D1）

线上 `forecast.lumixraku.org` 的数据库和本地 `data/forecast.db` **同一套结构、同一份内容**。

## 数据模型

全库只有 5 张业务表，定义在 `src/forecast_log/db.py::ensure_schema`，部署用 SQL 是
`deploy/schema.sql`（两边要一起改）：

| 表 | 内容 |
|---|---|
| `bars` | 全品种日线 OHLCV、`is_closed`、`fetched_at` |
| `signals` | 三个信号市场的信号列 + `factors_packed` 位图 |
| `news` | 每日新闻标题、正文与来源 |
| `agent_eval` | LLM 盲测结果扁平表（一题 × 一个预测模型一行） |
| `chat_messages` | DeepSeek × Qwen 群聊历史 |

旧版的 `market` / `session` / `market_node` / `contexts` / `factor_key` /
`timeline_*` / `agent_dataset` / `agent_node` 已全部废弃，D1 上不再存在。

### 读路径

| 接口 | 数据来源 |
|---|---|
| `/api/cross/{ndx,qqq,ssec}` | `signals LEFT JOIN bars`，汇总/分档/对比在 Worker 里算 |
| `/api/agent/{deepseek,qwen}` | `agent_eval`，Brier / LogLoss / ECE 在 Worker 里算 |
| `/api/timeline` | `_meta/timeline.json` 静态资源（VXN 类比模型是 ~900 行 numpy，部署时算好） |
| `/api/state` | 五张表的水位线，给增量同步用 |
| `/api/chat/threads`、`/api/chat/messages` | `chat_messages` |
| `/api/refresh` | 只补 `bars` + 结算已有信号，不生成新交易日的信号 |
| `/api/ingest` | 只接受对五张表的 `INSERT [OR REPLACE\|IGNORE]`，单条 SQL ≤ 95 KB |

`_meta/cross-*.json` 只放静态定义（问题文案、开关列表、日历/新闻来源说明），每天的数值仍来自 D1。

## 用法

所有线上命令均以平台为前缀区分操作对象，并以 `deploy/sync`（增量）和 `reseed`（全量清空重灌）区分更新方式：

### 日常部署 / 数据更新（Supabase + Cloudflare Worker，当前线上）

```bash
# 1. 先把本地数据推进一步（拉 Yahoo 行情、重算信号写入 data/forecast.db）
make refresh

# 2. 打包页面与派生数据 + 增量同步 Supabase + 更新 Hyperdrive + 部署 CF Worker
make cf-supabase-deploy
```

需要全量清空重灌 Supabase 并部署 Worker：

```bash
make cf-supabase-reseed
```

只操作单边：

```bash
make supabase-sync        # 仅增量同步本地库 → 线上 Supabase（不动 CF Worker）
make supabase-reseed      # 仅清空并全量重灌线上 Supabase（不动 CF Worker）
make supabase-schema      # 仅在线上 Supabase 应用 deploy/schema.pg.sql
make cf-worker-deploy     # 仅打包静态页/派生数据并部署 CF Worker（不动数据库）
make cf-worker-dev        # 本地 workerd 调试 CF Worker（端口 8787）
make cf-build             # 仅本地生成 deploy/dist
```

### 旧版 Cloudflare D1（仅作回滚备用）

```bash
make cf-d1-deploy         # 打包 + 部署 CF Worker + 增量同步到线上 CF D1
make cf-d1-reseed         # 删库重建并全量导入线上 CF D1 + 部署 CF Worker
make cf-d1-seed           # 仅从 data/forecast.db 生成 deploy/seed/*.sql
make cf-d1-local-schema   # 仅在本地 wrangler dev 的 D1 建表
make cf-d1-local-sync     # 仅把本地库增量推送到本地 wrangler dev 的 D1
```

> D1 全量导入按 `deploy/seed/*.sql` 逐片执行，单文件约 3 MB。网络中断时脚本会停在当前
> 分片；已写入的分片可安全重跑（`INSERT OR REPLACE`）。恢复后可只重跑失败的分片，
> 再执行一次 `make cf-d1-deploy` 跳过种子继续部署。

### 增量同步范围

`scripts/publish_d1.py` 不做全量上传：

- `bars`：每个品种最近 20 天的尾部（覆盖盘中修订和 `is_closed`）
- `signals`：最近 20 天 + 所有 `outcome IS NULL` 的行
- `news`：最近 3 天的 `as_of`（同一天可能补抓文档）
- `agent_eval`：`COUNT` 或 `generated_at` 变了才整份重发
- `chat_messages`：水位线之后的新消息

## 文件职责

| 文件 | 作用 |
|---|---|
| `deploy/schema.sql` | 线上建表 SQL（和 `db.py::ensure_schema` 对应） |
| `deploy/worker.mjs` | Worker：页面路由 + 五张表的读接口 + 增量写入 |
| `scripts/build_static.py` | 把 `web/` 复制到 `deploy/dist` |
| `scripts/build_d1_payloads.py` | 生成 `_meta/cross-*.json` 和 `_meta/timeline.json` |
| `scripts/export_d1_seed.py` | 本地库 → `deploy/seed/*.sql`（按 60 KB/语句、3 MB/文件切片） |
| `scripts/deploy_supabase.sh` | `make cf-supabase-deploy` / `make cf-supabase-reseed` 的完整流程 |
| `scripts/sync_supabase.py` | 本地库 → 线上 Supabase Postgres（`COPY` + `ON CONFLICT` 增量/全量同步） |
| `scripts/deploy_cf.sh` | 旧版 `make cf-d1-deploy` / `make cf-d1-reseed` 的完整流程 |
| `scripts/publish_d1.py` | 旧版 D1 增量推送五张表 |
| `deploy/dist/`、`deploy/seed/` | 生成物，已在 `.gitignore` |

## 验证

本地 `wrangler dev --local` 和线上都做过逐字段比对：

- `/api/cross/{ndx,qqq,ssec}`、`/api/agent/{deepseek,qwen}`、`/api/timeline`
  与本地 `load_cross_market` / `load_agent_eval` / `load_timeline` 完全一致
- Worker 里按 Python 语义实现了 `round()`（half-even）和 `statistics.fmean`（精确求和），
  浮点末位不漂移
- 页面 `/`、`/ndx`、`/qqq`、`/a-shares`、`/vxn`、`/agent`、`/sim` 均 200

## 注意

- `/api/timeline` 和 cross 静态定义随**静态资源**发布，所以改完数据要 `make cf-supabase-deploy`（或至少 `make cf-worker-deploy`）；
  只跑 `make supabase-sync` 不会更新 timeline。
- `/api/chat/run`（真正的多模型群聊）仍只在本地 Python 服务里，线上 Worker 不跑 LLM。
- `/api/refresh` 只结算已有信号；要推进到新交易日必须本地 `make refresh` 后重新部署。
- 若切回旧版 D1 且修改了 `wrangler.jsonc` 的 `database_id`，本地 `wrangler dev` 的 D1 是新库，
  需要重新 `make cf-d1-local-schema` + 导入种子。
- 更多部署细节见 `deploy/README.md`。
