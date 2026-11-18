# Cloudflare 部署（D1 + Worker + Static Assets）

线上数据库和本地 `data/forecast.db` **同一套结构、同一份内容**：

| 表 | 内容 | 本地行数（示例） |
|---|---|---|
| `bars` | 全品种客观日线 OHLCV | ~107,600 |
| `signals` | 三个信号市场的信号列 + `factors_packed` | ~13,700 |
| `news` | 每日新闻文档 | ~127,900 |
| `agent_eval` | LLM 盲测结果扁平表 | ~1,150 |
| `chat_messages` | DeepSeek × Qwen 群聊记录 | 少量 |

结构定义来自 `src/forecast_log/db.py::ensure_schema`，部署用的 SQL 是
`deploy/schema.sql`（全部 `CREATE TABLE IF NOT EXISTS`，可重复执行）。本地库新增表/列时，
两处要一起改。

## 一条命令

```bash
make cf              # 常规部署：建库（不存在时）、建表、空库时灌种子、部署、增量同步
RESEED=1 make cf     # 删掉线上 D1，按本地 data/forecast.db 重新全量导入
```

`scripts/deploy_cf.sh` 的步骤：

1. 检查 `wrangler login`
2. `scripts/build_static.py` + `scripts/build_d1_payloads.py` 打包页面与派生数据
3. D1 建库/复用（`RESEED=1` 时先 `wrangler d1 delete`），把 `database_id` 写回 `wrangler.jsonc`
4. `wrangler d1 execute --file=deploy/schema.sql`
5. 空库时 `scripts/export_d1_seed.py` 生成 `deploy/seed/*.sql` 并逐片导入
6. `wrangler secret put INGEST_TOKEN`
7. `wrangler deploy`
8. `scripts/publish_d1.py` 做一次增量同步（刚灌过种子时基本空转）

`deploy/seed/` 是 70 MB 左右的生成物，已在 `.gitignore` 里；`make seed` 可单独重新生成。

## 数据怎么读

Worker 直接查上面五张表，不再有 `market` / `session` / `timeline_*` 这些旧表：

| 接口 | 数据来源 | 说明 |
|---|---|---|
| `/api/cross/ndx\|qqq\|ssec` | `signals LEFT JOIN bars` | 逐日序列、命中率汇总、A 股分档在 Worker 里算 |
| `/api/agent/deepseek\|qwen` | `agent_eval` | Brier / LogLoss / ECE 在 Worker 里重算 |
| `/api/timeline` | `_meta/timeline.json` 静态资源 | VXN 类比模型是 ~900 行 numpy，部署时算好 |
| `/api/state` | 五张表 | 给增量同步用的水位线 |
| `/api/chat/threads\|messages` | `chat_messages` | 群聊历史 |

`_meta/` 下的 `cross-*.json` 只放**静态定义**（问题文案、开关列表、日历/新闻来源说明）；
每天的数值仍然来自 D1。`_meta/timeline.json` 是唯一整份缓存的派生数据——它无法在
Worker 的 CPU 预算里重算，数据变了要重新部署静态资源（`make cf`）。

`/api/chat/run`（真正的多模型群聊）仍只在本地 Python 服务里有，线上 Worker 不跑 LLM。

## 增量同步

`scripts/publish_d1.py` 不是全量上传：

- `bars`：每个品种回传最近 20 天的尾部（覆盖盘中修订和 `is_closed`）
- `signals`：最近 20 天 + 所有 `outcome IS NULL` 的行
- `news`：最近 3 天的 `as_of`（同一天可能补抓文档）
- `agent_eval`：`COUNT` 或 `generated_at` 变了才整份重发
- `chat_messages`：水位线之后的新消息

写入走 `POST /api/ingest`，Worker 只接受对五张表的 `INSERT [OR REPLACE|IGNORE]`，
单条 SQL 上限 95 KB。token 泄漏也删不了库。

## `/api/refresh`

线上只做两件安全的事：从 Yahoo 补 `bars`，以及按新 K 线重算已存在信号行的
`entry/close3/close5/hit3/outcome/peak`。**它不会生成新交易日的信号**——`flag` / `score`
依赖完整的 Python 因子计算。要推进到新的一天：

```bash
make refresh     # 本地拉行情并重算信号
make cf          # 重新打包派生数据 + 增量同步
```

## 本地验证

`wrangler dev --local` + 本地 D1，逐接口和 Python 产物对比：

- `/api/cross/ndx|qqq|ssec`、`/api/agent/deepseek|qwen`、`/api/timeline` 与本地
  `load_cross_market` / `load_agent_eval` / `load_timeline` **逐字段一致**（含浮点末位；
  `round()` 和 `statistics.fmean` 都在 Worker 里按 Python 语义实现）
- 页面 `/`、`/ndx`、`/qqq`、`/a-shares`、`/vxn`、`/agent`、`/sim` 均返回 200
- 群聊历史、`/api/state` 从 D1 读出

## 额度

全库约 100 MB，远低于 D1 单库上限。`news` 占大头（约 62 MB），写入主要发生在全量
seed；日常增量每天只有几十到几百行。
