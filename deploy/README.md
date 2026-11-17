# Cloudflare 部署（数据在 D1）

信号页的三个市场（`^NDX` / `QQQ` / `000001.SS`）已经改成**从 D1 读数据**，
不再读任何 `artifacts/*.json`。D1 是 Cloudflare 的 SQLite，免费额度内。

## 一条命令

```bash
INGEST_TOKEN=$(openssl rand -hex 24) make cf
```

`scripts/deploy_cf.sh` 按顺序做 8 件事，任何一步不通就停下并说清楚缺什么：

1. 检查能不能连上 `api.cloudflare.com`
2. 检查 `wrangler login` 是否有效
3. 打包 `deploy/dist`（页面 + 字体 + 图表库，6.22 MB）
4. D1 建库（已存在就复用），把 `database_id` 写回 `wrangler.jsonc`
5. 建表：`deploy/schema.sql`（全是 `CREATE TABLE IF NOT EXISTS`，可重复跑）
6. `wrangler secret put INGEST_TOKEN`
7. `wrangler deploy`，从输出里抓出 `*.workers.dev` 地址
8. `scripts/publish_d1.py` 把当天算好的信号写进 D1

## 数据怎么进的库

```
本地 Python 算信号          线上
──────────────            ─────────────────────────────
artifacts/cross*.json  →  /api/ingest  →  D1 表
                                            ↓
                          页面 → /api/cross/ndx|qqq|ssec → Worker 拼 JSON 返回
```

- **写**：`scripts/publish_d1.py`。它把 JSON 读进来、拆成行，用一条条 `INSERT` 通过
  `/api/ingest` 发给 Worker，Worker 用 Bearer Token 校验后写 D1。本地和线上走同一套代码。
  `/api/ingest` 只接受 `INSERT INTO session|market|market_node|contexts|factor_key`，
  别的 SQL 一律 400，token 泄漏也不能删库。
- **读**：Worker 从 D1 查出来，按原来的 JSON 形状拼回去，页面代码只改了 URL。

## 表结构（`deploy/schema.sql`）

| 表 | 干什么 | 行数 |
|---|---|---|
| `market` | 每个标签页一行：标的、问题、规则、首末交易日 | 3 |
| `session` | 每个交易日一行：OHLCV + 信号列 + `factors_packed` | 13,737 |
| `factor_key` | `factors_packed` 的哪个 4 bit 槽位是哪个开关 | 29 |
| `contexts` | 页面上那些开关的定义（标签、阈值、可用起点） | 27 |
| `market_node` | 所有嵌套值，一行一个标量，路径带下标（`bands.0.hit`） | 386 |

`factors_packed` 是把 8–11 个开关压进一个整数，每个开关占 4 bit（`riskoff_hits`
最大是 5，不是 0/1）。11 个开关 44 bit，还在 JS 安全整数范围内——所以一天一行就够，
不用给每个开关建一行（那会是 11 万行，超过免费额度每天 10 万行的写入上限）。

没有任何一列存 JSON 字符串：`market_node` 的 `path` 就是结构本身，
`kind` 说明该按数字、整数、布尔还是字符串还原。

## 现在还没搬的两页

`/vxn`（`artifacts/timeline.json`，3.5 MB，七层嵌套）和 `/agent`
（两份 `agent_eval*_summary.json`）仍然读静态文件。把这两份也搬进 D1 需要各自的表
（timeline 是 rolling / ladder / forward / vxn / ndx / daily / questions 七组序列），
是下一步的工作，不影响信号页。

## 本地验证过什么

`wrangler dev` + 本地 D1（不需要登录、不需要网络）：

- 三个市场写入 D1 后，`/api/cross/<market>` 返回的 JSON 与 Python 产物**逐字段一致**
  （13,737 个交易日 × 全部数组、components、summary、contexts、context、bands、comparison）
- 页面在浏览器里渲染正常，图表、统计表、开关都从数据库出来
- `/api/ingest` 无 token → 401，非 INSERT → 400，重复推送先 `DELETE` 再写

## 关于 /api/refresh

线上这个按钮仍然返回 501。原因没变：它是 31 秒的 numpy 计算，Workers 的 CPU 预算
放不下，容器才行。现在的做法是本地 `make refresh`（算完写 `artifacts/`），
再 `make cf`（或只重跑第 8 步 `publish_d1.py`）把新数据推进 D1。
