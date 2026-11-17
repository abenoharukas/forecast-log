# Cloudflare 部署

这个项目是「静态页面 + 一个 Python 接口」。前端可以原样搬到 Cloudflare，
`/api/refresh` 那条刷新链路不行——它要 numpy 和 Yahoo 行情。

结论先写在这里：

| 部分 | 能上 Cloudflare 吗 | 用什么 |
|---|---|---|
| 5 个页面（`/`、`/qqq`、`/a-shares`、`/volatility`、`/agent`） | 能，和本地一模一样 | Workers + Workers Static Assets（静态资源） |
| 干净地址、旧地址 301 | 能 | `deploy/worker.mjs` 里的路由（和 `server.py` 的规则一一对应） |
| `artifacts/*.json` | 能 | 同上，当静态文件发 |
| `/api/refresh`（拉 Yahoo + numpy 重算） | 不能 | 要么本地跑完再部署，要么上 Containers（见下） |
| `data/forecast.db`（81 MB，浏览器从不请求） | 不需要上 | 留在本地 / 换 R2 或 D1 |

## 包有多大

`python3 scripts/build_static.py` 生成 `deploy/dist`：20 个文件、6.22 MB，最大单文件
`artifacts/timeline.json` 3.53 MB。Workers Static Assets 的文档限制是单文件 25 MiB、
最多 20,000 个文件，这个包远在限内。

## 现在已经能跑的部分

`wrangler.jsonc` + `deploy/worker.mjs` 已经写好，`dist` 是构建产物。

```bash
python3 scripts/build_static.py     # 或 make static
wrangler dev --port 8787            # 本地用 workerd 跑一遍，不需要登录
```

本地实测：`/`、`/ndx`、`/qqq`、`/a-shares`、`/volatility`、`/agent` 全 200，
`/theme.css`、`/ctxbar.js`、`/fonts/*`、`/vendor/*`、`/artifacts/*.json` 全 200，
旧地址 `/web/points.html?market=qqq` 等 301 到新地址，页面在浏览器里渲染正常。

部署（需要先 `wrangler login`）：

```bash
python3 scripts/build_static.py
wrangler deploy                     # 或 make deploy
```

刷新按钮在这个部署里会返回 501 并说明原因——不是坏了，是刻意写死的，
免得用户以为数据更新了。

## 如果刷新也要在云上

三条路，按代价从小到大：

1. **本地刷新 + 重新部署（现在就是这条）**
   `make refresh` 重算 `artifacts/cross*.json`，然后 `wrangler deploy`。免费。
   代价是数据的新鲜度取决于你什么时候想起来跑。

2. **加一个 GitHub Action 定时任务**
   每天美东收盘后跑 `forecast-log cross`，把 `artifacts/` 提交回仓库，再用
   API Token 触发 `wrangler deploy`。仍然是静态站，免费，但每天会有一次提交。
   注意 Yahoo 在 CI 的网络环境下能否取到，需要实测。

3. **Containers（真跑 Python 服务）**
   把 `serve.py` 打进镜像，Worker 转发请求，`/api/refresh` 就真的能用。
   - 需要 Workers Paid 计划（Containers 按用量计费，另外可能还有实例空闲成本）。
   - 容器实例的文件系统**不持久**，重启就没了：重算出来的 `artifacts/*.json`
     要写到 R2（或其他存储），页面再从 R2/Worker 读，不能指望写磁盘。
   - 冷启动是秒级，刷新这种一天一次的操作可以接受。
   - 镜像要带 python3 + numpy，体积大概 200 MB 起。

如果目的只是「让别人能看这几张图」，方案 1 就够；如果目的是「网页上点一下就
能看到最新信号」，才需要方案 3。

## 需要你自己确认的两件事

- `wrangler login` 能不能过——我这边连不上 Cloudflare 的 auth 服务
  （网络/代理不通），所以没实际部署，只做了本地 `wrangler dev` 验证。
- 上面的限额和计费请以部署时 `wrangler` 的报错和 dashboard 为准；我这份写的时候
  取不到 developers.cloudflare.com 的页面。
