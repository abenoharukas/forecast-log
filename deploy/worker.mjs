/**
 * Cloudflare Worker 前端：路由规则和 forecast_log/server.py 保持一致。
 *
 * 静态资源走 Workers Static Assets（`env.ASSETS`），这里只做三件事：
 *   1. 干净路径 → 磁盘上的真实文件（服务端重写，地址栏不变）
 *   2. 旧 .html 地址 → 301 到新地址
 *   3. /api/refresh 在纯静态部署里没有后端，返回 501 并说明原因
 */

const PAGES = {
  "/": "/web/points.html",
  "/signals": "/web/points.html",
  "/ndx": "/web/points.html",
  "/qqq": "/web/points.html",
  "/a-shares": "/web/points.html",
  "/ssec": "/web/points.html",
  "/volatility": "/web/index.html",
  "/vxn": "/web/index.html",
  "/agent": "/web/agent.html",
};

const LEGACY_MARKET = { ndx: "/ndx", qqq: "/qqq", ssec: "/a-shares" };

const LEGACY = {
  "/web": "/volatility",
  "/web/index.html": "/volatility",
  "/web/index": "/volatility",
  "/index.html": "/volatility",
  "/web/points": "/ndx",
  "/points": "/ndx",
  "/web/qqq.html": "/qqq",
  "/web/qqq": "/qqq",
  "/qqq.html": "/qqq",
  "/web/ssec.html": "/a-shares",
  "/web/ssec": "/a-shares",
  "/ssec.html": "/a-shares",
  "/web/agent.html": "/agent",
  "/web/agent": "/agent",
  "/agent.html": "/agent",
};

const ASSET_FILES = new Set(["/theme.css", "/theme.js", "/ctxbar.js"]);
const ASSET_DIRS = ["/fonts/", "/vendor/"];

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const clean = url.pathname.replace(/\/+$/, "") || "/";

    if (clean === "/api/refresh") {
      return Response.json(
        {
          ok: false,
          static_deploy: true,
          error: "/api/refresh needs the Python pipeline (numpy + Yahoo bars), which is not in this Worker.",
          hint: "Refresh locally with `make refresh`, push artifacts, redeploy. See deploy/README.md.",
        },
        { status: 501 },
      );
    }

    if (clean === "/web/points.html" || clean === "/points.html") {
      const market = url.searchParams.get("market") || "";
      return redirect(LEGACY_MARKET[market] || "/ndx");
    }

    if (LEGACY[clean]) return redirect(LEGACY[clean]);

    const assetPath = ASSET_FILES.has(clean) || ASSET_DIRS.some((d) => clean.startsWith(d))
      ? `/web${clean}`
      : PAGES[clean];

    if (assetPath) {
      const assetUrl = new URL(assetPath, url);
      const res = await env.ASSETS.fetch(new Request(assetUrl, request));
      const headers = new Headers(res.headers);
      headers.set("Cache-Control", "no-store, no-cache, must-revalidate");
      return new Response(res.body, { status: res.status, headers });
    }

    return env.ASSETS.fetch(request);
  },
};

function redirect(location) {
  return new Response(null, {
    status: 301,
    headers: { Location: location, "Cache-Control": "no-store" },
  });
}
