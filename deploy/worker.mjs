/**
 * Cloudflare Worker for forecast-log.
 *
 * Two jobs:
 *   1. Serve the pages on clean URLs, with the numbers read from D1 (Cloudflare's
 *      SQLite), not from static JSON files.
 *   2. Accept writes from scripts/publish_d1.py at /api/ingest (bearer token).
 *
 * Data layout is described in deploy/schema.sql. Everything here is plain SQL +
 * a path-assembly step, so no artefact is ever stored or served as a JSON blob.
 */

const MARKET_BY_SLUG = { ndx: "ndx", qqq: "qqq", ssec: "ssec", "a-shares": "ssec" };

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

// 每个交易日一行的列；有值的才变成数组，和 Python 产物保持同形
const ARRAY_COLUMNS = [
  "open",
  "high",
  "low",
  "close",
  "volume",
  "entry",
  "entry_at",
  "base_close",
  "close3",
  "close5",
  "hit3",
  "score",
  "threshold",
  "flag",
  "flag_price_only",
  "news_veto",
  "outcome",
  "target",
  "peak",
  "peak_at",
  "us_date",
];

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const clean = url.pathname.replace(/\/+$/, "") || "/";

    const api = clean.match(/^\/api\/cross\/([\w-]+)$/);
    if (api) {
      const market = MARKET_BY_SLUG[api[1]];
      if (!market) return json({ ok: false, error: "unknown market" }, 404);
      const doc = await loadMarket(env, market);
      if (!doc) return json({ ok: false, error: `no rows for ${market}; run scripts/publish_d1.py` }, 404);
      return new Response(JSON.stringify(doc), {
        headers: { "Content-Type": "application/json; charset=utf-8", "Cache-Control": "no-store" },
      });
    }

    if (clean === "/api/ingest") {
      if (request.method !== "POST") return json({ ok: false, error: "POST only" }, 405);
      return ingest(request, env);
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
      const res = await env.ASSETS.fetch(new Request(new URL(assetPath, url), request));
      const headers = new Headers(res.headers);
      headers.set("Cache-Control", "no-store, no-cache, must-revalidate");
      return new Response(res.body, { status: res.status, headers });
    }

    return env.ASSETS.fetch(request);
  },
};

function redirect(location) {
  return new Response(null, { status: 301, headers: { Location: location, "Cache-Control": "no-store" } });
}

function json(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json; charset=utf-8", "Cache-Control": "no-store" },
  });
}

function cast(kind, value) {
  if (kind === "list") return [];
  if (kind === "obj") return {};
  if (value === null || value === undefined) return null;
  if (kind === "num") return Number.parseFloat(value);
  if (kind === "int") return Number.parseInt(value, 10);
  if (kind === "bool") return value === "1" || value === "true";
  if (kind === "list") return [];
  if (kind === "obj") return {};
  return value;
}

/** "bands.0.hit" -> doc.bands[0].hit, creating arrays or objects as needed. */
function setPath(root, path, value) {
  const parts = path.split(".");
  let cursor = root;
  for (let i = 0; i < parts.length - 1; i += 1) {
    const key = /^\d+$/.test(parts[i]) ? Number(parts[i]) : parts[i];
    if (cursor[key] === undefined) {
      cursor[key] = /^\d+$/.test(parts[i + 1]) ? [] : {};
    }
    cursor = cursor[key];
  }
  const last = parts[parts.length - 1];
  cursor[/^\d+$/.test(last) ? Number(last) : last] = value;
}

async function loadMarket(env, market) {
  const [metaRes, sessionRes, nodeRes, contextRes, factorRes] = await env.DB.batch([
    env.DB.prepare("SELECT * FROM market WHERE market = ?").bind(market),
    env.DB.prepare("SELECT * FROM session WHERE market = ? ORDER BY date").bind(market),
    env.DB.prepare("SELECT path, kind, value FROM market_node WHERE market = ?").bind(market),
    env.DB.prepare("SELECT * FROM contexts WHERE market = ? ORDER BY position").bind(market),
    env.DB.prepare("SELECT position, key FROM factor_key WHERE market = ? ORDER BY position").bind(market),
  ]);

  const meta = metaRes.results?.[0];
  const sessions = sessionRes.results ?? [];
  if (!meta || sessions.length === 0) return null;

  const doc = {};
  if (meta.name) doc.name = meta.name;
  doc.instrument = meta.instrument;
  doc.question = { horizon: meta.horizon, text: meta.question, rule: meta.rule };
  doc.fetched_at = meta.fetched_at;
  doc.first_session = meta.first_session;
  doc.last_session = meta.last_session;
  doc.time = sessions.map((row) => row.date);

  for (const column of ARRAY_COLUMNS) {
    if (!(column in sessions[0])) continue;
    const values = sessions.map((row) => row[column]);
    if (values.some((value) => value !== null && value !== undefined)) doc[column] = values;
  }

  doc.components = {};
  for (const { position, key } of factorRes.results ?? []) {
    // 不能用 >> ：JS 位运算是 32 位的，而 11 个开关要 44 位，所以走除法
    doc.components[key] = sessions.map((row) => Math.floor(row.factors_packed / 2 ** (4 * position)) % 16);
  }

  doc.contexts = (contextRes.results ?? []).map((row) => {
    const out = { key: row.key };
    if (row.label) out.label = row.label;
    if (row.kind) out.kind = row.kind;
    if (row.required) out.required = true;
    if (row.threshold !== null) out.threshold = row.threshold;
    if (row.group_name) {
      out.group = row.group_name;
      out.group_label = row.group_label;
    }
    if (row.note) out.note = row.note;
    if (row.available_from) out.available_from = row.available_from;
    return out;
  });

  for (const { path, kind, value } of nodeRes.results ?? []) {
    setPath(doc, path, cast(kind, value));
  }

  return doc;
}

async function ingest(request, env) {
  const token = (request.headers.get("Authorization") || "").replace(/^Bearer\s+/i, "");
  if (!env.INGEST_TOKEN) return json({ ok: false, error: "INGEST_TOKEN is not set on the Worker" }, 500);
  if (token !== env.INGEST_TOKEN) return json({ ok: false, error: "bad token" }, 401);

  let body;
  try {
    body = await request.json();
  } catch {
    return json({ ok: false, error: "body must be JSON" }, 400);
  }
  const market = body.market;
  if (!market) return json({ ok: false, error: "market is required" }, 400);

  try {
    return await applyOp(env, market, body);
  } catch (error) {
    return json({ ok: false, error: String(error && error.message ? error.message : error) }, 400);
  }
}

async function applyOp(env, market, body) {
  if (body.op === "clear") {
    await env.DB.batch([
      env.DB.prepare("DELETE FROM session WHERE market = ?").bind(market),
      env.DB.prepare("DELETE FROM market_node WHERE market = ?").bind(market),
      env.DB.prepare("DELETE FROM contexts WHERE market = ?").bind(market),
      env.DB.prepare("DELETE FROM factor_key WHERE market = ?").bind(market),
      env.DB.prepare("DELETE FROM market WHERE market = ?").bind(market),
    ]);
    return json({ ok: true, op: "clear", market });
  }

  if (body.op === "sql") {
    if (typeof body.sql !== "string") return json({ ok: false, error: "sql must be a string" }, 400);
    // 只允许写这几张表，避免 token 泄漏后能删库
    if (!/^\s*INSERT\s+(OR\s+REPLACE\s+)?INTO\s+(session|market|market_node|contexts|factor_key)\b/i.test(body.sql)) {
      return json({ ok: false, error: "only INSERT INTO session|market|market_node|contexts|factor_key is allowed" }, 400);
    }
    const result = await env.DB.exec(body.sql);
    return json({ ok: true, op: "sql", count: result.count ?? null });
  }

  return json({ ok: false, error: "unknown op" }, 400);
}
