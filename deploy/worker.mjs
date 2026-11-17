/**
 * Cloudflare Worker for forecast-log.
 *
 * Three jobs:
 *   1. Serve the pages on clean URLs, with all dashboard data read from D1 (Cloudflare's
 *      SQLite), not from static JSON files.
 *   2. Expose GET/POST /api/refresh (and /api/fetch) to incrementally fetch missing
 *      daily bars from Yahoo Finance after D1's latest date and upsert only the
 *      new/unresolved tail rows into D1.
 *   3. Expose GET /api/state and POST /api/ingest for incremental syncing from publish_d1.py.
 */

const MARKET_BY_SLUG = { ndx: "ndx", qqq: "qqq", ssec: "ssec", "a-shares": "ssec" };

const PAGES = {
  "/": "/web/points.html",
  "/signals": "/web/points.html",
  "/ndx": "/web/points.html",
  "/qqq": "/web/points.html",
  "/a-shares": "/web/points.html",
  "/ssec": "/web/points.html",
  "/vxn": "/web/index.html",
  "/agent": "/web/agent.html",
};

const LEGACY_MARKET = { ndx: "/ndx", qqq: "/qqq", ssec: "/a-shares" };

const LEGACY = {
  "/volatility": "/vxn",
  "/web": "/vxn",
  "/web/index.html": "/vxn",
  "/web/index": "/vxn",
  "/index.html": "/vxn",
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
      return json(doc);
    }

    if (clean === "/api/timeline") {
      const doc = await loadTimeline(env);
      if (!doc) return json({ ok: false, error: "no timeline rows; run scripts/publish_d1.py" }, 404);
      return json(doc);
    }

    const agentApi = clean.match(/^\/api\/agent\/([a-z0-9_-]+)$/);
    if (agentApi) {
      const doc = await loadAgentEval(env, agentApi[1]);
      if (!doc) return json({ ok: false, error: `no agent rows for ${agentApi[1]}; run scripts/publish_d1.py` }, 404);
      return json(doc);
    }

    if (clean === "/api/state") {
      return getD1State(env);
    }

    if (clean === "/api/ingest") {
      if (request.method !== "POST") return json({ ok: false, error: "POST only" }, 405);
      return ingest(request, env);
    }

    if (clean === "/api/refresh" || clean === "/api/fetch") {
      const force = url.searchParams.get("force") === "1";
      return refreshFromYahoo(env, force);
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
  return value;
}

function setPath(root, path, value) {
  const parts = path.includes("\x1f") ? path.split("\x1f") : path.split(".");
  let cursor = root;
  for (let i = 0; i < parts.length - 1; i += 1) {
    const key = /^\d+$/.test(parts[i]) ? Number(parts[i]) : parts[i];
    if (cursor[key] === undefined || cursor[key] === null) {
      cursor[key] = /^\d+$/.test(parts[i + 1]) ? [] : {};
    }
    cursor = cursor[key];
  }
  const last = parts[parts.length - 1];
  cursor[/^\d+$/.test(last) ? Number(last) : last] = value;
}

async function getD1State(env) {
  try {
    const [marketsRes, unresolvedRes, tlRes, agentRes] = await env.DB.batch([
      env.DB.prepare("SELECT market, last_session, fetched_at FROM market"),
      env.DB.prepare(
        "SELECT market, COUNT(*) AS cnt, MAX(date) AS max_date, MIN(CASE WHEN outcome IS NULL THEN date END) AS first_unresolved FROM session GROUP BY market",
      ),
      env.DB.prepare("SELECT COUNT(*) AS cnt FROM timeline_close"),
      env.DB.prepare("SELECT dataset, generated_at, n FROM agent_dataset"),
    ]);

    const markets = {};
    for (const r of marketsRes.results ?? []) {
      markets[r.market] = { last_session: r.last_session, fetched_at: r.fetched_at, count: 0, first_unresolved: null };
    }
    for (const r of unresolvedRes.results ?? []) {
      markets[r.market] = {
        ...(markets[r.market] || {}),
        count: r.cnt,
        max_date: r.max_date,
        first_unresolved: r.first_unresolved,
      };
    }
    const agents = {};
    for (const r of agentRes.results ?? []) {
      agents[r.dataset] = { generated_at: r.generated_at, n: r.n };
    }
    return json({
      ok: true,
      markets,
      timeline_count: tlRes.results?.[0]?.cnt ?? 0,
      agents,
    });
  } catch (err) {
    return json({ ok: true, markets: {}, timeline_count: 0, agents: {}, note: String(err) });
  }
}

function beijingDateStr(date = new Date()) {
  return new Intl.DateTimeFormat("en-CA", {
    timeZone: "Asia/Shanghai",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).format(date);
}

async function fetchYahooBars(symbol, startUnix, endUnix) {
  const url = `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(symbol)}?interval=1d&period1=${startUnix}&period2=${endUnix}`;
  const res = await fetch(url, { headers: { "User-Agent": "Mozilla/5.0 forecast-log/1.0" } });
  if (!res.ok) throw new Error(`Yahoo ${symbol} HTTP ${res.status}`);
  const payload = await res.json();
  const result = payload?.chart?.result?.[0];
  if (!result) return [];
  const tz = result?.meta?.exchangeTimezoneName || "America/New_York";
  const stamps = result.timestamp || [];
  const q = result.indicators?.quote?.[0] || {};
  const fmt = new Intl.DateTimeFormat("en-CA", {
    timeZone: tz,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  });
  const byDate = new Map();
  for (let i = 0; i < stamps.length; i += 1) {
    const close = q.close?.[i];
    if (close === null || close === undefined) continue;
    const d = fmt.format(new Date(stamps[i] * 1000));
    byDate.set(d, {
      date: d,
      open: q.open?.[i] != null ? Math.round(q.open[i] * 100) / 100 : null,
      high: q.high?.[i] != null ? Math.round(q.high[i] * 100) / 100 : null,
      low: q.low?.[i] != null ? Math.round(q.low[i] * 100) / 100 : null,
      close: Math.round(close * 100) / 100,
      volume: q.volume?.[i] != null ? Math.trunc(q.volume[i]) : null,
    });
  }
  return Array.from(byDate.values()).sort((a, b) => a.date.localeCompare(b.date));
}

async function refreshFromYahoo(env, force = false) {
  try {
    const todayBj = beijingDateStr();
    const metaRows = (await env.DB.prepare("SELECT market, instrument, last_session, fetched_at FROM market").all()).results ?? [];
    if (metaRows.length === 0) {
      return json({ ok: false, error: "D1 尚未初始化基础数据，请先运行 make cf" }, 404);
    }

    if (!force) {
      const allFresh = metaRows.every((m) => m.fetched_at && m.fetched_at.slice(0, 10) >= todayBj);
      if (allFresh) {
        const sessions = Object.fromEntries(metaRows.map((m) => [m.market, m.last_session]));
        return json({ ok: true, cached: true, beijing_date: todayBj, sessions });
      }
    }

    const nowSec = Math.floor(Date.now() / 1000) + 172800;
    const updated = {};
    const errors = {};
    const insertedCounts = {};

    for (const meta of metaRows) {
      const mkt = meta.market;
      const sym = meta.instrument;
      try {
        const tailRows = (
          await env.DB.prepare("SELECT * FROM session WHERE market = ? ORDER BY date DESC LIMIT 25").bind(mkt).all()
        ).results ?? [];
        if (tailRows.length === 0) continue;
        tailRows.reverse();
        const oldestTailDate = tailRows[0].date;
        const startUnix = Math.floor(new Date(`${oldestTailDate}T00:00:00Z`).getTime() / 1000) - 86400 * 5;
        const bars = await fetchYahooBars(sym, startUnix, nowSec);
        if (bars.length === 0) continue;

        const byDate = new Map(tailRows.map((r) => [r.date, { ...r }]));
        for (const b of bars) {
          if (b.date < oldestTailDate) continue;
          const prev = byDate.get(b.date);
          if (prev) {
            prev.open = b.open;
            prev.high = b.high;
            prev.low = b.low;
            prev.close = b.close;
            prev.volume = b.volume;
          } else {
            const lastRow = tailRows[tailRows.length - 1];
            byDate.set(b.date, {
              market: mkt,
              date: b.date,
              open: b.open,
              high: b.high,
              low: b.low,
              close: b.close,
              volume: b.volume,
              entry: null,
              entry_at: null,
              base_close: null,
              close3: null,
              close5: null,
              hit3: null,
              flag: 0,
              flag_price_only: 0,
              news_veto: 0,
              score: lastRow?.score ?? null,
              threshold: lastRow?.threshold ?? null,
              outcome: null,
              target: null,
              peak: null,
              peak_at: null,
              us_date: null,
              factors_packed: 0,
            });
          }
        }

        const merged = Array.from(byDate.values()).sort((a, b) => a.date.localeCompare(b.date));
        const H = 5;
        for (let i = 0; i < merged.length; i += 1) {
          const row = merged[i];
          if (mkt === "ssec") {
            if (i + H < merged.length && row.open != null) {
              const win = merged.slice(i + 1, i + 1 + H);
              if (win.every((w) => w.close != null)) {
                let bestIdx = 0;
                for (let k = 1; k < win.length; k += 1) {
                  if (win[k].close > win[bestIdx].close) bestIdx = k;
                }
                row.peak = win[bestIdx].close;
                row.peak_at = win[bestIdx].date;
                row.outcome = merged[i + H].close > row.close ? 1 : 0;
                if (row.flag) {
                  row.base_close = row.close;
                  row.close3 = merged[i + 3].close;
                  row.close5 = merged[i + H].close;
                  row.hit3 = row.close3 > row.base_close ? 1 : 0;
                }
              }
            }
          } else if (i + H + 1 < merged.length) {
            const nextBar = merged[i + 1];
            const win = merged.slice(i + 2, i + 2 + H);
            if (nextBar.open != null && win.every((w) => w.close != null)) {
              let bestIdx = 0;
              for (let k = 1; k < win.length; k += 1) {
                if (win[k].close > win[bestIdx].close) bestIdx = k;
              }
              row.entry = nextBar.open;
              row.entry_at = nextBar.date;
              row.base_close = row.close;
              row.close3 = merged[i + 3].close;
              row.close5 = merged[i + H].close;
              row.hit3 = row.close3 > row.base_close ? 1 : 0;
              row.outcome = row.close5 > row.base_close ? 1 : 0;
              row.target = merged[i + H].date;
              row.peak = win[bestIdx].close;
              row.peak_at = win[bestIdx].date;
            }
          }
        }

        const toWrite = merged.slice(-12);
        const stmts = toWrite.map((r) =>
          env.DB.prepare(
            `INSERT OR REPLACE INTO session (
              market, date, open, high, low, close, volume, entry, entry_at,
              base_close, close3, close5, hit3, flag, flag_price_only, news_veto,
              score, threshold, outcome, target, peak, peak_at, us_date, factors_packed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
          ).bind(
            mkt,
            r.date,
            r.open,
            r.high,
            r.low,
            r.close,
            r.volume,
            r.entry,
            r.entry_at,
            r.base_close,
            r.close3,
            r.close5,
            r.hit3,
            r.flag,
            r.flag_price_only,
            r.news_veto,
            r.score,
            r.threshold,
            r.outcome,
            r.target,
            r.peak,
            r.peak_at,
            r.us_date,
            r.factors_packed ?? 0,
          ),
        );
        const lastBar = merged[merged.length - 1];
        const fetchedIso = `${todayBj}T${new Date().toISOString().slice(11, 19)}+08:00`;
        stmts.push(
          env.DB.prepare("UPDATE market SET last_session = ?, fetched_at = ? WHERE market = ?").bind(
            lastBar.date,
            fetchedIso,
            mkt,
          ),
        );
        await env.DB.batch(stmts);
        updated[sym] = lastBar.date;
        insertedCounts[mkt] = toWrite.length;
      } catch (e) {
        errors[sym] = String(e && e.message ? e.message : e);
      }
    }

    const afterMeta = (await env.DB.prepare("SELECT market, last_session FROM market").all()).results ?? [];
    const sessions = Object.fromEntries(afterMeta.map((m) => [m.market, m.last_session]));
    return json({
      ok: true,
      refreshed_at: new Date().toISOString(),
      beijing_date: todayBj,
      sessions,
      updated,
      upserted_rows: insertedCounts,
      errors,
    });
  } catch (err) {
    return json({ ok: false, error: String(err && err.message ? err.message : err) }, 500);
  }
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

async function loadTimeline(env) {
  const [closeRes, rollingRes, dailyRes, ladderRes, nodeRes] = await env.DB.batch([
    env.DB.prepare("SELECT symbol, date, close FROM timeline_close ORDER BY symbol, idx"),
    env.DB.prepare("SELECT * FROM timeline_rolling ORDER BY symbol, idx"),
    env.DB.prepare("SELECT * FROM timeline_daily ORDER BY symbol, idx"),
    env.DB.prepare("SELECT * FROM timeline_ladder ORDER BY horizon, idx"),
    env.DB.prepare("SELECT path, kind, value FROM timeline_node"),
  ]);

  const closes = closeRes.results ?? [];
  if (closes.length === 0) return null;

  const doc = {
    rolling: { vxn: {}, ndx: {} },
    ladder: { vxn: { columns: {} } },
    forward: {},
    vxn: { time: [], close: [] },
    ndx: { time: [], close: [] },
    daily: { vxn: {}, ndx: {} },
    questions: {},
  };

  for (const row of closes) {
    doc[row.symbol].time.push(row.date);
    doc[row.symbol].close.push(row.close);
  }

  const rollingRows = rollingRes.results ?? [];
  const dailyRows = dailyRes.results ?? [];
  for (const sym of ["vxn", "ndx"]) {
    const r = rollingRows.filter((row) => row.symbol === sym);
    doc.rolling[sym] = {
      time: r.map((row) => row.time),
      as_of: r.map((row) => row.as_of),
      anchor: r.map((row) => row.anchor),
      median: r.map((row) => row.median),
      low: r.map((row) => row.low),
      high: r.map((row) => row.high),
      actual: r.map((row) => row.actual),
    };
    const d = dailyRows.filter((row) => row.symbol === sym);
    doc.daily[sym] = {
      time: d.map((row) => row.time),
      median: d.map((row) => row.median),
      low: d.map((row) => row.low),
      high: d.map((row) => row.high),
      actual: d.map((row) => row.actual),
      previous: d.map((row) => row.previous),
      risk: d.map((row) => row.risk),
    };
  }

  const cols = doc.ladder.vxn.columns;
  for (const row of ladderRes.results ?? []) {
    const h = String(row.horizon);
    if (!cols[h]) cols[h] = { median: [], low: [], high: [] };
    cols[h].median.push(row.median);
    cols[h].low.push(row.low);
    cols[h].high.push(row.high);
  }

  for (const { path, kind, value } of nodeRes.results ?? []) {
    setPath(doc, path, cast(kind, value));
  }

  return doc;
}

async function loadAgentEval(env, dataset) {
  const [metaRes, nodeRes] = await env.DB.batch([
    env.DB.prepare("SELECT * FROM agent_dataset WHERE dataset = ?").bind(dataset),
    env.DB.prepare("SELECT path, kind, value FROM agent_node WHERE dataset = ?").bind(dataset),
  ]);
  const meta = metaRes.results?.[0];
  if (!meta) return null;
  const doc = {
    n: meta.n,
    scored_on_shared: meta.scored_on_shared,
    generated_at: meta.generated_at,
  };
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

  try {
    return await applyOp(env, body);
  } catch (error) {
    return json({ ok: false, error: String(error && error.message ? error.message : error) }, 400);
  }
}

async function applyOp(env, body) {
  if (body.op === "clear_meta") {
    const market = body.market;
    if (!market) return json({ ok: false, error: "market is required" }, 400);
    await env.DB.batch([
      env.DB.prepare("DELETE FROM market_node WHERE market = ?").bind(market),
      env.DB.prepare("DELETE FROM contexts WHERE market = ?").bind(market),
      env.DB.prepare("DELETE FROM factor_key WHERE market = ?").bind(market),
    ]);
    return json({ ok: true, op: "clear_meta", market });
  }

  if (body.op === "clear") {
    const market = body.market;
    if (!market) return json({ ok: false, error: "market is required" }, 400);
    await env.DB.batch([
      env.DB.prepare("DELETE FROM session WHERE market = ?").bind(market),
      env.DB.prepare("DELETE FROM market_node WHERE market = ?").bind(market),
      env.DB.prepare("DELETE FROM contexts WHERE market = ?").bind(market),
      env.DB.prepare("DELETE FROM factor_key WHERE market = ?").bind(market),
      env.DB.prepare("DELETE FROM market WHERE market = ?").bind(market),
    ]);
    return json({ ok: true, op: "clear", market });
  }

  if (body.op === "clear_timeline") {
    await env.DB.batch([
      env.DB.prepare("DELETE FROM timeline_close"),
      env.DB.prepare("DELETE FROM timeline_rolling"),
      env.DB.prepare("DELETE FROM timeline_daily"),
      env.DB.prepare("DELETE FROM timeline_ladder"),
      env.DB.prepare("DELETE FROM timeline_node"),
    ]);
    return json({ ok: true, op: "clear_timeline" });
  }

  if (body.op === "clear_agent") {
    const dataset = body.dataset;
    if (!dataset) return json({ ok: false, error: "dataset is required" }, 400);
    await env.DB.batch([
      env.DB.prepare("DELETE FROM agent_node WHERE dataset = ?").bind(dataset),
      env.DB.prepare("DELETE FROM agent_dataset WHERE dataset = ?").bind(dataset),
    ]);
    return json({ ok: true, op: "clear_agent", dataset });
  }

  if (body.op === "sql") {
    if (typeof body.sql !== "string") return json({ ok: false, error: "sql must be a string" }, 400);
    if (
      !/^\s*INSERT\s+(OR\s+REPLACE\s+)?INTO\s+(session|market|market_node|contexts|factor_key|timeline_close|timeline_rolling|timeline_daily|timeline_ladder|timeline_node|agent_dataset|agent_node)\b/i.test(
        body.sql,
      )
    ) {
      return json({ ok: false, error: "INSERT only on dashboard tables" }, 400);
    }
    const result = await env.DB.exec(body.sql);
    return json({ ok: true, op: "sql", count: result.count ?? null });
  }

  return json({ ok: false, error: "unknown op" }, 400);
}
