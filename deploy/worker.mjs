/**
 * Cloudflare Worker for forecast-log.
 *
 * Data model: the same five tables as the local `data/forecast.db`
 * (`bars`, `signals`, `news`, `agent_eval`, `chat_messages`).
 * Supports Supabase PostgreSQL (via Cloudflare Hyperdrive or `DATABASE_URL`)
 * with automatic fallback to Cloudflare D1 (`env.DB`).
 *
 * What is read from the DB vs. served as an asset:
 *   - /api/cross/<market>  DB: signals + bars joined on (instrument, date). The per-day
 *                          series, hit-rate summaries and A-share bands are computed here.
 *                          `_meta/cross-<market>.json` only supplies the static bits
 *                          (question text, context switch definitions, provenance).
 *   - /api/agent/<dataset> DB: the flat `agent_eval` table. Brier / log-loss / ECE are
 *                          recomputed here, exactly like the local Python server.
 *   - /api/timeline        Asset: the VXN analogue-model fan charts are ~900 lines of
 *                          numpy and are built once at deploy time by
 *                          `scripts/build_d1_payloads.py`.
 *   - /api/state, /api/chat/*, /api/ingest, /api/refresh use the five tables directly.
 */

import { createDb } from "./db.mjs";
import { recomputeAllSignals } from "./signals.mjs";

const MARKET_BY_SLUG = { ndx: "ndx", qqq: "qqq", ssec: "ssec", "a-shares": "ssec" };
const INSTRUMENT = { ndx: "^NDX", qqq: "QQQ", ssec: "000001.SS" };
const HORIZON = 5;
const HOLDOUT = "2015-01-01";

const PAGES = {
  "/": "/web/points.html",
  "/signals": "/web/points.html",
  "/ndx": "/web/points.html",
  "/qqq": "/web/points.html",
  "/a-shares": "/web/points.html",
  "/ssec": "/web/points.html",
  "/vxn": "/web/index.html",
  "/agent": "/web/agent.html",
  "/sim": "/web/sim.html",
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
  "/web/sim.html": "/sim",
  "/web/sim": "/sim",
  "/sim.html": "/sim",
};

const ASSET_FILES = new Set(["/theme.css", "/theme.js", "/ctxbar.js"]);
const ASSET_DIRS = ["/fonts/", "/vendor/"];

const PRICE_COLUMNS = new Set(["open", "high", "low", "close"]);
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
  "flag",
  "flag_price_only",
  "sell_flag",
  "sell_threshold",
  "news_veto",
  "score",
  "threshold",
  "outcome",
  "target",
  "peak",
  "peak_at",
  "us_date",
  "is_closed",
];

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    const clean = url.pathname.replace(/\/+$/, "") || "/";

    if (clean === "/api/timeline") {
      const res = await env.ASSETS.fetch(new Request(new URL("/_meta/timeline.json", url)));
      if (!res.ok) return json({ ok: false, error: "timeline payload missing; run make cf-supabase-deploy or make cf-worker-deploy" }, 404);
      return new Response(res.body, {
        status: 200,
        headers: { "Content-Type": "application/json; charset=utf-8", "Cache-Control": "no-store" },
      });
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

    if (!clean.startsWith("/api/")) {
      return env.ASSETS.fetch(request);
    }

    const db = createDb(env);
    try {
      const api = clean.match(/^\/api\/cross\/([\w-]+)$/);
      if (api) {
        const market = MARKET_BY_SLUG[api[1]];
        if (!market) return json({ ok: false, error: "unknown market" }, 404);
        const doc = await loadMarket(db, env, url, market);
        if (!doc) return json({ ok: false, error: `no rows for ${market}; run sync first` }, 404);
        return json(doc);
      }

      const agentApi = clean.match(/^\/api\/agent\/([a-z0-9_-]+)$/);
      if (agentApi) {
        const doc = await loadAgentEval(db, agentApi[1]);
        if (!doc) return json({ ok: false, error: `no agent rows for ${agentApi[1]}; run sync first` }, 404);
        return json(doc);
      }

      if (clean === "/api/state") return await getDbState(db);

      if (clean === "/api/chat/threads") return await chatThreads(db, url);

      if (clean === "/api/chat/messages") return await chatMessages(db, url);

      if (clean === "/api/ingest") {
        if (request.method !== "POST") return json({ ok: false, error: "POST only" }, 405);
        return await ingest(request, db, env);
      }

      if (clean === "/api/refresh" || clean === "/api/fetch") {
        const force = url.searchParams.get("force") === "1";
        return await refreshFromYahoo(db, env, url, force);
      }

      if (clean === "/api/news") {
        const market = url.searchParams.get("market") || "ndx";
        const limit = Math.max(10, Math.min(Number(url.searchParams.get("limit") || 50) || 50, 120));
        return await loadNewsBundle(db, env, url, { market, limit, cached: true });
      }

      if (clean === "/api/news/refresh") {
        const force = url.searchParams.get("force") === "1";
        const market = url.searchParams.get("market") || "ndx";
        const limit = Math.max(10, Math.min(Number(url.searchParams.get("limit") || 50) || 50, 120));
        return await refreshNews(db, env, url, { force, market, limit });
      }

      return env.ASSETS.fetch(request);
    } finally {
      const closing = db.close();
      if (ctx && typeof ctx.waitUntil === "function") ctx.waitUntil(closing);
      else await closing;
    }
  },
};

// ------------------------------------------------------------------ utilities

function redirect(location) {
  return new Response(null, { status: 301, headers: { Location: location, "Cache-Control": "no-store" } });
}

function json(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json; charset=utf-8", "Cache-Control": "no-store" },
  });
}

/**
 * Python's round(value, digits): correctly-rounded decimal, ties to even.
 * `Math.round(x * 10**d)` is not the same thing — it rounds the scaled binary
 * product, which drifts by one cent on values such as 2941.115.
 */
function rnd(value, digits) {
  if (value === null || value === undefined) return null;
  const x = Number(value);
  if (!Number.isFinite(x)) return null;
  const negative = x < 0;
  const abs = Math.abs(x);
  const text = abs.toFixed(digits + 25);
  const dot = text.indexOf(".");
  const integerPart = text.slice(0, dot);
  const fraction = text.slice(dot + 1);
  const head = fraction.slice(0, digits);
  const rest = fraction.slice(digits);
  let scaled = BigInt(integerPart + head);
  const first = rest[0] || "0";
  const tail = rest.slice(1);
  let roundUp = false;
  if (first > "5") roundUp = true;
  else if (first === "5") {
    if (tail.replace(/0+$/, "") !== "") roundUp = true;
    else roundUp = scaled % 2n === 1n;
  }
  if (roundUp) scaled += 1n;
  const out = Number(scaled) / 10 ** digits;
  return negative ? -out : out;
}

/** math.fsum: exactly-rounded float summation (Python's `statistics.fmean` uses it). */
function fsum(values) {
  const partials = [];
  for (let value of values) {
    let x = Number(value);
    let index = 0;
    for (let j = 0; j < partials.length; j += 1) {
      let y = partials[j];
      if (Math.abs(x) < Math.abs(y)) {
        const swap = x;
        x = y;
        y = swap;
      }
      const hi = x + y;
      const lo = y - (hi - x);
      if (lo !== 0) {
        partials[index] = lo;
        index += 1;
      }
      x = hi;
    }
    partials.length = index;
    partials.push(x);
  }
  let n = partials.length - 1;
  let hi = partials[n] ?? 0;
  let lo = 0;
  while (n > 0) {
    const x = hi;
    n -= 1;
    hi = x + partials[n];
    lo = partials[n] - (hi - x);
    if (lo !== 0) break;
  }
  if (n > 0 && ((lo < 0 && partials[n - 1] < 0) || (lo > 0 && partials[n - 1] > 0))) {
    const y = lo * 2;
    const x = hi + y;
    if (y === x - hi) hi = x;
  }
  return hi;
}

function mean(values) {
  return values.length ? fsum(values) / values.length : null;
}

function sum(values) {
  return values.reduce((acc, value) => acc + value, 0);
}

function daysAgoStr(days) {
  return new Date(Date.now() - days * 86400 * 1000).toISOString().slice(0, 10);
}

function timingSafeEqualStr(a, b) {
  const enc = new TextEncoder();
  const ba = enc.encode(String(a || ""));
  const bb = enc.encode(String(b || ""));
  if (ba.length !== bb.length) return false;
  let diff = 0;
  for (let i = 0; i < ba.length; i += 1) diff |= ba[i] ^ bb[i];
  return diff === 0;
}

async function assetJson(env, url, path) {
  const res = await env.ASSETS.fetch(new Request(new URL(path, url)));
  if (!res.ok) return null;
  return res.json();
}

// ------------------------------------------------------------------ /api/cross

async function loadMarket(db, env, url, market) {
  const instrument = INSTRUMENT[market];
  const rows = await db.all(
    `SELECT s.*, b.open, b.high, b.low, b.close, b.volume,
            COALESCE(b.is_closed, 1) AS is_closed,
            COALESCE(NULLIF(s.fetched_at, ''), b.fetched_at, '') AS eff_fetched_at
     FROM signals s
     LEFT JOIN bars b ON b.symbol = ? AND b.date = s.date
     WHERE s.market = ?
     ORDER BY s.date ASC`,
    [instrument, market],
  );
  if (!rows.length) return null;

  const meta = (await assetJson(env, url, `/_meta/cross-${market}.json`)) || {};
  const ndxDoc = market === "ndx" ? null : await loadMarketRows(db, "ndx");

  const doc = {};
  if (meta.name) doc.name = meta.name;
  doc.instrument = meta.instrument || instrument;
  doc.question = meta.question;
  doc.fetched_at = String(rows[rows.length - 1].eff_fetched_at || "");
  doc.first_session = rows[0].date;
  doc.last_session = rows[rows.length - 1].date;
  doc.time = rows.map((row) => row.date);

  for (const column of ARRAY_COLUMNS) {
    if (!(column in rows[0])) continue;
    const values = rows.map((row) =>
      PRICE_COLUMNS.has(column) && row[column] !== null ? rnd(row[column], 2) : row[column],
    );
    if (values.some((value) => value !== null && value !== undefined)) doc[column] = values;
  }

  const contexts = meta.contexts || [];
  doc.components = {};
  for (let position = 0; position < contexts.length; position += 1) {
    doc.components[contexts[position].key] = rows.map((row) =>
      Math.floor(Number(row.factors_packed || 0) / 2 ** (4 * position)) % 16,
    );
  }
  if (market === "ssec") {
    doc.components.sell_composite = rows.map((row) => Number(row.sell_flag || 0));
  } else {
    doc.components.sell_composite = rows.map((row) =>
      row.score !== null && row.score !== undefined && row.sell_threshold !== null && row.sell_threshold !== undefined && Number(row.score) <= Number(row.sell_threshold) ? 1 : 0,
    );
  }
  doc.contexts = contexts;

  const last = rows[rows.length - 1];
  if (market === "ssec") {
    const ssec = ssecSummary(rows, ndxDoc);
    doc.summary = ssec.summary;
    doc.summary_sell = ssecSellSummary(rows);
    doc.bands = ssec.bands;
    if (ssec.reference) doc.reference = ssec.reference;
    doc.signals_seen = ssec.signalsSeen;
    doc.latest = {
      date: last.date,
      open: rnd(last.open, 2),
      high: rnd(last.high, 2),
      low: rnd(last.low, 2),
      close: rnd(last.close, 2),
      volume: last.volume,
      us_date: last.us_date || previousSession(ndxDoc?.time, last.date),
      flag: Number(last.flag || 0),
      sell_flag: Number(last.sell_flag || 0),
      is_closed: Boolean(last.is_closed),
    };
  } else {
    doc.summary = usTally(rows, "flag");
    doc.summary_price_only = usTally(rows, "flag_price_only");
    doc.summary_sell = usSellTally(rows);
    if (meta.calendar) doc.calendar = meta.calendar;
    if (meta.news) {
      doc.news = {
        ...meta.news,
        vetoed_signals: sum(rows.filter((row) => row.outcome !== null).map((row) => Number(row.news_veto || 0))),
      };
    }
    if (meta.context) doc.context = meta.context;
    if (market === "qqq" && ndxDoc && sameArray(ndxDoc.time, doc.time)) {
      doc.comparison = comparison(ndxDoc, doc);
    }
    doc.latest = {
      date: last.date,
      open: rnd(last.open, 2),
      high: rnd(last.high, 2),
      low: rnd(last.low, 2),
      close: rnd(last.close, 2),
      volume: last.volume,
      score: rnd(last.score, 4),
      threshold: rnd(last.threshold, 4),
      sell_threshold: rnd(last.sell_threshold, 4),
      flag: last.flag === null ? 0 : Number(last.flag || 0),
      sell_flag: last.sell_flag === null ? 0 : Number(last.sell_flag || 0),
      flag_price_only: last.flag_price_only === null ? 0 : Number(last.flag_price_only || 0),
      news_veto: Boolean(last.news_veto),
      is_closed: Boolean(last.is_closed),
    };
  }

  return doc;
}

/** Load only the per-session rows (no meta) — used for the NDX cross-reference. */
async function loadMarketRows(db, market) {
  const instrument = INSTRUMENT[market];
  const rows = await db.all(
    `SELECT s.*, b.open, b.high, b.low, b.close, b.volume,
            COALESCE(b.is_closed, 1) AS is_closed
     FROM signals s
     LEFT JOIN bars b ON b.symbol = ? AND b.date = s.date
     WHERE s.market = ?
     ORDER BY s.date ASC`,
    [instrument, market],
  );
  if (!rows.length) return null;
  const doc = {
    instrument,
    time: rows.map((row) => row.date),
    flag: rows.map((row) => row.flag),
    flag_price_only: rows.map((row) => row.flag_price_only),
    outcome: rows.map((row) => row.outcome),
    hit3: rows.map((row) => row.hit3),
    close: rows.map((row) => rnd(row.close, 2)),
    entry: rows.map((row) => row.entry),
    target: rows.map((row) => row.target),
    peak: rows.map((row) => rnd(row.peak, 2)),
    score: rows.map((row) => row.score),
    summary: usTally(rows, "flag"),
  };
  return doc;
}

function sameArray(a, b) {
  if (!a || !b || a.length !== b.length) return false;
  for (let i = 0; i < a.length; i += 1) if (a[i] !== b[i]) return false;
  return true;
}

function previousSession(days, day) {
  if (!days || !days.length) return null;
  let low = 0;
  let high = days.length;
  while (low < high) {
    const mid = (low + high) >> 1;
    if (days[mid] < day) low = mid + 1;
    else high = mid;
  }
  return low > 0 ? days[low - 1] : null;
}

function usTally(rows, flagColumn) {
  const resolved = rows.filter((row) => row.outcome !== null && row.outcome !== undefined);

  const split = (sample) => {
    if (!sample.length) return { hit3: null, hit45: null, miss: null };
    const three = sample.filter((row) => Number(row.hit3) === 1).length / sample.length;
    const later = sample.filter((row) => Number(row.outcome) === 1 && row.hit3 != null && Number(row.hit3) === 0).length / sample.length;
    return { hit3: rnd(three, 4), hit45: rnd(later, 4), miss: rnd(1 - three - later, 4) };
  };

  const flagged = resolved.filter((row) => Number(row[flagColumn] || 0));
  const held = resolved.filter((row) => row.date >= HOLDOUT);
  const heldFlag = held.filter((row) => Number(row[flagColumn] || 0));
  const fSplit = split(flagged);
  const bSplit = split(resolved);

  return {
    n: resolved.length,
    base_rate: resolved.length ? rnd(mean(resolved.map((row) => Number(row.outcome))) , 4) : null,
    flagged: flagged.length,
    hit: flagged.length ? rnd(mean(flagged.map((row) => Number(row.outcome))), 4) : null,
    holdout_n: held.length,
    holdout_base_rate: held.length ? rnd(mean(held.map((row) => Number(row.outcome))), 4) : null,
    holdout_flagged: heldFlag.length,
    holdout_hit: heldFlag.length ? rnd(mean(heldFlag.map((row) => Number(row.outcome))), 4) : null,
    hit3: fSplit.hit3,
    hit45: fSplit.hit45,
    miss: fSplit.miss,
    base3: bSplit.hit3,
    base45: bSplit.hit45,
    holdout_from: HOLDOUT,
  };
}

function usSellTally(rows) {
  const usable = rows.filter((row) => row.base_close != null && row.close5 != null);
  const down5 = (row) => (Number(row.close5) < Number(row.base_close) ? 1 : 0);
  const down3 = (row) => (row.close3 != null && Number(row.close3) < Number(row.base_close) ? 1 : 0);
  const split = (sample) => {
    if (!sample.length) return { hit3: null, hit45: null, miss: null };
    const three = sample.filter((row) => down3(row) === 1).length / sample.length;
    const later = sample.filter((row) => down5(row) === 1 && down3(row) === 0).length / sample.length;
    return { hit3: rnd(three, 4), hit45: rnd(later, 4), miss: rnd(1 - three - later, 4) };
  };
  const flagged = usable.filter((row) => Number(row.sell_flag || 0));
  const held = usable.filter((row) => row.date >= HOLDOUT);
  const heldFlag = held.filter((row) => Number(row.sell_flag || 0));
  const fSplit = split(flagged);
  const bSplit = split(usable);
  return {
    n: usable.length,
    base_rate: usable.length ? rnd(mean(usable.map(down5)), 4) : null,
    flagged: flagged.length,
    hit: flagged.length ? rnd(mean(flagged.map(down5)), 4) : null,
    holdout_n: held.length,
    holdout_base_rate: held.length ? rnd(mean(held.map(down5)), 4) : null,
    holdout_flagged: heldFlag.length,
    holdout_hit: heldFlag.length ? rnd(mean(heldFlag.map(down5)), 4) : null,
    hit3: fSplit.hit3,
    hit45: fSplit.hit45,
    miss: fSplit.miss,
    base3: bSplit.hit3,
    base45: bSplit.hit45,
    holdout_from: HOLDOUT,
  };
}

function ssecSellSummary(rows) {
  const n = rows.length;
  const pair = (i) => {
    const r = rows[i];
    const base = r.base_close != null ? Number(r.base_close) : (r.close != null ? Number(r.close) : null);
    const c3 = r.close3 != null ? Number(r.close3) : (i + 3 < n && rows[i + 3].close != null ? Number(rows[i + 3].close) : null);
    const c5 = r.close5 != null ? Number(r.close5) : (i + 5 < n && rows[i + 5].close != null ? Number(rows[i + 5].close) : null);
    if (base == null || c5 == null) return null;
    return [base, c3 != null ? c3 : base, c5];
  };
  const split = (indices) => {
    if (!indices.length) return { hit3: null, hit45: null, miss: null };
    let three = 0;
    let later = 0;
    for (const i of indices) {
      const p = pair(i);
      if (!p) continue;
      if (p[1] < p[0]) three += 1;
      else if (p[2] < p[0]) later += 1;
    }
    const t = three / indices.length;
    const l = later / indices.length;
    return { hit3: rnd(t, 4), hit45: rnd(l, 4), miss: rnd(1 - t - l, 4) };
  };
  const win = (low) => {
    const usable = [];
    rows.forEach((r, i) => {
      if (r.date >= low && r.outcome != null && pair(i) != null) usable.push(i);
    });
    const chosen = usable.filter((i) => Number(rows[i].sell_flag || 0));
    const base = usable.length ? rnd(mean(usable.map((i) => (pair(i)[2] < pair(i)[0] ? 1 : 0))), 4) : null;
    const hit = chosen.length ? rnd(mean(chosen.map((i) => (pair(i)[2] < pair(i)[0] ? 1 : 0))), 4) : null;
    const cs = split(chosen);
    const us = split(usable);
    return {
      sessions: usable.length,
      signals: chosen.length,
      hit,
      base_rate: base,
      lift: hit == null || base == null ? null : rnd(hit - base, 4),
      hit3: cs.hit3,
      hit45: cs.hit45,
      miss: cs.miss,
      base3: us.hit3,
      base45: us.hit45,
    };
  };
  return { all: win("1900"), holdout: win(HOLDOUT), holdout_from: HOLDOUT };
}

function ssecSummary(rows, ndxDoc) {
  const n = rows.length;

  const hit3At = (index) => {
    const row = rows[index];
    if (row.hit3 !== null && row.hit3 !== undefined) return Number(row.hit3) === 1;
    if (index + 3 < n && row.close !== null && rows[index + 3].close !== null) {
      return Number(rows[index + 3].close) > Number(row.close);
    }
    return false;
  };

  const hit5At = (index) => {
    const row = rows[index];
    if (row.outcome !== null && row.outcome !== undefined) return Number(row.outcome) === 1;
    if (index + 5 < n && row.close !== null && rows[index + 5].close !== null) {
      return Number(rows[index + 5].close) > Number(row.close);
    }
    return false;
  };

  const split = (indices) => {
    if (!indices.length) return { hit3: null, hit45: null, miss: null };
    const three = indices.filter((index) => hit3At(index)).length / indices.length;
    const later = indices.filter((index) => !hit3At(index) && hit5At(index)).length / indices.length;
    return { hit3: rnd(three, 4), hit45: rnd(later, 4), miss: rnd(1 - three - later, 4) };
  };

  const windowSummary = (low) => {
    const usable = [];
    rows.forEach((row, index) => {
      if (row.date >= low && row.outcome !== null && row.outcome !== undefined) usable.push(index);
    });
    const chosen = usable.filter((index) => Number(rows[index].flag || 0));
    const base = usable.length ? rnd(mean(usable.map((index) => Number(rows[index].outcome))), 4) : null;
    const hit = chosen.length ? rnd(mean(chosen.map((index) => Number(rows[index].outcome))), 4) : null;
    const cSplit = split(chosen);
    const uSplit = split(usable);
    return {
      sessions: usable.length,
      signals: chosen.length,
      hit,
      base_rate: base,
      lift: hit === null || base === null ? null : rnd(hit - base, 4),
      hit3: cSplit.hit3,
      hit45: cSplit.hit45,
      miss: cSplit.miss,
      base3: uSplit.hit3,
      base45: uSplit.hit45,
    };
  };

  const bands = [];
  let reference = null;
  if (ndxDoc && ndxDoc.time && ndxDoc.time.length && ndxDoc.score) {
    const usDays = ndxDoc.time;
    const usScores = ndxDoc.score;
    const scored = [];
    let pointer = 0;
    for (const row of rows) {
      if (!Number(row.flag || 0) || row.outcome === null || row.outcome === undefined) continue;
      const day = row.date;
      while (pointer + 1 < usDays.length && usDays[pointer + 1] < day) pointer += 1;
      if (usScores[pointer] !== null && usScores[pointer] !== undefined) {
        scored.push([Number(usScores[pointer]), Number(row.outcome)]);
      }
    }
    if (scored.length) {
      const ordered = scored.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
      const size = Math.floor(ordered.length / 10);
      if (size > 0) {
        for (let band = 0; band < 10; band += 1) {
          const chunk = band < 9 ? ordered.slice(band * size, (band + 1) * size) : ordered.slice(9 * size);
          bands.push({ band: band + 1, hit: rnd(mean(chunk.map(([, hit]) => hit)), 4) });
        }
      }
    }
    const ns = ndxDoc.summary || {};
    if (ns.hit !== null && ns.hit !== undefined && ns.base_rate !== null && ns.base_rate !== undefined) {
      reference = {
        instrument: "^NDX",
        signals: ns.flagged,
        hit: ns.hit,
        base_rate: ns.base_rate,
        lift: rnd(ns.hit - ns.base_rate, 4),
        holdout_signals: ns.holdout_flagged,
        holdout_hit: ns.holdout_hit,
        holdout_base_rate: ns.holdout_base_rate,
        holdout_lift:
          ns.holdout_hit !== null && ns.holdout_hit !== undefined &&
          ns.holdout_base_rate !== null && ns.holdout_base_rate !== undefined
            ? rnd(ns.holdout_hit - ns.holdout_base_rate, 4)
            : null,
      };
    }
  }

  return {
    summary: {
      all: windowSummary("1900"),
      holdout: windowSummary(HOLDOUT),
      holdout_from: HOLDOUT,
    },
    bands,
    reference,
    signalsSeen: rows.filter((row) => Number(row.flag || 0)).length,
  };
}

function instrumentRow(doc) {
  const summary = doc.summary;
  const hit = summary.hit;
  const base = summary.base_rate;
  const heldHit = summary.holdout_hit;
  const heldBase = summary.holdout_base_rate;
  return {
    instrument: doc.instrument,
    signals: summary.flagged,
    precision: hit,
    base_rate: base,
    lift: hit === null || base === null ? null : rnd(hit - base, 4),
    holdout_signals: summary.holdout_flagged,
    holdout_precision: heldHit,
    holdout_base_rate: heldBase,
    holdout_lift: heldHit === null || heldBase === null ? null : rnd(heldHit - heldBase, 4),
  };
}

function comparison(primary, other) {
  const disagreements = [];
  for (let index = 0; index < primary.time.length; index += 1) {
    if (!Number(primary.flag[index] || 0)) continue;
    if (primary.outcome[index] == null || other.outcome[index] == null) continue;
    if (Number(primary.outcome[index]) === Number(other.outcome[index])) continue;
    disagreements.push({
      date: primary.time[index],
      target: primary.target[index],
      a_return: rnd(Number(primary.peak[index]) / Number(primary.entry[index]) - 1, 6),
      b_return: rnd(Number(other.peak[index]) / Number(other.entry[index]) - 1, 6),
      a_hit: primary.outcome[index],
      b_hit: other.outcome[index],
    });
  }
  return {
    a: primary.instrument,
    b: other.instrument,
    rows: [instrumentRow(primary), instrumentRow(other)],
    same_days: sameArray(primary.time, other.time),
    same_flags: sameArray(primary.flag, other.flag),
    disagreements,
  };
}

// ------------------------------------------------------------------ /api/agent

async function loadAgentEval(db, dataset) {
  const rows = await db.all(
    "SELECT * FROM agent_eval WHERE dataset = ? ORDER BY as_of ASC, horizon ASC, question_id ASC",
    [dataset],
  );
  if (!rows.length) return null;

  const first = rows[0];
  const generatedAt = String(first.generated_at || "");
  const configModel = String(first.config_model || "");
  const mergedList = String(first.models_merged || "").split(",").filter(Boolean);

  const questionMeta = {};
  const answers = {};
  const telemetry = {};

  for (const row of rows) {
    const qid = String(row.question_id);
    const predictor = String(row.predictor);
    if (!(qid in questionMeta)) {
      questionMeta[qid] = {
        as_of: row.as_of,
        as_of_close: row.as_of_close,
        family: row.family,
        horizon: Number(row.horizon),
        outcome: Number(row.outcome),
        resolve_session: row.resolve_session,
        symbol: row.symbol,
        text: row.question_text,
        threshold: Number(row.threshold),
      };
    }
    answers[predictor] = answers[predictor] || {};
    if (!(predictor in telemetry)) telemetry[predictor] = row;
    if (row.probability !== null && row.probability !== undefined) {
      answers[predictor][qid] = Number(row.probability);
    }
  }

  const bench = Object.keys(questionMeta);
  const alive = Object.values(answers).filter((item) => Object.keys(item).length).map((item) => new Set(Object.keys(item)));
  let shared = alive.length ? [...alive[0]].filter((qid) => alive.every((set) => set.has(qid))) : [];
  shared.sort((a, b) => {
    const ra = questionMeta[a].resolve_session;
    const rb = questionMeta[b].resolve_session;
    if (ra !== rb) return ra < rb ? -1 : 1;
    return a < b ? -1 : a > b ? 1 : 0;
  });

  const outcomes = {};
  for (const qid of shared) outcomes[qid] = questionMeta[qid].outcome;

  const predictors = {};
  for (const name of Object.keys(answers).sort()) {
    const ans = answers[name];
    const tel = telemetry[name];
    if (!Object.keys(ans).length || !shared.length) {
      predictors[name] = {
        brier: null,
        log_loss: null,
        ece: null,
        hit_direction: null,
        mean_probability: null,
        observed_rate: Object.keys(outcomes).length ? rnd(mean(Object.values(outcomes)), 4) : null,
        cost_usd_total: Number(tel.cost_usd_total || 0),
        cost_usd_per_forecast: tel.cost_usd_per_forecast,
        latency_p50_s: tel.latency_p50_s,
        latency_p95_s: tel.latency_p95_s,
        tool_calls_mean: Number(tel.tool_calls_mean || 0),
        gate_hits: Number(tel.gate_hits || 0),
        failures: Number(tel.failures || 0),
        reliability: [],
        answers: {},
        pairs: {},
        by_family: {},
        n: 0,
      };
      continue;
    }

    const pairs = shared.map((qid) => [ans[qid], questionMeta[qid].outcome]);
    const families = [...new Set(shared.map((qid) => questionMeta[qid].family))].sort();
    const byFamily = {};
    for (const family of families) {
      const sample = shared
        .filter((qid) => questionMeta[qid].family === family)
        .map((qid) => [ans[qid], questionMeta[qid].outcome]);
      if (sample.length) {
        byFamily[family] = {
          n: sample.length,
          brier: rnd(brier(sample), 4),
          observed: rnd(mean(sample.map(([, outcome]) => outcome)), 3),
        };
      }
    }

    const answersRounded = {};
    for (const qid of Object.keys(ans)) answersRounded[qid] = rnd(ans[qid], 6);
    const pairsRounded = {};
    for (const qid of shared) pairsRounded[qid] = rnd(ans[qid], 6);

    predictors[name] = {
      brier: rnd(brier(pairs), 4),
      log_loss: rnd(logLoss(pairs), 4),
      ece: rnd(ece(pairs), 4),
      hit_direction: rnd(mean(pairs.map(([p, o]) => ((p >= 0.5) === (o === 1) ? 1 : 0))), 4),
      mean_probability: rnd(mean(pairs.map(([p]) => p)), 4),
      observed_rate: rnd(mean(pairs.map(([, o]) => o)), 4),
      cost_usd_total: Number(tel.cost_usd_total || 0),
      cost_usd_per_forecast: tel.cost_usd_per_forecast,
      latency_p50_s: tel.latency_p50_s,
      latency_p95_s: tel.latency_p95_s,
      tool_calls_mean: Number(tel.tool_calls_mean || 0),
      gate_hits: Number(tel.gate_hits || 0),
      failures: Number(tel.failures || 0),
      reliability: reliability(pairs),
      answers: answersRounded,
      pairs: pairsRounded,
      by_family: byFamily,
      n: shared.length,
    };
  }

  let references = {};
  if (shared.length) {
    const half = shared.map((qid) => [0.5, questionMeta[qid].outcome]);
    references = {
      "constant_0.5": { brier: rnd(brier(half), 4), log_loss: rnd(logLoss(half), 4) },
      observed_rate: rnd(mean(shared.map((qid) => questionMeta[qid].outcome)), 4),
    };
  }

  const modes = [...new Set(Object.values(questionMeta).map((item) => item.family))].sort();
  return {
    n: shared.length,
    scored_on_shared: shared.length,
    generated_at: generatedAt,
    bench,
    questions: shared,
    outcomes,
    question_meta: questionMeta,
    predictors,
    references,
    config: {
      symbol: String(first.symbol || "^VXN"),
      limit: bench.length,
      modes,
      model: configModel,
      models_merged: mergedList,
    },
  };
}

function brier(pairs) {
  return sum(pairs.map(([p, o]) => (p - o) ** 2)) / pairs.length;
}

function logLoss(pairs) {
  let total = 0;
  for (const [p, o] of pairs) {
    const clipped = Math.min(Math.max(p, 1e-6), 1 - 1e-6);
    total += -(o * Math.log(clipped) + (1 - o) * Math.log(1 - clipped));
  }
  return total / pairs.length;
}

function reliability(pairs, bins = 5) {
  const table = [];
  for (let index = 0; index < bins; index += 1) {
    const low = index / bins;
    const high = (index + 1) / bins;
    const bucket = pairs.filter(
      ([p]) => (low <= p && p < high) || (index === bins - 1 && p === 1.0),
    );
    if (!bucket.length) continue;
    table.push({
      bin: `${low.toFixed(1)}-${high.toFixed(1)}`,
      n: bucket.length,
      mean_predicted: rnd(mean(bucket.map(([p]) => p)), 4),
      observed: rnd(mean(bucket.map(([, o]) => o)), 4),
    });
  }
  return table;
}

function ece(pairs, bins = 5) {
  const table = reliability(pairs, bins);
  if (!table.length) return null;
  const total = sum(table.map((row) => row.n));
  return sum(table.map((row) => (row.n / total) * Math.abs(row.mean_predicted - row.observed)));
}

// ------------------------------------------------------------------ state / chat

async function getDbState(db) {
  try {
    const [marketRows, barRows, newsRows, agentRows, chatRows] = await db.batch([
      {
        sql: `SELECT s.market,
                COUNT(*) AS cnt,
                MAX(s.date) AS max_date,
                MIN(CASE WHEN s.outcome IS NULL THEN s.date END) AS first_unresolved,
                MAX(s.fetched_at) AS fetched_at,
                SUM(CASE WHEN COALESCE(b.is_closed, 1) = 0 THEN 1 ELSE 0 END) AS unclosed_count
         FROM signals s
         LEFT JOIN bars b ON b.symbol = CASE s.market WHEN 'ndx' THEN '^NDX' WHEN 'qqq' THEN 'QQQ' ELSE '000001.SS' END
                          AND b.date = s.date
         GROUP BY s.market`,
      },
      { sql: "SELECT symbol, COUNT(*) AS cnt, MAX(date) AS max_date FROM bars GROUP BY symbol" },
      { sql: "SELECT COUNT(*) AS cnt, MAX(as_of) AS max_as_of FROM news" },
      { sql: "SELECT dataset, COUNT(*) AS cnt, MAX(generated_at) AS generated_at FROM agent_eval GROUP BY dataset" },
      { sql: "SELECT COUNT(*) AS cnt, MAX(created_at) AS max_created_at FROM chat_messages" },
    ]);

    const markets = {};
    for (const row of marketRows) {
      markets[row.market] = {
        count: row.cnt,
        max_date: row.max_date,
        first_unresolved: row.first_unresolved,
        fetched_at: row.fetched_at,
        unclosed_count: row.unclosed_count ?? 0,
      };
    }
    const bars = {};
    for (const row of barRows) bars[row.symbol] = { count: row.cnt, max_date: row.max_date };
    const news = newsRows[0] ?? { cnt: 0, max_as_of: null };
    const agents = {};
    for (const row of agentRows) agents[row.dataset] = { count: row.cnt, generated_at: row.generated_at };
    const chat = chatRows[0] ?? { cnt: 0, max_created_at: null };

    return json({ ok: true, backend: db.backend, markets, bars, news, agents, chat });
  } catch (err) {
    return json({ ok: false, error: String(err && err.message ? err.message : err) }, 500);
  }
}

async function chatThreads(db, url) {
  const market = url.searchParams.get("market") || "ndx";
  try {
    const threads = await db.all(
      `SELECT thread_id, market, MAX(as_of) AS as_of, MIN(created_at) AS created_at,
              MAX(created_at) AS updated_at, COUNT(*) AS message_count
       FROM chat_messages WHERE market = ?
       GROUP BY thread_id, market ORDER BY updated_at DESC, thread_id DESC LIMIT 20`,
      [market],
    );
    let messages = [];
    if (threads.length) {
      messages = await db.all(
        "SELECT thread_id, seq, market, as_of, role, model, stance, content, created_at FROM chat_messages WHERE thread_id = ? ORDER BY seq ASC",
        [threads[0].thread_id],
      );
    }
    return json({
      ok: true,
      market,
      context: await chatContext(db, market),
      threads,
      active_thread_id: threads.length ? threads[0].thread_id : null,
      messages,
    });
  } catch (err) {
    return json({ ok: false, error: String(err && err.message ? err.message : err) }, 500);
  }
}

async function chatMessages(db, url) {
  const threadId = url.searchParams.get("thread_id") || "";
  if (!threadId) return json({ ok: true, thread_id: "", messages: [] });
  const messages = await db.all(
    "SELECT thread_id, seq, market, as_of, role, model, stance, content, created_at FROM chat_messages WHERE thread_id = ? ORDER BY seq ASC",
    [threadId],
  );
  return json({ ok: true, thread_id: threadId, messages });
}

/** Short market snapshot for the group-chat side panel (the full Python builder is not portable). */
function fmtPrice(value) {
  return value === null || value === undefined ? "-" : Number(value).toFixed(2);
}

async function chatContext(db, market) {
  const instrument = INSTRUMENT[market] || INSTRUMENT.ndx;
  try {
    const [bars, sigRows, recentNews] = await db.batch([
      { sql: "SELECT date, open, high, low, close, is_closed FROM bars WHERE symbol = ? ORDER BY date DESC LIMIT 2", params: [instrument] },
      { sql: "SELECT date, flag, score, threshold FROM signals WHERE market = ? ORDER BY date DESC LIMIT 1", params: [market] },
      { sql: "SELECT published_at, title FROM news WHERE doc_id != '' ORDER BY substr(published_at,1,10) DESC, published_at DESC LIMIT 5" },
    ]);
    const signal = sigRows[0];
    if (!bars.length) {
      return { market, symbol: instrument, as_of: "", text: `标的：${instrument}\n暂无行情数据。` };
    }
    const latest = bars[0];
    const prev = bars[1];
    const change = prev && prev.close ? latest.close / prev.close - 1 : 0;
    const pct = `${change >= 0 ? "+" : ""}${(change * 100).toFixed(2)}%`;
    const signalLine = signal
      ? `量化因子参考：${Number(signal.flag || 0) ? "触发买入信号 (BUY)" : "观望 / 无信号 (NO SIGNAL)"}` +
        (signal.score !== null && signal.threshold !== null
          ? `（合成得分 ${Number(signal.score).toFixed(3)} / 阈值 ${Number(signal.threshold).toFixed(3)}）`
          : "")
      : "";
    const newsLine = recentNews.length
      ? `3. 最新新闻参考：` + recentNews.map((n) => `[${String(n.published_at || "").slice(0, 10)}] ${String(n.title || "").slice(0, 80)}`).join("；")
      : "";
    return {
      market,
      symbol: instrument,
      as_of: latest.date,
      text: [
        `【标的】${instrument} · 基准日 ${latest.date}（${Number(latest.is_closed) ? "已收盘" : "盘中未收盘"}）`,
        `1. 当前盘面：收/现价 ${Number(latest.close).toFixed(2)}（当日 ${pct}），开 ${fmtPrice(latest.open)} / 高 ${fmtPrice(latest.high)} / 低 ${fmtPrice(latest.low)}`,
        signalLine ? `2. ${signalLine}` : "",
        newsLine,
      ]
        .filter(Boolean)
        .join("\n"),
    };
  } catch (err) {
    return { market, symbol: instrument, as_of: "", text: `上下文读取失败：${err.message || err}` };
  }
}

// ------------------------------------------------------------------ ingest

async function ingest(request, db, env) {
  const token = (request.headers.get("Authorization") || "").replace(/^Bearer\s+/i, "");
  if (!env.INGEST_TOKEN) return json({ ok: false, error: "INGEST_TOKEN is not set on the Worker" }, 500);
  if (!timingSafeEqualStr(token, env.INGEST_TOKEN)) return json({ ok: false, error: "bad token" }, 401);

  let body;
  try {
    body = await request.json();
  } catch {
    return json({ ok: false, error: "body must be JSON" }, 400);
  }

  try {
    if (body.op === "state") return getDbState(db);
    if (body.op !== "sql") return json({ ok: false, error: "unknown op" }, 400);
    if (typeof body.sql !== "string") return json({ ok: false, error: "sql must be a string" }, 400);
    if (body.sql.length > 95_000) return json({ ok: false, error: "sql too large; send fewer rows per statement" }, 400);
    if (db.dialect === "postgres" && /^\s*INSERT\s+OR\s+(REPLACE|IGNORE)\b/i.test(body.sql)) {
      return json({ ok: false, error: "Postgres backend active; use scripts/sync_supabase.py or Postgres INSERT ... ON CONFLICT syntax" }, 400);
    }
    if (
      !/^\s*INSERT\s+(OR\s+(REPLACE|IGNORE)\s+)?INTO\s+(bars|signals|news|agent_eval|chat_messages)\b/i.test(body.sql)
    ) {
      return json({ ok: false, error: "INSERT only on bars/signals/news/agent_eval/chat_messages" }, 400);
    }
    const result = await db.execRaw(body.sql);
    return json({ ok: true, op: "sql", count: result.count ?? null });
  } catch (error) {
    return json({ ok: false, error: String(error && error.message ? error.message : error) }, 400);
  }
}

// ------------------------------------------------------------------ /api/refresh

function tzForSymbol(symbol) {
  return symbol.endsWith(".SS") || symbol.endsWith(".SZ") ? "Asia/Shanghai" : "America/New_York";
}

function localDateTimeParts(date, timeZone) {
  const parts = new Intl.DateTimeFormat("en-CA", {
    timeZone,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    weekday: "short",
    hour12: false,
  }).formatToParts(date);
  const get = (type) => parts.find((part) => part.type === type)?.value || "00";
  const wd = get("weekday");
  const isWeekday = wd !== "Sat" && wd !== "Sun";
  const hour = Number(get("hour")) % 24;
  const minute = Number(get("minute"));
  return {
    dateStr: `${get("year")}-${get("month")}-${get("day")}`,
    mins: hour * 60 + minute,
    isWeekday,
  };
}

function beijingDateStr(date = new Date()) {
  return localDateTimeParts(date, "Asia/Shanghai").dateStr;
}

function isSessionClosed(sessionDate, tzName = "America/New_York", now = new Date()) {
  const isChina = tzName === "Asia/Shanghai";
  const targetTz = isChina ? "Asia/Shanghai" : "America/New_York";
  const closeHm = isChina ? 15 * 60 + 5 : 16 * 60 + 15;
  const { dateStr: localDate, mins } = localDateTimeParts(now, targetTz);
  if (sessionDate < localDate) return 1;
  if (sessionDate > localDate) return 0;
  return mins >= closeHm ? 1 : 0;
}

/**
 * Mirrors local `server.py::_needs_refresh` + `db.py::refresh_check_summary`:
 *   1. Any market whose `signals` lags behind `bars` -> refresh immediately
 *   2. Any unclosed bar past market close (overdue) -> refresh immediately
 *   3. Any active intraday unclosed bar older than 5 min -> refresh
 *   4. Any market whose `signals.fetched_at` hasn't crossed today's open/close -> refresh
 */
async function checkNeedsRefresh(db, now = new Date()) {
  const freshCutMs = now.getTime() - 5 * 60 * 1000;
  const [unclosed, markets, barHeads] = await db.batch([
    { sql: "SELECT symbol, date, fetched_at FROM bars WHERE is_closed = 0" },
    { sql: "SELECT market, MAX(date) AS last_session, MAX(fetched_at) AS fetched_at FROM signals GROUP BY market" },
    { sql: "SELECT symbol, MAX(date) AS max_date FROM bars WHERE symbol IN ('^NDX', 'QQQ', '000001.SS') GROUP BY symbol" },
  ]);

  const sessions = Object.fromEntries(markets.map((row) => [row.market, row.last_session]));
  if (markets.length < 3) return { needed: true, sessions };

  const barMaxBySym = new Map(barHeads.map((r) => [r.symbol, r.max_date]));
  for (const row of markets) {
    const sym = INSTRUMENT[row.market];
    const barMax = barMaxBySym.get(sym);
    if (barMax && (!row.last_session || row.last_session < barMax)) {
      return { needed: true, sessions };
    }
  }

  for (const row of unclosed) {
    if (isSessionClosed(row.date, tzForSymbol(row.symbol), now)) {
      return { needed: true, sessions };
    }
    const fetchedMs = row.fetched_at ? Date.parse(row.fetched_at) : NaN;
    if (!Number.isFinite(fetchedMs) || fetchedMs < freshCutMs) {
      return { needed: true, sessions };
    }
  }

  for (const row of markets) {
    const fetchedMs = row.fetched_at ? Date.parse(row.fetched_at) : NaN;
    if (!Number.isFinite(fetchedMs)) return { needed: true, sessions };
    const isChina = row.market === "ssec";
    const tz = isChina ? "Asia/Shanghai" : "America/New_York";
    const openHm = isChina ? 9 * 60 + 25 : 9 * 60 + 30;
    const closeHm = isChina ? 15 * 60 + 5 : 16 * 60 + 15;
    const localNow = localDateTimeParts(now, tz);
    const localFetched = localDateTimeParts(new Date(fetchedMs), tz);

    if (localFetched.dateStr < localNow.dateStr) {
      if (localFetched.mins < closeHm) return { needed: true, sessions };
      if (localNow.isWeekday && localNow.mins >= openHm) return { needed: true, sessions };
    } else if (localFetched.dateStr === localNow.dateStr) {
      if (localFetched.mins < closeHm && localNow.mins >= closeHm) return { needed: true, sessions };
    }
  }

  return { needed: false, sessions };
}

async function fetchYahooBars(symbol, startUnix, endUnix) {
  const url = `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(symbol)}?interval=1d&period1=${startUnix}&period2=${endUnix}`;
  const res = await fetch(url, { headers: { "User-Agent": "Mozilla/5.0 forecast-log/1.0" } });
  if (!res.ok) throw new Error(`Yahoo ${symbol} HTTP ${res.status}`);
  const payload = await res.json();
  const result = payload?.chart?.result?.[0];
  if (!result) return [];
  const tz = result?.meta?.exchangeTimezoneName || tzForSymbol(symbol);
  const stamps = result.timestamp || [];
  const quote = result.indicators?.quote?.[0] || {};
  const fmt = new Intl.DateTimeFormat("en-CA", {
    timeZone: tz,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  });
  const now = new Date();
  const byDate = new Map();
  for (let i = 0; i < stamps.length; i += 1) {
    const closeRaw = quote.close?.[i];
    if (closeRaw === null || closeRaw === undefined) continue;
    const close = rnd(closeRaw, 4);
    if (!Number.isFinite(close) || close <= 0) continue;
    const day = fmt.format(new Date(stamps[i] * 1000));
    const open = quote.open?.[i] != null && Number(quote.open[i]) > 0 ? rnd(quote.open[i], 4) : null;
    let high = quote.high?.[i] != null && Number(quote.high[i]) > 0 ? rnd(quote.high[i], 4) : null;
    let low = quote.low?.[i] != null && Number(quote.low[i]) > 0 ? rnd(quote.low[i], 4) : null;
    if (high !== null && low !== null && low > high) [low, high] = [high, low];
    const volRaw = quote.volume?.[i];
    const volume = volRaw != null && Number(volRaw) >= 0 ? Math.trunc(volRaw) : null;
    byDate.set(day, {
      date: day,
      // Same precision as forecast_log.yahoo: bars store 4 decimals; the API
      // rounds to 2 when it builds the chart series.
      open,
      high,
      low,
      close,
      volume,
      is_closed: isSessionClosed(day, tz, now),
    });
  }
  return [...byDate.values()].sort((a, b) => a.date.localeCompare(b.date));
}

async function mapConcurrent(items, limit, fn) {
  const results = new Array(items.length);
  let nextIdx = 0;
  const workers = Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (nextIdx < items.length) {
      const idx = nextIdx;
      nextIdx += 1;
      results[idx] = await fn(items[idx], idx);
    }
  });
  await Promise.all(workers);
  return results;
}

async function refreshFromYahoo(db, env, url, force = false) {
  try {
    const now = new Date();
    const todayBj = beijingDateStr(now);
    const symbolStats = await db.all(
      `SELECT symbol,
              MAX(date) AS max_date,
              MIN(CASE WHEN is_closed = 0 THEN date END) AS min_unclosed
       FROM bars
       GROUP BY symbol
       ORDER BY symbol`,
    );
    if (!symbolStats.length) return json({ ok: false, error: "数据库尚未初始化，请先同步数据" }, 404);

    if (!force) {
      const check = await checkNeedsRefresh(db, now);
      if (!check.needed) {
        return json({
          ok: true,
          cached: true,
          beijing_date: todayBj,
          sessions: check.sessions,
        });
      }
    }

    const updated = {};
    const errors = {};
    const nowSec = Math.floor(now.getTime() / 1000) + 172800;
    const chunkSize = db.dialect === "postgres" ? 50 : 10;

    await mapConcurrent(symbolStats, 6, async ({ symbol, max_date: maxDate, min_unclosed: minUnclosed }) => {
      try {
        if (!maxDate) return;
        const anchor = minUnclosed && minUnclosed < maxDate ? minUnclosed : maxDate;
        const startUnix = Math.floor(new Date(`${anchor}T00:00:00Z`).getTime() / 1000) - 86400 * 6;
        const bars = await fetchYahooBars(symbol, startUnix, nowSec);
        if (!bars.length) return;
        const fetchedAt = new Date().toISOString();
        for (let offset = 0; offset < bars.length; offset += chunkSize) {
          const slice = bars.slice(offset, offset + chunkSize);
          const tuples = slice.map(() => "(?, ?, ?, ?, ?, ?, ?, ?, ?)").join(", ");
          const params = [];
          for (const bar of slice) {
            params.push(symbol, bar.date, bar.open, bar.high, bar.low, bar.close, bar.volume, bar.is_closed, fetchedAt);
          }
          await db.run(
            `INSERT INTO bars (symbol, date, open, high, low, close, volume, is_closed, fetched_at)
             VALUES ${tuples}
             ON CONFLICT (symbol, date) DO UPDATE SET
               open = excluded.open,
               high = excluded.high,
               low = excluded.low,
               close = excluded.close,
               volume = excluded.volume,
               is_closed = excluded.is_closed,
               fetched_at = excluded.fetched_at`,
            params,
          );
        }
        const latestDate = bars[bars.length - 1].date;
        if (minUnclosed && minUnclosed < latestDate) {
          // Close any stale historical intraday bar that Yahoo no longer emits
          const tz = tzForSymbol(symbol);
          if (isSessionClosed(minUnclosed, tz, now)) {
            await db.run(
              "UPDATE bars SET is_closed = 1, fetched_at = ? WHERE symbol = ? AND is_closed = 0 AND date < ?",
              [fetchedAt, symbol, latestDate],
            );
          }
        }
        updated[symbol] = latestDate;
      } catch (err) {
        errors[symbol] = String(err && err.message ? err.message : err);
      }
    });

    const [eventsMeta, newsIndicators, ndxMeta, qqqMeta, ssecMeta] = await Promise.all([
      assetJson(env, url, "/_meta/events.json"),
      assetJson(env, url, "/_meta/news-indicators.json"),
      assetJson(env, url, "/_meta/cross-ndx.json"),
      assetJson(env, url, "/_meta/cross-qqq.json"),
      assetJson(env, url, "/_meta/cross-ssec.json"),
    ]);

    const refreshedAt = new Date().toISOString();
    const signalsUpdated = await recomputeAllSignals(db, {
      calendar: eventsMeta?.calendar || null,
      newsIndicators,
      contextsByMarket: {
        ndx: (ndxMeta?.contexts || []).map((c) => c.key),
        qqq: (qqqMeta?.contexts || []).map((c) => c.key),
        ssec: (ssecMeta?.contexts || []).map((c) => c.key),
      },
      fetchedAt: refreshedAt,
      rnd,
      scanKeywords,
    });

    const sessions = await db.all("SELECT market, MAX(date) AS last_session FROM signals GROUP BY market");
    return json({
      ok: true,
      refreshed_at: refreshedAt,
      beijing_date: beijingDateStr(),
      force,
      sessions: Object.fromEntries(sessions.map((row) => [row.market, row.last_session])),
      updated,
      signals_updated: signalsUpdated,
      errors,
    });
  } catch (err) {
    return json({ ok: false, error: String(err && err.message ? err.message : err) }, 500);
  }
}

// ------------------------------------------------------------------ /api/news & /api/news/refresh

const NEWS_COOLDOWN_MS = 15 * 60 * 1000;
const LIVE_NEWS_FEEDS = [
  ["wsj-markets", "https://feeds.a.dj.com/rss/RSSMarketsMain.xml"],
  ["cnbc-markets", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258"],
  ["cnbc-world", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100727362"],
  ["fed-monetary", "https://www.federalreserve.gov/feeds/press_monetary.xml"],
  ["sec-press", "https://www.sec.gov/news/pressreleases.rss"],
  ["yahoo-ndx", "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5ENDX,QQQ,SPY&region=US&lang=en-US"],
  ["yahoo-ssec", "https://feeds.finance.yahoo.com/rss/2.0/headline?s=000001.SS,FXI,KWEB&region=US&lang=en-US"],
];

const KEYWORD_RULES = [
  ["fed", /\bfed\b|federal reserve|fomc/i],
  ["rate hike", /raise rates|rate hike|hike rates|tighten/i],
  ["rate cut", /cut rates|rate cut|easing|stimulus/i],
  ["inflation", /inflation|consumer prices/i],
  ["recession", /recession|downturn|slowdown/i],
  ["tariff", /tariff|trade war|trade dispute/i],
  ["jobs", /unemployment|payrolls|jobless|jobs report|job cuts/i],
  ["gdp", /\bgdp\b|economic growth/i],
  ["oil", /oil price|crude|opec|barrel/i],
  ["dollar", /dollar|currency market/i],
  ["bond yields", /bond yield|treasury yield|yields/i],
  ["stocks up", /record high|shares (rise|surge|gain)|rally|stocks (rise|jump|climb)/i],
  ["stocks down", /sell-?off|plunge|slump|crash|tumble|stocks (fall|drop|slide)/i],
  ["earnings", /earnings|profit warning|quarterly results/i],
  ["ipo", /\bipo\b|initial public offering/i],
  ["bankruptcy", /bankrupt|insolven|default/i],
  ["china", /\bchina\b|beijing|pboc/i],
  ["europe", /\becb\b|eurozone|european central bank/i],
  ["geopolitics", /sanction|invasion|missile|air ?strike|military operation|\bwar\b/i],
  ["trade deal", /trade deal|trade agreement|exports|imports/i],
  ["crypto", /bitcoin|crypto|blockchain/i],
  ["housing", /housing|home sales|mortgage/i],
  ["banks", /\bbank\b|banking crisis|lender/i],
  ["election", /election|referendum|vote/i],
];
const RISK_OFF_KEYS = new Set(["geopolitics", "recession", "bankruptcy", "tariff", "stocks down", "banks"]);
const RISK_ON_KEYS = new Set(["rate cut", "stocks up", "trade deal", "housing"]);

function decodeXml(str) {
  return String(str || "")
    .replace(/<!\[CDATA\[([\s\S]*?)\]\]>/g, "$1")
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#39;|&apos;/g, "'")
    .replace(/&#(\d+);/g, (_, n) => String.fromCharCode(Number(n)))
    .replace(/<[^>]+>/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function hashId(input) {
  let h1 = 0xdeadbeef ^ input.length;
  let h2 = 0x41c6ce57 ^ input.length;
  for (let i = 0; i < input.length; i += 1) {
    const ch = input.charCodeAt(i);
    h1 = Math.imul(h1 ^ ch, 2654435761);
    h2 = Math.imul(h2 ^ ch, 1597334677);
  }
  h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507) ^ Math.imul(h2 ^ (h2 >>> 13), 3266489909);
  h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507) ^ Math.imul(h1 ^ (h1 >>> 13), 3266489909);
  return ((h2 >>> 0).toString(16).padStart(8, "0") + (h1 >>> 0).toString(16).padStart(8, "0")).slice(0, 12);
}

function parseRssXml(name, url, xml) {
  const docs = [];
  const blocks = xml.match(/<(item|entry)\b[\s\S]*?<\/\1>/gi) || [];
  const fallbackDomain = url.split("/")[2] || name;
  for (const block of blocks) {
    const titleMatch = block.match(/<title\b[^>]*>([\s\S]*?)<\/title>/i);
    const linkHref = block.match(/<link\b[^>]*href=["']([^"']+)["'][^>]*\/?>/i);
    const linkBody = block.match(/<link\b[^>]*>([\s\S]*?)<\/link>/i);
    const guidMatch = block.match(/<guid\b[^>]*>([\s\S]*?)<\/guid>/i);
    const dateMatch = block.match(/<(pubDate|updated|published|dc:date)\b[^>]*>([\s\S]*?)<\/\1>/i);
    const title = decodeXml(titleMatch?.[1] || "").slice(0, 400);
    let link = decodeXml(linkHref?.[1] || linkBody?.[1] || "");
    if (!link && guidMatch?.[1] && /^https?:\/\//i.test(guidMatch[1].trim())) link = guidMatch[1].trim();
    const rawDate = decodeXml(dateMatch?.[2] || "");
    if (!title || !link || !rawDate) continue;
    const parsed = new Date(rawDate);
    const publishedAt = Number.isNaN(parsed.getTime())
      ? (/^\d{4}-\d{2}-\d{2}/.test(rawDate) ? rawDate.slice(0, 10) : "")
      : parsed.toISOString().replace(/\.\d{3}Z$/, "+00:00");
    if (!publishedAt) continue;
    let domain = fallbackDomain;
    try { domain = new URL(link).hostname; } catch {}
    docs.push({
      id: hashId(`${name}|${link}|${title}`),
      published_at: publishedAt,
      title,
      text: title,
      url: link,
      domain,
      source: name,
    });
  }
  return docs;
}

function scanKeywords(titles) {
  const joined = titles.join(" ");
  const hits = KEYWORD_RULES.filter(([, re]) => re.test(joined)).map(([k]) => k);
  const riskOff = hits.filter((k) => RISK_OFF_KEYS.has(k));
  const riskOn = hits.filter((k) => RISK_ON_KEYS.has(k));
  return {
    items: titles.length,
    hits,
    risk_off_hits: riskOff,
    risk_on_hits: riskOn,
    tone: rnd((riskOn.length - riskOff.length) / Math.max(titles.length, 1), 4),
    veto_watch: riskOff.length >= 2,
  };
}

function classifyNewsRow(row) {
  const title = String(row.title || "");
  const tags = KEYWORD_RULES.filter(([, re]) => re.test(title)).map(([k]) => k);
  const off = tags.filter((k) => RISK_OFF_KEYS.has(k));
  const on = tags.filter((k) => RISK_ON_KEYS.has(k));
  const sentiment = off.length && !on.length ? "risk-off" : on.length && !off.length ? "risk-on" : off.length && on.length ? "mixed" : "neutral";
  const src = String(row.source || "");
  const dom = String(row.domain || "").toLowerCase();
  let category = "global";
  if (src.includes("yahoo-ssec") || /\b(china|chinese|beijing|pboc|shanghai|csi|hang seng|alibaba|tencent)\b/i.test(title)) category = "china";
  else if (src.includes("fed-monetary") || src.includes("sec-press") || dom.includes("federalreserve.gov") || dom.includes("sec.gov")) category = "macro";
  else if (src.includes("wikipedia") || dom.includes("wikipedia.org")) category = "global";
  else if (/wsj|cnbc-markets|yahoo-ndx/.test(src) || /wsj\.com|cnbc\.com|yahoo\.com/.test(dom)) category = "us";
  return { ...row, category, tags: tags.slice(0, 5), sentiment };
}

async function crossAssetsSnapshot(db) {
  const syms = ["^TNX", "DX-Y.NYB", "HYG", "TLT", "^VIX", "^VXN", "GC=F"];
  const placeholders = syms.map(() => "?").join(",");
  const cutoff = daysAgoStr(400);
  const rows = await db.all(
    `SELECT symbol, date, close, is_closed FROM bars WHERE symbol IN (${placeholders}) AND date >= ? ORDER BY symbol ASC, date ASC`,
    [...syms, cutoff],
  );
  const bySym = new Map();
  for (const r of rows) {
    if (!bySym.has(r.symbol)) bySym.set(r.symbol, []);
    bySym.get(r.symbol).push(r);
  }
  const pct = (a, b) => (b ? rnd((a / b - 1) * 100, 2) : null);
  const buildSingle = (symbol, label, unit, dir) => {
    const arr = bySym.get(symbol);
    if (!arr || !arr.length) return null;
    const last = arr[arr.length - 1];
    const c0 = Number(last.close);
    const c1 = arr.length >= 2 ? Number(arr[arr.length - 2].close) : null;
    const c5 = arr.length >= 6 ? Number(arr[arr.length - 6].close) : null;
    const c20 = arr.length >= 21 ? Number(arr[arr.length - 21].close) : null;
    const chg1 = c1 != null ? (unit === "diff" ? rnd(c0 - c1, 3) : pct(c0, c1)) : null;
    const chg5 = c5 != null ? (unit === "diff" ? rnd(c0 - c5, 3) : pct(c0, c5)) : null;
    const chg20 = c20 != null ? (unit === "diff" ? rnd(c0 - c20, 3) : pct(c0, c20)) : null;
    let hint = "中性";
    if ((symbol === "^VIX" || symbol === "^VXN") && arr.length >= 60) {
      const w = arr.slice(-252).map((x) => Number(x.close)).sort((a, b) => a - b);
      const med = w[Math.floor(w.length / 2)];
      const ratio = med ? (c0 / med - 1) * 100 : 0;
      hint = `较年均中位 ${ratio >= 0 ? "+" : ""}${ratio.toFixed(1)}%` + (ratio > 5 ? "（偏高→恐慌反向偏多）" : ratio < -5 ? "（偏低→平静）" : "");
    } else if (chg20 != null) {
      if (dir === "down") hint = chg20 < 0 ? "20日下行（偏多因子）" : "20日上行（偏空压力）";
      else if (dir === "up") hint = chg20 > 0 ? "20日走强（偏多因子）" : "20日走弱（偏空压力）";
    }
    return { symbol, label, date: last.date, close: rnd(c0, 2), unit, chg_1d: chg1, chg_5d: chg5, chg_20d: chg20, is_closed: Boolean(last.is_closed), hint };
  };

  const out = [];
  for (const spec of [
    ["^TNX", "10Y 美债收益率", "diff", "down"],
    ["DX-Y.NYB", "美元指数 DXY", "pct", "down"],
    ["^VIX", "标普波动率 VIX", "pct", "high"],
    ["^VXN", "纳指波动率 VXN", "pct", "high"],
    ["GC=F", "黄金期货 GC=F", "pct", "neutral"],
  ]) {
    const item = buildSingle(...spec);
    if (item) out.push(item);
  }
  const hyg = bySym.get("HYG") || [];
  const tlt = bySym.get("TLT") || [];
  if (hyg.length && tlt.length) {
    const tltMap = new Map(tlt.map((r) => [r.date, Number(r.close)]));
    const ratios = hyg.filter((r) => tltMap.get(r.date) > 0).map((r) => [r.date, Number(r.close) / tltMap.get(r.date)]);
    if (ratios.length) {
      const [d0, r0] = ratios[ratios.length - 1];
      const r1 = ratios.length >= 2 ? ratios[ratios.length - 2][1] : null;
      const r5 = ratios.length >= 6 ? ratios[ratios.length - 6][1] : null;
      const r20 = ratios.length >= 21 ? ratios[ratios.length - 21][1] : null;
      const c20 = r20 != null ? pct(r0, r20) : null;
      out.splice(2, 0, {
        symbol: "HYG/TLT",
        label: "信用风险偏好 (HYG/TLT)",
        date: d0,
        close: rnd(r0, 4),
        unit: "pct",
        chg_1d: r1 != null ? pct(r0, r1) : null,
        chg_5d: r5 != null ? pct(r0, r5) : null,
        chg_20d: c20,
        is_closed: Boolean(hyg[hyg.length - 1].is_closed && tlt[tlt.length - 1].is_closed),
        hint: (c20 || 0) > 0 ? "20日走强（偏多因子）" : "20日走弱（避险偏空）",
      });
    }
  }
  return out;
}

async function macroCalendarSnapshot(env, url, todayStr) {
  const evDoc = await assetJson(env, url, "/_meta/events.json");
  const cal = evDoc?.calendar;
  if (!cal) return { available: false, as_of: todayStr, events: [] };
  const todayMs = new Date(`${todayStr}T00:00:00Z`).getTime();
  const specs = [
    ["cpi", "CPI 通胀数据", "CPI 前 0–3 天且复合分 ≥ 中位触发买入增补"],
    ["nfp", "NFP 非农就业", "NFP 前 3–7 天且复合分 ≥ 中位触发买入增补"],
    ["fomc", "FOMC 联储决议", "FOMC 决议当天否决所有买卖信号"],
  ];
  return {
    available: true,
    as_of: todayStr,
    events: specs.map(([key, label, ruleHint]) => {
      const nextDate = (cal[key] || []).find((d) => d >= todayStr) || null;
      const daysAway = nextDate ? Math.round((new Date(`${nextDate}T00:00:00Z`).getTime() - todayMs) / 86400000) : null;
      let inWindow = false;
      let windowStatus = "常规期";
      if (daysAway != null) {
        if (key === "cpi" && daysAway >= 0 && daysAway <= 3) { inWindow = true; windowStatus = "触发窗口（临近 CPI）"; }
        else if (key === "nfp" && daysAway >= 3 && daysAway <= 7) { inWindow = true; windowStatus = "触发窗口（NFP 前 3–7 天）"; }
        else if (key === "fomc" && daysAway === 0) { inWindow = true; windowStatus = "否决窗口（FOMC 决议日）"; }
        else if (key === "fomc" && daysAway >= 1 && daysAway <= 3) { windowStatus = "临近决议（观望留意）"; }
      }
      return { key, label, next_date: nextDate, days_away: daysAway, in_window: inWindow, window_status: windowStatus, rule_hint: ruleHint };
    }),
  };
}

async function loadNewsBundle(db, env, url, { market = "ndx", limit = 50, cached = true, fetchMeta = {} } = {}) {
  const todayBj = beijingDateStr();
  const newsCutoff = daysAgoStr(20);
  const [aggRows, recentDays, itemRows, crossAssets, macroCalendar, ndxMeta, llmIndicators] = await Promise.all([
    db.all("SELECT COUNT(*) AS total_rows, MAX(CASE WHEN doc_id != '' THEN as_of END) AS latest_as_of, MAX(CASE WHEN doc_id != '' THEN fetched_at END) AS latest_fetched_at FROM news"),
    db.all("SELECT as_of, COUNT(*) AS count, MAX(source) AS source, MAX(fetched_at) AS fetched_at FROM news WHERE doc_id != '' GROUP BY as_of ORDER BY as_of DESC LIMIT 7"),
    db.all(
      "SELECT as_of, doc_id AS id, published_at, title, text, url, domain, source, fetched_at FROM news WHERE doc_id != '' AND as_of >= ? ORDER BY substr(published_at,1,10) DESC, published_at DESC, as_of DESC, fetched_at DESC LIMIT ?",
      [newsCutoff, Math.max(limit * 3, 120)],
    ),
    crossAssetsSnapshot(db),
    macroCalendarSnapshot(env, url, todayBj),
    assetJson(env, url, "/_meta/cross-ndx.json"),
    assetJson(env, url, "/_meta/news-indicators.json"),
  ]);

  const agg = aggRows[0] || {};
  const seen = new Set();
  const deduped = [];
  for (const row of itemRows) {
    const title = String(row.title || "").trim();
    if (!title) continue;
    const u = String(row.url || "").trim();
    const key = u && !u.includes("en.wikipedia.org") ? u.toLowerCase() : title.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    deduped.push(classifyNewsRow(row));
    if (deduped.length >= limit) break;
  }

  const kw = scanKeywords(deduped.slice(0, 40).map((it) => it.title));
  const sourcesSeen = [];
  for (const it of deduped) {
    for (const s of String(it.source || "").split("+")) {
      const clean = s.trim();
      if (clean && clean !== "none" && !sourcesSeen.includes(clean)) sourcesSeen.push(clean);
    }
  }

  return json({
    ok: true,
    cached,
    market,
    beijing_date: todayBj,
    refreshed_at: fetchMeta.refreshed_at || agg.latest_fetched_at || new Date().toISOString(),
    news: {
      latest_as_of: agg.latest_as_of || null,
      fetched_at: agg.latest_fetched_at || "",
      total_rows: Number(agg.total_rows || 0),
      latest_day_docs: recentDays[0]?.as_of === agg.latest_as_of ? Number(recentDays[0].count || 0) : 0,
      recent_days: recentDays,
      sources: fetchMeta.providers || sourcesSeen,
      backfilled_days: fetchMeta.backfilled_days || [],
      new_count: fetchMeta.new_count || 0,
      errors: fetchMeta.errors || [],
      items: deduped,
    },
    keywords: kw,
    llm_indicators: llmIndicators || { running: false, deepseek: null, qwen: null, recent_days: [], errors: [] },
    macro_calendar: macroCalendar,
    cross_assets: crossAssets,
    daily_llm: ndxMeta?.context || null,
  });
}

async function refreshNews(db, env, url, { force = false, market = "ndx", limit = 50 } = {}) {
  try {
    const todayBj = beijingDateStr();
    if (!force) {
      const agg = await db.first(
        "SELECT MAX(CASE WHEN doc_id != '' THEN as_of END) AS latest_as_of, MAX(CASE WHEN doc_id != '' THEN fetched_at END) AS latest_fetched_at FROM news",
      );
      const latestDate = agg?.latest_as_of || "";
      const fetchedMs = agg?.latest_fetched_at ? new Date(agg.latest_fetched_at).getTime() : 0;
      if (latestDate >= todayBj && fetchedMs && Date.now() - fetchedMs < NEWS_COOLDOWN_MS) {
        return loadNewsBundle(db, env, url, { market, limit, cached: true });
      }
    }

    const fetchedUtc = new Date().toISOString().replace(/\.\d{3}Z$/, "+00:00");
    const providersOk = [];
    const errors = [];
    const allDocs = [];

    const results = await Promise.allSettled(
      LIVE_NEWS_FEEDS.map(async ([name, feedUrl]) => {
        const res = await fetch(feedUrl, { headers: { "User-Agent": "forecast-log/1.0 (+market news)" } });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const xml = await res.text();
        return { name, docs: parseRssXml(name, feedUrl, xml) };
      }),
    );

    for (let i = 0; i < results.length; i += 1) {
      const [name] = LIVE_NEWS_FEEDS[i];
      const r = results[i];
      if (r.status === "fulfilled") {
        if (r.value.docs.length) {
          providersOk.push(name);
          allDocs.push(...r.value.docs);
        }
      } else {
        errors.push(`${name}: ${String(r.reason?.message || r.reason)}`);
      }
    }

    const seenUrl = new Set();
    const seenId = new Set();
    const unique = [];
    allDocs.sort((a, b) => (b.published_at || "").localeCompare(a.published_at || ""));
    for (const doc of allDocs) {
      const k = doc.url.toLowerCase();
      if (seenUrl.has(k) || seenId.has(doc.id)) continue;
      seenUrl.add(k);
      seenId.add(doc.id);
      unique.push(doc);
      if (unique.length >= 100) break;
    }

    if (unique.length) {
      await db.run("DELETE FROM news WHERE as_of = ? AND doc_id = ''", [todayBj]);
      const chunkSize = db.dialect === "postgres" ? 50 : 8;
      for (let offset = 0; offset < unique.length; offset += chunkSize) {
        const slice = unique.slice(offset, offset + chunkSize);
        const tuples = slice.map(() => "(?, ?, ?, ?, ?, ?, ?, NULL, ?, '', ?)").join(", ");
        const params = [];
        for (const doc of slice) {
          params.push(todayBj, doc.id, doc.published_at, doc.title, doc.text, doc.url, doc.domain, doc.source, fetchedUtc);
        }
        await db.run(
          `INSERT INTO news (as_of, doc_id, published_at, title, text, url, domain, depth, source, page, fetched_at)
           VALUES ${tuples}
           ON CONFLICT (as_of, doc_id) DO UPDATE SET
             published_at = excluded.published_at,
             title = excluded.title,
             text = excluded.text,
             url = excluded.url,
             domain = excluded.domain,
             depth = excluded.depth,
             source = excluded.source,
             page = excluded.page,
             fetched_at = excluded.fetched_at`,
          params,
        );
      }
    }

    return loadNewsBundle(db, env, url, {
      market,
      limit,
      cached: false,
      fetchMeta: {
        refreshed_at: fetchedUtc,
        providers: providersOk,
        backfilled_days: [],
        new_count: unique.length,
        errors,
      },
    });
  } catch (err) {
    return json({ ok: false, error: String(err && err.message ? err.message : err) }, 500);
  }
}
