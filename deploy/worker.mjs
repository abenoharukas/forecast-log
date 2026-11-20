/**
 * Cloudflare Worker for forecast-log.
 *
 * Data model: the same five tables as the local `data/forecast.db`
 * (`bars`, `signals`, `news`, `agent_eval`, `chat_messages`). Nothing else lives in D1.
 *
 * What is read from D1 vs. served as an asset:
 *   - /api/cross/<market>  D1: signals + bars joined on (instrument, date). The per-day
 *                          series, hit-rate summaries and A-share bands are computed here.
 *                          `_meta/cross-<market>.json` only supplies the static bits
 *                          (question text, context switch definitions, provenance).
 *   - /api/agent/<dataset> D1: the flat `agent_eval` table. Brier / log-loss / ECE are
 *                          recomputed here, exactly like the local Python server.
 *   - /api/timeline        Asset: the VXN analogue-model fan charts are ~900 lines of
 *                          numpy and are built once at deploy time by
 *                          `scripts/build_d1_payloads.py`.
 *   - /api/state, /api/chat/*, /api/ingest, /api/refresh use the five tables directly.
 */

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
  async fetch(request, env) {
    const url = new URL(request.url);
    const clean = url.pathname.replace(/\/+$/, "") || "/";

    const api = clean.match(/^\/api\/cross\/([\w-]+)$/);
    if (api) {
      const market = MARKET_BY_SLUG[api[1]];
      if (!market) return json({ ok: false, error: "unknown market" }, 404);
      const doc = await loadMarket(env, url, market);
      if (!doc) return json({ ok: false, error: `no rows for ${market}; run make cf` }, 404);
      return json(doc);
    }

    if (clean === "/api/timeline") {
      const res = await env.ASSETS.fetch(new Request(new URL("/_meta/timeline.json", url)));
      if (!res.ok) return json({ ok: false, error: "timeline payload missing; run make cf" }, 404);
      return new Response(res.body, {
        status: 200,
        headers: { "Content-Type": "application/json; charset=utf-8", "Cache-Control": "no-store" },
      });
    }

    const agentApi = clean.match(/^\/api\/agent\/([a-z0-9_-]+)$/);
    if (agentApi) {
      const doc = await loadAgentEval(env, agentApi[1]);
      if (!doc) return json({ ok: false, error: `no agent rows for ${agentApi[1]}; run make cf` }, 404);
      return json(doc);
    }

    if (clean === "/api/state") return getD1State(env);

    if (clean === "/api/chat/threads") return chatThreads(env, url);

    if (clean === "/api/chat/messages") return chatMessages(env, url);

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

async function assetJson(env, url, path) {
  const res = await env.ASSETS.fetch(new Request(new URL(path, url)));
  if (!res.ok) return null;
  return res.json();
}

// ------------------------------------------------------------------ /api/cross

async function loadMarket(env, url, market) {
  const instrument = INSTRUMENT[market];
  const res = await env.DB.prepare(
    `SELECT s.*, b.open, b.high, b.low, b.close, b.volume,
            COALESCE(b.is_closed, 1) AS is_closed,
            COALESCE(NULLIF(s.fetched_at, ''), b.fetched_at, '') AS eff_fetched_at
     FROM signals s
     LEFT JOIN bars b ON b.symbol = ? AND b.date = s.date
     WHERE s.market = ?
     ORDER BY s.date ASC`,
  )
    .bind(instrument, market)
    .all();
  const rows = res.results ?? [];
  if (!rows.length) return null;

  const meta = (await assetJson(env, url, `/_meta/cross-${market}.json`)) || {};
  const ndxDoc = market === "ndx" ? null : await loadMarketRows(env, "ndx");

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
async function loadMarketRows(env, market) {
  const instrument = INSTRUMENT[market];
  const res = await env.DB.prepare(
    `SELECT s.*, b.open, b.high, b.low, b.close, b.volume,
            COALESCE(b.is_closed, 1) AS is_closed
     FROM signals s
     LEFT JOIN bars b ON b.symbol = ? AND b.date = s.date
     WHERE s.market = ?
     ORDER BY s.date ASC`,
  )
    .bind(instrument, market)
    .all();
  const rows = res.results ?? [];
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

async function loadAgentEval(env, dataset) {
  const res = await env.DB.prepare(
    "SELECT * FROM agent_eval WHERE dataset = ? ORDER BY as_of ASC, horizon ASC, question_id ASC",
  )
    .bind(dataset)
    .all();
  const rows = res.results ?? [];
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

async function getD1State(env) {
  try {
    const [marketRes, barRes, newsRes, agentRes, chatRes] = await env.DB.batch([
      env.DB.prepare(
        `SELECT s.market,
                COUNT(*) AS cnt,
                MAX(s.date) AS max_date,
                MIN(CASE WHEN s.outcome IS NULL THEN s.date END) AS first_unresolved,
                MAX(s.fetched_at) AS fetched_at,
                SUM(CASE WHEN COALESCE(b.is_closed, 1) = 0 THEN 1 ELSE 0 END) AS unclosed_count
         FROM signals s
         LEFT JOIN bars b ON b.symbol = CASE s.market WHEN 'ndx' THEN '^NDX' WHEN 'qqq' THEN 'QQQ' ELSE '000001.SS' END
                          AND b.date = s.date
         GROUP BY s.market`,
      ),
      env.DB.prepare("SELECT symbol, COUNT(*) AS cnt, MAX(date) AS max_date FROM bars GROUP BY symbol"),
      env.DB.prepare("SELECT COUNT(*) AS cnt, MAX(as_of) AS max_as_of FROM news"),
      env.DB.prepare(
        "SELECT dataset, COUNT(*) AS cnt, MAX(generated_at) AS generated_at FROM agent_eval GROUP BY dataset",
      ),
      env.DB.prepare("SELECT COUNT(*) AS cnt, MAX(created_at) AS max_created_at FROM chat_messages"),
    ]);

    const markets = {};
    for (const row of marketRes.results ?? []) {
      markets[row.market] = {
        count: row.cnt,
        max_date: row.max_date,
        first_unresolved: row.first_unresolved,
        fetched_at: row.fetched_at,
        unclosed_count: row.unclosed_count ?? 0,
      };
    }
    const bars = {};
    for (const row of barRes.results ?? []) bars[row.symbol] = { count: row.cnt, max_date: row.max_date };
    const news = newsRes.results?.[0] ?? { cnt: 0, max_as_of: null };
    const agents = {};
    for (const row of agentRes.results ?? []) agents[row.dataset] = { count: row.cnt, generated_at: row.generated_at };
    const chat = chatRes.results?.[0] ?? { cnt: 0, max_created_at: null };

    return json({ ok: true, markets, bars, news, agents, chat });
  } catch (err) {
    return json({ ok: false, error: String(err && err.message ? err.message : err) }, 500);
  }
}

async function chatThreads(env, url) {
  const market = url.searchParams.get("market") || "ndx";
  try {
    const res = await env.DB.prepare(
      `SELECT thread_id, market, MAX(as_of) AS as_of, MIN(created_at) AS created_at,
              MAX(created_at) AS updated_at, COUNT(*) AS message_count
       FROM chat_messages WHERE market = ?
       GROUP BY thread_id, market ORDER BY updated_at DESC, thread_id DESC LIMIT 20`,
    )
      .bind(market)
      .all();
    const threads = res.results ?? [];
    let messages = [];
    if (threads.length) {
      const msg = await env.DB.prepare(
        "SELECT thread_id, seq, market, as_of, role, model, stance, content, created_at FROM chat_messages WHERE thread_id = ? ORDER BY seq ASC",
      )
        .bind(threads[0].thread_id)
        .all();
      messages = msg.results ?? [];
    }
    return json({
      ok: true,
      market,
      context: await chatContext(env, market),
      threads,
      active_thread_id: threads.length ? threads[0].thread_id : null,
      messages,
    });
  } catch (err) {
    return json({ ok: false, error: String(err && err.message ? err.message : err) }, 500);
  }
}

async function chatMessages(env, url) {
  const threadId = url.searchParams.get("thread_id") || "";
  if (!threadId) return json({ ok: true, thread_id: "", messages: [] });
  const res = await env.DB.prepare(
    "SELECT thread_id, seq, market, as_of, role, model, stance, content, created_at FROM chat_messages WHERE thread_id = ? ORDER BY seq ASC",
  )
    .bind(threadId)
    .all();
  return json({ ok: true, thread_id: threadId, messages: res.results ?? [] });
}

/** Short market snapshot for the group-chat side panel (the full Python builder is not portable). */
function fmtPrice(value) {
  return value === null || value === undefined ? "-" : Number(value).toFixed(2);
}

async function chatContext(env, market) {
  const instrument = INSTRUMENT[market] || INSTRUMENT.ndx;
  try {
    const [barRes, sigRes] = await env.DB.batch([
      env.DB.prepare("SELECT date, open, high, low, close, is_closed FROM bars WHERE symbol = ? ORDER BY date DESC LIMIT 2").bind(instrument),
      env.DB.prepare("SELECT date, flag, score, threshold FROM signals WHERE market = ? ORDER BY date DESC LIMIT 1").bind(market),
    ]);
    const bars = barRes.results ?? [];
    const signal = sigRes.results?.[0];
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
    return {
      market,
      symbol: instrument,
      as_of: latest.date,
      text: [
        `【标的】${instrument} · 基准日 ${latest.date}（${Number(latest.is_closed) ? "已收盘" : "盘中未收盘"}）`,
        `1. 当前盘面：收/现价 ${Number(latest.close).toFixed(2)}（当日 ${pct}），开 ${fmtPrice(latest.open)} / 高 ${fmtPrice(latest.high)} / 低 ${fmtPrice(latest.low)}`,
        signalLine ? `2. ${signalLine}` : "",
      ]
        .filter(Boolean)
        .join("\n"),
    };
  } catch (err) {
    return { market, symbol: instrument, as_of: "", text: `上下文读取失败：${err.message || err}` };
  }
}

// ------------------------------------------------------------------ ingest

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
    if (body.op === "state") return getD1State(env);
    if (body.op !== "sql") return json({ ok: false, error: "unknown op" }, 400);
    if (typeof body.sql !== "string") return json({ ok: false, error: "sql must be a string" }, 400);
    if (body.sql.length > 95_000) return json({ ok: false, error: "sql too large; send fewer rows per statement" }, 400);
    if (
      !/^\s*INSERT\s+(OR\s+(REPLACE|IGNORE)\s+)?INTO\s+(bars|signals|news|agent_eval|chat_messages)\b/i.test(body.sql)
    ) {
      return json({ ok: false, error: "INSERT only on bars/signals/news/agent_eval/chat_messages" }, 400);
    }
    const result = await env.DB.exec(body.sql);
    return json({ ok: true, op: "sql", count: result.count ?? null });
  } catch (error) {
    return json({ ok: false, error: String(error && error.message ? error.message : error) }, 400);
  }
}

// ------------------------------------------------------------------ /api/refresh

function beijingDateStr(date = new Date()) {
  return new Intl.DateTimeFormat("en-CA", {
    timeZone: "Asia/Shanghai",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).format(date);
}

function isSessionClosed(sessionDate, tzName = "America/New_York", now = new Date()) {
  const isChina = tzName === "Asia/Shanghai";
  const targetTz = isChina ? "Asia/Shanghai" : "America/New_York";
  const closeHm = isChina ? 15 * 60 + 5 : 16 * 60 + 15;
  const parts = new Intl.DateTimeFormat("en-CA", {
    timeZone: targetTz,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).formatToParts(now);
  const get = (type) => parts.find((part) => part.type === type)?.value || "00";
  const localDate = `${get("year")}-${get("month")}-${get("day")}`;
  if (sessionDate < localDate) return 1;
  if (sessionDate > localDate) return 0;
  const mins = Number(get("hour")) * 60 + Number(get("minute"));
  return mins >= closeHm ? 1 : 0;
}

async function fetchYahooBars(symbol, startUnix, endUnix) {
  const url = `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(symbol)}?interval=1d&period1=${startUnix}&period2=${endUnix}`;
  const res = await fetch(url, { headers: { "User-Agent": "Mozilla/5.0 forecast-log/1.0" } });
  if (!res.ok) throw new Error(`Yahoo ${symbol} HTTP ${res.status}`);
  const payload = await res.json();
  const result = payload?.chart?.result?.[0];
  if (!result) return [];
  const tz =
    result?.meta?.exchangeTimezoneName ||
    (symbol.endsWith(".SS") || symbol.endsWith(".SZ") ? "Asia/Shanghai" : "America/New_York");
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
    const close = quote.close?.[i];
    if (close === null || close === undefined) continue;
    const day = fmt.format(new Date(stamps[i] * 1000));
    byDate.set(day, {
      date: day,
      // Same precision as forecast_log.yahoo: bars store 4 decimals; the API
      // rounds to 2 when it builds the chart series.
      open: quote.open?.[i] != null ? rnd(quote.open[i], 4) : null,
      high: quote.high?.[i] != null ? rnd(quote.high[i], 4) : null,
      low: quote.low?.[i] != null ? rnd(quote.low[i], 4) : null,
      close: rnd(close, 4),
      volume: quote.volume?.[i] != null ? Math.trunc(quote.volume[i]) : null,
      is_closed: isSessionClosed(day, tz, now),
    });
  }
  return [...byDate.values()].sort((a, b) => a.date.localeCompare(b.date));
}

async function refreshFromYahoo(env, force = false) {
  try {
    const todayBj = beijingDateStr();
    const symbols = (await env.DB.prepare("SELECT DISTINCT symbol FROM bars ORDER BY symbol").all()).results ?? [];
    if (!symbols.length) return json({ ok: false, error: "D1 尚未初始化，请先运行 make cf" }, 404);

    // 盘中 5 分钟冷却：页面每次加载都会 non-force 调一次，不能每次都打 Yahoo。
    if (!force) {
      const freshCut = new Date(Date.now() - 5 * 60 * 1000).toISOString();
      const unclosed = (await env.DB.prepare("SELECT symbol, date, fetched_at FROM bars WHERE is_closed = 0").all())
        .results ?? [];
      const stale = unclosed.filter((row) => !row.fetched_at || row.fetched_at < freshCut);
      const markets = (await env.DB.prepare("SELECT market, MAX(date) AS last_session, MAX(fetched_at) AS fetched_at FROM signals GROUP BY market").all())
        .results ?? [];
      const allFresh = markets.length > 0 && markets.every((row) => (row.fetched_at || "").slice(0, 10) >= todayBj);
      if (stale.length === 0 && allFresh) {
        return json({
          ok: true,
          cached: true,
          beijing_date: todayBj,
          sessions: Object.fromEntries(markets.map((row) => [row.market, row.last_session])),
        });
      }
    }

    const updated = {};
    const errors = {};
    const nowSec = Math.floor(Date.now() / 1000) + 172800;

    for (const { symbol } of symbols) {
      try {
        const tail = (
          await env.DB.prepare("SELECT date FROM bars WHERE symbol = ? ORDER BY date DESC LIMIT 1")
            .bind(symbol)
            .all()
        ).results ?? [];
        if (!tail.length) continue;
        const startUnix = Math.floor(new Date(`${tail[0].date}T00:00:00Z`).getTime() / 1000) - 86400 * 10;
        const bars = await fetchYahooBars(symbol, startUnix, nowSec);
        if (!bars.length) continue;
        const statements = bars.map((bar) =>
          env.DB.prepare(
            `INSERT OR REPLACE INTO bars (symbol, date, open, high, low, close, volume, is_closed, fetched_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
          ).bind(
            symbol,
            bar.date,
            bar.open,
            bar.high,
            bar.low,
            bar.close,
            bar.volume,
            bar.is_closed,
            new Date().toISOString(),
          ),
        );
        await env.DB.batch(statements);
        updated[symbol] = bars[bars.length - 1].date;
      } catch (err) {
        errors[symbol] = String(err && err.message ? err.message : err);
      }
    }

    const settled = await settleSignals(env);
    const sessions = (await env.DB.prepare("SELECT market, MAX(date) AS last_session FROM signals GROUP BY market").all())
      .results ?? [];
    return json({
      ok: true,
      refreshed_at: new Date().toISOString(),
      beijing_date: beijingDateStr(),
      force,
      sessions: Object.fromEntries(sessions.map((row) => [row.market, row.last_session])),
      updated,
      settled,
      errors,
      note: "线上只增量抓取 bars 并结算已存在信号；新交易日的信号仍需本地 make refresh 后 publish。",
    });
  } catch (err) {
    return json({ ok: false, error: String(err && err.message ? err.message : err) }, 500);
  }
}

/** Fill close3/close5/hit3/outcome/entry/peak for existing signal rows from the bars table. */
async function settleSignals(env) {
  const settled = {};
  for (const market of Object.keys(INSTRUMENT)) {
    const instrument = INSTRUMENT[market];
    const bars = (await env.DB.prepare("SELECT date, open, close FROM bars WHERE symbol = ? ORDER BY date ASC").bind(instrument).all())
      .results ?? [];
    if (!bars.length) continue;
    const index = new Map(bars.map((bar, i) => [bar.date, i]));
    const tail = (
      await env.DB.prepare("SELECT * FROM signals WHERE market = ? AND outcome IS NULL ORDER BY date DESC LIMIT 60")
        .bind(market)
        .all()
    ).results ?? [];
    let changed = 0;
    for (const row of tail) {
      const i = index.get(row.date);
      if (i === undefined) continue;
      const baseClose = bars[i].close;
      if (baseClose === null) continue;
      const H = HORIZON;
      if (market === "ssec") {
        if (i + H >= bars.length) continue;
        const window = bars.slice(i + 1, i + 1 + H);
        let peak = null;
        let peakAt = null;
        for (const bar of window) {
          if (bar.close === null) continue;
          if (peak === null || bar.close > peak) {
            peak = bar.close;
            peakAt = bar.date;
          }
        }
        const close5 = bars[i + H].close;
        const outcome = close5 > baseClose ? 1 : 0;
        const close3 = bars[i + 3]?.close ?? null;
        const flagged = Number(row.flag || 0) ? 1 : 0;
        await env.DB.prepare(
          `UPDATE signals SET outcome = ?, peak = ?, peak_at = ?,
                              base_close = CASE WHEN ? THEN ? ELSE base_close END,
                              close3 = CASE WHEN ? THEN ? ELSE close3 END,
                              close5 = CASE WHEN ? THEN ? ELSE close5 END,
                              hit3 = CASE WHEN ? THEN ? ELSE hit3 END
           WHERE market = ? AND date = ?`,
        )
          .bind(
            outcome,
            rnd(peak, 2),
            peakAt,
            flagged,
            rnd(baseClose, 2),
            flagged,
            rnd(close3, 2),
            flagged,
            rnd(close5, 2),
            flagged,
            close3 !== null && close3 > baseClose ? 1 : 0,
            market,
            row.date,
          )
          .run();
        changed += 1;
      } else {
        if (i + H + 1 >= bars.length) continue;
        const next = bars[i + 1];
        const window = bars.slice(i + 2, i + 2 + H);
        if (next.open === null || window.some((bar) => bar.close === null)) continue;
        let peak = null;
        let peakAt = null;
        for (const bar of window) {
          if (peak === null || bar.close > peak) {
            peak = bar.close;
            peakAt = bar.date;
          }
        }
        const close3 = bars[i + 3]?.close ?? null;
        const close5 = bars[i + H].close;
        await env.DB.prepare(
          `UPDATE signals SET entry = ?, entry_at = ?, base_close = ?, close3 = ?, close5 = ?, hit3 = ?,
                              outcome = ?, target = ?, peak = ?, peak_at = ?
           WHERE market = ? AND date = ?`,
        )
          .bind(
            rnd(next.open, 2),
            next.date,
            rnd(baseClose, 2),
            rnd(close3, 2),
            rnd(close5, 2),
            close3 !== null && close3 > baseClose ? 1 : 0,
            close5 > baseClose ? 1 : 0,
            bars[i + H].date,
            rnd(peak, 2),
            peakAt,
            market,
            row.date,
          )
          .run();
        changed += 1;
      }
    }
    settled[market] = changed;
  }
  return settled;
}
