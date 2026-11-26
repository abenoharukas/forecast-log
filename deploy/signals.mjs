/**
 * Cross-asset & A-share signal recomputation for Cloudflare Worker.
 *
 * Mirrors Python's `forecast_log/cross.py`, `forecast_log/china.py`,
 * `forecast_log/stats.py`, and `forecast_log/artifacts_db.py::save_cross_market`
 * so `/api/refresh` on the edge can generate new trading-day `signals` rows
 * directly from `bars`, `_meta/events.json`, and `news` without waiting for a
 * local `make refresh && make supabase-sync`.
 */

const HORIZON = 5;
const QUANTILE = 0.9;
const SELL_QUANTILE = 0.1;
const MIN_HISTORY = 252;
const EXPANDING_Z_MIN_ROWS = 30;
const CPI_WINDOW = 3;
const NFP_LOW = 3;
const NFP_HIGH = 7;

const FACTORS = [
  ["^TNX", "diff20", -1.0],
  ["DX-Y.NYB", "ret20", -1.0],
  ["HYG/TLT", "ret20", 1.0],
  ["^VIX", "offset", 1.0],
];

const DEFAULT_US_CONTEXTS = [
  "composite",
  "cal_cpi",
  "cal_nfp",
  "fomc_day",
  "news_bearish",
  "news_bullish",
  "riskoff_hits",
];

const SIGNAL_WRITE_COLS = [
  "market",
  "date",
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
  "factors_packed",
  "fetched_at",
];

const COMPARE_COLS = [
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
  "factors_packed",
];

function lowerBound(arr, target) {
  let lo = 0;
  let hi = arr.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (arr[mid] < target) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

function upperBound(arr, target) {
  let lo = 0;
  let hi = arr.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (arr[mid] <= target) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

function insertSorted(arr, val) {
  let lo = 0;
  let hi = arr.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (arr[mid] < val) lo = mid + 1;
    else hi = mid;
  }
  arr.splice(lo, 0, val);
}

function removeSorted(arr, val) {
  let lo = 0;
  let hi = arr.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (arr[mid] < val) lo = mid + 1;
    else hi = mid;
  }
  if (lo < arr.length && arr[lo] === val) arr.splice(lo, 1);
}

function quantileSorted(sorted, q) {
  const n = sorted.length;
  const pos = (n - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (pos - lo) * (sorted[hi] - sorted[lo]);
}

function unpackSlot(packed, slot) {
  return Math.floor(Number(packed || 0) / 2 ** (4 * slot)) % 16;
}

function packFactors(orderedKeys, valuesByKey) {
  let packed = 0;
  for (let slot = 0; slot < orderedKeys.length; slot += 1) {
    const raw = valuesByKey[orderedKeys[slot]] || 0;
    const clamped = Math.max(0, Math.min(15, Math.trunc(raw)));
    if (clamped) packed += clamped * 2 ** (4 * slot);
  }
  return packed;
}

function alignSeries(dates, bars) {
  const n = dates.length;
  const out = new Float64Array(n);
  out.fill(NaN);
  let ptr = 0;
  let last = NaN;
  for (let i = 0; i < n; i += 1) {
    const day = dates[i];
    while (ptr < bars.length && bars[ptr].date <= day) {
      last = Number(bars[ptr].close);
      ptr += 1;
    }
    out[i] = last;
  }
  return out;
}

function factorPrices(factorBars, symbol, dates) {
  if (symbol.includes("/")) {
    const [left, right] = symbol.split("/");
    const a = alignSeries(dates, factorBars.get(left) || []);
    const b = alignSeries(dates, factorBars.get(right) || []);
    const out = new Float64Array(dates.length);
    for (let i = 0; i < dates.length; i += 1) out[i] = a[i] / b[i];
    return out;
  }
  return alignSeries(dates, factorBars.get(symbol) || []);
}

function factorLeg(values, how) {
  const n = values.length;
  const out = new Float64Array(n);
  out.fill(NaN);
  if (how === "ret20") {
    for (let i = 20; i < n; i += 1) {
      out[i] = Math.log(values[i]) - Math.log(values[i - 20]);
    }
  } else if (how === "diff20") {
    for (let i = 20; i < n; i += 1) {
      out[i] = values[i] - values[i - 20];
    }
  } else if (how === "offset") {
    if (n <= 252) return out;
    // Matches Python: for i in range(252, n): window = values[i - 251 : i + 1] (252 elems)
    const sortedWin = [];
    let nanCount = 0;
    for (let k = 1; k <= 252; k += 1) {
      const v = values[k];
      if (Number.isFinite(v)) insertSorted(sortedWin, v);
      else nanCount += 1;
    }
    if (nanCount === 0) {
      const med = (sortedWin[125] + sortedWin[126]) * 0.5;
      out[252] = Math.log(values[252] / med);
    }
    for (let i = 253; i < n; i += 1) {
      const oldVal = values[i - 252];
      if (Number.isFinite(oldVal)) removeSorted(sortedWin, oldVal);
      else nanCount -= 1;
      const newVal = values[i];
      if (Number.isFinite(newVal)) insertSorted(sortedWin, newVal);
      else nanCount += 1;
      if (nanCount === 0) {
        const med = (sortedWin[125] + sortedWin[126]) * 0.5;
        out[i] = Math.log(newVal / med);
      }
    }
  } else {
    throw new Error(`unknown leg method: ${how}`);
  }
  return out;
}

/** O(N) Welford expanding z-score matching numpy population std (ddof=0). */
function expandingZ(values) {
  const n = values.length;
  const out = new Float64Array(n);
  out.fill(NaN);
  let count = 0;
  let mean = 0;
  let m2 = 0;
  for (let i = 0; i < n; i += 1) {
    const v = values[i];
    if (!Number.isFinite(v)) continue;
    count += 1;
    const delta = v - mean;
    mean += delta / count;
    const delta2 = v - mean;
    m2 += delta * delta2;
    if (count < EXPANDING_Z_MIN_ROWS) continue;
    const spread = Math.sqrt(m2 / count);
    if (spread <= 0) continue;
    out[i] = (v - mean) / spread;
  }
  return out;
}

export function computeComposite(dates, factorBars) {
  const n = dates.length;
  const parts = FACTORS.map(([symbol, how, sign]) => {
    const raw = factorLeg(factorPrices(factorBars, symbol, dates), how);
    const z = expandingZ(raw);
    if (sign !== 1.0) {
      for (let i = 0; i < n; i += 1) z[i] *= sign;
    }
    return z;
  });
  const out = new Float64Array(n);
  for (let i = 0; i < n; i += 1) {
    const a = parts[0][i];
    const b = parts[1][i];
    const c = parts[2][i];
    const d = parts[3][i];
    if (Number.isNaN(a) || Number.isNaN(b) || Number.isNaN(c) || Number.isNaN(d)) {
      out[i] = NaN;
    } else {
      out[i] = (a + b + c + d) * 0.25;
    }
  }
  return out;
}

function isoToEpochDay(iso) {
  return Math.floor(Date.parse(`${iso}T00:00:00Z`) / 86400000);
}

function calendarDistances(calendar, dates) {
  if (!calendar || typeof calendar !== "object") return null;
  const dayEpochs = dates.map(isoToEpochDay);
  const out = {};
  for (const key of ["fomc", "cpi", "nfp"]) {
    const list = Array.isArray(calendar[key]) ? [...calendar[key]].sort() : [];
    const evEpochs = list.map(isoToEpochDay);
    const dists = new Float64Array(dates.length);
    let ptr = 0;
    for (let i = 0; i < dates.length; i += 1) {
      const d = dayEpochs[i];
      while (ptr < evEpochs.length && evEpochs[ptr] < d) ptr += 1;
      dists[i] = ptr < evEpochs.length ? evEpochs[ptr] - d : NaN;
    }
    out[key] = dists;
  }
  return out;
}

function buildRecentNewsLabelMap(newsIndicators) {
  const map = new Map();
  const list = Array.isArray(newsIndicators?.recent_days) ? newsIndicators.recent_days : [];
  for (const item of list) {
    if (!item || !item.as_of) continue;
    map.set(item.as_of, {
      bearish: item.deepseek?.veto ? 1 : 0,
      bullish: item.deepseek?.boost ? 1 : 0,
      qwenVeto: item.qwen?.veto ? 1 : 0,
    });
  }
  return map;
}

/**
 * Pure computation of NDX or QQQ signals (mirrors `cross.py::calls` + `document`).
 */
export function computeUsMarket({
  market,
  ndxBars,
  targetBars,
  scores,
  calendar,
  contextKeys,
  existingByDate,
  recentNewsLabels,
  riskoffByDate,
  fetchedAt,
  rnd,
}) {
  const dates = ndxBars.map((b) => b.date);
  const n = dates.length;
  const opens = new Float64Array(n);
  const closes = new Float64Array(n);

  if (!targetBars) {
    for (let i = 0; i < n; i += 1) {
      opens[i] = ndxBars[i].open != null ? Number(ndxBars[i].open) : NaN;
      closes[i] = Number(ndxBars[i].close);
    }
  } else {
    const byDate = new Map(targetBars.map((b) => [b.date, b]));
    for (let i = 0; i < n; i += 1) {
      const b = byDate.get(dates[i]);
      opens[i] = b && b.open != null ? Number(b.open) : NaN;
      closes[i] = b && b.close != null ? Number(b.close) : NaN;
    }
  }

  // Settlement outcomes (exact match to cross.py lines 425-437)
  const entryArr = new Float64Array(n).fill(NaN);
  const entryAtArr = new Array(n).fill(null);
  const baseCloseArr = new Float64Array(n).fill(NaN);
  const close3Arr = new Float64Array(n).fill(NaN);
  const close5Arr = new Float64Array(n).fill(NaN);
  const peakArr = new Float64Array(n).fill(NaN);
  const peakAtArr = new Array(n).fill(null);

  for (let i = 0; i < n - HORIZON - 1; i += 1) {
    const j = i + 1;
    if (!Number.isFinite(opens[j])) continue;
    let allFinite = true;
    let maxVal = -Infinity;
    let maxOffset = 0;
    for (let k = 0; k < HORIZON; k += 1) {
      const c = closes[j + 1 + k];
      if (!Number.isFinite(c)) {
        allFinite = false;
        break;
      }
      if (c > maxVal) {
        maxVal = c;
        maxOffset = k;
      }
    }
    if (!allFinite) continue;
    entryArr[i] = opens[j];
    entryAtArr[i] = dates[j];
    baseCloseArr[i] = closes[i];
    close3Arr[i] = closes[i + 3];
    close5Arr[i] = closes[i + 5];
    peakArr[i] = maxVal;
    peakAtArr[i] = dates[j + 1 + maxOffset];
  }

  const calDists = calendarDistances(calendar, dates);

  const orderedKeys = [...(contextKeys && contextKeys.length ? contextKeys : DEFAULT_US_CONTEXTS)];
  if (targetBars) {
    if (!orderedKeys.includes("qqq_below_ma20")) orderedKeys.push("qqq_below_ma20");
    if (!orderedKeys.includes("qqq_gap_veto")) orderedKeys.push("qqq_gap_veto");
  }
  if (!orderedKeys.includes("sell_composite")) orderedKeys.push("sell_composite");

  const slotOf = new Map(orderedKeys.map((k, idx) => [k, idx]));
  const extraNewsKeys = orderedKeys.filter(
    (k) => k.startsWith("news_") && k !== "news_bearish" && k !== "news_bullish",
  );

  const sortedHistory = [];
  const rawRows = [];

  for (let i = 0; i < n; i += 1) {
    const sc = scores[i];
    if (!Number.isFinite(sc)) continue;
    insertSorted(sortedHistory, sc);
    if (i < MIN_HISTORY || sortedHistory.length < MIN_HISTORY) continue;

    const day = dates[i];
    const threshold = quantileSorted(sortedHistory, QUANTILE);
    const sellThreshold = quantileSorted(sortedHistory, SELL_QUANTILE);
    const middle = quantileSorted(sortedHistory, 0.5);

    const compositeFlag = sc >= threshold ? 1 : 0;
    const sellCompositeFlag = sc <= sellThreshold ? 1 : 0;

    let calCpi = 0;
    let calNfp = 0;
    let fomcDay = 0;
    if (calDists) {
      const aboveMiddle = sc >= middle;
      calCpi = aboveMiddle && calDists.cpi[i] <= CPI_WINDOW ? 1 : 0;
      calNfp = aboveMiddle && calDists.nfp[i] >= NFP_LOW && calDists.nfp[i] <= NFP_HIGH ? 1 : 0;
      fomcDay = calDists.fomc[i] === 0 ? 1 : 0;
    }

    const prev = existingByDate.get(day);
    const recent = recentNewsLabels.get(day);
    let bearish = 0;
    let bullish = 0;
    if (prev) {
      if (slotOf.has("news_bearish")) bearish = unpackSlot(prev.factors_packed, slotOf.get("news_bearish"));
      if (slotOf.has("news_bullish")) bullish = unpackSlot(prev.factors_packed, slotOf.get("news_bullish"));
    } else if (recent) {
      bearish = recent.bearish;
      bullish = recent.bullish;
    }

    let riskoffHits = 0;
    if (riskoffByDate.has(day)) {
      riskoffHits = riskoffByDate.get(day);
    } else if (prev && slotOf.has("riskoff_hits")) {
      riskoffHits = unpackSlot(prev.factors_packed, slotOf.get("riskoff_hits"));
    }

    const extraNewsVals = {};
    for (const nk of extraNewsKeys) {
      if (prev) {
        extraNewsVals[nk] = unpackSlot(prev.factors_packed, slotOf.get(nk));
      } else if (recent && nk.startsWith("news_qwen")) {
        extraNewsVals[nk] = recent.qwenVeto;
      } else {
        extraNewsVals[nk] = 0;
      }
    }

    const priceOnly = (compositeFlag || calCpi || calNfp) && !fomcDay ? 1 : 0;
    const flag = priceOnly && !bearish ? 1 : 0;
    const sellFlag = sellCompositeFlag && !fomcDay && !bullish ? 1 : 0;
    const newsVeto = priceOnly && !flag ? 1 : 0;

    const resolved = Number.isFinite(peakArr[i]) && Number.isFinite(entryArr[i]);
    const rawEntry = resolved ? entryArr[i] : null;
    const rawClose = closes[i];

    rawRows.push({
      date: day,
      rawClose,
      rawEntry,
      entry: resolved ? rnd(entryArr[i], 2) : null,
      entry_at: entryAtArr[i],
      base_close: resolved ? rnd(baseCloseArr[i], 2) : null,
      close3: resolved ? rnd(close3Arr[i], 2) : null,
      close5: resolved ? rnd(close5Arr[i], 2) : null,
      hit3: resolved ? (close3Arr[i] > baseCloseArr[i] ? 1 : 0) : null,
      flag,
      flag_price_only: priceOnly,
      sell_flag: sellFlag,
      sell_threshold: rnd(sellThreshold, 4),
      news_veto: newsVeto,
      score: rnd(sc, 4),
      threshold: rnd(threshold, 4),
      outcome: resolved ? (close5Arr[i] > baseCloseArr[i] ? 1 : 0) : null,
      target: resolved ? dates[i + HORIZON] : null,
      peak: resolved ? rnd(peakArr[i], 2) : null,
      peak_at: peakAtArr[i],
      us_date: null,
      comp: {
        composite: compositeFlag,
        sell_composite: sellCompositeFlag,
        cal_cpi: calCpi,
        cal_nfp: calNfp,
        fomc_day: fomcDay,
        news_bearish: bearish,
        news_bullish: bullish,
        riskoff_hits: riskoffHits,
        ...extraNewsVals,
      },
    });
  }

  // QQQ extra filters (exact match to cross.py lines 657-672)
  if (targetBars) {
    for (let idx = 0; idx < rawRows.length; idx += 1) {
      let belowMa20 = 0;
      if (idx >= 19) {
        let sum20 = 0;
        let ok = true;
        for (let k = idx - 19; k <= idx; k += 1) {
          const c = rawRows[k].rawClose;
          if (!Number.isFinite(c)) {
            ok = false;
            break;
          }
          sum20 += c;
        }
        if (ok && rawRows[idx].rawClose < sum20 / 20) belowMa20 = 1;
      }
      let gapVeto = 0;
      const r = rawRows[idx];
      if (r.rawEntry !== null && Number.isFinite(r.rawClose) && r.rawClose !== 0) {
        if (r.rawEntry / r.rawClose - 1.0 > 0.015) gapVeto = 1;
      }
      r.comp.qqq_below_ma20 = belowMa20;
      r.comp.qqq_gap_veto = gapVeto;
    }
  }

  const rows = rawRows.map((r) => ({
    market,
    date: r.date,
    entry: r.entry,
    entry_at: r.entry_at,
    base_close: r.base_close,
    close3: r.close3,
    close5: r.close5,
    hit3: r.hit3,
    flag: r.flag,
    flag_price_only: r.flag_price_only,
    sell_flag: r.sell_flag,
    sell_threshold: r.sell_threshold,
    news_veto: r.news_veto,
    score: r.score,
    threshold: r.threshold,
    outcome: r.outcome,
    target: r.target,
    peak: r.peak,
    peak_at: r.peak_at,
    us_date: null,
    factors_packed: packFactors(orderedKeys, r.comp),
    fetched_at: fetchedAt,
  }));

  return { rows, rawRows, orderedKeys };
}

/**
 * Pure computation of A-share (`ssec`) signals mapped from `ndx` (mirrors `china.py::document`).
 */
export function computeSsecMarket({ ndxResult, ssecBars, contextKeys, fetchedAt, rnd }) {
  const ndxRaw = ndxResult.rawRows;
  if (!ndxRaw.length || !ssecBars.length) return { rows: [] };

  const ndxDates = ndxRaw.map((r) => r.date);
  const dates = ssecBars.map((b) => b.date);
  const n = dates.length;
  const opens = new Float64Array(n);
  const closes = new Float64Array(n);
  for (let i = 0; i < n; i += 1) {
    opens[i] = ssecBars[i].open != null ? Number(ssecBars[i].open) : NaN;
    closes[i] = Number(ssecBars[i].close);
  }

  const peakArr = new Float64Array(n).fill(NaN);
  const peakAtArr = new Array(n).fill(null);
  const baseCloseArr = new Float64Array(n).fill(NaN);
  const close3Arr = new Float64Array(n).fill(NaN);
  const close5Arr = new Float64Array(n).fill(NaN);

  for (let i = 0; i < n - HORIZON; i += 1) {
    if (!Number.isFinite(opens[i])) continue;
    let allFinite = true;
    let maxVal = -Infinity;
    let maxOffset = 0;
    for (let k = 0; k < HORIZON; k += 1) {
      const c = closes[i + 1 + k];
      if (!Number.isFinite(c)) {
        allFinite = false;
        break;
      }
      if (c > maxVal) {
        maxVal = c;
        maxOffset = k;
      }
    }
    if (!allFinite) continue;
    peakArr[i] = maxVal;
    peakAtArr[i] = dates[i + 1 + maxOffset];
    baseCloseArr[i] = closes[i];
    close3Arr[i] = closes[i + 3];
    close5Arr[i] = closes[i + 5];
  }

  function mapEntries(flagProp) {
    const entries = [];
    let lastEntry = -1;
    for (let u = 0; u < ndxRaw.length; u += 1) {
      if (!ndxRaw[u][flagProp]) continue;
      const usDay = ndxDates[u];
      const idx = upperBound(dates, usDay);
      if (idx >= n || idx <= lastEntry) continue;
      lastEntry = idx;
      entries.push({ idx, us_date: usDay });
    }
    return entries;
  }

  const buyEntries = mapEntries("flag");
  const sellEntries = mapEntries("sell_flag");
  const allFlagged = new Set(buyEntries.map((e) => e.idx));
  const allSellFlagged = new Set(sellEntries.map((e) => e.idx));
  const signalOf = new Map();
  for (const e of sellEntries) signalOf.set(e.idx, e.us_date);
  for (const e of buyEntries) signalOf.set(e.idx, e.us_date);

  const first = lowerBound(dates, ndxDates[0]);
  const kept = [];
  for (let i = first; i < n; i += 1) kept.push(i);

  const ndxCompKeys = Object.keys(ndxRaw[0].comp);
  const orderedKeys = [...(contextKeys && contextKeys.length ? contextKeys : [])];
  if (!orderedKeys.length) {
    for (const k of ndxResult.orderedKeys) {
      if (k !== "sell_composite") orderedKeys.push(k);
    }
    orderedKeys.push("ssec_below_ma20", "ssec_gap_veto", "ssec_oversold_boost");
  }
  if (!orderedKeys.includes("sell_composite")) orderedKeys.push("sell_composite");

  const rows = [];
  for (let pos = 0; pos < kept.length; pos += 1) {
    const i = kept[pos];
    const prevDate = pos > 0 ? dates[kept[pos - 1]] : null;
    const high = lowerBound(ndxDates, dates[i]);
    const low = prevDate === null ? Math.max(0, high - 1) : lowerBound(ndxDates, prevDate);

    const compVals = {};
    if (high <= low) {
      for (const k of ndxCompKeys) compVals[k] = 0;
    } else {
      let chosen = high - 1;
      for (let j = high - 1; j >= low; j -= 1) {
        const c = ndxRaw[j].comp;
        if (c.composite || c.cal_cpi || c.cal_nfp || c.sell_composite) {
          chosen = j;
          break;
        }
      }
      const chosenComp = ndxRaw[chosen].comp;
      for (const k of ndxCompKeys) compVals[k] = chosenComp[k] || 0;
    }

    // Local A-share context filters (exact match to china.py lines 328-355)
    let belowMa20 = 0;
    if (i >= 19) {
      let sum20 = 0;
      let ok = true;
      for (let k = i - 19; k <= i; k += 1) {
        if (!Number.isFinite(closes[k])) {
          ok = false;
          break;
        }
        sum20 += closes[k];
      }
      if (ok && closes[i] < sum20 / 20) belowMa20 = 1;
    }
    const gapVeto =
      i > 0 && Number.isFinite(opens[i]) && Number.isFinite(closes[i - 1]) && closes[i - 1] > 0 && opens[i] / closes[i - 1] - 1.0 > 0.01
        ? 1
        : 0;
    const oversoldBoost =
      i >= 5 && Number.isFinite(closes[i]) && Number.isFinite(closes[i - 5]) && closes[i - 5] > 0 && closes[i] / closes[i - 5] - 1.0 < -0.03
        ? 1
        : 0;

    compVals.ssec_below_ma20 = belowMa20;
    compVals.ssec_gap_veto = gapVeto;
    compVals.ssec_oversold_boost = oversoldBoost;

    const flagged = allFlagged.has(i) ? 1 : 0;
    const sellFlagged = allSellFlagged.has(i) ? 1 : 0;
    const resolved = Number.isFinite(peakArr[i]);
    let usDate = signalOf.has(i) ? signalOf.get(i) : null;
    // Matches china.py `latest.us_date` + artifacts_db.py `build_cross_bundle` fallback on latest row
    if (usDate === null && pos === kept.length - 1 && high > 0) {
      usDate = ndxDates[high - 1];
    }

    rows.push({
      market: "ssec",
      date: dates[i],
      entry: (flagged || sellFlagged) && Number.isFinite(opens[i]) ? rnd(opens[i], 2) : null,
      entry_at: null,
      base_close: resolved ? rnd(baseCloseArr[i], 2) : null,
      close3: resolved ? rnd(close3Arr[i], 2) : null,
      close5: resolved ? rnd(close5Arr[i], 2) : null,
      hit3: resolved ? (close3Arr[i] > baseCloseArr[i] ? 1 : 0) : null,
      flag: flagged,
      flag_price_only: null,
      sell_flag: sellFlagged,
      sell_threshold: null,
      news_veto: null,
      score: null,
      threshold: null,
      outcome: resolved ? (close5Arr[i] > baseCloseArr[i] ? 1 : 0) : null,
      target: null,
      peak: resolved ? rnd(peakArr[i], 2) : null,
      peak_at: peakAtArr[i],
      us_date: usDate,
      factors_packed: packFactors(orderedKeys, compVals),
      fetched_at: fetchedAt,
    });
  }

  return { rows, orderedKeys };
}

function sameVal(a, b) {
  if (a === b) return true;
  if (a === null || a === undefined || b === null || b === undefined) return a == null && b == null;
  if (typeof a === "number" || typeof b === "number") return Number(a) === Number(b);
  return String(a) === String(b);
}

function rowChanged(prev, next) {
  if (!prev) return true;
  for (const col of COMPARE_COLS) {
    if (!sameVal(prev[col], next[col])) return true;
  }
  return false;
}

async function upsertSignalRows(db, rows) {
  if (!rows.length) return 0;
  // D1 has a 100 bound-parameter limit per query (22 cols * 4 = 88 <= 100).
  const chunkSize = db.dialect === "postgres" ? 50 : 4;
  const colList = SIGNAL_WRITE_COLS.join(", ");
  const updateCols = SIGNAL_WRITE_COLS.slice(2)
    .map((c) => `${c} = excluded.${c}`)
    .join(", ");
  const tupleSql = `(${SIGNAL_WRITE_COLS.map(() => "?").join(", ")})`;

  for (let offset = 0; offset < rows.length; offset += chunkSize) {
    const slice = rows.slice(offset, offset + chunkSize);
    const tuples = slice.map(() => tupleSql).join(", ");
    const params = [];
    for (const r of slice) {
      for (const col of SIGNAL_WRITE_COLS) params.push(r[col] ?? null);
    }
    await db.run(
      `INSERT INTO signals (${colList}) VALUES ${tuples}
       ON CONFLICT (market, date) DO UPDATE SET ${updateCols}`,
      params,
    );
  }
  return rows.length;
}

/**
 * Recompute `ndx`, `qqq`, and `ssec` signals from `bars` + `_meta/events.json` + `news`
 * and incrementally upsert all new or changed rows (plus the latest watermark row).
 */
export async function recomputeAllSignals(
  db,
  { calendar = null, newsIndicators = null, contextsByMarket = {}, fetchedAt, rnd, scanKeywords },
) {
  const [targetBarRows, factorBarRows, existingSignalRows] = await db.batch([
    {
      sql: "SELECT symbol, date, open, close, is_closed FROM bars WHERE symbol IN ('^NDX', 'QQQ', '000001.SS') ORDER BY symbol ASC, date ASC",
    },
    {
      sql: "SELECT symbol, date, close FROM bars WHERE symbol IN ('^TNX', 'DX-Y.NYB', 'HYG', 'TLT', '^VIX') ORDER BY symbol ASC, date ASC",
    },
    {
      sql: `SELECT market, date, entry, entry_at, base_close, close3, close5, hit3,
                   flag, flag_price_only, sell_flag, sell_threshold, news_veto,
                   score, threshold, outcome, target, peak, peak_at, us_date,
                   factors_packed, fetched_at
            FROM signals
            ORDER BY market ASC, date ASC`,
    },
  ]);

  const targetBarsBySym = new Map();
  for (const r of targetBarRows) {
    if (!targetBarsBySym.has(r.symbol)) targetBarsBySym.set(r.symbol, []);
    targetBarsBySym.get(r.symbol).push(r);
  }
  const factorBarsBySym = new Map();
  for (const r of factorBarRows) {
    if (!factorBarsBySym.has(r.symbol)) factorBarsBySym.set(r.symbol, []);
    factorBarsBySym.get(r.symbol).push(r);
  }
  const existingByMarket = { ndx: new Map(), qqq: new Map(), ssec: new Map() };
  for (const r of existingSignalRows) {
    if (existingByMarket[r.market]) existingByMarket[r.market].set(r.date, r);
  }

  const ndxBars = targetBarsBySym.get("^NDX") || [];
  const qqqBars = targetBarsBySym.get("QQQ") || [];
  const ssecBars = targetBarsBySym.get("000001.SS") || [];
  if (!ndxBars.length) return { ndx: 0, qqq: 0, ssec: 0 };

  // Load recent news titles (last ~20 calendar days) for riskoff_hits on new/tail sessions
  const riskoffByDate = new Map();
  if (scanKeywords) {
    const latestNdxDate = ndxBars[ndxBars.length - 1].date;
    const tailCutoff = new Date(Date.parse(`${latestNdxDate}T00:00:00Z`) - 20 * 86400000).toISOString().slice(0, 10);
    const newsRows = await db.all("SELECT as_of, title FROM news WHERE as_of >= ? AND title <> ''", [tailCutoff]);
    const titlesByDay = new Map();
    for (const r of newsRows) {
      if (!titlesByDay.has(r.as_of)) titlesByDay.set(r.as_of, []);
      titlesByDay.get(r.as_of).push(r.title);
    }
    for (const [day, titles] of titlesByDay) {
      riskoffByDate.set(day, scanKeywords(titles).risk_off_hits.length);
    }
  }

  const dates = ndxBars.map((b) => b.date);
  const scores = computeComposite(dates, factorBarsBySym);
  const recentNewsLabels = buildRecentNewsLabelMap(newsIndicators);

  const ndxRes = computeUsMarket({
    market: "ndx",
    ndxBars,
    targetBars: null,
    scores,
    calendar,
    contextKeys: contextsByMarket.ndx || [],
    existingByDate: existingByMarket.ndx,
    recentNewsLabels,
    riskoffByDate,
    fetchedAt,
    rnd,
  });

  const qqqRes = computeUsMarket({
    market: "qqq",
    ndxBars,
    targetBars: qqqBars,
    scores,
    calendar,
    contextKeys: contextsByMarket.qqq || [],
    existingByDate: existingByMarket.qqq,
    recentNewsLabels,
    riskoffByDate,
    fetchedAt,
    rnd,
  });

  const ssecRes = computeSsecMarket({
    ndxResult: ndxRes,
    ssecBars,
    contextKeys: contextsByMarket.ssec || [],
    fetchedAt,
    rnd,
  });

  const written = {};
  const computed = { ndx: ndxRes.rows, qqq: qqqRes.rows, ssec: ssecRes.rows };
  for (const market of ["ndx", "qqq", "ssec"]) {
    const rows = computed[market];
    const existing = existingByMarket[market];
    const dirty = [];
    for (let i = 0; i < rows.length; i += 1) {
      const row = rows[i];
      const prev = existing.get(row.date);
      if (rowChanged(prev, row) || i === rows.length - 1) {
        dirty.push(row);
      }
    }
    written[market] = await upsertSignalRows(db, dirty);
  }

  return written;
}
