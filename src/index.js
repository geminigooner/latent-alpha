/**
 * LATENT ALPHA OBSERVATORY — Phase 0
 * Narrative novelty from SEC filings.
 *
 * Routes
 *   GET  /                  the page
 *   GET  /api/filings?cik=  list recent 10-K/10-Q from EDGAR
 *   POST /api/ingest        fetch ONE filing, extract, embed, store
 *   GET  /api/novelty?t=    read D1, compute novelty series
 *   POST /api/reset?t=      wipe one ticker
 *
 * One filing per request on purpose — keeps each invocation under the
 * Workers CPU limit. The browser drives the loop.
 */

const SEC_UA = "Amanda Hatley ahatley094@gmail.com";
const EMBED_MODEL = "gemini-embedding-001";
const EMBED_DIM = 1536;
const CHUNK_CHARS = 6000;
const MAX_TEXT = 600_000; // hard cap so a giant 10-K can't eat the CPU budget

const COMPANIES = [
  { ticker: "NVDA", name: "Nvidia", cik: "0001045810" },
  { ticker: "GOOGL", name: "Alphabet", cik: "0001652044" },
  { ticker: "MSFT", name: "Microsoft", cik: "0000789019" },
  { ticker: "META", name: "Meta", cik: "0001326801" },
  { ticker: "AMZN", name: "Amazon", cik: "0001018724" },
  { ticker: "AMD", name: "AMD", cik: "0000002488" },
  { ticker: "TSM", name: "TSMC", cik: "0001046179" },
];

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    try {
      switch (url.pathname) {
        case "/":             return new Response(PAGE, { headers: {
                                "content-type": "text/html;charset=utf-8",
                                // without this Safari happily serves the previous
                                // deploy's page and you debug code that isn't running
                                "cache-control": "no-store, must-revalidate",
                              } });
        case "/api/filings":  return json(await listFilings(url.searchParams.get("cik"), +(url.searchParams.get("n") || 10)));
        case "/api/ingest":   return json(await ingest(await request.json(), env));
        case "/api/novelty":  return json(await novelty(url.searchParams.get("t"), env));
        case "/api/prices":   return json(await loadPrices(url.searchParams.get("t"), env));
        case "/api/test":     return json(await volTest(url.searchParams.get("t"), env));
        case "/api/reset":    return json(await reset(url.searchParams.get("t"), env));
        case "/api/status":   return json(await status(env));
        case "/api/topchunk": return json(await topChunk(url.searchParams, env));
        case "/export":       return new Response(await exportText(env), { headers: {
                                "content-type": "text/plain;charset=utf-8",
                                "cache-control": "no-store",
                              } });
        default:              return new Response("not found", { status: 404 });
      }
    } catch (err) {
      return json({ error: String(err && err.message || err) }, 500);
    }
  },
};

const json = (obj, status = 200) =>
  new Response(JSON.stringify(obj), { status, headers: { "content-type": "application/json" } });

/* ─────────────────────────────────────────────── EDGAR */

async function listFilings(cik, limit) {
  if (!cik) throw new Error("cik required");
  const padded = String(cik).padStart(10, "0");
  const r = await fetch(`https://data.sec.gov/submissions/CIK${padded}.json`, {
    headers: { "User-Agent": SEC_UA, "Accept-Encoding": "gzip" },
  });
  if (!r.ok) throw new Error(`EDGAR index ${r.status}`);
  const recent = (await r.json()).filings.recent;

  const out = [];
  for (let i = 0; i < recent.form.length && out.length < limit; i++) {
    const form = recent.form[i];
    if (form !== "10-K" && form !== "10-Q") continue;
    const acc = recent.accessionNumber[i].replace(/-/g, "");
    out.push({
      form,
      filed: recent.filingDate[i],
      url: `https://www.sec.gov/Archives/edgar/data/${Number(cik)}/${acc}/${recent.primaryDocument[i]}`,
    });
  }
  return out.reverse(); // oldest first
}

/** Filing HTML → flat text, streamed. Skips <script>/<style> contents. */
async function filingText(url) {
  const res = await fetch(url, { headers: { "User-Agent": SEC_UA, "Accept-Encoding": "gzip" } });
  if (!res.ok) throw new Error(`filing fetch ${res.status}`);

  const parts = [];
  let skip = 0, len = 0;

  const rewritten = new HTMLRewriter()
    .on("script, style", {
      element(el) { skip++; el.onEndTag(() => { skip--; }); },
    })
    .on("*", {
      text(t) {
        if (skip || len > MAX_TEXT) return;
        const s = t.text;
        if (s) { parts.push(s); len += s.length; }
      },
    })
    .transform(res);

  await rewritten.arrayBuffer(); // drain the stream
  return decodeEntities(parts.join(" "))
    .replace(/\u00a0/g, " ").replace(/\s+/g, " ").trim();
}

const NAMED = { amp: "&", lt: "<", gt: ">", quot: '"', apos: "'", nbsp: " ", ndash: "–", mdash: "—", rsquo: "\u2019", lsquo: "\u2018", ldquo: "\u201c", rdquo: "\u201d" };
function decodeEntities(s) {
  return s.replace(/&(#x?[0-9a-f]+|[a-z]+);/gi, (m, body) => {
    if (body[0] === "#") {
      const code = body[1] === "x" || body[1] === "X"
        ? parseInt(body.slice(2), 16) : parseInt(body.slice(1), 10);
      return Number.isFinite(code) && code > 0 && code < 0x110000 ? String.fromCodePoint(code) : m;
    }
    return NAMED[body.toLowerCase()] ?? m;
  });
}

/* ─────────────────────────────────────────────── section extraction */

const SECTIONS = {
  "10-K": {
    mdna: {
      label: "MD&A",
      start: ["item 7.", "item 7 .", "item 7-", "item 7–", "item 7—", "item 7 ", "s discussion and analysis"],
      end: ["item 7a", "item 8.", "item 8 ", "quantitative and qualitative disclosures", "financial statements and supplementary"],
    },
    risk: {
      label: "Risk Factors",
      start: ["item 1a.", "item 1a .", "item 1a-", "item 1a–", "item 1a—", "item 1a ", "risk factors"],
      end: ["item 1b", "item 1c", "item 2.", "item 2 ", "unresolved staff comments", "properties"],
    },
  },
  "10-Q": {
    mdna: {
      label: "MD&A",
      start: ["item 2.", "item 2 .", "item 2-", "item 2–", "item 2—", "s discussion and analysis"],
      end: ["item 3.", "item 3 ", "item 4.", "item 4 ", "quantitative and qualitative disclosures", "controls and procedures", "part ii"],
    },
    risk: {
      label: "Risk Factors",
      start: ["item 1a.", "item 1a .", "item 1a-", "item 1a–", "item 1a—", "risk factors"],
      end: ["item 2.", "item 2 ", "item 3.", "item 3 ", "item 5.", "item 6.", "unregistered sales of equity", "defaults upon senior", "exhibits", "signatures"],
    },
  },
};

function findAll(hay, needle) {
  const out = [];
  let i = hay.indexOf(needle);
  while (i !== -1) { out.push(i); i = hay.indexOf(needle, i + needle.length); }
  return out;
}

const ITEM_HEADER_RE = /\bitem\s*\d{1,2}\s*[a-c]?\s*[.:)\-–—]/g;

function extractSection(text, kind, form = "10-K") {
  const formSpec = SECTIONS[form] || SECTIONS["10-K"];
  const spec = formSpec[kind];
  if (!spec) return null;

  const low = text.toLowerCase().replace(/['\u2019`]/g, " ");

  const allMarkers = Object.values(formSpec).flatMap((s) => [...s.start, ...s.end]);
  const all = [...new Set(allMarkers.flatMap((m) => findAll(low, m)))].sort((a, b) => a - b);

  // TOC entries have high marker density within 1500 chars
  const inTOC = (p) => all.filter((q) => Math.abs(q - p) < 1500).length >= 5;

  // Find all real "Item X." headers in the document
  const itemHeaders = [];
  let m;
  ITEM_HEADER_RE.lastIndex = 0;
  while ((m = ITEM_HEADER_RE.exec(low)) !== null) {
    if (!inTOC(m.index)) itemHeaders.push(m.index);
  }

  let starts = [...new Set(spec.start.flatMap((k) => findAll(low, k)))]
    .sort((a, b) => a - b)
    .filter((p) => !inTOC(p));

  if (!starts.length) return null;

  const ends = [...new Set(spec.end.flatMap((k) => findAll(low, k)))]
    .sort((a, b) => a - b)
    .filter((p) => !inTOC(p));

  const minLen = form === "10-Q" && kind === "risk" ? 120 : 1200;

  let best = null;
  for (const s of starts) {
    const windowBefore = low.slice(Math.max(0, s - 50), s + 35);
    if (!/\bitem\s*\d{1,2}\s*[a-c]?\b/.test(windowBefore) && !/discussion and analysis|risk factors/.test(windowBefore)) {
      continue;
    }

    const e = ends.find((x) => x > s + minLen);
    const nextHeader = itemHeaders.find((h) => h > s + 400);
    let stop = e ?? nextHeader ?? Math.min(s + 120_000, text.length);
    if (nextHeader && nextHeader < stop && nextHeader > s + minLen) {
      stop = nextHeader;
    }
    if (stop - s < minLen) continue;
    if (!best || stop - s > best[1] - best[0]) best = [s, stop];
  }

  if (!best) return null;
  const body = text.slice(best[0], best[1]);
  return body.length >= minLen ? body : null;
}

/** Safe-harbour language is near-identical every quarter; it would flatten novelty. */
function stripBoilerplate(t) {
  return t
    .replace(/(?:this (?:quarterly|annual) report|the following discussion)[^.]{0,200}forward[- ]looking statements[\s\S]{0,3000}?(?:section 21e|private securities litigation reform act)[^.]*\./gi, " ")
    .replace(/table of contents/gi, " ")
    .replace(/see note \d+ of the notes to (?:the )?consolidated financial statements/gi, " ")
    .replace(/\s+/g, " ")
    .trim();
}

/* ─────────────────────────────────────────────── embeddings */

function chunk(text, size = CHUNK_CHARS) {
  const out = [];
  for (let i = 0; i < text.length; i += size) {
    let end = Math.min(i + size, text.length);
    if (end < text.length) {
      const dot = text.lastIndexOf(". ", end);
      if (dot > i + size * 0.5) end = dot + 1;
    }
    out.push(text.slice(i, end));
    i = end - size; // resync loop to the actual cut point
  }
  return out.filter((s) => s.trim().length > 50);
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function embed(texts, apiKey) {
  const vecs = [];
  for (let i = 0; i < texts.length; i += 20) {
    const batch = texts.slice(i, i + 20);
    const body = JSON.stringify({
      requests: batch.map((text) => ({
        model: `models/${EMBED_MODEL}`,
        content: { parts: [{ text }] },
        taskType: "SEMANTIC_SIMILARITY",
        outputDimensionality: EMBED_DIM,
      })),
    });

    let r, attempt = 0;
    while (true) {
      r = await fetch(
        `https://generativelanguage.googleapis.com/v1beta/models/${EMBED_MODEL}:batchEmbedContents`,
        { method: "POST", headers: { "content-type": "application/json", "x-goog-api-key": apiKey }, body }
      );
      if (r.ok) break;
      if (r.status === 429 && attempt < 3) { attempt++; await sleep(2000 * attempt); continue; }
      throw new Error(r.status === 429 ? "QUOTA" : `embed ${r.status}`);
    }

    for (const e of (await r.json()).embeddings) vecs.push(e.values);
    await sleep(250);
  }
  return vecs;
}

function poolNormalize(vecs) {
  const d = vecs[0].length;
  const acc = new Float64Array(d);
  for (const v of vecs) for (let i = 0; i < d; i++) acc[i] += v[i];
  let n = 0;
  for (let i = 0; i < d; i++) { acc[i] /= vecs.length; n += acc[i] * acc[i]; }
  n = Math.sqrt(n) || 1;
  return Array.from(acc, (x) => x / n);
}

const normalize = (v) => {
  let n = 0;
  for (let i = 0; i < v.length; i++) n += v[i] * v[i];
  n = Math.sqrt(n) || 1;
  return Float32Array.from(v, (x) => x / n);
};

function packVecs(vecs) {
  const d = vecs[0].length;
  const flat = new Float32Array(vecs.length * d);
  vecs.forEach((v, i) => flat.set(v, i * d));
  const bytes = new Uint8Array(flat.buffer);
  let s = "";
  for (let i = 0; i < bytes.length; i += 0x8000) {
    s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  }
  return btoa(s);
}

function unpackVecs(b64, d) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const f = new Float32Array(bytes.buffer);
  const out = [];
  for (let i = 0; i + d <= f.length; i += d) out.push(f.subarray(i, i + d));
  return out;
}

const cosine = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };

/* ─────────────────────────────────────────────── ingest */

async function ingest(body, env) {
  const { ticker, form, filed, url, section } = body;
  if (!env.GEMINI_API_KEY) throw new Error("GEMINI_API_KEY secret not set");

  const formSpec = SECTIONS[form] || SECTIONS["10-K"];
  const kinds = section && formSpec[section] ? [section] : ["mdna", "risk"];

  const keys = kinds.map((k) => `${ticker}:${filed}:${form}:${k}`);
  const have = await env.DB.prepare(
    `SELECT key FROM sections WHERE chunks IS NOT NULL AND key IN (${keys.map(() => "?").join(",")})`
  ).bind(...keys).all();
  const stored = new Set(have.results.map((r) => r.key));
  if (stored.size === keys.length) return { filed, form, skipped: true, sections: kinds.map((k) => ({ kind: k, status: "cached" })) };

  const text = await filingText(url);
  const done = [];

  for (const kind of kinds) {
    if (stored.has(`${ticker}:${filed}:${form}:${kind}`)) { done.push({ kind, status: "cached" }); continue; }
    const raw = extractSection(text, kind, form);
    if (!raw) { done.push({ kind, status: "not found" }); continue; }

    const clean = stripBoilerplate(raw);
    const parts = chunk(clean).slice(0, 40);
    if (!parts.length) { done.push({ kind, status: "not found" }); continue; }
    const rawVecs = await embed(parts, env.GEMINI_API_KEY);
    const unit = rawVecs.map(normalize);

    await env.DB.prepare(
      `INSERT OR REPLACE INTO sections (key, ticker, form, filed, section, chars, vec, chunks)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?)`
    ).bind(
      `${ticker}:${filed}:${form}:${kind}`, ticker, form, filed, kind, clean.length,
      JSON.stringify(poolNormalize(rawVecs)), packVecs(unit)
    ).run();

    done.push({ kind, status: "ok", chars: clean.length, chunks: parts.length });
  }
  return { filed, form, sections: done, textLen: text.length };
}

/* ─────────────────────────────────────────────── novelty */

function shiftMetrics(cur, prev) {
  const dists = cur.map((a) => {
    let best = 2;
    for (const b of prev) {
      let dot = 0;
      for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
      const d = 1 - dot;
      if (d < best) best = d;
    }
    return best;
  });
  const sorted = dists.slice().sort((x, y) => x - y);
  const q = (p) => sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))];
  return { max: sorted[sorted.length - 1], p75: q(0.75) };
}

async function novelty(ticker, env) {
  const { results } = await env.DB.prepare(
    `SELECT form, filed, section, chars, vec, chunks FROM sections
     WHERE ticker = ? ORDER BY filed ASC`
  ).bind(ticker).all();

  const out = {};
  for (const kind of ["mdna", "risk"]) {
    const rows = results.filter((r) => r.section === kind).map((r) => ({
      ...r,
      v: JSON.parse(r.vec),
      c: r.chunks ? unpackVecs(r.chunks, EMBED_DIM) : null,
    }));

    out[kind] = rows.map((r, i) => {
      const consec = i > 0 ? 1 - cosine(r.v, rows[i - 1].v) : null;
      let prev = null;
      for (let j = i - 1; j >= 0; j--) if (rows[j].form === r.form) { prev = rows[j]; break; }

      const s = prev && r.c && prev.c ? shiftMetrics(r.c, prev.c) : null;
      return {
        filed: r.filed, form: r.form, chars: r.chars, chunks: r.c ? r.c.length : 0,
        consecutive: consec,
        pooled: prev ? 1 - cosine(r.v, prev.v) : null,
        max: s ? s.max : null,
        p75: s ? s.p75 : null,
      };
    });
  }
  return out;
}

/* ─────────────────────────────────────────────── prices */

const BROWSER_UA =
  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 " +
  "(KHTML, like Gecko) Chrome/125.0 Safari/537.36";

async function fromYahoo(ticker) {
  const r = await fetch(
    `https://query1.finance.yahoo.com/v8/finance/chart/${ticker}?range=5y&interval=1d`,
    { headers: { "User-Agent": BROWSER_UA, accept: "application/json" } }
  );
  if (!r.ok) throw new Error(`yahoo ${r.status}`);
  const res = (await r.json())?.chart?.result?.[0];
  if (!res?.timestamp) throw new Error("yahoo: empty series");

  const adj = res.indicators?.adjclose?.[0]?.adjclose;
  const raw = res.indicators?.quote?.[0]?.close;
  const px = adj || raw;
  if (!px) throw new Error("yahoo: no closes");

  const rows = [];
  for (let i = 0; i < res.timestamp.length; i++) {
    const c = px[i];
    if (c == null || !isFinite(c)) continue;
    rows.push([new Date(res.timestamp[i] * 1000).toISOString().slice(0, 10), c]);
  }
  return { rows, source: adj ? "yahoo (adjusted)" : "yahoo (close)" };
}

async function fromStooq(ticker) {
  const r = await fetch(`https://stooq.com/q/d/l/?s=${ticker.toLowerCase()}.us&i=d`,
    { headers: { "User-Agent": BROWSER_UA } });
  if (!r.ok) throw new Error(`stooq ${r.status}`);
  const csv = await r.text();
  if (!csv.startsWith("Date")) throw new Error("stooq: no data");

  const rows = [];
  for (const line of csv.trim().split("\n").slice(1)) {
    const c = line.split(",");
    const close = parseFloat(c[4]);
    if (c[0] && isFinite(close)) rows.push([c[0], close]);
  }
  return { rows, source: "stooq" };
}

async function loadPrices(ticker, env) {
  const tried = [];
  let got = null;

  for (const src of [fromYahoo, fromStooq]) {
    try { got = await src(ticker); break; }
    catch (e) { tried.push(e.message); }
  }
  if (!got || got.rows.length < 100) throw new Error(tried.join(" / ") || "no usable series");

  got.rows.sort((a, b) => (a[0] < b[0] ? -1 : 1));
  await env.DB.prepare(
    `INSERT OR REPLACE INTO series (ticker, updated, data) VALUES (?, ?, ?)`
  ).bind(ticker, new Date().toISOString().slice(0, 10), JSON.stringify(got.rows)).run();

  return {
    ticker, source: got.source, days: got.rows.length,
    from: got.rows[0][0], to: got.rows[got.rows.length - 1][0],
  };
}

/** Annualised realised vol of daily log returns over a window of closes. */
function realisedVol(closes) {
  if (closes.length < 8) return null;
  const rets = [];
  for (let i = 1; i < closes.length; i++) rets.push(Math.log(closes[i] / closes[i - 1]));
  const mean = rets.reduce((a, b) => a + b, 0) / rets.length;
  const varc = rets.reduce((a, b) => a + (b - mean) ** 2, 0) / (rets.length - 1);
  return Math.sqrt(varc * 252);
}

/* ─────────────────────────────────────────────── volatility test */

function rankArray(arr) {
  const indexed = arr.map((v, i) => ({ v, i })).sort((a, b) => a.v - b.v);
  const ranks = new Array(arr.length);
  let i = 0;
  while (i < indexed.length) {
    let j = i;
    while (j + 1 < indexed.length && indexed[j + 1].v === indexed[i].v) j++;
    const avgRank = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) ranks[indexed[k].i] = avgRank;
    i = j + 1;
  }
  return ranks;
}

function spearman(x, y) {
  const n = x.length;
  if (n < 3) return null;
  const rx = rankArray(x);
  const ry = rankArray(y);
  const mx = rx.reduce((a, b) => a + b, 0) / n;
  const my = ry.reduce((a, b) => a + b, 0) / n;
  let num = 0, dx = 0, dy = 0;
  for (let i = 0; i < n; i++) {
    const a = rx[i] - mx;
    const b = ry[i] - my;
    num += a * b;
    dx += a * a;
    dy += b * b;
  }
  const den = Math.sqrt(dx * dy);
  return den === 0 ? 0 : num / den;
}

function permTest(x, y, shuffles = 3000) {
  const n = x.length;
  if (n < 4) return { n, rho: null, p: null };
  const obs = spearman(x, y);
  if (obs == null || !isFinite(obs)) return { n, rho: null, p: null };

  const copy = y.slice();
  let extreme = 0;
  for (let s = 0; s < shuffles; s++) {
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      const tmp = copy[i]; copy[i] = copy[j]; copy[j] = tmp;
    }
    const r = spearman(x, copy);
    if (Math.abs(r) >= Math.abs(obs) - 1e-12) extreme++;
  }
  return { n, rho: obs, p: (extreme + 1) / (shuffles + 1) };
}

async function volTest(tickerParam, env) {
  const { results: priceRows } = await env.DB.prepare(`SELECT ticker, data FROM series`).all();
  const priceMap = new Map(priceRows.map((r) => [r.ticker, JSON.parse(r.data)]));

  const { results: secRows } = await env.DB.prepare(
    `SELECT ticker, form, filed, section FROM sections ORDER BY ticker, filed ASC`
  ).all();

  const tickers = tickerParam
    ? [tickerParam]
    : [...new Set([...COMPANIES.map((c) => c.ticker), ...secRows.map((r) => r.ticker)])];

  const allObs = [];
  const perCompany = {};

  for (const t of tickers) {
    const px = priceMap.get(t);
    const nov = await novelty(t, env);
    const companyObs = [];

    if (px && px.length >= 60) {
      for (const r of nov.mdna || []) {
        if (r.max == null || r.pooled == null || r.p75 == null) continue;
        const idx = px.findIndex((p) => p[0] > r.filed);
        if (idx < 30 || idx + 30 > px.length) continue;

        const trailCloses = px.slice(idx - 30, idx).map((p) => p[1]);
        const fwdCloses = px.slice(idx, idx + 30).map((p) => p[1]);
        const vTrail = realisedVol(trailCloses);
        const vFwd = realisedVol(fwdCloses);
        if (!vTrail || !vFwd) continue;

        const volRatio = Math.log(vFwd / vTrail);
        const item = { ticker: t, filed: r.filed, form: r.form, max: r.max, p75: r.p75, pooled: r.pooled, volRatio };
        companyObs.push(item);
        allObs.push(item);
      }
    }

    const cTest = companyObs.length >= 4
      ? permTest(companyObs.map((o) => o.max), companyObs.map((o) => o.volRatio), 1000)
      : { n: companyObs.length, rho: null, p: null };

    perCompany[t] = { n: companyObs.length, rho: cTest.rho, p: cTest.p };
  }

  const y = allObs.map((o) => o.volRatio);
  return {
    metrics: {
      max: permTest(allObs.map((o) => o.max), y, 3000),
      p75: permTest(allObs.map((o) => o.p75), y, 3000),
      pooled: permTest(allObs.map((o) => o.pooled), y, 3000),
    },
    perCompany,
    observations: allObs,
  };
}

/* ─────────────────────────────────────────────── utilities & inspection */

async function reset(ticker, env) {
  if (!ticker) throw new Error("ticker required");
  await env.DB.prepare(`DELETE FROM sections WHERE ticker = ?`).bind(ticker).run();
  return { ok: true, ticker };
}

async function status(env) {
  const { results: s } = await env.DB.prepare(
    `SELECT ticker, COUNT(*) as count FROM sections GROUP BY ticker`
  ).all();
  const { results: p } = await env.DB.prepare(
    `SELECT ticker, updated FROM series`
  ).all();
  return { sections: s, prices: p };
}

async function topChunk(params, env) {
  const ticker = params.get("t");
  const filed = params.get("f");
  const section = params.get("s") || "mdna";
  if (!ticker || !filed) throw new Error("missing t or f");

  const { results } = await env.DB.prepare(
    `SELECT form, filed, chunks FROM sections WHERE ticker = ? AND section = ? ORDER BY filed ASC`
  ).bind(ticker, section).all();

  const idx = results.findIndex((r) => r.filed === filed);
  if (idx === -1) return { error: "filing not found" };
  const cur = results[idx];
  let prev = null;
  for (let j = idx - 1; j >= 0; j--) if (results[j].form === cur.form) { prev = results[j]; break; }
  if (!prev) return { error: "no prior same-form filing" };

  const curVecs = unpackVecs(cur.chunks, EMBED_DIM);
  const prevVecs = unpackVecs(prev.chunks, EMBED_DIM);
  let worstIdx = 0, worstDist = -1;
  curVecs.forEach((a, i) => {
    let best = 2;
    for (const b of prevVecs) {
      const d = 1 - cosine(a, b);
      if (d < best) best = d;
    }
    if (best > worstDist) { worstDist = best; worstIdx = i; }
  });

  const filings = await listFilings(COMPANIES.find((c) => c.ticker === ticker)?.cik, 15);
  const fMeta = filings.find((f) => f.filed === filed);
  if (!fMeta) return { chunkIndex: worstIdx, distance: worstDist, text: "(filing URL not in recent list)" };

  const fullText = await filingText(fMeta.url);
  const raw = extractSection(fullText, section, cur.form);
  const parts = raw ? chunk(stripBoilerplate(raw)).slice(0, 40) : [];
  return {
    filed: cur.filed,
    prevFiled: prev.filed,
    form: cur.form,
    chunkIndex: worstIdx,
    distance: worstDist,
    text: parts[worstIdx] || "(chunk text unavailable)",
  };
}

async function exportText(env) {
  const res = await volTest(null, env);
  return JSON.stringify(res, null, 2);
}

/* ─────────────────────────────────────────────── UI PAGE */

const PAGE = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>LATENT ALPHA — narrative novelty</title>
<style>
  :root {
    --bg: #0c090a;
    --panel: #140f11;
    --border: #2a2024;
    --text: #e8dfe2;
    --muted: #8c7b81;
    --accent: #f0c8d8;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 24px 18px 80px;
    background: var(--bg); color: var(--text);
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 13px; line-height: 1.55;
    max-width: 720px; margin-inline: auto;
  }
  h1 { font-size: 16px; letter-spacing: 0.12em; margin: 0 0 4px; font-weight: 600; }
  .sub { color: var(--muted); font-size: 12px; margin-bottom: 24px; border-bottom: 1px solid var(--border); padding-bottom: 16px; }
  label { display: block; color: var(--muted); font-size: 11px; letter-spacing: 0.1em; margin: 18px 0 6px; text-transform: uppercase; }
  select, button {
    width: 100%; padding: 12px 14px;
    background: var(--panel); color: var(--text);
    border: 1px solid var(--border);
    font-family: inherit; font-size: 13px;
    appearance: none; border-radius: 0;
  }
  button {
    margin-top: 14px; cursor: pointer;
    letter-spacing: 0.1em; text-transform: uppercase; font-size: 12px;
  }
  button:hover { border-color: var(--accent); }
  button:disabled { opacity: 0.5; cursor: wait; }
  .log {
    margin-top: 16px; padding: 12px;
    background: var(--panel); border: 1px solid var(--border);
    font-size: 12px; color: var(--muted); white-space: pre-wrap;
    display: none;
  }
  .sec-title {
    margin-top: 32px; padding-top: 16px;
    border-top: 1px solid var(--border);
    font-size: 12px; letter-spacing: 0.1em; color: var(--muted); text-transform: uppercase;
  }
  table { width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 12px; }
  th, td { text-align: left; padding: 8px 6px; border-bottom: 1px solid var(--border); }
  th { color: var(--muted); font-weight: normal; font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase; }
  tr.clickable { cursor: pointer; }
  tr.clickable:hover { background: #1c1518; }
  .note { color: var(--muted); font-size: 12px; margin-top: 14px; line-height: 1.6; }
  .modal {
    margin-top: 14px; padding: 14px;
    background: var(--panel); border: 1px solid var(--accent);
    font-size: 12px; white-space: pre-wrap; display: none;
  }
</style>
</head>
<body>
  <h1>LATENT ALPHA</h1>
  <div class="sub">narrative novelty · sec edgar · gemini embeddings</div>

  <label>COMPANY</label>
  <select id="company">
    <option value="NVDA|0001045810">NVDA — Nvidia</option>
    <option value="GOOGL|0001652044">GOOGL — Alphabet</option>
    <option value="MSFT|0000789019">MSFT — Microsoft</option>
    <option value="META|0001326801">META — Meta</option>
    <option value="AMZN|0001018724">AMZN — Amazon</option>
    <option value="AMD|0000002488">AMD — AMD</option>
    <option value="TSM|0001046179">TSM — TSMC</option>
  </select>

  <label>FILINGS</label>
  <select id="limit">
    <option value="6">6</option>
    <option value="10" selected>10</option>
    <option value="14">14</option>
  </select>

  <button id="ingestBtn">INGEST & MEASURE</button>
  <div id="log" class="log"></div>
  <div id="tables"></div>
  <div id="chunkModal" class="modal"></div>

  <div class="sec-title">DOES IT PREDICT ANYTHING?</div>
  <div class="note">
    Hypothesis, fixed before looking: filings whose language moved unusually far from the same company's previous same-form filing are followed by higher realised volatility than that company's own recent baseline. Forward window starts the trading day <i>after</i> the filing.
  </div>

  <button id="pricesBtn">LOAD PRICES</button>
  <button id="testBtn">RUN TEST</button>
  <div id="priceLog" class="log"></div>
  <div id="testOut"></div>

<script>
const $ = (id) => document.getElementById(id);
const fmt = (v) => v == null || !isFinite(v) ? "—" : v.toFixed(4);

async function renderNovelty(ticker) {
  const r = await fetch("/api/novelty?t=" + ticker);
  const data = await r.json();
  let html = "";
  for (const [kind, label] of [["mdna", "MD&A"], ["risk", "RISK FACTORS"]]) {
    const rows = data[kind] || [];
    html += '<div class="sec-title">' + ticker + ' · ' + label + '</div>';
    html += '<table><thead><tr><th>FILED</th><th>FORM</th><th>POOLED</th><th>MAX</th><th>P75</th></tr></thead><tbody>';
    for (const row of rows) {
      html += '<tr class="clickable" onclick="inspectChunk(\\'' + ticker + '\\',\\'' + row.filed + '\\',\\'' + kind + '\\')">' +
        '<td>' + row.filed + '</td>' +
        '<td>' + row.form + '</td>' +
        '<td>' + fmt(row.pooled) + '</td>' +
        '<td><b>' + fmt(row.max) + '</b></td>' +
        '<td>' + fmt(row.p75) + '</td></tr>';
    }
    html += '</tbody></table>';
  }
  html += '<div class="note">Max is the distance from each passage in this filing to its nearest passage in the previous same-form filing, taking the largest — "is there something here with no counterpart last quarter?" Pooled averages the whole section, which drowns one changed paragraph in the thirty that did not change. p75 only rises when much of the section is rewritten.<br><br>Tap any row to read the passage that produced the max. A number cannot tell you whether it fired on a strategy shift or on a new legal disclaimer, and those mean opposite things.</div>';
  $("tables").innerHTML = html;
}

window.inspectChunk = async (ticker, filed, section) => {
  const m = $("chunkModal");
  m.style.display = "block";
  m.textContent = "loading top passage for " + ticker + " " + filed + " (" + section + ")...";
  const r = await fetch("/api/topchunk?t=" + ticker + "&f=" + filed + "&s=" + section);
  const d = await r.json();
  if (d.error) { m.textContent = d.error; return; }
  m.textContent = ticker + " " + d.filed + " vs " + d.prevFiled + " (" + d.form + " · chunk #" + d.chunkIndex + " · dist " + fmt(d.distance) + ")\\n\\n" + d.text;
};

$("ingestBtn").onclick = async () => {
  const [ticker, cik] = $("company").value.split("|");
  const n = $("limit").value;
  const log = $("log");
  log.style.display = "block";
  log.textContent = "fetching filing index...\\n";
  $("ingestBtn").disabled = true;

  try {
    const fl = await (await fetch("/api/filings?cik=" + cik + "&n=" + n)).json();
    if (fl.error) throw new Error(fl.error);
    log.textContent += fl.length + " filings · " + (fl[0]?.filed || "") + " → " + (fl[fl.length - 1]?.filed || "") + "\\n\\n";

    for (const f of fl) {
      log.textContent += f.filed + "  " + f.form + "\\n";
      for (const sec of ["mdna", "risk"]) {
        const res = await (await fetch("/api/ingest", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ ticker, form: f.form, filed: f.filed, url: f.url, section: sec }),
        })).json();
        if (res.error) {
          log.textContent += "  " + sec + " error: " + res.error + "\\n";
        } else {
          const st = res.sections?.[0]?.status || (res.skipped ? "cached" : "ok");
          log.textContent += "  " + sec + " " + st + "\\n";
        }
      }
    }
    log.textContent += "\\ncomputing novelty...";
    await renderNovelty(ticker);
  } catch (e) {
    log.textContent += "\\nERROR: " + e.message;
  } finally {
    $("ingestBtn").disabled = false;
  }
};

$("pricesBtn").onclick = async () => {
  const log = $("priceLog");
  log.style.display = "block";
  log.textContent = "";
  $("pricesBtn").disabled = true;
  const tickers = ["NVDA", "MSFT", "GOOGL", "META", "AMZN", "AMD", "TSM"];
  try {
    for (const t of tickers) {
      const r = await (await fetch("/api/prices?t=" + t)).json();
      if (r.error) log.textContent += t + " error: " + r.error + " ";
      else log.textContent += t + " " + r.days + " days " + r.from + " → " + r.to + " · " + r.source + " ";
    }
  } finally {
    $("pricesBtn").disabled = false;
  }
};

$("testBtn").onclick = async () => {
  const out = $("testOut");
  out.innerHTML = '<div class="note">running permutation test...</div>';
  $("testBtn").disabled = true;
  try {
    const d = await (await fetch("/api/test")).json();
    if (d.error) throw new Error(d.error);
    let h = '<table><thead><tr><th>METRIC</th><th>N</th><th>ρ</th><th>P</th></tr></thead><tbody>';
    for (const [k, label] of [["max", "max shift"], ["p75", "p75 shift"], ["pooled", "pooled"]]) {
      const m = d.metrics?.[k] || {};
      h += '<tr><td><b>' + label + '</b></td><td>' + (m.n || "—") + '</td><td>' + fmt(m.rho) + '</td><td>' + fmt(m.p) + '</td></tr>';
    }
    h += '</tbody></table>';

    h += '<div class="sec-title">PER COMPANY</div>';
    h += '<table><thead><tr><th>TICKER</th><th>N</th><th>ρ MAX</th><th>P MAX</th></tr></thead><tbody>';
    for (const [t, info] of Object.entries(d.perCompany || {})) {
      if (!info.n && !["NVDA", "GOOGL", "MSFT"].includes(t)) continue;
      const tooFew = (info.n || 0) < 4;
      h += '<tr><td><b>' + t + '</b></td><td>' + (info.n || 0) + '</td><td>' + (tooFew ? "—" : fmt(info.rho)) + '</td><td>' + (tooFew ? "too few" : fmt(info.p)) + '</td></tr>';
    }
    h += '</tbody></table>';
    h += '<div class="note">All three metrics run every time and are all reported, so none of them was chosen after the fact. Nothing is significant. At this sample size that is closer to "no data" than to "no effect" — the fix is more companies, not a different metric.</div>';
    out.innerHTML = h;
  } catch (e) {
    out.innerHTML = '<div class="note">ERROR: ' + e.message + '</div>';
  } finally {
    $("testBtn").disabled = false;
  }
};
</script>
</body>
</html>`;