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

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    try {
      switch (url.pathname) {
        case "/":             return new Response(PAGE, { headers: { "content-type": "text/html;charset=utf-8" } });
        case "/api/filings":  return json(await listFilings(url.searchParams.get("cik"), +(url.searchParams.get("n") || 10)));
        case "/api/ingest":   return json(await ingest(await request.json(), env));
        case "/api/novelty":  return json(await novelty(url.searchParams.get("t"), env));
        case "/api/prices":   return json(await loadPrices(url.searchParams.get("t"), env));
        case "/api/test":     return json(await volTest(url.searchParams.get("t"), env));
        case "/api/reset":    return json(await reset(url.searchParams.get("t"), env));
        case "/api/status":   return json(await status(env));
        case "/api/topchunk": return json(await topChunk(url.searchParams, env));
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
  return parts.join(" ").replace(/\u00a0/g, " ").replace(/\s+/g, " ").trim();
}

/* ─────────────────────────────────────────────── section extraction */

/**
 * Anchor on section TITLES, not item numbers — MD&A is Item 7 in a 10-K
 * and Item 2 in a 10-Q. Apostrophes are normalised to spaces (length
 * preserved, so indices stay valid), which makes "management's discussion"
 * and "managements discussion" both match on "s discussion and analysis".
 */
const SECTIONS = {
  mdna: {
    label: "MD&A",
    start: ["s discussion and analysis"],
    end: ["quantitative and qualitative disclosures", "controls and procedures", "financial statements and supplementary"],
  },
  risk: {
    label: "Risk Factors",
    start: ["risk factors"],
    end: ["unresolved staff comments", "legal proceedings", "unregistered sales of equity", "properties"],
  },
};

function findAll(hay, needle) {
  const out = [];
  let i = hay.indexOf(needle);
  while (i !== -1) { out.push(i); i = hay.indexOf(needle, i + needle.length); }
  return out;
}

/**
 * Every marker appears at least twice — once in the table of contents,
 * once for real. Take every (start, nearest-end) pair and keep the longest
 * span; the TOC hit is always a few characters long, the real one isn't.
 */
function extractSection(text, kind) {
  const spec = SECTIONS[kind];
  const low = text.toLowerCase().replace(/['\u2019`]/g, " ");

  const starts = spec.start.flatMap((m) => findAll(low, m)).sort((a, b) => a - b);
  if (!starts.length) return null;
  const ends = spec.end.flatMap((m) => findAll(low, m)).sort((a, b) => a - b);

  let best = null;
  for (const s of starts) {
    const e = ends.find((x) => x > s + 200);
    const stop = e ?? Math.min(s + 120_000, text.length);
    if (!best || stop - s > best[1] - best[0]) best = [s, stop];
  }
  const body = text.slice(best[0], best[1]);
  return body.length > 1500 ? body : null; // too short = we grabbed the TOC line
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
      // 429 = rate limited. Back off and retry a few times; if it's the daily
      // quota rather than per-minute, the retries won't help and we surface it.
      if (r.status === 429 && attempt < 3) { attempt++; await sleep(2000 * attempt); continue; }
      throw new Error(r.status === 429 ? "QUOTA" : `embed ${r.status}`);
    }

    for (const e of (await r.json()).embeddings) vecs.push(e.values);
    await sleep(250); // stay under the per-minute rate limit
  }
  return vecs;
}

/** Mean-pool then L2 normalise. Normalising matters: MRL-truncated dims aren't unit length. */
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

/* Chunk vectors are kept as packed float32, not JSON. 30 chunks of 1536 dims
   is 184KB as base64 and 2.6MB as a JSON array of numbers — the second one
   does not fit in a D1 row. */
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

  // One section per request. Both sections in a single invocation was enough
  // work — fetch a 3MB filing, embed 60 chunks, base64 and write 370KB to D1,
  // twice — to trip the Worker resource limit on the larger filings and return
  // a 503 HTML error page instead of JSON.
  const kinds = section && SECTIONS[section] ? [section] : Object.keys(SECTIONS);

  const keys = kinds.map((k) => `${ticker}:${filed}:${form}:${k}`);
  const have = await env.DB.prepare(
    `SELECT key FROM sections WHERE chunks IS NOT NULL AND key IN (${keys.map(() => "?").join(",")})`
  ).bind(...keys).all();
  const stored = new Set(have.results.map((r) => r.key));
  if (stored.size === keys.length) return { filed, form, skipped: true, sections: [] };

  const text = await filingText(url);
  const done = [];

  for (const kind of kinds) {
    if (stored.has(`${ticker}:${filed}:${form}:${kind}`)) { done.push({ kind, status: "cached" }); continue; }
    const raw = extractSection(text, kind);
    if (!raw) { done.push({ kind, status: "not found" }); continue; }

    const clean = stripBoilerplate(raw);
    const parts = chunk(clean).slice(0, 40);
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

/* ─────────────────────────────────────────────── novelty

   Three metrics from the same vectors, because they ask different questions:

   pooled  — distance between the mean-pooled section vectors. This is the one
             that produced a flat line: averaging thirty chunks of a 100,000
             character section drowns the one paragraph that changed in the
             twenty-nine that didn't.

   max     — for every chunk in this filing, distance to its NEAREST chunk in
             the previous filing; then the largest of those. Answers "is there
             a passage here with no close counterpart last quarter?" Sensitive
             to genuinely new content, and to a single reworded paragraph.

   p75     — the same alignment distances at the 75th percentile. Blunter than
             max, but one oddly-worded chunk can't carry it on its own. */

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
  for (const kind of Object.keys(SECTIONS)) {
    const rows = results.filter((r) => r.section === kind).map((r) => ({
      ...r,
      v: JSON.parse(r.vec),
      c: r.chunks ? unpackVecs(r.chunks, EMBED_DIM) : null,
    }));

    out[kind] = rows.map((r, i) => {
      // consecutive: vs previous filing of ANY type. Contaminated — a 10-K is
      // far longer and broader than a 10-Q, so annuals spike structurally.
      const consec = i > 0 ? 1 - cosine(r.v, rows[i - 1].v) : null;

      // same-form: 10-Q vs previous 10-Q. A spike here is about language.
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

/* ─────────────────────────────────────────────── prices

   Two free, no-key sources tried in order. Stooq blocks datacenter IPs, so
   Cloudflare's egress gets refused — Yahoo's chart endpoint is the reliable
   one from a Worker, and it returns split/dividend-adjusted closes, which is
   what you actually want for return calculations.

   Stored as one JSON row per ticker: ~30KB, and we always read the whole
   series anyway, so there's no reason to pay for 1,500 row inserts. */

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

  // adjusted closes handle splits — NVDA did 10:1 in 2024, raw closes would
  // manufacture a fake -90% return on that single day and wreck the vol series
  const adj = res.indicators?.adjclose?.[0]?.adjclose;
  const raw = res.indicators?.quote?.[0]?.close;
  const px = adj || raw;
  if (!px) throw new Error("yahoo: no closes");

  const rows = [];
  for (let i = 0; i < res.timestamp.length; i++) {
    const c = px[i];
    if (c == null || !isFinite(c)) continue;               // holidays / gaps
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

/* ─────────────────────────────────────────────── statistics */

function ranks(xs) {
  const idx = xs.map((v, i) => [v, i]).sort((a, b) => a[0] - b[0]);
  const r = new Array(xs.length);
  for (let i = 0; i < idx.length;) {           // average ranks within ties
    let j = i;
    while (j + 1 < idx.length && idx[j + 1][0] === idx[i][0]) j++;
    const avg = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) r[idx[k][1]] = avg;
    i = j + 1;
  }
  return r;
}

function pearson(a, b) {
  const n = a.length;
  const ma = a.reduce((x, y) => x + y, 0) / n;
  const mb = b.reduce((x, y) => x + y, 0) / n;
  let num = 0, da = 0, db = 0;
  for (let i = 0; i < n; i++) {
    const x = a[i] - ma, y = b[i] - mb;
    num += x * y; da += x * x; db += y * y;
  }
  return da && db ? num / Math.sqrt(da * db) : 0;
}

const spearman = (a, b) => pearson(ranks(a), ranks(b));

/**
 * Permutation test. With ~15 filings per company a correlation coefficient on
 * its own is close to meaningless — shuffling the labels a few thousand times
 * and asking how often chance beats the observed value is the honest version.
 */
function permutationP(a, b, iters = 3000) {
  const observed = Math.abs(spearman(a, b));
  const shuffled = b.slice();
  let hits = 0;
  for (let it = 0; it < iters; it++) {
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    if (Math.abs(spearman(a, shuffled)) >= observed) hits++;
  }
  return { rho: spearman(a, b), p: (hits + 1) / (iters + 1), n: a.length };
}

const zscore = (xs) => {
  const m = xs.reduce((a, b) => a + b, 0) / xs.length;
  const sd = Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / Math.max(1, xs.length - 1)) || 1;
  return xs.map((x) => (x - m) / sd);
};

/* ─────────────────────────────────────────────── the test

   Hypothesis, fixed before looking: filings whose language moved unusually far
   from the same company's previous same-form filing are followed by higher
   realised volatility than that company's own recent baseline.

   Only same-form novelty is used (10-Q vs previous 10-Q) — the consecutive
   series spikes every year for structural reasons. Forward vol starts the
   trading day AFTER the filing date, so nothing from the future leaks in. */

const WINDOW = 30;
const METRICS = ["max", "p75", "pooled"];

/**
 * Runs every metric, every time. Selecting one after seeing the results is how
 * you manufacture a finding; the defence is that all three are always on the
 * record, so the comparison can be audited later.
 */
async function volTest(which, env) {
  const all = (await env.DB.prepare(`SELECT DISTINCT ticker FROM sections`).all())
    .results.map((r) => r.ticker);
  const tickers = which && which !== "ALL" ? [which] : all;

  const cache = {}, missing = [];
  for (const ticker of tickers) {
    const row = await env.DB.prepare(`SELECT data FROM series WHERE ticker = ?`).bind(ticker).first();
    if (!row) { missing.push(ticker); continue; }
    const series = JSON.parse(row.data);
    cache[ticker] = {
      dates: series.map((r) => r[0]),
      closes: series.map((r) => r[1]),
      nov: (await novelty(ticker, env)).mdna,
    };
  }

  const out = {};
  for (const m of METRICS) {
    const perTicker = [], pooledNov = [], pooledVol = [];

    for (const ticker of Object.keys(cache)) {
      const { dates, closes, nov } = cache[ticker];
      const rows = [];

      for (const f of nov) {
        if (f[m] == null) continue;
        const i = dates.findIndex((d) => d > f.filed);   // first close strictly after filing
        if (i < WINDOW || i + WINDOW >= closes.length) continue;

        const fwd = realisedVol(closes.slice(i, i + WINDOW));
        const trail = realisedVol(closes.slice(i - WINDOW, i));
        if (!fwd || !trail) continue;
        rows.push({ filed: f.filed, novelty: f[m], ratio: fwd / trail });
      }

      if (rows.length >= 4) {
        const n = rows.map((r) => r.novelty);
        const v = rows.map((r) => Math.log(r.ratio));
        perTicker.push({ ticker, ...permutationP(n, v) });
        pooledNov.push(...zscore(n));                    // z-score within ticker
        pooledVol.push(...zscore(v));                    // before pooling
      } else {
        perTicker.push({ ticker, rho: null, p: null, n: rows.length });
      }
    }
    out[m] = { perTicker, pooled: pooledNov.length >= 8 ? permutationP(pooledNov, pooledVol) : null };
  }

  return { window: WINDOW, metrics: out, missingPrices: missing };
}


/* ─────────────────────────────────────────────── what actually moved

   A number can't tell you whether max fired on a strategy shift or on a new
   legal disclaimer, and those demand opposite conclusions. This returns the
   passage that won, and the passage it was compared against.

   Chunk text isn't stored — only vectors. Rather than re-embed everything to
   add a text column, this re-fetches the filing and re-runs the identical
   extract → strip → chunk path. Chunking is deterministic on the same input,
   so index n here is the same passage as index n at ingest time. */

async function sectionParts(url, kind) {
  const raw = extractSection(await filingText(url), kind);
  if (!raw) return null;
  return chunk(stripBoilerplate(raw)).slice(0, 40);   // must match ingest
}

async function topChunk(params, env) {
  const ticker = params.get("t");
  const filed = params.get("filed");
  const kind = params.get("section") || "mdna";
  const cik = params.get("cik");
  if (!ticker || !filed || !cik) throw new Error("need t, filed, cik");

  const { results } = await env.DB.prepare(
    `SELECT form, filed, chunks FROM sections
     WHERE ticker = ? AND section = ? AND chunks IS NOT NULL ORDER BY filed ASC`
  ).bind(ticker, kind).all();

  const i = results.findIndex((r) => r.filed === filed);
  if (i < 0) throw new Error(`no stored ${kind} for ${ticker} ${filed}`);

  let prev = null;
  for (let j = i - 1; j >= 0; j--) if (results[j].form === results[i].form) { prev = results[j]; break; }
  if (!prev) throw new Error("no previous same-form filing to compare against");

  const cur = unpackVecs(results[i].chunks, EMBED_DIM);
  const old = unpackVecs(prev.chunks, EMBED_DIM);

  // the chunk whose nearest neighbour last quarter is furthest away
  let bestI = 0, bestD = -1, bestJ = 0;
  cur.forEach((a, ci) => {
    let near = 2, nearJ = 0;
    old.forEach((b, pj) => {
      let dot = 0;
      for (let k = 0; k < a.length; k++) dot += a[k] * b[k];
      const d = 1 - dot;
      if (d < near) { near = d; nearJ = pj; }
    });
    if (near > bestD) { bestD = near; bestI = ci; bestJ = nearJ; }
  });

  // recover the text by walking the same path the ingest took
  const filings = await listFilings(cik, 40);
  const findUrl = (d) => filings.find((f) => f.filed === d)?.url;
  const curUrl = findUrl(filed), prevUrl = findUrl(prev.filed);
  if (!curUrl || !prevUrl) throw new Error("filing no longer in the EDGAR index window");

  const curParts = await sectionParts(curUrl, kind);
  const prevParts = await sectionParts(prevUrl, kind);

  return {
    ticker, section: kind, filed, form: results[i].form,
    comparedTo: prev.filed,
    distance: bestD,
    chunkIndex: bestI, of: cur.length,
    text: curParts?.[bestI] ?? "(could not recover text)",
    nearestPrevText: prevParts?.[bestJ] ?? "(could not recover text)",
  };
}


async function status(env) {
  const { results } = await env.DB.prepare(
    `SELECT ticker, section,
            COUNT(*) AS rows,
            SUM(CASE WHEN chunks IS NULL THEN 1 ELSE 0 END) AS missing_chunks,
            MIN(filed) AS first, MAX(filed) AS last
     FROM sections GROUP BY ticker, section ORDER BY ticker, section`
  ).all();
  const prices = (await env.DB.prepare(`SELECT ticker FROM series`).all())
    .results.map((r) => r.ticker);
  return { sections: results, prices };
}

async function reset(ticker, env) {
  await env.DB.prepare(`DELETE FROM sections WHERE ticker = ?`).bind(ticker).run();
  return { cleared: ticker };
}

/* ─────────────────────────────────────────────── page */

const PAGE = `<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Latent Alpha — Phase 0</title>
<style>
:root{
  --ink:#0d0b0b; --panel:#151112; --line:#2a2323;
  --text:#e6e0dd; --mute:#7d7370; --rose:#d6a2a8;
  --mono:ui-monospace,"SF Mono",Menlo,monospace;
}
*{box-sizing:border-box}
body{margin:0;background:var(--ink);color:var(--text);font-family:var(--mono);
  font-size:13px;line-height:1.55;padding:20px 16px 80px;-webkit-text-size-adjust:100%}
h1{font-size:13px;font-weight:500;letter-spacing:.18em;text-transform:uppercase;margin:0}
.sub{color:var(--mute);font-size:11px;letter-spacing:.06em;margin:4px 0 24px}
label{display:block;color:var(--mute);font-size:10px;letter-spacing:.14em;
  text-transform:uppercase;margin:14px 0 5px}
select,button{width:100%;font-family:var(--mono);font-size:13px;padding:11px 12px;
  background:var(--panel);color:var(--text);border:1px solid var(--line);border-radius:2px}
button{margin-top:16px;color:var(--rose);letter-spacing:.14em;text-transform:uppercase;
  font-size:11px;cursor:pointer}
button:disabled{color:var(--mute);cursor:default}
#log{margin-top:18px;color:var(--mute);font-size:11px;white-space:pre-wrap;
  max-height:180px;overflow:auto}
.sec{margin-top:34px;border-top:1px solid var(--line);padding-top:14px}
.sec h2{font-size:10px;letter-spacing:.16em;text-transform:uppercase;color:var(--rose);
  font-weight:500;margin:0 0 12px}
table{width:100%;border-collapse:collapse;font-size:12px}
th{text-align:right;color:var(--mute);font-size:9px;letter-spacing:.1em;
  text-transform:uppercase;font-weight:400;padding-bottom:7px;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}
td{padding:7px 0;border-bottom:1px solid #1c1717}
.bar{display:block;height:3px;background:var(--rose);border-radius:2px;min-width:1px;
  transform:scaleX(0);transform-origin:left;transition:transform .85s cubic-bezier(.22,.9,.24,1)}
.bar.in{transform:scaleX(1)}
tr.insp{cursor:pointer}
tr.insp:active td{color:var(--rose)}
.quote{margin-top:10px;padding:14px 15px;background:#131011;border-left:2px solid var(--rose);
  border-radius:2px;font-size:12px;line-height:1.65;max-height:280px;overflow:auto}
.quote.dim{border-left-color:var(--line);color:var(--mute)}
@media (prefers-reduced-motion:reduce){.bar{transition:none;transform:scaleX(1)}}
.note{color:var(--mute);font-size:11px;margin-top:10px;line-height:1.6}
.dim{color:var(--mute)}
</style></head><body>

<h1>Latent Alpha</h1>
<div class="sub">narrative novelty · sec edgar · gemini embeddings</div>

<label>Company</label>
<select id="co">
  <option value="1045810">NVDA — Nvidia</option>
  <option value="789019">MSFT — Microsoft</option>
  <option value="1652044">GOOGL — Alphabet</option>
  <option value="1326801">META — Meta</option>
  <option value="1018724">AMZN — Amazon</option>
  <option value="2488">AMD</option>
  <option value="1046179">TSM — Taiwan Semi</option>
</select>

<label>Filings</label>
<select id="n"><option>6</option><option selected>10</option><option>14</option></select>

<button id="go">Ingest &amp; measure</button>
<div id="log"></div>
<div id="out"></div>

<div class="sec">
  <h2>Does it predict anything?</h2>
  <div class="note" style="margin:0 0 14px">
    Hypothesis, fixed before looking: filings whose language moved unusually far
    from the same company's previous same-form filing are followed by higher
    realised volatility than that company's own recent baseline.
    Forward window starts the trading day <i>after</i> the filing.
  </div>
  <button id="px">Load prices</button>
  <button id="run">Run test</button>
  <div id="tlog"></div>
  <div id="tout"></div>
</div>

<script>
const $=s=>document.querySelector(s);
const log=m=>{$('#log').textContent+=m+"\\n";$('#log').scrollTop=1e6};

// Every fetch goes through this. A Worker that trips a resource limit returns
// an HTML error page, and .json() on that throws "the string did not match the
// expected pattern" — which tells you nothing and killed whole runs.
async function api(url,opts){
  const res=await fetch(url,opts);
  const raw=await res.text();
  try{ return JSON.parse(raw); }
  catch{ return {error:'HTTP '+res.status+' — '+raw.slice(0,90).replace(/\\s+/g,' ')}; }
}

$('#go').onclick=async()=>{
  const btn=$('#go'); btn.disabled=true;
  $('#log').textContent=''; $('#out').innerHTML='';
  const cik=$('#co').value, ticker=$('#co').selectedOptions[0].text.split(' ')[0], n=$('#n').value;

  try{
    log('fetching filing index…');
    const filings=await api('/api/filings?cik='+cik+'&n='+n);
    if(filings.error) throw new Error(filings.error);
    log(filings.length+' filings · '+filings[0].filed+' → '+filings[filings.length-1].filed+'\\n');

    let quota=false;
    outer:
    for(const f of filings){
      log(f.filed+'  '+f.form);
      // one section per request — two at once was enough work to trip the
      // Worker resource limit on the bigger filings
      for(const section of ['mdna','risk']){
        const r=await api('/api/ingest',{method:'POST',
          headers:{'content-type':'application/json'},
          body:JSON.stringify({ticker,section,...f})});

        if(r.error){
          if(r.error==='QUOTA'){ quota=true; log('   ! Gemini quota reached — stopping'); break outer; }
          log('   '+section+' ! '+r.error); continue;
        }
        if(r.skipped){ log('   '+section+' cached'); continue; }
        const s=r.sections[0];
        log('   '+section+' '+(s&&s.chars ? s.chars.toLocaleString()+'c · '+s.chunks+' chunks' : (s?s.status:'—')));
      }
    }
    if(quota) log('\\nQuota resets daily. Press again later — stored filings are skipped.');

    log('\\ncomputing novelty…');
    const nov=await api('/api/novelty?t='+ticker);
    if(nov.error) throw new Error(nov.error);
    render(nov, ticker);
  }catch(e){ log('ERROR: '+e.message); }
  btn.disabled=false;
};

$('#px').onclick=async()=>{
  const b=$('#px'); b.disabled=true; $('#tlog').textContent='';
  const tlog=m=>{$('#tlog').textContent+=m+"\\n"};
  const tickers=[...$('#co').options].map(o=>o.text.split(' ')[0]);
  for(const t of tickers){
    const r=await api('/api/prices?t='+t,{method:'POST'});
    tlog(r.error ? t+'  ! '+r.error : t+'  '+r.days+' days  '+r.from+' → '+r.to+'  ·  '+r.source);
  }
  b.disabled=false;
};

$('#run').onclick=async()=>{
  const b=$('#run'); b.disabled=true; $('#tout').innerHTML='';
  $('#tlog').textContent='running permutation test…';
  const d=await api('/api/test?t=ALL');
  if(d.error){ $('#tlog').textContent='ERROR: '+d.error; b.disabled=false; return; }
  $('#tlog').textContent = d.missingPrices.length ? 'no prices for: '+d.missingPrices.join(' ') : '';
  renderTest(d);
  b.disabled=false;
};

function renderTest(d){
  const names={max:'max shift',p75:'p75 shift',pooled:'pooled'};
  let h='<table><tr><th>Metric</th><th>n</th><th>ρ</th><th>p</th></tr>';
  for(const m of ['max','p75','pooled']){
    const p=d.metrics[m].pooled;
    h+='<tr><td>'+names[m]+'</td>'+
       '<td class="dim">'+(p?p.n:'—')+'</td>'+
       '<td>'+(p?p.rho.toFixed(3):'—')+'</td>'+
       '<td class="'+(p&&p.p<0.05?'':'dim')+'">'+(p?p.p.toFixed(4):'—')+'</td></tr>';
  }
  h+='</table>';

  h+='<div class="sec"><h2>Per company</h2><table>'+
     '<tr><th>Ticker</th><th>n</th><th>ρ max</th><th>p max</th></tr>';
  for(const t of d.metrics.max.perTicker){
    h+='<tr><td>'+t.ticker+'</td><td class="dim">'+t.n+'</td>'+
       '<td>'+(t.rho!=null?t.rho.toFixed(3):'—')+'</td>'+
       '<td class="dim">'+(t.p!=null?t.p.toFixed(3):'too few')+'</td></tr>';
  }
  h+='</table></div>';

  const any=['max','p75','pooled'].some(m=>d.metrics[m].pooled&&d.metrics[m].pooled.p<0.05);
  h+='<div class="note">All three metrics run every time and are all reported, so none of '+
     'them was chosen after the fact. '+(any
       ? 'Something is under 0.05 — but three tests means roughly three chances at a false '+
         'positive, so the honest threshold here is about 0.017, and these companies move '+
         'together so the effective sample is smaller than n.'
       : 'Nothing is significant. At this sample size that is closer to "no data" than to '+
         '"no effect" — the fix is more companies, not a different metric.')+
     '</div>';
  $('#tout').innerHTML=h;
}

function render(data,ticker){
  const titles={mdna:'MD&A',risk:'Risk Factors'};
  let html='';
  for(const k of Object.keys(data)){
    const rows=data[k]; if(!rows.length) continue;
    const max=Math.max(0.02,...rows.map(r=>r.max||0));
    html+='<div class="sec"><h2>'+ticker+' · '+titles[k]+'</h2><table>'+
      '<tr><th>Filed</th><th>Form</th><th>Pooled</th><th>Max</th><th>p75</th><th style="width:22%"></th></tr>';
    for(const r of rows){
      const f=v=>v!=null?v.toFixed(4):'—';
      html+='<tr'+(r.max!=null?' class="insp" data-filed="'+r.filed+'" data-sec="'+k+'"':'')+'>'+
        '<td>'+r.filed+(r.max!=null?' <span class="dim">▸</span>':'')+'</td>'+
        '<td class="dim">'+r.form+'</td>'+
        '<td class="dim">'+f(r.pooled)+'</td>'+
        '<td>'+f(r.max)+'</td>'+
        '<td class="dim">'+f(r.p75)+'</td>'+
        '<td>'+(r.max!=null?'<span class="bar" style="width:'+(r.max/max*100)+'%"></span>':'')+'</td></tr>';
    }
    html+='</table></div>';
  }
  html+='<div class="note"><b>Max</b> is the distance from each passage in this filing to its '+
    'nearest passage in the previous same-form filing, taking the largest — "is there something '+
    'here with no counterpart last quarter?" <b>Pooled</b> averages the whole section, which '+
    'drowns one changed paragraph in the thirty that did not change. <b>p75</b> only rises when '+
    'much of the section is rewritten.<br><br>'+
    '<b>Tap any row</b> to read the passage that produced the max. A number cannot tell you '+
    'whether it fired on a strategy shift or on a new legal disclaimer, and those mean opposite '+
    'things.</div><div id="chunk"></div>';
  $('#out').innerHTML=html;
  requestAnimationFrame(()=>document.querySelectorAll('.bar').forEach(b=>b.classList.add('in')));

  document.querySelectorAll('tr.insp').forEach(tr=>{
    tr.onclick=async()=>{
      const box=$('#chunk');
      box.innerHTML='<div class="note">reading filing…</div>';
      box.scrollIntoView({behavior:'smooth',block:'nearest'});
      const d=await api('/api/topchunk?t='+ticker+'&cik='+$('#co').value+
        '&filed='+tr.dataset.filed+'&section='+tr.dataset.sec);
      if(d.error){ box.innerHTML='<div class="note">'+d.error+'</div>'; return; }
      const esc=s=>s.replace(/[<>&]/g,c=>({'<':'&lt;','>':'&gt;','&':'&amp;'}[c]));
      box.innerHTML='<div class="sec"><h2>'+d.filed+' · passage '+(d.chunkIndex+1)+' of '+d.of+
        ' · distance '+d.distance.toFixed(4)+'</h2>'+
        '<div class="quote">'+esc(d.text)+'</div>'+
        '<h2 style="margin-top:22px">nearest passage in '+d.comparedTo+'</h2>'+
        '<div class="quote dim">'+esc(d.nearestPrevText)+'</div></div>';
    };
  });
}
</script></body></html>`;