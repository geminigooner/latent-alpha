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

/* Some filings double-encode, so HTMLRewriter hands back a literal "&#8217;".
   Left alone it shows up in the passage viewer and pollutes the embeddings. */
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

/**
 * Form-aware section extraction.
 * 10-K: MD&A is Item 7, Risk Factors is Item 1A
 * 10-Q: MD&A is Part I Item 2, Risk Factors is Part II Item 1A
 */
const SECTIONS = {
  "10-K": {
    mdna: {
      label: "MD&A",
      start: ["item 7.", "item 7 .", "item7."],
      end: ["item 7a.", "item 7a .", "item 8.", "item 8 ."],
      fallbackStart: ["s discussion and analysis"],
    },
    risk: {
      label: "Risk Factors",
      start: ["item 1a.", "item 1a .", "item1a."],
      end: ["item 1b.", "item 1b .", "item 2.", "item 2 ."],
      fallbackStart: ["risk factors"],
    },
  },
  "10-Q": {
    mdna: {
      label: "MD&A",
      start: ["item 2.", "item 2 .", "item2."],
      end: ["item 3.", "item 3 .", "item 4.", "item 4 .", "part ii"],
      fallbackStart: ["s discussion and analysis"],
    },
    risk: {
      label: "Risk Factors",
      start: ["item 1a.", "item 1a .", "item1a."],
      end: ["item 2.", "item 2 .", "item 3.", "item 3 .", "item 6.", "item 6 .", "signatures"],
      fallbackStart: ["risk factors"],
    },
  },
};

function findAll(hay, needle) {
  const out = [];
  let i = hay.indexOf(needle);
  while (i !== -1) { out.push(i); i = hay.indexOf(needle, i + needle.length); }
  return out;
}

const HEADER_PREFIX = /item\s*\d{1,2}\s*[a-c]?\s*[.:)\-–—]?\s*$/;

function extractSection(text, kind, form) {
  const spec = SECTIONS[form]?.[kind];
  if (!spec) return null;
  
  const low = text.toLowerCase().replace(/['\u2019`]/g, " ");

  // Collect all possible markers for this form type to check for TOC density
  const allMarkers = [...spec.start, ...spec.end, ...spec.fallbackStart];
  const all = allMarkers.flatMap((m) => findAll(low, m)).sort((a, b) => a - b);
  
  // If 4+ markers appear within 1500 chars, it's a Table of Contents, ignore them.
  const inTOC = (p) => all.filter((q) => Math.abs(q - p) < 1500).length >= 4;
  
  // Find real section starts
  let starts = spec.start.flatMap((m) => findAll(low, m)).sort((a, b) => a - b).filter((p) => !inTOC(p));
  if (!starts.length) {
    starts = spec.fallbackStart.flatMap((m) => findAll(low, m)).sort((a, b) => a - b).filter((p) => !inTOC(p));
  }
  if (!starts.length) return null;
  
  const ends = spec.end.flatMap((m) => findAll(low, m)).sort((a, b) => a - b).filter((p) => !inTOC(p));

  // Find the longest valid span that doesn't swallow another Item header
  const headers = all.filter((p) => HEADER_PREFIX.test(low.slice(Math.max(0, p - 40), p)));
  
  let best = null;
  for (const s of starts) {
    const e = ends.find((x) => x > s + 200);
    const stop = e ?? Math.min(s + 120_000, text.length);
    if (headers.some((o) => o > s + 200 && o < stop - 50)) continue; // Swallowed a different section
    if (!best || stop - s > best[1] - best[0]) best = [s, stop];
  }
  
  if (!best) return null;

  const body = text.slice(best[0], best[1]);
  
  // 10-Q Risk Factors are sometimes just "There have been no material changes..."
  // If it's a 10-Q and it's short, allow it through (min 100 chars instead of 1500)
  const minLength = form === "10-Q" && kind === "risk" ? 100 : 1500;
  
  return body.length > minLength ? body : null;
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

  // Only try to ingest if we actually have specs for this form type
  if (!SECTIONS[form]) throw new Error(`Unsupported form type: ${form}`);

  const kinds = section && SECTIONS[form][section] ? [section] : Object.keys(SECTIONS[form]);

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
    const raw = extractSection(text, kind, form);
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
  
  // We extract keys from both 10-K and 10-Q since they share the 'mdna' and 'risk' namespaces
  const allKinds = [...new Set([
    ...Object.keys(SECTIONS["10-K"]), 
    ...Object.keys(SECTIONS["10-Q"])
  ])];

  for (const kind of allKinds) {
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

/* ─────────
