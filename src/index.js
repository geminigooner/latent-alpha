/**
 * LATENT ALPHA OBSERVATORY — Phase 0
 * Narrative novelty from SEC filings.
 *
 * Routes
 *   GET  /                  the page
 *   GET  /api/filings?cik=  list recent 10-K/10-Q from EDGAR
 *   POST /api/extract       fetch ONE filing once, extract & chunk sections
 *   POST /api/ingest        embed chunks for ONE section, store in D1
 *   GET  /api/novelty?t=    read D1, compute & cache novelty series
 *   GET  /api/prices?t=     fetch daily closes (Yahoo -> Stooq) into D1
 *   GET  /api/test          pooled + per-company Spearman permutation test
 *   POST /api/reset?t=      wipe one ticker
 *
 * Split into extract + per-section ingest on purpose — keeps every invocation
 * comfortably under the Workers 10ms Free-tier CPU & 128MB memory limits.
 */

const SEC_UA = "Amanda Hatley ahatley094@gmail.com";
const EMBED_MODEL = "gemini-embedding-001";
const EMBED_DIM = 1536;
const CHUNK_CHARS = 6000;
const MAX_TEXT = 900_000; // visible text cap (hidden XBRL & tag indentation excluded)
const B64_PER_CHUNK = (EMBED_DIM * 4 / 3) * 4; // 8192 base64 chars per 1536-float32 chunk

const COMPANIES = [
  { ticker: "NVDA", name: "Nvidia", cik: "0001045810" },
  { ticker: "GOOGL", name: "Alphabet", cik: "0001652044" },
  { ticker: "MSFT", name: "Microsoft", cik: "0000789019" },
  { ticker: "META", name: "Meta", cik: "0001326801" },
  { ticker: "AMZN", name: "Amazon", cik: "0001018724" },
  { ticker: "AMD", name: "AMD", cik: "0000002488" },
  { ticker: "AVGO", name: "Broadcom", cik: "0001730168" },
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
        case "/api/extract":  return json(await extractFiling(await request.json(), env));
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
      return json({ error: String((err && err.message) || err) }, 500);
    }
  },
};

const json = (obj, status = 200) =>
  new Response(JSON.stringify(obj), { status, headers: { "content-type": "application/json" } });

async function ensureTables(env) {
  await env.DB.batch([
    env.DB.prepare(
      `CREATE TABLE IF NOT EXISTS series (ticker TEXT PRIMARY KEY, updated TEXT NOT NULL, data TEXT NOT NULL)`
    ),
    env.DB.prepare(
      `CREATE TABLE IF NOT EXISTS shift_cache (key TEXT PRIMARY KEY, max REAL, p75 REAL)`
    ),
  ]);
}

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

/**
 * Filing HTML -> flat visible text, streamed without buffering output HTML.
 * Skips <script>, <style>, and hidden iXBRL metadata blocks (display:none / hidden).
 */
async function filingText(url) {
  const res = await fetch(url, { headers: { "User-Agent": SEC_UA, "Accept-Encoding": "gzip" } });
  if (!res.ok) throw new Error(`filing fetch ${res.status}`);

  const parts = [];
  let skip = 0, len = 0, cur = "";

  const rewritten = new HTMLRewriter()
    .on('script, style, [hidden], [style*="display:none"], [style*="display: none"]', {
      element(el) { skip++; el.onEndTag(() => { skip--; }); },
    })
    .onDocument({
      text(t) {
        if (skip || len >= MAX_TEXT) return;
        if (t.text) cur += t.text;
        if (t.lastInTextNode && cur) {
          const trimmed = cur.trim();
          if (trimmed) {
            parts.push(trimmed);
            len += trimmed.length + 1;
          }
          cur = "";
        }
      },
    })
    .transform(res);

  // Drain chunk-by-chunk and discard immediately so 25MB+ 10-Ks never buffer in RAM
  const reader = rewritten.body.getReader();
  while (true) {
    const { done } = await reader.read();
    if (done || len >= MAX_TEXT) {
      await reader.cancel();
      break;
    }
  }
  if (cur.trim() && len < MAX_TEXT) parts.push(cur.trim());

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
  if (typeof bytes.toBase64 === "function") return bytes.toBase64();
  let s = "";
  for (let i = 0; i < bytes.length; i += 0x8000) {
    s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  }
  return btoa(s);
}

function unpackVecs(b64, d) {
  let bytes;
  if (typeof Uint8Array.fromBase64 === "function") {
    bytes = Uint8Array.fromBase64(b64);
  } else {
    const bin = atob(b64);
    bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  }
  const f = new Float32Array(bytes.buffer, bytes.byteOffset, Math.floor(bytes.byteLength / 4));
  const out = [];
  for (let i = 0; i + d <= f.length; i += d) out.push(f.subarray(i, i + d));
  return out;
}

const cosine = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };

/* ─────────────────────────────────────────────── extract + ingest */

/**
 * Step 1: Fetch the SEC filing ONCE, extract MD&A + Risk Factors, return text chunks.
 * Keeps HTML streaming in a separate invocation from embedding & vector packing.
 */
async function extractFiling(body, env) {
  const { ticker, form, filed, url } = body;
  const kinds = ["mdna", "risk"];
  const keys = kinds.map((k) => `${ticker}:${filed}:${form}:${k}`);
  const have = await env.DB.prepare(
    `SELECT key FROM sections WHERE chunks IS NOT NULL AND key IN (${keys.map(() => "?").join(",")})`
  ).bind(...keys).all();
  const stored = new Set(have.results.map((r) => r.key));
  if (stored.size === keys.length) {
    return { filed, form, cached: true, sections: { mdna: { cached: true }, risk: { cached: true } } };
  }

  const text = await filingText(url);
  const sections = {};
  for (const kind of kinds) {
    if (stored.has(`${ticker}:${filed}:${form}:${kind}`)) {
      sections[kind] = { cached: true };
      continue;
    }
    const raw = extractSection(text, kind, form);
    if (!raw) { sections[kind] = { status: "not found" }; continue; }
    const clean = stripBoilerplate(raw);
    const parts = chunk(clean).slice(0, 40);
    if (!parts.length) { sections[kind] = { status: "not found" }; continue; }
    sections[kind] = { status: "ready", chars: clean.length, parts };
  }
  return { filed, form, cached: false, textLen: text.length, sections };
}

/**
 * Step 2: Embed chunks for a single section and store in D1.
 * Also supports legacy direct { ticker, form, filed, url, section } calls.
 */
async function ingest(body, env) {
  const { ticker, form, filed, url, section, parts: preParts, chars: preChars } = body;
  if (!env.GEMINI_API_KEY) throw new Error("GEMINI_API_KEY secret not set");
  await ensureTables(env);

  if (section && Array.isArray(preParts)) {
    if (!preParts.length) return { filed, form, sections: [{ kind: section, status: "not found" }] };
    const rawVecs = await embed(preParts, env.GEMINI_API_KEY);
    const unit = rawVecs.map(normalize);
    const cleanChars = preChars || preParts.reduce((a, s) => a + s.length, 0);

    await env.DB.batch([
      env.DB.prepare(
        `INSERT OR REPLACE INTO sections (key, ticker, form, filed, section, chars, vec, chunks)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?)`
      ).bind(
        `${ticker}:${filed}:${form}:${section}`, ticker, form, filed, section, cleanChars,
        JSON.stringify(poolNormalize(rawVecs)), packVecs(unit)
      ),
      env.DB.prepare(`DELETE FROM shift_cache WHERE key LIKE ?`).bind(`${ticker}:${section}:%`),
    ]);

    return { filed, form, sections: [{ kind: section, status: "ok", chars: cleanChars, chunks: preParts.length }] };
  }

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

    await env.DB.batch([
      env.DB.prepare(
        `INSERT OR REPLACE INTO sections (key, ticker, form, filed, section, chars, vec, chunks)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?)`
      ).bind(
        `${ticker}:${filed}:${form}:${kind}`, ticker, form, filed, kind, clean.length,
        JSON.stringify(poolNormalize(rawVecs)), packVecs(unit)
      ),
      env.DB.prepare(`DELETE FROM shift_cache WHERE key LIKE ?`).bind(`${ticker}:${kind}:%`),
    ]);

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

async function novelty(ticker, env, onlyKinds = ["mdna", "risk"]) {
  await ensureTables(env);

  // Read lightweight metadata + pooled vector first; only pull 330KB chunk blobs on cache miss
  const { results: metaRows } = await env.DB.prepare(
    `SELECT key, form, filed, section, chars, vec, LENGTH(chunks) AS chunks_len
     FROM sections WHERE ticker = ? ORDER BY filed ASC`
  ).bind(ticker).all();

  const { results: cachedRows } = await env.DB.prepare(
    `SELECT key, max, p75 FROM shift_cache WHERE key LIKE ?`
  ).bind(`${ticker}:%`).all();
  const shiftMap = new Map(cachedRows.map((r) => [r.key, r]));

  const out = {};
  for (const kind of onlyKinds) {
    const rows = metaRows.filter((r) => r.section === kind).map((r) => ({
      ...r,
      v: JSON.parse(r.vec),
      nChunks: r.chunks_len ? Math.round(r.chunks_len / B64_PER_CHUNK) : 0,
    }));

    const needBlobKeys = new Set();
    const pairs = rows.map((r, i) => {
      let prev = null;
      for (let j = i - 1; j >= 0; j--) if (rows[j].form === r.form) { prev = rows[j]; break; }
      const cacheKey = prev ? `${ticker}:${kind}:${r.filed}:${prev.filed}:${r.form}` : null;
      if (prev && r.nChunks > 0 && prev.nChunks > 0 && !shiftMap.has(cacheKey)) {
        needBlobKe