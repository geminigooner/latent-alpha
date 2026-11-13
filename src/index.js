/**
 * LATENT ALPHA OBSERVATORY — Phase 0
 * Narrative novelty from SEC filings · Dual-model (Gemma 4 Workers AI + Gemini API).
 *
 * Routes
 *   GET  /                    the page
 *   GET  /api/filings?cik=    list recent 10-K/10-Q from EDGAR
 *   GET  /api/raw?url=        zero-CPU SEC stream passthrough (browser fallback for giant 10-Ks)
 *   POST /api/extract         fetch ONE filing once, extract & chunk sections
 *   POST /api/ingest          embed chunks for ONE section (Gemma 4 or Gemini), store in D1
 *   GET  /api/novelty?t=&m=   read D1, compute & cache novelty series for model m
 *   GET  /api/prices?t=       fetch daily closes (Yahoo -> Stooq) into D1
 *   GET  /api/test?m=         pooled + per-company Spearman permutation test for model m
 *   POST /api/reset?t=&m=     wipe one ticker for model m
 */

const SEC_UA = "Amanda Hatley ahatley094@gmail.com";
const CHUNK_CHARS = 6000;
const MAX_TEXT = 650_000;

const MODELS = {
  gemma2: {
    id: "gemma2",
    label: "Gemma 4 · EmbeddingGemma 2 (Workers AI · 768d)",
    short: "EmbeddingGemma 2 (Gemma 4 · 768d)",
    provider: "workers-ai",
    modelName: "@cf/google/embeddinggemma-2",
    dim: 768,
  },
  gemini: {
    id: "gemini",
    label: "Gemini · gemini-embedding-001 (API · 1536d)",
    short: "gemini-embedding-001 (1536d)",
    provider: "gemini-api",
    modelName: "gemini-embedding-001",
    dim: 1536,
  },
};

const getModel = (m) => MODELS[m] || MODELS.gemini;
const secKey = (mId, ticker, filed, form, kind) =>
  mId === "gemini" ? `${ticker}:${filed}:${form}:${kind}` : `${mId}:${ticker}:${filed}:${form}:${kind}`;
const secPrefix = (mId, ticker) =>
  mId === "gemini" ? `${ticker}:%` : `${mId}:${ticker}:%`;
const shiftKey = (mId, ticker, kind, filed, prevFiled, form) =>
  `${mId}:${ticker}:${kind}:${filed}:${prevFiled}:${form}`;
const shiftPrefix = (mId, ticker, kind = "") =>
  `${mId}:${ticker}:${kind ? kind + ":" : ""}%`;

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
                                "cache-control": "no-store, must-revalidate",
                              } });
        case "/api/filings":  return json(await listFilings(url.searchParams.get("cik"), +(url.searchParams.get("n") || 10)));
        case "/api/raw":      return proxySec(url.searchParams.get("url"));
        case "/api/extract":  return json(await extractFiling(await request.json(), env));
        case "/api/ingest":   return json(await ingest(await request.json(), env));
        case "/api/novelty":  return json(await novelty(url.searchParams.get("t"), env, ["mdna", "risk"], url.searchParams.get("m")));
        case "/api/prices":   return json(await loadPrices(url.searchParams.get("t"), env));
        case "/api/test":     return json(await volTest(url.searchParams.get("t"), env, url.searchParams.get("m")));
        case "/api/reset":    return json(await reset(url.searchParams.get("t"), env, url.searchParams.get("m")));
        case "/api/status":   return json(await status(env));
        case "/api/topchunk": return json(await topChunk(url.searchParams, env));
        case "/export":       return new Response(await exportText(env, url.searchParams.get("m")), { headers: {
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
      `CREATE TABLE IF NOT EXISTS sections (
         key TEXT PRIMARY KEY, ticker TEXT NOT NULL, form TEXT NOT NULL,
         filed TEXT NOT NULL, section TEXT NOT NULL, chars INTEGER NOT NULL,
         vec TEXT NOT NULL, chunks TEXT
       )`
    ),
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
  return out.reverse();
}

/** Zero-CPU stream passthrough restricted strictly to SEC EDGAR archives. */
async function proxySec(secUrl) {
  if (!secUrl || !secUrl.startsWith("https://www.sec.gov/Archives/edgar/data/")) {
    throw new Error("invalid SEC url");
  }
  const r = await fetch(secUrl, { headers: { "User-Agent": SEC_UA, "Accept-Encoding": "gzip" } });
  if (!r.ok) throw new Error(`SEC fetch ${r.status}`);
  return new Response(r.body, {
    headers: { "content-type": "text/plain;charset=utf-8", "cache-control": "no-store" },
  });
}

const BLOCK_OPEN_RE = /<(ix:header|ix:hidden|script|style)\b/i;

/**
 * Chunked HTML -> visible text stripper with zero per-tag JS callbacks.
 * Jumps straight over multi-megabyte <ix:header> XBRL blocks in C++ indexOf.
 */
async function filingText(url) {
  const res = await fetch(url, { headers: { "User-Agent": SEC_UA, "Accept-Encoding": "gzip" } });
  if (!res.ok) throw new Error(`filing fetch ${res.status}`);

  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  const parts = [];
  let carry = "";
  let waitClose = null;
  let len = 0;

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    let s = carry + value;
    carry = "";

    while (s.length > 0) {
      if (waitClose) {
        const idx = s.toLowerCase().indexOf(waitClose);
        if (idx === -1) {
          carry = s.slice(-16);
          s = "";
          break;
        }
        const gt = s.indexOf(">", idx + waitClose.length);
        if (gt === -1) {
          carry = s.slice(idx);
          s = "";
          break;
        }
        s = s.slice(gt + 1);
        waitClose = null;
        continue;
      }

      const m = BLOCK_OPEN_RE.exec(s);
      if (m) {
        const before = s.slice(0, m.index);
        if (before) {
          const txt = before.replace(/<[^>]*>/g, " ").replace(/\s+/g, " ").trim();
          if (txt) { parts.push(txt); len += txt.length + 1; }
        }
        waitClose = "</" + m[1].toLowerCase();
        s = s.slice(m.index + m[0].length);
        continue;
      }

      const lastLt = s.lastIndexOf("<");
      const lastGt = s.lastIndexOf(">");
      if (lastLt > lastGt) {
        carry = s.slice(lastLt);
        s = s.slice(0, lastLt);
      } else {
        const tailAmp = s.lastIndexOf("&");
        if (tailAmp !== -1 && s.length - tailAmp < 12 && s.indexOf(";", tailAmp) === -1) {
          carry = s.slice(tailAmp);
          s = s.slice(0, tailAmp);
        }
      }

      if (s) {
        const txt = s.replace(/<[^>]*>/g, " ").replace(/\s+/g, " ").trim();
        if (txt) { parts.push(txt); len += txt.length + 1; }
      }
      break;
    }

    if (len >= MAX_TEXT) {
      await reader.cancel();
      break;
    }
  }

  return decodeEntities(parts.join(" "))
    .replace(/\u00a0/g, " ")
    .replace(/\s+/g, " ")
    .trim();
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

/** Pre-indexes markers & Item headers ONCE per filing so MD&A + Risk don't scan twice. */
function prepareDocIndex(text, form = "10-K") {
  const formSpec = SECTIONS[form] || SECTIONS["10-K"];
  const low = text.toLowerCase().replace(/['\u2019`]/g, " ");
  const allMarkers = Object.values(formSpec).flatMap((s) => [...s.start, ...s.end]);
  const all = [...new Set(allMarkers.flatMap((m) => findAll(low, m)))].sort((a, b) => a - b);
  const inTOC = (p) => all.filter((q) => Math.abs(q - p) < 1500).length >= 5;

  const itemHeaders = [];
  let m;
  ITEM_HEADER_RE.lastIndex = 0;
  while ((m = ITEM_HEADER_RE.exec(low)) !== null) {
    if (!inTOC(m.index)) itemHeaders.push(m.index);
  }
  return { text, low, form, formSpec, inTOC, itemHeaders };
}

function extractFromIndex(docIdx, kind) {
  const { text, low, form, formSpec, inTOC, itemHeaders } = docIdx;
  const spec = formSpec[kind];
  if (!spec) return null;

  const starts = [...new Set(spec.start.flatMap((k) => findAll(low, k)))]
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

function extractSection(text, kind, form = "10-K") {
  return extractFromIndex(prepareDocIndex(text, form), kind);
}

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
    i = end - size;
  }
  return out.filter((s) => s.trim().length > 50);
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function embedGemini(texts, apiKey, m) {
  if (!apiKey) throw new Error("GEMINI_API_KEY secret not set");
  const vecs = [];
  for (let i = 0; i < texts.length; i += 20) {
    const batch = texts.slice(i, i + 20);
    const body = JSON.stringify({
      requests: batch.map((text) => ({
        model: `models/${m.modelName}`,
        content: { parts: [{ text }] },
        taskType: "SEMANTIC_SIMILARITY",
        outputDimensionality: m.dim,
      })),
    });

    let r, attempt = 0;
    while (true) {
      r = await fetch(
        `https://generativelanguage.googleapis.com/v1beta/models/${m.modelName}:batchEmbedContents`,
        { method: "POST", headers: { "content-type": "application/json", "x-goog-api-key": apiKey }, body }
      );
      if (r.ok) break;
      if (r.status === 429 && attempt < 3) { attempt++; await sleep(2000 * attempt); continue; }
      throw new Error(r.status === 429 ? "QUOTA" : `gemini embed ${r.status}`);
    }

    for (const e of (await r.json()).embeddings) vecs.push(e.values);
    await sleep(250);
  }
  return vecs;
}

async function embedGemma(texts, ai, m) {
  if (!ai) {
    throw new Error('Workers AI binding (env.AI) missing — check [ai] binding = "AI" in wrangler.toml');
  }
  const vecs = [];
  for (let i = 0; i < texts.length; i += 10) {
    const batch = texts.slice(i, i + 10).map(
      (t) => "task: sentence similarity | query: " + t.trim()
    );

    let res, attempt = 0;
    while (true) {
      try {
        res = await ai.run(m.modelName, { input: batch });
        break;
      } catch (e) {
        if (attempt < 2) { attempt++; await sleep(1000 * attempt); continue; }
        throw new Error("gemma embed: " + ((e && e.message) || e));
      }
    }

    const items = (res && res.data) || [];
    if (items.length !== batch.length) {
      throw new Error(`gemma embed: expected ${batch.length} vectors, got ${items.length}`);
    }
    const ordered = items.slice().sort((a, b) => (a.index ?? 0) - (b.index ?? 0));
    for (const item of ordered) {
      const v = item.embedding || item;
      if (!v || !v.length) throw new Error("gemma embed: empty vector");
      vecs.push(v);
    }
  }
  return vecs;
}

async function embedChunks(texts, m, env) {
  return m.provider === "workers-ai"
    ? embedGemma(texts, env.AI, m)
    : embedGemini(texts, env.GEMINI_API_KEY, m);
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

function decodeB64Bytes(s) {
  if (typeof Uint8Array.fromBase64 === "function") return Uint8Array.fromBase64(s);
  const bin = atob(s);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes;
}

function parseVec(raw) {
  if (!raw || typeof raw !== "string") return null;
  const s = raw.trim();
  if (!s) return null;
  if (s[0] === "[") {
    try { return normalize(JSON.parse(s)); } catch { return null; }
  }
  try {
    const bytes = decodeB64Bytes(s);
    if (bytes.byteLength < 16 || bytes.byteLength % 4 !== 0) return null;
    return normalize(new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4));
  } catch {
    return null;
  }
}

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
  if (!b64 || typeof b64 !== "string") return [];
  const s = b64.trim();
  if (s[0] === "[") {
    try {
      return JSON.parse(s).map((v) => normalize(v)).filter((v) => v.length === d);
    } catch { return []; }
  }
  try {
    const bytes = decodeB64Bytes(s);
    const f = new Float32Array(bytes.buffer, bytes.byteOffset, Math.floor(bytes.byteLength / 4));
    const out = [];
    for (let i = 0; i + d <= f.length; i += d) out.push(f.subarray(i, i + d));
    return out;
  } catch {
    return [];
  }
}

const cosine = (a, b) => {
  if (!a || !b || a.length !== b.length) return 0;
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
};

/* ─────────────────────────────────────────────── extract + ingest */

async function extractFiling(body, env) {
  const { ticker, form, filed, url, model, preText } = body;
  const m = getModel(model);
  await ensureTables(env);

  const kinds = ["mdna", "risk"];
  const keys = kinds.map((k) => secKey(m.id, ticker, filed, form, k));
  const have = await env.DB.prepare(
    `SELECT key, vec FROM sections WHERE chunks IS NOT NULL AND key IN (${keys.map(() => "?").join(",")})`
  ).bind(...keys).all();
  const stored = new Set(
    have.results
      .filter((r) => { const v = parseVec(r.vec); return v && v.length === m.dim; })
      .map((r) => r.key)
  );
  if (stored.size === keys.length) {
    return { filed, form, model: m.id, cached: true, sections: { mdna: { cached: true }, risk: { cached: true } } };
  }

  const text = typeof preText === "string" ? preText : await filingText(url);
  const docIdx = prepareDocIndex(text, form);
  const sections = {};
  for (const kind of kinds) {
    if (stored.has(secKey(m.id, ticker, filed, form, kind))) {
      sections[kind] = { cached: true };
      continue;
    }
    const raw = extractFromIndex(docIdx, kind);
    if (!raw) { sections[kind] = { status: "not found" }; continue; }
    const clean = stripBoilerplate(raw);
    const parts = chunk(clean).slice(0, 40);
    if (!parts.length) { sections[kind] = { status: "not found" }; continue; }
    sections[kind] = { status: "ready", chars: clean.length, parts };
  }
  return { filed, form, model: m.id, cached: false, textLen: text.length, sections };
}

async function ingest(body, env) {
  const { ticker, form, filed, url, section, parts: preParts, chars: preChars, model } = body;
  const m = getModel(model);
  await ensureTables(env);

  if (section && Array.isArray(preParts)) {
    if (!preParts.length) return { filed, form, model: m.id, sections: [{ kind: section, status: "not found" }] };
    const rawVecs = await embedChunks(preParts, m, env);
    const unit = rawVecs.map(normalize);
    const cleanChars = preChars || preParts.reduce((a, s) => a + s.length, 0);

    await env.DB.batch([
      env.DB.prepare(
        `INSERT OR REPLACE INTO sections (key, ticker, form, filed, section, chars, vec, chunks)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?)`
      ).bind(
        secKey(m.id, ticker, filed, form, section), ticker, form, filed, section, cleanChars,
        JSON.stringify(poolNormalize(rawVecs)), packVecs(unit)
      ),
      env.DB.prepare(`DELETE FROM shift_cache WHERE key LIKE ?`).bind(shiftPrefix(m.id, ticker, section)),
    ]);

    return { filed, form, model: m.id, sections: [{ kind: section, status: "ok", chars: cleanChars, chunks: preParts.length }] };
  }

  const formSpec = SECTIONS[form] || SECTIONS["10-K"];
  const kinds = section && formSpec[section] ? [section] : ["mdna", "risk"];
  const keys = kinds.map((k) => secKey(m.id, ticker, filed, form, k));
  const have = await env.DB.prepare(
    `SELECT key, vec FROM sections WHERE chunks IS NOT NULL AND key IN (${keys.map(() => "?").join(",")})`
  ).bind(...keys).all();
  const stored = new Set(
    have.results
      .filter((r) => { const v = parseVec(r.vec); return v && v.length === m.dim; })
      .map((r) => r.key)
  );
  if (stored.size === keys.length) {
    return { filed, form, model: m.id, skipped: true, sections: kinds.map((k) => ({ kind: k, status: "cached" })) };
  }

  const text = await filingText(url);
  const docIdx = prepareDocIndex(text, form);
  const done = [];

  for (const kind of kinds) {
    if (stored.has(secKey(m.id, ticker, filed, form, kind))) { done.push({ kind, status: "cached" }); continue; }
    const raw = extractFromIndex(docIdx, kind);
    if (!raw) { done.push({ kind, status: "not found" }); continue; }

    const clean = stripBoilerplate(raw);
    const parts = chunk(clean).slice(0, 40);
    if (!parts.length) { done.push({ kind, status: "not found" }); continue; }
    const rawVecs = await embedChunks(parts, m, env);
    const unit = rawVecs.map(normalize);

    await env.DB.batch([
      env.DB.prepare(
        `INSERT OR REPLACE INTO sections (key, ticker, form, filed, section, chars, vec, chunks)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?)`
      ).bind(
        secKey(m.id, ticker, filed, form, kind), ticker, form, filed, kind, clean.length,
        JSON.stringify(poolNormalize(rawVecs)), packVecs(unit)
      ),
      env.DB.prepare(`DELETE FROM shift_cache WHERE key LIKE ?`).bind(shiftPrefix(m.id, ticker, kind)),
    ]);

    done.push({ kind, status: "ok", chars: clean.length, chunks: parts.length });
  }
  return { filed, form, model: m.id, sections: done, textLen: text.length };
}

/* ─────────────────────────────────────────────── novelty */

function shiftMetrics(cur, prev) {
  if (!cur.length || !prev.length) return null;
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

async function novelty(ticker, env, onlyKinds = ["mdna", "risk"], modelParam = "gemini") {
  const m = getModel(modelParam);
  const b64PerChunk = (m.dim * 4 / 3) * 4;
  await ensureTables(env);

  const { results: metaRows } = await env.DB.prepare(
    `SELECT key, form, filed, section, chars, vec, LENGTH(chunks) AS chunks_len
     FROM sections WHERE ticker = ? AND key LIKE ? ORDER BY filed ASC`
  ).bind(ticker, secPrefix(m.id, ticker)).all();

  const { results: cachedRows } = await env.DB.prepare(
    `SELECT key, max, p75 FROM shift_cache WHERE key LIKE ?`
  ).bind(shiftPrefix(m.id, ticker)).all();
  const shiftMap = new Map(cachedRows.map((r) => [r.key, r]));

  const out = { model: m.id };
  for (const kind of onlyKinds) {
    const rows = metaRows
      .filter((r) => r.section === kind)
      .map((r) => ({
        ...r,
        v: parseVec(r.vec),
        nChunks: r.chunks_len ? Math.round(r.chunks_len / b64PerChunk) : 0,
      }))
      .filter((r) => r.v && r.v.length === m.dim);

    const needBlobKeys = new Set();
    const pairs = rows.map((r, i) => {
      let prev = null;
      for (let j = i - 1; j >= 0; j--) if (rows[j].form === r.form) { prev = rows[j]; break; }
      const cKey = prev ? shiftKey(m.id, ticker, kind, r.filed, prev.filed, r.form) : null;
      if (prev && r.nChunks > 0 && prev.nChunks > 0 && !shiftMap.has(cKey)) {
        needBlobKeys.add(r.key);
        needBlobKeys.add(prev.key);
      }
      return { r, i, prev, cKey };
    });

    const chunkMap = new Map();
    if (needBlobKeys.size > 0) {
      const arr = [...needBlobKeys];
      for (let b = 0; b < arr.length; b += 40) {
        const slice = arr.slice(b, b + 40);
        const { results: blobs } = await env.DB.prepare(
          `SELECT key, chunks FROM sections WHERE key IN (${slice.map(() => "?").join(",")})`
        ).bind(...slice).all();
        for (const row of blobs) {
          if (row.chunks) {
            const unpacked = unpackVecs(row.chunks, m.dim);
            if (unpacked.length) chunkMap.set(row.key, unpacked);
          }
        }
      }
    }

    const writes = [];
    out[kind] = pairs.map(({ r, i, prev, cKey }) => {
      const consec = i > 0 ? 1 - cosine(r.v, rows[i - 1].v) : null;
      let s = cKey ? shiftMap.get(cKey) : null;
      if (!s && prev && chunkMap.has(r.key) && chunkMap.has(prev.key)) {
        s = shiftMetrics(chunkMap.get(r.key), chunkMap.get(prev.key));
        if (s) {
          shiftMap.set(cKey, s);
          writes.push(
            env.DB.prepare(`INSERT OR REPLACE INTO shift_cache (key, max, p75) VALUES (?, ?, ?)`)
              .bind(cKey, s.max, s.p75)
          );
        }
      }
      return {
        filed: r.filed, form: r.form, chars: r.chars, chunks: r.nChunks,
        consecutive: consec,
        pooled: prev ? 1 - cosine(r.v, prev.v) : null,
        max: s ? s.max : null,
        p75: s ? s.p75 : null,
      };
    });

    if (writes.length > 0) await env.DB.batch(writes);
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
  if (!ticker) throw new Error("ticker required");
  await ensureTables(env);
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

  const rx = rankArray(x), ry = rankArray(y);
  const mx = rx.reduce((a, b) => a + b, 0) / n;
  const my = ry.reduce((a, b) => a + b, 0) / n;
  const ax = new Float64Array(n), ay = new Float64Array(n);
  let dx = 0, dy = 0;
  for (let i = 0; i < n; i++) {
    ax[i] = rx[i] - mx;
    ay[i] = ry[i] - my;
    dx += ax[i] * ax[i];
    dy += ay[i] * ay[i];
  }
  const den = Math.sqrt(dx * dy);
  if (den === 0) return { n, rho: 0, p: 1 };

  let extreme = 0;
  for (let s = 0; s < shuffles; s++) {
    for (let i = n - 1; i > 0; i--) {
      const j = (Math.random() * (i + 1)) | 0;
      const tmp = ay[i]; ay[i] = ay[j]; ay[j] = tmp;
    }
    let num = 0;
    for (let i = 0; i < n; i++) num += ax[i] * ay[i];
    if (Math.abs(num / den) >= Math.abs(obs) - 1e-12) extreme++;
  }
  return { n, rho: obs, p: (extreme + 1) / (shuffles + 1) };
}

async function volTest(tickerParam, env, modelParam = "gemini") {
  const m = getModel(modelParam);
  await ensureTables(env);
  const { results: priceRows } = await env.DB.prepare(`SELECT ticker, data FROM series`).all();
  const priceMap = new Map();
  for (const r of priceRows) {
    try { priceMap.set(r.ticker, JSON.parse(r.data)); } catch {}
  }

  const { results: secRows } = await env.DB.prepare(
    `SELECT DISTINCT ticker FROM sections`
  ).all();

  const tickers = tickerParam
    ? [tickerParam]
    : [...new Set([...COMPANIES.map((c) => c.ticker), ...secRows.map((r) => r.ticker)])];

  const observations = [];

  for (const t of tickers) {
    const px = priceMap.get(t);
    if (!px || px.length < 100) continue;

    const nov = await novelty(t, env, ["mdna"], m.id);
    for (const r of nov.mdna || []) {
      if (r.max == null || r.pooled == null || r.p75 == null) continue;
      const idx = px.findIndex((p) => p[0] > r.filed);
      if (idx < 30 || idx + 30 > px.length) continue;

      const preCloses = px.slice(idx - 30, idx).map((p) => p[1]);
      const postCloses = px.slice(idx, idx + 30).map((p) => p[1]);
      const preVol = realisedVol(preCloses);
      const postVol = realisedVol(postCloses);
      if (!preVol || !postVol || preVol === 0) continue;

      observations.push({
        ticker: t, filed: r.filed, form: r.form,
        max: r.max, p75: r.p75, pooled: r.pooled,
        preVol, postVol,
        volRatio: Math.log(postVol / preVol),
      });
    }
  }

  const ys = observations.map((o) => o.volRatio);
  const pooledTests = {};
  for (const metric of ["max", "p75", "pooled"]) {
    const xs = observations.map((o) => o[metric]);
    pooledTests[metric] = permTest(xs, ys, 3000);
  }

  const perCompany = {};
  for (const t of tickers) {
    const sub = observations.filter((o) => o.ticker === t);
    perCompany[t] = permTest(sub.map((o) => o.max), sub.map((o) => o.volRatio), 1000);
  }

  return { model: m.id, modelLabel: m.short, n: observations.length, pooled: pooledTests, perCompany, observations };
}

/* ─────────────────────────────────────────────── utilities */

async function reset(ticker, env, modelParam = "gemini") {
  if (!ticker) throw new Error("ticker required");
  const m = getModel(modelParam);
  await ensureTables(env);
  const r = await env.DB.prepare(
    `DELETE FROM sections WHERE ticker = ? AND key LIKE ?`
  ).bind(ticker, secPrefix(m.id, ticker)).run();
  await env.DB.prepare(`DELETE FROM shift_cache WHERE key LIKE ?`).bind(shiftPrefix(m.id, ticker)).run();
  return { model: m.id, deleted: r.meta.changes };
}

async function status(env) {
  await ensureTables(env);
  const secs = await env.DB.prepare(
    `SELECT CASE WHEN key LIKE 'gemma2:%' THEN 'gemma2' ELSE 'gemini' END AS model,
            ticker, section, COUNT(*) as n, MIN(filed) as from_d, MAX(filed) as to_d,
            SUM(CASE WHEN chunks IS NOT NULL THEN 1 ELSE 0 END) as with_chunks
     FROM sections GROUP BY model, ticker, section`
  ).all();
  const px = await env.DB.prepare(`SELECT ticker, updated, LENGTH(data) as bytes FROM series`).all();
  return { sections: secs.results, prices: px.results };
}

async function topChunk(params, env) {
  const ticker = params.get("t");
  const filed = params.get("filed");
  const section = params.get("s") || "mdna";
  const m = getModel(params.get("m"));
  if (!ticker || !filed) throw new Error("t and filed required");

  const { results } = await env.DB.prepare(
    `SELECT form, filed, chunks FROM sections
     WHERE ticker = ? AND section = ? AND key LIKE ? ORDER BY filed ASC`
  ).bind(ticker, section, secPrefix(m.id, ticker)).all();

  const idx = results.findIndex((r) => r.filed === filed);
  if (idx < 0) throw new Error("filing not in DB for " + m.short);
  const cur = results[idx];
  let prev = null;
  for (let j = idx - 1; j >= 0; j--) if (results[j].form === cur.form) { prev = results[j]; break; }
  if (!prev || !cur.chunks || !prev.chunks) throw new Error("no same-form predecessor with chunks");

  const curV = unpackVecs(cur.chunks, m.dim);
  const prevV = unpackVecs(prev.chunks, m.dim);
  if (!curV.length || !prevV.length) throw new Error("chunk vectors have incompatible dimension");

  let worstIdx = 0, worstDist = -1;
  for (let i = 0; i < curV.length; i++) {
    let best = 2;
    for (const b of prevV) {
      const d = 1 - cosine(curV[i], b);
      if (d < best) best = d;
    }
    if (best > worstDist) { worstDist = best; worstIdx = i; }
  }

  const filings = await listFilings(COMPANIES.find((c) => c.ticker === ticker)?.cik, 20);
  const curUrl = filings.find((f) => f.filed === filed)?.url;
  let snippet = "(filing no longer in recent index)";
  if (curUrl) {
    const text = await filingText(curUrl);
    const raw = extractSection(text, section, cur.form);
    if (raw) {
      const parts = chunk(stripBoilerplate(raw)).slice(0, 40);
      snippet = (parts[worstIdx] || "").slice(0, 700);
    }
  }

  return {
    model: m.id, modelLabel: m.short,
    ticker, section, form: cur.form, filed, comparedTo: prev.filed,
    chunkIndex: worstIdx, totalChunks: curV.length,
    distance: +worstDist.toFixed(4),
    excerpt: snippet,
  };
}

async function exportText(env, modelParam) {
  const st = await status(env);
  const targetModels = modelParam ? [getModel(modelParam).id] : ["gemma2", "gemini"];
  const modelsOut = {};
  for (const mId of targetModels) {
    const vt = await volTest(null, env, mId);
    const nov = {};
    for (const c of COMPANIES) nov[c.ticker] = await novelty(c.ticker, env, ["mdna", "risk"], mId);
    modelsOut[mId] = { volTest: vt, novelty: nov };
  }
  return JSON.stringify({ exported: new Date().toISOString(), status: st, models: modelsOut }, null, 2);
}

/* ─────────────────────────────────────────────── page */

const MODEL_OPTIONS = Object.values(MODELS)
  .map((m) => `<option value="${m.id}">${m.label}</option>`)
  .join("\n      ");
const COMPANY_OPTIONS = COMPANIES
  .map((c) => `<option value="${c.ticker}|${c.cik}">${c.ticker} — ${c.name}</option>`)
  .join("\n      ");
const ALL_TICKERS_JSON = JSON.stringify(COMPANIES.map((c) => c.ticker));

const PAGE = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Latent Alpha — Phase 0</title>
<style>
  :root {
    --bg: #0c090a; --panel: #140f11; --line: #2a1f24;
    --fg: #ede4e7; --dim: #8c7780; --accent: #f0c8d8; --hot: #ff8fb8;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 20px 16px 64px; background: var(--bg); color: var(--fg);
    font: 14px/1.55 ui-monospace, "SF Mono", Menlo, monospace;
    max-width: 760px; margin-inline: auto;
  }
  h1 { font-size: 13px; letter-spacing: .18em; text-transform: uppercase; color: var(--accent); margin: 0 0 4px; font-weight: 600; }
  .sub { color: var(--dim); font-size: 12px; margin-bottom: 18px; }
  .bar { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; }
  select, button {
    background: var(--panel); color: var(--fg); border: 1px solid var(--line);
    padding: 10px 12px; font: inherit; font-size: 13px; border-radius: 3px;
  }
  select { flex: 1; min-width: 150px; }
  select.model-pick { flex: 1 1 100%; border-color: var(--accent); color: var(--accent); }
  button { cursor: pointer; letter-spacing: .06em; }
  button.go { border-color: var(--accent); color: var(--accent); }
  button:disabled { opacity: .4; cursor: default; }
  #log {
    background: var(--panel); border: 1px solid var(--line); padding: 12px;
    font-size: 12px; color: var(--dim); white-space: pre-wrap;
    max-height: 220px; overflow-y: auto; display: none; margin-bottom: 18px;
  }
  h2 { font-size: 11px; letter-spacing: .16em; text-transform: uppercase; color: var(--dim); margin: 26px 0 8px; font-weight: 500; }
  table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
  th, td { padding: 7px 6px; text-align: left; border-bottom: 1px solid var(--line); }
  th { color: var(--dim); font-weight: 400; font-size: 11px; }
  td.num { text-align: right; font-variant-numeric: tabular-nums; }
  .barcell { width: 34%; }
  .spark { height: 7px; background: var(--accent); display: inline-block; vertical-align: middle; }
  .spark.hot { background: var(--hot); }
  .note { color: var(--dim); font-size: 11.5px; margin-top: 22px; line-height: 1.65; }
  .sc { cursor: pointer; text-decoration: underline dotted var(--dim); }
  #peek {
    background: var(--panel); border: 1px solid var(--accent); padding: 14px;
    font-size: 12px; line-height: 1.65; margin: 16px 0; display: none;
  }
  #peek .hdr { color: var(--accent); font-size: 11px; letter-spacing: .1em; text-transform: uppercase; margin-bottom: 8px; }
</style>
</head>
<body>

<h1>Latent Alpha Observatory</h1>
<div class="sub">Phase 0 · SEC filing language shift · Gemma 4 (Workers AI) &amp; Gemini cross-model embeddings</div>

<div class="bar">
  <select id="model" class="model-pick" title="Embedding model">
    ${MODEL_OPTIONS}
  </select>
  <select id="company">
    ${COMPANY_OPTIONS}
  </select>
  <select id="count" style="flex:0;min-width:95px">
    <option value="6">6 filings</option>
    <option value="10" selected>10 filings</option>
    <option value="14">14 filings</option>
  </select>
  <button class="go" id="ingestBtn">INGEST &amp; MEASURE</button>
  <button id="resetBtn" title="wipe cached vectors for this company &amp; model">RESET</button>
</div>

<div id="log"></div>
<div id="peek"></div>
<div id="tables"></div>

<hr style="border:0;border-top:1px solid var(--line);margin:32px 0 20px">

<h2>Does it predict anything?</h2>
<div class="note" style="margin-top:0;margin-bottom:14px">
  Hypothesis, fixed before looking: filings whose language moved unusually far
  from the same company's previous same-form filing are followed by higher
  realised volatility than that company's own recent baseline. Forward window
  starts the trading day <em>after</em> the filing.
</div>
<div class="bar">
  <button id="pricesBtn">LOAD PRICES</button>
  <button class="go" id="testBtn">RUN TEST</button>
</div>
<div id="testOut"></div>

<div class="note">
  <b>max shift</b> = largest cosine distance from any chunk (~6k chars) of this
  filing to its closest match anywhere in the previous same-form filing — catches
  a single new paragraph that averaging would wash out. <b>p75</b> = 75th-percentile
  chunk distance. <b>pooled</b> = whole-section mean-vector distance. All three
  metrics run every time and are all reported, so none of them was chosen after
  the fact. 10-K is only ever compared to 10-K, 10-Q to 10-Q. Tap any
  <b>max shift</b> number to read the paragraph that moved most.
</div>

<script>
const $ = (id) => document.getElementById(id);
const log = $("log");
const ALL_TICKERS = ${ALL_TICKERS_JSON};

try {
  const saved = localStorage.getItem("la_model");
  if (saved && $("model").querySelector('option[value="' + saved + '"]')) $("model").value = saved;
} catch {}

const curModel = () => $("model").value;
const curTicker = () => $("company").value.split("|")[0];

async function api(url, opts) {
  const r = await fetch(url, opts);
  const txt = await r.text();
  let data;
  try { data = JSON.parse(txt); }
  catch {
    throw new Error("HTTP " + r.status + (txt.includes("1102") ? " (Worker resource limit)" : " non-JSON response"));
  }
  if (!r.ok || (data && data.error)) throw new Error((data && data.error) || ("HTTP " + r.status));
  return data;
}

/** Browser-side HTML stripper fallback if a 50MB+ filing ever hits Worker 10ms CPU cap. */
async function browserStripSec(secUrl) {
  const r = await fetch("/api/raw?url=" + encodeURIComponent(secUrl));
  if (!r.ok) throw new Error("raw fetch HTTP " + r.status);
  const html = await r.text();
  const doc = new DOMParser().parseFromString(html, "text/html");
  doc.querySelectorAll('script, style, [hidden], [style*="display:none"], [style*="display: none"]').forEach((e) => e.remove());
  for (const tag of ["ix:header", "ix:hidden"]) {
    const els = doc.getElementsByTagName(tag);
    while (els.length) els[0].remove();
  }
  const rawTxt = (doc.body ? doc.body.textContent : doc.documentElement.textContent) || "";
  return rawTxt.replace(/\\s+/g, " ").trim().slice(0, 650000);
}

function showTable(title, kind, ticker, mId, rows) {
  if (!rows.length) return "<h2>" + title + "</h2><div class=note>no sections stored yet for this model — tap INGEST &amp; MEASURE</div>";
  const vals = rows.map((r) => r.max).filter((v) => v != null);
  const hi = Math.max(...vals, 0.001);
  const mean = vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
  const sd = vals.length > 1 ? Math.sqrt(vals.reduce((a, b) => a + (b - mean) ** 2, 0) / vals.length) : 0;

  let html = "<h2>" + title + "</h2><table><tr><th>filed</th><th>form</th><th>chunks</th><th class=num>max shift</th><th class=num>p75</th><th class=num>pooled</th><th class=barcell></th></tr>";
  for (const r of rows) {
    const d = r.max;
    const hot = sd > 0 && d != null && d > mean + 0.9 * sd;
    const w = d == null ? 0 : Math.max(2, Math.round((d / hi) * 100));
    const cell = d == null ? "—"
      : '<span class="sc" data-top="' + ticker + "|" + kind + "|" + r.filed + "|" + mId + '">' + d.toFixed(4) + "</span>";
    html += "<tr>"
      + "<td>" + r.filed + "</td>"
      + "<td>" + r.form + "</td>"
      + '<td style="color:var(--dim)">' + (r.chunks || "—") + "</td>"
      + '<td class="num">' + cell + "</td>"
      + '<td class="num" style="color:var(--dim)">' + (r.p75 == null ? "—" : r.p75.toFixed(4)) + "</td>"
      + '<td class="num" style="color:var(--dim)">' + (r.pooled == null ? "—" : r.pooled.toFixed(4)) + "</td>"
      + '<td class="barcell">' + (d == null ? "" : '<span class="spark' + (hot ? " hot" : "") + '" style="width:' + w + '%"></span>') + "</td>"
      + "</tr>";
  }
  return html + "</table>";
}

async function renderNovelty(ticker) {
  const mId = curModel();
  const nov = await api("/api/novelty?t=" + ticker + "&m=" + mId);
  $("tables").innerHTML =
    showTable("MD&A — Management's Discussion & Analysis", "mdna", ticker, mId, nov.mdna || []) +
    showTable("Item 1A — Risk Factors", "risk", ticker, mId, nov.risk || []);
}

$("model").onchange = () => {
  try { localStorage.setItem("la_model", curModel()); } catch {}
  $("peek").style.display = "none";
  $("testOut").innerHTML = "";
  renderNovelty(curTicker()).catch(() => {});
};

$("company").onchange = () => {
  $("peek").style.display = "none";
  renderNovelty(curTicker()).catch(() => {});
};

$("tables").onclick = async (ev) => {
  const el = ev.target.closest("[data-top]");
  if (!el) return;
  const [t, s, filed, mId] = el.dataset.top.split("|");
  const peek = $("peek");
  peek.style.display = "block";
  peek.innerHTML = '<div class="hdr">loading most-novel chunk…</div>';
  try {
    const d = await api("/api/topchunk?t=" + t + "&s=" + s + "&filed=" + filed + "&m=" + (mId || curModel()));
    peek.innerHTML = '<div class="hdr">' + d.ticker + " " + d.form + " " + d.filed
      + " · " + d.modelLabel
      + " · chunk " + (d.chunkIndex + 1) + "/" + d.totalChunks
      + " · distance " + d.distance + " vs " + d.comparedTo + "</div>"
      + d.excerpt.replace(/</g, "&lt;") + "…";
  } catch (e) {
    peek.textContent = "error: " + e.message;
  }
};

$("ingestBtn").onclick = async () => {
  const [ticker, cik] = $("company").value.split("|");
  const n = $("count").value;
  const mId = curModel();
  $("ingestBtn").disabled = true;
  log.style.display = "block";
  log.textContent = "model: " + mId + "\\nfetching filing index…\\n";

  try {
    const fl = await api("/api/filings?cik=" + cik + "&n=" + n);
    log.textContent += fl.length + " filings · " + (fl[0]?.filed || "") + " → " + (fl[fl.length - 1]?.filed || "") + "\\n\\n";

    for (const f of fl) {
      log.textContent += f.filed + "  " + f.form + "\\n";
      try {
        let ex;
        try {
          ex = await api("/api/extract", {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({ ticker, form: f.form, filed: f.filed, url: f.url, model: mId }),
          });
        } catch (exErr) {
          if (String(exErr.message).includes("503") || String(exErr.message).includes("1102")) {
            log.textContent += "  large filing · browser fallback…\\n";
            log.scrollTop = log.scrollHeight;
            const preText = await browserStripSec(f.url);
            ex = await api("/api/extract", {
              method: "POST",
              headers: { "content-type": "application/json" },
              body: JSON.stringify({ ticker, form: f.form, filed: f.filed, model: mId, preText }),
            });
          } else {
            throw exErr;
          }
        }

        for (const sec of ["mdna", "risk"]) {
          const info = ex.sections && ex.sections[sec];
          if (!info || info.cached) {
            log.textContent += "  " + sec + " cached\\n";
            log.scrollTop = log.scrollHeight;
            continue;
          }
          if (info.status !== "ready") {
            log.textContent += "  " + sec + " " + (info.status || "not found") + "\\n";
            log.scrollTop = log.scrollHeight;
            continue;
          }
          const res = await api("/api/ingest", {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({
              ticker, form: f.form, filed: f.filed, section: sec,
              chars: info.chars, parts: info.parts, model: mId,
            }),
          });
          const r = res.sections && res.sections[0];
          log.textContent += "  " + sec + " " + (r ? r.status + (r.chunks ? " · " + r.chunks + " chunks" : "") : "?") + "\\n";
          log.scrollTop = log.scrollHeight;
        }
      } catch (fileErr) {
        log.textContent += "  error: " + fileErr.message + "\\n";
        log.scrollTop = log.scrollHeight;
      }
    }

    log.textContent += "\\ncomputing novelty…";
    await renderNovelty(ticker);
    log.textContent += " done.\\n";
  } catch (e) {
    log.textContent += "\\nERROR: " + e.message;
  } finally {
    $("ingestBtn").disabled = false;
  }
};

$("pricesBtn").onclick = async () => {
  $("pricesBtn").disabled = true;
  const out = $("testOut");
  out.innerHTML = '<div class="note">fetching daily closes…</div>';
  const lines = [];
  for (const t of ALL_TICKERS) {
    try {
      const r = await api("/api/prices?t=" + t);
      lines.push(t + ": " + r.days + " days (" + r.from + " → " + r.to + ") via " + r.source);
    } catch (e) {
      lines.push(t + ": ERROR " + e.message);
    }
    out.innerHTML = '<div class="note">' + lines.join("<br>") + "</div>";
  }
  $("pricesBtn").disabled = false;
};

$("testBtn").onclick = async () => {
  $("testBtn").disabled = true;
  const mId = curModel();
  const out = $("testOut");
  out.innerHTML = '<div class="note">running 3,000-shuffle permutation test (' + mId + ")…</div>";
  try {
    const d = await api("/api/test?m=" + mId);
    const fmt = (x) => (x == null ? "—" : x.toFixed(3));
    let h = "<h2>Pooled across companies · " + d.modelLabel + " · n = " + d.n + " filings · target = log(vol₊₃₀ / vol₋₃₀)</h2>";
    h += "<table><tr><th>MD&amp;A novelty metric</th><th class=num>Spearman ρ</th><th class=num>two-sided p (perm)</th></tr>";
    for (const [k, v] of Object.entries(d.pooled)) {
      const sig = v.p != null && v.p < 0.05;
      h += "<tr><td>" + k + '</td><td class="num">' + fmt(v.rho) + '</td><td class="num" style="color:' + (sig ? "var(--hot)" : "var(--dim)") + '">' + fmt(v.p) + "</td></tr>";
    }
    h += "</table>";

    h += "<h2>Within each company (max shift vs vol ratio)</h2><table><tr><th>ticker</th><th class=num>n</th><th class=num>Spearman ρ</th><th class=num>p</th></tr>";
    for (const [t, info] of Object.entries(d.perCompany || {})) {
      if (!info.n && !ALL_TICKERS.includes(t)) continue;
      const tooFew = (info.n || 0) < 4;
      h += "<tr><td><b>" + t + '</b></td><td class="num">' + (info.n || 0) + '</td><td class="num">' + (tooFew ? "—" : fmt(info.rho)) + '</td><td class="num" style="color:var(--dim)">' + (tooFew ? "too few" : fmt(info.p)) + "</td></tr>";
    }
    h += "</table>";

    if (d.observations.length) {
      h += "<h2>Every observation</h2><table><tr><th>filing</th><th class=num>max shift</th><th class=num>pre vol</th><th class=num>post vol</th><th class=num>log ratio</th></tr>";
      for (const o of d.observations.slice().sort((a, b) => b.max - a.max)) {
        const up = o.volRatio > 0;
        h += "<tr><td>" + o.ticker + " " + o.form + " " + o.filed + "</td>"
          + '<td class="num">' + o.max.toFixed(4) + "</td>"
          + '<td class="num" style="color:var(--dim)">' + (o.preVol * 100).toFixed(1) + "%</td>"
          + '<td class="num">' + (o.postVol * 100).toFixed(1) + "%</td>"
          + '<td class="num" style="color:' + (up ? "var(--accent)" : "var(--dim)") + '">' + (up ? "+" : "") + o.volRatio.toFixed(3) + "</td></tr>";
      }
      h += "</table>";
    }
    out.innerHTML = h;
  } catch (e) {
    out.innerHTML = '<div class="note">ERROR: ' + e.message + "</div>";
  }
  $("testBtn").disabled = false;
};

$("resetBtn").onclick = async () => {
  const ticker = curTicker();
  const mId = curModel();
  if (!confirm("Delete cached " + mId + " vectors for " + ticker + "?")) return;
  await api("/api/reset?t=" + ticker + "&m=" + mId, { method: "POST" });
  $("tables").innerHTML = "";
  $("peek").style.display = "none";
  log.style.display = "block";
  log.textContent = ticker + " (" + mId + ") cleared.";
};

renderNovelty(curTicker()).catch(() => {});
</script>
</body>
</html>`;