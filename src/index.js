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

const cosine = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };

/* ─────────────────────────────────────────────── ingest */

async function ingest(body, env) {
  const { ticker, form, filed, url } = body;
  if (!env.GEMINI_API_KEY) throw new Error("GEMINI_API_KEY secret not set");

  // Resume: if every section of this filing is already stored, don't refetch
  // or re-embed it. Lets you press the button again after a quota stall.
  const keys = Object.keys(SECTIONS).map((k) => `${ticker}:${filed}:${form}:${k}`);
  const have = await env.DB.prepare(
    `SELECT key FROM sections WHERE key IN (${keys.map(() => "?").join(",")})`
  ).bind(...keys).all();
  const stored = new Set(have.results.map((r) => r.key));
  if (stored.size === keys.length) return { filed, form, skipped: true, sections: [] };

  const text = await filingText(url);
  const done = [];

  for (const kind of Object.keys(SECTIONS)) {
    if (stored.has(`${ticker}:${filed}:${form}:${kind}`)) { done.push({ kind, status: "cached" }); continue; }
    const raw = extractSection(text, kind);
    if (!raw) { done.push({ kind, status: "not found" }); continue; }

    const clean = stripBoilerplate(raw);
    const vec = poolNormalize(await embed(chunk(clean), env.GEMINI_API_KEY));

    await env.DB.prepare(
      `INSERT OR REPLACE INTO sections (key, ticker, form, filed, section, chars, vec)
       VALUES (?, ?, ?, ?, ?, ?, ?)`
    ).bind(`${ticker}:${filed}:${form}:${kind}`, ticker, form, filed, kind, clean.length, JSON.stringify(vec)).run();

    done.push({ kind, status: "ok", chars: clean.length });
  }
  return { filed, form, sections: done, textLen: text.length };
}

/* ─────────────────────────────────────────────── novelty */

async function novelty(ticker, env) {
  const { results } = await env.DB.prepare(
    `SELECT form, filed, section, chars, vec FROM sections WHERE ticker = ? ORDER BY filed ASC`
  ).bind(ticker).all();

  const out = {};
  for (const kind of Object.keys(SECTIONS)) {
    const rows = results.filter((r) => r.section === kind).map((r) => ({ ...r, v: JSON.parse(r.vec) }));
    out[kind] = rows.map((r, i) => {
      // consecutive: vs previous filing of ANY type. Contaminated — a 10-K is
      // far longer and broader than a 10-Q, so annuals spike structurally.
      const consec = i > 0 ? 1 - cosine(r.v, rows[i - 1].v) : null;
      // same-form: 10-Q vs previous 10-Q. Slower, but a spike here is language.
      let prev = null;
      for (let j = i - 1; j >= 0; j--) if (rows[j].form === r.form) { prev = rows[j]; break; }
      return {
        filed: r.filed, form: r.form, chars: r.chars,
        consecutive: consec,
        sameForm: prev ? 1 - cosine(r.v, prev.v) : null,
      };
    });
  }
  return out;
}

/* ─────────────────────────────────────────────── prices

   Stooq: free daily CSV, no key. We store the whole series as one JSON row
   per ticker — it's ~30KB and we always want the whole thing anyway, so
   there's no reason to pay for 1,500 row inserts. */

async function loadPrices(ticker, env) {
  const r = await fetch(`https://stooq.com/q/d/l/?s=${ticker.toLowerCase()}.us&i=d`);
  if (!r.ok) throw new Error(`stooq ${r.status}`);
  const csv = await r.text();
  if (!csv.startsWith("Date")) throw new Error("stooq returned no data for " + ticker);

  const rows = [];
  for (const line of csv.trim().split("\n").slice(1)) {
    const c = line.split(",");
    const close = parseFloat(c[4]);
    if (c[0] && isFinite(close)) rows.push([c[0], close]);
  }
  rows.sort((a, b) => (a[0] < b[0] ? -1 : 1));

  await env.DB.prepare(
    `INSERT OR REPLACE INTO series (ticker, updated, data) VALUES (?, ?, ?)`
  ).bind(ticker, new Date().toISOString().slice(0, 10), JSON.stringify(rows)).run();

  return { ticker, days: rows.length, from: rows[0]?.[0], to: rows[rows.length - 1]?.[0] };
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

async function volTest(which, env) {
  const all = (await env.DB.prepare(`SELECT DISTINCT ticker FROM sections`).all())
    .results.map((r) => r.ticker);
  const tickers = which && which !== "ALL" ? [which] : all;

  const perTicker = [];
  const pooledNov = [], pooledVol = [];
  const missing = [];

  for (const ticker of tickers) {
    const row = await env.DB.prepare(`SELECT data FROM series WHERE ticker = ?`).bind(ticker).first();
    if (!row) { missing.push(ticker); continue; }
    const series = JSON.parse(row.data);
    const dates = series.map((r) => r[0]);
    const closes = series.map((r) => r[1]);

    const nov = (await novelty(ticker, env)).mdna.filter((r) => r.sameForm != null);
    const rows = [];

    for (const f of nov) {
      let i = dates.findIndex((d) => d > f.filed);      // first close strictly after filing
      if (i < WINDOW || i + WINDOW >= closes.length) continue;

      const fwd = realisedVol(closes.slice(i, i + WINDOW));
      const trail = realisedVol(closes.slice(i - WINDOW, i));
      if (!fwd || !trail) continue;

      rows.push({ filed: f.filed, form: f.form, novelty: f.sameForm, fwd, trail, ratio: fwd / trail });
    }

    if (rows.length >= 4) {
      const n = rows.map((r) => r.novelty);
      const v = rows.map((r) => Math.log(r.ratio));
      perTicker.push({ ticker, rows, ...permutationP(n, v) });
      pooledNov.push(...zscore(n));                      // z-score within ticker
      pooledVol.push(...zscore(v));                      // before pooling
    } else {
      perTicker.push({ ticker, rows, rho: null, p: null, n: rows.length });
    }
  }

  const pooled = pooledNov.length >= 8 ? permutationP(pooledNov, pooledVol) : null;
  return { window: WINDOW, perTicker, pooled, missingPrices: missing };
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
.bar{display:block;height:3px;background:var(--rose);border-radius:2px;min-width:1px}
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

$('#go').onclick=async()=>{
  const btn=$('#go'); btn.disabled=true;
  $('#log').textContent=''; $('#out').innerHTML='';
  const cik=$('#co').value, ticker=$('#co').selectedOptions[0].text.split(' ')[0], n=$('#n').value;

  try{
    log('fetching filing index…');
    const filings=await (await fetch('/api/filings?cik='+cik+'&n='+n)).json();
    if(filings.error) throw new Error(filings.error);
    log(filings.length+' filings · '+filings[0].filed+' → '+filings[filings.length-1].filed+'\\n');

    let quota=false;
    for(const f of filings){
      log(f.filed+'  '+f.form+'  …');
      let r;
      try{
        // Read as text first — a Worker that blows a resource limit returns an
        // HTML error page, and .json() on that throws a useless parse error.
        const res=await fetch('/api/ingest',{method:'POST',
          headers:{'content-type':'application/json'},
          body:JSON.stringify({ticker,...f})});
        const raw=await res.text();
        try{ r=JSON.parse(raw); }
        catch{ log('   ! worker returned non-JSON ('+res.status+'): '+raw.slice(0,120)); continue; }
      }catch(e){ log('   ! request failed: '+e.message); continue; }

      if(r.error){
        if(r.error==='QUOTA'){ quota=true; log('   ! Gemini quota reached — stopping here'); break; }
        log('   ! '+r.error); continue;
      }
      if(r.skipped){ log('   already stored'); continue; }
      log('   '+r.sections.map(s=>s.kind+' '+(s.chars?s.chars.toLocaleString()+'c':s.status)).join('  ·  '));
    }
    if(quota) log('\\nQuota resets daily. Press the button again later —\\nstored filings are skipped, so it resumes where it stopped.');

    log('\\ncomputing novelty…');
    render(await (await fetch('/api/novelty?t='+ticker)).json(), ticker);
  }catch(e){ log('ERROR: '+e.message); }
  btn.disabled=false;
};

$('#px').onclick=async()=>{
  const b=$('#px'); b.disabled=true; $('#tlog').textContent='';
  const tlog=m=>{$('#tlog').textContent+=m+"\\n"};
  const tickers=[...$('#co').options].map(o=>o.text.split(' ')[0]);
  for(const t of tickers){
    try{
      const r=await (await fetch('/api/prices?t='+t,{method:'POST'})).json();
      tlog(r.error ? t+'  ! '+r.error : t+'  '+r.days+' days  '+r.from+' → '+r.to);
    }catch(e){ tlog(t+'  ! '+e.message); }
  }
  b.disabled=false;
};

$('#run').onclick=async()=>{
  const b=$('#run'); b.disabled=true; $('#tout').innerHTML='';
  $('#tlog').textContent='running permutation test…';
  try{
    const d=await (await fetch('/api/test?t=ALL')).json();
    if(d.error) throw new Error(d.error);
    $('#tlog').textContent = d.missingPrices.length ? 'no prices for: '+d.missingPrices.join(' ') : '';
    renderTest(d);
  }catch(e){ $('#tlog').textContent='ERROR: '+e.message; }
  b.disabled=false;
};

function renderTest(d){
  let h='<table><tr><th>Ticker</th><th>n</th><th>ρ</th><th>p</th><th style="width:30%"></th></tr>';
  for(const t of d.perTicker){
    const sig = t.p!=null && t.p<0.05;
    h+='<tr><td>'+t.ticker+'</td><td class="dim">'+t.n+'</td>'+
       '<td>'+(t.rho!=null?t.rho.toFixed(3):'—')+'</td>'+
       '<td class="'+(sig?'':'dim')+'">'+(t.p!=null?t.p.toFixed(3):'—')+'</td>'+
       '<td class="dim">'+(t.n<4?'too few filings':'')+'</td></tr>';
  }
  h+='</table>';

  if(d.pooled){
    const p=d.pooled, sig=p.p<0.05;
    h+='<div class="sec"><h2>Pooled · '+p.n+' observations</h2>'+
       '<table><tr><td>Spearman ρ</td><td style="text-align:right">'+p.rho.toFixed(3)+'</td></tr>'+
       '<tr><td>permutation p</td><td style="text-align:right'+(sig?';color:var(--rose)':'')+'">'+p.p.toFixed(4)+'</td></tr>'+
       '</table>'+
       '<div class="note">'+(sig
         ? 'Below 0.05. That is one test on a small, correlated sample — treat it as worth pursuing, not as a finding. Companies in this sector move together, so the effective sample is smaller than n suggests.'
         : 'Not significant. Novelty as currently measured does not track forward volatility here. That is a real result, and cheaper to learn now than after building five tabs on top of it.')+
       '</div></div>';
  }else{
    h+='<div class="note">Not enough filings with prices yet. Ingest more companies first.</div>';
  }
  $('#tout').innerHTML=h;
}

function render(data,ticker){
  const titles={mdna:'MD&A',risk:'Risk Factors'};
  let html='';
  for(const k of Object.keys(data)){
    const rows=data[k]; if(!rows.length) continue;
    const max=Math.max(0.02,...rows.map(r=>r.sameForm||0));
    html+='<div class="sec"><h2>'+ticker+' · '+titles[k]+'</h2><table>'+
      '<tr><th>Filed</th><th>Form</th><th>Consec</th><th>Same&nbsp;form</th><th style="width:34%"></th></tr>';
    for(const r of rows){
      const sf=r.sameForm;
      html+='<tr><td>'+r.filed+'</td><td class="dim">'+r.form+'</td>'+
        '<td class="dim">'+(r.consecutive!=null?r.consecutive.toFixed(4):'—')+'</td>'+
        '<td>'+(sf!=null?sf.toFixed(4):'—')+'</td>'+
        '<td>'+(sf!=null?'<span class="bar" style="width:'+(sf/max*100)+'%"></span>':'')+'</td></tr>';
    }
    html+='</table></div>';
  }
  html+='<div class="note">Trust the <b>same-form</b> column — 10-Q against the previous 10-Q. '+
    'The consecutive column compares a 10-K to the 10-Q before it, so it spikes every year for '+
    'structural reasons, not news. It is shown so you can see the artifact.<br><br>'+
    'Now check the top spikes by hand. If they land on quarters where something real happened, '+
    'the measurement works.</div>';
  $('#out').innerHTML=html;
}
</script></body></html>`;