
// Controlled Transformation Suite — Latent Alpha / ELENS extension
//
// Embeds each before/after passage pair with Gemini, then computes:
//   - cosine distance (what Latent Alpha already tracks)
//   - displacement magnitude (Euclidean norm of the raw vector difference)
//   - the raw displacement vector itself (saved to disk for later directional
//     analysis — angular consistency within a category, PCA across categories, etc.)
//
// This does NOT touch the Latent Alpha pipeline. It's a standalone harness so
// you can see category-level signatures before wiring anything into D1.
//
// Usage:
//   GEMINI_API_KEY=your_key node run_suite.mjs
//
// Match EMBED_MODEL / OUTPUT_DIM to whatever Latent Alpha's pipeline already
// uses — the whole point of this suite is comparability with your existing
// 1536-dim vectors, so don't let this drift from that config.

import { readFile, writeFile } from "fs/promises";

const API_KEY = process.env.GEMINI_API_KEY;
if (!API_KEY) {
  console.error("Set GEMINI_API_KEY in your environment before running.");
  process.exit(1);
}

const EMBED_MODEL = "gemini-embedding-001";
const OUTPUT_DIM = 1536;

async function embed(text) {
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${EMBED_MODEL}:embedContent?key=${API_KEY}`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: `models/${EMBED_MODEL}`,
      content: { parts: [{ text }] },
      outputDimensionality: OUTPUT_DIM,
    }),
  });

  if (!res.ok) {
    const errText = await res.text();
    throw new Error(`Embedding request failed (${res.status}): ${errText}`);
  }

  const data = await res.json();
  return data.embedding.values;
}

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function norm(a) {
  return Math.sqrt(dot(a, a));
}

function cosineDistance(a, b) {
  return 1 - dot(a, b) / (norm(a) * norm(b));
}

function subtract(a, b) {
  return a.map((v, i) => v - b[i]);
}

function mean(xs) {
  return xs.reduce((s, x) => s + x, 0) / xs.length;
}

function stdev(xs) {
  const m = mean(xs);
  return Math.sqrt(mean(xs.map((x) => (x - m) ** 2)));
}

async function main() {
  const pairs = JSON.parse(
    await readFile(new URL("./pairs.json", import.meta.url))
  );

  const results = {};

  for (const [category, items] of Object.entries(pairs)) {
    results[category] = [];

    for (const { id, before, after } of items) {
      console.log(`Embedding ${id}...`);

      const [vBefore, vAfter] = await Promise.all([
        embed(before),
        embed(after),
      ]);

      const diff = subtract(vAfter, vBefore);

      const record = {
        id,
        cosineDistance: cosineDistance(vBefore, vAfter),
        displacementMagnitude: norm(diff),
        displacementVector: diff,
      };

      results[category].push(record);
    }
  }

  // Save full results (including raw vectors) for later directional analysis
  await writeFile(
    new URL("./results_full.json", import.meta.url),
    JSON.stringify(results, null, 2)
  );

  // Print a compact summary table
  console.log("\n=== Category Summary ===\n");

  console.log(
    "category".padEnd(24) +
      "n".padEnd(4) +
      "mean cos.dist".padEnd(16) +
      "mean disp.mag".padEnd(16) +
      "std disp.mag"
  );

  for (const [category, records] of Object.entries(results)) {
    const cosDists = records.map((r) => r.cosineDistance);
    const mags = records.map((r) => r.displacementMagnitude);

    console.log(
      category.padEnd(24) +
        String(records.length).padEnd(4) +
        mean(cosDists).toFixed(4).padEnd(16) +
        mean(mags).toFixed(4).padEnd(16) +
        stdev(mags).toFixed(4)
    );
  }

  console.log(
    "\nFull per-pair results (including raw displacement vectors) saved to results_full.json"
  );
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
