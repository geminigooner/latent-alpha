# Latent Alpha

**An experimental system for measuring narrative change in SEC filings with semantic embeddings — and testing whether unusual changes in corporate language are associated with subsequent market volatility.**

> Status: exploratory / work in progress.  
> This is a research prototype, not a trading system or investment strategy.

## The question

Companies describe themselves repeatedly through 10-K and 10-Q filings.

Most of that language is stable from one filing to the next — but sometimes the semantic structure changes.

Latent Alpha asks:

**Can unusually large changes in how a company describes its business, risks, or performance be measured in embedding space, and are those changes associated with changes in realized volatility afterward?**

The project began as an experiment in applying Gemini embeddings to financial text. The methodology is still being tested and revised.

## How it works

```text
SEC EDGAR
    ↓
10-K / 10-Q filings
    ↓
extract MD&A + Risk Factors
    ↓
remove recurring boilerplate
    ↓
chunk filing text
    ↓
Gemini embeddings (1536d)
    ↓
compare with previous same-form filing
    ↓
semantic novelty metrics
    ↓
market price series
    ↓
forward vs trailing realized volatility
    ↓
Spearman correlation + permutation testing
```

## Current result

No significant relationship.

Pooled across NVDA and GOOGL  
*(n = 15 filings, 30-day windows)*

| Metric | ρ | p |
|---|---:|---:|
| Max shift | 0.029 | 0.93 |
| p75 shift | −0.411 | 0.13 |
| Pooled | −0.275 | 0.32 |

p-values are permutation tests using 3,000 shuffles.

All three metrics are computed and reported on every run so that none can be selected after the fact.

At **n = 15**, this is closer to **“not enough data”** than **“no effect.”**

More companies is the next step, not a different metric.

## Known limitations

- **Section extraction is heuristic.** Filings are inconsistent HTML, and headers can be indistinguishable from table-of-contents entries. An early version silently included Legal Proceedings inside Risk Factors, which made the metric fire on litigation updates rather than narrative change.

- **Small, correlated sample.** These companies move together, so the effective sample is smaller than *n* suggests.

- **Boilerplate.** A rewritten legal disclaimer can move the metric in the same way a strategy shift does. Every score can be traced back to the passage that produced it for exactly this reason.