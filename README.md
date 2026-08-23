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