CREATE TABLE IF NOT EXISTS sections (
  key     TEXT PRIMARY KEY,
  ticker  TEXT NOT NULL,
  form    TEXT NOT NULL,
  filed   TEXT NOT NULL,
  section TEXT NOT NULL,
  chars   INTEGER NOT NULL,
  vec     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ticker_filed ON sections (ticker, filed);
