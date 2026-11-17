-- Cloudflare D1 schema for the forecast-log dashboard.
--
-- Nothing here is a JSON blob: every number the pages need lives in a typed column.
-- The two generic tables exist because the three markets genuinely disagree about
-- shape (the A-share artefact has `bands`/`reference`, QQQ has `comparison`), so a
-- fixed table would be half NULLs.
--
--   market        one row per tab (ndx / qqq / ssec): the question and its rule
--   session       one row per trading day: OHLCV plus the derived signal columns
--   factor_key    which 4-bit slot of session.factors_packed means which switch
--   contexts      the switches the page renders as checkboxes
--   market_node   every nested value as one scalar row, addressed by path
--
-- Rows: ~13,900 sessions, ~30 kv per market, ~11 factor keys. Well inside the free tier.

CREATE TABLE IF NOT EXISTS market (
  market        TEXT PRIMARY KEY,       -- ndx | qqq | ssec
  instrument    TEXT NOT NULL,          -- ^NDX, QQQ, 000001.SS
  name          TEXT,                   -- human label from the artefact
  horizon       INTEGER,
  question      TEXT,
  rule          TEXT,
  first_session TEXT,
  last_session  TEXT,
  fetched_at    TEXT
);

CREATE TABLE IF NOT EXISTS session (
  market          TEXT NOT NULL,
  date            TEXT NOT NULL,
  open            REAL,
  high            REAL,
  low             REAL,
  close           REAL,
  volume          INTEGER,
  entry           REAL,
  entry_at        TEXT,
  base_close      REAL,
  close3          REAL,
  close5          REAL,
  hit3            INTEGER,
  flag            INTEGER,
  flag_price_only INTEGER,
  news_veto       INTEGER,
  score           REAL,
  threshold       REAL,
  outcome         INTEGER,
  target          TEXT,
  peak            REAL,
  peak_at         TEXT,
  us_date         TEXT,
  factors_packed  INTEGER NOT NULL DEFAULT 0,  -- 4 bits per switch, matching factor_key.position
  PRIMARY KEY (market, date)
);

CREATE INDEX IF NOT EXISTS session_by_market_date ON session (market, date);

CREATE TABLE IF NOT EXISTS factor_key (
  market   TEXT NOT NULL,
  position INTEGER NOT NULL,
  key      TEXT NOT NULL,
  PRIMARY KEY (market, position)
);

CREATE TABLE IF NOT EXISTS contexts (
  market         TEXT NOT NULL,
  position       INTEGER NOT NULL,
  key            TEXT NOT NULL,
  label          TEXT,
  kind           TEXT,
  required       INTEGER,
  threshold      REAL,
  group_name     TEXT,
  group_label    TEXT,
  note           TEXT,
  available_from TEXT,
  PRIMARY KEY (market, position)
);

-- Everything nested goes here as one scalar per row, with a path: list indexes are
-- path segments ("bands.0.hit", "question.factors.3"), so arrays keep their order and
-- nothing is stored as a JSON blob.
CREATE TABLE IF NOT EXISTS market_node (
  market TEXT NOT NULL,
  path   TEXT NOT NULL,               -- summary.n | calendar.counts.fomc | comparison.rows.0.hit
  kind   TEXT NOT NULL,               -- num | int | str | bool | null
  value  TEXT,
  PRIMARY KEY (market, path)
);
