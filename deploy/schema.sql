-- Cloudflare D1 schema for the forecast-log dashboard.
--
-- Nothing here is a JSON blob: every number the pages need lives in a typed column.
-- The generic node tables exist because nested metadata/summaries vary in shape
-- across markets, timelines, and evaluation runs.
--
-- Cross-asset signals (/api/cross/<market>):
--   market        one row per tab (ndx / qqq / ssec): the question and its rule
--   session       one row per trading day: OHLCV plus the derived signal columns
--   contexts      which 4-bit slot of session.factors_packed means which switch
--   contexts      the switches the page renders as checkboxes
--   market_node   every nested value as one scalar row, addressed by path
--
-- VXN timeline (/api/timeline):
--   timeline_close    vxn / ndx daily closes
--   timeline_rolling  rolling horizon forecasts vs actuals
--   timeline_daily    daily risk-band series
--   timeline_ladder   per-horizon (1..7) median/low/high series
--   timeline_node     nested metadata (summary, pending, forward, questions, tomorrow)
--
-- Agent evaluation (/api/agent/<dataset>):
--   agent_dataset     one row per dataset (deepseek / qwen)
--   agent_node        all evaluation fields as scalar rows addressed by path

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
  factors_packed  INTEGER NOT NULL DEFAULT 0,  -- 4 bits per switch, matching contexts.position
  is_closed       INTEGER NOT NULL DEFAULT 1,  -- 1 = closed final print, 0 = intraday unclosed
  PRIMARY KEY (market, date)
);

CREATE INDEX IF NOT EXISTS session_by_market_date ON session (market, date);

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
  kind   TEXT NOT NULL,               -- num | int | str | bool | null | list | obj
  value  TEXT,
  PRIMARY KEY (market, path)
);

-- ------------------------------------------------------------------ VXN timeline

CREATE TABLE IF NOT EXISTS timeline_close (
  symbol TEXT NOT NULL,               -- vxn | ndx
  idx    INTEGER NOT NULL,
  date   TEXT NOT NULL,
  close  REAL NOT NULL,
  PRIMARY KEY (symbol, idx)
);

CREATE TABLE IF NOT EXISTS timeline_rolling (
  symbol TEXT NOT NULL,               -- vxn | ndx
  idx    INTEGER NOT NULL,
  time   TEXT NOT NULL,
  as_of  TEXT,
  anchor REAL,
  median REAL,
  low    REAL,
  high   REAL,
  actual REAL,
  PRIMARY KEY (symbol, idx)
);

CREATE TABLE IF NOT EXISTS timeline_daily (
  symbol   TEXT NOT NULL,             -- vxn | ndx
  idx      INTEGER NOT NULL,
  time     TEXT NOT NULL,
  median   REAL,
  low      REAL,
  high     REAL,
  actual   REAL,
  previous REAL,
  risk     INTEGER,
  PRIMARY KEY (symbol, idx)
);

CREATE TABLE IF NOT EXISTS timeline_ladder (
  symbol  TEXT NOT NULL,              -- vxn
  horizon INTEGER NOT NULL,           -- 1..7
  idx     INTEGER NOT NULL,
  median  REAL,
  low     REAL,
  high    REAL,
  PRIMARY KEY (symbol, horizon, idx)
);

CREATE TABLE IF NOT EXISTS timeline_node (
  path  TEXT NOT NULL PRIMARY KEY,
  kind  TEXT NOT NULL,
  value TEXT
);

-- ------------------------------------------------------------------ Agent eval

CREATE TABLE IF NOT EXISTS agent_dataset (
  dataset          TEXT PRIMARY KEY,  -- deepseek | qwen
  n                INTEGER,
  scored_on_shared INTEGER,
  generated_at     TEXT
);

CREATE TABLE IF NOT EXISTS agent_node (
  dataset TEXT NOT NULL,
  path    TEXT NOT NULL,
  kind    TEXT NOT NULL,
  value   TEXT,
  PRIMARY KEY (dataset, path)
);
