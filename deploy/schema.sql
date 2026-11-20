-- Cloudflare D1 schema — mirrors the local `data/forecast.db` exactly (5 domain tables).
-- Kept in sync with `src/forecast_log/db.py::ensure_schema`.
-- Every statement is idempotent so this file can be re-applied at any time.

CREATE TABLE IF NOT EXISTS bars (
    symbol     TEXT NOT NULL,
    date       TEXT NOT NULL,
    open       REAL CHECK (open IS NULL OR open > 0),
    high       REAL CHECK (high IS NULL OR high > 0),
    low        REAL CHECK (low IS NULL OR low > 0),
    close      REAL NOT NULL CHECK (close > 0),
    volume     INTEGER CHECK (volume IS NULL OR volume >= 0),
    is_closed  INTEGER NOT NULL DEFAULT 1,
    fetched_at TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (symbol, date),
    CHECK (high IS NULL OR low IS NULL OR low <= high)
);

CREATE INDEX IF NOT EXISTS idx_bars_date
    ON bars(date);

CREATE TABLE IF NOT EXISTS signals (
    market          TEXT NOT NULL,
    date            TEXT NOT NULL,
    entry           REAL,
    entry_at        TEXT,
    base_close      REAL,
    close3          REAL,
    close5          REAL,
    hit3            INTEGER,
    flag            INTEGER,
    flag_price_only INTEGER,
    sell_flag       INTEGER,
    sell_threshold  REAL,
    news_veto       INTEGER,
    score           REAL,
    threshold       REAL,
    outcome         INTEGER,
    target          TEXT,
    peak            REAL,
    peak_at         TEXT,
    us_date         TEXT,
    factors_packed  INTEGER NOT NULL DEFAULT 0,
    fetched_at      TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (market, date)
);

CREATE TABLE IF NOT EXISTS news (
    as_of        TEXT NOT NULL,
    doc_id       TEXT NOT NULL DEFAULT '',
    published_at TEXT NOT NULL DEFAULT '',
    title        TEXT NOT NULL DEFAULT '',
    text         TEXT NOT NULL DEFAULT '',
    url          TEXT NOT NULL DEFAULT '',
    domain       TEXT NOT NULL DEFAULT '',
    depth        INTEGER,
    source       TEXT NOT NULL DEFAULT '',
    page         TEXT NOT NULL DEFAULT '',
    fetched_at   TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (as_of, doc_id)
);

CREATE INDEX IF NOT EXISTS idx_news_published_at
    ON news(published_at);

CREATE INDEX IF NOT EXISTS idx_news_domain
    ON news(domain);

CREATE TABLE IF NOT EXISTS agent_eval (
    dataset               TEXT NOT NULL,
    question_id           TEXT NOT NULL,
    predictor             TEXT NOT NULL,
    symbol                TEXT NOT NULL,
    as_of                 TEXT NOT NULL,
    as_of_close           REAL,
    resolve_session       TEXT NOT NULL,
    horizon               INTEGER NOT NULL,
    family                TEXT NOT NULL,
    threshold             REAL NOT NULL,
    question_text         TEXT NOT NULL,
    outcome               INTEGER NOT NULL,
    probability           REAL,
    cost_usd_total        REAL NOT NULL DEFAULT 0.0,
    cost_usd_per_forecast REAL,
    latency_p50_s         REAL,
    latency_p95_s         REAL,
    tool_calls_mean       REAL NOT NULL DEFAULT 0.0,
    gate_hits             INTEGER NOT NULL DEFAULT 0,
    failures              INTEGER NOT NULL DEFAULT 0,
    generated_at          TEXT NOT NULL DEFAULT '',
    config_model          TEXT NOT NULL DEFAULT '',
    models_merged         TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (dataset, question_id, predictor)
);

CREATE TABLE IF NOT EXISTS chat_messages (
    thread_id  TEXT NOT NULL,
    seq        INTEGER NOT NULL,
    market     TEXT NOT NULL,
    as_of      TEXT NOT NULL DEFAULT '',
    role       TEXT NOT NULL CHECK (role IN ('context', 'user', 'deepseek', 'qwen')),
    model      TEXT NOT NULL DEFAULT '',
    stance     TEXT NOT NULL DEFAULT '',
    content    TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (thread_id, seq)
);

CREATE INDEX IF NOT EXISTS idx_chat_messages_market_created
    ON chat_messages(market, created_at DESC, thread_id, seq);
