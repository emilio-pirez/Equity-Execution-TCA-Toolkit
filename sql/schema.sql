-- Equity Execution & TCA Toolkit - SQLite schema
-- Conventions: costs in bps (positive = cost to the trader); timestamps are ISO-8601
-- strings in exchange local time (ts_local) and UTC (ts_utc); dates are YYYY-MM-DD.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS instruments (
    ticker       TEXT PRIMARY KEY,          -- yfinance ticker
    name         TEXT NOT NULL,
    bbg_ticker   TEXT,
    market       TEXT NOT NULL CHECK (market IN ('MX', 'US')),
    asset_type   TEXT NOT NULL,
    tz           TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS bars_1m (
    ticker             TEXT NOT NULL REFERENCES instruments(ticker),
    source             TEXT NOT NULL,       -- 'yf' | 'bbg' | 'synthetic'
    ts_local           TEXT NOT NULL,
    ts_utc             TEXT NOT NULL,
    date               TEXT NOT NULL,
    minute_of_session  INTEGER NOT NULL,    -- minutes since the continuous open
    open REAL, high REAL, low REAL, close REAL,
    volume             REAL,
    exec_volume        REAL,                -- excl. auction bars and block/bad prints
    bid REAL, ask REAL,                     -- NULL when the source has no quotes
    is_open_auction    INTEGER NOT NULL,
    is_close_auction   INTEGER NOT NULL,
    vol_outlier        INTEGER NOT NULL,
    PRIMARY KEY (ticker, source, ts_utc)
);
CREATE INDEX IF NOT EXISTS ix_bars_date ON bars_1m (ticker, date);

CREATE TABLE IF NOT EXISTS daily (
    ticker TEXT NOT NULL,
    date   TEXT NOT NULL,
    open REAL, high REAL, low REAL, close REAL, volume REAL,
    PRIMARY KEY (ticker, date)
);

CREATE TABLE IF NOT EXISTS liquidity_summary (
    ticker TEXT PRIMARY KEY REFERENCES instruments(ticker),
    market TEXT, days_1m INTEGER, price_local REAL, adv_shares REAL, adv_usd_m REAL,
    daily_vol_bps REAL, cs_spread_bps REAL, roll_spread_bps REAL, model_spread_bps REAL,
    amihud_bps_per_usd_m REAL, usd_per_bp_1m REAL, pct_minutes_traded REAL,
    coverage_1m_vs_daily REAL, coverage_exec_vs_daily REAL, outlier_volume_share REAL
);

CREATE TABLE IF NOT EXISTS volume_profile (
    ticker TEXT NOT NULL, bucket_min INTEGER NOT NULL, volume_share REAL, spread_bps REAL,
    PRIMARY KEY (ticker, bucket_min)
);

CREATE TABLE IF NOT EXISTS sim_orders (
    order_id         INTEGER PRIMARY KEY,
    ticker           TEXT NOT NULL REFERENCES instruments(ticker),
    date             TEXT NOT NULL,
    algo             TEXT NOT NULL,         -- TWAP | VWAP | POV10 | POV20 | IS
    pct_adv          REAL NOT NULL,
    side             INTEGER NOT NULL,      -- +1 buy, -1 sell
    qty              INTEGER NOT NULL,
    adv REAL, sigma_bps REAL,
    arrival_px REAL, avg_px REAL, vwap_px REAL, close_px REAL,
    vs_arrival_bps REAL, vs_vwap_bps REAL, vs_close_bps REAL,
    timing_bps REAL, spread_bps REAL, impact_bps REAL, opportunity_bps REAL,
    is_total_bps REAL, is_filled_bps REAL, fill_rate REAL,
    duration_min INTEGER, avg_participation REAL
);
CREATE INDEX IF NOT EXISTS ix_orders ON sim_orders (ticker, algo, pct_adv);

CREATE TABLE IF NOT EXISTS spread_validation (
    ticker TEXT NOT NULL, reference TEXT NOT NULL, level TEXT NOT NULL, estimator TEXT NOT NULL,
    n INTEGER, ref_mean_bps REAL, est_mean_bps REAL, bias_bps REAL, mae_bps REAL, ratio REAL, corr REAL,
    PRIMARY KEY (ticker, reference, level, estimator)
);
