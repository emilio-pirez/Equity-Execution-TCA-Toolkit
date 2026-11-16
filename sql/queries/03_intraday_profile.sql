-- Q3. Intraday liquidity: share of executable continuous-session volume and
-- average 1-minute absolute return by 30-minute bucket since the open (yfinance bars).
WITH b AS (
    SELECT ticker, date,
           (minute_of_session / 30) * 30 AS bucket_min,
           exec_volume,
           (high - low) / ((high + low) / 2.0) * 10000.0 AS range_bps
    FROM bars_1m
    WHERE source = 'yf' AND is_open_auction = 0 AND is_close_auction = 0
),
per_day AS (
    SELECT ticker, date, bucket_min, SUM(exec_volume) AS v, AVG(range_bps) AS range_bps
    FROM b GROUP BY ticker, date, bucket_min
),
lvl AS (
    SELECT ticker, bucket_min, AVG(v) AS avg_v, AVG(range_bps) AS avg_range_bps
    FROM per_day GROUP BY ticker, bucket_min
)
SELECT l.ticker,
       l.bucket_min,
       ROUND(100.0 * l.avg_v / SUM(l.avg_v) OVER (PARTITION BY l.ticker), 2) AS volume_share_pct,
       ROUND(l.avg_range_bps, 2)                                             AS avg_1m_range_bps,
       ROUND(p.spread_bps, 2)                                                AS model_spread_bps
FROM lvl AS l
LEFT JOIN volume_profile AS p ON p.ticker = l.ticker AND p.bucket_min = l.bucket_min
ORDER BY l.ticker, l.bucket_min;
