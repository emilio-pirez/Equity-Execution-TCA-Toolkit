-- Q2. Mexican vs US names: spread, depth proxy and modeled impact of a 5% ADV VWAP order.
SELECT
    l.market,
    l.ticker,
    ROUND(l.adv_usd_m, 1)                AS adv_usd_m,
    ROUND(l.cs_spread_bps, 2)            AS cs_spread_bps,
    ROUND(l.roll_spread_bps, 2)          AS roll_spread_bps,
    ROUND(l.model_spread_bps, 2)         AS model_spread_bps,
    ROUND(l.usd_per_bp_1m, 0)            AS usd_per_bp_1m,
    ROUND(l.amihud_bps_per_usd_m, 3)     AS amihud_bps_per_usd_m,
    ROUND(l.daily_vol_bps, 0)            AS daily_vol_bps,
    ROUND(l.coverage_exec_vs_daily * 100, 0) AS exec_volume_pct_of_adv,
    ROUND(AVG(o.impact_bps), 2)          AS vwap_5pct_impact_bps,
    ROUND(AVG(o.is_total_bps), 2)        AS vwap_5pct_total_bps
FROM liquidity_summary AS l
LEFT JOIN sim_orders AS o
       ON o.ticker = l.ticker AND o.algo = 'VWAP' AND ABS(o.pct_adv - 0.05) < 1e-9
GROUP BY l.ticker
ORDER BY l.market, l.adv_usd_m DESC;
