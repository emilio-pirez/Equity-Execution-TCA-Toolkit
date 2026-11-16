-- Implementation-shortfall decomposition (bps of parent notional):
-- total = timing + spread + impact + opportunity (exact identity, see src/tca/analysis.py).
SELECT
    ticker, algo,
    ROUND(pct_adv * 100, 0)          AS pct_adv,
    ROUND(AVG(spread_bps), 2)        AS spread_bps,
    ROUND(AVG(impact_bps), 2)        AS impact_bps,
    ROUND(AVG(timing_bps), 2)        AS timing_bps,
    ROUND(AVG(opportunity_bps), 2)   AS opportunity_bps,
    ROUND(AVG(is_total_bps), 2)      AS total_bps,
    ROUND(AVG(spread_bps + impact_bps + timing_bps + opportunity_bps) - AVG(is_total_bps), 6) AS identity_check
FROM sim_orders
GROUP BY ticker, algo, pct_adv
ORDER BY ticker, pct_adv, algo;
