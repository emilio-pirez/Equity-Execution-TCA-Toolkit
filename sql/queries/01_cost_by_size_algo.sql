-- Q1. Expected cost (implementation shortfall vs arrival, bps) of executing
-- 1/5/10/20% of ADV, by ticker and algorithm. Buys and sells pooled so that the
-- market drift of the sample averages out; risk = std dev across orders.
SELECT
    o.ticker,
    o.algo,
    ROUND(o.pct_adv * 100, 0)                                        AS pct_adv,
    COUNT(*)                                                         AS n_orders,
    ROUND(AVG(o.is_total_bps), 2)                                    AS mean_is_bps,
    ROUND(SQRT(AVG(o.is_total_bps * o.is_total_bps)
               - AVG(o.is_total_bps) * AVG(o.is_total_bps)), 2)      AS std_is_bps,
    ROUND(AVG(o.spread_bps + o.impact_bps), 2)                       AS mean_spread_plus_impact_bps,
    ROUND(AVG(o.fill_rate) * 100, 1)                                 AS fill_pct,
    ROUND(AVG(o.duration_min), 0)                                    AS avg_minutes
FROM sim_orders AS o
GROUP BY o.ticker, o.algo, o.pct_adv
ORDER BY o.ticker, o.pct_adv, mean_is_bps;
