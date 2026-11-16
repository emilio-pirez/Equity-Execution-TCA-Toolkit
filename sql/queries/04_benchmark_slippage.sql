-- Slippage of each algorithm against the three standard benchmarks (bps, + = cost),
-- buys only, so the sign of drift is visible. Arrival = decision price,
-- VWAP = market VWAP over the execution interval, Close = official close.
SELECT
    ticker, algo,
    ROUND(pct_adv * 100, 0)       AS pct_adv,
    ROUND(AVG(vs_arrival_bps), 2) AS vs_arrival_bps,
    ROUND(AVG(vs_vwap_bps), 2)    AS vs_vwap_bps,
    ROUND(AVG(vs_close_bps), 2)   AS vs_close_bps,
    COUNT(*)                      AS n
FROM sim_orders
WHERE side = 1
GROUP BY ticker, algo, pct_adv
ORDER BY ticker, pct_adv, algo;
