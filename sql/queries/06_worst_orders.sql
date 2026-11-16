-- Outlier review: the 20 most expensive simulated orders vs arrival, with context.
SELECT order_id, ticker, date, algo, ROUND(pct_adv * 100, 0) AS pct_adv,
       CASE side WHEN 1 THEN 'BUY' ELSE 'SELL' END AS side,
       ROUND(is_total_bps, 1) AS is_bps, ROUND(timing_bps, 1) AS timing_bps,
       ROUND(impact_bps, 1) AS impact_bps, ROUND(fill_rate * 100, 0) AS fill_pct
FROM sim_orders
ORDER BY is_total_bps DESC
LIMIT 20;
