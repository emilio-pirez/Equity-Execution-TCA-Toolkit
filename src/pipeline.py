"""End-to-end pipeline: data -> liquidity -> simulations -> validation -> DB -> charts -> findings.

    python -m src.pipeline              # use cached data in data/cache
    python -m src.pipeline --refresh    # re-download from yfinance first (merges into cache)

Outputs: results/*.csv, results/findings.md, reports/figures/*.png, data/tca.db.
Deterministic: same cache in, same numbers out.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from src.config import (
    BPS, DB_PATH, DEFAULT_SEED, FIGURES_DIR, FX_TICKER, IMPACT_K, ORDER_SIZES_PCT_ADV,
    RESULTS_DIR, UNIVERSE, BY_TICKER,
)
from src.data import db
from src.data.bbg_loader import has_bbg, load_bbg_1m
from src.data.loaders import available_tickers, enrich_bars, load_enriched
from src.data.synthetic import generate_bars
from src.data.yf_loader import load_yf_daily, refresh_universe
from src.execution import algos
from src.execution.impact import pretrade_cost_bps
from src.execution.simulator import ALGO_SET, TickerContext, run_grid, run_order, spread_curve_bps
from src.liquidity.metrics import intraday_realized_vol, liquidity_summary, volume_coverage, volume_profile
from src.liquidity.spread_estimators import validate_estimators
from src.reporting import charts, findings
from src.tca.analysis import cost_table

log = logging.getLogger("pipeline")
AC_TICKER = "GFNORTEO.MX"
AC_SIZE = 0.10
AC_URGENCIES = (0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0)


def build_contexts(tickers: list[str], fx: pd.DataFrame) -> tuple[dict[str, TickerContext], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-ticker simulator contexts plus liquidity, profile and data-quality tables."""
    ctxs, liq_rows, prof_rows, dq_rows = {}, [], [], []
    for t in tickers:
        bars = load_enriched(t)
        daily = load_yf_daily(t)
        liq = liquidity_summary(t, bars, daily, fx)
        spread_tod = spread_curve_bps(bars, liq["pct_minutes_traded"])
        prof = volume_profile(bars)
        liq["model_spread_bps"] = float((spread_tod.reindex(prof.index).ffill().bfill() * prof).sum())
        liq_rows.append(liq)
        prof_rows.append(pd.DataFrame({
            "ticker": t, "market": BY_TICKER[t].market, "bucket_min": prof.index,
            "volume_share": prof.to_numpy(), "spread_bps": spread_tod.reindex(prof.index).ffill().bfill().to_numpy(),
        }))
        cov = volume_coverage(bars, daily)
        g = bars.groupby("date")
        dq_rows.append(pd.DataFrame({
            "ticker": t,
            "date": g.size().index.strftime("%Y-%m-%d"),
            "session_open": g["session_open"].first().to_numpy(),
            "first_bar": g["timestamp"].min().dt.strftime("%H:%M").to_numpy(),
            "last_bar": g["timestamp"].max().dt.strftime("%H:%M").to_numpy(),
            "bars": g.size().to_numpy(),
            "coverage_all": cov["coverage_all"].to_numpy(),
            "coverage_exec": cov["coverage_exec"].to_numpy(),
        }))
        ctxs[t] = TickerContext(t, bars, daily, spread_tod)
    return ctxs, pd.DataFrame(liq_rows), pd.concat(prof_rows, ignore_index=True), pd.concat(dq_rows, ignore_index=True)


def synthetic_validation(seeds=range(5), tpms=(2, 5, 10, 30, 100), spreads=(2.0, 5.0, 10.0, 20.0)) -> pd.DataFrame:
    """Estimator recovery on synthetic bars with a known spread (daily-level accuracy)."""
    rows = []
    for tpm in tpms:
        for sp in spreads:
            for s in seeds:
                b = enrich_bars(generate_bars(n_days=10, trades_per_minute=tpm, mean_spread_bps=sp,
                                              seed=DEFAULT_SEED + s), market="US")
                summ = validate_estimators(b, ref_col="true_spread_bps")["summary"]
                for r in summ[summ["level"] == "daily"].to_dict("records"):
                    rows.append({"trades_per_minute": tpm, "true_spread_bps": sp, "seed": s, **r})
    return pd.DataFrame(rows)


def bloomberg_validation(tickers: list[str]) -> pd.DataFrame:
    """Corwin-Schultz/Roll vs Bloomberg quoted spread. Only aggregated statistics are returned."""
    rows = []
    for t in tickers:
        if not has_bbg(t):
            continue
        bars = enrich_bars(load_bbg_1m(t), ticker=t)
        if bars[["bid", "ask"]].notna().sum().min() == 0:
            log.warning("%s: Bloomberg file has no bid/ask, skipped", t)
            continue
        summ = validate_estimators(bars)["summary"]
        rows += [{"ticker": t, "reference": "bbg_quoted", **r} for r in summ.to_dict("records")]
    if not rows:
        log.info("No Bloomberg files in data/raw/bbg - spread validation vs quoted spreads skipped.")
    return pd.DataFrame(rows)


def ac_analysis(ctx: TickerContext, pct_adv: float = AC_SIZE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Analytic AC frontier + the same schedules (and VWAP/POV) simulated on history."""
    dates = ctx.dates
    n_bars = int(np.median([len(ctx.grid(d)) for d in dates]))
    sigma_d = float(np.median([ctx.prior_stats(d)["sigma_bps"] for d in dates]))
    adv_ = float(np.median([ctx.prior_stats(d)["adv"] for d in dates]))
    rv = intraday_realized_vol(ctx.bars)
    n_per_day = ctx.bars[ctx.bars["is_continuous"]].groupby("date").size()
    sigma_bar = float(np.median(rv / np.sqrt(n_per_day)) * BPS)
    exec_day = float(ctx.bars.groupby("date")["exec_volume"].sum().median())
    q = pct_adv * adv_
    rho = q / (q + exec_day)
    eta = n_bars * ctx.k * sigma_d * np.sqrt(rho)  # linear temp impact calibrated at the uniform rate
    half_spread = float(ctx.spread_tod.mean()) / 2.0
    fr = pd.DataFrame(algos.ac_frontier(n_bars, sigma_bar, eta))
    fr["cost_bps"] += half_spread
    fr = fr.assign(ticker=ctx.ticker, pct_adv=pct_adv, sigma_bar_bps=sigma_bar, eta_bps=eta, n_bars=n_bars)

    strategies = [(f"IS u={u:g}", "is", {"urgency": u}) for u in AC_URGENCIES]
    strategies += [("VWAP", "vwap", {}), ("POV10", "pov", {"rate": 0.10}), ("POV20", "pov", {"rate": 0.20})]
    rows = []
    for label, algo, params in strategies:
        res = [run_order(ctx, d, algo, pct_adv, side, **params)[0] for d in dates for side in (1, -1)]
        res = pd.DataFrame(res)
        rows.append({"label": label, "algo": algo, "urgency": params.get("urgency", np.nan),
                     "cost_bps": res["is_total_bps"].mean(), "risk_bps": res["is_total_bps"].std(),
                     "fill_rate": res["fill_rate"].mean(), "n": len(res)})
    return fr, pd.DataFrame(rows)


def impact_sensitivity(liq: pd.DataFrame) -> pd.DataFrame:
    """Pre-trade model cost by ticker x size for k in {0.5, 0.7, 1.0}."""
    rows = []
    for r in liq.to_dict("records"):
        for pct in ORDER_SIZES_PCT_ADV:
            for k in (0.5, IMPACT_K, 1.0):
                c = pretrade_cost_bps(pct * r["adv_shares"], r["adv_shares"], r["daily_vol_bps"], r["model_spread_bps"], k)
                rows.append({"ticker": r["ticker"], "market": r["market"], "pct_adv": pct, "k": k, **c})
    return pd.DataFrame(rows)


def write_db(ctxs, liq, prof, orders, bbg_val) -> None:
    con = db.connect(DB_PATH)
    db.init_db(con, reset=True)
    for t, ctx in ctxs.items():
        db.insert_df(con, "bars_1m", db.bars_to_db_frame(ctx.bars, t, "yf"))
        d = ctx.daily.assign(ticker=t, date=ctx.daily["date"].dt.strftime("%Y-%m-%d"))
        db.insert_df(con, "daily", d)
    db.insert_df(con, "liquidity_summary", liq)
    db.insert_df(con, "volume_profile", prof)
    db.insert_df(con, "sim_orders", orders.reset_index(drop=True).rename_axis("order_id").reset_index())
    if not bbg_val.empty:
        db.insert_df(con, "spread_validation", bbg_val)
    con.close()


def main(refresh: bool = False) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if refresh:
        log.info("Refreshing yfinance cache: %s", refresh_universe())
    tickers = available_tickers([i.yf for i in UNIVERSE])
    fx = load_yf_daily(FX_TICKER)

    log.info("Liquidity metrics for %s", tickers)
    ctxs, liq, prof, dq = build_contexts(tickers, fx)
    liq.to_csv(RESULTS_DIR / "liquidity_summary.csv", index=False)
    prof.to_csv(RESULTS_DIR / "intraday_profile.csv", index=False)
    dq.to_csv(RESULTS_DIR / "data_quality.csv", index=False)

    log.info("Simulating %d algos x %d sizes x 2 sides per ticker-day", len(ALGO_SET), len(ORDER_SIZES_PCT_ADV))
    orders = pd.concat([run_grid(ctx, ORDER_SIZES_PCT_ADV)[0] for ctx in ctxs.values()], ignore_index=True)
    orders.to_csv(RESULTS_DIR / "orders.csv.gz", index=False)
    costs = cost_table(orders)
    costs.to_csv(RESULTS_DIR / "cost_table.csv", index=False)
    sens = impact_sensitivity(liq)
    sens.to_csv(RESULTS_DIR / "impact_sensitivity.csv", index=False)

    log.info("Spread-estimator validation (synthetic + Bloomberg if present)")
    syn = synthetic_validation()
    syn.to_csv(RESULTS_DIR / "synthetic_validation.csv", index=False)
    bbg_val = bloomberg_validation(tickers)
    if not bbg_val.empty:
        bbg_val.to_csv(RESULTS_DIR / "bbg_validation.csv", index=False)

    log.info("Almgren-Chriss frontier for %s", AC_TICKER)
    ac_t = AC_TICKER if AC_TICKER in ctxs else next(t for t in ctxs if BY_TICKER[t].market == "MX")
    frontier, ac_sim = ac_analysis(ctxs[ac_t])
    frontier.to_csv(RESULTS_DIR / "ac_frontier.csv", index=False)
    ac_sim.to_csv(RESULTS_DIR / "ac_simulated.csv", index=False)

    log.info("Writing SQLite database %s", DB_PATH)
    write_db(ctxs, liq, prof, orders, bbg_val)

    log.info("Charts and findings")
    charts.save(charts.cost_curve(costs, "VWAP"), FIGURES_DIR / "cost_curve_vwap.png")
    charts.save(charts.algo_decomposition(costs, ac_t, 0.10), FIGURES_DIR / "algo_decomposition.png")
    charts.save(charts.liquidity_bars(liq), FIGURES_DIR / "mx_vs_us_liquidity.png", width=1000)
    charts.save(charts.intraday_profile(prof, "volume_share"), FIGURES_DIR / "intraday_volume.png")
    charts.save(charts.intraday_profile(prof, "spread_bps"), FIGURES_DIR / "intraday_spread.png")
    charts.save(charts.ac_frontier_chart(frontier, ac_sim, ac_t), FIGURES_DIR / "ac_frontier.png")
    charts.save(charts.estimator_recovery(syn), FIGURES_DIR / "estimator_recovery.png")
    text = findings.build(liq, costs, prof, dq, syn, frontier, ac_sim, sens, bbg_val, ac_t)
    (RESULTS_DIR / "findings.md").write_text(text)
    log.info("Done. See results/findings.md")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--refresh", action="store_true", help="re-download data from yfinance before running")
    main(p.parse_args().refresh)
