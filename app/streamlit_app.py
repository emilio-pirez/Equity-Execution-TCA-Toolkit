"""Equity Execution & TCA Toolkit - Streamlit dashboard.

Run locally:  streamlit run app/streamlit_app.py
Reads the yfinance cache (data/cache) and pipeline outputs (results/).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import BY_TICKER, FX_TICKER, IMPACT_K, RESULTS_DIR, UNIVERSE  # noqa: E402
from src.data.loaders import load_enriched  # noqa: E402
from src.data.yf_loader import cache_path, load_yf_daily  # noqa: E402
from src.execution.impact import pretrade_cost_bps  # noqa: E402
from src.execution.simulator import TickerContext, run_order, spread_curve_bps  # noqa: E402
from src.liquidity.metrics import liquidity_summary  # noqa: E402
from src.reporting import charts  # noqa: E402

st.set_page_config(page_title="Execution & TCA Toolkit", page_icon="📈", layout="wide")

ALGOS = {"VWAP": ("vwap", {}), "TWAP": ("twap", {}), "POV 10%": ("pov", {"rate": 0.10}),
         "POV 20%": ("pov", {"rate": 0.20}), "IS (urgency 3)": ("is", {"urgency": 3.0})}


@st.cache_data
def results(name: str) -> pd.DataFrame | None:
    for p in (RESULTS_DIR / f"{name}.csv", RESULTS_DIR / f"{name}.csv.gz"):
        if p.exists():
            return pd.read_csv(p)
    return None


@st.cache_resource
def context(ticker: str) -> tuple[TickerContext, dict]:
    bars = load_enriched(ticker)
    daily = load_yf_daily(ticker)
    liq = liquidity_summary(ticker, bars, daily, load_yf_daily(FX_TICKER))
    return TickerContext(ticker, bars, daily, spread_curve_bps(bars, liq["pct_minutes_traded"])), liq


available = [i.yf for i in UNIVERSE if cache_path(i.yf).exists()]
label = {t: f"{charts.SHORT[t]} · {BY_TICKER[t].name} ({BY_TICKER[t].market})" for t in available}

st.title("Equity Execution & TCA Toolkit")
st.caption("Best execution in Mexican (BMV) and US equities/ETFs: liquidity, algorithm cost simulation and "
           "transaction cost analysis on 1-minute yfinance bars. Costs in bps; positive = cost to the trader. "
           "Spreads are *estimated* and impact is *modeled* — see the README for methodology and limitations.")
if not available:
    st.error("No cached data. Run `python -m src.data.yf_loader` then `python -m src.pipeline`.")
    st.stop()

tabs = st.tabs(["Liquidity", "Pre-trade cost", "TCA explorer", "Intraday", "Impact vs risk", "Spread estimators"])

# ---------------------------------------------------------------- Liquidity
with tabs[0]:
    liq = results("liquidity_summary")
    if liq is None:
        st.info("Run `python -m src.pipeline` to generate results.")
    else:
        st.plotly_chart(charts.liquidity_bars(liq), width="stretch")
        show = liq[["ticker", "market", "adv_usd_m", "model_spread_bps", "cs_spread_bps", "roll_spread_bps",
                    "usd_per_bp_1m", "daily_vol_bps", "pct_minutes_traded", "coverage_exec_vs_daily"]]
        st.dataframe(show.style.format({
            "adv_usd_m": "{:,.1f}", "model_spread_bps": "{:.2f}", "cs_spread_bps": "{:.2f}", "roll_spread_bps": "{:.2f}",
            "usd_per_bp_1m": "{:,.0f}", "daily_vol_bps": "{:.0f}", "pct_minutes_traded": "{:.0%}",
            "coverage_exec_vs_daily": "{:.0%}"}), hide_index=True, width="stretch")
        st.caption("coverage_exec_vs_daily: share of official daily volume visible as executable continuous-session "
                   "1-min volume (excludes auctions, block crosses / bad prints, and BMV's last hour missing in yfinance).")

# ---------------------------------------------------------------- Pre-trade
with tabs[1]:
    c1, c2, c3 = st.columns([2, 1, 1])
    t = c1.selectbox("Ticker", available, format_func=label.get, key="pt_t")
    pct = c2.slider("Order size (% ADV)", 0.5, 30.0, 5.0, 0.5) / 100
    algo_name = c3.selectbox("Algorithm", list(ALGOS), key="pt_a")
    ctx, lq = context(t)
    pre = pretrade_cost_bps(pct * lq["adv_shares"], lq["adv_shares"], lq["daily_vol_bps"],
                            float(ctx.spread_tod.mean()), IMPACT_K)
    algo, params = ALGOS[algo_name]
    with st.spinner("Simulating on every session in the sample (buys and sells)…"):
        sims = pd.DataFrame([run_order(ctx, d, algo, pct, s, **params)[0] for d in ctx.dates for s in (1, -1)])
    m = st.columns(5)
    m[0].metric("Order", f"{pct * lq['adv_shares']:,.0f} sh", f"USD {pct * lq['adv_usd_m']:,.2f}m", delta_color="off")
    m[1].metric("Pre-trade model", f"{pre['total_bps']:.1f} bps", "spread/2 + k·σ·√(Q/ADV)", delta_color="off")
    m[2].metric("Simulated mean IS", f"{sims['is_total_bps'].mean():.1f} bps")
    m[3].metric("Risk (std of IS)", f"{sims['is_total_bps'].std():.0f} bps")
    m[4].metric("Avg fill rate", f"{sims['fill_rate'].mean():.0%}")
    fig = go.Figure(go.Histogram(x=sims["is_total_bps"], nbinsx=25, marker_color=charts.SERIES[0],
                                 marker_line={"color": charts.SURFACE, "width": 2},
                                 hovertemplate="%{x} bps: %{y} orders<extra></extra>"))
    fig.add_vline(x=sims["is_total_bps"].mean(), line={"color": charts.INK_2, "dash": "dot"})
    fig.update_xaxes(title="Implementation shortfall vs arrival (bps)")
    fig.update_yaxes(title="Orders")
    st.plotly_chart(charts._style(fig, f"Distribution of simulated cost — {charts.SHORT[t]}, {algo_name}, {pct:.1%} ADV",
                                  "Width = timing risk (market drift while working the order); mean = spread + impact"),
                    width="stretch")
    comp = sims[["spread_bps", "impact_bps", "timing_bps", "opportunity_bps"]].mean().rename("mean bps").to_frame().T
    st.dataframe(comp.style.format("{:.2f}"), hide_index=True)

# ---------------------------------------------------------------- TCA explorer
with tabs[2]:
    c = st.columns(5)
    t = c[0].selectbox("Ticker", available, format_func=label.get, key="tca_t")
    ctx, lq = context(t)
    d = c[1].selectbox("Date", ctx.dates, index=len(ctx.dates) - 1, format_func=lambda x: x.strftime("%Y-%m-%d"))
    algo_name = c[2].selectbox("Algorithm", list(ALGOS), key="tca_a")
    pct = c[3].select_slider("Size (% ADV)", [1, 5, 10, 20], 5, key="tca_s") / 100
    side = 1 if c[4].radio("Side", ["Buy", "Sell"], horizontal=True) == "Buy" else -1
    algo, params = ALGOS[algo_name]
    s, fills = run_order(ctx, d, algo, pct, side, **params)
    g = ctx.grid(d)
    k = st.columns(6)
    k[0].metric("vs Arrival", f"{s['vs_arrival_bps']:.1f} bps")
    k[1].metric("vs Interval VWAP", f"{s['vs_vwap_bps']:.1f} bps")
    k[2].metric("vs Close", f"{s['vs_close_bps']:.1f} bps")
    k[3].metric("Fill", f"{s['fill_rate']:.0%}")
    k[4].metric("Duration", f"{s['duration_min']} min")
    k[5].metric("Avg participation", f"{s['avg_participation']:.1%}")
    traded = fills[fills["qty"] > 0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g["timestamp"], y=g["ref_px"], name="Market (typical price)", mode="lines",
                             line={"width": 1.5, "color": charts.INK_2}))
    fig.add_trace(go.Scatter(x=traded["timestamp"], y=traded["fill_px"], name="Our fills", mode="markers",
                             marker={"size": 6 + 14 * np.sqrt(traded["qty"] / traded["qty"].max()), "color": charts.SERIES[1],
                                     "line": {"color": charts.SURFACE, "width": 1}},
                             customdata=np.c_[traded["qty"], traded["participation"] * 100],
                             hovertemplate="%{x|%H:%M}<br>%{customdata[0]:,.0f} sh @ %{y:.2f}<br>participation %{customdata[1]:.1f}%<extra></extra>"))
    for name, px_, dash in (("Arrival", s["arrival_px"], "dot"), ("Interval VWAP", s["vwap_px"], "dash")):
        fig.add_hline(y=px_, line={"color": charts.SERIES[0], "dash": dash, "width": 1},
                      annotation_text=name, annotation_font_color=charts.INK_2)
    fig.update_yaxes(title=f"Price ({'MXN' if BY_TICKER[t].market == 'MX' else 'USD'})")
    st.plotly_chart(charts._style(fig, f"{charts.SHORT[t]} {d:%Y-%m-%d} — {algo_name} {'buy' if side == 1 else 'sell'} {pct:.0%} ADV",
                                  "Fill = reference price ± (half spread + k·σ·√participation)"), width="stretch")
    cum = go.Figure()
    cum.add_trace(go.Scatter(x=fills["timestamp"], y=fills["qty"].cumsum() / s["qty"] * 100, name="Our execution",
                             line={"width": 2, "color": charts.SERIES[1]}))
    mv = g["exec_volume"].cumsum()
    cum.add_trace(go.Scatter(x=g["timestamp"], y=mv / mv.iloc[-1] * 100, name="Market executable volume",
                             line={"width": 2, "color": charts.SERIES[0]}))
    cum.update_yaxes(title="% complete", range=[0, 102])
    st.plotly_chart(charts._style(cum, "Execution progress vs market volume", height=360), width="stretch")
    st.dataframe(pd.DataFrame([{k_: s[k_] for k_ in ("timing_bps", "spread_bps", "impact_bps", "opportunity_bps", "is_total_bps")}])
                 .style.format("{:.2f}"), hide_index=True)

# ---------------------------------------------------------------- Intraday
with tabs[3]:
    prof = results("intraday_profile")
    if prof is not None:
        a, b = st.columns(2)
        a.plotly_chart(charts.intraday_profile(prof, "volume_share"), width="stretch")
        b.plotly_chart(charts.intraday_profile(prof, "spread_bps"), width="stretch")
        t = st.selectbox("Single ticker", available, format_func=label.get, key="id_t")
        one = prof[prof["ticker"] == t].assign(market=charts.SHORT[t])
        one_fig = go.Figure(go.Bar(x=one["bucket_min"] / 60, y=one["volume_share"] * 100, marker_color=charts.SERIES[0],
                                   customdata=one["spread_bps"], marker_line={"color": charts.SURFACE, "width": 2},
                                   hovertemplate="+%{x:.1f} h: %{y:.1f}% of volume, spread %{customdata:.2f} bps<extra></extra>"))
        one_fig.update_xaxes(title="Hours since open")
        one_fig.update_yaxes(title="% of continuous volume")
        st.plotly_chart(charts._style(one_fig, f"{charts.SHORT[t]} volume profile (hover for spread)", height=360),
                        width="stretch")

# ---------------------------------------------------------------- AC
with tabs[4]:
    fr, sim = results("ac_frontier"), results("ac_simulated")
    if fr is not None and sim is not None:
        st.plotly_chart(charts.ac_frontier_chart(fr, sim, fr["ticker"].iloc[0]), width="stretch")
        st.dataframe(sim.style.format({"cost_bps": "{:.1f}", "risk_bps": "{:.1f}", "fill_rate": "{:.0%}", "urgency": "{:g}"}),
                     hide_index=True)
        st.markdown("Holdings follow $x_j = \\sinh(\\kappa(N-j))/\\sinh(\\kappa N)$; urgency $u=\\kappa N$. "
                    "$u=0$ is TWAP. Higher urgency trades faster: more impact, less exposure to price drift.")

# ---------------------------------------------------------------- Spread estimators
with tabs[5]:
    syn = results("synthetic_validation")
    if syn is not None:
        st.plotly_chart(charts.estimator_recovery(syn), width="stretch")
    bbg = results("bbg_validation")
    if bbg is None:
        st.info("Bloomberg validation pending: export 1-min bars with bid/ask to data/raw/bbg/ "
                "(see docs/bloomberg_extraction.md) and re-run the pipeline. Only aggregated statistics are published.")
    else:
        st.subheader("Estimators vs Bloomberg quoted spread")
        st.dataframe(bbg, hide_index=True, width="stretch")
