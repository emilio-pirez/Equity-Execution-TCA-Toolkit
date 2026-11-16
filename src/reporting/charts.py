"""Plotly figures shared by the README export (PNG) and the Streamlit dashboard.

Design rules: categorical colours in a fixed order (colour follows the entity, never
its rank), one y-axis per chart, thin marks, legends for >= 2 series, recessive grid.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import UNIVERSE

# Validated categorical palette (light surface), fixed slot order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKET_COLOR = {"MX": SERIES[0], "US": SERIES[1]}
TICKER_COLOR = {i.yf: SERIES[k] for k, i in enumerate(UNIVERSE)}
ALGO_ORDER = ["TWAP", "VWAP", "POV10", "POV20", "IS"]
ALGO_COLOR = {a: SERIES[k] for k, a in enumerate(ALGO_ORDER)}
COMPONENT_COLOR = {"Spread": SERIES[0], "Impact": SERIES[1], "Opportunity": SERIES[2]}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
SHORT = {i.yf: i.yf.replace(".MX", "").replace("ISHRS", "") for i in UNIVERSE}


def _style(fig: go.Figure, title: str, subtitle: str | None = None, height: int = 460) -> go.Figure:
    text = f"<b>{title}</b>" + (f"<br><span style='font-size:12px;color:{INK_2}'>{subtitle}</span>" if subtitle else "")
    fig.update_layout(
        title={"text": text, "x": 0.01, "xanchor": "left"},
        template="plotly_white",
        font={"family": "Inter, Helvetica, Arial, sans-serif", "size": 13, "color": INK},
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        height=height,
        margin={"l": 60, "r": 30, "t": 90 if subtitle else 60, "b": 55},
        legend={"orientation": "h", "y": -0.18, "x": 0, "font": {"color": INK_2}},
        hoverlabel={"bgcolor": "white", "font": {"color": INK}},
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, tickfont={"color": INK_2})
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, tickfont={"color": INK_2})
    return fig


def cost_curve(costs: pd.DataFrame, algo: str = "VWAP") -> go.Figure:
    """Mean IS (bps) vs order size (% ADV) for one algorithm, one line per ticker."""
    d = costs[costs["algo"] == algo]
    fig = go.Figure()
    for t, g in d.groupby("ticker", sort=False):
        g = g.sort_values("pct_adv")
        fig.add_trace(go.Scatter(
            x=g["pct_adv"] * 100, y=g["is_total_bps"], name=SHORT[t], mode="lines+markers",
            line={"width": 2, "color": TICKER_COLOR[t]}, marker={"size": 8},
            hovertemplate=f"{SHORT[t]}<br>%{{x:.0f}}% ADV<br>%{{y:.1f}} bps<extra></extra>",
        ))
    fig.update_xaxes(title="Order size (% of ADV)", tickvals=[1, 5, 10, 20], range=[0, 21])
    fig.update_yaxes(title="Expected cost vs arrival (bps)", rangemode="tozero")
    return _style(fig, f"Expected execution cost by order size — {algo}",
                  "Simulated on ~20 sessions of 1-min bars; buys and sells pooled; square-root impact, k = 0.7")


def algo_decomposition(costs: pd.DataFrame, ticker: str, pct_adv: float) -> go.Figure:
    """Stacked cost components by algorithm for one ticker/size, with risk (std) annotated."""
    d = costs[(costs["ticker"] == ticker) & np.isclose(costs["pct_adv"], pct_adv)].set_index("algo")
    d = d.reindex([a for a in ALGO_ORDER if a in d.index])
    fig = go.Figure()
    for comp, col in (("Spread", "spread_bps"), ("Impact", "impact_bps"), ("Opportunity", "opportunity_bps")):
        fig.add_trace(go.Bar(x=d.index, y=d[col].clip(lower=0), name=comp, marker_color=COMPONENT_COLOR[comp],
                             marker_line={"color": SURFACE, "width": 2},
                             hovertemplate=f"%{{x}} · {comp}: %{{y:.1f}} bps<extra></extra>"))
    for a, r in d.iterrows():
        top = r[["spread_bps", "impact_bps", "opportunity_bps"]].clip(lower=0).sum()
        fig.add_annotation(x=a, y=top, yshift=14, showarrow=False, font={"size": 11, "color": INK_2},
                           text=f"{r['is_total_bps']:.1f} bps · σ {r['is_std_bps']:.0f} · fill {r['fill_rate'] * 100:.0f}%")
    fig.update_layout(barmode="stack", bargap=0.45)
    fig.update_yaxes(title="Mean cost vs arrival (bps)")
    return _style(fig, f"{SHORT[ticker]}: {pct_adv * 100:.0f}% ADV — cost by algorithm",
                  "Stacked = expected cost components; σ = std dev of implementation shortfall across orders (timing risk)")


def liquidity_bars(liq: pd.DataFrame) -> go.Figure:
    """Two panels: modeled spread (bps) and depth proxy (USD per bp, log scale), by ticker."""
    d = liq.sort_values(["market", "usd_per_bp_1m"], ascending=[True, False])
    names = [SHORT[t] for t in d["ticker"]]
    colors = [MARKET_COLOR[m] for m in d["market"]]
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.14,
                        subplot_titles=("Spread estimate (bps)", "Depth proxy: USD traded per 1 bp move (log)"))
    fig.add_trace(go.Bar(y=names, x=d["model_spread_bps"], orientation="h", marker_color=colors, showlegend=False,
                         hovertemplate="%{y}: %{x:.2f} bps<extra></extra>"), 1, 1)
    fig.add_trace(go.Bar(y=names, x=d["usd_per_bp_1m"], orientation="h", marker_color=colors, showlegend=False,
                         hovertemplate="%{y}: $%{x:,.0f} per bp<extra></extra>"), 1, 2)
    for m, c in MARKET_COLOR.items():
        fig.add_trace(go.Bar(x=[None], y=[None], marker_color=c, name="Mexico (BMV)" if m == "MX" else "United States"))
    fig.update_xaxes(type="log", row=1, col=2)
    fig.update_yaxes(autorange="reversed")
    fig.update_annotations(font={"size": 12, "color": INK_2})
    return _style(fig, "Mexican vs US liquidity",
                  "Spread: Corwin-Schultz (Roll for sparse names), floored at 1 tick. Depth: median daily Σ$volume / Σ|1-min return|")


def intraday_profile(profile: pd.DataFrame, value: str = "volume_share", title: str | None = None) -> go.Figure:
    """Market-average intraday curve by 30-min bucket since the open (one line per market)."""
    fig = go.Figure()
    for m, g in profile.groupby("market"):
        g = g.groupby("bucket_min")[value].mean().reset_index()
        y = g[value] * (100 if value == "volume_share" else 1)
        label = "Mexico (BMV)" if m == "MX" else "United States"
        fig.add_trace(go.Scatter(x=g["bucket_min"] / 60, y=y, name=label, mode="lines+markers",
                                 line={"width": 2, "color": MARKET_COLOR[m]}, marker={"size": 8},
                                 hovertemplate=f"{label}<br>+%{{x:.1f}} h<br>%{{y:.2f}}<extra></extra>"))
    unit = "% of continuous-session volume" if value == "volume_share" else "bps"
    fig.update_xaxes(title="Hours since the continuous open (30-min buckets)")
    fig.update_yaxes(title=unit, rangemode="tozero")
    return _style(fig, title or ("Intraday volume profile" if value == "volume_share" else "Intraday spread profile"),
                  "Average of names in each market; BMV buckets are measured from its open, which moves with US DST")


def ac_frontier_chart(analytic: pd.DataFrame, simulated: pd.DataFrame, ticker: str) -> go.Figure:
    """Almgren-Chriss efficient frontier (analytic) with simulated strategies overlaid."""
    keep = {"IS u=0", "IS u=3", "IS u=12", "VWAP", "POV10"}
    simulated = simulated.assign(tag=[(("TWAP (u=0)" if l == "IS u=0" else l) if l in keep else "") for l in simulated["label"]])
    max_u = float(simulated["urgency"].max()) * 1.5
    analytic = analytic[analytic["urgency"] <= max_u]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=analytic["risk_bps"], y=analytic["cost_bps"], mode="lines", name="AC frontier (analytic)",
                             line={"width": 2, "color": SERIES[0]},
                             hovertemplate="risk %{x:.1f} bps<br>cost %{y:.1f} bps<extra></extra>"))
    fig.add_trace(go.Scatter(x=simulated["risk_bps"], y=simulated["cost_bps"], mode="markers+text", name="Simulated on history",
                             marker={"size": 10, "color": SERIES[1], "line": {"color": SURFACE, "width": 2}},
                             text=simulated["tag"], textposition=["bottom center" if l == "VWAP" else "top center" for l in simulated["label"]], textfont={"size": 11, "color": INK_2},
                             customdata=simulated["label"],
                             hovertemplate="%{customdata}<br>risk %{x:.1f} bps<br>cost %{y:.1f} bps<extra></extra>"))
    fig.update_xaxes(title="Timing risk: std dev of cost (bps)", rangemode="tozero")
    fig.update_yaxes(title="Expected impact + spread cost (bps)", rangemode="tozero")
    return _style(fig, f"Impact vs timing risk — {SHORT[ticker]}, 10% ADV",
                  "Line: Almgren-Chriss with linear impact calibrated to the square-root model. Dots: same schedules simulated on 1-min history")


BLUE_RAMP = ["#9ec5f4", "#5598e7", "#256abf", "#0d366b"]  # sequential, ordinal (light -> dark)


def estimator_recovery(syn: pd.DataFrame) -> go.Figure:
    """Synthetic test: Corwin-Schultz estimate / true spread vs trades per minute, one line per
    true spread level (ordinal blue ramp), plus Roll averaged over levels."""
    fig = go.Figure()
    cs = syn[syn["estimator"] == "cs_bps"]
    levels = sorted(cs["true_spread_bps"].unique())
    for k, lvl in enumerate(levels):
        g = cs[cs["true_spread_bps"] == lvl].groupby("trades_per_minute")["ratio"].mean().reset_index()
        name = f"Corwin-Schultz, true spread {lvl:g} bps"
        fig.add_trace(go.Scatter(x=g["trades_per_minute"], y=g["ratio"], name=name, mode="lines+markers",
                                 line={"width": 2, "color": BLUE_RAMP[k % len(BLUE_RAMP)]}, marker={"size": 8},
                                 hovertemplate=f"{name}<br>%{{x}} trades/min<br>ratio %{{y:.2f}}<extra></extra>"))
    r = syn[syn["estimator"] == "roll_bps"].groupby("trades_per_minute")["ratio"].mean().reset_index()
    fig.add_trace(go.Scatter(x=r["trades_per_minute"], y=r["ratio"], name="Roll (all levels)", mode="lines+markers",
                             line={"width": 2, "color": SERIES[1], "dash": "dash"}, marker={"size": 8},
                             hovertemplate="Roll<br>%{x} trades/min<br>ratio %{y:.2f}<extra></extra>"))
    fig.add_hline(y=1.0, line={"color": INK_2, "width": 1, "dash": "dot"})
    fig.update_xaxes(title="Trades per minute (log)", type="log")
    fig.update_yaxes(title="Estimated / true spread", rangemode="tozero")
    fig.update_layout(legend={"y": -0.22})
    return _style(fig, "Can high/low estimators recover a known spread?",
                  "Synthetic bars, daily vol 2%, 5 seeds. Ratio 1 = unbiased. CS under-reads when trading is sparse, over-reads tight spreads", height=500)


def save(fig: go.Figure, path: Path, width: int = 900) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(path), width=width, scale=2)
