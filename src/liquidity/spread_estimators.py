"""Bid-ask spread estimators from OHLC bars, and their validation.

Corwin & Schultz (2012), "A Simple Way to Estimate Bid-Ask Spreads from Daily High
and Low Prices", Journal of Finance 67(2). Applied here to consecutive 1-minute bars.

    beta  = [ln(H_t / L_t)]^2 + [ln(H_{t+1} / L_{t+1})]^2
    gamma = [ln(max(H_t, H_{t+1}) / min(L_t, L_{t+1}))]^2
    alpha = (sqrt(2 beta) - sqrt(beta)) / (3 - 2 sqrt 2)  -  sqrt(gamma / (3 - 2 sqrt 2))
    S     = 2 (e^alpha - 1) / (1 + e^alpha)                 (fraction of price)

Intuition: the high/low ratio contains volatility AND spread; volatility scales with
the length of the interval but the spread does not, so comparing one 2-bar range with
two 1-bar ranges separates them.

Gap adjustment (CS 2012, sec. II.B, applied between bars): if bar t+1 trades entirely
above close_t, H_{t+1} and L_{t+1} are shifted down by (L_{t+1} - C_t); if entirely
below, shifted up by (C_t - H_{t+1}).

Negative estimates: each two-bar estimate S < 0 is set to 0 BEFORE averaging over the
aggregation window (CS 2012's main specification). This biases the mean upward in
low-spread names, which is why `clip_of_mean` is also offered.

Roll (1984), "A Simple Implicit Measure of the Effective Bid-Ask Spread":
    S = 2 * sqrt(-Cov(r_t, r_{t-1}))   if Cov < 0, else 0
with r_t the 1-minute log return (relative spread). Computed per aggregation window.

Pairs/returns never span two trading days. Bars with zero volume (no trade: OHLC is a
carried value) and auction bars are excluded before estimation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import BPS

K_CS = 3.0 - 2.0 * np.sqrt(2.0)


def quoted_spread_bps(bid: pd.Series, ask: pd.Series) -> pd.Series:
    """Quoted spread in bps of the mid: (ask - bid) / ((ask + bid) / 2) * 10,000.

    Crossed/locked or missing quotes (ask <= bid, NaN) return NaN.
    """
    mid = (ask + bid) / 2.0
    s = (ask - bid) / mid * BPS
    return s.where((ask > bid) & (bid > 0))


def _estimation_sample(bars: pd.DataFrame) -> pd.DataFrame:
    """Continuous-session bars with at least one trade, sorted, with a `date` column."""
    df = bars
    if "is_continuous" in df:
        df = df[df["is_continuous"]]
    df = df[(df["volume"] > 0) & (df["high"] >= df["low"]) & (df["low"] > 0)]
    if "date" not in df:
        df = df.assign(date=df["timestamp"].dt.tz_localize(None).dt.normalize())
    return df.sort_values("timestamp")


def corwin_schultz_pairs(bars: pd.DataFrame, adjust_gaps: bool = True) -> pd.Series:
    """Two-bar Corwin-Schultz spread (fraction of price), indexed by the first bar of each pair.

    Pairs are consecutive *available* bars within the same day. Unclipped (can be < 0).
    """
    df = _estimation_sample(bars)
    h, l, c = df["high"].to_numpy(float), df["low"].to_numpy(float), df["close"].to_numpy(float)
    h1, l1 = np.roll(h, -1), np.roll(l, -1)
    if adjust_gaps:
        gap_up = np.clip(l1 - c, 0.0, None)
        gap_dn = np.clip(c - h1, 0.0, None)
        h1, l1 = h1 - gap_up + gap_dn, l1 - gap_up + gap_dn
    beta = np.log(h / l) ** 2 + np.log(h1 / l1) ** 2
    gamma = np.log(np.maximum(h, h1) / np.minimum(l, l1)) ** 2
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / K_CS - np.sqrt(gamma / K_CS)
    s = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    same_day = df["date"].to_numpy() == np.roll(df["date"].to_numpy(), -1)
    same_day[-1] = False
    return pd.Series(np.where(same_day, s, np.nan), index=df.index, name="cs_spread").dropna()


def corwin_schultz_bps(
    bars: pd.DataFrame,
    by: str | list[str] | None = "date",
    method: str = "mean_of_clipped",
    adjust_gaps: bool = True,
) -> pd.Series | float:
    """Corwin-Schultz spread in bps aggregated over `by` (None = whole sample).

    method="mean_of_clipped": mean(max(S_pair, 0))       [CS 2012 main specification]
    method="clip_of_mean":    max(mean(S_pair), 0)       [less upward bias, noisier sign]
    """
    s = corwin_schultz_pairs(bars, adjust_gaps)
    if method == "mean_of_clipped":
        s = s.clip(lower=0.0)
    elif method != "clip_of_mean":
        raise ValueError("method must be 'mean_of_clipped' or 'clip_of_mean'")
    if by is None:
        out = s.mean()
        return max(out, 0.0) * BPS if method == "clip_of_mean" else out * BPS
    cols = [by] if isinstance(by, str) else list(by)
    keys = bars.loc[s.index, cols]
    out = s.groupby([keys[k] for k in cols]).mean()
    if method == "clip_of_mean":
        out = out.clip(lower=0.0)
    return (out * BPS).rename("cs_spread_bps")


def roll_from_returns(r: np.ndarray, r_lag: np.ndarray) -> float:
    """Roll spread (fraction) from paired returns: 2*sqrt(-cov) if cov < 0 else 0."""
    mask = np.isfinite(r) & np.isfinite(r_lag)
    if mask.sum() < 3:
        return np.nan
    cov = np.cov(r[mask], r_lag[mask], ddof=1)[0, 1]
    return 2.0 * np.sqrt(-cov) if cov < 0 else 0.0


def roll_bps(bars: pd.DataFrame, by: str | list[str] | None = "date") -> pd.Series | float:
    """Roll (1984) spread in bps over each `by` window (None = whole sample).

    Returns are 1-minute log close-to-close within a day (consecutive available bars).
    """
    df = _estimation_sample(bars).copy()
    r = np.log(df["close"]).groupby(df["date"]).diff()
    df["r"] = r
    df["r_lag"] = r.groupby(df["date"]).shift(1)
    if by is None:
        return roll_from_returns(df["r"].to_numpy(), df["r_lag"].to_numpy()) * BPS
    out = df.groupby(by).apply(
        lambda g: roll_from_returns(g["r"].to_numpy(), g["r_lag"].to_numpy()), include_groups=False
    )
    return (out * BPS).rename("roll_spread_bps")


def add_tod_bucket(bars: pd.DataFrame, bucket_min: int = 30) -> pd.DataFrame:
    """Add `tod_bucket` = minutes since continuous open floored to `bucket_min`."""
    return bars.assign(tod_bucket=(bars["minute_of_session"] // bucket_min) * bucket_min)


def compare(est: pd.Series, ref: pd.Series) -> dict[str, float]:
    """Accuracy of an estimator vs a reference spread, both in bps, aligned on index.

    bias_bps = mean(est - ref); mae_bps = mean|est - ref|; ratio = mean(est) / mean(ref);
    corr = Pearson correlation across windows; n = number of windows.
    """
    j = pd.concat([est.rename("est"), ref.rename("ref")], axis=1).dropna()
    if j.empty:
        return {"n": 0, "ref_mean_bps": np.nan, "est_mean_bps": np.nan, "bias_bps": np.nan,
                "mae_bps": np.nan, "ratio": np.nan, "corr": np.nan}
    d = j["est"] - j["ref"]
    corr = j["est"].corr(j["ref"]) if len(j) > 2 and j["est"].std() > 0 and j["ref"].std() > 0 else np.nan
    return {
        "n": int(len(j)),
        "ref_mean_bps": float(j["ref"].mean()),
        "est_mean_bps": float(j["est"].mean()),
        "bias_bps": float(d.mean()),
        "mae_bps": float(d.abs().mean()),
        "ratio": float(j["est"].mean() / j["ref"].mean()) if j["ref"].mean() else np.nan,
        "corr": float(corr) if corr == corr else np.nan,
    }


def validate_estimators(
    bars: pd.DataFrame, ref_col: str | None = None, bucket_min: int = 30
) -> dict[str, pd.DataFrame]:
    """Compare Corwin-Schultz and Roll against a reference spread.

    `bars` must be enriched (date, minute_of_session, is_continuous). The reference is
    `ref_col` if given (e.g. "true_spread_bps" for synthetic data) else the quoted
    spread from bid/ask (Bloomberg). Returns
      daily   : per-day estimates and reference
      tod     : per time-of-day bucket estimates and reference
      summary : accuracy table (estimator x level) with bias/MAE/ratio/correlation
    """
    df = add_tod_bucket(bars, bucket_min)
    if "is_continuous" in df:
        df = df[df["is_continuous"]]
    ref = df[ref_col] if ref_col else quoted_spread_bps(df["bid"], df["ask"])
    df = df.assign(ref_bps=ref)
    tables = {}
    for level in ("date", "tod_bucket"):
        t = pd.DataFrame(
            {
                "reference_bps": df.groupby(level)["ref_bps"].mean(),
                "cs_bps": corwin_schultz_bps(df, by=level),
                "cs_clip_of_mean_bps": corwin_schultz_bps(df, by=level, method="clip_of_mean"),
                "roll_bps": roll_bps(df, by=level),
            }
        )
        tables["daily" if level == "date" else "tod"] = t
    rows = []
    for level, t in (("daily", tables["daily"]), ("time_of_day", tables["tod"])):
        for est in ("cs_bps", "cs_clip_of_mean_bps", "roll_bps"):
            rows.append({"level": level, "estimator": est, **compare(t[est], t["reference_bps"])})
    tables["summary"] = pd.DataFrame(rows)
    return tables
