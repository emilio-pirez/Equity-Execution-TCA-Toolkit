---
title: "What does it cost to trade Mexican equities? An execution & TCA toolkit"
---

# What does it cost to trade Mexican equities?

*Draft write-up for emilio-pirez.github.io. Code, data and methodology:
[github.com/emilio-pirez/equity-execution-tca-toolkit](https://github.com/emilio-pirez/equity-execution-tca-toolkit).*

An execution trader's job is to turn a portfolio manager's order into fills at the lowest
total cost: the spread you pay, the price you push, and the market drift you're exposed to
while you work the order. I built a small toolkit to put numbers on those three costs for
five BMV names (Walmex, Grupo México, América Móvil, Banorte, NAFTRAC) against IVV and Apple.
It runs on public 1-minute data with a documented formula behind every figure.

**How it works.** Liquidity is measured from 1-minute bars: ADV, volatility, Amihud
illiquidity, a depth proxy, intraday volume curves, and spreads estimated with Corwin-Schultz
and Roll, since public data has no quotes. Then TWAP, VWAP, POV and an Almgren-Chriss
implementation-shortfall schedule are replayed against every session in the sample, with
square-root market impact (cost ≈ spread/2 + 0.7·σ·√participation). Each simulated order gets
a TCA breakdown into spread, impact, timing and opportunity cost.

**What came out.**

1. *Cost scales with √size, and Mexico's constraint is notional.* A full-day VWAP for 10% of ADV
   costs ~41 bps in Banorte or Walmex, about the same as Apple (38). But 5% of ADV is roughly
   USD 2m in a typical Mexican name versus hundreds of millions in the US.
2. *Depth differs by ~400x.* Per basis point of 1-minute price move, US names absorb about 400
   times more dollars. Only about half of a Mexican name's daily volume trades on the lit
   continuous book. The rest goes through auctions and block crosses.
3. *Avoid the open.* Estimated spreads in the first 30 minutes are 1.6x the midday level in Mexico
   and 3.2x in the US, while volume builds into the close.
4. *Urgency has a price.* For a 10%-ADV Banorte order, front-loading (Almgren-Chriss urgency 3)
   costs about 6 bps more than TWAP and cuts timing risk by ~19%. Schedules replayed on real
   history land on the theoretical efficient frontier.
5. *Know your estimator.* On synthetic data with a known spread, Corwin-Schultz under-reads
   badly when a stock trades only a few times a minute. That is exactly the case in less
   liquid Mexican names. Validating it against Bloomberg quoted spreads is the next step.

**Caveats.** Spreads are estimated, impact is modeled rather than observed, and the sample
is about 20 sessions of 1-minute bars. Public data also lacks BMV's last trading hour after
the US daylight-saving switch. The value is in the method and the traceability, not in any
single number.
