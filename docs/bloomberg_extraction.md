# Bloomberg extraction (validation data)

Bloomberg data is used **only** to validate the spread estimators against the real quoted
spread. It is licensed: raw bars are never committed (`data/raw/` is in `.gitignore`), and
only aggregated statistics (`results/bbg_validation.csv`) are published.

## What to export

One CSV per ticker, `data/raw/bbg/<STEM>_1min.csv`, where `STEM` is the first token of the
Bloomberg ticker without `*`:

| Name | Bloomberg | File |
|---|---|---|
| Walmex | `WALMEX* MF Equity` | `WALMEX_1min.csv` |
| Grupo México | `GMEXICOB MF Equity` | `GMEXICOB_1min.csv` |
| América Móvil | `AMXB MF Equity` | `AMXB_1min.csv` |
| Banorte | `GFNORTEO MF Equity` | `GFNORTEO_1min.csv` |
| NAFTRAC | `NAFTRACISHRS MF Equity` | `NAFTRACISHRS_1min.csv` |
| IVV | `IVV US Equity` | `IVV_1min.csv` |
| Apple | `AAPL US Equity` | `AAPL_1min.csv` |

Columns (header names are case-insensitive; `date`/`datetime`, `last_price`/`px_last`,
`px_bid`, `px_ask` are accepted as aliases):

```
timestamp,open,high,low,close,volume,bid,ask
2026-11-10 08:31:00,54.12,54.49,54.01,54.29,79287,54.28,54.30
```

* `timestamp`: bar **start**, in exchange local time (CDMX for MF, New York for US), no offset.
* `open/high/low/close/volume`: 1-minute **trade** bars.
* `bid` / `ask`: the last bid / ask of the same minute (close of the 1-minute bid and ask bars).

## Excel add-in (no Python on the terminal)

The easiest route is **Bloomberg ribbon → Spreadsheet Builder → Historical Intraday Bars**,
which writes the formula for you. Pull three blocks per ticker over the same window, all with
1-minute bars:

1. Trade bars: fields open, high, low, close, volume, bar type *Trade*.
2. Bid bars: field close, bar type *Bid*.
3. Ask bars: field close, bar type *Ask*.

The generated formulas look like
`=BDH("WALMEX* MF Equity","OPEN,HIGH,LOW,LAST_PRICE,VOLUME","11/10/2026 08:30","11/14/2026 15:10","BarSz=1","BarTp=T")`
(check the exact override names the wizard produces on your terminal). Then:

* Align the three blocks on the timestamp column (bid/ask bars can be missing in minutes with
  no quote change: fill forward the last quote).
* **Paste as values**, delete the formulas, save as CSV (UTF-8, comma separated).
* Check the terminal's time zone setting (`TZDF<GO>`): export in exchange local time, or
  convert before saving.
* Intraday bar history on the terminal is limited (roughly six months), so export the same
  window as the yfinance sample when possible; the validation does not require it.

## Run

```bash
python -m src.pipeline
```

If files are present, the pipeline writes `results/bbg_validation.csv` (bias in bps, MAE,
ratio and correlation of Corwin-Schultz and Roll vs the quoted spread, by day and by
time-of-day bucket) and the dashboard's *Spread estimators* tab shows it. Without files the
step is skipped with a log message and everything else runs normally.
