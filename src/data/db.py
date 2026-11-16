"""SQLite persistence: create the schema in sql/schema.sql and load DataFrames.

The database (data/tca.db) is a build artifact: `python -m src.pipeline` recreates it.
It is the source for the SQL queries in sql/queries/ and for an optional Power BI file.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from src.config import DB_PATH, ROOT, UNIVERSE

SCHEMA = ROOT / "sql" / "schema.sql"
QUERIES = ROOT / "sql" / "queries"


def connect(path: Path | str = DB_PATH) -> sqlite3.Connection:
    con = sqlite3.connect(str(path))
    con.execute("PRAGMA foreign_keys = ON")
    return con


def init_db(con: sqlite3.Connection, reset: bool = False) -> None:
    """Create all tables (dropping them first when `reset`)."""
    if reset:
        tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        con.execute("PRAGMA foreign_keys = OFF")
        for t in tables:
            con.execute(f'DROP TABLE IF EXISTS "{t}"')
        con.execute("PRAGMA foreign_keys = ON")
    con.executescript(SCHEMA.read_text())
    con.executemany(
        "INSERT OR REPLACE INTO instruments VALUES (?,?,?,?,?,?)",
        [(i.yf, i.name, i.bbg, i.market, i.asset_type, i.tz) for i in UNIVERSE],
    )
    con.commit()


def _columns(con: sqlite3.Connection, table: str) -> list[str]:
    return [r[1] for r in con.execute(f'PRAGMA table_info("{table}")')]


def insert_df(con: sqlite3.Connection, table: str, df: pd.DataFrame) -> int:
    """Insert the columns of `df` that exist in `table` (INSERT OR REPLACE). Returns rows."""
    cols = [c for c in _columns(con, table) if c in df.columns]
    if not cols or df.empty:
        return 0
    data = df[cols].astype(object).where(df[cols].notna(), None).itertuples(index=False, name=None)
    q = f'INSERT OR REPLACE INTO "{table}" ({",".join(cols)}) VALUES ({",".join("?" * len(cols))})'
    con.executemany(q, data)
    con.commit()
    return len(df)


def bars_to_db_frame(bars: pd.DataFrame, ticker: str, source: str) -> pd.DataFrame:
    """Enriched bars -> bars_1m rows."""
    out = bars.copy()
    out["ticker"] = ticker
    out["source"] = source
    out["ts_local"] = out["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    out["ts_utc"] = out["timestamp"].dt.tz_convert("UTC").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    for c in ("is_open_auction", "is_close_auction", "vol_outlier"):
        out[c] = out[c].astype(int)
    return out


def run_query(con: sqlite3.Connection, name: str, params: dict | None = None) -> pd.DataFrame:
    """Run sql/queries/<name>.sql and return a DataFrame."""
    sql = (QUERIES / f"{name}.sql").read_text()
    return pd.read_sql_query(sql, con, params=params or {})
