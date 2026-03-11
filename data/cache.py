"""
data/cache.py — SQLite persistence layer
  • get_conn()               : open/init DB
  • read_cache()             : load ticker from DB → DataFrame
  • write_cache()            : upsert rows into DB
  • batch_download_missing() : first-run bulk fetch via yfinance batch API
  • delta_update_parallel()  : daily incremental fetch (parallel HTTP, serial write)
  • load_ohlcv()             : cache-first loader used by universe builder
"""

import sqlite3
import concurrent.futures
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from config import (
    DB_PATH, BATCH_SIZE, MAX_WORKERS,
    TICKER_LOOKBACK, LOOKBACK_DEFAULT,
)

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
#  CONNECTION
# ══════════════════════════════════════════════════════════════
def get_conn() -> sqlite3.Connection:
    """Open (or create) the SQLite DB and ensure schema exists."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-64000")   # 64 MB page cache
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ohlcv (
            ticker  TEXT NOT NULL,
            date    TEXT NOT NULL,
            open    REAL,
            high    REAL,
            low     REAL,
            close   REAL,
            volume  INTEGER,
            PRIMARY KEY (ticker, date)
        )
    """)
    conn.commit()
    return conn


# ══════════════════════════════════════════════════════════════
#  READ / WRITE
# ══════════════════════════════════════════════════════════════
def read_cache(ticker: str, conn: sqlite3.Connection) -> pd.DataFrame:
    """Return full cached OHLCV as DataFrame (float32). Empty DF if nothing cached."""
    try:
        df = pd.read_sql(
            "SELECT date,open,high,low,close,volume "
            "FROM ohlcv WHERE ticker=? ORDER BY date",
            conn, params=(ticker,), parse_dates=["date"],
        )
        if df.empty:
            return pd.DataFrame()
        df = df.set_index("date")
        df.index = pd.DatetimeIndex(df.index)
        df.columns = ["Open", "High", "Low", "Close", "Volume"]
        for c in ["Open", "High", "Low", "Close"]:
            df[c] = df[c].astype("float32")
        df["Volume"] = df["Volume"].astype("int64")
        return df
    except Exception as e:
        log.warning(f"read_cache({ticker}): {e}")
        return pd.DataFrame()


def write_cache(ticker: str, df: pd.DataFrame,
                conn: sqlite3.Connection) -> None:
    """Upsert OHLCV rows for a ticker. Always commits."""
    if df.empty:
        return
    rows = [
        (
            ticker,
            str(idx.date()),
            float(r.Open), float(r.High),
            float(r.Low),  float(r.Close),
            int(r.Volume),
        )
        for idx, r in df.iterrows()
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO ohlcv"
        "(ticker,date,open,high,low,close,volume) VALUES(?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    log.debug(f"write_cache({ticker}): {len(rows)} rows committed")


# ══════════════════════════════════════════════════════════════
#  RAW FETCH  (yfinance)
# ══════════════════════════════════════════════════════════════
def _clean_yf(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalise a raw yfinance DataFrame to clean OHLCV float32."""
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw.droplevel(1, axis=1)
    needed = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in raw.columns]
    if len(needed) < 5:
        return pd.DataFrame()
    df = raw[needed].copy().dropna()
    for c in ["Open", "High", "Low", "Close"]:
        df[c] = df[c].astype("float32")
    df["Volume"] = df["Volume"].astype("int64")
    return df


def fetch_single(ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
    """Download one ticker from yfinance."""
    try:
        kw = dict(start=start, auto_adjust=True, progress=False)
        if end:
            kw["end"] = end
        raw = yf.download(ticker, **kw)
        return _clean_yf(raw)
    except Exception as e:
        log.warning(f"fetch_single({ticker}): {e}")
        return pd.DataFrame()


def _fetch_batch(tickers: list[str], start: str) -> dict[str, pd.DataFrame]:
    """Download multiple tickers in one yfinance call. Returns {ticker: DataFrame}."""
    results: dict[str, pd.DataFrame] = {}
    try:
        raw = yf.download(
            tickers,
            start=start,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
        )
        if raw.empty:
            return results
        for ticker in tickers:
            try:
                if len(tickers) == 1:
                    sub = raw
                else:
                    sub = raw[ticker] if ticker in raw.columns.get_level_values(0) else pd.DataFrame()
                results[ticker] = _clean_yf(sub)
            except Exception:
                results[ticker] = pd.DataFrame()
    except Exception as e:
        log.warning(f"_fetch_batch failed: {e}")
        for t in tickers:
            results[t] = fetch_single(t, start)
    return results


# ══════════════════════════════════════════════════════════════
#  SMART LOADER  (cache-first)
# ══════════════════════════════════════════════════════════════
def load_ohlcv(ticker: str, conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Load OHLCV from cache.
    NOTE: Returns the FULL cached history — trimming is intentionally removed.
    Indicators need as much history as possible (EMA200 needs 200 bars minimum).
    The trim to lookback window was causing data to appear missing on reload.
    """
    return read_cache(ticker, conn)


# ══════════════════════════════════════════════════════════════
#  BATCH DOWNLOAD  (first-run, tickers not yet in cache)
# ══════════════════════════════════════════════════════════════
def batch_download_missing(
    tickers: list[str],
    conn: sqlite3.Connection,
    status_fn=None,
) -> None:
    """
    Check which tickers are absent from the DB and bulk-download them.
    Downloads are parallel (HTTP), but writes are serial (SQLite is not thread-safe).
    """
    existing = {
        row[0]
        for row in conn.execute("SELECT DISTINCT ticker FROM ohlcv").fetchall()
    }
    missing = [t for t in tickers if t not in existing]
    if not missing:
        log.info(f"batch_download_missing: all {len(tickers)} tickers cached, skipping")
        return

    log.info(f"batch_download_missing: {len(missing)} tickers missing from cache")
    total_batches = (len(missing) + BATCH_SIZE - 1) // BATCH_SIZE

    if status_fn:
        status_fn(
            f"📥 First-time download: {len(missing)} tickers "
            f"in {total_batches} batches …"
        )

    for b_idx in range(total_batches):
        batch     = missing[b_idx * BATCH_SIZE: (b_idx + 1) * BATCH_SIZE]
        max_years = max(TICKER_LOOKBACK.get(t, LOOKBACK_DEFAULT) for t in batch)
        start     = str(datetime.today().date() - timedelta(days=365 * max_years))

        if status_fn:
            status_fn(
                f"📥 Batch {b_idx + 1}/{total_batches} — "
                f"{len(batch)} tickers from {start} …"
            )

        # Fetch all in batch (parallel HTTP inside yfinance)
        batch_data = _fetch_batch(batch, start)

        # Write serially — SQLite connections are NOT thread-safe
        for ticker, df in batch_data.items():
            if not df.empty:
                write_cache(ticker, df, conn)
                log.info(f"  cached {ticker}: {len(df)} rows")
            else:
                log.warning(f"  {ticker}: empty response from yfinance")

    if status_fn:
        status_fn("")   # clear message


# ══════════════════════════════════════════════════════════════
#  DELTA UPDATE  (daily incremental)
# ══════════════════════════════════════════════════════════════
def delta_update_parallel(
    tickers: list[str],
    conn: sqlite3.Connection,
    status_fn=None,
) -> None:
    """
    For tickers already in cache that are stale (last bar > 1 day ago),
    fetch only the missing days.

    IMPORTANT: HTTP fetches run in parallel, but DB writes are serial.
    SQLite connections are NOT thread-safe — never write from multiple threads.
    """
    today = datetime.today().date()
    rows  = conn.execute(
        "SELECT ticker, MAX(date) FROM ohlcv GROUP BY ticker"
    ).fetchall()
    last_dates = {row[0]: row[1] for row in rows}

    stale = []
    for t in tickers:
        if t not in last_dates:
            continue
        try:
            ld = datetime.strptime(last_dates[t], "%Y-%m-%d").date()
            # Stale if last bar is older than yesterday (accounts for weekends)
            if ld < today - timedelta(days=1):
                stale.append((t, str(ld + timedelta(days=1))))
        except Exception:
            continue

    if not stale:
        log.info("delta_update: all tickers up to date")
        return

    if status_fn:
        status_fn(f"🔄 Updating {len(stale)} stale tickers …")

    log.info(f"delta_update: fetching {len(stale)} stale tickers")

    # Fetch in parallel (HTTP only — no DB operations in threads)
    def _fetch_one(args):
        ticker, start = args
        return ticker, fetch_single(ticker, start)

    fetched = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for ticker, delta in pool.map(_fetch_one, stale):
            fetched[ticker] = delta

    # Write serially — one ticker at a time, same connection
    updated = 0
    for ticker, delta in fetched.items():
        if not delta.empty:
            write_cache(ticker, delta, conn)
            updated += 1

    log.info(f"delta_update: updated {updated}/{len(stale)} tickers")

    if status_fn:
        status_fn("")


# ══════════════════════════════════════════════════════════════
#  CACHE STATS
# ══════════════════════════════════════════════════════════════
def cache_stats(conn: sqlite3.Connection) -> dict:
    """Return DB statistics for display."""
    try:
        n_rows  = conn.execute("SELECT COUNT(*) FROM ohlcv").fetchone()[0]
        n_tkrs  = conn.execute("SELECT COUNT(DISTINCT ticker) FROM ohlcv").fetchone()[0]
        size_mb = DB_PATH.stat().st_size / 1024 / 1024 if DB_PATH.exists() else 0
        oldest  = conn.execute("SELECT MIN(date) FROM ohlcv").fetchone()[0] or "—"
        newest  = conn.execute("SELECT MAX(date) FROM ohlcv").fetchone()[0] or "—"
        return dict(rows=n_rows, tickers=n_tkrs,
                    size_mb=round(size_mb, 1),
                    oldest=oldest, newest=newest)
    except Exception:
        return dict(rows=0, tickers=0, size_mb=0, oldest="—", newest="—")
