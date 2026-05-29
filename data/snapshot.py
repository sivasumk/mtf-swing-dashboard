"""
data/snapshot.py — repo-committed cache priming for Streamlit Cloud.

Streamlit Community Cloud wipes the container disk on every cold start (after a
redeploy or when the app wakes from sleep), so the SQLite cache would otherwise
be rebuilt from a full yfinance download — the 30-90s blank-screen problem.

We commit a gzipped snapshot of the DB to the repo and restore it on startup
when the live DB is empty. `delta_update_parallel` then only tops up the few
days between the snapshot's date and today.

Stdlib gzip + sqlite3 backup API only — no extra dependencies.
"""

import gzip
import shutil
import sqlite3
import tempfile
from pathlib import Path

from config import DB_PATH

SNAPSHOT_PATH = Path(__file__).parent / "cache_snapshot.db.gz"


def _row_count(db_path: Path) -> int:
    """Rows in the ohlcv table, or 0 if the DB is missing/empty/unreadable."""
    if not db_path.exists():
        return 0
    try:
        conn = sqlite3.connect(str(db_path))
        try:
            return conn.execute("SELECT COUNT(*) FROM ohlcv").fetchone()[0]
        finally:
            conn.close()
    except sqlite3.Error:
        return 0


def export_snapshot(conn: sqlite3.Connection, dest: Path = SNAPSHOT_PATH) -> int:
    """Write a consistent, gzipped copy of the live DB to `dest`.

    Run from scripts/make_snapshot.py after the daily fetch, then commit `dest`.
    backup() is WAL-safe and yields a single self-contained file. Returns the
    snapshot size in bytes.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        clean_db = Path(tmp) / "snapshot.db"
        dest_conn = sqlite3.connect(str(clean_db))
        try:
            conn.backup(dest_conn)
        finally:
            dest_conn.close()
        with open(clean_db, "rb") as f_in, \
             gzip.open(dest, "wb", compresslevel=9) as f_out:
            shutil.copyfileobj(f_in, f_out)
    return dest.stat().st_size


def restore_if_empty(db_path: Path = DB_PATH, snapshot: Path = SNAPSHOT_PATH) -> bool:
    """Restore the snapshot into `db_path` when the live DB has no rows.

    Safe to call on every startup: a no-op once the DB is populated, and a no-op
    if no snapshot is committed. Never clobbers a non-empty DB. Returns True only
    when a restore actually ran.
    """
    if not snapshot.exists():
        return False
    if _row_count(db_path) > 0:
        return False
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(snapshot, "rb") as f_in, open(db_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    return True
