"""
scripts/make_snapshot.py — build / refresh the committed cache snapshot.

Run from the repo root after your daily data fetch:

    python scripts/make_snapshot.py                     # snapshot the existing DB
    python scripts/make_snapshot.py --universe nifty100 # download, then snapshot
    python scripts/make_snapshot.py --universe full

Then commit data/cache_snapshot.db.gz. On a Streamlit Cloud cold start the app
restores from it instead of doing a full yfinance download.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import INDEX_TICKERS, NIFTY50, NIFTY_NEXT50, NIFTY_MIDCAP
from data.cache import (
    get_conn, batch_download_missing, delta_update_parallel, cache_stats,
)
from data.snapshot import export_snapshot, SNAPSHOT_PATH

UNIVERSES = {
    "indices":  [],
    "nifty50":  NIFTY50,
    "next50":   NIFTY_NEXT50,
    "midcap":   NIFTY_MIDCAP,
    "nifty100": NIFTY50 + NIFTY_NEXT50,
    "full":     NIFTY50 + NIFTY_NEXT50 + NIFTY_MIDCAP,
}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build/refresh the committed cache snapshot.")
    ap.add_argument(
        "--universe", choices=sorted(UNIVERSES), default=None,
        help="Download this universe before snapshotting. "
             "Omit to snapshot the existing DB as-is.")
    args = ap.parse_args()

    conn = get_conn()
    try:
        if args.universe is not None:
            tickers = list(dict.fromkeys(
                UNIVERSES[args.universe] + list(INDEX_TICKERS.values())))
            print(f"Building cache for '{args.universe}' "
                  f"({len(tickers)} tickers incl. indices) …")
            batch_download_missing(tickers, conn, status_fn=print)
            delta_update_parallel(tickers, conn, status_fn=print)

        stats = cache_stats(conn)
        if stats["rows"] == 0:
            print("DB is empty — run the app once, or pass --universe to "
                  "populate it before snapshotting.")
            return 1

        size = export_snapshot(conn)
        print(f"\nSnapshot written: {SNAPSHOT_PATH}")
        print(f"  tickers : {stats['tickers']}")
        print(f"  rows    : {stats['rows']:,}")
        print(f"  range   : {stats['oldest']} -> {stats['newest']}")
        print(f"  size    : {size / 1024 / 1024:.1f} MB (gzipped)")
        print("\nNow commit it:")
        print(f"  git add {SNAPSHOT_PATH.as_posix()}")
        print('  git commit -m "Refresh cache snapshot"')
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
