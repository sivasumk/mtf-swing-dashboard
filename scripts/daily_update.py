"""
scripts/daily_update.py — one daily run that feeds the learning log.

Intended to run unattended (GitHub Action, after NSE close):

    1. restore the committed snapshot if the live DB is empty,
    2. delta-update every ticker to today's bar,
    3. score the universe (ML off — fast, deterministic),
    4. append today's signals to data/signal_log.csv,
    5. back-fill realised 5-day outcomes for matured rows,
    6. refresh the committed cache snapshot.

Then the workflow commits data/signal_log.csv + data/cache_snapshot.db.gz so the
record survives Streamlit Cloud's ephemeral disk. Stdlib + existing deps only.
"""

import argparse
import sys
from datetime import date
from pathlib import Path

# Emoji in status strings would crash on a cp1252 Windows console; harmless on CI.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import INDEX_TICKERS, NIFTY50, NIFTY_NEXT50, NIFTY_MIDCAP
from data.cache import (
    get_conn, batch_download_missing, delta_update_parallel, cache_stats,
)
from data.snapshot import restore_if_empty, export_snapshot, SNAPSHOT_PATH
from utils.universe import build_universe_df
from utils.track_record import append_signals, backfill_outcomes, load_log

UNIVERSES = {
    "nifty50":  NIFTY50,
    "next50":   NIFTY_NEXT50,
    "midcap":   NIFTY_MIDCAP,
    "nifty100": NIFTY50 + NIFTY_NEXT50,
    "full":     NIFTY50 + NIFTY_NEXT50 + NIFTY_MIDCAP,
}


class _Prog:
    """Minimal stand-in for st.progress — prints every ~10%."""
    def progress(self, pct, text=""):
        if int(pct * 100) % 10 == 0:
            print(f"  ... {int(pct * 100):3d}%  {text}", flush=True)


def _status(msg):
    if msg:
        print(" ", msg, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Daily learning-log update.")
    ap.add_argument("--universe", choices=sorted(UNIVERSES), default="nifty100",
                    help="Universe to scan (default: nifty100).")
    ap.add_argument("--run-date", default=None,
                    help="Override the run date (YYYY-MM-DD); defaults to today.")
    args = ap.parse_args()

    universe = UNIVERSES[args.universe]
    all_tickers = list(dict.fromkeys(universe + list(INDEX_TICKERS.values())))
    run_date = args.run_date or date.today().isoformat()

    conn = get_conn()
    try:
        if restore_if_empty():
            print("Restored cache from committed snapshot.", flush=True)

        print(f"Universe: {len(universe)} stocks (+{len(INDEX_TICKERS)} indices)",
              flush=True)
        print("Downloading any missing history ...", flush=True)
        batch_download_missing(all_tickers, conn, status_fn=_status)
        print("Delta-updating to today ...", flush=True)
        delta_update_parallel(all_tickers, conn, status_fn=_status)

        print("Scoring universe (ML off) ...", flush=True)
        df = build_universe_df(universe, conn, run_ml=False, progress_bar=_Prog())
        if df.empty:
            print("No data computed — aborting.", flush=True)
            return 1

        added = append_signals(df, conn, run_date=run_date)
        filled = backfill_outcomes(conn)
        print(f"\nLogged {added} signals for {run_date}; "
              f"back-filled {filled} matured outcomes.", flush=True)

        log = load_log()
        scored = int((log["outcome_filled"].fillna(0).astype(int) == 1).sum()) \
            if not log.empty else 0
        print(f"Signal log now holds {len(log)} rows ({scored} scored).", flush=True)

        size = export_snapshot(conn)
        stats = cache_stats(conn)
        print(f"\nSnapshot refreshed: {SNAPSHOT_PATH.name} "
              f"({size / 1024 / 1024:.1f} MB, {stats['rows']:,} rows, "
              f"{stats['oldest']} -> {stats['newest']}).", flush=True)
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
