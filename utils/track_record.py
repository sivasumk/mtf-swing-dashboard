"""
utils/track_record.py — persistent signal-outcome log + learning analytics.

Records each daily scan's signals, back-fills the realised 5-day forward return
once the horizon elapses, then aggregates empirical hit-rate / expectancy per
"setup signature". The app uses this to annotate today's candidates with how
that kind of setup has actually performed (annotate-only; ranking untouched).

Persistence is a committed CSV (data/signal_log.csv) refreshed by a daily
GitHub Action, so it survives Streamlit Cloud's ephemeral disk. Stdlib + pandas
only — no new dependencies.
"""

import re
from datetime import date as _date

import numpy as np
import pandas as pd

from config import BASE_DIR
from data.cache import read_cache

LOG_PATH = BASE_DIR / "data" / "signal_log.csv"
HORIZON = 5   # trading days forward used to score a signal

LOG_COLS = [
    "run_date", "bar_date", "ticker", "trade", "trend", "regime",
    "price", "rank_score", "mom_score", "rs_score", "rsi",
    "rsi_zone", "adx_str", "vol_spurt", "pattern", "setup_sig",
    "fwd_ret", "dir_hit", "outcome_filled",
]

_NON_ASCII = re.compile(r"[^\x00-\x7F]+")


# ── small coercion helpers ────────────────────────────────────
def _txt(v) -> str:
    """Strip emoji / non-ASCII so the CSV stays clean and git-diffable."""
    return _NON_ASCII.sub("", str(v)).strip()


def _f(v):
    try:
        return round(float(v), 4)
    except (TypeError, ValueError):
        return np.nan


def _direction(trade, trend) -> str:
    t = str(trade)
    if "LONG" in t:
        return "LONG"
    if "SHORT" in t:
        return "SHORT"
    tr = str(trend)
    if "Bullish" in tr:
        return "LONG"
    if "Bearish" in tr:
        return "SHORT"
    return "NEUTRAL"


def _rsi_bucket(rsi) -> str:
    try:
        r = float(rsi)
    except (TypeError, ValueError):
        return "NA"
    if r < 30:
        return "OS"
    if r < 45:
        return "Lo"
    if r < 55:
        return "Mid"
    if r < 70:
        return "Hi"
    return "OB"


def _regime_tok(regime) -> str:
    r = str(regime)
    if "Trend" in r:
        return "Trend"
    if "Rang" in r:
        return "Range"
    return "Neut"


def _vol_tok(vol_spurt) -> str:
    v = str(vol_spurt)
    if "SPURT" in v:
        return "Spurt"
    if "DUMP" in v:
        return "Dump"
    if "Abv5" in v or "Abv20" in v or "High" in v:
        return "High"
    return "Norm"


def setup_signature(direction, regime, rsi, vol_spurt) -> str:
    """Compact categorical key used to bucket and score similar setups."""
    return f"{direction}|{_regime_tok(regime)}|RSI{_rsi_bucket(rsi)}|{_vol_tok(vol_spurt)}"


def _row_signature(row) -> str:
    return setup_signature(
        _direction(row.get("Trade"), row.get("D_Trend")),
        row.get("Regime"), row.get("RSI"), row.get("VolSpurt"),
    )


def _cache_key(ticker) -> str:
    """Map a display ticker back to its cache key (the universe strips '.NS')."""
    t = str(ticker)
    if t.startswith("^") or t.endswith(".NS"):
        return t
    return t + ".NS"


# ── log IO ────────────────────────────────────────────────────
def load_log() -> pd.DataFrame:
    if not LOG_PATH.exists():
        return pd.DataFrame(columns=LOG_COLS)
    try:
        return pd.read_csv(LOG_PATH)
    except Exception:
        return pd.DataFrame(columns=LOG_COLS)


def _save_log(df: pd.DataFrame) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(LOG_PATH, index=False)


# ── append today's signals ────────────────────────────────────
def append_signals(daily_df: pd.DataFrame, conn, run_date: str | None = None) -> int:
    """Append one row per scanned ticker for `run_date` (defaults to today).

    Re-running on the same day overwrites that day's rows (idempotent).
    Outcomes are left blank — filled later by backfill_outcomes().
    """
    if daily_df is None or daily_df.empty:
        return 0
    run_date = run_date or _date.today().isoformat()

    recs = []
    for _, r in daily_df.iterrows():
        ticker = r.get("Ticker")
        if not ticker:
            continue
        cached = read_cache(_cache_key(ticker), conn)
        bar_date = str(cached.index.max().date()) if not cached.empty else run_date
        direction = _direction(r.get("Trade"), r.get("D_Trend"))
        recs.append({
            "run_date": run_date,
            "bar_date": bar_date,
            "ticker": ticker,
            "trade": direction,
            "trend": _txt(r.get("D_Trend")),
            "regime": _regime_tok(r.get("Regime")),
            "price": _f(r.get("Price")),
            "rank_score": _f(r.get("Rank_Score")),
            "mom_score": _f(r.get("MomScore")),
            "rs_score": _f(r.get("RS_Score")),
            "rsi": _f(r.get("RSI")),
            "rsi_zone": _txt(r.get("RSI_Zone")),
            "adx_str": _txt(r.get("ADX_Str")),
            "vol_spurt": _txt(r.get("VolSpurt")),
            "pattern": _txt(r.get("Pattern")),
            "setup_sig": _row_signature(r),
            "fwd_ret": np.nan,
            "dir_hit": np.nan,
            "outcome_filled": 0,
        })

    new = pd.DataFrame(recs, columns=LOG_COLS)
    log = load_log()
    if not log.empty:
        log = log[log["run_date"].astype(str) != run_date]
        out = pd.concat([log, new], ignore_index=True)
    else:
        out = new
    _save_log(out)
    return len(new)


# ── back-fill realised outcomes ───────────────────────────────
def backfill_outcomes(conn, horizon: int = HORIZON) -> int:
    """Fill fwd_ret + dir_hit for logged rows whose horizon has elapsed.

    Direction-aware hit: LONG hits if price rose, SHORT if it fell. Uses the
    cached close series, anchored on the signal's bar_date. Rows without enough
    forward bars yet are left pending. Returns the number newly filled.
    """
    log = load_log()
    if log.empty:
        return 0
    filled_mask = log["outcome_filled"].fillna(0).astype(int) == 1
    pending = log[~filled_mask]
    if pending.empty:
        return 0

    close_cache: dict[str, pd.Series] = {}
    filled = 0
    for idx, r in pending.iterrows():
        t = r["ticker"]
        if t not in close_cache:
            df = read_cache(_cache_key(t), conn)
            close_cache[t] = df["Close"] if not df.empty else pd.Series(dtype=float)
        close = close_cache[t]
        if close.empty:
            continue
        bar = pd.Timestamp(r["bar_date"])
        pos = close.index.searchsorted(bar, side="right") - 1
        if pos < 0 or pos + horizon >= len(close):
            continue   # not enough forward data yet
        p0 = float(close.iloc[pos])
        p1 = float(close.iloc[pos + horizon])
        if p0 <= 0:
            continue
        fwd = p1 / p0 - 1.0
        direction = r["trade"]
        if direction == "LONG":
            hit = 1 if fwd > 0 else 0
        elif direction == "SHORT":
            hit = 1 if fwd < 0 else 0
        else:
            hit = np.nan
        log.at[idx, "fwd_ret"] = round(fwd, 4)
        log.at[idx, "dir_hit"] = hit
        log.at[idx, "outcome_filled"] = 1
        filled += 1

    if filled:
        _save_log(log)
    return filled


# ── aggregate learning ────────────────────────────────────────
def setup_stats(log: pd.DataFrame | None = None, min_n: int = 1) -> pd.DataFrame:
    """Per-setup hit-rate and direction-adjusted average return (filled only)."""
    if log is None:
        log = load_log()
    if log.empty:
        return pd.DataFrame()
    done = log[log["outcome_filled"].fillna(0).astype(int) == 1].copy()
    done = done[done["trade"].isin(["LONG", "SHORT"])]
    if done.empty:
        return pd.DataFrame()

    sign = done["trade"].map({"LONG": 1, "SHORT": -1})
    done["edge_ret"] = done["fwd_ret"] * sign

    g = done.groupby("setup_sig").agg(
        n=("fwd_ret", "size"),
        hit_rate=("dir_hit", "mean"),
        avg_ret=("edge_ret", "mean"),
    ).reset_index()
    g["hit_rate"] = (g["hit_rate"] * 100).round(1)
    g["avg_ret"] = (g["avg_ret"] * 100).round(2)
    g = g[g["n"] >= min_n].sort_values(["hit_rate", "n"], ascending=False)
    return g


def annotate(df: pd.DataFrame, stats: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add Edge%/EdgeRet%/EdgeN to candidates from historical setup stats.

    No-op (returns df unchanged) when nothing has been scored yet, so the app
    behaves exactly as before until the log matures.
    """
    if df is None or df.empty:
        return df
    if stats is None:
        stats = setup_stats()
    if stats is None or stats.empty:
        return df

    out = df.copy()
    sigs = out.apply(_row_signature, axis=1)
    m = stats.set_index("setup_sig")
    out["Edge%"] = sigs.map(m["hit_rate"])
    out["EdgeRet%"] = sigs.map(m["avg_ret"])
    out["EdgeN"] = sigs.map(m["n"])
    return out
