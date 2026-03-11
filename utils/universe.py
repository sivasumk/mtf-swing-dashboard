"""
utils/universe.py — Universe Builder (Orchestration)

Loops over tickers, calls indicators → signals → ML,
assembles the final dashboard DataFrame.
Memory-efficient: del df + gc.collect() every 20 tickers.
"""

import gc
import logging
import sqlite3

import numpy as np
import pandas as pd
import streamlit as st

from config import ML_STRONG_BUY_PROB, ML_STRONG_SELL_PROB
from data.cache import load_ohlcv
from indicators.engine import add_indicators, resample_to_tf
from indicators.signals import (
    compute_signals, format_row, compute_rs_score,
    detect_bullish_engulfing, detect_bearish_engulfing,
    detect_doji, detect_hammer,
)
from ml.model import train_and_predict

log = logging.getLogger(__name__)


# Signals are computed fresh each full run; filtering/sorting is in-memory


def _compute_market_features(
    benchmark_df: pd.DataFrame,
    vix_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute market-wide features once (shared across all tickers).
    These columns are merged into each stock's df before ML training.
    """
    mkt = pd.DataFrame(index=benchmark_df.index)
    bc = benchmark_df["Close"]

    mkt["nifty_ret_5d"]  = bc.pct_change(5).astype("float32")
    mkt["nifty_ret_20d"] = bc.pct_change(20).astype("float32")

    ema200 = bc.ewm(span=200, adjust=False).mean()
    mkt["market_above_ema200"] = (bc > ema200).astype("float32")

    if not vix_df.empty and "Close" in vix_df.columns:
        vix_close = vix_df["Close"].reindex(benchmark_df.index, method="ffill")
        # Normalize VIX: divide by 20 (typical mean) so values center ~1.0
        mkt["vix_level"]      = (vix_close / 20.0).astype("float32")
        mkt["vix_change_5d"]  = vix_close.pct_change(5).astype("float32")
    else:
        mkt["vix_level"]      = np.float32(1.0)
        mkt["vix_change_5d"]  = np.float32(0.0)

    return mkt


def build_universe_df(
    tickers: list[str],
    conn: sqlite3.Connection,
    run_ml: bool,
    progress_bar,
    benchmark_ticker: str = "^NSEI",
) -> pd.DataFrame:
    """
    Main loop: for each ticker →
      1. Load OHLCV from cache
      2. Compute indicators
      3. Compute signals
      4. Detect patterns
      5. Train+predict ML (if enabled)
      6. Assemble row
    Returns complete dashboard DataFrame.
    """
    # Load Nifty 50 benchmark + VIX once for RS and ML market features
    benchmark_df = load_ohlcv(benchmark_ticker, conn)
    vix_df       = load_ohlcv("^INDIAVIX", conn)
    market_cols  = _compute_market_features(benchmark_df, vix_df)

    rows  = []
    n     = len(tickers)
    fails = []

    for i, ticker in enumerate(tickers):
        pct  = (i + 1) / n
        name = ticker.replace(".NS", "")
        progress_bar.progress(pct, text=f"[{i+1}/{n}] {name} — indicators …")

        try:
            # ── Load ─────────────────────────────────────────
            df = load_ohlcv(ticker, conn)
            if df.empty or len(df) < 60:
                fails.append(ticker)
                continue

            # ── Indicators ───────────────────────────────────
            df = add_indicators(df)

            # ── Market regime features (for ML) ─────────────
            for col in market_cols.columns:
                df.loc[:, col] = market_cols[col].reindex(df.index, method="ffill")

            # ── Signals ──────────────────────────────────────
            signals = compute_signals(df)
            if not signals:
                fails.append(ticker)
                continue

            # ── Patterns ─────────────────────────────────────
            b_eng  = detect_bullish_engulfing(df)
            br_eng = detect_bearish_engulfing(df)
            doji   = detect_doji(df)
            hammer = detect_hammer(df)

            # ── ML ───────────────────────────────────────────
            if run_ml:
                progress_bar.progress(pct, text=f"[{i+1}/{n}] {name} — ML …")
                try:
                    ml_prob, ml_acc, ml_reason = _cached_ml(ticker, df)
                except Exception as ml_e:
                    log.warning(f"ML failed {ticker}: {ml_e}")
                    ml_prob, ml_acc, ml_reason = 0.5, 0.0, "ML error"
            else:
                ml_prob, ml_acc, ml_reason = 0.5, 0.0, "ML off"

            # ── Relative Strength ────────────────────────────
            rs = compute_rs_score(df, benchmark_df)

            # ── Assemble row ──────────────────────────────────
            row = format_row(
                ticker, signals,
                b_eng, br_eng, doji, hammer,
                ml_prob, ml_acc, ml_reason,
                rs=rs,
            )
            rows.append(row)

        except Exception as e:
            log.warning(f"Skipped {ticker}: {e}")
            fails.append(ticker)

        finally:
            # Memory cleanup every 20 tickers
            if i % 20 == 19:
                gc.collect()

    if fails:
        log.info(f"Skipped {len(fails)} tickers: {fails[:5]} …")

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)

    # ── RS Rank: percentile rank within the universe (1=strongest) ──
    if "RS_Score" in result.columns and len(result) > 1:
        result["RS_Rank"] = result["RS_Score"].rank(ascending=False, method="min").astype(int)

    # ── Overall Ranking Score (composite) ───────────────────────────
    # Combines MomScore + RS_Score + ML_Prob% into one rank
    if all(c in result.columns for c in ["MomScore","RS_Score","ML_Prob%"]):
        result["Rank_Score"] = (
            result["MomScore"]  * 0.35 +
            result["RS_Score"]  * 0.40 +
            result["ML_Prob%"]  * 0.25
        ).round(1)
        result["Rank"] = result["Rank_Score"].rank(ascending=False, method="min").astype(int)
    elif "MomScore" in result.columns:
        result["Rank"] = result["MomScore"].rank(ascending=False, method="min").astype(int)

    return result


# File-based ML cache: {ticker}_{date}.json in data/ml_cache/
import json
from datetime import date as _date
from pathlib import Path as _Path

_ML_CACHE_DIR = _Path(__file__).parent.parent / 'data' / 'ml_cache'
_ML_CACHE_VER = "v3"   # Bump when features/models change to invalidate cache

def _cached_ml(ticker: str, df) -> tuple[float, float, str]:
    """
    File-based ML cache keyed by ticker + today date + model version.
    Avoids Streamlit hashing the entire DataFrame (which is slow/broken).
    Retrain once per day per ticker.
    """
    _ML_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_key  = f"{ticker.replace('.','_')}_{_date.today().isoformat()}_{_ML_CACHE_VER}"
    cache_file = _ML_CACHE_DIR / f"{cache_key}.json"

    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text())
            return cached['prob'], cached['acc'], cached['reason']
        except Exception:
            pass

    prob, acc, reason = train_and_predict(df)

    try:
        cache_file.write_text(json.dumps({'prob': prob, 'acc': acc, 'reason': reason}))
        # Clean old cache files (keep only today)
        for f in _ML_CACHE_DIR.glob('*.json'):
            if _date.today().isoformat() not in f.name:
                f.unlink(missing_ok=True)
    except Exception:
        pass

    return prob, acc, reason


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """
    Apply all sidebar filter checkboxes to the universe DataFrame.
    filters: dict of {filter_name: bool}
    """
    f = df.copy()

    if filters.get("bullish_d"):
        f = f[f["_trend"] == "Bullish"]
    if filters.get("bearish_d"):
        f = f[f["_trend"] == "Bearish"]
    if filters.get("rsi_bull"):
        f = f[f["_rsi_zone"] == "Bull"]
    if filters.get("rsi_bear"):
        f = f[f["_rsi_zone"] == "Bear"]
    if filters.get("oversold"):
        f = f[f["RSI"] < 30]
    if filters.get("overbought"):
        f = f[f["RSI"] > 70]
    if filters.get("engulf_bull") and "Pattern" in f.columns:
        f = f[f["Pattern"].str.contains("BullEng", na=False)]
    if filters.get("engulf_bear") and "Pattern" in f.columns:
        f = f[f["Pattern"].str.contains("BearEng", na=False)]
    if filters.get("hammer") and "Pattern" in f.columns:
        f = f[f["Pattern"].str.contains("Hammer", na=False)]
    if filters.get("vol_expansion"):
        f = f[f["_vol"] == "Expansion"]
    if filters.get("squeeze"):
        f = f[f["_vol"] == "Squeeze"]
    if filters.get("high_mom"):
        f = f[f["MomScore"] > 65]
    if filters.get("low_mom"):
        f = f[f["MomScore"] < 35]
    if filters.get("trending"):
        f = f[f["_regime"] == "Trending"]
    if filters.get("ranging"):
        f = f[f["_regime"] == "Ranging"]
    if filters.get("ml_buy"):
        f = f[f["_ml_prob"] > ML_STRONG_BUY_PROB]
    if filters.get("ml_sell"):
        f = f[f["_ml_prob"] < ML_STRONG_SELL_PROB]
    if filters.get("above_ema200"):
        f = f[f["_above_ema200"] == 1]
    if filters.get("macd_bull") and "MACD" in f.columns:
        f = f[f["MACD"].astype(str).str.startswith("🟢", na=False)]
    if filters.get("conflict"):
        f = f[f["⚠️Conflict"] == "⚠️"]
    if filters.get("vol_spurt") and "VolSpurt" in f.columns:
        f = f[f["VolSpurt"].isin(["🟢 SPURT×2", "🔴 DUMP×2", "⚡ Abv5+20"])]
    if filters.get("smi_bull") and "SMI" in f.columns:
        f = f[f["SMI"] > 0]
    if filters.get("smi_os") and "SMI" in f.columns:
        f = f[f["SMI"] < -40]
    if filters.get("smi_cross") and "SMI_Cross" in f.columns:
        f = f[f["SMI_Cross"] == 1]

    # RSI range slider
    rmin = filters.get("rsi_min", 0)
    rmax = filters.get("rsi_max", 100)
    f = f[(f["RSI"] >= rmin) & (f["RSI"] <= rmax)]

    return f


# Columns where LOWER value = better (ascending = rank 1 at top)
_ASCENDING_COLS = {"Rank", "RS_Rank"}

def sort_df(df: pd.DataFrame, col: str, ascending: bool) -> pd.DataFrame:
    if col not in df.columns:
        return df

    # For Rank/RS_Rank: override to always ascending (1 = best at top)
    # User checkbox still works for everything else
    if col in _ASCENDING_COLS:
        ascending = True

    # Secondary sort tiebreaker
    secondary = {
        "MTF_Score" : "RS_Score",
        "Rank"      : "MTF_Score",
        "RS_Score"  : "MTF_Score",
        "MomScore"  : "RS_Score",
    }
    sec = secondary.get(col)
    if sec and sec in df.columns:
        sec_asc = sec in _ASCENDING_COLS   # secondary also respects direction
        return df.sort_values(
            [col, sec],
            ascending=[ascending, sec_asc],
        )
    return df.sort_values(col, ascending=ascending)


def universe_stats(df: pd.DataFrame) -> dict:
    """Compute summary statistics for the stats bar."""
    if df.empty:
        return {}
    return {
        "total"    : len(df),
        "bullish"  : (df["_trend"] == "Bullish").sum(),
        "bearish"  : (df["_trend"] == "Bearish").sum(),
        "neutral"  : (df["_trend"] == "Neutral").sum(),
        "trending" : (df["_regime"] == "Trending").sum(),
        "ranging"  : (df["_regime"] == "Ranging").sum(),
        "squeeze"  : (df["_vol"] == "Squeeze").sum(),
        "expansion": (df["_vol"] == "Expansion").sum(),
        "rsi_bull" : (df["_rsi_zone"] == "Bull").sum(),
        "rsi_bear" : (df["_rsi_zone"] == "Bear").sum(),
        "ml_buy"   : (df["_ml_prob"] > ML_STRONG_BUY_PROB).sum() if "_ml_prob" in df.columns else 0,
        "ml_sell"  : (df["_ml_prob"] < ML_STRONG_SELL_PROB).sum() if "_ml_prob" in df.columns else 0,
    }


# ══════════════════════════════════════════════════════════════
#  MTF BUILDER  — runs for Weekly or Monthly timeframe
# ══════════════════════════════════════════════════════════════
def build_universe_tf(
    tickers: list[str],
    conn,
    tf: str,                # "W" or "ME"
    run_ml: bool,
    progress_bar,
    benchmark_ticker: str = "^NSEI",
) -> pd.DataFrame:
    """
    Same pipeline as build_universe_df but resamples daily OHLCV
    to weekly (tf="W") or monthly (tf="ME") before computing indicators.

    Returns compact DataFrame with:
      Ticker, Price, Chg%, Trend, RSI_Zone, ADX_Str, MomScore,
      VolStatus, RSI, ADX, ATR%, ML_Signal, ML_Prob%
    """
    TF_LABEL = {"W": "Weekly", "ME": "Monthly"}
    label = TF_LABEL.get(tf, tf)

    benchmark_daily = load_ohlcv(benchmark_ticker, conn)
    benchmark_tf    = resample_to_tf(benchmark_daily, tf)
    vix_daily       = load_ohlcv("^INDIAVIX", conn)
    vix_tf          = resample_to_tf(vix_daily, tf) if not vix_daily.empty else pd.DataFrame()
    market_cols     = _compute_market_features(benchmark_tf, vix_tf)

    rows  = []
    n     = len(tickers)
    fails = []

    for i, ticker in enumerate(tickers):
        pct  = (i + 1) / n
        name = ticker.replace(".NS", "")
        progress_bar.progress(pct, text=f"[{i+1}/{n}] {name} — {label} indicators …")

        try:
            daily = load_ohlcv(ticker, conn)
            if daily.empty or len(daily) < 60:
                fails.append(ticker)
                continue

            df = resample_to_tf(daily, tf)
            if df.empty:
                fails.append(ticker)
                continue

            df = add_indicators(df)

            # Market regime features (for ML)
            for col in market_cols.columns:
                df.loc[:, col] = market_cols[col].reindex(df.index, method="ffill")

            signals = compute_signals(df)
            if not signals:
                fails.append(ticker)
                continue

            # ML on weekly/monthly — only if enabled
            if run_ml:
                progress_bar.progress(pct, text=f"[{i+1}/{n}] {name} — {label} ML …")
                try:
                    ml_prob, ml_acc, ml_reason = _cached_ml(f"{ticker}_{tf}", df)
                except Exception as ml_e:
                    log.warning(f"ML {tf} failed {ticker}: {ml_e}")
                    ml_prob, ml_acc, ml_reason = 0.5, 0.0, "ML error"
            else:
                ml_prob, ml_acc, ml_reason = 0.5, 0.0, "ML off"

            # Patterns (weekly/monthly engulfing is very significant)
            b_eng  = detect_bullish_engulfing(df)
            br_eng = detect_bearish_engulfing(df)
            doji   = detect_doji(df)
            hammer = detect_hammer(df)

            rs = compute_rs_score(daily, benchmark_daily)   # RS always on daily

            row = format_row(
                ticker, signals,
                b_eng, br_eng, doji, hammer,
                ml_prob, ml_acc, ml_reason,
                rs=rs,
            )
            # Prefix weekly/monthly columns so they don't clash when merged
            tf_row = {"Ticker": row["Ticker"]}
            prefix = tf.replace("ME", "M")   # W_ or M_
            skip   = {"Ticker", "RS_Score", "RS_Rank", "RS_Trend",
                      "RS_1M", "RS_3M", "Rank", "Rank_Score"}
            for k, v in row.items():
                if k in skip:
                    continue
                tf_row[f"{prefix}_{k}"] = v

            rows.append(tf_row)

        except Exception as e:
            log.warning(f"Skipped {ticker} [{tf}]: {e}")
            fails.append(ticker)

        finally:
            if (i + 1) % 20 == 0:
                gc.collect()

    if fails:
        log.info(f"[{label}] Skipped {len(fails)}: {fails[:5]} …")

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    return result


# ══════════════════════════════════════════════════════════════
#  MTF MERGE — combines Daily + Weekly + Monthly into one table
#  and computes MTF_Score (0-3 timeframes aligned)
# ══════════════════════════════════════════════════════════════
def merge_mtf(
    daily_df: pd.DataFrame,
    weekly_df: pd.DataFrame | None = None,
    monthly_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Merge daily, weekly, monthly into a single DataFrame.
    Adds MTF_Score: count of timeframes where Trend agrees with Daily trend.

    MTF_Score:
      3 = Daily + Weekly + Monthly all agree  ← strongest signal
      2 = Two timeframes agree
      1 = Only daily signal (W or M conflicting)
      0 = All timeframes disagree (avoid)
    """
    result = daily_df.copy()

    if weekly_df is not None and not weekly_df.empty:
        result = result.merge(weekly_df, on="Ticker", how="left")

    if monthly_df is not None and not monthly_df.empty:
        result = result.merge(monthly_df, on="Ticker", how="left")

    # Compute MTF_Score
    def _score(row):
        base = row.get("D_Trend", "")
        if not base or base == "Neutral":
            return 1   # neutral daily = no strong view

        score = 1   # daily itself counts as 1
        w_trend = row.get("W_D_Trend", "")
        m_trend = row.get("M_D_Trend", "")

        if w_trend and w_trend == base:
            score += 1
        if m_trend and m_trend == base:
            score += 1
        return score

    if "W_D_Trend" in result.columns or "M_D_Trend" in result.columns:
        result["MTF_Score"] = result.apply(_score, axis=1)
        # Move MTF_Score near the front
        cols = result.columns.tolist()
        if "MTF_Score" in cols:
            cols.remove("MTF_Score")
            insert_after = cols.index("D_Trend") + 1 if "D_Trend" in cols else 4
            cols.insert(insert_after, "MTF_Score")
            result = result[cols]

        # Update Rank to weight MTF_Score
        if all(c in result.columns for c in ["MomScore","RS_Score","ML_Prob%"]):
            result["Rank_Score"] = (
                result["MomScore"]   * 0.30 +
                result["RS_Score"]   * 0.35 +
                result["ML_Prob%"]   * 0.20 +
                result["MTF_Score"]  * (100/3) * 0.15   # scale 0-3 to 0-100
            ).round(1)
            result["Rank"] = result["Rank_Score"].rank(ascending=False, method="min").astype(int)

    return result
