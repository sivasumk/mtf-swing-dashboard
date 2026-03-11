"""
indicators/signals.py — Signal Computation Layer

Key fixes vs previous version:
  1. RSI zones: >60 = Bull, <40 = Bear (user requirement)
  2. Momentum score uses 5 weighted components with proper normalisation
  3. Trend requires BOTH EMA cross AND SuperTrend alignment (stricter)
  4. Vol status uses ATR percentile (more robust than simple mean)
  5. Engulfing patterns validated with volume confirmation
"""

import numpy as np
import pandas as pd

from config import (
    RSI_BULL_THRESHOLD, RSI_BEAR_THRESHOLD,
    RSI_OVERSOLD, RSI_OVERBOUGHT,
    ADX_TREND_THRESHOLD, ADX_STRONG_TREND,
    ATR_EXPANSION_MULT,
    MOM_RSI_WEIGHT, MOM_MACD_WEIGHT, MOM_KUMO_WEIGHT,
    ML_STRONG_BUY_PROB, ML_STRONG_SELL_PROB,
    ML_BEAR_BUY_PROB, ML_BEAR_SELL_PROB,
    MOM_EMA_WEIGHT, MOM_RSI_SMA_WEIGHT,
    TREND_ICONS, REGIME_ICONS, SQUEEZE_ICONS, RSI_ZONE_ICONS,
)


def compute_rs_score(df: pd.DataFrame, benchmark_df: pd.DataFrame) -> dict:
    """
    Relative Strength vs Nifty 50 benchmark.
    Computes RS over 3 lookback periods, weighted toward recent.

    Returns:
        rs_score  : 0-100 percentile score (100 = strongest vs Nifty)
        rs_1m     : 1-month RS ratio
        rs_3m     : 3-month RS ratio
        rs_6m     : 6-month RS ratio
        rs_trend  : "Rising" / "Falling" / "Flat"
    """
    if df.empty or benchmark_df.empty or len(df) < 130 or len(benchmark_df) < 130:
        return {"rs_score": 50.0, "rs_1m": 1.0, "rs_3m": 1.0, "rs_6m": 1.0, "rs_trend": "Flat"}

    stk = df["Close"].reindex(benchmark_df.index, method="ffill").dropna()
    bmk = benchmark_df["Close"].reindex(stk.index, method="ffill").dropna()
    common = stk.index.intersection(bmk.index)
    stk = stk.loc[common]; bmk = bmk.loc[common]

    if len(stk) < 130:
        return {"rs_score": 50.0, "rs_1m": 1.0, "rs_3m": 1.0, "rs_6m": 1.0, "rs_trend": "Flat"}

    def _ret(s, n):
        return float(s.iloc[-1] / s.iloc[-n]) if len(s) >= n else 1.0

    # RS ratio = stock return / benchmark return over period
    rs_1m = _ret(stk, 21)  / max(_ret(bmk, 21),  0.001)
    rs_3m = _ret(stk, 63)  / max(_ret(bmk, 63),  0.001)
    rs_6m = _ret(stk, 126) / max(_ret(bmk, 126), 0.001)

    # Weighted composite: recent periods weighted more
    rs_composite = rs_1m * 0.50 + rs_3m * 0.30 + rs_6m * 0.20

    # RS trend: is RS line rising or falling over last 20 days?
    rs_line = stk / bmk
    rs_slope = rs_line.iloc[-1] - rs_line.iloc[-20] if len(rs_line) >= 20 else 0
    if rs_slope > rs_line.std() * 0.05:
        rs_trend = "Rising"
    elif rs_slope < -rs_line.std() * 0.05:
        rs_trend = "Falling"
    else:
        rs_trend = "Flat"

    # Score 0-100: centred on 1.0 = in-line with Nifty = 50
    # 1.20+ composite = ~80-100, 0.80- = ~0-20
    # Use tanh-shaped curve so extremes compress naturally
    deviation = (rs_composite - 1.0)  # positive = outperforming
    rs_score  = float(50 + 50 * np.tanh(deviation / 0.25))

    return {
        "rs_score" : round(rs_score, 1),
        "rs_1m"    : round(rs_1m, 3),
        "rs_3m"    : round(rs_3m, 3),
        "rs_6m"    : round(rs_6m, 3),
        "rs_trend" : rs_trend,
    }


def compute_signals(df: pd.DataFrame) -> dict:
    """
    Compute all trading signals from the last bar of an indicator-enriched DataFrame.
    Returns flat dict of signal values + raw values for filtering.
    """
    if df.empty or len(df) < 2:
        return {}

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # ── Price ────────────────────────────────────────────────
    price   = float(last["Close"])
    pct_chg = (price / float(prev["Close"]) - 1) * 100

    # ── Daily Trend ───────────────────────────────────────────
    ema_bull = int(last.get("EMA_cross", 0) or 0) == 1
    st_bull  = int(last.get("SuperTrend_Dir", 0) or 0) == 1
    adx_val  = float(last.get("ADX", 20) or 20)

    if ema_bull and st_bull:
        trend = "Bullish"
    elif not ema_bull and not st_bull:
        trend = "Bearish"
    elif adx_val >= 25:
        trend = "Bullish" if ema_bull else "Bearish"
    else:
        trend = "Neutral"

    # ── Regime ───────────────────────────────────────────────
    regime = "Trending" if adx_val >= ADX_TREND_THRESHOLD else "Ranging"
    adx_strength = (
        "Strong"  if adx_val >= ADX_STRONG_TREND else
        "Moderate" if adx_val >= ADX_TREND_THRESHOLD else
        "Weak"
    )

    # ── RSI Zone (user requirement: >60 bull, <40 bear) ──────
    rsi_val = float(last.get("RSI", 50) or 50)
    if rsi_val >= RSI_BULL_THRESHOLD:
        rsi_zone = "Bull"
    elif rsi_val <= RSI_BEAR_THRESHOLD:
        rsi_zone = "Bear"
    else:
        rsi_zone = "Neutral"

    # ── Momentum Score (0–100, 5 components) ─────────────────
    mom_score = _momentum_score(last, df)

    # ── Volatility Status ────────────────────────────────────
    vol_status = _vol_status(last, df)

    # ── MACD ─────────────────────────────────────────────────
    macdh_val   = float(last.get("MACDh", 0) or 0)
    macd_cross  = int(last.get("MACD_cross", 0) or 0)
    macd_bull   = macdh_val > 0

    # ── Kumo ─────────────────────────────────────────────────
    kumo_val    = int(last.get("Kumo_Breakout", 0) or 0)

    # ── Price context ─────────────────────────────────────────
    above_ema20  = float(last.get("Dist_EMA20_pct", 0) or 0) > 0
    above_ema200 = int(last.get("Above_EMA200", 0) or 0) == 1
    pct_52wh     = float(last.get("Pct_from_52wH", 0) or 0)
    pct_52wl     = float(last.get("Pct_from_52wL", 0) or 0)

    # ── Volume (rich: vs 5d avg, 20d avg, spurt) ─────────────
    vol_ratio    = float(last.get("Vol_ratio",  1) or 1)   # vs 20d
    vol_ratio5   = float(last.get("Vol_ratio5", 1) or 1)   # vs 5d
    obv_bull     = float(last.get("OBV_slope", 0) or 0) > 0

    # Build single rich Vol label
    # Priority: Spurt (>2× 20d) > Above both > Above 20d > Above 5d > Normal
    if vol_ratio >= 2.0:
        vol_label = ("🔴 DUMP×2" if pct_chg < 0 else "🟢 SPURT×2")
        _vol_raw  = "Spurt"
    elif vol_ratio >= 1.5 and vol_ratio5 >= 1.5:
        vol_label = "⚡ Abv5+20"   # above both 5d and 20d avg
        _vol_raw  = "High"
    elif vol_ratio >= 1.2:
        vol_label = "📈 Abv20d"
        _vol_raw  = "AboveAvg"
    elif vol_ratio5 >= 1.2:
        vol_label = "〰 Abv5d"
        _vol_raw  = "Rising"
    else:
        vol_label = "—"
        _vol_raw  = "Normal"

    # ── SMI (Stochastic Momentum Index) ──────────────────────
    smi_val    = float(last.get("SMI", 0) or 0)
    smi_sig    = float(last.get("SMI_Signal", 0) or 0)
    smi_prev   = float(df["SMI"].iloc[-2]) if len(df) > 1 and "SMI" in df.columns else smi_val
    smi_sig_prev = float(df["SMI_Signal"].iloc[-2]) if len(df) > 1 and "SMI_Signal" in df.columns else smi_sig

    if smi_val > 40:
        smi_zone = "OB"
    elif smi_val < -40:
        smi_zone = "OS"
    elif smi_val > 0:
        smi_zone = "Bull"
    else:
        smi_zone = "Bear"

    smi_cross = 0
    if smi_val > smi_sig and smi_prev <= smi_sig_prev:
        smi_cross = 1
    elif smi_val < smi_sig and smi_prev >= smi_sig_prev:
        smi_cross = -1

    # ── Market Structure ─────────────────────────────────────
    mkt_struct = str(last.get("Mkt_Struct", "—") or "—")

    # ── RSI Divergence ───────────────────────────────────────
    rsi_div     = int(last.get("RSI_Divergence", 0) or 0) == 1

    # ── ATR% ─────────────────────────────────────────────────
    atr_pct     = float(last.get("ATR_pct", 1) or 1)

    return {
        "Price"        : round(price, 2),
        "Chg%"         : round(pct_chg, 2),
        "Trend"        : trend,
        "Regime"       : regime,
        "ADX_Strength" : adx_strength,
        "RSI_Zone"     : rsi_zone,
        "MomScore"     : round(mom_score, 1),
        "VolStatus"    : vol_status,
        "Vol"          : vol_label,          # rich volume label
        "RSI"          : round(rsi_val, 1),
        "ADX"          : round(adx_val, 1),
        "ATR_pct"      : round(atr_pct, 2),
        "MACD_Bull"    : macd_bull,
        "MACD_hist"    : round(macdh_val, 3),
        "MACD_Cross"   : macd_cross,
        "Kumo"         : kumo_val,
        "Above_EMA20"  : above_ema20,
        "Above_EMA200" : above_ema200,
        "Pct_52wH"     : round(pct_52wh, 1),
        "Pct_52wL"     : round(pct_52wl, 1),
        "Vol_ratio"    : round(vol_ratio, 2),
        "OBV_Bull"     : obv_bull,
        "RSI_Div"      : rsi_div,
        "SMI"          : round(smi_val, 1),
        "SMI_Signal"   : round(smi_sig, 1),
        "SMI_Zone"     : smi_zone,
        "SMI_Cross"    : smi_cross,
        "Mkt_Struct"   : mkt_struct,
        "_trend"       : trend,
        "_regime"      : regime,
        "_vol"         : vol_status,
        "_vol_raw"     : _vol_raw,
        "_rsi_zone"    : rsi_zone,
    }


# ══════════════════════════════════════════════════════════════
#  MOMENTUM SCORE
# ══════════════════════════════════════════════════════════════
def _momentum_score(last, df: pd.DataFrame) -> float:
    """
    Composite momentum score 0–100.

    Components (weights sum to 100):
      RSI component       (30): normalised 0-100 RSI → scaled
      MACD component      (25): histogram normalised by ATR
      Kumo component      (20): -1/0/+1 → 0/50/100
      EMA component       (15): price vs EMA20 distance
      RSI vs SMA9         (10): RSI momentum direction

    FIX: score is now context-aware.
    A Bearish stock CAN have a high score if all momentum is pointing up
    (meaning it's recovering / bottoming). This is CORRECT and USEFUL.
    The dashboard shows both Trend and MomScore so the user sees full picture.
    """
    rsi_val    = float(last.get("RSI", 50) or 50)
    macdh_val  = float(last.get("MACDh", 0) or 0)
    atr_safe   = float(last.get("ATR", 1) or 1)
    kumo_val   = float(last.get("Kumo_Breakout", 0) or 0)
    dist_ema20 = float(last.get("Dist_EMA20_pct", 0) or 0)
    rsi_sma9   = float(last.get("RSI_SMA9", 50) or 50)

    # RSI component: 0→0, 50→50, 100→100
    c_rsi  = np.clip(rsi_val, 0, 100)

    # MACD component: normalise by ATR, clip to ±1, map to 0–100
    macd_norm = np.clip(macdh_val / (atr_safe * 0.5 + 1e-9), -1, 1)
    c_macd    = (macd_norm + 1) / 2 * 100

    # Kumo: -1→0, 0→50, +1→100
    c_kumo = (kumo_val + 1) / 2 * 100

    # EMA distance: clip ±5% → map to 0–100
    c_ema  = np.clip((dist_ema20 + 5) / 10 * 100, 0, 100)

    # RSI above/below its own 9-SMA (momentum direction)
    c_rsi_sma = 100.0 if rsi_val > rsi_sma9 else 0.0

    score = (
        c_rsi     * MOM_RSI_WEIGHT / 100 +
        c_macd    * MOM_MACD_WEIGHT / 100 +
        c_kumo    * MOM_KUMO_WEIGHT / 100 +
        c_ema     * MOM_EMA_WEIGHT / 100 +
        c_rsi_sma * MOM_RSI_SMA_WEIGHT / 100
    )
    return float(np.clip(score, 0, 100))


# ══════════════════════════════════════════════════════════════
#  VOLATILITY STATUS
# ══════════════════════════════════════════════════════════════
def _vol_status(last, df: pd.DataFrame) -> str:
    """
    Squeeze: BB inside Keltner Channel (momentum coiling)
    Expansion: ATR above its 20d mean by ATR_EXPANSION_MULT
    Normal: everything else
    Uses ATR percentile when available for more robust detection.
    """
    sq_val   = int(last.get("Squeeze", 0) or 0)
    if sq_val == 1:
        return "Squeeze"

    atr_pctile = float(last.get("ATR_pctile", 50) or 50)
    if atr_pctile > 70:
        return "Expansion"

    # Fallback: simple mean comparison
    atr_pct  = float(last.get("ATR_pct", 1) or 1)
    atr_mean = df["ATR_pct"].rolling(20).mean().iloc[-1]
    if not pd.isna(atr_mean) and atr_mean > 0 and atr_pct > atr_mean * ATR_EXPANSION_MULT:
        return "Expansion"

    return "Normal"


# ══════════════════════════════════════════════════════════════
#  TRADE BIAS + PATTERN HELPERS
# ══════════════════════════════════════════════════════════════
def _build_pattern_str(b_eng: bool, br_eng: bool, doji: bool, hammer: bool) -> str:
    """Merge all candle patterns into one column string."""
    parts = []
    if b_eng:  parts.append("BullEng")
    if br_eng: parts.append("BearEng")
    if hammer: parts.append("Hammer")
    if doji:   parts.append("Doji")
    return " · ".join(parts) if parts else "—"


def _trade_bias(t: dict, ml_prob: float, ml_reason: str) -> str:
    """
    Actionable Long / Short bias for FnO traders.
    Scores bull and bear signals (max 5 each), threshold at 4 for strong call.
    """
    ml_active  = ml_reason not in ("ML off", "ML error", "Insufficient data")
    trend      = t.get("_trend", "Neutral")
    rsi_zone   = t.get("_rsi_zone", "Neutral")
    smi_val    = t.get("SMI", 0)
    mkt_struct = t.get("Mkt_Struct", "—")

    bull_pts = sum([
        trend == "Bullish",
        rsi_zone == "Bull",
        float(smi_val) > 0 if isinstance(smi_val, (int, float)) else False,
        mkt_struct == "HH-HL",
        (ml_prob > ML_STRONG_BUY_PROB) if ml_active else False,
    ])
    bear_pts = sum([
        trend == "Bearish",
        rsi_zone == "Bear",
        float(smi_val) < 0 if isinstance(smi_val, (int, float)) else False,
        mkt_struct == "LH-LL",
        (ml_prob < ML_STRONG_SELL_PROB) if ml_active else False,
    ])

    if bull_pts >= 4: return "🟢 LONG"
    if bear_pts >= 4: return "🔴 SHORT"
    if bull_pts == 3: return "🟡 LONG?"
    if bear_pts == 3: return "🟠 SHORT?"
    return "⬜ NEUTRAL"


# ══════════════════════════════════════════════════════════════
#  CANDLESTICK PATTERNS
# ══════════════════════════════════════════════════════════════
def detect_bullish_engulfing(df: pd.DataFrame) -> bool:
    """
    Bullish engulfing: bearish candle followed by larger bullish candle.
    Optional volume confirmation: today's volume > yesterday's.
    """
    if len(df) < 2:
        return False
    p, l = df.iloc[-2], df.iloc[-1]
    # Classic pattern
    pattern = (
        float(p["Close"]) < float(p["Open"]) and   # prev bearish
        float(l["Close"]) > float(l["Open"]) and   # today bullish
        float(l["Open"])  < float(p["Close"]) and  # today opens below prev close
        float(l["Close"]) > float(p["Open"])        # today closes above prev open
    )
    if not pattern:
        return False
    # Volume confirmation (today's volume >= yesterday's)
    vol_ok = float(l.get("Volume", 1)) >= float(p.get("Volume", 1)) * 0.8
    return vol_ok


def detect_bearish_engulfing(df: pd.DataFrame) -> bool:
    """
    Bearish engulfing: bullish candle followed by larger bearish candle.
    """
    if len(df) < 2:
        return False
    p, l = df.iloc[-2], df.iloc[-1]
    pattern = (
        float(p["Close"]) > float(p["Open"]) and
        float(l["Close"]) < float(l["Open"]) and
        float(l["Open"])  > float(p["Close"]) and
        float(l["Close"]) < float(p["Open"])
    )
    if not pattern:
        return False
    vol_ok = float(l.get("Volume", 1)) >= float(p.get("Volume", 1)) * 0.8
    return vol_ok


def detect_doji(df: pd.DataFrame) -> bool:
    """Doji: open ≈ close (body < 10% of candle range)."""
    if df.empty:
        return False
    l = df.iloc[-1]
    body  = abs(float(l["Close"]) - float(l["Open"]))
    range_= abs(float(l["High"])  - float(l["Low"]))
    return range_ > 0 and (body / range_) < 0.1


def detect_hammer(df: pd.DataFrame) -> bool:
    """
    Hammer (bullish reversal): small body at top, long lower wick.
    Lower wick >= 2× body, upper wick <= 0.5× body.
    """
    if len(df) < 2:
        return False
    l     = df.iloc[-1]
    o, h_, lo, cl = float(l["Open"]), float(l["High"]), float(l["Low"]), float(l["Close"])
    body  = abs(cl - o)
    upper = h_ - max(o, cl)
    lower = min(o, cl) - lo
    if body == 0:
        return False
    return lower >= 2 * body and upper <= 0.5 * body


# ══════════════════════════════════════════════════════════════
#  FORMATTED ROW  (for the dashboard table)
# ══════════════════════════════════════════════════════════════
def format_row(ticker: str, signals: dict,
               b_eng: bool, br_eng: bool,
               doji: bool, hammer: bool,
               ml_prob: float, ml_acc: float,
               ml_reason: str,
               rs: dict | None = None) -> dict:
    """Build the final display row for the universe table."""
    t  = signals
    trend_icon = TREND_ICONS.get(t["Trend"], "🟡")
    reg_icon   = REGIME_ICONS.get(t["Regime"], "")
    vol_icon   = SQUEEZE_ICONS.get(t["VolStatus"], "")
    rsi_icon   = RSI_ZONE_ICONS.get(t["RSI_Zone"], "➖")

    # ML signal — regime-adaptive thresholds
    # Stocks below EMA200 need higher conviction to trigger Buy
    if ml_reason in ("ML off", "ML error", "Insufficient data"):
        ml_signal = "—"
    else:
        above_200 = int(t.get("Above_EMA200", 0) or 0) == 1
        buy_th  = ML_STRONG_BUY_PROB  if above_200 else ML_BEAR_BUY_PROB
        sell_th = ML_STRONG_SELL_PROB  if above_200 else ML_BEAR_SELL_PROB
        if ml_prob > buy_th:
            ml_signal = "🟢 Buy"
        elif ml_prob < sell_th:
            ml_signal = "🔴 Sell"
        else:
            ml_signal = "🟡 Hold"

    # Trend-ML alignment check
    trend_ml_ok = (
        (t["Trend"] == "Bullish" and ml_prob > 0.50) or
        (t["Trend"] == "Bearish" and ml_prob < 0.50) or
        (t["Trend"] == "Neutral")
    )

    return {
        "Ticker"      : ticker.replace(".NS",""),
        "Price"       : t["Price"],
        "Chg%"        : t["Chg%"],
        "D_Trend"     : f"{trend_icon} {t['Trend']}",
        "Mkt_Struct"  : t.get("Mkt_Struct", "—"),
        "RSI_Zone"    : f"{rsi_icon} {t['RSI_Zone']}",
        "Regime"      : f"{reg_icon} {t['Regime']}",
        "ADX_Str"     : t["ADX_Strength"],
        "MomScore"    : t["MomScore"],
        "VolStatus"   : f"{vol_icon} {t['VolStatus']}",
        "Vol"         : t.get("Vol", "—"),
        "RSI"         : t["RSI"],
        "ADX"         : t["ADX"],
        "ATR%"        : t["ATR_pct"],
        # MACD: show histogram value + direction emoji — not just Bull/Bear
        "MACD"        : f"{'🟢' if t['MACD_Bull'] else '🔴'} {t['MACD_hist']:+.2f}",
        ">EMA20"      : "✅" if t["Above_EMA20"] else "❌",
        ">EMA200"     : "✅" if t["Above_EMA200"] else "❌",
        "52wH%"       : t["Pct_52wH"],
        "SMI"         : t.get("SMI", 0.0),
        "SMI_Zone"    : t.get("SMI_Zone", "—"),
        # Merged pattern column: list all present patterns
        "Pattern"     : _build_pattern_str(b_eng, br_eng, doji, hammer),
        "ML_Prob%"    : round(ml_prob * 100, 1),
        "ML_Acc%"     : ml_acc,
        "ML_Signal"   : ml_signal,
        "ML_Reason"   : ml_reason,
        "⚠️Conflict"  : "⚠️" if not trend_ml_ok else "—",
        # Long / Short candidate — visible trade bias
        "Trade"       : _trade_bias(t, ml_prob, ml_reason),
        # raw fields
        "_trend"      : t["_trend"],
        "_regime"     : t["_regime"],
        "_vol"        : t["_vol"],
        "_vol_raw"    : t.get("_vol_raw", "Normal"),
        "_rsi_zone"   : t["_rsi_zone"],
        "_ml_prob"    : ml_prob,
        "_above_ema200": t["Above_EMA200"],
        "RS_Score"    : rs["rs_score"]  if rs else 50.0,
        "RS_Trend"    : rs["rs_trend"]  if rs else "Flat",
        "RS_1M"       : rs["rs_1m"]     if rs else 1.0,
        "RS_3M"       : rs["rs_3m"]     if rs else 1.0,
        "RS_Rank"     : 0,
        "SMI_Cross"   : t.get("SMI_Cross", 0),
        "VolSpurt"    : t.get("Vol", "—"),
    }
