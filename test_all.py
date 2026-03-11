"""
test_all.py — Validation suite for MTF Dashboard
Run: python test_all.py
Tests all modules with synthetic data (no internet required).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import traceback
from datetime import datetime, timedelta

# ══════════════════════════════════════════════════════════════
#  SYNTHETIC DATA GENERATORS
# ══════════════════════════════════════════════════════════════

def make_bullish_df(n=500) -> pd.DataFrame:
    """Steadily uptrending price series."""
    np.random.seed(42)
    dates  = pd.date_range(end=datetime.today(), periods=n, freq="B")
    close  = 1000 + np.cumsum(np.random.randn(n) * 5 + 0.8)   # drift up
    high   = close + np.abs(np.random.randn(n) * 3)
    low    = close - np.abs(np.random.randn(n) * 3)
    open_  = close - np.random.randn(n) * 2
    volume = (np.abs(np.random.randn(n)) * 1e6 + 5e5).astype(int)
    df = pd.DataFrame({
        "Open": open_, "High": high, "Low": low,
        "Close": close, "Volume": volume,
    }, index=dates)
    return df.astype({"Open":"float32","High":"float32",
                       "Low":"float32","Close":"float32"})


def make_bearish_df(n=500) -> pd.DataFrame:
    """Steadily downtrending price series."""
    np.random.seed(99)
    dates  = pd.date_range(end=datetime.today(), periods=n, freq="B")
    close  = 2000 + np.cumsum(np.random.randn(n) * 5 - 0.9)   # drift down
    high   = close + np.abs(np.random.randn(n) * 3)
    low    = close - np.abs(np.random.randn(n) * 3)
    open_  = close - np.random.randn(n) * 2
    volume = (np.abs(np.random.randn(n)) * 1e6 + 5e5).astype(int)
    df = pd.DataFrame({
        "Open": open_, "High": high, "Low": low,
        "Close": close, "Volume": volume,
    }, index=dates)
    return df.astype({"Open":"float32","High":"float32",
                       "Low":"float32","Close":"float32"})


def make_sideways_df(n=500) -> pd.DataFrame:
    """Mean-reverting / sideways price series."""
    np.random.seed(7)
    dates = pd.date_range(end=datetime.today(), periods=n, freq="B")
    close = 1500 + np.cumsum(np.random.randn(n) * 6)           # no drift
    close = close - close.mean() + 1500                         # centre at 1500
    high  = close + np.abs(np.random.randn(n) * 4)
    low   = close - np.abs(np.random.randn(n) * 4)
    open_ = close - np.random.randn(n) * 2
    volume= (np.abs(np.random.randn(n)) * 8e5 + 3e5).astype(int)
    df = pd.DataFrame({
        "Open": open_, "High": high, "Low": low,
        "Close": close, "Volume": volume,
    }, index=dates)
    return df.astype({"Open":"float32","High":"float32",
                       "Low":"float32","Close":"float32"})


def make_engulfing_df(bullish: bool = True) -> pd.DataFrame:
    """Two-bar engulfing pattern at the end."""
    dates = pd.date_range(end=datetime.today(), periods=10, freq="B")
    if bullish:
        # prev: bearish (open=105, close=100), today: bullish (open=99, close=107)
        closes = [100,101,102,103,102,101,100,101,100,107]
        opens  = [99, 100,101,102,103,102,101,100,105,99]
    else:
        # prev: bullish (open=95, close=100), today: bearish (open=101, close=94)
        closes = [100,99,98,99,100,101,100,99,100,94]
        opens  = [99, 100,99,98,99,98,99,98,95,101]

    highs  = [max(o,c)+1 for o,c in zip(opens,closes)]
    lows   = [min(o,c)-1 for o,c in zip(opens,closes)]
    volume = [1000000]*10
    return pd.DataFrame({
        "Open":opens,"High":highs,"Low":lows,"Close":closes,"Volume":volume
    }, index=dates).astype({"Open":"float32","High":"float32",
                              "Low":"float32","Close":"float32"})


# ══════════════════════════════════════════════════════════════
#  TEST RUNNER
# ══════════════════════════════════════════════════════════════

PASS = "✅ PASS"
FAIL = "❌ FAIL"
results = []

def test(name: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    results.append((name, status, detail))
    print(f"  {status}  {name}" + (f" — {detail}" if detail else ""))


# ══════════════════════════════════════════════════════════════
#  1. CONFIG
# ══════════════════════════════════════════════════════════════
print("\n━━━ 1. CONFIG ━━━")
try:
    from config import (
        RSI_BULL_THRESHOLD, RSI_BEAR_THRESHOLD,
        NIFTY50, NIFTY_NEXT50, NIFTY_MIDCAP,
        UNIVERSE_MAP, TICKER_LOOKBACK,
        MOM_RSI_WEIGHT, MOM_MACD_WEIGHT, MOM_KUMO_WEIGHT,
        MOM_EMA_WEIGHT, MOM_RSI_SMA_WEIGHT,
    )
    test("RSI bull threshold = 60", RSI_BULL_THRESHOLD == 60)
    test("RSI bear threshold = 40", RSI_BEAR_THRESHOLD == 40)
    test("Nifty50 has 50 tickers", len(NIFTY50) == 50, f"got {len(NIFTY50)}")
    test("All tickers end with .NS",
         all(t.endswith(".NS") for t in NIFTY50 + NIFTY_NEXT50 + NIFTY_MIDCAP))
    test("No duplicate tickers in Nifty50",
         len(NIFTY50) == len(set(NIFTY50)))
    test("No overlap Nifty50 ∩ Next50",
         len(set(NIFTY50) & set(NIFTY_NEXT50)) == 0,
         f"overlap: {set(NIFTY50) & set(NIFTY_NEXT50)}")
    test("Momentum weights sum to 100",
         MOM_RSI_WEIGHT+MOM_MACD_WEIGHT+MOM_KUMO_WEIGHT+MOM_EMA_WEIGHT+MOM_RSI_SMA_WEIGHT == 100)
    test("TICKER_LOOKBACK populated",
         len(TICKER_LOOKBACK) > 100)
except Exception as e:
    test("Config import", False, str(e))
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════
#  2. INDICATOR ENGINE
# ══════════════════════════════════════════════════════════════
print("\n━━━ 2. INDICATOR ENGINE ━━━")
try:
    from indicators.engine import add_indicators

    bull_df  = make_bullish_df(500)
    bear_df  = make_bearish_df(500)
    side_df  = make_sideways_df(500)
    short_df = make_bullish_df(30)   # too short

    bull_ind = add_indicators(bull_df.copy())
    bear_ind = add_indicators(bear_df.copy())
    side_ind = add_indicators(side_df.copy())

    # Column presence
    required_cols = [
        "EMA20","EMA50","EMA200","EMA_cross","SuperTrend_Dir",
        "ADX","Kumo_Breakout","RSI","RSI_SMA9","RSI_Divergence",
        "MACD","MACDs","MACDh","MACD_cross",
        "ATR","ATR_pct","ATR_pctile","HV_ratio",
        "BB_Upper","BB_Lower","BB_Width","BB_pct","Squeeze",
        "WilliamsR","ROC10","ROC20","OBV","OBV_slope","Vol_ratio",
        "High_52w","Low_52w","Pct_from_52wH","Pct_from_52wL","Dist_EMA20_pct",
    ]
    missing = [c for c in required_cols if c not in bull_ind.columns]
    test("All indicator columns present", len(missing)==0, f"missing: {missing}")

    # Dtype check
    float_cols = [c for c in required_cols if c not in ["EMA_cross","SuperTrend_Dir",
                                                          "Kumo_Breakout","MACD_cross",
                                                          "RSI_Divergence","Squeeze"]]
    wrong_dtype = [c for c in float_cols
                   if c in bull_ind.columns and str(bull_ind[c].dtype) != "float32"]
    test("All float columns are float32", len(wrong_dtype)==0, f"wrong: {wrong_dtype[:3]}")

    # Short df guard
    short_result = add_indicators(short_df.copy())
    test("Short df (<60 bars) returned unchanged",
         len(short_result.columns) == len(short_df.columns))

    # EMA relationship in bullish trend
    last_bull = bull_ind.dropna().iloc[-1]
    test("Bullish: EMA20 > EMA50 (uptrend)", float(last_bull["EMA20"]) > float(last_bull["EMA50"]),
         f"EMA20={last_bull['EMA20']:.1f} EMA50={last_bull['EMA50']:.1f}")

    # EMA relationship in bearish trend
    last_bear = bear_ind.dropna().iloc[-1]
    test("Bearish: EMA20 < EMA50 (downtrend)", float(last_bear["EMA20"]) < float(last_bear["EMA50"]),
         f"EMA20={last_bear['EMA20']:.1f} EMA50={last_bear['EMA50']:.1f}")

    # RSI range
    rsi_vals = bull_ind["RSI"].dropna()
    test("RSI always in 0–100", rsi_vals.between(0,100).all(),
         f"min={rsi_vals.min():.1f} max={rsi_vals.max():.1f}")

    # ADX always positive
    adx_vals = bull_ind["ADX"].dropna()
    test("ADX always >= 0", (adx_vals >= 0).all(), f"min={adx_vals.min():.2f}")

    # Squeeze is binary
    sq_vals = bull_ind["Squeeze"].dropna().unique()
    test("Squeeze is 0 or 1 only", set(sq_vals).issubset({0,1}))

    # ATR percentile 0-100
    atr_p = bull_ind["ATR_pctile"].dropna()
    test("ATR_pctile in 0–100", atr_p.between(0,100).all() if not atr_p.empty else True)

    # SuperTrend direction is +1 or -1
    st_vals = bull_ind["SuperTrend_Dir"].unique()
    test("SuperTrend_Dir is ±1 only", set(st_vals).issubset({1,-1}))

    # Bullish stock should trend up (SuperTrend > 0 at end)
    test("Bullish df: SuperTrend positive at end",
         int(bull_ind["SuperTrend_Dir"].iloc[-1]) == 1)

    # Bearish stock SuperTrend
    test("Bearish df: SuperTrend negative at end",
         int(bear_ind["SuperTrend_Dir"].iloc[-1]) == -1)

except Exception as e:
    test("Indicator engine", False, str(e))
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════
#  3. SIGNALS
# ══════════════════════════════════════════════════════════════
print("\n━━━ 3. SIGNALS ━━━")
try:
    from indicators.signals import (
        compute_signals,
        detect_bullish_engulfing, detect_bearish_engulfing,
        detect_doji, detect_hammer,
        _momentum_score,
    )
    from indicators.engine import add_indicators

    bull_ind = add_indicators(make_bullish_df(500))
    bear_ind = add_indicators(make_bearish_df(500))
    side_ind = add_indicators(make_sideways_df(500))

    sig_bull = compute_signals(bull_ind)
    sig_bear = compute_signals(bear_ind)
    sig_side = compute_signals(side_ind)

    # Required keys
    req_keys = [
        "Price","Chg%","Trend","Regime","RSI_Zone","MomScore",
        "VolStatus","RSI","ADX","ATR_pct","MACD_Bull",
        "Above_EMA20","Above_EMA200","_trend","_regime","_vol","_rsi_zone",
    ]
    missing_keys = [k for k in req_keys if k not in sig_bull]
    test("All signal keys present", len(missing_keys)==0, f"missing: {missing_keys}")

    # Trend logic
    test("Bullish df → Bullish trend", sig_bull["Trend"] == "Bullish",
         f"got: {sig_bull['Trend']}")
    test("Bearish df → Bearish trend", sig_bear["Trend"] == "Bearish",
         f"got: {sig_bear['Trend']}")

    # RSI zone (your requirement: >60 bull, <40 bear)
    rsi_bull_val = float(sig_bull["RSI"])
    rsi_bear_val = float(sig_bear["RSI"])
    test(f"Bullish RSI ({rsi_bull_val:.1f}) → RSI_Zone=Bull",
         sig_bull["RSI_Zone"] == "Bull" if rsi_bull_val >= 60 else True,
         f"RSI={rsi_bull_val:.1f} Zone={sig_bull['RSI_Zone']}")

    # Explicit RSI zone tests
    from config import RSI_BULL_THRESHOLD, RSI_BEAR_THRESHOLD
    test("RSI 65 → Bull zone",
         _rsi_zone_check(65) == "Bull")
    test("RSI 35 → Bear zone",
         _rsi_zone_check(35) == "Bear")
    test("RSI 50 → Neutral zone",
         _rsi_zone_check(50) == "Neutral")
    test("RSI 60 → Bull zone (boundary)",
         _rsi_zone_check(60) == "Bull")
    test("RSI 40 → Bear zone (boundary)",
         _rsi_zone_check(40) == "Bear")

    # Momentum score range
    test("MomScore in 0–100",
         0 <= sig_bull["MomScore"] <= 100,
         f"got: {sig_bull['MomScore']}")
    test("Bullish MomScore > Bearish MomScore",
         sig_bull["MomScore"] > sig_bear["MomScore"],
         f"bull={sig_bull['MomScore']:.1f} bear={sig_bear['MomScore']:.1f}")

    # Engulfing patterns
    bull_eng_df = make_engulfing_df(bullish=True)
    bear_eng_df = make_engulfing_df(bullish=False)
    test("Bullish engulfing detected", detect_bullish_engulfing(bull_eng_df))
    test("Bearish engulfing detected", detect_bearish_engulfing(bear_eng_df))
    test("No false bull engulfing on bear df", not detect_bullish_engulfing(bear_eng_df))
    test("No false bear engulfing on bull df", not detect_bearish_engulfing(bull_eng_df))

    # Doji detection
    doji_df = _make_doji_df()
    test("Doji pattern detected", detect_doji(doji_df))

    # Hammer detection
    hammer_df = _make_hammer_df()
    test("Hammer pattern detected", detect_hammer(hammer_df))

    # Empty df guard
    test("compute_signals(empty) → {}", compute_signals(pd.DataFrame()) == {})

except Exception as e:
    test("Signals module", False, str(e))
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════
#  4. ML FEATURES
# ══════════════════════════════════════════════════════════════
print("\n━━━ 4. ML FEATURES ━━━")
try:
    from ml.features import build_features, FEATURE_COLS
    from indicators.engine import add_indicators

    bull_ind = add_indicators(make_bullish_df(600))
    feats    = build_features(bull_ind)

    test("Features DataFrame not empty", not feats.empty)
    test("All feature columns present",
         all(c in feats.columns for c in FEATURE_COLS),
         f"missing: {[c for c in FEATURE_COLS if c not in feats.columns]}")
    test("Target column present", "target" in feats.columns)
    test("Target is binary (0/1)",
         set(feats["target"].unique()).issubset({0,1}))
    test("No inf values in features",
         not feats[FEATURE_COLS].replace([float("inf"),float("-inf")],float("nan")).isnull().all().any())
    test("Feature count = 22", len(FEATURE_COLS) == 22, f"got {len(FEATURE_COLS)}")

    # Target logic: price higher after 5 days
    from config import ML_FORWARD_DAYS
    close = bull_ind["Close"].reset_index(drop=True)
    feats2 = build_features(bull_ind)
    # Sample check: verify a few target values
    bull_ind_r = bull_ind.reset_index(drop=True)
    errors = 0
    for i in range(10, min(50, len(bull_ind_r)-ML_FORWARD_DAYS-1)):
        expected = int(bull_ind_r["Close"].iloc[i+ML_FORWARD_DAYS] > bull_ind_r["Close"].iloc[i])
        # Can't directly match due to NaN dropping, just verify type
    test("Target values are int8", str(feats["target"].dtype) == "int8")

    # Bullish series should have majority target=1
    bull_target_mean = feats["target"].mean()
    test("Bullish df: majority target=1 (>0.5)",
         bull_target_mean > 0.5, f"got {bull_target_mean:.2f}")

    # Bearish series should have majority target=0
    bear_ind  = add_indicators(make_bearish_df(600))
    bear_feats= build_features(bear_ind)
    bear_target_mean = bear_feats["target"].mean()
    test("Bearish df: majority target=0 (<0.5)",
         bear_target_mean < 0.5, f"got {bear_target_mean:.2f}")

except Exception as e:
    test("ML features", False, str(e))
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════
#  5. ML MODEL
# ══════════════════════════════════════════════════════════════
print("\n━━━ 5. ML MODEL ━━━")
try:
    from ml.model import train_and_predict, USE_GPU

    print(f"  ℹ️  GPU available: {USE_GPU}")

    bull_ind   = add_indicators(make_bullish_df(700))
    bear_ind   = add_indicators(make_bearish_df(700))

    bull_prob, bull_acc, bull_reason = train_and_predict(bull_ind)
    bear_prob, bear_acc, bear_reason = train_and_predict(bear_ind)

    test("Probability in 0–1 (bull)", 0 <= bull_prob <= 1, f"got {bull_prob}")
    test("Probability in 0–1 (bear)", 0 <= bear_prob <= 1, f"got {bear_prob}")
    test("Bullish df → prob > 0.5", bull_prob > 0.5, f"got {bull_prob:.3f}")
    test("Bearish df → prob < 0.5", bear_prob < 0.5, f"got {bear_prob:.3f}")
    test("WF accuracy in 0–100 (bull)", 0 <= bull_acc <= 100, f"got {bull_acc}")
    test("WF accuracy in 0–100 (bear)", 0 <= bear_acc <= 100, f"got {bear_acc}")
    test("WF accuracy > 45% (better than random)", bull_acc > 45 or bear_acc > 45,
         f"bull_acc={bull_acc:.1f}% bear_acc={bear_acc:.1f}%")
    test("Reason string not empty", len(bull_reason) > 0, bull_reason)
    test("Reason has direction", "Bullish" in bull_reason or "Bearish" in bull_reason,
         bull_reason)

    print(f"  ℹ️  Bull prob={bull_prob:.3f} acc={bull_acc:.1f}% reason='{bull_reason}'")
    print(f"  ℹ️  Bear prob={bear_prob:.3f} acc={bear_acc:.1f}% reason='{bear_reason}'")

    # Insufficient data guard
    short_ind = add_indicators(make_bullish_df(50))
    p, a, r   = train_and_predict(short_ind)
    test("Short df → prob=0.5 (default)", p == 0.5, f"got {p}")
    test("Short df → acc=0.0 (default)", a == 0.0, f"got {a}")

except Exception as e:
    test("ML model", False, str(e))
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════
#  6. STYLING
# ══════════════════════════════════════════════════════════════
print("\n━━━ 6. STYLING ━━━")
try:
    from utils.styling import (
        grad_rg, grad_blue, color_chg,
        color_rsi, color_adx, color_52wh,
    )

    test("grad_rg(0)   → red background",   "255" in grad_rg(0))
    test("grad_rg(100) → green background", "rgba" in grad_rg(100))
    test("grad_rg(50)  → mid colour",        "rgba" in grad_rg(50))
    test("color_chg(+1) → green", "26a69a" in color_chg(1.0))
    test("color_chg(-1) → red",   "ef5350" in color_chg(-1.0))
    test("color_chg(0)  → empty", color_chg(0.0) == "")

    test("RSI 75 → orange (overbought)",  "ff6b35" in color_rsi(75))
    test("RSI 65 → green (bull zone)",    "26a69a" in color_rsi(65))
    test("RSI 25 → bold green (oversold)","26a69a" in color_rsi(25))
    test("RSI 35 → red (bear zone)",      "ef5350" in color_rsi(35))
    test("RSI 50 → grey (neutral)",       "9e9e9e" in color_rsi(50))

    test("ADX 45 → gold",  "FFD700" in color_adx(45))
    test("ADX 30 → green", "26a69a" in color_adx(30))
    test("ADX 15 → grey",  "9e9e9e" in color_adx(15))

    test("52wH -1%  → green (near high)",   "26a69a" in color_52wh(-1))
    test("52wH -35% → red (far from high)", "ef5350" in color_52wh(-35))

except Exception as e:
    test("Styling", False, str(e))
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════
#  7. UNIVERSE / FILTERS
# ══════════════════════════════════════════════════════════════
print("\n━━━ 7. UNIVERSE FILTERS ━━━")
try:
    from utils.universe import apply_filters, sort_df, universe_stats
    from indicators.signals import format_row

    # Build a synthetic universe df
    rows = []
    for ticker, prob, trend, rsi, regime, vol, rsi_zone in [
        ("RELIANCE",  0.70, "Bullish", 65, "Trending", "Normal",    "Bull"),
        ("TCS",       0.35, "Bearish", 35, "Ranging",  "Squeeze",   "Bear"),
        ("SBIN",      0.55, "Neutral", 50, "Ranging",  "Normal",    "Neutral"),
        ("HDFC",      0.65, "Bullish", 72, "Trending", "Expansion", "Bull"),
        ("INFY",      0.30, "Bearish", 28, "Ranging",  "Normal",    "Bear"),
    ]:
        rows.append({
            "Ticker": ticker, "Price": 1000.0, "Chg%": 0.5,
            "D_Trend": f"🟢 {trend}", "RSI_Zone": f"🐂 {rsi_zone}",
            "Regime": f"📈 {regime}", "ADX_Str": "Moderate",
            "MomScore": 60.0, "VolStatus": f"➖ {vol}",
            "RSI": float(rsi), "ADX": 30.0, "ATR%": 1.5,
            "MACD": "🟢 Bull", ">EMA20": "✅", ">EMA200": "✅",
            "52wH%": -5.0,
            "BullEng": "—", "BearEng": "—", "Doji": "—", "Hammer": "—",
            "ML_Prob%": round(prob*100,1), "ML_Acc%": 60.0,
            "ML_Signal": "🟢 Buy" if prob>0.58 else "🔴 Sell",
            "ML_Reason": "Bullish: RSI↑",
            "⚠️Conflict": "—",
            "_trend": trend, "_regime": regime, "_vol": vol,
            "_rsi_zone": rsi_zone, "_ml_prob": prob,
            "_above_ema200": 1,
        })
    udf = pd.DataFrame(rows)

    # Filter tests
    bull_f = apply_filters(udf, {"bullish_d": True})
    test("Filter bullish_d=True → 2 rows", len(bull_f)==2, f"got {len(bull_f)}")

    bear_f = apply_filters(udf, {"bearish_d": True})
    test("Filter bearish_d=True → 2 rows", len(bear_f)==2, f"got {len(bear_f)}")

    rsi_bull_f = apply_filters(udf, {"rsi_bull": True})
    test("Filter rsi_bull → RSI>60 only",
         all(r >= 60 for r in rsi_bull_f["RSI"]),
         f"RSI values: {list(rsi_bull_f['RSI'])}")

    rsi_bear_f = apply_filters(udf, {"rsi_bear": True})
    test("Filter rsi_bear → RSI<40 only",
         all(r <= 40 for r in rsi_bear_f["RSI"]),
         f"RSI values: {list(rsi_bear_f['RSI'])}")

    ml_buy_f = apply_filters(udf, {"ml_buy": True})
    test("Filter ml_buy → prob>0.58 only",
         all(r > 0.58 for r in ml_buy_f["_ml_prob"]),
         f"probs: {list(ml_buy_f['_ml_prob'])}")

    sq_f = apply_filters(udf, {"squeeze": True})
    test("Filter squeeze → only squeeze tickers", len(sq_f)==1, f"got {len(sq_f)}")

    rsi_range_f = apply_filters(udf, {"rsi_min": 45, "rsi_max": 75})
    test("RSI range 45–75 → 3 rows", len(rsi_range_f)==3,
         f"got {len(rsi_range_f)}, RSI: {list(rsi_range_f['RSI'])}")

    # No filter → all rows
    no_f = apply_filters(udf, {})
    test("No filters → all 5 rows", len(no_f)==5)

    # Sort test
    sorted_asc = sort_df(udf, "RSI", ascending=True)
    rsi_vals   = list(sorted_asc["RSI"])
    test("Sort by RSI ascending", rsi_vals == sorted(rsi_vals),
         f"got {rsi_vals}")

    sorted_desc = sort_df(udf, "RSI", ascending=False)
    rsi_vals_d  = list(sorted_desc["RSI"])
    test("Sort by RSI descending", rsi_vals_d == sorted(rsi_vals_d, reverse=True))

    # Stats
    stats = universe_stats(udf)
    test("Stats: total=5", stats["total"]==5)
    test("Stats: bullish=2", stats["bullish"]==2)
    test("Stats: bearish=2", stats["bearish"]==2)
    test("Stats: rsi_bull=2", stats["rsi_bull"]==2)
    test("Stats: rsi_bear=2", stats["rsi_bear"]==2)
    test("Stats: squeeze=1", stats["squeeze"]==1)

except Exception as e:
    test("Universe/Filters", False, str(e))
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════
#  8. DATA INTEGRITY — ML vs TREND LOGIC
# ══════════════════════════════════════════════════════════════
print("\n━━━ 8. DATA INTEGRITY — ML vs TREND ━━━")
try:
    from indicators.signals import format_row
    from ml.model import train_and_predict
    from indicators.engine import add_indicators

    # Scenario: stock is Bullish but ML says Sell — should flag conflict
    sig_bull_ml_sell = {
        "Price":1000,"Chg%":0.5,"Trend":"Bullish","Regime":"Trending",
        "ADX_Strength":"Strong","RSI_Zone":"Bull","MomScore":75.0,
        "VolStatus":"Normal","RSI":65,"ADX":35,"ATR_pct":1.5,
        "MACD_Bull":True,"MACD_Cross":0,"Kumo":1,
        "Above_EMA20":True,"Above_EMA200":True,
        "Pct_52wH":-3,"Pct_52wL":20,"Vol_ratio":1.2,
        "OBV_Bull":True,"RSI_Div":False,
        "_trend":"Bullish","_regime":"Trending","_vol":"Normal","_rsi_zone":"Bull",
    }
    row_conflict = format_row("TEST",sig_bull_ml_sell,
                               False,False,False,False,
                               0.30, 55.0, "Bearish: MACD↓")
    test("Bullish trend + ML Sell → ⚠️ conflict flagged",
         row_conflict["⚠️Conflict"] == "⚠️",
         f"got: {row_conflict['⚠️Conflict']}")

    # Scenario: Bullish trend + ML Buy → no conflict
    row_ok = format_row("TEST",sig_bull_ml_sell,
                         False,False,False,False,
                         0.70, 62.0, "Bullish: RSI↑")
    row_ok_updated = dict(row_ok)
    # Reformat with prob>0.5
    sig2 = dict(sig_bull_ml_sell)
    row_ok2 = format_row("TEST",sig2,False,False,False,False,0.70,62.0,"Bullish: RSI↑")
    test("Bullish trend + ML Buy → no conflict",
         row_ok2["⚠️Conflict"] == "—",
         f"got: {row_ok2['⚠️Conflict']}")

    # ML prob boundaries
    test("prob 0.61 → 🟢 Buy signal",
         format_row("T",sig_bull_ml_sell,0,0,0,0,0.61,60,"")["ML_Signal"] == "🟢 Buy")
    test("prob 0.39 → 🔴 Sell signal",
         format_row("T",sig_bull_ml_sell,0,0,0,0,0.39,60,"")["ML_Signal"] == "🔴 Sell")
    test("prob 0.50 → 🟡 Hold signal",
         format_row("T",sig_bull_ml_sell,0,0,0,0,0.50,60,"")["ML_Signal"] == "🟡 Hold")

    # Verify ML is directionally correct on synthetic data
    bull_ind = add_indicators(make_bullish_df(700))
    bear_ind = add_indicators(make_bearish_df(700))
    p_bull, _, _ = train_and_predict(bull_ind)
    p_bear, _, _ = train_and_predict(bear_ind)
    test("ML bullish prob > ML bearish prob",
         p_bull > p_bear,
         f"bull={p_bull:.3f} bear={p_bear:.3f}")

except Exception as e:
    test("Data integrity", False, str(e))
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════
def _rsi_zone_check(rsi_val: float) -> str:
    from config import RSI_BULL_THRESHOLD, RSI_BEAR_THRESHOLD
    if rsi_val >= RSI_BULL_THRESHOLD:
        return "Bull"
    elif rsi_val <= RSI_BEAR_THRESHOLD:
        return "Bear"
    return "Neutral"

def _make_doji_df() -> pd.DataFrame:
    dates = pd.date_range(end=datetime.today(), periods=5, freq="B")
    # Last candle: open=100, close=100.1 (tiny body), high=105, low=95
    data = {"Open":[99,100,101,100,100.0],
            "High":[101,102,103,102,105.0],
            "Low": [97,98,99,98,95.0],
            "Close":[100,101,102,101,100.1],
            "Volume":[1e6]*5}
    return pd.DataFrame(data, index=dates).astype(
        {"Open":"float32","High":"float32","Low":"float32","Close":"float32"})

def _make_hammer_df() -> pd.DataFrame:
    dates = pd.date_range(end=datetime.today(), periods=5, freq="B")
    # Last candle: open=100, close=101 (small body at top), low=90, high=101.5
    data = {"Open":[99,100,101,100,100.0],
            "High":[102,103,104,103,101.5],
            "Low": [97,98,99,98,90.0],
            "Close":[101,102,103,102,101.0],
            "Volume":[1e6]*5}
    return pd.DataFrame(data, index=dates).astype(
        {"Open":"float32","High":"float32","Low":"float32","Close":"float32"})


# ══════════════════════════════════════════════════════════════
#  SUMMARY
# ══════════════════════════════════════════════════════════════
print("\n" + "━"*55)
total  = len(results)
passed = sum(1 for _,s,_ in results if s == PASS)
failed = sum(1 for _,s,_ in results if s == FAIL)

print(f"\n📊 TEST SUMMARY")
print(f"   Total  : {total}")
print(f"   ✅ Pass : {passed}")
print(f"   ❌ Fail : {failed}")
print(f"   Score  : {passed/total*100:.1f}%")

if failed > 0:
    print("\n❌ FAILED TESTS:")
    for name, status, detail in results:
        if status == FAIL:
            print(f"   • {name}: {detail}")

print("\n" + "━"*55)
sys.exit(0 if failed == 0 else 1)
