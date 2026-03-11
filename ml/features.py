"""
ml/features.py — Feature Engineering for ML Models

Features (25 total):
  Price momentum  : lagged returns 1d, 5d, 10d, 20d
  RSI             : level + distance from 9-SMA + divergence flag
  Volatility      : ATR%, vol_z-score, HV_ratio, BB_Width, ATR_percentile
  Trend           : dist_ema20, dist_ema200, EMA_cross, Kumo_breakout
  MACD            : histogram normalised, MACD cross signal
  Volume          : OBV slope, volume ratio
  Calendar        : day_of_week, month (seasonal effects)
"""

import numpy as np
import pandas as pd

from config import ML_FORWARD_DAYS


FEATURE_COLS = [
    # Price momentum
    "ret_1d","ret_5d","ret_10d","ret_20d",
    # RSI
    "rsi","rsi_dist_sma","rsi_divergence",
    # SMI (replaces Williams%R — smoother, more informative)
    "smi","smi_dist_signal","smi_zone",
    # Volatility
    "atr_pct","vol_z","hv_ratio","bb_width","atr_pctile",
    # Trend
    "dist_ema20","dist_ema200","ema_cross","kumo",
    # MACD
    "macdh_norm","macd_cross",
    # Volume
    "obv_slope_z","vol_ratio",
    # Calendar
    "day_of_week","month",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build ML feature matrix from indicator-enriched OHLCV DataFrame.
    Returns DataFrame with FEATURE_COLS + 'target' column.
    Rows with NaN are dropped.
    """
    feats = pd.DataFrame(index=df.index)
    c     = df["Close"]

    # ── Price momentum ───────────────────────────────────────
    feats["ret_1d"]  = c.pct_change(1).astype("float32")
    feats["ret_5d"]  = c.pct_change(5).astype("float32")
    feats["ret_10d"] = c.pct_change(10).astype("float32")
    feats["ret_20d"] = c.pct_change(20).astype("float32")

    # ── RSI ──────────────────────────────────────────────────
    rsi               = _get(df, "RSI", 50)
    rsi_sma           = _get(df, "RSI_SMA9", 50)
    feats["rsi"]      = rsi.astype("float32")
    feats["rsi_dist_sma"] = (rsi - rsi_sma).astype("float32")
    feats["rsi_divergence"] = _get(df, "RSI_Divergence", 0).astype("float32")

    # ── SMI ──────────────────────────────────────────────────
    smi_val    = _get(df, "SMI", 0)
    smi_signal = _get(df, "SMI_Signal", 0)
    feats["smi"]           = smi_val.clip(-100, 100).astype("float32")
    feats["smi_dist_signal"]= (smi_val - smi_signal).clip(-50, 50).astype("float32")
    # Encode zone as numeric: OS=-2, Bear=-1, Bull=1, OB=2
    feats["smi_zone"] = pd.cut(
        smi_val, bins=[-np.inf, -40, 0, 40, np.inf],
        labels=[-2, -1, 1, 2]
    ).astype("float32")

    # ── Volatility ───────────────────────────────────────────
    atr_pct      = _get(df, "ATR_pct", 1).fillna(1)
    atr_roll_mean= atr_pct.rolling(30).mean().replace(0, np.nan)
    atr_roll_std = atr_pct.rolling(30).std().replace(0, np.nan)
    feats["atr_pct"]   = atr_pct.astype("float32")
    feats["vol_z"]     = ((atr_pct - atr_roll_mean) / atr_roll_std).astype("float32")
    feats["hv_ratio"]  = _get(df, "HV_ratio", 1).astype("float32")
    feats["bb_width"]  = _get(df, "BB_Width", 5).astype("float32")
    feats["atr_pctile"]= _get(df, "ATR_pctile", 50).astype("float32")

    # ── Trend ────────────────────────────────────────────────
    feats["dist_ema20"]  = _get(df, "Dist_EMA20_pct", 0).astype("float32")
    ema200 = _get(df, "EMA200", c)
    feats["dist_ema200"] = ((c - ema200) / ema200.replace(0, np.nan) * 100).astype("float32")
    feats["ema_cross"]   = _get(df, "EMA_cross", 0).astype("float32")
    feats["kumo"]        = _get(df, "Kumo_Breakout", 0).astype("float32")

    # ── MACD ─────────────────────────────────────────────────
    atr_safe = _get(df, "ATR", 1).fillna(1).replace(0, 1)
    macdh    = _get(df, "MACDh", 0)
    feats["macdh_norm"]  = (macdh / (atr_safe * 0.5 + 1e-9)).clip(-3, 3).astype("float32")
    feats["macd_cross"]  = _get(df, "MACD_cross", 0).astype("float32")

    # ── Volume ───────────────────────────────────────────────
    obv_slope = _get(df, "OBV_slope", 0)
    obv_std   = obv_slope.rolling(20).std().replace(0, np.nan)
    feats["obv_slope_z"] = (obv_slope / obv_std).clip(-3, 3).astype("float32")
    feats["vol_ratio"]   = _get(df, "Vol_ratio", 1).clip(0, 5).astype("float32")

    # ── Calendar ─────────────────────────────────────────────
    feats["day_of_week"] = pd.Series(df.index.dayofweek, index=df.index).astype("float32")
    feats["month"]       = pd.Series(df.index.month,     index=df.index).astype("float32")

    # ── Target: price higher in ML_FORWARD_DAYS days ─────────
    feats["target"] = (c.shift(-ML_FORWARD_DAYS) > c).astype("int8")

    # Drop rows with any NaN
    feats = feats.replace([np.inf, -np.inf], np.nan).dropna()
    return feats


def _get(df: pd.DataFrame, col: str, default) -> pd.Series:
    """Safely get a column from df, filling missing with default."""
    if col in df.columns:
        return df[col].fillna(default)
    if hasattr(default, "__len__"):
        return default
    return pd.Series(default, index=df.index)


def get_feature_names() -> list[str]:
    return FEATURE_COLS.copy()
