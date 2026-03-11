"""
indicators/engine.py — Technical Indicator Engine
Computes all indicators using the `ta` library (Python 3.12 compatible).
All outputs stored as float32 to minimise RAM.
"""

import logging
import numpy as np
import pandas as pd

from ta.trend import EMAIndicator, MACD, ADXIndicator, IchimokuIndicator, PSARIndicator
from ta.momentum import RSIIndicator, WilliamsRIndicator, ROCIndicator
from ta.volatility import AverageTrueRange, BollingerBands, KeltnerChannel
from ta.volume import OnBalanceVolumeIndicator

from config import (
    EMA_FAST, EMA_SLOW, EMA_TREND,
    ST_PERIOD, ST_MULTIPLIER,
    ADX_TREND_THRESHOLD,
    BB_PERIOD, BB_STD,
    KC_PERIOD, KC_ATR_PERIOD,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    DONCHIAN_PERIOD, PSAR_STEP, PSAR_MAX_STEP, VWMA_PERIOD,
)

log = logging.getLogger(__name__)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical indicators in-place.
    Requires columns: Open, High, Low, Close, Volume
    Minimum rows: 60 (ideally 200+)
    Returns enriched DataFrame (float32 columns).
    """
    if df.empty or len(df) < 60:
        return df

    df = df.copy()
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]

    # ── EMA Cross ────────────────────────────────────────────
    df.loc[:, "EMA20"]  = _safe(EMAIndicator(c, window=EMA_FAST).ema_indicator())
    df.loc[:, "EMA50"]  = _safe(EMAIndicator(c, window=EMA_SLOW).ema_indicator())
    df.loc[:, "EMA200"] = _safe(EMAIndicator(c, window=EMA_TREND).ema_indicator())

    df.loc[:, "EMA_cross"]    = (df["EMA20"] > df["EMA50"]).astype("int8")
    df.loc[:, "Above_EMA200"] = (c > df["EMA200"]).astype("int8")

    # ── SuperTrend ───────────────────────────────────────────
    df.loc[:, "SuperTrend_Dir"] = _supertrend(h, l, c, ST_PERIOD, ST_MULTIPLIER)

    # ── ADX ──────────────────────────────────────────────────
    adx_ind = ADXIndicator(h, l, c, window=14)
    df.loc[:, "ADX"]       = _safe(adx_ind.adx())
    df.loc[:, "ADX_pos"]   = _safe(adx_ind.adx_pos())
    df.loc[:, "ADX_neg"]   = _safe(adx_ind.adx_neg())
    df.loc[:, "ADX_slope"] = df["ADX"].diff(3).astype("float32")

    # ── Ichimoku ─────────────────────────────────────────────
    df.loc[:, "Kumo_Breakout"] = _ichimoku_breakout(h, l, c, df.index)

    # ── RSI ──────────────────────────────────────────────────
    rsi = RSIIndicator(c, window=14).rsi()
    df.loc[:, "RSI"]      = _safe(rsi)
    df.loc[:, "RSI_SMA9"] = _safe(rsi.rolling(9).mean())
    price_dir = c.diff(5).apply(lambda x: 1 if x > 0 else -1)
    rsi_dir   = rsi.diff(5).apply(lambda x: 1 if x > 0 else -1)
    df.loc[:, "RSI_Divergence"] = (price_dir != rsi_dir).astype("int8")

    # ── MACD ─────────────────────────────────────────────────
    macd_ind = MACD(c, window_fast=MACD_FAST, window_slow=MACD_SLOW, window_sign=MACD_SIGNAL)
    df.loc[:, "MACD"]  = _safe(macd_ind.macd())
    df.loc[:, "MACDs"] = _safe(macd_ind.macd_signal())
    df.loc[:, "MACDh"] = _safe(macd_ind.macd_diff())
    df.loc[:, "MACD_cross"] = _macd_cross(df["MACD"], df["MACDs"])

    # ── ATR ──────────────────────────────────────────────────
    df.loc[:, "ATR"]     = _safe(AverageTrueRange(h, l, c, window=14).average_true_range())
    df.loc[:, "ATR_pct"] = _safe((df["ATR"] / c * 100))
    df.loc[:, "HV_ratio"] = _safe(
        df["ATR_pct"].rolling(10).mean() /
        (df["ATR_pct"].rolling(30).mean().replace(0, np.nan))
    )

    # ATR percentile rank (0-100) over last 252 days
    df.loc[:, "ATR_pctile"] = (
        df["ATR_pct"]
        .rolling(252, min_periods=50)
        .apply(lambda x: (x[-1] > x[:-1]).mean() * 100, raw=True)
        .astype("float32")
    )

    # ── Bollinger Bands ──────────────────────────────────────
    bb = BollingerBands(c, window=BB_PERIOD, window_dev=BB_STD)
    df.loc[:, "BB_Upper"] = _safe(bb.bollinger_hband())
    df.loc[:, "BB_Lower"] = _safe(bb.bollinger_lband())
    df.loc[:, "BB_Mid"]   = _safe(bb.bollinger_mavg())
    df.loc[:, "BB_Width"] = _safe(bb.bollinger_wband())
    df.loc[:, "BB_pct"]   = _safe(bb.bollinger_pband())

    # ── Keltner + Squeeze ────────────────────────────────────
    try:
        kc = KeltnerChannel(h, l, c, window=KC_PERIOD, window_atr=KC_ATR_PERIOD)
        df.loc[:, "KC_Upper"] = _safe(kc.keltner_channel_hband())
        df.loc[:, "KC_Lower"] = _safe(kc.keltner_channel_lband())
        df.loc[:, "Squeeze"]  = (
            (df["BB_Upper"] < df["KC_Upper"]) &
            (df["BB_Lower"] > df["KC_Lower"])
        ).astype("int8")
    except Exception:
        df.loc[:, "Squeeze"] = np.int8(0)

    # ── Stochastic Momentum Index (SMI) ─────────────────────
    smi_val, smi_signal = _smi(h, l, c, length_k=10, length_d=3, length_ema=3)
    df.loc[:, "SMI"]        = smi_val
    df.loc[:, "SMI_Signal"] = smi_signal

    # ── Williams %R ──────────────────────────────────────────
    df.loc[:, "WilliamsR"] = _safe(WilliamsRIndicator(h, l, c, lbp=14).williams_r())

    # ── Rate of Change ───────────────────────────────────────
    df.loc[:, "ROC10"] = _safe(ROCIndicator(c, window=10).roc())
    df.loc[:, "ROC20"] = _safe(ROCIndicator(c, window=20).roc())

    # ── Volume indicators ────────────────────────────────────
    df.loc[:, "OBV"]       = _safe(OnBalanceVolumeIndicator(c, v).on_balance_volume())
    df.loc[:, "OBV_slope"] = _safe(df["OBV"].diff(5))
    vol_ma5  = v.rolling(5).mean().replace(0, 1)
    vol_ma20 = v.rolling(20).mean().replace(0, 1)
    df.loc[:, "Vol_ratio"]  = _safe((v / vol_ma20))
    df.loc[:, "Vol_ratio5"] = _safe((v / vol_ma5))

    # ── Market Structure (HH/HL/LH/LL) ──────────────────────
    df.loc[:, "Swing_High"] = h.rolling(10, center=True).max()
    df.loc[:, "Swing_Low"]  = l.rolling(10, center=True).min()
    df.loc[:, "Mkt_Struct"] = _market_structure(h, l, c)

    # ── Price context ────────────────────────────────────────
    df.loc[:, "High_52w"]       = h.rolling(252, min_periods=50).max().astype("float32")
    df.loc[:, "Low_52w"]        = l.rolling(252, min_periods=50).min().astype("float32")
    df.loc[:, "Pct_from_52wH"]  = _safe(((c - df["High_52w"]) / df["High_52w"] * 100))
    df.loc[:, "Pct_from_52wL"]  = _safe(((c - df["Low_52w"])  / df["Low_52w"]  * 100))
    df.loc[:, "Dist_EMA20_pct"] = _safe(((c - df["EMA20"])    / df["EMA20"]    * 100))

    # ── DI Crossover ───────────────────────────────────────
    _di_above = df["ADX_pos"] > df["ADX_neg"]
    _di_cross_up   = _di_above & ~_di_above.shift(1, fill_value=False)
    _di_cross_down = ~_di_above & _di_above.shift(1, fill_value=False)
    di_cross = pd.Series(np.int8(0), index=df.index)
    di_cross[_di_cross_up]   = np.int8(1)
    di_cross[_di_cross_down] = np.int8(-1)
    df.loc[:, "DI_Cross"] = di_cross

    # ── Donchian Channel ───────────────────────────────────
    df.loc[:, "Donchian_High"]  = h.rolling(DONCHIAN_PERIOD).max().astype("float32")
    df.loc[:, "Donchian_Low"]   = l.rolling(DONCHIAN_PERIOD).min().astype("float32")
    df.loc[:, "Donchian_Break"] = np.where(
        c >= df["Donchian_High"], np.int8(1),
        np.where(c <= df["Donchian_Low"], np.int8(-1), np.int8(0))
    ).astype("int8")

    # ── VWMA (daily VWAP proxy) ────────────────────────────
    _cv   = (c * v).rolling(VWMA_PERIOD).sum()
    _vsum = v.rolling(VWMA_PERIOD).sum().replace(0, np.nan)
    df.loc[:, "VWMA20"]     = _safe(_cv / _vsum)
    df.loc[:, "Above_VWMA"] = (c > df["VWMA20"]).astype("int8")

    # ── Parabolic SAR ──────────────────────────────────────
    try:
        _psar = PSARIndicator(h, l, c, step=PSAR_STEP, max_step=PSAR_MAX_STEP)
        _psar_up   = _psar.psar_up()
        _psar_down = _psar.psar_down()
        df.loc[:, "PSAR"] = _safe(_psar_up.fillna(_psar_down))
        _psar_dir = np.where(_psar_up.notna(), np.int8(1), np.int8(-1))
        df.loc[:, "PSAR_Dir"] = pd.Series(_psar_dir, index=df.index, dtype="int8")
        _prev_dir = df["PSAR_Dir"].shift(1)
        df.loc[:, "PSAR_Flip"] = np.where(
            (df["PSAR_Dir"] == 1) & (_prev_dir == -1), np.int8(1),
            np.where((df["PSAR_Dir"] == -1) & (_prev_dir == 1), np.int8(-1), np.int8(0))
        ).astype("int8")
    except Exception:
        df.loc[:, "PSAR"]      = _safe(c * np.nan)
        df.loc[:, "PSAR_Dir"]  = np.int8(0)
        df.loc[:, "PSAR_Flip"] = np.int8(0)

    return df


# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════
def _safe(series: pd.Series) -> pd.Series:
    """Cast to float32, propagate NaN safely."""
    return series.astype("float32") if series is not None else pd.Series(dtype="float32")


def _supertrend(h, l, c, period: int, multiplier: float) -> pd.Series:
    """
    Manual SuperTrend implementation (ta library has no SuperTrend).
    Returns +1 (bullish) or -1 (bearish).
    """
    atr  = AverageTrueRange(h, l, c, window=period).average_true_range()
    hl2  = (h + l) / 2
    upper= (hl2 + multiplier * atr).astype("float32")
    lower= (hl2 - multiplier * atr).astype("float32")

    direction = pd.Series(1, index=c.index, dtype="int8")
    for i in range(1, len(c)):
        prev_c = c.iloc[i - 1]
        if prev_c > upper.iloc[i - 1]:
            direction.iloc[i] = 1
        elif prev_c < lower.iloc[i - 1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i - 1]
    return direction


def _ichimoku_breakout(h, l, c, index) -> pd.Series:
    """
    Returns:  +1 price above Kumo (bullish)
              -1 price below Kumo (bearish)
               0 price inside Kumo (neutral)
    """
    try:
        ich     = IchimokuIndicator(h, l, window1=9, window2=26, window3=52)
        ka      = ich.ichimoku_a().reindex(index)
        kb      = ich.ichimoku_b().reindex(index)
        top     = pd.concat([ka, kb], axis=1).max(axis=1)
        bot     = pd.concat([ka, kb], axis=1).min(axis=1)
        result  = np.where(c > top, 1, np.where(c < bot, -1, 0))
        return pd.Series(result, index=index, dtype="int8")
    except Exception:
        return pd.Series(0, index=index, dtype="int8")


def _market_structure(h: pd.Series, l: pd.Series, c: pd.Series,
                      swing_bars: int = 10) -> pd.Series:
    """
    Classify market structure on each bar using recent swing pivots.
    Looks back over last 2×swing_bars to find prior and current pivots.

    Returns string series:
      "HH-HL"  — Higher High + Higher Low  (bullish structure)
      "LH-LL"  — Lower High  + Lower Low   (bearish structure)
      "HH-LL"  — Higher High + Lower Low   (expanding / volatile)
      "LH-HL"  — Lower High  + Higher Low  (contracting / coiling)
      "—"      — insufficient data
    """
    n      = len(c)
    result = pd.Series("—", index=c.index, dtype=object)

    lookback = swing_bars * 4
    for i in range(lookback, n):
        window_h = h.iloc[i - lookback : i]
        window_l = l.iloc[i - lookback : i]

        # Split into two halves — prior vs recent
        mid   = lookback // 2
        prior_h = float(window_h.iloc[:mid].max())
        prior_l = float(window_l.iloc[:mid].min())
        cur_h   = float(window_h.iloc[mid:].max())
        cur_l   = float(window_l.iloc[mid:].min())

        hh = cur_h > prior_h   # higher high
        hl = cur_l > prior_l   # higher low

        if hh and hl:
            result.iloc[i] = "HH-HL"
        elif not hh and not hl:
            result.iloc[i] = "LH-LL"
        elif hh and not hl:
            result.iloc[i] = "HH-LL"
        else:
            result.iloc[i] = "LH-HL"

    return result


def _smi(h: pd.Series, l: pd.Series, c: pd.Series,
         length_k: int = 10, length_d: int = 3, length_ema: int = 3
         ) -> tuple[pd.Series, pd.Series]:
    """
    Stochastic Momentum Index — TradingView Pine Script equivalent.

    Pine: emaEma(s, n) = ta.ema(ta.ema(s, n), n)
          SMI = 200 * emaEma(relRange, D) / emaEma(hlRange, D)
          Signal = ta.ema(SMI, lengthEMA)

    Better than standard Stochastic because:
      - Centred on range midpoint (not bottom) → doesn't hug extremes in trends
      - Double EMA smoothing → fewer false crossovers vs SMA-based Stoch
      - Zero-line crossover is a clean signal

    Range: -100 to +100  |  Overbought > +40  |  Oversold < -40
    """
    def _ema_ema(s: pd.Series, n: int) -> pd.Series:
        return s.ewm(span=n, adjust=False).mean().ewm(span=n, adjust=False).mean()

    highest_high = h.rolling(length_k).max()
    lowest_low   = l.rolling(length_k).min()
    hl_range     = (highest_high - lowest_low).replace(0, np.nan)
    rel_range    = c - (highest_high + lowest_low) / 2

    smi_val    = _safe(200 * _ema_ema(rel_range, length_d) / _ema_ema(hl_range, length_d))
    smi_signal = _safe(smi_val.ewm(span=length_ema, adjust=False).mean())
    return smi_val, smi_signal


def _macd_cross(macd: pd.Series, signal: pd.Series) -> pd.Series:
    """
    +1 = bullish crossover (MACD crossed above signal)
    -1 = bearish crossover (MACD crossed below signal)
     0 = no cross
    """
    above = macd > signal
    cross_up   = above & ~above.shift(1, fill_value=False)   # False→True
    cross_down = ~above & above.shift(1, fill_value=False)   # True→False
    cross = pd.Series(np.int8(0), index=macd.index)
    cross[cross_up]   = np.int8(1)
    cross[cross_down] = np.int8(-1)
    return cross


# ══════════════════════════════════════════════════════════════
#  TIMEFRAME RESAMPLER
# ══════════════════════════════════════════════════════════════
def resample_to_tf(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """
    Resample daily OHLCV to weekly (W) or monthly (ME) bars.
    Uses standard OHLCV aggregation rules.

    Args:
        df : daily OHLCV DataFrame with DatetimeIndex
        tf : "D" (no-op), "W" (weekly), "ME" (month-end)

    Returns:
        Resampled DataFrame, same column structure as input.
        Minimum 60 bars required for indicators; returns empty if insufficient.
    """
    if tf == "D" or df.empty:
        return df

    agg = {
        "Open"   : "first",
        "High"   : "max",
        "Low"    : "min",
        "Close"  : "last",
        "Volume" : "sum",
    }
    # Keep only OHLCV cols before resampling (drop indicator cols if any)
    ohlcv_cols = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
    try:
        resampled = (
            df[ohlcv_cols]
            .resample(tf)
            .agg(agg)
            .dropna()
        )
    except Exception as e:
        log.warning(f"resample_to_tf({tf}) failed: {e}")
        return pd.DataFrame()

    # Maintain float32 for price cols
    for c in ["Open","High","Low","Close"]:
        if c in resampled.columns:
            resampled[c] = resampled[c].astype("float32")

    MIN_BARS = {"W": 52, "ME": 24}   # min bars for reliable indicators
    if len(resampled) < MIN_BARS.get(tf, 30):
        log.debug(f"resample_to_tf({tf}): only {len(resampled)} bars — skipping")
        return pd.DataFrame()

    return resampled
