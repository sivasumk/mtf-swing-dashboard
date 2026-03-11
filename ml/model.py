"""
ml/model.py — XGBoost + Ensemble ML Model

Design:
  1. XGBoost (GPU if available) — primary model
  2. Random Forest               — diversity / stability
  3. Logistic Regression         — linear baseline
  Ensemble: weighted average probabilities (0.5 / 0.3 / 0.2)

  Time-decay sample weighting: recent bars get higher weight
  Walk-forward validation:      3 expanding folds for accuracy estimate

Returns:
  probability (float)    — ensemble bullish probability 0-1
  wf_accuracy (float)    — walk-forward accuracy %
  top_reasons (str)      — top 2 feature contributions
"""

import gc
import logging
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from datetime import datetime, timedelta
from config import (
    ML_FORWARD_DAYS, ML_MIN_SAMPLES, ML_TIME_DECAY,
    ML_TRAIN_YEARS,
    WF_N_SPLITS, WF_TEST_SIZE,
)
from ml.features import build_features, FEATURE_COLS

log = logging.getLogger(__name__)

# XGBoost with GPU auto-detection
try:
    import xgboost as xgb
    _test_model = xgb.XGBClassifier(device="cuda", n_estimators=1, verbosity=0)
    _test_model.fit([[0]*5], [0])
    USE_GPU    = True
    XGB_DEVICE = "cuda"
    log.info("XGBoost GPU (CUDA) enabled")
except Exception:
    try:
        import xgboost as xgb
        USE_GPU    = False
        XGB_DEVICE = "cpu"
        log.info("XGBoost CPU mode")
    except ImportError:
        xgb        = None
        USE_GPU    = False
        XGB_DEVICE = "none"
        log.warning("XGBoost not available — using RF + LR only")

# Ensemble weights
W_XGB = 0.50
W_RF  = 0.30
W_LR  = 0.20


# ══════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════
def train_and_predict(df_with_indicators: pd.DataFrame) -> tuple[float, float, str]:
    """
    Train ensemble on last ML_TRAIN_YEARS of data only (regime-aware).
    Using full 15yr history causes bull-market bias: model never predicts Sell
    because 65%+ of historical 5d returns are positive.
    Capping to 3yr gives a more honest current-regime signal.

    Returns:
        (probability, wf_accuracy_pct, reason_string)
        probability : 0.0–1.0  (>ML_STRONG_BUY_PROB=Buy, <ML_STRONG_SELL_PROB=Sell)
    """
    feats = build_features(df_with_indicators)

    if len(feats) < ML_MIN_SAMPLES:
        return 0.5, 0.0, "Insufficient data"

    # ── Regime-aware: use last N trading days only (bull-bias fix) ──
    # tail() is robust — no dtype/timezone comparison needed
    TRADING_DAYS_PER_YEAR = 252
    recent_bars = ML_TRAIN_YEARS * TRADING_DAYS_PER_YEAR
    if len(feats) > recent_bars + ML_MIN_SAMPLES:
        feats = feats.tail(recent_bars)
    # If not enough recent data, use whatever we have (already checked above)

    X_all = feats[FEATURE_COLS].values.astype("float32")
    y_all = feats["target"].values

    # Time-decay sample weights
    n          = len(X_all)
    indices    = np.arange(n)
    weights    = np.exp(ML_TIME_DECAY * (indices - n)).astype("float32")
    weights   /= weights.sum()
    weights   *= n   # scale so sum = n (sklearn expects sample_weight ≈ n)

    # Walk-forward accuracy (on historical data)
    wf_acc = _walk_forward_accuracy(X_all, y_all, weights)

    # Train final model on ALL data except last ML_FORWARD_DAYS
    X_train = X_all[:-ML_FORWARD_DAYS]
    y_train = y_all[:-ML_FORWARD_DAYS]
    w_train = weights[:-ML_FORWARD_DAYS]
    X_pred  = X_all[-1].reshape(1, -1)

    # Scale
    scaler       = StandardScaler()
    X_train_sc   = scaler.fit_transform(X_train)
    X_pred_sc    = scaler.transform(X_pred)

    prob   = _ensemble_predict(X_train_sc, y_train, w_train, X_pred_sc)
    reason = _explain(feats.iloc[-1], prob)

    del X_all, y_all, X_train, y_train, w_train, X_pred
    gc.collect()

    return round(float(prob), 3), round(wf_acc, 1), reason


# ══════════════════════════════════════════════════════════════
#  ENSEMBLE PREDICT
# ══════════════════════════════════════════════════════════════
def _ensemble_predict(X_train, y_train, weights, X_pred) -> float:
    probs = []
    w_used = []

    # XGBoost
    if xgb is not None:
        try:
            model_xgb = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.04,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                gamma=0.1,
                device=XGB_DEVICE,
                tree_method="hist",
                eval_metric="logloss",
                verbosity=0,
                random_state=42,
            )
            model_xgb.fit(X_train, y_train, sample_weight=weights)
            p = float(model_xgb.predict_proba(X_pred)[0][1])
            probs.append(p)
            w_used.append(W_XGB)
            del model_xgb
        except Exception as e:
            log.warning(f"XGBoost failed: {e}")

    # Random Forest
    try:
        model_rf = RandomForestClassifier(
            n_estimators=80,
            max_depth=6,
            min_samples_leaf=8,
            max_features="sqrt",
            n_jobs=-1,
            random_state=42,
        )
        model_rf.fit(X_train, y_train, sample_weight=weights)
        p = float(model_rf.predict_proba(X_pred)[0][1])
        probs.append(p)
        w_used.append(W_RF)
        del model_rf
    except Exception as e:
        log.warning(f"RF failed: {e}")

    # Logistic Regression
    try:
        model_lr = LogisticRegression(
            C=0.1,
            max_iter=500,
            random_state=42,
            solver="lbfgs",
        )
        model_lr.fit(X_train, y_train, sample_weight=weights)
        p = float(model_lr.predict_proba(X_pred)[0][1])
        probs.append(p)
        w_used.append(W_LR)
        del model_lr
    except Exception as e:
        log.warning(f"LR failed: {e}")

    if not probs:
        return 0.5

    # Weighted average
    total_w = sum(w_used)
    prob    = sum(p * w for p, w in zip(probs, w_used)) / total_w
    return float(np.clip(prob, 0, 1))


# ══════════════════════════════════════════════════════════════
#  WALK-FORWARD ACCURACY  (fast — RF only, batch predict)
# ══════════════════════════════════════════════════════════════
def _walk_forward_accuracy(X, y, weights) -> float:
    """
    Fast walk-forward: RF only + batch predict (no per-sample loop).
    ~50x faster than full ensemble per sample.
    Full ensemble still used for the live prediction.
    """
    n        = len(X)
    test_sz  = max(30, int(n * WF_TEST_SIZE))
    min_train= ML_MIN_SAMPLES
    accuracies = []

    for fold in range(WF_N_SPLITS):
        test_end   = n - fold * test_sz - ML_FORWARD_DAYS
        test_start = test_end - test_sz
        if test_start < min_train:
            break
        X_tr = X[:test_start]; y_tr = y[:test_start]; w_tr = weights[:test_start]
        X_te = X[test_start:test_end]; y_te = y[test_start:test_end]
        if len(X_te) < 10:
            continue
        scaler  = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_te_sc = scaler.transform(X_te)
        rf = RandomForestClassifier(
            n_estimators=80, max_depth=5, min_samples_leaf=8,
            max_features="sqrt", n_jobs=-1, random_state=42,
        )
        rf.fit(X_tr_sc, y_tr, sample_weight=w_tr)
        preds = rf.predict(X_te_sc)
        accuracies.append(float(np.mean(preds == y_te)))
        del rf

    return float(np.mean(accuracies) * 100) if accuracies else 0.0


# ══════════════════════════════════════════════════════════════
#  EXPLANATION  (top driving features)
# ══════════════════════════════════════════════════════════════
_FEATURE_LABELS = {
    "rsi"         : "RSI",
    "ret_5d"      : "5d Return",
    "dist_ema20"  : "EMA20 Dist",
    "macdh_norm"  : "MACD",
    "vol_z"       : "Vol Z",
    "kumo"        : "Ichimoku",
    "ema_cross"   : "EMA Cross",
    "rsi_dist_sma": "RSI Momentum",
    "obv_slope_z" : "OBV",
    "dist_ema200" : "EMA200 Dist",
    "atr_pctile"  : "ATR Rank",
    "ret_1d"      : "1d Return",
}

def _explain(last_row: pd.Series, prob: float) -> str:
    """
    Rule-based explanation of the top 2 signal drivers.
    Simple and interpretable — no SHAP needed.
    """
    drivers = []

    rsi = float(last_row.get("rsi", 50))
    if rsi > 60:   drivers.append(("RSI↑", abs(rsi - 50)))
    elif rsi < 40: drivers.append(("RSI↓", abs(rsi - 50)))

    macdh = float(last_row.get("macdh_norm", 0))
    if abs(macdh) > 0.3:
        drivers.append(("MACD" + ("↑" if macdh > 0 else "↓"), abs(macdh)))

    dist = float(last_row.get("dist_ema20", 0))
    if abs(dist) > 1.5:
        drivers.append(("EMA20" + ("↑" if dist > 0 else "↓"), abs(dist)))

    ret5 = float(last_row.get("ret_5d", 0))
    if abs(ret5) > 0.02:
        drivers.append(("5d" + ("↑" if ret5 > 0 else "↓"), abs(ret5) * 100))

    kumo = float(last_row.get("kumo", 0))
    if kumo != 0:
        drivers.append(("Kumo" + ("↑" if kumo > 0 else "↓"), 1.0))

    obv = float(last_row.get("obv_slope_z", 0))
    if abs(obv) > 0.5:
        drivers.append(("OBV" + ("↑" if obv > 0 else "↓"), abs(obv)))

    if not drivers:
        return "Mixed signals"

    # Sort by magnitude, take top 2
    drivers.sort(key=lambda x: x[1], reverse=True)
    top = " · ".join(d[0] for d in drivers[:2])
    direction = "Bullish" if prob > 0.5 else "Bearish"
    return f"{direction}: {top}"
