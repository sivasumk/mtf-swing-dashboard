"""
utils/styling.py — Color and formatting helpers for Streamlit DataFrames
No matplotlib dependency — pure RGB computation.
"""


def grad_rg(val, vmin: float = 0, vmax: float = 100) -> str:
    """Red→Yellow→Green gradient background."""
    try:
        norm = max(0.0, min(1.0, (float(val) - vmin) / (vmax - vmin + 1e-9)))
        r = int(255 * (1 - norm))
        g = int(180 * norm + 55)
        return f"background-color:rgba({r},{g},60,0.30)"
    except Exception:
        return ""


def grad_blue(val, vmin: float = 30, vmax: float = 70) -> str:
    """Same R→G gradient used for ML probability."""
    return grad_rg(val, vmin, vmax)


def color_chg(val) -> str:
    """Green for positive, red for negative % change."""
    try:
        return "color:#26a69a;font-weight:600" if float(val) > 0 else (
               "color:#ef5350;font-weight:600" if float(val) < 0 else "")
    except Exception:
        return ""


def color_rsi(val) -> str:
    """
    RSI coloring:
      > 60 → green (bull zone)
      < 40 → red (bear zone)
      > 70 → bold red (overbought)
      < 30 → bold green (oversold)
    """
    try:
        v = float(val)
        if v >= 70:  return "color:#ff6b35;font-weight:700"
        if v >= 60:  return "color:#26a69a;font-weight:600"
        if v <= 30:  return "color:#26a69a;font-weight:700"
        if v <= 40:  return "color:#ef5350;font-weight:600"
        return "color:#9e9e9e"
    except Exception:
        return ""


def color_adx(val) -> str:
    """ADX: >40 strong green, >25 mild green, else grey."""
    try:
        v = float(val)
        if v >= 40: return "color:#FFD700;font-weight:700"
        if v >= 25: return "color:#26a69a"
        return "color:#9e9e9e"
    except Exception:
        return ""


def color_52wh(val) -> str:
    """Distance from 52-week high: near high = green, far = red."""
    try:
        v = float(val)   # negative value (0 = at high, -20 = 20% below)
        if v >= -3:  return "color:#26a69a;font-weight:700"   # near high
        if v >= -10: return "color:#26a69a"
        if v <= -30: return "color:#ef5350;font-weight:700"
        return "color:#9e9e9e"
    except Exception:
        return ""


def color_conflict(val) -> str:
    """Highlight ML-Trend conflicts."""
    return "color:#FF6B35;font-weight:700" if val == "⚠️" else ""


def color_rs(val) -> str:
    """
    RS_Score: centred at 50 (in-line with Nifty).
    >70 = strong outperformer (green), <30 = laggard (red), 40-60 = neutral.
    """
    try:
        v = float(val)
        if v >= 70: return "background-color:rgba(38,166,154,0.35);font-weight:700"
        if v >= 60: return "background-color:rgba(38,166,154,0.18)"
        if v <= 30: return "background-color:rgba(239,83,80,0.35);font-weight:700"
        if v <= 40: return "background-color:rgba(239,83,80,0.18)"
        return "color:#9e9e9e"  # neutral band 40-60
    except Exception:
        return ""


def color_trade(val) -> str:
    """Colour Trade bias column."""
    s = str(val)
    if "LONG" in s and "?" not in s:   return "color:#26a69a;font-weight:700"
    if "SHORT" in s and "?" not in s:  return "color:#ef5350;font-weight:700"
    if "LONG?" in s:                   return "color:#80cbc4"
    if "SHORT?" in s:                  return "color:#ef9a9a"
    return "color:#616161"


def color_struct(val) -> str:
    """Colour market structure column."""
    s = str(val)
    if s == "HH-HL": return "color:#26a69a;font-weight:600"   # bullish
    if s == "LH-LL": return "color:#ef5350;font-weight:600"   # bearish
    if s == "LH-HL": return "color:#ffcc02"                   # coiling
    if s == "HH-LL": return "color:#ff9800"                   # expanding
    return ""


def apply_table_style(styler, run_ml: bool, cols: list):
    """Apply all column styles to a pandas Styler object."""

    # Safe formatter: never crashes on string/None values
    def _sf(fmt_str):
        def _f(v):
            try:
                if v is None or (isinstance(v, float) and v != v):
                    return "—"
                return fmt_str.format(v)
            except (ValueError, TypeError):
                return "" if v is None else str(v)
        return _f

    # Safe color mapper: skip non-numeric values silently
    def _sc(fn):
        def _f(v):
            try:
                if v is None or (isinstance(v, float) and v != v):
                    return ""
                return fn(v)
            except (ValueError, TypeError):
                return ""
        return _f

    # Build format dict — only for columns that exist
    fmt_spec = [
        ("MACD",      "{:.2f}"),
        ("Price",     "{:,.2f}"),
        ("Chg%",      "{:+.2f}%"),
        ("MomScore",  "{:.1f}"),
        ("RSI",       "{:.1f}"),
        ("ADX",       "{:.1f}"),
        ("ATR%",      "{:.2f}%"),
        ("52wH%",     "{:.1f}%"),
        ("RS_Score",  "{:.1f}"),
        ("RS_1M",     "{:.3f}"),
        ("RS_3M",     "{:.3f}"),
        ("RS_Rank",   "{:.0f}"),
        ("MTF_Score", "{:.0f}"),
        ("Rank",      "{:.0f}"),
        ("ML_Prob%",  "{:.1f}%"),
        ("ML_Acc%",   "{:.1f}%"),
        ("SMI",       "{:.1f}"),
        ("SMI_Signal","{:.1f}"),
    ]
    fmt = {col: _sf(fs) for col, fs in fmt_spec if col in cols}

    # W_ / M_ numeric prefixed columns
    for col in cols:
        for suffix, fs in [("MomScore","{:.1f}"),("RSI","{:.1f}"),
                           ("ADX","{:.1f}"),("ATR%","{:.2f}%"),
                           ("ML_Prob%","{:.1f}%"),("ML_Acc%","{:.1f}%")]:
            if col in (f"W_{suffix}", f"M_{suffix}"):
                fmt[col] = _sf(fs)

    styler = styler.format(fmt, na_rep="—")

    if "Chg%"      in cols: styler = styler.map(_sc(color_chg),                 subset=["Chg%"])
    if "RSI"       in cols: styler = styler.map(_sc(color_rsi),                 subset=["RSI"])
    if "ADX"       in cols: styler = styler.map(_sc(color_adx),                 subset=["ADX"])
    if "52wH%"     in cols: styler = styler.map(_sc(color_52wh),                subset=["52wH%"])
    if "MomScore"  in cols: styler = styler.map(_sc(lambda v: grad_rg(v,0,100)),subset=["MomScore"])
    # RS_Score: 50=neutral (grey), >70=strong green, <30=red  — centred at 50 not 0
    if "RS_Score"  in cols: styler = styler.map(_sc(color_rs),                  subset=["RS_Score"])
    if "MTF_Score" in cols: styler = styler.map(
        _sc(lambda v: grad_rg(v/3*100, 0, 100)), subset=["MTF_Score"])
    if "SMI"       in cols: styler = styler.map(
        _sc(lambda v: grad_rg(v + 100, 0, 200)), subset=["SMI"])
    if "Trade"     in cols: styler = styler.map(_sc(color_trade),               subset=["Trade"])
    if "Mkt_Struct"in cols: styler = styler.map(_sc(color_struct),              subset=["Mkt_Struct"])

    if run_ml:
        if "ML_Prob%" in cols:
            styler = styler.map(_sc(lambda v: grad_blue(v,30,70)), subset=["ML_Prob%"])
        if "ML_Acc%" in cols:
            styler = styler.map(_sc(lambda v: grad_rg(v,50,85)), subset=["ML_Acc%"])
        if "⚠️Conflict" in cols:
            styler = styler.map(color_conflict, subset=["⚠️Conflict"])
    return styler


DASHBOARD_CSS = """
<style>
/* ── Hide Streamlit deploy toolbar & top padding it creates ── */
[data-testid="stToolbar"]          { display: none !important; }
[data-testid="stDecoration"]       { display: none !important; }
#MainMenu                          { display: none !important; }
header[data-testid="stHeader"]     { display: none !important; }
.stApp > header                    { display: none !important; }

.stApp { background-color: #0e1117; }
div[data-testid="stSidebar"] { background-color: #141820; }
.stDataFrame { font-size: 12px; }
.block-container { padding-top: 0.5rem !important; }
div[data-testid="metric-container"] {
    background: #1e2130;
    border-radius: 8px;
    padding: 10px 14px;
    border-left: 3px solid #4CAF50;
}
</style>
"""
