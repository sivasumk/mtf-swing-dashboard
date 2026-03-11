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


def grad_rg_v2(val, vmin: float = 0, vmax: float = 100) -> str:
    """Enhanced gradient: deep red → amber → emerald green with higher saturation."""
    try:
        norm = max(0.0, min(1.0, (float(val) - vmin) / (vmax - vmin + 1e-9)))
        if norm < 0.4:
            r, g, b = 220, int(60 + 140 * (norm / 0.4)), 40
            alpha = 0.35
        elif norm < 0.6:
            r, g, b = 160, 160, 80
            alpha = 0.15
        else:
            t = (norm - 0.6) / 0.4
            r, g, b = int(60 * (1 - t)), int(160 + 40 * t), int(80 + 40 * t)
            alpha = 0.30
        return f"background-color:rgba({r},{g},{b},{alpha});font-weight:600"
    except Exception:
        return ""


def grad_blue(val, vmin: float = 30, vmax: float = 70) -> str:
    """Same R→G gradient used for ML probability."""
    return grad_rg_v2(val, vmin, vmax)


def color_chg(val) -> str:
    """Green for positive, red for negative % change."""
    try:
        return "color:#26a69a;font-weight:600" if float(val) > 0 else (
               "color:#ef5350;font-weight:600" if float(val) < 0 else "")
    except Exception:
        return ""


def color_rsi(val) -> str:
    """RSI coloring: >70 overbought, >60 bull, <30 oversold, <40 bear."""
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
    """ADX: >40 strong gold, >25 mild green, else grey."""
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
        v = float(val)
        if v >= -3:  return "color:#26a69a;font-weight:700"
        if v >= -10: return "color:#26a69a"
        if v <= -30: return "color:#ef5350;font-weight:700"
        return "color:#9e9e9e"
    except Exception:
        return ""


def color_conflict(val) -> str:
    """Highlight ML-Trend conflicts."""
    return "color:#FF6B35;font-weight:700" if val == "⚠️" else ""


def color_rs(val) -> str:
    """RS_Score: >70 strong outperformer (green), <30 laggard (red), 40-60 neutral."""
    try:
        v = float(val)
        if v >= 70: return "background-color:rgba(38,166,154,0.35);font-weight:700"
        if v >= 60: return "background-color:rgba(38,166,154,0.18)"
        if v <= 30: return "background-color:rgba(239,83,80,0.35);font-weight:700"
        if v <= 40: return "background-color:rgba(239,83,80,0.18)"
        return "color:#9e9e9e"
    except Exception:
        return ""


def color_trade(val) -> str:
    """Trade bias with background tints."""
    s = str(val)
    if "LONG" in s and "?" not in s:
        return "background-color:rgba(38,166,154,0.2);color:#26a69a;font-weight:700"
    if "SHORT" in s and "?" not in s:
        return "background-color:rgba(239,83,80,0.2);color:#ef5350;font-weight:700"
    if "LONG?" in s:  return "color:#80cbc4"
    if "SHORT?" in s: return "color:#ef9a9a"
    return "color:#616161"


def color_struct(val) -> str:
    """Colour market structure column."""
    s = str(val)
    if s == "HH-HL": return "color:#26a69a;font-weight:600"
    if s == "LH-LL": return "color:#ef5350;font-weight:600"
    if s == "LH-HL": return "color:#ffcc02"
    if s == "HH-LL": return "color:#ff9800"
    return ""


def color_volspurt(val) -> str:
    """VolSpurt: bright colors for volume events."""
    s = str(val)
    if "SPURT" in s:
        return "background-color:rgba(0,230,118,0.25);color:#00e676;font-weight:700"
    if "DUMP" in s:
        return "background-color:rgba(255,82,82,0.25);color:#ff5252;font-weight:700"
    if "Abv5+20" in s:
        return "background-color:rgba(255,193,7,0.2);color:#ffc107;font-weight:600"
    return "color:#555"


def color_ml_signal(val) -> str:
    """ML signal styling with background."""
    s = str(val)
    if "Buy" in s:
        return "background-color:rgba(0,230,118,0.2);color:#00e676;font-weight:700"
    if "Sell" in s:
        return "background-color:rgba(255,82,82,0.2);color:#ff5252;font-weight:700"
    if "Hold" in s:
        return "color:#ffc107"
    return "color:#555"


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

    # --- Color mappings ---
    if "Chg%"       in cols: styler = styler.map(_sc(color_chg),    subset=["Chg%"])
    if "RSI"        in cols: styler = styler.map(_sc(color_rsi),    subset=["RSI"])
    if "ADX"        in cols: styler = styler.map(_sc(color_adx),    subset=["ADX"])
    if "52wH%"      in cols: styler = styler.map(_sc(color_52wh),   subset=["52wH%"])
    if "RS_Score"   in cols: styler = styler.map(_sc(color_rs),     subset=["RS_Score"])
    if "Trade"      in cols: styler = styler.map(_sc(color_trade),  subset=["Trade"])
    if "Mkt_Struct" in cols: styler = styler.map(_sc(color_struct), subset=["Mkt_Struct"])

    # Enhanced gradients (v2)
    if "MomScore"   in cols: styler = styler.map(_sc(lambda v: grad_rg_v2(v, 0, 100)), subset=["MomScore"])
    if "MTF_Score"  in cols: styler = styler.map(_sc(lambda v: grad_rg_v2(v/3*100, 0, 100)), subset=["MTF_Score"])
    if "SMI"        in cols: styler = styler.map(_sc(lambda v: grad_rg_v2(v + 100, 0, 200)), subset=["SMI"])

    # New column stylers
    if "VolSpurt"   in cols: styler = styler.map(_sc(color_volspurt),  subset=["VolSpurt"])
    if "ML_Signal"  in cols: styler = styler.map(_sc(color_ml_signal), subset=["ML_Signal"])

    # W_/M_ MomScore gradients
    for pfx in ("W_", "M_"):
        c = f"{pfx}MomScore"
        if c in cols:
            styler = styler.map(_sc(lambda v: grad_rg_v2(v, 0, 100)), subset=[c])

    if run_ml:
        if "ML_Prob%" in cols:
            styler = styler.map(_sc(lambda v: grad_blue(v, 30, 70)), subset=["ML_Prob%"])
        if "ML_Acc%" in cols:
            styler = styler.map(_sc(lambda v: grad_rg_v2(v, 50, 85)), subset=["ML_Acc%"])
        if "⚠️Conflict" in cols:
            styler = styler.map(color_conflict, subset=["⚠️Conflict"])
    return styler


DASHBOARD_CSS = """
<style>
/* ── Hide Streamlit chrome ── */
[data-testid="stToolbar"],
[data-testid="stDecoration"],
#MainMenu,
header[data-testid="stHeader"],
.stApp > header { display: none !important; }

/* ── Base theme ── */
.stApp { background-color: #0a0e14; }
div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1318 0%, #141820 100%);
    border-right: 1px solid #1e2535;
}
.block-container { padding-top: 0.5rem !important; }

/* ── Metric cards ── */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1a1f2e 0%, #151a26 100%);
    border-radius: 10px;
    padding: 12px 16px;
    border-left: 3px solid #5c7cfa;
    border-bottom: 1px solid #1e2535;
}

/* ── Stats bar pills ── */
.stats-bar {
    display: flex; gap: 12px; flex-wrap: wrap;
    padding: 10px 0; margin-bottom: 8px;
}
.stat-pill {
    background: #141820;
    border: 1px solid #1e2535;
    border-radius: 20px;
    padding: 6px 16px;
    display: flex; flex-direction: column; align-items: center;
    min-width: 80px;
}
.stat-label {
    font-size: 9px; color: #667788;
    text-transform: uppercase; letter-spacing: 0.8px;
}
.stat-val {
    font-size: 18px; font-weight: 700;
}

/* ── Table ── */
.stDataFrame {
    font-size: 12.5px !important;
}
.stDataFrame th {
    background: #1a1f2e !important;
    color: #8899bb !important;
    font-size: 10.5px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
    border-bottom: 2px solid #2a3550 !important;
    padding: 8px 6px !important;
}
.stDataFrame td {
    border-bottom: 1px solid #151a24 !important;
    padding: 6px !important;
}
.stDataFrame tr:hover td {
    background: #1a2035 !important;
}

/* ── Sidebar expanders ── */
section[data-testid="stSidebar"] .stExpander {
    margin-bottom: 4px;
    border: 1px solid #1e2535;
    border-radius: 8px;
}
</style>
"""
