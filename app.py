"""
app.py — MTF Swing Trading Dashboard  (UI v2)
Key UI improvements:
  - Market context bar (Nifty + VIX) always at top
  - Quick-scan preset buttons (1-click filter combos)
  - Active filter count badge
  - Watchlist tab (pinned tickers, session-persistent)
  - Reorganised compact sidebar
  - Market regime colour coding
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from datetime import datetime
import streamlit as st
import pandas as pd

from config import (
    PAGE_CFG, UNIVERSE_MAP, INDEX_TICKERS,
    DB_PATH, TABLE_ROW_HEIGHT, TABLE_MAX_HEIGHT,
)
from data.cache import (
    get_conn, load_ohlcv,
    batch_download_missing, delta_update_parallel,
    cache_stats,
)
from utils.universe import (
    build_universe_df, build_universe_tf, merge_mtf,
    apply_filters, sort_df, universe_stats,
)
from utils.styling import apply_table_style, DASHBOARD_CSS
from ml.model import USE_GPU

# ══════════════════════════════════════════════════════════════
#  PAGE SETUP
# ══════════════════════════════════════════════════════════════
st.set_page_config(**PAGE_CFG)
st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)

# Extra CSS for UI v2
st.markdown("""
<style>
/* Market context bar */
.mkt-bar {
    display: flex; gap: 20px; align-items: center;
    background: linear-gradient(135deg, #141820 0%, #1a1f2e 100%);
    border-radius: 12px;
    padding: 12px 24px; margin-bottom: 12px;
    border: 1px solid #1e2535;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}
.mkt-item { display: flex; flex-direction: column; align-items: center; }
.mkt-label { font-size: 10px; color: #8899aa; text-transform: uppercase; letter-spacing: 1px; }
.mkt-val   { font-size: 18px; font-weight: 700; color: #e8eaf0; }
.mkt-chg.up   { color: #4caf50; font-size: 12px; }
.mkt-chg.down { color: #f44336; font-size: 12px; }

/* Regime badge with glow */
.regime-bull { background:#1a3325; color:#4caf50; border:1px solid #4caf50;
               border-radius:8px; padding:4px 14px; font-size:12px; font-weight:700;
               box-shadow:0 0 8px rgba(76,175,80,0.3); letter-spacing:0.5px; }
.regime-bear { background:#2d1b1b; color:#f44336; border:1px solid #f44336;
               border-radius:8px; padding:4px 14px; font-size:12px; font-weight:700;
               box-shadow:0 0 8px rgba(244,67,54,0.3); letter-spacing:0.5px; }
.regime-neut { background:#1e2030; color:#ffcc02; border:1px solid #ffcc02;
               border-radius:8px; padding:4px 14px; font-size:12px; font-weight:700;
               box-shadow:0 0 8px rgba(255,204,2,0.2); letter-spacing:0.5px; }

/* Preset buttons row */
.preset-row { display: flex; gap: 8px; flex-wrap: wrap; margin: 8px 0; }

/* Filter badge with gradient */
.filter-badge {
    background: linear-gradient(90deg, #e65100, #ff8f00);
    color: white; border-radius: 16px;
    padding: 3px 14px; font-size: 11px; font-weight: 700;
    display: inline-block; margin-left: 6px;
    box-shadow: 0 1px 4px rgba(230,81,0,0.4);
}

/* Watchlist chip */
.wl-chip {
    display: inline-block; background: #1e2540; color: #90caf9;
    border: 1px solid #3d5afe; border-radius: 20px;
    padding: 3px 12px; margin: 3px; font-size: 13px;
}

/* Sidebar section headers */
.sidebar-section {
    font-size: 11px; color: #8899bb; text-transform: uppercase;
    letter-spacing: 1.5px; margin: 12px 0 4px 0; font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  CONSTANTS  (must be at top level — used in sidebar AND main area)
# ══════════════════════════════════════════════════════════════
PRESETS = {
    "🐂 Bull Setup"     : dict(bullish_d=True, rsi_bull=True, trending=True),
    "📉 Oversold Bounce": dict(rsi_bear=True, oversold=True),
    "💥 Vol Spurt"      : dict(vol_spurt=True),
    "🔭 MTF Aligned"    : dict(bullish_d=True),
    "🔴 Sell Setups"    : dict(bearish_d=True, rsi_bear=True),
    "🗜️ Squeeze"        : dict(squeeze=True),
}

# Compact view: 8 core columns only
COMPACT_COLS = ["Rank", "Ticker", "Price", "Chg%",
                "Trade", "D_Trend", "ML_Signal",
                "MomScore", "RS_Score", "RSI_Zone"]

# Column glossary for tooltips
COL_GLOSSARY = {
    "Rank"      : "Overall rank 1=best. Composite: MomScore(35%) + RS_Score(40%) + ML_Prob%(25%).",
    "D_Trend"   : "Daily trend. Needs EMA20>EMA50 AND SuperTrend bullish to show Bullish.",
    "MTF_Score" : "Multi-timeframe alignment 0-3. Score 3 = Daily+Weekly+Monthly all agree.",
    "RSI_Zone"  : "Bull=RSI>60, Bear=RSI<40, Neutral=40-60.",
    "MomScore"  : "Composite 0-100. RSI(30%)+MACD(25%)+Kumo(20%)+EMA20 dist(15%)+RSI_SMA(10%).",
    "VolStatus" : "Squeeze=BB inside Keltner (coiling energy). Expansion=ATR above 70th pctile.",
    "VolSpurt"  : "🟢Spurt=>2× avg vol on up day. 🔴Dump=>2× on down day. ⚡High=1.5-2×.",
    "RS_Score"  : "Relative Strength vs Nifty50. >60=outperforming. Weighted 1M(50%)+3M(30%)+6M(20%).",
    "RS_Trend"  : "Direction of RS line over 20 days. Rising=accelerating vs Nifty.",
    "SMI"       : "Stochastic Momentum Index -100 to +100. Better than Stoch: double-EMA smoothed.",
    "SMI_Zone"  : "OB=Overbought(>40), Bull=0-40, Bear=-40 to 0, OS=Oversold(<-40).",
    "ADX_Str"   : "Trend strength. Strong=ADX>40, Moderate=25-40, Weak<25.",
    "ATR%"      : "Volatility as % of price. Multiply by 2 to get ATR stop distance.",
    "ML_Prob%"  : "Probability price is higher in 5 days (XGBoost+RF+LR ensemble).",
    "ML_Acc%"   : "Walk-forward historical accuracy % for this ticker's ML model.",
    "ML_Signal" : "Buy=ML_Prob>57%, Sell=<45%, Hold=in between.",
    "⚠️Conflict" : "Trend direction disagrees with ML signal — proceed with caution.",
    "52wH%"     : "% from 52-week high. -5% = stock is 5% below its peak.",
    "BullEng"   : "Bullish engulfing candle with volume confirmation.",
    "Hammer"    : "Hammer: small body top, lower wick ≥ 2× body — bullish reversal.",
}

# ══════════════════════════════════════════════════════════════
#  SESSION STATE DEFAULTS
# ══════════════════════════════════════════════════════════════
if "watchlist" not in st.session_state:
    st.session_state["watchlist"] = []
if "preset" not in st.session_state:
    st.session_state["preset"] = None
if "compact_mode" not in st.session_state:
    st.session_state["compact_mode"] = False


# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:

    # ── Top controls ─────────────────────────────────────────
    c_ref, c_ml = st.columns(2)
    with c_ref:
        if st.button("🔄 Refresh", use_container_width=True, type="primary"):
            for k in ["batch_dl_done", "delta_done", "weekly_done",
                      "monthly_done", "weekly_df", "monthly_df"]:
                st.session_state.pop(k, None)
            st.cache_data.clear()
            st.rerun()
    with c_ml:
        run_ml = st.toggle("🤖 ML", value=False,
                           help="XGBoost+RF ensemble. ~2-4 sec/stock.")

    if run_ml:
        st.success("ML ON")
    else:
        st.caption("ML OFF — fast mode")

    # ── Universe ──────────────────────────────────────────────
    st.divider()
    default_idx = next(
        (i for i, k in enumerate(UNIVERSE_MAP) if "Nifty Indices Only" in k), 0
    )
    universe_choice = st.selectbox("🌐 Universe", list(UNIVERSE_MAP.keys()),
                                   index=default_idx, label_visibility="visible")
    tickers = UNIVERSE_MAP[universe_choice]
    st.caption(f"{len(tickers)} tickers · {datetime.now().strftime('%d %b %Y %H:%M')}")

    # ── Cache info ─────────────────────────────────────────────
    conn_s = get_conn()
    sc = cache_stats(conn_s)
    conn_s.close()
    st.caption(f"💾 {sc['size_mb']} MB · {sc['tickers']} tkrs · {sc['rows']:,} rows")
    if st.button("🗑️ Clear ML Cache", use_container_width=True):
        import shutil
        ml_cache = Path(__file__).parent / "data" / "ml_cache"
        if ml_cache.exists():
            shutil.rmtree(ml_cache)
        st.cache_data.clear()
        st.rerun()

    # ── Quick-Scan Presets ────────────────────────────────────
    st.divider()
    st.markdown('<div class="sidebar-section">Quick Scans</div>', unsafe_allow_html=True)

    # Show presets as 2-column button grid
    p_names = list(PRESETS.keys())
    for i in range(0, len(p_names), 2):
        pc1, pc2 = st.columns(2)
        for j, col in enumerate([pc1, pc2]):
            idx = i + j
            if idx < len(p_names):
                name = p_names[idx]
                is_active = st.session_state["preset"] == name
                btn_type = "primary" if is_active else "secondary"
                if col.button(name, key=f"preset_{idx}",
                              use_container_width=True, type=btn_type):
                    # Toggle: click same preset again to clear
                    if st.session_state["preset"] == name:
                        st.session_state["preset"] = None
                    else:
                        st.session_state["preset"] = name
                    st.rerun()

    if st.session_state["preset"]:
        if st.button("✖ Clear Preset", use_container_width=True):
            st.session_state["preset"] = None
            st.rerun()

    # ── Signal Filters ────────────────────────────────────────
    st.divider()

    # Count active filters for badge
    def _n_active(**kwargs):
        return sum(1 for v in kwargs.values() if v)

    with st.expander("📈 Trend & Regime"):
        f_bullish_d   = st.checkbox("Bullish Trend")
        f_bearish_d   = st.checkbox("Bearish Trend")
        f_trending    = st.checkbox("Trending (ADX>25)")
        f_ranging     = st.checkbox("Ranging")
        f_above_ema200= st.checkbox("Above EMA 200")

    with st.expander("📊 RSI / SMI"):
        f_rsi_bull  = st.checkbox("RSI Zone Bull (>60)")
        f_rsi_bear  = st.checkbox("RSI Zone Bear (<40)")
        f_oversold  = st.checkbox("RSI Oversold (<30)")
        f_overbought= st.checkbox("RSI Overbought (>70)")
        rsi_min, rsi_max = st.slider("RSI Range", 0, 100, (0, 100))
        f_smi_bull  = st.checkbox("SMI Bullish (>0)")
        f_smi_os    = st.checkbox("SMI Oversold (<-40)")
        f_smi_cross = st.checkbox("SMI Bullish Cross")

    with st.expander("⚡ Volume & Momentum"):
        f_high_mom  = st.checkbox("High Momentum (>60)")
        f_low_mom   = st.checkbox("Low Momentum (<30)")
        f_vol_break = st.checkbox("Vol Expansion")
        f_squeeze   = st.checkbox("Volatility Squeeze")
        f_macd_bull = st.checkbox("MACD Bull Cross")
        f_vol_spurt = st.checkbox("Volume Spurt (>2×)")

    with st.expander("🕯️ Patterns"):
        f_engulf_b  = st.checkbox("Bullish Engulfing")
        f_engulf_br = st.checkbox("Bearish Engulfing")
        f_hammer    = st.checkbox("Hammer")

    with st.expander("🤖 ML"):
        f_ml_buy   = st.checkbox("ML Buy")
        f_ml_sell  = st.checkbox("ML Sell")
        f_conflict = st.checkbox("Conflicts Only")

    # Count active manual filters
    _manual_filters = [f_bullish_d, f_bearish_d, f_trending, f_ranging,
                       f_above_ema200, f_rsi_bull, f_rsi_bear, f_oversold,
                       f_overbought, f_smi_bull, f_smi_os, f_smi_cross,
                       f_high_mom, f_low_mom, f_vol_break, f_squeeze,
                       f_macd_bull, f_vol_spurt, f_engulf_b, f_engulf_br,
                       f_hammer, f_ml_buy, f_ml_sell, f_conflict,
                       (rsi_min > 0 or rsi_max < 100)]
    n_active = sum(1 for v in _manual_filters if v)
    active_preset = st.session_state["preset"]
    if n_active or active_preset:
        badge_txt = f"{n_active} filter{'s' if n_active != 1 else ''} active"
        if active_preset:
            badge_txt = f"Preset: {active_preset}" + (f" + {n_active}" if n_active else "")
        st.markdown(f'<span class="filter-badge">🔍 {badge_txt}</span>',
                    unsafe_allow_html=True)
        if st.button("✖ Clear All Filters", use_container_width=True):
            st.session_state["preset"] = None
            st.rerun()

    # ── Sort ─────────────────────────────────────────────────
    st.divider()
    sort_col = st.selectbox("🔃 Sort By",
        ["Rank","MTF_Score","RS_Score","MomScore","SMI","RSI","ADX",
         "Chg%","ATR%","ML_Prob%","52wH%"],
    )
    _rank_cols = {"Rank", "RS_Rank"}
    sort_asc = st.checkbox("Ascending ↑", value=(sort_col in _rank_cols))

    # ── Columns ───────────────────────────────────────────────
    st.divider()
    st.markdown('<div class="sidebar-section">Show Columns</div>', unsafe_allow_html=True)
    compact_mode = st.toggle("⚡ Compact View (8 cols)",
                             value=st.session_state["compact_mode"],
                             help="Show only the 8 most important columns. "
                                  "Uncheck to see full detail.")
    st.session_state["compact_mode"] = compact_mode
    if not compact_mode:
        cc1, cc2 = st.columns(2)
        show_patterns = cc1.checkbox("Patterns",  value=True)
        show_ml       = cc2.checkbox("ML Cols",   value=True)
        show_context  = cc1.checkbox("52wH/EMA",  value=True)
        show_rs       = cc2.checkbox("RS Nifty",  value=True)
        show_smi      = cc1.checkbox("SMI",       value=True)
        show_mtf_full = cc2.checkbox("MTF Detail",value=False)
    else:
        # Compact: fixed column set, hide toggles
        show_patterns = show_ml = show_context = show_rs = show_smi = show_mtf_full = False


# ══════════════════════════════════════════════════════════════
#  BUILD FILTER DICT  (merge preset + manual)
# ══════════════════════════════════════════════════════════════
filter_dict = dict(
    bullish_d   = f_bullish_d,
    bearish_d   = f_bearish_d,
    rsi_bull    = f_rsi_bull,
    rsi_bear    = f_rsi_bear,
    oversold    = f_oversold,
    overbought  = f_overbought,
    engulf_bull = f_engulf_b,
    engulf_bear = f_engulf_br,
    hammer      = f_hammer,
    vol_expansion=f_vol_break,
    squeeze     = f_squeeze,
    high_mom    = f_high_mom,
    low_mom     = f_low_mom,
    trending    = f_trending,
    ranging     = f_ranging,
    ml_buy      = f_ml_buy,
    ml_sell     = f_ml_sell,
    above_ema200= f_above_ema200,
    macd_bull   = f_macd_bull,
    conflict    = f_conflict,
    rsi_min     = rsi_min,
    rsi_max     = rsi_max,
    vol_spurt   = f_vol_spurt,
    smi_bull    = f_smi_bull,
    smi_os      = f_smi_os,
    smi_cross   = f_smi_cross,
)

# Merge preset on top
if active_preset and active_preset in PRESETS:
    for k, v in PRESETS[active_preset].items():
        filter_dict[k] = filter_dict.get(k) or v


# ══════════════════════════════════════════════════════════════
#  DATA PIPELINE
# ══════════════════════════════════════════════════════════════
conn = get_conn()
_all_tickers    = list(dict.fromkeys(tickers + list(INDEX_TICKERS.values())))
_dl_done_key    = f"batch_dl_done_{universe_choice}"
_delta_done_key = f"delta_done_{universe_choice}"

if not st.session_state.get(_dl_done_key):
    dl_msg = st.empty()
    batch_download_missing(_all_tickers, conn,
        status_fn=lambda m: dl_msg.info(m) if m else dl_msg.empty())
    st.session_state[_dl_done_key] = True
    dl_msg.empty()

if not st.session_state.get(_delta_done_key):
    upd_msg = st.empty()
    delta_update_parallel(_all_tickers, conn,
        status_fn=lambda m: upd_msg.info(m) if m else upd_msg.empty())
    st.session_state[_delta_done_key] = True
    upd_msg.empty()



# ══════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS  (all defined before first call)
# ══════════════════════════════════════════════════════════════

def _sector_heatmap():
    """Compact sector RS bar — green = outperforming Nifty, red = lagging."""
    sector_map = {
        "Bank"   : "^NSEBANK",
        "IT"     : "^CNXIT",
        "Pharma" : "^CNXPHARMA",
        "Auto"   : "^CNXAUTO",
        "FMCG"   : "^CNXFMCG",
        "Metal"  : "^CNXMETAL",
        "Realty" : "^CNXREALTY",
    }
    nifty_df = load_ohlcv("^NSEI", conn)
    if nifty_df.empty or len(nifty_df) < 63:
        return

    rows = []
    for name, sym in sector_map.items():
        df_s = load_ohlcv(sym, conn)
        if df_s.empty or len(df_s) < 63:
            continue
        c = df_s["Close"]
        n = nifty_df["Close"]
        common = c.index.intersection(n.index)
        c, n = c.loc[common], n.loc[common]
        if len(c) < 63:
            continue
        rs_1m  = (c.iloc[-1]/c.iloc[-21]) / max(n.iloc[-1]/n.iloc[-21], 0.001)
        rs_3m  = (c.iloc[-1]/c.iloc[-63]) / max(n.iloc[-1]/n.iloc[-63], 0.001)
        rs     = rs_1m * 0.6 + rs_3m * 0.4
        chg_1d = (c.iloc[-1]/c.iloc[-2] - 1) * 100 if len(c) >= 2 else 0
        rows.append({"Sector": name, "RS": rs, "1D%": chg_1d})

    if not rows:
        return

    rows.sort(key=lambda x: x["RS"], reverse=True)
    st.markdown("**\U0001f504 Sector Rotation** \u2014 RS vs Nifty50 (green = outperforming)")
    cols = st.columns(len(rows))
    for col, row in zip(cols, rows):
        rs  = row["RS"]
        chg = row["1D%"]
        if rs > 1.05:
            bg, fg = "#1a3325", "#4caf50"
        elif rs > 1.0:
            bg, fg = "#1e2820", "#81c784"
        elif rs > 0.95:
            bg, fg = "#2a2510", "#ffcc02"
        else:
            bg, fg = "#2d1b1b", "#f44336"
        arrow   = "\u25b2" if chg >= 0 else "\u25bc"
        chg_col = "#4caf50" if chg >= 0 else "#f44336"
        col.markdown(
            f'<div style="background:{bg};border-radius:8px;padding:8px 4px;'
            f'text-align:center;border:1px solid {fg}22;">'
            f'<div style="color:{fg};font-weight:700;font-size:13px">{row["Sector"]}</div>'
            f'<div style="color:{fg};font-size:11px">RS {rs:.2f}</div>'
            f'<div style="color:{chg_col};font-size:11px">{arrow}{abs(chg):.1f}%</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    st.markdown("<br>", unsafe_allow_html=True)


def _ticker_detail_card(df, key_suffix):
    """Pick any ticker from dropdown and render full signal + trade level card."""
    if df.empty or "Ticker" not in df.columns:
        return

    selected = st.selectbox(
        "Inspect ticker",
        ["— pick a ticker —"] + df["Ticker"].tolist(),
        key=f"detail_{key_suffix}",
        label_visibility="collapsed",
    )
    if selected == "— pick a ticker —":
        return

    row = df[df["Ticker"] == selected].iloc[0]

    price   = float(row.get("Price", 0))
    atr_pct_val = row.get("ATR%", 1)
    if isinstance(atr_pct_val, str):
        try:
            atr_pct_val = float(atr_pct_val.replace("%",""))
        except Exception:
            atr_pct_val = 1.0
    atr_pct = float(atr_pct_val) / 100
    atr_abs = price * atr_pct
    stop    = price - 2 * atr_abs
    target  = price + 3 * atr_abs

    trend  = str(row.get("D_Trend", "\u2014"))
    rsi    = row.get("RSI", "\u2014")
    mom    = row.get("MomScore", "\u2014")
    rs     = row.get("RS_Score", "\u2014")
    rs_tr  = row.get("RS_Trend", "\u2014")
    smi    = row.get("SMI", "\u2014")
    smi_z  = row.get("SMI_Zone", "\u2014")
    ml_sig = row.get("ML_Signal", "\u2014")
    ml_prob= row.get("ML_Prob%", "\u2014")
    ml_why = row.get("ML_Reason", "\u2014")
    mtf    = row.get("MTF_Score", "\u2014")
    spurt  = row.get("VolSpurt", "\u2014")
    conflict = row.get("\u26a0\ufe0fConflict", "\u2014")

    def _is_num(v):
        return isinstance(v, (int, float))

    signals_bull = [
        "Bullish" in str(trend),
        float(rsi) > 60 if _is_num(rsi) else False,
        float(mom) > 60 if _is_num(mom) else False,
        float(rs)  > 60 if _is_num(rs)  else False,
        str(rs_tr) == "Rising",
        float(smi) > 0  if _is_num(smi) else False,
        mtf in (2, 3)   if _is_num(mtf) else False,
        "Buy" in str(ml_sig),
    ]
    conviction = sum(signals_bull)
    conv_color = "#4caf50" if conviction >= 6 else "#ffcc02" if conviction >= 4 else "#f44336"
    conv_label = "High" if conviction >= 6 else "Medium" if conviction >= 4 else "Low"

    with st.container():
        st.markdown(f"### {selected} \u2014 Signal Detail")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Price",    f"\u20b9{price:,.2f}")
        c1.metric("D_Trend",  trend)
        c1.metric("RSI",      f"{rsi:.1f}" if _is_num(rsi) else str(rsi))
        c2.metric("MomScore", f"{mom:.1f}" if _is_num(mom) else str(mom))
        c2.metric("RS_Score", f"{rs:.1f}"  if _is_num(rs)  else str(rs))
        c2.metric("RS_Trend", str(rs_tr))
        c3.metric("SMI",      f"{smi:.1f}" if _is_num(smi) else str(smi))
        c3.metric("SMI_Zone", str(smi_z))
        c3.metric("MTF_Score",str(mtf))
        c4.metric("ML Signal",str(ml_sig))
        c4.metric("ML Prob%", f"{ml_prob:.1f}%" if _is_num(ml_prob) else str(ml_prob))
        c4.metric("ML Reason",str(ml_why)[:30] if ml_why else "\u2014")

        st.divider()
        t1, t2, t3, t4, t5 = st.columns(5)
        t1.metric("Entry Zone",     f"\u20b9{price:,.2f}")
        t2.metric("ATR Stop (2\u00d7)",  f"\u20b9{stop:,.2f}",
                  delta=f"-{2*atr_pct*100:.1f}%", delta_color="inverse")
        t3.metric("Target (3\u00d7ATR)", f"\u20b9{target:,.2f}",
                  delta=f"+{3*atr_pct*100:.1f}%")
        t4.metric("R:R Ratio", "1 : 1.5")
        t5.markdown(
            f'<div style="text-align:center;padding-top:8px;">'
            f'<div style="font-size:11px;color:#8899aa">Conviction</div>'
            f'<div style="font-size:22px;font-weight:700;color:{conv_color}">'
            f'{conv_label} ({conviction}/8)</div></div>',
            unsafe_allow_html=True,
        )
        if str(conflict) == "\u26a0\ufe0f":
            st.warning("\u26a0\ufe0f Conflict \u2014 Trend and ML disagree. Wait for alignment.", icon="\u26a0\ufe0f")
        if str(spurt) in ("\U0001f7e2 Spurt", "\u26a1 High"):
            st.success(f"Volume confirmation: {spurt} \u2014 move has institutional backing.")
        st.divider()


def _column_glossary():
    with st.expander("\U0001f4d6 Column Glossary"):
        g1, g2 = st.columns(2)
        items = list(COL_GLOSSARY.items())
        mid   = len(items) // 2
        for col, subset in [(g1, items[:mid]), (g2, items[mid:])]:
            for name, desc in subset:
                col.markdown(f"**`{name}`** \u2014 {desc}")

# ══════════════════════════════════════════════════════════════
#  MARKET CONTEXT BAR  (Nifty + VIX always visible)
# ══════════════════════════════════════════════════════════════
def _market_context_bar():
    """Render sticky top bar with Nifty/VIX/sector snapshot."""
    idx_data = {}
    for name, sym in INDEX_TICKERS.items():
        df_i = load_ohlcv(sym, conn)
        if not df_i.empty and len(df_i) >= 2:
            price = float(df_i["Close"].iloc[-1])
            chg   = (price / float(df_i["Close"].iloc[-2]) - 1) * 100
            # Nifty regime: price vs EMA200
            ema200 = df_i["Close"].ewm(span=200, adjust=False).mean().iloc[-1]
            idx_data[name] = {"price": price, "chg": chg,
                              "above_ema200": price > ema200}

    if not idx_data:
        return "Neutral"

    # Determine Nifty regime
    nifty = idx_data.get("Nifty 50", {})
    vix   = idx_data.get("India VIX", {})
    vix_val = vix.get("price", 15)
    nifty_above = nifty.get("above_ema200", True)
    nifty_chg   = nifty.get("chg", 0)

    if nifty_above and nifty_chg > -1:
        regime_html = '<span class="regime-bull">🟢 BULL</span>'
        mkt_regime  = "Bull"
    elif not nifty_above or nifty_chg < -1.5:
        regime_html = '<span class="regime-bear">🔴 BEAR</span>'
        mkt_regime  = "Bear"
    else:
        regime_html = '<span class="regime-neut">🟡 MIXED</span>'
        mkt_regime  = "Neutral"

    # VIX badge
    vix_color = "#f44336" if vix_val > 20 else "#ffcc02" if vix_val > 15 else "#4caf50"
    vix_html = f'<span style="color:{vix_color};font-weight:700">VIX {vix_val:.1f}</span>'

    # Render as columns for cleaner layout
    n_cols = min(len(idx_data) + 1, 10)
    cols = st.columns(n_cols)

    # First col: regime badge
    with cols[0]:
        st.markdown(f"{regime_html}<br>{vix_html}", unsafe_allow_html=True)

    for i, (name, data) in enumerate(list(idx_data.items())[:n_cols-1]):
        price = data["price"]
        chg   = data["chg"]
        chg_color = "#4caf50" if chg >= 0 else "#f44336"
        arrow = "▲" if chg >= 0 else "▼"
        cols[i+1].metric(
            name,
            f"{price:,.0f}" if price > 100 else f"{price:.2f}",
            f"{arrow} {abs(chg):.2f}%",
            delta_color="normal" if chg >= 0 else "inverse",
        )

    return mkt_regime

mkt_regime = _market_context_bar()

# ── Sector Rotation Heatmap ───────────────────────────────────
_sector_heatmap()
st.divider()


# ══════════════════════════════════════════════════════════════
#  COLUMN SETS
# ══════════════════════════════════════════════════════════════
BASE_COLS    = ["Rank","Ticker","Price","Chg%",
                # Decision signals
                "Trade","D_Trend","ML_Signal",
                # Conviction scores
                "MomScore","RS_Score",
                # Market context
                "RSI_Zone","Regime","Mkt_Struct","ADX_Str","VolStatus","VolSpurt",
                # Technical detail
                "RSI","ADX","ATR%","MACD",">EMA20",
                # Volume
                "Vol"]
SMI_COLS     = ["SMI","SMI_Zone"]
CONTEXT_COLS = [">EMA200","52wH%"]
RS_COLS      = ["RS_Rank","RS_Trend","RS_1M","RS_3M"]
PATTERN_COLS = ["Pattern"]
ML_COLS      = ["ML_Signal","ML_Prob%","ML_Acc%","ML_Reason","⚠️Conflict"]

def _tf_cols(prefix, full=False):
    compact = [f"{prefix}_D_Trend", f"{prefix}_RSI_Zone",
               f"{prefix}_ADX_Str", f"{prefix}_MomScore",
               f"{prefix}_VolStatus", f"{prefix}_ML_Signal"]
    extra   = [f"{prefix}_RSI", f"{prefix}_ADX", f"{prefix}_ATR%",
               f"{prefix}_ML_Prob%", f"{prefix}_SMI"]
    return compact + extra if full else compact

def _build_display_cols(df_cols, prefix=None, full_mtf=False):
    # Compact mode: fixed 8-col view
    if st.session_state.get("compact_mode"):
        base = COMPACT_COLS.copy()
        if "MTF_Score" in df_cols and "D_Trend" in base:
            base.insert(base.index("D_Trend") + 1, "MTF_Score")
        if prefix:
            base += _tf_cols(prefix, full=False)
        return [c for c in base if c in df_cols]

    cols = BASE_COLS.copy()
    if "MTF_Score" in df_cols:
        cols.insert(cols.index("D_Trend") + 1, "MTF_Score")
    if show_smi:       cols += SMI_COLS
    if prefix:         cols += _tf_cols(prefix, full=full_mtf)
    if show_context:   cols += CONTEXT_COLS
    if show_rs:        cols += RS_COLS
    if show_patterns:  cols += PATTERN_COLS
    if show_ml and run_ml: cols += ML_COLS
    # Deduplicate (ML_Signal is in both BASE and ML_COLS)
    seen = set()
    cols = [c for c in cols if not (c in seen or seen.add(c))]
    return [c for c in cols if c in df_cols]


# ══════════════════════════════════════════════════════════════
#  RENDER TABLE  (shared across all tabs)
# ══════════════════════════════════════════════════════════════
def render_table(df: pd.DataFrame, display_cols: list, key_suffix: str = "d"):
    if df.empty:
        st.info("No stocks match the current filters. Try relaxing them.")
        return

    # ── Stats bar (pill badges) ────────────────────────────────
    stats  = universe_stats(df)
    items = [
        ("Showing", f"{len(df)}/{stats.get('total', len(df))}", "#5c7cfa"),
        ("Bullish", stats.get("bullish", 0), "#4caf50"),
        ("Bearish", stats.get("bearish", 0), "#f44336"),
        ("Trending", stats.get("trending", 0), "#7c4dff"),
        ("RSI Bull", stats.get("rsi_bull", 0), "#26a69a"),
        ("RSI Bear", stats.get("rsi_bear", 0), "#ef5350"),
        ("ML Buy", stats.get("ml_buy", 0), "#00e676"),
        ("ML Sell", stats.get("ml_sell", 0), "#ff5252"),
    ]
    pills = "".join(
        f'<div class="stat-pill">'
        f'<span class="stat-label">{label}</span>'
        f'<span class="stat-val" style="color:{color}">{val}</span>'
        f'</div>'
        for label, val, color in items
    )
    st.markdown(f'<div class="stats-bar">{pills}</div>', unsafe_allow_html=True)

    # ── Market regime warning ────────────────────────────────
    if mkt_regime == "Bear" and not filter_dict.get("bearish_d"):
        st.warning("⚠️ **Market in Bear regime** — Nifty below EMA200. "
                   "Buy signals have lower reliability. Consider Sell Setups preset.", icon="⚠️")

    # ── Compact mode indicator ───────────────────────────────
    if st.session_state.get("compact_mode"):
        st.caption("⚡ **Compact View** — showing 8 core columns. "
                   "Toggle off in sidebar for full detail.")

    st.divider()

    # ── Pre-format numeric cols as strings ───────────────────
    display_df = df[display_cols].copy()
    def _fmt_num(series, fmt):
        return series.apply(
            lambda v: fmt.format(v) if pd.notna(v) and isinstance(v, (int, float)) else "—"
        )
    for col, fmt in [
        ("ML_Prob%", "{:.1f}%"), ("ML_Acc%",  "{:.1f}%"),
        ("RS_Score", "{:.1f}"),  ("RS_1M",    "{:.3f}"),
        ("RS_3M",    "{:.3f}"),  ("MomScore", "{:.1f}"),
        ("RSI",      "{:.1f}"),  ("ADX",      "{:.1f}"),
        ("ATR%",     "{:.2f}%"), ("52wH%",    "{:.1f}%"),
        ("SMI",      "{:.1f}"),
        ("SMI_Signal","{:.1f}"),
    ]:
        if col in display_df.columns:
            display_df[col] = _fmt_num(display_df[col], fmt)

    # ── Watchlist: highlight pinned tickers ──────────────────
    wl = st.session_state.get("watchlist", [])
    if wl and "Ticker" in display_df.columns:
        def _highlight_wl(row):
            if row.get("Ticker") in wl:
                return ["background-color: #1a2540"] * len(row)
            return [""] * len(row)
        styler = display_df.style.apply(_highlight_wl, axis=1)
    else:
        styler = display_df.style

    styled = apply_table_style(styler, run_ml=run_ml and show_ml, cols=display_cols)
    height = min(80 + len(df) * TABLE_ROW_HEIGHT, TABLE_MAX_HEIGHT)
    st.dataframe(styled, width="stretch", height=height)

    # ── Actions row ──────────────────────────────────────────
    act1, act2, act3 = st.columns([2, 2, 4])
    with act1:
        csv = df[display_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            f"⬇️ Download CSV ({len(df)} rows)",
            data=csv,
            file_name=f"mtf_{key_suffix}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            key=f"dl_{key_suffix}",
        )
    with act2:
        if "Ticker" in df.columns:
            if st.button(f"📌 Pin top {min(5,len(df))} to Watchlist",
                         key=f"pin_{key_suffix}"):
                top5 = df["Ticker"].head(5).tolist()
                existing = st.session_state.get("watchlist", [])
                new_wl = list(dict.fromkeys(existing + top5))[:20]
                st.session_state["watchlist"] = new_wl
                st.rerun()

    # ── Ticker detail card ───────────────────────────────────
    st.divider()
    _ticker_detail_card(df, key_suffix)

    # ── Column glossary ──────────────────────────────────────
    _column_glossary()


# ══════════════════════════════════════════════════════════════
#  4 TABS  (Daily | Weekly | Monthly | Watchlist)
# ══════════════════════════════════════════════════════════════
tab_d, tab_w, tab_m, tab_wl = st.tabs([
    "📅 Daily",
    "📆 Weekly",
    "🗓️ Monthly",
    f"📌 Watchlist ({len(st.session_state.get('watchlist',[]))})",
])


# ─────────────────────────────────────────────────────────────
#  TAB 1 — DAILY
# ─────────────────────────────────────────────────────────────
with tab_d:
    st.subheader(f"📅 Daily Signals — {universe_choice}")
    if active_preset:
        st.info(f"⚡ Preset active: **{active_preset}** — "
                f"click preset again to toggle off")

    prog_d   = st.progress(0, text="Loading daily signals …")
    daily_df = build_universe_df(tickers, conn, run_ml=run_ml, progress_bar=prog_d)
    prog_d.empty()

    if daily_df.empty:
        st.error("No daily data. Click 🔄 Refresh.")
    else:
        filtered_d = apply_filters(daily_df, filter_dict)
        # MTF Aligned preset: requires MTF_Score >= 2
        if active_preset == "🔭 MTF Aligned" and "MTF_Score" in filtered_d.columns:
            filtered_d = filtered_d[filtered_d["MTF_Score"] >= 2]
        filtered_d = sort_df(filtered_d, sort_col, sort_asc)
        dcols = _build_display_cols(filtered_d.columns.tolist())
        render_table(filtered_d, dcols, key_suffix="daily")


# ─────────────────────────────────────────────────────────────
#  TAB 2 — WEEKLY
# ─────────────────────────────────────────────────────────────
with tab_w:
    st.subheader(f"📆 Weekly Signals — {universe_choice}")
    st.caption("Weekly bars resampled from daily cache — no extra download.")

    col_btn, col_info = st.columns([2, 5])
    with col_btn:
        run_weekly = st.button("▶ Run Weekly", use_container_width=True, type="primary")
    with col_info:
        if not st.session_state.get("weekly_done"):
            st.info("Click **▶ Run Weekly** to compute weekly indicators.")

    if run_weekly or st.session_state.get("weekly_done"):
        if "weekly_df" not in st.session_state or run_weekly:
            prog_w = st.progress(0, text="Computing weekly indicators …")
            weekly_df = build_universe_tf(
                tickers, conn, tf="W", run_ml=run_ml, progress_bar=prog_w)
            prog_w.empty()
            st.session_state["weekly_df"]   = weekly_df
            st.session_state["weekly_done"] = True
        else:
            weekly_df = st.session_state["weekly_df"]

        if weekly_df.empty:
            st.error("No weekly data computed.")
        else:
            merged_w   = merge_mtf(daily_df, weekly_df=weekly_df) if not daily_df.empty else weekly_df
            filtered_w = apply_filters(merged_w, filter_dict)
            filtered_w = sort_df(filtered_w, sort_col, sort_asc)
            wcols = _build_display_cols(filtered_w.columns.tolist(),
                                        prefix="W", full_mtf=show_mtf_full)
            render_table(filtered_w, wcols, key_suffix="weekly")


# ─────────────────────────────────────────────────────────────
#  TAB 3 — MONTHLY
# ─────────────────────────────────────────────────────────────
with tab_m:
    st.subheader(f"🗓️ Monthly Signals — {universe_choice}")
    st.caption("Monthly bars resampled from cache. Needs 24+ months per ticker.")

    col_btn, col_info = st.columns([2, 5])
    with col_btn:
        run_monthly = st.button("▶ Run Monthly", use_container_width=True, type="primary")
    with col_info:
        if not st.session_state.get("monthly_done"):
            st.info("Click **▶ Run Monthly** to compute monthly indicators.")

    if run_monthly or st.session_state.get("monthly_done"):
        if "monthly_df" not in st.session_state or run_monthly:
            prog_m = st.progress(0, text="Computing monthly indicators …")
            monthly_df = build_universe_tf(
                tickers, conn, tf="ME", run_ml=run_ml, progress_bar=prog_m)
            prog_m.empty()
            st.session_state["monthly_df"]   = monthly_df
            st.session_state["monthly_done"] = True
        else:
            monthly_df = st.session_state["monthly_df"]

        if monthly_df.empty:
            st.error("No monthly data computed.")
        else:
            merged_m   = merge_mtf(daily_df, monthly_df=monthly_df) if not daily_df.empty else monthly_df
            filtered_m = apply_filters(merged_m, filter_dict)
            filtered_m = sort_df(filtered_m, sort_col, sort_asc)
            mcols = _build_display_cols(filtered_m.columns.tolist(),
                                        prefix="M", full_mtf=show_mtf_full)
            render_table(filtered_m, mcols, key_suffix="monthly")


# ─────────────────────────────────────────────────────────────
#  TAB 4 — WATCHLIST
# ─────────────────────────────────────────────────────────────
with tab_wl:
    st.subheader("📌 Watchlist")
    wl = st.session_state.get("watchlist", [])

    if not wl:
        st.info("No tickers pinned yet. Use **📌 Pin top 5** button below any table, "
                "or add manually below.")
    else:
        # Show pinned chips
        chips_html = "".join(f'<span class="wl-chip">{t}</span>' for t in wl)
        st.markdown(chips_html, unsafe_allow_html=True)
        st.caption(f"{len(wl)} tickers pinned")

        # Action buttons
        wl_c1, wl_c2, wl_c3 = st.columns([2, 2, 4])
        with wl_c1:
            if st.button("🗑️ Clear Watchlist"):
                st.session_state["watchlist"] = []
                st.rerun()
        with wl_c2:
            remove_ticker = st.selectbox("Remove ticker", ["—"] + wl,
                                         key="wl_remove", label_visibility="collapsed")
            if remove_ticker != "—":
                wl2 = [t for t in wl if t != remove_ticker]
                st.session_state["watchlist"] = wl2
                st.rerun()

        st.divider()

        # Filter daily_df to watchlist and show full detail
        if not daily_df.empty:
            wl_tickers_ns = [t + ".NS" if not t.startswith("^") else t for t in wl]
            wl_tickers_plain = wl  # plain names like "RELIANCE"
            wl_df = daily_df[daily_df["Ticker"].isin(wl + wl_tickers_ns + wl_tickers_plain)]

            if wl_df.empty:
                st.warning("Pinned tickers not found in current universe. "
                           "Switch to a larger universe or check ticker names.")
            else:
                wl_df = sort_df(wl_df, sort_col, sort_asc)
                wl_cols = _build_display_cols(wl_df.columns.tolist())
                render_table(wl_df, wl_cols, key_suffix="watchlist")

    # Manual add
    st.divider()
    st.markdown("**➕ Add ticker manually**")
    add_col1, add_col2 = st.columns([3, 1])
    with add_col1:
        new_ticker = st.text_input("Ticker (e.g. RELIANCE, TCS)",
                                   key="wl_add_input", label_visibility="collapsed",
                                   placeholder="Enter ticker…")
    with add_col2:
        if st.button("Add", key="wl_add_btn") and new_ticker.strip():
            t = new_ticker.strip().upper()
            if t not in wl:
                st.session_state["watchlist"] = wl + [t]
                st.rerun()


# ─────────────────────────────────────────────────────────────
#  FULL MTF ALIGNMENT  (when all 3 timeframes loaded)
# ─────────────────────────────────────────────────────────────
w_done = st.session_state.get("weekly_done")
m_done = st.session_state.get("monthly_done")

if w_done and m_done and not daily_df.empty:
    st.divider()
    with st.expander("🔭 Full MTF Alignment View (Daily + Weekly + Monthly)", expanded=False):
        st.caption("**MTF_Score 3** = All timeframes aligned — strongest signal. "
                   "**MTF_Score 1** = Daily only — lower conviction.")
        full_mtf = merge_mtf(
            daily_df,
            weekly_df =st.session_state.get("weekly_df"),
            monthly_df=st.session_state.get("monthly_df"),
        )
        full_mtf = sort_df(full_mtf, "MTF_Score", ascending=False)
        full_cols = ["Rank","MTF_Score","Ticker","Price","Chg%",
                     "Trade","D_Trend","W_D_Trend","M_D_Trend",
                     "ML_Signal",
                     "MomScore","W_MomScore","M_MomScore",
                     "RS_Score",
                     "RSI_Zone","W_RSI_Zone","M_RSI_Zone",
                     "SMI"]
        full_cols = [c for c in full_cols if c in full_mtf.columns]
        render_table(full_mtf, full_cols, key_suffix="full_mtf")

conn.close()

st.divider()
st.caption(
    "⚠️ **Disclaimer:** Educational & research use only. Not financial advice. "
    "Data sourced from Yahoo Finance (NSE). Past ML accuracy ≠ future results."
)
