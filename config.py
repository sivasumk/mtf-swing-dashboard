"""
config.py — Central configuration for MTF Swing Dashboard
All constants, universe definitions, thresholds in one place.
"""

from pathlib import Path

# ══════════════════════════════════════════════════════════════
#  PATHS
# ══════════════════════════════════════════════════════════════
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
DB_PATH    = DATA_DIR / "market_cache.db"

# ══════════════════════════════════════════════════════════════
#  DATA SETTINGS
# ══════════════════════════════════════════════════════════════
# Tiered lookback by stock maturity (years of history to fetch)
LOOKBACK_LARGE_CAP  = 15   # Nifty 50
LOOKBACK_NEXT50     = 10   # Nifty Next 50
LOOKBACK_MIDCAP     = 7    # Midcap 100
LOOKBACK_INDEX      = 20   # Indices (^NSEI etc.)
LOOKBACK_DEFAULT    = 10   # Fallback

BATCH_SIZE   = 20          # Tickers per yfinance batch call
MAX_WORKERS  = 8           # Parallel download threads

# ══════════════════════════════════════════════════════════════
#  ML SETTINGS
# ══════════════════════════════════════════════════════════════
ML_FORWARD_DAYS     = 5    # Predict N-day forward direction
ML_CACHE_TTL        = 86400  # 24 hours in seconds
ML_MIN_SAMPLES      = 200  # Minimum bars needed to train
ML_STRONG_BUY_PROB  = 0.57  # Buy signal threshold (normal regime)
ML_STRONG_SELL_PROB = 0.45  # Sell signal threshold (normal regime)
# Regime-adaptive: tighter thresholds when stock is below EMA200
ML_BEAR_BUY_PROB    = 0.63  # Harder to trigger Buy when below EMA200
ML_BEAR_SELL_PROB   = 0.48  # Easier to trigger Sell when below EMA200
ML_TIME_DECAY       = 0.0003  # Exponential decay rate for sample weighting
ML_TRAIN_YEARS      = 3     # Only use last N years for training (regime-aware)

# Walk-forward settings
WF_N_SPLITS     = 5        # Number of walk-forward folds (increased from 3)
WF_TEST_SIZE    = 0.15     # Each fold test size (15% of total)
WF_PURGE_GAP    = ML_FORWARD_DAYS  # Purge gap between train/test (prevents target leakage)

# ══════════════════════════════════════════════════════════════
#  INDICATOR THRESHOLDS
# ══════════════════════════════════════════════════════════════
# RSI Zones (your requirement: >60 bull, <40 bear)
RSI_BULL_THRESHOLD  = 60
RSI_BEAR_THRESHOLD  = 40
RSI_OVERSOLD        = 30
RSI_OVERBOUGHT      = 70

# ADX
ADX_TREND_THRESHOLD = 25   # Above = Trending, Below = Ranging
ADX_STRONG_TREND    = 40   # Strong trend

# ATR expansion threshold (vs 20-day mean)
ATR_EXPANSION_MULT  = 1.3

# SuperTrend parameters
ST_PERIOD       = 10
ST_MULTIPLIER   = 3.0

# EMA periods
EMA_FAST        = 20
EMA_SLOW        = 50
EMA_TREND       = 200      # Long-term trend filter

# Bollinger Band
BB_PERIOD       = 20
BB_STD          = 2.0

# Keltner Channel (for squeeze detection)
KC_PERIOD       = 20
KC_ATR_PERIOD   = 10

# MACD
MACD_FAST       = 12
MACD_SLOW       = 26
MACD_SIGNAL     = 9

# ══════════════════════════════════════════════════════════════
#  SIGNAL LOGIC
# ══════════════════════════════════════════════════════════════
# Momentum Score weights (must sum to 100)
MOM_RSI_WEIGHT    = 30
MOM_MACD_WEIGHT   = 25
MOM_KUMO_WEIGHT   = 20
MOM_EMA_WEIGHT    = 15
MOM_RSI_SMA_WEIGHT= 10

# ══════════════════════════════════════════════════════════════
#  UI SETTINGS
# ══════════════════════════════════════════════════════════════
PAGE_CFG = dict(
    page_title="MTF Swing Dashboard · Nifty",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

TABLE_ROW_HEIGHT = 35
TABLE_MAX_HEIGHT = 750

# Icon maps
TREND_ICONS   = {"Bullish": "🟢", "Bearish": "🔴", "Neutral": "🟡"}
REGIME_ICONS  = {"Trending": "📈", "Ranging": "↔️"}
SQUEEZE_ICONS = {"Squeeze": "🗜️", "Expansion": "💥", "Normal": "➖"}
RSI_ZONE_ICONS= {"Bull": "🐂", "Bear": "🐻", "Neutral": "➖"}

# ══════════════════════════════════════════════════════════════
#  TICKER UNIVERSES
# ══════════════════════════════════════════════════════════════
INDEX_TICKERS = {
    "Nifty 50"    : "^NSEI",
    "Nifty Bank"  : "^NSEBANK",
    "Nifty IT"    : "^CNXIT",
    "Nifty Pharma": "^CNXPHARMA",
    "Nifty Auto"  : "^CNXAUTO",
    "Nifty FMCG"  : "^CNXFMCG",
    "Nifty Metal" : "^CNXMETAL",
    "Nifty Realty": "^CNXREALTY",
    "India VIX"   : "^INDIAVIX",
}

NIFTY50 = [
    # From NSE official list (Image 3 cross-referenced)
    "RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS",
    "HINDUNILVR.NS","SBIN.NS","BHARTIARTL.NS","ITC.NS","KOTAKBANK.NS",
    "LT.NS","AXISBANK.NS","ASIANPAINT.NS","MARUTI.NS","TITAN.NS",
    "SUNPHARMA.NS","BAJFINANCE.NS","ULTRACEMCO.NS","WIPRO.NS","HCLTECH.NS",
    "POWERGRID.NS","NTPC.NS","NESTLEIND.NS","TECHM.NS","ONGC.NS",
    "TATAMOTORS.NS","ADANIENT.NS","TATASTEEL.NS","JSWSTEEL.NS","BAJAJFINSV.NS",
    "HINDALCO.NS","GRASIM.NS","INDUSINDBK.NS","DRREDDY.NS","CIPLA.NS",
    "DIVISLAB.NS","APOLLOHOSP.NS","TRENT.NS","COALINDIA.NS","BRITANNIA.NS",
    "EICHERMOT.NS","HEROMOTOCO.NS","BPCL.NS","SHREECEM.NS","M&M.NS",
    "ADANIPORTS.NS","BAJAJ-AUTO.NS","HDFCLIFE.NS","SBILIFE.NS","BEL.NS",
]

NIFTY_NEXT50 = [
    # Updated from Image 3 (Nifty Next 50 / Nifty 100 FnO eligible)
    "ABB.NS","ADANIGREEN.NS","ADANIPOWER.NS","AMBUJACEM.NS","DMART.NS",
    "BAJAJHLDNG.NS","BAJAJHFL.NS","BANKBARODA.NS","BOSCHLTD.NS",
    "CANBK.NS","CHOLAFIN.NS","CGPOWER.NS","DLF.NS","GODREJCP.NS",
    "GODREJPROP.NS","HAVELLS.NS","HAL.NS","HINDZINC.NS","HYUNDAI.NS",
    "ICICIGI.NS","ICICIPRULI.NS","INDUSTOWER.NS","IOC.NS","IRFC.NS",
    "NAUKRI.NS","JIOFIN.NS","LICI.NS","LODHA.NS","MARICO.NS",
    "MAZDOCK.NS","MOTHERSON.NS","MUTHOOTFIN.NS","SIEMENS.NS","SIEMENSENERGY.NS",
    "PERSISTENT.NS","PETRONET.NS","PIDILITIND.NS","PNB.NS","RECLTD.NS",
    "TATAPOWER.NS","TORNTPHARM.NS","TVSMOTOR.NS","UNITDSPR.NS","VBL.NS",
    "VEDL.NS","ZYDUSLIFE.NS","SOLARINDS.NS","INDHOTEL.NS","PFC.NS",
]

NIFTY_MIDCAP = [
    "ABCAPITAL.NS","ABFRL.NS","ACC.NS","AIAENG.NS","ALKEM.NS",
    "APLLTD.NS","ASTRAL.NS","ATUL.NS","AUBANK.NS","BALKRISIND.NS",
    "BATAINDIA.NS","BHARATFORG.NS","BHEL.NS","BLUESTARCO.NS","BSOFT.NS",
    "CANFINHOME.NS","CASTROLIND.NS","CEATLTD.NS","CESC.NS",
    "CROMPTON.NS","CUMMINSIND.NS","DEEPAKNTR.NS","EDELWEISS.NS","EMAMILTD.NS",
    "ESCORTS.NS","EXIDEIND.NS","FLUOROCHEM.NS","GMRINFRA.NS","GNFC.NS",
    "GRANULES.NS","GSPL.NS","HFCL.NS","HINDPETRO.NS","IDFCFIRSTB.NS",
    "IEX.NS","INDIANB.NS","INDIGO.NS","ISEC.NS","JKCEMENT.NS",
    "JSWENERGY.NS","KAJARIACER.NS","KANSAINER.NS","KARURVYSYA.NS","KEI.NS",
    "KPITTECH.NS","LAURUSLABS.NS","LICHSGFIN.NS","LINDEINDIA.NS","LTTS.NS",
    "LUXIND.NS","MANAPPURAM.NS","METROPOLIS.NS","MFSL.NS","MGL.NS",
    "MPHASIS.NS","MRF.NS","NATIONALUM.NS","NAVINFLUOR.NS",
    "NBCC.NS","NCC.NS","NLCINDIA.NS","NMDC.NS","OBEROIRLTY.NS",
    "OFSS.NS","OIL.NS","PFIZER.NS",
    "PHOENIXLTD.NS","PVRINOX.NS","RADICO.NS","RAMCOCEM.NS","RATNAMANI.NS",
    "RITES.NS","ROSSARI.NS","SJVN.NS","SKFINDIA.NS","SONACOMS.NS",
    "STLTECH.NS","SUNDARMFIN.NS","SUPREMEIND.NS","SYNGENE.NS","TANLA.NS",
    "TIINDIA.NS","TTKPRESTIG.NS","VGUARD.NS","VINATIORGA.NS","VSTIND.NS",
    "WABCOINDIA.NS","WELCORP.NS","ZENSARTECH.NS","CDSL.NS","DELHIVERY.NS",
    "BIKAJI.NS","CAMPUS.NS","NYKAA.NS","POLICYBZR.NS","PAYTM.NS",
]

# Lookback map by universe
TICKER_LOOKBACK = {}
for t in NIFTY50:
    TICKER_LOOKBACK[t] = LOOKBACK_LARGE_CAP
for t in NIFTY_NEXT50:
    TICKER_LOOKBACK[t] = LOOKBACK_NEXT50
for t in NIFTY_MIDCAP:
    TICKER_LOOKBACK[t] = LOOKBACK_MIDCAP
for t in INDEX_TICKERS.values():
    TICKER_LOOKBACK[t] = LOOKBACK_INDEX

UNIVERSE_MAP = {
    "📊 Nifty Indices Only"          : list(INDEX_TICKERS.values()),
    "🔵 Nifty 50"                    : NIFTY50,
    "🟡 Nifty Next 50"               : NIFTY_NEXT50,
    "🟠 Nifty Midcap 100"            : NIFTY_MIDCAP,
    "🔵🟡 Nifty 100"                 : NIFTY50 + NIFTY_NEXT50,
    "🌐 Full Universe (~200 stocks)" : NIFTY50 + NIFTY_NEXT50 + NIFTY_MIDCAP,
}
