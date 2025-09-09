# analisa_saham.py
"""
Analisa Teknikal Saham - Refactor Layout Baru
Layout:
- Bagian atas: input parameter
- Bagian bawah: hasil analisis lengkap
Fitur:
- Indikator: RSI, MACD, Bollinger, ATR, OBV, ADX, VWAP
- Scoring System (Composite Score)
- Support/Resistance
- Bandarmology (volume spikes, smart money patterns, volume profile)
- Breakout detection (basic)
- Chart candlestick + Bollinger + Volume
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from scipy import stats

st.set_page_config(layout="wide", page_title="Analisa Teknikal Saham (Refactor)")

# =========================
# Konstanta
# =========================
INDICATOR_WEIGHTS = {
    'rsi': 0.15,
    'macd_cross': 0.25,
    'macd_hist': 0.10,
    'bollinger': 0.15,
    'volume': 0.20,
    'obv': 0.10,
    'adx': 0.05
}

# =========================
# Data Fetch
# =========================
@st.cache_data(show_spinner=False)
def get_stock_data_yf(ticker_no_suffix: str, end_date: datetime, days_back=360):
    try:
        ticker = yf.Ticker(f"{ticker_no_suffix}.JK")
        start = end_date - timedelta(days=days_back)
        df = ticker.history(start=start.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
        if df.empty:
            return None
        df = df.rename(columns={c: c.title() for c in df.columns})
        return df[['Open','High','Low','Close','Volume']].copy()
    except:
        return None

# =========================
# Indicators
# =========================
def compute_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return (100 - (100 / (1 + rs))).fillna(50)

def compute_macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def compute_bollinger(close, window=20, num_std=2):
    sma = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    bandwidth = (upper - lower) / sma
    percent_b = (close - lower) / (upper - lower)
    return upper, sma, lower, bandwidth, percent_b

def compute_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1,tr2,tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def compute_obv(close, volume):
    return (volume * np.sign(close.diff().fillna(0))).cumsum()

def compute_adx(high, low, close, period=14):
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period).mean() / atr)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    adx = dx.ewm(alpha=1/period).mean()
    return adx, plus_di, minus_di

def compute_vwap(high, low, close, volume):
    tp = (high+low+close)/3
    return (tp*volume).cumsum() / volume.cumsum()

# =========================
# Scoring
# =========================
class IndicatorScoringSystem:
    def __init__(self, weights=None):
        self.weights = weights or INDICATOR_WEIGHTS

    def score_rsi(self, rsi):
        val = rsi.iloc[-1]
        score = (50 - val) / 50
        return np.sign(score), min(1.0, abs(score))

    def score_macd(self, macd, signal, hist):
        cross = 1 if macd.iloc[-1] > signal.iloc[-1] else -1
        hist_score = 0.5 if hist.iloc[-1] > 0 else -0.5
        return cross, 1.0, hist_score, 1.0

    def composite(self, scores):
        total, wsum = 0, 0
        for k,(s,strg) in scores.items():
            w = self.weights.get(k,0)
            total += s*strg*w
            wsum += w
        return total/wsum if wsum else 0

# =========================
# Support / Resistance
# =========================
def calculate_support_resistance(df):
    current = df['Close'].iloc[-1]
    ma20 = df['Close'].rolling(20).mean().iloc[-1]
    ma50 = df['Close'].rolling(50).mean().iloc[-1]
    support = [lvl for lvl in [ma20, ma50] if lvl < current]
    resistance = [lvl for lvl in [ma20, ma50] if lvl > current]
    return {"Support":support,"Resistance":resistance}

# =========================
# Bandarmology
# =========================
def analyze_bandarmology(df, period=30):
    vol_ma = df['Volume'].rolling(period).mean()
    vol_std = df['Volume'].rolling(period).std()
    df['Volume_Z'] = (df['Volume'] - vol_ma) / (vol_std.replace(0,1))

    volume_spikes = int((df['Volume_Z'] > 2.5).iloc[-5:].sum())

    buying_days = int(((df['Close'] > df['Open']) & (df['Volume'] > vol_ma)).iloc[-period:].sum())
    selling_days = int(((df['Close'] < df['Open']) & (df['Volume'] > vol_ma)).iloc[-period:].sum())

    price_min = df['Low'].iloc[-period:].min()
    price_max = df['High'].iloc[-period:].max()
    poc = (price_min + price_max)/2

    return {
        "Volume Spikes (5 hari)": volume_spikes,
        "Hari Buying Pressure": buying_days,
        "Hari Selling Pressure": selling_days,
        "POC (approx)": poc
    }

# =========================
# Chart
# =========================
def create_chart(df, sr):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,open=df['Open'],high=df['High'],
                                 low=df['Low'],close=df['Close'],name="OHLC"))
    upper,sma,lower,_,_ = compute_bollinger(df['Close'])
    fig.add_trace(go.Scatter(x=df.index,y=upper,name="BB Upper",line=dict(color='blue',dash='dot')))
    fig.add_trace(go.Scatter(x=df.index,y=sma,name="BB Mid",line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=df.index,y=lower,name="BB Lower",line=dict(color='blue',dash='dot')))
    for s in sr['Support']:
        fig.add_hline(y=s,line_dash="dash",line_color="green")
    for r in sr['Resistance']:
        fig.add_hline(y=r,line_dash="dash",line_color="red")
    fig.update_layout(height=600,xaxis_rangeslider_visible=False)
    return fig

# =========================
# Main App
# =========================
def app():
    st.title("📊 Analisa Teknikal Saham - Layout Baru")

    # ---- Input di atas ----
    with st.form("params"):
        col1, col2, col3 = st.columns(3)
        with col1:
            ticker = st.text_input("Kode Saham (tanpa .JK)", "BBCA")
            modal = st.number_input("Modal (Rp)", value=100_000_000, step=1_000_000)
        with col2:
            risiko = st.slider("Risiko per Trade (%)",0.5,5.0,2.0)/100
            multi_tf = st.checkbox("Gunakan Multi-Timeframe", value=True)
        with col3:
            days = st.number_input("Ambil Data (hari)",90,2000,360)
            date = st.date_input("Tanggal Analisis", datetime.today())
        submitted = st.form_submit_button("🚀 Mulai Analisis")

    if not submitted: return

    # ---- Analisis di bawah ----
    df = get_stock_data_yf(ticker, date, days_back=days)
    if df is None or df.empty:
        st.warning("Data tidak tersedia.")
        return

    # Indikator
    df['RSI'] = compute_rsi(df['Close'])
    df['MACD'],df['Signal'],df['Hist'] = compute_macd(df['Close'])
    df['OBV'] = compute_obv(df['Close'],df['Volume'])
    df['ADX'],df['+DI'],df['-DI'] = compute_adx(df['High'],df['Low'],df['Close'])
    df['VWAP'] = compute_vwap(df['High'],df['Low'],df['Close'],df['Volume'])
    df['ATR'] = compute_atr(df['High'],df['Low'],df['Close'])

    # Scoring
    sc = IndicatorScoringSystem()
    scores = {
        'rsi': sc.score_rsi(df['RSI']),
        'macd_cross': (1 if df['MACD'].iloc[-1]>df['Signal'].iloc[-1] else -1,1),
        'macd_hist': (0.5 if df['Hist'].iloc[-1]>0 else -0.5,1),
        'obv': (1 if df['OBV'].iloc[-1]>df['OBV'].iloc[-5] else -1,1),
        'adx': (1 if df['ADX'].iloc[-1]>25 else 0,1)
    }
    comp = sc.composite(scores)

    st.subheader("🎯 Hasil Analisis Cross-Confirmation")
    st.metric("Composite Score", f"{comp:.2f}")
    st.table(pd.DataFrame([
        {"Indicator":k,"Dir":"Bullish" if v[0]>0 else "Bearish" if v[0]<0 else "Netral","Strength":f"{v[1]:.2f}"}
        for k,v in scores.items()
    ]))

    # Support/Resistance
    sr = calculate_support_resistance(df)
    st.subheader("📈 Support & Resistance")
    st.write(sr)

    # Bandarmology
    st.subheader("🕵️ Analisis Bandarmology")
    report = analyze_bandarmology(df)
    st.write(report)

    # Chart
    st.subheader("📊 Chart Teknikal")
    st.plotly_chart(create_chart(df,sr),use_container_width=True)

    # Volume
    st.subheader("📊 Volume")
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Bar(x=df.index,y=df['Volume'],name="Volume"))
    fig_vol.add_trace(go.Scatter(x=df.index,y=df['Volume'].rolling(20).mean(),name="MA20 Volume"))
    spikes = df[df['Volume_Z']>2.5].index if 'Volume_Z' in df else []
    if len(spikes)>0:
        fig_vol.add_trace(go.Scatter(x=spikes,y=df.loc[spikes,'Volume'],mode='markers',marker=dict(color='red',size=8),name="Spikes"))
    st.plotly_chart(fig_vol,use_container_width=True)

if __name__=="__main__":
    app()
