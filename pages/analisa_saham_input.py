import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import plotly.graph_objects as go
from scipy.signal import argrelextrema
import time

# =========================
# --- INDIKATOR TEKNIKAL ---
# =========================

def compute_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_mfi(df, period=14):
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = tp * df['Volume']
    positive_flow, negative_flow = [0], [0]
    for i in range(1, len(tp)):
        if tp.iloc[i] > tp.iloc[i-1]:
            positive_flow.append(money_flow.iloc[i])
            negative_flow.append(0)
        elif tp.iloc[i] < tp.iloc[i-1]:
            positive_flow.append(0)
            negative_flow.append(money_flow.iloc[i])
        else:
            positive_flow.append(0)
            negative_flow.append(0)
    pos_mf = pd.Series(positive_flow).rolling(window=period, min_periods=1).sum()
    neg_mf = pd.Series(negative_flow).rolling(window=period, min_periods=1).sum()
    ratio = np.where(neg_mf > 0, pos_mf / neg_mf, 1.0)
    mfi = 100 - (100 / (1 + ratio))
    return pd.Series(mfi, index=df.index)

def compute_macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def compute_bollinger_bands(close, window=20, num_std=2):
    sma = close.rolling(window=window).mean()
    std = close.rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return upper_band, sma, lower_band

def compute_obv(df):
    obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    return obv

def compute_adl(df):
    mfm = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low']).replace(0, np.nan)
    mfv = mfm * df['Volume']
    adl = mfv.cumsum()
    return adl

def compute_adx(df, period=14):
    df['TR'] = np.maximum(df['High'] - df['Low'], np.maximum(abs(df['High'] - df['Close'].shift()), abs(df['Low'] - df['Close'].shift())))
    df['+DM'] = np.where((df['High'] - df['High'].shift()) > (df['Low'].shift() - df['Low']), df['High'] - df['High'].shift(), 0)
    df['-DM'] = np.where((df['Low'].shift() - df['Low']) > (df['High'] - df['High'].shift()), df['Low'].shift() - df['Low'], 0)
    df['+DI'] = 100 * (df['+DM'].ewm(alpha=1/period).mean() / df['TR'].ewm(alpha=1/period).mean())
    df['-DI'] = 100 * (df['-DM'].ewm(alpha=1/period).mean() / df['TR'].ewm(alpha=1/period).mean())
    dx = (abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])) * 100
    adx = dx.ewm(alpha=1/period).mean()
    return adx

def compute_atr(df, period=14):
    tr = np.maximum(df['High'] - df['Low'], np.maximum(abs(df['High'] - df['Close'].shift()), abs(df['Low'] - df['Close'].shift())))
    atr = tr.rolling(window=period).mean()
    return atr

def calculate_vwap(df):
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    vwap = (tp * df['Volume']).cumsum() / df['Volume'].cumsum()
    return vwap

# ================================
# --- SISTEM CROSS CONFIRMATION ---
# ================================

class IndicatorScoringSystem:
    def __init__(self, df):
        self.df = df

    def score_rsi(self):
        val = self.df['RSI'].iloc[-1]
        if val < 30: return 1
        elif val > 70: return -1
        return 0

    def score_macd(self):
        if self.df['MACD'].iloc[-1] > self.df['Signal'].iloc[-1]:
            return 1
        elif self.df['MACD'].iloc[-1] < self.df['Signal'].iloc[-1]:
            return -1
        return 0

    def score_bollinger(self):
        if self.df['Close'].iloc[-1] < self.df['BB_Lower'].iloc[-1]:
            return 1
        elif self.df['Close'].iloc[-1] > self.df['BB_Upper'].iloc[-1]:
            return -1
        return 0

    def score_volume(self):
        if self.df['Volume'].iloc[-1] > 1.5 * self.df['Avg_Volume_20'].iloc[-1]:
            return 1
        return 0

    def score_obv(self):
        return 1 if self.df['OBV'].iloc[-1] > self.df['OBV'].iloc[-5] else -1

    def score_adx(self):
        val = self.df['ADX'].iloc[-1]
        return 1 if val > 25 else 0

    def calculate_composite_score(self):
        scores = [self.score_rsi(), self.score_macd(), self.score_bollinger(), self.score_volume(), self.score_obv(), self.score_adx()]
        return np.mean(scores)

    def get_confidence_level(self, composite_score):
        if abs(composite_score) > 0.7: return "Tinggi"
        elif abs(composite_score) > 0.3: return "Sedang"
        return "Rendah"

# ===========================
# --- BREAKOUT DETECTION ---
# ===========================

class BreakoutDetector:
    def __init__(self, df, sr):
        self.df = df
        self.sr = sr

    def detect_breakout(self):
        last_close = self.df['Close'].iloc[-1]
        last_volume = self.df['Volume'].iloc[-1]
        avg_volume = self.df['Avg_Volume_20'].iloc[-1]
        breakout = None

        for r in self.sr['Resistance']:
            if last_close > r and last_volume > 1.5 * avg_volume:
                breakout = f"🚀 Breakout Resistance {r:,.2f} dengan volume tinggi"
        for s in self.sr['Support']:
            if last_close < s and last_volume > 1.5 * avg_volume:
                breakout = f"⚠️ Breakdown Support {s:,.2f} dengan volume tinggi"
        return breakout

# ======================
# --- KESIMPULAN NARATIF
# ======================

def generate_conclusion(df, composite_score, confidence, breakout, market_trend):
    conclusions = []
    conclusions.append(f"🌐 **Konteks Pasar**: {market_trend[1]}")
    conclusions.append(f"📊 **Composite Score**: {composite_score:.2f} (Kepercayaan: {confidence})")
    if breakout:
        conclusions.append(breakout)
    if composite_score > 0.5:
        conclusions.append("✅ Rekomendasi: Bias Bullish — pertimbangkan akumulasi dengan manajemen risiko.")
    elif composite_score < -0.5:
        conclusions.append("❌ Rekomendasi: Bias Bearish — pertimbangkan take profit atau wait and see.")
    else:
        conclusions.append("⏺️ Rekomendasi: Netral — tunggu konfirmasi lebih lanjut.")
    return conclusions

# ===================
# --- APP STREAMLIT ---
# ===================

def app():
    st.title("📈 Analisa Teknikal Saham Hybrid")
    ticker_input = st.text_input("Masukkan Kode Saham (contoh: BBCA)", value="BBCA")
    analysis_date = st.date_input("📅 Tanggal Analisis", value=datetime.today())

    if st.button("🔍 Mulai Analisis"):
        with st.spinner('Sedang mengambil data...'):
            time.sleep(1)
            ticker = ticker_input.strip().upper()
            if not ticker.endswith(".JK"):
                ticker += ".JK"
            stock = yf.Ticker(ticker)
            start_date = analysis_date - timedelta(days=180)
            df = stock.history(start=start_date, end=analysis_date)
            if df.empty:
                st.warning("Data tidak tersedia.")
                return

            # Hitung indikator
            df['RSI'] = compute_rsi(df['Close'])
            df['MFI'] = compute_mfi(df)
            df['MACD'], df['Signal'], df['Hist'] = compute_macd(df['Close'])
            df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = compute_bollinger_bands(df['Close'])
            df['Avg_Volume_20'] = df['Volume'].rolling(20).mean()
            df['OBV'] = compute_obv(df)
            df['ADL'] = compute_adl(df)
            df['ADX'] = compute_adx(df)
            df['ATR'] = compute_atr(df)
            df['VWAP'] = calculate_vwap(df)

            # Support/Resistance sederhana
            sr = {"Support": [df['Close'].min()], "Resistance": [df['Close'].max()]}

            # Sistem scoring
            scorer = IndicatorScoringSystem(df)
            composite_score = scorer.calculate_composite_score()
            confidence = scorer.get_confidence_level(composite_score)

            # Breakout detection
            breakout = BreakoutDetector(df, sr).detect_breakout()

            # Market trend (IHSG)
            idx = yf.Ticker("^JKSE")
            idx_data = idx.history(period="1mo")
            idx_change = ((idx_data['Close'].iloc[-1] - idx_data['Close'].iloc[0]) / idx_data['Close'].iloc[0]) * 100
            market_trend = ("Bullish" if idx_change > 3 else "Bearish" if idx_change < -3 else "Netral", f"Perubahan {idx_change:.2f}% dalam 1 bulan")

            # Kesimpulan naratif
            conclusions = generate_conclusion(df, composite_score, confidence, breakout, market_trend)

            # --- OUTPUT ---
            st.subheader("📋 Kesimpulan Analisis")
            for c in conclusions:
                st.markdown(f"- {c}")

            st.subheader("📊 Chart")
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Candlestick'))
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(color='red', dash='dot'), name='BB Upper'))
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Middle'], line=dict(color='purple'), name='BB Middle'))
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color='green', dash='dot'), name='BB Lower'))
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    app()
