import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import percentileofscore
import plotly.graph_objs as go

# =============================
# Fungsi Hitung Indikator
# =============================
def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def macd(df):
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    macd_line = exp1 - exp2
    signal = macd_line.ewm(span=9, adjust=False).mean()
    hist = macd_line - signal
    return macd_line, signal, hist

def bollinger_bands(series, window=20, num_std=2):
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return upper, lower

def atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr_val = true_range.rolling(period).mean()
    return atr_val

# =============================
# Streamlit UI
# =============================
col1, col2 = st.columns([2,1])

with col2:
    ticker = st.text_input("Kode Saham (contoh: BBCA.JK)", "BBCA.JK")
    periode = st.selectbox("Periode data", ["1mo","3mo","6mo","1y"], index=1)
    modal = st.number_input("Modal / Dana (Rp)", value=10000000, step=1000000)
    risk_pct = st.slider("Risk per trade (%)", 0.5, 5.0, 1.0, 0.1)
    run = st.button("🚀 Mulai Analisis")

with col1:
    st.title("📈 Analisa Teknikal Saham (Streamlit)")

    if run:
        data = yf.download(ticker, period=periode, interval="1d")

        if data.empty:
            st.error("Data tidak ditemukan.")
        else:
            data['RSI'] = rsi(data['Close'])
            data['MACD'], data['Signal'], data['Hist'] = macd(data)
            data['BB_upper'], data['BB_lower'] = bollinger_bands(data['Close'])
            data['ATR'] = atr(data)

            last_close = data.iloc[-1]['Close']
            last_rsi = data.iloc[-1]['RSI']
            last_macd = data.iloc[-1]['MACD']
            last_signal = data.iloc[-1]['Signal']
            last_hist = data.iloc[-1]['Hist']

            # Kesimpulan sederhana
            st.subheader("📊 Ringkasan Analisa")
            if last_rsi < 30:
                st.success("RSI Oversold → Potensi Rebound")
            elif last_rsi > 70:
                st.warning("RSI Overbought → Waspada Reversal")

            if last_macd > last_signal and last_hist > 0:
                st.success("MACD Bullish Crossover")
            elif last_macd < last_signal and last_hist < 0:
                st.error("MACD Bearish Crossover")

            # Plot Candlestick + Bollinger Bands
            fig = go.Figure(data=[go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Candlestick')])

            fig.add_trace(go.Scatter(x=data.index, y=data['BB_upper'], line=dict(color='blue', width=1), name='Upper BB'))
            fig.add_trace(go.Scatter(x=data.index, y=data['BB_lower'], line=dict(color='blue', width=1), name='Lower BB'))
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'].rolling(20).mean(), line=dict(color='orange', width=1), name='SMA20'))

            fig.update_layout(title=f"{ticker} - Analisa Teknikal", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

            # Tabel ringkasan indikator
            summary_df = pd.DataFrame({
                "Indikator": ["Close", "RSI", "MACD", "Signal", "Hist", "ATR"],
                "Nilai": [last_close, last_rsi, last_macd, last_signal, last_hist, data.iloc[-1]['ATR']]
            })
            st.dataframe(summary_df)
