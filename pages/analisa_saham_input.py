import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# =========================
# Fungsi indikator teknikal
# =========================
def calculate_indicators(df):
    df["MA20"] = df["Close"].rolling(window=20).mean()
    df["MA50"] = df["Close"].rolling(window=50).mean()

    # RSI
    delta = df["Close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # MFI
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    money_flow = typical_price * df["Volume"]
    positive_flow = []
    negative_flow = []
    for i in range(1, len(typical_price)):
        if typical_price[i] > typical_price[i-1]:
            positive_flow.append(money_flow.iloc[i-1])
            negative_flow.append(0)
        else:
            positive_flow.append(0)
            negative_flow.append(money_flow.iloc[i-1])
    positive_mf = pd.Series(positive_flow).rolling(14).sum()
    negative_mf = pd.Series(negative_flow).rolling(14).sum()
    mfi = 100 * (positive_mf / (positive_mf + negative_mf))
    df["MFI"] = mfi.values

    # MACD
    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["Histogram"] = df["MACD"] - df["Signal"]

    # Bollinger Bands
    df["BB_Mid"] = df["MA20"]
    df["BB_Upper"] = df["MA20"] + 2 * df["Close"].rolling(window=20).std()
    df["BB_Lower"] = df["MA20"] - 2 * df["Close"].rolling(window=20).std()
    df["BB_Width"] = df["BB_Upper"] - df["BB_Lower"]

    return df

# =========================
# Fungsi kesimpulan
# =========================
def get_summary(df):
    latest = df.iloc[-1]
    signal = []

    # MACD
    if latest["MACD"] > latest["Signal"]:
        signal.append("MACD Bullish")
    else:
        signal.append("MACD Bearish")

    # RSI
    if latest["RSI"] > 70:
        signal.append("RSI Overbought")
    elif latest["RSI"] < 30:
        signal.append("RSI Oversold")

    # MFI
    if latest["MFI"] > 80:
        signal.append("MFI Overbought")
    elif latest["MFI"] < 20:
        signal.append("MFI Oversold")

    # Bollinger Bands
    if latest["Close"] > latest["BB_Upper"]:
        signal.append("Breakout ↑ (Upper Band)")
    elif latest["Close"] < latest["BB_Lower"]:
        signal.append("Breakout ↓ (Lower Band)")
    elif abs(latest["BB_Width"]) < latest["MA20"] * 0.05:
        signal.append("Konsolidasi (BB squeeze)")
    else:
        if abs(latest["Close"] - latest["BB_Upper"]) / latest["BB_Upper"] < 0.01:
            signal.append("Reversal ↓ (Near Upper Band)")
        elif abs(latest["Close"] - latest["BB_Lower"]) / latest["BB_Lower"] < 0.01:
            signal.append("Reversal ↑ (Near Lower Band)")

    return signal

# =========================
# Streamlit App
# =========================
st.set_page_config(page_title="Analisa Saham", layout="wide")
st.title("📊 Analisa Saham dengan Indikator Teknikal")

ticker = st.text_input("Masukkan kode saham (contoh: BBCA.JK):", "BBCA.JK")

if st.button("Analisa"):
    df = yf.download(ticker, period="6mo", interval="1d")
    df = calculate_indicators(df)

    # =========================
    # Kesimpulan di atas
    # =========================
    signals = get_summary(df)
    st.subheader("📌 Kesimpulan Analisa")
    for s in signals:
        if "Bullish" in s or "Breakout ↑" in s or "Reversal ↑" in s:
            st.success(s)
        elif "Bearish" in s or "Breakout ↓" in s or "Reversal ↓" in s:
            st.error(s)
        else:
            st.warning(s)

    # =========================
    # Plot harga + Bollinger Bands
    # =========================
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
                                 low=df["Low"], close=df["Close"], name="Candlestick"))
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"], line=dict(color="blue", width=1), name="Upper Band"))
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Mid"], line=dict(color="orange", width=1), name="MA20"))
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"], line=dict(color="blue", width=1), name="Lower Band"))
    st.plotly_chart(fig, use_container_width=True)

    # =========================
    # Indikator teknikal
    # =========================
    st.subheader("📈 Indikator Teknis")
    latest = df.iloc[-1]
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("MA20", f"{latest['MA20']:.2f}")
        st.metric("MA50", f"{latest['MA50']:.2f}")
    with col2:
        st.metric("RSI", f"{latest['RSI']:.2f}")
        st.metric("MFI", f"{latest['MFI']:.2f}")
    with col3:
        st.metric("MACD", f"{latest['MACD']:.2f}")
        st.metric("Signal", f"{latest['Signal']:.2f}")
