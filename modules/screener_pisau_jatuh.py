# modules/screener_pisau_jatuh.py
import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.cluster import KMeans

# ===================== FUNGSI TEKNIKAL =====================

def calculate_ma(data, period=20):
    data[f"MA{period}"] = data['Close'].rolling(window=period).mean()
    return data

def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(period).mean()
    avg_loss = pd.Series(loss).rolling(period).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    return data

def calculate_mfi(data, period=14):
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    money_flow = typical_price * data['Volume']
    positive_flow = []
    negative_flow = []

    for i in range(1, len(typical_price)):
        if typical_price[i] > typical_price[i - 1]:
            positive_flow.append(money_flow[i - 1])
            negative_flow.append(0)
        elif typical_price[i] < typical_price[i - 1]:
            negative_flow.append(money_flow[i - 1])
            positive_flow.append(0)
        else:
            positive_flow.append(0)
            negative_flow.append(0)

    positive_mf = pd.Series(positive_flow).rolling(period).sum()
    negative_mf = pd.Series(negative_flow).rolling(period).sum()

    mfi = 100 * (positive_mf / (positive_mf + negative_mf))
    data['MFI'] = mfi
    return data

def calculate_obv(data):
    obv = [0]
    for i in range(1, len(data)):
        if data['Close'][i] > data['Close'][i-1]:
            obv.append(obv[-1] + data['Volume'][i])
        elif data['Close'][i] < data['Close'][i-1]:
            obv.append(obv[-1] - data['Volume'][i])
        else:
            obv.append(obv[-1])
    data['OBV'] = obv
    data['OBV_Interpretation'] = data['OBV'].diff().apply(
        lambda x: "Tekanan Beli" if x > 0 else ("Tekanan Jual" if x < 0 else "Netral")
    )
    return data

def calculate_volume_profile(data, bins=12):
    prices = data['Close']
    volumes = data['Volume']
    hist, bin_edges = np.histogram(prices, bins=bins, weights=volumes)
    vol_profile = pd.DataFrame({
        'Price_Level': (bin_edges[:-1] + bin_edges[1:]) / 2,
        'Volume': hist
    })
    vol_profile = vol_profile.sort_values(by='Volume', ascending=False)
    support = vol_profile.iloc[-1]['Price_Level']
    resistance = vol_profile.iloc[0]['Price_Level']
    return support, resistance

def calculate_fibonacci(data):
    max_price = data['High'].max()
    min_price = data['Low'].min()
    diff = max_price - min_price
    levels = {
        '0.0%': max_price,
        '23.6%': max_price - 0.236 * diff,
        '38.2%': max_price - 0.382 * diff,
        '50.0%': max_price - 0.5 * diff,
        '61.8%': max_price - 0.618 * diff,
        '100.0%': min_price
    }
    return levels

# ===================== FUNGSI SCREENING =====================

def screen_falling_knife(df):
    results = []
    for ticker in df['Ticker']:
        try:
            data = yf.download(ticker, period="3mo", interval="1d", progress=False)
            if data.empty or len(data) < 20:
                continue

            data = calculate_ma(data)
            data = calculate_rsi(data)
            data = calculate_mfi(data)
            data = calculate_obv(data)

            support, resistance = calculate_volume_profile(data)
            fibo_levels = calculate_fibonacci(data)

            latest = data.iloc[-1]
            price = latest['Close']

            # Deteksi pola "Pisau Jatuh" (turun berturut-turut min 3 hari)
            close_changes = data['Close'].diff()
            falling_days = (close_changes < 0).tail(3).sum()
            is_falling_knife = falling_days >= 3

            if is_falling_knife:
                results.append({
                    'Ticker': ticker,
                    'Price': price,
                    'MA20': latest['MA20'],
                    'RSI': latest['RSI'],
                    'MFI': latest['MFI'],
                    'OBV Interpretasi': latest['OBV_Interpretation'],
                    'Support (VP)': support,
                    'Resistance (VP)': resistance,
                    'Fibo Levels': fibo_levels
                })
        except Exception as e:
            st.warning(f"Error memproses {ticker}: {e}")
    return pd.DataFrame(results)

# ===================== FUNGSI STREAMLIT =====================

def app():
    st.title("📉 Screener Pisau Jatuh")
    st.caption("Screening saham jatuh berturut-turut + Analisis Lanjutan")

    uploaded_file = st.file_uploader("Upload daftar saham (.xlsx dengan kolom 'Ticker')", type=["xlsx"])
    if uploaded_file is not None:
        df_list = pd.read_excel(uploaded_file)
        if 'Ticker' not in df_list.columns:
            st.error("File harus memiliki kolom 'Ticker'")
            return

        with st.spinner("⏳ Sedang memproses screening..."):
            results_df = screen_falling_knife(df_list)

        if not results_df.empty:
            st.success(f"✅ Ditemukan {len(results_df)} saham pola Pisau Jatuh")
            st.dataframe(results_df)
        else:
            st.info("Tidak ada saham yang memenuhi kriteria.")
