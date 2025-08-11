# modules/screener_pisau_jatuh.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.cluster import KMeans

# ===================== Fungsi Analisis Teknis ===================== #

def calculate_ma20(df):
    df['MA20'] = df['Close'].rolling(window=20).mean()
    return df

def calculate_rsi(df, period=14):
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def calculate_obv(df):
    obv = [0]
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
            obv.append(obv[-1] + df['Volume'].iloc[i])
        elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
            obv.append(obv[-1] - df['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    df['OBV'] = obv

    # Interpretasi OBV
    if df['OBV'].iloc[-1] > df['OBV'].iloc[-5]:
        df['OBV_Interpretasi'] = "Tekanan Beli"
    elif df['OBV'].iloc[-1] < df['OBV'].iloc[-5]:
        df['OBV_Interpretasi'] = "Tekanan Jual"
    else:
        df['OBV_Interpretasi'] = "Netral"

    return df

def calculate_mfi(df, period=14):
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = typical_price * df['Volume']

    positive_flow = []
    negative_flow = []
    for i in range(1, len(df)):
        if typical_price.iloc[i] > typical_price.iloc[i-1]:
            positive_flow.append(money_flow.iloc[i])
            negative_flow.append(0)
        elif typical_price.iloc[i] < typical_price.iloc[i-1]:
            positive_flow.append(0)
            negative_flow.append(money_flow.iloc[i])
        else:
            positive_flow.append(0)
            negative_flow.append(0)

    positive_mf = pd.Series(positive_flow).rolling(window=period).sum()
    negative_mf = pd.Series(negative_flow).rolling(window=period).sum()
    mfi = 100 * (positive_mf / (positive_mf + negative_mf))
    mfi = mfi.reindex(df.index, method='pad')
    df['MFI'] = mfi
    return df

def calculate_volume_profile(df, bins=12):
    price = df['Close']
    volume = df['Volume']

    hist, bin_edges = np.histogram(price, bins=bins, weights=volume)
    vol_profile = pd.DataFrame({'price_level': bin_edges[:-1], 'volume': hist})
    high_vol = vol_profile.loc[vol_profile['volume'].idxmax(), 'price_level']
    low_vol = vol_profile.loc[vol_profile['volume'].idxmin(), 'price_level']
    return round(low_vol, 2), round(high_vol, 2)

def calculate_fibonacci_levels(df):
    high_price = df['High'].max()
    low_price = df['Low'].min()
    diff = high_price - low_price
    levels = {
        'Fibo_0.236': high_price - 0.236 * diff,
        'Fibo_0.382': high_price - 0.382 * diff,
        'Fibo_0.5': high_price - 0.5 * diff,
        'Fibo_0.618': high_price - 0.618 * diff,
        'Fibo_0.786': high_price - 0.786 * diff
    }
    return {k: round(v, 2) for k, v in levels.items()}

# ===================== Screening Pisau Jatuh ===================== #

def screening_pisau_jatuh(symbols):
    results = []

    for symbol in symbols:
        try:
            df = yf.download(symbol, period="3mo", interval="1d", progress=False)
            if df.empty or len(df) < 20:
                continue

            df = calculate_ma20(df)
            df = calculate_rsi(df)
            df = calculate_obv(df)
            df = calculate_mfi(df)

            low_vol, high_vol = calculate_volume_profile(df)
            fibo_levels = calculate_fibonacci_levels(df)

            # Logika screening "Pisau Jatuh" — contoh sederhana:
            last_close = df['Close'].iloc[-1]
            ma20 = df['MA20'].iloc[-1]
            rsi = df['RSI'].iloc[-1]

            if (last_close < ma20) and (rsi < 30):
                results.append({
                    "Kode": symbol,
                    "Close": round(last_close, 2),
                    "MA20": round(ma20, 2),
                    "RSI": round(rsi, 2),
                    "OBV": df['OBV_Interpretasi'].iloc[-1],
                    "MFI": round(df['MFI'].iloc[-1], 2),
                    "Support_VP": low_vol,
                    "Resistance_VP": high_vol,
                    **fibo_levels
                })

        except Exception as e:
            print(f"Error {symbol}: {e}")
            continue

    return pd.DataFrame(results)

# ===================== Fungsi Utama untuk Streamlit ===================== #

def run():
    st.title("📉 Screener Pisau Jatuh + Analisis Lanjutan")

    symbols_input = st.text_area("Masukkan kode saham (pisahkan dengan koma)", "BBCA.JK, BBRI.JK, TLKM.JK")
    symbols = [s.strip() for s in symbols_input.split(",") if s.strip()]

    if st.button("Screening"):
        with st.spinner("Sedang memproses..."):
            results_df = screening_pisau_jatuh(symbols)

        if not results_df.empty:
            st.success(f"Ditemukan {len(results_df)} saham yang memenuhi kriteria.")
            st.dataframe(results_df)
        else:
            st.warning("Tidak ada saham yang memenuhi kriteria.")
