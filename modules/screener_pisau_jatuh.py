import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from scipy.signal import argrelextrema
import warnings

# ==================== PENGATURAN AWAL ====================
warnings.filterwarnings("ignore", category=UserWarning)

# Inisialisasi session state untuk hasil screening
if "screening_results" not in st.session_state:
    st.session_state.screening_results = None

# ==================== PARAMETER ====================
DEFAULT_PERIOD = "6mo"
DEFAULT_INTERVAL = "1d"

# ==================== FUNGSI AMBIL DATA ====================
def get_stock_data(ticker, period=DEFAULT_PERIOD, interval=DEFAULT_INTERVAL):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        df.dropna(inplace=True)
        return df
    except Exception:
        return pd.DataFrame()

# ==================== FUNGSI DETEKSI PISAU JATUH ====================
def detect_falling_knife(df):
    if df.empty:
        return False

    # Syarat: Penurunan signifikan dalam beberapa hari terakhir
    recent_close = df["Close"].iloc[-1]
    prev_close = df["Close"].iloc[-4]
    drop_pct = ((recent_close - prev_close) / prev_close) * 100

    return drop_pct <= -10  # misal turunnya ≥ 10%

# ==================== FUNGSI FIBONACCI RETRACEMENT ====================
def fibonacci_retracement(df):
    if df.empty:
        return None
    max_price = df['High'].max()
    min_price = df['Low'].min()
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

# ==================== FUNGSI VOLUME PROFILE ====================
def volume_profile(df, bins=20):
    if df.empty:
        return None
    prices = (df['High'] + df['Low']) / 2
    volumes = df['Volume']
    data = pd.DataFrame({'Price': prices, 'Volume': volumes})
    data['Price_bin'] = pd.cut(data['Price'], bins=bins)
    profile = data.groupby('Price_bin')['Volume'].sum().reset_index()
    max_vol_bin = profile.loc[profile['Volume'].idxmax(), 'Price_bin']
    return max_vol_bin

# ==================== FUNGSI SCREENING ====================
def screening(tickers):
    results = []

    for ticker in tickers:
        df = get_stock_data(ticker)

        if detect_falling_knife(df):
            fibo_levels = fibonacci_retracement(df)
            vol_profile = volume_profile(df)

            results.append({
                "Ticker": ticker,
                "Close": df["Close"].iloc[-1],
                "Fib Levels": fibo_levels,
                "Volume Profile": str(vol_profile)
            })

    return pd.DataFrame(results)

# ==================== HALAMAN STREAMLIT ====================
def run():
    st.title("📉 Screener Pisau Jatuh")

    ticker_input = st.text_area(
        "Masukkan daftar ticker saham (pisahkan dengan koma)",
        value="BBRI.JK, BBCA.JK, TLKM.JK"
    )

    if st.button("🔍 Jalankan Screening"):
        tickers = [t.strip() for t in ticker_input.split(",") if t.strip()]
        df_results = screening(tickers)

        st.session_state.screening_results = df_results

        if not df_results.empty:
            st.success("Screening selesai ✅")
            st.dataframe(df_results)
        else:
            st.warning("Tidak ada saham yang memenuhi kriteria.")

    if st.session_state.screening_results is not None and not st.session_state.screening_results.empty:
        st.subheader("Hasil Screening Terakhir")
        st.dataframe(st.session_state.screening_results)

# Eksekusi modul jika dijalankan langsung
if __name__ == "__main__":
    run()
