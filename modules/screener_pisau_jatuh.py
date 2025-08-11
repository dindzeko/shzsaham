# modules/screener_pisau_jatuh.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.cluster import KMeans

# =======================
# Fungsi Load Data Google Sheets
# =======================
def load_google_drive_excel(file_url):
    try:
        file_id = file_url.split("/d/")[1].split("/")[0]
        export_url = f"https://drive.google.com/uc?id={file_id}&export=download"
        df = pd.read_excel(export_url)
        return df
    except Exception as e:
        st.error(f"Gagal memuat file Google Drive: {e}")
        return pd.DataFrame()

# =======================
# Indikator Teknis
# =======================
def calculate_ma20(df):
    df['MA20'] = df['Close'].rolling(window=20).mean()
    return df

def calculate_rsi(df, period=14):
    delta = df['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def calculate_obv(df):
    obv = [0]
    for i in range(1, len(df)):
        if df['Close'][i] > df['Close'][i - 1]:
            obv.append(obv[-1] + df['Volume'][i])
        elif df['Close'][i] < df['Close'][i - 1]:
            obv.append(obv[-1] - df['Volume'][i])
        else:
            obv.append(obv[-1])
    df['OBV'] = obv
    df['OBV_Interpretasi'] = np.where(df['OBV'].diff() > 0, 'Tekanan Beli',
                                      np.where(df['OBV'].diff() < 0, 'Tekanan Jual', 'Netral'))
    return df

def calculate_mfi(df, period=14):
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = typical_price * df['Volume']
    positive_flow = []
    negative_flow = []
    for i in range(1, len(typical_price)):
        if typical_price[i] > typical_price[i-1]:
            positive_flow.append(money_flow[i-1])
            negative_flow.append(0)
        elif typical_price[i] < typical_price[i-1]:
            positive_flow.append(0)
            negative_flow.append(money_flow[i-1])
        else:
            positive_flow.append(0)
            negative_flow.append(0)
    positive_mf = pd.Series(positive_flow).rolling(window=period).sum()
    negative_mf = pd.Series(negative_flow).rolling(window=period).sum()
    mfi = 100 * (positive_mf / (positive_mf + negative_mf))
    df['MFI'] = mfi.reindex(df.index, fill_value=np.nan)
    return df

# =======================
# Fibonacci Retracement
# =======================
def calculate_fibonacci(df):
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

# =======================
# Volume Profile
# =======================
def calculate_volume_profile(df, bins=10):
    df['price_bin'] = pd.cut(df['Close'], bins=bins)
    volume_profile = df.groupby('price_bin')['Volume'].sum().reset_index()
    max_volume_row = volume_profile.loc[volume_profile['Volume'].idxmax()]
    return {
        "Level Tertinggi Volume": max_volume_row['price_bin'],
        "Volume Tertinggi": max_volume_row['Volume']
    }

# =======================
# Screening Pisau Jatuh
# =======================
def is_falling_knife(df):
    if len(df) < 4:
        return False
    last4 = df['Close'].tail(4).reset_index(drop=True)
    return all(last4[i] < last4[i-1] for i in range(1, 4))

# =======================
# Fungsi Utama App
# =======================
def app():
    st.title("📉 Screener Pisau Jatuh + Analisis Lanjutan")

    file_url = "https://docs.google.com/spreadsheets/d/1t6wgBIcPEUWMq40GdIH1GtZ8dvI9PZ2v/edit?usp=drive_link"
    df_tickers = load_google_drive_excel(file_url)

    if df_tickers.empty or 'Ticker' not in df_tickers.columns:
        st.error("File tidak memiliki kolom 'Ticker'.")
        return

    tickers = df_tickers['Ticker'].dropna().unique().tolist()
    hasil_screening = []

    end_date = datetime.today()
    start_date = end_date - timedelta(days=100)

    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            if data.empty:
                continue

            data = calculate_ma20(data)
            data = calculate_rsi(data)
            data = calculate_obv(data)
            data = calculate_mfi(data)

            if is_falling_knife(data):
                fibo_levels = calculate_fibonacci(data)
                vol_profile = calculate_volume_profile(data)

                hasil_screening.append({
                    "Ticker": ticker,
                    "Close": data['Close'].iloc[-1],
                    "MA20": data['MA20'].iloc[-1],
                    "RSI": data['RSI'].iloc[-1],
                    "OBV Interpretasi": data['OBV_Interpretasi'].iloc[-1],
                    "MFI": data['MFI'].iloc[-1],
                    "Fibonacci Levels": fibo_levels,
                    "Volume Profile": vol_profile
                })
        except Exception as e:
            st.warning(f"Error memproses {ticker}: {e}")

    if hasil_screening:
        df_hasil = pd.DataFrame(hasil_screening)
        st.dataframe(df_hasil)
    else:
        st.info("Tidak ada saham yang memenuhi kriteria pisau jatuh.")

