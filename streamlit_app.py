import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
from sklearn.linear_model import LinearRegression

# ====== HILANGKAN MENU MULTI-PAGE DEFAULT STREAMLIT ======
hide_pages_style = """
    <style>
        [data-testid="stSidebarNav"] {display: none;}
    </style>
"""
st.markdown(hide_pages_style, unsafe_allow_html=True)

# ====== MENU SIDEBAR MANUAL ======
menu = st.sidebar.radio(
    "Pilih Fitur",
    ["📊 Screener Multi", "🔪 Screener Pisau Jatuh"]
)

# ====== FUNGSI PEMBACAAN DATA GOOGLE DRIVE ======
def load_google_drive_excel(file_url):
    try:
        file_id = file_url.split("/d/")[1].split("/")[0]
        download_url = f"https://drive.google.com/uc?id={file_id}"
        return pd.read_excel(download_url)
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        return None

# ====== FUNGSI ANALISIS TEKNIKAL ======
def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_ma(data, period=20):
    return data['Close'].rolling(window=period).mean()

def calculate_obv(data):
    obv = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()
    if obv.iloc[-1] > obv.iloc[-20:].mean():
        return "Tekanan Beli"
    elif obv.iloc[-1] < obv.iloc[-20:].mean():
        return "Tekanan Jual"
    else:
        return "Netral"

def calculate_mfi(data, period=14):
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    money_flow = typical_price * data['Volume']
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
    return mfi

# ====== HALAMAN SCREENER MULTI ======
if menu == "📊 Screener Multi":
    st.title("📊 Screener Multi")
    file_url = st.text_input("Masukkan URL Google Drive untuk data saham")

    if file_url:
        df_list = load_google_drive_excel(file_url)
        if df_list is not None:
            results = []
            for ticker in df_list['Ticker']:
                try:
                    data = yf.download(ticker, period="3mo")
                    data['MA20'] = calculate_ma(data)
                    data['RSI'] = calculate_rsi(data)
                    data['OBV_Status'] = calculate_obv(data)
                    data['MFI'] = calculate_mfi(data)

                    last_row = data.iloc[-1]
                    results.append({
                        "Ticker": ticker,
                        "Close": last_row['Close'],
                        "MA20": last_row['MA20'],
                        "RSI": last_row['RSI'],
                        "OBV": last_row['OBV_Status'],
                        "MFI": last_row['MFI']
                    })
                except Exception as e:
                    st.warning(f"Gagal memproses {ticker}: {e}")

            st.dataframe(pd.DataFrame(results))

# ====== HALAMAN SCREENER PISAU JATUH ======
elif menu == "🔪 Screener Pisau Jatuh":
    st.title("🔪 Screener Pisau Jatuh")
    tickers = st.text_area("Masukkan daftar ticker (pisahkan dengan koma)")

    if tickers:
        ticker_list = [t.strip() for t in tickers.split(",")]
        results = []
        for ticker in ticker_list:
            try:
                data = yf.download(ticker, period="3mo")
                change = (data['Close'].iloc[-1] - data['Close'].iloc[-5]) / data['Close'].iloc[-5] * 100
                if change < -10:  # Contoh kriteria "pisau jatuh"
                    results.append({"Ticker": ticker, "Perubahan 5 Hari (%)": change})
            except Exception as e:
                st.warning(f"Gagal memproses {ticker}: {e}")

        if results:
            st.dataframe(pd.DataFrame(results))
        else:
            st.info("Tidak ada saham yang terdeteksi sebagai 'Pisau Jatuh'.")
