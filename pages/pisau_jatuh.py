# pages/pisau_jatuh.py
import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
from scipy.signal import argrelextrema
import io
import re

# --- FUNGSI UTILITY ---
def clean_numeric_value(value):
    """Membersihkan nilai numerik dari format string dengan koma"""
    if isinstance(value, str):
        # Hapus karakter non-numeric kecuali titik dan minus
        cleaned = re.sub(r'[^\d.-]', '', value)
        try:
            return float(cleaned)
        except ValueError:
            return 0.0
    return float(value)

# --- FUNGSI DETEKSI POLA ---
def detect_pattern(data):
    if len(data) < 4:
        return False
        
    recent = data.tail(4)
    c1, c2, c3, c4 = recent.iloc[0], recent.iloc[1], recent.iloc[2], recent.iloc[3]

    is_c1_bullish = c1['Close'] > c1['Open'] and (c1['Close'] - c1['Open']) > 0.015 * c1['Open']
    is_c2_bearish = c2['Close'] < c2['Open'] and c2['Close'] < c1['Close']
    is_c3_bearish = c3['Close'] < c3['Open']
    is_c4_bearish = c4['Close'] < c4['Open']
    is_uptrend = data['Close'].iloc[-20:].mean() > data['Close'].iloc[-50:-20].mean() if len(data) >= 50 else False
    is_close_sequence = c2['Close'] > c3['Close'] > c4['Close']

    return all([
        is_c1_bullish,
        is_c2_bearish,
        is_c3_bearish,
        is_c4_bearish,
        is_uptrend,
        is_close_sequence
    ])

def get_stock_data(ticker, end_date):
    try:
        stock = yf.Ticker(f"{ticker}.JK")
        start_date = end_date - timedelta(days=90)
        data = stock.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
        
        # Pastikan data numerik sudah dalam format yang benar
        if data is not None and not data.empty:
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_columns:
                if col in data.columns:
                    data[col] = data[col].apply(clean_numeric_value)
                    
        return data if not data.empty else None
    except Exception as e:
        st.error(f"Gagal mengambil data untuk {ticker}: {e}")
        return None

def load_google_drive_excel(file_url):
    try:
        file_id = file_url.split("/d/")[1].split("/")[0]
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        df = pd.read_excel(download_url, engine='openpyxl')

        if 'Ticker' not in df.columns or 'Papan Pencatatan' not in df.columns:
            st.error("Kolom 'Ticker' dan 'Papan Pencatatan' harus ada di file Excel.")
            return None

        # Bersihkan nilai numerik jika ada di kolom lain
        numeric_columns = ['Close', 'Volume', 'MA5', 'MA20']  # Kolom yang mungkin mengandung nilai numerik
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].apply(clean_numeric_value)

        st.success("✅ Berhasil memuat data dari Google Drive!")
        st.info(f"Jumlah baris: {len(df)}")
        return df

    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        return None

# --- FUNGSI UTAMA APP ---
def app():
    st.title("🔪 Pisau Jatuh Screener")

    # Inisialisasi session state
    if 'screening_results' not in st.session_state:
        st.session_state.screening_results = None

    file_url = "https://docs.google.com/spreadsheets/d/1t6wgBIcPEUWMq40GdIH1GtZ8dvI9PZ2v/edit?usp=drive_link"
    df = load_google_drive_excel(file_url)

    if df is None or 'Ticker' not in df.columns:
        return

    tickers = df['Ticker'].dropna().unique().tolist()
    analysis_date = st.date_input("📅 Tanggal Analisis", value=datetime.today())

    if st.button("🔍 Mulai Screening"):
        results = []
        progress_bar = st.progress(0)
        progress_text = st.empty()

        for i, ticker in enumerate(tickers):
            data = get_stock_data(ticker, analysis_date)

            if data is not None and len(data) >= 50:
                # Hitung MA untuk harga
                data['MA5'] = data['Close'].rolling(window=5).mean()
                data['MA20'] = data['Close'].rolling(window=20).mean()
                
                # Hitung MA untuk volume
                data['Volume_MA5'] = data['Volume'].rolling(window=5).mean()
                data['Volume_MA20'] = data['Volume'].rolling(window=20).mean()
                
                if detect_pattern(data):
                    papan = df[df['Ticker'] == ticker]['Papan Pencatatan'].values[0]
                    
                    # Dapatkan data terbaru
                    latest = data.iloc[-1]
                    
                    # Dapatkan data pada tanggal analisa
                    analysis_data = data[data.index.date == analysis_date]
                    if len(analysis_data) > 0:
                        close_analisa = analysis_data['Close'].iloc[0]
                    else:
                        # Jika tidak ada data pada tanggal analisa, gunakan data terakhir sebelum tanggal itu
                        analysis_data = data[data.index.date <= analysis_date]
                        if len(analysis_data) > 0:
                            close_analisa = analysis_data['Close'].iloc[-1]
                        else:
                            close_analisa = latest['Close']
                    
                    # Hitung Volume Lot (1 lot = 100 saham)
                    volume_lot = latest['Volume'] / 100
                    volume_rp = latest['Volume'] * latest['Close']
                    
                    results.append({
                        "Ticker": ticker,
                        "Papan": papan,
                        "Close Analisa": round(close_analisa, 2),
                        "Close Last": round(latest['Close'], 2),
                        "MA5": round(latest['MA5'], 2) if not pd.isna(latest['MA5']) else 0,
                        "MA20": round(latest['MA20'], 2) if not pd.isna(latest['MA20']) else 0,
                        "Volume Lot": int(volume_lot),
                        "Volume Rp (M)": round(volume_rp / 1000000, 2),  # Dalam juta
                        "Vol MA5": int(latest['Volume_MA5'] / 100) if not pd.isna(latest['Volume_MA5']) else 0,
                        "Vol MA20": int(latest['Volume_MA20'] / 100) if not pd.isna(latest['Volume_MA20']) else 0
                    })

            progress = (i + 1) / len(tickers)
            progress_bar.progress(progress)
            progress_text.text(f"Progress: {int(progress * 100)}% - Memproses {ticker}")

        if results:
            st.session_state.screening_results = pd.DataFrame(results)
        else:
            st.warning("Tidak ada saham yang cocok dengan pola.")

    # Tampilkan hasil
    if st.session_state.screening_results is not None:
        st.subheader("✅ Saham yang Memenuhi Pola Pisau Jatuh")
        
        # Format tampilan angka
        display_df = st.session_state.screening_results.copy()
        
        # Konversi ke format string dengan pemisah ribuan
        display_df['Volume Lot'] = display_df['Volume Lot'].apply(lambda x: f"{x:,}")
        display_df['Vol MA5'] = display_df['Vol MA5'].apply(lambda x: f"{x:,}")
        display_df['Vol MA20'] = display_df['Vol MA20'].apply(lambda x: f"{x:,}")
        display_df['Volume Rp (M)'] = display_df['Volume Rp (M)'].apply(lambda x: f"Rp {x:,.2f}")
        
        # Terapkan gradasi warna pada kolom numerik
        numeric_columns = ['Close Analisa', 'Close Last', 'MA5', 'MA20']
        styled_df = display_df.style.background_gradient(
            subset=numeric_columns,
            cmap='YlOrRd'
        )
        
        st.dataframe(styled_df)

        # --- DOWNLOAD HASIL KE EXCEL ---
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            st.session_state.screening_results.to_excel(writer, sheet_name='Hasil Screening', index=False)
        
        st.download_button(
            label="📥 Unduh Hasil Screening (Excel)",
            data=output.getvalue(),
            file_name=f"pisau_jatuh_{datetime.today().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
