# pages/pisau_jatuh.py
import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
from scipy.signal import argrelextrema
import io

# --- FUNGSI DETEKSI POLA (TAHAP 1) ---
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
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None

    file_url = "https://docs.google.com/spreadsheets/d/1t6wgBIcPEUWMq40GdIH1GtZ8dvI9PZ2v/edit?usp=drive_link"
    df = load_google_drive_excel(file_url)

    if df is None or 'Ticker' not in df.columns:
        return

    tickers = df['Ticker'].dropna().unique().tolist()
    analysis_date = st.date_input("📅 Tanggal Analisis", value=datetime.today())

    if st.button("🔍 Mulai Screening"):
        # TAHAP 1: SCREENING POLA
        screening_results = []
        progress_bar = st.progress(0)
        progress_text = st.empty()

        for i, ticker in enumerate(tickers):
            data = get_stock_data(ticker, analysis_date)

            if data is not None and len(data) >= 50:
                if detect_pattern(data):
                    papan = df[df['Ticker'] == ticker]['Papan Pencatatan'].values[0]
                    screening_results.append({
                        "Ticker": ticker,
                        "Papan": papan,
                        "Data": data  # Simpan data lengkap untuk analisis tahap 2
                    })

            progress = (i + 1) / len(tickers)
            progress_bar.progress(progress)
            progress_text.text(f"Progress: {int(progress * 100)}% - Memproses {ticker}")

        st.session_state.screening_results = screening_results
        
        # TAHAP 2: ANALISIS HASIL SCREENING
        if screening_results:
            analysis_results = []
            for result in screening_results:
                data = result['Data']
                last_close = data['Close'].iloc[-1]
                
                # Hitung indikator tambahan
                analysis_data = {
                    "Ticker": result['Ticker'],
                    "Papan": result['Papan'],
                    "Harga Analisa": data['Close'].iloc[-2],  # Harga 1 hari sebelum tanggal input
                    "Harga Terakhir": last_close,
                    "Volume Rp": last_close * data['Volume'].iloc[-1],
                    "Volume Lot": data['Volume'].iloc[-1] / 100,
                    "MA 5": data['Close'].tail(5).mean(),
                    "MA 20": data['Close'].tail(20).mean()
                }
                analysis_results.append(analysis_data)
            
            st.session_state.analysis_results = pd.DataFrame(analysis_results)
            st.success(f"✅ Ditemukan {len(analysis_results)} saham yang memenuhi pola!")
        else:
            st.warning("Tidak ada saham yang cocok dengan pola.")

    # Tampilkan hasil screening
    if st.session_state.screening_results is not None:
        st.subheader("📊 Hasil Screening Tahap 1")
        screening_df = pd.DataFrame([{
            "Ticker": r["Ticker"], 
            "Papan": r["Papan"]
        } for r in st.session_state.screening_results])
        st.dataframe(screening_df)

    # Tampilkan hasil analisis
    if st.session_state.analysis_results is not None:
        st.subheader("📈 Hasil Analisis Tahap 2")
        
        # Format angka untuk tampilan
        display_df = st.session_state.analysis_results.copy()
        display_df['Harga Analisa'] = display_df['Harga Analisa'].round(2)
        display_df['Harga Terakhir'] = display_df['Harga Terakhir'].round(2)
        display_df['Volume Rp'] = display_df['Volume Rp'].apply(lambda x: f"Rp {x:,.0f}")
        display_df['Volume Lot'] = display_df['Volume Lot'].apply(lambda x: f"{x:,.0f}")
        display_df['MA 5'] = display_df['MA 5'].round(2)
        display_df['MA 20'] = display_df['MA 20'].round(2)
        
        st.dataframe(display_df)

        # --- DOWNLOAD HASIL KE EXCEL ---
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            st.session_state.analysis_results.to_excel(writer, sheet_name='Hasil Analisis', index=False)
        
        st.download_button(
            label="📥 Unduh Hasil Analisis (Excel)",
            data=output.getvalue(),
            file_name=f"pisau_jatuh_analysis_{datetime.today().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

if __name__ == "__main__":
    app()
