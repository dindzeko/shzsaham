# pages/pisau_jatuh.py
import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
from scipy.signal import argrelextrema
import io

# --- FUNGSI DETEKSI POLA (TIDAK BERUBAH SAMA SEKALI) ---
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

# --- FUNGSI PENGAMBILAN DATA SAHAM ---
def get_stock_data(ticker, end_date):
    try:
        stock = yf.Ticker(f"{ticker}.JK")
        start_date = end_date - timedelta(days=90)
        data = stock.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
        return data if not data.empty else None
    except Exception as e:
        st.error(f"Gagal mengambil data untuk {ticker}: {e}")
        return None

# --- FUNGSI MEMUAT DATA DARI GOOGLE DRIVE ---
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

# --- FUNGSI UNTUK ANALISIS TAMBAHAN SETELAH SCREENING (DENGAN VALIDASI KETAT) ---
def analyze_results(screening_results, analysis_date):
    enhanced_results = []

    for _, row in screening_results.iterrows():
        ticker = row['Ticker']
        try:
            # Ambil data dari 90 hari ke belakang hingga hari ini
            stock = yf.Ticker(f"{ticker}.JK")
            end_date = datetime.today().date()
            start_date = end_date - timedelta(days=90)
            data = stock.history(start=start_date.strftime('%Y-%m-%d'), end=(end_date + timedelta(days=1)).strftime('%Y-%m-%d'))

            if data.empty or len(data) < 20:
                continue

            # ✅ Harga Terakhir = harga closing terbaru (hari perdagangan terakhir)
            last_close = data['Close'].iloc[-1]

            # ✅ Harga Analisa = harga closing 1 hari perdagangan sebelum tanggal analisis
            target_date = analysis_date - timedelta(days=1)
            
            # Cari data perdagangan terakhir sebelum target_date
            trading_days_before = data[data.index <= pd.Timestamp(target_date)]
            
            if trading_days_before.empty:
                st.warning(f"⚠️ Tidak ada data untuk {ticker} sebelum tanggal {target_date}")
                continue
                
            # Ambil harga closing dari hari perdagangan terakhir sebelum target_date
            analysis_close = trading_days_before['Close'].iloc[-1]

            # ✅ Hitung volume dan MA
            latest_volume = data['Volume'].iloc[-1]
            volume_lot = latest_volume // 100
            volume_rp = last_close * latest_volume

            ma5 = data['Close'].tail(5).mean()
            ma20 = data['Close'].tail(20).mean()

            enhanced_results.append({
                "Ticker": ticker,
                "Papan": row['Papan'],
                "Harga Terakhir": round(last_close, 2),
                "Harga Analisa": round(analysis_close, 2),
                "Volume (Rp)": round(volume_rp, 2),
                "Volume (Lot)": volume_lot,
                "MA 5": round(ma5, 2),
                "MA 20": round(ma20, 2)
            })

        except Exception as e:
            st.error(f"⚠️ Gagal menganalisis {ticker}: {str(e)}")

    return pd.DataFrame(enhanced_results)

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
                if detect_pattern(data):
                    papan = df[df['Ticker'] == ticker]['Papan Pencatatan'].values[0]
                    results.append({
                        "Ticker": ticker,
                        "Papan": papan
                    })

            progress = (i + 1) / len(tickers)
            progress_bar.progress(progress)
            progress_text.text(f"Progress: {int(progress * 100)}% - Memproses {ticker}")

        if results:
            # Tahap 2: Analisis tambahan dengan validasi ketat
            temp_df = pd.DataFrame(results)
            final_df = analyze_results(temp_df, analysis_date)
            st.session_state.screening_results = final_df
        else:
            st.warning("Tidak ada saham yang cocok dengan pola.")

    # Tampilkan hasil
    if st.session_state.screening_results is not None and not st.session_state.screening_results.empty:
        st.subheader("✅ Saham yang Memenuhi Pola Pisau Jatuh")
        st.dataframe(st.session_state.screening_results)

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
    elif st.session_state.screening_results is not None:
        st.info("Tidak ada data yang memenuhi kriteria setelah analisis.")
