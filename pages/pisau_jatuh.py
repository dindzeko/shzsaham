# pages/pisau_jatuh.py
import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import io
import re
import time

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

def get_stock_data(ticker, end_date):
    try:
        stock = yf.Ticker(f"{ticker}.JK")
        start_date = end_date - timedelta(days=60)
        data = stock.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
        return data if not data.empty else None
    except Exception as e:
        st.error(f"Gagal mengambil data untuk {ticker}: {e}")
        return None

def detect_pattern(data):
    recent = data.tail(4)
    if recent.shape[0] != 4:
        return False

    c1, c2, c3, c4 = recent.iloc[0], recent.iloc[1], recent.iloc[2], recent.iloc[3]

    is_c1_bullish = c1['Close'] > c1['Open'] and (c1['Close'] - c1['Open']) > 0.02 * c1['Open']
    is_c2_bearish = c2['Close'] < c2['Open'] and c2['Close'] < c1['Close']
    is_c3_bearish = c3['Close'] < c3['Open']
    is_c4_bearish = c4['Close'] < c4['Open']
    is_uptrend = c4['Close'] < c1['Close']
    is_close_sequence = c2['Close'] > c3['Close'] > c4['Close']

    return all([
        is_c1_bullish,
        is_c2_bearish,
        is_c3_bearish,
        is_c4_bearish,
        is_uptrend,
        is_close_sequence
    ])

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
            # Dapatkan data untuk deteksi pola (hingga tanggal analisa)
            data_for_analysis = get_stock_data(ticker, analysis_date)
            
            # Dapatkan data terbaru untuk Close Last (hingga hari ini)
            try:
                stock = yf.Ticker(f"{ticker}.JK")
                data_current = stock.history(period="1d")
            except:
                data_current = None

            if data_for_analysis is not None and len(data_for_analysis) >= 20 and data_current is not None and not data_current.empty:
                # Hitung MA untuk harga pada data analisa
                data_for_analysis['MA5'] = data_for_analysis['Close'].rolling(window=5).mean()
                data_for_analysis['MA20'] = data_for_analysis['Close'].rolling(window=20).mean()
                
                # Hitung MA untuk volume pada data terkini
                data_current['Volume_MA5'] = data_current['Volume'].rolling(window=5).mean()
                data_current['Volume_MA20'] = data_current['Volume'].rolling(window=20).mean()
                
                if detect_pattern(data_for_analysis):
                    papan = df[df['Ticker'] == ticker]['Papan Pencatatan'].values[0]
                    
                    # Dapatkan data analisa (pada tanggal analisa)
                    analysis_day_data = data_for_analysis[data_for_analysis.index.date <= analysis_date]
                    if len(analysis_day_data) > 0:
                        close_analisa = analysis_day_data['Close'].iloc[-1]
                        ma5_analisa = analysis_day_data['MA5'].iloc[-1] if not pd.isna(analysis_day_data['MA5'].iloc[-1]) else 0
                        ma20_analisa = analysis_day_data['MA20'].iloc[-1] if not pd.isna(analysis_day_data['MA20'].iloc[-1]) else 0
                    else:
                        close_analisa = data_for_analysis['Close'].iloc[-1]
                        ma5_analisa = data_for_analysis['MA5'].iloc[-1] if not pd.isna(data_for_analysis['MA5'].iloc[-1]) else 0
                        ma20_analisa = data_for_analysis['MA20'].iloc[-1] if not pd.isna(data_for_analysis['MA20'].iloc[-1]) else 0
                    
                    # Dapatkan data terkini (hari ini)
                    latest = data_current.iloc[-1]
                    
                    # Hitung Volume Lot (1 lot = 100 saham)
                    volume_lot = latest['Volume'] / 100
                    volume_rp = latest['Volume'] * latest['Close']
                    
                    # Format angka
                    volume_lot_formatted = f"{volume_lot:,.0f}".replace(",", ".")
                    volume_rp_formatted = f"Rp {volume_rp/1e9:,.2f}M".replace(",", ".")
                    vol_ma5_formatted = f"{latest['Volume_MA5']/100:,.0f}".replace(",", ".") if not pd.isna(latest['Volume_MA5']) else "0"
                    vol_ma20_formatted = f"{latest['Volume_MA20']/100:,.0f}".replace(",", ".") if not pd.isna(latest['Volume_MA20']) else "0"
                    
                    results.append({
                        "Ticker": ticker,
                        "Papan": papan,
                        "Close Analisa": f"Rp {close_analisa:,.0f}".replace(",", "."),
                        "Close Last": f"Rp {latest['Close']:,.0f}".replace(",", "."),
                        "MA5": f"Rp {ma5_analisa:,.0f}".replace(",", "."),
                        "MA20": f"Rp {ma20_analisa:,.0f}".replace(",", "."),
                        "Volume Lot": volume_lot_formatted,
                        "Volume Rp": volume_rp_formatted,
                        "Vol MA5": vol_ma5_formatted,
                        "Vol MA20": vol_ma20_formatted
                    })

            progress = (i + 1) / len(tickers)
            progress_bar.progress(progress)
            progress_text.text(f"Progress: {int(progress * 100)}% - Memproses {ticker}")
            
            # Tambahkan delay untuk menghindari rate limiting
            time.sleep(0.1)

        if results:
            st.session_state.screening_results = pd.DataFrame(results)
        else:
            st.warning("Tidak ada saham yang cocok dengan pola.")

    # Tampilkan hasil
    if st.session_state.screening_results is not None:
        st.subheader("✅ Saham yang Memenuhi Pola Pisau Jatuh")
        
        # Terapkan gradasi warna pada kolom numerik
        styled_df = st.session_state.screening_results.style.background_gradient(
            subset=['Close Analisa', 'Close Last', 'MA5', 'MA20', 'Volume Lot', 'Volume Rp', 'Vol MA5', 'Vol MA20'],
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
