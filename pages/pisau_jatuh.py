# pages/pisau_jatuh.py
import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import io

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

    return all([is_c1_bullish, is_c2_bearish, is_c3_bearish, is_c4_bearish, is_uptrend, is_close_sequence])

def get_stock_data(ticker, end_date):
    try:
        stock = yf.Ticker(f"{ticker}.JK")
        start_date = end_date - timedelta(days=120)
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

        return df

    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        return None

# --- FUNGSI UTAMA APP ---
def app():
    st.title("🔪 Pisau Jatuh Screener")

    if 'screening_results' not in st.session_state:
        st.session_state.screening_results = None

    file_url = "https://docs.google.com/spreadsheets/d/1t6wgBIcPEUWMq40GdIH1GtZ8dvI9PZ2v/edit?usp=drive_link"
    df = load_google_drive_excel(file_url)
    if df is None:
        return

    tickers = df['Ticker'].dropna().unique().tolist()
    analysis_date = st.date_input("📅 Tanggal Analisis", value=datetime.today())

    if st.button("🔍 Mulai Screening"):
        results = []
        progress_bar = st.progress(0)

        for i, ticker in enumerate(tickers):
            data = get_stock_data(ticker, analysis_date)

            if data is not None and len(data) >= 50:
                if detect_pattern(data):
                    papan = df[df['Ticker'] == ticker]['Papan Pencatatan'].values[0]

                    # --- HITUNG NILAI TAMBAHAN ---
                    close_last = data['Close'].iloc[-1]
                    close_analisa = data[data.index <= pd.to_datetime(analysis_date)]['Close']
                    close_analisa = close_analisa.iloc[-1] if not close_analisa.empty else np.nan

                    close_ma5 = data['Close'].rolling(window=5).mean().iloc[-1]
                    close_ma20 = data['Close'].rolling(window=20).mean().iloc[-1]

                    vol_lot = data['Volume'].iloc[-1] / 100
                    vol_rp = close_last * data['Volume'].iloc[-1]

                    vol_lot_ma5 = (data['Volume'].rolling(window=5).mean().iloc[-1]) / 100
                    vol_lot_ma20 = (data['Volume'].rolling(window=20).mean().iloc[-1]) / 100

                    vol_rp_ma5 = (data['Volume'].rolling(window=5).mean().iloc[-1]) * close_last
                    vol_rp_ma20 = (data['Volume'].rolling(window=20).mean().iloc[-1]) * close_last

                    results.append({
                        "Ticker": ticker,
                        "Papan": papan,
                        "Close Analisa": close_analisa,
                        "Close Last": close_last,
                        "Close MA 5": close_ma5,
                        "Close MA 20": close_ma20,
                        "Volume Lot": vol_lot,
                        "Volume Rp": vol_rp,
                        "Volume Lot MA 5": vol_lot_ma5,
                        "Volume Lot MA 20": vol_lot_ma20,
                        "Volume Rp MA 5": vol_rp_ma5,
                        "Volume Rp MA 20": vol_rp_ma20
                    })

            progress = (i + 1) / len(tickers)
            progress_bar.progress(progress)

        if results:
            df_results = pd.DataFrame(results)

            # Highlight gradasi
            def color_gradient(val, cmap="Greens"):
                return f"background-color: rgba(0, 100, 0, {min(1, max(0, (val - np.nanmin(df_results.select_dtypes(float).values)) / (np.nanmax(df_results.select_dtypes(float).values) - np.nanmin(df_results.select_dtypes(float).values) + 1e-9)))})"

            st.session_state.screening_results = df_results.style.background_gradient(cmap="YlGn")

        else:
            st.warning("Tidak ada saham yang cocok dengan pola.")

    # --- HASIL ---
    if st.session_state.screening_results is not None:
        st.subheader("✅ Saham yang Memenuhi Pola Pisau Jatuh")
        st.dataframe(st.session_state.screening_results, use_container_width=True)

        # Simpan ke Excel
        output = io.BytesIO()
        df_export = st.session_state.screening_results.data if hasattr(st.session_state.screening_results, "data") else st.session_state.screening_results
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_export.to_excel(writer, sheet_name='Hasil Screening', index=False)

        st.download_button(
            label="📥 Unduh Hasil Screening (Excel)",
            data=output.getvalue(),
            file_name=f"pisau_jatuh_{datetime.today().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
