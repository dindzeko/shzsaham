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

    return all([
        is_c1_bullish,
        is_c2_bearish,
        is_c3_bearish,
        is_c4_bearish,
        is_uptrend,
        is_close_sequence
    ])

# --- AMBIL & PROSES DATA SAHAM ---
def get_stock_data(ticker, analysis_date):
    """
    Ambil data OHLCV dari yfinance untuk ticker (format TICK.JK).
    Mengembalikan DataFrame (index datetime tanpa tz) dengan kolom MA dan volume yg dihitung.
    """
    try:
        # pastikan analysis_date jadi pandas.Timestamp tanpa timezone
        analysis_ts = pd.to_datetime(analysis_date).normalize()

        start_date = analysis_ts - timedelta(days=120)
        # ambil sampai hari setelah analysis_date agar inclusive
        end_for_yf = (analysis_ts + timedelta(days=1)).strftime("%Y-%m-%d")

        # gunakan yf.download (lebih stabil untuk range)
        data = yf.download(ticker + ".JK", start=start_date.strftime("%Y-%m-%d"),
                           end=end_for_yf, progress=False, threads=False)

        if data is None or data.empty:
            return None

        # jika index timezone-aware, jadikan naive (tanpa tz)
        if getattr(data.index, "tz", None) is not None:
            try:
                data.index = data.index.tz_convert(None)
            except Exception:
                # kalau gagal convert, coba tz_localize None
                data.index = data.index.tz_localize(None)

        # pastikan kolom numerik
        data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
        data['Open'] = pd.to_numeric(data['Open'], errors='coerce')
        data['Volume'] = pd.to_numeric(data['Volume'], errors='coerce').fillna(0)

        # hitung MA harga & volume
        data['MA5'] = data['Close'].rolling(window=5, min_periods=1).mean()
        data['MA20'] = data['Close'].rolling(window=20, min_periods=1).mean()
        data['Volume_MA5'] = data['Volume'].rolling(window=5, min_periods=1).mean()
        data['Volume_MA20'] = data['Volume'].rolling(window=20, min_periods=1).mean()

        return data

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

# --- APLIKASI UTAMA ---
def app():
    st.title("🔪 Pisau Jatuh Screener")

    # session state
    if 'screening_results' not in st.session_state:
        st.session_state.screening_results = None

    # ambil daftar ticker dari file Google Drive (sama seperti kode lama)
    file_url = "https://docs.google.com/spreadsheets/d/1t6wgBIcPEUWMq40GdIH1GtZ8dvI9PZ2v/edit?usp=drive_link"
    df_tickers = load_google_drive_excel(file_url)
    if df_tickers is None:
        return

    tickers = df_tickers['Ticker'].dropna().unique().tolist()
    analysis_date = st.date_input("📅 Tanggal Analisis", value=datetime.today())

    if st.button("🔍 Mulai Screening"):
        results = []
        progress_bar = st.progress(0)
        progress_text = st.empty()

        # konversi sekali supaya consistent
        analysis_ts = pd.to_datetime(analysis_date).normalize()

        for i, ticker in enumerate(tickers):
            data = get_stock_data(ticker, analysis_ts)
            # jika tidak ada data atau sedikit, skip
            if data is None or len(data) < 4:
                progress = (i + 1) / len(tickers)
                progress_bar.progress(progress)
                progress_text.text(f"Progress: {int(progress * 100)}% - Memproses {ticker} (no data)")
                continue

            # deteksi pola pakai fungsi yg sama (data berisi Open/Close)
            try:
                pattern_ok = detect_pattern(data)
            except Exception:
                pattern_ok = False

            if pattern_ok:
                papan = df_tickers[df_tickers['Ticker'] == ticker]['Papan Pencatatan'].values
                papan = papan[0] if len(papan) > 0 else ""

                latest = data.iloc[-1]  # bar paling akhir (Close Last)
                # ambil close pada tanggal analisa (atau terakhir sebelum tgl itu)
                close_before = data.loc[:analysis_ts.strftime("%Y-%m-%d")]['Close']
                if not close_before.empty:
                    close_analisa = close_before.iloc[-1]
                else:
                    # fallback: gunakan earliest available
                    close_analisa = np.nan

                # hitung nilai yang diminta
                close_last = latest['Close']
                ma5 = latest.get('MA5', np.nan)
                ma20 = latest.get('MA20', np.nan)

                volume_lot = latest['Volume'] / 100  # lot
                # volume rp dalam juta (user sebelumnya pakai Juta / M), aku simpan sebagai juta
                volume_rp_million = (latest['Volume'] * latest['Close']) / 1e6

                vol_ma5_lot = data['Volume_MA5'].iloc[-1] / 100
                vol_ma20_lot = data['Volume_MA20'].iloc[-1] / 100

                results.append({
                    "Ticker": ticker,
                    "Papan": papan,
                    "Close Analisa": float(np.nan if pd.isna(close_analisa) else round(close_analisa, 2)),
                    "Close Last": float(round(close_last, 2)),
                    "MA5": float(round(ma5, 2)) if not pd.isna(ma5) else np.nan,
                    "MA20": float(round(ma20, 2)) if not pd.isna(ma20) else np.nan,
                    "Volume Lot": float(round(volume_lot, 0)),
                    "Volume Rp (M)": float(round(volume_rp_million, 2)),  # jutaan (M = juta)
                    "Vol MA5": float(round(vol_ma5_lot, 0)),
                    "Vol MA20": float(round(vol_ma20_lot, 0)),
                })

            progress = (i + 1) / len(tickers)
            progress_bar.progress(progress)
            progress_text.text(f"Progress: {int(progress * 100)}% - Memproses {ticker}")

        if results:
            st.session_state.screening_results = pd.DataFrame(results)
        else:
            st.session_state.screening_results = None
            st.warning("Tidak ada saham yang cocok dengan pola.")

    # --- TAMPILKAN HASIL (jika ada) ---
    if st.session_state.screening_results is not None:
        st.subheader("✅ Saham yang Memenuhi Pola Pisau Jatuh")

        df_results = st.session_state.screening_results.copy()

        # Format angka untuk display tetapi JANGAN ubah df_results menjadi string (agar gradient bekerja)
        format_dict = {
            "Close Analisa": "{:,.2f}",
            "Close Last": "{:,.2f}",
            "MA5": "{:,.2f}",
            "MA20": "{:,.2f}",
            "Volume Lot": "{:,.0f}",
            "Vol MA5": "{:,.0f}",
            "Vol MA20": "{:,.0f}",
            # Volume Rp (M) disajikan sebagai "Rp X,XXX.XX M"
            "Volume Rp (M)": lambda x: f"Rp {x:,.2f} M"
        }

        # Pilih kolom numerik untuk gradient
        numeric_subset = ["Close Analisa", "Close Last", "MA5", "MA20",
                          "Volume Lot", "Volume Rp (M)", "Vol MA5", "Vol MA20"]

        # Pastikan kolom ada sebelum dipakai (defensive)
        numeric_subset = [c for c in numeric_subset if c in df_results.columns]

        # Apply styling (gradient + format). Gradient uses numeric underlying values.
        styled = df_results.style.format(format_dict, na_rep="-").background_gradient(subset=numeric_subset, cmap="YlOrRd")

        st.dataframe(styled, use_container_width=True)

        # --- DOWNLOAD (export raw numeric df, tanpa styling) ---
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # tulis df_results mentah (angka) ke excel
            df_results.to_excel(writer, sheet_name='Hasil Screening', index=False)
        st.download_button(
            label="📥 Unduh Hasil Screening (Excel)",
            data=output.getvalue(),
            file_name=f"pisau_jatuh_{datetime.today().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
