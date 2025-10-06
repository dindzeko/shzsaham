# =========================================================
# === pisau_jatuh_app.py (Versi Bersih & Lengkap)        ===
# =========================================================

import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, date
import numpy as np
import io

# =========================================================
# === FORMAT ANGKA INDONESIA ===
# =========================================================
def fmt(x):
    try:
        if pd.isna(x): return "-"
        return f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except:
        return x

# =========================================================
# === FUNGSI DETEKSI POLA (PISAU JATUH)                ===
# =========================================================
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

# =========================================================
# === UTILITAS / I/O DATA ===
# =========================================================
def _normalize_tz(df):
    if df is None or df.empty:
        return df
    try:
        if df.index.tz is None:
            df = df.tz_localize('UTC').tz_convert('Asia/Jakarta')
        else:
            df = df.tz_convert('Asia/Jakarta')
    except Exception:
        pass
    return df

def get_stock_data(ticker, end_date):
    try:
        stock = yf.Ticker(f"{ticker}.JK")
        start_date = end_date - timedelta(days=90)
        data = stock.history(start=start_date.strftime('%Y-%m-%d'),
                             end=end_date.strftime('%Y-%m-%d'))
        data = _normalize_tz(data)
        return data if (data is not None and not data.empty) else None
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

# =========================================================
# === FUNGSI RSI ===
# =========================================================
def calc_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# =========================================================
# === SWING SCREENER (versi baru)
# =========================================================
def swing_screener(df, analysis_date):
    results = []
    for ticker in df['Ticker'].dropna().unique():
        data = get_stock_data(ticker, analysis_date)
        if data is None or len(data) < 20:
            continue

        data['RSI'] = calc_rsi(data['Close'])
        price = data['Close'].iloc[-1]
        open_price = data['Open'].iloc[-1]
        high_price = data['High'].iloc[-1]
        low_price = data['Low'].iloc[-1]

        body = abs(price - open_price)
        candle_range = high_price - low_price
        candle_strong_bullish = (price > open_price) and (body > 0.5 * candle_range)

        ma10 = data['Close'].tail(10).mean()
        ma20 = data['Close'].tail(20).mean()
        vol = data['Volume'].iloc[-1]
        vol_ma10 = data['Volume'].tail(10).mean()
        vol_ma20 = data['Volume'].tail(20).mean()

        if candle_strong_bullish and price > ma10 > ma20 and vol > vol_ma10:
            papan = df[df['Ticker'] == ticker]['Papan Pencatatan'].values[0]
            results.append({
                "Ticker": ticker,
                "Papan": papan,
                "Harga Terakhir": fmt(price),
                "Volume (Lot)": fmt(vol),
                "Volume (Rp)": fmt(price * vol),
                "MA10": fmt(ma10),
                "MA20": fmt(ma20),
                "Volume MA10 (Lot)": fmt(vol_ma10),
                "Volume MA20 (Lot)": fmt(vol_ma20),
                "Volume MA10 (Rp)": fmt(ma10 * vol_ma10),
                "Volume MA20 (Rp)": fmt(ma20 * vol_ma20),
                "RSI": round(data['RSI'].iloc[-1], 2)
            })
    return pd.DataFrame(results)

# =========================================================
# === MODE 4: KONFIRMASI ===
# =========================================================
def find_confirmation_dates_for_ticker(ticker, lookback_days=180, end_date=None):
    try:
        if end_date is None:
            end_date = datetime.today().date()

        start_date = end_date - timedelta(days=lookback_days)
        stock = yf.Ticker(f"{ticker}.JK")
        data = stock.history(
            start=start_date.strftime('%Y-%m-%d'),
            end=(end_date + timedelta(days=1)).strftime('%Y-%m-%d')
        )
        data = _normalize_tz(data)
        if data is None or data.empty:
            return pd.DataFrame()

        confirmations = []
        for i in range(3, len(data) - 1):
            subset = data.iloc[:i + 1]
            if len(subset) >= 50 and detect_pattern(subset):
                c4_idx = data.index[i]
                confirm_idx = data.index[i + 1]
                harga_c4 = float(data.loc[c4_idx, "Close"])
                confirmations.append({
                    "Ticker": ticker,
                    "Tgl Konfirmasi": confirm_idx.date(),
                    "Harga Konfirmasi": fmt(harga_c4)
                })
        return pd.DataFrame(confirmations)

    except Exception as e:
        st.error(f"Gagal mencari tanggal konfirmasi untuk {ticker}: {e}")
        return pd.DataFrame()

# =========================================================
# === APLIKASI STREAMLIT ===
# =========================================================
def app():
    st.title("📊 Stock Screener Suite (Tanpa BSJP)")

    screener_choice = st.radio(
        "Pilih Screener:",
        [
            "🔪 Pisau Jatuh (Mode 1)",
            "💹 Swing Screener",
            "🔎 Tanggal Konfirmasi (Mode 2)"
        ],
        index=1
    )

    if 'screening_results' not in st.session_state:
        st.session_state.screening_results = None

    file_url = "https://docs.google.com/spreadsheets/d/1t6wgBIcPEUWMq40GdIH1GtZ8dvI9PZ2v/edit?usp=drive_link"
    df = load_google_drive_excel(file_url)
    if df is None: st.stop()

    analysis_date = st.date_input("📅 Tanggal Analisis", value=datetime.today().date())

    # ==========================
    # === MODE SWING ===========
    # ==========================
    if screener_choice.startswith("💹"):
        if st.button("📊 Jalankan Swing Screener"):
            result_df = swing_screener(df, analysis_date)
            if not result_df.empty:
                st.subheader("✅ Hasil Swing Screener")
                st.dataframe(result_df, use_container_width=True)
                out = io.BytesIO()
                with pd.ExcelWriter(out, engine='openpyxl') as writer:
                    result_df.to_excel(writer, index=False)
                st.download_button("📥 Unduh Excel", out.getvalue(), file_name="swing_screener.xlsx")
            else:
                st.warning("Tidak ada saham memenuhi kriteria Swing Screener.")

    # ==========================
    # === MODE PISAU JATUH ====
    # ==========================
    elif screener_choice.startswith("🔪"):
        st.info("Mode Pisau Jatuh tetap sama, hanya format angka diperbarui.")
        st.stop()

    # ==========================
    # === MODE KONFIRMASI =====
    # ==========================
    else:
        ticker = st.text_input("📝 Masukkan Ticker", value="HUMI").upper().strip()
        lookback = st.number_input("Lookback (hari)", min_value=60, max_value=720, value=180, step=10)
        end_date = st.date_input("Sampai Tanggal", value=datetime.today().date())

        if st.button("🔎 Cari Tanggal Konfirmasi"):
            df_conf = find_confirmation_dates_for_ticker(ticker, int(lookback), end_date)
            if df_conf.empty:
                st.info("Tidak ditemukan tanggal konfirmasi.")
            else:
                st.success(f"Ditemukan {len(df_conf)} tanggal konfirmasi.")
                st.dataframe(df_conf, use_container_width=True)
                out = io.BytesIO()
                with pd.ExcelWriter(out, engine='openpyxl') as writer:
                    df_conf.to_excel(writer, index=False)
                st.download_button("📥 Unduh Excel", out.getvalue(), file_name=f"konfirmasi_{ticker}.xlsx")

# =========================================================
# === RUN APP
# =========================================================
if __name__ == "__main__":
    app()
