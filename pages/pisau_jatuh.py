# =========================================================
# === pisau_jatuh_app.py (Versi Lengkap 4 Screener)     ===
# =========================================================

import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, date
import numpy as np
import io

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
# === ANALISIS TAMBAHAN (MODE 1)
# =========================================================
def analyze_results(screening_results, analysis_date):
    enhanced_results = []

    for _, row in screening_results.iterrows():
        ticker = row['Ticker']
        try:
            stock = yf.Ticker(f"{ticker}.JK")
            end_date = datetime.today().date()
            start_date = end_date - timedelta(days=90)
            data = stock.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=(end_date + timedelta(days=1)).strftime('%Y-%m-%d')
            )
            data = _normalize_tz(data)
            if data is None or data.empty or len(data) < 20:
                continue

            last_close = float(data['Close'].iloc[-1])
            target_date = pd.Timestamp(analysis_date - timedelta(days=1)).tz_localize('Asia/Jakarta')
            trading_days_before = data[data.index <= target_date]
            if trading_days_before.empty:
                continue
            analysis_close = float(trading_days_before['Close'].iloc[-1])
            latest_volume = float(data['Volume'].iloc[-1])
            volume_rp = last_close * latest_volume

            ma5 = float(data['Close'].tail(5).mean())
            ma20 = float(data['Close'].tail(20).mean())

            enhanced_results.append({
                "Ticker": ticker,
                "Papan": row['Papan'],
                "Harga Terakhir": round(last_close, 2),
                "Harga Analisa": round(analysis_close, 2),
                "Volume (Rp)": round(volume_rp, 2),
                "MA 5": round(ma5, 2),
                "MA 20": round(ma20, 2)
            })

        except Exception as e:
            st.error(f"⚠️ Gagal menganalisis {ticker}: {str(e)}")

    return pd.DataFrame(enhanced_results)

# =========================================================
# === MODE 2: CARI TANGGAL KONFIRMASI
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

        today = datetime.today().date()
        today_hist = stock.history(
            start=(today - timedelta(days=30)).strftime('%Y-%m-%d'),
            end=(today + timedelta(days=1)).strftime('%Y-%m-%d')
        )
        today_hist = _normalize_tz(today_hist)
        if today_hist is None or today_hist.empty:
            last_close_today = np.nan
            last_close_date = None
        else:
            last_close_today = float(today_hist['Close'].iloc[-1])
            last_close_date = today_hist.index[-1].date()

        confirmations = []
        for i in range(3, len(data) - 1):
            subset = data.iloc[:i + 1]
            if len(subset) >= 50 and detect_pattern(subset):
                c4_idx = data.index[i]
                confirm_idx = data.index[i + 1]
                harga_c4 = float(data.loc[c4_idx, "Close"])

                if pd.notna(last_close_today):
                    chg_rp = last_close_today - harga_c4
                    chg_pct = (chg_rp / harga_c4) * 100 if harga_c4 != 0 else np.nan
                else:
                    chg_rp, chg_pct = np.nan, np.nan

                days_since = (last_close_date - confirm_idx.date()).days if last_close_date else None

                confirmations.append({
                    "Ticker": ticker,
                    "Tgl Konfirmasi (Hari ke-5)": confirm_idx.date(),
                    "Harga Konfirmasi (Close)": round(harga_c4, 2),
                    "Harga Today (Last Close)": round(last_close_today, 2) if pd.notna(last_close_today) else None,
                    "Perubahan (Rp)": round(chg_rp, 2) if pd.notna(chg_rp) else None,
                    "Perubahan (%)": round(chg_pct, 2) if pd.notna(chg_pct) else None,
                    "Hari sejak Konfirmasi": days_since
                })

        return pd.DataFrame(confirmations)

    except Exception as e:
        st.error(f"Gagal mencari tanggal konfirmasi untuk {ticker}: {e}")
        return pd.DataFrame()

# =========================================================
# === APLIKASI STREAMLIT (4 SCREENER)
# =========================================================
def app():
    st.title("📊 Stock Screener Suite")

    screener_choice = st.radio(
        "Pilih Screener:",
        [
            "🔪 Pisau Jatuh (Mode 1)",
            "📈 BSJP Screener",
            "💹 Swing Screener",
            "🔎 Tanggal Konfirmasi (Mode 2)"
        ],
        index=0
    )

    if 'screening_results' not in st.session_state:
        st.session_state.screening_results = None

    # =====================================================
    # === MODE 1: PISAU JATUH
    # =====================================================
    if screener_choice.startswith("🔪"):
        file_url = "https://docs.google.com/spreadsheets/d/1t6wgBIcPEUWMq40GdIH1GtZ8dvI9PZ2v/edit?usp=drive_link"
        df = load_google_drive_excel(file_url)
        if df is None: st.stop()

        tickers = df['Ticker'].dropna().unique().tolist()
        analysis_date = st.date_input("📅 Tanggal Analisis", value=datetime.today().date())

        if st.button("🔍 Mulai Screening Pisau Jatuh"):
            results = []
            progress = st.progress(0)
            for i, ticker in enumerate(tickers):
                data = get_stock_data(ticker, analysis_date)
                if data is not None and len(data) >= 50 and detect_pattern(data):
                    papan = df[df['Ticker'] == ticker]['Papan Pencatatan'].values[0]
                    results.append({"Ticker": ticker, "Papan": papan})
                progress.progress((i + 1) / len(tickers))
            if results:
                st.session_state.screening_results = analyze_results(pd.DataFrame(results), analysis_date)
            else:
                st.session_state.screening_results = pd.DataFrame()
                st.warning("Tidak ada saham yang cocok dengan pola.")

        if st.session_state.screening_results is not None and not st.session_state.screening_results.empty:
            st.subheader("✅ Hasil Screener Pisau Jatuh")
            st.dataframe(st.session_state.screening_results, use_container_width=True)
            out = io.BytesIO()
            with pd.ExcelWriter(out, engine='openpyxl') as writer:
                st.session_state.screening_results.to_excel(writer, index=False)
            st.download_button("📥 Unduh Hasil (Excel)", out.getvalue(), file_name="pisau_jatuh.xlsx")

    # =====================================================
    # === MODE 2: BSJP
    # =====================================================
    elif screener_choice.startswith("📈"):
        st.subheader("📈 BSJP Screener (Momentum & Value > 5M)")
        file_url = "https://docs.google.com/spreadsheets/d/1t6wgBIcPEUWMq40GdIH1GtZ8dvI9PZ2v/edit?usp=drive_link"
        df = load_google_drive_excel(file_url)
        if df is None: st.stop()
        tickers = df['Ticker'].dropna().unique().tolist()
        analysis_date = st.date_input("📅 Tanggal Analisis", value=datetime.today().date())

        if st.button("🚀 Jalankan Screener BSJP"):
            results = []
            for ticker in tickers:
                data = get_stock_data(ticker, analysis_date)
                if data is None or len(data) < 20: continue

                price, prev_price = data['Close'].iloc[-1], data['Close'].iloc[-2]
                ma20 = data['Close'].tail(20).mean()
                vol, prev_vol = data['Volume'].iloc[-1], data['Volume'].iloc[-2]
                value = price * vol

                if vol > prev_vol and prev_price < price and price > ma20 and value > 5_000_000_000:
                    papan = df[df['Ticker'] == ticker]['Papan Pencatatan'].values[0]
                    results.append({"Ticker": ticker, "Papan": papan, "Harga": price, "MA20": ma20, "Value": value})

            if results: st.dataframe(pd.DataFrame(results), use_container_width=True)
            else: st.warning("Tidak ada saham memenuhi kriteria BSJP.")

    # =====================================================
    # === MODE 3: SWING
    # =====================================================
    elif screener_choice.startswith("💹"):
        st.subheader("💹 Swing Screener (MA10>MA20 & Volume naik)")
        file_url = "https://docs.google.com/spreadsheets/d/1t6wgBIcPEUWMq40GdIH1GtZ8dvI9PZ2v/edit?usp=drive_link"
        df = load_google_drive_excel(file_url)
        if df is None: st.stop()
        tickers = df['Ticker'].dropna().unique().tolist()
        analysis_date = st.date_input("📅 Tanggal Analisis", value=datetime.today().date())

        if st.button("📊 Jalankan Swing Screener"):
            results = []
            for ticker in tickers:
                data = get_stock_data(ticker, analysis_date)
                if data is None or len(data) < 20: continue

                price, prev_price = data['Close'].iloc[-1], data['Close'].iloc[-2]
                ma10, ma20 = data['Close'].tail(10).mean(), data['Close'].tail(20).mean()
                vol, vol_ma20 = data['Volume'].iloc[-1], data['Volume'].tail(20).mean()

                if price > ma10 and ma10 > ma20 and prev_price < ma10 and vol > vol_ma20:
                    papan = df[df['Ticker'] == ticker]['Papan Pencatatan'].values[0]
                    results.append({"Ticker": ticker, "Papan": papan, "Harga": price, "MA10": ma10, "MA20": ma20})

            if results: st.dataframe(pd.DataFrame(results), use_container_width=True)
            else: st.warning("Tidak ada saham memenuhi kriteria Swing Screener.")

    # =====================================================
    # === MODE 4: TANGGAL KONFIRMASI
    # =====================================================
    else:
        col1, col2, col3 = st.columns([2,1,1])
        with col1:
            ticker = st.text_input("📝 Masukkan Ticker", value="HUMI").upper().strip()
        with col2:
            lookback = st.number_input("Lookback (hari)", min_value=60, max_value=720, value=180, step=10)
        with col3:
            end_date = st.date_input("Sampai Tanggal", value=datetime.today().date())

        if st.button("🔎 Cari Tanggal Konfirmasi"):
            df_conf = find_confirmation_dates_for_ticker(ticker, int(lookback), end_date)
            if df_conf.empty:
                st.info("Tidak ditemukan tanggal konfirmasi.")
            else:
                st.success(f"Ditemukan {len(df_conf)} tanggal konfirmasi.")
                df_conf = df_conf.sort_values("Tgl Konfirmasi (Hari ke-5)", ascending=False)
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
