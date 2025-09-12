# pisau_jatuh_app.py
import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, date
import numpy as np
from scipy.signal import argrelextrema
import io

# =========================================================
# === FUNGSI DETEKSI POLA (TIDAK BERUBAH SAMA SEKALI)  ===
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
    """Pastikan index time-series berada di Asia/Jakarta untuk konsistensi tanggal perdagangan."""
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
    """Ambil ~90 hari data hingga end_date (exclusive), return DataFrame atau None."""
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
# === ANALISIS TAMBAHAN SETELAH SCREENING (MODE 1)     ===
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

            # Harga Terakhir = closing hari perdagangan terakhir
            last_close = float(data['Close'].iloc[-1])

            # Harga Analisa = closing hari perdagangan terakhir SEBELUM tanggal analisis
            target_date = pd.Timestamp(analysis_date - timedelta(days=1)).tz_localize('Asia/Jakarta')
            trading_days_before = data[data.index <= target_date]
            if trading_days_before.empty:
                st.warning(f"⚠️ Tidak ada data untuk {ticker} sebelum tanggal {target_date.date()}")
                continue
            analysis_close = float(trading_days_before['Close'].iloc[-1])

            latest_volume = float(data['Volume'].iloc[-1])
            volume_lot = int(latest_volume // 100)
            volume_rp = last_close * latest_volume

            ma5 = float(data['Close'].tail(5).mean())
            ma20 = float(data['Close'].tail(20).mean())

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

# =========================================================
# === MODE 2: CARI TANGGAL KONFIRMASI (HARI KE-5)      ===
# =========================================================
def find_confirmation_dates_for_ticker(ticker: str,
                                       lookback_days: int = 180,
                                       end_date: date = None) -> pd.DataFrame:
    """
    Cari semua kejadian pola (4 candle terakhir valid menurut detect_pattern),
    lalu kembalikan:
      - Tanggal Konfirmasi = bar berikutnya (hari ke-5)
      - HARGA KONFIRMASI = harga penutupan C4 (candle ke-4)
      - Perubahan = dari harga C4 ke harga last close hari ini
    """
    try:
        if end_date is None:
            end_date = datetime.today().date()

        # 1) Data utama untuk deteksi pola (hingga end_date)
        start_date = end_date - timedelta(days=lookback_days)
        stock = yf.Ticker(f"{ticker}.JK")
        data = stock.history(
            start=start_date.strftime('%Y-%m-%d'),
            end=(end_date + timedelta(days=1)).strftime('%Y-%m-%d')
        )
        data = _normalize_tz(data)

        if data is None or data.empty:
            return pd.DataFrame()

        # 2) Harga last close "today" (hari perdagangan terakhir saat ini)
        today = datetime.today().date()
        today_hist = stock.history(
            start=(today - timedelta(days=30)).strftime('%Y-%m-%d'),
            end=(today + timedelta(days=1)).strftime('%Y-%m-%d')
        )
        today_hist = _normalize_tz(today_hist)
        if today_hist is None or today_hist.empty:
            last_close_today = np.nan
            last_close_date  = None
        else:
            last_close_today = float(today_hist['Close'].iloc[-1])
            last_close_date  = today_hist.index[-1].date()

        confirmations = []
        # iterasi i sebagai bar ke-4 (0-based), pastikan i+1 ada utk konfirmasi
        for i in range(3, len(data) - 1):
            subset = data.iloc[:i + 1]
            if len(subset) >= 50 and detect_pattern(subset):
                c4_idx = data.index[i]       # C4 = bar ke-4
                confirm_idx = data.index[i + 1]  # hari ke-5 (tanggal konfirmasi)
                harga_c4 = float(data.loc[c4_idx, "Close"])  # harga konfirmasi = harga C4

                # perubahan dari C4 ke harga today
                if pd.notna(last_close_today):
                    chg_rp = last_close_today - harga_c4
                    chg_pct = (chg_rp / harga_c4) * 100 if harga_c4 != 0 else np.nan
                else:
                    chg_rp = np.nan
                    chg_pct = np.nan

                days_since = (last_close_date - confirm_idx.date()).days if last_close_date else None

                confirmations.append({
                    "Ticker": ticker,
                    "Tgl Konfirmasi (Hari ke-5)": confirm_idx.date(),
                    "Harga Konfirmasi (Close)": round(harga_c4, 2),  # ← harga C4
                    "Harga Today (Last Close)": round(float(last_close_today), 2) if pd.notna(last_close_today) else None,
                    "Perubahan (Rp)": round(float(chg_rp), 2) if pd.notna(chg_rp) else None,
                    "Perubahan (%)": round(float(chg_pct), 2) if pd.notna(chg_pct) else None,
                    "Hari sejak Konfirmasi": days_since
                })

        return pd.DataFrame(confirmations)

    except Exception as e:
        st.error(f"Gagal mencari tanggal konfirmasi untuk {ticker}: {e}")
        return pd.DataFrame()

# =========================================================
# === APLIKASI STREAMLIT                                ===
# =========================================================
def app():
    st.title("🔪 Pisau Jatuh Screener")

    # Session state untuk hasil mode 1
    if 'screening_results' not in st.session_state:
        st.session_state.screening_results = None

    # Pilih mode
    mode = st.radio(
        "Pilih Mode Analisis",
        options=["🗓️ Cari Saham by Tanggal (Mode 1)", "🔎 Cari Tanggal Konfirmasi by Ticker (Mode 2)"],
        index=0
    )

    # ==========================
    # MODE 1: BY TANGGAL (ASLI)
    # ==========================
    if mode.startswith("🗓️"):
        file_url = "https://docs.google.com/spreadsheets/d/1t6wgBIcPEUWMq40GdIH1GtZ8dvI9PZ2v/edit?usp=drive_link"
        df = load_google_drive_excel(file_url)
        if df is None or 'Ticker' not in df.columns:
            st.stop()

        tickers = df['Ticker'].dropna().unique().tolist()

        default_date = datetime.today().date()
        analysis_date = st.date_input("📅 Tanggal Analisis", value=default_date)

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
                temp_df = pd.DataFrame(results)
                final_df = analyze_results(temp_df, analysis_date)
                st.session_state.screening_results = final_df
            else:
                st.session_state.screening_results = pd.DataFrame()
                st.warning("Tidak ada saham yang cocok dengan pola.")

        # Tampilkan hasil Mode 1
        if st.session_state.screening_results is not None and not st.session_state.screening_results.empty:
            st.subheader("✅ Saham yang Memenuhi Pola Pisau Jatuh (Mode 1)")
            st.dataframe(st.session_state.screening_results, use_container_width=True)

            # Download Excel
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

    # =============================================
    # MODE 2: BY TICKER → CARI TANGGAL KONFIRMASI
    # =============================================
    else:
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            ticker_input = st.text_input("📝 Masukkan Ticker (tanpa .JK)", value="HUMI").strip().upper()
        with col2:
            lookback_days = st.number_input("Lookback (hari kalender)", min_value=60, max_value=730, value=180, step=10)
        with col3:
            end_date = st.date_input("Sampai Tanggal", value=datetime.today().date())

        run = st.button("🔎 Cari Tanggal Konfirmasi (Hari ke-5)")

        if run:
            if not ticker_input:
                st.error("Mohon isi ticker terlebih dahulu.")
                st.stop()

            with st.spinner(f"Mencari tanggal konfirmasi untuk {ticker_input}.JK ..."):
                df_conf = find_confirmation_dates_for_ticker(
                    ticker=ticker_input,
                    lookback_days=int(lookback_days),
                    end_date=end_date
                )

            if df_conf.empty:
                st.info(f"Tidak ditemukan tanggal konfirmasi untuk {ticker_input}.JK dalam {lookback_days} hari ke belakang.")
            else:
                st.success(f"Ditemukan {len(df_conf)} tanggal konfirmasi untuk {ticker_input}.JK")

                # urutkan terbaru
                df_conf = df_conf.sort_values("Tgl Konfirmasi (Hari ke-5)", ascending=False).reset_index(drop=True)

                display_cols = [
                    "Ticker",
                    "Tgl Konfirmasi (Hari ke-5)",
                    "Harga Konfirmasi (Close)",   # ini = harga C4
                    "Harga Today (Last Close)",
                    "Perubahan (Rp)",
                    "Perubahan (%)",
                    "Hari sejak Konfirmasi"
                ]
                st.dataframe(df_conf[display_cols], use_container_width=True)

                st.caption(
                    "Tanggal konfirmasi = **hari perdagangan berikutnya** (hari ke-5). "
                    "**Harga Konfirmasi (Close)** = **harga penutupan candle ke-4 (C4)**. "
                    "Perubahan dihitung dari harga C4 ke harga penutupan hari perdagangan terakhir saat ini."
                )

                # Download Excel
                out2 = io.BytesIO()
                with pd.ExcelWriter(out2, engine='openpyxl') as writer:
                    df_conf[display_cols].to_excel(writer, sheet_name=f'{ticker_input}_Konfirmasi', index=False)
                st.download_button(
                    "📥 Unduh Hasil (Excel)",
                    data=out2.getvalue(),
                    file_name=f"konfirmasi_pisau_jatuh_{ticker_input}_{datetime.today().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

# Run app
if __name__ == "__main__":
    app()
