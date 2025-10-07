# =========================================================
# === pisau_jatuh_app.py — Versi Final (4 Tab Aktif) ======
# =========================================================
import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, date
import numpy as np
import io

# =========================================================
# === UTILITAS FORMAT & I/O ===============================
# =========================================================
def fmt(x):
    """Format angka Indonesia 1.234,56"""
    try:
        if pd.isna(x):
            return "-"
        return f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return x

def _normalize_tz(df):
    if df is None or df.empty:
        return df
    try:
        if df.index.tz is None:
            df = df.tz_localize("UTC").tz_convert("Asia/Jakarta")
        else:
            df = df.tz_convert("Asia/Jakarta")
    except Exception:
        pass
    return df

def get_stock_data(ticker: str, end_date: date, days: int = 90):
    """Ambil data harian sampai end_date (exclusive)"""
    try:
        stock = yf.Ticker(f"{ticker}.JK")
        start_date = end_date - timedelta(days=days)
        data = stock.history(
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d")
        )
        data = _normalize_tz(data)
        return data if (data is not None and not data.empty) else None
    except Exception as e:
        st.error(f"Gagal mengambil data untuk {ticker}: {e}")
        return None

def load_google_drive_excel(file_url: str):
    try:
        file_id = file_url.split("/d/")[1].split("/")[0]
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        df = pd.read_excel(download_url, engine="openpyxl")
        if "Ticker" not in df.columns or "Papan Pencatatan" not in df.columns:
            st.error("Kolom 'Ticker' dan 'Papan Pencatatan' harus ada di file Excel.")
            return None
        st.success("✅ Data Google Drive berhasil dimuat.")
        st.info(f"Jumlah saham: {len(df)}")
        return df
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        return None

# =========================================================
# === INDIKATOR / TEKNIKAL ================================
# =========================================================
def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_fibo_levels(last60: pd.DataFrame) -> dict:
    high = float(last60["High"].max())
    low = float(last60["Low"].min())
    rng = high - low
    return {
        0.0: high,
        0.236: high - 0.236 * rng,
        0.382: high - 0.382 * rng,
        0.5:  high - 0.5 * rng,
        0.618: high - 0.618 * rng,
        0.786: high - 0.786 * rng,
        1.0: low,
    }

# =========================================================
# === DETEKSI POLA (VERSI KAMU — STABIL) ==================
# =========================================================
def detect_pattern(data: pd.DataFrame) -> bool:
    if data is None or len(data) < 4:
        return False
    recent = data.tail(4)
    if len(recent) < 4:
        return False

    c1, c2, c3, c4 = recent.iloc[0], recent.iloc[1], recent.iloc[2], recent.iloc[3]

    is_c1_bullish = c1["Close"] > c1["Open"] and (c1["Close"] - c1["Open"]) > 0.015 * c1["Open"]
    is_c2_bearish = c2["Close"] < c2["Open"] and c2["Close"] < c1["Close"]
    is_c3_bearish = c3["Close"] < c3["Open"]
    is_c4_bearish = c4["Close"] < c4["Open"]
    is_uptrend = data["Close"].iloc[-20:].mean() > data["Close"].iloc[-50:-20].mean() if len(data) >= 50 else False
    is_close_sequence = c2["Close"] > c3["Close"] > c4["Close"]

    return all([
        is_c1_bullish,
        is_c2_bearish,
        is_c3_bearish,
        is_c4_bearish,
        is_uptrend,
        is_close_sequence
    ])

# =========================================================
# === ANALISIS LANJUT (MODE 1) ============================
# =========================================================
def analyze_results(screening_results: pd.DataFrame, analysis_date: date) -> pd.DataFrame:
    enhanced_results = []
    for _, row in screening_results.iterrows():
        ticker = row["Ticker"]
        try:
            stock = yf.Ticker(f"{ticker}.JK")
            end_date = datetime.today().date()
            start_date = end_date - timedelta(days=90)
            data = stock.history(
                start=start_date.strftime("%Y-%m-%d"),
                end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d")
            )
            data = _normalize_tz(data)
            if data is None or data.empty or len(data) < 20:
                continue

            last_close = float(data["Close"].iloc[-1])

            # harga analisa = closing hari perdagangan terakhir SEBELUM analysis_date
            target_date = pd.Timestamp(analysis_date - timedelta(days=1)).tz_localize("Asia/Jakarta")
            trading_days_before = data[data.index <= target_date]
            if trading_days_before.empty:
                continue
            analysis_close = float(trading_days_before["Close"].iloc[-1])

            latest_volume = float(data["Volume"].iloc[-1])
            volume_lot = int(latest_volume // 100)
            volume_rp = last_close * latest_volume

            ma5 = float(data["Close"].tail(5).mean())
            ma20 = float(data["Close"].tail(20).mean())

            enhanced_results.append({
                "Ticker": ticker,
                "Papan": row["Papan"],
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
# === MODE 2: KONFIRMASI (HARI KE-5) ======================
# =========================================================
def find_confirmation_dates_for_ticker(ticker: str,
                                       lookback_days: int = 180,
                                       end_date: date | None = None) -> pd.DataFrame:
    try:
        if end_date is None:
            end_date = datetime.today().date()

        start_date = end_date - timedelta(days=lookback_days)
        stock = yf.Ticker(f"{ticker}.JK")
        data = stock.history(
            start=start_date.strftime("%Y-%m-%d"),
            end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d")
        )
        data = _normalize_tz(data)
        if data is None or data.empty:
            return pd.DataFrame()

        # harga last close terbaru (untuk hitung perubahan)
        today = datetime.today().date()
        today_hist = stock.history(
            start=(today - timedelta(days=30)).strftime("%Y-%m-%d"),
            end=(today + timedelta(days=1)).strftime("%Y-%m-%d")
        )
        today_hist = _normalize_tz(today_hist)
        if today_hist is None or today_hist.empty:
            last_close_today = np.nan
            last_close_date = None
        else:
            last_close_today = float(today_hist["Close"].iloc[-1])
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
                    "Harga Konfirmasi (Close)": round(harga_c4, 2),  # harga C4
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
# === SWING SCREENER ======================================
# =========================================================
def swing_screener(df_list: pd.DataFrame, analysis_date: date):
    results = []
    for t in df_list["Ticker"].dropna().unique():
        d = get_stock_data(t, analysis_date, days=90)
        if d is None or len(d) < 20:
            continue

        price = float(d["Close"].iloc[-1])
        open_price = float(d["Open"].iloc[-1])
        high_price = float(d["High"].iloc[-1])
        low_price = float(d["Low"].iloc[-1])

        # candle bullish kuat: body > 50% range
        body = abs(price - open_price)
        candle_range = max(high_price - low_price, 1e-9)
        candle_strong_bullish = (price > open_price) and (body > 0.5 * candle_range)

        ma10 = float(d["Close"].tail(10).mean())
        ma20 = float(d["Close"].tail(20).mean())
        vol = float(d["Volume"].iloc[-1])
        vol_ma10 = float(d["Volume"].tail(10).mean())
        rsi_last = float(calc_rsi(d["Close"]).iloc[-1])

        if candle_strong_bullish and ma10 > ma20 and vol > vol_ma10:
            papan = df_list[df_list["Ticker"] == t]["Papan Pencatatan"].values[0]
            results.append({
                "Ticker": t,
                "Papan": papan,
                "Harga Terakhir (num)": price,
                "MA10 (num)": ma10,
                "MA20 (num)": ma20,
                "Volume (Rp) (num)": price * vol,
                "RSI": round(rsi_last, 2),
            })

    raw = pd.DataFrame(results)
    if raw.empty:
        return pd.DataFrame(), pd.DataFrame()

    raw = raw.sort_values("Volume (Rp) (num)", ascending=False)
    show = raw.copy()
    for c in list(raw.columns):
        if "(num)" in c:
            show[c.replace(" (num)", "")] = raw[c].apply(fmt)
    show = show[[c for c in show.columns if "(num)" not in c]]
    return show, raw

# =========================================================
# === FIBO SUPPORT SCREENER (≤ 1% dari 1.0) ===============
# =========================================================
def fibo_screener(df_list: pd.DataFrame, analysis_date: date):
    results = []
    for t in df_list["Ticker"].dropna().unique():
        d = get_stock_data(t, analysis_date, days=90)
        if d is None or len(d) < 60:
            continue

        last60 = d.tail(60)
        fibo = compute_fibo_levels(last60)
        price = float(d["Close"].iloc[-1])
        rsi_last = float(calc_rsi(d["Close"]).iloc[-1])

        fibo_1_0 = float(fibo[1.0])  # low
        if price <= fibo_1_0 * 1.01:  # dekat support <= 1%
            papan = df_list[df_list["Ticker"] == t]["Papan Pencatatan"].values[0]
            selisih_pct = ((price - fibo_1_0) / fibo_1_0) * 100 if fibo_1_0 else np.nan
            results.append({
                "Ticker": t,
                "Papan": papan,
                "Harga Terakhir (num)": price,
                "Fibo 1.0 (num)": fibo_1_0,
                "Selisih (%) (num)": selisih_pct,
                "RSI": round(rsi_last, 2),
            })

    raw = pd.DataFrame(results)
    if raw.empty:
        return pd.DataFrame(), pd.DataFrame()

    raw = raw.sort_values("Selisih (%) (num)", ascending=True)
    show = raw.copy()
    for c in list(raw.columns):
        if "(num)" in c:
            show[c.replace(" (num)", "")] = raw[c].apply(fmt)
    show = show[[c for c in show.columns if "(num)" not in c]]
    return show, raw

# =========================================================
# === STREAMLIT APP =======================================
# =========================================================
def app():
    st.title("🔪 Pisau Jatuh Suite")

    tabs = st.tabs([
        "🗓️ Mode 1: Screening Pola",
        "🔎 Mode 2: Konfirmasi",
        "💹 Swing",
        "🧭 Fibo Support"
    ])

    # ---------------------------
    # MODE 1 — Screening Pola
    # ---------------------------
    with tabs[0]:
        st.subheader("Mode 1 — Screening Pola Pisau Jatuh")
        file_url = "https://docs.google.com/spreadsheets/d/1t6wgBIcPEUWMq40GdIH1GtZ8dvI9PZ2v/edit?usp=drive_link"
        df = load_google_drive_excel(file_url)
        if df is None:
            st.stop()

        tickers = df["Ticker"].dropna().unique().tolist()
        analysis_date = st.date_input("📅 Tanggal Analisis", value=datetime.today().date())

        if st.button("🚀 Jalankan Screening"):
            results = []
            progress_bar = st.progress(0)
            progress_text = st.empty()
            for i, t in enumerate(tickers):
                d = get_stock_data(t, analysis_date, days=90)
                if d is not None and len(d) >= 50 and detect_pattern(d):
                    papan = df[df["Ticker"] == t]["Papan Pencatatan"].values[0]
                    results.append({"Ticker": t, "Papan": papan})
                progress = (i + 1) / len(tickers)
                progress_bar.progress(progress)
                progress_text.text(f"Progress: {int(progress*100)}% - {t}")

            if results:
                final_df = analyze_results(pd.DataFrame(results), analysis_date)
                st.success(f"✅ Ditemukan {len(final_df)} saham memenuhi pola.")
                st.dataframe(final_df, use_container_width=True)
                out = io.BytesIO()
                with pd.ExcelWriter(out, engine="openpyxl") as w:
                    final_df.to_excel(w, sheet_name="Hasil Screening", index=False)
                st.download_button("📥 Unduh Hasil (Excel)", out.getvalue(),
                                   file_name=f"pisau_jatuh_mode1_{datetime.today().strftime('%Y%m%d')}.xlsx")
            else:
                st.info("Tidak ada saham yang memenuhi pola.")

    # ---------------------------
    # MODE 2 — Konfirmasi
    # ---------------------------
    with tabs[1]:
        st.subheader("Mode 2 — Cari Tanggal Konfirmasi (Hari ke-5)")
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            ticker_input = st.text_input("📝 Ticker (tanpa .JK)", value="HUMI").strip().upper()
        with col2:
            lookback = st.number_input("Lookback (hari)", min_value=60, max_value=720, value=180, step=10)
        with col3:
            end_date = st.date_input("Sampai Tanggal", value=datetime.today().date())

        if st.button("🔍 Cari Tanggal Konfirmasi"):
            if not ticker_input:
                st.error("Mohon isi ticker terlebih dahulu.")
            else:
                with st.spinner(f"Mencari tanggal konfirmasi untuk {ticker_input}.JK ..."):
                    dfc = find_confirmation_dates_for_ticker(ticker_input, int(lookback), end_date)

                if dfc.empty:
                    st.info(f"Tidak ada konfirmasi untuk {ticker_input}.JK.")
                else:
                    st.success(f"Ditemukan {len(dfc)} tanggal konfirmasi.")
                    dfc = dfc.sort_values("Tgl Konfirmasi (Hari ke-5)", ascending=False).reset_index(drop=True)
                    show_cols = [
                        "Ticker", "Tgl Konfirmasi (Hari ke-5)", "Harga Konfirmasi (Close)",
                        "Harga Today (Last Close)", "Perubahan (Rp)", "Perubahan (%)", "Hari sejak Konfirmasi"
                    ]
                    st.dataframe(dfc[show_cols], use_container_width=True)
                    out = io.BytesIO()
                    with pd.ExcelWriter(out, engine="openpyxl") as w:
                        dfc[show_cols].to_excel(w, index=False)
                    st.download_button("📥 Unduh Hasil (Excel)", out.getvalue(),
                                       file_name=f"konfirmasi_{ticker_input}.xlsx")

    # ---------------------------
    # SWING — Momentum Bullish
    # ---------------------------
    with tabs[2]:
        st.subheader("💹 Swing Screener — Bullish Momentum")
        file_url = "https://docs.google.com/spreadsheets/d/1t6wgBIcPEUWMq40GdIH1GtZ8dvI9PZ2v/edit?usp=drive_link"
        df = load_google_drive_excel(file_url)
        if df is None:
            st.stop()
        analysis_date = st.date_input("📅 Tanggal Analisis (Swing)", value=datetime.today().date())
        if st.button("📊 Jalankan Swing Screener"):
            show, raw = swing_screener(df, analysis_date)
            if raw.empty:
                st.info("Tidak ada saham yang memenuhi kriteria Swing.")
            else:
                st.success(f"✅ {len(raw)} saham memenuhi kriteria Swing.")
                st.dataframe(show, use_container_width=True)
                out = io.BytesIO()
                with pd.ExcelWriter(out, engine="openpyxl") as w:
                    raw.to_excel(w, index=False)
                st.download_button("📥 Unduh Hasil (Excel)", out.getvalue(), "swing_screener.xlsx")

    # ---------------------------
    # FIBO — Dekat Support 1.0
    # ---------------------------
    with tabs[3]:
        st.subheader("🧭 Fibo Support Screener (≤ 1% dari Fibo 1.0, 60 bar)")
        file_url = "https://docs.google.com/spreadsheets/d/1t6wgBIcPEUWMq40GdIH1GtZ8dvI9PZ2v/edit?usp=drive_link"
        df = load_google_drive_excel(file_url)
        if df is None:
            st.stop()
        analysis_date = st.date_input("📅 Tanggal Analisis (Fibo)", value=datetime.today().date())
        if st.button("🧭 Jalankan Fibo Support Screener"):
            show, raw = fibo_screener(df, analysis_date)
            if raw.empty:
                st.info("Tidak ada saham yang mendekati support Fibo 1.0.")
            else:
                st.success(f"✅ {len(raw)} saham dekat dengan Fibo 1.0 (≤ 1%).")
                st.dataframe(show, use_container_width=True)
                out = io.BytesIO()
                with pd.ExcelWriter(out, engine="openpyxl") as w:
                    raw.to_excel(w, index=False)
                st.download_button("📥 Unduh Hasil (Excel)", out.getvalue(), "fibo_support.xlsx")

# =========================================================
if __name__ == "__main__":
    app()
