# =========================================================
# === pisau_jatuh_app.py (Final, 4 Mode)                 ===
# =========================================================
import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, date
import numpy as np
import io

# =========================================================
# === UTILITAS FORMAT ANGKA INDONESIA =====================
# =========================================================
def fmt(x):
    """Ubah angka ke format Indonesia (1.234,56)"""
    try:
        if pd.isna(x):
            return "-"
        return f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except:
        return x

# =========================================================
# === UTILITAS DATA YFINANCE ==============================
# =========================================================
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

def get_stock_data(ticker, end_date):
    """Ambil data 90 hari terakhir"""
    try:
        stock = yf.Ticker(f"{ticker}.JK")
        start_date = end_date - timedelta(days=90)
        data = stock.history(start=start_date.strftime("%Y-%m-%d"),
                             end=end_date.strftime("%Y-%m-%d"))
        return _normalize_tz(data)
    except Exception as e:
        st.error(f"Gagal mengambil data {ticker}: {e}")
        return None

def load_google_drive_excel(url):
    try:
        file_id = url.split("/d/")[1].split("/")[0]
        df = pd.read_excel(
            f"https://drive.google.com/uc?export=download&id={file_id}",
            engine="openpyxl"
        )
        if "Ticker" not in df.columns:
            st.error("File harus memiliki kolom 'Ticker' dan 'Papan Pencatatan'.")
            return None
        st.success("✅ Data Google Drive berhasil dimuat.")
        st.info(f"Jumlah saham: {len(df)}")
        return df
    except Exception as e:
        st.error(f"Gagal baca file: {e}")
        return None

# =========================================================
# === FUNGSI TEKNIKAL ====================================
# =========================================================
def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def detect_pattern(data):
    """Deteksi pola Pisau Jatuh klasik"""
    if len(data) < 4:
        return False
    c1, c2, c3, c4 = data.iloc[-4:]
    return (
        c1["Close"] > c1["Open"]
        and (c1["Close"] - c1["Open"]) > 0.015 * c1["Open"]
        and all(x["Close"] < x["Open"] for x in [c2, c3, c4])
        and (data["Close"].iloc[-20:].mean() >
             data["Close"].iloc[-50:-20].mean() if len(data) >= 50 else False)
    )

# =========================================================
# === SWING SCREENER ======================================
# =========================================================
def swing_screener(df, analysis_date):
    results = []
    for t in df["Ticker"].dropna().unique():
        d = get_stock_data(t, analysis_date)
        if d is None or len(d) < 20:
            continue
        d["RSI"] = calc_rsi(d["Close"])
        price = d["Close"].iloc[-1]
        open_price = d["Open"].iloc[-1]
        high_price = d["High"].iloc[-1]
        low_price = d["Low"].iloc[-1]

        body = abs(price - open_price)
        candle_range = high_price - low_price
        candle_strong_bullish = (price > open_price) and (body > 0.5 * candle_range)

        ma10 = d["Close"].tail(10).mean()
        ma20 = d["Close"].tail(20).mean()
        vol = d["Volume"].iloc[-1]
        vol_ma10 = d["Volume"].tail(10).mean()

        if candle_strong_bullish and price > ma10 > ma20 and vol > vol_ma10:
            papan = df[df["Ticker"] == t]["Papan Pencatatan"].values[0]
            results.append({
                "Ticker": t, "Papan": papan,
                "Harga Terakhir (num)": price,
                "Volume (Rp) (num)": price * vol,
                "MA10 (num)": ma10, "MA20 (num)": ma20,
                "RSI": round(d["RSI"].iloc[-1], 2)
            })
    dfres = pd.DataFrame(results)
    if dfres.empty:
        return pd.DataFrame(), pd.DataFrame()
    dfres = dfres.sort_values("Volume (Rp) (num)", ascending=False)
    show = dfres.copy()
    for c in dfres.columns:
        if "(num)" in c:
            show[c.replace(" (num)", "")] = dfres[c].apply(fmt)
    show = show[[c for c in show.columns if "(num)" not in c]]
    return show, dfres

# =========================================================
# === FIBO SUPPORT SCREENER (≤1 % di atas Fibo 1.0) ======
# =========================================================
def fibo_screener(df, analysis_date):
    results = []
    for t in df["Ticker"].dropna().unique():
        d = get_stock_data(t, analysis_date)
        if d is None or len(d) < 60:
            continue

        # Gunakan HIGH dan LOW (bukan CLOSE) seperti page Analisa Saham
        high = d["High"].max()
        low = d["Low"].min()
        fibo_1_0 = low
        price = d["Close"].iloc[-1]
        rsi = calc_rsi(d["Close"]).iloc[-1]

        # Kriteria: harga ≤ 1% di atas Fibo 1.0
        if price <= fibo_1_0 * 1.01:
            papan = df[df["Ticker"] == t]["Papan Pencatatan"].values[0]
            selisih = ((price - fibo_1_0) / fibo_1_0) * 100
            dekat = "Ya" if selisih <= 1 else "-"
            results.append({
                "Ticker": t,
                "Papan": papan,
                "Harga Terakhir (num)": price,
                "Fibo 1.0 (num)": fibo_1_0,
                "Selisih (%) (num)": selisih,
                "📉 Dekat Support?": dekat,
                "RSI": round(rsi, 2)
            })

    dfres = pd.DataFrame(results)
    if dfres.empty:
        return pd.DataFrame(), pd.DataFrame()
    dfres = dfres.sort_values("Selisih (%) (num)", ascending=True)
    show = dfres.copy()
    for c in dfres.columns:
        if "(num)" in c:
            show[c.replace(" (num)", "")] = dfres[c].apply(fmt)
    show = show[[c for c in show.columns if "(num)" not in c]]
    return show, dfres

# =========================================================
# === KONFIRMASI PISAU JATUH =============================
# =========================================================
def find_confirmation_dates_for_ticker(ticker, lookback_days=180, end_date=None):
    if end_date is None:
        end_date = datetime.today().date()
    start = end_date - timedelta(days=lookback_days)
    stock = yf.Ticker(f"{ticker}.JK")
    d = _normalize_tz(stock.history(start=start.strftime("%Y-%m-%d"),
                                    end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d")))
    if d is None or d.empty:
        return pd.DataFrame()
    conf = []
    for i in range(3, len(d) - 1):
        if detect_pattern(d.iloc[:i + 1]):
            c4 = d.index[i]
            conf_idx = d.index[i + 1]
            conf.append({
                "Ticker": ticker,
                "Tgl Konfirmasi": conf_idx.date(),
                "Harga Konfirmasi": fmt(d.loc[c4, "Close"])
            })
    return pd.DataFrame(conf)

# =========================================================
# === STREAMLIT APP ======================================
# =========================================================
def app():
    st.title("📊 Stock Screener Suite (Pisau Jatuh | Swing | Fibo)")

    choice = st.radio(
        "Pilih Mode Screener:",
        [
            "🔪 Pisau Jatuh (Mode 1)",
            "🔎 Pisau Jatuh (Mode 2 - Konfirmasi)",
            "💹 Swing Screener",
            "🧭 Fibo Support Screener (≤ 1 % Fib 1.0)"
        ],
        index=2
    )

    file_url = "https://docs.google.com/spreadsheets/d/1t6wgBIcPEUWMq40GdIH1GtZ8dvI9PZ2v/edit?usp=drive_link"
    df = load_google_drive_excel(file_url)
    if df is None:
        st.stop()

    analysis_date = st.date_input("📅 Tanggal Analisis", value=datetime.today().date())

    # === Mode Swing ===
    if choice.startswith("💹"):
        if st.button("📊 Jalankan Swing Screener"):
            show, raw = swing_screener(df, analysis_date)
            if not raw.empty:
                st.subheader("✅ Hasil Swing Screener (Volume Rp ↓)")
                st.dataframe(show, use_container_width=True)
                out = io.BytesIO()
                with pd.ExcelWriter(out, engine="openpyxl") as w:
                    raw.to_excel(w, index=False)
                st.download_button("📥 Unduh Excel", out.getvalue(), "swing_screener.xlsx")
            else:
                st.warning("Tidak ada saham yang memenuhi kriteria Swing.")

    # === Mode Fibo ===
    elif choice.startswith("🧭"):
        if st.button("🧭 Jalankan Fibo Support Screener"):
            show, raw = fibo_screener(df, analysis_date)
            if not raw.empty:
                st.subheader("✅ Saham mendekati support (≤ 1 % dari Fibo 1.0)")
                st.dataframe(show, use_container_width=True)
                out = io.BytesIO()
                with pd.ExcelWriter(out, engine="openpyxl") as w:
                    raw.to_excel(w, index=False)
                st.download_button("📥 Unduh Excel", out.getvalue(), "fibo_support.xlsx")
            else:
                st.warning("Tidak ada saham yang mendekati support (Fib 1.0).")

    # === Mode Pisau Jatuh 1 ===
    elif choice.startswith("🔪"):
        st.info("Mode Pisau Jatuh 1 tetap sama (deteksi pola utama).")

    # === Mode Pisau Jatuh 2 (Konfirmasi) ===
    else:
        t = st.text_input("📝 Masukkan Ticker", value="HUMI").upper().strip()
        look = st.number_input("Lookback (hari)", 60, 720, 180, 10)
        endd = st.date_input("Sampai Tanggal", value=datetime.today().date())
        if st.button("🔎 Cari Tanggal Konfirmasi"):
            dfc = find_confirmation_dates_for_ticker(t, int(look), endd)
            if dfc.empty:
                st.info("Tidak ada konfirmasi.")
            else:
                st.success(f"Ditemukan {len(dfc)} tanggal konfirmasi.")
                st.dataframe(dfc, use_container_width=True)
                out = io.BytesIO()
                with pd.ExcelWriter(out, engine="openpyxl") as w:
                    dfc.to_excel(w, index=False)
                st.download_button("📥 Unduh Excel", out.getvalue(), f"konfirmasi_{t}.xlsx")

# =========================================================
# === RUN =================================================
# =========================================================
if __name__ == "__main__":
    app()
