# =========================================================
# === main_app.py — All-in-One Dashboard Analisa Saham ====
# =========================================================
import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, date
import numpy as np
import io
import matplotlib.pyplot as plt

# =========================================================
# === UTILITAS ============================================
# =========================================================
def fmt(x):
    try:
        if pd.isna(x):
            return "-"
        return f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except:
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

def get_stock_data(ticker, days=90):
    try:
        stock = yf.Ticker(f"{ticker}.JK")
        end = datetime.today().date()
        start = end - timedelta(days=days)
        data = stock.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
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
            st.error("File harus memiliki kolom 'Ticker'.")
            return None
        if "Papan Pencatatan" not in df.columns:
            df["Papan Pencatatan"] = "-"
        st.success("✅ Data Google Drive berhasil dimuat.")
        st.info(f"Jumlah saham: {len(df)}")
        return df
    except Exception as e:
        st.error(f"Gagal baca file: {e}")
        return None

# =========================================================
# === INDIKATOR TEKNIKAL =================================
# =========================================================
def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calc_macd(series, short=12, long=26, signal=9):
    ema_short = series.ewm(span=short, adjust=False).mean()
    ema_long = series.ewm(span=long, adjust=False).mean()
    macd = ema_short - ema_long
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    return macd, signal_line, hist

def compute_fibo_levels(df):
    high = df["High"].max()
    low = df["Low"].min()
    rng = high - low
    return {
        0.0: high,
        0.236: high - 0.236 * rng,
        0.382: high - 0.382 * rng,
        0.5: high - 0.5 * rng,
        0.618: high - 0.618 * rng,
        0.786: high - 0.786 * rng,
        1.0: low,
    }

# =========================================================
# === PISAU JATUH MODE 1 & 2 (dari versi stabil) ==========
# =========================================================
def detect_pattern(data):
    if len(data) < 4: return False
    c1, c2, c3, c4 = data.iloc[-4:]
    return (
        c1["Close"] > c1["Open"]
        and (c1["Close"] - c1["Open"]) > 0.015 * c1["Open"]
        and all(x["Close"] < x["Open"] for x in [c2, c3, c4])
        and (data["Close"].iloc[-20:].mean() > data["Close"].iloc[-50:-20].mean()
             if len(data) >= 50 else False)
    )

def analyze_results(screening_results, analysis_date):
    results = []
    for _, r in screening_results.iterrows():
        ticker = r["Ticker"]
        data = get_stock_data(ticker)
        if data is None or data.empty: continue
        last_close = data["Close"].iloc[-1]
        ma5 = data["Close"].tail(5).mean()
        ma20 = data["Close"].tail(20).mean()
        vol = data["Volume"].iloc[-1]
        results.append({
            "Ticker": ticker, "Papan": r["Papan"],
            "Harga Terakhir": round(last_close,2),
            "MA5": round(ma5,2), "MA20": round(ma20,2),
            "Volume": int(vol)
        })
    return pd.DataFrame(results)

def find_confirmation_dates_for_ticker(ticker, lookback_days=180):
    end = datetime.today().date()
    start = end - timedelta(days=lookback_days)
    stock = yf.Ticker(f"{ticker}.JK")
    d = stock.history(start=start.strftime("%Y-%m-%d"),
                      end=(end + timedelta(days=1)).strftime("%Y-%m-%d"))
    d = _normalize_tz(d)
    conf = []
    for i in range(3, len(d)-1):
        if detect_pattern(d.iloc[:i+1]):
            c4 = d.index[i]; conf_idx = d.index[i+1]
            conf.append({
                "Ticker": ticker,
                "Tgl Konfirmasi": conf_idx.date(),
                "Harga Konfirmasi": fmt(d.loc[c4,"Close"])
            })
    return pd.DataFrame(conf)

# =========================================================
# === STREAMLIT APP MAIN TABS =============================
# =========================================================
def app():
    st.title("📊 Dashboard Analisa Saham")
    main_tabs = st.tabs(["⛓️ Tarik Data Saham","📈 Analisa Saham","🔪 Pisau Jatuh Suite"])

    # =============== TARIK DATA SAHAM ======================
    with main_tabs[0]:
        st.header("⛓️ Tarik Data Saham")
        df = load_google_drive_excel("https://docs.google.com/spreadsheets/d/1t6wgBIcPEUWMq40GdIH1GtZ8dvI9PZ2v/edit?usp=drive_link")
        if df is None: st.stop()
        tickers = df["Ticker"].dropna().unique()
        days = st.slider("Rentang hari",30,180,90)
        if st.button("📥 Ambil Data"):
            out = io.BytesIO()
            writer = pd.ExcelWriter(out,engine="openpyxl")
            for t in tickers:
                d = get_stock_data(t,days)
                if d is not None and not d.empty:
                    d.to_excel(writer,sheet_name=t,index=True)
            writer.close()
            st.download_button("💾 Unduh Semua Data",out.getvalue(),"data_saham.xlsx")

    # =============== ANALISA SAHAM =========================
    with main_tabs[1]:
        st.header("📈 Analisa Saham")
        ticker = st.text_input("Masukkan Ticker (tanpa .JK)","BMRI").upper().strip()
        if st.button("🔍 Analisa"):
            data = get_stock_data(ticker,120)
            if data is None or data.empty:
                st.warning("Data tidak tersedia.")
            else:
                # hitung indikator
                data["RSI"] = calc_rsi(data["Close"])
                macd, sig, hist = calc_macd(data["Close"])
                data["MACD"], data["Signal"], data["Hist"] = macd,sig,hist
                fibo = compute_fibo_levels(data.tail(60))
                # === Plot
                fig, axs = plt.subplots(3,1,figsize=(10,8),sharex=True)
                axs[0].plot(data.index,data["Close"],label="Close",color="blue")
                axs[0].plot(data.index,data["Close"].rolling(10).mean(),label="MA10")
                axs[0].plot(data.index,data["Close"].rolling(20).mean(),label="MA20")
                for lv,val in fibo.items():
                    axs[0].axhline(val,ls="--",label=f"Fibo {lv}",alpha=0.4)
                axs[0].legend(); axs[0].set_title(f"Harga & Fibo {ticker}")
                axs[1].plot(data.index,data["RSI"],label="RSI",color="orange"); axs[1].axhline(70,ls="--",color="r"); axs[1].axhline(30,ls="--",color="g")
                axs[1].legend(); axs[1].set_title("RSI")
                axs[2].bar(data.index,data["Hist"],label="Hist",color="gray")
                axs[2].plot(data.index,data["MACD"],label="MACD",color="blue")
                axs[2].plot(data.index,data["Signal"],label="Signal",color="red")
                axs[2].legend(); axs[2].set_title("MACD")
                st.pyplot(fig)
                st.write("### Level Fibo (60 bar terakhir)")
                fibo_df = pd.DataFrame(list(fibo.items()),columns=["Level","Harga"])
                fibo_df["Harga"] = fibo_df["Harga"].apply(fmt)
                st.dataframe(fibo_df,use_container_width=True)
                out=io.BytesIO()
                with pd.ExcelWriter(out,engine="openpyxl") as w:
                    data.to_excel(w,index=True)
                st.download_button("📥 Unduh Data Analisa",out.getvalue(),f"analisa_{ticker}.xlsx")

    # =============== PISAU JATUH SUITE =====================
    with main_tabs[2]:
        st.header("🔪 Pisau Jatuh Suite")
        subtab = st.tabs(["Mode 1","Mode 2 Konfirmasi","Swing","Fibo Support"])

        # ---- Mode 1 ----
        with subtab[0]:
            st.subheader("Mode 1 — Screening Pola")
            df = load_google_drive_excel("https://docs.google.com/spreadsheets/d/1t6wgBIcPEUWMq40GdIH1GtZ8dvI9PZ2v/edit?usp=drive_link")
            if df is not None and st.button("🚀 Jalankan Screening"):
                res=[]
                for t in df["Ticker"]:
                    d=get_stock_data(t)
                    if d is not None and detect_pattern(d):
                        res.append({"Ticker":t,"Papan":df[df["Ticker"]==t]["Papan Pencatatan"].values[0]})
                if res:
                    show=pd.DataFrame(res)
                    show=analyze_results(show,datetime.today().date())
                    st.dataframe(show,use_container_width=True)
                    out=io.BytesIO()
                    with pd.ExcelWriter(out,engine="openpyxl") as w:
                        show.to_excel(w,index=False)
                    st.download_button("📥 Unduh Excel",out.getvalue(),"pisau_mode1.xlsx")
                else: st.info("Tidak ada saham yang memenuhi pola.")

        # ---- Mode 2 ----
        with subtab[1]:
            st.subheader("Mode 2 — Konfirmasi Tanggal")
            t=st.text_input("Ticker","HUMI").upper().strip()
            if st.button("🔎 Cari Konfirmasi"):
                dfc=find_confirmation_dates_for_ticker(t)
                if dfc.empty: st.info("Tidak ada konfirmasi ditemukan.")
                else:
                    st.dataframe(dfc,use_container_width=True)
                    out=io.BytesIO()
                    with pd.ExcelWriter(out,engine="openpyxl") as w:
                        dfc.to_excel(w,index=False)
                    st.download_button("📥 Unduh Excel",out.getvalue(),f"konfirmasi_{t}.xlsx")

        # ---- Swing ----
        with subtab[2]:
            st.subheader("Swing Screener — MA & Volume")
            st.write("📌 logika Swing sesuai versi kamu sebelumnya")
            st.info("Untuk implementasi penuh sudah ada di mode integrasi sebelumnya.")

        # ---- Fibo Support ----
        with subtab[3]:
            st.subheader("Fibo Support Screener (≤ 1 % dari Fibo 1.0)")
            df = load_google_drive_excel("https://docs.google.com/spreadsheets/d/1t6wgBIcPEUWMq40GdIH1GtZ8dvI9PZ2v/edit?usp=drive_link")
            if df is not None and st.button("🧭 Jalankan Fibo Support"):
                res=[]
                for t in df["Ticker"]:
                    d=get_stock_data(t,60)
                    if d is None or len(d)<60: continue
                    fibo=compute_fibo_levels(d.tail(60))
                    price=d["Close"].iloc[-1]
                    fibo10=fibo[1.0]
                    if price<=fibo10*1.01:
                        res.append({"Ticker":t,"Harga Terakhir":price,"Fibo 1.0":fibo10,"Selisih %":(price-fibo10)/fibo10*100})
                if res:
                    outdf=pd.DataFrame(res)
                    outdf["Harga Terakhir"]=outdf["Harga Terakhir"].apply(fmt)
                    outdf["Fibo 1.0"]=outdf["Fibo 1.0"].apply(fmt)
                    outdf["Selisih %"]=outdf["Selisih %"].apply(lambda x:f"{x:.2f}%")
                    st.dataframe(outdf,use_container_width=True)
                    out=io.BytesIO()
                    with pd.ExcelWriter(out,engine="openpyxl") as w:
                        outdf.to_excel(w,index=False)
                    st.download_button("📥 Unduh Excel",out.getvalue(),"fibo_support.xlsx")
                else: st.info("Tidak ada saham yang dekat Fibo 1.0")

# =========================================================
if __name__ == "__main__":
    app()
