# =========================================================
# === pisau_jatuh_app.py (Versi Terbaru, 4 Mode)        ===
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
    try:
        if pd.isna(x): return "-"
        return f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except:
        return x

# =========================================================
# === UTILITAS DATA YFINANCE ==============================
# =========================================================
def _normalize_tz(df):
    if df is None or df.empty: return df
    try:
        if df.index.tz is None:
            df = df.tz_localize("UTC").tz_convert("Asia/Jakarta")
        else:
            df = df.tz_convert("Asia/Jakarta")
    except Exception: pass
    return df

def get_stock_data(ticker, end_date):
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
            st.error("File harus punya kolom 'Ticker' dan 'Papan Pencatatan'")
            return None
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
    if len(data) < 4: return False
    c1, c2, c3, c4 = data.iloc[-4:]
    return (
        c1["Close"] > c1["Open"] and
        (c1["Close"] - c1["Open"]) > 0.015 * c1["Open"] and
        all(x["Close"] < x["Open"] for x in [c2, c3, c4]) and
        data["Close"].iloc[-20:].mean() >
        data["Close"].iloc[-50:-20].mean()
        if len(data) >= 50 else False
    )

# =========================================================
# === SWING SCREENER =====================================
# =========================================================
def swing_screener(df, analysis_date):
    res=[]
    for t in df["Ticker"].dropna().unique():
        d=get_stock_data(t,analysis_date)
        if d is None or len(d)<20:continue
        d["RSI"]=calc_rsi(d["Close"])
        p=d["Close"].iloc[-1]; o=d["Open"].iloc[-1]
        r=abs(p-o)/(d["High"].iloc[-1]-d["Low"].iloc[-1])
        ma10=d["Close"].tail(10).mean(); ma20=d["Close"].tail(20).mean()
        v=d["Volume"].iloc[-1]; v10=d["Volume"].tail(10).mean()
        if p>ma10>ma20 and v>v10 and p>o and r>0.5:
            papan=df[df["Ticker"]==t]["Papan Pencatatan"].values[0]
            res.append({"Ticker":t,"Papan":papan,
                        "Harga Terakhir (num)":p,
                        "Volume (Rp) (num)":p*v,
                        "MA10 (num)":ma10,"MA20 (num)":ma20,
                        "RSI":round(d["RSI"].iloc[-1],2)})
    dfres=pd.DataFrame(res)
    if dfres.empty:return pd.DataFrame(),pd.DataFrame()
    dfres=dfres.sort_values("Volume (Rp) (num)",ascending=False)
    show=dfres.copy()
    for c in dfres.columns:
        if "(num)" in c: show[c.replace(" (num)","")]=dfres[c].apply(fmt)
    show=show[[c for c in show.columns if "(num)" not in c]]
    return show,dfres

# =========================================================
# === FIBO SUPPORT SCREENER (≤ 1 % di atas Fib 0.786) ====
# =========================================================
def fibo_screener(df,analysis_date):
    res=[]
    for t in df["Ticker"].dropna().unique():
        d=get_stock_data(t,analysis_date)
        if d is None or len(d)<60:continue
        high,low=d["Close"].max(),d["Close"].min()
        fib_0786=high-(high-low)*0.786
        fib_10=low
        p=d["Close"].iloc[-1]
        rsi=calc_rsi(d["Close"]).iloc[-1]
        if p<=fib_0786*1.01:
            papan=df[df["Ticker"]==t]["Papan Pencatatan"].values[0]
            diff=((p-fib_0786)/fib_0786)*100
            dekat="Ya" if diff<=1 else "-"
            res.append({
                "Ticker":t,"Papan":papan,
                "Harga Terakhir (num)":p,
                "Fibo 0.786 (num)":fib_0786,
                "Fibo 1.0 (num)":fib_10,
                "Selisih (%) (num)":diff,
                "📉 Dekat Support?":dekat,
                "RSI":round(rsi,2)
            })
    dfres=pd.DataFrame(res)
    if dfres.empty:return pd.DataFrame(),pd.DataFrame()
    dfres=dfres.sort_values("Selisih (%) (num)",ascending=True)
    show=dfres.copy()
    for c in dfres.columns:
        if "(num)" in c: show[c.replace(" (num)","")]=dfres[c].apply(fmt)
    show=show[[c for c in show.columns if "(num)" not in c]]
    return show,dfres

# =========================================================
# === KONFIRMASI PISAU JATUH =============================
# =========================================================
def find_confirmation_dates_for_ticker(ticker,lookback_days=180,end_date=None):
    if end_date is None:end_date=datetime.today().date()
    start=end_date-timedelta(days=lookback_days)
    stock=yf.Ticker(f"{ticker}.JK")
    d=_normalize_tz(stock.history(start=start.strftime("%Y-%m-%d"),
                                  end=(end_date+timedelta(days=1)).strftime("%Y-%m-%d")))
    if d is None or d.empty:return pd.DataFrame()
    conf=[]
    for i in range(3,len(d)-1):
        if detect_pattern(d.iloc[:i+1]):
            c4=d.index[i];conf_idx=d.index[i+1]
            conf.append({"Ticker":ticker,
                         "Tgl Konfirmasi":conf_idx.date(),
                         "Harga Konfirmasi":fmt(d.loc[c4,"Close"])})
    return pd.DataFrame(conf)

# =========================================================
# === STREAMLIT APP ======================================
# =========================================================
def app():
    st.title("📊 Stock Screener Suite (Pisau Jatuh 1-2 | Swing | Fibo)")
    choice=st.radio("Pilih Mode Screener:",
        ["🔪 Pisau Jatuh (M1)","🔎 Pisau Jatuh (M2)",
         "💹 Swing Screener","🧭 Fibo Support Screener (≤1 % Fib 0.786)"],
        index=2)

    file_url="https://docs.google.com/spreadsheets/d/1t6wgBIcPEUWMq40GdIH1GtZ8dvI9PZ2v/edit?usp=drive_link"
    df=load_google_drive_excel(file_url)
    if df is None:st.stop()
    analysis_date=st.date_input("📅 Tanggal Analisis",value=datetime.today().date())

    # --- Swing ---
    if choice.startswith("💹"):
        if st.button("📊 Jalankan Swing Screener"):
            show,raw=swing_screener(df,analysis_date)
            if not raw.empty:
                st.subheader("✅ Hasil Swing Screener (Volume Rp ↓)")
                st.dataframe(show,use_container_width=True)
                out=io.BytesIO(); 
                with pd.ExcelWriter(out,engine="openpyxl") as w: raw.to_excel(w,index=False)
                st.download_button("📥 Unduh Excel",out.getvalue(),"swing_screener.xlsx")
            else: st.warning("Tidak ada hasil.")

    # --- Fibo ---
    elif choice.startswith("🧭"):
        if st.button("🧭 Jalankan Fibo Support Screener"):
            show,raw=fibo_screener(df,analysis_date)
            if not raw.empty:
                st.subheader("✅ Saham mendekati support (≤ 1 % Fib 0.786)")
                st.dataframe(show,use_container_width=True)
                out=io.BytesIO()
                with pd.ExcelWriter(out,engine="openpyxl") as w: raw.to_excel(w,index=False)
                st.download_button("📥 Unduh Excel",out.getvalue(),"fibo_support.xlsx")
            else: st.warning("Tidak ada saham mendekati support.")

    # --- Pisau Jatuh Mode 1 ---
    elif choice.startswith("🔪"):
        st.info("Mode Pisau Jatuh 1 masih menggunakan pola deteksi lama dan hasil analisis standar.")

    # --- Pisau Jatuh Mode 2 ---
    else:
        t=st.text_input("📝 Ticker",value="HUMI").upper().strip()
        look=st.number_input("Lookback (hari)",60,720,180,10)
        endd=st.date_input("Sampai Tanggal",value=datetime.today().date())
        if st.button("🔎 Cari Tanggal Konfirmasi"):
            dfc=find_confirmation_dates_for_ticker(t,int(look),endd)
            if dfc.empty:st.info("Tidak ada konfirmasi.")
            else:
                st.success(f"Ditemukan {len(dfc)} konfirmasi.")
                st.dataframe(dfc,use_container_width=True)
                out=io.BytesIO()
                with pd.ExcelWriter(out,engine="openpyxl") as w: dfc.to_excel(w,index=False)
                st.download_button("📥 Unduh Excel",out.getvalue(),f"konfirmasi_{t}.xlsx")

# =========================================================
# === RUN =================================================
# =========================================================
if __name__=="__main__":
    app()
