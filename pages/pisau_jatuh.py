import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import io

# --- FUNGSI DETEKSI POLA (DIPERBAIKI) ---
def detect_pattern(data, full_data=None):
    """
    Fungsi untuk mendeteksi pola pisau jatuh
    data: 4 hari terakhir untuk pola candle
    full_data: data lengkap untuk menghitung moving average (opsional)
    """
    if len(data) < 4:
        return False
    
    # Jika full_data tidak diberikan, gunakan data yang ada
    if full_data is None:
        full_data = data
        
    recent = data.tail(4)
    c1, c2, c3, c4 = recent.iloc[0], recent.iloc[1], recent.iloc[2], recent.iloc[3]

    # Kriteria pola pisau jatuh
    is_c1_bullish = c1['Close'] > c1['Open'] and (c1['Close'] - c1['Open']) > 0.015 * c1['Open']
    is_c2_bearish = c2['Close'] < c2['Open'] and c2['Close'] < c1['Close']
    is_c3_bearish = c3['Close'] < c3['Open']
    is_c4_bearish = c4['Close'] < c4['Open']
    
    # Perbaikan: Gunakan full_data untuk menghitung trend
    if len(full_data) >= 50:
        ma_short = full_data['Close'].iloc[-20:].mean()
        ma_long = full_data['Close'].iloc[-50:-20].mean()
        is_uptrend = ma_short > ma_long
    else:
        is_uptrend = False
        
    # Perbaikan: Kriteria urutan harga close yang lebih tepat
    is_close_sequence = c2['Close'] > c3['Close'] and c3['Close'] > c4['Close']

    return all([
        is_c1_bullish,
        is_c2_bearish,
        is_c3_bearish,
        is_c4_bearish,
        is_uptrend,
        is_close_sequence
    ])

# --- FUNGSI PENGAMBILAN DATA (MODE 1) ---
def get_stock_data(ticker, end_date):
    try:
        stock = yf.Ticker(f"{ticker}.JK")
        # Ambil data lebih banyak untuk menghitung moving average
        start_date = end_date - timedelta(days=60)
        data = stock.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
        data = data[data.index.weekday < 5]
        if len(data) >= 4:
            # Kembalikan data lengkap untuk analisis trend
            return data
        return None
    except Exception as e:
        st.error(f"Gagal mengambil data untuk {ticker}: {e}")
        return None

# --- FUNGSI MEMUAT EXCEL GOOGLE DRIVE ---
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

# --- ✅ PERBAIKAN MODE 2: SIMPAN pattern_date & confirmation_date ---
def find_confirmation_dates_for_ticker(ticker, start_date, end_date):
    try:
        stock = yf.Ticker(f"{ticker}.JK")
        # Ambil data yang lebih panjang untuk analisis trend
        data_start_date = start_date - timedelta(days=60)
        data = stock.history(start=data_start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
        
        if data.empty or len(data) < 4:
            return []

        # Bersihkan index
        data.index = pd.to_datetime(data.index).tz_localize(None)
        data = data[data.index.weekday < 5]  # Hanya hari kerja

        results = []
        # Perbaikan: Iterasi melalui data dengan window 4 hari
        for i in range(4, len(data)):
            window_data = data.iloc[i-4:i].copy()  # Ambil 4 hari untuk pola
            
            # Pastikan hari berurutan (tidak ada missing day)
            date_diff = (window_data.index[-1] - window_data.index[0]).days
            if date_diff > 4:
                continue
                
            # Gunakan data lengkap untuk analisis trend
            full_data_for_trend = data.iloc[:i]  # Semua data sampai hari ke-i
            
            if detect_pattern(window_data, full_data_for_trend):
                pattern_date = window_data.index[-1].date()
                
                # Cari hari trading berikutnya
                next_trading_day = pattern_date + timedelta(days=1)
                while next_trading_day <= end_date.date():
                    if next_trading_day.weekday() < 5:
                        # Pastikan hari tersebut ada dalam data
                        if next_trading_day in data.index.date:
                            results.append({
                                "pattern_date": pattern_date,
                                "confirmation_date": next_trading_day
                            })
                        break
                    next_trading_day += timedelta(days=1)
                    
        return results
    except Exception as e:
        st.error(f"Gagal menganalisis {ticker}: {e}")
        return []

# --- ANALISA HASIL MODE 2 ---
def analyze_pattern_dates(ticker, results):
    enhanced_results = []
    for r in results:
        pattern_date = r["pattern_date"]
        conf_date = r["confirmation_date"]
        try:
            stock = yf.Ticker(f"{ticker}.JK")
            start_date = pattern_date - timedelta(days=90)
            end_date = conf_date + timedelta(days=1)
            data = stock.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
            
            if data.empty or len(data) < 20:
                continue

            # Pastikan tanggal konfirmasi ada dalam data
            if pd.Timestamp(conf_date) not in data.index:
                continue
                
            last_close = data.loc[pd.Timestamp(conf_date)]['Close']
            
            # Dapatkan harga penutupan hari sebelumnya
            prev_day = data[data.index < pd.Timestamp(conf_date)]
            if prev_day.empty:
                continue
            analysis_close = prev_day['Close'].iloc[-1]

            latest_volume = data.loc[pd.Timestamp(conf_date)]['Volume']
            volume_lot = latest_volume // 100
            volume_rp = last_close * latest_volume

            # Hitung moving averages
            ma5 = data['Close'].tail(5).mean()
            ma20 = data['Close'].tail(20).mean()
            ma200 = data['Close'].tail(200).mean() if len(data) >= 200 else data['Close'].mean()

            enhanced_results.append({
                "Ticker": ticker,
                "Tanggal Pola": pattern_date,
                "Tanggal Konfirmasi": conf_date,
                "Harga Terakhir": round(last_close, 2),
                "Harga Analisa": round(analysis_close, 2),
                "Volume (Rp)": round(volume_rp, 2),
                "Volume (Lot)": int(volume_lot),
                "MA 5": round(ma5, 2),
                "MA 20": round(ma20, 2),
                "MA 200": round(ma200, 2)
            })
        except Exception as e:
            st.error(f"⚠️ Gagal menganalisis {ticker} pada {conf_date}: {str(e)}")
            continue
    return pd.DataFrame(enhanced_results)

# --- ANALISA HASIL MODE 1 ---
def analyze_results(screening_results, analysis_date):
    enhanced_results = []
    for _, row in screening_results.iterrows():
        ticker = row['Ticker']
        try:
            stock = yf.Ticker(f"{ticker}.JK")
            end_date = datetime.today().date()
            start_date = end_date - timedelta(days=90)
            data = stock.history(start=start_date.strftime('%Y-%m-%d'), end=(end_date + timedelta(days=1)).strftime('%Y-%m-%d'))
            
            if data.empty or len(data) < 20:
                continue

            last_close = data['Close'].iloc[-1]
            target_date = pd.Timestamp(analysis_date - timedelta(days=1))
            
            # Pastikan tanggal analisis ada dalam data
            if target_date.date() not in data.index.date:
                continue
                
            analysis_close = data.loc[target_date]['Close']

            latest_volume = data['Volume'].iloc[-1]
            volume_lot = latest_volume // 100
            volume_rp = last_close * latest_volume

            ma5 = data['Close'].tail(5).mean()
            ma20 = data['Close'].tail(20).mean()
            ma200 = data['Close'].tail(200).mean() if len(data) >= 200 else data['Close'].mean()

            enhanced_results.append({
                "Ticker": ticker,
                "Papan": row['Papan'],
                "Harga Terakhir": round(last_close, 2),
                "Harga Analisa": round(analysis_close, 2),
                "Volume (Rp)": round(volume_rp, 2),
                "Volume (Lot)": volume_lot,
                "MA 5": round(ma5, 2),
                "MA 20": round(ma20, 2),
                "MA 200": round(ma200, 2)
            })
        except Exception as e:
            st.error(f"⚠️ Gagal menganalisis {ticker}: {str(e)}")
    return pd.DataFrame(enhanced_results)

# --- APP ---
def app():
    st.title("🔪 Pisau Jatuh Screener")
    if 'screening_results' not in st.session_state:
        st.session_state.screening_results = None

    file_url = "https://docs.google.com/spreadsheets/d/1t6wgBIcPEUWMq40GdIH1GtZ8dvI9PZ2v/edit?usp=drive_link"
    df = load_google_drive_excel(file_url)
    if df is None or 'Ticker' not in df.columns:
        return

    mode = st.radio("Pilih Mode Screening:", ("📅 Screening by Tanggal", "🔍 Cari Tanggal Konfirmasi untuk Ticker Tertentu"))

    if mode == "📅 Screening by Tanggal":
        tickers = df['Ticker'].dropna().unique().tolist()
        analysis_date = st.date_input("📅 Tanggal Analisis", value=datetime.today())
        
        # Tambahkan opsi untuk menyesuaikan kriteria
        st.subheader("Kriteria Pola Pisau Jatuh")
        min_volume_lot = st.number_input("Volume Minimum (Lot)", min_value=100, value=1000, step=100)
        
        if st.button("🔍 Mulai Screening"):
            results = []
            progress_bar = st.progress(0)
            progress_text = st.empty()
            
            for i, ticker in enumerate(tickers):
                data = get_stock_data(ticker, analysis_date)
                if data is not None and len(data) >= 4:
                    # Gunakan data lengkap untuk analisis trend
                    if detect_pattern(data.tail(4), data):
                        papan = df[df['Ticker'] == ticker]['Papan Pencatatan'].values[0]
                        results.append({"Ticker": ticker, "Papan": papan})
                
                progress = (i + 1) / len(tickers)
                progress_bar.progress(progress)
                progress_text.text(f"Progress: {int(progress * 100)}% - Memproses {ticker}")
            
            if results:
                temp_df = pd.DataFrame(results)
                final_df = analyze_results(temp_df, analysis_date)
                
                # Filter berdasarkan volume minimum
                final_df = final_df[final_df['Volume (Lot)'] >= min_volume_lot]
                
                st.session_state.screening_results = final_df
            else:
                st.warning("Tidak ada saham yang cocok dengan pola.")
    else:
        ticker_input = st.text_input("📌 Masukkan Ticker Saham (tanpa .JK):", "").strip().upper()
        today = datetime.today().date()
        start_date = today - timedelta(days=90)  # Periode lebih panjang untuk analisis
        end_date = today
        
        st.info(f"📅 Mencari tanggal konfirmasi dari {start_date} hingga {today}")
        
        if st.button("🔍 Cari Tanggal Konfirmasi Pisau Jatuh"):
            if not ticker_input:
                st.warning("❗ Silakan masukkan ticker saham.")
            else:
                with st.spinner(f"Mencari tanggal konfirmasi untuk **{ticker_input}**..."):
                    results = find_confirmation_dates_for_ticker(ticker_input, start_date, end_date)
                
                if results:
                    st.success(f"✅ Ditemukan {len(results)} pola untuk **{ticker_input}**")
                    final_df = analyze_pattern_dates(ticker_input, results)
                    st.session_state.screening_results = final_df
                else:
                    st.warning(f"❌ Tidak ada pola ditemukan untuk **{ticker_input}**.")

    if st.session_state.screening_results is not None and not st.session_state.screening_results.empty:
        st.subheader("✅ Hasil Screening")
        st.dataframe(st.session_state.screening_results)

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            st.session_state.screening_results.to_excel(writer, sheet_name='Hasil Screening', index=False)
        
        mode_label = "by_tanggal" if mode == "📅 Screening by Tanggal" else f"by_ticker_{ticker_input}" if 'ticker_input' in locals() and ticker_input else "unknown"
        st.download_button(
            label="📥 Unduh Hasil Screening (Excel)",
            data=output.getvalue(),
            file_name=f"pisau_jatuh_{mode_label}_{datetime.today().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    elif st.session_state.screening_results is not None:
        st.info("ℹ️ Tidak ada data yang memenuhi kriteria setelah analisis.")

if __name__ == "__main__":
    app()
