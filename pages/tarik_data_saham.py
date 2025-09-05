# pages/tarik_data.py
import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import io
import time

# Konfigurasi halaman — HARUS DI LUAR app() untuk menghindari error
st.set_page_config(
    page_title="Halaman Tarik Data Saham",
    page_icon="📊",
    layout="centered"
)

def app():
    # Judul aplikasi — cukup pakai st.title, tanpa banner HTML
    st.title("📊 Halaman Tarik Data Saham")

    # Pilih Time Frame
    st.subheader("⚙️ Pengaturan Data")
    col1, col2 = st.columns(2)
    with col1:
        timeframe = st.radio("**Interval **", ["30m", "60m", "1d"], index=2, horizontal=True)
    with col2:
        trading_days = st.number_input("**Jumlah hari perdagangan:**", min_value=1, max_value=60, value=10)

    # Informasi kolom harga
    with st.expander("ℹ️ Penjelasan Kolom Harga", expanded=False):
        st.markdown(f"""
        **Kolom Harga:**
        - **Close**: Harga penutupan aktual (sesuai tampilan web)
        - **Adj Close**: Harga penutupan yang disesuaikan (untuk corporate actions)
        - **Open/High/Low**: Harga pembukaan/tertinggi/terendah aktual
        
        Aplikasi ini akan mengambil data untuk **{trading_days} hari perdagangan** terakhir.
        """)

    # Input metode ticker
    st.subheader("📋 Input Ticker Saham")
    ticker_input_method = st.radio("Pilih cara input ticker:", ["Upload Excel", "Input Manual"], 
                                  horizontal=True, label_visibility="collapsed")

    tickers_list = []

    # Input Ticker via Upload Excel
    if ticker_input_method == "Upload Excel":
        uploaded_file = st.file_uploader("Upload file Excel (.xlsx) yang berisi kolom 'Ticker'", type=["xlsx"])
        if uploaded_file:
            try:
                df_tickers = pd.read_excel(uploaded_file)
                if 'Ticker' not in df_tickers.columns:
                    st.error("❌ File Excel harus memiliki kolom bernama 'Ticker'")
                else:
                    tickers_list = df_tickers['Ticker'].dropna().astype(str).str.strip().str.upper().tolist()
                    st.success(f"✅ Ditemukan {len(tickers_list)} ticker")
                    with st.expander("Lihat Daftar Ticker"):
                        st.write(tickers_list)
            except Exception as e:
                st.error(f"Terjadi kesalahan saat membaca file: {e}")

    # Input Ticker Manual
    else:
        manual_tickers = st.text_area("Masukkan daftar ticker (pisahkan dengan koma):", "BBCA.JK, TLKM.JK, PTBA.JK")
        if manual_tickers:
            tickers_list = [x.strip().upper() for x in manual_tickers.split(",") if x.strip()]
            st.info(f"ℹ️ {len(tickers_list)} ticker siap diambil")

    # Tombol ambil data
    if st.button("🚀 Ambil Data Saham", use_container_width=True, type="primary"):
        if not tickers_list:
            st.warning("Silakan input ticker saham terlebih dahulu.")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            end_date = datetime.today()
            buffer_days = trading_days * 3
            start_date = end_date - timedelta(days=buffer_days)
            
            data_frames = []
            failed_tickers = []
            success_count = 0
            
            for i, ticker in enumerate(tickers_list):
                try:
                    status_text.text(f"⏳ Mengambil data {ticker} ({i+1}/{len(tickers_list)})")
                    progress_bar.progress((i+1) / len(tickers_list))
                    
                    stock = yf.Ticker(ticker)
                    hist = stock.history(
                        start=start_date, 
                        end=end_date, 
                        interval=timeframe,
                        auto_adjust=False
                    )
                    
                    if hist.empty:
                        st.warning(f"⚠️ Data kosong untuk {ticker}")
                        failed_tickers.append(ticker)
                        time.sleep(0.3)
                        continue
                    
                    hist = hist.reset_index()
                    if 'Datetime' in hist.columns:
                        hist.rename(columns={'Datetime': 'Date'}, inplace=True)
                    
                    if 'Adj Close' not in hist.columns:
                        hist['Adj Close'] = hist['Close']
                    
                    hist['Date'] = hist['Date'].dt.tz_localize(None)
                    
                    if timeframe == "1d":
                        unique_dates = hist['Date'].dt.date.unique()
                        selected_dates = sorted(unique_dates, reverse=True)[:trading_days]
                        hist = hist[hist['Date'].dt.date.isin(selected_dates)]
                    else:
                        unique_dates = hist['Date'].dt.date.unique()
                        selected_dates = sorted(unique_dates, reverse=True)[:trading_days]
                        hist = hist[hist['Date'].dt.date.isin(selected_dates)]
                    
                    hist.insert(0, 'Ticker', ticker)
                    
                    if timeframe == "1d":
                        hist['Date'] = hist['Date'].dt.strftime('%Y-%m-%d')
                    else:
                        hist['Date'] = hist['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
                    
                    data_frames.append(hist)
                    success_count += 1
                    time.sleep(0.5)
                    
                except Exception as e:
                    st.error(f"❌ Gagal mengambil data {ticker}: {str(e)}")
                    failed_tickers.append(ticker)
                    time.sleep(1)

            if data_frames:
                result_df = pd.concat(data_frames, ignore_index=True)
                
                # Urutkan kolom
                column_order = ['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
                result_df = result_df[column_order]
                
                # Urutkan berdasarkan tanggal (terbaru di atas)
                result_df = result_df.sort_values(by='Date', ascending=False)
                
                st.success(f"✅ Berhasil mengambil data untuk {trading_days} hari perdagangan terakhir")
                
                if failed_tickers:
                    st.warning(f"⚠️ Gagal mengambil data untuk: {', '.join(failed_tickers)}")
                
                # Tampilkan informasi jumlah data
                unique_dates = result_df['Date'].nunique() if timeframe == "1d" else len(result_df['Date'].unique())
                st.info(f"📅 Jumlah hari perdagangan yang diambil: {unique_dates} hari")
                
                # Tampilkan data
                with st.expander("📊 Lihat Data", expanded=True):
                    st.dataframe(
                        result_df.style.format({
                            'Open': '{:.2f}',
                            'High': '{:.2f}',
                            'Low': '{:.2f}',
                            'Close': '{:.2f}',
                            'Adj Close': '{:.2f}',
                            'Volume': '{:,.0f}'
                        }),
                        use_container_width=True,
                        height=400
                    )
                
                # Download Excel
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    result_df.to_excel(writer, index=False)
                
                st.download_button(
                    label="💾 Download Data Excel",
                    data=output.getvalue(),
                    file_name=f"data_saham_{timeframe}_{trading_days}hari.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    type="primary"
                )
            else:
                st.error("❌ Tidak ada data yang berhasil diambil. Silakan cek koneksi atau ticker Anda")
            
            progress_bar.empty()
            status_text.empty()
