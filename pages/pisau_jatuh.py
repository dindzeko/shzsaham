# pages/pisau_jatuh.py
import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
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

# --- AMBIL DATA SAHAM ---
def get_stock_data(ticker, analysis_date):
    try:
        stock = yf.Ticker(f"{ticker}.JK")
        start_date = analysis_date - timedelta(days=90)

        # Ambil sampai hari ini (supaya dapat harga terbaru)
        data = stock.history(
            start=start_date.strftime('%Y-%m-%d'),
            end=(datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d')
        )
        return data if not data.empty else None
    except Exception as e:
        st.error(f"Gagal mengambil data untuk {ticker}: {e}")
        return None

# --- BACA FILE EXCEL DARI GOOGLE DRIVE ---
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

# --- FUNGSI UTAMA APP ---
def app():
    st.title("🔪 Pisau Jatuh Screener")

    if 'screening_results' not in st.session_state:
        st.session_state.screening_results = None

    file_url = "https://docs.google.com/spreadsheets/d/1t6wgBIcPEUWMq40GdIH1GtZ8dvI9PZ2v/edit?usp=drive_link"
    df = load_google_drive_excel(file_url)
    if df is None:
        return

    tickers = df['Ticker'].dropna().unique().tolist()
    analysis_date = st.date_input("📅 Tanggal Analisis", value=datetime.today())

    if st.button("🔍 Mulai Screening"):
        results = []
        progress_bar = st.progress(0)
        progress_text = st.empty()

        for i, ticker in enumerate(tickers):
            data = get_stock_data(ticker, analysis_date)

            if data is not None and len(data) >= 50:
                if detect_pattern(data):
                    papan = df[df['Ticker'] == ticker]['Papan Pencatatan'].values[0]

                    # Harga Analisa = close terakhir sebelum/tanggal analisa
                    harga_analisa = data.loc[:str(analysis_date)].iloc[-1]['Close'] if not data.loc[:str(analysis_date)].empty else None
                    # Harga Terakhir = close paling baru dari yfinance
                    harga_terakhir = data['Close'].iloc[-1]

                    if harga_analisa is not None:
                        volume = int(data['Volume'].iloc[-1])
                        results.append({
                            "Ticker": ticker,
                            "Papan": papan,
                            "Harga Terakhir": round(harga_terakhir, 2),
                            "Harga Analisa": round(harga_analisa, 2),
                            "Volume Rp": int(harga_terakhir * volume),
                            "Volume Lot": int(volume / 100),
                            "MA 5 Close": round(data['Close'].tail(5).mean(), 2),
                            "MA 20 Close": round(data['Close'].tail(20).mean(), 2)
                        })

            progress = (i + 1) / len(tickers)
            progress_bar.progress(progress)
            progress_text.text(f"Progress: {int(progress * 100)}% - Memproses {ticker}")

        if results:
            st.session_state.screening_results = pd.DataFrame(results)
        else:
            st.warning("Tidak ada saham yang cocok dengan pola.")

    if st.session_state.screening_results is not None:
        st.subheader("✅ Saham yang Memenuhi Pola Pisau Jatuh")
        st.dataframe(st.session_state.screening_results)

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            st.session_state.screening_results.to_excel(writer, sheet_name='Hasil Screening', index=False)

        st.download_button(
            label="📥 Unduh Hasil Screening (Excel)",
            data=output.getvalue(),
            file_name=f"pisau_jatuh_{analysis_date.strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

if __name__ == "__main__":
    app()
