# pages/home.py
import streamlit as st

def app():
    st.title("🏡 Selamat Datang di Aplikasi Saham SHZ")
    st.write("""
    Aplikasi ini merupakan berisi fitur:

    - **Pisau Jatuh**: Screener otomatis dari daftar ticker di Google Sheet — deteksi pola candlestick "Pisau Jatuh".
    - **Analisa Saham Input**: Input ticker → tampilkan analisis teknikal lengkap (MA, Fibonacci, RSI, MFI, grafik interaktif) + export PNG/PDF.
    - **Tarik Data Saham**: Input satu atau banyak ticker → unduh data historis ke Excel (multi-sheet).

    Pilih menu di sidebar untuk mulai !
    """)
