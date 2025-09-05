import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import datetime

# Fungsi untuk menambahkan CSS
def add_css(css):
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# CSS styling (opsional)
css_styles = """
<style>
/* Styling untuk judul utama */
h1 {
    font-size: 2.5rem;
    color: #333;
}

/* Styling untuk deskripsi */
p {
    font-size: 1.2rem;
    color: #555;
}

/* Styling untuk sidebar */
.sidebar .sidebar-content {
    padding: 20px;
    background-color: #f9f9f9;
}
</style>
"""
add_css(css_styles)

# Impor modul-modul halaman dari folder `pages/`
try:
    from pages.home import app as home_app
    from pages.pisau_jatuh import app as pisau_jatuh_app
    from pages.analisa_saham_input import app as analisa_saham_input_app
    from pages.tarik_data_saham import app as tarik_data_saham_app
except ImportError as e:
    st.error(f"Error importing modules: {str(e)}")
    st.stop()

# Inisialisasi session state
if "subpages" not in st.session_state:
    st.session_state["subpages"] = None

# ----------- HALAMAN UTAMA -----------
def main_pages():
    st.title("Selamat Datang di Aplikasi Saham SHZ")
    st.write("""
    Aplikasi ini merupakan berisi fitur:
    - **Pisau Jatuh**: Screener otomatis dari daftar ticker di Google Sheet — deteksi pola candlestick "Pisau Jatuh".
    - **Analisa Saham Input**: Input ticker → tampilkan analisis teknikal lengkap (MA, Fibonacci, RSI, MFI, grafik interaktif) + export PNG/PDF.
    - **Tarik Data Saham**: Input satu atau banyak ticker → unduh data historis ke Excel (multi-sheet).
    
    Pilih menu di sidebar untuk mulai!
    """)

# ----------- HALAMAN PISAU JATUH -----------
def pisau_jatuh():
    st.title("Pisau Jatuh")
    try:
        pisau_jatuh_app()
    except Exception as e:
        st.error(f"Error: {str(e)}")

# ----------- HALAMAN ANALISA SAHAM INPUT -----------
def analisa_saham_input():
    st.title("Analisa Saham Input")
    try:
        analisa_saham_input_app()
    except Exception as e:
        st.error(f"Error: {str(e)}")

# ----------- HALAMAN TARIK DATA SAHAM -----------
def tarik_data_saham():
    st.title("Tarik Data Saham")
    try:
        tarik_data_saham_app()
    except Exception as e:
        st.error(f"Error: {str(e)}")

# ----------- KONFIGURASI NAVIGASI -----------
pages_config = {
    "Main pages": main_pages,
    "Pisau Jatuh": pisau_jatuh,
    "Analisa Saham Input": analisa_saham_input,
    "Tarik Data Saham": tarik_data_saham,
}

# ----------- SIDEBAR -----------
with st.sidebar:
    selected = option_menu(
        menu_title="Saham SHZ",
        options=list(pages_config.keys()),
        icons=["house", "knife", "chart-line", "download"],
        menu_icon="cast",
        default_index=0,
    )

# Reset session state jika kembali ke halaman utama
if selected == "Main pages":
    st.session_state["subpages"] = None

# ----------- RENDER HALAMAN -----------
try:
    if selected in pages_config:
        pages_config[selected]()
    else:
        st.error("Halaman tidak ditemukan. Silakan pilih halaman lain dari menu navigasi.")
except KeyError as e:
    st.error(f"Kesalahan: Halaman '{selected}' tidak ditemukan.")
except Exception as e:
    st.error(f"Terjadi kesalahan: {str(e)}")
