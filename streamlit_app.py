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

# Impor modul-modul halaman dari folder `page/`
try:
    from page.home import app as home_app
    from page.pisau_jatuh import app as pisau_jatuh_app
    from page.analisa_saham_input import app as analisa_saham_input_app
    from page.tarik_data_saham import app as tarik_data_saham_app
except ImportError as e:
    st.error(f"Error importing modules: {str(e)}")
    st.stop()

# Inisialisasi session state
if "subpage" not in st.session_state:
    st.session_state["subpage"] = None

# ----------- HALAMAN UTAMA -----------
def main_page():
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
page_config = {
    "Main Page": main_page,
    "Pisau Jatuh": pisau_jatuh,
    "Analisa Saham Input": analisa_saham_input,
    "Tarik Data Saham": tarik_data_saham,
}

# ----------- SIDEBAR -----------
with st.sidebar:
    selected = option_menu(
        menu_title="Saham SHZ",
        options=list(page_config.keys()),
        icons=["house", "knife", "chart-line", "download"],
        menu_icon="cast",
        default_index=0,
    )

# Reset session state jika kembali ke halaman utama
if selected == "Main Page":
    st.session_state["subpage"] = None

# ----------- RENDER HALAMAN -----------
try:
    if selected in page_config:
        page_config[selected]()
    else:
        st.error("Halaman tidak ditemukan. Silakan pilih halaman lain dari menu navigasi.")
except KeyError as e:
    st.error(f"Kesalahan: Halaman '{selected}' tidak ditemukan.")
except Exception as e:
    st.error(f"Terjadi kesalahan: {str(e)}")
