import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import datetime

# ---------------- KONFIGURASI HALAMAN ----------------
st.set_page_config(
    page_title="Saham SHZ",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hilangkan default "streamlit app" di sidebar
hide_default_format = """
    <style>
    [data-testid="stSidebarNav"] {
        display: none;
    }
    </style>
"""
st.markdown(hide_default_format, unsafe_allow_html=True)

# Fungsi untuk menambahkan CSS
def add_css(css):
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# CSS styling
css_styles = """
<style>
h1 {
    font-size: 2.5rem;
    color: #333;
}
p {
    font-size: 1.2rem;
    color: #555;
}
.sidebar .sidebar-content {
    padding: 20px;
    background-color: #f9f9f9;
}
</style>
"""
add_css(css_styles)

# ---------------- IMPORT MODUL ----------------
try:
    from pages.home import app as home_app
    from pages.pisau_jatuh import app as pisau_jatuh_app
    from pages.analisa_saham_input import app as analisa_saham_input_app
    from pages.tarik_data_saham import app as tarik_data_saham_app
except ImportError as e:
    st.error(f"Error importing modules: {str(e)}")
    st.stop()

# ---------------- SESSION STATE ----------------
if "subpages" not in st.session_state:
    st.session_state["subpages"] = None

# ---------------- HALAMAN ----------------
def main_pages():
    st.title("Selamat Datang di Aplikasi Saham SHZ")
    st.write("""
    Aplikasi ini berisi fitur:
    - **Pisau Jatuh**: Screener otomatis dari daftar ticker di Google Sheet — deteksi pola candlestick "Pisau Jatuh".
    - **Analisa Saham**: Input ticker → tampilkan analisis teknikal lengkap (MA, Fibonacci, RSI, MFI, grafik interaktif) + export PNG/PDF.
    - **Tarik Data Saham**: Input satu atau banyak ticker → unduh data historis ke Excel (multi-sheet).
    
    Pilih menu di sidebar untuk mulai!
    """)

def pisau_jatuh():
    try:
        pisau_jatuh_app()
    except Exception as e:
        st.error(f"Error: {str(e)}")

def analisa_saham():
    try:
        analisa_saham_input_app()
    except Exception as e:
        st.error(f"Error: {str(e)}")

def tarik_data_saham():
    try:
        tarik_data_saham_app()
    except Exception as e:
        st.error(f"Error: {str(e)}")

# ---------------- NAVIGASI ----------------
pages_config = {
    "Main pages": main_pages,
    "Pisau Jatuh": pisau_jatuh,
    "Analisa Saham": analisa_saham,
    "Tarik Data Saham": tarik_data_saham,
}

# Sidebar Navigation
with st.sidebar:
    selected = option_menu(
        menu_title="📊 Saham SHZ",
        options=list(pages_config.keys()),
        icons=["house", "scissors", "bar-chart", "download"],
        menu_icon="cast",
        default_index=0,
    )

# Reset subpages saat kembali ke Main Page
if selected == "Main pages":
    st.session_state["subpages"] = None

# ---------------- RENDER HALAMAN ----------------
try:
    if selected in pages_config:
        pages_config[selected]()
    else:
        st.error("Halaman tidak ditemukan.")
except KeyError as e:
    st.error(f"Kesalahan: Halaman '{selected}' tidak ditemukan.")
except Exception as e:
    st.error(f"Terjadi kesalahan: {str(e)}")
