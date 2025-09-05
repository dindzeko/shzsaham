import streamlit as st
from utils import *

# Inisialisasi session state
if 'screening_results' not in st.session_state:
    st.session_state.screening_results = None
if 'selected_ticker' not in st.session_state:
    st.session_state.selected_ticker = None

# Konfigurasi halaman utama
st.set_page_config(
    page_title="Analisis Saham Sholikhuddin",
    layout="wide"
)

# Dictionary mapping halaman utama ke fungsi
PAGE_CONFIG = {
    "📊 Stock Screener": stock_screener.app,
    "📈 Analisa Saham": stock_analysis.app,
    "📥 Tarik Data Saham": download_stock.app,
}

# Sidebar navigasi
with st.sidebar:
    selected_page = st.selectbox(
        "Pilih Halaman",
        list(PAGE_CONFIG.keys()),
        index=0
    )

# Render halaman yang dipilih
if selected_page in PAGE_CONFIG:
    PAGE_CONFIG[selected_page]()
else:
    st.error("Halaman tidak ditemukan!")
