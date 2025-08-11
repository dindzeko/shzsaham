import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import datetime

# ========== CSS ==========
def add_css(css):
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

css_styles = """
/* Styling untuk judul utama */
h1 {
    font-size: 2.2rem;
    color: #333;
}
/* Styling untuk deskripsi */
p {
    font-size: 1rem;
    color: #555;
}
/* Sidebar */
.sidebar .sidebar-content {
    padding: 20px;
    background-color: #f9f9f9;
}
/* Tombol lebar penuh */
.stButton > button {
    width: 100%;
}
"""
add_css(css_styles)

# ========== IMPORT PAGE ==========
try:
    from pages.screener_pisau_jatuh import app as pisau_jatuh_app
except ImportError:
    def pisau_jatuh_app():
        st.warning("🚧 Halaman Pisau Jatuh belum tersedia.")

# Placeholder Multi Screener
def multi_screener_app():
    st.info("🚧 Halaman Multi Screener sedang dalam pengembangan.")

# Placeholder Analisa
def analisa_app():
    st.info("📊 Halaman Analisa belum diimplementasikan.")

# Placeholder Tarik Data
def tarik_data_app():
    st.info("📥 Halaman Tarik Data belum diimplementasikan.")

# ========== SESSION STATE ==========
if "subpage" not in st.session_state:
    st.session_state["subpage"] = None

# ========== HALAMAN ==========
def main_page():
    st.title("🎯 Selamat Datang di Aplikasi Screener & Analisis")
    st.write("""
    Aplikasi ini membantu proses **screening data**, **analisis cepat**, dan **pengambilan data**.
    
    Modul utama:
    - **Screener**: Screening saham / data dengan berbagai metode
    - **Analisa**: Analisis data hasil screening
    - **Tarik Data**: Ambil data dari sumber eksternal
    """)

def screener():
    st.title("🔍 Screener")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Pisau Jatuh", use_container_width=True):
            st.session_state["subpage"] = "Pisau Jatuh"
    with col2:
        if st.button("Multi Screener", use_container_width=True):
            st.session_state["subpage"] = "Multi Screener"

    if st.session_state["subpage"] == "Pisau Jatuh":
        st.subheader("⚙️ Screener Pisau Jatuh")
        pisau_jatuh_app()
    elif st.session_state["subpage"] == "Multi Screener":
        st.subheader("⚙️ Screener Multi")
        multi_screener_app()

def analisa():
    st.title("📊 Analisa")
    st.session_state["subpage"] = None
    analisa_app()

def tarik_data():
    st.title("📥 Tarik Data")
    st.session_state["subpage"] = None
    tarik_data_app()

# ========== PAGE CONFIG ==========
page_config = {
    "Halaman Utama": main_page,
    "Screener": screener,
    "Analisa": analisa,
    "Tarik Data": tarik_data
}

# ========== SIDEBAR ==========
with st.sidebar:
    selected = option_menu(
        menu_title="📊 Screener & Analisis",
        options=list(page_config.keys()),
        icons=["house", "search", "graph-up", "download"],
        menu_icon="cast",
        default_index=0
    )

# Reset subpage kalau kembali ke Halaman Utama
if selected == "Halaman Utama":
    st.session_state["subpage"] = None

# ========== RENDER HALAMAN ==========
try:
    if selected in page_config:
        page_config[selected]()
    else:
        st.error("Halaman tidak ditemukan.")
except Exception as e:
    st.error(f"Terjadi kesalahan: {str(e)}")
