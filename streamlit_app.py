import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import datetime

# Fungsi untuk menambahkan CSS
def add_css(css):
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# CSS styling
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

/* Styling untuk tombol */
.stButton>button {
    width: 100%;
}
</style>
"""
add_css(css_styles)

# Impor halaman dari folder `pages/`
try:
    from pages.screener_pisau_jatuh import app as pisau_jatuh_app
    # Fungsi dummy untuk multi screener
    def multi_screener_app(): 
        st.info("🚧 Halaman Multi Screener sedang dalam pengembangan")
        st.write("Fitur ini akan segera hadir dalam versi berikutnya")
except ImportError as e:
    st.error(f"Error importing modules: {str(e)}")
    # Fallback: fungsi dummy
    def pisau_jatuh_app(): 
        st.write("🔧 Halaman Pisau Jatuh belum diimplementasikan.")
    def multi_screener_app(): 
        st.write("🔧 Halaman Multi Screener belum diimplementasikan.")

# Fungsi dummy untuk halaman lain
def analisa_app(): st.write("📊 Halaman Analisa belum diimplementasikan.")
def tarik_data_app(): st.write("📥 Halaman Tarik Data belum diimplementasikan.")

# Inisialisasi session state
if "subpage" not in st.session_state:
    st.session_state["subpage"] = None

# ----------- HALAMAN UTAMA -----------
def main_page():
    st.title("🎯 Selamat Datang di Aplikasi Screener & Analisis")
    st.write("""
    Aplikasi ini dirancang untuk membantu proses **screening data**, **analisis cepat**, dan **pengambilan data** secara efisien.
    
    Pilih menu di sidebar untuk mulai:
    
    - **Screener**: Lakukan screening data dengan berbagai metode
    - **Analisa**: Analisis data yang sudah diambil
    - **Tarik Data**: Ekstraksi data dari sumber eksternal
    
    Gunakan sub-menu jika tersedia.
    """)

# ----------- HALAMAN SCREENER -----------
def screener_page():
    st.title("🔍 Screener")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Pisau Jatuh", use_container_width=True):
            st.session_state["subpage"] = "Pisau Jatuh"
    
    with col2:
        if st.button("Multi Screener", use_container_width=True):
            st.session_state["subpage"] = "Multi Screener"
    
    # Render subpage
    if st.session_state.get("subpage") == "Pisau Jatuh":
        st.subheader("⚙️ Pisau Jatuh")
        try:
            pisau_jatuh_app()
        except Exception as e:
            st.error(f"Error: {str(e)}")
    elif st.session_state.get("subpage") == "Multi Screener":
        st.subheader("⚙️ Multi Screener")
        multi_screener_app()

# ----------- HALAMAN ANALISA -----------
def analisa_page():
    st.title("📊 Analisa")
    st.session_state["subpage"] = None  # Reset subpage
    analisa_app()

# ----------- HALAMAN TARIK DATA -----------
def tarik_data_page():
    st.title("📥 Tarik Data")
    st.session_state["subpage"] = None  # Reset subpage
    tarik_data_app()

# ----------- KONFIGURASI NAVIGASI -----------
page_config = {
    "Halaman Utama": main_page,
    "Screener": screener_page,
    "Analisa": analisa_page,
    "Tarik Data": tarik_data_page,
}

# ----------- SIDEBAR -----------
with st.sidebar:
    st.image("https://via.placeholder.com/150/007BFF/FFFFFF?text=AppLogo", width=120)
    st.markdown("### 📊 Audit & Screening Tools")
    st.markdown("---")
    
    selected = option_menu(
        menu_title=None,  # Tidak ada judul menu
        options=list(page_config.keys()),
        icons=["house", "search", "graph-up", "download"],
        menu_icon="cast",
        default_index=0,
    )

# Reset session state jika kembali ke halaman utama
if selected == "Halaman Utama":
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
