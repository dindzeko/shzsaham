import streamlit as st
from streamlit_option_menu import option_menu

# Fungsi untuk menambahkan CSS (opsional)
def add_css(css):
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# CSS styling (bisa dikembangkan nanti)
css_styles = """
<style>
h1 {
    font-size: 2.5rem;
    color: #2c3e50;
}
.sidebar .sidebar-content {
    background-color: #f8f9fa;
}
</style>
"""
add_css(css_styles)

# Impor halaman (akan diisi nanti, kita gunakan placeholder dulu)
# Pastikan file-file ini ada di folder `pages/`:
# - screener_pisau_jatuh.py
# - screener_multi.py
# - analisa.py
# - tarik_data.py

try:
    from pages.screener_pisau_jatuh import app as pisau_jatuh_app
    from pages.screener_multi import app as multi_screener_app
    from pages.analisa import app as analisa_app
    from pages.tarik_data import app as tarik_data_app
except ImportError as e:
    st.warning(f"Beberapa modul belum tersedia: {str(e)}. Pastikan file halaman sudah dibuat di folder 'pages/'.")
    # Definisikan fungsi dummy sebagai fallback
    def pisau_jatuh_app(): st.write("🔧 Halaman Pisau Jatuh belum diimplementasikan.")
    def multi_screener_app(): st.write("🔧 Halaman Multi Screener belum diimplementasikan.")
    def analisa_app(): st.write("📊 Halaman Analisa belum diimplementasikan.")
    def tarik_data_app(): st.write("📥 Halaman Tarik Data belum diimplementasikan.")

# Inisialisasi session state untuk subpage
if "subpage" not in st.session_state:
    st.session_state["subpage"] = None

# =========== HALAMAN-HALAMAN UTAMA ===========

def main_page():
    st.title("🎯 Selamat Datang di Aplikasi Screener & Analisis")
    st.markdown("""
    Aplikasi ini dirancang untuk membantu proses **screening data**, **analisis cepat**, dan **pengambilan data** secara efisien.
    
    Pilih menu di sidebar untuk mulai:
    
    - **Screener**: Lakukan screening data dengan berbagai metode
    - **Analisa**: Analisis data yang sudah diambil
    - **Tarik Data**: Ekstraksi data dari sumber eksternal
    
    Gunakan sub-menu jika tersedia.
    """)

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
    if st.session_state["subpage"] == "Pisau Jatuh":
        st.subheader("⚙️ Pisau Jatuh")
        pisau_jatuh_app()
    elif st.session_state["subpage"] == "Multi Screener":
        st.subheader("⚙️ Multi Screener")
        multi_screener_app()

def analisa_page():
    st.title("📊 Analisa")
    st.session_state["subpage"] = None  # Reset subpage
    analisa_app()

def tarik_data_page():
    st.title("📥 Tarik Data")
    st.session_state["subpage"] = None  # Reset subpage
    tarik_data_app()

# =========== KONFIGURASI MENU NAVIGASI ===========
pages = {
    "Main Page": main_page,
    "Screener": screener_page,
    "Analisa": analisa_page,
    "Tarik Data": tarik_data_page,
}

# =========== SIDEBAR MENU ===========
with st.sidebar:
    st.image("https://via.placeholder.com/150/007BFF/FFFFFF?text=AppLogo", width=120)  # opsional: ganti dengan logo Anda
    st.markdown("### 📊 Audit & Screening Tools")
    st.markdown("---")

    selected = option_menu(
        menu_title="Navigasi",
        options=list(pages.keys()),
        icons=["house", "search", "graph-up", "download"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "orange", "font-size": "20px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#eee",
            },
            "nav-link-selected": {"background-color": "#007BFF"},
        },
    )

# Reset subpage jika kembali ke halaman utama
if selected == "Main Page":
    st.session_state["subpage"] = None

# =========== RENDER HALAMAN YANG DIPILIH ===========
try:
    pages[selected]()
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat halaman: {str(e)}")
