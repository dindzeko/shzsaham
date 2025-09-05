import streamlit as st
from streamlit_option_menu import option_menu
import os

# Konfigurasi halaman utama
st.set_page_config(
    page_title="SHZ SAHAM",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="expanded"
)

# CSS styling (opsional)
def add_css(css):
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

css_styles = """
<style>
/* Styling judul */
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

# Daftar halaman yang tersedia (sesuai nama file di folder 'pages/')
pages = {
    "Home": "home",
    "Pisau Jatuh": "pisau_jatuh",
    "Analisa Saham Input": "analisa_saham_input",
    "Tarik Data Saham": "tarik_data_saham"
}

# Sidebar navigation
with st.sidebar:
    selected = option_menu(
        menu_title="Saham SHZ",
        options=list(pages.keys()),
        icons=[
            "house", "knife", "chart-line", "download"
        ],
        menu_icon="cast",
        default_index=0,
    )

# Ambil nama file Python dari pilihan
page_name = pages[selected]

# Cek apakah file ada di direktori pages/
pages_dir = "pages"
if not os.path.exists(pages_dir):
    st.error("Folder 'pages' tidak ditemukan.")
else:
    try:
        # Impor dinamis berdasarkan nama file
        module = __import__(f"pages.{page_name}", fromlist=["app"])
        if hasattr(module, "app"):
            module.app()
        else:
            st.error(f"Modul '{page_name}.py' tidak memiliki fungsi 'app()'.")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat halaman {selected}: {str(e)}")
