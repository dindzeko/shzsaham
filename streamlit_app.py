import streamlit as st
from streamlit_option_menu import option_menu
import os

# --- CSS ---
def add_css(css):
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

css_styles = """
<style>
h1 { font-size: 2.5rem; color: #333; text-align: center; }
p { font-size: 1.2rem; color: #555; }
.sidebar .sidebar-content { padding: 20px; background-color: #f9f9f9; }

/* Sembunyikan header dan navigasi otomatis */
[data-testid="stSidebarHeader"] {
    display: none !important;
}
[data-testid="stSidebarNav"] > div {
    display: none !important;
}
</style>
"""
add_css(css_styles)

# --- Menu Manual ---
pages = {
    "Home": "home",
    "Pisau Jatuh": "pisau_jatuh",
    "Analisa Saham Input": "analisa_saham_input",
    "Tarik Data Saham": "tarik_data_saham"
}

with st.sidebar:
    selected = option_menu(
        menu_title="Saham SHZ",
        options=list(pages.keys()),
        icons=["house", "knife", "chart-line", "download"],
        menu_icon="cast",
        default_index=0,
    )

# --- Load Module ---
pages_dir = "pages"
if not os.path.exists(pages_dir):
    st.error("Folder 'pages' tidak ditemukan.")
else:
    try:
        module = __import__(f"pages.{pages[selected]}", fromlist=["app"])
        if hasattr(module, "app"):
            module.app()
        else:
            st.error(f"Modul tidak memiliki fungsi 'app()'.")
    except Exception as e:
        st.error(f"Error memuat halaman: {str(e)}")
