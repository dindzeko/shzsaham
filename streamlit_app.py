import streamlit as st
from streamlit_option_menu import option_menu
import os

# ====================
# Fungsi CSS Styling
# ====================
def add_css(css):
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# ====================
# FULL CSS STYLING (Bersih & Profesional)
# ====================
modern_css = """
<style>
/* --- Reset & Font --- */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Segoe UI', sans-serif;
    background-color: #f5f7fa;
    color: #333;
}

/* --- Header Utama --- */
header {
    background-color: #fff;
    padding: 1.2rem 2rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    position: sticky;
    top: 0;
    z-index: 100;
    display: flex;
    justify-content: center;
    align-items: center;
}

header h1 {
    font-size: 2.4rem;
    color: #2c3e50;
    font-weight: 600;
    letter-spacing: -0.5px;
}

/* --- Sidebar --- */
.sidebar .sidebar-content {
    background-color: #f8f9fa !important;
    padding: 25px !important;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    margin-top: 20px;
}

/* --- HILANGKAN STREAMLIT.APP DAN HEADER DEFAULT --- */
[data-testid="stSidebarHeader"] {
    display: none !important;
}
[data-testid="stSidebarNav"] > div {
    display: none !important;
}

/* --- MENU OPTION --- */
.sidebar .sidebar-content ul {
    list-style: none;
    padding: 0;
}

.sidebar .sidebar-content ul li a {
    display: flex;
    align-items: center;
    padding: 14px 20px;
    text-decoration: none;
    color: #2c3e50;
    font-size: 1.05rem;
    border-radius: 8px;
    margin-bottom: 8px;
    transition: all 0.3s ease;
    font-weight: 500;
}

.sidebar .sidebar-content ul li a:hover,
.sidebar .sidebar-content ul li a.active {
    background-color: #2980b9;
    color: white;
    transform: translateX(4px);
}

.sidebar .sidebar-content ul li a i {
    margin-right: 12px;
    font-size: 1.2rem;
    width: 24px;
    text-align: center;
}

/* --- CONTENT UTAMA --- */
.main .block-container {
    padding-top: 2rem !important;
    padding-left: 3rem !important;
    padding-right: 3rem !important;
}

/* --- JUDUL UTAMA --- */
h1 {
    font-size: 2.8rem;
    color: #2c3e50;
    text-align: center;
    margin-bottom: 1.5rem;
    font-weight: 700;
    letter-spacing: -1px;
}

h2 {
    font-size: 2.1rem;
    color: #2980b9;
    margin-top: 2rem;
    margin-bottom: 1rem;
    font-weight: 600;
}

/* --- PARAGRAF --- */
p {
    font-size: 1.1rem;
    line-height: 1.7;
    color: #555;
    margin-bottom: 1.2rem;
    text-align: justify;
}

/* --- FOOTER --- */
footer {
    text-align: center;
    padding: 30px;
    color: #7f8c8d;
    font-size: 0.95rem;
    margin-top: 3rem;
    border-top: 1px solid #eee;
}
</style>
"""
add_css(modern_css)

# ====================
# IMPOR MODUL HALAMAN
# ====================
try:
    from pages.home import app as home_app
    from pages.pisau_jatuh import app as pisau_jatuh_app
    from pages.analisa_saham_input import app as analisa_saham_input_app
    from pages.tarik_data_saham import app as tarik_data_saham_app
except ImportError as e:
    st.error(f"Error importing modules: {str(e)}")
    st.stop()

# ====================
# SESSION STATE
# ====================
if "selected_page" not in st.session_state:
    st.session_state["selected_page"] = "Home"

# ====================
# NAVIGASI SIDEBAR
# ====================
with st.sidebar:
    selected = option_menu(
        menu_title="Saham SHZ",
        options=["Home", "Pisau Jatuh", "Analisa Saham Input", "Tarik Data Saham"],
        icons=["house", "knife", "chart-line", "download"],
        menu_icon="cast",
        default_index=0,
    )
    st.session_state["selected_page"] = selected

# ====================
# RENDER HALAMAN
# ====================
if st.session_state["selected_page"] == "Home":
    st.title("Selamat Datang di Aplikasi Saham SHZ")
    st.write("""
    Aplikasi ini merupakan berisi fitur:
    - **Pisau Jatuh**: Screener otomatis dari daftar ticker di Google Sheet — deteksi pola candlestick "Pisau Jatuh".
    - **Analisa Saham Input**: Input ticker → tampilkan analisis teknikal lengkap (MA, Fibonacci, RSI, MFI, grafik interaktif) + export PNG/PDF.
    - **Tarik Data Saham**: Input satu atau banyak ticker → unduh data historis ke Excel (multi-sheet).
    
    Pilih menu di sidebar untuk mulai!
    """)
elif st.session_state["selected_page"] == "Pisau Jatuh":
    try:
        pisau_jatuh_app()
    except Exception as e:
        st.error(f"Error: {str(e)}")
elif st.session_state["selected_page"] == "Analisa Saham Input":
    try:
        analisa_saham_input_app()
    except Exception as e:
        st.error(f"Error: {str(e)}")
elif st.session_state["selected_page"] == "Tarik Data Saham":
    try:
        tarik_data_saham_app()
    except Exception as e:
        st.error(f"Error: {str(e)}")
