import streamlit as st

# Atur konfigurasi halaman utama
st.set_page_config(
    page_title="SHZ Saham",
    page_icon="📈",
    layout="wide"
)

# CSS untuk header dan background
st.markdown("""
    <style>
    .main-header {
        font-size: 24px;
        font-weight: bold;
        color: white;
        padding: 10px 20px;
        background-color: black;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin: -60px -15px 20px -15px;
        position: relative;
        z-index: 100;
    }
    .menu {
        list-style: none;
        display: flex;
        gap: 20px;
        padding: 0;
        margin: 0;
    }
    .menu a {
        color: #FF5733;
        text-decoration: none;
        font-size: 14px;
        font-weight: 500;
    }
    .menu a:hover {
        text-decoration: underline;
    }
    .overlay-text {
        position: absolute;
        top: 40%;
        left: 50%;
        transform: translate(-50%, -50%);
        color: white;
        text-align: center;
        z-index: 1;
        font-family: 'Arial', sans-serif;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.7);
    }
    .name {
        font-size: 48px;
        font-weight: bold;
        margin: 0;
    }
    .desc {
        font-size: 18px;
        color: #ddd;
        margin: 10px 0 0;
    }
    .stImage > img {
        border-radius: 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="main-header">
        <span>SHZ Saham</span>
        <ul class="menu">
            <li><a href="?page=home">Home</a></li>
            <li><a href="?page=screener">Screener</a></li>
            <li><a href="?page=tarik_data">Tarik Data</a></li>
            <li><a href="?page=analisa">Analisa Saham</a></li>
        </ul>
    </div>
""", unsafe_allow_html=True)

# Baca parameter halaman dari query string
from streamlit import experimental_get_query_params as get_query_params
from streamlit import experimental_set_query_params as set_query_params

params = get_query_params()
page = params.get("page", ["home"])[0]

# Background hanya di home
if page == "home":
    try:
        # Gunakan gambar dari URL
        st.image("https://storage.googleapis.com/flip-prod-mktg-strapi/media-library/apa_itu_investasi_saham_d3716da8f1/apa_itu_investasi_saham_d3716da8f1.jpeg", use_column_width=True)
    except Exception as e:
        st.error(f"Error loading image: {e}")

    st.markdown("""
        <div class="overlay-text">
            <p class="name">SHZ Saham</p>
            <p class="desc">Analisa & Screening Saham Otomatis</p>
        </div>
    """, unsafe_allow_html=True)

# Routing ke halaman
if page == "screener":
    st.switch_page("pages/screener.py")
elif page == "tarik_data":
    st.switch_page("pages/tarik_data.py")
elif page == "analisa":
    st.switch_page("pages/analisa_saham.py")
