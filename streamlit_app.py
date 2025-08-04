import streamlit as st

# Set page config
st.set_page_config(
    page_title="SHZ Saham",
    page_icon="📈",
    layout="wide"
)

# Custom CSS untuk header
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
    .menu button {
        background: none;
        border: none;
        color: #FF5733;
        font-size: 14px;
        font-weight: 500;
        cursor: pointer;
        padding: 0;
    }
    .menu button:hover {
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
    </style>
""", unsafe_allow_html=True)

# Header dengan tombol navigasi
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.write("SHZ Saham")

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Screener"):
        st.switch_page("screener")
with col2:
    if st.button("Tarik Data"):
        st.switch_page("tarik_data")
with col3:
    if st.button("Analisa Saham"):
        st.switch_page("analisa_saham")
st.markdown('</div>', unsafe_allow_html=True)

# Halaman utama (home)
if "page" not in st.query_params:
    # Background dari URL
    try:
        st.image(
            "https://storage.googleapis.com/flip-prod-mktg-strapi/media-library/apa_itu_investasi_saham_d3716da8f1/apa_itu_investasi_saham_d3716da8f1.jpeg",
            use_container_width=True
        )
    except:
        st.image("https://via.placeholder.com/1920x1080/000000/FFFFFF?text=Background+SHZ+Sah
