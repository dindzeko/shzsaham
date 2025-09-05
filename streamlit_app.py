import streamlit as st
from pages import *

def main():
    st.set_page_config(page_title="Analisis Saham Sholikhuddin", layout="wide")
    
    # Navigation
    st.sidebar.title("🧭 Navigasi")
    page = st.sidebar.radio("Pilih Halaman:", ["🏠 Home", "📊 Analisa Pisau Jatuh", "📈 Analisa Saham Input", "📥 Tarik Data Saham"])
    
    if page == "🏠 Home":
        show_home_page()
    elif page == "📊 Analisa Pisau Jatuh":
        show_pisau_jatuh_page()
    elif page == "📈 Analisa Saham Input":
        show_analisa_saham_input_page()
    elif page == "📥 Tarik Data Saham":
        show_tarik_data_saham_page()

if __name__ == "__main__":
    main()
