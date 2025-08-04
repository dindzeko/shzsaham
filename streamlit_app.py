# main.py
import streamlit as st

st.set_page_config(
    page_title="Aplikasi Saham",
    page_icon="📈",
    layout="wide"
)

# Redirect ke halaman utama
from pages import Main_Page
Main_Page.app()
