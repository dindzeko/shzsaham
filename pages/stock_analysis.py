import streamlit as st
from utils import *

def app():
    st.title("📈 Analisa Saham")
    
    # Kolom untuk subpage
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Teknikal", use_container_width=True):
            st.session_state["subpage"] = "teknikal"
    with col2:
        if st.button("Fundamental", use_container_width=True):
            st.session_state["subpage"] = "fundamental"
    
    # Render subpage berdasarkan session state
    if st.session_state.get("subpage") == "teknikal":
        st.subheader("Analisis Teknikal")
        st.write("Implementasi analisis teknikal di sini")
    elif st.session_state.get("subpage") == "fundamental":
        st.subheader("Analisis Fundamental")
        st.write("Implementasi analisis fundamental di sini")

if __name__ == "__main__":
    app()
