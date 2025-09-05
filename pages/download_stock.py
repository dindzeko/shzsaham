import streamlit as st
from utils import *

def app():
    st.title("📥 Tarik Data Saham")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Historis", use_container_width=True):
            st.session_state["subpage"] = "historis"
    with col2:
        if st.button("Real-time", use_container_width=True):
            st.session_state["subpage"] = "realtime"
    
    if st.session_state.get("subpage") == "historis":
        st.subheader("Unduh Data Historis")
        # Implementasi logika unduh data historis
    elif st.session_state.get("subpage") == "realtime":
        st.subheader("Unduh Data Real-time")
        # Implementasi logika unduh data real-time

if __name__ == "__main__":
    app()
