# main.py
import streamlit as st

# Konfigurasi halaman utama
st.set_page_config(
    page_title="Aplikasi Saham",
    page_icon="📈",
    layout="wide"
)

# Header
st.title("📈 Aplikasi Analisis Saham")
st.markdown("---")

# Deskripsi Home
st.header("Selamat Datang di Aplikasi Analisis Saham")
st.write("""
Aplikasi ini membantu Anda dalam menganalisis saham di pasar modal dengan tiga fitur utama:
""")

# Tampilkan menu sebagai tombol navigasi
st.subheader("Navigasi Fitur")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🔍 Screener Saham", use_container_width=True):
        st.switch_page("pages/1_Screener.py")

with col2:
    if st.button("📥 Tarik Data", use_container_width=True):
        st.switch_page("pages/2_Tarik_Data.py")

with col3:
    if st.button("📊 Analisa Individu Saham", use_container_width=True):
        st.switch_page("pages/3_Analisa_Individu.py")

# Informasi tambahan
st.markdown("---")
st.markdown("""
**Fitur:**
- **Screener**: Filter saham berdasarkan kriteria tertentu (misalnya: PER, ROE, dll).
- **Tarik Data**: Ambil data historis harga saham dari sumber seperti Yahoo Finance.
- **Analisa Individu**: Analisis teknikal/fundamental untuk satu saham tertentu.

Gunakan tombol di atas untuk membuka fitur yang diinginkan.
""")
