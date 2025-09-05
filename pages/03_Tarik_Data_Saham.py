from utils import *

def show_tarik_data_saham_page():
    st.title("📥 Tarik Data Saham")
    
    st.write("""
    Fitur ini memungkinkan Anda untuk secara manual mengunduh data historis saham. 
    Silakan masukkan kode saham dan rentang tanggal yang Anda inginkan.
    """)
    
    ticker_input = st.text_input("Masukkan kode saham (misal: BBCA)", placeholder="Contoh: BBCA")
    start_date = st.date_input("Tanggal Mulai", value=datetime.today() - timedelta(days=365))
    end_date = st.date_input("Tanggal Akhir", value=datetime.today())
    
    if st.button("Unduh Data"):
        if ticker_input:
            data = get_stock_data(ticker_input, end_date)
            if data is not None and not data.empty:
                st.success(f"Data untuk {ticker_input} berhasil diunduh!")
                st.write(f"Rentang tanggal: {start_date.strftime('%Y-%m-%d')} sampai {end_date.strftime('%Y-%m-%d')}")
                st.write(f"Jumlah baris data: {len(data)}")
                
                # Tampilkan preview data
                st.subheader("Preview Data")
                st.dataframe(data.head())
                
                # Opsi unduh CSV
                csv = data.to_csv(index=True)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"{ticker_input}_stock_data.csv",
                    mime="text/csv"
                )
            else:
                st.error(f"Tidak dapat mengambil data untuk {ticker_input}")
        else:
            st.warning("Silakan masukkan kode saham terlebih dahulu")
