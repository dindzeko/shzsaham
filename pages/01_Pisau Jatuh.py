from utils import *

def show_pisau_jatuh_page():
    st.title("📊 Analisa Pisau Jatuh - Screening Saham")
    file_url = "https://docs.google.com/spreadsheets/d/1t6wgBIcPEUWMq40GdIH1GtZ8dvI9PZ2v/edit?usp=drive_link"
    df = load_google_drive_excel(file_url)
    if df is None or 'Ticker' not in df.columns:
        return
    
    tickers = df['Ticker'].dropna().unique().tolist()
    analysis_date = st.date_input("📅 Tanggal Analisis", value=datetime.today())
    
    # Tombol screening
    if st.button("🔍 Mulai Screening"):
        results = []
        progress_bar = st.progress(0)
        progress_text = st.empty()
        for i, ticker in enumerate(tickers):
            data = get_stock_data(ticker, analysis_date)
            if data is not None and len(data) >= 50:
                if detect_pattern(data):
                    metrics = calculate_additional_metrics(data)
                    papan = df[df['Ticker'] == ticker]['Papan Pencatatan'].values[0]
                    fib = metrics["Fibonacci"]
                    results.append({
                        "Ticker": ticker,
                        "Papan": papan,
                        "Last Close": round(data['Close'].iloc[-1], 2),
                        "MA20": metrics["MA20"],
                        "MA50": metrics["MA50"],
                        "RSI": metrics["RSI"],
                        "MFI": metrics["MFI"],
                        "MFI Signal": metrics["MFI_Signal"],
                        "Vol Anomali": "🚨 Ya" if metrics["Volume_Anomali"] else "-",
                        "Volume": metrics["Volume"],
                        "Support": " | ".join([f"{s:.2f}" for s in metrics["Support"]]),
                        "Resistance": " | ".join([f"{r:.2f}" for r in metrics["Resistance"]]),
                        "Fib 0.382": fib['Fib_0.382'],
                        "Fib 0.5": fib['Fib_0.5'],
                        "Fib 0.618": fib['Fib_0.618']
                    })
            progress = (i + 1) / len(tickers)
            progress_bar.progress(progress)
            progress_text.text(f"Progress: {int(progress * 100)}% - Memproses {ticker}")
        if results:
            st.session_state.screening_results = pd.DataFrame(results)
            st.session_state.selected_ticker = None
        else:
            st.warning("Tidak ada saham yang cocok dengan pola.")
    
    # Tampilkan hasil screening jika ada
    if st.session_state.screening_results is not None:
        st.subheader("✅ Saham yang Memenuhi Kriteria")
        st.dataframe(st.session_state.screening_results)
        
        # Dropdown untuk memilih saham
        ticker_list = st.session_state.screening_results['Ticker'].tolist()
        selected_ticker = st.selectbox(
            "Pilih Saham untuk Detail",
            options=ticker_list,
            index=ticker_list.index(st.session_state.selected_ticker) if st.session_state.selected_ticker in ticker_list else 0,
            key='ticker_selector'
        )
        
        # Simpan ticker yang dipilih di session state
        st.session_state.selected_ticker = selected_ticker
        
        # Tombol untuk menampilkan detail
        if st.button("Tampilkan Analisis Detail"):
            if st.session_state.selected_ticker:
                show_stock_details(st.session_state.selected_ticker, analysis_date)
