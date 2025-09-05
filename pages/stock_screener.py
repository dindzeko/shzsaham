import streamlit as st
from utils import *

def app():
    st.title("📊 Stock Screener - Pisau Jatuh SHZ 2nd Gen Edition ")
    file_url = "https://docs.google.com/spreadsheets/d/1t6wgBIcPEUWMq40GdIH1GtZ8dvI9PZ2v/edit?usp=drive_link"
    df = load_google_drive_excel(file_url)
    if df is None or 'Ticker' not in df.columns:
        return
    tickers = df['Ticker'].dropna().tolist()
    analysis_date = st.date_input("📅 Tanggal Analisis", value=datetime.today())
    
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
    
    if st.session_state.screening_results is not None:
        st.subheader("✅ Saham yang Memenuhi Kriteria")
        st.dataframe(st.session_state.screening_results)
        
        ticker_list = st.session_state.screening_results['Ticker'].tolist()
        selected_ticker = st.selectbox(
            "Pilih Saham untuk Detail",
            options=ticker_list,
            index=ticker_list.index(st.session_state.selected_ticker) if st.session_state.selected_ticker in ticker_list else 0,
            key='ticker_selector'
        )
        
        st.session_state.selected_ticker = selected_ticker
        
        if st.button("Tampilkan Analisis Detail"):
            if st.session_state.selected_ticker:
                show_stock_details(st.session_state.selected_ticker, analysis_date)

def show_stock_details(ticker, end_date):
    data = get_stock_data(ticker, end_date)
    if data is None or data.empty:
        st.warning(f"Data untuk {ticker} tidak tersedia")
        return
        
    st.subheader(f"Analisis Teknis: {ticker}")
    
    fig = go.Figure()
    
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Candlestick'
    ))
    
    data['MA20'] = data['Close'].rolling(20).mean()
    data['MA50'] = data['Close'].rolling(50).mean()
    fig.add_trace(go.Scatter(
        x=data.index, 
        y=data['MA20'], 
        name='MA20',
        line=dict(color='blue', width=1)
    ))
    fig.add_trace(go.Scatter(
        x=data.index, 
        y=data['MA50'], 
        name='MA50',
        line=dict(color='orange', width=1)
    ))
    
    try:
        sr = calculate_support_resistance(data.tail(60))
        fib = sr['Fibonacci']
        
        for level in sr['Support']:
            fig.add_hline(
                y=level, 
                line_dash="dash", 
                line_color="green",
                annotation_text=f"Support: {level:.2f}",
                annotation_position="bottom right"
            )
        for level in sr['Resistance']:
            fig.add_hline(
                y=level, 
                line_dash="dash", 
                line_color="red",
                annotation_text=f"Resistance: {level:.2f}",
                annotation_position="top right"
            )
        
        for key, value in fib.items():
            if "Fib" in key:
                fig.add_hline(
                    y=value,
                    line_dash="dot",
                    line_color="purple",
                    annotation_text=f"{key}: {value:.2f}",
                    annotation_position="top left" if "0." in key else "bottom left"
                )
    except Exception as e:
        st.warning(f"Gagal menghitung support/resistance: {e}")
    
    fig.update_layout(
        title=f"{ticker} Price Analysis",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    try:
        metrics = calculate_additional_metrics(data)
        fib = metrics.get("Fibonacci", {})
        
        col1, col2, col3 = st.columns(3)
        col1.metric("MA20", f"{metrics.get('MA20', 0):.2f}")
        col1.metric("MA50", f"{metrics.get('MA50', 0):.2f}")
        col2.metric("RSI", f"{metrics.get('RSI', 0):.2f}")
        col2.metric("MFI", f"{metrics.get('MFI', 0):.2f}", metrics.get('MFI_Signal', 'N/A'))
        col3.metric("Volume", f"{metrics.get('Volume', 0):,}")
        col3.metric("Volume Anomali", "🚨 Ya" if metrics.get('Volume_Anomali', False) else "-")
        
        st.subheader("Level Penting")
        st.write(f"**Support:** {' | '.join([f'{s:.2f}' for s in metrics.get('Support', [])])}")
        st.write(f"**Resistance:** {' | '.join([f'{r:.2f}' for r in metrics.get('Resistance', [])])}")
        
        st.subheader("Level Fibonacci")
        fib_cols = st.columns(4)
        fib_cols[0].metric("Fib 0.236", f"{fib.get('Fib_0.236', 0):.2f}")
        fib_cols[1].metric("Fib 0.382", f"{fib.get('Fib_0.382', 0):.2f}")
        fib_cols[2].metric("Fib 0.5", f"{fib.get('Fib_0.5', 0):.2f}")
        fib_cols[3].metric("Fib 0.618", f"{fib.get('Fib_0.618', 0):.2f}")
    except Exception as e:
        st.error(f"Gagal menampilkan indikator: {e}")

if __name__ == "__main__":
    app()
