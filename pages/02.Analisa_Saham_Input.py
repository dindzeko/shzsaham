from utils import *

def show_analisa_saham_input_page():
    st.title("📈 Analisa Saham Sesuai Kode Input")
    
    st.write("""
    Masukkan kode saham yang ingin Anda analisis. Sistem akan menggunakan logika analisa yang sama seperti pada fitur screening.
    """)
    
    ticker_input = st.text_input("Masukkan kode saham (misal: BBCA)", placeholder="Contoh: BBCA")
    analysis_date = st.date_input("📅 Tanggal Analisis", value=datetime.today())
    
    if st.button("🔍 Lakukan Analisa"):
        if ticker_input:
            data = get_stock_data(ticker_input, analysis_date)
            if data is not None and not data.empty:
                st.success(f"Data untuk {ticker_input} berhasil diambil!")
                
                # Tampilkan chart
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name='Candlestick'
                ))
                
                # Tambahkan MA
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
                
                # Support/Resistance dan Fibonacci
                sr = calculate_support_resistance(data)
                fib = sr['Fibonacci']
                
                # Tambahkan level Support/Resistance
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
                
                # Tambahkan level Fibonacci
                for key, value in fib.items():
                    if "Fib" in key:
                        fig.add_hline(
                            y=value,
                            line_dash="dot",
                            line_color="purple",
                            annotation_text=f"{key}: {value:.2f}",
                            annotation_position="top left" if "0." in key else "bottom left"
                        )
                
                # Layout chart
                fig.update_layout(
                    title=f"{ticker_input} Price Analysis",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    xaxis_rangeslider_visible=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Tampilkan indikator
                metrics = calculate_additional_metrics(data)
                fib = metrics.get("Fibonacci", {})
                
                col1, col2, col3 = st.columns(3)
                col1.metric("MA20", f"{metrics.get('MA20', 0):.2f}")
                col1.metric("MA50", f"{metrics.get('MA50', 0):.2f}")
                col2.metric("RSI", f"{metrics.get('RSI', 0):.2f}")
                col2.metric("MFI", f"{metrics.get('MFI', 0):.2f}", metrics.get('MFI_Signal', 'N/A'))
                col3.metric("Volume", f"{metrics.get('Volume', 0):,}")
                col3.metric("Volume Anomali", "Ya" if metrics.get('Volume_Anomali', False) else "Tidak")
                
                st.subheader("Level Penting")
                st.write(f"**Support:** {' | '.join([f'{s:.2f}' for s in metrics.get('Support', [])])}")
                st.write(f"**Resistance:** {' | '.join([f'{r:.2f}' for r in metrics.get('Resistance', [])])}")
                
                st.subheader("Level Fibonacci")
                fib_cols = st.columns(4)
                fib_cols[0].metric("Fib 0.236", f"{fib.get('Fib_0.236', 0):.2f}")
                fib_cols[1].metric("Fib 0.382", f"{fib.get('Fib_0.382', 0):.2f}")
                fib_cols[2].metric("Fib 0.5", f"{fib.get('Fib_0.5', 0):.2f}")
                fib_cols[3].metric("Fib 0.618", f"{fib.get('Fib_0.618', 0):.2f}")
            else:
                st.error(f"Tidak dapat mengambil data untuk {ticker_input}")
        else:
            st.warning("Silakan masukkan kode saham terlebih dahulu")
