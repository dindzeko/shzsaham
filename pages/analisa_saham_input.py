import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go

# ==============================
# 🔹 Fungsi Indikator
# ==============================
def compute_rsi(data, window=14):
    delta = data.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=window).mean()
    avg_loss = pd.Series(loss).rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return pd.Series(rsi, index=data.index)

def compute_mfi(df, window=14):
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = typical_price * df['Volume']
    positive_flow = []
    negative_flow = []
    for i in range(1, len(df)):
        if typical_price[i] > typical_price[i-1]:
            positive_flow.append(money_flow[i])
            negative_flow.append(0)
        else:
            positive_flow.append(0)
            negative_flow.append(money_flow[i])
    positive_mf = pd.Series(positive_flow).rolling(window=window).sum()
    negative_mf = pd.Series(negative_flow).rolling(window=window).sum()
    mfi = 100 * (positive_mf / (positive_mf + negative_mf))
    mfi = pd.Series([np.nan] + list(mfi), index=df.index)
    return mfi

def compute_macd(close, short=12, long=26, signal=9):
    ema_short = close.ewm(span=short, adjust=False).mean()
    ema_long = close.ewm(span=long, adjust=False).mean()
    macd = ema_short - ema_long
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    return macd, signal_line, hist

def compute_ichimoku(df):
    nine_high = df['High'].rolling(window=9).max()
    nine_low = df['Low'].rolling(window=9).min()
    tenkan_sen = (nine_high + nine_low) / 2

    period26_high = df['High'].rolling(window=26).max()
    period26_low = df['Low'].rolling(window=26).min()
    kijun_sen = (period26_high + period26_low) / 2

    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)

    period52_high = df['High'].rolling(window=52).max()
    period52_low = df['Low'].rolling(window=52).min()
    senkou_span_b = ((period52_high + period52_low) / 2).shift(26)

    chikou_span = df['Close'].shift(-26)

    return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span

def fibonacci_levels(df):
    max_price = df['Close'].max()
    min_price = df['Close'].min()
    diff = max_price - min_price
    levels = {
        "0.0%": max_price,
        "23.6%": max_price - 0.236 * diff,
        "38.2%": max_price - 0.382 * diff,
        "50.0%": max_price - 0.5 * diff,
        "61.8%": max_price - 0.618 * diff,
        "100.0%": min_price
    }
    return levels

# ==============================
# 🔹 Aplikasi Utama
# ==============================
def app():
    st.title("📈 Analisa Saham Otomatis dengan Indikator Teknis")

    ticker = st.text_input("Masukkan kode saham (contoh: BBCA.JK):", "BBCA.JK")
    start_date = st.date_input("Tanggal mulai", pd.to_datetime("2023-01-01"))
    end_date = st.date_input("Tanggal akhir", pd.to_datetime("today"))

    if st.button("Analisa"):
        df = yf.download(ticker, start=start_date, end=end_date)

        if df.empty:
            st.error("Data tidak tersedia. Coba ganti ticker atau tanggal.")
            return

        # =========================
        # Hitung indikator teknikal
        # =========================
        df['RSI'] = compute_rsi(df['Close'])
        df['MFI'] = compute_mfi(df)
        macd, macd_signal, macd_hist = compute_macd(df['Close'])
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = macd, macd_signal, macd_hist
        tenkan, kijun, senkou_a, senkou_b, chikou = compute_ichimoku(df)
        df['Tenkan'], df['Kijun'], df['Senkou_A'], df['Senkou_B'], df['Chikou'] = tenkan, kijun, senkou_a, senkou_b, chikou

        # =========================
        # Plot Candlestick + MA + Ichimoku + Fibonacci
        # =========================
        fig = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'],
            name="Candlestick"
        )])

        # Ichimoku
        fig.add_trace(go.Scatter(x=df.index, y=df['Tenkan'], mode="lines", name="Tenkan-sen", line=dict(color="red", width=1)))
        fig.add_trace(go.Scatter(x=df.index, y=df['Kijun'], mode="lines", name="Kijun-sen", line=dict(color="blue", width=1)))
        fig.add_trace(go.Scatter(x=df.index, y=df['Senkou_A'], mode="lines", name="Senkou A", line=dict(color="green", width=1), opacity=0.5))
        fig.add_trace(go.Scatter(x=df.index, y=df['Senkou_B'], mode="lines", name="Senkou B", line=dict(color="orange", width=1), opacity=0.5))
        fig.add_traces([
            go.Scatter(x=df.index, y=df['Senkou_A'], line=dict(width=0), showlegend=False),
            go.Scatter(x=df.index, y=df['Senkou_B'], fill='tonexty', mode='none', fillcolor='rgba(200,200,250,0.2)', showlegend=False)
        ])

        # Fibonacci retracement
        fibo = fibonacci_levels(df)
        for level, val in fibo.items():
            fig.add_hline(y=val, line_dash="dot", annotation_text=f"Fibo {level}", annotation_position="top right")

        fig.update_layout(title=f"Chart {ticker}", xaxis_rangeslider_visible=False, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        # =========================
        # MACD Chart
        # =========================
        macd_fig = go.Figure()
        macd_fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode="lines", name="MACD", line=dict(color="blue")))
        macd_fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], mode="lines", name="Signal", line=dict(color="red")))
        macd_fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name="Histogram", marker_color="gray"))
        macd_fig.update_layout(title="MACD Indicator", xaxis_title="Date", yaxis_title="Value", template="plotly_white", height=300)
        st.plotly_chart(macd_fig, use_container_width=True)

        # =========================
        # Indikator & Kesimpulan
        # =========================
        st.subheader("📊 Indikator Teknis")
        rsi_val, mfi_val = df['RSI'].iloc[-1], df['MFI'].iloc[-1]
        st.write(f"RSI terakhir: {rsi_val:.2f}")
        st.write(f"MFI terakhir: {mfi_val:.2f}")

        st.subheader("📝 Kesimpulan Analisis")
        kesimpulan = []

        if rsi_val < 30:
            kesimpulan.append("RSI menunjukkan **Oversold** → potensi rebound.")
        elif rsi_val > 70:
            kesimpulan.append("RSI menunjukkan **Overbought** → potensi koreksi.")

        if mfi_val < 20:
            kesimpulan.append("MFI di area oversold → peluang beli.")
        elif mfi_val > 80:
            kesimpulan.append("MFI di area overbought → waspada jual.")

        if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1]:
            kesimpulan.append("MACD **Bullish Crossover** → momentum naik.")
        else:
            kesimpulan.append("MACD **Bearish Crossover** → momentum turun.")

        if df['Close'].iloc[-1] > df['Senkou_A'].iloc[-1] and df['Close'].iloc[-1] > df['Senkou_B'].iloc[-1]:
            kesimpulan.append("Harga berada **di atas Ichimoku Cloud** → tren naik.")
        elif df['Close'].iloc[-1] < df['Senkou_A'].iloc[-1] and df['Close'].iloc[-1] < df['Senkou_B'].iloc[-1]:
            kesimpulan.append("Harga berada **di bawah Ichimoku Cloud** → tren turun.")
        else:
            kesimpulan.append("Harga berada **di dalam Ichimoku Cloud** → konsolidasi.")

        if not kesimpulan:
            st.write("Tidak ada sinyal kuat saat ini.")
        else:
            for k in kesimpulan:
                st.write(f"- {k}")

# ==============================
if __name__ == "__main__":
    app()
