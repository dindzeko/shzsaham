# pages/analisa_saham_input.py
import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import plotly.graph_objects as go
from scipy.signal import argrelextrema

# --- FUNGSI ANALISIS TEKNIKAL ---
def compute_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_mfi(df, period=14):
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = tp * df['Volume']
    positive_flow = [0]
    negative_flow = [0]
    for i in range(1, len(tp)):
        if tp.iloc[i] > tp.iloc[i-1]:
            positive_flow.append(money_flow.iloc[i-1])
            negative_flow.append(0)
        elif tp.iloc[i] < tp.iloc[i-1]:
            positive_flow.append(0)
            negative_flow.append(money_flow.iloc[i-1])
        else:
            positive_flow.append(money_flow.iloc[i-1])
            negative_flow.append(money_flow.iloc[i-1])
    pos_series = pd.Series(positive_flow)
    neg_series = pd.Series(negative_flow)
    pos_mf = pos_series.rolling(window=period, min_periods=1).sum()
    neg_mf = neg_series.rolling(window=period, min_periods=1).sum()
    ratio = np.where(neg_mf > 0, pos_mf / neg_mf, 1.0)
    mfi = 100 - (100 / (1 + ratio))
    return pd.Series(mfi, index=df.index)

def interpret_mfi(mfi_value):
    if mfi_value >= 80:
        return "🔴 Overbought"
    elif mfi_value >= 65:
        return "🟢 Bullish"
    elif mfi_value <= 20:
        return "🟢 Oversold"
    elif mfi_value <= 35:
        return "🔴 Bearish"
    else:
        return "⚪ Neutral"

def identify_significant_swings(df, window=60, min_swing_size=0.05):
    highs = df['High']
    lows = df['Low']
    max_idx = argrelextrema(highs.values, np.greater, order=5)[0]
    min_idx = argrelextrema(lows.values, np.less, order=5)[0]
    recent_highs = highs.iloc[max_idx][-10:] if len(max_idx) > 0 else pd.Series()
    recent_lows = lows.iloc[min_idx][-10:] if len(min_idx) > 0 else pd.Series()
    if len(recent_highs) == 0 or len(recent_lows) == 0:
        return df['High'].max(), df['Low'].min()
    significant_highs = []
    significant_lows = []
    for i in range(1, len(recent_highs)):
        change = (recent_highs.iloc[i] - recent_highs.iloc[i-1]) / recent_highs.iloc[i-1]
        if abs(change) > min_swing_size:
            significant_highs.append(recent_highs.iloc[i])
    for i in range(1, len(recent_lows)):
        change = (recent_lows.iloc[i] - recent_lows.iloc[i-1]) / recent_lows.iloc[i-1]
        if abs(change) > min_swing_size:
            significant_lows.append(recent_lows.iloc[i])
    swing_high = max(significant_highs) if significant_highs else recent_highs.max()
    swing_low = min(significant_lows) if significant_lows else recent_lows.min()
    return swing_high, swing_low

def calculate_fibonacci_levels(swing_high, swing_low):
    diff = swing_high - swing_low
    return {
        'Fib_0.0': round(swing_high, 2),
        'Fib_0.236': round(swing_high - 0.236 * diff, 2),
        'Fib_0.382': round(swing_high - 0.382 * diff, 2),
        'Fib_0.5': round(swing_high - 0.5 * diff, 2),
        'Fib_0.618': round(swing_high - 0.618 * diff, 2),
        'Fib_0.786': round(swing_high - 0.786 * diff, 2),
        'Fib_1.0': round(swing_low, 2)
    }

def calculate_vwap(df):
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    vwap = (tp * df['Volume']).cumsum() / df['Volume'].cumsum()
    return vwap

def find_psychological_levels(close_price):
    levels = [50, 100, 200, 500, 1000, 2000, 5000]
    closest_level = min(levels, key=lambda x: abs(x - close_price))
    return closest_level

def calculate_support_resistance(data):
    df = data.copy()
    current_price = df['Close'].iloc[-1]
    swing_high, swing_low = identify_significant_swings(df.tail(60))
    fib_levels = calculate_fibonacci_levels(swing_high, swing_low)
    ma20 = df['Close'].rolling(20).mean().iloc[-1]
    ma50 = df['Close'].rolling(50).mean().iloc[-1]
    vwap = calculate_vwap(df).iloc[-1]
    psych_level = find_psychological_levels(current_price)
    support_levels = [
        fib_levels['Fib_0.618'], 
        fib_levels['Fib_0.786'],
        ma20,
        vwap,
        psych_level
    ]
    resistance_levels = [
        fib_levels['Fib_0.236'], 
        fib_levels['Fib_0.382'],
        ma50,
        vwap,
        psych_level
    ]
    # Tambahkan Fib_0.0 sebagai resistance & Fib_1.0 sebagai support
    if not np.isnan(fib_levels['Fib_0.0']) and fib_levels['Fib_0.0'] > current_price:
        resistance_levels.append(fib_levels['Fib_0.0'])
    if not np.isnan(fib_levels['Fib_1.0']) and fib_levels['Fib_1.0'] < current_price:
        support_levels.append(fib_levels['Fib_1.0'])

    valid_support = [lvl for lvl in support_levels if not np.isnan(lvl) and lvl < current_price]
    valid_resistance = [lvl for lvl in resistance_levels if not np.isnan(lvl) and lvl > current_price]
    valid_support.sort(reverse=True)
    valid_resistance.sort()
    return {
        'Support': valid_support[:3] if valid_support else [],
        'Resistance': valid_resistance[:3] if valid_resistance else [],
        'Fibonacci': fib_levels
    }

def get_stock_data(ticker, end_date):
    try:
        stock = yf.Ticker(f"{ticker}.JK")
        start_date = end_date - timedelta(days=90)
        data = stock.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
        return data if not data.empty else None
    except Exception as e:
        st.error(f"Gagal mengambil data untuk {ticker}: {e}")
        return None

# --- FUNGSI UTAMA ---
def app():
    st.title("📈 Analisa Teknikal Saham")

    ticker_input = st.text_input("Masukkan Kode Saham (contoh: BBCA.JK)", value="BBCA.JK")
    analysis_date = st.date_input("📅 Tanggal Analisis", value=datetime.today())

    if st.button("Mulai Analisis"):
        if not ticker_input.strip():
            st.warning("Silakan masukkan kode saham.")
            return

        ticker = ticker_input.replace(".JK", "") + ".JK"
        data = get_stock_data(ticker.replace(".JK", ""), analysis_date)

        if data is None or data.empty:
            st.warning(f"Data untuk {ticker} tidak tersedia.")
            return

        # Hitung semua indikator
        df = data.copy()
        df['MA20'] = df['Close'].rolling(20).mean()
        df['MA50'] = df['Close'].rolling(50).mean()
        df['RSI'] = compute_rsi(df['Close'])
        df['MFI'] = compute_mfi(df, 14)
        sr = calculate_support_resistance(df)
        fib = sr['Fibonacci']
        mfi_value = df['MFI'].iloc[-1] if not df['MFI'].empty else np.nan
        mfi_signal = interpret_mfi(mfi_value) if not np.isnan(mfi_value) else "N/A"

        # Volume Anomali
        df['Avg_Volume_20'] = df['Volume'].rolling(window=20).mean()
        vol_anomali = (df['Volume'].iloc[-1] > 1.7 * df['Avg_Volume_20'].iloc[-1]) if not df['Avg_Volume_20'].isna().iloc[-1] else False

        # --- PLOT GRAFIK (PLOTLY) ---
        fig = go.Figure()

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Candlestick',
            increasing_line_color='green',
            decreasing_line_color='red'
        ))

        # MA20 & MA50
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['MA20'],
            mode='lines',
            name='MA20',
            line=dict(color='blue', width=1)
        ))
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['MA50'],
            mode='lines',
            name='MA50',
            line=dict(color='orange', width=1)
        ))

        # Support & Resistance
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

        # Fibonacci Levels — TERMASUK FIB_0.0 DAN FIB_1.0
        fib_keys = ['Fib_0.0', 'Fib_0.236', 'Fib_0.382', 'Fib_0.5', 'Fib_0.618', 'Fib_0.786', 'Fib_1.0']
        for key in fib_keys:
            if key in fib and not np.isnan(fib[key]):
                # Warna khusus untuk Fib_0.0 (magenta) dan Fib_1.0 (blue)
                color = "magenta" if key == 'Fib_0.0' else "blue" if key == 'Fib_1.0' else "purple"
                dash = "solid" if key in ['Fib_0.0', 'Fib_1.0'] else "dot"
                position = "top left" if key in ['Fib_0.0', 'Fib_0.236', 'Fib_0.382'] else "bottom left"
                position = "bottom left" if key == 'Fib_1.0' else position

                fig.add_hline(
                    y=fib[key],
                    line_dash=dash,
                    line_color=color,
                    annotation_text=f"{key}: {fib[key]:.2f}",
                    annotation_position=position
                )

        # Layout
        fig.update_layout(
            title=f"{ticker} Price Analysis",
            xaxis_title="Date",
            yaxis_title="Price (Rp)",
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_white",
            height=600
        )

        # Tampilkan di Streamlit
        st.plotly_chart(fig, use_container_width=True)

        # --- TAMPILKAN INDIKATOR TEKNIKAL ---
        st.subheader("📊 Indikator Teknikal")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("MA20", f"{df['MA20'].iloc[-1]:.2f}" if not np.isnan(df['MA20'].iloc[-1]) else "N/A")
            st.metric("MA50", f"{df['MA50'].iloc[-1]:.2f}" if not np.isnan(df['MA50'].iloc[-1]) else "N/A")

        with col2:
            st.metric("RSI", f"{df['RSI'].iloc[-1]:.2f}" if not np.isnan(df['RSI'].iloc[-1]) else "N/A")
            st.metric("MFI", f"{mfi_value:.2f}" if not np.isnan(mfi_value) else "N/A", mfi_signal)

        with col3:
            st.metric("Volume", f"{int(df['Volume'].iloc[-1]):,}" if not np.isnan(df['Volume'].iloc[-1]) else "N/A")
            st.metric("Volume Anomali", "🚨 Ya" if vol_anomali else "Tidak")

        # --- LEVEL PENTING ---
        st.subheader("📍 Level Penting")
        if sr['Support']:
            st.write(f"**Support:** {' | '.join([f'{s:.2f}' for s in sr['Support']])}")
        if sr['Resistance']:
            st.write(f"**Resistance:** {' | '.join([f'{r:.2f}' for r in sr['Resistance']])}")

        # --- LEVEL FIBONACCI ---
        st.subheader("🔢 Level Fibonacci")
        fib_display = {k: v for k, v in fib.items() if k in ['Fib_0.0', 'Fib_0.236', 'Fib_0.382', 'Fib_0.5', 'Fib_0.618', 'Fib_1.0']}
        if fib_display:
            cols = st.columns(len(fib_display))
            for i, (key, value) in enumerate(fib_display.items()):
                cols[i].metric(key.replace('Fib_', 'Fib '), f"{value:.2f}")
