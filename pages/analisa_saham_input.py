# pages/analisa_saham_input.py
import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import plotly.graph_objects as go
from scipy.signal import argrelextrema

# ==================== INDIKATOR TEKNIKAL ====================

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

def compute_macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def compute_ichimoku(df):
    high_9 = df['High'].rolling(window=9).max()
    low_9 = df['Low'].rolling(window=9).min()
    tenkan = (high_9 + low_9) / 2

    high_26 = df['High'].rolling(window=26).max()
    low_26 = df['Low'].rolling(window=26).min()
    kijun = (high_26 + low_26) / 2

    senkou_span_a = ((tenkan + kijun) / 2).shift(26)

    high_52 = df['High'].rolling(window=52).max()
    low_52 = df['Low'].rolling(window=52).min()
    senkou_span_b = ((high_52 + low_52) / 2).shift(26)

    chikou = df['Close'].shift(-26)

    return tenkan, kijun, senkou_span_a, senkou_span_b, chikou

# ==================== LEVEL HARGA ====================

def identify_significant_swings(df, window=60, min_swing_size=0.05):
    highs = df['High']
    lows = df['Low']
    max_idx = argrelextrema(highs.values, np.greater, order=5)[0]
    min_idx = argrelextrema(lows.values, np.less, order=5)[0]
    recent_highs = highs.iloc[max_idx][-10:] if len(max_idx) > 0 else pd.Series()
    recent_lows = lows.iloc[min_idx][-10:] if len(min_idx) > 0 else pd.Series()
    if len(recent_highs) == 0 or len(recent_lows) == 0:
        return df['High'].max(), df['Low'].min()
    swing_high = recent_highs.max()
    swing_low = recent_lows.min()
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

    support_levels = [fib_levels['Fib_0.618'], fib_levels['Fib_0.786'], ma20, vwap, psych_level]
    resistance_levels = [fib_levels['Fib_0.236'], fib_levels['Fib_0.382'], ma50, vwap, psych_level]

    if fib_levels['Fib_0.0'] > current_price:
        resistance_levels.append(fib_levels['Fib_0.0'])
    if fib_levels['Fib_1.0'] < current_price:
        support_levels.append(fib_levels['Fib_1.0'])

    valid_support = [lvl for lvl in support_levels if lvl < current_price]
    valid_resistance = [lvl for lvl in resistance_levels if lvl > current_price]
    valid_support.sort(reverse=True)
    valid_resistance.sort()
    return {'Support': valid_support[:3], 'Resistance': valid_resistance[:3], 'Fibonacci': fib_levels}

# ==================== DATA ====================

def get_stock_data(ticker, end_date):
    try:
        stock = yf.Ticker(f"{ticker}.JK")
        start_date = end_date - timedelta(days=180)
        data = stock.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
        return data if not data.empty else None
    except Exception as e:
        st.error(f"Gagal mengambil data untuk {ticker}: {e}")
        return None

# ==================== APLIKASI ====================

def app():
    st.title("📈 Analisa Teknikal Saham")

    ticker_input = st.text_input("Masukkan Kode Saham (contoh: BBCA)", value="BBCA")
    analysis_date = st.date_input("📅 Tanggal Analisis", value=datetime.today())

    if st.button("Mulai Analisis"):
        ticker = ticker_input.strip().upper()
        data = get_stock_data(ticker, analysis_date)

        if data is None or data.empty:
            st.warning(f"Data untuk {ticker} tidak tersedia.")
            return

        df = data.copy()
        df['MA20'] = df['Close'].rolling(20).mean()
        df['MA50'] = df['Close'].rolling(50).mean()
        df['RSI'] = compute_rsi(df['Close'])
        df['MFI'] = compute_mfi(df, 14)
        macd_line, signal_line, hist = compute_macd(df['Close'])
        df['MACD'], df['Signal'], df['Hist'] = macd_line, signal_line, hist
        tenkan, kijun, senkou_a, senkou_b, chikou = compute_ichimoku(df)
        df['Tenkan'], df['Kijun'], df['SenkouA'], df['SenkouB'], df['Chikou'] = tenkan, kijun, senkou_a, senkou_b, chikou

        sr = calculate_support_resistance(df)
        fib = sr['Fibonacci']
        mfi_value = df['MFI'].iloc[-1]
        mfi_signal = interpret_mfi(mfi_value)

        # Volume Anomali
        df['Avg_Volume_20'] = df['Volume'].rolling(20).mean()
        vol_anomali = df['Volume'].iloc[-1] > 1.7 * df['Avg_Volume_20'].iloc[-1]

        # Plot Chart
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                     low=df['Low'], close=df['Close'],
                                     name='Candlestick',
                                     increasing_line_color='green',
                                     decreasing_line_color='red'))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], mode='lines', name='MA20', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], mode='lines', name='MA50', line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=df.index, y=tenkan, mode='lines', name='Tenkan', line=dict(color='purple')))
        fig.add_trace(go.Scatter(x=df.index, y=kijun, mode='lines', name='Kijun', line=dict(color='brown')))
        fig.add_trace(go.Scatter(x=df.index, y=senkou_a, mode='lines', name='Senkou A', line=dict(color='green', dash='dot')))
        fig.add_trace(go.Scatter(x=df.index, y=senkou_b, mode='lines', name='Senkou B', line=dict(color='red', dash='dot')))

        # Support / Resistance
        for level in sr['Support']:
            fig.add_hline(y=level, line_dash="dash", line_color="green", annotation_text=f"S: {level:.2f}")
        for level in sr['Resistance']:
            fig.add_hline(y=level, line_dash="dash", line_color="red", annotation_text=f"R: {level:.2f}")

        fig.update_layout(title=f"{ticker}.JK - Analisa Teknikal",
                          xaxis_rangeslider_visible=False,
                          template="plotly_white",
                          height=700)

        st.plotly_chart(fig, use_container_width=True)

        # Indikator
        st.subheader("📊 Indikator Teknikal")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("MA20", f"{df['MA20'].iloc[-1]:.2f}")
            st.metric("MA50", f"{df['MA50'].iloc[-1]:.2f}")
        with c2:
            st.metric("RSI", f"{df['RSI'].iloc[-1]:.2f}")
            st.metric("MFI", f"{mfi_value:.2f}", mfi_signal)
        with c3:
            st.metric("MACD", f"{df['MACD'].iloc[-1]:.2f}")
            st.metric("Signal", f"{df['Signal'].iloc[-1]:.2f}")

        # Kesimpulan
        st.subheader("📌 Kesimpulan")
        kesimpulan = []
        if df['RSI'].iloc[-1] < 30: kesimpulan.append("RSI menunjukkan **Oversold** → potensi rebound.")
        elif df['RSI'].iloc[-1] > 70: kesimpulan.append("RSI menunjukkan **Overbought** → potensi koreksi.")
        if df['MACD'].iloc[-1] > df['Signal'].iloc[-1]: kesimpulan.append("MACD bullish crossover.")
        else: kesimpulan.append("MACD bearish crossover.")
        if df['Close'].iloc[-1] > df['SenkouA'].iloc[-1] and df['Close'].iloc[-1] > df['SenkouB'].iloc[-1]:
            kesimpulan.append("Harga berada **di atas awan Ichimoku** → tren bullish.")
        else:
            kesimpulan.append("Harga berada **di bawah/di dalam awan Ichimoku** → tren lemah/bearish.")
        if vol_anomali: kesimpulan.append("🚨 Terdapat lonjakan volume anomali.")

        if kesimpulan:
            for k in kesimpulan:
                st.write("- " + k)
