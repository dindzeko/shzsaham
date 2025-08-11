import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import plotly.graph_objects as go
from scipy.signal import argrelextrema

# Inisialisasi session state
if 'screening_results' not in st.session_state:
    st.session_state.screening_results = None
if 'selected_ticker' not in st.session_state:
    st.session_state.selected_ticker = None

# 1. PERBAIKAN MFI
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

# 2. SUPPORT/RESISTANCE DAN FIBONACCI
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
    
    valid_support = [lvl for lvl in support_levels if not np.isnan(lvl) and lvl < current_price]
    valid_resistance = [lvl for lvl in resistance_levels if not np.isnan(lvl) and lvl > current_price]
    
    valid_support.sort(reverse=True)
    valid_resistance.sort()
    
    return {
        'Support': valid_support[:3] if valid_support else [],
        'Resistance': valid_resistance[:3] if valid_resistance else [],
        'Fibonacci': fib_levels
    }

def load_google_drive_excel(file_url):
    try:
        file_id = file_url.split("/d/")[1].split("/")[0]
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        df = pd.read_excel(download_url, engine='openpyxl')

        if 'Ticker' not in df.columns or 'Papan Pencatatan' not in df.columns:
            st.error("Kolom 'Ticker' dan 'Papan Pencatatan' harus ada di file Excel.")
            return None

        st.success("✅ Berhasil memuat data dari Google Drive!")
        st.info(f"Jumlah baris: {len(df)}")
        return df

    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        return None

def get_stock_data(ticker, end_date):
    try:
        stock = yf.Ticker(f"{ticker}.JK")
        start_date = end_date - timedelta(days=90)
        data = stock.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
        return data if not data.empty else None
    except Exception as e:
        st.error(f"Gagal mengambil data untuk {ticker}: {e}")
        return None

def detect_pattern(data):
    if len(data) < 4:
        return False
        
    recent = data.tail(4)
    c1, c2, c3, c4 = recent.iloc[0], recent.iloc[1], recent.iloc[2], recent.iloc[3]

    is_c1_bullish = c1['Close'] > c1['Open'] and (c1['Close'] - c1['Open']) > 0.015 * c1['Open']
    is_c2_bearish = c2['Close'] < c2['Open'] and c2['Close'] < c1['Close']
    is_c3_bearish = c3['Close'] < c3['Open']
    is_c4_bearish = c4['Close'] < c4['Open']
    is_uptrend = data['Close'].iloc[-20:].mean() > data['Close'].iloc[-50:-20].mean() if len(data) >= 50 else False
    is_close_sequence = c2['Close'] > c3['Close'] > c4['Close']

    return all([
        is_c1_bullish,
        is_c2_bearish,
        is_c3_bearish,
        is_c4_bearish,
        is_uptrend,
        is_close_sequence
    ])

def compute_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_additional_metrics(data):
    df = data.copy()
    metrics = {}
    
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = compute_rsi(df['Close'])
    df['MFI'] = compute_mfi(df, 14)
    
    df['Avg_Volume_20'] = df['Volume'].rolling(window=20).mean()
    vol_anomali = (df['Volume'].iloc[-1] > 1.7 * df['Avg_Volume_20'].iloc[-1]) if not df['Avg_Volume_20'].isna().iloc[-1] else False
    
    sr_levels = calculate_support_resistance(df)
    
    mfi_value = df['MFI'].iloc[-1] if not df['MFI'].empty else np.nan
    mfi_signal = interpret_mfi(mfi_value) if not np.isnan(mfi_value) else "N/A"
    
    last_row = df.iloc[-1]
    
    return {
        "MA20": round(last_row['MA20'], 2) if not np.isnan(last_row['MA20']) else None,
        "MA50": round(last_row['MA50'], 2) if not np.isnan(last_row['MA50']) else None,
        "RSI": round(last_row['RSI'], 2) if not np.isnan(last_row['RSI']) else None,
        "MFI": round(mfi_value, 2) if not np.isnan(mfi_value) else None,
        "MFI_Signal": mfi_signal,
        "Volume": int(last_row['Volume']) if not np.isnan(last_row['Volume']) else None,
        "Volume_Anomali": vol_anomali,
        "Support": sr_levels['Support'],
        "Resistance": sr_levels['Resistance'],
        "Fibonacci": sr_levels['Fibonacci']
    }

# =========== FUNGSI UTAMA: app() ===========
def app():
    st.markdown("### 🔍 Pisau Jatuh - Stock Screener")
    st.write("Gunakan alat ini untuk mendeteksi saham dengan pola *Pisau Jatuh* dan analisis teknikal mendalam.")

    file_url = "https://docs.google.com/spreadsheets/d/1t6wgBIcPEUWMq40GdIH1GtZ8dvI9PZ2v/edit?usp=drive_link"
    df = load_google_drive_excel(file_url)

    if df is None or 'Ticker' not in df.columns:
        st.info("Silakan periksa kembali file Google Spreadsheet Anda.")
        return

    tickers = df['Ticker'].dropna().unique().tolist()
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
        st.dataframe(st.session_state.screening_results, use_container_width=True)
        
        ticker_list = st.session_state.screening_results['Ticker'].tolist()
        selected_ticker = st.selectbox(
            "Pilih Saham untuk Detail",
            options=ticker_list,
            index=ticker_list.index(st.session_state.selected_ticker) if st.session_state.selected_ticker in ticker_list else 0,
            key='ticker_selector'
        )
        
        st.session_state.selected_ticker = selected_ticker
        
        if st.button("📊 Tampilkan Analisis Detail"):
            if st.session_state.selected_ticker:
                show_stock_details(st.session_state.selected_ticker, analysis_date)

def show_stock_details(ticker, end_date):
    data = get_stock_data(ticker, end_date)
    if data is None or data.empty:
        st.warning(f"Data untuk {ticker} tidak tersedia")
        return
        
    st.markdown(f"### 📊 Analisis Teknis: {ticker}")
    
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
        sr = calculate_support_resistance(data)
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
        title=f"{ticker} - Price & Technical Analysis",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 📈 Indikator Teknikal")
    try:
        metrics = calculate_additional_metrics(data)
        fib = metrics.get("Fibonacci", {})
        
        col1, col2, col3 = st.columns(3)
        col1.metric("MA20", f"{metrics.get('MA20', 0):.2f}")
        col1.metric("MA50", f"{metrics.get('MA50', 0):.2f}")
        col2.metric("RSI", f"{metrics.get('RSI', 0):.2f}")
        col2.metric("MFI", f"{metrics.get('MFI', 0):.2f}", metrics.get('MFI_Signal', 'N/A'))
        col3.metric("Volume", f"{metrics.get('Volume', 0):,}")
        col3.metric("Volume Anomali", "Ya" if metrics.get('Volume_Anomali', False) else "Tidak")
        
        st.markdown("### 🔼 Level Penting")
        st.write(f"**Support:** {' | '.join([f'{s:.2f}' for s in metrics.get('Support', [])])}")
        st.write(f"**Resistance:** {' | '.join([f'{r:.2f}' for r in metrics.get('Resistance', [])])}")
        
        st.markdown("### 🌀 Level Fibonacci")
        fib_cols = st.columns(4)
        fib_cols[0].metric("Fib 0.236", f"{fib.get('Fib_0.236', 0):.2f}")
        fib_cols[1].metric("Fib 0.382", f"{fib.get('Fib_0.382', 0):.2f}")
        fib_cols[2].metric("Fib 0.5", f"{fib.get('Fib_0.5', 0):.2f}")
        fib_cols[3].metric("Fib 0.618", f"{fib.get('Fib_0.618', 0):.2f}")
    except Exception as e:
        st.error(f"Gagal menampilkan indikator: {e}")
