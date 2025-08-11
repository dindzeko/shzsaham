# modules/screener_pisau_jatuh.py
import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import plotly.graph_objects as go
from scipy.signal import argrelextrema

# -------------------- HELPERS --------------------
def _fmt_num(v, d=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    try:
        # show thousands separator
        if isinstance(v, int):
            return f"{v:,}"
        return f"{v:,.{d}f}"
    except Exception:
        return str(v)

# -------------------- MFI --------------------
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

    pos_series = pd.Series(positive_flow, index=df.index[:len(positive_flow)])
    neg_series = pd.Series(negative_flow, index=df.index[:len(negative_flow)])

    pos_mf = pos_series.rolling(window=period, min_periods=1).sum()
    neg_mf = neg_series.rolling(window=period, min_periods=1).sum()

    ratio = np.where(neg_mf > 0, pos_mf / neg_mf, 1.0)
    mfi = 100 - (100 / (1 + ratio))

    # return aligned series
    return pd.Series(mfi, index=pos_mf.index)

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

# -------------------- SUPPORT / FIB --------------------
def identify_significant_swings(df, order=5, min_swing_size=0.05):
    highs = df['High']
    lows = df['Low']

    try:
        max_idx = argrelextrema(highs.values, np.greater, order=order)[0]
        min_idx = argrelextrema(lows.values, np.less, order=order)[0]
    except Exception:
        max_idx = np.array([])
        min_idx = np.array([])

    recent_highs = highs.iloc[max_idx][-10:] if len(max_idx) > 0 else pd.Series(dtype=float)
    recent_lows = lows.iloc[min_idx][-10:] if len(min_idx) > 0 else pd.Series(dtype=float)

    if recent_highs.empty or recent_lows.empty:
        return df['High'].max(), df['Low'].min()

    significant_highs = []
    significant_lows = []

    for i in range(1, len(recent_highs)):
        prev = recent_highs.iloc[i-1]
        cur = recent_highs.iloc[i]
        if prev != 0 and abs((cur - prev) / prev) > min_swing_size:
            significant_highs.append(cur)

    for i in range(1, len(recent_lows)):
        prev = recent_lows.iloc[i-1]
        cur = recent_lows.iloc[i]
        if prev != 0 and abs((cur - prev) / prev) > min_swing_size:
            significant_lows.append(cur)

    swing_high = max(significant_highs) if significant_highs else recent_highs.max()
    swing_low = min(significant_lows) if significant_lows else recent_lows.min()

    return swing_high, swing_low

def calculate_fibonacci_levels(swing_high, swing_low):
    diff = swing_high - swing_low if swing_high is not None and swing_low is not None else 0
    return {
        'Fib_0.0': round(swing_high, 2) if diff != 0 else None,
        'Fib_0.236': round(swing_high - 0.236 * diff, 2) if diff != 0 else None,
        'Fib_0.382': round(swing_high - 0.382 * diff, 2) if diff != 0 else None,
        'Fib_0.5': round(swing_high - 0.5 * diff, 2) if diff != 0 else None,
        'Fib_0.618': round(swing_high - 0.618 * diff, 2) if diff != 0 else None,
        'Fib_0.786': round(swing_high - 0.786 * diff, 2) if diff != 0 else None,
        'Fib_1.0': round(swing_low, 2) if diff != 0 else None
    }

def calculate_vwap(df):
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    cum_vol = df['Volume'].cumsum()
    # avoid division by zero
    with np.errstate(invalid='ignore', divide='ignore'):
        vwap = (tp * df['Volume']).cumsum() / cum_vol
    return vwap

def find_psychological_levels(close_price):
    levels = [50, 100, 200, 500, 1000, 2000, 5000]
    return min(levels, key=lambda x: abs(x - close_price))

def calculate_support_resistance(data):
    df = data.copy()
    if df.empty:
        return {'Support': [], 'Resistance': [], 'Fibonacci': {}}

    current_price = df['Close'].iloc[-1]

    swing_high, swing_low = identify_significant_swings(df.tail(60))
    fib_levels = calculate_fibonacci_levels(swing_high, swing_low)

    ma20 = df['Close'].rolling(20).mean().iloc[-1] if len(df) >= 20 else np.nan
    ma50 = df['Close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else np.nan
    vwap = calculate_vwap(df).iloc[-1] if not calculate_vwap(df).empty else np.nan
    psych_level = find_psychological_levels(current_price)

    support_levels = [fib_levels.get('Fib_0.618'), fib_levels.get('Fib_0.786'), ma20, vwap, psych_level]
    resistance_levels = [fib_levels.get('Fib_0.236'), fib_levels.get('Fib_0.382'), ma50, vwap, psych_level]

    valid_support = [lvl for lvl in support_levels if lvl is not None and not (isinstance(lvl, float) and np.isnan(lvl)) and lvl < current_price]
    valid_resistance = [lvl for lvl in resistance_levels if lvl is not None and not (isinstance(lvl, float) and np.isnan(lvl)) and lvl > current_price]

    valid_support.sort(reverse=True)
    valid_resistance.sort()

    return {
        'Support': valid_support[:3] if valid_support else [],
        'Resistance': valid_resistance[:3] if valid_resistance else [],
        'Fibonacci': fib_levels
    }

# -------------------- DATA FETCH --------------------
@st.cache_data
def load_google_drive_excel(file_url):
    try:
        file_id = file_url.split("/d/")[1].split("/")[0]
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        df = pd.read_excel(download_url, engine='openpyxl')
        return df
    except Exception:
        return None

@st.cache_data
def get_stock_data(ticker, end_date):
    try:
        # end_date may be datetime.date -> ok
        stock = yf.Ticker(f"{ticker}.JK")
        start_date = end_date - timedelta(days=90)
        data = stock.history(start=start_date.strftime('%Y-%m-%d'), end=(end_date + timedelta(days=1)).strftime('%Y-%m-%d'))
        if data is None or data.empty:
            return None
        data = data.dropna(subset=['Close'])
        return data
    except Exception:
        return None

# -------------------- PATTERN DETECTION --------------------
def detect_pattern(data):
    if data is None or len(data) < 4:
        return False
    recent = data.tail(4)
    c1, c2, c3, c4 = recent.iloc[0], recent.iloc[1], recent.iloc[2], recent.iloc[3]
    is_c1_bullish = c1['Close'] > c1['Open'] and (c1['Close'] - c1['Open']) > 0.015 * c1['Open']
    is_c2_bearish = c2['Close'] < c2['Open'] and c2['Close'] < c1['Close']
    is_c3_bearish = c3['Close'] < c3['Open']
    is_c4_bearish = c4['Close'] < c4['Open']
    is_uptrend = False
    if len(data) >= 50:
        is_uptrend = data['Close'].iloc[-20:].mean() > data['Close'].iloc[-50:-20].mean()
    is_close_sequence = (c2['Close'] > c3['Close'] > c4['Close'])
    return all([is_c1_bullish, is_c2_bearish, is_c3_bearish, is_c4_bearish, is_uptrend, is_close_sequence])

# -------------------- INDICATORS --------------------
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
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = compute_rsi(df['Close'])
    df['MFI'] = compute_mfi(df, 14)
    df['Avg_Volume_20'] = df['Volume'].rolling(window=20).mean()
    vol_anomali = False
    if not df['Avg_Volume_20'].isna().iloc[-1]:
        vol_anomali = df['Volume'].iloc[-1] > 1.7 * df['Avg_Volume_20'].iloc[-1]
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

# -------------------- SHOW DETAILS --------------------
def show_stock_details(ticker, end_date):
    data = get_stock_data(ticker, end_date)
    if data is None or data.empty:
        st.warning(f"Data untuk {ticker} tidak tersedia.")
        return

    st.markdown(f"### 📊 Analisis Teknis: {ticker}")

    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name='Candlestick'))

    data['MA20'] = data['Close'].rolling(20).mean()
    data['MA50'] = data['Close'].rolling(50).mean()
    fig.add_trace(go.Scatter(x=data.index, y=data['MA20'], name='MA20', line=dict(width=1)))
    fig.add_trace(go.Scatter(x=data.index, y=data['MA50'], name='MA50', line=dict(width=1)))

    try:
        sr = calculate_support_resistance(data)
        fib = sr.get('Fibonacci', {})
        for level in sr.get('Support', []):
            fig.add_hline(y=level, line_dash="dash", annotation_text=f"Support: {level:.2f}", annotation_position="bottom right")
        for level in sr.get('Resistance', []):
            fig.add_hline(y=level, line_dash="dash", annotation_text=f"Resistance: {level:.2f}", annotation_position="top right")
        for key, value in fib.items():
            if value is not None:
                fig.add_hline(y=value, line_dash="dot", annotation_text=f"{key}: {value:.2f}", annotation_position="top left")
    except Exception:
        pass

    fig.update_layout(title=f"{ticker} - Price & Technical Analysis", xaxis_title="Date", yaxis_title="Price", xaxis_rangeslider_visible=False, height=600)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📈 Indikator Teknikal")
    metrics = calculate_additional_metrics(data)
    col1, col2, col3 = st.columns(3)
    col1.metric("MA20", _fmt_num(metrics.get('MA20')))
    col1.metric("MA50", _fmt_num(metrics.get('MA50')))
    col2.metric("RSI", _fmt_num(metrics.get('RSI')))
    col2.metric("MFI", _fmt_num(metrics.get('MFI')), metrics.get('MFI_Signal', 'N/A'))
    col3.metric("Volume", _fmt_num(metrics.get('Volume'), d=0))
    col3.metric("Volume Anomali", "Ya" if metrics.get('Volume_Anomali') else "Tidak")

    st.markdown("### 🔼 Level Penting")
    st.write(f"**Support:** {' | '.join([f'{s:.2f}' for s in metrics.get('Support', [])]) or '—'}")
    st.write(f"**Resistance:** {' | '.join([f'{r:.2f}' for r in metrics.get('Resistance', [])]) or '—'}")

    st.markdown("### 🌀 Level Fibonacci")
    fib = metrics.get('Fibonacci', {})
    fib_cols = st.columns(4)
    fib_cols[0].metric("Fib 0.236", _fmt_num(fib.get('Fib_0.236')))
    fib_cols[1].metric("Fib 0.382", _fmt_num(fib.get('Fib_0.382')))
    fib_cols[2].metric("Fib 0.5", _fmt_num(fib.get('Fib_0.5')))
    fib_cols[3].metric("Fib 0.618", _fmt_num(fib.get('Fib_0.618')))

# -------------------- MAIN APP --------------------
def app(file_url: str = "https://docs.google.com/spreadsheets/d/1t6wgBIcPEUWMq40GdIH1GtZ8dvI9PZ2v/edit?usp=drive_link"):
    # initialize session_state keys safely (fix for the error you saw)
    st.session_state.setdefault('screening_results', None)
    st.session_state.setdefault('selected_ticker', None)

    st.markdown("### 🔍 Pisau Jatuh - Stock Screener")

    df = load_google_drive_excel(file_url)
    if df is None:
        st.error("Gagal memuat Google Sheet. Periksa URL / akses berbagi.")
        return

    if 'Ticker' not in df.columns or 'Papan Pencatatan' not in df.columns:
        st.error("Kolom 'Ticker' dan 'Papan Pencatatan' harus ada di file.")
        return

    tickers = df['Ticker'].dropna().unique().tolist()
    analysis_date = st.date_input("📅 Tanggal Analisis", value=datetime.today())

    if st.button("🔍 Mulai Screening"):
        results = []
        progress_bar = st.progress(0)
        progress_text = st.empty()
        for i, ticker in enumerate(tickers):
            data = get_stock_data(ticker, analysis_date)
            if data is not None and len(data) >= 50 and detect_pattern(data):
                metrics = calculate_additional_metrics(data)
                papan_vals = df.loc[df['Ticker'] == ticker, 'Papan Pencatatan'].values
                papan = papan_vals[0] if len(papan_vals) > 0 else "-"
                fib = metrics.get("Fibonacci", {})
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
                    "Support": " | ".join([f"{s:.2f}" for s in metrics["Support"]]) if metrics["Support"] else "-",
                    "Resistance": " | ".join([f"{r:.2f}" for r in metrics["Resistance"]]) if metrics["Resistance"] else "-",
                    "Fib 0.382": fib.get('Fib_0.382'),
                    "Fib 0.5": fib.get('Fib_0.5'),
                    "Fib 0.618": fib.get('Fib_0.618')
                })
            progress = (i + 1) / max(1, len(tickers))
            progress_bar.progress(progress)
            progress_text.text(f"Progress: {int(progress * 100)}% - Memproses {ticker}")

        if results:
            st.session_state['screening_results'] = pd.DataFrame(results)
        else:
            st.session_state['screening_results'] = None
            st.warning("Tidak ada saham yang cocok dengan pola.")

    # tampilkan hasil jika ada
    if st.session_state.get('screening_results') is not None:
        st.subheader("✅ Saham yang Memenuhi Kriteria")
        st.dataframe(st.session_state['screening_results'], use_container_width=True)

        ticker_list = st.session_state['screening_results']['Ticker'].tolist()
        selected_ticker = st.selectbox(
            "Pilih Saham untuk Detail",
            options=ticker_list,
            index=ticker_list.index(st.session_state['selected_ticker']) if st.session_state.get('selected_ticker') in ticker_list else 0,
            key='ticker_selector'
        )
        st.session_state['selected_ticker'] = selected_ticker

        if st.button("📊 Tampilkan Analisis Detail"):
            if st.session_state['selected_ticker']:
                show_stock_details(st.session_state['selected_ticker'], analysis_date)

# allow running module directly for debugging
if __name__ == "__main__":
    app()
