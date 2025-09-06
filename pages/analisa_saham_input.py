# pages/analisa_saham_streamlit.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from scipy.signal import argrelextrema
import plotly.graph_objects as go

# -----------------------------
# Utility / Indicator Functions
# -----------------------------
@st.cache_data
def fetch_data(ticker: str, period: str = "6mo") -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=period, interval="1d")
        if df.empty:
            return pd.DataFrame()
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        st.error(f"Gagal mengambil data: {e}")
        return pd.DataFrame()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    mf = tp * df['Volume']
    positive_flow = []
    negative_flow = []
    # start from index 1 because we compare current tp with previous
    for i in range(1, len(tp)):
        if tp.iat[i] > tp.iat[i-1]:
            positive_flow.append(mf.iat[i])
            negative_flow.append(0)
        elif tp.iat[i] < tp.iat[i-1]:
            positive_flow.append(0)
            negative_flow.append(mf.iat[i])
        else:
            positive_flow.append(0)
            negative_flow.append(0)
    # align index
    pf = pd.Series([0] + positive_flow, index=df.index)
    nf = pd.Series([0] + negative_flow, index=df.index)
    pos_mf = pf.rolling(window=period, min_periods=1).sum()
    neg_mf = nf.rolling(window=period, min_periods=1).sum()
    ratio = pos_mf / neg_mf.replace(0, np.nan)
    ratio = ratio.fillna(1.0)
    mfi = 100 - (100 / (1 + ratio))
    return mfi


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def compute_bollinger(series: pd.Series, window: int = 20, num_std: int = 2):
    ma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = ma + std * num_std
    lower = ma - std * num_std
    bandwidth = (upper - lower) / ma.replace(0, np.nan)
    return ma, upper, lower, bandwidth


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr


def compute_obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df['Close'].diff()).fillna(0)
    obv = (direction * df['Volume']).cumsum()
    return obv


def compute_adl(df: pd.DataFrame) -> pd.Series:
    denom = (df['High'] - df['Low']).replace(0, np.nan)
    mfm = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / denom
    mfv = mfm * df['Volume']
    adl = mfv.cumsum()
    return adl


def volume_price_histogram(df: pd.DataFrame, days: int = 5, bins: int = 12):
    recent = df.tail(days).copy()
    if recent.empty:
        return None, None
    prices = recent['Close']
    vols = recent['Volume']
    bin_edges = np.linspace(prices.min(), prices.max(), bins + 1)
    hist, edges = np.histogram(prices, bins=bin_edges, weights=vols)
    # return the bin range with maximum volume
    max_idx = np.argmax(hist)
    return (edges[max_idx], edges[max_idx + 1]), (hist, edges)


# -----------------------------
# Fusion & Rules
# -----------------------------

def score_indicators(df: pd.DataFrame) -> dict:
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    scores = {}

    # RSI
    rsi = latest['RSI']
    if pd.notna(rsi):
        if rsi < 30:
            scores['RSI'] = (1, 0.9, f"RSI {rsi:.1f} -> Oversold (bullish)")
        elif rsi > 70:
            scores['RSI'] = (-1, 0.9, f"RSI {rsi:.1f} -> Overbought (bearish)")
        else:
            scores['RSI'] = (0, 0.3, f"RSI {rsi:.1f} -> Netral")
    else:
        scores['RSI'] = (0, 0.0, 'RSI N/A')

    # MACD
    macd, signal, hist = latest['MACD'], latest['Signal'], latest['Hist']
    pmacd, psignal = prev['MACD'], prev['Signal']
    if pd.notna(macd) and pd.notna(signal):
        if macd > signal and pmacd <= psignal:
            scores['MACD'] = (1, 1.0, 'MACD cross up')
        elif macd < signal and pmacd >= psignal:
            scores['MACD'] = (-1, 1.0, 'MACD cross down')
        else:
            # momentum of histogram
            if hist > 0:
                scores['MACD'] = (1, 0.4, f'MACD hist {hist:.3f} positive')
            elif hist < 0:
                scores['MACD'] = (-1, 0.4, f'MACD hist {hist:.3f} negative')
            else:
                scores['MACD'] = (0, 0.2, 'MACD neutral')
    else:
        scores['MACD'] = (0, 0.0, 'MACD N/A')

    # Bollinger (position + squeeze)
    up, mid, lo = latest['BB_Up'], latest['BB_MA'], latest['BB_Lo']
    if pd.notna(up) and pd.notna(lo):
        if latest['Close'] > up:
            scores['Bollinger'] = (1, 0.7, 'Close above upper band')
        elif latest['Close'] < lo:
            scores['Bollinger'] = (-1, 0.7, 'Close below lower band')
        else:
            scores['Bollinger'] = (0, 0.3, 'Inside bands')
        # squeeze detection (percentile-based)
        bw = df['BB_BW'].dropna()
        if len(bw) > 30:
            p = bw.iloc[-20:].quantile(0.25)
            if bw.iloc[-1] < p:
                # strong squeeze
                scores['Bollinger_Squeeze'] = (0, 0.6, 'Bollinger squeeze detected')
    else:
        scores['Bollinger'] = (0, 0.0, 'BB N/A')

    # MFI
    mfi = latest['MFI']
    if pd.notna(mfi):
        if mfi > 65:
            scores['MFI'] = (1, 0.6, f'MFI {mfi:.1f} bullish')
        elif mfi < 35:
            scores['MFI'] = (-1, 0.6, f'MFI {mfi:.1f} bearish')
        else:
            scores['MFI'] = (0, 0.2, f'MFI {mfi:.1f} neutral')
    else:
        scores['MFI'] = (0, 0.0, 'MFI N/A')

    # OBV & ADL slope (last 5 days average change)
    obv_slope = df['OBV'].diff().tail(5).mean()
    adl_slope = df['ADL'].diff().tail(5).mean()
    if obv_slope > 0:
        scores['OBV'] = (1, 0.4, f'OBV slope {obv_slope:.0f} up')
    elif obv_slope < 0:
        scores['OBV'] = (-1, 0.4, f'OBV slope {obv_slope:.0f} down')
    else:
        scores['OBV'] = (0, 0.1, 'OBV flat')
    if adl_slope > 0:
        scores['ADL'] = (1, 0.4, f'ADL slope {adl_slope:.0f} up')
    elif adl_slope < 0:
        scores['ADL'] = (-1, 0.4, f'ADL slope {adl_slope:.0f} down')
    else:
        scores['ADL'] = (0, 0.1, 'ADL flat')

    return scores


def composite_from_scores(scores: dict, weights: dict = None):
    # default weights if not provided
    default_weights = {
        'MACD': 0.25,
        'RSI': 0.15,
        'Bollinger': 0.15,
        'MFI': 0.12,
        'OBV': 0.12,
        'ADL': 0.11,
        'Bollinger_Squeeze': 0.1
    }
    if weights is None:
        weights = default_weights

    total_w = 0
    score_sum = 0
    reasons = []
    for k, (s, strength, reason) in scores.items():
        w = weights.get(k, 0.05)
        score_sum += s * strength * w
        total_w += strength * w
        if strength > 0 and abs(s) > 0:
            reasons.append((k, s, strength, reason))
    composite = score_sum / total_w if total_w != 0 else 0
    # normalize to -1..1 (should already be in this range)
    return composite, reasons


# -----------------------------
# Breakout, Entry & Risk
# -----------------------------

def find_support_resistance(df: pd.DataFrame, lookback: int = 60):
    data = df.tail(lookback)
    highs = data['High']
    lows = data['Low']
    # Simple SR: recent rolling max/min + swing points
    recent_res = highs.rolling(window=20, min_periods=1).max().iloc[-2]
    recent_sup = lows.rolling(window=20, min_periods=1).min().iloc[-2]
    return recent_sup, recent_res


def breakout_rule(df: pd.DataFrame, multiplier: float = 1.5, buffer_pct: float = 0.005):
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    support, resistance = find_support_resistance(df, lookback=60)
    vol_avg20 = df['Volume'].rolling(20).mean().iloc[-2]
    atr = df['ATR'].iloc[-1]
    buffer = buffer_pct * latest['Close']

    breakout = False
    breakout_type = None
    if latest['Close'] > resistance + buffer and latest['Volume'] > multiplier * vol_avg20 and latest['Hist'] > 0:
        breakout = True
        breakout_type = 'up'
    elif latest['Close'] < support - buffer and latest['Volume'] > multiplier * vol_avg20 and latest['Hist'] < 0:
        breakout = True
        breakout_type = 'down'

    # entry suggestion
    entry = None
    stop = None
    tp1 = None
    tp2 = None
    if breakout:
        entry = latest['Close']
        if breakout_type == 'up':
            stop = entry - 1.5 * atr
            tp1 = entry + 2 * (entry - stop)
            tp2 = entry + 3 * (entry - stop)
        else:
            stop = entry + 1.5 * atr
            tp1 = entry - 2 * (stop - entry)
            tp2 = entry - 3 * (stop - entry)

    # retest suggestions (if no breakout)
    retest = None
    if not breakout:
        # define plausible retest zone near resistance/support
        if latest['Close'] > (resistance * 0.98):
            retest = {'type': 'retest_up', 'zone': (resistance * 0.995, resistance * 1.02)}
        elif latest['Close'] < (support * 1.02):
            retest = {'type': 'retest_down', 'zone': (support * 0.98, support * 1.005)}

    return {
        'breakout': breakout,
        'type': breakout_type,
        'entry': entry,
        'stop': stop,
        'tp1': tp1,
        'tp2': tp2,
        'resistance': resistance,
        'support': support,
        'retest': retest
    }


# -----------------------------
# Streamlit App
# -----------------------------

def app():
    st.set_page_config(page_title="Analisa Saham SHZ - Streamlit", layout="wide")
    st.title("📈 Analisa Teknikal Saham (Streamlit)")

    with st.sidebar:
        st.header("Input")
        ticker = st.text_input("Kode Saham (contoh: BBCA.JK)", value="BBCA.JK").strip().upper()
        period = st.selectbox("Periode data", options=["3mo", "6mo", "1y", "2y"], index=1)
        risk_input = st.number_input("Modal / Dana (Rp)", value=10000000, step=1000000, min_value=100000)
        risk_pct = st.slider("Risk per trade (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
        run = st.button("🔍 Mulai Analisis")

    if not run:
        st.info("Masukkan ticker dan klik 'Mulai Analisis' untuk melihat hasil.")
        st.stop()

    # Fetch data
    df = fetch_data(ticker, period)
    if df.empty:
        st.error("Data tidak tersedia. Pastikan ticker benar dan ada koneksi internet.")
        st.stop()

    # Compute indicators
    df['RSI'] = compute_rsi(df['Close'])
    df['MFI'] = compute_mfi(df)
    df['MACD'], df['Signal'], df['Hist'] = compute_macd(df['Close'])
    df['BB_MA'], df['BB_Up'], df['BB_Lo'], df['BB_BW'] = compute_bollinger(df['Close'])
    df['ATR'] = compute_atr(df)
    df['OBV'] = compute_obv(df)
    df['ADL'] = compute_adl(df)

    # Score and composite
    scores = score_indicators(df)
    composite, reasons = composite_from_scores(scores)

    # Breakout & trade plan
    br = breakout_rule(df)

    # Volume profile last 5 days
    vol_area, vph = volume_price_histogram(df, days=5, bins=12)

    # Layout: left chart, right summary
    col1, col2 = st.columns((2, 1))

    with col1:
        st.subheader(f"Chart & Indikator: {ticker}")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Candle'))
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_MA'], mode='lines', name='MA20', line=dict(width=1)))
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Up'], mode='lines', name='BB Upper', line=dict(width=1, dash='dot')))
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lo'], mode='lines', name='BB Lower', line=dict(width=1, dash='dot')))
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'].rolling(50).mean(), mode='lines', name='MA50', line=dict(width=1, dash='dash')))

        # SR lines
        fig.add_hline(y=br['resistance'], line_dash='dash', line_color='red', annotation_text=f"Resistance: Rp {br['resistance']:.2f}", annotation_position='top right')
        fig.add_hline(y=br['support'], line_dash='dash', line_color='green', annotation_text=f"Support: Rp {br['support']:.2f}", annotation_position='bottom right')

        # Highlight breakout annotation
        if br['breakout']:
            fig.add_annotation(x=df.index[-1], y=df['Close'].iloc[-1], text="⚠️ BREAKOUT", showarrow=True, arrowcolor='purple')

        fig.update_layout(xaxis_rangeslider_visible=False, height=600, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

        # Subcharts: MACD, RSI, Volume
        macd_fig = go.Figure()
        macd_fig.add_trace(go.Bar(x=df.index, y=df['Hist'], name='MACD Hist'))
        macd_fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD'))
        macd_fig.add_trace(go.Scatter(x=df.index, y=df['Signal'], mode='lines', name='Signal'))
        macd_fig.update_layout(height=250, template='plotly_white', title='MACD')
        st.plotly_chart(macd_fig, use_container_width=True)

        rsi_fig = go.Figure()
        rsi_fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI'))
        rsi_fig.add_hline(y=70, line_dash='dot', line_color='red')
        rsi_fig.add_hline(y=30, line_dash='dot', line_color='green')
        rsi_fig.update_layout(height=200, template='plotly_white', title='RSI')
        st.plotly_chart(rsi_fig, use_container_width=True)

        vol_fig = go.Figure()
        vol_fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume'))
        vol_fig.update_layout(height=200, template='plotly_white', title='Volume')
        st.plotly_chart(vol_fig, use_container_width=True)

    with col2:
        st.subheader("Ringkasan & Rekomendasi")
        st.metric("Composite Score", f"{composite:.2f}", delta=None)
        # show top reasons
        st.markdown("**Alasan utama (indikator yang mendukung):**")
        if reasons:
            for k, s, strength, text in reasons:
                tag = "Bullish" if s > 0 else "Bearish"
                st.write(f"- **{k}** ({tag}, strength {strength}) — {text}")
        else:
            st.write("- Tidak ada indikasi kuat saat ini.")

        st.markdown("---")

        st.markdown("**Status Breakout / Rencana Trading**")
        if br['breakout']:
            st.success(f"BREAKOUT terdeteksi ({br['type']}). Entry: Rp {br['entry']:.2f}")
            st.write(f"Stop: Rp {br['stop']:.2f} — TP1: Rp {br['tp1']:.2f} — TP2: Rp {br['tp2']:.2f}")
            # position sizing example
            risk_amount = (risk_input * (risk_pct/100))
            position = int(risk_amount / abs(br['entry'] - br['stop'])) if br['stop'] and br['entry'] else 0
            st.write(f"Contoh ukuran posisi (berdasarkan risk {risk_pct}%): {position} lembar (risk Rp {risk_amount:,.0f})")
        else:
            st.info("Belum ada breakout yang memenuhi syarat. Berikut kondisi retest / zona yang bisa diawasi:")
            if br['retest']:
                z = br['retest']['zone']
                st.write(f"- {br['retest']['type']} di zona Rp {z[0]:.2f} — Rp {z[1]:.2f}")
            else:
                st.write("- Tidak ada retest terdekat. Pantau resistance/support dan volume untuk konfirmasi breakout.")

        st.markdown('---')
        st.markdown("**Analisis Akumulasi / Distribusi (5 hari)**")
        adl_chg = df['ADL'].iloc[-1] - df['ADL'].iloc[-5] if len(df) >= 5 else np.nan
        obv_chg = df['OBV'].iloc[-1] - df['OBV'].iloc[-5] if len(df) >= 5 else np.nan
        avg_vol5 = df['Volume'].tail(5).mean()
        avg_vol20 = df['Volume'].tail(20).mean()
        mfi5 = df['MFI'].tail(5).mean()
        st.write(f"ADL change (5d): {adl_chg:.0f}")
        st.write(f"OBV change (5d): {obv_chg:.0f}")
        st.write(f"Avg Vol 5 / Avg Vol 20: {avg_vol5/avg_vol20:.2f} (ratio)")
        st.write(f"MFI 5-day average: {mfi5:.1f}")

        acc_score = 0
        if adl_chg > 0: acc_score += 1
        if obv_chg > 0: acc_score += 1
        if avg_vol5 > 1.2 * avg_vol20: acc_score += 1
        if mfi5 > 60: acc_score += 1
        label = 'Netral'
        if acc_score >= 3:
            label = 'Akumulasi'
            st.success('Indikasi AKUMULASI dalam 5 hari terakhir')
        elif acc_score <= 1:
            label = 'Distribusi'
            st.error('Indikasi DISTRIBUSI dalam 5 hari terakhir')
        else:
            st.info('Indikasi NETRAL')
        st.write(f"Kesimpulan: **{label}** (score {acc_score}/4)")

        if vol_area:
            st.markdown(f"Harga dengan volume terbanyak (last 5d): Rp {vol_area[0]:.2f} - Rp {vol_area[1]:.2f}")

        st.markdown('---')
        st.markdown('**Tabel indikator terakhir**')
        last = df[['Close','RSI','MFI','MACD','Signal','Hist','BB_MA','BB_Up','BB_Lo','ATR','OBV','ADL']].iloc[-1:]
        st.dataframe(last.T.style.format({c: ':.2f' for c in last.columns}))

    # Footer / Disclaimer
    st.info("⚠️ Disclaimer: Ini adalah alat bantu edukasi. Bukan rekomendasi investasi. Periksa data dan gunakan manajemen risiko.")


if __name__ == '__main__':
    app()
