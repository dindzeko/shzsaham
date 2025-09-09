# analisa_saham.py
"""
Analisa Teknikal Saham - Versi Refactor (Input di Atas, Hasil di Bawah)
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from scipy import stats

# ===========================
# KONSTANTA
# ===========================
INDICATOR_WEIGHTS = {
    'rsi': 0.15,
    'macd_cross': 0.25,
    'macd_hist': 0.10,
    'bollinger': 0.15,
    'volume': 0.20,
    'obv': 0.10,
    'adx': 0.05
}

# ===========================
# INDIKATOR
# ===========================
def compute_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def compute_mfi(df, period=14):
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = tp * df['Volume']
    pos_flow = money_flow.where(tp > tp.shift(1), 0)
    neg_flow = money_flow.where(tp < tp.shift(1), 0)
    pos_mf = pos_flow.rolling(window=period, min_periods=period).sum()
    neg_mf = neg_flow.rolling(window=period, min_periods=period).sum()
    mfi = pd.Series(50, index=df.index, dtype=float)
    mask = neg_mf > 0
    mfi[mask] = 100 - (100 / (1 + pos_mf[mask] / neg_mf[mask]))
    mask = (pos_mf > 0) & (neg_mf == 0)
    mfi[mask] = 100
    return mfi.fillna(50)

def compute_macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line.fillna(0), signal_line.fillna(0), hist.fillna(0)

def compute_bollinger(close, window=20, num_std=2):
    sma = close.rolling(window=window).mean()
    std = close.rolling(window=window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    bandwidth = (upper - lower) / sma
    percent_b = (close - lower) / (upper - lower)
    return upper.fillna(0), sma.fillna(0), lower.fillna(0), bandwidth.fillna(0), percent_b.fillna(0)

def compute_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr.fillna(0)

def compute_obv(close, volume):
    change = close.diff().fillna(0)
    obv = (volume * np.sign(change)).cumsum()
    return obv.fillna(0)

def compute_adx(high, low, close, period=14):
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period).mean() / atr)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.ewm(alpha=1/period).mean()
    return adx.fillna(0), plus_di.fillna(0), minus_di.fillna(0)

def compute_vwap(high, low, close, volume):
    tp = (high + low + close) / 3
    return (tp * volume).cumsum() / volume.cumsum()

# ===========================
# SCORING
# ===========================
class IndicatorScoring:
    def __init__(self, weights=None):
        self.weights = weights or INDICATOR_WEIGHTS

    def calc_trend(self, values, period=3):
        if len(values) < period:
            return 0.0
        x = np.arange(period)
        y = values[-period:].values
        slope, _, _, _, _ = stats.linregress(x, y)
        return slope / np.mean(y) if np.mean(y) else 0

    def score_rsi(self, rsi):
        val = rsi.iloc[-1]
        score = (50 - val) / 50
        return np.sign(score), min(1.0, abs(score))

    def score_macd(self, macd, signal, hist):
        cross = 1 if macd.iloc[-1] > signal.iloc[-1] else -1
        trend = self.calc_trend(hist, 3)
        hist_score = 0.5 if trend > 0 else -0.5
        if macd.iloc[-1] < 0:
            cross *= 0.8
        return cross, 1.0, hist_score, min(1.0, abs(trend) * 2)

    def composite(self, scores):
        total = 0; wsum = 0
        for ind, (sc, strength) in scores.items():
            if ind in self.weights:
                total += sc * strength * self.weights[ind]
                wsum += self.weights[ind]
        return total / wsum if wsum else 0

# ===========================
# BREAKOUT & POSITION SIZE
# ===========================
class BreakoutDetector:
    def __init__(self, atr_period=14, buffer=0.005, max_pos=0.1):
        self.atr_period = atr_period
        self.buffer = buffer
        self.max_pos = max_pos

    def pos_size(self, entry, stop, acct, risk=0.02):
        risk_amt = acct * risk
        risk_share = abs(entry - stop)
        if risk_share == 0: return 0
        size = risk_amt / risk_share
        return min(size, acct * self.max_pos / entry)

# ===========================
# HELPER DATA
# ===========================
def get_stock_data(ticker, end, days_back=360):
    try:
        t = yf.Ticker(f"{ticker}.JK")
        start = end - timedelta(days=days_back)
        df = t.history(start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'))
        return df if not df.empty else None
    except:
        return None

# ===========================
# STREAMLIT APP
# ===========================
def app():
    st.title("📊 Analisa Teknikal Saham - Versi Refactor (Input di Atas, Hasil di Bawah)")

    # Input Form di Atas
    ticker = st.text_input("Kode Saham", "BBCA").upper()
    capital = st.number_input("Modal (Rp)", value=100_000_000, step=1_000_000)
    risk_percent = st.slider("Risiko per Trade (%)", 0.5, 5.0, 2.0) / 100
    use_multi_timeframe = st.checkbox("Gunakan Multi-Timeframe (Weekly/Monthly)", value=True)
    lookback_days = st.number_input("Ambil data (hari)", min_value=90, max_value=2000, value=360)
    date = st.date_input("Tanggal Analisis", datetime.today())
    run_button = st.button("🚀 Mulai Analisis")

    # Hasil Analisis di Bawah
    if run_button:
        df = get_stock_data(ticker, date, lookback_days)
        if df is None or len(df) < 50:
            st.warning("Data historis tidak cukup")
            return

        # Hitung indikator
        df['RSI'] = compute_rsi(df['Close'])
        df['MACD'], df['Signal'], df['Hist'] = compute_macd(df['Close'])
        df['OBV'] = compute_obv(df['Close'], df['Volume'])
        df['MFI'] = compute_mfi(df)
        df['ADX'], df['+DI'], df['-DI'] = compute_adx(df['High'], df['Low'], df['Close'])
        df['ATR'] = compute_atr(df['High'], df['Low'], df['Close'])
        df['VWAP'] = compute_vwap(df['High'], df['Low'], df['Close'], df['Volume'])

        # Scoring
        sc = IndicatorScoring()
        scores = {
            'rsi': sc.score_rsi(df['RSI']),
            'macd_cross': (1 if df['MACD'].iloc[-1] > df['Signal'].iloc[-1] else -1, 1),
            'macd_hist': (0.5 if df['Hist'].iloc[-1] > 0 else -0.5, 1),
            'obv': (1 if sc.calc_trend(df['OBV'], 5) > 0 else -1, 1),
            'adx': (1 if df['ADX'].iloc[-1] > 25 else 0, 1)
        }
        composite = sc.composite(scores)

        st.subheader("🎯 Hasil Analisis Cross-Confirmation")
        st.metric("Composite Score", f"{composite:.2f}")

        # Chart Candlestick
        st.subheader("📈 Grafik Harga dengan Bollinger Bands")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'],
            name="OHLC"
        ))
        upper, sma, lower, _, _ = compute_bollinger(df['Close'])
        fig.add_trace(go.Scatter(x=df.index, y=upper, line=dict(color='blue', width=1), name='Upper BB'))
        fig.add_trace(go.Scatter(x=df.index, y=sma, line=dict(color='orange', width=1), name='SMA20'))
        fig.add_trace(go.Scatter(x=df.index, y=lower, line=dict(color='blue', width=1), name='Lower BB'))
        fig.update_layout(xaxis_rangeslider_visible=False, template="plotly_white", height=600)
        st.plotly_chart(fig, use_container_width=True)

        # Ringkasan indikator terakhir
        st.subheader("📋 Ringkasan Indikator Terakhir")
        last = df.iloc[-1]
        table = pd.DataFrame({
            'RSI (real)': [last['RSI']],
            'RSI Strength': [scores['rsi'][1]],
            'MACD': [last['MACD']],
            'Signal': [last['Signal']],
            'Hist': [last['Hist']],
            'MFI': [last['MFI']],
            'ADX': [last['ADX']],
            'ATR': [last['ATR']],
            'OBV': [last['OBV']]
        })
        st.table(table)

if __name__ == "__main__":
    app()
