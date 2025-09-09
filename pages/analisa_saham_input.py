# analisa_saham.py (versi refactor)
import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import plotly.graph_objects as go
from scipy.signal import argrelextrema
from scipy import stats
import time

# --- KONSTANTA DAN KONFIGURASI ---
INDICATOR_WEIGHTS = {
    'rsi': 0.15,
    'macd_cross': 0.25,
    'macd_hist': 0.10,
    'bollinger': 0.15,
    'volume': 0.20,
    'obv': 0.10,
    'adx': 0.05
}

# --- FUNGSI ANALISIS TEKNIKAL ---
def compute_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(0)

def compute_mfi(df, period=14):
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = tp * df['Volume']
    pos_flow = money_flow.where(tp > tp.shift(1), 0)
    neg_flow = money_flow.where(tp < tp.shift(1), 0)
    pos_mf = pos_flow.rolling(window=period, min_periods=period).sum()
    neg_mf = neg_flow.rolling(window=period, min_periods=period).sum()

    # handle division by zero
    mfi = 50 * np.ones(len(df))
    mask = neg_mf > 0
    mfi[mask] = 100 - (100 / (1 + (pos_mf[mask] / neg_mf[mask])))
    mask = (pos_mf > 0) & (neg_mf == 0)
    mfi[mask] = 100
    return pd.Series(mfi, index=df.index).fillna(50)

def compute_macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line.fillna(0), signal_line.fillna(0), histogram.fillna(0)

def compute_bollinger_bands(close, window=20, num_std=2):
    sma = close.rolling(window=window).mean()
    std = close.rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    bandwidth = (upper_band - lower_band) / sma
    percent_b = (close - lower_band) / (upper_band - lower_band)
    return upper_band.fillna(0), sma.fillna(0), lower_band.fillna(0), bandwidth.fillna(0), percent_b.fillna(0)

def compute_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr.fillna(0)

def compute_obv(close, volume):
    price_change = close.diff().fillna(0)
    obv = (volume * np.sign(price_change)).cumsum()
    return obv.fillna(0)

def compute_adx(high, low, close, period=14):
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period).mean() / tr.ewm(alpha=1/period).mean())
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period).mean() / tr.ewm(alpha=1/period).mean())
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1/period).mean()
    return adx.fillna(0), plus_di.fillna(0), minus_di.fillna(0)

def compute_vwap(high, low, close, volume):
    tp = (high + low + close) / 3
    vwap = (tp * volume).cumsum() / volume.cumsum()
    return vwap.fillna(0)

def compute_adl(high, low, close, volume):
    mfm = ((close - low) - (high - close)) / (high - low)
    mfm = mfm.replace([np.inf, -np.inf], 0).fillna(0)
    mfv = mfm * volume
    adl = mfv.cumsum()
    return adl.fillna(0)

# --- SISTEM SCORING ---
class IndicatorScoringSystem:
    def __init__(self, weights=None):
        self.weights = weights or INDICATOR_WEIGHTS

    def score_rsi(self, rsi_values, period=3):
        current = rsi_values.iloc[-1]
        trend = self.calculate_trend(rsi_values, period)
        # linear scoring
        score = (50 - current) / 50  # normalize to -1..1
        strength = min(1.0, abs(score))
        # adjust with trend
        if trend > 0.1 and score >= 0:
            strength += 0.2
        elif trend < -0.1 and score <= 0:
            strength += 0.2
        return np.sign(score), min(1.0, strength)

    def score_macd(self, macd_line, signal_line, histogram):
        cross_score = 1.0 if macd_line.iloc[-1] > signal_line.iloc[-1] else -1.0
        hist_trend = self.calculate_trend(histogram[-5:], 3)
        hist_score = 0.5 if hist_trend > 0 else -0.5 if hist_trend < 0 else 0
        hist_strength = min(1.0, abs(hist_trend) * 2)
        # tambahan zero line filter
        if macd_line.iloc[-1] > 0:
            cross_score *= 1
        else:
            cross_score *= 0.8  # lemah kalau di bawah nol
        return cross_score, 1.0, hist_score, hist_strength

    def calculate_trend(self, values, period):
        if len(values) < period:
            return 0.0
        x = np.arange(len(values[-period:]))
        y = values[-period:].values
        slope, _, _, _, _ = stats.linregress(x, y)
        return slope / np.mean(y) if np.mean(y) != 0 else 0.0

    def calculate_composite_score(self, scores):
        weighted_scores = []
        total_weight = 0
        for ind, (score, strength) in scores.items():
            if ind in self.weights:
                weighted_scores.append(score * strength * self.weights[ind])
                total_weight += self.weights[ind]
        return sum(weighted_scores) / total_weight if total_weight > 0 else 0

# --- BREAKOUT DETECTOR ---
class BreakoutDetector:
    def __init__(self, atr_period=14, buffer_percent=0.005, max_position_percent=0.1):
        self.atr_period = atr_period
        self.buffer_percent = buffer_percent
        self.max_position_percent = max_position_percent

    def calculate_position_size(self, entry, stop, account_size, risk_percent=0.02):
        risk_amount = account_size * risk_percent
        risk_per_share = abs(entry - stop)
        if risk_per_share == 0:
            return 0
        size = risk_amount / risk_per_share
        max_size = (account_size * self.max_position_percent) / entry
        return min(size, max_size)

# --- DATA FETCH ---
def get_stock_data(ticker, end_date):
    try:
        stock = yf.Ticker(f"{ticker}.JK")
        start = end_date - timedelta(days=180)
        data = stock.history(start=start.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
        if data.empty:
            start = end_date - timedelta(days=360)
            data = stock.history(start=start.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
        return data if not data.empty else None
    except Exception as e:
        st.error(f"Gagal mengambil data {ticker}: {e}")
        return None

# --- STREAMLIT APP (ringkas) ---
def app():
    st.title("📊 Analisa Teknikal Saham - Versi Refactor")
    ticker = st.text_input("Kode Saham", "BBCA").upper()
    analysis_date = st.date_input("Tanggal", value=datetime.today())
    account_size = st.number_input("Modal", value=100_000_000)

    if st.button("Mulai Analisis"):
        df = get_stock_data(ticker, analysis_date)
        if df is None: return

        # hitung indikator
        df['RSI'] = compute_rsi(df['Close'])
        df['MACD'], df['Signal'], df['Hist'] = compute_macd(df['Close'])
        df['OBV'] = compute_obv(df['Close'], df['Volume'])
        df['MFI'] = compute_mfi(df)
        df['ADX'], df['+DI'], df['-DI'] = compute_adx(df['High'], df['Low'], df['Close'])

        # scoring
        scoring = IndicatorScoringSystem()
        scores = {
            'rsi': scoring.score_rsi(df['RSI']),
            'macd_cross': (1.0 if df['MACD'].iloc[-1] > df['Signal'].iloc[-1] else -1.0, 1.0),
            'macd_hist': (0.5 if df['Hist'].iloc[-1] > 0 else -0.5, 1.0),
            'obv': (1.0 if scoring.calculate_trend(df['OBV'], 5) > 0 else -1.0, 1.0),
            'adx': (1.0 if df['ADX'].iloc[-1] > 25 else 0.0, 1.0)
        }
        composite = scoring.calculate_composite_score(scores)

        st.metric("Composite Score", f"{composite:.2f}")

if __name__ == "__main__":
    app()
