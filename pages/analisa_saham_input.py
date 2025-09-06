# analisa_saham.py
import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import plotly.graph_objects as go
from scipy.signal import argrelextrema
import time
from scipy import stats

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
    """Menghitung Relative Strength Index (RSI) dengan Wilder smoothing"""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    # Wilder smoothing dengan EMA
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_mfi_corrected(df, period=14):
    """Menghitung Money Flow Index (MFI) dengan benar"""
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = tp * df['Volume']
    
    # Gunakan perbandingan typical price hari ini vs kemarin
    positive_flow = (money_flow.where(tp > tp.shift(1), 0))
    negative_flow = (money_flow.where(tp < tp.shift(1), 0))
    
    # Rolling sum dengan periode
    pos_mf = positive_flow.rolling(window=period, min_periods=1).sum()
    neg_mf = negative_flow.rolling(window=period, min_periods=1).sum()
    
    mfi_ratio = pos_mf / neg_mf
    mfi = 100 - (100 / (1 + mfi_ratio))
    return mfi

def compute_macd(close, fast=12, slow=26, signal=9):
    """Menghitung Moving Average Convergence Divergence (MACD)"""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def compute_bollinger_bands(close, window=20, num_std=2):
    """Menghitung Bollinger Bands dengan standar deviasi"""
    sma = close.rolling(window=window).mean()
    std = close.rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    
    # Hitung bandwidth dan %b
    bandwidth = (upper_band - lower_band) / sma
    percent_b = (close - lower_band) / (upper_band - lower_band)
    
    return upper_band, sma, lower_band, bandwidth, percent_b

def compute_atr(high, low, close, period=14):
    """Menghitung Average True Range (ATR)"""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr

def compute_obv(close, volume):
    """Menghitung On-Balance Volume (OBV)"""
    obv = np.zeros(len(close))
    obv[0] = volume.iloc[0]
    
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]:
            obv[i] = obv[i-1] + volume.iloc[i]
        elif close.iloc[i] < close.iloc[i-1]:
            obv[i] = obv[i-1] - volume.iloc[i]
        else:
            obv[i] = obv[i-1]
    
    return pd.Series(obv, index=close.index)

def compute_adx(high, low, close, period=14):
    """Menghitung Average Directional Index (ADX)"""
    # Menghitung +DM dan -DM
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Menghitung True Range (TR)
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Smoothing +DM, -DM, dan TR
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period).mean() / 
                    tr.ewm(alpha=1/period).mean())
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period).mean() / 
                     tr.ewm(alpha=1/period).mean())
    
    # Menghitung DX dan ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/period).mean()
    
    return adx, plus_di, minus_di

def compute_vwap(high, low, close, volume):
    """Menghitung Volume Weighted Average Price (VWAP)"""
    tp = (high + low + close) / 3
    vwap = (tp * volume).cumsum() / volume.cumsum()
    return vwap

def compute_adl(high, low, close, volume):
    """Menghitung Accumulation/Distribution Line (ADL)"""
    mfm = ((close - low) - (high - close)) / (high - low)
    mfm = mfm.replace([np.inf, -np.inf], 0).fillna(0)
    mfv = mfm * volume
    adl = mfv.cumsum()
    return adl

# --- SISTEM SCORING DAN CROSS-CONFIRMATION ---
class IndicatorScoringSystem:
    def __init__(self, weights=None):
        # Bobot untuk setiap indikator
        self.weights = weights or INDICATOR_WEIGHTS
        
    def score_rsi(self, rsi_values, period=3):
        """Memberikan skor untuk RSI berdasarkan nilai terakhir dan trend"""
        current_rsi = rsi_values.iloc[-1]
        rsi_trend = self.calculate_trend(rsi_values, period)
        
        # Skor berdasarkan level RSI
        if current_rsi < 30:
            score = 1.0  # Bullish kuat (oversold)
            strength = min(1.0, (30 - current_rsi) / 30)
        elif current_rsi > 70:
            score = -1.0  # Bearish kuat (overbought)
            strength = min(1.0, (current_rsi - 70) / 30)
        elif current_rsi > 45:
            score = 0.5  # Bullish lemah
            strength = (current_rsi - 45) / 25
        elif current_rsi < 55:
            score = -0.5  # Bearish lemah
            strength = (55 - current_rsi) / 25
        else:
            score = 0.0  # Netral
            strength = 0.0
            
        # Adjust score berdasarkan trend
        if rsi_trend > 0.1 and score >= 0:
            strength = min(1.0, strength + 0.2)
        elif rsi_trend < -0.1 and score <= 0:
            strength = min(1.0, strength + 0.2)
            
        return score, strength
    
    def score_macd(self, macd_line, signal_line, histogram):
        """Memberikan skor untuk MACD berdasarkan crossover dan momentum histogram"""
        # Skor untuk crossover
        macd_cross_score = 1.0 if macd_line.iloc[-1] > signal_line.iloc[-1] else -1.0
        
        # Skor untuk momentum histogram
        hist_values = histogram.iloc[-5:]  # 5 periode terakhir
        hist_trend = self.calculate_trend(hist_values, 3)
        
        if hist_trend > 0:
            hist_score = 0.5
            hist_strength = min(1.0, abs(hist_trend) * 2)
        elif hist_trend < 0:
            hist_score = -0.5
            hist_strength = min(1.0, abs(hist_trend) * 2)
        else:
            hist_score = 0.0
            hist_strength = 0.0
            
        return macd_cross_score, 1.0, hist_score, hist_strength
    
    def score_bollinger(self, price, upper_band, lower_band, percent_b, bandwidth, bandwidth_history):
        """Memberikan skor untuk Bollinger Bands"""
        current_price = price.iloc[-1]
        current_pct_b = percent_b.iloc[-1]
        current_bandwidth = bandwidth.iloc[-1]
        
        # Hitung bandwidth percentile
        bandwidth_percentile = stats.percentileofscore(bandwidth_history, current_bandwidth) / 100
        
        # Deteksi squeeze (bandwidth di bawah percentile 20)
        is_squeeze = bandwidth_percentile < 0.2
        
        # Skor berdasarkan posisi harga
        if current_pct_b > 0.8:
            score = -1.0  # Mendekati upper band (overbought)
            strength = min(1.0, (current_pct_b - 0.8) / 0.2)
        elif current_pct_b < 0.2:
            score = 1.0  # Mendekati lower band (oversold)
            strength = min(1.0, (0.2 - current_pct_b) / 0.2)
        elif current_pct_b > 0.5:
            score = -0.5  # Di atas middle band
            strength = (current_pct_b - 0.5) / 0.3
        else:
            score = 0.5  # Di bawah middle band
            strength = (0.5 - current_pct_b) / 0.3
            
        # Adjust untuk squeeze
        if is_squeeze:
            strength = min(1.0, strength + 0.3)
            
        return score, strength, is_squeeze
    
    def score_volume(self, volume, volume_ma, price_change):
        """Memberikan skor untuk volume berdasarkan anomaly dan konfirmasi harga"""
        volume_ratio = volume.iloc[-1] / volume_ma.iloc[-1]
        
        # Hitung z-score volume
        volume_zscore = (volume.iloc[-1] - volume_ma.iloc[-1]) / volume_ma.iloc[-1]
        
        # Skor berdasarkan volume anomaly dan konfirmasi harga
        if volume_ratio > 1.7 and price_change > 0:
            score = 1.0  # Volume tinggi dengan harga naik (bullish)
            strength = min(1.0, (volume_ratio - 1.7) / 0.8)
        elif volume_ratio > 1.7 and price_change < 0:
            score = -1.0  # Volume tinggi dengan harga turun (bearish)
            strength = min(1.0, (volume_ratio - 1.7) / 0.8)
        elif volume_ratio > 1.3 and price_change > 0:
            score = 0.5  # Volume sedang dengan harga naik
            strength = (volume_ratio - 1.3) / 0.4
        elif volume_ratio > 1.3 and price_change < 0:
            score = -0.5  # Volume sedang dengan harga turun
            strength = (volume_ratio - 1.3) / 0.4
        else:
            score = 0.0  # Volume normal
            strength = 0.0
            
        return score, strength
    
    def score_obv(self, obv_values, period=5):
        """Memberikan skor untuk OBV berdasarkan trend"""
        obv_trend = self.calculate_trend(obv_values, period)
        
        if obv_trend > 0.05:
            score = 1.0  # OBV trending up (bullish)
            strength = min(1.0, obv_trend * 5)
        elif obv_trend < -0.05:
            score = -1.0  # OBV trending down (bearish)
            strength = min(1.0, abs(obv_trend) * 5)
        else:
            score = 0.0  # OBV flat
            strength = 0.0
            
        return score, strength
    
    def score_adx(self, adx_values, plus_di, minus_di, threshold=25):
        """Memberikan skor untuk ADX berdasarkan strength trend dan direction"""
        current_adx = adx_values.iloc[-1]
        current_plus_di = plus_di.iloc[-1]
        current_minus_di = minus_di.iloc[-1]
        
        # Tentukan arah trend berdasarkan DI+ dan DI-
        direction = 1.0 if current_plus_di > current_minus_di else -1.0
        
        # Skor berdasarkan strength trend
        if current_adx > threshold:
            score = direction  # Trend kuat
            strength = min(1.0, (current_adx - threshold) / (100 - threshold))
        else:
            score = 0.0  # Trend lemah atau sideways
            strength = 0.0
            
        return score, strength
    
    def calculate_trend(self, values, period):
        """Menghitung trend dari seri nilai (slope linear regression)"""
        if len(values) < period:
            return 0.0
            
        x = np.arange(len(values[-period:]))
        y = values[-period:].values
        slope, _, _, _, _ = stats.linregress(x, y)
        
        # Normalisasi slope terhadap nilai rata-rata
        if np.mean(y) != 0:
            normalized_slope = slope / np.mean(y)
        else:
            normalized_slope = 0.0
            
        return normalized_slope
    
    def calculate_composite_score(self, scores):
        """Menghitung composite score berbobot"""
        weighted_scores = []
        total_weight = 0
        
        for indicator, score_strength in scores.items():
            if indicator in self.weights:
                score, strength = score_strength
                weighted_score = score * strength * self.weights[indicator]
                weighted_scores.append(weighted_score)
                total_weight += self.weights[indicator]
        
        if total_weight > 0:
            composite_score = sum(weighted_scores) / total_weight
        else:
            composite_score = 0
            
        return composite_score
    
    def get_confidence_level(self, composite_score, scores):
        """Menentukan level confidence berdasarkan konsistensi sinyal"""
        # Hitung persentase indikator yang setuju dengan arah composite score
        if composite_score > 0:
            agreeing = sum(1 for score, strength in scores.values() 
                          if score > 0 and strength > 0.3)
        elif composite_score < 0:
            agreeing = sum(1 for score, strength in scores.values() 
                          if score < 0 and strength > 0.3)
        else:
            agreeing = 0
            
        total_indicators = len(scores)
        agreement_ratio = agreeing / total_indicators if total_indicators > 0 else 0
        
        # Tentukan confidence level berdasarkan agreement ratio
        if agreement_ratio >= 0.7:
            return "Tinggi"
        elif agreement_ratio >= 0.5:
            return "Sedang"
        else:
            return "Rendah"

# --- FUNGSI BREAKOUT DETECTION ---
class BreakoutDetector:
    def __init__(self, atr_period=14, buffer_percent=0.005):
        self.atr_period = atr_period
        self.buffer_percent = buffer_percent
    
    def detect_breakout(self, df, resistance_level, support_level):
        """Mendeteksi breakout dari level support/resistance"""
        current_close = df['Close'].iloc[-1]
        current_volume = df['Volume'].iloc[-1]
        avg_volume = df['Volume'].rolling(20).mean().iloc[-1]
        
        # Hitung ATR untuk menentukan buffer dinamis
        atr = compute_atr(df['High'], df['Low'], df['Close'], self.atr_period).iloc[-1]
        atr_buffer = atr / df['Close'].iloc[-1]  # Buffer sebagai % dari harga
        
        # Gunakan buffer yang lebih besar antara fixed % dan ATR-based
        buffer = max(self.buffer_percent, atr_buffer * 0.5)
        
        # Kondisi breakout resistance
        resistance_breakout = current_close > resistance_level * (1 + buffer)
        volume_confirm = current_volume > 1.5 * avg_volume
        
        # Kondisi breakdown support
        support_breakdown = current_close < support_level * (1 - buffer)
        
        return resistance_breakout, support_breakdown, volume_confirm, buffer
    
    def calculate_position_size(self, entry_price, stop_price, account_size, risk_percent=0.02):
        """Menghitung ukuran posisi berdasarkan risiko"""
        risk_amount = account_size * risk_percent
        risk_per_share = abs(entry_price - stop_price)
        
        if risk_per_share > 0:
            position_size = risk_amount / risk_per_share
        else:
            position_size = 0
            
        return position_size
    
    def generate_trading_plan(self, df, breakout_type, level, buffer, account_size=100000000):
        """Membuat rencana trading untuk breakout"""
        current_close = df['Close'].iloc[-1]
        atr = compute_atr(df['High'], df['Low'], df['Close'], self.atr_period).iloc[-1]
        
        if breakout_type == "resistance":
            entry_price = level * (1 + buffer)  # Entry di atas resistance dengan buffer
            stop_loss = entry_price - 2 * atr  # Stop loss 2 ATR di bawah entry
            target_1 = entry_price + 2 * atr  # Target 1: 2 ATR
            target_2 = entry_price + 4 * atr  # Target 2: 4 ATR
            
            position_size = self.calculate_position_size(entry_price, stop_loss, account_size)
            
            return {
                "type": "Bullish Breakout",
                "entry": round(entry_price, 2),
                "stop_loss": round(stop_loss, 2),
                "target_1": round(target_1, 2),
                "target_2": round(target_2, 2),
                "position_size": round(position_size),
                "risk_reward": round((target_1 - entry_price) / (entry_price - stop_loss), 2)
            }
            
        elif breakout_type == "support":
            entry_price = level * (1 - buffer)  # Entry di bawah support dengan buffer
            stop_loss = entry_price + 2 * atr  # Stop loss 2 ATR di atas entry
            target_1 = entry_price - 2 * atr  # Target 1: 2 ATR
            target_2 = entry_price - 4 * atr  # Target 2: 4 ATR
            
            position_size = self.calculate_position_size(entry_price, stop_loss, account_size)
            
            return {
                "type": "Bearish Breakdown",
                "entry": round(entry_price, 2),
                "stop_loss": round(stop_loss, 2),
                "target_1": round(target_1, 2),
                "target_2": round(target_2, 2),
                "position_size": round(position_size),
                "risk_reward": round((entry_price - target_1) / (stop_loss - entry_price), 2)
            }
        
        return None

# --- FUNGSI AKUMULASI/DISTRIBUSI ---
def detect_accumulation_distribution(df, period=5):
    """Mendeteksi pola akumulasi atau distribusi"""
    # Hitung indikator akumulasi/distribusi
    adl = compute_adl(df['High'], df['Low'], df['Close'], df['Volume'])
    obv = compute_obv(df['Close'], df['Volume'])
    mfi = compute_mfi_corrected(df, 14)
    
    # Hitung perubahan 5 hari terakhir
    adl_change = (adl.iloc[-1] - adl.iloc[-period]) / adl.iloc[-period] * 100 if adl.iloc[-period] != 0 else 0
    obv_change = (obv.iloc[-1] - obv.iloc[-period]) / abs(obv.iloc[-period]) * 100 if obv.iloc[-period] != 0 else 0
    mfi_avg = mfi.iloc[-period:].mean()
    
    # Volume analysis
    vol_ratio = df['Volume'].iloc[-period:].mean() / df['Volume'].rolling(20).mean().iloc[-1]
    
    # Price action analysis
    price_change = (df['Close'].iloc[-1] - df['Close'].iloc[-period]) / df['Close'].iloc[-period] * 100
    vwap = compute_vwap(df['High'], df['Low'], df['Close'], df['Volume'])
    price_vs_vwap = (df['Close'].iloc[-1] - vwap.iloc[-1]) / vwap.iloc[-1] * 100
    
    # Skor akumulasi/distribusi
    accumulation_score = 0
    distribution_score = 0
    
    # Aturan untuk akumulasi
    if adl_change > 2: accumulation_score += 1
    if obv_change > 3: accumulation_score += 1
    if mfi_avg > 60: accumulation_score += 1
    if vol_ratio > 1.2: accumulation_score += 1
    if price_change > 0 and price_vs_vwap > 0: accumulation_score += 1
    
    # Aturan untuk distribusi
    if adl_change < -2: distribution_score += 1
    if obv_change < -3: distribution_score += 1
    if mfi_avg < 40: distribution_score += 1
    if vol_ratio > 1.2: distribution_score += 1  # Volume tinggi bisa distribusi atau akumulasi
    if price_change < 0 and price_vs_vwap < 0: distribution_score += 1
    
    # Tentukan hasil
    if accumulation_score >= 3 and accumulation_score > distribution_score:
        return "Akumulasi", accumulation_score, distribution_score
    elif distribution_score >= 3 and distribution_score > accumulation_score:
        return "Distribusi", accumulation_score, distribution_score
    else:
        return "Netral", accumulation_score, distribution_score

def calculate_volume_profile(df, period=5, bins=10):
    """Menghitung volume profile untuk periode tertentu"""
    recent_data = df.iloc[-period:]
    
    # Buat bins berdasarkan range harga
    price_min = recent_data['Low'].min()
    price_max = recent_data['High'].max()
    bin_edges = np.linspace(price_min, price_max, bins + 1)
    
    # Distribusikan volume ke setiap bin
    volume_by_bin = np.zeros(bins)
    
    for _, row in recent_data.iterrows():
        # Untuk setiap bar, distribusikan volume berdasarkan perbandingan waktu
        # di setiap level harga (sederhana: bagi rata volume ke semua level dalam range)
        price_range = row['High'] - row['Low']
        if price_range > 0:
            volume_per_price = row['Volume'] / price_range
            for i in range(bins):
                bin_low = bin_edges[i]
                bin_high = bin_edges[i+1]
                
                # Hitung overlap antara bin dan range harga bar
                overlap_low = max(bin_low, row['Low'])
                overlap_high = min(bin_high, row['High'])
                overlap = max(0, overlap_high - overlap_low)
                
                volume_by_bin[i] += overlap * volume_per_price
    
    # Temukan bin dengan volume tertinggi
    max_volume_bin = np.argmax(volume_by_bin)
    price_zone_low = bin_edges[max_volume_bin]
    price_zone_high = bin_edges[max_volume_bin+1]
    
    return volume_by_bin, bin_edges, price_zone_low, price_zone_high

# --- FUNGSI BANTUAN DATA DAN ANALISIS ---
def get_stock_data(ticker, end_date):
    """Mengambil data saham dari Yahoo Finance"""
    try:
        stock = yf.Ticker(f"{ticker}.JK")
        start_date = end_date - timedelta(days=180)  # Ambil data 6 bulan untuk analisis lebih baik
        data = stock.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
        
        if data.empty:
            # Coba lagi dengan tanggal yang lebih awal jika diperlukan
            start_date = end_date - timedelta(days=360)
            data = stock.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
        
        return data if not data.empty else None
    except Exception as e:
        st.error(f"Gagal mengambil data untuk {ticker}: {e}")
        return None

def identify_significant_swings(df, window=60, min_swing_size=0.05):
    """Mengidentifikasi swing signifikan dengan ukuran minimum perubahan"""
    highs = df['High']
    lows = df['Low']
    
    # Gunakan order yang lebih dinamis berdasarkan window
    order = max(5, window // 12)
    
    max_idx = argrelextrema(highs.values, np.greater, order=order)[0]
    min_idx = argrelextrema(lows.values, np.less, order=order)[0]
    
    recent_highs = highs.iloc[max_idx][-10:] if len(max_idx) > 0 else pd.Series()
    recent_lows = lows.iloc[min_idx][-10:] if len(min_idx) > 0 else pd.Series()
    
    if len(recent_highs) == 0 or len(recent_lows) == 0:
        return df['High'].max(), df['Low'].min()
    
    # Filter swings berdasarkan ukuran minimum
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
    """Menghitung level Fibonacci berdasarkan swing high dan swing low"""
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

def find_psychological_levels(close_price):
    """Menemukan level psikologis terdekat"""
    levels = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
    closest_level = min(levels, key=lambda x: abs(x - close_price))
    return closest_level

def calculate_support_resistance(data):
    """Menghitung level support dan resistance dengan berbagai metode"""
    df = data.copy()
    current_price = df['Close'].iloc[-1]
    
    # Identifikasi swing points
    swing_high, swing_low = identify_significant_swings(df.tail(60))
    fib_levels = calculate_fibonacci_levels(swing_high, swing_low)
    
    # Moving averages
    ma20 = df['Close'].rolling(20).mean().iloc[-1]
    ma50 = df['Close'].rolling(50).mean().iloc[-1]
    ma100 = df['Close'].rolling(100).mean().iloc[-1]
    
    # VWAP
    vwap = compute_vwap(df['High'], df['Low'], df['Close'], df['Volume']).iloc[-1]
    
    # Psychological levels
    psych_level = find_psychological_levels(current_price)
    
    # Support levels
    support_levels = [
        fib_levels['Fib_0.618'], 
        fib_levels['Fib_0.786'],
        ma20,
        ma50,
        vwap,
        psych_level
    ]
    
    # Resistance levels
    resistance_levels = [
        fib_levels['Fib_0.236'], 
        fib_levels['Fib_0.382'],
        ma100,
        vwap,
        psych_level
    ]
    
    # Tambahkan Fibonacci levels jika relevan
    if not np.isnan(fib_levels['Fib_0.0']) and fib_levels['Fib_0.0'] > current_price:
        resistance_levels.append(fib_levels['Fib_0.0'])
    if not np.isnan(fib_levels['Fib_1.0']) and fib_levels['Fib_1.0'] < current_price:
        support_levels.append(fib_levels['Fib_1.0'])

    # Filter level yang valid
    valid_support = [lvl for lvl in support_levels if not np.isnan(lvl) and lvl < current_price]
    valid_resistance = [lvl for lvl in resistance_levels if not np.isnan(lvl) and lvl > current_price]
    
    # Urutkan dan ambil 3 level terkuat
    valid_support.sort(reverse=True)
    valid_resistance.sort()
    
    return {
        'Support': valid_support[:3] if valid_support else [current_price * 0.95],
        'Resistance': valid_resistance[:3] if valid_resistance else [current_price * 1.05],
        'Fibonacci': fib_levels
    }

def create_technical_chart(df, sr, is_squeeze):
    """Membuat chart teknikal lengkap dengan semua indikator"""
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
    
    # Moving averages
    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], mode='lines', name='MA20', line=dict(color='blue', width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], mode='lines', name='MA50', line=dict(color='orange', width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA100'], mode='lines', name='MA100', line=dict(color='purple', width=1.5)))
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], mode='lines', name='BB Upper', 
                            line=dict(color='red', width=1, dash='dot')))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Middle'], mode='lines', name='BB Middle', 
                            line=dict(color='purple', width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], mode='lines', name='BB Lower', 
                            line=dict(color='green', width=1, dash='dot'),
                            fill='tonexty', fillcolor='rgba(173, 216, 230, 0.1)'))
    
    # Support & Resistance
    for i, level in enumerate(sr['Support']):
        fig.add_hline(y=level, line_dash="dash", line_color="green",
                      annotation_text=f"Support {i+1}: Rp {level:,.2f}", 
                      annotation_position="bottom right")
    
    for i, level in enumerate(sr['Resistance']):
        fig.add_hline(y=level, line_dash="dash", line_color="red",
                      annotation_text=f"Resistance {i+1}: Rp {level:,.2f}", 
                      annotation_position="top right")
    
    # Tambahkan indikator Squeeze jika terdeteksi
    if is_squeeze:
        fig.add_annotation(
            x=df.index[-1],
            y=df['Close'].iloc[-1],
            text="⚠️ BOLLINGER SQUEEZE",
            showarrow=True,
            arrowhead=1,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="purple",
            font=dict(size=12, color="purple", weight="bold"),
            yshift=20
        )
    
    fig.update_layout(
        title="Chart Teknikal Lengkap",
        xaxis_title="Tanggal",
        yaxis_title="Harga (Rp)",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        height=700,
        hovermode='x unified'
    )
    
    return fig

# --- FUNGSI UTAMA APLIKASI STREAMLIT ---
    st.title("📈 Analisa Teknikal Saham dengan Cross-Confirmation")
    
    # Input parameter
    col1, col2 = st.columns(2)
    with col1:
        ticker_input = st.text_input("Masukkan Kode Saham (contoh: BBCA)", value="BBCA")
        account_size = st.number_input("Modal (Rp)", value=100000000, step=10000000)
        risk_percent = st.slider("Risiko per Trade (%)", 0.5, 5.0, 2.0) / 100
    
    with col2:
        analysis_date = st.date_input("📅 Tanggal Analisis", value=datetime.today())
        use_multi_timeframe = st.checkbox("Gunakan Konfirmasi Multi-Timeframe", value=True)
    
    if st.button("🔍 Mulai Analisis"):
        with st.spinner('Sedang mengambil dan menganalisis data...'):
            time.sleep(1)
            
            if not ticker_input.strip():
                st.warning("Silakan masukkan kode saham.")
                return

            # Format ticker dengan benar
            ticker = ticker_input.strip().upper()
            if not ticker.endswith(".JK"):
                ticker += ".JK"
            
            # Ambil data saham
            data = get_stock_data(ticker.replace(".JK", ""), analysis_date)

            if data is None or data.empty:
                st.warning(f"Data untuk {ticker} tidak tersedia. Coba kode saham lain atau tanggal berbeda.")
                return

            # Hitung semua indikator
            df = data.copy()
            df['MA20'] = df['Close'].rolling(20).mean()
            df['MA50'] = df['Close'].rolling(50).mean()
            df['MA100'] = df['Close'].rolling(100).mean()
            df['RSI'] = compute_rsi(df['Close'])
            df['MACD'], df['Signal'], df['Hist'] = compute_macd(df['Close'])
            df['BB_Upper'], df['BB_Middle'], df['BB_Lower'], df['BB_Bandwidth'], df['BB_%B'] = compute_bollinger_bands(df['Close'])
            df['ATR'] = compute_atr(df['High'], df['Low'], df['Close'])
            df['OBV'] = compute_obv(df['Close'], df['Volume'])
            df['ADX'], df['Plus_DI'], df['Minus_DI'] = compute_adx(df['High'], df['Low'], df['Close'])
            df['VWAP'] = compute_vwap(df['High'], df['Low'], df['Close'], df['Volume'])
            df['MFI'] = compute_mfi_corrected(df)
            
            # Hitung volume moving average
            df['Volume_MA20'] = df['Volume'].rolling(20).mean()
            
            # Hitung support/resistance
            sr = calculate_support_resistance(df)
            
            # Inisialisasi sistem scoring
            scoring_system = IndicatorScoringSystem()
            
            # Hitung skor untuk setiap indikator
            scores = {}
            
            # RSI
            scores['rsi'] = scoring_system.score_rsi(df['RSI'])
            
            # MACD
            macd_cross_score, macd_cross_strength, macd_hist_score, macd_hist_strength = scoring_system.score_macd(
                df['MACD'], df['Signal'], df['Hist'])
            scores['macd_cross'] = (macd_cross_score, macd_cross_strength)
            scores['macd_hist'] = (macd_hist_score, macd_hist_strength)
            
            # Bollinger Bands
            bb_score, bb_strength, is_squeeze = scoring_system.score_bollinger(
                df['Close'], df['BB_Upper'], df['BB_Lower'], df['BB_%B'], 
                df['BB_Bandwidth'], df['BB_Bandwidth'].iloc[:-1])  # Historical bandwidth
            scores['bollinger'] = (bb_score, bb_strength)
            
            # Volume
            price_change = (df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100
            volume_score, volume_strength = scoring_system.score_volume(
                df['Volume'], df['Volume_MA20'], price_change)
            scores['volume'] = (volume_score, volume_strength)
            
            # OBV
            obv_score, obv_strength = scoring_system.score_obv(df['OBV'])
            scores['obv'] = (obv_score, obv_strength)
            
            # ADX
            adx_score, adx_strength = scoring_system.score_adx(df['ADX'], df['Plus_DI'], df['Minus_DI'])
            scores['adx'] = (adx_score, adx_strength)
            
            # Hitung composite score dan confidence
            composite_score = scoring_system.calculate_composite_score(scores)
            confidence = scoring_system.get_confidence_level(composite_score, scores)
            
            # Deteksi breakout
            breakout_detector = BreakoutDetector()
            resistance_breakout, support_breakdown, volume_confirm, buffer = breakout_detector.detect_breakout(
                df, sr['Resistance'][0] if sr['Resistance'] else df['Close'].iloc[-1] * 1.1,
                sr['Support'][0] if sr['Support'] else df['Close'].iloc[-1] * 0.9)
            
            # Deteksi akumulasi/distribusi
            accumulation_status, acc_score, dist_score = detect_accumulation_distribution(df)
            volume_profile, bin_edges, price_zone_low, price_zone_high = calculate_volume_profile(df)
            
            # Tampilkan hasil
            st.subheader("🎯 Hasil Analisis Cross-Confirmation")
            
            # Tampilkan composite score dengan gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = composite_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Composite Score", 'font': {'size': 24}},
                delta = {'reference': 0},
                gauge = {
                    'axis': {'range': [-1, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [-1, -0.5], 'color': 'red'},
                        {'range': [-0.5, 0], 'color': 'lightcoral'},
                        {'range': [0, 0.5], 'color': 'lightgreen'},
                        {'range': [0.5, 1], 'color': 'green'}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': composite_score}}))
            
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Tampilkan confidence level
            st.metric("Tingkat Keyakinan", confidence)
            
            # Tampilkan detail skor indikator
            st.subheader("📊 Detail Skor Indikator")
            indicator_cols = st.columns(3)
            
            with indicator_cols[0]:
                st.metric("RSI", f"{df['RSI'].iloc[-1]:.2f}", 
                         delta=f"Skor: {scores['rsi'][0]:.2f} (Kekuatan: {scores['rsi'][1]:.2f})")
                st.metric("MACD Cross", 
                         f"{'Bullish' if scores['macd_cross'][0] > 0 else 'Bearish' if scores['macd_cross'][0] < 0 else 'Netral'}",
                         delta=f"Skor: {scores['macd_cross'][0]:.2f} (Kekuatan: {scores['macd_cross'][1]:.2f})")
            
            with indicator_cols[1]:
                st.metric("Bollinger Bands", 
                         f"{'Bullish' if scores['bollinger'][0] > 0 else 'Bearish' if scores['bollinger'][0] < 0 else 'Netral'}",
                         delta=f"Skor: {scores['bollinger'][0]:.2f} (Kekuatan: {scores['bollinger'][1]:.2f})")
                st.metric("Volume", 
                         f"{'Bullish' if scores['volume'][0] > 0 else 'Bearish' if scores['volume'][0] < 0 else 'Netral'}",
                         delta=f"Skor: {scores['volume'][0]:.2f} (Kekuatan: {scores['volume'][1]:.2f})")
            
            with indicator_cols[2]:
                st.metric("OBV", 
                         f"{'Bullish' if scores['obv'][0] > 0 else 'Bearish' if scores['obv'][0] < 0 else 'Netral'}",
                         delta=f"Skor: {scores['obv'][0]:.2f} (Kekuatan: {scores['obv'][1]:.2f})")
                st.metric("ADX", 
                         f"{'Bullish' if scores['adx'][0] > 0 else 'Bearish' if scores['adx'][0] < 0 else 'Netral'}",
                         delta=f"Skor: {scores['adx'][0]:.2f} (Kekuatan: {scores['adx'][1]:.2f})")
            
            # Tampilkan analisis akumulasi/distribusi
            st.subheader("📦 Analisis Akumulasi/Distribusi")
            acc_cols = st.columns(2)
            
            with acc_cols[0]:
                st.metric("Status", accumulation_status)
                st.metric("Skor Akumulasi", acc_score)
                st.metric("Skor Distribusi", dist_score)
            
            with acc_cols[1]:
                st.write("**Volume Profile (5 Hari Terakhir)**")
                st.write(f"Zona harga dengan volume tertinggi: Rp {price_zone_low:,.2f} - Rp {price_zone_high:,.2f}")
                
                # Buat chart volume profile
                fig_volume = go.Figure(go.Bar(
                    x=bin_edges[:-1],
                    y=volume_profile,
                    name="Volume Profile"
                ))
                fig_volume.update_layout(
                    title="Distribusi Volume per Level Harga",
                    xaxis_title="Harga",
                    yaxis_title="Volume",
                    height=300
                )
                st.plotly_chart(fig_volume, use_container_width=True)
            
            # Tampilkan rekomendasi trading
            st.subheader("🎯 Rekomendasi Trading")
            
            # Generate trading plan jika ada breakout
            trading_plan = None
            if resistance_breakout and volume_confirm:
                trading_plan = breakout_detector.generate_trading_plan(
                    df, "resistance", sr['Resistance'][0], buffer, account_size)
            elif support_breakdown and volume_confirm:
                trading_plan = breakout_detector.generate_trading_plan(
                    df, "support", sr['Support'][0], buffer, account_size)
            
            if trading_plan:
                st.success("🚀 **Sinyal Breakout Terdeteksi!**")
                
                plan_cols = st.columns(2)
                with plan_cols[0]:
                    st.metric("Jenis Sinyal", trading_plan["type"])
                    st.metric("Harga Entry", f"Rp {trading_plan['entry']:,.2f}")
                    st.metric("Stop Loss", f"Rp {trading_plan['stop_loss']:,.2f}")
                
                with plan_cols[1]:
                    st.metric("Target 1", f"Rp {trading_plan['target_1']:,.2f}")
                    st.metric("Target 2", f"Rp {trading_plan['target_2']:,.2f}")
                    st.metric("Risk/Reward", trading_plan["risk_reward"])
                
                st.metric("Ukuran Posisi", f"{trading_plan['position_size']:,} saham")
                
                # Tampilkan penjelasan manajemen risiko
                st.info(f"""
                **Manajemen Risiko:**
                - Ukuran posisi dihitung berdasarkan risiko {risk_percent*100}% dari modal (Rp {account_size:,.0f})
                - Risk/Reward Ratio: {trading_plan['risk_reward']}:1
                - Pertimbangkan untuk mengambil profit sebagian di Target 1 dan trail stop untuk sisanya
                """)
            else:
                # Tampilkan kondisi saat ini dan level untuk monitor
                st.warning("⏳ **Belum Terdeteksi Breakout yang Kuat**")
                
                if sr['Resistance']:
                    st.write(f"**Resistance Terdekat:** Rp {sr['Resistance'][0]:,.2f}")
                    st.write(f"**Syarat Breakout Bullish:** Close > Rp {sr['Resistance'][0] * (1 + buffer):,.2f} dengan volume > 1.5x rata-rata")
                
                if sr['Support']:
                    st.write(f"**Support Terdekat:** Rp {sr['Support'][0]:,.2f}")
                    st.write(f"**Syarat Breakdown Bearish:** Close < Rp {sr['Support'][0] * (1 - buffer):,.2f} dengan volume > 1.5x rata-rata")
            
            # Tampilkan chart teknikal lengkap
            st.subheader("📈 Chart Teknikal Lengkap")
            
            # Buat chart dengan Plotly
            fig = create_technical_chart(df, sr, is_squeeze)
            st.plotly_chart(fig, use_container_width=True)
            
            # Disclaimer
            st.info("⚠️ **Disclaimer**: Analisis ini hanya untuk tujuan edukasi dan bukan sebagai rekomendasi investasi. "
                    "Harga saham bisa berubah sewaktu-waktu. Lakukan riset tambahan dan konsultasi dengan "
                    "penasihat keuangan sebelum mengambil keputusan investasi.")

if __name__ == "__main__":
    main()
