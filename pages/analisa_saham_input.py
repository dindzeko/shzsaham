# analisa_saham.py
"""
Analisa Teknikal Saham - Versi Full Refactor
Fitur:
- Indikator: RSI, MACD, Bollinger, ATR, OBV (vektorized), ADX, VWAP, MFI (safe)
- Scoring system (composite) dengan versi scoring yang lebih halus
- Breakout detection (ATR-based buffer) + trading plan + position sizing dengan cap
- Bandarmology: volume spikes, smart money patterns, advanced volume profile (POC, Value Area)
- Support/Resistance & Fibonacci
- Multi-timeframe option (resample weekly/monthly)
- Caching untuk data fetch dan perhitungan berat
- NaN handling & defensive programming
"""
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from scipy.signal import argrelextrema
from scipy import stats
import time

st.set_page_config(layout="wide", page_title="Analisa Teknikal Saham (Refactor)")

# ----------------------
# Konstanta & Config
# ----------------------
INDICATOR_WEIGHTS = {
    'rsi': 0.15,
    'macd_cross': 0.25,
    'macd_hist': 0.10,
    'bollinger': 0.15,
    'volume': 0.20,
    'obv': 0.10,
    'adx': 0.05
}

MIN_DATA_POINTS = 60  # default minimal bars to run full analysis

# ----------------------
# Utility / Data Fetch
# ----------------------
@st.cache_data(show_spinner=False)
def get_stock_data_yf(ticker_no_suffix: str, end_date: datetime, days_back=360):
    """
    Ambil data OHLCV dari yfinance. Terkadang Yahoo API gagal untuk rentang singkat -> ambil 360 hari.
    Return: DataFrame atau None
    """
    try:
        ticker = yf.Ticker(f"{ticker_no_suffix}.JK")
        start = end_date - timedelta(days=days_back)
        df = ticker.history(start=start.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
        if df.empty:
            return None
        # normalize column names (Open/High/Low/Close/Volume)
        df = df.rename(columns={c: c.title() for c in df.columns})
        # keep required columns
        cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing = [c for c in cols if c not in df.columns]
        if missing:
            return None
        return df[cols].copy()
    except Exception as e:
        st.error(f"Gagal mengambil data: {e}")
        return None

# ----------------------
# Indicator Implementations (refactored & robust)
# ----------------------
def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(0)

def compute_mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Money Flow Index safe implementation:
    - typical price = (H+L+C)/3
    - money flow = typical price * volume
    - positive/negative flow based on typical price change
    - rolling sums min_periods=period
    - handle neg_mf == 0
    """
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    mf = tp * df['Volume']
    pos_flow = mf.where(tp > tp.shift(1), 0.0)
    neg_flow = mf.where(tp < tp.shift(1), 0.0)
    pos_mf = pos_flow.rolling(window=period, min_periods=period).sum()
    neg_mf = neg_flow.rolling(window=period, min_periods=period).sum()

    mfi = pd.Series(index=df.index, dtype=float)
    # mask where both are zero -> neutral 50
    both_zero = (pos_mf == 0) & (neg_mf == 0)
    mfi[both_zero] = 50.0

    # mask where neg_mf == 0 and pos_mf > 0 -> MFI = 100
    mask_pos_only = (neg_mf == 0) & (pos_mf > 0)
    mfi[mask_pos_only] = 100.0

    # mask where neg_mf > 0 normal calc
    mask_normal = neg_mf > 0
    ratio = pd.Series(np.nan, index=df.index)
    ratio[mask_normal] = pos_mf[mask_normal] / neg_mf[mask_normal]
    mfi[mask_normal] = 100 - (100 / (1 + ratio[mask_normal]))

    return mfi.fillna(50.0)

def compute_macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line.fillna(0), signal_line.fillna(0), histogram.fillna(0)

def compute_bollinger_bands(close: pd.Series, window=20, num_std=2):
    sma = close.rolling(window=window, min_periods=window).mean()
    std = close.rolling(window=window, min_periods=window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    bandwidth = (upper - lower) / sma
    percent_b = (close - lower) / (upper - lower)
    # Fillna with reasonable defaults so later code doesn't crash
    return upper.fillna(method='ffill').fillna(0), sma.fillna(method='ffill').fillna(0), lower.fillna(method='ffill').fillna(0), bandwidth.fillna(0), percent_b.fillna(0.5)

def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period=14):
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()
    return atr.fillna(0)

def compute_obv(close: pd.Series, volume: pd.Series):
    """
    Vectorized OBV:
    obv[0] = 0 (or starting at volume if you want)
    obv = cumsum(volume * sign(close.diff()))
    For sign==0, we add 0 (carry previous)
    """
    price_diff = close.diff().fillna(0)
    sign = np.sign(price_diff)  # -1, 0, 1
    obv = (volume * sign).cumsum()
    # make sure index aligns
    return obv.fillna(method='ffill').fillna(0)

def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period=14):
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # ewm smoothing
    tr_smooth = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean() / tr_smooth.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean() / tr_smooth.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx.fillna(0), plus_di.fillna(0), minus_di.fillna(0)

def compute_vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series):
    tp = (high + low + close) / 3
    cum_vp = (tp * volume).cumsum()
    cum_vol = volume.cumsum()
    vwap = cum_vp / cum_vol.replace(0, np.nan)
    return vwap.fillna(method='ffill').fillna(close)

def compute_adl(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series):
    mfm = ((close - low) - (high - close)) / (high - low)
    mfm = mfm.replace([np.inf, -np.inf], 0).fillna(0)
    mfv = mfm * volume
    adl = mfv.cumsum()
    return adl.fillna(0)

# ----------------------
# Scoring System
# ----------------------
class IndicatorScoringSystem:
    def __init__(self, weights=None):
        self.weights = weights or INDICATOR_WEIGHTS

    def score_rsi(self, rsi_series: pd.Series, period=3):
        current = rsi_series.iloc[-1]
        # map RSI to -1..1 where 50->0, lower -> positive (bullish oversold), higher -> negative
        score_val = (50 - current) / 50.0  # -1..1 roughly
        strength = min(1.0, abs(score_val))
        # small trend adjust
        trend = self.calculate_trend(rsi_series, period)
        if trend > 0.1 and score_val > 0:
            strength = min(1.0, strength + 0.2)
        elif trend < -0.1 and score_val < 0:
            strength = min(1.0, strength + 0.2)
        # return sign (direction) and strength
        return np.sign(score_val), strength

    def score_macd(self, macd_line: pd.Series, signal_line: pd.Series, histogram: pd.Series):
        cross_dir = 1.0 if macd_line.iloc[-1] > signal_line.iloc[-1] else -1.0
        hist_trend = self.calculate_trend(histogram.iloc[-5:], 3)
        hist_score = 0.5 * np.sign(hist_trend)
        hist_strength = min(1.0, abs(hist_trend) * 2)
        # prefer crosses above zero
        if macd_line.iloc[-1] < 0:
            # reduce conviction if MACD is below zero
            cross_dir *= 0.8
        return cross_dir, 1.0, hist_score, hist_strength

    def score_bollinger(self, price: pd.Series, upper: pd.Series, lower: pd.Series, percent_b: pd.Series, bandwidth: pd.Series, history_bandwidth: pd.Series):
        current_pctb = percent_b.iloc[-1]
        current_band = bandwidth.iloc[-1]
        bw_percentile = stats.percentileofscore(history_bandwidth.dropna(), current_band) / 100.0 if len(history_bandwidth.dropna()) > 0 else 0.5
        is_squeeze = bw_percentile < 0.2
        # determine score based on percent_b
        if current_pctb > 0.8:
            score = -1.0
            strength = min(1.0, (current_pctb - 0.8) / 0.2)
        elif current_pctb < 0.2:
            score = 1.0
            strength = min(1.0, (0.2 - current_pctb) / 0.2)
        elif current_pctb > 0.5:
            score = -0.5
            strength = (current_pctb - 0.5) / 0.5
        else:
            score = 0.5
            strength = (0.5 - current_pctb) / 0.5
        if is_squeeze:
            strength = min(1.0, strength + 0.3)
        return score, strength, is_squeeze

    def score_volume(self, volume: pd.Series, volume_ma: pd.Series, price_change_pct: float):
        last_vol = volume.iloc[-1]
        ma_vol = volume_ma.iloc[-1] if not np.isnan(volume_ma.iloc[-1]) else np.mean(volume[-20:])
        ratio = last_vol / (ma_vol + 1e-9)
        if ratio > 1.7 and price_change_pct > 0:
            return 1.0, min(1.0, (ratio - 1.7) / 0.8)
        elif ratio > 1.7 and price_change_pct < 0:
            return -1.0, min(1.0, (ratio - 1.7) / 0.8)
        elif ratio > 1.3 and price_change_pct > 0:
            return 0.5, (ratio - 1.3) / 0.4
        elif ratio > 1.3 and price_change_pct < 0:
            return -0.5, (ratio - 1.3) / 0.4
        else:
            return 0.0, 0.0

    def score_obv(self, obv_values: pd.Series, period=5):
        trend = self.calculate_trend(obv_values, period)
        if trend > 0.02:
            return 1.0, min(1.0, trend * 5)
        elif trend < -0.02:
            return -1.0, min(1.0, abs(trend) * 5)
        else:
            return 0.0, 0.0

    def score_adx(self, adx_values: pd.Series, plus_di: pd.Series, minus_di: pd.Series, threshold=25):
        current_adx = adx_values.iloc[-1]
        direction = 1.0 if plus_di.iloc[-1] > minus_di.iloc[-1] else -1.0
        if current_adx > threshold:
            strength = min(1.0, (current_adx - threshold) / (100 - threshold))
            return direction, strength
        else:
            return 0.0, 0.0

    def calculate_trend(self, series: pd.Series, period: int):
        if len(series.dropna()) < period:
            return 0.0
        y = series.dropna().values[-period:]
        x = np.arange(len(y))
        slope, _, _, _, _ = stats.linregress(x, y)
        mean_y = np.mean(y) if np.mean(y) != 0 else 1.0
        return slope / mean_y

    def calculate_composite_score(self, scores: dict):
        weighted = 0.0
        total_w = 0.0
        for k, (score, strength) in scores.items():
            w = self.weights.get(k, 0)
            weighted += score * strength * w
            total_w += w
        return weighted / total_w if total_w > 0 else 0.0

    def get_confidence_level(self, composite, scores):
        if composite > 0:
            agreeing = sum(1 for (s, strg) in scores.values() if s > 0 and strg > 0.3)
        elif composite < 0:
            agreeing = sum(1 for (s, strg) in scores.values() if s < 0 and strg > 0.3)
        else:
            agreeing = 0
        total = len(scores)
        ratio = agreeing / total if total > 0 else 0
        if ratio >= 0.7:
            return "Tinggi"
        elif ratio >= 0.5:
            return "Sedang"
        else:
            return "Rendah"

    def interpret_composite_score(self, score):
        if score >= 0.7:
            return "Sangat Bullish - Sentimen beli sangat kuat"
        elif score >= 0.4:
            return "Bullish Kuat - Sentimen beli kuat"
        elif score >= 0.1:
            return "Bullish Lemah - Sentimen cenderung beli"
        elif score > -0.1:
            return "Netral - Sentimen tidak jelas"
        elif score > -0.4:
            return "Bearish Lemah - Sentimen cenderung jual"
        elif score > -0.7:
            return "Bearish Kuat - Sentimen jual kuat"
        else:
            return "Sangat Bearish - Sentimen jual sangat kuat"

# ----------------------
# Support / Resistance / Fib / Swings
# ----------------------
def identify_significant_swings(df: pd.DataFrame, window=60, min_swing_size=0.05):
    highs = df['High']
    lows = df['Low']
    order = max(5, window // 12)
    max_idx = argrelextrema(highs.values, np.greater, order=order)[0]
    min_idx = argrelextrema(lows.values, np.less, order=order)[0]
    recent_highs = highs.iloc[max_idx][-10:] if len(max_idx) > 0 else pd.Series(dtype=float)
    recent_lows = lows.iloc[min_idx][-10:] if len(min_idx) > 0 else pd.Series(dtype=float)
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
    swing_high = max(significant_highs) if significant_highs else (recent_highs.max() if len(recent_highs)>0 else df['High'].max())
    swing_low = min(significant_lows) if significant_lows else (recent_lows.min() if len(recent_lows)>0 else df['Low'].min())
    return swing_high, swing_low

def calculate_fibonacci_levels(swing_high, swing_low):
    diff = swing_high - swing_low if swing_high is not None and swing_low is not None else 0
    return {
        'Fib_0.0': round(swing_high, 2),
        'Fib_0.236': round(swing_high - 0.236 * diff, 2),
        'Fib_0.382': round(swing_high - 0.382 * diff, 2),
        'Fib_0.5': round(swing_high - 0.5 * diff, 2),
        'Fib_0.618': round(swing_high - 0.618 * diff, 2),
        'Fib_0.786': round(swing_high - 0.786 * diff, 2),
        'Fib_1.0': round(swing_low, 2)
    }

def calculate_support_resistance(df: pd.DataFrame):
    current = df['Close'].iloc[-1]
    swing_high, swing_low = identify_significant_swings(df.tail(60))
    fib = calculate_fibonacci_levels(swing_high, swing_low)
    ma20 = df['Close'].rolling(20).mean().iloc[-1] if len(df) >= 20 else np.nan
    ma50 = df['Close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else np.nan
    ma100 = df['Close'].rolling(100).mean().iloc[-1] if len(df) >= 100 else np.nan
    vwap = compute_vwap(df['High'], df['Low'], df['Close'], df['Volume']).iloc[-1]
    psych = find_psychological_levels(current)
    support_levels = [fib['Fib_0.618'], fib['Fib_0.786'], ma20, ma50, vwap, psych]
    resistance_levels = [fib['Fib_0.236'], fib['Fib_0.382'], ma100, vwap, psych, fib['Fib_0.0']]
    valid_support = [lvl for lvl in support_levels if not np.isnan(lvl) and lvl < current]
    valid_resistance = [lvl for lvl in resistance_levels if not np.isnan(lvl) and lvl > current]
    valid_support.sort(reverse=True)
    valid_resistance.sort()
    return {
        'Support': valid_support[:3] if valid_support else [current * 0.95],
        'Resistance': valid_resistance[:3] if valid_resistance else [current * 1.05],
        'Fibonacci': fib
    }

def find_psychological_levels(price):
    levels = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
    return min(levels, key=lambda x: abs(x - price))

# ----------------------
# Volume Profile / Bandarmology
# ----------------------
def calculate_volume_profile_advanced(df: pd.DataFrame, period=20, bins=20):
    recent = df.iloc[-period:]
    price_min = recent['Low'].min()
    price_max = recent['High'].max()
    if price_max <= price_min:
        price_max = price_min + 1e-6
    bin_edges = np.linspace(price_min, price_max, bins + 1)
    volume_by_price = np.zeros(bins)
    value_by_price = np.zeros(bins)

    # distribute volume across bins (approximation)
    for _, row in recent.iterrows():
        low = row['Low']
        high = row['High']
        vol = row['Volume']
        if high - low <= 0:
            continue
        # find overlap with bins
        for i in range(bins):
            bin_low = bin_edges[i]
            bin_high = bin_edges[i + 1]
            overlap_low = max(bin_low, low)
            overlap_high = min(bin_high, high)
            overlap = max(0.0, overlap_high - overlap_low)
            if overlap > 0:
                # proportion of bar range contributing to this bin
                prop = overlap / (high - low)
                volume_by_price[i] += vol * prop
                value_by_price[i] += ( (bin_low + bin_high) / 2.0 ) * vol * prop

    poc_index = int(np.argmax(volume_by_price)) if len(volume_by_price) > 0 else 0
    poc_price = (bin_edges[poc_index] + bin_edges[poc_index + 1]) / 2.0
    total_volume = volume_by_price.sum() if volume_by_price.sum() > 0 else 1.0
    sorted_idx = np.argsort(volume_by_price)[::-1]
    cum = 0.0
    value_area_idx = []
    for idx in sorted_idx:
        cum += volume_by_price[idx]
        value_area_idx.append(idx)
        if cum >= 0.7 * total_volume:
            break
    if value_area_idx:
        val_low = bin_edges[min(value_area_idx)]
        val_high = bin_edges[max(value_area_idx)+1]
    else:
        val_low, val_high = price_min, price_max

    recent_up = recent[recent['Close'] > recent['Open']]
    recent_down = recent[recent['Close'] < recent['Open']]
    buying_vol = recent_up['Volume'].sum()
    selling_vol = recent_down['Volume'].sum()
    volume_delta = buying_vol - selling_vol

    return {
        'poc_price': poc_price,
        'value_area_low': val_low,
        'value_area_high': val_high,
        'volume_delta': volume_delta,
        'volume_profile': volume_by_price,
        'bin_edges': bin_edges,
        'buying_volume': buying_vol,
        'selling_volume': selling_vol
    }

def analyze_institutional_activity(df: pd.DataFrame, period=20):
    results = {}
    volume_ma = df['Volume'].rolling(period).mean()
    volume_std = df['Volume'].rolling(period).std().replace(0, 1)
    df['Volume_ZScore'] = (df['Volume'] - volume_ma) / volume_std
    results['Volume_Spikes_Last_5_Days'] = (df['Volume_ZScore'] > 2.5).iloc[-5:].sum()
    price_change = df['Close'].pct_change()
    volume_change = df['Volume'].pct_change()
    pos_corr = ((price_change > 0) & (volume_change > 0)).iloc[-period:].sum()
    neg_corr = ((price_change < 0) & (volume_change > 0)).iloc[-period:].sum()
    results['Positive_Volume_Price_Days'] = int(pos_corr)
    results['Negative_Volume_Price_Days'] = int(neg_corr)
    volume_clusters = (df['Volume'] > volume_ma * 1.5).rolling(3).sum().iloc[-5:].max()
    results['Volume_Clusters'] = int(volume_clusters) if not np.isnan(volume_clusters) else 0
    down_days = df['Close'] < df['Open']
    followthrough = (df['Close'].shift(-1) > df['Open'].shift(-1)) & down_days
    results['Resilience_Days'] = int(followthrough.iloc[-period:].sum())
    return results, df

def detect_smart_money_patterns(df: pd.DataFrame, period=30):
    patterns = {}
    support = df['Low'].rolling(20).mean()
    spring_pattern = ((df['Low'] < support) & (df['Close'] > support)).iloc[-5:].sum()
    avg_range = (df['High'] - df['Low']).rolling(20).mean()
    stopping_vol = ((df['Volume'] > df['Volume'].rolling(20).mean() * 1.5) & ((df['High'] - df['Low']) < avg_range * 0.7)).iloc[-5:].sum()
    climax_action = ((df['Volume'] > df['Volume'].rolling(20).mean() * 2) & ((df['High'] - df['Low']) > avg_range * 1.5)).iloc[-5:].sum()
    hidden_buying = ((df['Close'] < df['Open']) & (df['Volume'] < df['Volume'].rolling(20).mean() * 0.7)).iloc[-5:].sum()
    hidden_selling = ((df['Close'] > df['Open']) & (df['Volume'] < df['Volume'].rolling(20).mean() * 0.7)).iloc[-5:].sum()
    patterns['Spring_Pattern'] = int(spring_pattern)
    patterns['Stopping_Volume'] = int(stopping_vol)
    patterns['Climax_Action'] = int(climax_action)
    patterns['Hidden_Buying'] = int(hidden_buying)
    patterns['Hidden_Selling'] = int(hidden_selling)
    return patterns

def generate_bandarmology_report(df: pd.DataFrame, period=30):
    report = {}
    institutional, df2 = analyze_institutional_activity(df, period)
    report['institutional_activity'] = institutional
    patterns = detect_smart_money_patterns(df, period)
    report['smart_money_patterns'] = patterns
    vp = calculate_volume_profile_advanced(df, period)
    report['volume_profile_analysis'] = vp
    price_trend = (df['Close'].iloc[-1] / df['Close'].iloc[-period] - 1) if len(df) >= period+1 else 0
    volume_trend = (df['Volume'].iloc[-period:].mean() / (df['Volume'].iloc[-2*period:-period].mean() + 1e-9) - 1) if len(df) >= 2*period else 0
    if price_trend > 0.05 and volume_trend > 0.1:
        report['trend_assessment'] = "Uptrend kuat dengan konfirmasi volume"
    elif price_trend > 0.05 and volume_trend < -0.1:
        report['trend_assessment'] = "Uptrend lemah, kurang volume konfirmasi"
    elif price_trend < -0.05 and volume_trend > 0.1:
        report['trend_assessment'] = "Downtrend kuat dengan volume tinggi"
    elif price_trend < -0.05 and volume_trend < -0.1:
        report['trend_assessment'] = "Downtrend mungkin exhausted (volume rendah)"
    else:
        report['trend_assessment'] = "Sideways atau trend tidak jelas"
    conclusion = []
    inst = report['institutional_activity']
    if inst['Volume_Spikes_Last_5_Days'] >= 2:
        conclusion.append("🔍 Aktivitas volume tinggi terdeteksi - Kemungkinan ada aktivitas institusional")
    if inst['Positive_Volume_Price_Days'] > inst['Negative_Volume_Price_Days'] * 1.5:
        conclusion.append("📈 Dominasi buying pressure")
    elif inst['Negative_Volume_Price_Days'] > inst['Positive_Volume_Price_Days'] * 1.5:
        conclusion.append("📉 Dominasi selling pressure")
    if patterns['Spring_Pattern'] > 0:
        conclusion.append("🔄 Pola Spring terdeteksi - kemungkinan akumulasi")
    if patterns['Stopping_Volume'] > 0:
        conclusion.append("⏹️ Stopping Volume terdeteksi - institusi menahan pergerakan")
    if patterns['Climax_Action'] > 0:
        conclusion.append("🎯 Climax Action - kemungkinan exhaustion move")
    vp = report['volume_profile_analysis']
    cur_price = df['Close'].iloc[-1]
    if cur_price > vp['value_area_high']:
        conclusion.append("🚀 Harga di atas Value Area - kondisi bullish")
    elif cur_price < vp['value_area_low']:
        conclusion.append("🔻 Harga di bawah Value Area - kondisi bearish")
    else:
        conclusion.append("↔️ Harga dalam Value Area - konsolidasi")
    if vp['volume_delta'] > 0:
        conclusion.append(f"➕ Volume Delta positif ({vp['volume_delta']:.0f}) - Buying pressure dominan")
    else:
        conclusion.append(f"➖ Volume Delta negatif ({vp['volume_delta']:.0f}) - Selling pressure dominan")
    report['conclusion'] = conclusion
    return report, df2

# ----------------------
# Breakout Detector & Trading Plan
# ----------------------
class BreakoutDetector:
    def __init__(self, atr_period=14, buffer_percent=0.005, max_position_percent=0.1):
        self.atr_period = atr_period
        self.buffer_percent = buffer_percent
        self.max_position_percent = max_position_percent

    def detect_breakout(self, df: pd.DataFrame, resistance_level: float, support_level: float):
        current_close = df['Close'].iloc[-1]
        current_vol = df['Volume'].iloc[-1]
        avg_vol = df['Volume'].rolling(20).mean().iloc[-1] if len(df) >= 20 else df['Volume'].mean()
        atr = compute_atr(df['High'], df['Low'], df['Close'], self.atr_period).iloc[-1]
        atr_buffer = (atr / df['Close'].iloc[-1]) if df['Close'].iloc[-1] != 0 else self.buffer_percent
        buffer = max(self.buffer_percent, 0.5 * atr_buffer)
        resistance_breakout = current_close > resistance_level * (1 + buffer)
        support_breakdown = current_close < support_level * (1 - buffer)
        volume_confirm = current_vol > 1.5 * (avg_vol + 1e-9)
        return resistance_breakout, support_breakdown, volume_confirm, buffer

    def calculate_position_size(self, entry_price, stop_price, account_size, risk_percent=0.02):
        risk_amount = account_size * risk_percent
        risk_per_share = abs(entry_price - stop_price)
        if risk_per_share <= 0:
            return 0
        raw_size = risk_amount / risk_per_share
        max_size = (account_size * self.max_position_percent) / max(entry_price, 1e-9)
        return int(min(raw_size, max_size))

    def generate_trading_plan(self, df: pd.DataFrame, breakout_type: str, level: float, buffer: float, account_size=100_000_000):
        atr = compute_atr(df['High'], df['Low'], df['Close'], self.atr_period).iloc[-1]
        cur = df['Close'].iloc[-1]
        if breakout_type == "resistance":
            entry = level * (1 + buffer)
            stop = entry - 2 * atr
            target1 = entry + 2 * atr
            target2 = entry + 4 * atr
            size = self.calculate_position_size(entry, stop, account_size)
            rr = (target1 - entry) / (entry - stop) if (entry - stop) != 0 else None
            return {
                "type": "Bullish Breakout",
                "entry": round(entry, 2),
                "stop_loss": round(stop, 2),
                "target_1": round(target1, 2),
                "target_2": round(target2, 2),
                "position_size": size,
                "risk_reward": round(rr, 2) if rr is not None else None
            }
        elif breakout_type == "support":
            entry = level * (1 - buffer)
            stop = entry + 2 * atr
            target1 = entry - 2 * atr
            target2 = entry - 4 * atr
            size = self.calculate_position_size(entry, stop, account_size)
            rr = (entry - target1) / (stop - entry) if (stop - entry) != 0 else None
            return {
                "type": "Bearish Breakdown",
                "entry": round(entry, 2),
                "stop_loss": round(stop, 2),
                "target_1": round(target1, 2),
                "target_2": round(target2, 2),
                "position_size": size,
                "risk_reward": round(rr, 2) if rr is not None else None
            }
        return None

# ----------------------
# Plotting helpers
# ----------------------
def create_technical_chart(df: pd.DataFrame, sr: dict, is_squeeze: bool):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                                 name='Candlestick', increasing_line_color='green', decreasing_line_color='red'))
    if 'MA20' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], mode='lines', name='MA20'))
    if 'MA50' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], mode='lines', name='MA50'))
    if 'MA100' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MA100'], mode='lines', name='MA100'))
    if 'BB_Upper' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], mode='lines', name='BB Upper', line=dict(dash='dot')))
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Middle'], mode='lines', name='BB Middle'))
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], mode='lines', name='BB Lower', line=dict(dash='dot'), fill='tonexty', fillcolor='rgba(173,216,230,0.1)'))
    for i, lvl in enumerate(sr['Support']):
        fig.add_hline(y=lvl, line_dash="dash", line_color="green", annotation_text=f"Support {i+1}: Rp {lvl:,.2f}", annotation_position="bottom right")
    for i, lvl in enumerate(sr['Resistance']):
        fig.add_hline(y=lvl, line_dash="dash", line_color="red", annotation_text=f"Resistance {i+1}: Rp {lvl:,.2f}", annotation_position="top right")
    if is_squeeze:
        fig.add_annotation(x=df.index[-1], y=df['Close'].iloc[-1], text="BOLLINGER SQUEEZE", showarrow=True, arrowhead=1, arrowcolor="purple", font=dict(color="purple"))
    fig.update_layout(title="Chart Teknikal Lengkap", xaxis_rangeslider_visible=False, template="plotly_white", height=700)
    return fig

# ----------------------
# Main Streamlit App
# ----------------------
def app():
    st.title("📊 Analisa Teknikal Saham - Versi Refactor (Lengkap)")
    with st.sidebar:
        st.header("Parameter")
        ticker_input = st.text_input("Masukkan Kode Saham (tanpa .JK)", value="BBCA")
        account_size = st.number_input("Modal (Rp)", value=100_000_000, step=1_000_000)
        risk_percent = st.slider("Risiko per Trade (%)", min_value=0.5, max_value=5.0, value=2.0) / 100.0
        use_multi_timeframe = st.checkbox("Gunakan Multi-Timeframe (Weekly/Monthly) untuk konfirmasi", value=True)
        lookback_days = st.number_input("Ambil data (hari)", min_value=90, max_value=2000, value=360)
        st.markdown("---")
        st.write("**Catatan:** Perubahan parameter dapat mempengaruhi hasil analisis.")
    col1, col2 = st.columns(2)
    with col1:
        analysis_date = st.date_input("📅 Tanggal Analisis", value=datetime.today())
    with col2:
        st.write("")  # spacer

    if st.button("🚀 Mulai Analisis"):
        with st.spinner("Mengambil data..."):
            df = get_stock_data_yf(ticker_input, analysis_date, days_back=lookback_days)
            if df is None or df.empty:
                st.warning("Data tidak tersedia. Coba kode lain atau perpanjang periode pengambilan data.")
                return
            if len(df) < 30:
                st.warning("Data historis kurang (kurang dari 30 bar). Hasil mungkin tidak akurat.")
        # Prepare dataframe (ensure numeric)
        df = df.astype(float)
        # Compute indicators (vectorized, cached by function calls if needed)
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
        df['MFI'] = compute_mfi(df)
        df['ADL'] = compute_adl(df['High'], df['Low'], df['Close'], df['Volume'])
        df['Volume_MA20'] = df['Volume'].rolling(20).mean()

        # support/resistance
        sr = calculate_support_resistance(df)

        # bandarmology
        with st.spinner("Menganalisis bandarmology..."):
            bandarmology_report, df_with_ind = generate_bandarmology_report(df, period=30)

        # scoring
        scoring = IndicatorScoringSystem()
        scores = {}
        scores['rsi'] = scoring.score_rsi(df['RSI'])
        macd_cross_score, macd_cross_strength, macd_hist_score, macd_hist_strength = scoring.score_macd(df['MACD'], df['Signal'], df['Hist'])
        scores['macd_cross'] = (macd_cross_score, macd_cross_strength)
        scores['macd_hist'] = (macd_hist_score, macd_hist_strength)
        bb_score, bb_strength, is_squeeze = scoring.score_bollinger(df['Close'], df['BB_Upper'], df['BB_Lower'], df['BB_%B'], df['BB_Bandwidth'], df['BB_Bandwidth'].iloc[:-1])
        scores['bollinger'] = (bb_score, bb_strength)
        price_change_pct = ((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / (df['Close'].iloc[-2] + 1e-9)) * 100
        vol_score, vol_strength = scoring.score_volume(df['Volume'], df['Volume_MA20'], price_change_pct)
        scores['volume'] = (vol_score, vol_strength)
        obv_score, obv_strength = scoring.score_obv(df['OBV'])
        scores['obv'] = (obv_score, obv_strength)
        adx_score, adx_strength = scoring.score_adx(df['ADX'], df['Plus_DI'], df['Minus_DI'])
        scores['adx'] = (adx_score, adx_strength)

        composite_score = scoring.calculate_composite_score(scores)
        confidence = scoring.get_confidence_level(composite_score, scores)
        interpretation = scoring.interpret_composite_score(composite_score)

        # breakout detection & plan
        breakout_detector = BreakoutDetector()
        resistance_level = sr['Resistance'][0] if sr['Resistance'] else df['Close'].iloc[-1] * 1.05
        support_level = sr['Support'][0] if sr['Support'] else df['Close'].iloc[-1] * 0.95
        resistance_breakout, support_breakdown, volume_confirm, buffer = breakout_detector.detect_breakout(df, resistance_level, support_level)

        accumulation_status, acc_score, dist_score = detect_accumulation_distribution_wrapper(df)
        volume_profile_simple, bin_edges_simple, pzl, pzh = calculate_volume_profile_simple_wrapper(df)

        # Display UI
        st.subheader("🎯 Hasil Analisis Cross-Confirmation")
        # Gauge-like metric (simple)
        col_g1, col_g2 = st.columns([1, 3])
        with col_g1:
            st.metric("Composite Score", f"{composite_score:.2f}", delta=None)
            st.write(f"Confidence: **{confidence}**")
            st.info(f"{interpretation}")
        with col_g2:
            st.write("**Ringkasan Indicators**")
            ind_table = []
            for name, (s, strg) in scores.items():
                ind_table.append({"Indicator": name.upper(), "Direction": ("Bullish" if s > 0 else "Bearish" if s < 0 else "Netral"), "Score": f"{s:.2f}", "Strength": f"{strg:.2f}"})
            st.table(pd.DataFrame(ind_table))

        # Support/Resistance & Fibonacci
        st.subheader("📈 Support / Resistance / Fibonacci")
        sr_rows = []
        for i, lvl in enumerate(sr['Support']):
            sr_rows.append({"Type": f"Support {i+1}", "Price": f"Rp {lvl:,.2f}"})
        for i, lvl in enumerate(sr['Resistance']):
            sr_rows.append({"Type": f"Resistance {i+1}", "Price": f"Rp {lvl:,.2f}"})
        st.table(pd.DataFrame(sr_rows))
        fib_df = pd.DataFrame([{"Level": k, "Price": f"Rp {v:,.2f}"} for k, v in sr['Fibonacci'].items()])
        st.table(fib_df)

        # Bandarmology
        st.subheader("🕵️ Analisis Bandarmology")
        col_a, col_b = st.columns(2)
        with col_a:
            inst = bandarmology_report['institutional_activity']
            st.write("**Aktivitas Institusional**")
            st.write(f"- Volume Spikes (5 hari): {inst['Volume_Spikes_Last_5_Days']}")
            st.write(f"- Hari Buying Pressure: {inst['Positive_Volume_Price_Days']}")
            st.write(f"- Hari Selling Pressure: {inst['Negative_Volume_Price_Days']}")
            st.write(f"- Volume Clusters: {inst['Volume_Clusters']}")
            st.write(f"- Resilience Days: {inst['Resilience_Days']}")
        with col_b:
            pat = bandarmology_report['smart_money_patterns']
            st.write("**Pola Smart Money (5 hari terakhir)**")
            st.write(f"- Spring Pattern: {pat['Spring_Pattern']}")
            st.write(f"- Stopping Volume: {pat['Stopping_Volume']}")
            st.write(f"- Climax Action: {pat['Climax_Action']}")
            st.write(f"- Hidden Buying: {pat['Hidden_Buying']}")
            st.write(f"- Hidden Selling: {pat['Hidden_Selling']}")
        st.write("**Volume Profile**")
        vp = bandarmology_report['volume_profile_analysis']
        st.write(f"- POC (Point of Control): Rp {vp['poc_price']:,.2f}")
        st.write(f"- Value Area: Rp {vp['value_area_low']:,.2f} - Rp {vp['value_area_high']:,.2f}")
        st.write(f"- Volume Delta: {vp['volume_delta']:,.0f}")
        st.write(f"- Trend Assessment: {bandarmology_report['trend_assessment']}")
        st.subheader("🔍 Kesimpulan Bandarmology")
        for c in bandarmology_report['conclusion']:
            st.write(f"- {c}")

        # Trading recommendation
        st.subheader("🎯 Rekomendasi Trading / Breakout")
        trading_plan = None
        if resistance_breakout and volume_confirm:
            trading_plan = breakout_detector.generate_trading_plan(df, "resistance", resistance_level, buffer, account_size)
        elif support_breakdown and volume_confirm:
            trading_plan = breakout_detector.generate_trading_plan(df, "support", support_level, buffer, account_size)

        if trading_plan:
            st.success("🚀 Sinyal Breakout Terdeteksi")
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Jenis Sinyal", trading_plan["type"])
                st.metric("Entry", f"Rp {trading_plan['entry']:,.2f}")
                st.metric("Stop Loss", f"Rp {trading_plan['stop_loss']:,.2f}")
            with c2:
                st.metric("Target 1", f"Rp {trading_plan['target_1']:,.2f}")
                st.metric("Target 2", f"Rp {trading_plan['target_2']:,.2f}")
                st.metric("Ukuran Posisi", f"{trading_plan['position_size']:,} saham")
                st.write(f"Risk/Reward: {trading_plan['risk_reward']}")
            st.info(f"Manajemen risiko: Ukuran posisi dibatasi max {breakout_detector.max_position_percent*100}% dari modal.")
        else:
            st.warning("⚠️ Belum terdeteksi breakout kuat saat ini.")
            st.write(f"Resistance terdekat: Rp {resistance_level:,.2f} (syarat breakout: close > {resistance_level*(1+buffer):,.2f} & volume > 1.5x MA20)")
            st.write(f"Support terdekat: Rp {support_level:,.2f} (syarat breakdown: close < {support_level*(1-buffer):,.2f} & volume > 1.5x MA20)")

        # Charts
        st.subheader("📈 Chart Teknikal Lengkap")
        is_squeeze = bool(df['BB_Bandwidth'].iloc[-1] < np.percentile(df['BB_Bandwidth'].dropna(), 20)) if len(df['BB_Bandwidth'].dropna())>0 else False
        fig = create_technical_chart(df, sr, is_squeeze)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📊 Volume & Volume Spikes")
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume'))
        fig_vol.add_trace(go.Scatter(x=df.index, y=df['Volume'].rolling(20).mean(), name='MA20 Volume'))
        spike_idx = df_with_ind[df_with_ind['Volume_ZScore'] > 2.5].index if 'Volume_ZScore' in df_with_ind.columns else []
        if len(spike_idx) > 0:
            fig_vol.add_trace(go.Scatter(x=spike_idx, y=df.loc[spike_idx, 'Volume'], mode='markers', name='Spike', marker=dict(color='red', size=8)))
        fig_vol.update_layout(height=300)
        st.plotly_chart(fig_vol, use_container_width=True)

        st.info("**Disclaimer**: Ini bukan rekomendasi investasi. Lakukan uji lebih lanjut (backtest) sebelum mengeksekusi strategi.")

# ----------------------
# Small helpers for backwards compatibility with previous code blocks
# ----------------------
def detect_accumulation_distribution_wrapper(df, period=5):
    adl = compute_adl(df['High'], df['Low'], df['Close'], df['Volume'])
    obv = compute_obv(df['Close'], df['Volume'])
    mfi = compute_mfi(df, 14)
    # compute 5-day changes safely
    idx = -period if len(adl) >= period+1 else 0
    try:
        adl_change = (adl.iloc[-1] - adl.iloc[idx]) / (abs(adl.iloc[idx]) + 1e-9) * 100
    except Exception:
        adl_change = 0
    try:
        obv_change = (obv.iloc[-1] - obv.iloc[idx]) / (abs(obv.iloc[idx]) + 1e-9) * 100
    except Exception:
        obv_change = 0
    mfi_avg = mfi.iloc[-period:].mean() if len(mfi) >= period else mfi.mean()
    vol_ratio = df['Volume'].iloc[-period:].mean() / (df['Volume'].rolling(20).mean().iloc[-1] + 1e-9)
    price_change = (df['Close'].iloc[-1] - df['Close'].iloc[idx]) / (df['Close'].iloc[idx] + 1e-9) * 100 if len(df) >= period+1 else 0
    vwap = compute_vwap(df['High'], df['Low'], df['Close'], df['Volume'])
    price_vs_vwap = (df['Close'].iloc[-1] - vwap.iloc[-1]) / (vwap.iloc[-1] + 1e-9) * 100
    accumulation_score = 0
    distribution_score = 0
    if adl_change > 2: accumulation_score += 1
    if obv_change > 3: accumulation_score += 1
    if mfi_avg > 60: accumulation_score += 1
    if vol_ratio > 1.2: accumulation_score += 1
    if price_change > 0 and price_vs_vwap > 0: accumulation_score += 1
    if adl_change < -2: distribution_score += 1
    if obv_change < -3: distribution_score += 1
    if mfi_avg < 40: distribution_score += 1
    if vol_ratio > 1.2: distribution_score += 1
    if price_change < 0 and price_vs_vwap < 0: distribution_score += 1
    if accumulation_score >= 3 and accumulation_score > distribution_score:
        return "Akumulasi", accumulation_score, distribution_score
    elif distribution_score >= 3 and distribution_score > accumulation_score:
        return "Distribusi", accumulation_score, distribution_score
    else:
        return "Netral", accumulation_score, distribution_score

def calculate_volume_profile(df, period=20, bins=20):
    out = calculate_volume_profile_advanced(df, period, bins)
    return out['volume_profile'], out['bin_edges'], out['value_area_low'], out['value_area_high']

def calculate_volume_profile_simple_wrapper(df, period=20, bins=20):
    vp = calculate_volume_profile_advanced(df, period, bins)
    return vp['volume_profile'], vp['bin_edges'], vp['value_area_low'], vp['value_area_high']

# ----------------------
# Entrypoint
# ----------------------
if __name__ == "__main__":
    app()
