import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import plotly.graph_objects as go
from scipy.signal import argrelextrema

# 1. PERBAIKAN MFI
def compute_mfi(df, period=14):
    """
    Menghitung Money Flow Index (MFI) dengan metode yang benar
    """
    # Hitung Typical Price
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = tp * df['Volume']
    
    positive_flow = [0]  # Hari pertama tidak ada perbandingan
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
    
    # Konversi ke pandas Series untuk perhitungan rolling
    pos_series = pd.Series(positive_flow)
    neg_series = pd.Series(negative_flow)
    
    # Rolling sum dengan periode tertentu
    pos_mf = pos_series.rolling(window=period, min_periods=1).sum()
    neg_mf = neg_series.rolling(window=period, min_periods=1).sum()
    
    # Hitung MFI dengan penanganan pembagian nol
    ratio = np.where(neg_mf > 0, pos_mf / neg_mf, 1.0)
    mfi = 100 - (100 / (1 + ratio))
    
    return pd.Series(mfi, index=df.index)

def interpret_mfi(mfi_value):
    """Memberikan interpretasi sinyal MFI untuk trading"""
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

# 2. PERBAIKAN SUPPORT/RESISTANCE DAN FIBONACCI
def identify_significant_swings(df, window=60, min_swing_size=0.05):
    """
    Mengidentifikasi swing high dan swing low yang signifikan
    """
    # 1. Identifikasi swing points awal
    highs = df['High']
    lows = df['Low']
    
    # Temukan maxima dan minima lokal
    max_idx = argrelextrema(highs.values, np.greater, order=5)[0]
    min_idx = argrelextrema(lows.values, np.less, order=5)[0]
    
    # Ambil swing points dalam window terbaru
    recent_highs = highs.iloc[max_idx][-10:] if len(max_idx) > 0 else pd.Series()
    recent_lows = lows.iloc[min_idx][-10:] if len(min_idx) > 0 else pd.Series()
    
    if len(recent_highs) == 0 or len(recent_lows) == 0:
        # Fallback jika tidak ditemukan swing points
        return df['High'].max(), df['Low'].min()
    
    # 2. Filter berdasarkan signifikansi pergerakan
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
    
    # 3. Pilih swing tertinggi dan terendah yang signifikan
    swing_high = max(significant_highs) if significant_highs else recent_highs.max()
    swing_low = min(significant_lows) if significant_lows else recent_lows.min()
    
    return swing_high, swing_low

def calculate_fibonacci_levels(swing_high, swing_low):
    """Menghitung level Fibonacci retracement yang akurat"""
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
    """Menghitung Volume Weighted Average Price (VWAP)"""
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    vwap = (tp * df['Volume']).cumsum() / df['Volume'].cumsum()
    return vwap

def find_psychological_levels(close_price):
    """Menemukan level psikologis terdekat"""
    levels = [50, 100, 200, 500, 1000, 2000, 5000]
    closest_level = min(levels, key=lambda x: abs(x - close_price))
    return closest_level

def calculate_support_resistance(data):
    """
    Menghitung level support dan resistance dengan berbagai metode
    """
    df = data.copy()
    current_price = df['Close'].iloc[-1]
    
    # 1. Swing Points dan Fibonacci
    swing_high, swing_low = identify_significant_swings(df.tail(60))
    fib_levels = calculate_fibonacci_levels(swing_high, swing_low)
    
    # 2. Moving Averages
    ma20 = df['Close'].rolling(20).mean().iloc[-1]
    ma50 = df['Close'].rolling(50).mean().iloc[-1]
    
    # 3. VWAP
    vwap = calculate_vwap(df).iloc[-1]
    
    # 4. Psychological Levels
    psych_level = find_psychological_levels(current_price)
    
    # Gabungkan semua level
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
    
    # Filter nilai valid dan ambil yang paling signifikan
    valid_support = [lvl for lvl in support_levels if not np.isnan(lvl) and lvl < current_price]
    valid_resistance = [lvl for lvl in resistance_levels if not np.isnan(lvl) and lvl > current_price]
    
    # Urutkan dan ambil 3 level terdekat
    valid_support.sort(reverse=True)
    valid_resistance.sort()
    
    return {
        'Support': valid_support[:3] if valid_support else [],
        'Resistance': valid_resistance[:3] if valid_resistance else [],
        'Fibonacci': fib_levels
    }

# 3. FUNGSI UTAMA DAN FUNGSI PENDUKUNG
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
    # Kriteria lebih fleksibel
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
    """Menghitung Relative Strength Index (RSI)"""
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
    
    # Moving Averages
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
    
    # Indikator Momentum
    df['RSI'] = compute_rsi(df['Close'])
    df['MFI'] = compute_mfi(df, 14)
    
    # Volume Analysis
    df['Avg_Volume_20'] = df['Volume'].rolling(window=20).mean()
    vol_anomali = (df['Volume'].iloc[-1] > 1.7 * df['Avg_Volume_20'].iloc[-1]) if not df['Avg_Volume_20'].isna().iloc[-1] else False
    
    # Support & Resistance
    sr_levels = calculate_support_resistance(df)
    
    # Interpretasi Indikator
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
        "Volume_Anomali": "🚨 Ya" if vol_anomali else "-",
        "Support": sr_levels['Support'],
        "Resistance": sr_levels['Resistance'],
        "Fibonacci": sr_levels['Fibonacci']
    }

def main():
    pass  # Fungsi main tidak digunakan karena menggunakan sistem multi-page

if __name__ == "__main__":
    main()
