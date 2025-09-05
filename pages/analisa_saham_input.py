# pages/analisa_saham_input.py
import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import plotly.graph_objects as go
from scipy.signal import argrelextrema
import time

# --- FUNGSI ANALISIS TEKNIKAL ---
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

def compute_mfi(df, period=14):
    """Menghitung Money Flow Index (MFI)"""
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
    """Interpretasi MFI dengan tingkat kepercayaan"""
    if mfi_value >= 80:
        return "🔴 Overbought (Tekanan Jual Kuat)"
    elif mfi_value >= 65:
        return "🟢 Bullish (Potensi Lanjutkan Kenaikan)"
    elif mfi_value <= 20:
        return "🟢 Oversold (Tekanan Beli Kuat)"
    elif mfi_value <= 35:
        return "🔴 Bearish (Potensi Lanjutkan Penurunan)"
    else:
        return "⚪ Netral (Tidak Ada Tekanan Ekstrem)"

def compute_macd(close, fast=12, slow=26, signal=9):
    """Menghitung Moving Average Convergence Divergence (MACD)"""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def interpret_macd(macd_line, signal_line, hist):
    """Interpretasi MACD dengan konfirmasi histogram"""
    if macd_line.iloc[-1] > signal_line.iloc[-1] and hist.iloc[-1] > 0:
        if hist.iloc[-1] > hist.iloc[-2] and hist.iloc[-2] > hist.iloc[-3]:
            return "🟢 **Bullish Kuat** (MACD > Signal & Histogram membesar)"
        return "🟢 Bullish (MACD > Signal & Histogram positif)"
    elif macd_line.iloc[-1] < signal_line.iloc[-1] and hist.iloc[-1] < 0:
        if hist.iloc[-1] < hist.iloc[-2] and hist.iloc[-2] < hist.iloc[-3]:
            return "🔴 **Bearish Kuat** (MACD < Signal & Histogram membesar negatif)"
        return "🔴 Bearish (MACD < Signal & Histogram negatif)"
    else:
        return "⚪ Netral (tidak ada sinyal kuat)"

def compute_bollinger_bands(close, window=20, num_std=2):
    """Menghitung Bollinger Bands dengan standar deviasi"""
    sma = close.rolling(window=window).mean()
    std = close.rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return upper_band, sma, lower_band

def interpret_bollinger_bands(price, upper_band, middle_band, lower_band, volume, avg_volume, hist):
    """
    Interpretasi Bollinger Bands yang komprehensif dengan:
    - Posisi harga relatif terhadap bands
    - Deteksi Bollinger Squeeze
    - Konfirmasi dengan volume
    - Analisis potensi reversal
    """
    current_upper = upper_band.iloc[-1]
    current_middle = middle_band.iloc[-1]
    current_lower = lower_band.iloc[-1]
    current_price = price.iloc[-1]
    current_volume = volume.iloc[-1]
    avg_vol = avg_volume.iloc[-1]
    
    # Hitung bandwidth untuk deteksi squeeze
    bandwidth = (current_upper - current_lower) / current_middle
    avg_bandwidth = ((upper_band - lower_band) / middle_band).mean()
    
    # Analisis posisi harga
    position = ""
    if current_price > current_upper:
        if current_volume > 1.5 * avg_vol:
            position = "🔴 **Harga di ATAS Upper Band dengan Volume Tinggi** (Overbought Ekstrem)"
        else:
            position = "🟡 **Harga di ATAS Upper Band** (Peringatan Overbought)"
    elif current_price < current_lower:
        if current_volume > 1.5 * avg_vol:
            position = "🟢 **Harga di BAWAH Lower Band dengan Volume Tinggi** (Oversold Ekstrem)"
        else:
            position = "🟡 **Harga di BAWAH Lower Band** (Peringatan Oversold)"
    elif current_price > current_middle:
        position = "🔵 **Harga di ATAS Middle Band** (Tren Bullish Jangka Pendek)"
    else:
        position = "🔵 **Harga di BAWAH Middle Band** (Tren Bearish Jangka Pendek)"
    
    # Deteksi Squeeze
    squeeze = ""
    squeeze_detected = False
    if bandwidth < 0.7 * avg_bandwidth:
        squeeze = "⚠️ **BOLLINGER SQUEEZE TERDETEKSI** - Penyempitan bands menunjukkan volatilitas rendah, potensi breakout besar dalam 1-5 hari"
        squeeze_detected = True
    
    # Analisis Reversal
    reversal = ""
    if current_price > current_upper and hist.iloc[-1] < 0:  # Harga di atas upper band tapi MACD histogram turun
        reversal = "🔄 **Potensi Reversal Bearish** - Harga melampaui upper band dengan konfirmasi MACD"
    elif current_price < current_lower and hist.iloc[-1] > 0:  # Harga di bawah lower band tapi MACD histogram naik
        reversal = "🔄 **Potensi Reversal Bullish** - Harga melampaui lower band dengan konfirmasi MACD"
    
    # Analisis Trend
    trend = ""
    if current_price > current_middle and current_price > current_upper * 0.95:
        trend = "📈 **Tren Bullish Kuat** - Harga mendekati upper band dalam uptrend"
    elif current_price < current_middle and current_price < current_lower * 1.05:
        trend = "📉 **Tekanan Jual Kuat** - Harga mendekati lower band dalam downtrend"
    
    # Analisis bandwidth
    bandwidth_status = ""
    if bandwidth < 0.5 * avg_bandwidth:
        bandwidth_status = "Sangat Rendah (Squeeze Ekstrem)"
    elif bandwidth < 0.7 * avg_bandwidth:
        bandwidth_status = "Rendah (Potensi Breakout)"
    elif bandwidth > 1.3 * avg_bandwidth:
        bandwidth_status = "Tinggi (Volatilitas Tinggi)"
    else:
        bandwidth_status = "Normal"
    
    return {
        'position': position,
        'squeeze': squeeze,
        'reversal': reversal,
        'trend': trend,
        'bandwidth': bandwidth,
        'avg_bandwidth': avg_bandwidth,
        'squeeze_detected': squeeze_detected,
        'bandwidth_status': bandwidth_status
    }

def identify_significant_swings(df, window=60, min_swing_size=0.05):
    """Mengidentifikasi swing signifikan dengan ukuran minimum perubahan"""
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
    """Menghitung level support dan resistance dengan berbagai metode"""
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

def get_market_context():
    """Mengambil konteks pasar secara keseluruhan dari IDX"""
    try:
        idx = yf.Ticker("^JKSE")
        idx_data = idx.history(period="1mo")
        idx_change = ((idx_data['Close'].iloc[-1] - idx_data['Close'].iloc[0]) / idx_data['Close'].iloc[0]) * 100
        volatility = (idx_data['High'] - idx_data['Low']).mean() / idx_data['Close'].mean() * 100
        
        if idx_change > 3:
            return "Bullish", f"Pasar bullish ({idx_change:.1f}%), volatilitas {volatility:.1f}%"
        elif idx_change < -3:
            return "Bearish", f"Pasar bearish ({idx_change:.1f}%), volatilitas {volatility:.1f}%"
        else:
            return "Netral", f"Pasar netral ({idx_change:.1f}%), volatilitas {volatility:.1f}%"
    except:
        return "Tidak Diketahui", "Data pasar tidak tersedia"

def analyze_volume_pattern(df):
    """Analisis pola volume selama beberapa periode"""
    volume_ma = df['Volume'].rolling(5).mean()
    volume_ratio = df['Volume'] / volume_ma
    
    # Hitung rata-rata volume ratio untuk 5 hari terakhir
    recent_vol_ratio = volume_ratio.iloc[-5:].mean()
    
    if recent_vol_ratio > 1.8:
        return "📈 **Volume Meningkat Signifikan** - Minat perdagangan meningkat tajam"
    elif recent_vol_ratio > 1.3:
        return "🔼 Volume Meningkat - Minat perdagangan meningkat"
    elif recent_vol_ratio < 0.7:
        return "🔽 Volume Menurun - Minat perdagangan berkurang"
    else:
        return "⚪ Volume Normal - Minat perdagangan stabil"

def validate_analysis_date(ticker, analysis_date):
    """Validasi apakah tanggal analisis adalah hari perdagangan aktif"""
    try:
        # Cek apakah tanggal analisis adalah hari libur pasar
        if analysis_date.weekday() >= 5:  # Sabtu/Minggu
            st.warning("Tanggal yang dipilih adalah hari libur. Memilih hari perdagangan terakhir sebelumnya.")
            # Cari hari perdagangan terakhir sebelum tanggal ini
            while analysis_date.weekday() >= 5:
                analysis_date -= timedelta(days=1)
            return analysis_date
        
        # Cek apakah ada data untuk tanggal ini
        stock = yf.Ticker(f"{ticker}.JK")
        test_data = stock.history(start=analysis_date, end=analysis_date + timedelta(days=1))
        if test_data.empty:
            st.warning("Tanggal yang dipilih mungkin hari libur pasar. Memilih hari perdagangan terakhir sebelumnya.")
            # Cari hari perdagangan terakhir sebelum tanggal ini
            prev_date = analysis_date
            while prev_date > (analysis_date - timedelta(days=7)):
                prev_date -= timedelta(days=1)
                test_data = stock.history(start=prev_date, end=prev_date + timedelta(days=1))
                if not test_data.empty:
                    return prev_date
            return analysis_date - timedelta(days=1)  # Default fallback
        
        return analysis_date
    except:
        return analysis_date

def get_stock_data(ticker, end_date):
    """Mengambil data saham dari Yahoo Finance"""
    try:
        stock = yf.Ticker(f"{ticker}.JK")
        start_date = end_date - timedelta(days=120)  # Ambil lebih banyak data untuk analisis yang lebih baik
        data = stock.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
        
        if data.empty:
            # Coba lagi dengan tanggal yang lebih awal jika diperlukan
            start_date = end_date - timedelta(days=180)
            data = stock.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
        
        return data if not data.empty else None
    except Exception as e:
        st.error(f"Gagal mengambil data untuk {ticker}: {e}")
        return None

def generate_conclusion(df, mfi_value, macd_signal, bb_analysis, vol_anomali, price_change, market_trend):
    """
    Sistem kesimpulan yang lebih cerdas dengan bobot dan konteks:
    - Analisis RSI dengan konteks pasar
    - Analisis MFI dengan konteks volume
    - Analisis MACD dengan konfirmasi histogram
    - Analisis Bollinger Bands yang lebih mendalam
    - Analisis Volume yang lebih detail
    - Rekomendasi Berbobot dengan Analisis Konvergensi
    """
    conclusions = []
    
    # 1. Konteks Pasar
    conclusions.append(f"🌐 **Konteks Pasar**: {market_trend[1]}")
    
    # 2. Analisis RSI dengan konteks pasar
    rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns and not np.isnan(df['RSI'].iloc[-1]) else None
    if rsi is not None:
        if rsi > 75:
            conclusions.append("🔴 **RSI Ekstrem (80+)** - Harga sangat overbought, koreksi signifikan mungkin terjadi")
        elif rsi > 70:
            conclusions.append("🟡 RSI Tinggi (70-75) - Potensi overbought, waspadai koreksi")
        elif rsi < 25:
            conclusions.append("🟢 **RSI Ekstrem (20-)** - Harga sangat oversold, rebound kuat mungkin terjadi")
        elif rsi < 30:
            conclusions.append("🟡 RSI Rendah (25-30) - Potensi oversold, waspadai rebound")
        else:
            conclusions.append("⚪ RSI Netral - Tidak menunjukkan kondisi overbought/oversold ekstrem")
    
    # 3. Analisis MFI dengan konteks volume
    if not np.isnan(mfi_value):
        if mfi_value >= 80:
            conclusions.append("🔴 **MFI Sangat Tinggi** - Tekanan jual kuat terkonfirmasi volume")
        elif mfi_value >= 65:
            conclusions.append("🟡 MFI Tinggi - Potensi tekanan jual")
        elif mfi_value <= 20:
            conclusions.append("🟢 **MFI Sangat Rendah** - Tekanan beli kuat terkonfirmasi volume")
        elif mfi_value <= 35:
            conclusions.append("🟡 MFI Rendah - Potensi tekanan beli")
        else:
            conclusions.append("⚪ MFI Netral - Tidak ada tekanan ekstrem")
    
    # 4. Analisis MACD dengan konfirmasi histogram
    conclusions.append(f"📊 **MACD**: {macd_signal}")
    
    # 5. Analisis Bollinger Bands yang lebih mendalam
    if bb_analysis['position']:
        conclusions.append(f"🔵 **Bollinger Bands**: {bb_analysis['position']}")
    if bb_analysis['squeeze']:
        conclusions.append(f"🔵 **Bollinger Squeeze**: {bb_analysis['squeeze']}")
    if bb_analysis['reversal']:
        conclusions.append(f"🔵 **Sinyal Reversal**: {bb_analysis['reversal']}")
    if bb_analysis['trend']:
        conclusions.append(f"🔵 **Tren Bollinger**: {bb_analysis['trend']}")
    
    # 6. Analisis Volume yang lebih detail
    if vol_anomali:
        if df['Close'].iloc[-1] > df['Open'].iloc[-1]:  # Harga naik
            conclusions.append("💹 **Volume Tinggi dengan Harga Naik** - Konfirmasi kuat untuk kelanjutan tren bullish")
        else:  # Harga turun
            conclusions.append("📉 **Volume Tinggi dengan Harga Turun** - Konfirmasi kuat untuk kelanjutan tren bearish")
    
    # 7. Analisis Perubahan Harga dengan Konteks Volatilitas
    if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
        volatility = df['BB_Upper'].iloc[-1] - df['BB_Lower'].iloc[-1]
        price_range = df['High'].max() - df['Low'].min()
        volatility_pct = (volatility / price_range) * 100 if price_range > 0 else 0
        
        if abs(price_change) > 2:
            if volatility_pct < 30:  # Volatilitas rendah
                conclusions.append(f"⚡ **Perubahan Harga Signifikan ({price_change:.2f}%)** - Pergerakan besar dalam kondisi volatilitas rendah, sinyal kuat")
            else:
                conclusions.append(f"📈 **Perubahan Harga ({price_change:.2f}%)** - Sesuai dengan tingkat volatilitas saham")
    
    # 8. Rekomendasi Berbobot dengan Analisis Konvergensi
    bullish_score = 0
    bearish_score = 0
    
    # Bobot berdasarkan kekuatan sinyal
    if rsi is not None:
        if rsi < 30: 
            bullish_score += 3 if rsi < 25 else 2
        elif rsi > 70: 
            bearish_score += 3 if rsi > 75 else 2
    
    if not np.isnan(mfi_value):
        if mfi_value <= 20: 
            bullish_score += 2
        elif mfi_value >= 80: 
            bearish_score += 2
    
    if "Bullish" in macd_signal: 
        bullish_score += 2
    elif "Bearish" in macd_signal: 
        bearish_score += 2
    
    if "Bullish" in bb_analysis['trend'] or "Reversal Bullish" in bb_analysis['reversal']:
        bullish_score += 2
    elif "Bearish" in bb_analysis['trend'] or "Reversal Bearish" in bb_analysis['reversal']:
        bearish_score += 2
    
    if vol_anomali and df['Close'].iloc[-1] > df['Open'].iloc[-1]:
        bullish_score += 1
    elif vol_anomali and df['Close'].iloc[-1] < df['Open'].iloc[-1]:
        bearish_score += 1
    
    # Rekomendasi berdasarkan skor
    if bullish_score >= 6:
        conclusions.append("**✅ REKOMENDASI: BULLISH KUAT** - Pertimbangkan untuk akumulasi atau entry long dengan manajemen risiko")
    elif bearish_score >= 6:
        conclusions.append("**❌ REKOMENDASI: BEARISH KUAT** - Pertimbangkan untuk take profit atau entry short dengan manajemen risiko")
    elif bullish_score >= bearish_score + 2:
        conclusions.append("**🔼 REKOMENDASI: BIAS BULLISH** - Potensi kenaikan lebih lanjut, pertimbangkan untuk hold atau akumulasi bertahap")
    elif bearish_score >= bullish_score + 2:
        conclusions.append("**🔽 REKOMENDASI: BIAS BEARISH** - Potensi penurunan lebih lanjut, pertimbangkan untuk wait and see atau take profit")
    else:
        conclusions.append("**⏺️ REKOMENDASI: NETRAL** - Tunggu konfirmasi lebih lanjut sebelum mengambil posisi")
    
    # Tambahkan rekomendasi manajemen risiko
    if bb_analysis['squeeze_detected']:
        risk_level = "TINGGI"
        risk_recommendation = "Gunakan stop loss lebih ketat (1-2%) karena potensi breakout besar"
    elif bullish_score >= 6 or bearish_score >= 6:
        risk_level = "SEDANG"
        risk_recommendation = "Gunakan stop loss standar (2-3%)"
    else:
        risk_level = "RENDAH"
        risk_recommendation = "Stop loss bisa lebih longgar (3-5%)"
    
    conclusions.append(f"**⚠️ TINGKAT RISIKO: {risk_level}** - {risk_recommendation}")
    
    return conclusions

# --- FUNGSI UTAMA ---
def app():
    st.title("📈 Analisa Teknikal Saham Lengkap")
    
    # Informasi di sidebar
    with st.sidebar:
        st.header("ℹ️ Informasi Aplikasi")
        st.markdown("""
        Aplikasi ini menyediakan analisis teknikal saham secara komprehensif dengan:
        - Analisis Bollinger Bands mendalam
        - Deteksi Bollinger Squeeze
        - Analisis konvergensi multi-indikator
        - Rekomendasi berbasis scoring
        - Konteks pasar secara keseluruhan
        """)
        st.markdown("---")
        st.caption("© 2023 Analisa Saham Indonesia")
    
    ticker_input = st.text_input("Masukkan Kode Saham (contoh: BBCA)", value="BBCA")
    analysis_date = st.date_input("📅 Tanggal Analisis", value=datetime.today())
    
    # Validasi tanggal analisis
    analysis_date = validate_analysis_date(ticker_input, analysis_date)
    
    if st.button("🔍 Mulai Analisis"):
        # Tampilkan spinner selama proses analisis
        with st.spinner('Sedang mengambil dan menganalisis data...'):
            time.sleep(1)  # Beri waktu untuk spinner terlihat
            
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

            # Ambil konteks pasar
            market_trend = get_market_context()
            
            # Hitung indikator
            df = data.copy()
            df['MA20'] = df['Close'].rolling(20).mean()
            df['MA50'] = df['Close'].rolling(50).mean()
            df['RSI'] = compute_rsi(df['Close'])
            df['MFI'] = compute_mfi(df, 14)
            df['MACD'], df['Signal'], df['Hist'] = compute_macd(df['Close'])
            
            # Bollinger Bands
            df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = compute_bollinger_bands(df['Close'])
            df['Avg_Volume_20'] = df['Volume'].rolling(window=20).mean()
            
            # Analisis Bollinger Bands
            bb_analysis = interpret_bollinger_bands(
                df['Close'], 
                df['BB_Upper'], 
                df['BB_Middle'], 
                df['BB_Lower'],
                df['Volume'],
                df['Avg_Volume_20'],
                df['Hist']
            )
            
            # Hitung level support/resistance
            sr = calculate_support_resistance(df)
            fib = sr['Fibonacci']
            
            # Ambil nilai terkini
            mfi_value = df['MFI'].iloc[-1] if not df['MFI'].empty and not np.isnan(df['MFI'].iloc[-1]) else np.nan
            mfi_signal = interpret_mfi(mfi_value) if not np.isnan(mfi_value) else "N/A"
            macd_signal = interpret_macd(df['MACD'], df['Signal'], df['Hist'])
            
            # Volume Anomali
            vol_anomali = (df['Volume'].iloc[-1] > 1.7 * df['Avg_Volume_20'].iloc[-1]) if not df['Avg_Volume_20'].isna().iloc[-1] else False
            
            # Harga sebelumnya
            previous_close = df['Close'].iloc[-2] if len(df) > 1 else df['Close'].iloc[-1]
            price_change = ((df['Close'].iloc[-1] - previous_close) / previous_close) * 100
            
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
            fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], mode='lines', name='MA20', line=dict(color='blue', width=1.5)))
            fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], mode='lines', name='MA50', line=dict(color='orange', width=1.5)))
            
            # Bollinger Bands dengan visualisasi lebih informatif
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], mode='lines', name='BB Upper', 
                                    line=dict(color='red', width=1, dash='dot')))
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Middle'], mode='lines', name='BB Middle', 
                                    line=dict(color='purple', width=1.5)))
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], mode='lines', name='BB Lower', 
                                    line=dict(color='green', width=1, dash='dot'),
                                    fill='tonexty', fillcolor='rgba(173, 216, 230, 0.1)'))

            # Tambahkan zona overbought/oversold
            fig.add_trace(go.Scatter(
                x=df.index, 
                y=np.where(df['Close'] > df['BB_Upper'], df['Close'], np.nan),
                mode='markers',
                name='Overbought',
                marker=dict(color='rgba(255, 0, 0, 0.6)', size=8, symbol='circle')
            ))
            fig.add_trace(go.Scatter(
                x=df.index, 
                y=np.where(df['Close'] < df['BB_Lower'], df['Close'], np.nan),
                mode='markers',
                name='Oversold',
                marker=dict(color='rgba(0, 255, 0, 0.6)', size=8, symbol='circle')
            ))
            
            # Support & Resistance
            for i, level in enumerate(sr['Support']):
                fig.add_hline(y=level, line_dash="dash", line_color="green",
                              annotation_text=f"Support {i+1}: Rp {level:,.2f}", 
                              annotation_position="bottom right")
            for i, level in enumerate(sr['Resistance']):
                fig.add_hline(y=level, line_dash="dash", line_color="red",
                              annotation_text=f"Resistance {i+1}: Rp {level:,.2f}", 
                              annotation_position="top right")

            # Fibonacci
            fib_keys = ['Fib_0.0', 'Fib_0.236', 'Fib_0.382', 'Fib_0.5', 'Fib_0.618', 'Fib_0.786', 'Fib_1.0']
            for key in fib_keys:
                if key in fib and not np.isnan(fib[key]):
                    color = "magenta" if key == 'Fib_0.0' else "blue" if key == 'Fib_1.0' else "purple"
                    dash = "solid" if key in ['Fib_0.0', 'Fib_1.0'] else "dot"
                    position = "top left" if key in ['Fib_0.0', 'Fib_0.236', 'Fib_0.382'] else "bottom left"
                    position = "bottom left" if key == 'Fib_1.0' else position
                    fig.add_hline(y=fib[key], line_dash=dash, line_color=color,
                                  annotation_text=f"{key.replace('Fib_', 'Fib ')}: Rp {fib[key]:,.2f}", 
                                  annotation_position=position)

            # Tambahkan indikator Squeeze jika terdeteksi
            if bb_analysis['squeeze_detected']:
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
                title=f"{ticker} - Analisa Teknikal Lengkap",
                xaxis_title="Tanggal",
                yaxis_title="Harga (Rp)",
                xaxis_rangeslider_visible=False,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                template="plotly_white",
                height=700,
                hovermode='x unified'
            )

            st.plotly_chart(fig, use_container_width=True)

            # --- INFORMASI HARGA ---
            st.subheader("💰 Informasi Harga Terkini")
            col_prev, col_curr, col_change, col_vol = st.columns(4)
            
            with col_prev:
                st.metric("Harga Penutupan Hari Sebelumnya", f"Rp {previous_close:,.2f}")
            with col_curr:
                st.metric("Harga Penutupan Hari Ini", f"Rp {df['Close'].iloc[-1]:,.2f}")
            with col_change:
                st.metric("Perubahan Harga", 
                         f"{price_change:.2f}%",
                         delta=f"{price_change:.2f}%",
                         delta_color="inverse" if price_change < 0 else "normal")
            with col_vol:
                vol_text = f"{df['Volume'].iloc[-1]:,.0f}"
                st.metric("Volume Perdagangan", vol_text)
            
            # --- INDIKATOR TEKNIKAL ---
            st.subheader("📊 Indikator Teknikal")
            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.metric("MA20", f"Rp {df['MA20'].iloc[-1]:,.2f}" if not np.isnan(df['MA20'].iloc[-1]) else "N/A")
                st.metric("MA50", f"Rp {df['MA50'].iloc[-1]:,.2f}" if not np.isnan(df['MA50'].iloc[-1]) else "N/A")

            with col2:
                rsi_value = df['RSI'].iloc[-1] if not df['RSI'].empty and not np.isnan(df['RSI'].iloc[-1]) else "N/A"
                st.metric("RSI", f"{rsi_value:.2f}" if rsi_value != "N/A" else "N/A")
                st.metric("MFI", f"{mfi_value:.2f}" if not np.isnan(mfi_value) else "N/A", mfi_signal)

            with col3:
                st.metric("MACD", f"{df['MACD'].iloc[-1]:.4f}")
                st.metric("Signal", f"{df['Signal'].iloc[-1]:.4f}")

            with col4:
                st.metric("Histogram", f"{df['Hist'].iloc[-1]:.4f}")
                st.metric("Volume Anomali", "Ya" if vol_anomali else "Tidak")
            
            with col5:
                st.metric("Bandwidth Status", bb_analysis['bandwidth_status'])
                if bb_analysis['squeeze_detected']:
                    st.metric("Kondisi", "Squeeze Terdeteksi")

            # --- LEVEL PENTING ---
            st.subheader("📍 Level Penting")
            col_sup, col_res = st.columns(2)
            
            with col_sup:
                if sr['Support']:
                    st.markdown("**Support Terkuat:**")
                    for i, s in enumerate(sr['Support'][:3]):
                        st.markdown(f"{i+1}. Rp {s:,.2f}")
                else:
                    st.markdown("Tidak ada level support yang teridentifikasi")
            
            with col_res:
                if sr['Resistance']:
                    st.markdown("**Resistance Terkuat:**")
                    for i, r in enumerate(sr['Resistance'][:3]):
                        st.markdown(f"{i+1}. Rp {r:,.2f}")
                else:
                    st.markdown("Tidak ada level resistance yang teridentifikasi")
            # --- LEVEL FIBONACCI ---
            st.markdown('<div style="font-size: 0.85em;">', unsafe_allow_html=True)
            st.subheader("🔢 Level Fibonacci")
            fib_display = {k: v for k, v in fib.items() if k in fib_keys}
            if fib_display:
                cols = st.columns(len(fib_display))
                for i, (key, value) in enumerate(fib_display.items()):
                    cols[i].metric(
                        key.replace('Fib_', 'Fib ').replace('_', ' '), 
                        f"{value:,.2f}",
                        help=f"Level {key.replace('Fib_', '')}"
                   )
            st.markdown('</div>', unsafe_allow_html=True)
       
            # --- ANALISIS BOLLINGER BANDS ---
            st.subheader("📊 Analisis Bollinger Bands")
            
            # Tampilkan hasil analisis
            if bb_analysis['position']:
                st.markdown(f"**Posisi Harga:** {bb_analysis['position']}")
            if bb_analysis['trend']:
                st.markdown(f"**Analisis Tren:** {bb_analysis['trend']}")
            if bb_analysis['squeeze']:
                st.markdown(f"**Kondisi Bandwidth:** {bb_analysis['squeeze']}")
            if bb_analysis['reversal']:
                st.markdown(f"**Sinyal Reversal:** {bb_analysis['reversal']}")
            
            # Tampilkan nilai Bollinger Bands
            col_bb1, col_bb2, col_bb3 = st.columns(3)
            with col_bb1:
                st.metric("Upper Band", f"Rp {df['BB_Upper'].iloc[-1]:,.2f}")
            with col_bb2:
                st.metric("Middle Band (MA20)", f"Rp {df['BB_Middle'].iloc[-1]:,.2f}")
            with col_bb3:
                st.metric("Lower Band", f"Rp {df['BB_Lower'].iloc[-1]:,.2f}")
            
            # --- KESIMPULAN LENGKAP ---
            st.subheader("📋 Kesimpulan Analisis Teknikal")
            
            # Generate conclusion dengan semua parameter
            conclusions = generate_conclusion(
                df, 
                mfi_value, 
                macd_signal, 
                bb_analysis, 
                vol_anomali,
                price_change,
                market_trend
            )
            
            # Tampilkan kesimpulan dengan format yang lebih baik
            for i, conclusion in enumerate(conclusions):
                # Highlight rekomendasi utama
                if "REKOMENDASI" in conclusion:
                    st.markdown(f"<div style='background-color: #f0f2f6; padding: 15px; border-radius: 10px; border-left: 4px solid #1f77b4; margin: 10px 0;'>{conclusion}</div>", 
                               unsafe_allow_html=True)
                else:
                    st.markdown(f"- {conclusion}")

            # --- REKOMENDASI TRADING ---
            st.subheader("🎯 Rekomendasi Trading")
            
            # Cari rekomendasi utama dari kesimpulan
            main_recommendation = "Netral"
            risk_level = "Sedang"
            for conclusion in conclusions:
                if "REKOMENDASI" in conclusion:
                    if "BULLISH KUAT" in conclusion:
                        main_recommendation = "Bullish Kuat"
                    elif "BEARISH KUAT" in conclusion:
                        main_recommendation = "Bearish Kuat"
                    elif "BIAS BULLISH" in conclusion:
                        main_recommendation = "Bias Bullish"
                    elif "BIAS BEARISH" in conclusion:
                        main_recommendation = "Bias Bearish"
                if "TINGKAT RISIKO" in conclusion:
                    if "TINGGI" in conclusion:
                        risk_level = "Tinggi"
                    elif "RENDAH" in conclusion:
                        risk_level = "Rendah"
            
            # Tampilkan rekomendasi dengan visual yang jelas
            if "Bullish" in main_recommendation:
                color = "#28a745"
                icon = "📈"
            elif "Bearish" in main_recommendation:
                color = "#dc3545"
                icon = "📉"
            else:
                color = "#6c757d"
                icon = "⏺️"
                
            st.markdown(f"""
            <div style="background-color: {color}22; border-left: 4px solid {color}; padding: 15px; border-radius: 0 8px 8px 0; margin: 15px 0;">
                <h4 style="color: {color}; margin: 0;">{icon} {main_recommendation}</h4>
                <p style="margin: 10px 0 0 0; color: #333;">Tingkat Risiko: <strong>{risk_level}</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Detail rekomendasi berdasarkan arah
            if "Bullish" in main_recommendation:
                st.markdown("### Strategi Bullish")
                st.markdown("""
                - **Entry Point**: Cari konfirmasi pada level support atau breakout resistance
                - **Target Harga**: Resistance terkuat berikutnya atau upper Bollinger Band
                - **Stop Loss**: Di bawah level support terdekat atau lower Bollinger Band
                - **Manajemen Risiko**: Gunakan 1-2% dari modal untuk posisi ini
                """)
            elif "Bearish" in main_recommendation:
                st.markdown("### Strategi Bearish")
                st.markdown("""
                - **Entry Point**: Cari konfirmasi pada level resistance atau breakdown support
                - **Target Harga**: Support terkuat berikutnya atau lower Bollinger Band
                - **Stop Loss**: Di atas level resistance terdekat atau upper Bollinger Band
                - **Manajemen Risiko**: Gunakan 1-2% dari modal untuk posisi ini
                """)
            else:
                st.markdown("### Strategi Netral")
                st.markdown("""
                - **Entry Point**: Tunggu konfirmasi sinyal yang lebih jelas
                - **Breakout Confirmation**: Tunggu penutupan di luar Bollinger Bands dengan volume tinggi
                - **Manajemen Risiko**: Pertahankan posisi kecil atau tidak berposisi
                - **Pantau**: Level support/resistance utama dan indikator momentum
                """)
            
            # Disclaimer
            st.info("⚠️ **Disclaimer**: Analisis ini hanya untuk tujuan edukasi dan bukan sebagai rekomendasi investasi. "
                    "Harga saham bisa berubah sewaktu-waktu. Lakukan riset tambahan dan konsultasi dengan "
                    "penasihat keuangan sebelum mengambil keputusan investasi.")

if __name__ == "__main__":
    app()
