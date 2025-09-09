import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from scipy import stats
from scipy.signal import argrelextrema

# Set page config
st.set_page_config(layout="wide", page_title="Analisa Teknikal Saham", page_icon="📊")

# =========================
# Konstanta
# =========================
INDICATOR_WEIGHTS = {
    'rsi': 0.15,
    'macd_cross': 0.25,
    'macd_hist': 0.10,
    'bollinger': 0.15,
    'volume': 0.20,
    'obv': 0.10,
    'adx': 0.05
}

# =========================
# Fungsi Bantuan Data
# =========================
@st.cache_data(show_spinner=False, ttl=3600)
def get_stock_data_yf(ticker_no_suffix: str, end_date: datetime, days_back=360):
    """Mengambil data saham dari Yahoo Finance"""
    try:
        ticker = yf.Ticker(f"{ticker_no_suffix}.JK")
        start = end_date - timedelta(days=days_back)
        df = ticker.history(start=start.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
        if df.empty:
            # Coba tanpa suffix .JK untuk saham luar
            ticker = yf.Ticker(ticker_no_suffix)
            df = ticker.history(start=start.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
            
        if df.empty:
            return None
            
        # Pastikan kolom ada dan dengan penamaan yang konsisten
        df = df.rename(columns={c: c.title() for c in df.columns})
        return df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    except Exception as e:
        st.error(f"Error mengambil data: {e}")
        return None

# =========================
# Fungsi Indikator Teknikal
# =========================
def compute_rsi(close, period=14):
    """Menghitung Relative Strength Index (RSI)"""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def compute_macd(close, fast=12, slow=26, signal=9):
    """Menghitung Moving Average Convergence Divergence (MACD)"""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def compute_bollinger_bands(close, window=20, num_std=2):
    """Menghitung Bollinger Bands"""
    sma = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
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
    price_change = close.diff()
    obv = (volume * np.sign(price_change)).cumsum()
    return obv.fillna(0)

def compute_adx(high, low, close, period=14):
    """Menghitung Average Directional Index (ADX)"""
    up_move = high.diff()
    down_move = low.diff().apply(lambda x: -x if x > 0 else 0)
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = true_range.rolling(period).mean()
    plus_di = 100 * (pd.Series(plus_dm).rolling(period).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm).rolling(period).mean() / atr)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(period).mean()
    return adx.fillna(0), plus_di.fillna(0), minus_di.fillna(0)

def compute_vwap(high, low, close, volume):
    """Menghitung Volume Weighted Average Price (VWAP)"""
    typical_price = (high + low + close) / 3
    vwap = (typical_price * volume).cumsum() / volume.cumsum()
    return vwap

# =========================
# Sistem Scoring
# =========================
class IndicatorScoringSystem:
    def __init__(self, weights=None):
        self.weights = weights or INDICATOR_WEIGHTS

    def score_rsi(self, rsi_values):
        """Memberikan skor untuk RSI"""
        current_rsi = rsi_values.iloc[-1]
        
        if current_rsi < 30:
            score = 1.0  # Bullish (oversold)
            strength = min(1.0, (30 - current_rsi) / 30)
        elif current_rsi > 70:
            score = -1.0  # Bearish (overbought)
            strength = min(1.0, (current_rsi - 70) / 30)
        elif current_rsi > 50:
            score = 0.5  # Mild bullish
            strength = (current_rsi - 50) / 20
        else:
            score = -0.5  # Mild bearish
            strength = (50 - current_rsi) / 20
            
        return score, strength

    def score_macd(self, macd_line, signal_line, histogram):
        """Memberikan skor untuk MACD"""
        # Skor untuk crossover
        macd_cross_score = 1.0 if macd_line.iloc[-1] > signal_line.iloc[-1] else -1.0
        macd_cross_strength = 1.0
        
        # Skor untuk momentum histogram
        hist_trend = self.calculate_trend(histogram.tail(5))
        if hist_trend > 0:
            hist_score = 0.5
            hist_strength = min(1.0, abs(hist_trend))
        elif hist_trend < 0:
            hist_score = -0.5
            hist_strength = min(1.0, abs(hist_trend))
        else:
            hist_score = 0.0
            hist_strength = 0.0
            
        return macd_cross_score, macd_cross_strength, hist_score, hist_strength

    def score_bollinger(self, close, upper_band, lower_band, percent_b):
        """Memberikan skor untuk Bollinger Bands"""
        current_price = close.iloc[-1]
        current_pct_b = percent_b.iloc[-1]
        
        if current_pct_b < 0.2:
            score = 1.0  # Near lower band (oversold)
            strength = min(1.0, (0.2 - current_pct_b) / 0.2)
        elif current_pct_b > 0.8:
            score = -1.0  # Near upper band (overbought)
            strength = min(1.0, (current_pct_b - 0.8) / 0.2)
        elif current_pct_b > 0.5:
            score = -0.5  # Above middle
            strength = (current_pct_b - 0.5) / 0.3
        else:
            score = 0.5  # Below middle
            strength = (0.5 - current_pct_b) / 0.3
            
        return score, strength

    def score_volume(self, volume, volume_ma, price_change):
        """Memberikan skor untuk Volume"""
        volume_ratio = volume.iloc[-1] / volume_ma.iloc[-1]
        
        if volume_ratio > 1.5 and price_change > 0:
            score = 1.0  # High volume with price up
            strength = min(1.0, (volume_ratio - 1.5) / 0.5)
        elif volume_ratio > 1.5 and price_change < 0:
            score = -1.0  # High volume with price down
            strength = min(1.0, (volume_ratio - 1.5) / 0.5)
        elif volume_ratio > 1.2 and price_change > 0:
            score = 0.5  # Moderate volume with price up
            strength = (volume_ratio - 1.2) / 0.3
        elif volume_ratio > 1.2 and price_change < 0:
            score = -0.5  # Moderate volume with price down
            strength = (volume_ratio - 1.2) / 0.3
        else:
            score = 0.0
            strength = 0.0
            
        return score, strength

    def score_obv(self, obv_values):
        """Memberikan skor untuk OBV"""
        obv_trend = self.calculate_trend(obv_values.tail(5))
        
        if obv_trend > 0.05:
            score = 1.0
            strength = min(1.0, obv_trend * 5)
        elif obv_trend < -0.05:
            score = -1.0
            strength = min(1.0, abs(obv_trend) * 5)
        else:
            score = 0.0
            strength = 0.0
            
        return score, strength

    def score_adx(self, adx_values, plus_di, minus_di):
        """Memberikan skor untuk ADX"""
        current_adx = adx_values.iloc[-1]
        current_plus_di = plus_di.iloc[-1]
        current_minus_di = minus_di.iloc[-1]
        
        if current_adx > 25:
            if current_plus_di > current_minus_di:
                score = 1.0  # Strong uptrend
            else:
                score = -1.0  # Strong downtrend
            strength = min(1.0, (current_adx - 25) / 25)
        else:
            score = 0.0  # Weak or no trend
            strength = 0.0
            
        return score, strength

    def calculate_trend(self, values):
        """Menghitung trend dari seri nilai"""
        if len(values) < 2:
            return 0.0
            
        x = np.arange(len(values))
        y = values.values
        slope, _, _, _, _ = stats.linregress(x, y)
        
        if np.mean(y) != 0:
            return slope / np.mean(y)
        return 0.0

    def calculate_composite_score(self, scores):
        """Menghitung composite score berbobot"""
        total_score = 0
        total_weight = 0
        
        for indicator, (score, strength) in scores.items():
            if indicator in self.weights:
                weight = self.weights[indicator]
                total_score += score * strength * weight
                total_weight += weight
                
        return total_score / total_weight if total_weight > 0 else 0

    def get_confidence_level(self, composite_score, scores):
        """Menentukan tingkat keyakinan berdasarkan konsistensi sinyal"""
        if composite_score > 0:
            agreeing = sum(1 for score, strength in scores.values() if score > 0 and strength > 0.3)
        elif composite_score < 0:
            agreeing = sum(1 for score, strength in scores.values() if score < 0 and strength > 0.3)
        else:
            return "Rendah"
            
        total_indicators = len(scores)
        agreement_ratio = agreeing / total_indicators
        
        if agreement_ratio >= 0.7:
            return "Tinggi"
        elif agreement_ratio >= 0.5:
            return "Sedang"
        else:
            return "Rendah"

    def interpret_composite_score(self, score):
        """Memberikan interpretasi untuk composite score"""
        if score >= 0.7:
            return "Sangat Bullish"
        elif score >= 0.4:
            return "Bullish Kuat"
        elif score >= 0.1:
            return "Bullish Lemah"
        elif score > -0.1:
            return "Netral"
        elif score > -0.4:
            return "Bearish Lemah"
        elif score > -0.7:
            return "Bearish Kuat"
        else:
            return "Sangat Bearish"

# =========================
# Support & Resistance
# =========================
def identify_significant_swings(high, low, window=20):
    """Mengidentifikasi swing points signifikan"""
    # Gunakan metode sederhana untuk menemukan high dan low lokal
    high_idx = argrelextrema(high.values, np.greater, order=window//2)[0]
    low_idx = argrelextrema(low.values, np.less, order=window//2)[0]
    
    significant_highs = high.iloc[high_idx][-3:] if len(high_idx) > 0 else pd.Series()
    significant_lows = low.iloc[low_idx][-3:] if len(low_idx) > 0 else pd.Series()
    
    return significant_highs, significant_lows

def calculate_support_resistance(df):
    """Menghitung level support dan resistance"""
    current_price = df['Close'].iloc[-1]
    
    # Moving averages sebagai dynamic support/resistance
    ma20 = df['Close'].rolling(20).mean().iloc[-1]
    ma50 = df['Close'].rolling(50).mean().iloc[-1]
    ma100 = df['Close'].rolling(100).mean().iloc[-1]
    
    # Swing points
    swing_highs, swing_lows = identify_significant_swings(df['High'], df['Low'])
    
    # Support levels (harga di bawah current price)
    support_levels = []
    if not swing_lows.empty:
        support_levels.extend([l for l in swing_lows if l < current_price])
    
    for ma in [ma20, ma50, ma100]:
        if ma < current_price:
            support_levels.append(ma)
    
    # Resistance levels (harga di atas current price)
    resistance_levels = []
    if not swing_highs.empty:
        resistance_levels.extend([h for h in swing_highs if h > current_price])
    
    for ma in [ma20, ma50, ma100]:
        if ma > current_price:
            resistance_levels.append(ma)
    
    # Urutkan dan ambil 3 level terdekat
    support_levels.sort(reverse=True)  # Dari yang tertinggi ke terendah
    resistance_levels.sort()  # Dari yang terendah ke tertinggi
    
    return {
        'Support': support_levels[:3] if support_levels else [],
        'Resistance': resistance_levels[:3] if resistance_levels else []
    }

# =========================
# Visualisasi
# =========================
def create_gauge_chart(score):
    """Membuat gauge chart untuk composite score"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Composite Score", 'font': {'size': 24}},
        delta={'reference': 0},
        gauge={
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
                'value': score}}
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def create_technical_chart(df, sr_levels):
    """Membuat chart teknikal lengkap"""
    fig = go.Figure()
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price',
        increasing_line_color='green',
        decreasing_line_color='red'
    ))
    
    # Moving averages
    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], mode='lines', 
                             name='MA20', line=dict(color='blue', width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], mode='lines', 
                             name='MA50', line=dict(color='orange', width=1)))
    
    # Bollinger Bands
    bb_upper, bb_middle, bb_lower, _, _ = compute_bollinger_bands(df['Close'])
    fig.add_trace(go.Scatter(x=df.index, y=bb_upper, mode='lines', 
                             name='BB Upper', line=dict(color='gray', width=1, dash='dash')))
    fig.add_trace(go.Scatter(x=df.index, y=bb_middle, mode='lines', 
                             name='BB Middle', line=dict(color='gray', width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=bb_lower, mode='lines', 
                             name='BB Lower', line=dict(color='gray', width=1, dash='dash'),
                             fill='tonexty', fillcolor='rgba(200, 200, 200, 0.1)'))
    
    # Support and Resistance levels
    for i, level in enumerate(sr_levels['Support']):
        fig.add_hline(y=level, line_dash="dash", line_color="green",
                      annotation_text=f"S{i+1}: {level:.2f}", annotation_position="bottom right")
    
    for i, level in enumerate(sr_levels['Resistance']):
        fig.add_hline(y=level, line_dash="dash", line_color="red",
                      annotation_text=f"R{i+1}: {level:.2f}", annotation_position="top right")
    
    fig.update_layout(
        title="Chart Teknikal",
        xaxis_title="Tanggal",
        yaxis_title="Harga",
        xaxis_rangeslider_visible=False,
        height=500,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def create_volume_chart(df):
    """Membuat chart volume"""
    fig = go.Figure()
    
    # Volume bars
    colors = ['red' if row['Open'] > row['Close'] else 'green' for _, row in df.iterrows()]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=colors))
    
    # Volume moving average
    fig.add_trace(go.Scatter(x=df.index, y=df['Volume_MA20'], mode='lines', 
                             name='Vol MA20', line=dict(color='blue', width=1)))
    
    fig.update_layout(
        title="Volume",
        xaxis_title="Tanggal",
        yaxis_title="Volume",
        height=300,
        showlegend=True
    )
    
    return fig

# =========================
# Fungsi Utama
# =========================
def app():
    st.title("📊 Analisa Teknikal Saham")
    
    # Input parameter di bagian atas
    with st.container():
        st.subheader("Parameter Analisis")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            ticker_input = st.text_input("Kode Saham (tanpa .JK)", "BBCA").upper()
            account_size = st.number_input("Modal (Rp)", min_value=1000000, value=100000000, step=1000000)
        
        with col2:
            risk_percent = st.slider("Risiko per Trade (%)", min_value=0.5, max_value=5.0, value=2.0, step=0.5) / 100
            use_multi_timeframe = st.checkbox("Gunakan Multi-Timeframe", value=True)
        
        with col3:
            days_back = st.number_input("Ambil Data (hari)", min_value=30, max_value=1000, value=360, step=30)
            analysis_date = st.date_input("Tanggal Analisis", value=datetime.today())
        
        analyze_btn = st.button("🚀 Mulai Analisis", type="primary", use_container_width=True)
    
    if not analyze_btn:
        st.info("Masukkan parameter analisis dan klik 'Mulai Analisis'")
        return
    
    # Ambil dan proses data
    with st.spinner("Mengambil data saham..."):
        df = get_stock_data_yf(ticker_input, analysis_date, days_back)
    
    if df is None or df.empty:
        st.error(f"Tidak dapat mengambil data untuk saham {ticker_input}")
        return
    
    # Hitung indikator teknikal
    with st.spinner("Menghitung indikator..."):
        df['MA20'] = df['Close'].rolling(20).mean()
        df['MA50'] = df['Close'].rolling(50).mean()
        df['RSI'] = compute_rsi(df['Close'])
        df['MACD'], df['Signal'], df['Histogram'] = compute_macd(df['Close'])
        df['OBV'] = compute_obv(df['Close'], df['Volume'])
        df['ADX'], df['Plus_DI'], df['Minus_DI'] = compute_adx(df['High'], df['Low'], df['Close'])
        df['VWAP'] = compute_vwap(df['High'], df['Low'], df['Close'], df['Volume'])
        df['Volume_MA20'] = df['Volume'].rolling(20).mean()
        df['ATR'] = compute_atr(df['High'], df['Low'], df['Close'])
    
    # Hitung support dan resistance
    sr_levels = calculate_support_resistance(df)
    
    # Hitung skor indikator
    scoring_system = IndicatorScoringSystem()
    
    price_change = (df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100
    
    scores = {
        'rsi': scoring_system.score_rsi(df['RSI']),
        'macd_cross': scoring_system.score_macd(df['MACD'], df['Signal'], df['Histogram'])[:2],
        'macd_hist': scoring_system.score_macd(df['MACD'], df['Signal'], df['Histogram'])[2:],
        'bollinger': scoring_system.score_bollinger(df['Close'], *compute_bollinger_bands(df['Close'])[:3]),
        'volume': scoring_system.score_volume(df['Volume'], df['Volume_MA20'], price_change),
        'obv': scoring_system.score_obv(df['OBV']),
        'adx': scoring_system.score_adx(df['ADX'], df['Plus_DI'], df['Minus_DI'])
    }
    
    composite_score = scoring_system.calculate_composite_score(scores)
    confidence = scoring_system.get_confidence_level(composite_score, scores)
    interpretation = scoring_system.interpret_composite_score(composite_score)
    
    # Tampilkan hasil analisis
    st.divider()
    st.subheader("🎯 Hasil Analisis Cross-Confirmation")
    
    # Tampilkan composite score dengan gauge chart
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.plotly_chart(create_gauge_chart(composite_score), use_container_width=True)
    
    with col2:
        st.metric("Composite Score", f"{composite_score:.3f}")
        st.metric("Tingkat Keyakinan", confidence)
        st.metric("Interpretasi", interpretation)
        
        # Tabel detail indikator
        indicator_data = []
        for indicator, (score, strength) in scores.items():
            direction = "Bullish" if score > 0 else "Bearish" if score < 0 else "Netral"
            indicator_data.append({
                "Indicator": indicator.upper(),
                "Direction": direction,
                "Score": f"{score:.2f}",
                "Strength": f"{strength:.2f}"
            })
        
        st.dataframe(pd.DataFrame(indicator_data), use_container_width=True)
    
    # Tampilkan support dan resistance
    st.subheader("📈 Support & Resistance")
    
    sr_col1, sr_col2 = st.columns(2)
    
    with sr_col1:
        st.write("**Level Support**")
        if sr_levels['Support']:
            for i, level in enumerate(sr_levels['Support']):
                st.write(f"{i+1}. Rp {level:,.2f}")
        else:
            st.write("Tidak ada level support yang teridentifikasi")
    
    with sr_col2:
        st.write("**Level Resistance**")
        if sr_levels['Resistance']:
            for i, level in enumerate(sr_levels['Resistance']):
                st.write(f"{i+1}. Rp {level:,.2f}")
        else:
            st.write("Tidak ada level resistance yang teridentifikasi")
    
    # Tampilkan chart
    st.subheader("📊 Chart Teknikal")
    st.plotly_chart(create_technical_chart(df, sr_levels), use_container_width=True)
    
    # Tampilkan volume chart
    st.plotly_chart(create_volume_chart(df), use_container_width=True)
    
    # Informasi tambahan
    st.subheader("📋 Informasi Tambahan")
    
    info_col1, info_col2, info_col3 = st.columns(3)
    
    with info_col1:
        st.metric("Harga Terakhir", f"Rp {df['Close'].iloc[-1]:,.2f}")
        st.metric("Volume", f"{df['Volume'].iloc[-1]:,.0f}")
    
    with info_col2:
        st.metric("RSI", f"{df['RSI'].iloc[-1]:.2f}")
        st.metric("ATR", f"{df['ATR'].iloc[-1]:.2f}")
    
    with info_col3:
        st.metric("VWAP", f"{df['VWAP'].iloc[-1]:.2f}")
        st.metric("ADX", f"{df['ADX'].iloc[-1]:.2f}")
    
    # Disclaimer
    st.divider()
    st.warning("""
    **Disclaimer:** Analisis ini hanya untuk tujuan edukasi dan bukan sebagai rekomendasi investasi. 
    Selalu lakukan penelitian sendiri dan pertimbangkan kondisi pasar sebelum membuat keputusan investasi.
    """)

if __name__ == "__main__":
    main()
