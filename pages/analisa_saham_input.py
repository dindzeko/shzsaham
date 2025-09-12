# analisa_saham.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta
from scipy import stats

# =========================
# KONFIGURASI & BOBOT
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
# UTILITAS FORMAT & IDX
# =========================
def fmt_rp(x: float) -> str:
    return f"Rp {x:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")

def fmt_int(x: float) -> str:
    return f"{int(round(x)):,}".replace(",", ".")

def shares_to_lot(shares: float) -> float:
    return shares / 100.0

def lot_to_shares(lot: float) -> int:
    return int(lot * 100)

def get_idx_tick(price: float) -> int:
    """Tick size aturan IDX (disederhanakan & umum dipakai)"""
    if price < 200: return 1
    if price < 500: return 2
    if price < 2000: return 5
    if price < 5000: return 10
    return 25

def round_to_tick(price: float) -> float:
    tick = get_idx_tick(price)
    return round(price / tick) * tick

# =========================
# DATA FETCH (dengan cache)
# =========================
@st.cache_data(ttl=600, show_spinner=False)
def fetch_history_yf(ticker_jk: str, start: datetime, end: datetime) -> pd.DataFrame:
    df = yf.download(ticker_jk, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"),
                     auto_adjust=False, progress=False)
    return df

def resolve_ticker(user_input: str) -> str:
    t = user_input.strip().upper()
    if not t.endswith(".JK"):
        t += ".JK"
    return t

# =========================
# INDIKATOR TEKNIKAL
# =========================
def rma(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(alpha=1/period, adjust=False).mean()

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI Wilder smoothing (praktik standar)"""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = rma(gain, period)
    avg_loss = rma(loss, period)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def compute_mfi_corrected(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """MFI dengan typical price vs hari sebelumnya + guard divide-by-zero"""
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = tp * df['Volume']
    pos_flow = money_flow.where(tp > tp.shift(1), 0.0)
    neg_flow = money_flow.where(tp < tp.shift(1), 0.0)
    pos = pos_flow.rolling(period, min_periods=1).sum()
    neg = neg_flow.rolling(period, min_periods=1).sum().replace(0, np.nan)
    ratio = pos / neg
    mfi = 100 - (100 / (1 + ratio))
    return mfi.fillna(50)

def compute_macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return macd, sig, hist

def compute_bollinger_bands(close: pd.Series, window=20, num_std=2):
    sma = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    width = (upper - lower)
    bandwidth = width / sma.replace(0, np.nan)
    percent_b = (close - lower) / width.replace(0, np.nan)
    percent_b = percent_b.clip(0, 1)
    return upper, sma, lower, bandwidth.fillna(0), percent_b.fillna(0)

def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period=14):
    tr = pd.concat([
        (high - low),
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return rma(tr, period)

def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    change = np.sign(close.diff().fillna(0))
    obv = (volume * change).cumsum()
    return obv.fillna(0)

def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period=14):
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)

    tr = pd.concat([
        (high - low),
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)

    atr = rma(tr, period).replace(0, np.nan)
    plus_di = 100 * rma(plus_dm, period) / atr
    minus_di = 100 * rma(minus_dm, period) / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = rma(dx, period)
    return adx.fillna(0), plus_di.fillna(0), minus_di.fillna(0)

def compute_vwap(high, low, close, volume):
    tp = (high + low + close) / 3
    cum_v = volume.cumsum().replace(0, np.nan)
    vwap = (tp * volume).cumsum() / cum_v
    return vwap.fillna(method="bfill").fillna(method="ffill")

def compute_adl(high, low, close, volume):
    mfm = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    mfm = mfm.replace([np.inf, -np.inf], 0).fillna(0)
    mfv = mfm * volume
    return mfv.cumsum()

# =========================
# S/R & FIBONACCI
# =========================
def identify_swings(df: pd.DataFrame, window: int = 60) -> tuple[float, float]:
    """Ambil swing high/low sederhana dari 60 bar terakhir."""
    last = df.tail(window)
    return last['High'].max(), last['Low'].min()

def fibonacci_levels(swing_high: float, swing_low: float) -> dict:
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

def psych_level(close_price: float) -> float:
    levels = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
    return min(levels, key=lambda x: abs(x - close_price))

def cluster_levels(levels, tol=0.01):
    """Cluster level yang berdekatan (±1%) lalu beri skor konfluensi."""
    if not levels: return []
    levels = sorted(levels)
    clusters, cluster = [], [levels[0]]
    for x in levels[1:]:
        if abs(x - cluster[-1]) / max(cluster[-1], 1e-9) <= tol:
            cluster.append(x)
        else:
            clusters.append(cluster); cluster = [x]
    clusters.append(cluster)
    # median cluster & skor = jumlah sumber level
    out = []
    for c in clusters:
        out.append((float(np.median(c)), len(c)))
    out.sort(key=lambda t: t[1], reverse=True)
    return out

def compute_support_resistance(df: pd.DataFrame) -> dict:
    close = df['Close'].iloc[-1]
    swing_high, swing_low = identify_swings(df, 60)
    fib = fibonacci_levels(swing_high, swing_low)
    ma20, ma50, ma100 = df['Close'].rolling(20).mean().iloc[-1], df['Close'].rolling(50).mean().iloc[-1], df['Close'].rolling(100).mean().iloc[-1]
    vwap = compute_vwap(df['High'], df['Low'], df['Close'], df['Volume']).iloc[-1]
    psych = psych_level(close)

    candidates = [
        fib['Fib_0.236'], fib['Fib_0.382'], fib['Fib_0.5'], fib['Fib_0.618'], fib['Fib_0.786'], fib['Fib_0.0'], fib['Fib_1.0'],
        ma20, ma50, ma100, vwap, psych
    ]
    candidates = [c for c in candidates if np.isfinite(c)]

    clustered = cluster_levels(candidates, tol=0.01)  # (level, score)

    supports = [lvl for (lvl, sc) in clustered if lvl < close]
    resistances = [lvl for (lvl, sc) in clustered if lvl > close]

    supports = sorted(supports, reverse=True)[:3] if supports else [close * 0.95]
    resistances = sorted(resistances)[:3] if resistances else [close * 1.05]

    return {'Support': supports, 'Resistance': resistances, 'Fibonacci': fib}

# =========================
# SCORING SISTEM
# =========================
class IndicatorScoringSystem:
    def __init__(self, weights=None):
        self.weights = weights or INDICATOR_WEIGHTS

    @staticmethod
    def _trend(values: pd.Series, period: int) -> float:
        if len(values) < max(3, period): return 0.0
        x = np.arange(period)
        y = values.iloc[-period:].values
        slope, _, _, _, _ = stats.linregress(x, y)
        mean = np.mean(y) if np.mean(y) != 0 else 1e-9
        return slope / mean

    def score_rsi(self, rsi: pd.Series):
        r = float(rsi.iloc[-1])
        if r <= 30: score, strg = 1.0, min(1.0, (30 - r) / 30)
        elif r <= 40: score, strg = 0.5, (40 - r) / 10
        elif r < 50:  score, strg = 0.2, (r - 40) / 10
        elif r < 60:  score, strg = -0.2, (60 - r) / 10
        elif r < 70:  score, strg = -0.5, (r - 60) / 10
        else:         score, strg = -1.0, min(1.0, (r - 70) / 30)
        t = self._trend(rsi, 5)
        if (score > 0 and t > 0) or (score < 0 and t < 0):
            strg = min(1.0, strg + min(abs(t)*2, 0.3))
        return score, float(strg)

    def score_macd(self, macd_line, signal_line, histogram):
        cross = 1.0 if macd_line.iloc[-1] > signal_line.iloc[-1] else -1.0
        hist_trend = self._trend(histogram, 5)
        if hist_trend > 0: hscore, hstr = 0.5, min(1.0, hist_trend*2)
        elif hist_trend < 0: hscore, hstr = -0.5, min(1.0, abs(hist_trend)*2)
        else: hscore, hstr = 0.0, 0.0
        return cross, 1.0, hscore, float(hstr)

    def score_boll(self, close, upper, lower, pct_b, bandwidth):
        cb = float(pct_b.iloc[-1]); bw = float(bandwidth.iloc[-1])
        hist_bw = bandwidth.iloc[-120:].dropna()
        if len(hist_bw) < 10:
            bw_pct = 0.5
        else:
            bw_pct = stats.percentileofscore(hist_bw, bw) / 100.0
        is_squeeze = bw_pct < 0.2
        if cb > 0.8: score, strg = -1.0, min(1.0, (cb - 0.8)/0.2)
        elif cb < 0.2: score, strg = 1.0, min(1.0, (0.2 - cb)/0.2)
        elif cb > 0.5: score, strg = -0.5, (cb - 0.5)/0.3
        else: score, strg = 0.5, (0.5 - cb)/0.3
        if is_squeeze: strg = min(1.0, strg + 0.3)
        return score, float(strg), bool(is_squeeze)

    def score_volume(self, vol: pd.Series, vol_ma: pd.Series, price_change_pct: float):
        eps = 1e-9
        ratio = float(vol.iloc[-1] / max(vol_ma.iloc[-1], eps))
        if ratio > 1.7 and price_change_pct > 0:
            score, strg = 1.0, min(1.0, (ratio - 1.7) / 0.8)
        elif ratio > 1.7 and price_change_pct < 0:
            score, strg = -1.0, min(1.0, (ratio - 1.7) / 0.8)
        elif ratio > 1.3 and price_change_pct > 0:
            score, strg = 0.5, (ratio - 1.3) / 0.4
        elif ratio > 1.3 and price_change_pct < 0:
            score, strg = -0.5, (ratio - 1.3) / 0.4
        else:
            score, strg = 0.0, 0.0
        return score, float(max(0.0, min(strg, 1.0)))

    def score_obv(self, obv: pd.Series):
        t = self._trend(obv, 10)
        if t > 0.05: return 1.0, min(1.0, t*5)
        if t < -0.05: return -1.0, min(1.0, abs(t)*5)
        return 0.0, 0.0

    def score_adx(self, adx: pd.Series, plus_di: pd.Series, minus_di: pd.Series, threshold=25):
        cur_adx, pdi, mdi = float(adx.iloc[-1]), float(plus_di.iloc[-1]), float(minus_di.iloc[-1])
        if cur_adx < threshold: return 0.0, 0.0
        direction = 1.0 if pdi > mdi else -1.0
        strength = min(1.0, (cur_adx - threshold) / (100 - threshold))
        return direction, float(strength)

    def composite(self, scores: dict) -> float:
        ws, tw = [], 0.0
        for k, (s, strg) in scores.items():
            if k in self.weights:
                ws.append(s * strg * self.weights[k]); tw += self.weights[k]
        return float(sum(ws) / tw) if tw > 0 else 0.0

    def confidence(self, composite_score: float, scores: dict) -> str:
        if composite_score > 0:
            agree = sum(1 for s, stg in scores.values() if s > 0 and stg > 0.3)
        elif composite_score < 0:
            agree = sum(1 for s, stg in scores.values() if s < 0 and stg > 0.3)
        else:
            agree = 0
        total = len(scores) if scores else 1
        ratio = agree / total
        if ratio >= 0.7: return "Tinggi"
        if ratio >= 0.5: return "Sedang"
        return "Rendah"

    @staticmethod
    def interpret(score: float) -> str:
        if score >= 0.7: return "Sangat Bullish - Sentimen beli sangat kuat"
        if score >= 0.4: return "Bullish Kuat - Sentimen beli kuat"
        if score >= 0.1: return "Bullish Lemah - Sentimen cenderung beli"
        if score > -0.1: return "Netral - Sentimen tidak jelas"
        if score > -0.4: return "Bearish Lemah - Sentimen cenderung jual"
        if score > -0.7: return "Bearish Kuat - Sentimen jual kuat"
        return "Sangat Bearish - Sentimen jual sangat kuat"

# =========================
# BREAKOUT & RISK
# =========================
class BreakoutDetector:
    def __init__(self, atr_period=14, min_buffer_pct=0.005):
        self.atr_period = atr_period
        self.min_buffer_pct = min_buffer_pct

    def _dynamic_buffer(self, df: pd.DataFrame) -> float:
        atr = compute_atr(df['High'], df['Low'], df['Close'], self.atr_period).iloc[-1]
        close = df['Close'].iloc[-1]
        atr_buf = (atr / max(close, 1e-9)) * 0.5
        return max(self.min_buffer_pct, float(atr_buf))

    def detect(self, df, res_level: float, sup_level: float, bars_confirm: int = 1):
        close = df['Close']
        vol = df['Volume']
        buffer = self._dynamic_buffer(df)
        avg_vol = vol.rolling(20).mean().iloc[-1]
        vol_ok = vol.iloc[-1] > 1.5 * max(avg_vol, 1e-9)

        res_break = (close.iloc[-1] > res_level * (1 + buffer))
        sup_break = (close.iloc[-1] < sup_level * (1 - buffer))

        def confirm(level, direction):
            if bars_confirm <= 1: return True
            if direction == "up":
                return (close.iloc[-bars_confirm:] > level).all()
            else:
                return (close.iloc[-bars_confirm:] < level).all()

        res_ok = res_break and confirm(res_level * (1 + buffer), "up")
        sup_ok = sup_break and confirm(sup_level * (1 - buffer), "down")
        return res_ok, sup_ok, vol_ok, buffer

    @staticmethod
    def position_size(entry: float, stop: float, account_size: float, risk_percent: float) -> int:
        risk_amt = account_size * risk_percent
        risk_per_share = max(abs(entry - stop), 1e-9)
        shares = risk_amt / risk_per_share
        lots = max(int(shares // 100), 0)
        return lots * 100

    def plan(self, df, breakout_type: str, level: float, buffer: float, account_size: float, risk_percent: float):
        atr = compute_atr(df['High'], df['Low'], df['Close'], self.atr_period).iloc[-1]
        if breakout_type == "resistance":
            entry = round_to_tick(level * (1 + buffer))
            stop = round_to_tick(entry - 2 * atr)
            t1 = round_to_tick(entry + 2 * atr)
            t2 = round_to_tick(entry + 4 * atr)
            size = self.position_size(entry, stop, account_size, risk_percent)
            rr = (t1 - entry) / max((entry - stop), 1e-9)
            return {"type": "Bullish Breakout", "entry": entry, "stop_loss": stop, "target_1": t1, "target_2": t2, "position_size": size, "risk_reward": round(rr, 2)}
        else:
            entry = round_to_tick(level * (1 - buffer))
            stop = round_to_tick(entry + 2 * atr)
            t1 = round_to_tick(entry - 2 * atr)
            t2 = round_to_tick(entry - 4 * atr)
            size = self.position_size(entry, stop, account_size, risk_percent)
            rr = (entry - t1) / max((stop - entry), 1e-9)
            return {"type": "Bearish Breakdown", "entry": entry, "stop_loss": stop, "target_1": t1, "target_2": t2, "position_size": size, "risk_reward": round(rr, 2)}

# =========================
# BANDARMOLOGY RINGKAS
# =========================
def bandarmology_brief(df: pd.DataFrame, period: int = 30) -> dict:
    out = {}
    vol_ma = df['Volume'].rolling(20).mean()
    vol_std = df['Volume'].rolling(20).std().replace(0, 1e-9)
    z = (df['Volume'] - vol_ma) / vol_std
    out['volume_spikes_5d'] = int((z.iloc[-5:] > 2.5).sum())

    price_chg = df['Close'].pct_change()
    vol_chg = df['Volume'].pct_change()
    pos = ((price_chg > 0) & (vol_chg > 0)).iloc[-period:].sum()
    neg = ((price_chg < 0) & (vol_chg > 0)).iloc[-period:].sum()
    out['pos_volume_price_days'] = int(pos)
    out['neg_volume_price_days'] = int(neg)

    adl = compute_adl(df['High'], df['Low'], df['Close'], df['Volume'])
    obv = compute_obv(df['Close'], df['Volume'])
    mfi = compute_mfi_corrected(df, 14)

    def pct_change_last(s: pd.Series, n=5):
        if len(s) <= n or s.iloc[-n] == 0: return 0.0
        return float((s.iloc[-1] - s.iloc[-n]) / abs(s.iloc[-n]) * 100)

    out['adl_5d_pct'] = round(pct_change_last(adl, 5), 2)
    out['obv_5d_pct'] = round(pct_change_last(obv, 5), 2)
    out['mfi_last'] = round(float(mfi.iloc[-1]), 2)

    # Volume profile sederhana 20 bar
    last = df.tail(20)
    price_min, price_max = last['Low'].min(), last['High'].max()
    bins = np.linspace(price_min, price_max, 21)
    vol_profile = np.zeros(20)
    for _, r in last.iterrows():
        rng = r['High'] - r['Low']
        if rng <= 0: continue
        vpp = r['Volume'] / rng
        for i in range(20):
            lo, hi = bins[i], bins[i+1]
            ol = max(lo, r['Low']); oh = min(hi, r['High'])
            ov = max(0, oh - ol)
            vol_profile[i] += ov * vpp
    poc_idx = int(np.argmax(vol_profile))
    poc_price = (bins[poc_idx] + bins[poc_idx+1]) / 2
    total_v = vol_profile.sum()
    sorted_idx = np.argsort(vol_profile)[::-1]
    acc, chosen = 0, []
    for idx in sorted_idx:
        acc += vol_profile[idx]
        chosen.append(idx)
        if acc >= total_v * 0.7: break
    va_low = bins[min(chosen)]
    va_high = bins[max(chosen)+1]

    out['poc'] = float(poc_price)
    out['va_low'] = float(va_low)
    out['va_high'] = float(va_high)
    out['in_value_area'] = (df['Close'].iloc[-1] >= va_low) and (df['Close'].iloc[-1] <= va_high)
    return out

# =========================
# DIVERGENCE DETECTOR (BARU)
# =========================
def _local_peaks_troughs(series: pd.Series):
    """Deteksi puncak/lembah sederhana: bandingkan dengan tetangga kiri-kanan."""
    peaks = (series.shift(1) < series) & (series.shift(-1) < series)
    troughs = (series.shift(1) > series) & (series.shift(-1) > series)
    return series[peaks], series[troughs]

def detect_divergence(price: pd.Series, indicator: pd.Series, lookback: int = 120) -> dict:
    """
    Mendeteksi divergensi bullish/bearish sederhana dalam lookback terakhir.
    - Bearish divergence: harga HH, indikator LH.
    - Bullish divergence: harga LL, indikator HL.
    Return contoh paling terbaru jika ada.
    """
    p = price.tail(lookback)
    i = indicator.reindex(p.index).fillna(method="ffill")
    p_peaks, p_troughs = _local_peaks_troughs(p)
    i_peaks, i_troughs = _local_peaks_troughs(i)

    out = {"bearish": None, "bullish": None}

    # Cari dua puncak terakhir (harga & indikator)
    if len(p_peaks) >= 2 and len(i_peaks) >= 2:
        p_last2 = p_peaks.iloc[-2:]
        i_last2 = i_peaks.loc[p_last2.index].dropna()
        if len(i_last2) == 2:
            # Harga HH? Indikator LH?
            if p_last2.iloc[-1] > p_last2.iloc[0] and i_last2.iloc[-1] < i_last2.iloc[0]:
                out["bearish"] = {
                    "price_points": (p_last2.index[0], p_last2.index[-1]),
                    "indicator_points": (i_last2.index[0], i_last2.index[-1]),
                    "desc": "Bearish divergence (Price HH vs Indicator LH)"
                }

    # Cari dua lembah terakhir (harga & indikator)
    if len(p_troughs) >= 2 and len(i_troughs) >= 2:
        p_last2 = p_troughs.iloc[-2:]
        i_last2 = i_troughs.loc[p_last2.index].dropna()
        if len(i_last2) == 2:
            # Harga LL? Indikator HL?
            if p_last2.iloc[-1] < p_last2.iloc[0] and i_last2.iloc[-1] > i_last2.iloc[0]:
                out["bullish"] = {
                    "price_points": (p_last2.index[0], p_last2.index[-1]),
                    "indicator_points": (i_last2.index[0], i_last2.index[-1]),
                    "desc": "Bullish divergence (Price LL vs Indicator HL)"
                }
    return out

# =========================
# RETEST LOGIC SETELAH BREAKOUT (BARU)
# =========================
def assess_retest(df: pd.DataFrame, level: float, direction: str, buffer_pct: float, lookback: int = 10) -> str:
    """
    Menilai apakah terjadi retest terhadap level setelah breakout.
    - direction: 'up' untuk breakout ke atas, 'down' untuk breakdown.
    - Kriteria sederhana:
      * 'Retested and held' jika low (untuk up) atau high (untuk down) menyentuh ±30% buffer di 1..lookback bar
        lalu harga kembali sesuai arah (close > level untuk up, close < level untuk down).
      * 'Retest happening' jika harga saat ini berada dekat level ±30% buffer.
      * 'No retest' jika tidak ada kondisi di atas.
    """
    if len(df) < 5: return "No data"
    recent = df.tail(lookback)
    band = level * (0.3 * buffer_pct)  # toleransi 30% dari buffer breakout
    if direction == "up":
        touched = ((recent['Low'] >= level - band) & (recent['Low'] <= level + band)).any()
        near_now = (df['Close'].iloc[-1] >= level - band) and (df['Close'].iloc[-1] <= level + band)
        if near_now: return "Retest happening (near level)"
        if touched and (df['Close'].iloc[-1] > level): return "Retested and held"
        return "No retest"
    else:
        touched = ((recent['High'] >= level - band) & (recent['High'] <= level + band)).any()
        near_now = (df['Close'].iloc[-1] >= level - band) and (df['Close'].iloc[-1] <= level + band)
        if near_now: return "Retest happening (near level)"
        if touched and (df['Close'].iloc[-1] < level): return "Retested and held"
        return "No retest"

# =========================
# BACKTEST RINGKAS EXPECTANCY (BARU)
# =========================
def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d['MA20'] = d['Close'].rolling(20).mean()
    d['MA50'] = d['Close'].rolling(50).mean()
    d['MA100'] = d['Close'].rolling(100).mean()
    d['RSI'] = compute_rsi(d['Close'], 14)
    d['MACD'], d['Signal'], d['Hist'] = compute_macd(d['Close'], 12, 26, 9)
    d['BB_Upper'], d['BB_Middle'], d['BB_Lower'], d['BB_BW'], d['BB_%B'] = compute_bollinger_bands(d['Close'], 20, 2)
    d['ATR'] = compute_atr(d['High'], d['Low'], d['Close'], 14)
    d['OBV'] = compute_obv(d['Close'], d['Volume'])
    d['ADX'], d['Plus_DI'], d['Minus_DI'] = compute_adx(d['High'], d['Low'], d['Close'], 14)
    d['VWAP'] = compute_vwap(d['High'], d['Low'], d['Close'], d['Volume'])
    d['MFI'] = compute_mfi_corrected(d, 14)
    d['Volume_MA20'] = d['Volume'].rolling(20).mean()
    return d

def backtest_composite(df: pd.DataFrame, threshold_long=0.4, threshold_short=-0.4, max_hold=20):
    """
    Backtest sangat ringkas:
    - Entry Long saat composite menyilang naik > 0.4; Stop = 2*ATR di bawah; Target = 2*ATR di atas.
    - Entry Short saat composite menyilang turun < -0.4; Stop = 2*ATR di atas; Target = 2*ATR di bawah.
    - Exit saat Stop/Target tercapai, atau composite kembali netral (<=0 untuk long, >=0 untuk short), atau timeout max_hold bar.
    - Hitung R per trade dan expectancy.
    """
    scorer = IndicatorScoringSystem()
    # Pra-hitung skor tiap bar
    comp = []
    for idx in range(len(df)):
        if idx < 100:  # need warmup for indicators
            comp.append(0.0); continue
        window = df.iloc[:idx+1]
        scores = {}
        scores['rsi'] = scorer.score_rsi(window['RSI'])
        mc, mcs, mh, mhs = scorer.score_macd(window['MACD'], window['Signal'], window['Hist'])
        scores['macd_cross'] = (mc, mcs)
        scores['macd_hist'] = (mh, mhs)
        bsc, bst, _ = scorer.score_boll(window['Close'], window['BB_Upper'], window['BB_Lower'],
                                        window['BB_%B'], window['BB_BW'])
        scores['bollinger'] = (bsc, bst)
        price_chg_pct = float((window['Close'].iloc[-1] - window['Close'].iloc[-2]) / max(window['Close'].iloc[-2], 1e-9) * 100)
        vsc, vst = scorer.score_volume(window['Volume'], window['Volume_MA20'], price_chg_pct)
        scores['volume'] = (vsc, vst)
        os, ost = scorer.score_obv(window['OBV'])
        scores['obv'] = (os, ost)
        axs, axst = scorer.score_adx(window['ADX'], window['Plus_DI'], window['Minus_DI'])
        scores['adx'] = (axs, axst)
        comp.append(scorer.composite(scores))
    comp = pd.Series(comp, index=df.index)

    trades = []
    i = 101
    while i < len(df):
        price = df['Close'].iloc[i]
        atr = df['ATR'].iloc[i]
        # Crossover rules
        enter_long = comp.iloc[i-1] <= threshold_long and comp.iloc[i] > threshold_long
        enter_short = comp.iloc[i-1] >= threshold_short and comp.iloc[i] < threshold_short
        if enter_long:
            entry = price
            stop = entry - 2*atr
            target = entry + 2*atr
            stop_dist = entry - stop
            j = i+1
            exit_price, reason = None, None
            while j < min(i+max_hold, len(df)):
                high = df['High'].iloc[j]; low = df['Low'].iloc[j]; cl = df['Close'].iloc[j]
                if low <= stop: exit_price, reason = stop, "Stop"; break
                if high >= target: exit_price, reason = target, "Target"; break
                if comp.iloc[j] <= 0: exit_price, reason = cl, "NeutralExit"; break
                j += 1
            if exit_price is None:
                exit_price, reason = df['Close'].iloc[min(i+max_hold-1, len(df)-1)], "Timeout"
            R = (exit_price - entry) / max(stop_dist, 1e-9)
            trades.append({"dir":"Long","entry_i":i,"exit_i":j,"R":R,"reason":reason})
            i = j + 1
        elif enter_short:
            entry = price
            stop = entry + 2*atr
            target = entry - 2*atr
            stop_dist = stop - entry
            j = i+1
            exit_price, reason = None, None
            while j < min(i+max_hold, len(df)):
                high = df['High'].iloc[j]; low = df['Low'].iloc[j]; cl = df['Close'].iloc[j]
                if high >= stop: exit_price, reason = stop, "Stop"; break
                if low <= target: exit_price, reason = target, "Target"; break
                if comp.iloc[j] >= 0: exit_price, reason = cl, "NeutralExit"; break
                j += 1
            if exit_price is None:
                exit_price, reason = df['Close'].iloc[min(i+max_hold-1, len(df)-1)], "Timeout"
            R = (entry - exit_price) / max(stop_dist, 1e-9)
            trades.append({"dir":"Short","entry_i":i,"exit_i":j,"R":R,"reason":reason})
            i = j + 1
        else:
            i += 1

    if len(trades) == 0:
        return {"num_trades":0,"win_rate":0.0,"avg_R":0.0,"median_R":0.0,"avg_hold":0.0,"detail":pd.DataFrame()}, comp

    tr_df = pd.DataFrame(trades)
    win_rate = float((tr_df['R'] > 0).mean()*100)
    avg_R = float(tr_df['R'].mean())
    median_R = float(tr_df['R'].median())
    avg_hold = float((tr_df['exit_i'] - tr_df['entry_i']).mean())
    summary = {"num_trades":int(len(trades)),"win_rate":win_rate,"avg_R":avg_R,"median_R":median_R,"avg_hold":avg_hold,"detail":tr_df}
    return summary, comp

# =========================
# CHART
# =========================
def make_main_chart(df: pd.DataFrame, sr: dict, is_squeeze: bool) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name="Candlestick",
        increasing_line_color='green', decreasing_line_color='red'
    ))
    for name, col, color in [
        ("MA20", 'MA20', 'blue'),
        ("MA50", 'MA50', 'orange'),
        ("MA100", 'MA100', 'purple'),
    ]:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=name, line=dict(color=color, width=1.5)))

    # Bollinger
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name="BB Upper", mode="lines", line=dict(color="red", width=1, dash="dot")))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Middle'], name="BB Middle", mode="lines", line=dict(color="purple", width=1.2)))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name="BB Lower", mode="lines", line=dict(color="green", width=1, dash="dot"), fill='tonexty', fillcolor='rgba(173,216,230,0.1)'))

    # S/R
    for i, lvl in enumerate(sr['Support']):
        fig.add_hline(y=lvl, line_dash="dash", line_color="green",
                      annotation_text=f"Support {i+1}: {fmt_rp(lvl)}", annotation_position="bottom right")
    for i, lvl in enumerate(sr['Resistance']):
        fig.add_hline(y=lvl, line_dash="dash", line_color="red",
                      annotation_text=f"Resistance {i+1}: {fmt_rp(lvl)}", annotation_position="top right")

    if is_squeeze:
        fig.add_annotation(x=df.index[-1], y=df['Close'].iloc[-1], text="BOLLINGER SQUEEZE",
                           showarrow=True, arrowhead=1, arrowcolor="purple", font=dict(color="purple"))

    fig.update_layout(
        title="Chart Teknikal",
        template="plotly_white",
        xaxis_rangeslider_visible=False,
        height=700,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='x unified'
    )
    return fig

# =========================
# APP STREAMLIT
# =========================
def app():
    st.title("📊 Analisa Teknikal Saham (IDX) + Divergence, Retest, & Backtest")

    c1, c2 = st.columns(2)
    with c1:
        ticker_in = st.text_input("Kode Saham", value="BBCA").strip().upper()
        account_size = st.number_input("Modal (Rp)", value=100_000_000, step=10_000_000)
        risk_percent = st.slider("Risiko per Trade (%)", 0.5, 5.0, 2.0) / 100
    with c2:
        analysis_date = st.date_input("📅 Tanggal Analisis", value=datetime.today())
        use_mtf = st.checkbox("Gunakan Konfirmasi Multi-Timeframe (Weekly)", value=True)

    if st.button("🚀 Mulai Analisis"):
        if not ticker_in:
            st.warning("Masukkan kode saham terlebih dahulu.")
            return

        ticker = resolve_ticker(ticker_in)  # tambah .JK jika belum
        end = datetime.combine(analysis_date, datetime.min.time()) + timedelta(days=1)  # inclusive date
        start = end - timedelta(days=365)  # 1 tahun
        df = fetch_history_yf(ticker, start, end)

        if df is None or df.empty:
            st.warning("Data tidak tersedia. Coba tanggal lain atau kode lain.")
            return

        last_dt = df.index[-1]
        st.caption(f"Data terakhir per: **{last_dt.date()}**")

        # Indikator lengkap
        base = compute_all_indicators(df)

        # Multi-timeframe bias (weekly sederhana)
        weekly_bias = 0
        if use_mtf:
            start_w = start - timedelta(days=100)
            dfw = fetch_history_yf(ticker, start_w, end)
            if not dfw.empty:
                w = dfw.resample('W-FRI').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
                if len(w) >= 30:
                    macd_w, sig_w, _ = compute_macd(w['Close'])
                    weekly_bias = 1 if macd_w.iloc[-1] > sig_w.iloc[-1] else -1

        # Support / Resistance
        sr = compute_support_resistance(base)

        # Scoring
        scorer = IndicatorScoringSystem()
        scores = {}
        scores['rsi'] = scorer.score_rsi(base['RSI'])
        mc, mcs, mh, mhs = scorer.score_macd(base['MACD'], base['Signal'], base['Hist'])
        scores['macd_cross'] = (mc, mcs)
        scores['macd_hist'] = (mh, mhs)
        bscore, bstr, is_squeeze = scorer.score_boll(base['Close'], base['BB_Upper'], base['BB_Lower'], base['BB_%B'], base['BB_BW'])
        scores['bollinger'] = (bscore, bstr)
        price_change_pct = float((base['Close'].iloc[-1] - base['Close'].iloc[-2]) / max(base['Close'].iloc[-2], 1e-9) * 100) if len(base) > 1 else 0.0
        vscore, vstr = scorer.score_volume(base['Volume'], base['Volume_MA20'], price_change_pct)
        scores['volume'] = (vscore, vstr)
        oscore, ostr = scorer.score_obv(base['OBV'])
        scores['obv'] = (oscore, ostr)
        adx_s, adx_str = scorer.score_adx(base['ADX'], base['Plus_DI'], base['Minus_DI'])
        scores['adx'] = (adx_s, adx_str)

        composite = scorer.composite(scores)
        if weekly_bias != 0:
            composite *= (1.15 if weekly_bias > 0 else 0.85)
        confidence = scorer.confidence(composite, scores)
        interp = scorer.interpret(composite)

        # Breakout detector (+ retest)
        detector = BreakoutDetector()
        res_lvl = sr['Resistance'][0] if sr['Resistance'] else base['Close'].iloc[-1] * 1.05
        sup_lvl = sr['Support'][0] if sr['Support'] else base['Close'].iloc[-1] * 0.95
        res_ok, sup_ok, vol_ok, buffer = detector.detect(base, res_lvl, sup_lvl, bars_confirm=1)
        retest_status = "N/A"
        if res_ok:
            retest_status = assess_retest(base, res_lvl, "up", buffer, lookback=10)
        elif sup_ok:
            retest_status = assess_retest(base, sup_lvl, "down", buffer, lookback=10)

        # Bandarmology ringkas
        bdm = bandarmology_brief(base, period=30)

        # ---------------- UI: SCORE GAUGE ----------------
        st.subheader("🎯 Hasil Analisis Cross-Confirmation")
        gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=composite, delta={'reference': 0},
            gauge={
                'axis': {'range': [-1, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [-1, -0.5], 'color': 'red'},
                    {'range': [-0.5, 0], 'color': 'lightcoral'},
                    {'range': [0, 0.5], 'color': 'lightgreen'},
                    {'range': [0.5, 1], 'color': 'green'}
                ]
            },
            title={'text': "Composite Score"}))
        gauge.update_layout(height=300, template="plotly_white")
        st.plotly_chart(gauge, use_container_width=True)
        st.info(f"**Interpretasi:** {interp}\n\n**Tingkat Keyakinan:** {confidence}")

        # ---------------- RINGKAS DATA HARGA/VOLUME ----------------
        st.subheader("📊 Data Harga & Volume Terkini")
        last_close = float(base['Close'].iloc[-1])
        prev_close = float(base['Close'].iloc[-2]) if len(base) > 1 else last_close
        change_abs = last_close - prev_close
        change_pct = (change_abs / prev_close * 100) if prev_close else 0.0
        last_vol_shares = float(base['Volume'].iloc[-1])
        last_vol_lot = shares_to_lot(last_vol_shares)
        avg_vol5_lot = shares_to_lot(base['Volume'].rolling(5).mean().iloc[-1]) if len(base) >= 5 else 0
        transaction_value = last_close * last_vol_shares  # Rp

        cA, cB, cC, cD = st.columns(4)
        with cA:
            st.write("**Last Close**")
            st.write(fmt_rp(last_close))
            arrow = "↑" if change_pct >= 0 else "↓"
            color = "green" if change_pct >= 0 else "red"
            st.markdown(f"<span style='color:{color};'>{arrow} {change_pct:.2f}% ({change_abs:.2f})</span>", unsafe_allow_html=True)
        with cB:
            st.write("**Volume (Lot)**")
            st.write(fmt_int(last_vol_lot))
        with cC:
            st.write("**Nilai Transaksi (Rp)**")
            st.write(fmt_rp(transaction_value))
        with cD:
            st.write("**Rata-rata Volume 5H (Lot)**")
            st.write(fmt_int(avg_vol5_lot))

        # ---------------- TABEL S/R & FIB ----------------
        st.subheader("📈 Support / Resistance")
        rows = []
        for i, lvl in enumerate(sr['Support']):
            rows.append({"Level": f"Support {i+1}", "Harga": fmt_rp(lvl)})
        for i, lvl in enumerate(sr['Resistance']):
            rows.append({"Level": f"Resistance {i+1}", "Harga": fmt_rp(lvl)})
        st.table(pd.DataFrame(rows))
        st.subheader("📊 Fibonacci Levels")
        fib_df = pd.DataFrame([{"Level": k, "Harga": fmt_rp(v)} for k, v in sr['Fibonacci'].items()])
        st.table(fib_df)

        # ---------------- DETAIL SKOR ----------------
        st.subheader("🔍 Detail Skor Indikator")
        ic1, ic2, ic3 = st.columns(3)
        with ic1:
            st.metric("RSI", f"{base['RSI'].iloc[-1]:.2f}", delta=f"Skor: {scores['rsi'][0]:.2f} | Str: {scores['rsi'][1]:.2f}")
            st.metric("MACD Cross", "Bullish" if scores['macd_cross'][0] > 0 else "Bearish", delta=f"Skor: {scores['macd_cross'][0]:.2f}")
        with ic2:
            st.metric("MACD Hist", "Bullish" if scores['macd_hist'][0] > 0 else "Bearish" if scores['macd_hist'][0] < 0 else "Netral",
                      delta=f"Skor: {scores['macd_hist'][0]:.2f} | Str: {scores['macd_hist'][1]:.2f}")
            st.metric("Bollinger", "Bullish" if scores['bollinger'][0] > 0 else "Bearish" if scores['bollinger'][0] < 0 else "Netral",
                      delta=f"Skor: {scores['bollinger'][0]:.2f} | Str: {scores['bollinger'][1]:.2f}")
        with ic3:
            st.metric("Volume", "Bullish" if scores['volume'][0] > 0 else "Bearish" if scores['volume'][0] < 0 else "Netral",
                      delta=f"Skor: {scores['volume'][0]:.2f} | Str: {scores['volume'][1]:.2f}")
            st.metric("OBV/ADX", f"{'Bullish' if scores['obv'][0] + scores['adx'][0] > 0 else 'Bearish' if scores['obv'][0] + scores['adx'][0] < 0 else 'Netral'}",
                      delta=f"OBV: {scores['obv'][0]:.2f}/{scores['obv'][1]:.2f} | ADX: {scores['adx'][0]:.2f}/{scores['adx'][1]:.2f}")

        # ---------------- BANDARMOLOGY RINGKAS ----------------
        st.subheader("🕵️ Bandarmology (Ringkas)")
        b1, b2, b3 = st.columns(3)
        with b1:
            st.write(f"Volume Spikes (5H): **{bdm['volume_spikes_5d']}**")
            st.write(f"ADL 5H Δ%: **{bdm['adl_5d_pct']}%**")
        with b2:
            st.write(f"Hari +Price & +Vol (30H): **{bdm['pos_volume_price_days']}**")
            st.write(f"OBV 5H Δ%: **{bdm['obv_5d_pct']}%**")
        with b3:
            st.write(f"MFI (last): **{bdm['mfi_last']}**")
            st.write(f"Value Area: **{fmt_rp(bdm['va_low'])} – {fmt_rp(bdm['va_high'])}**")

        if base['Close'].iloc[-1] > bdm['va_high']:
            st.info("🚀 Harga **di atas** Value Area → bias bullish lanjutan.")
        elif base['Close'].iloc[-1] < bdm['va_low']:
            st.warning("🔻 Harga **di bawah** Value Area → bias bearish lanjutan.")
        else:
            st.write("↔️ Harga **dalam** Value Area → konsolidasi.")

        # ---------------- DIVERGENCE (BARU) ----------------
        st.subheader("🧭 Divergence Detector (RSI & MACD)")
        div_rsi = detect_divergence(base['Close'], base['RSI'], lookback=120)
        div_macd = detect_divergence(base['Close'], base['MACD'], lookback=120)

        def _desc_div(div):
            items = []
            if div["bearish"]:
                a, b = div["bearish"]["price_points"]
                items.append(f"**Bearish** (puncak {a.date()} → {b.date()})")
            if div["bullish"]:
                a, b = div["bullish"]["price_points"]
                items.append(f"**Bullish** (lembah {a.date()} → {b.date()})")
            return " | ".join(items) if items else "Tidak terdeteksi"

        st.write(f"**RSI**: {_desc_div(div_rsi)}")
        st.write(f"**MACD**: {_desc_div(div_macd)}")

        # ---------------- REKOM / RENCANA TRADING ----------------
        st.subheader("🎯 Rekomendasi Trading")
        plan = None
        if res_ok and vol_ok:
            plan = detector.plan(base, "resistance", res_lvl, buffer, account_size, risk_percent)
        elif sup_ok and vol_ok:
            plan = detector.plan(base, "support", sup_lvl, buffer, account_size, risk_percent)

        if plan:
            st.success("🚀 **Sinyal Breakout Terdeteksi**")
            p1, p2 = st.columns(2)
            with p1:
                st.metric("Jenis", plan['type'])
                st.metric("Entry", fmt_rp(plan['entry']))
                st.metric("Stop", fmt_rp(plan['stop_loss']))
            with p2:
                st.metric("Target 1", fmt_rp(plan['target_1']))
                st.metric("Target 2", fmt_rp(plan['target_2']))
                st.metric("Risk/Reward", f"{plan['risk_reward']}:1")
            st.metric("Ukuran Posisi", f"{fmt_int(plan['position_size'])} saham ({fmt_int(plan['position_size']/100)} lot)")
            st.info(f"- Ukuran posisi mengikuti risiko **{int(risk_percent*100)}%** dari modal {fmt_rp(account_size)} "
                    f"\n- Harga sudah dibulatkan ke **tick size IDX**"
                    f"\n- Status Retest: **{retest_status}**"
                    f"\n- Pertimbangkan ambil sebagian di Target 1 dan **trail stop** untuk sisa")
        else:
            st.warning("Belum ada breakout kuat dengan konfirmasi volume.")
            st.write(f"**Resistance utama**: {fmt_rp(res_lvl)} → butuh close > {fmt_rp(res_lvl * (1 + buffer))} + volume > 1.5×MA20")
            st.write(f"**Support utama**: {fmt_rp(sup_lvl)} → butuh close < {fmt_rp(sup_lvl * (1 - buffer))} + volume > 1.5×MA20")
            st.write(f"**Status Retest:** {retest_status}")

        # ---------------- CHART ----------------
        st.subheader("📈 Chart Teknikal")
        fig = make_main_chart(base, sr, is_squeeze)
        st.plotly_chart(fig, use_container_width=True)

        # ---------------- BACKTEST RINGKAS (BARU) ----------------
        st.subheader("🧪 Backtest Ringkas (Composite Threshold)")
        bt_summary, comp_series = backtest_composite(base, threshold_long=0.4, threshold_short=-0.4, max_hold=20)
        if bt_summary["num_trades"] == 0:
            st.info("Belum ada trade yang memenuhi kriteria backtest pada horizon data saat ini.")
        else:
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1: st.metric("Jumlah Trade", bt_summary["num_trades"])
            with c2: st.metric("Win Rate", f"{bt_summary['win_rate']:.1f}%")
            with c3: st.metric("Avg R", f"{bt_summary['avg_R']:.2f}")
            with c4: st.metric("Median R", f"{bt_summary['median_R']:.2f}")
            with c5: st.metric("Rata-rata Lama (bar)", f"{bt_summary['avg_hold']:.1f}")

            # Plot composite historis
            comp_fig = go.Figure()
            comp_fig.add_trace(go.Scatter(x=comp_series.index, y=comp_series.values, mode='lines', name='Composite'))
            comp_fig.add_hline(y=0.4, line_dash="dash", line_color="green", annotation_text="Long Thresh 0.4")
            comp_fig.add_hline(y=-0.4, line_dash="dash", line_color="red", annotation_text="Short Thresh -0.4")
            comp_fig.update_layout(height=260, template="plotly_white", title="Composite Score (Historis)")
            st.plotly_chart(comp_fig, use_container_width=True)

        # ---------------- KESIMPULAN TAMBAHAN ----------------
        st.subheader("🧾 Kesimpulan Tambahan")
        bullets = []
        # Divergence
        if div_rsi["bearish"] or div_macd["bearish"]:
            bullets.append("⚠️ Terindikasi **bearish divergence** (RSI/MACD vs harga) → waspada potensi koreksi/kelemahan momentum.")
        if div_rsi["bullish"] or div_macd["bullish"]:
            bullets.append("✅ Terindikasi **bullish divergence** (RSI/MACD vs harga) → peluang rebound/awal pembalikan.")
        if not bullets:
            bullets.append("ℹ️ Tidak ada divergence signifikan yang terdeteksi pada 120 bar terakhir.")
        # Retest
        bullets.append(f"🔁 **Retest** terhadap level kunci: **{retest_status}**.")
        # Backtest
        if bt_summary["num_trades"] > 0:
            verdict = "positif" if bt_summary["avg_R"] > 0 else "negatif"
            bullets.append(f"📈 Backtest composite (±1 tahun): **{bt_summary['num_trades']} trade**, win rate **{bt_summary['win_rate']:.1f}%**, "
                           f"avg R **{bt_summary['avg_R']:.2f}** → expectancy historis **{verdict}**.")
        else:
            bullets.append("📉 Backtest tidak menemukan trade memenuhi kriteria—pertimbangkan periode data lebih panjang atau ubah threshold.")
        st.write("\n".join([f"- {b}" for b in bullets]))

        # ---------------- DISCLAIMER ----------------
        st.info("**Disclaimer**: Ini materi edukasi, bukan rekomendasi beli/jual. "
                "Selalu lakukan riset mandiri dan sesuaikan dengan profil risiko.")

if __name__ == "__main__":
    app()
