# analisa_saham.py
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta
from scipy import stats

# ====== Opsional: Model volatilitas ======
try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except Exception:
    ARCH_AVAILABLE = False

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
    try:
        return f"Rp {float(x):,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")
    except Exception:
        return str(x)

def fmt_int(x: float) -> str:
    try:
        return f"{int(round(float(x))):,}".replace(",", ".")
    except Exception:
        return str(x)

def shares_to_lot(shares: float) -> float:
    return float(shares) / 100.0

def as_series(obj) -> pd.Series:
    if isinstance(obj, pd.DataFrame):
        s = obj.squeeze("columns")
    else:
        s = obj
    if not isinstance(s, pd.Series):
        s = pd.Series(s)
    return s

def round_to_tick(price: float) -> float:
    p = float(price)
    if p < 200: tick = 1
    elif p < 500: tick = 2
    elif p < 2000: tick = 5
    elif p < 5000: tick = 10
    else: tick = 25
    return round(p / tick) * tick

# =========================
# NORMALISASI DATA YFINANCE
# =========================
def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return df

    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.droplevel(1, axis=1)
        except Exception:
            df = df.droplevel(0, axis=1)

    cols_lower = {c.lower(): c for c in df.columns}
    def pick(name):
        return cols_lower.get(name.lower())
    rename_map = {}
    for nm in ['Open','High','Low','Close','Adj Close','Volume']:
        src = pick(nm)
        if src: rename_map[src] = nm
    df = df.rename(columns=rename_map)

    if 'Close' not in df.columns and 'Adj Close' in df.columns:
        df['Close'] = df['Adj Close']

    required = ['Open','High','Low','Close','Volume']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns after normalization: {missing}")

    for c in required + (['Adj Close'] if 'Adj Close' in df.columns else []):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df[required + (['Adj Close'] if 'Adj Close' in df.columns else [])].dropna(how='all')

@st.cache_data(ttl=900, show_spinner=False)
def fetch_history_yf(ticker_jk: str, start: datetime, end: datetime) -> pd.DataFrame:
    raw = yf.download(
        ticker_jk,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False,
        group_by="column",
    )
    if raw is None or raw.empty: return raw
    return normalize_ohlcv(raw)

def resolve_ticker(user_input: str) -> str:
    t = user_input.strip().upper()
    if not t.endswith(".JK"):
        t += ".JK"
    return t

# =========================
# INDIKATOR TEKNIKAL
# =========================
def rma(series: pd.Series, period: int) -> pd.Series:
    series = as_series(series)
    return series.ewm(alpha=1/period, adjust=False).mean()

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    close = as_series(close)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = rma(gain, period) / rma(loss, period).replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def compute_mfi_corrected(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    mf = tp * df['Volume']
    pos = mf.where(tp > tp.shift(1), 0.0)
    neg = mf.where(tp < tp.shift(1), 0.0)
    pos_s = pos.rolling(period, min_periods=1).sum()
    neg_s = neg.rolling(period, min_periods=1).sum().replace(0, np.nan)
    ratio = pos_s / neg_s
    mfi = 100 - (100 / (1 + ratio))
    return as_series(mfi).fillna(50)

def compute_macd(close: pd.Series, fast=12, slow=26, signal=9):
    close = as_series(close)
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return as_series(macd), as_series(sig), as_series(hist)

def compute_bollinger_bands(close: pd.Series, window=20, num_std=2):
    close = as_series(close)
    sma = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    width = (upper - lower)
    bandwidth = width / sma.replace(0, np.nan)
    percent_b = (close - lower) / width.replace(0, np.nan)
    percent_b = percent_b.clip(0, 1)
    return as_series(upper), as_series(sma), as_series(lower), as_series(bandwidth).fillna(0), as_series(percent_b).fillna(0)

def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period=14):
    high, low, close = as_series(high), as_series(low), as_series(close)
    tr = pd.concat([
        (high - low),
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return rma(tr, period)

def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    close, volume = as_series(close), as_series(volume)
    sign = np.sign(close.diff().fillna(0))
    obv = (volume * sign).cumsum()
    return as_series(obv).fillna(0)

def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period=14):
    high, low, close = as_series(high), as_series(low), as_series(close)
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
    return as_series(adx).fillna(0), as_series(plus_di).fillna(0), as_series(minus_di).fillna(0)

def compute_vwap(high, low, close, volume):
    high, low, close, volume = as_series(high), as_series(low), as_series(close), as_series(volume)
    tp = (high + low + close) / 3
    cum_v = volume.cumsum().replace(0, np.nan)
    vwap = (tp * volume).cumsum() / cum_v
    return as_series(vwap).fillna(method="bfill").fillna(method="ffill")

def compute_adl(high, low, close, volume):
    high, low, close, volume = as_series(high), as_series(low), as_series(close), as_series(volume)
    mfm = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    mfm = mfm.replace([np.inf, -np.inf], 0).fillna(0)
    mfv = mfm * volume
    return as_series(mfv.cumsum())

# =========================
# S/R & FIBONACCI
# =========================
def identify_swings(df: pd.DataFrame, window: int = 60) -> tuple[float, float]:
    last = df.tail(window)
    return float(last['High'].max()), float(last['Low'].min())

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
    return float(min(levels, key=lambda x: abs(x - close_price)))

def cluster_levels(levels, tol=0.01):
    if not levels: return []
    levels = sorted(levels)
    clusters, cluster = [], [levels[0]]
    for x in levels[1:]:
        if abs(x - cluster[-1]) / max(cluster[-1], 1e-9) <= tol:
            cluster.append(x)
        else:
            clusters.append(cluster); cluster = [x]
    clusters.append(cluster)
    out = []
    for c in clusters:
        out.append((float(np.median(c)), len(c)))
    out.sort(key=lambda t: t[1], reverse=True)
    return out

def compute_support_resistance(df: pd.DataFrame) -> dict:
    close = float(df['Close'].iloc[-1])
    swing_high, swing_low = identify_swings(df, 60)
    fib = fibonacci_levels(swing_high, swing_low)
    ma20 = float(df['Close'].rolling(20).mean().iloc[-1])
    ma50 = float(df['Close'].rolling(50).mean().iloc[-1])
    ma100 = float(df['Close'].rolling(100).mean().iloc[-1])
    vwap = float(compute_vwap(df['High'], df['Low'], df['Close'], df['Volume']).iloc[-1])
    psych = psych_level(close)

    candidates = [fib['Fib_0.236'], fib['Fib_0.382'], fib['Fib_0.5'], fib['Fib_0.618'], fib['Fib_0.786'],
                  fib['Fib_0.0'], fib['Fib_1.0'], ma20, ma50, ma100, vwap, psych]
    candidates = [c for c in candidates if np.isfinite(c)]
    clustered = cluster_levels(candidates, tol=0.01)
    supports = [lvl for (lvl, _) in clustered if lvl < close]
    resistances = [lvl for (lvl, _) in clustered if lvl > close]
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
        s = as_series(values).dropna()
        if len(s) < max(3, period): return 0.0
        y = s.iloc[-period:].to_numpy(dtype=float).ravel()
        if y.size < 2: return 0.0
        x = np.arange(y.size, dtype=float)
        slope, _, _, _, _ = stats.linregress(x, y)
        mean = float(np.mean(y)) if np.mean(y) != 0 else 1e-9
        return float(slope / mean)

    def score_rsi(self, rsi: pd.Series):
        rsi = as_series(rsi)
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
        macd_line, signal_line, histogram = as_series(macd_line), as_series(signal_line), as_series(histogram)
        cross = 1.0 if macd_line.iloc[-1] > signal_line.iloc[-1] else -1.0
        hist_trend = self._trend(histogram, 5)
        if hist_trend > 0: hscore, hstr = 0.5, min(1.0, hist_trend*2)
        elif hist_trend < 0: hscore, hstr = -0.5, min(1.0, abs(hist_trend)*2)
        else: hscore, hstr = 0.0, 0.0
        return cross, 1.0, hscore, float(hstr)

    def score_boll(self, close, upper, lower, pct_b, bandwidth):
        pct_b, bandwidth = as_series(pct_b), as_series(bandwidth)
        cb = float(pct_b.iloc[-1]); bw = float(bandwidth.iloc[-1])
        hist_bw = bandwidth.iloc[-120:].dropna()
        bw_pct = 0.5 if len(hist_bw) < 10 else stats.percentileofscore(hist_bw.to_numpy().ravel(), bw) / 100.0
        is_squeeze = bw_pct < 0.2
        if cb > 0.8: score, strg = -1.0, min(1.0, (cb - 0.8)/0.2)
        elif cb < 0.2: score, strg = 1.0, min(1.0, (0.2 - cb)/0.2)
        elif cb > 0.5: score, strg = -0.5, (cb - 0.5)/0.3
        else: score, strg = 0.5, (0.5 - cb)/0.3
        if is_squeeze: strg = min(1.0, strg + 0.3)
        return score, float(strg), bool(is_squeeze)

    def score_volume(self, vol: pd.Series, vol_ma: pd.Series, price_change_pct: float):
        vol, vol_ma = as_series(vol), as_series(vol_ma)
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
        obv = as_series(obv)
        t = self._trend(obv, 10)
        if t > 0.05: return 1.0, min(1.0, t*5)
        if t < -0.05: return -1.0, min(1.0, abs(t)*5)
        return 0.0, 0.0

    def score_adx(self, adx: pd.Series, plus_di: pd.Series, minus_di: pd.Series, threshold=25):
        adx, plus_di, minus_di = as_series(adx), as_series(plus_di), as_series(minus_di)
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
# BREAKOUT, RETEST, BANDAR
# =========================
class BreakoutDetector:
    def __init__(self, atr_period=14, min_buffer_pct=0.005):
        self.atr_period = atr_period
        self.min_buffer_pct = min_buffer_pct

    def _dynamic_buffer(self, df: pd.DataFrame) -> float:
        atr = float(compute_atr(df['High'], df['Low'], df['Close'], self.atr_period).iloc[-1])
        close = float(df['Close'].iloc[-1])
        atr_buf = (atr / max(close, 1e-9)) * 0.5
        return max(self.min_buffer_pct, float(atr_buf))

    def detect(self, df, res_level: float, sup_level: float, bars_confirm: int = 1):
        close = as_series(df['Close'])
        vol = as_series(df['Volume'])
        buffer = self._dynamic_buffer(df)
        avg_vol = float(vol.rolling(20).mean().iloc[-1])
        vol_ok = float(vol.iloc[-1]) > 1.5 * max(avg_vol, 1e-9)

        res_break = float(close.iloc[-1]) > res_level * (1 + buffer)
        sup_break = float(close.iloc[-1]) < sup_level * (1 - buffer)

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
        atr = float(compute_atr(df['High'], df['Low'], df['Close'], self.atr_period).iloc[-1])
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

def bandarmology_brief(df: pd.DataFrame, period: int = 30) -> dict:
    out = {}
    vol = as_series(df['Volume'])
    vol_ma = vol.rolling(20).mean()
    vol_std = vol.rolling(20).std().replace(0, 1e-9)
    z = (vol - vol_ma) / vol_std
    out['volume_spikes_5d'] = int((z.iloc[-5:] > 2.5).sum())

    price_chg = as_series(df['Close']).pct_change()
    vol_chg = vol.pct_change()
    pos = ((price_chg > 0) & (vol_chg > 0)).iloc[-period:].sum()
    neg = ((price_chg < 0) & (vol_chg > 0)).iloc[-period:].sum()
    out['pos_volume_price_days'] = int(pos)
    out['neg_volume_price_days'] = int(neg)

    adl = compute_adl(df['High'], df['Low'], df['Close'], df['Volume'])
    obv = compute_obv(df['Close'], df['Volume'])
    mfi = compute_mfi_corrected(df, 14)

    def pct_change_last(s: pd.Series, n=5):
        s = as_series(s)
        if len(s) <= n or s.iloc[-n] == 0: return 0.0
        return float((s.iloc[-1] - s.iloc[-n]) / abs(s.iloc[-n]) * 100)

    out['adl_5d_pct'] = round(pct_change_last(adl, 5), 2)
    out['obv_5d_pct'] = round(pct_change_last(obv, 5), 2)
    out['mfi_last'] = round(float(as_series(mfi).iloc[-1]), 2)

    last = df.tail(20)
    price_min, price_max = float(last['Low'].min()), float(last['High'].max())
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
        acc += vol_profile[idx]; chosen.append(idx)
        if acc >= total_v * 0.7: break
    va_low = bins[min(chosen)]
    va_high = bins[max(chosen)+1]

    out['poc'] = float(poc_price)
    out['va_low'] = float(va_low)
    out['va_high'] = float(va_high)
    out['in_value_area'] = (float(df['Close'].iloc[-1]) >= va_low) and (float(df['Close'].iloc[-1]) <= va_high)
    return out

# =========================
# DIVERGENCE DETECTOR
# =========================
def _local_peaks_troughs(series: pd.Series):
    s = as_series(series)
    peaks = (s.shift(1) < s) & (s.shift(-1) < s)
    troughs = (s.shift(1) > s) & (s.shift(-1) > s)
    return s[peaks], s[troughs]

def detect_divergence(price: pd.Series, indicator: pd.Series, lookback: int = 120) -> dict:
    p = as_series(price).tail(lookback)
    i = as_series(indicator).reindex(p.index).ffill()
    p_peaks, p_troughs = _local_peaks_troughs(p)
    i_peaks, i_troughs = _local_peaks_troughs(i)
    out = {"bearish": None, "bullish": None}
    if len(p_peaks) >= 2 and len(i_peaks) >= 2:
        p_last2 = p_peaks.iloc[-2:]
        i_last2 = i_peaks.reindex(p_last2.index)
        if i_last2.notna().all():
            if (p_last2.iloc[-1] > p_last2.iloc[0]) and (i_last2.iloc[-1] < i_last2.iloc[0]):
                out["bearish"] = {"price_points": (p_last2.index[0], p_last2.index[-1])}
    if len(p_troughs) >= 2 and len(i_troughs) >= 2:
        p_last2 = p_troughs.iloc[-2:]
        i_last2 = i_troughs.reindex(p_last2.index)
        if i_last2.notna().all():
            if (p_last2.iloc[-1] < p_last2.iloc[0]) and (i_last2.iloc[-1] > i_last2.iloc[0]):
                out["bullish"] = {"price_points": (p_last2.index[0], p_last2.index[-1])}
    return out

# =========================
# BACKTEST RINGKAS (Composite)
# =========================
def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d['MA20'] = as_series(d['Close']).rolling(20).mean()
    d['MA50'] = as_series(d['Close']).rolling(50).mean()
    d['MA100'] = as_series(d['Close']).rolling(100).mean()
    d['RSI'] = compute_rsi(d['Close'], 14)
    d['MACD'], d['Signal'], d['Hist'] = compute_macd(d['Close'], 12, 26, 9)
    d['BB_Upper'], d['BB_Middle'], d['BB_Lower'], d['BB_BW'], d['BB_%B'] = compute_bollinger_bands(d['Close'], 20, 2)
    d['ATR'] = compute_atr(d['High'], d['Low'], d['Close'], 14)
    d['OBV'] = compute_obv(d['Close'], d['Volume'])
    d['ADX'], d['Plus_DI'], d['Minus_DI'] = compute_adx(d['High'], d['Low'], d['Close'], 14)
    d['VWAP'] = compute_vwap(d['High'], d['Low'], d['Close'], d['Volume'])
    d['MFI'] = compute_mfi_corrected(d, 14)
    d['Volume_MA20'] = as_series(d['Volume']).rolling(20).mean()
    return d

def backtest_composite(df: pd.DataFrame, threshold_long=0.4, threshold_short=-0.4, max_hold=20):
    scorer = IndicatorScoringSystem()
    comp = []
    for idx in range(len(df)):
        if idx < 100:
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
        price_chg_pct = float((window['Close'].iloc[-1] - window['Close'].iloc[-2]) /
                              max(window['Close'].iloc[-2], 1e-9) * 100)
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
        price = float(df['Close'].iloc[i])
        atr = float(df['ATR'].iloc[i])
        enter_long = comp.iloc[i-1] <= threshold_long and comp.iloc[i] > threshold_long
        enter_short = comp.iloc[i-1] >= threshold_short and comp.iloc[i] < threshold_short
        if enter_long:
            entry = price; stop = entry - 2*atr; target = entry + 2*atr
            stop_dist = entry - stop; j = i+1; exit_price = None; reason = None
            while j < min(i+max_hold, len(df)):
                high = float(df['High'].iloc[j]); low = float(df['Low'].iloc[j]); cl = float(df['Close'].iloc[j])
                if low <= stop: exit_price, reason = stop, "Stop"; break
                if high >= target: exit_price, reason = target, "Target"; break
                if comp.iloc[j] <= 0: exit_price, reason = cl, "NeutralExit"; break
                j += 1
            if exit_price is None:
                exit_price, reason = float(df['Close'].iloc[min(i+max_hold-1, len(df)-1)]), "Timeout"
            R = (exit_price - entry) / max(stop_dist, 1e-9)
            trades.append({"dir":"Long","entry_i":i,"exit_i":j,"R":R,"reason":reason})
            i = j + 1
        elif enter_short:
            entry = price; stop = entry + 2*atr; target = entry - 2*atr
            stop_dist = stop - entry; j = i+1; exit_price = None; reason = None
            while j < min(i+max_hold, len(df)):
                high = float(df['High'].iloc[j]); low = float(df['Low'].iloc[j]); cl = float(df['Close'].iloc[j])
                if high >= stop: exit_price, reason = stop, "Stop"; break
                if low <= target: exit_price, reason = target, "Target"; break
                if comp.iloc[j] >= 0: exit_price, reason = cl, "NeutralExit"; break
                j += 1
            if exit_price is None:
                exit_price, reason = float(df['Close'].iloc[min(i+max_hold-1, len(df)-1)]), "Timeout"
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
# GARCH – PROYEKSI 1..5 HARI
# =========================
@st.cache_data(ttl=900, show_spinner=False)
def garch_forecast_paths(adj_close: pd.Series, sims=10000, horizon=5, seed=42):
    """
    Mengembalikan:
      - dict summary (agregat 5H)
      - dict fan_prices (harga kuantil per hari untuk fan chart)
      - r5 (return kumulatif 5H, proporsi)
      - paths_p (matrix harga simulasi, shape [sims, horizon])
      - paths_r (matrix return harian simulasi, proporsi, shape [sims, horizon])  <-- BARU
      - return_quantiles (dict {5,25,50,75,95} -> array kuantil return harian [horizon])  <-- BARU
    """
    if not ARCH_AVAILABLE:
        raise RuntimeError("Paket 'arch' tidak tersedia. Jalankan: pip install arch")

    px = as_series(adj_close).dropna()
    if len(px) < 250:
        raise RuntimeError("Data terlalu pendek (< 250 bar) untuk GARCH yang stabil.")

    r = np.log(px/px.shift(1)).dropna()
    am = arch_model(r*100, vol="GARCH", p=1, q=1, dist="t", mean="Zero")
    res = am.fit(disp="off")

    np.random.seed(seed)
    nu = float(res.params['nu'])
    omega = float(res.params['omega'])
    alpha = float(res.params['alpha[1]'])
    beta  = float(res.params['beta[1]'])

    h_last = float(res.conditional_volatility.iloc[-1]**2)  # %^2

    sims = int(sims)
    horizon = int(horizon)
    paths_r_pct = np.zeros((sims, horizon))   # dalam %
    paths_p = np.zeros((sims, horizon))       # harga

    p0 = float(px.iloc[-1])
    var = np.full(sims, h_last)               # %^2
    price = np.full(sims, p0)
    for t in range(horizon):
        z = np.random.standard_t(nu, size=sims) * np.sqrt((nu-2)/nu)
        sigma = np.sqrt(var)                  # %
        eps = sigma * z                       # shock %
        paths_r_pct[:, t] = eps
        price = price * np.exp(eps/100.0)
        paths_p[:, t] = price
        var = omega + alpha*(eps**2) + beta*var

    # Kuantil harian (return & harga)
    qs = [5, 25, 50, 75, 95]
    # Return harian dalam proporsi
    paths_r = paths_r_pct / 100.0
    return_quantiles = {q: np.percentile(paths_r, q, axis=0) for q in qs}
    # Kumulatif return utk fan chart harga (pakai quantile-of-sum approx by cum of daily quantiles)
    quant_r_for_fan = {q: np.percentile(paths_r, q, axis=0) for q in qs}
    cum_q = {q: np.cumsum(quant_r_for_fan[q], axis=0) for q in qs}
    fan_prices = {q: p0 * np.exp(cum_q[q]) for q in qs}

    # Agregat 5H
    r5 = paths_r.sum(axis=1)
    p5 = p0 * np.exp(r5)
    summary = {
        "P5_return": float(np.percentile(r5, 5)),
        "P50_return": float(np.percentile(r5, 50)),
        "P95_return": float(np.percentile(r5, 95)),
        "P5_price": float(np.percentile(p5, 5)),
        "P50_price": float(np.percentile(p5, 50)),
        "P95_price": float(np.percentile(p5, 95)),
        "VaR95_return": float(np.percentile(r5, 5)),
        "CVaR95_return": float(r5[r5 <= np.percentile(r5,5)].mean()),
        "p0": p0
    }
    return summary, fan_prices, r5, paths_p, paths_r, return_quantiles

def garch_prob_level(paths_prices: np.ndarray, levels: dict) -> dict:
    sims, horizon = paths_prices.shape
    out = {}
    if levels.get("target"):
        target = float(levels["target"])
        out["prob_up_by_day"] = [(paths_prices[:, d] >= target).mean() for d in range(horizon)]
        out["prob_up_5d"] = (paths_prices[:, -1] >= target).mean()
    if levels.get("stop"):
        stop = float(levels["stop"])
        out["prob_down_by_day"] = [(paths_prices[:, d] <= stop).mean() for d in range(horizon)]
        out["prob_down_5d"] = (paths_prices[:, -1] <= stop).mean()
    return out

# =========================
# CHART
# =========================
def make_main_chart(df: pd.DataFrame, sr: dict, is_squeeze: bool) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name="Candlestick", increasing_line_color='green', decreasing_line_color='red'
    ))
    for name, col, color in [("MA20",'MA20','blue'), ("MA50",'MA50','orange'), ("MA100",'MA100','purple')]:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=name, line=dict(color=color, width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name="BB Upper", mode="lines", line=dict(color="red", width=1, dash="dot")))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Middle'], name="BB Middle", mode="lines", line=dict(color="purple", width=1.2)))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name="BB Lower", mode="lines", line=dict(color="green", width=1, dash="dot"),
                             fill='tonexty', fillcolor='rgba(173,216,230,0.1)'))
    for i, lvl in enumerate(sr['Support']):
        fig.add_hline(y=lvl, line_dash="dash", line_color="green",
                      annotation_text=f"Support {i+1}: {fmt_rp(lvl)}", annotation_position="bottom right")
    for i, lvl in enumerate(sr['Resistance']):
        fig.add_hline(y=lvl, line_dash="dash", line_color="red",
                      annotation_text=f"Resistance {i+1}: {fmt_rp(lvl)}", annotation_position="top right")
    if is_squeeze:
        fig.add_annotation(x=df.index[-1], y=df['Close'].iloc[-1], text="BOLLINGER SQUEEZE",
                           showarrow=True, arrowhead=1, arrowcolor="purple", font=dict(color="purple"))
    fig.update_layout(title="Chart Teknikal", template="plotly_white", xaxis_rangeslider_visible=False,
                      height=700, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                      hovermode='x unified')
    return fig

def make_fan_chart(fan_prices: dict) -> go.Figure:
    fig = go.Figure()
    h = len(fan_prices[50])
    x = list(range(1, h+1))
    fig.add_trace(go.Scatter(x=x, y=fan_prices[50], mode="lines", name="Median"))
    fig.add_trace(go.Scatter(x=x, y=fan_prices[95], mode="lines", name="P95", line=dict(width=0)))
    fig.add_trace(go.Scatter(x=x, y=fan_prices[5], mode="lines", name="P5", fill="tonexty", line=dict(width=0)))
    fig.add_trace(go.Scatter(x=x, y=fan_prices[75], mode="lines", name="P75", line=dict(width=0)))
    fig.add_trace(go.Scatter(x=x, y=fan_prices[25], mode="lines", name="P25", fill="tonexty", line=dict(width=0)))
    fig.update_layout(title="Forecast Probabilistik Harga 1–5 Hari (GARCH(1,1)-t)",
                      xaxis_title="Hari ke-", yaxis_title="Harga (Rp)", template="plotly_white", height=380)
    return fig

# =========================
# APP STREAMLIT
# =========================
def app():
    st.set_page_config(page_title="Analisa Saham IDX: Teknikal + Probabilistik 5H", layout="wide")
    st.title("📊 Analisa Saham IDX – Teknikal + Proyeksi Probabilistik 1–5 Hari")

    c1, c2 = st.columns(2)
    with c1:
        ticker_in = st.text_input("Kode Saham", value="BBCA").strip().upper()
        account_size = st.number_input("Modal (Rp)", value=100_000_000, step=10_000_000)
        risk_percent = st.slider("Risiko per Trade (%)", 0.5, 5.0, 2.0) / 100
    with c2:
        as_of = st.date_input("📅 Tanggal Analisis (as-of)", value=datetime.today())
        use_mtf = st.checkbox("Gunakan Konfirmasi Weekly", value=True)

    adv = st.expander("Pengaturan Lanjutan (Forecast)")
    with adv:
        sims = st.slider("Jumlah simulasi GARCH", 2000, 20000, 10000, step=1000)
        horizon = st.slider("Horizon hari trading", 3, 5, 5)
        user_target = st.number_input("Target harga (opsional; default: Resistance 1)", value=0.0, step=10.0)
        user_stop = st.number_input("Stop harga (opsional; default: Support 1)", value=0.0, step=10.0)

    if st.button("🚀 Mulai Analisis"):
        if not ticker_in:
            st.warning("Masukkan kode saham terlebih dahulu."); return

        ticker = resolve_ticker(ticker_in)
        end_all = datetime.combine(as_of, datetime.min.time()) + timedelta(days=1)
        start_all = end_all - timedelta(days=3*365)

        data_all = fetch_history_yf(ticker, start_all, end_all)
        if data_all is None or data_all.empty:
            st.warning("Data tidak tersedia. Coba kode atau tanggal lain."); return

        df_hist = data_all[data_all.index <= pd.Timestamp(as_of) + pd.Timedelta(days=1)]
        if len(df_hist) < 120:
            st.warning("Data historis terlalu pendek (<120 bar). Tambah rentang tanggal."); return

        st.caption(f"Data terakhir sampai as-of: **{df_hist.index[-1].date()}** (n={len(df_hist)})")

        # ======== INDICATOR PACK =========
        base = compute_all_indicators(df_hist)

        # Weekly bias (opsional)
        weekly_bias = 0
        if use_mtf:
            try:
                dfw = df_hist.resample('W-FRI').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
                if len(dfw) >= 30:
                    macd_w, sig_w, _ = compute_macd(dfw['Close'])
                    weekly_bias = 1 if float(macd_w.iloc[-1]) > float(sig_w.iloc[-1]) else -1
            except Exception as e:
                st.warning(f"Konfirmasi weekly dimatikan: {e}")
                weekly_bias = 0

        # S/R
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

        # Breakout + Retest
        detector = BreakoutDetector()
        res_lvl = sr['Resistance'][0] if sr['Resistance'] else float(base['Close'].iloc[-1]) * 1.05
        sup_lvl = sr['Support'][0] if sr['Support'] else float(base['Close'].iloc[-1]) * 0.95
        res_ok, sup_ok, vol_ok, buffer = detector.detect(base, res_lvl, sup_lvl, bars_confirm=1)
        # Bandarmology & Divergence
        bdm = bandarmology_brief(base, period=30)
        div_rsi = detect_divergence(base['Close'], base['RSI'], lookback=120)
        div_macd = detect_divergence(base['Close'], base['MACD'], lookback=120)

        # ======== OUTPUT TEKNIKAL =========
        st.subheader("🎯 Hasil Analisa Cross-Confirmation (as-of)")
        gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=composite, delta={'reference': 0},
            gauge={'axis': {'range': [-1, 1]},
                   'bar': {'color': "darkblue"},
                   'steps': [{'range': [-1, -0.5], 'color': 'red'},
                             {'range': [-0.5, 0], 'color': 'lightcoral'},
                             {'range': [0, 0.5], 'color': 'lightgreen'},
                             {'range': [0.5, 1], 'color': 'green'}]},
            title={'text': "Composite Score"}))
        gauge.update_layout(height=300, template="plotly_white")
        st.plotly_chart(gauge, use_container_width=True)
        st.info(f"**Interpretasi:** {interp} • **Keyakinan:** {confidence}")

        # Ringkas harga & volume
        st.subheader("📊 Ringkas Harga & Volume (as-of)")
        last_close = float(base['Close'].iloc[-1])
        prev_close = float(base['Close'].iloc[-2]) if len(base) > 1 else last_close
        change_abs = last_close - prev_close
        change_pct = (change_abs / prev_close * 100) if prev_close else 0.0
        last_vol_shares = float(base['Volume'].iloc[-1])
        last_vol_lot = shares_to_lot(last_vol_shares)
        avg_vol5_lot = shares_to_lot(as_series(base['Volume']).rolling(5).mean().iloc[-1]) if len(base) >= 5 else 0
        transaction_value = last_close * last_vol_shares

        cA, cB, cC, cD = st.columns(4)
        with cA:
            st.write("**Last Close**"); st.write(fmt_rp(last_close))
            arrow = "↑" if change_pct >= 0 else "↓"; color = "green" if change_pct >= 0 else "red"
            st.markdown(f"<span style='color:{color};'>{arrow} {change_pct:.2f}% ({change_abs:.2f})</span>", unsafe_allow_html=True)
        with cB: st.write("**Volume (Lot)**"); st.write(fmt_int(last_vol_lot))
        with cC: st.write("**Nilai Transaksi (Rp)**"); st.write(fmt_rp(transaction_value))
        with cD: st.write("**Rata-rata Volume 5H (Lot)**"); st.write(fmt_int(avg_vol5_lot))

        # SR & Fib
        st.subheader("📈 Support / Resistance (as-of)")
        rows = []
        for i, lvl in enumerate(sr['Support']): rows.append({"Level": f"Support {i+1}", "Harga": fmt_rp(lvl)})
        for i, lvl in enumerate(sr['Resistance']): rows.append({"Level": f"Resistance {i+1}", "Harga": fmt_rp(lvl)})
        st.table(pd.DataFrame(rows))
        st.subheader("📊 Fibonacci Levels")
        st.table(pd.DataFrame([{"Level": k, "Harga": fmt_rp(v)} for k, v in sr['Fibonacci'].items()]))

        # Detail skor
        st.subheader("🔍 Detail Skor Indikator")
        ic1, ic2, ic3 = st.columns(3)
        with ic1:
            st.metric("RSI", f"{float(base['RSI'].iloc[-1]):.2f}", delta=f"Skor: {scores['rsi'][0]:.2f} | Str: {scores['rsi'][1]:.2f}")
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

        # Bandarmology
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
            st.write("Status:", "📦 konsolidasi." if bdm['in_value_area'] else ("🚀 di atas VA" if last_close > bdm['va_high'] else "🔻 di bawah VA"))

        # Divergence
        st.subheader("🧭 Divergence (RSI & MACD)")
        def _desc_div(div):
            items = []
            if div["bearish"]:
                a, b = div["bearish"]["price_points"]; items.append(f"**Bearish** (puncak {a.date()} → {b.date()})")
            if div["bullish"]:
                a, b = div["bullish"]["price_points"]; items.append(f"**Bullish** (lembah {a.date()} → {b.date()})")
            return " | ".join(items) if items else "Tidak terdeteksi"
        st.write(f"**RSI**: {_desc_div(div_rsi)}")
        st.write(f"**MACD**: {_desc_div(div_macd)}")

        # Rekomendasi Trading (as-of)
        st.subheader("🎯 Rekomendasi Trading (as-of)")
        plan = None
        if res_ok and vol_ok:
            plan = detector.plan(base, "resistance", res_lvl, buffer, account_size, risk_percent)
        elif sup_ok and vol_ok:
            plan = detector.plan(base, "support", sup_lvl, buffer, account_size, risk_percent)
        if plan:
            st.success("🚀 **Sinyal Breakout Terdeteksi (as-of)**")
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
            st.info(f"- Risiko **{int(risk_percent*100)}%** dari modal {fmt_rp(account_size)}"
                    f"\n- Dibulatkan ke **tick size IDX**"
                    f"\n- Retest: (as-of) belum dapat dinilai sesudahnya")
        else:
            st.warning("Belum ada breakout kuat (as-of).")
            st.write(f"**Resistance utama**: {fmt_rp(res_lvl)} → butuh close > {fmt_rp(res_lvl * (1 + buffer))} + volume > 1.5×MA20")
            st.write(f"**Support utama**: {fmt_rp(sup_lvl)} → butuh close < {fmt_rp(sup_lvl * (1 - buffer))} + volume > 1.5×MA20")

        # Chart teknikal
        st.subheader("📈 Chart Teknikal (as-of)")
        fig_main = make_main_chart(base, sr, is_squeeze)
        st.plotly_chart(fig_main, use_container_width=True)

        # ======== PROYEKSI PROBABILISTIK 1..5 HARI (as-of) ========
        st.subheader("🔮 Proyeksi Probabilistik 1–5 Hari (GARCH) – as-of")
        st.caption("Fan chart: **garis tengah = median (P50)**, **area abu-abu = band P5–P95**. Ini *bukan jalur harga pasti*, melainkan **rentang kemungkinan**; biasanya median ≈ datar seperti random walk, sementara pita melebar seiring h bertambah (volatilitas terakumulasi).")

        if not ARCH_AVAILABLE:
            st.error("Paket 'arch' belum terpasang. Jalankan: pip install arch")
        else:
            px_series = df_hist['Adj Close'] if 'Adj Close' in df_hist.columns else df_hist['Close']

            try:
                summary, fan_prices, r5, paths_p, paths_r, return_quantiles = garch_forecast_paths(px_series, sims=sims, horizon=horizon, seed=42)
            except Exception as e:
                st.error(f"Gagal menghitung GARCH: {e}")
                paths_p = None; fan_prices = None; summary = None; paths_r = None; return_quantiles = None

            if summary and fan_prices is not None:
                # Tabel kuantil HARGA per hari
                rows_price = []
                for d in range(horizon):
                    rows_price.append({
                        "Hari ke-": d+1,
                        "P5": fmt_rp(fan_prices[5][d]),
                        "P50": fmt_rp(fan_prices[50][d]),
                        "P95": fmt_rp(fan_prices[95][d]),
                    })
                st.table(pd.DataFrame(rows_price))

                # Fan chart
                fan_fig = make_fan_chart(fan_prices)
                st.plotly_chart(fan_fig, use_container_width=True)

                # ======= TAMBAHAN: Probabilitas naik/turun harian & kuantil RETURN harian =======
                if paths_r is not None and return_quantiles is not None:
                    # Prob naik/turun per hari
                    prob_rows = []
                    for d in range(paths_r.shape[1]):
                        p_up = float((paths_r[:, d] > 0).mean())
                        p_down = 1.0 - p_up
                        prob_rows.append({"Hari ke-": d+1,
                                          "Pr(r>0)": f"{p_up*100:.1f}%",
                                          "Pr(r<0)": f"{p_down*100:.1f}%"})
                    st.write("**Probabilitas return harian** (dari simulasi):")
                    st.table(pd.DataFrame(prob_rows))

                    # Kuantil return harian (dalam %)
                    ret_rows = []
                    for d in range(paths_r.shape[1]):
                        ret_rows.append({
                            "Hari ke-": d+1,
                            "P5": f"{return_quantiles[5][d]*100:.2f}%",
                            "P50": f"{return_quantiles[50][d]*100:.2f}%",
                            "P95": f"{return_quantiles[95][d]*100:.2f}%"
                        })
                    st.write("**Rentang return harian** (kuantil):")
                    st.table(pd.DataFrame(ret_rows))

                # Prob > target/stop (default SR1/S1 jika kosong)
                target_level = user_target if user_target > 0 else sr['Resistance'][0]
                stop_level = user_stop if user_stop > 0 else sr['Support'][0]
                probs = garch_prob_level(paths_p, {"target": target_level, "stop": stop_level})

                colA, colB, colC = st.columns(3)
                with colA:
                    st.metric("Harga as-of", fmt_rp(summary["p0"]))
                with colB:
                    st.metric("Target (default: R1)", fmt_rp(target_level))
                with colC:
                    st.metric("Stop (default: S1)", fmt_rp(stop_level))

                if "prob_up_by_day" in probs:
                    st.write(f"Prob. **≥ Target** per-hari: " +
                             ", ".join([f"D{d+1}: {p*100:.1f}%" for d, p in enumerate(probs['prob_up_by_day'])]) +
                             f" • **5H**: {probs['prob_up_5d']*100:.1f}%")
                if "prob_down_by_day" in probs:
                    st.write(f"Prob. **≤ Stop** per-hari: " +
                             ", ".join([f"D{d+1}: {p*100:.1f}%" for d, p in enumerate(probs['prob_down_by_day'])]) +
                             f" • **5H**: {probs['prob_down_5d']*100:.1f}%")

                st.caption(f"5H VaR95: {summary['VaR95_return']*100:.2f}%,  CVaR95: {summary['CVaR95_return']*100:.2f}%")

                # ======== EVALUASI (jika ada data setelah as_of) ========
                df_future = data_all[data_all.index > df_hist.index[-1]].head(horizon)
                if not df_future.empty:
                    st.subheader("🧪 Evaluasi Akurasi (as-of vs realisasi)")
                    realized = df_future['Adj Close'] if 'Adj Close' in df_future.columns else df_future['Close']
                    cover = []
                    mae = []
                    for d in range(len(realized)):
                        r = float(realized.iloc[d])
                        lo, md, hi = fan_prices[5][d], fan_prices[50][d], fan_prices[95][d]
                        cover.append(1.0 if (lo <= r <= hi) else 0.0)
                        mae.append(abs(r - md) / md)
                    cov_rate = np.mean(cover) * 100 if cover else 0.0
                    mae_pct = np.mean(mae) * 100 if mae else 0.0
                    c1, c2 = st.columns(2)
                    with c1: st.metric("Coverage band 90% (hari real tersedia)", f"{cov_rate:.1f}%")
                    with c2: st.metric("MAE vs Median", f"{mae_pct:.2f}%")
                    rows_eval = []
                    idxs = realized.index
                    for d in range(len(realized)):
                        rows_eval.append({
                            "Tanggal": str(idxs[d].date()),
                            "Real": fmt_rp(realized.iloc[d]),
                            "Band90%": f"{fmt_rp(fan_prices[5][d])} – {fmt_rp(fan_prices[95][d])}",
                            "Dalam Band?": "✅" if cover[d] == 1.0 else "❌",
                            "Abs Err vs P50": f"{mae[d]*100:.2f}%",
                        })
                    st.table(pd.DataFrame(rows_eval))

        # Disclaimer
        st.info("**Disclaimer**: Edukasi, bukan rekomendasi. Trading mengandung risiko. Gunakan kebijaksanaan dan sesuaikan dengan profil risiko Anda.")

if __name__ == "__main__":
    app()
