# analisa_swing_narasi.py
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta
from scipy import stats

# =========================
# KONFIG & KONSTANTA
# =========================
SR_WINDOW = 60               # S/R & Fibonacci dari 60 bar (≈ 3 bulan)
PROFILE_DAYS = 20           # Volume Profile window (hari)
VOL_ANOMALY_MULT = 1.5      # Volume anomali = ≥ 1.5 × MA20
VIEW_DAYS = 90              # jendela tampilan chart (≤ 3 bulan)

# =========================
# UTILITAS FORMAT & IDX
# =========================
def fmt_rp(x: float) -> str:
    try:
        return f"Rp {float(x):,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")
    except Exception:
        return str(x)

def fmt_rp_short(x: float) -> str:
    """Format singkat untuk anotasi/label agar tidak kepotong."""
    try:
        n = abs(float(x))
    except Exception:
        return str(x)
    s = "-" if x < 0 else ""
    if n >= 1_000_000_000_000:
        return f"{s}Rp {n/1_000_000_000_000:.2f}T"
    if n >= 1_000_000_000:
        return f"{s}Rp {n/1_000_000_000:.2f}M"
    if n >= 1_000_000:
        return f"{s}Rp {n/1_000_000:.2f}Jt"
    return f"{s}Rp {n:,.0f}".replace(",", ".")

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

def tick_size(price: float) -> int:
    p = float(price)
    if p < 200: return 1
    elif p < 500: return 2
    elif p < 2000: return 5
    elif p < 5000: return 10
    else: return 25

def resolve_ticker(user_input: str) -> str:
    t = user_input.strip().upper()
    if not t.endswith(".JK"):
        t += ".JK"
    return t

# =========================
# FETCH DATA
# =========================
def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return df
    if isinstance(df.columns, pd.MultiIndex):
        try: df = df.droplevel(1, axis=1)
        except Exception: df = df.droplevel(0, axis=1)
    cols_lower = {c.lower(): c for c in df.columns}
    def pick(name): return cols_lower.get(name.lower())
    rename_map = {}
    for nm in ['Open','High','Low','Close','Adj Close','Volume']:
        src = pick(nm)
        if src: rename_map[src] = nm
    df = df.rename(columns=rename_map)
    if 'Close' not in df.columns and 'Adj Close' in df.columns:
        df['Close'] = df['Adj Close']
    required = ['Open','High','Low','Close','Volume']
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
    if raw is None or raw.empty:
        return raw
    return normalize_ohlcv(raw)

@st.cache_data(ttl=600, show_spinner=False)
def fetch_intraday_5m_yf(ticker_jk: str, days: int = 30) -> pd.DataFrame:
    period = f"{min(days, 60)}d"   # batas 5m yfinance ≈ 60 hari
    raw = yf.download(
        ticker_jk,
        period=period,
        interval="5m",
        auto_adjust=False,
        progress=False,
        group_by="column",
        prepost=False
    )
    if raw is None or raw.empty:
        return raw
    return normalize_ohlcv(raw)

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
    up = high.diff(); down = -low.diff()
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

# =========================
# S/R & FIBONACCI
# =========================
def identify_swings(df: pd.DataFrame, window: int = SR_WINDOW) -> tuple[float, float]:
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

def compute_support_resistance(df: pd.DataFrame, window: int = SR_WINDOW) -> dict:
    dfw = df.tail(window)
    close = float(dfw['Close'].iloc[-1])
    swing_high, swing_low = identify_swings(dfw, window)
    fib = fibonacci_levels(swing_high, swing_low)
    ma20 = float(dfw['Close'].rolling(20).mean().iloc[-1])
    ma50 = float(dfw['Close'].rolling(50).mean().iloc[-1])
    ma100 = float(dfw['Close'].rolling(100).mean().iloc[-1]) if len(dfw) >= 100 else np.nan
    vwap = float(compute_vwap(dfw['High'], dfw['Low'], dfw['Close'], dfw['Volume']).iloc[-1])
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
INDICATOR_WEIGHTS = {
    'rsi': 0.15, 'macd_cross': 0.25, 'macd_hist': 0.10,
    'bollinger': 0.15, 'volume': 0.20, 'obv': 0.10, 'adx': 0.05
}

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
        rsi = as_series(rsi); r = float(rsi.iloc[-1])
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
        if ratio > 1.7 and price_change_pct > 0:   score, strg = 1.0, min(1.0, (ratio - 1.7) / 0.8)
        elif ratio > 1.7 and price_change_pct < 0: score, strg = -1.0, min(1.0, (ratio - 1.7) / 0.8)
        elif ratio > 1.3 and price_change_pct > 0: score, strg = 0.5, (ratio - 1.3) / 0.4
        elif ratio > 1.3 and price_change_pct < 0: score, strg = -0.5, (ratio - 1.3) / 0.4
        else: score, strg = 0.0, 0.0
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
        if not scores: return "Rendah"
        if composite_score > 0:
            agree = sum(1 for (s, stg) in scores.values() if (s > 0 and stg > 0.3))
        elif composite_score < 0:
            agree = sum(1 for (s, stg) in scores.values() if (s < 0 and stg > 0.3))
        else:
            agree = 0
        ratio = agree / max(len(scores), 1)
        if ratio >= 0.7: return "Tinggi"
        if ratio >= 0.5: return "Sedang"
        return "Rendah"

    @staticmethod
    def interpret(score: float) -> str:
        if score >= 0.7:  return "Sangat Bullish - Sentimen beli sangat kuat"
        if score >= 0.4:  return "Bullish Kuat - Sentimen beli kuat"
        if score >= 0.1:  return "Bullish Lemah - Sentimen cenderung beli"
        if score >  -0.1: return "Netral - Sentimen tidak jelas"
        if score >  -0.4: return "Bearish Lemah - Sentimen cenderung jual"
        if score >  -0.7: return "Bearish Kuat - Sentimen jual kuat"
        return "Sangat Bearish - Sentimen jual sangat kuat"

# =========================
# BREAKOUT & FALSE BREAK
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
        close = as_series(df['Close']); vol = as_series(df['Volume'])
        buffer = self._dynamic_buffer(df)
        avg_vol = float(vol.rolling(20).mean().iloc[-1])
        vol_ok = float(vol.iloc[-1]) >= VOL_ANOMALY_MULT * max(avg_vol, 1e-9)

        res_break = float(close.iloc[-1]) > res_level * (1 + buffer)
        sup_break = float(close.iloc[-1]) < sup_level * (1 - buffer)

        def confirm(level, direction):
            if bars_confirm <= 1: return True
            if direction == "up":   return (close.iloc[-bars_confirm:] > level).all()
            else:                   return (close.iloc[-bars_confirm:] < level).all()

        res_ok = res_break and confirm(res_level * (1 + buffer), "up")
        sup_ok = sup_break and confirm(sup_level * (1 - buffer), "down")
        return res_ok, sup_ok, vol_ok, buffer

def detect_false_breaks(df: pd.DataFrame, sr: dict, lookback: int = SR_WINDOW, atr_period: int = 14):
    if df is None or df.empty: return []
    win = df.tail(lookback).copy()
    atr = compute_atr(win['High'], win['Low'], win['Close'], atr_period)
    close = win['Close']; high = win['High']; low = win['Low']; vol = win['Volume']
    vol_ma20 = vol.rolling(20).mean()
    eps = 1e-9
    buf_pct = np.maximum((atr / np.maximum(close, eps)) * 0.5, 0.005)
    events = []
    levels_up   = sr.get('Resistance', [])[:3]
    levels_down = sr.get('Support', [])[:3]
    for dt in win.index:
        i = win.index.get_loc(dt)
        b = float(buf_pct.iloc[i]) if i < len(buf_pct) else 0.01
        v_ratio = float(vol.iloc[i] / max(vol_ma20.iloc[i], 1.0))
        for lvl in levels_up:
            if (float(high.iloc[i]) > lvl * (1 + b)) and (float(close.iloc[i]) <= lvl):
                events.append({"date": dt, "type": "false_break_up", "level": float(lvl),
                               "high": float(high.iloc[i]), "close": float(close.iloc[i]),
                               "buffer_pct": b, "vol_ratio": v_ratio})
                break
        for lvl in levels_down:
            if (float(low.iloc[i]) < lvl * (1 - b)) and (float(close.iloc[i]) >= lvl):
                events.append({"date": dt, "type": "false_break_down", "level": float(lvl),
                               "low": float(low.iloc[i]), "close": float(close.iloc[i]),
                               "buffer_pct": b, "vol_ratio": v_ratio})
                break
    return events

# =========================
# VOLUME PROFILE (20H, TF 5-minute) – VERTICAL CHART STYLE
# =========================
def compute_volume_profile_vertical(df_price: pd.DataFrame,
                                    ticker_jk: str,
                                    days: int = PROFILE_DAYS):
    """
    Output untuk chart vertikal:
      - volume_profile: array [nbins] (lembar)
      - bin_edges: array [nbins+1] (harga Rupiah, sumbu-X)
      - poc_price, value_area_low, value_area_high
      - in_value_area (bool), total_volume (lembar)
    """
    intr = fetch_intraday_5m_yf(ticker_jk, days=days+5)
    data = intr if intr is not None and not intr.empty else df_price.copy()
    data = data.tail(days)

    if data.empty or len(data) < 5:
        return {"volume_profile": np.array([]), "bin_edges": np.array([]),
                "poc_price": np.nan, "value_area_low": np.nan, "value_area_high": np.nan,
                "in_value_area": False, "total_volume": 0.0}

    lo_raw = float(np.nanpercentile(data['Low'], 1))
    hi_raw = float(np.nanpercentile(data['High'], 99))
    mid = float(data['Close'].tail(50).mean())
    step = max(tick_size(mid), (hi_raw - lo_raw) / 35.0)
    lo = np.floor(lo_raw / step) * step
    hi = np.ceil(hi_raw / step) * step
    if hi <= lo:
        hi = lo + step
    bin_edges = np.arange(lo, hi + step, step)
    vols = np.zeros(len(bin_edges) - 1, dtype=float)

    for _, r in data.iterrows():
        low = float(r['Low']); high = float(r['High']); vol = float(r['Volume'])
        if vol <= 0: continue
        if high <= low:
            c = float(r['Close'])
            idx = int(np.clip(np.searchsorted(bin_edges, c) - 1, 0, len(vols)-1))
            vols[idx] += vol; continue
        start = int(np.clip(np.searchsorted(bin_edges, low) - 1, 0, len(vols)-1))
        end   = int(np.clip(np.searchsorted(bin_edges, high) - 1, 0, len(vols)-1))
        rng = max(high - low, 1e-9)
        for i in range(start, end+1):
            seg_lo = bin_edges[i]; seg_hi = bin_edges[i+1]
            ov_len = max(0.0, min(high, seg_hi) - max(low, seg_lo))
            if ov_len > 0: vols[i] += vol * (ov_len / rng)

    total_v = float(vols.sum())
    if total_v <= 0:
        return {"volume_profile": vols, "bin_edges": bin_edges,
                "poc_price": np.nan, "value_area_low": np.nan, "value_area_high": np.nan,
                "in_value_area": False, "total_volume": 0.0}

    poc_idx = int(np.argmax(vols))
    poc_price = float((bin_edges[poc_idx] + bin_edges[poc_idx+1]) / 2)

    order = np.argsort(vols)[::-1]; cum = 0.0; picked = []
    for i in order:
        picked.append(i); cum += vols[i]
        if cum >= total_v * 0.7: break
    va_low = float(bin_edges[min(picked)])
    va_high = float(bin_edges[max(picked)+1])

    close_last = float(df_price['Close'].iloc[-1])
    in_va = bool((close_last >= va_low) and (close_last <= va_high))

    return {
        "volume_profile": vols,
        "bin_edges": bin_edges,
        "poc_price": poc_price,
        "value_area_low": va_low,
        "value_area_high": va_high,
        "in_value_area": in_va,
        "total_volume": total_v
    }

# =========================
# BANDARMOLOGY (tanpa tape, berbasis OHLCV & 5m)
# =========================
def _effort_result_today(df: pd.DataFrame, lookback_ref: int = 60) -> dict:
    """Effort ~ nilai (Close*Volume), Result ~ True Range. Bandingkan z-score terhadap 60H."""
    d = df.tail(max(lookback_ref, 21)).copy()
    tr = (d['High'] - d['Low']).abs()
    val = (d['Close'] * d['Volume']).astype(float)
    if len(d) < 10:
        return {"class":"Normal","z_effort":0.0,"z_result":0.0}
    z_eff = (val.iloc[-1] - val.iloc[:-1].median()) / (val.iloc[:-1].std(ddof=0)+1e-9)
    z_res = (tr.iloc[-1]  - tr.iloc[:-1].median())  / (tr.iloc[:-1].std(ddof=0)+1e-9)
    if z_eff > 1.0 and z_res < 0.0:
        cls = "Absorption (High Effort, Low Result)"
    elif z_eff < -0.5 and z_res < -0.5:
        cls = "No-Supply/No-Demand (Low Effort, Low Result)"
    elif z_eff > 1.0 and z_res > 1.0:
        cls = "Climactic (High Effort, High Result)"
    else:
        cls = "Normal"
    return {"class": cls, "z_effort": float(z_eff), "z_result": float(z_res)}

def _ad_days(df: pd.DataFrame, n: int = 20) -> dict:
    """Hari naik+volume naik (accumulation) vs turun+volume naik (distribution) 20H."""
    d = df.tail(n+1).copy()
    if len(d) < 2:
        return {"acc_days":0, "dist_days":0, "net":0}
    px_chg = d['Close'].diff()
    vol_chg = d['Volume'].diff()
    acc = ((px_chg > 0) & (vol_chg > 0)).sum()
    dist = ((px_chg < 0) & (vol_chg > 0)).sum()
    return {"acc_days": int(acc), "dist_days": int(dist), "net": int(acc - dist)}

def _poc_migration_5m(intraday_5m: pd.DataFrame, last_n_sessions: int = 3) -> dict:
    """POC harian untuk N sesi terakhir (5m)."""
    out = {"series": [], "direction": "↔"}
    if intraday_5m is None or intraday_5m.empty:
        return out
    daily_groups = []
    for dt, g in intraday_5m.groupby(intraday_5m.index.date):
        if len(g) >= 30:
            daily_groups.append((pd.Timestamp(dt), g))
    daily_groups = daily_groups[-last_n_sessions:]
    if not daily_groups:
        return out

    def _poc_of_day(g: pd.DataFrame) -> float:
        lo = float(np.nanpercentile(g['Low'], 1)); hi = float(np.nanpercentile(g['High'], 99))
        mid = float(g['Close'].mean()); step = max(tick_size(mid), (hi-lo)/30.0)
        lo = np.floor(lo/step)*step; hi = np.ceil(hi/step)*step
        edges = np.arange(lo, hi+step, step); vols = np.zeros(len(edges)-1)
        for _, r in g.iterrows():
            low, high, vol = float(r['Low']), float(r['High']), float(r['Volume'])
            if vol <= 0: continue
            if high <= low:
                idx = int(np.clip(np.searchsorted(edges, float(r['Close']))-1, 0, len(vols)-1))
                vols[idx] += vol; continue
            s = int(np.clip(np.searchsorted(edges, low)-1, 0, len(vols)-1))
            e = int(np.clip(np.searchsorted(edges, high)-1, 0, len(vols)-1))
            rng = max(high-low, 1e-9)
            for i in range(s, e+1):
                seg_lo, seg_hi = edges[i], edges[i+1]
                ov = max(0.0, min(high, seg_hi)-max(low, seg_lo))
                if ov>0: vols[i] += vol*(ov/rng)
        if vols.sum() <= 0: return np.nan
        i = int(np.argmax(vols))
        return float((edges[i]+edges[i+1])/2)

    series = []
    for dt, g in daily_groups:
        series.append((dt, _poc_of_day(g)))
    out["series"] = series
    if len(series) >= 2 and np.isfinite(series[-1][1]) and np.isfinite(series[0][1]):
        diff = series[-1][1] - series[0][1]
        out["direction"] = "↑" if diff > 0 else ("↓" if diff < 0 else "↔")
    return out

def _top_1pct_bars_5m(intraday_5m: pd.DataFrame) -> dict:
    """Sebaran Top-1% bar by nilai transaksi (Close*Vol) dalam 20H: Low/Mid/High."""
    if intraday_5m is None or intraday_5m.empty:
        return {"low":0, "mid":0, "high":0, "count":0}
    g = intraday_5m.copy()
    g["value"] = (g["Close"]*g["Volume"]).astype(float)
    thr = float(np.nanpercentile(g["value"], 99))
    top = g[g["value"] >= thr].copy()
    if top.empty:
        return {"low":0, "mid":0, "high":0, "count":0}
    pos = []
    rng = (top["High"] - top["Low"]).replace(0, np.nan)
    pct = (top["Close"] - top["Low"]) / rng
    for x in pct.fillna(0.5):
        if x <= 0.33: pos.append("low")
        elif x >= 0.67: pos.append("high")
        else: pos.append("mid")
    return {"low": int(pos.count("low")),
            "mid": int(pos.count("mid")),
            "high": int(pos.count("high")),
            "count": int(len(pos))}

def compute_bandarmology(df_daily: pd.DataFrame, ticker_jk: str) -> dict:
    """Kompilasi metrik bandarmology untuk UI & skor 0–100."""
    intr = fetch_intraday_5m_yf(ticker_jk, days=PROFILE_DAYS+5)

    d = df_daily.copy()
    ma20v = d["Volume"].rolling(20).mean()
    k = VOL_ANOMALY_MULT
    streak = 0
    for v, m in zip(d["Volume"][::-1], ma20v[::-1]):
        if m <= 0: break
        if v >= k*m: streak += 1
        else: break

    ad = _ad_days(d, 20)
    er = _effort_result_today(d, lookback_ref=60)
    mig = _poc_migration_5m(intr, last_n_sessions=3)
    top = _top_1pct_bars_5m(intr)

    score = 50
    score += np.clip(ad["net"]*2, -20, 20)
    score += np.clip(streak*3, 0, 15)
    score += 10 if "Absorption" in er["class"] else (5 if "No-Supply" in er["class"] else 0)
    score += 8 if mig["direction"] == "↑" else (-8 if mig["direction"] == "↓" else 0)
    score += 5 if top["low"] >= max(top["mid"], top["high"]) else (-5 if top["high"] > top["low"] else 0)
    score = int(np.clip(score, 0, 100))
    label = "Accumulation" if score >= 66 else ("Distribution" if score <= 34 else "Neutral")

    return {
        "score": score, "label": label,
        "streak_anom": streak, "ad_days": ad,
        "effort_result": er, "poc_migration": mig, "top1pct": top
    }

# =========================
# PAKET INDIKATOR (MA5/20/200 + BB + lainnya)
# =========================
def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d['MA5'] = as_series(d['Close']).rolling(5).mean()
    d['MA20'] = as_series(d['Close']).rolling(20).mean()
    d['MA50'] = as_series(d['Close']).rolling(50).mean()
    d['MA200'] = as_series(d['Close']).rolling(200).mean()
    d['RSI'] = compute_rsi(d['Close'], 14)
    d['MACD'], d['Signal'], d['Hist'] = compute_macd(d['Close'], 12, 26, 9)
    d['BB_Upper'], d['BB_Middle'], d['BB_Lower'], d['BB_BW'], d['BB_%B'] = compute_bollinger_bands(d['Close'], 20, 2)
    d['ATR'] = compute_atr(d['High'], d['Low'], d['Close'], 14)
    d['OBV'] = compute_obv(d['Close'], d['Volume'])
    d['ADX'], d['Plus_DI'], d['Minus_DI'] = compute_adx(d['High'], d['Low'], d['Close'], 14)
    d['VWAP'] = compute_vwap(d['High'], d['Low'], d['Close'], d['Volume'])
    d['MFI'] = compute_mfi_corrected(d, 14)
    d['Volume_MA20'] = as_series(d['Volume']).rolling(20).mean()
    d['Value_Rp'] = (as_series(d['Close']) * as_series(d['Volume']))
    d['Value_MA20'] = as_series(d['Value_Rp']).rolling(20).mean()
    return d

# =========================
# CHART HARGA (≤ 3 bulan) + clamp Y & filter level jauh
# =========================
def make_main_chart(df: pd.DataFrame, sr: dict, is_squeeze: bool,
                    fb_events: list | None = None) -> tuple[go.Figure, list]:
    y_min = float(np.nanmin([df['Low'].tail(VIEW_DAYS).min(), df['BB_Lower'].tail(VIEW_DAYS).min()]))
    y_max = float(np.nanmax([df['High'].tail(VIEW_DAYS).max(), df['BB_Upper'].tail(VIEW_DAYS).max()]))
    pad = (y_max - y_min) * 0.08
    y_lo, y_hi = y_min - pad, y_max + pad

    def in_view(level: float) -> bool:
        return (level >= y_lo) and (level <= y_hi)

    out_of_view = []

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name="Candlestick", increasing_line_color='green', decreasing_line_color='red'
    ))
    for name, col, color in [("MA5",'MA5','teal'), ("MA20",'MA20','blue'), ("MA200",'MA200','black')]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=name,
                                     line=dict(color=color, width=1.6)))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name="BB Upper",
                             mode="lines", line=dict(color="red", width=1, dash="dot")))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Middle'], name="BB Middle",
                             mode="lines", line=dict(color="purple", width=1.2)))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name="BB Lower",
                             mode="lines", line=dict(color="green", width=1, dash="dot"),
                             fill='tonexty', fillcolor='rgba(173,216,230,0.10)'))

    for i, lvl in enumerate(sr['Support']):
        if in_view(lvl):
            fig.add_hline(y=lvl, line_dash="dash", line_color="green",
                          annotation_text=f"S{i+1}: {fmt_rp_short(lvl)}", annotation_position="bottom right")
        else:
            out_of_view.append(("S"+str(i+1), lvl))
    for i, lvl in enumerate(sr['Resistance']):
        if in_view(lvl):
            fig.add_hline(y=lvl, line_dash="dash", line_color="red",
                          annotation_text=f"R{i+1}: {fmt_rp_short(lvl)}", annotation_position="top right")
        else:
            out_of_view.append(("R"+str(i+1), lvl))

    if fb_events:
        for ev in fb_events[-6:]:
            y = ev.get("high") or ev.get("low") or float(df['Close'].iloc[-1])
            if in_view(y):
                txt = "FB↑" if ev["type"] == "false_break_up" else "FB↓"
                fig.add_annotation(x=ev["date"], y=y, text=txt, showarrow=True, arrowhead=2)

    if is_squeeze:
        fig.add_annotation(x=df.index[-1], y=df['Close'].iloc[-1], text="BOLLINGER SQUEEZE",
                           showarrow=True, arrowhead=1, arrowcolor="purple", font=dict(color="purple"))

    fig.update_layout(
        title="Chart Teknikal (≤ 3 bulan)",
        template="plotly_white",
        xaxis_rangeslider_visible=False,
        height=700,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='x unified'
    )
    fig.update_yaxes(range=[y_lo, y_hi])
    return fig, out_of_view

# =========================
# NARASI
# =========================
def build_narrative(ticker: str, base: pd.DataFrame, scores: dict, composite: float,
                    confidence: str, sr: dict, profile: dict) -> str:
    last_close = float(base['Close'].iloc[-1])
    prev_close = float(base['Close'].iloc[-2]) if len(base) > 1 else last_close
    chg_pct = (last_close - prev_close) / prev_close * 100 if prev_close else 0.0
    sma200 = base['Close'].rolling(200).mean()
    in_uptrend = (last_close > float(sma200.iloc[-1])) if not np.isnan(sma200.iloc[-1]) else False
    adx_now = float(base['ADX'].iloc[-1]); rsi_now = float(base['RSI'].iloc[-1])
    macd_cross = scores['macd_cross'][0] > 0

    vol_today_shares = float(base['Volume'].iloc[-1]); vol_today_lot = shares_to_lot(vol_today_shares)
    value_today = float((base['Close'] * base['Volume']).iloc[-1])

    R1 = sr['Resistance'][0] if sr['Resistance'] else last_close * 1.05
    S1 = sr['Support'][0] if sr['Support'] else last_close * 0.95

    poc = profile.get("poc_price", np.nan); vaL = profile.get("value_area_low", np.nan); vaH = profile.get("value_area_high", np.nan)
    in_va = profile.get("in_value_area", False)

    interp = IndicatorScoringSystem.interpret(composite)

    parts = []
    parts.append(f"**{ticker}** {('uptrend' if in_uptrend else 'di bawah MA200')}, {interp.lower()} (keyakinan {confidence}).")
    parts.append(f"Bar terakhir **{('naik' if chg_pct>=0 else 'turun')} {abs(chg_pct):.2f}%**; RSI **{rsi_now:.0f}**, ADX **{int(adx_now)}**, MACD {'cross up' if macd_cross else 'cross down'}.")
    if np.isfinite(poc): parts.append(f"Profil volume 20H: **POC {fmt_rp_short(poc)}**, **VA {fmt_rp_short(vaL)} – {fmt_rp_short(vaH)}**; harga kini **{'di dalam' if in_va else 'di luar'} VA**.")
    parts.append(f"Saat ini transaksi **~{fmt_int(vol_today_lot)} lot** (≈ {fmt_rp_short(value_today)}).")
    parts.append(f"Level kunci: **R1 {fmt_rp_short(R1)}**, **S1 {fmt_rp_short(S1)}**. Hindari false break: tunggu close tegas di luar level + volume ≥ {VOL_ANOMALY_MULT}× MA20.")
    return " ".join(parts)

# =========================
# APP
# =========================
def app():
    st.set_page_config(page_title="Analisa Saham IDX – Narasi Swing + Volume Profile", layout="wide")
    st.title("📝 Analisa Saham IDX – Narasi Swing + Volume Profile 20H (5m)")

    st.markdown("""
    <style>
    div[data-testid="stMetricValue"] { white-space: nowrap; }
    </style>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([1.1, 1])
    with c1:
        ticker_in = st.text_input("Kode Saham", value="BBCA").strip().upper()
        as_of = st.date_input("📅 Tanggal Analisis (as-of)", value=datetime.today())
    with c2:
        account_size = st.number_input("Modal (Rp)", value=100_000_000, step=10_000_000)
        risk_percent = st.slider("Risiko per Trade (%)", 0.5, 5.0, 2.0) / 100

    if st.button("🚀 Mulai Analisis", use_container_width=True):
        if not ticker_in:
            st.warning("Masukkan kode saham terlebih dahulu."); return

        ticker = resolve_ticker(ticker_in)
        end_all = datetime.combine(as_of, datetime.min.time()) + timedelta(days=1)
        start_all = end_all - timedelta(days=365)
        data_all = fetch_history_yf(ticker, start_all, end_all)
        if data_all is None or data_all.empty:
            st.error("Data tidak tersedia dari yfinance."); return

        df_hist = data_all[data_all.index <= pd.Timestamp(as_of) + pd.Timedelta(days=1)]
        if len(df_hist) < 200:
            st.warning("Data historis terlalu pendek (<200 bar) untuk MA200 yang stabil."); return

        st.caption(f"Data terakhir sampai as-of: **{df_hist.index[-1].date()}** (n={len(df_hist)})")

        base = compute_all_indicators(df_hist)
        view_df = base[base.index >= (base.index[-1] - pd.Timedelta(days=VIEW_DAYS))].copy()
        if view_df.empty: view_df = base.tail(VIEW_DAYS).copy()

        sr = compute_support_resistance(view_df, window=SR_WINDOW)
        false_breaks = detect_false_breaks(view_df, sr, lookback=SR_WINDOW, atr_period=14)

        if false_breaks:
            last_fb = false_breaks[-1]
            if last_fb["type"] == "false_break_up":
                st.warning(f"⚠️ False break ATAS di sekitar R (≈ {fmt_rp_short(last_fb['level'])}) "
                           f"{last_fb['date'].date()} — High {fmt_rp_short(last_fb.get('high',0))}, "
                           f"Close {fmt_rp_short(last_fb['close'])}, Vol~{last_fb['vol_ratio']:.1f}×MA20.")
            else:
                st.warning(f"⚠️ False break BAWAH di sekitar S (≈ {fmt_rp_short(last_fb['level'])}) "
                           f"{last_fb['date'].date()} — Low {fmt_rp_short(last_fb.get('low',0))}, "
                           f"Close {fmt_rp_short(last_fb['close'])}, Vol~{last_fb['vol_ratio']:.1f}×MA20.")
            st.caption(f"Total false break 60 bar terakhir: **{len(false_breaks)}**")
        else:
            st.info("Tidak ada false break pada 60 bar terakhir.")

        scorer = IndicatorScoringSystem()
        scores = {}
        scores['rsi'] = scorer.score_rsi(base['RSI'])
        mc, mcs, mh, mhs = scorer.score_macd(base['MACD'], base['Signal'], base['Hist'])
        scores['macd_cross'] = (mc, mcs); scores['macd_hist'] = (mh, mhs)
        bscore, bstr, _ = scorer.score_boll(base['Close'], base['BB_Upper'], base['BB_Lower'], base['BB_%B'], base['BB_BW'])
        scores['bollinger'] = (bscore, bstr)
        price_change_pct = float((base['Close'].iloc[-1] - base['Close'].iloc[-2]) / max(base['Close'].iloc[-2], 1e-9) * 100) if len(base) > 1 else 0.0
        vscore, vstr = scorer.score_volume(base['Volume'], base['Volume_MA20'], price_change_pct)
        scores['volume'] = (vscore, vstr)
        oscore, ostr = scorer.score_obv(base['OBV'])
        scores['obv'] = (oscore, ostr)
        adx_s, adx_str = scorer.score_adx(base['ADX'], base['Plus_DI'], base['Minus_DI'])
        scores['adx'] = (adx_s, adx_str)
        composite = scorer.composite(scores)
        confidence = scorer.confidence(composite, scores)

        # ===== Panel Indikator
        st.subheader("📟 Indikator Teknikal")
        a1, a2 = st.columns(2)
        with a1:
            st.metric("MA20", f"{float(base['MA20'].iloc[-1]):.2f}")
            st.metric("MA50", f"{float(base['MA50'].iloc[-1]):.2f}")
        with a2:
            st.metric("RSI", f"{float(base['RSI'].iloc[-1]):.2f}")
            st.metric("MFI", f"{float(base['MFI'].iloc[-1]):.2f}")

        # ===== Panel Ringkas Bar Terakhir
        last_dt = base.index[-1]
        last_close = float(base['Close'].iloc[-1])
        last_vol_shares = float(base['Volume'].iloc[-1])
        last_vol_lot = shares_to_lot(last_vol_shares)
        last_value_rp = last_close * last_vol_shares
        ma5_val = float(base['MA5'].iloc[-1])

        st.subheader("🧾 Ringkas Perdagangan (Bar Terakhir)")
        p1, p2, p3 = st.columns([1, 1, 1.3])
        with p1: st.metric("MA5 (Harga)", f"{ma5_val:.2f}")
        with p2: st.metric("Volume Perdagangan (Lot)", fmt_int(last_vol_lot))
        with p3: st.metric("Nilai Transaksi (Rp)", fmt_rp(last_value_rp))
        st.caption(f"Tanggal: **{last_dt.date()}**")

        vol_ma20 = float(base['Volume_MA20'].iloc[-1]) if not np.isnan(base['Volume_MA20'].iloc[-1]) else 0.0
        vol_anom = last_vol_shares >= VOL_ANOMALY_MULT * max(vol_ma20, 1.0)
        st.info(f"**Volume Anomali:** {'Ya' if vol_anom else 'Tidak'} "
                f"(hari ini {fmt_int(last_vol_lot)} lot; ambang {VOL_ANOMALY_MULT}×MA20 ≈ {fmt_int(shares_to_lot(vol_ma20))} lot)")

        # ===== Level Penting & Fibo
        st.subheader("🧭 Level Penting")
        st.write("**Support:** " + " | ".join([fmt_rp(x) for x in sr['Support']]))
        st.write("**Resistance:** " + " | ".join([fmt_rp(x) for x in sr['Resistance']]))
        st.subheader("📐 Level Fibonacci (60 bar)")
        st.table(pd.DataFrame([{"Level": k, "Harga": fmt_rp(v)} for k, v in sr['Fibonacci'].items()]))

        # ===== Narasi
        st.subheader(f"✍️ Narasi – {ticker_in.upper()} ({df_hist.index[-1].date()})")
        st.write(build_narrative(ticker_in.upper(), base, scores, composite, confidence, sr, {
            "poc_price": np.nan, "value_area_low": np.nan, "value_area_high": np.nan, "in_value_area": False
        }))

        # ===== Chart Harga 3 Bulan
        st.subheader("📈 Chart Teknikal (≤ 3 bulan)")
        bw_hist = view_df['BB_BW'].dropna()
        is_squeeze = (stats.percentileofscore(bw_hist.to_numpy().ravel(), float(view_df['BB_BW'].iloc[-1]))/100.0 < 0.2) if len(bw_hist) >= 30 else False
        fig_main, out_of_view = make_main_chart(view_df, sr, is_squeeze, false_breaks)
        st.plotly_chart(fig_main, use_container_width=True)
        if out_of_view:
            st.caption("Level di luar tampilan: " +
                       " | ".join([f"{name}: {fmt_rp_short(val)}" for name,val in out_of_view]))

        # ===== Volume Profile (5m, 20H) – Vertical style
        st.subheader("📊 Volume Profile (20 Hari Terakhir)")
        vp = compute_volume_profile_vertical(df_hist, ticker, days=PROFILE_DAYS)
        if vp["bin_edges"].size == 0:
            st.warning("Volume Profile tidak tersedia (data terlalu sedikit).")
        else:
            fig_vp = go.Figure(go.Bar(
                x=vp['bin_edges'][:-1],
                y=vp['volume_profile'],
                name="Volume Profile"
            ))
            if np.isfinite(vp['poc_price']):
                fig_vp.add_vline(x=vp['poc_price'], line_dash="dash", line_color="red", annotation_text="POC")
            if np.isfinite(vp['value_area_low']) and np.isfinite(vp['value_area_high']):
                fig_vp.add_vrect(x0=vp['value_area_low'], x1=vp['value_area_high'],
                                 fillcolor="green", opacity=0.12, line_width=0, annotation_text="Value Area")
            fig_vp.add_vline(x=last_close, line_dash="dot", line_color="blue", annotation_text="Current Price")
            fig_vp.update_layout(height=360, template="plotly_white",
                                 xaxis_title="Harga", yaxis_title="Volume (lembar)")
            st.plotly_chart(fig_vp, use_container_width=True)

            centers = (vp['bin_edges'][:-1] + vp['bin_edges'][1:]) / 2
            df_top = pd.DataFrame({"Harga": centers, "Volume": vp["volume_profile"]})
            df_top = df_top[df_top["Volume"] > 0].copy()
            df_top["Volume (Lembar)"] = df_top["Volume"].map(lambda v: fmt_int(v))
            df_top["Volume (Lot)"] = df_top["Volume"].map(lambda v: fmt_int(shares_to_lot(v)))
            df_top["Harga"] = df_top["Harga"].map(fmt_rp)
            df_top = df_top.sort_values("Volume", ascending=False).head(10)
            st.table(df_top[["Harga","Volume (Lembar)","Volume (Lot)"]])

            narasi_vp = build_narrative(ticker_in.upper(), base, scores, composite, confidence, sr, {
                "poc_price": vp["poc_price"],
                "value_area_low": vp["value_area_low"],
                "value_area_high": vp["value_area_high"],
                "in_value_area": vp["in_value_area"]
            })
            st.subheader("🗒️ Ringkasan + VP")
            st.write(narasi_vp)

        # ===== Panel Bandarmology
        st.subheader("🕵️ Bandarmology (Ringkas)")
        bdm = compute_bandarmology(df_hist, ticker)
        c1b, c2b, c3b, c4b = st.columns(4)
        with c1b:
            st.metric("Skor", f"{bdm['score']}/100")
            st.caption(bdm["label"])
        with c2b:
            st.metric("A/D Days (20H)", f"+{bdm['ad_days']['acc_days']} / -{bdm['ad_days']['dist_days']}")
            st.caption(f"Net: {bdm['ad_days']['net']:+d}")
        with c3b:
            st.metric("Streak Volume Anomali", f"{bdm['streak_anom']} hari")
            st.caption(f"Ambang: {VOL_ANOMALY_MULT}×MA20")
        with c4b:
            st.metric("POC Migration (3 sesi)", bdm["poc_migration"]["direction"])
            series = bdm["poc_migration"]["series"]
            if series:
                mini = " → ".join([fmt_rp_short(p) if np.isfinite(p) else "-" for _, p in series])
                st.caption(mini)

        e1, e2 = st.columns(2)
        with e1:
            st.write(f"**Effort vs Result:** {bdm['effort_result']['class']} "
                     f"(zEff {bdm['effort_result']['z_effort']:.2f} | zRes {bdm['effort_result']['z_result']:.2f})")
        with e2:
            t = bdm["top1pct"]; total = t["count"]
            st.write(f"**Top-1% 5m bars (nilai):** Low={t['low']}, Mid={t['mid']}, High={t['high']} (n={total})")

        st.info("**Disclaimer**: Edukasi, bukan rekomendasi. Trading mengandung risiko.")

if __name__ == "__main__":
    app()
