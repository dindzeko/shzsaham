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
SR_WINDOW = 60                 # S/R & Fibonacci dari 60 bar (≈ 3 bulan)
PROFILE_DAYS_DEFAULT = 20     # Window Volume Profile
INTRADAY_DEFAULT = True       # Default profil pakai intraday 5m (fallback harian)
VOL_ANOMALY_MULT = 1.5        # Volume anomali = ≥ 1.5 × MA20

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
    period = f"{min(days, 60)}d"   # limit yfinance untuk 5m ≈ 60 hari
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

def compute_adl(high, low, close, volume):
    high, low, close, volume = as_series(high), as_series(low), as_series(close), as_series(volume)
    mfm = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    mfm = mfm.replace([np.inf, -np.inf], 0).fillna(0)
    mfv = mfm * volume
    return as_series(mfv.cumsum())

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
        """Keyakinan = proporsi indikator yang searah & cukup kuat."""
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
        if score >= 0.7: return "Sangat Bullish - Sentimen beli sangat kuat"
        if score >= 0.4: return "Bullish Kuat - Sentimen beli kuat"
        if score >= 0.1: return "Bullish Lemah - Sentimen cenderung beli"
        if score > -0.1: return "Netral - Sentimen tidak jelas"
        if score > -0.4: return "Bearish Lemah - Sentimen cenderung jual"
        if score > -0.7: return "Bearish Kuat - Sentimen jual kuat"
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
        close = as_series(df['Close'])
        vol = as_series(df['Volume'])
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
# BANDARMOLOGY (ringkas, untuk panel cepat)
# =========================
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
    close_last = float(df['Close'].iloc[-1])
    out['in_value_area'] = bool((close_last >= out['va_low']) and (close_last <= out['va_high']))
    return out

# =========================
# VOLUME PROFILE 20H
# =========================
def compute_volume_profile(df_price: pd.DataFrame, days: int = PROFILE_DAYS_DEFAULT,
                           use_intraday_5m: bool = INTRADAY_DEFAULT,
                           ticker_jk: str | None = None,
                           bin_step: str | int = "tick"):
    # Pilih sumber data
    if use_intraday_5m and ticker_jk:
        intr = fetch_intraday_5m_yf(ticker_jk, days=days+5)
        data = intr if intr is not None and not intr.empty else df_price.copy()
        if 'date' in data.columns: data = data.set_index('date')
    else:
        data = df_price.copy()
    data = data.tail(days if '5m' not in str(data.index.freq) else days*78)

    if data.empty or len(data) < 5:
        return {"levels": pd.DataFrame(columns=["price","volume","volume_pct","cum_pct"]),
                "poc": np.nan, "va_low": np.nan, "va_high": np.nan,
                "in_value_area": False, "total_volume": 0.0}

    mid_price = float(data['Close'].tail(20).mean())
    if bin_step == "tick": step = tick_size(mid_price)
    elif isinstance(bin_step, (int, float)) and bin_step > 0: step = float(bin_step)
    else:
        hi = float(data['High'].max()); lo = float(data['Low'].min())
        step = max((hi - lo) / max(30, 1), tick_size(mid_price))

    hi = float(data['High'].max()); lo = float(data['Low'].min())
    lo = round_to_tick(lo - step); hi = round_to_tick(hi + step)
    bins = np.arange(lo, hi + step, step)
    vols = np.zeros(len(bins)-1)

    for _, r in data.iterrows():
        low = float(r['Low']); high = float(r['High']); vol = float(r['Volume'])
        if vol <= 0: continue
        if high <= low:
            c = float(r['Close'])
            idx = int(np.clip(np.searchsorted(bins, c) - 1, 0, len(vols)-1))
            vols[idx] += vol; continue
        start = int(np.clip(np.searchsorted(bins, low) - 1, 0, len(vols)-1))
        end   = int(np.clip(np.searchsorted(bins, high) - 1, 0, len(vols)-1))
        rng = high - low
        for i in range(start, end+1):
            seg_lo = bins[i]; seg_hi = bins[i+1]
            ov_len = max(0.0, min(high, seg_hi) - max(low, seg_lo))
            if ov_len > 0: vols[i] += vol * (ov_len / rng)

    price_levels = (bins[:-1] + bins[1:]) / 2
    total_v = float(vols.sum())

    if total_v <= 0:
        df_levels = pd.DataFrame({"price": price_levels, "volume": vols})
        df_levels["volume_pct"] = 0.0; df_levels["cum_pct"] = 0.0
        return {"levels": df_levels, "poc": np.nan, "va_low": np.nan, "va_high": np.nan,
                "in_value_area": False, "total_volume": 0.0}

    order = np.argsort(vols)[::-1]
    cum = 0.0; picked = []
    for i in order:
        picked.append(i); cum += vols[i]
        if cum >= total_v * 0.7: break
    va_low = float(bins[min(picked)])
    va_high = float(bins[max(picked)+1])
    poc_idx = int(np.argmax(vols))
    poc = float(price_levels[poc_idx])

    df_levels = pd.DataFrame({"price": price_levels, "volume": vols}).sort_values("price")
    df_levels["volume_pct"] = df_levels["volume"] / total_v
    df_levels["cum_pct"] = df_levels["volume_pct"].cumsum()
    close_last = float(df_price['Close'].iloc[-1])
    in_va = bool((close_last >= va_low) and (close_last <= va_high))

    return {"levels": df_levels, "poc": poc, "va_low": va_low, "va_high": va_high,
            "in_value_area": in_va, "total_volume": total_v}

# =========================
# PAKET INDIKATOR (MA5/20/50/100/200 + BB)
# =========================
def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d['MA5'] = as_series(d['Close']).rolling(5).mean()
    d['MA20'] = as_series(d['Close']).rolling(20).mean()
    d['MA50'] = as_series(d['Close']).rolling(50).mean()
    d['MA100'] = as_series(d['Close']).rolling(100).mean()
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
# CHART UTAMA (≤ 3 bulan)
# =========================
def make_main_chart(df: pd.DataFrame, sr: dict, is_squeeze: bool,
                    fb_events: list | None = None, profile: dict | None = None) -> go.Figure:
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
        fig.add_hline(y=lvl, line_dash="dash", line_color="green",
                      annotation_text=f"S{i+1}: {fmt_rp(lvl)}", annotation_position="bottom right")
    for i, lvl in enumerate(sr['Resistance']):
        fig.add_hline(y=lvl, line_dash="dash", line_color="red",
                      annotation_text=f"R{i+1}: {fmt_rp(lvl)}", annotation_position="top right")
    if profile and np.isfinite(profile.get("va_low", np.nan)) and np.isfinite(profile.get("va_high", np.nan)):
        fig.add_hline(y=profile["va_low"], line_dash="dot", line_color="gray",
                      annotation_text=f"VAL {fmt_rp(profile['va_low'])}", annotation_position="bottom left")
        fig.add_hline(y=profile["va_high"], line_dash="dot", line_color="gray",
                      annotation_text=f"VAH {fmt_rp(profile['va_high'])}", annotation_position="top left")
    if profile and np.isfinite(profile.get("poc", np.nan)):
        fig.add_hline(y=profile["poc"], line_dash="dot", line_color="black",
                      annotation_text=f"POC {fmt_rp(profile['poc'])}", annotation_position="top left")
    if fb_events:
        for ev in fb_events[-6:]:
            y = ev.get("high") or ev.get("low") or float(df['Close'].iloc[-1])
            txt = "FB↑" if ev["type"] == "false_break_up" else "FB↓"
            fig.add_annotation(x=ev["date"], y=y, text=txt, showarrow=True, arrowhead=2)
    if is_squeeze:
        fig.add_annotation(x=df.index[-1], y=df['Close'].iloc[-1], text="BOLLINGER SQUEEZE",
                           showarrow=True, arrowhead=1, arrowcolor="purple", font=dict(color="purple"))
    fig.update_layout(
        title="Chart Teknikal (≤ 3 bulan)",
        template="plotly_white", xaxis_rangeslider_visible=False,
        height=700, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='x unified'
    )
    return fig

# =========================
# VIZ: VOLUME PROFILE (bar horizontal)
# =========================
def make_profile_bar(profile: dict) -> go.Figure:
    df = profile["levels"]
    fig = go.Figure(go.Bar(
        x=df["volume"], y=df["price"], orientation="h", name="Vol",
        hovertemplate="Harga %{y}<br>Volume %{x:.0f}<extra></extra>"
    ))
    fig.update_layout(
        title="Volume by Price – 20 Hari",
        xaxis_title="Volume (lembar)", yaxis_title="Harga",
        template="plotly_white", height=450
    )
    return fig

# =========================
# NARASI
# =========================
def build_narrative(ticker: str, base: pd.DataFrame, scores: dict, composite: float,
                    confidence: str, sr: dict, weekly_bias: int,
                    plan: dict | None, res_ok: bool, sup_ok: bool, vol_ok: bool,
                    profile: dict) -> str:
    last_close = float(base['Close'].iloc[-1])
    prev_close = float(base['Close'].iloc[-2]) if len(base) > 1 else last_close
    chg_pct = (last_close - prev_close) / prev_close * 100 if prev_close else 0.0
    sma200 = base['Close'].rolling(200).mean()
    in_uptrend = (last_close > float(sma200.iloc[-1])) if not np.isnan(sma200.iloc[-1]) else False
    adx_now = float(base['ADX'].iloc[-1]); rsi_now = float(base['RSI'].iloc[-1])
    macd_cross = scores['macd_cross'][0] > 0
    # Volume & nilai (MA20)
    vol_today_shares = float(base['Volume'].iloc[-1]); vol_today_lot = shares_to_lot(vol_today_shares)
    vol_ma20_shares = float(base['Volume_MA20'].iloc[-1]); vol_ma20_lot = shares_to_lot(vol_ma20_shares)
    value_today = float((base['Close'] * base['Volume']).iloc[-1]); value_ma20 = float(base['Value_MA20'].iloc[-1])
    vol_anom = bool(vol_today_shares >= VOL_ANOMALY_MULT * max(vol_ma20_shares, 1e-9))
    R1 = sr['Resistance'][0] if sr['Resistance'] else last_close * 1.05
    S1 = sr['Support'][0] if sr['Support'] else last_close * 0.95
    poc = profile.get("poc", np.nan); vaL = profile.get("va_low", np.nan); vaH = profile.get("va_high", np.nan)
    in_va = profile.get("in_value_area", False)
    bias_txt = "bias mingguan mendukung" if weekly_bias > 0 else ("bias mingguan melemahkan" if weekly_bias < 0 else "tanpa bias mingguan")
    interp = IndicatorScoringSystem.interpret(composite)

    parts = []
    parts.append(f"**{ticker}** {('uptrend' if in_uptrend else 'di bawah MA200')}, {interp.lower()} (keyakinan {confidence}); {bias_txt}.")
    parts.append(f"Perubahan bar terakhir **{('naik' if chg_pct>=0 else 'turun')} {abs(chg_pct):.2f}%**; RSI **{rsi_now:.0f}**, ADX **{int(adx_now)}**, MACD {'cross up' if macd_cross else 'cross down'}.")
    if np.isfinite(poc): parts.append(f"Profil volume 20H: **POC {fmt_rp(poc)}**, **VA {fmt_rp(vaL)} – {fmt_rp(vaH)}**; harga kini **{'di dalam' if in_va else 'di luar'} VA**.")
    parts.append(f"Konfirmasi breakout lebih valid bila **volume ≥ {VOL_ANOMALY_MULT}× MA20** → hari ini **{fmt_int(vol_today_lot)} lot** (≈ {fmt_int(vol_today_shares)} lembar) vs MA20 **{fmt_int(vol_ma20_lot)} lot** (≈ {fmt_int(vol_ma20_shares)} lembar); nilai **{fmt_rp(value_today)}** (MA20 nilai **{fmt_rp(value_ma20)}**).")
    parts.append(f"Level kunci: **R1 {fmt_rp(R1)}**, **S1 {fmt_rp(S1)}**.")
    if plan: parts.append(f"Rencana: {plan['type']} dengan RR ≈ **{plan['risk_reward']}:1**; disiplin **stop-loss ATR** & amati retest.")
    else: parts.append("Belum ada trigger kuat; tunggu close di luar level kunci + volume memadai untuk menghindari false break.")
    return " ".join(parts)

# =========================
# APP
# =========================
def app():
    st.set_page_config(page_title="Analisa Saham IDX – Narasi Swing + Volume Profile", layout="wide")
    st.title("📝 Analisa Saham IDX – Narasi Swing + Volume Profile 20H")

    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c1:
        ticker_in = st.text_input("Kode Saham", value="BBCA").strip().upper()
        as_of = st.date_input("📅 Tanggal Analisis (as-of)", value=datetime.today())
    with c2:
        account_size = st.number_input("Modal (Rp)", value=100_000_000, step=10_000_000)
        risk_percent = st.slider("Risiko per Trade (%)", 0.5, 5.0, 2.0) / 100
    with c3:
        use_mtf = st.checkbox("Gunakan Konfirmasi Weekly", value=True)
        use_intraday_profile = st.checkbox("Volume Profile pakai intraday 5m (fallback harian)", value=INTRADAY_DEFAULT)

    exp_profile = st.expander("Pengaturan Volume Profile 20H")
    with exp_profile:
        profile_days = st.slider("Window hari", 10, 30, PROFILE_DAYS_DEFAULT)
        bin_choice = st.selectbox("Lebar bin harga", ["Tick IDX (auto)", "10", "25", "Auto (≈30 bin)"], index=0)

    if st.button("🚀 Mulai Analisis", use_container_width=True):
        if not ticker_in:
            st.warning("Masukkan kode saham terlebih dahulu."); return

        ticker = resolve_ticker(ticker_in)
        end_all = datetime.combine(as_of, datetime.min.time()) + timedelta(days=1)
        start_all = end_all - timedelta(days=365)  # ambil panjang untuk stabilitas indikator
        data_all = fetch_history_yf(ticker, start_all, end_all)
        if data_all is None or data_all.empty:
            st.error("Data tidak tersedia dari yfinance."); return

        df_hist = data_all[data_all.index <= pd.Timestamp(as_of) + pd.Timedelta(days=1)]
        if len(df_hist) < 200:
            st.warning("Data historis terlalu pendek (<200 bar) untuk MA200 yang stabil."); return

        st.caption(f"Data terakhir sampai as-of: **{df_hist.index[-1].date()}** (n={len(df_hist)})")

        # Hitung indikator lengkap
        base = compute_all_indicators(df_hist)

        # Window chart 3 bulan
        view_days = 90
        view_df = base[base.index >= (base.index[-1] - pd.Timedelta(days=view_days))].copy()
        if view_df.empty: view_df = base.tail(90).copy()

        # Weekly bias (opsional)
        weekly_bias = 0
        if use_mtf:
            try:
                dfw = df_hist.resample('W-FRI').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
                if len(dfw) >= 30:
                    macd_w, sig_w, _ = compute_macd(dfw['Close'])
                    weekly_bias = 1 if float(macd_w.iloc[-1]) > float(sig_w.iloc[-1]) else -1
            except Exception as e:
                st.warning(f"Konfirmasi weekly dimatikan: {e}"); weekly_bias = 0

        # S/R & false break
        sr = compute_support_resistance(view_df, window=SR_WINDOW)
        false_breaks = detect_false_breaks(view_df, sr, lookback=SR_WINDOW, atr_period=14)
        if false_breaks:
            last_fb = false_breaks[-1]
            if last_fb["type"] == "false_break_up":
                st.warning(f"⚠️ False break ATAS di sekitar R (≈ {fmt_rp(last_fb['level'])}) "
                           f"{last_fb['date'].date()} — High {fmt_rp(last_fb.get('high',0))}, "
                           f"Close {fmt_rp(last_fb['close'])}, Vol~{last_fb['vol_ratio']:.1f}×MA20.")
            else:
                st.warning(f"⚠️ False break BAWAH di sekitar S (≈ {fmt_rp(last_fb['level'])}) "
                           f"{last_fb['date'].date()} — Low {fmt_rp(last_fb.get('low',0))}, "
                           f"Close {fmt_rp(last_fb['close'])}, Vol~{last_fb['vol_ratio']:.1f}×MA20.")
            st.caption(f"Total false break 60 bar terakhir: **{len(false_breaks)}**")
        else:
            st.info("Tidak ada false break pada 60 bar terakhir.")

        # Skoring komposit
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
        if weekly_bias != 0:
            composite *= (1.15 if weekly_bias > 0 else 0.85)
        confidence = scorer.confidence(composite, scores)

        # Breakout plan
        detector = BreakoutDetector()
        res_lvl = sr['Resistance'][0] if sr['Resistance'] else float(view_df['Close'].iloc[-1]) * 1.05
        sup_lvl = sr['Support'][0] if sr['Support'] else float(view_df['Close'].iloc[-1]) * 0.95
        res_ok, sup_ok, vol_ok, buffer = detector.detect(view_df, res_lvl, sup_lvl, bars_confirm=1)
        plan = None
        if res_ok and vol_ok:   plan = detector.plan(view_df, "resistance", res_lvl, buffer, account_size, risk_percent)
        elif sup_ok and vol_ok: plan = detector.plan(view_df, "support", sup_lvl, buffer, account_size, risk_percent)

        # Volume Profile 20H
        bs = "tick" if bin_choice == "Tick IDX (auto)" else (10 if bin_choice=="10" else (25 if bin_choice=="25" else "auto"))
        profile = compute_volume_profile(df_hist, days=profile_days,
                                         use_intraday_5m=use_intraday_profile, ticker_jk=ticker,
                                         bin_step=bs)

        # ===== Panel Indikator =====
        st.subheader("📟 Indikator Teknikal")
        a1, a2, a3, a4 = st.columns(4)
        with a1:
            st.metric("MA20", f"{float(base['MA20'].iloc[-1]):.2f}")
            st.metric("MA50", f"{float(base['MA50'].iloc[-1]):.2f}")
        with a2:
            st.metric("RSI", f"{float(base['RSI'].iloc[-1]):.2f}")
            st.metric("MFI", f"{float(base['MFI'].iloc[-1]):.2f}")
        with a3:
            st.metric("Volume (Lembar)", fmt_int(float(base['Volume'].iloc[-1])))
            st.metric("MA20 Volume (Lot)", fmt_int(shares_to_lot(float(base['Volume_MA20'].iloc[-1]))))
            st.caption(f"(≈ {fmt_int(float(base['Volume_MA20'].iloc[-1]))} lembar)")
        with a4:
            st.metric("Nilai (Rp)", fmt_rp(float((base['Close']*base['Volume']).iloc[-1])))
            st.metric("MA20 Nilai (Rp)", fmt_rp(float(base['Value_MA20'].iloc[-1])))
            vol_anom = bool(float(base['Volume'].iloc[-1]) >= VOL_ANOMALY_MULT * max(float(base['Volume_MA20'].iloc[-1]), 1e-9))
            st.markdown(f"**Volume Anomali:** {'Ya' if vol_anom else 'Tidak'}")

        # ===== Level Penting & Fibo =====
        st.subheader("🧭 Level Penting")
        st.write("**Support:** " + " | ".join([fmt_rp(x) for x in sr['Support']]))
        st.write("**Resistance:** " + " | ".join([fmt_rp(x) for x in sr['Resistance']]))
        st.subheader("📐 Level Fibonacci (60 bar)")
        st.table(pd.DataFrame([{"Level": k, "Harga": fmt_rp(v)} for k, v in sr['Fibonacci'].items()]))

        # ===== Narasi =====
        st.subheader(f"✍️ Narasi – {ticker_in.upper()} ({df_hist.index[-1].date()})")
        narrative = build_narrative(ticker_in.upper(), base, scores, composite, confidence, sr,
                                    weekly_bias, plan, res_ok, sup_ok, vol_ok, profile)
        st.write(narrative)

        # ===== Chart Harga 3 Bulan =====
        st.subheader("📈 Chart Teknikal (≤ 3 bulan)")
        bw_hist = view_df['BB_BW'].dropna()
        is_squeeze = (stats.percentileofscore(bw_hist.to_numpy().ravel(), float(view_df['BB_BW'].iloc[-1]))/100.0 < 0.2) if len(bw_hist) >= 30 else False
        fig_main = make_main_chart(view_df, sr, is_squeeze, false_breaks, profile)
        st.plotly_chart(fig_main, use_container_width=True)

        # ===== Volume Profile =====
        st.subheader("🏗️ Volume Profile 20 Hari")
        if profile["levels"].empty:
            st.warning("Profil tidak tersedia (data terlalu sedikit).")
        else:
            st.caption(f"POC: **{fmt_rp(profile['poc'])}**, VA70%: **{fmt_rp(profile['va_low'])} – {fmt_rp(profile['va_high'])}**, "
                       f"Harga kini **{'di dalam' if profile['in_value_area'] else 'di luar'}** VA.")
            st.plotly_chart(make_profile_bar(profile), use_container_width=True)
            top = profile["levels"].sort_values("volume", ascending=False).head(10).copy()
            top["Harga"] = top["price"].map(fmt_rp); top["Volume (Lembar)"] = top["volume"].map(lambda v: fmt_int(v))
            st.table(top[["Harga","Volume (Lembar)"]])

        # ===== Rencana Trading =====
        st.subheader("🎯 Rencana Trading")
        if plan:
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Jenis", plan['type']); st.metric("Entry", fmt_rp(plan['entry']))
            with c2: st.metric("Stop Loss", fmt_rp(plan['stop_loss'])); st.metric("Target 1", fmt_rp(plan['target_1']))
            with c3: st.metric("Target 2", fmt_rp(plan['target_2'])); st.metric("Risk/Reward", f"{plan['risk_reward']}:1")
            st.metric("Ukuran Posisi", f"{fmt_int(plan['position_size'])} saham ({fmt_int(plan['position_size']/100)} lot)")
            st.info(f"- Risiko **{int(risk_percent*100)}%** dari modal {fmt_rp(account_size)}"
                    f"\n- Dibulatkan ke **tick size IDX**\n- Amati retest & keberlanjutan volume")
        else:
            st.warning("Belum ada rencana eksekusi karena trigger belum valid.\n"
                       f"- **Long**: butuh close > {fmt_rp(res_lvl * (1 + buffer))} + volume ≥ {VOL_ANOMALY_MULT}× MA20 "
                       f"({fmt_int(shares_to_lot(float(base['Volume_MA20'].iloc[-1])))} lot / ≈ {fmt_int(float(base['Volume_MA20'].iloc[-1]))} lembar).\n"
                       f"- **Short**: butuh close < {fmt_rp(sup_lvl * (1 - buffer))} + volume ≥ {VOL_ANOMALY_MULT}× MA20.")

        st.info("**Disclaimer**: Edukasi, bukan rekomendasi. Trading mengandung risiko.")

if __name__ == "__main__":
    app()
