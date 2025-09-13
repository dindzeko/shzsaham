# analisa_saham.py
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta
from scipy import stats

# =========================
# KONFIGURASI GLOBAL
# =========================
st.set_page_config(page_title="Analisa Saham IDX", layout="wide")

# Ambang volume anomali (ganti 1.5 -> 1.3 / 2.0 sesuai kebutuhan)
VOLUME_ANOMALY_MULT = 1.5

# =========================
# UTILITAS FORMAT
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
# DATA
# =========================
def resolve_ticker(user_input: str) -> str:
    t = user_input.strip().upper()
    if not t.endswith(".JK"):
        t += ".JK"
    return t

@st.cache_data(ttl=600, show_spinner=False)
def fetch_history_daily(ticker_jk: str, end: datetime) -> pd.DataFrame:
    start = end - timedelta(days=120)  # <= ~3 bulan kalender
    raw = yf.download(
        ticker_jk,
        start=start.strftime("%Y-%m-%d"),
        end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False,
        group_by="column",
        interval="1d",
    )
    if raw is None or raw.empty:
        return raw
    df = normalize_ohlcv(raw)
    return df

@st.cache_data(ttl=900, show_spinner=False)
def fetch_history_5m_20d(ticker_jk: str) -> pd.DataFrame:
    # ambil 60 hari 5m (batas yfinance), lalu pilih 20 hari perdagangan terakhir
    raw = yf.download(
        ticker_jk, period="60d", interval="5m",
        auto_adjust=False, progress=False, group_by="column"
    )
    if raw is None or raw.empty: 
        return raw
    df = normalize_ohlcv(raw)
    # Pilih 20 sesi terakhir
    sessions = df.index.normalize().unique()
    sessions = sessions[-20:]
    df = df[df.index.normalize().isin(sessions)]
    return df

def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return df
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.droplevel(1, axis=1)
        except Exception:
            df = df.droplevel(0, axis=1)
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
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")
    for c in required + (['Adj Close'] if 'Adj Close' in df.columns else []):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df[required + (['Adj Close'] if 'Adj Close' in df.columns else [])].dropna(how='all')
    df.index = pd.to_datetime(df.index)
    return df

# =========================
# INDIKATOR
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

def compute_macd(close: pd.Series, fast=12, slow=26, signal=9):
    close = as_series(close)
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return macd, sig, hist

def compute_bollinger_bands(close: pd.Series, window=20, num_std=2):
    close = as_series(close)
    sma = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    width = (upper - lower)
    percent_b = (close - lower) / width.replace(0, np.nan)
    return upper, sma, lower, (width/sma.replace(0,np.nan)).fillna(0), percent_b.clip(0,1).fillna(0)

def compute_atr(high, low, close, period=14):
    high, low, close = as_series(high), as_series(low), as_series(close)
    tr = pd.concat([
        (high - low),
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return rma(tr, period)

def compute_obv(close, volume):
    sign = np.sign(as_series(close).diff().fillna(0))
    return (as_series(volume) * sign).cumsum().fillna(0)

def compute_adl(high, low, close, volume):
    high, low, close, volume = as_series(high), as_series(low), as_series(close), as_series(volume)
    mfm = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    mfm = mfm.replace([np.inf, -np.inf], 0).fillna(0)
    mfv = mfm * volume
    return mfv.cumsum()

# =========================
# SUPPORT/RESIST (60 bar) + FIB
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

def compute_support_resistance(df: pd.DataFrame, lookback=60) -> dict:
    close = float(df['Close'].iloc[-1])
    swing_high, swing_low = identify_swings(df, lookback)
    fib = fibonacci_levels(swing_high, swing_low)
    ma20 = float(df['Close'].rolling(20).mean().iloc[-1])
    ma50 = float(df['Close'].rolling(50).mean().iloc[-1])
    ma100 = float(df['Close'].rolling(100).mean().iloc[-1])
    vwap = compute_vwap(df['High'], df['Low'], df['Close'], df['Volume']).iloc[-1]
    candidates = [fib[k] for k in ['Fib_0.236','Fib_0.382','Fib_0.5','Fib_0.618','Fib_0.786','Fib_0.0','Fib_1.0']]
    candidates += [ma20, ma50, ma100, float(vwap)]
    candidates = [c for c in candidates if np.isfinite(c)]
    clustered = cluster_levels(candidates, tol=0.01)
    supports = [lvl for (lvl, _) in clustered if lvl < close]
    resistances = [lvl for (lvl, _) in clustered if lvl > close]
    supports = sorted(supports, reverse=True)[:3] if supports else [close*0.95]
    resistances = sorted(resistances)[:3] if resistances else [close*1.05]
    return {'Support': supports, 'Resistance': resistances, 'Fibonacci': fib}

def compute_vwap(high, low, close, volume):
    high, low, close, volume = as_series(high), as_series(low), as_series(close), as_series(volume)
    tp = (high + low + close) / 3
    cum_v = volume.cumsum().replace(0, np.nan)
    vwap = (tp * volume).cumsum() / cum_v
    return vwap.fillna(method="bfill").fillna(method="ffill")

# =========================
# SCORING TEKNIKAL
# =========================
class IndicatorScoringSystem:
    def __init__(self):
        pass

    @staticmethod
    def _trend(values: pd.Series, period: int) -> float:
        s = as_series(values).dropna()
        if len(s) < max(3, period): return 0.0
        y = s.iloc[-period:].to_numpy(dtype=float).ravel()
        x = np.arange(y.size, dtype=float)
        slope, _, _, _, _ = stats.linregress(x, y)
        mean = float(np.mean(y)) if np.mean(y) != 0 else 1e-9
        return float(slope / mean)

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
# FALSE BREAK DETECTOR
# =========================
def detect_false_breaks(df: pd.DataFrame, sr: dict, lookback=60):
    """
    Kriteria:
      - Upthrust: High > R1 lalu Close kembali < R1 (vol >= 1.5×MA20)
      - Spring  : Low < S1 lalu Close kembali > S1 (vol >= 1.5×MA20)
    """
    if df is None or df.empty: 
        return []
    d = df.copy().iloc[-lookback:]
    ma20v = d['Volume'].rolling(20).mean()
    r1 = sr['Resistance'][0] if sr['Resistance'] else None
    s1 = sr['Support'][0] if sr['Support'] else None
    events = []

    for idx, row in d.iterrows():
        vol_ok = row['Volume'] >= VOLUME_ANOMALY_MULT * (ma20v.loc[idx] if not np.isnan(ma20v.loc[idx]) else 0)
        if r1 and (row['High'] > r1) and (row['Close'] < r1) and vol_ok:
            events.append({"type":"Upthrust","date":idx.date(),"level":r1,
                           "high":row['High'],"close":row['Close'],"vol":row['Volume']})
        if s1 and (row['Low'] < s1) and (row['Close'] > s1) and vol_ok:
            events.append({"type":"Spring","date":idx.date(),"level":s1,
                           "low":row['Low'],"close":row['Close'],"vol":row['Volume']})
    return events

# =========================
# VOLUME PROFILE (5m, 20 hari)
# =========================
def volume_profile_from_5m(df5m: pd.DataFrame, bins=40):
    """
    Distribusi volume per level harga dengan pendekatan overlap (fair-division).
    Return:
      volume_by_bin, bin_edges, poc_price, va_low, va_high, top10_df
    """
    if df5m is None or df5m.empty:
        return None, None, None, None, None, pd.DataFrame()

    recent = df5m.copy()
    price_min = recent['Low'].min()
    price_max = recent['High'].max()
    if not np.isfinite(price_min) or not np.isfinite(price_max) or price_max <= price_min:
        return None, None, None, None, None, pd.DataFrame()

    bin_edges = np.linspace(price_min, price_max, bins + 1)
    volume_by_bin = np.zeros(bins)

    for _, r in recent.iterrows():
        rng = r['High'] - r['Low']
        if rng <= 0: 
            continue
        vpp = r['Volume'] / rng
        for i in range(bins):
            lo, hi = bin_edges[i], bin_edges[i+1]
            ol = max(lo, r['Low']); oh = min(hi, r['High'])
            ov = max(0.0, oh - ol)
            volume_by_bin[i] += ov * vpp

    # POC & Value Area (70% rule)
    poc_idx = int(np.argmax(volume_by_bin))
    poc_price = (bin_edges[poc_idx] + bin_edges[poc_idx+1]) / 2
    total_v = volume_by_bin.sum()
    sorted_idx = np.argsort(volume_by_bin)[::-1]
    acc, chosen = 0.0, []
    for idx in sorted_idx:
        acc += volume_by_bin[idx]
        chosen.append(idx)
        if acc >= total_v * 0.7:
            break
    va_low = bin_edges[min(chosen)]
    va_high = bin_edges[max(chosen) + 1]

    # Top 10 harga (bin) terbesar
    top_idx = sorted_idx[:10]
    rows = []
    for i in top_idx:
        price_mid = (bin_edges[i] + bin_edges[i+1]) / 2
        rows.append({"Harga": price_mid, "Volume (Lembar)": volume_by_bin[i]})
    top10_df = pd.DataFrame(rows).sort_values("Volume (Lembar)", ascending=False)

    return volume_by_bin, bin_edges, float(poc_price), float(va_low), float(va_high), top10_df

# =========================
# CHARTS
# =========================
def make_main_chart(df: pd.DataFrame, sr: dict, ma5, ma20, ma200, bbU, bbM, bbL, false_breaks) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name="Candlestick", increasing_line_color='green', decreasing_line_color='red'
    ))
    # MA lines
    fig.add_trace(go.Scatter(x=df.index, y=ma5, mode='lines', name="MA5", line=dict(color='green', width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=ma20, mode='lines', name="MA20", line=dict(color='blue', width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=ma200, mode='lines', name="MA200", line=dict(color='black', width=1.5)))
    # Bollinger
    fig.add_trace(go.Scatter(x=df.index, y=bbU, mode='lines', name="BB Upper", line=dict(color="red", width=1, dash="dot")))
    fig.add_trace(go.Scatter(x=df.index, y=bbM, mode='lines', name="BB Middle", line=dict(color="purple", width=1.2)))
    fig.add_trace(go.Scatter(x=df.index, y=bbL, mode='lines', name="BB Lower", line=dict(color="green", width=1, dash="dot")))

    # S/R lines
    for i, lvl in enumerate(sr['Support']):
        fig.add_hline(y=lvl, line_dash="dash", line_color="green",
                      annotation_text=f"S{i+1}: {fmt_rp(lvl)}", annotation_position="bottom right")
    for i, lvl in enumerate(sr['Resistance']):
        fig.add_hline(y=lvl, line_dash="dash", line_color="red",
                      annotation_text=f"R{i+1}: {fmt_rp(lvl)}", annotation_position="top right")

    # Anotasi false break terakhir (jika ada)
    for ev in false_breaks[-3:]:  # tampilkan maks 3
        txt = "FB↑" if ev["type"]=="Upthrust" else "FB↓"
        ts = pd.Timestamp(ev['date'])
        if ts in df.index:
            y_ = df.loc[ts, 'High'] if ev["type"]=="Upthrust" else df.loc[ts, 'Low']
            fig.add_annotation(x=ts, y=y_, text=txt, showarrow=True, arrowhead=2, ax=0, ay=-20, font=dict(color="gray"))

    # Layout anti-terpotong
    fig.update_layout(
        title="Chart Teknikal (≤ 3 bulan)",
        template="plotly_white",
        xaxis_rangeslider_visible=False,
        height=560,
        margin=dict(l=40, r=140, t=60, b=40),  # r besar agar label 'Rp ...' tidak terpotong
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='x unified'
    )
    return fig

def make_volume_profile_chart(vp, edges, poc, vaL, vaH, last_price) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=edges[:-1], y=vp, name="Volume Profile"
    ))
    fig.add_vline(x=poc, line_dash="dash", line_color="red", annotation_text="POC")
    fig.add_vrect(x0=vaL, x1=vaH, fillcolor="green", opacity=0.12, line_width=0, annotation_text="Value Area")
    fig.add_vline(x=last_price, line_dash="dot", line_color="blue", annotation_text="Harga")
    fig.update_layout(
        title="📊 Volume Profile (5m • 20 hari)",
        xaxis_title="Harga",
        yaxis_title="Volume (lembar)",
        template="plotly_white",
        height=320,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig

# =========================
# PANEL RINGKASAN
# =========================
def technical_summary(df, ma5, ma20, ma200, rsi, macd_line, macd_sig, adx, sr, false_breaks):
    close = float(df['Close'].iloc[-1])
    bbU, bbM, bbL, bbw, _ = compute_bollinger_bands(df['Close'])
    squeeze = float(np.nanpercentile(bbw.tail(120).dropna(), 20)) > 0 and float(bbw.iloc[-1]) <= float(np.nanpercentile(bbw.tail(120).dropna(), 20))
    fb_txt = ""
    if false_breaks:
        last = false_breaks[-1]
        if last["type"]=="Upthrust":
            fb_txt = f"• False break **atas** terakhir {last['date']} dekat R1 ({fmt_rp(last['level'])}). "
        else:
            fb_txt = f"• False break **bawah** terakhir {last['date']} dekat S1 ({fmt_rp(last['level'])}). "
    trend_str = "di atas MA200 (bullish struktural)" if close >= ma200.iloc[-1] else "di bawah MA200 (bearish struktural)"
    mom = f"RSI {rsi.iloc[-1]:.0f}, MACD {'cross up' if macd_line.iloc[-1]>macd_sig.iloc[-1] else 'cross down'}"
    adx_str = f"ADX {adx.iloc[-1]:.0f}"
    bb_str = "BB squeeze" if squeeze else "BB tidak squeeze"
    r1 = fmt_rp(sr['Resistance'][0]) if sr['Resistance'] else "-"
    s1 = fmt_rp(sr['Support'][0]) if sr['Support'] else "-"
    lines = [
        f"**Trend**: {trend_str}; MA5 {fmt_rp(ma5.iloc[-1])}, MA20 {fmt_rp(ma20.iloc[-1])}, MA200 {fmt_rp(ma200.iloc[-1])}, {bb_str}.",
        f"**Momentum**: {mom}, {adx_str}.",
        f"**Level kunci**: R1 {r1}, S1 {s1}.",
        fb_txt.strip()
    ]
    return " ".join([x for x in lines if x])

def bandarmology_summary(df_daily, df5m, vp, edges, poc, vaL, vaH, top10_df):
    # A/D days 20H
    cl = df_daily['Close']
    chg = cl.diff()
    up_days = int((chg.tail(20) > 0).sum())
    dn_days = int((chg.tail(20) < 0).sum())

    # OBV slope (10)
    obv = compute_obv(df_daily['Close'], df_daily['Volume'])
    obv_slope = IndicatorScoringSystem._trend(obv, 10)

    # Volume anomaly streak
    v = df_daily['Volume']
    ma20v = v.rolling(20).mean()
    anom = v >= (VOLUME_ANOMALY_MULT * ma20v)
    streak = 0
    for val in anom[::-1]:
        if val: streak += 1
        else: break

    # Posisi harga relatif VA
    close = float(df_daily['Close'].iloc[-1])
    if poc is None:
        pos_va = "—"
    else:
        if close > vaH: pos_va = "di atas VAH (attempt acceptance)"
        elif close < vaL: pos_va = "di bawah VAL (rejection risk)"
        else: pos_va = "di dalam VA"

    # Skor sederhana 0..100
    score = 50
    score += np.clip(obv_slope*80, -15, 15)
    score += np.clip((up_days - dn_days)*2, -20, 20)
    if close > vaH: score += 10
    if close < vaL: score -= 10
    score = int(np.clip(score, 0, 100))
    label = "Accumulation" if score >= 66 else ("Neutral" if score >= 40 else "Distribution")

    # Big bars footprint (5m): top 10% nilai -> lokasi close dalam bar
    footprint = "—"
    if df5m is not None and not df5m.empty:
        val5 = df5m['Close']*df5m['Volume']
        thr = np.nanpercentile(val5.dropna(), 90)
        top = df5m[val5 >= thr].copy()
        if not top.empty:
            clv = (top['Close'] - top['Low']) / (top['High'] - top['Low']).replace(0, np.nan)
            low = (clv <= 0.33).mean()
            mid = ((clv>0.33) & (clv<0.66)).mean()
            high= (clv >= 0.66).mean()
            footprint = f"Low {low*100:.0f}% • Mid {mid*100:.0f}% • High {high*100:.0f}%"

    # Rangkai
    lines = [
        f"**Status**: Skor **{score}/100 – {label}**.",
        f"**A/D 20H**: +{up_days}/-{dn_days} • **OBV slope**: {obv_slope:+.2f}.",
        f"**Posisi terhadap VA**: {pos_va}.",
        f"**POC/VA**: POC {fmt_rp(poc) if poc else '-'}, VA {fmt_rp(vaL) if vaL else '-'} – {fmt_rp(vaH) if vaH else '-'}.",
        f"**Volume anomaly streak**: {streak} hari (≥ {VOLUME_ANOMALY_MULT:.1f}×MA20).",
        f"**Where big bars hit (5m)**: {footprint}."
    ]

    # Aksi jika sudah pegang di harga close
    atr = compute_atr(df_daily['High'], df_daily['Low'], df_daily['Close']).iloc[-1]
    stop_level = (vaL - atr) if vaL else (close - 2*atr)
    action_lines = []
    if label == "Accumulation":
        action_lines = [
            f"**Hold & tambah selektif** saat retest **POC/VAH** dengan volume ≥ {VOLUME_ANOMALY_MULT:.1f}×MA20.",
            f"**Target**: R1 lalu R2; **Stop** awal di **{fmt_rp(stop_level)}** (≈ VAL − 1×ATR)."
        ]
    elif label == "Neutral":
        action_lines = [
            f"**Hold ringan**; trading tepi kotak **VAL↔VAH**. Tambah hanya di retest valid (volume normal–tinggi).",
            f"**Stop**: break **VAL − 1×ATR ≈ {fmt_rp(stop_level)}**."
        ]
    else:
        action_lines = [
            f"**Kurangi saat pantulan** ke **VAH/R1** (risiko upthrust).",
            f"Tambah baru **tidak disarankan** sebelum re-acceptance di atas R1 + volume ambang."
        ]

    return " ".join(lines), " ".join(action_lines), score, label

# =========================
# APP
# =========================
def app():
    st.title("📊 Analisa Saham IDX – Teknikal + Bandarmology")
    c1, c2 = st.columns([2,1])
    with c1:
        ticker_in = st.text_input("Kode Saham", value="BBCA").strip().upper()
    with c2:
        as_of = st.date_input("📅 Tanggal Analisis (as-of)", value=datetime.today())

    if not ticker_in:
        st.stop()

    ticker = resolve_ticker(ticker_in)
    df = fetch_history_daily(ticker, datetime.combine(as_of, datetime.min.time()))
    if df is None or df.empty:
        st.error("Data tidak tersedia."); st.stop()

    # ===== Indikator dasar
    base = df.copy()
    ma5 = base['Close'].rolling(5).mean()
    ma20 = base['Close'].rolling(20).mean()
    ma200 = base['Close'].rolling(200).mean()
    bbU, bbM, bbL, _, _ = compute_bollinger_bands(base['Close'], 20, 2)
    rsi = compute_rsi(base['Close'], 14)
    macd_line, macd_sig, _ = compute_macd(base['Close'])
    # ADX sederhana: gunakan ATR-based DI (approx) — agar ringan, pakai proxy
    # (Untuk ringkas, tampilkan nilai ATR sebagai indikator volatilitas)
    atr = compute_atr(base['High'], base['Low'], base['Close'])

    # S/R & Fibo (60 bar)
    sr = compute_support_resistance(base, 60)

    # False breaks (60 bar)
    fb_events = detect_false_breaks(base, sr, 60)

    # ===== Panel indikator singkat
    last_close = float(base['Close'].iloc[-1])
    last_vol_shares = float(base['Volume'].iloc[-1])
    last_val = last_close * last_vol_shares
    colA, colB, colC = st.columns(3)
    with colA:
        st.metric("MA5 (harga)", fmt_rp(ma5.iloc[-1]))
    with colB:
        st.metric("Volume Perdagangan (Lot)", fmt_int(shares_to_lot(last_vol_shares)))
    with colC:
        st.metric("Nilai Transaksi (Rp)", fmt_rp(last_val))

    st.caption(f"Data terakhir: **{base.index[-1].date()}** (n={len(base)})")

    # ===== Chart utama (≤3 bulan)
    st.subheader("🖼️ Chart Teknikal (≤ 3 bulan)")
    fig_main = make_main_chart(base, sr, ma5, ma20, ma200, bbU, bbM, bbL, fb_events)
    st.plotly_chart(fig_main, use_container_width=True)

    # ===== Volume Profile 5m (20 hari)
    df5m = fetch_history_5m_20d(ticker)
    vp, edges, poc, vaL, vaH, top10 = volume_profile_from_5m(df5m, bins=40)
    if vp is not None:
        fig_vp = make_volume_profile_chart(vp, edges, poc, vaL, vaH, last_close)
        st.plotly_chart(fig_vp, use_container_width=True)
        st.write("**Top 10 harga paling aktif (5m • 20 hari)**")
        if not top10.empty:
            td = top10.copy()
            td['Harga'] = td['Harga'].map(fmt_rp)
            td['Volume (Lembar)'] = td['Volume (Lembar)'].map(fmt_int)
            st.table(td.reset_index(drop=True))
        else:
            st.info("Belum ada data top-10 harga.")

    # ===== Dua panel kesimpulan (baru)
    st.subheader("🧠 Kesimpulan Analisa Teknikal")
    # ADX proxy: gunakan slope MA20 terhadap MA200 sebagai kekuatan trend sederhana
    adx_proxy = (ma20.iloc[-1] - ma20.iloc[-14]) / max(ma20.iloc[-14], 1e-9) * 100 if len(ma20.dropna())>20 else 0
    teknikal = technical_summary(base, ma5, ma20, ma200, rsi, macd_line, macd_sig, atr, sr, fb_events)
    st.write(teknikal)

    st.subheader("🕵️ Kesimpulan Bandarmology + Aksi (anggap Anda pegang di harga Close)")
    band_sum, action_txt, score, label = bandarmology_summary(base, df5m, vp, edges, poc, vaL, vaH, top10)
    st.markdown(band_sum)
    st.markdown(f"**Aksi (harga pegang = {fmt_rp(last_close)}):** {action_txt}")

    # ===== Level Fibonacci (60 bar) – ringkas
    with st.expander("🔢 Level Fibonacci (60 bar)"):
        fib_rows = [{"Level": k.replace('_',' ').replace('Fib ','Fib '), "Harga": fmt_rp(v)} for k,v in sr['Fibonacci'].items()]
        st.table(pd.DataFrame(fib_rows))

    # ===== False break log
    with st.expander("⚠️ False Break log (60 bar terakhir)"):
        if not fb_events:
            st.info("Tidak ada upthrust/spring valid (≥ ambang volume).")
        else:
            rows = []
            for ev in fb_events:
                if ev["type"]=="Upthrust":
                    rows.append({"Tanggal": ev['date'], "Tipe":"Upthrust", "Level": fmt_rp(ev['level']),
                                 "High": fmt_rp(ev['high']), "Close": fmt_rp(ev['close']),
                                 "Vol": fmt_int(ev['vol']), "Kriteria": f"≥ {VOLUME_ANOMALY_MULT:.1f}×MA20"})
                else:
                    rows.append({"Tanggal": ev['date'], "Tipe":"Spring", "Level": fmt_rp(ev['level']),
                                 "Low": fmt_rp(ev['low']), "Close": fmt_rp(ev['close']),
                                 "Vol": fmt_int(ev['vol']), "Kriteria": f"≥ {VOLUME_ANOMALY_MULT:.1f}×MA20"})
            st.table(pd.DataFrame(rows))

    st.info("**Disclaimer**: Edukasi, bukan rekomendasi. Selalu sesuaikan dengan profil risiko Anda.")

if __name__ == "__main__":
    app()
