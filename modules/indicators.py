# modules/indicators.py
import numpy as np
import pandas as pd

def compute_mfi(df, period=14):
    """
    Menghitung Money Flow Index (MFI) dengan metode yang benar.
    Input: DataFrame dengan kolom High, Low, Close, Volume.
    Output: Series MFI.
    """
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
    """
    Memberikan interpretasi sinyal MFI untuk trading.
    """
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


def compute_rsi(close, period=14):
    """
    Menghitung Relative Strength Index (RSI).
    """
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
