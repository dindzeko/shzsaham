# modules/utils.py
import numpy as np
from .indicators import compute_mfi, compute_rsi, interpret_mfi
from .support_resistance import calculate_support_resistance

def calculate_additional_metrics(data):
    """
    Menghitung semua metrik teknikal tambahan untuk satu saham.
    Digunakan saat screening dan detail.
    """
    df = data.copy()
    metrics = {}
    
    # Moving Averages
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    
    # Indikator Momentum
    df['RSI'] = compute_rsi(df['Close'])
    df['MFI'] = compute_mfi(df, 14)
    
    # Volume Analysis
    df['Avg_Volume_20'] = df['Volume'].rolling(window=20).mean()
    vol_anomali = (df['Volume'].iloc[-1] > 1.7 * df['Avg_Volume_20'].iloc[-1]) if not df['Avg_Volume_20'].isna().iloc[-1] else False
    
    # Support & Resistance
    sr_levels = calculate_support_resistance(df)
    
    # Interpretasi MFI
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
        "Volume_Anomali": vol_anomali,
        "Support": sr_levels['Support'],
        "Resistance": sr_levels['Resistance'],
        "Fibonacci": sr_levels['Fibonacci']
    }
