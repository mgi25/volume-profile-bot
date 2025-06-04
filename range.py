import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import MetaTrader5 as mt5
import numpy as np
import ta
import time

# === MT5 CONFIG ===
SYMBOL = "XAUUSDm"
TIMEFRAME = mt5.TIMEFRAME_M1
LENGTH = 8              # Reduced for faster detection
ATR_LEN = 300           # Slightly tighter ATR window
MULT = 0.9              # Slightly narrower bands

# === Initialize MT5 ===
mt5.initialize()
mt5.symbol_select(SYMBOL)

# === LuxAlgo Detection Logic (Tuned) ===
def get_lux_ranges(df_bars, length=10, mult=1.0, atr_len=500, min_duration=3):
    closes = df_bars['close'].values
    highs = df_bars['high'].values
    lows = df_bars['low'].values
    times = pd.to_datetime(df_bars['time'], unit='s')

    ma = pd.Series(closes).rolling(length).mean().values
    atr = ta.volatility.average_true_range(
        pd.Series(highs), pd.Series(lows), pd.Series(closes), atr_len
    ).values * mult

    locked_ranges = []
    current_range = None
    duration = 0

    for i in range(length, len(closes)):
        ma_now = ma[i]
        atr_now = atr[i]
        if np.isnan(ma_now) or np.isnan(atr_now):
            continue

        top = ma_now + atr_now
        bottom = ma_now - atr_now
        count = sum(abs(closes[i - length:i] - ma_now) > atr_now)

        if count <= 1:  # allow 1 outlier (faster react to tight zones)
            if current_range is None:
                current_range = {
                    "start": times[i - length],
                    "end": times[i],
                    "top": top,
                    "bottom": bottom
                }
                duration = 1
            else:
                current_range["end"] = times[i]
                current_range["top"] = max(current_range["top"], top)
                current_range["bottom"] = min(current_range["bottom"], bottom)
                duration += 1
        elif current_range is not None:
            if duration >= min_duration:
                locked_ranges.append(current_range)
            current_range = None
            duration = 0

    if current_range is not None and duration >= min_duration:
        locked_ranges.append(current_range)

    return locked_ranges

# === Live Plot ===
def live_lux_plot():
    plt.ion()
    fig, ax = plt.subplots(figsize=(14, 6))
    while True:
        bars = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 600)
        if bars is None or len(bars) < 100:
            time.sleep(1)
            continue

        df_bars = pd.DataFrame(bars)
        df_bars['time'] = pd.to_datetime(df_bars['time'], unit='s')

        closes = df_bars['close']
        times = df_bars['time']

        locked_ranges = get_lux_ranges(df_bars, LENGTH, MULT, ATR_LEN)

        ax.clear()
        ax.plot(times, closes, label='Close', color='black')

        for idx, r in enumerate(locked_ranges):
            ax.axhline(r["top"], color='green', linestyle='--', alpha=0.6, label='Range Top' if idx == 0 else "")
            ax.axhline(r["bottom"], color='red', linestyle='--', alpha=0.6, label='Range Bottom' if idx == 0 else "")
            ax.fill_between(times, r["top"], r["bottom"],
                            where=(times >= r["start"]) & (times <= r["end"]),
                            color='blue', alpha=0.1, label='Locked Zone' if idx == 0 else "")

        ax.set_title(f'LuxAlgo Range Zone (Live) - {datetime.now().strftime("%H:%M:%S")}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax.grid(True)
        ax.legend(loc='upper right')
        plt.tight_layout()
        plt.pause(1)

live_lux_plot()
