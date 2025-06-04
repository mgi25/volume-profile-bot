import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, time, timezone
import time as ptime
import logging
import sys
from statsmodels.tsa.stattools import adfuller
import os
from datetime import datetime, timedelta
import ta
from ta.volatility import AverageTrueRange
from datetime import timezone
from decimal import Decimal
sys.stdout.reconfigure(encoding='utf-8')

# === CONFIGURATION ===
LOGIN = 204215535
PASSWORD = "Mgi@2005"
SERVER = "Exness-MT5Trial7"
SYMBOL = "XAUUSDm"
TIMEFRAME = mt5.TIMEFRAME_M1

# === STRATEGY SETTINGS ===
LOT_SIZE = 0.01
PRICE_STEP = 0.01
SKEW_THRESHOLD = 0.05
VOL_SPIKE_FACTOR = 0.8
MAGIC = 123456
HEDGE_TRIGGER_PERCENT = 2
PROFIT_SCALP_TARGET = 0.50
SESSION_START = time(0, 0)   # 00:00 UTC -> 5:30 AM IST
SESSION_END = time(23, 59)   # 23:59 UTC -> 5:29 AM IST next day
# Global Day session range -> 9:30 AM to 8:30 PM IST (04:00 to 15:00 UTC)
DAY_SESSION_START = time(0, 0)
DAY_SESSION_END = time(23, 59)

SPREAD_LIMIT = 0.20
# === RECOVERY SNIPER SETTINGS ===
last_sniper_time = datetime.min
recovery_snipers_fired = 0
MAX_RECOVERY_SNIPERS = 3
current_leg = 1
entry_sequence = []
hedge_mode = False
last_price_snapshot = None
last_entry_side = None
price_left_zone = True
last_sniper_time = datetime.min
recovery_snipers_fired = 0
locked_active = False
locked_loss = 0.0
recovery_attempts = 0
post_lock_recovery = False
base_entry_price = None  # 🔑 This tracks the fixed starting price for all hedge legs
hedged_tickets = set()
# Map recovery sniper ticket -> hedge ticket for grouped management
recovery_hedge_pairs = {}

# === CONFIGURATION ===
LOT_SIZE = 0.01
SPREAD_LIMIT = 1.5
PROFIT_SCALP_TARGET = 0.25
HEDGE_TRIGGER_PIPS = 50  # <--- 🔥 Add this
post_lock_recovery_pnl = 0.0

# === LOGGING ===
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler('live_bot.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[console_handler, file_handler]
)

# === GLOBAL STATE ===
entry_sequence = []
current_leg = 1
hedge_mode = False
last_price_snapshot = 0
last_entry_side = None
price_left_zone = True

locked_active = False
locked_loss = 0.0

# === CONSOLIDATION DETECTION ===
def hurst_exponent(prices):
    prices = np.asarray(prices)
    N = len(prices)
    if N < 20:
        return 0.5
    var1 = np.var(np.diff(prices))
    Hs = []
    for k in [2, 5, 10]:
        if k < N:
            var_k = np.var(prices[k:] - prices[:-k])
            if var1 > 0 and var_k > 0:
                Hs.append(0.5 * np.log(var_k / var1) / np.log(k))
    return max(0.0, min(1.0, np.mean(Hs))) if Hs else 0.5

def detect_consolidation(prices, window=100):
    prices = np.asarray(prices)
    if len(prices) < window:
        return 0.0, False

    # ATR Compression (low volatility)
    atrs = np.abs(np.diff(prices))
    atr_current = np.mean(atrs[-14:])
    atr_prev = np.mean(atrs[-50:-14]) + 1e-9
    atr_compression = 1.0 - min(1.0, atr_current / atr_prev)

    # Bollinger Band Width
    rolling_mean = pd.Series(prices).rolling(20).mean()
    rolling_std = pd.Series(prices).rolling(20).std()
    upper = rolling_mean + 2 * rolling_std
    lower = rolling_mean - 2 * rolling_std
    bbw = np.mean((upper - lower) / (rolling_mean + 1e-9))
    bbw_score = 1.0 - min(1.0, bbw / (np.std(prices) + 1e-9))

    # Volume profile flatness - simulated using price range compactness
    price_range = max(prices[-window:]) - min(prices[-window:])
    price_std = np.std(prices[-window:])
    vp_flatness = 1.0 - min(1.0, price_std / (price_range + 1e-9))

    # Hurst exponent
    hurst = hurst_exponent(prices)
    hurst_score = 1.0 - min(1.0, abs(hurst - 0.5) / 0.5)

    # ADF stationarity
    try:
        adf_p = adfuller(prices)[1]
    except:
        adf_p = 1.0
    adf_score = max(0.0, 1.0 - adf_p)

    # Final Score
    score = np.mean([atr_compression, bbw_score, vp_flatness, hurst_score, adf_score])
    return score, score >= 0.7  # Now uses stricter 0.7 threshold

# === UTILITIES ===
def get_effective_balance():
    info = mt5.account_info()
    return 100 if info is None or info.balance < 110 else info.balance

def is_momentum_candle(closes, highs, lows, volumes):
    if len(closes) < 20:
        return False

    closes = np.array(closes)
    highs = np.array(highs)
    lows = np.array(lows)
    volumes = np.array(volumes)

    opens = closes[:-1]
    body_size = abs(closes[-1] - opens[-1])
    full_size = highs[-1] - lows[-1] + 1e-9
    body_ratio = body_size / full_size

    atr = np.mean(np.maximum(
        highs[-14:] - lows[-14:],
        np.maximum(
            np.abs(highs[-14:] - closes[-15:-1]),
            np.abs(lows[-14:] - closes[-15:-1])
        )
    ))

    volume_now = volumes[-1]
    avg_volume = np.mean(volumes[-20:])

    return (body_ratio > 0.6) and (full_size > atr) and (volume_now > avg_volume * 1.1)

def calc_skew(df_ticks, return_poc=False, min_tick_volume=1):
    if df_ticks.empty:
        return (None, None) if return_poc else None

    df = df_ticks.copy()
    df['price'] = (df['bid'] + df['ask']) / 2
    df['volume'] = df['volume'].replace(0, min_tick_volume)
    df['bin'] = (df['price'] / PRICE_STEP).round() * PRICE_STEP

    vp = df.groupby('bin')['volume'].sum().sort_index()
    if vp.empty:
        return (None, None) if return_poc else None

    poc = vp.idxmax()
    total_volume = vp.sum()
    cum_volume = 0.0
    va_bins = []

    for p, v in vp.sort_values(ascending=False).items():
        cum_volume += v
        va_bins.append(p)
        if cum_volume >= 0.7 * total_volume:
            break

    val, vah = min(va_bins), max(va_bins)
    width = vah - val
    mid = (val + vah) / 2
    skew = (poc - mid) / width if width > 0 else 0

    # Additional: POC drift signal
    last_poc = getattr(calc_skew, "last_poc", None)
    calc_skew.last_poc = poc
    poc_drift = abs(poc - last_poc) if last_poc else 0

    if return_poc:
        return skew, poc, poc_drift
    return skew

def is_trap_zone_expanding(entry_sequence, threshold=3.0):
    if len(entry_sequence) < 3:
        return False
    prices = [p[0] for p in entry_sequence]
    trap_range = max(prices) - min(prices)
    return trap_range > threshold
def get_higher_tf_bias():
    bars = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M15, 0, 30)
    if not bars or len(bars) < 20:
        return None

    closes = np.array([b['close'] for b in bars])
    ema_short = pd.Series(closes).ewm(span=5).mean().iloc[-1]
    ema_long = pd.Series(closes).ewm(span=20).mean().iloc[-1]

    return 'BULLISH' if ema_short > ema_long else 'BEARISH'

# === MT5 FUNCTIONS ===
def initialize():
    if not mt5.initialize(server=SERVER, login=LOGIN, password=PASSWORD):
        logging.error(f"MT5 init failed: {mt5.last_error()}")
        raise SystemExit
    mt5.symbol_select(SYMBOL, True)
    logging.info("Bot initialized")

def shutdown():
    mt5.shutdown()
    logging.info("MT5 shutdown")

def should_block_hedge_final(entry_sequence, bars, df_ticks, 
                                 drift_threshold_factor=1.5, max_legs_in_chop=3):
    """
    Block hedging if:
    - Still in consolidation
    - Not enough price drift outside VA
    - Already has too many legs
    """
    if not entry_sequence or len(bars) < 30 or df_ticks.empty:
        return False

    entry_price = entry_sequence[-1][0]
    latest_price = df_ticks['ask'].iloc[-1] if 'ask' in df_ticks.columns else df_ticks['price'].iloc[-1]

    # === 1. Consolidation confirmation using volatility ratio
    highs = [b['high'] for b in bars[-30:]]
    lows = [b['low'] for b in bars[-30:]]
    total_range = max(highs) - min(lows)
    avg_range = np.mean([h - l for h, l in zip(highs, lows)])
    chop_score = avg_range / total_range if total_range != 0 else 1

    is_choppy = chop_score < 0.2  # tighter range = more consolidation

    # === 2. Volume profile drift calc
    df_ticks['price'] = (df_ticks['bid'] + df_ticks['ask']) / 2
    df_ticks['volume'] = df_ticks['volume'].replace(0, 1)
    df_ticks['bin'] = (df_ticks['price'] / 0.01).round() * 0.01
    vp = df_ticks.groupby('bin')['volume'].sum().sort_index()

    if vp.empty:
        return False

    total_volume = vp.sum()
    cum_volume = 0
    va_bins = []
    for price, vol in vp.sort_values(ascending=False).items():
        cum_volume += vol
        va_bins.append(price)
        if cum_volume >= 0.7 * total_volume:
            break

    val, vah = min(va_bins), max(va_bins)
    va_width = vah - val
    price_drift = abs(latest_price - entry_price)
    drifted = price_drift > drift_threshold_factor * va_width

    # === 3. Final Decision
    if is_choppy and len(entry_sequence) >= max_legs_in_chop and not drifted:
        logging.info("[BLOCKED] Too many legs in consolidation with no breakout")
        return True
    return False

def place_entry(side, price, volume=LOT_SIZE, comment='VP_ENTRY'):
    order_type = mt5.ORDER_TYPE_BUY if side == 'BUY' else mt5.ORDER_TYPE_SELL
    req = {
        'action': mt5.TRADE_ACTION_DEAL,
        'symbol': SYMBOL,
        'volume': volume,
        'type': order_type,
        'price': price,
        'deviation': 10,
        'magic': MAGIC,
        'comment': comment,
        'type_time': mt5.ORDER_TIME_GTC,
        'type_filling': mt5.ORDER_FILLING_IOC
    }
    result = mt5.order_send(req)
    logging.info(f"{comment} -> {side} at {price:.3f}, vol={volume}, retcode={result.retcode}")
    return result

def close_position(pos, comment=''):
    tick = mt5.symbol_info_tick(SYMBOL)
    close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
    close_price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask
    req = {
        'action': mt5.TRADE_ACTION_DEAL,
        'symbol': SYMBOL,
        'volume': pos.volume,
        'type': close_type,
        'position': pos.ticket,
        'price': close_price,
        'deviation': 10,
        'magic': MAGIC,
        'comment': comment,
        'type_time': mt5.ORDER_TIME_GTC,
        'type_filling': mt5.ORDER_FILLING_IOC
    }
    mt5.order_send(req)
    logging.warning(f"{comment} @ {close_price:.2f} for ticket {pos.ticket}")

def close_all_positions():
    for pos in mt5.positions_get(symbol=SYMBOL) or []:
        close_position(pos, comment='AUTO_EXIT')

def rebuild_state():
    global entry_sequence, current_leg, hedge_mode, last_entry_side, price_left_zone, base_entry_price
    positions = mt5.positions_get(symbol=SYMBOL)
    if not positions:
        logging.info("No open positions. Fresh start.")
        return
    entry_sequence.clear()
    for pos in sorted(positions, key=lambda x: x.time):
        side = 'BUY' if pos.type == mt5.POSITION_TYPE_BUY else 'SELL'
        entry_sequence.append((pos.price_open, side, pos.volume))
    current_leg = len(entry_sequence)
    last_entry_side = entry_sequence[-1][1]
    hedge_mode = len(entry_sequence) > 1
    price_left_zone = True
    base_entry_price = entry_sequence[0][0] if entry_sequence else None


def log_live_equity(spread=0.0, skew=0.0, vol_check=False, momentum_ok=False, 
                    consolidation_score=0.0, is_consolidating=False, 
                    action_taken="HOLD"):
    account = mt5.account_info()
    positions = mt5.positions_get(symbol=SYMBOL) or []
    lot_size = positions[-1].volume if positions else 0
    pnl = sum(p.profit for p in positions)
    timestamp = datetime.now(timezone.utc)

    file_exists = os.path.isfile("ml_trading_log.csv")
    with open("ml_trading_log.csv", "a") as f:
        if not file_exists:
            f.write("timestamp,equity,balance,lot_size,pnl,spread,skew,vol_check,momentum_ok,consolidation_score,is_consolidating,hedge_mode,current_leg,action_taken\n")
        f.write(f"{timestamp},{account.equity:.2f},{account.balance:.2f},{lot_size:.2f},{pnl:.2f},{spread:.5f},{skew:.5f},{int(vol_check)},{int(momentum_ok)},{consolidation_score:.5f},{int(is_consolidating)},{int(hedge_mode)},{current_leg},{action_taken}\n")

def init_csv_file():
    """Ensure the ML logging file exists and has the correct header."""
    filepath = "ml_trading_log.csv"
    if not os.path.isfile(filepath):
        with open(filepath, "w") as f:
            f.write("timestamp,equity,balance,lot_size,pnl,spread,skew,vol_check,momentum_ok,consolidation_score,is_consolidating,hedge_mode,current_leg,action_taken\n")
def is_consolidation(bars, atr_period=14, window=20, atr_contract_thresh=0.8):
    if len(bars) < atr_period + 1 or len(bars) < window:
        return False

    tr = []
    for i in range(1, len(bars)):
        h, l, pc = bars[i]['high'], bars[i]['low'], bars[i - 1]['close']
        tr.append(max(h - l, abs(h - pc), abs(l - pc)))

    curr_atr = sum(tr[-atr_period:]) / atr_period
    long_term = sum(tr) / len(tr)
    vol_contracted = (long_term > 0) and (curr_atr / long_term < atr_contract_thresh)

    highs = [b['high'] for b in bars[-window:]]
    lows = [b['low'] for b in bars[-window:]]
    band = max(highs) - min(lows)
    tight_band = band < 1.5 * curr_atr

    return vol_contracted and tight_band

def should_force_hedge_consolidation(tick_price, entry_price, df_ticks, bars, entry_sequence,
                                     min_breakout_factor=2.0, min_volume_ratio=1.2, momentum_lookback=2):
    if df_ticks.empty or len(bars) < 20 or not entry_sequence:
        return False

    # === Volume Profile Setup
    df_ticks['price'] = (df_ticks['bid'] + df_ticks['ask']) / 2
    df_ticks['volume'] = df_ticks['volume'].replace(0, 1)
    df_ticks['bin'] = (df_ticks['price'] / PRICE_STEP).round() * PRICE_STEP
    vp = df_ticks.groupby('bin')['volume'].sum().sort_index()
    if vp.empty:
        return False

    total_vol = vp.sum()
    cum_vol = 0
    va_bins = []
    for price, vol in vp.sort_values(ascending=False).items():
        cum_vol += vol
        va_bins.append(price)
        if cum_vol >= 0.7 * total_vol:
            break
    val, vah = min(va_bins), max(va_bins)
    va_width = vah - val + 1e-9
    price_drift = abs(tick_price - entry_price)
    breakout_factor = price_drift / va_width

    # === Trap Rejection Check
    trap_low, trap_high = get_trap_zone(entry_sequence)
    if trap_low and trap_high and trap_low < tick_price < trap_high:
        logging.info("[TRAP] Price still inside trap bounds. No hedge.")
        return False

    # === Volume Confirmation
    recent_volumes = [b['tick_volume'] for b in bars[-20:]]
    vol_now = recent_volumes[-1]
    avg_vol = np.mean(recent_volumes[:-1])
    volume_ok = vol_now > min_volume_ratio * avg_vol

    # === Momentum Check
    closes = [b['close'] for b in bars[-(momentum_lookback + 1):]]
    if len(closes) < momentum_lookback + 1:
        return False
    momentum_ok = abs(closes[-1] - closes[-2]) > 0.1

    # === Strong Candle Confirmation
    strong_candle = is_strong_candle([b['close'] for b in bars], [b['high'] for b in bars], [b['low'] for b in bars])

    # === Final Logic
    if breakout_factor >= min_breakout_factor and volume_ok and momentum_ok and strong_candle:
        logging.info(f"[HEDGE CONFIRMED] BreakoutFactor={breakout_factor:.2f} | VolOK={volume_ok} | Momentum={momentum_ok} | CandleStrong={strong_candle}")
        return True

    logging.info(f"[HEDGE BLOCKED] BreakoutFactor={breakout_factor:.2f} | VolOK={volume_ok} | Momentum={momentum_ok} | CandleStrong={strong_candle}")
    return False


def handle_recovery_trap(entry_sequence, bars, df_ticks, place_entry_func,
                         trap_zone_width=2.0, momentum_lookback=2, sniper_cooldown_minutes=5,
                         max_snipers=2, max_volume=0.16):
    global last_sniper_time, recovery_snipers_fired

    if not entry_sequence or len(bars) < 20 or df_ticks.empty:
        return False

    trap_prices = [entry[0] for entry in entry_sequence]
    trap_center = np.mean(trap_prices)
    trap_upper = trap_center + trap_zone_width
    trap_lower = trap_center - trap_zone_width
    tick_price = df_ticks['ask'].iloc[-1] if 'ask' in df_ticks.columns else df_ticks['price'].iloc[-1]

    if trap_lower <= tick_price <= trap_upper:
        return False  # Still in trap, wait for breakout

    if datetime.now() - last_sniper_time < timedelta(minutes=sniper_cooldown_minutes):
        return False
    if recovery_snipers_fired >= max_snipers:
        return False

    closes = [b['close'] for b in bars[-(momentum_lookback + 1):]]
    if len(closes) < momentum_lookback + 1:
        return False
    direction = closes[-1] - closes[-2]
    if abs(direction) < 0.1:
        return False

    if not is_strong_candle([b['close'] for b in bars], [b['high'] for b in bars], [b['low'] for b in bars]):
        return False

    if not value_area_breakout(df_ticks, tick_price):
        return False

    recent_volumes = [b['tick_volume'] for b in bars[-20:]]
    vol_now = recent_volumes[-1]
    avg_vol = np.mean(recent_volumes[:-1])
    if vol_now < 1.2 * avg_vol:
        return False

    # 🔁 Exponential lot size with cap
    volume = min(max_volume, round(0.01 * (2 ** recovery_snipers_fired), 2))
    side = 'BUY' if direction > 0 else 'SELL'
    price = tick_price

    logging.info(f"[SNIPER READY] Side={side}, Price={price:.2f}, Vol={vol_now:.2f}, Dir={direction:.2f}, Lot={volume}")

    result = place_entry_func(side, price, volume, comment='TRAP_RECOVERY_SNIPER_V3')
    if hasattr(result, 'retcode') and result.retcode == 10009:
        recovery_snipers_fired += 1
        last_sniper_time = datetime.now()
        entry_sequence.append((price, side, volume))
        return True

    return False


def is_strong_candle(closes, highs, lows):
    if len(closes) < 2:
        return False
    open_price = closes.iloc[-2] if hasattr(closes, 'iloc') else closes[-2]
    close_price = closes.iloc[-1] if hasattr(closes, 'iloc') else closes[-1]
    high = highs.iloc[-1] if hasattr(highs, 'iloc') else highs[-1]
    low = lows.iloc[-1] if hasattr(lows, 'iloc') else lows[-1]

    body = abs(close_price - open_price)
    wick = (high - low) - body
    if (body > wick) and (body > (np.std(closes[-20:]) * 0.5)):
        return True
    return False

def detect_trap_escape(entry_sequence, current_price, recent_volumes, threshold=1.5):
    if not entry_sequence or len(recent_volumes) < 10:
        return False
    trap_prices = [p[0] for p in entry_sequence]
    trap_center = np.mean(trap_prices)
    trap_range = max(abs(current_price - min(trap_prices)), abs(current_price - max(trap_prices)))

    # Check volume breakout
    avg_vol = np.mean(recent_volumes[:-1])
    vol_now = recent_volumes[-1]
    escaped = (trap_range > 1.5) and (vol_now > threshold * avg_vol)
    return escaped

def value_area_breakout(df_ticks, price, factor=1.5):
    if df_ticks.empty:
        return False

    df_ticks['price'] = (df_ticks['bid'] + df_ticks['ask']) / 2
    df_ticks['volume'] = df_ticks['volume'].replace(0, 1)
    df_ticks['bin'] = (df_ticks['price'] / PRICE_STEP).round() * PRICE_STEP

    vp = df_ticks.groupby('bin')['volume'].sum().sort_index()
    if vp.empty:
        return False

    total_vol = vp.sum()
    cum_vol = 0
    va_bins = []
    for bin_price, vol in vp.sort_values(ascending=False).items():
        cum_vol += vol
        va_bins.append(bin_price)
        if cum_vol >= 0.7 * total_vol:
            break

    val = min(va_bins)
    vah = max(va_bins)
    va_width = vah - val
    outside_va = price < val or price > vah
    distance = abs(price - ((val + vah) / 2))
    return outside_va and distance > (factor * va_width)

def is_high_confidence_entry(skew, spread, vol_check, momentum_ok, is_consolidating, strong_candle, poc_drift=0.0):
    confirmations = [
        abs(skew) >= SKEW_THRESHOLD,
        spread <= SPREAD_LIMIT,
        vol_check,
        momentum_ok,
        strong_candle,
        not is_consolidating,
        poc_drift > 0.02  # new confirmation: is POC moving?
    ]
    score = sum(confirmations)
    logging.info(f"[CONFIRMATION SCORE] {score}/7 -> {'PASS' if score >= 4 else 'FAIL'}")
    return score >= 4
def should_avoid_hedge_on_chop(entry_sequence, bars, df_ticks, skew, max_range=1.0):
    if len(bars) < 20 or not entry_sequence:
        return False

    closes = [b['close'] for b in bars]
    price_range = max(closes) - min(closes)
    recent_skew_flat = abs(skew) < 0.02

    trap_score = get_trap_score(entry_sequence)
    return (price_range < max_range) and recent_skew_flat and trap_score > 0.6


def get_value_area_bounds(df_ticks):
    if df_ticks.empty:
        return None, None

    df_ticks['price'] = (df_ticks['bid'] + df_ticks['ask']) / 2
    df_ticks['volume'] = df_ticks['volume'].replace(0, 1)
    df_ticks['bin'] = (df_ticks['price'] / PRICE_STEP).round() * PRICE_STEP

    vp = df_ticks.groupby('bin')['volume'].sum().sort_index()
    if vp.empty:
        return None, None

    total_vol = vp.sum()
    cum_vol = 0
    va_bins = []
    for bin_price, vol in vp.sort_values(ascending=False).items():
        cum_vol += vol
        va_bins.append(bin_price)
        if cum_vol >= 0.7 * total_vol:
            break

    return min(va_bins), max(va_bins)

def get_trap_zone(entry_sequence):
    if not entry_sequence:
        return None, None
    prices = [p[0] for p in entry_sequence]
    center = np.mean(prices)
    width = max(abs(center - min(prices)), abs(center - max(prices)))
    return center - width, center + width

def log_entry_criteria(skew, spread, vol_check, momentum_ok, is_consolidating, strong_candle):
    log_str = "[CONFIRMATION] -> "
    log_str += f"Skew={abs(skew) >= SKEW_THRESHOLD}, "
    log_str += f"Spread={spread <= SPREAD_LIMIT}, "
    log_str += f"Volume={vol_check}, "
    log_str += f"Momentum={momentum_ok}, "
    log_str += f"StrongCandle={strong_candle}, "
    log_str += f"NoConsolidation={not is_consolidating}"
    logging.info(log_str)

def get_trade_bias(closes, skew):
    if skew > 0.05 and closes[-1] > closes[-2]:
        return "BULLISH"
    elif skew < -0.05 and closes[-1] < closes[-2]:
        return "BEARISH"
    return "NEUTRAL"
def draw_console_debug_info(price, va_low, va_high, trap_low, trap_high):
    logging.info(f"[ZONE] Price={price:.2f}, VA=[{va_low:.2f}, {va_high:.2f}], Trap=[{trap_low:.2f}, {trap_high:.2f}]")

def is_stack_allowed(entry_sequence, bars, df_ticks):
    if len(entry_sequence) < 2:
        return True

    # Check if we’re in consolidation and already have too many entries
    if is_consolidation(bars) and len(entry_sequence) >= 3:
        logging.warning("[STACK BLOCK] Too many entries inside consolidation")
        return False

    # Check if trap zone is too wide
    trap_prices = [entry[0] for entry in entry_sequence]
    trap_width = max(trap_prices) - min(trap_prices)
    if trap_width > 5.0:
        logging.warning(f"[STACK BLOCK] Trap zone width too high: {trap_width:.2f}")
        return False

    return True

def is_entry_bias_valid(skew, closes):
    bias = get_trade_bias(closes, skew)
    if bias == "BULLISH" and skew < 0:
        logging.info("[BIAS FILTER] Bias is BULLISH but skew is SELL -> Blocked")
        return False
    elif bias == "BEARISH" and skew > 0:
        logging.info("[BIAS FILTER] Bias is BEARISH but skew is BUY -> Blocked")
        return False
    return True

def should_throttle_due_to_drawdown():
    #acc = mt5.account_info()
    #if not acc:
     #   return False
    #drawdown = acc.balance - acc.equity
    #if drawdown >= 5:  # tweak as needed
     #   logging.warning(f"[THROTTLE] Drawdown too high: {drawdown:.2f} -> throttle entries")
      #  return True
    return False

def is_night_session():
    hour = datetime.now(timezone.utc).hour
    return not (DAY_SESSION_START <= datetime.now(timezone.utc).time() <= DAY_SESSION_END)

def is_strong_body_candle(close, open_, high, low, min_body_ratio=0.6):
    body = abs(close - open_)
    full_range = high - low + 1e-9
    return (body / full_range) >= min_body_ratio

def is_sniper_entry(closes, opens, highs, lows, volumes, skew, spread, is_consolidating):
    if len(closes) < 20:
        return False

    close = closes[-1]
    open_ = opens[-1]
    high = highs[-1]
    low = lows[-1]
    volume_now = volumes[-1]
    avg_volume = np.mean(volumes[-20:])

    body_ok = is_strong_body_candle(close, open_, high, low)
    vol_ok = volume_now > avg_volume * VOL_SPIKE_FACTOR
    momentum_ok = is_momentum_candle(closes, highs, lows, volumes)
    strong_candle = is_strong_candle(closes, highs, lows)
    skew_ok = abs(skew) >= SKEW_THRESHOLD
    spread_ok = spread <= SPREAD_LIMIT
    no_consolidation = not is_consolidating

    # ✅ HTF Bias filter
    bias = get_higher_tf_bias()
    bias_valid = True
    if bias:
        bias_valid = (bias == 'BULLISH' and skew > 0) or (bias == 'BEARISH' and skew < 0)

    confirmations = [
        body_ok,
        vol_ok,
        momentum_ok,
        strong_candle,
        skew_ok,
        spread_ok,
        no_consolidation
    ]

    passed = sum(confirmations)
    logging.info(f"[SNIPER CHECK] Body={body_ok} | Vol={vol_ok} | Momentum={momentum_ok} | Strong={strong_candle} | Skew={skew_ok} | Spread={spread_ok} | NoConsolidation={no_consolidation} | BiasMatch={bias_valid} -> Score: {passed}/7")

    return passed >= 6 and bias_valid

def get_higher_tf_bias():
    bars_15m = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M15, 0, 50)
    if not bars_15m or len(bars_15m) < 2:
        return None
    close_now = bars_15m[-1]['close']
    close_prev = bars_15m[-2]['close']
    return 'BULLISH' if close_now > close_prev else 'BEARISH'

def trap_breakout_confirmed(entry_sequence, current_price, recent_prices, confirmation_candles=3, min_distance=0.5):
    if not entry_sequence or len(recent_prices) < confirmation_candles:
        return False

    trap_low, trap_high = get_trap_zone(entry_sequence)
    if not trap_low or not trap_high:
        return False

    # Price must close outside trap for confirmation_candles straight
    outside_count = 0
    for p in recent_prices[-confirmation_candles:]:
        if p < trap_low - min_distance or p > trap_high + min_distance:
            outside_count += 1

    return outside_count >= confirmation_candles

def get_trap_score(entry_sequence, max_allowed_entries=5, max_zone_width=3.0):
    if not entry_sequence:
        return 0.0

    prices = [p[0] for p in entry_sequence]
    zone_width = max(prices) - min(prices)
    entry_count = len(entry_sequence)

    width_score = min(1.0, zone_width / max_zone_width)
    entry_score = min(1.0, entry_count / max_allowed_entries)

    trap_score = 0.5 * width_score + 0.5 * entry_score
    return trap_score  # closer to 1 = stale, dangerous trap
def get_lux_ranges(df_bars, length=10, mult=1.0, atr_len=500, min_duration=3):
    closes = df_bars['close'].values
    highs = df_bars['high'].values
    lows = df_bars['low'].values
    times = pd.to_datetime(df_bars['time'], unit='s', utc=True)

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

        if count <= 1:  # allow 1 outlier
            if current_range is None:
                current_range = {
                    "start": times[i - length].replace(tzinfo=timezone.utc),
                    "end": times[i].replace(tzinfo=timezone.utc),
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


def is_inside_locked_range(price, time_now, locked_ranges):
    """Check if price is inside *any* of the currently active LuxAlgo range zones."""
    active_ranges = [r for r in locked_ranges if r["start"] <= time_now <= r["end"]]
    for r in active_ranges:
        if r["bottom"] <= price <= r["top"]:
            return True
    return False
last_action_time = datetime.now()

def update_last_action():
    global last_action_time
    last_action_time = datetime.now()
def is_100_percent_trade(skew, spread, vol_check, momentum_ok, is_consolidating, strong_candle, poc_drift):
    confirmations = [
        abs(skew) >= SKEW_THRESHOLD,
        spread <= SPREAD_LIMIT,
        vol_check,
        momentum_ok,
        strong_candle,
        not is_consolidating,
        poc_drift > 0.02
    ]
    return sum(confirmations) == 7
def price_left_lux_range(price, locked_ranges, current_time):
    """Check if price has moved outside any locked LuxAlgo range."""
    for r in locked_ranges:
        if r['start'] <= current_time <= r['end']:
            if price < r['bottom'] or price > r['top']:
                return True
    return False

def get_confirmation_score(skew, spread, vol_ok, momentum_ok, is_consolidating, strong_candle, poc_drift):
    confirmations = [
        abs(skew) >= SKEW_THRESHOLD,
        spread <= SPREAD_LIMIT,
        vol_ok,
        momentum_ok,
        strong_candle,
        not is_consolidating,
        poc_drift > 0.02
    ]
    score = sum(confirmations)
    logging.info(f"[CONFIRMATION SCORE] {score}/7 -> {'PASS' if score >= 4 else 'FAIL'}")
    return score

def get_trade_direction_based_on_skew_or_momentum(skew, closes):
    """Smart direction chooser for recovery"""
    if skew > 0.05 and closes[-1] > closes[-2]:
        return 'BUY'
    elif skew < -0.05 and closes[-1] < closes[-2]:
        return 'SELL'
    return 'BUY' if closes[-1] > closes[-2] else 'SELL'

def calculate_cumulative_hedge_lot(entry_sequence):
    base_lot = Decimal("0.01")
    hedge_lots = next_hedge_lot = round(LOT_SIZE * (current_leg + 1), 2)
    return round(base_lot + sum(hedge_lots), 2) if hedge_lots else base_lot * 2
def reset_state():
    global entry_sequence, locked_active, locked_loss, post_lock_recovery, recovery_attempts
    global base_entry_price, current_leg, post_lock_recovery_pnl
    global recovery_snipers_fired, recovery_hedge_count, recovery_hedge_pairs

    post_lock_recovery_pnl = 0.0
    entry_sequence = []
    locked_active = False
    locked_loss = 0.0
    post_lock_recovery = False
    recovery_attempts = 0
    base_entry_price = None
    current_leg = 0
    recovery_snipers_fired = 0
    recovery_hedge_count = 0
    recovery_hedge_pairs = {}


def log_hedge_debug(price_now, expected_price, base_entry_price, current_leg):
    logging.info(f"[HEDGE DEBUG] Now={price_now:.2f}, Expected={expected_price:.2f}, Base={base_entry_price:.2f}, Leg={current_leg}")

def log_trade(entry_sequence):
    import csv
    from datetime import datetime
    import os

    if not entry_sequence:
        return

    filename = "Log.csv"
    file_exists = os.path.isfile(filename)

    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Group_ID", "Timestamp", "Price", "Side", "Volume", "Type"])

        group_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamp = datetime.now(timezone.utc).isoformat()

        for trade in entry_sequence:
            row = [
                group_id,
                timestamp,
                trade[0],  # Price
                trade[1],  # Side
                trade[2],  # Volume
                trade[3] if len(trade) > 3 else ("LOCK" if locked_active else "ENTRY")
            ]
            writer.writerow(row)
def update_locked_loss_with_profit(pnl_gain):
    global locked_loss
    locked_loss = round(locked_loss + pnl_gain, 2)
    logging.info(f"[RECOVERY UPDATE] Locked Loss Adjusted: New Locked Loss = {locked_loss:.2f}")
def get_last_position():
    pos = mt5.positions_get(symbol=SYMBOL)
    return pos[-1] if pos else None

def main():
    global current_leg, entry_sequence, hedge_mode, last_price_snapshot, last_entry_side, price_left_zone
    global last_sniper_time, recovery_snipers_fired, locked_active, locked_loss
    global recovery_attempts, post_lock_recovery, recovery_hedge_count, hedge_price_levels, base_entry_price
    global recovery_hedge_pairs, hedged_tickets

    initialize()
    init_csv_file()
    rebuild_state()

    current_leg = 0
    recovery_hedge_count = 0
    hedge_price_levels = []
    base_entry_price = None
    recovery_hedge_pairs = {}
    hedged_tickets = set()
    post_lock_recovery_pnl = 0.0

    logging.info("Bot initialized")

    while True:
        tick = mt5.symbol_info_tick(SYMBOL)
        if not tick:
            ptime.sleep(1)
            continue

        now = datetime.now(timezone.utc).time()
        if now < SESSION_START or now > SESSION_END:
            logging.info(f"[SESSION BLOCKED] Time={now} outside trading session")
            ptime.sleep(1)
            continue

        spread = tick.ask - tick.bid
        if spread > SPREAD_LIMIT:
            logging.info(f"[SPREAD BLOCKED] Spread={spread:.2f} > Limit")
            ptime.sleep(1)
            continue

        price = tick.ask
        positions = mt5.positions_get(symbol=SYMBOL) or []
        total_pnl = sum(p.profit for p in positions)

        bars = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 600)
        if bars is None or len(bars) < 100:
            ptime.sleep(1)
            continue

        df_bars = pd.DataFrame(bars)
        df_bars['time'] = pd.to_datetime(df_bars['time'], unit='s')
        closes = df_bars['close'].tolist()
        time_now = df_bars['time'].iloc[-1].to_pydatetime().replace(tzinfo=timezone.utc)

        locked_ranges = get_lux_ranges(df_bars)
        active_range = next((r for r in locked_ranges if r['start'] <= time_now <= r['end']), None)
        in_active_range = bool(active_range and active_range['bottom'] <= price <= active_range['top'])

        # ✅ Auto-lock after 3 entries
        if not locked_active and len(entry_sequence) == 3:
            buy_lots = sum(p.volume for p in positions if p.type == mt5.POSITION_TYPE_BUY)
            sell_lots = sum(p.volume for p in positions if p.type == mt5.POSITION_TYPE_SELL)
            diff = round(abs(buy_lots - sell_lots), 2)
            if diff > 0:
                lock_side = 'SELL' if buy_lots > sell_lots else 'BUY'
                lock_price = tick.bid if lock_side == 'SELL' else tick.ask
                result = place_entry(lock_side, lock_price, diff, comment='AUTO_LOCK_4TH_ENTRY')
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    logging.warning(f"[FORCED LOCK] {lock_side} {diff:.2f} @ {lock_price:.2f}")
                    entry_sequence.append((lock_price, lock_side, diff, "AUTO_LOCK_4TH_ENTRY"))
                    locked_active = True
                    locked_loss = abs(total_pnl)
                    post_lock_recovery = True
                    current_leg = 0
                    recovery_hedge_count = 0
                    continue

        # 📌 Exit if profit hit
        if positions and not locked_active and total_pnl >= PROFIT_SCALP_TARGET:
            logging.warning(f"[EXIT] Profit ${total_pnl:.2f} hit -> Closing all")
            log_trade(entry_sequence)
            close_all_positions()
            reset_state()
            continue

        # 🔒 Lock if inside Lux range and unbalanced
        if positions and not locked_active and in_active_range:
            buy_lots = sum(p.volume for p in positions if p.type == mt5.POSITION_TYPE_BUY)
            sell_lots = sum(p.volume for p in positions if p.type == mt5.POSITION_TYPE_SELL)
            diff = round(abs(buy_lots - sell_lots), 2)
            if diff > 0:
                lock_side = 'SELL' if buy_lots > sell_lots else 'BUY'
                lock_price = tick.bid if lock_side == 'SELL' else tick.ask
                result = place_entry(lock_side, lock_price, diff, comment='LOCK_INTO_RANGE')
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    logging.warning(f"[LOCK] {lock_side} {diff:.2f} @ {lock_price:.2f}")
                    entry_sequence.append((lock_price, lock_side, diff, "LOCK_INTO_RANGE"))
                    locked_active = True
                    locked_loss = abs(total_pnl)
                    post_lock_recovery = True
                    current_leg = 0
                    recovery_hedge_count = 0
                    continue

        # 🔁 Recovery mode with sniper + hedge group handling
        if locked_active and post_lock_recovery:
            snipers = [p for p in positions if p.comment == "RECOVERY_SNIPER"]
            hedges = [p for p in positions if p.comment == "RECOVERY_HEDGE"]

            for sniper in snipers:
                if sniper.ticket in recovery_hedge_pairs:
                    hedge_ticket = recovery_hedge_pairs[sniper.ticket]
                    hedge = next((h for h in hedges if h.ticket == hedge_ticket), None)
                    if hedge:
                        combined_pnl = sniper.profit + hedge.profit
                        if combined_pnl >= PROFIT_SCALP_TARGET:
                            close_position(sniper, comment="[GROUP TP - SNIPER]")
                            close_position(hedge, comment="[GROUP TP - HEDGE]")
                            logging.info(f"[GROUP TP] Sniper+Hedge profit: ${combined_pnl:.2f}")
                            locked_loss -= combined_pnl
                            del recovery_hedge_pairs[sniper.ticket]
                    else:
                        # Hedge missing, treat sniper individually
                        if sniper.profit >= PROFIT_SCALP_TARGET:
                            close_position(sniper, comment="[SNIPER TP]")
                            logging.info(f"[SNIPER TP] +${sniper.profit:.2f}")
                            locked_loss -= sniper.profit
                            recovery_hedge_pairs.pop(sniper.ticket, None)
                else:
                    if sniper.profit >= PROFIT_SCALP_TARGET:
                        close_position(sniper, comment="[SNIPER TP]")
                        logging.info(f"[SNIPER TP] +${sniper.profit:.2f}")
                        locked_loss -= sniper.profit

            if locked_loss <= 0 and total_pnl >= 0:
                logging.info(f"[RECOVERY COMPLETE] Locked loss recovered → Closing all")
                log_trade(entry_sequence)
                close_all_positions()
                reset_state()
                continue

            if len(snipers) + len(hedges) >= 4:
                logging.warning("[MAX RECOVERY REACHED] Closing all trades")
                log_trade(entry_sequence)
                close_all_positions()
                reset_state()
                continue

            for p in snipers:
                if p.ticket in recovery_hedge_pairs:
                    continue
                loss_pips = (p.price_open - tick.bid) * 100 if p.type == mt5.POSITION_TYPE_BUY else (tick.ask - p.price_open) * 100
                if loss_pips >= HEDGE_TRIGGER_PIPS:
                    hedge_side = 'SELL' if p.type == mt5.POSITION_TYPE_BUY else 'BUY'
                    hedge_price = tick.bid if hedge_side == 'SELL' else tick.ask
                    lot = round(p.volume + 0.01, 2)
                    result = place_entry(hedge_side, hedge_price, lot, comment="RECOVERY_HEDGE")
                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        logging.warning(f"[RECOVERY HEDGE] {hedge_side} {lot:.2f} @ {hedge_price:.2f}")
                        entry_sequence.append((hedge_price, hedge_side, lot, "RECOVERY_HEDGE"))
                        recovery_hedge_pairs[p.ticket] = result.order
                        recovery_hedge_count += 1
                        current_leg += 1

            if in_active_range:
                logging.info("[SNIPER BLOCKED] Price still inside Lux range")
            elif len(snipers) >= 1:
                logging.info("[WAITING] One sniper trade already active")
            else:
                from_time = datetime.now(timezone.utc) - timedelta(minutes=5)
                ticks = mt5.copy_ticks_from(SYMBOL, from_time, 3000, mt5.COPY_TICKS_ALL)
                df_ticks = pd.DataFrame(ticks)
                if not df_ticks.empty:
                    skew, poc, poc_drift = calc_skew(df_ticks, return_poc=True)
                    vol_ok = df_bars['tick_volume'].iloc[-1] > df_bars['tick_volume'].mean()
                    momentum_ok = is_momentum_candle(closes, df_bars['high'], df_bars['low'], df_bars['tick_volume'])
                    strong = is_strong_candle(closes, df_bars['high'], df_bars['low'])
                    confidence = get_confirmation_score(skew, spread, vol_ok, momentum_ok, False, strong, poc_drift)

                    if confidence >= 4:
                        side = get_trade_direction_based_on_skew_or_momentum(skew, closes)
                        entry_price = tick.ask if side == "BUY" else tick.bid
                        lot = round(0.01 + 0.01 * recovery_hedge_count, 2)
                        result = place_entry(side, entry_price, lot, comment="RECOVERY_SNIPER")
                        if result.retcode == mt5.TRADE_RETCODE_DONE:
                            logging.info(f"[RECOVERY SNIPER] {side} {lot:.2f} @ {entry_price:.2f}")
                            entry_sequence.append((entry_price, side, lot, "RECOVERY_SNIPER"))
                            last_entry_side = side
                            current_leg += 1
                            recovery_attempts += 1
            ptime.sleep(0.3)
            continue

        # 🛡️ Hedge any open trade
        if not locked_active and positions:
            for p in positions:
                if p.ticket in hedged_tickets:
                    continue
                loss_pips = (p.price_open - tick.bid) * 100 if p.type == mt5.POSITION_TYPE_BUY else (tick.ask - p.price_open) * 100
                if loss_pips >= HEDGE_TRIGGER_PIPS:
                    hedge_side = 'SELL' if p.type == mt5.POSITION_TYPE_BUY else 'BUY'
                    hedge_price = tick.bid if hedge_side == 'SELL' else tick.ask
                    lot = round(p.volume + 0.01, 2)
                    result = place_entry(hedge_side, hedge_price, lot, comment="HEDGE_LAYER")
                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        logging.warning(f"[HEDGE] {hedge_side} {lot:.2f} @ {hedge_price:.2f}")
                        entry_sequence.append((hedge_price, hedge_side, lot, "HEDGE_LAYER"))
                        hedged_tickets.add(p.ticket)
                        current_leg += 1

        # 🚀 VP Entry
        if not positions and not locked_active and not in_active_range:
            from_time = datetime.now(timezone.utc) - timedelta(minutes=5)
            ticks = mt5.copy_ticks_from(SYMBOL, from_time, 3000, mt5.COPY_TICKS_ALL)
            df_ticks = pd.DataFrame(ticks)
            if df_ticks.empty:
                ptime.sleep(1)
                continue

            skew, poc, poc_drift = calc_skew(df_ticks, return_poc=True)
            vol_ok = df_bars['tick_volume'].iloc[-1] > df_bars['tick_volume'].mean()
            momentum_ok = is_momentum_candle(closes, df_bars['high'], df_bars['low'], df_bars['tick_volume'])
            strong = is_strong_candle(closes, df_bars['high'], df_bars['low'])

            if is_high_confidence_entry(skew, spread, vol_ok, momentum_ok, False, strong, poc_drift):
                side = "BUY" if skew > 0 else "SELL"
                entry_price = tick.ask if side == "BUY" else tick.bid
                lot = 0.01
                result = place_entry(side, entry_price, lot, comment="VP_ENTRY")
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    logging.info(f"[VP_ENTRY] {side} {lot:.2f} @ {entry_price:.2f}")
                    entry_sequence.append((entry_price, side, lot, "VP_ENTRY"))
                    base_entry_price = entry_price
                    last_entry_side = side
                    current_leg = 1
                    continue

        ptime.sleep(1)

    shutdown()



if __name__ == '__main__':
    try:
        main()
    except Exception:
        logging.exception("Bot crashed")
    finally:
        shutdown()
        
        