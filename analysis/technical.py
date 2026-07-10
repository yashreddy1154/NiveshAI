"""
NiveshAI — Technical Analysis
Compute and interpret technical indicators from OHLCV data.
"""

import pandas as pd
import numpy as np
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Core Indicator Computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(close: pd.Series, fast=12, slow=26, signal=9) -> pd.DataFrame:
    ema_fast   = close.ewm(span=fast, adjust=False).mean()
    ema_slow   = close.ewm(span=slow, adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram  = macd_line - signal_line
    return pd.DataFrame({
        "MACD":        macd_line,
        "MACD_Signal": signal_line,
        "MACD_Hist":   histogram,
    })


def compute_bollinger(close: pd.Series, period=20, std_dev=2) -> pd.DataFrame:
    sma    = close.rolling(period).mean()
    std    = close.rolling(period).std()
    upper  = sma + std_dev * std
    lower  = sma - std_dev * std
    pct_b  = (close - lower) / (upper - lower + 1e-9)
    bw     = (upper - lower) / sma   # Bandwidth = volatility indicator
    return pd.DataFrame({
        "BB_Upper":  upper,
        "BB_Middle": sma,
        "BB_Lower":  lower,
        "BB_PctB":   pct_b,
        "BB_Width":  bw,
    })


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period=14) -> pd.Series:
    """Average True Range — measures volatility."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period=14) -> pd.Series:
    """Average Directional Index — trend strength (>25 = strong trend)."""
    up_move   = high - high.shift(1)
    down_move = low.shift(1) - low

    pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr    = compute_atr(high, low, close, period)
    pos_di = 100 * pd.Series(pos_dm, index=close.index).rolling(period).mean() / atr
    neg_di = 100 * pd.Series(neg_dm, index=close.index).rolling(period).mean() / atr

    dx  = 100 * (pos_di - neg_di).abs() / (pos_di + neg_di + 1e-9)
    adx = dx.rolling(period).mean()
    return adx


def compute_stochastic(high, low, close, k_period=14, d_period=3) -> pd.DataFrame:
    """Stochastic Oscillator %K and %D."""
    lowest  = low.rolling(k_period).min()
    highest = high.rolling(k_period).max()
    k = 100 * (close - lowest) / (highest - lowest + 1e-9)
    d = k.rolling(d_period).mean()
    return pd.DataFrame({"Stoch_K": k, "Stoch_D": d})


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume — cumulative volume direction indicator."""
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


# ─────────────────────────────────────────────────────────────────────────────
# Full Indicator Suite
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all technical indicators to an OHLCV DataFrame.
    Requires columns: Open, High, Low, Close, Volume
    """
    if df.empty or len(df) < 26:
        return df

    df = df.copy()
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]

    # Moving averages
    for p in [10, 20, 50, 100, 200]:
        df[f"SMA_{p}"] = c.rolling(p).mean()
        df[f"EMA_{p}"] = c.ewm(span=p, adjust=False).mean()

    # Momentum
    df["RSI"]    = compute_rsi(c)
    macd_df      = compute_macd(c)
    df           = pd.concat([df, macd_df], axis=1)

    # Volatility
    boll_df      = compute_bollinger(c)
    df           = pd.concat([df, boll_df], axis=1)
    df["ATR"]    = compute_atr(h, l, c)

    # Trend strength
    df["ADX"]    = compute_adx(h, l, c)

    # Stochastic
    stoch_df     = compute_stochastic(h, l, c)
    df           = pd.concat([df, stoch_df], axis=1)

    # Volume
    df["OBV"]        = compute_obv(c, v)
    df["Volume_MA20"] = v.rolling(20).mean()

    # Returns
    df["Daily_Return"] = c.pct_change() * 100
    df["Cumulative_Return"] = (1 + c.pct_change()).cumprod()

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Signal Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_signals(df: pd.DataFrame) -> dict:
    """
    Analyse the latest indicators and produce structured buy/sell/hold signals.

    Returns:
        {
          "overall":  "BUY" | "SELL" | "HOLD",
          "strength": float (0-1),
          "score":    int   (positive = bullish, negative = bearish),
          "signals":  list of {"indicator": str, "signal": str, "value": str, "bullish": bool}
        }
    """
    if df.empty:
        return {"overall": "HOLD", "strength": 0.5, "score": 0, "signals": []}

    row = df.iloc[-1]
    signals = []
    score   = 0  # positive = bullish votes, negative = bearish

    def add(indicator, label, value_str, bullish: bool, weight: int = 1):
        nonlocal score
        signals.append({
            "indicator": indicator,
            "signal":    label,
            "value":     value_str,
            "bullish":   bullish,
        })
        score += weight if bullish else -weight

    # ── RSI ──────────────────────────────────────────────────────────────────
    rsi = row.get("RSI", np.nan)
    if not np.isnan(rsi):
        if rsi < 30:
            add("RSI", f"Oversold ({rsi:.1f})", f"{rsi:.1f}", True, 2)
        elif rsi > 70:
            add("RSI", f"Overbought ({rsi:.1f})", f"{rsi:.1f}", False, 2)
        elif rsi > 50:
            add("RSI", f"Bullish ({rsi:.1f})", f"{rsi:.1f}", True, 1)
        else:
            add("RSI", f"Bearish ({rsi:.1f})", f"{rsi:.1f}", False, 1)

    # ── MACD ─────────────────────────────────────────────────────────────────
    macd_h = row.get("MACD_Hist", np.nan)
    macd   = row.get("MACD", np.nan)
    sig    = row.get("MACD_Signal", np.nan)
    if not np.isnan(macd_h):
        bullish = macd_h > 0
        label   = "Bullish crossover" if bullish else "Bearish crossover"
        add("MACD", label, f"Hist={macd_h:.4f}", bullish, 2)

    # ── Moving Averages ───────────────────────────────────────────────────────
    close  = row.get("Close", np.nan)
    sma20  = row.get("SMA_20", np.nan)
    sma50  = row.get("SMA_50", np.nan)
    sma200 = row.get("SMA_200", np.nan)

    if not np.isnan(close) and not np.isnan(sma20):
        bullish = close > sma20
        add("SMA 20", "Price above" if bullish else "Price below",
            f"₹{close:.0f} vs SMA₹{sma20:.0f}", bullish, 1)

    if not np.isnan(sma20) and not np.isnan(sma50):
        bullish = sma20 > sma50
        add("SMA 20/50", "Golden cross" if bullish else "Death cross",
            f"SMA20={sma20:.0f} SMA50={sma50:.0f}", bullish, 2)

    if not np.isnan(close) and not np.isnan(sma200):
        bullish = close > sma200
        add("SMA 200", "Long-term uptrend" if bullish else "Long-term downtrend",
            f"₹{close:.0f} vs SMA₹{sma200:.0f}", bullish, 1)

    # ── Bollinger Bands ────────────────────────────────────────────────────────
    pct_b = row.get("BB_PctB", np.nan)
    if not np.isnan(pct_b):
        if pct_b < 0.1:
            add("Bollinger Bands", "Price near lower band", f"%B={pct_b:.2f}", True, 1)
        elif pct_b > 0.9:
            add("Bollinger Bands", "Price near upper band", f"%B={pct_b:.2f}", False, 1)

    # ── ADX (Trend Strength) ──────────────────────────────────────────────────
    adx = row.get("ADX", np.nan)
    if not np.isnan(adx):
        if adx > 25:
            add("ADX", f"Strong trend ({adx:.1f})", f"{adx:.1f}", score > 0, 1)
        else:
            add("ADX", f"Weak/sideways trend ({adx:.1f})", f"{adx:.1f}", False, 0)

    # ── Stochastic ────────────────────────────────────────────────────────────
    stoch_k = row.get("Stoch_K", np.nan)
    stoch_d = row.get("Stoch_D", np.nan)
    if not np.isnan(stoch_k):
        if stoch_k < 20:
            add("Stochastic", f"Oversold (%K={stoch_k:.0f})", f"{stoch_k:.1f}", True, 1)
        elif stoch_k > 80:
            add("Stochastic", f"Overbought (%K={stoch_k:.0f})", f"{stoch_k:.1f}", False, 1)

    # ── Overall ───────────────────────────────────────────────────────────────
    max_score = sum(abs(s["bullish"] * 1 - (1 - int(s["bullish"]))) for s in signals) or 1
    strength  = min(1.0, abs(score) / max(1, len(signals)))

    if score >= 3:
        overall = "BUY"
    elif score <= -3:
        overall = "SELL"
    else:
        overall = "HOLD"

    return {
        "overall":  overall,
        "strength": round(strength, 2),
        "score":    score,
        "signals":  signals,
    }


def get_support_resistance(df: pd.DataFrame, lookback: int = 90) -> dict:
    """
    Compute simple support and resistance levels from recent price history.
    Support  = recent local lows cluster
    Resistance = recent local highs cluster
    """
    if df.empty or len(df) < lookback:
        return {"support": [], "resistance": []}

    recent = df.tail(lookback)
    highs  = recent["High"]
    lows   = recent["Low"]

    # Find local extremes (simple: rolling max/min peaks)
    resist_level = highs.rolling(5, center=True).max()
    support_level = lows.rolling(5, center=True).min()

    # Top 3 resistance and support levels
    resistance = sorted(
        resist_level.dropna().nlargest(5).unique().tolist(), reverse=True
    )[:3]
    support = sorted(
        support_level.dropna().nsmallest(5).unique().tolist()
    )[:3]

    return {
        "support":    [round(v, 2) for v in support],
        "resistance": [round(v, 2) for v in resistance],
    }
