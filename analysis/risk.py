"""
NiveshAI — Risk Assessment
Computes volatility metrics, drawdown, VaR, and an overall risk score
from a stock's historical price data.
"""

import numpy as np
import pandas as pd
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Core Risk Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_volatility(returns: pd.Series, annualise: bool = True) -> float:
    """
    Compute standard deviation of daily returns.
    Annualised by multiplying by √252 (trading days).
    """
    vol = returns.std()
    return float(vol * np.sqrt(252) if annualise else vol)


def compute_max_drawdown(prices: pd.Series) -> float:
    """
    Maximum peak-to-trough decline as a fraction (e.g. -0.35 = -35%).
    """
    cummax = prices.cummax()
    drawdown = (prices - cummax) / cummax
    return float(drawdown.min())


def compute_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Historical Value at Risk at the given confidence level.
    Returns the loss threshold (negative float, e.g. -0.03 = 3% one-day loss).
    """
    return float(np.percentile(returns.dropna(), (1 - confidence) * 100))


def compute_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Conditional VaR (Expected Shortfall): average loss beyond the VaR threshold.
    """
    var = compute_var(returns, confidence)
    tail = returns[returns <= var]
    return float(tail.mean()) if len(tail) > 0 else var


def compute_beta(stock_returns: pd.Series, market_returns: pd.Series) -> float:
    """
    Beta vs market index (NIFTY 50).
    Beta > 1 = more volatile than market.
    """
    if len(stock_returns) < 20 or len(market_returns) < 20:
        return 1.0
    # Align
    combined = pd.concat([stock_returns, market_returns], axis=1).dropna()
    if combined.empty or combined.shape[1] < 2:
        return 1.0
    s = combined.iloc[:, 0]
    m = combined.iloc[:, 1]
    cov  = np.cov(s, m)
    if cov[1, 1] == 0:
        return 1.0
    return float(cov[0, 1] / cov[1, 1])


def compute_sharpe(returns: pd.Series, risk_free_rate: float = 0.065) -> float:
    """
    Sharpe Ratio: (annualised return - risk-free rate) / annualised volatility.
    Indian 10-year G-Sec ≈ 6.5% as risk-free rate.
    """
    ann_return = float(returns.mean() * 252)
    ann_vol    = compute_volatility(returns)
    if ann_vol == 0:
        return 0.0
    return round((ann_return - risk_free_rate) / ann_vol, 2)


def compute_sortino(returns: pd.Series, risk_free_rate: float = 0.065) -> float:
    """
    Sortino Ratio: like Sharpe but only penalises downside volatility.
    """
    ann_return   = float(returns.mean() * 252)
    downside     = returns[returns < 0]
    downside_vol = float(downside.std() * np.sqrt(252)) if len(downside) > 1 else 1e-9
    return round((ann_return - risk_free_rate) / downside_vol, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Risk Score
# ─────────────────────────────────────────────────────────────────────────────

def _score_volatility(vol: float) -> tuple[int, str, str]:
    """Score volatility (annualised). Returns (penalty, level, comment)."""
    pct = vol * 100
    if pct < 15:
        return 0,  "Low",    f"Annualised volatility {pct:.1f}% — stable"
    elif pct < 25:
        return 2,  "Medium", f"Annualised volatility {pct:.1f}% — moderate"
    elif pct < 40:
        return 4,  "High",   f"Annualised volatility {pct:.1f}% — volatile"
    else:
        return 6,  "Very High", f"Annualised volatility {pct:.1f}% — very volatile"


def _score_drawdown(dd: float) -> tuple[int, str]:
    """Max drawdown penalty."""
    pct = abs(dd) * 100
    if pct < 15:
        return 0, f"Max drawdown {pct:.1f}% — low"
    elif pct < 30:
        return 2, f"Max drawdown {pct:.1f}% — moderate"
    elif pct < 50:
        return 3, f"Max drawdown {pct:.1f}% — significant"
    else:
        return 4, f"Max drawdown {pct:.1f}% — severe (>50%)"


def _score_beta(beta: float) -> tuple[int, str]:
    """Beta penalty vs market."""
    if beta < 0.5:
        return 0, f"Beta {beta:.2f} — very defensive"
    elif beta < 0.8:
        return 1, f"Beta {beta:.2f} — defensive"
    elif beta <= 1.2:
        return 2, f"Beta {beta:.2f} — market-like"
    elif beta <= 1.8:
        return 3, f"Beta {beta:.2f} — aggressive (moves more than market)"
    else:
        return 4, f"Beta {beta:.2f} — very aggressive (highly amplified)"


# ─────────────────────────────────────────────────────────────────────────────
# Main Public Function
# ─────────────────────────────────────────────────────────────────────────────

def assess_risk(
    history: pd.DataFrame,
    fundamentals: Optional[dict] = None,
    market_history: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Full risk assessment for a stock.

    Args:
        history:        OHLCV DataFrame with a 'Close' column.
        fundamentals:   Optional fundamentals dict (for beta from yfinance).
        market_history: Optional NIFTY 50 OHLCV for beta calculation.

    Returns:
        {
          "risk_score":    int  (0-10, lower = safer)
          "risk_level":    str  ("Low" | "Medium" | "High" | "Very High")
          "risk_emoji":    str
          "volatility":    float (annualised)
          "max_drawdown":  float (fraction)
          "var_95":        float (1-day, fraction)
          "cvar_95":       float
          "beta":          float
          "sharpe":        float
          "sortino":       float
          "factors":       list of str (risk commentary)
          "metrics":       dict (formatted for display)
        }
    """
    if history.empty or len(history) < 30:
        return _empty_risk()

    prices  = history["Close"].dropna()
    returns = prices.pct_change().dropna()

    # ── Compute metrics ───────────────────────────────────────────────────────
    vol     = compute_volatility(returns)
    dd      = compute_max_drawdown(prices)
    var95   = compute_var(returns, 0.95)
    cvar95  = compute_cvar(returns, 0.95)
    sharpe  = compute_sharpe(returns)
    sortino = compute_sortino(returns)

    # Beta: prefer yfinance value, else compute vs market if available
    beta = 1.0
    if fundamentals and fundamentals.get("beta"):
        beta = float(fundamentals["beta"])
    elif market_history is not None and not market_history.empty:
        mkt_returns = market_history["Close"].pct_change().dropna()
        beta = compute_beta(returns, mkt_returns)

    # ── Risk scoring (penalty points — higher = riskier) ─────────────────────
    factors    = []
    penalty    = 0

    vol_pen, vol_level, vol_comment = _score_volatility(vol)
    penalty += vol_pen
    factors.append(vol_comment)

    dd_pen, dd_comment = _score_drawdown(dd)
    penalty += dd_pen
    factors.append(dd_comment)

    beta_pen, beta_comment = _score_beta(beta)
    penalty += beta_pen
    factors.append(beta_comment)

    # VaR risk
    var_pct = abs(var95) * 100
    if var_pct > 5:
        factors.append(f"95% VaR {var_pct:.1f}% — can lose more than 5% in a bad day")
        penalty += 2
    else:
        factors.append(f"95% VaR {var_pct:.1f}% — single-day loss risk manageable")

    # Sharpe quality
    if sharpe >= 1.5:
        factors.append(f"Sharpe ratio {sharpe:.2f} — excellent risk-adjusted return")
    elif sharpe >= 0.5:
        factors.append(f"Sharpe ratio {sharpe:.2f} — decent risk-adjusted return")
    else:
        factors.append(f"Sharpe ratio {sharpe:.2f} — poor risk-adjusted return")
        penalty += 1

    # ── Risk score (0-10, capped) ─────────────────────────────────────────────
    risk_score = min(10, penalty)

    if risk_score <= 2:
        risk_level, risk_emoji = "Low",       "🟢"
    elif risk_score <= 4:
        risk_level, risk_emoji = "Medium",    "🟡"
    elif risk_score <= 6:
        risk_level, risk_emoji = "High",      "🟠"
    else:
        risk_level, risk_emoji = "Very High", "🔴"

    # ── Formatted display metrics ─────────────────────────────────────────────
    metrics = {
        "Volatility (Annual)": f"{vol*100:.1f}%",
        "Max Drawdown":        f"{dd*100:.1f}%",
        "VaR (95%, 1-day)":   f"{var95*100:.2f}%",
        "CVaR (95%, 1-day)":  f"{cvar95*100:.2f}%",
        "Beta":                f"{beta:.2f}",
        "Sharpe Ratio":        f"{sharpe:.2f}",
        "Sortino Ratio":       f"{sortino:.2f}",
        "Risk Score":          f"{risk_score}/10",
        "Risk Level":          risk_level,
    }

    return {
        "risk_score":   risk_score,
        "risk_level":   risk_level,
        "risk_emoji":   risk_emoji,
        "volatility":   round(vol, 4),
        "max_drawdown": round(dd, 4),
        "var_95":       round(var95, 4),
        "cvar_95":      round(cvar95, 4),
        "beta":         round(beta, 2),
        "sharpe":       round(sharpe, 2),
        "sortino":      round(sortino, 2),
        "factors":      factors,
        "metrics":      metrics,
    }


def _empty_risk() -> dict:
    return {
        "risk_score": 5, "risk_level": "Unknown", "risk_emoji": "⚪",
        "volatility": 0, "max_drawdown": 0, "var_95": 0, "cvar_95": 0,
        "beta": 1, "sharpe": 0, "sortino": 0,
        "factors": ["Insufficient data for risk assessment"],
        "metrics": {},
    }
