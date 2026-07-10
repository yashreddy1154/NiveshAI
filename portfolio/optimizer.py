"""
NiveshAI — Portfolio Optimizer
Modern Portfolio Theory (Markowitz) optimization using scipy.
Computes optimal weights, efficient frontier, and risk/return metrics.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Optional
import warnings

warnings.filterwarnings("ignore")

TRADING_DAYS = 252
RISK_FREE_RATE = 0.065   # Indian 10-year G-Sec ~6.5%


# ─────────────────────────────────────────────────────────────────────────────
# Return & Covariance helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_price_matrix(histories: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Combine per-stock Close price series into a single wide DataFrame.
    histories: {symbol: ohlcv_df, ...}
    """
    frames = {}
    for sym, df in histories.items():
        if df is not None and not df.empty and "Close" in df.columns:
            frames[sym] = df["Close"]

    if not frames:
        return pd.DataFrame()

    prices = pd.DataFrame(frames)
    prices = prices.dropna(how="all").fillna(method="ffill").dropna()
    return prices


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily percentage returns from price matrix."""
    return prices.pct_change().dropna()


def annualise(mean_daily: np.ndarray, cov_daily: np.ndarray):
    """Scale daily returns/covariance to annualised values."""
    return mean_daily * TRADING_DAYS, cov_daily * TRADING_DAYS


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio Statistics
# ─────────────────────────────────────────────────────────────────────────────

def portfolio_stats(weights: np.ndarray, ann_returns: np.ndarray, ann_cov: np.ndarray) -> dict:
    """Compute return, volatility, and Sharpe for a weight vector."""
    w   = np.array(weights)
    ret = float(np.dot(w, ann_returns))
    vol = float(np.sqrt(w @ ann_cov @ w))
    sharpe = (ret - RISK_FREE_RATE) / vol if vol > 0 else 0
    return {"return": round(ret, 4), "volatility": round(vol, 4), "sharpe": round(sharpe, 4)}


def neg_sharpe(weights, ann_returns, ann_cov):
    s = portfolio_stats(weights, ann_returns, ann_cov)
    return -s["sharpe"]


def portfolio_volatility(weights, ann_cov):
    w = np.array(weights)
    return float(np.sqrt(w @ ann_cov @ w))


# ─────────────────────────────────────────────────────────────────────────────
# Optimization
# ─────────────────────────────────────────────────────────────────────────────

def _constraints_and_bounds(n: int, min_weight: float = 0.02, max_weight: float = 0.40):
    """Standard long-only constraints: weights sum to 1, each in [min, max]."""
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(min_weight, max_weight)] * n
    return constraints, bounds


def _optimize(
    objective,
    n: int,
    ann_returns: np.ndarray,
    ann_cov: np.ndarray,
    extra_constraint=None,
    min_weight: float = 0.02,
    max_weight: float = 0.40,
) -> Optional[np.ndarray]:
    """Run scipy minimize with multiple random starts for robustness."""
    constraints, bounds = _constraints_and_bounds(n, min_weight, max_weight)
    if extra_constraint:
        constraints.append(extra_constraint)

    best_result = None
    best_val    = np.inf

    for _ in range(20):   # 20 random starts
        w0 = np.random.dirichlet(np.ones(n))   # random valid start
        res = minimize(
            objective,
            w0,
            args=(ann_returns, ann_cov) if "ann_returns" in objective.__code__.co_varnames else (ann_cov,),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-9},
        )
        if res.success and res.fun < best_val:
            best_val    = res.fun
            best_result = res.x

    return best_result


def optimize_max_sharpe(
    ann_returns: np.ndarray,
    ann_cov: np.ndarray,
    min_weight: float = 0.02,
    max_weight: float = 0.40,
) -> Optional[np.ndarray]:
    """Find weights that maximise the Sharpe ratio."""
    n = len(ann_returns)
    constraints, bounds = _constraints_and_bounds(n, min_weight, max_weight)
    best = None
    best_val = np.inf

    for _ in range(30):
        w0  = np.random.dirichlet(np.ones(n))
        res = minimize(
            neg_sharpe,
            w0,
            args=(ann_returns, ann_cov),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-10},
        )
        if res.success and res.fun < best_val:
            best_val = res.fun
            best = res.x

    return best


def optimize_min_volatility(
    ann_cov: np.ndarray,
    min_weight: float = 0.02,
    max_weight: float = 0.40,
) -> Optional[np.ndarray]:
    """Find weights that minimise portfolio volatility."""
    n = ann_cov.shape[0]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(min_weight, max_weight)] * n
    best = None
    best_val = np.inf

    for _ in range(20):
        w0  = np.random.dirichlet(np.ones(n))
        res = minimize(
            lambda w: portfolio_volatility(w, ann_cov),
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000},
        )
        if res.success and res.fun < best_val:
            best_val = res.fun
            best = res.x

    return best


def optimize_target_return(
    target_return: float,
    ann_returns: np.ndarray,
    ann_cov: np.ndarray,
    min_weight: float = 0.02,
    max_weight: float = 0.40,
) -> Optional[np.ndarray]:
    """Minimise volatility subject to achieving a target annualised return."""
    n = len(ann_returns)
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "eq", "fun": lambda w: np.dot(w, ann_returns) - target_return},
    ]
    bounds = [(min_weight, max_weight)] * n
    best = None
    best_val = np.inf

    for _ in range(20):
        w0  = np.random.dirichlet(np.ones(n))
        res = minimize(
            lambda w: portfolio_volatility(w, ann_cov),
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000},
        )
        if res.success and res.fun < best_val:
            best_val = res.fun
            best = res.x

    return best


# ─────────────────────────────────────────────────────────────────────────────
# Efficient Frontier
# ─────────────────────────────────────────────────────────────────────────────

def compute_efficient_frontier(
    ann_returns: np.ndarray,
    ann_cov: np.ndarray,
    n_points: int = 40,
    min_weight: float = 0.02,
    max_weight: float = 0.40,
) -> pd.DataFrame:
    """
    Compute the efficient frontier: n_points portfolios from min-vol to max-return.
    Returns DataFrame with columns: return, volatility, sharpe.
    """
    # Range of target returns
    min_ret = ann_returns.min()
    max_ret = ann_returns.max()
    targets = np.linspace(min_ret * 1.01, max_ret * 0.95, n_points)

    points = []
    for target in targets:
        w = optimize_target_return(target, ann_returns, ann_cov, min_weight, max_weight)
        if w is not None:
            s = portfolio_stats(w, ann_returns, ann_cov)
            points.append(s)

    return pd.DataFrame(points)


# ─────────────────────────────────────────────────────────────────────────────
# Risk-Tolerance Mapping
# ─────────────────────────────────────────────────────────────────────────────

RISK_TOLERANCE_PARAMS = {
    "conservative": {"min_weight": 0.05, "max_weight": 0.30, "strategy": "min_vol"},
    "moderate":     {"min_weight": 0.03, "max_weight": 0.35, "strategy": "max_sharpe"},
    "aggressive":   {"min_weight": 0.02, "max_weight": 0.45, "strategy": "max_sharpe"},
}


# ─────────────────────────────────────────────────────────────────────────────
# Main Public Function
# ─────────────────────────────────────────────────────────────────────────────

def optimize_portfolio(
    histories: dict[str, pd.DataFrame],
    investment_amount: float = 1_000_000,
    risk_tolerance: str = "moderate",
) -> dict:
    """
    Full portfolio optimization.

    Args:
        histories:          {symbol: ohlcv_df} — use fetch_history per symbol
        investment_amount:  Total amount in INR to invest
        risk_tolerance:     'conservative' | 'moderate' | 'aggressive'

    Returns:
        {
          "weights":              {symbol: float}
          "allocation_inr":       {symbol: float}
          "num_shares":           {symbol: int}    (approximate)
          "expected_return":      float (annualised)
          "expected_volatility":  float (annualised)
          "sharpe_ratio":         float
          "efficient_frontier":   pd.DataFrame
          "correlation_matrix":   pd.DataFrame
          "individual_returns":   {symbol: float}
          "error":                str or None
        }
    """
    try:
        # Build price matrix
        prices = build_price_matrix(histories)
        if prices.empty or prices.shape[1] < 2:
            return _error_result("Need at least 2 stocks with price data")

        symbols = list(prices.columns)
        n = len(symbols)

        # Daily returns + annualise
        daily_ret = compute_returns(prices)
        mean_d    = daily_ret.mean().values
        cov_d     = daily_ret.cov().values
        ann_ret, ann_cov = annualise(mean_d, cov_d)

        # Correlation matrix for display
        corr_matrix = daily_ret.corr()

        # Get optimization params
        params = RISK_TOLERANCE_PARAMS.get(risk_tolerance, RISK_TOLERANCE_PARAMS["moderate"])
        min_w  = params["min_weight"]
        max_w  = params["max_weight"]

        # Optimize
        if params["strategy"] == "min_vol":
            weights = optimize_min_volatility(ann_cov, min_w, max_w)
        else:
            weights = optimize_max_sharpe(ann_ret, ann_cov, min_w, max_w)

        if weights is None:
            # Fallback to equal weights
            weights = np.ones(n) / n

        # Clip and renormalize
        weights = np.clip(weights, 0, None)
        weights = weights / weights.sum()

        # Portfolio stats
        stats = portfolio_stats(weights, ann_ret, ann_cov)

        # Efficient frontier
        ef = compute_efficient_frontier(ann_ret, ann_cov, n_points=40, min_weight=min_w, max_weight=max_w)

        # Allocation in INR
        allocation = {sym: round(w * investment_amount, 2) for sym, w in zip(symbols, weights)}

        return {
            "weights":             {sym: round(float(w), 4) for sym, w in zip(symbols, weights)},
            "allocation_inr":      allocation,
            "expected_return":     stats["return"],
            "expected_volatility": stats["volatility"],
            "sharpe_ratio":        stats["sharpe"],
            "efficient_frontier":  ef,
            "correlation_matrix":  corr_matrix,
            "individual_returns":  {sym: round(float(r) * 100, 2) for sym, r in zip(symbols, ann_ret)},
            "symbols":             symbols,
            "risk_tolerance":      risk_tolerance,
            "investment_amount":   investment_amount,
            "error":               None,
        }

    except Exception as e:
        return _error_result(str(e))


def _error_result(msg: str) -> dict:
    return {
        "weights": {}, "allocation_inr": {},
        "expected_return": 0, "expected_volatility": 0, "sharpe_ratio": 0,
        "efficient_frontier": pd.DataFrame(), "correlation_matrix": pd.DataFrame(),
        "individual_returns": {}, "symbols": [], "error": msg,
    }
