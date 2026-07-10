"""
NiveshAI — Fundamental Analysis
Summarise, score, and rate company fundamentals fetched from yfinance.
"""

import numpy as np
from typing import Optional


# ── Sector-specific P/E benchmarks (approximate Indian market averages) ────────
SECTOR_PE_BENCHMARKS = {
    "Information Technology": 28,
    "Financial Services":     18,
    "Healthcare":             32,
    "Energy":                 12,
    "Consumer Goods":         45,
    "FMCG":                   50,
    "Automobile":             20,
    "Metals & Mining":        10,
    "Telecom":                30,
    "Infrastructure":         25,
    "Real Estate":            30,
    "Utilities":              14,
    "Capital Goods":          28,
    "Chemicals":              25,
    "Default":                25,
}

# ── Score Thresholds ──────────────────────────────────────────────────────────
RATING_MAP = [
    (80, "Strong Buy",  "🟢"),
    (60, "Buy",         "🟩"),
    (40, "Hold",        "🟡"),
    (20, "Underperform","🟠"),
    (0,  "Sell",        "🔴"),
]


def _safe(val, default=None):
    """Return val if it's a real number, else default."""
    if val is None:
        return default
    try:
        f = float(val)
        return default if (np.isnan(f) or np.isinf(f)) else f
    except (TypeError, ValueError):
        return default


def _score_pe(pe: Optional[float], sector: str) -> tuple[int, str]:
    """Score P/E ratio relative to sector benchmark."""
    benchmark = SECTOR_PE_BENCHMARKS.get(sector, SECTOR_PE_BENCHMARKS["Default"])
    if pe is None:
        return 5, "P/E unavailable"
    if pe <= 0:
        return 0, f"Negative P/E ({pe:.1f}) — losses"
    ratio = pe / benchmark
    if ratio < 0.7:
        return 20, f"P/E {pe:.1f} — trading at discount vs sector avg {benchmark}x"
    elif ratio < 1.0:
        return 15, f"P/E {pe:.1f} — slightly below sector avg {benchmark}x"
    elif ratio < 1.3:
        return 10, f"P/E {pe:.1f} — in line with sector avg {benchmark}x"
    elif ratio < 1.8:
        return 5,  f"P/E {pe:.1f} — premium to sector avg {benchmark}x"
    else:
        return 0,  f"P/E {pe:.1f} — significantly overvalued vs sector avg {benchmark}x"


def _score_roe(roe: Optional[float]) -> tuple[int, str]:
    """Return on Equity — measures profitability."""
    if roe is None:
        return 5, "ROE unavailable"
    pct = roe * 100
    if pct >= 20:
        return 20, f"ROE {pct:.1f}% — excellent (≥20%)"
    elif pct >= 15:
        return 15, f"ROE {pct:.1f}% — good (15-20%)"
    elif pct >= 10:
        return 10, f"ROE {pct:.1f}% — acceptable (10-15%)"
    elif pct >= 5:
        return 5,  f"ROE {pct:.1f}% — below average (5-10%)"
    else:
        return 0,  f"ROE {pct:.1f}% — poor (<5%)"


def _score_debt_equity(de: Optional[float]) -> tuple[int, str]:
    """Debt-to-Equity — lower is safer."""
    if de is None:
        return 5, "D/E unavailable"
    if de < 0.3:
        return 15, f"D/E {de:.2f} — very low debt"
    elif de < 0.8:
        return 12, f"D/E {de:.2f} — manageable debt"
    elif de < 1.5:
        return 8,  f"D/E {de:.2f} — moderate debt"
    elif de < 3.0:
        return 3,  f"D/E {de:.2f} — high debt"
    else:
        return 0,  f"D/E {de:.2f} — dangerously high debt"


def _score_profit_margin(margin: Optional[float]) -> tuple[int, str]:
    """Net profit margin — higher = more efficient."""
    if margin is None:
        return 5, "Profit margin unavailable"
    pct = margin * 100
    if pct >= 20:
        return 15, f"Net margin {pct:.1f}% — excellent"
    elif pct >= 12:
        return 12, f"Net margin {pct:.1f}% — good"
    elif pct >= 5:
        return 8,  f"Net margin {pct:.1f}% — acceptable"
    elif pct >= 0:
        return 3,  f"Net margin {pct:.1f}% — thin"
    else:
        return 0,  f"Net margin {pct:.1f}% — losses"


def _score_revenue_growth(growth: Optional[float]) -> tuple[int, str]:
    """Revenue growth YoY."""
    if growth is None:
        return 5, "Revenue growth unavailable"
    pct = growth * 100
    if pct >= 20:
        return 10, f"Revenue growth {pct:.1f}% YoY — high growth"
    elif pct >= 10:
        return 8,  f"Revenue growth {pct:.1f}% YoY — healthy"
    elif pct >= 0:
        return 5,  f"Revenue growth {pct:.1f}% YoY — stable"
    else:
        return 0,  f"Revenue growth {pct:.1f}% YoY — declining"


def _get_rating(score: int) -> tuple[str, str]:
    for threshold, label, emoji in RATING_MAP:
        if score >= threshold:
            return label, emoji
    return "Sell", "🔴"


# ─────────────────────────────────────────────────────────────────────────────
# Main Public Function
# ─────────────────────────────────────────────────────────────────────────────

def analyze_fundamentals(fundamentals: dict) -> dict:
    """
    Score and interpret a fundamentals dict (from stock_data.fetch_fundamentals).

    Returns:
        {
          "rating":         str  ("Strong Buy" | "Buy" | "Hold" | "Underperform" | "Sell")
          "rating_emoji":   str
          "score":          int  (0-100)
          "breakdown":      list of {"metric": str, "score": int, "comment": str}
          "strengths":      list[str]
          "weaknesses":     list[str]
          "summary":        str  (2-3 sentence text summary)
          "metrics":        dict (cleaned, formatted key metrics for display)
        }
    """
    sector = fundamentals.get("sector", "Default")
    breakdown  = []
    strengths  = []
    weaknesses = []
    total = 0

    def add(metric, score_fn_result):
        sc, comment = score_fn_result
        breakdown.append({"metric": metric, "score": sc, "comment": comment})
        if sc >= 12:
            strengths.append(comment)
        elif sc <= 3:
            weaknesses.append(comment)
        return sc

    total += add("P/E Ratio",        _score_pe(_safe(fundamentals.get("pe_ratio")), sector))
    total += add("Return on Equity",  _score_roe(_safe(fundamentals.get("roe"))))
    total += add("Debt-to-Equity",    _score_debt_equity(_safe(fundamentals.get("debt_to_equity"))))
    total += add("Profit Margin",     _score_profit_margin(_safe(fundamentals.get("profit_margin"))))
    total += add("Revenue Growth",    _score_revenue_growth(_safe(fundamentals.get("revenue_growth"))))

    # Normalize to 0-100
    max_possible = 20 + 20 + 15 + 15 + 10   # max from each scorer
    score = int(round(total / max_possible * 100))
    score = max(0, min(100, score))

    rating, emoji = _get_rating(score)

    # ── Format display metrics ────────────────────────────────────────────────
    def fmt_pct(v, mult=100):
        return f"{v * mult:.1f}%" if v is not None else "N/A"

    def fmt_cr(v):
        if v is None: return "N/A"
        if abs(v) >= 1e7:  return f"₹{v/1e7:,.1f} Cr"
        if abs(v) >= 1e5:  return f"₹{v/1e5:,.1f} L"
        return f"₹{v:,.0f}"

    metrics = {
        "Market Cap":        fmt_cr(_safe(fundamentals.get("market_cap"))),
        "P/E Ratio":         f"{_safe(fundamentals.get('pe_ratio'), 0):.1f}x" if _safe(fundamentals.get("pe_ratio")) else "N/A",
        "Forward P/E":       f"{_safe(fundamentals.get('forward_pe'), 0):.1f}x" if _safe(fundamentals.get("forward_pe")) else "N/A",
        "EPS":               f"₹{_safe(fundamentals.get('eps'), 0):.2f}" if _safe(fundamentals.get("eps")) else "N/A",
        "P/B Ratio":         f"{_safe(fundamentals.get('pb_ratio'), 0):.2f}x" if _safe(fundamentals.get("pb_ratio")) else "N/A",
        "ROE":               fmt_pct(_safe(fundamentals.get("roe"))),
        "ROA":               fmt_pct(_safe(fundamentals.get("roa"))),
        "Profit Margin":     fmt_pct(_safe(fundamentals.get("profit_margin"))),
        "Op. Margin":        fmt_pct(_safe(fundamentals.get("op_margin"))),
        "Revenue Growth":    fmt_pct(_safe(fundamentals.get("revenue_growth"))),
        "Earnings Growth":   fmt_pct(_safe(fundamentals.get("earnings_growth"))),
        "Debt/Equity":       f"{_safe(fundamentals.get('debt_to_equity'), 0):.2f}" if _safe(fundamentals.get("debt_to_equity")) else "N/A",
        "Current Ratio":     f"{_safe(fundamentals.get('current_ratio'), 0):.2f}" if _safe(fundamentals.get("current_ratio")) else "N/A",
        "Dividend Yield":    fmt_pct(_safe(fundamentals.get("dividend_yield"))),
        "52W High":          f"₹{_safe(fundamentals.get('week_52_high'), 0):,.2f}" if _safe(fundamentals.get("week_52_high")) else "N/A",
        "52W Low":           f"₹{_safe(fundamentals.get('week_52_low'), 0):,.2f}" if _safe(fundamentals.get("week_52_low")) else "N/A",
        "Beta":              f"{_safe(fundamentals.get('beta'), 1.0):.2f}" if _safe(fundamentals.get("beta")) else "N/A",
    }

    # ── Text summary ──────────────────────────────────────────────────────────
    name = fundamentals.get("company_name", "This company")
    top_str  = f"{strengths[0]}" if strengths else "no major strengths identified"
    weak_str = f"{weaknesses[0]}" if weaknesses else "no major concerns"

    summary = (
        f"{name} scores {score}/100 on our fundamental scorecard, earning a '{rating}' rating. "
        f"Key strength: {top_str}. "
        f"Main concern: {weak_str}."
    )

    return {
        "rating":        rating,
        "rating_emoji":  emoji,
        "score":         score,
        "breakdown":     breakdown,
        "strengths":     strengths,
        "weaknesses":    weaknesses,
        "summary":       summary,
        "metrics":       metrics,
    }
