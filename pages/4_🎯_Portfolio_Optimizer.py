import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Portfolio Optimizer — NiveshAI",
    page_icon="🎯",
    layout="wide",
)

# ──────────────────────────────────────────────
# CUSTOM CSS — Premium Dark Glassmorphism
# ──────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background: #0E1117;
    }

    /* ── Sidebar ── */
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0E1117 0%, #1A1F2E 100%);
        border-right: 1px solid rgba(108, 99, 255, 0.12);
    }
    div[data-testid="stSidebar"] .stMarkdown h3 {
        color: #6C63FF !important;
    }

    /* ── Metric cards ── */
    .stMetric {
        background: linear-gradient(135deg, rgba(108, 99, 255, 0.1), rgba(0, 212, 170, 0.05));
        border: 1px solid rgba(108, 99, 255, 0.2);
        border-radius: 12px;
        padding: 16px;
    }
    div[data-testid="stMetricValue"] {
        font-weight: 700;
    }

    /* ── Glass card ── */
    .glass-card {
        background: rgba(26, 31, 46, 0.8);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(108, 99, 255, 0.15);
        border-radius: 16px;
        padding: 24px;
        margin: 8px 0;
    }
    .glass-card-green {
        background: rgba(0, 212, 170, 0.06);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 212, 170, 0.25);
        border-radius: 16px;
        padding: 24px;
        margin: 8px 0;
    }
    .glass-card-red {
        background: rgba(255, 107, 107, 0.06);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 107, 107, 0.25);
        border-radius: 16px;
        padding: 24px;
        margin: 8px 0;
    }
    .glass-card-purple {
        background: rgba(108, 99, 255, 0.08);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(108, 99, 255, 0.3);
        border-radius: 16px;
        padding: 24px;
        margin: 8px 0;
    }

    /* ── Gradient text ── */
    .gradient-text {
        background: linear-gradient(135deg, #6C63FF, #00D4AA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    .gradient-text-gold {
        background: linear-gradient(135deg, #FFB347, #FFE66D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(14, 17, 23, 0.5);
        border-radius: 12px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(108, 99, 255, 0.08);
        border-radius: 8px;
        color: #A0AEC0;
        font-weight: 500;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(108, 99, 255, 0.3) !important;
        color: #FFFFFF !important;
        font-weight: 600;
    }

    /* ── Tables ── */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #6C63FF, #5A54D4) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        letter-spacing: 0.3px;
        transition: all 0.3s ease !important;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #7B73FF, #6C63FF) !important;
        box-shadow: 0 4px 20px rgba(108, 99, 255, 0.4) !important;
        transform: translateY(-1px);
    }

    /* ── Misc ── */
    .subtitle {
        color: #A0AEC0;
        font-size: 1.1rem;
        margin-top: -8px;
        margin-bottom: 24px;
    }
    .section-label {
        color: #6C63FF;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .big-number {
        font-size: 2.4rem;
        font-weight: 800;
        line-height: 1.1;
    }
    .var-card {
        text-align: center;
        padding: 32px;
    }
    .score-badge {
        display: inline-block;
        background: linear-gradient(135deg, #6C63FF, #00D4AA);
        border-radius: 30px;
        padding: 6px 20px;
        color: white;
        font-weight: 700;
        font-size: 1.3rem;
    }

    /* ── Radio horizontal ── */
    div[data-testid="stRadio"] > div {
        flex-direction: column;
        gap: 4px;
    }

    /* ── Hide default decoration ── */
    #MainMenu, footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# REPRODUCIBLE MOCK DATA
# ──────────────────────────────────────────────
np.random.seed(42)

STOCK_UNIVERSE = [
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
    "BHARTIARTL", "SBIN", "ITC", "KOTAKBANK", "LT",
    "HINDUNILVR", "AXISBANK", "BAJFINANCE", "MARUTI", "SUNPHARMA",
    "TITAN", "WIPRO", "ADANIENT", "NTPC", "POWERGRID",
]

DEFAULT_STOCKS = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ITC"]

STOCK_META = {
    "RELIANCE":   {"sector": "Energy",         "ret": 0.195, "vol": 0.22, "beta": 1.12},
    "TCS":        {"sector": "IT",             "ret": 0.165, "vol": 0.17, "beta": 0.78},
    "INFY":       {"sector": "IT",             "ret": 0.175, "vol": 0.19, "beta": 0.82},
    "HDFCBANK":   {"sector": "Banking",        "ret": 0.155, "vol": 0.16, "beta": 0.90},
    "ICICIBANK":  {"sector": "Banking",        "ret": 0.180, "vol": 0.20, "beta": 1.05},
    "BHARTIARTL": {"sector": "Telecom",        "ret": 0.210, "vol": 0.24, "beta": 0.95},
    "SBIN":       {"sector": "Banking",        "ret": 0.200, "vol": 0.26, "beta": 1.18},
    "ITC":        {"sector": "FMCG",           "ret": 0.125, "vol": 0.14, "beta": 0.65},
    "KOTAKBANK":  {"sector": "Banking",        "ret": 0.145, "vol": 0.18, "beta": 0.88},
    "LT":         {"sector": "Infrastructure", "ret": 0.185, "vol": 0.21, "beta": 1.10},
    "HINDUNILVR": {"sector": "FMCG",           "ret": 0.120, "vol": 0.13, "beta": 0.55},
    "AXISBANK":   {"sector": "Banking",        "ret": 0.170, "vol": 0.22, "beta": 1.08},
    "BAJFINANCE": {"sector": "NBFC",           "ret": 0.225, "vol": 0.28, "beta": 1.35},
    "MARUTI":     {"sector": "Auto",           "ret": 0.160, "vol": 0.20, "beta": 0.92},
    "SUNPHARMA":  {"sector": "Pharma",         "ret": 0.150, "vol": 0.19, "beta": 0.70},
    "TITAN":      {"sector": "Consumer",       "ret": 0.205, "vol": 0.23, "beta": 1.02},
    "WIPRO":      {"sector": "IT",             "ret": 0.135, "vol": 0.20, "beta": 0.75},
    "ADANIENT":   {"sector": "Conglomerate",   "ret": 0.280, "vol": 0.38, "beta": 1.55},
    "NTPC":       {"sector": "Power",          "ret": 0.140, "vol": 0.16, "beta": 0.72},
    "POWERGRID":  {"sector": "Power",          "ret": 0.130, "vol": 0.14, "beta": 0.60},
}

# ──────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────

def _build_correlation_matrix(stocks):
    """Generate a reproducible, realistic correlation matrix."""
    rng = np.random.RandomState(42)
    n = len(stocks)
    # Start with identity and fill
    raw = rng.uniform(0.25, 0.85, size=(n, n))
    corr = (raw + raw.T) / 2
    np.fill_diagonal(corr, 1.0)
    # Boost intra-sector correlations
    for i in range(n):
        for j in range(n):
            if i != j and STOCK_META[stocks[i]]["sector"] == STOCK_META[stocks[j]]["sector"]:
                corr[i, j] = min(corr[i, j] + 0.15, 0.95)
    return pd.DataFrame(corr, index=stocks, columns=stocks).round(2)


def _generate_efficient_frontier(stocks, risk_profile):
    """Return mock efficient frontier points, individual stocks, optimal & current portfolios."""
    rng = np.random.RandomState(42)
    # Efficient frontier curve
    vols = np.linspace(0.08, 0.35, 80)
    # Parabolic-ish upper boundary
    rets = -1.8 * (vols - 0.22) ** 2 + 0.23 + rng.normal(0, 0.003, len(vols))
    rets = np.clip(rets, 0.04, 0.30)
    # Sort by vol for a clean curve
    idx_sort = np.argsort(vols)
    vols, rets = vols[idx_sort], rets[idx_sort]
    # Keep only the upper envelope
    max_ret_so_far = np.maximum.accumulate(rets)
    mask = rets >= max_ret_so_far - 0.005
    ef_vol, ef_ret = vols[mask], rets[mask]

    # Individual stocks
    stock_vols = [STOCK_META[s]["vol"] for s in stocks]
    stock_rets = [STOCK_META[s]["ret"] for s in stocks]

    # Optimal portfolio (risk-adjusted placement)
    risk_scale = {"Conservative": 0.70, "Moderate": 1.0, "Aggressive": 1.3}
    scale = risk_scale.get(risk_profile, 1.0)
    opt_vol = 0.142 * scale
    opt_ret = 0.185 * scale
    # Current portfolio (slightly worse)
    cur_vol = opt_vol + 0.025
    cur_ret = opt_ret - 0.015

    return ef_vol, ef_ret, stock_vols, stock_rets, (opt_vol, opt_ret), (cur_vol, cur_ret)


def _mock_allocations(stocks, risk_profile):
    """Generate current & optimised weight allocations."""
    rng = np.random.RandomState(42)
    n = len(stocks)
    current_raw = rng.dirichlet(np.ones(n))
    # Optimised depends on risk profile
    alpha_map = {"Conservative": 0.5, "Moderate": 1.0, "Aggressive": 2.0}
    alpha = alpha_map.get(risk_profile, 1.0)
    rets = np.array([STOCK_META[s]["ret"] for s in stocks])
    vols = np.array([STOCK_META[s]["vol"] for s in stocks])
    # Higher alpha → favour higher return stocks
    scores = (rets ** alpha) / (vols ** (1 / alpha))
    opt_raw = scores / scores.sum()
    current_w = (current_raw * 100).round(1)
    opt_w = (opt_raw * 100).round(1)
    # Normalise to exactly 100
    current_w[-1] += 100 - current_w.sum()
    opt_w[-1] += 100 - opt_w.sum()
    return current_w, opt_w


def _mock_drawdown_series():
    """Generate 252 trading-day mock drawdown series."""
    rng = np.random.RandomState(42)
    daily = rng.normal(0.0005, 0.012, 252)
    prices = 100 * np.exp(np.cumsum(daily))
    running_max = np.maximum.accumulate(prices)
    drawdown = (prices - running_max) / running_max * 100  # %
    dates = pd.bdate_range(end=pd.Timestamp("2026-07-08"), periods=252)
    return dates, drawdown


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; margin-bottom:12px;">
        <span style="font-size:2rem;">🎯</span><br>
        <span class="gradient-text" style="font-size:1.3rem;">NiveshAI</span><br>
        <span style="color:#A0AEC0; font-size:0.78rem; letter-spacing:1px;">PORTFOLIO OPTIMIZER</span>
    </div>
    <hr style="border-color:rgba(108,99,255,0.15); margin:0 0 16px 0;">
    """, unsafe_allow_html=True)

    st.markdown("### 📊 Select Stocks")
    selected_stocks = st.multiselect(
        "Choose stocks from NSE",
        options=STOCK_UNIVERSE,
        default=DEFAULT_STOCKS,
        help="Select 2–20 stocks to include in the optimisation.",
    )
    if len(selected_stocks) < 2:
        st.warning("⚠️ Select at least 2 stocks.")
        st.stop()

    st.markdown("---")
    st.markdown("### 💰 Investment Amount")
    investment = st.number_input(
        "Amount (₹)",
        min_value=100_000,
        max_value=100_000_000,
        value=10_00_000,
        step=100_000,
        format="%d",
    )

    st.markdown("---")
    st.markdown("### ⚖️ Risk Tolerance")
    risk_profile = st.radio(
        "Choose your risk appetite",
        options=["Conservative", "Moderate", "Aggressive"],
        index=1,
        help="Controls how the optimiser balances return vs risk.",
    )

    st.markdown("---")
    optimise_clicked = st.button("✨  Optimize Portfolio", use_container_width=True)

    st.markdown("""
    <div style="margin-top:28px; padding:14px; border-radius:10px;
                background:rgba(108,99,255,0.06); border:1px solid rgba(108,99,255,0.12);
                font-size:0.75rem; color:#A0AEC0; text-align:center;">
        Powered by <b style="color:#6C63FF;">Modern Portfolio Theory</b><br>
        Markowitz Mean-Variance Optimisation
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────
st.markdown('<h1><span class="gradient-text">🎯 Portfolio Optimizer</span></h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Optimize your portfolio using Modern Portfolio Theory (Markowitz) — powered by AI-driven analytics for the Indian market.</p>', unsafe_allow_html=True)

# ──────────────────────────────────────────────
# COMPUTE MOCK DATA
# ──────────────────────────────────────────────
current_weights, opt_weights = _mock_allocations(selected_stocks, risk_profile)
ef_vol, ef_ret, s_vol, s_ret, opt_pt, cur_pt = _generate_efficient_frontier(selected_stocks, risk_profile)
corr_matrix = _build_correlation_matrix(selected_stocks)
dd_dates, dd_values = _mock_drawdown_series()

# Risk-adjusted metrics
risk_scale = {"Conservative": 0.85, "Moderate": 1.0, "Aggressive": 1.18}
rs = risk_scale[risk_profile]
exp_return = round(18.5 * rs, 1)
port_risk = round(14.2 * rs, 1)
sharpe = round(1.32 * (rs ** 0.5), 2)
var_daily = round(52340 * rs * (investment / 1_000_000), 0)
max_dd = round(-12.3 * rs, 1)
beta_port = round(1.05 * rs, 2)

# ──────────────────────────────────────────────
# TOP METRICS BAR
# ──────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric("💰 Investment", f"₹{investment:,.0f}")
m2.metric("📈 Expected Return", f"{exp_return}%", delta=f"+{exp_return}% p.a.")
m3.metric("📊 Volatility (Risk)", f"{port_risk}%", delta=f"{port_risk}%", delta_color="inverse")
m4.metric("⚡ Sharpe Ratio", f"{sharpe}", delta="Good" if sharpe > 1 else "Fair")

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────
tab_alloc, tab_ef, tab_risk, tab_corr = st.tabs([
    "📊 Allocation", "📈 Efficient Frontier", "🛡️ Risk Analysis", "🔗 Correlation"
])

# ═══════════════════════════════════════════════
# TAB 1 — ALLOCATION
# ═══════════════════════════════════════════════
with tab_alloc:
    st.markdown('<p class="section-label">Current vs Optimized Allocation</p>', unsafe_allow_html=True)

    col_cur, col_opt = st.columns(2)

    donut_colors = ["#6C63FF", "#00D4AA", "#FFB347", "#FF6B6B", "#4ECDC4",
                    "#FFE66D", "#A78BFA", "#F472B6", "#34D399", "#60A5FA",
                    "#FBBF24", "#F87171", "#818CF8", "#2DD4BF", "#FB923C",
                    "#C084FC", "#38BDF8", "#E879F9", "#4ADE80", "#FDA4AF"]

    with col_cur:
        fig_cur = go.Figure(go.Pie(
            labels=selected_stocks,
            values=current_weights,
            hole=0.55,
            marker=dict(colors=donut_colors[:len(selected_stocks)],
                        line=dict(color="#0E1117", width=2)),
            textinfo="label+percent",
            textfont=dict(size=12, color="white"),
            hovertemplate="<b>%{label}</b><br>Weight: %{value:.1f}%<br>Amount: ₹%{customdata:,.0f}<extra></extra>",
            customdata=[w / 100 * investment for w in current_weights],
        ))
        fig_cur.update_layout(
            title=dict(text="Current Portfolio", font=dict(size=16, color="#A0AEC0"), x=0.5),
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=380,
            margin=dict(t=50, b=20, l=20, r=20),
            annotations=[dict(text=f"₹{investment / 100_000:.1f}L", x=0.5, y=0.5,
                              font=dict(size=18, color="#6C63FF", family="Inter"),
                              showarrow=False)],
        )
        st.plotly_chart(fig_cur, use_container_width=True)

    with col_opt:
        fig_opt = go.Figure(go.Pie(
            labels=selected_stocks,
            values=opt_weights,
            hole=0.55,
            marker=dict(colors=donut_colors[:len(selected_stocks)],
                        line=dict(color="#0E1117", width=2)),
            textinfo="label+percent",
            textfont=dict(size=12, color="white"),
            hovertemplate="<b>%{label}</b><br>Weight: %{value:.1f}%<br>Amount: ₹%{customdata:,.0f}<extra></extra>",
            customdata=[w / 100 * investment for w in opt_weights],
        ))
        fig_opt.update_layout(
            title=dict(text="✨ Optimized Portfolio", font=dict(size=16, color="#00D4AA"), x=0.5),
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=380,
            margin=dict(t=50, b=20, l=20, r=20),
            annotations=[dict(text=f"₹{investment / 100_000:.1f}L", x=0.5, y=0.5,
                              font=dict(size=18, color="#00D4AA", family="Inter"),
                              showarrow=False)],
        )
        st.plotly_chart(fig_opt, use_container_width=True)

    # ── Allocation table ──
    st.markdown('<p class="section-label" style="margin-top:16px;">Detailed Breakdown</p>', unsafe_allow_html=True)
    tbl = pd.DataFrame({
        "Stock": selected_stocks,
        "Sector": [STOCK_META[s]["sector"] for s in selected_stocks],
        "Current Wt %": current_weights,
        "Optimized Wt %": opt_weights,
        "Change %": (opt_weights - current_weights).round(1),
        "Exp. Return %": [round(STOCK_META[s]["ret"] * 100, 1) for s in selected_stocks],
        "Alloc Amount (₹)": [f"₹{w / 100 * investment:,.0f}" for w in opt_weights],
    })
    st.dataframe(
        tbl.style.format({"Current Wt %": "{:.1f}", "Optimized Wt %": "{:.1f}", "Change %": "{:+.1f}"})
            .applymap(lambda v: "color: #00D4AA" if isinstance(v, (int, float)) and v > 0 else
                      ("color: #FF6B6B" if isinstance(v, (int, float)) and v < 0 else ""),
                      subset=["Change %"]),
        use_container_width=True,
        hide_index=True,
        height=min(38 * (len(selected_stocks) + 1), 420),
    )

    # ── Suggested Actions ──
    changes = list(zip(selected_stocks, (opt_weights - current_weights).round(1)))
    increases = sorted([(s, c) for s, c in changes if c > 0], key=lambda x: -x[1])
    decreases = sorted([(s, c) for s, c in changes if c < 0], key=lambda x: x[1])

    suggestions = []
    for s, c in increases[:3]:
        suggestions.append(f"<span style='color:#00D4AA'>▲ Increase <b>{s}</b> by {c:+.1f}%</span>")
    for s, c in decreases[:3]:
        suggestions.append(f"<span style='color:#FF6B6B'>▼ Reduce <b>{s}</b> by {abs(c):.1f}%</span>")

    st.markdown(f"""
    <div class="glass-card-purple">
        <p class="section-label">💡 Suggested Rebalancing Actions</p>
        <div style="display:flex; flex-wrap:wrap; gap:12px 28px; font-size:0.95rem; margin-top:8px;">
            {''.join(f'<div>{s}</div>' for s in suggestions)}
        </div>
        <p style="color:#A0AEC0; font-size:0.8rem; margin-top:14px;">
            Rebalancing towards the optimised weights can improve your risk-adjusted returns by an estimated
            <b style="color:#FFB347;">{abs(sharpe - 1.05):.2f}</b> Sharpe ratio points.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# TAB 2 — EFFICIENT FRONTIER
# ═══════════════════════════════════════════════
with tab_ef:
    st.markdown('<p class="section-label">Markowitz Efficient Frontier</p>', unsafe_allow_html=True)

    fig_ef = go.Figure()

    # Frontier curve
    fig_ef.add_trace(go.Scatter(
        x=ef_vol * 100, y=ef_ret * 100,
        mode="lines",
        line=dict(color="#6C63FF", width=3),
        name="Efficient Frontier",
        fill="tonexty",
        fillcolor="rgba(108,99,255,0.06)",
        hovertemplate="Vol: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>",
    ))

    # Shaded region below (baseline)
    fig_ef.add_trace(go.Scatter(
        x=ef_vol * 100, y=[4] * len(ef_vol),
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    ))

    # Individual stocks
    fig_ef.add_trace(go.Scatter(
        x=[v * 100 for v in s_vol],
        y=[r * 100 for r in s_ret],
        mode="markers+text",
        marker=dict(size=12, color="#4ECDC4", line=dict(color="white", width=1.5)),
        text=selected_stocks,
        textposition="top center",
        textfont=dict(size=10, color="#A0AEC0"),
        name="Individual Stocks",
        hovertemplate="<b>%{text}</b><br>Vol: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>",
    ))

    # Optimal portfolio
    fig_ef.add_trace(go.Scatter(
        x=[opt_pt[0] * 100], y=[opt_pt[1] * 100],
        mode="markers",
        marker=dict(size=20, color="#FFE66D", symbol="star",
                    line=dict(color="#FFB347", width=2)),
        name="⭐ Optimal Portfolio",
        hovertemplate="<b>Optimal Portfolio</b><br>Vol: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>",
    ))

    # Current portfolio
    fig_ef.add_trace(go.Scatter(
        x=[cur_pt[0] * 100], y=[cur_pt[1] * 100],
        mode="markers",
        marker=dict(size=16, color="#FF6B6B", symbol="x",
                    line=dict(color="#FF6B6B", width=2)),
        name="❌ Current Portfolio",
        hovertemplate="<b>Current Portfolio</b><br>Vol: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>",
    ))

    # Capital Market Line (approx)
    rf = 6.5  # risk-free rate %
    cml_x = np.linspace(0, 35, 50)
    cml_slope = (opt_pt[1] * 100 - rf) / (opt_pt[0] * 100)
    cml_y = rf + cml_slope * cml_x
    fig_ef.add_trace(go.Scatter(
        x=cml_x, y=cml_y,
        mode="lines",
        line=dict(color="#FFB347", width=1.5, dash="dash"),
        name="Capital Market Line",
        hoverinfo="skip",
    ))

    fig_ef.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(14,17,23,0.8)",
        height=560,
        xaxis=dict(title="Risk — Volatility (%)", gridcolor="rgba(108,99,255,0.08)",
                   range=[5, 40]),
        yaxis=dict(title="Expected Return (%)", gridcolor="rgba(108,99,255,0.08)",
                   range=[4, 30]),
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center",
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        margin=dict(t=30, b=80),
    )
    st.plotly_chart(fig_ef, use_container_width=True)

    # Info cards
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="glass-card" style="text-align:center;">
            <p class="section-label">⭐ Optimal Portfolio</p>
            <p style="color:#FFE66D; font-size:1.5rem; font-weight:700;">
                {opt_pt[1]*100:.1f}% Return</p>
            <p style="color:#A0AEC0; font-size:0.9rem;">{opt_pt[0]*100:.1f}% Volatility</p>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="glass-card" style="text-align:center;">
            <p class="section-label">❌ Current Portfolio</p>
            <p style="color:#FF6B6B; font-size:1.5rem; font-weight:700;">
                {cur_pt[1]*100:.1f}% Return</p>
            <p style="color:#A0AEC0; font-size:0.9rem;">{cur_pt[0]*100:.1f}% Volatility</p>
        </div>""", unsafe_allow_html=True)
    with c3:
        improvement = (opt_pt[1] - cur_pt[1]) * 100
        st.markdown(f"""
        <div class="glass-card-green" style="text-align:center;">
            <p class="section-label">🚀 Improvement</p>
            <p style="color:#00D4AA; font-size:1.5rem; font-weight:700;">
                +{improvement:.1f}% Return</p>
            <p style="color:#A0AEC0; font-size:0.9rem;">
                −{(cur_pt[0]-opt_pt[0])*100:.1f}% less risk</p>
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# TAB 3 — RISK ANALYSIS
# ═══════════════════════════════════════════════
with tab_risk:
    st.markdown('<p class="section-label">Portfolio Risk Metrics</p>', unsafe_allow_html=True)

    # ── Top risk metric cards ──
    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown(f"""
        <div class="glass-card-red var-card">
            <p class="section-label">Value at Risk (5 % Daily)</p>
            <p class="big-number" style="color:#FF6B6B;">₹{var_daily:,.0f}</p>
            <p style="color:#A0AEC0; font-size:0.82rem; margin-top:8px;">
                You could lose up to this amount on 1 in 20 trading days.</p>
        </div>""", unsafe_allow_html=True)
    with r2:
        st.markdown(f"""
        <div class="glass-card var-card">
            <p class="section-label">Maximum Drawdown</p>
            <p class="big-number" style="color:#FFB347;">{max_dd}%</p>
            <p style="color:#A0AEC0; font-size:0.82rem; margin-top:8px;">
                Worst peak-to-trough decline in the past year.</p>
        </div>""", unsafe_allow_html=True)
    with r3:
        beta_color = "#00D4AA" if beta_port <= 1.0 else "#FFB347" if beta_port <= 1.2 else "#FF6B6B"
        st.markdown(f"""
        <div class="glass-card var-card">
            <p class="section-label">Portfolio Beta</p>
            <p class="big-number" style="color:{beta_color};">{beta_port}</p>
            <p style="color:#A0AEC0; font-size:0.82rem; margin-top:8px;">
                Sensitivity to NIFTY 50 benchmark moves.</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    # ── Volatility comparison bar chart ──
    vol_left, vol_right = st.columns([3, 2])
    with vol_left:
        st.markdown('<p class="section-label">Volatility Comparison — Stock vs Portfolio</p>', unsafe_allow_html=True)
        stock_vols = [STOCK_META[s]["vol"] * 100 for s in selected_stocks]
        bar_colors = ["#FF6B6B" if v > port_risk else "#00D4AA" for v in stock_vols]

        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(
            x=selected_stocks, y=stock_vols,
            marker=dict(color=bar_colors, line=dict(width=0),
                        cornerradius=6),
            name="Stock Volatility",
            hovertemplate="<b>%{x}</b><br>Volatility: %{y:.1f}%<extra></extra>",
        ))
        # Portfolio line
        fig_vol.add_hline(
            y=port_risk, line_dash="dash", line_color="#6C63FF", line_width=2,
            annotation_text=f"Portfolio: {port_risk}%",
            annotation_font=dict(color="#6C63FF", size=12),
            annotation_position="top right",
        )
        fig_vol.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(14,17,23,0.8)",
            height=380,
            yaxis=dict(title="Annualised Volatility (%)", gridcolor="rgba(108,99,255,0.08)"),
            xaxis=dict(gridcolor="rgba(108,99,255,0.08)"),
            margin=dict(t=30, b=40),
            showlegend=False,
        )
        st.plotly_chart(fig_vol, use_container_width=True)

    with vol_right:
        st.markdown('<p class="section-label">Risk Breakdown</p>', unsafe_allow_html=True)
        for s in selected_stocks:
            vol_s = STOCK_META[s]["vol"] * 100
            beta_s = STOCK_META[s]["beta"]
            color = "#00D4AA" if vol_s <= port_risk else "#FF6B6B"
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; align-items:center;
                        padding:8px 12px; border-bottom:1px solid rgba(108,99,255,0.08);">
                <div>
                    <span style="font-weight:600; color:white;">{s}</span>
                    <span style="color:#A0AEC0; font-size:0.78rem; margin-left:6px;">β {beta_s:.2f}</span>
                </div>
                <span style="color:{color}; font-weight:600;">{vol_s:.1f}%</span>
            </div>""", unsafe_allow_html=True)

    # ── Drawdown chart ──
    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
    st.markdown('<p class="section-label">Historical Drawdown — Last 1 Year</p>', unsafe_allow_html=True)

    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=dd_dates, y=dd_values,
        fill="tozeroy",
        fillcolor="rgba(255,107,107,0.15)",
        line=dict(color="#FF6B6B", width=1.5),
        hovertemplate="Date: %{x|%b %d, %Y}<br>Drawdown: %{y:.2f}%<extra></extra>",
    ))
    fig_dd.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(14,17,23,0.8)",
        height=320,
        yaxis=dict(title="Drawdown (%)", gridcolor="rgba(108,99,255,0.08)"),
        xaxis=dict(gridcolor="rgba(108,99,255,0.08)"),
        margin=dict(t=10, b=40, l=60, r=20),
        showlegend=False,
    )
    st.plotly_chart(fig_dd, use_container_width=True)


# ═══════════════════════════════════════════════
# TAB 4 — CORRELATION
# ═══════════════════════════════════════════════
with tab_corr:
    st.markdown('<p class="section-label">Stock Correlation Matrix</p>', unsafe_allow_html=True)

    fig_corr = go.Figure(go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale=[
            [0.0, "#FF6B6B"],
            [0.5, "#1A1F2E"],
            [1.0, "#00D4AA"],
        ],
        zmin=0, zmax=1,
        text=corr_matrix.values.round(2),
        texttemplate="%{text}",
        textfont=dict(size=12, color="white"),
        hovertemplate="<b>%{x} × %{y}</b><br>Correlation: %{z:.2f}<extra></extra>",
        colorbar=dict(
            title="ρ",
            tickvals=[0, 0.25, 0.5, 0.75, 1.0],
            ticktext=["0.0", "0.25", "0.50", "0.75", "1.0"],
            len=0.6,
        ),
    ))
    fig_corr.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(14,17,23,0.8)",
        height=max(400, 70 * len(selected_stocks)),
        margin=dict(t=20, b=20, l=10, r=10),
        xaxis=dict(side="bottom"),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # ── Diversification Score ──
    avg_corr = corr_matrix.values[np.triu_indices(len(selected_stocks), k=1)].mean()
    div_score = round(10 * (1 - avg_corr), 1)
    div_score = max(1.0, min(10.0, div_score))

    bar_pct = div_score * 10
    bar_color = "#00D4AA" if div_score >= 7 else "#FFB347" if div_score >= 5 else "#FF6B6B"

    sc1, sc2 = st.columns([1, 2])
    with sc1:
        st.markdown(f"""
        <div class="glass-card" style="text-align:center;">
            <p class="section-label">Diversification Score</p>
            <div class="score-badge">{div_score} / 10</div>
            <div style="margin-top:16px; background:rgba(255,255,255,0.05); border-radius:8px;
                        height:10px; overflow:hidden;">
                <div style="width:{bar_pct}%; height:100%; background:{bar_color};
                            border-radius:8px; transition:width 0.5s;"></div>
            </div>
        </div>""", unsafe_allow_html=True)

    with sc2:
        if div_score >= 7:
            verdict = "🟢 <b>Well Diversified</b> — Your portfolio has low average correlation between holdings, which helps reduce unsystematic risk."
            tip = "You're in great shape! Consider adding a defensive sector (Pharma/FMCG) to further smooth volatility."
        elif div_score >= 5:
            verdict = "🟡 <b>Moderately Diversified</b> — There is moderate overlap between your holdings. Some sector concentration exists."
            tip = "Consider replacing one correlated stock with an uncorrelated sector — e.g., add Power/Pharma if you're heavy on Banking."
        else:
            verdict = "🔴 <b>Poorly Diversified</b> — High correlation among holdings means your portfolio moves in tandem, amplifying drawdowns."
            tip = "Urgently diversify! Spread allocation across IT, FMCG, Energy, and Banking to reduce portfolio-level risk."

        st.markdown(f"""
        <div class="glass-card-green">
            <p style="font-size:1.05rem; color:white; margin-bottom:10px;">{verdict}</p>
            <p style="color:#A0AEC0; font-size:0.88rem;">
                Average pairwise correlation: <b style="color:white;">{avg_corr:.2f}</b><br>
                Number of stock pairs: <b style="color:white;">{len(selected_stocks) * (len(selected_stocks)-1) // 2}</b>
            </p>
            <div style="margin-top:12px; padding:10px 14px; border-radius:8px;
                        background:rgba(108,99,255,0.08); border-left:3px solid #6C63FF;">
                <span style="color:#FFE66D; font-weight:600;">💡 Tip:</span>
                <span style="color:#A0AEC0; font-size:0.85rem;"> {tip}</span>
            </div>
        </div>""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; margin-top:48px; padding:18px;
            border-top:1px solid rgba(108,99,255,0.1);">
    <span class="gradient-text" style="font-size:0.9rem;">NiveshAI</span>
    <span style="color:#A0AEC0; font-size:0.78rem;"> · Portfolio Optimizer · Markowitz MPT Engine</span><br>
    <span style="color:rgba(160,174,192,0.5); font-size:0.7rem;">
        Disclaimer: This is for educational purposes only. Not financial advice. Past performance ≠ future results.
    </span>
</div>
""", unsafe_allow_html=True)
