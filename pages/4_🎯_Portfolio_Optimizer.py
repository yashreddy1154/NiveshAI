import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Internal modules
from portfolio.optimizer import optimize_portfolio
from data.stock_data import fetch_history
from data.company_db import get_display_options, parse_display_option, get_company

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Portfolio Optimizer — NiveshAI",
    page_icon="🎯",
    layout="wide",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
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

    /* ── Metric cards ── */
    .stMetric {
        background: linear-gradient(135deg, rgba(108, 99, 255, 0.1), rgba(0, 212, 170, 0.05));
        border: 1px solid rgba(108, 99, 255, 0.2);
        border-radius: 12px;
        padding: 16px;
    }

    /* ── Glass card ── */
    .glass-card {
        background: rgba(26, 31, 46, 0.8);
        backdrop-filter: blur(10px);
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
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(108, 99, 255, 0.3) !important;
        color: #FFFFFF !important;
        font-weight: 600;
    }

    .stButton > button {
        background: linear-gradient(135deg, #6C63FF, #5A54D4) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #7B73FF, #6C63FF) !important;
        box-shadow: 0 4px 20px rgba(108, 99, 255, 0.4) !important;
    }

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
</style>
""", unsafe_allow_html=True)

# Predefined Beta values for Indian Stocks fallback (used in Risk Analysis)
STOCK_BETAS = {
    "RELIANCE": 1.12, "TCS": 0.78, "INFY": 0.82, "HDFCBANK": 0.90, "ICICIBANK": 1.05,
    "BHARTIARTL": 0.95, "SBIN": 1.18, "ITC": 0.65, "KOTAKBANK": 0.88, "LT": 1.10,
    "HINDUNILVR": 0.55, "AXISBANK": 1.08, "BAJFINANCE": 1.35, "MARUTI": 0.92,
    "SUNPHARMA": 0.70, "TITAN": 1.02, "WIPRO": 0.75, "ADANIENT": 1.55, "NTPC": 0.72,
    "POWERGRID": 0.60,
}

# Initialize session state for persistent results
if "portfolio_result" not in st.session_state:
    st.session_state.portfolio_result = None
if "portfolio_histories" not in st.session_state:
    st.session_state.portfolio_histories = None
if "portfolio_symbols" not in st.session_state:
    st.session_state.portfolio_symbols = None
if "portfolio_warnings" not in st.session_state:
    st.session_state.portfolio_warnings = []

# ─── Sidebar Settings ───
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 18px 0 8px 0;">
        <span style="font-size:2rem;">🎯</span><br>
        <span class="gradient-text" style="font-size:1.3rem;">NiveshAI</span><br>
        <span style="color:#A0AEC0; font-size:0.78rem; letter-spacing:1px;">PORTFOLIO OPTIMIZER</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("### ⚙️ Optimization Settings")
    investment_amount = st.number_input(
        "Investment Amount (₹)",
        min_value=100000,
        max_value=10000000,
        value=1000000,
        step=100000,
        format="%d"
    )
    
    risk_tolerance = st.select_slider(
        "Risk Tolerance",
        options=["conservative", "moderate", "aggressive"],
        value="moderate"
    )
    
    period = st.selectbox(
        "Data Period",
        options=["1y", "2y", "3y"],
        index=1
    )
    
    st.markdown("---")
    btn_optimize = st.button("🚀 Optimize Portfolio")

# ─── Main Page Header & Title ───
st.markdown('<h1><span class="gradient-text">🎯 Portfolio Optimizer</span></h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Optimize your portfolio using Modern Portfolio Theory (Markowitz) — powered by yfinance historical returns.</p>', unsafe_allow_html=True)

# ─── Multi-stock selector in main page ───
try:
    all_options = get_display_options()
except Exception:
    all_options = ["RELIANCE — Reliance Industries Ltd", "TCS — Tata Consultancy Services Ltd", "INFY — Infosys Ltd", "HDFCBANK — HDFC Bank Ltd", "ITC — ITC Ltd"]

default_selections = []
for opt in all_options:
    sym = parse_display_option(opt)
    if sym in ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ITC"]:
        default_selections.append(opt)
if not default_selections:
    default_selections = all_options[:min(5, len(all_options))]

selected_options = st.multiselect(
    "Select stocks for portfolio (2-15 stocks)",
    all_options,
    default=default_selections,
    max_selections=15
)
symbols = [parse_display_option(o) for o in selected_options]

# ─── Check scipy and selection requirements ───
try:
    import scipy
    has_scipy = True
except ImportError:
    has_scipy = False

if not has_scipy:
    st.error("❌ Scipy is not installed. Please install it using: `pip install scipy`")
    st.stop()

if len(symbols) < 2:
    st.warning("⚠️ Please select at least 2 stocks.")
    st.stop()

# ─── Run Optimization ───
if btn_optimize or st.session_state.portfolio_result is None:
    with st.spinner("Downloading stock data and optimizing..."):
        @st.cache_data(ttl=3600)
        def run_optimization(syms_tuple, amount, risk, period_val):
            hist_data = {}
            warnings_list = []
            for sym in syms_tuple:
                try:
                    h = fetch_history(sym, period=period_val)
                    if not h.empty and "Close" in h.columns:
                        hist_data[sym] = h
                    else:
                        warnings_list.append(f"No data found for {sym}. Skipping from portfolio.")
                except Exception as e:
                    warnings_list.append(f"Failed to fetch data for {sym}: {e}. Skipping from portfolio.")
            
            res = optimize_portfolio(hist_data, amount, risk)
            return res, hist_data, warnings_list

        result, histories_dict, warnings_list = run_optimization(tuple(symbols), investment_amount, risk_tolerance, period)
        
        # Save to session state
        st.session_state.portfolio_result = result
        st.session_state.portfolio_histories = histories_dict
        st.session_state.portfolio_symbols = symbols
        st.session_state.portfolio_warnings = warnings_list

# Load from session state
result = st.session_state.portfolio_result
histories = st.session_state.portfolio_histories
symbols_used = st.session_state.portfolio_symbols
warnings_list = st.session_state.portfolio_warnings

# Display warnings if any
for w in warnings_list:
    st.warning(f"⚠️ {w}")

if result.get("error") is not None:
    st.error(f"Portfolio Optimization failed: {result['error']}")
    st.info("💡 Could not find optimal weights, using equal weights.")
    st.stop()

# Extract Results
symbols = result["symbols"]
weights = result["weights"]
allocation_inr = result["allocation_inr"]
expected_return = result["expected_return"]
expected_vol = result["expected_volatility"]
sharpe = result["sharpe_ratio"]
ef_df = result["efficient_frontier"]
corr_matrix = result["correlation_matrix"]
individual_returns = result["individual_returns"]

# Map investment variable for compatibility
investment = investment_amount

# Calculate risk metrics (VaR & Drawdown)
# Re-extract clean daily returns matching the portfolio
combined_prices = pd.DataFrame({sym: histories[sym]["Close"] for sym in symbols}).dropna()
daily_returns = combined_prices.pct_change().dropna()

# Weight array
w_array = np.array([weights[sym] for sym in symbols])
portfolio_daily_returns = daily_returns @ w_array

# Daily VaR (5%) in INR
var_pct_5 = np.percentile(portfolio_daily_returns, 5)
var_inr = -var_pct_5 * investment

# Cumulative returns & Max Drawdown
cum_returns = (1 + portfolio_daily_returns).cumprod()
running_max = cum_returns.cummax()
drawdowns = (cum_returns - running_max) / running_max
max_drawdown = drawdowns.min() * 100

# Weighted Beta
beta_port = sum(weights[sym] * STOCK_BETAS.get(sym, 1.00) for sym in symbols)

# Equal weights allocations for comparison
n_stocks = len(symbols)
current_weights = np.ones(n_stocks) / n_stocks

# ─── Top metrics Row ──────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric("💰 Total Allocation", f"₹{investment:,.0f}")
m2.metric("📈 Expected Return (p.a.)", f"{expected_return * 100:.2f}%")
m3.metric("📊 Volatility (Risk)", f"{expected_vol * 100:.2f}%", delta_color="inverse")
m4.metric("⚡ Sharpe Ratio", f"{sharpe:.2f}")

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# Tabs
tab_alloc, tab_ef, tab_risk, tab_corr = st.tabs([
    "📊 Allocation", "📈 Efficient Frontier", "🛡️ Risk Analysis", "🔗 Correlation"
])

# ═══════════════════════════════════════════════
# TAB 1 — Allocation
# ═══════════════════════════════════════════════
with tab_alloc:
    st.markdown('<p class="section-label">Equal Weight Baseline vs Optimized Allocation</p>', unsafe_allow_html=True)

    col_cur, col_opt = st.columns(2)
    donut_colors = ["#6C63FF", "#00D4AA", "#FFB347", "#FF6B6B", "#4ECDC4",
                    "#FFE66D", "#A78BFA", "#F472B6", "#34D399", "#60A5FA",
                    "#FBBF24", "#F87171", "#818CF8", "#2DD4BF", "#FB923C",
                    "#C084FC", "#38BDF8", "#E879F9", "#4ADE80", "#FDA4AF"]

    with col_cur:
        fig_cur = go.Figure(go.Pie(
            labels=symbols,
            values=current_weights * 100,
            hole=0.55,
            marker=dict(colors=donut_colors[:len(symbols)], line=dict(color="#0E1117", width=2)),
            textinfo="label+percent",
            textfont=dict(size=12, color="white"),
            hovertemplate="<b>%{label}</b><br>Weight: %{value:.1f}%<extra></extra>",
        ))
        fig_cur.update_layout(
            title=dict(text="Equal-Weighted Portfolio", font=dict(size=16, color="#A0AEC0"), x=0.5),
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=380,
            margin=dict(t=50, b=20, l=20, r=20),
            annotations=[dict(text="Baseline", x=0.5, y=0.5, font=dict(size=18, color="#6C63FF", family="Inter"), showarrow=False)],
        )
        st.plotly_chart(fig_cur, use_container_width=True)

    with col_opt:
        opt_values_pct = [weights[sym] * 100 for sym in symbols]
        fig_opt = go.Figure(go.Pie(
            labels=symbols,
            values=opt_values_pct,
            hole=0.55,
            marker=dict(colors=donut_colors[:len(symbols)], line=dict(color="#0E1117", width=2)),
            textinfo="label+percent",
            textfont=dict(size=12, color="white"),
            hovertemplate="<b>%{label}</b><br>Weight: %{value:.1f}%<br>Amount: ₹%{customdata:,.2f}<extra></extra>",
            customdata=[allocation_inr[sym] for sym in symbols],
        ))
        fig_opt.update_layout(
            title=dict(text="✨ Optimized Portfolio", font=dict(size=16, color="#00D4AA"), x=0.5),
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=380,
            margin=dict(t=50, b=20, l=20, r=20),
            annotations=[dict(text="MPT Opt", x=0.5, y=0.5, font=dict(size=18, color="#00D4AA", family="Inter"), showarrow=False)],
        )
        st.plotly_chart(fig_opt, use_container_width=True)

    # Allocation table
    st.markdown('<p class="section-label" style="margin-top:16px;">Detailed Allocation Breakdown</p>', unsafe_allow_html=True)
    
    table_rows = []
    for idx, sym in enumerate(symbols):
        c_info = get_company(sym) or {}
        sec = c_info.get("Sector", "N/A")
        cur_wt = 100.0 / n_stocks
        opt_wt = weights[sym] * 100
        change = opt_wt - cur_wt
        table_rows.append({
            "Stock": sym,
            "Sector": sec,
            "Baseline Wt %": cur_wt,
            "Optimized Wt %": opt_wt,
            "Change %": change,
            "Annualised Return": f"{individual_returns[sym]:.1f}%",
            "Allocated Amount": f"₹{allocation_inr[sym]:,.2f}"
        })
    
    tbl_df = pd.DataFrame(table_rows)
    st.dataframe(
        tbl_df.style.format({"Baseline Wt %": "{:.1f}", "Optimized Wt %": "{:.1f}", "Change %": "{:+.1f}"})
            .applymap(lambda v: "color: #00D4AA" if isinstance(v, (int, float)) and v > 0 else
                      ("color: #FF6B6B" if isinstance(v, (int, float)) and v < 0 else ""),
                      subset=["Change %"]),
        use_container_width=True,
        hide_index=True,
    )

    # Rebalancing insights
    suggestions = []
    for row in table_rows:
        s = row["Stock"]
        c = row["Change %"]
        if c > 0.5:
            suggestions.append(f"<span style='color:#00D4AA'>▲ Overweight <b>{s}</b> by {c:+.1f}%</span>")
        elif c < -0.5:
            suggestions.append(f"<span style='color:#FF6B6B'>▼ Underweight <b>{s}</b> by {abs(c):.1f}%</span>")

    st.markdown(f"""
    <div class="glass-card-purple">
        <p class="section-label">💡 Rebalancing Adjustments</p>
        <div style="display:flex; flex-wrap:wrap; gap:12px 28px; font-size:0.95rem; margin-top:8px;">
            {''.join(f'<div>{s}</div>' for s in suggestions) if suggestions else "No significant deviations from baseline weights found."}
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# TAB 2 — Efficient Frontier
# ═══════════════════════════════════════════════
with tab_ef:
    st.markdown('<p class="section-label">Markowitz Efficient Frontier Curve</p>', unsafe_allow_html=True)

    if ef_df.empty:
        st.info("Efficient Frontier curve details are empty for this selection.")
    else:
        fig_ef = go.Figure()

        # Frontier Curve
        fig_ef.add_trace(go.Scatter(
            x=ef_df["volatility"] * 100, y=ef_df["return"] * 100,
            mode="lines", line=dict(color="#6C63FF", width=3),
            name="Efficient Frontier",
            fill="tonexty", fillcolor="rgba(108,99,255,0.06)",
            hovertemplate="Vol: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>",
        ))

        # Baseline
        fig_ef.add_trace(go.Scatter(
            x=ef_df["volatility"] * 100, y=[4] * len(ef_df),
            mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"
        ))

        # Individual Stocks
        s_vols = [np.sqrt(daily_returns[sym].var() * 252) * 100 for sym in symbols]
        s_rets = [individual_returns[sym] for sym in symbols]
        fig_ef.add_trace(go.Scatter(
            x=s_vols, y=s_rets,
            mode="markers+text",
            marker=dict(size=12, color="#4ECDC4", line=dict(color="white", width=1.5)),
            text=symbols, textposition="top center",
            textfont=dict(size=10, color="#A0AEC0"),
            name="Individual Assets",
            hovertemplate="<b>%{text}</b><br>Vol: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>",
        ))

        # Optimal MPT Portfolio
        fig_ef.add_trace(go.Scatter(
            x=[expected_vol * 100], y=[expected_return * 100],
            mode="markers",
            marker=dict(size=20, color="#FFE66D", symbol="star", line=dict(color="#FFB347", width=2)),
            name="⭐ Optimal Portfolio",
            hovertemplate="<b>Optimal</b><br>Vol: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>",
        ))

        fig_ef.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(14,17,23,0.8)",
            height=520,
            xaxis=dict(title="Volatility - Risk (%)", gridcolor="rgba(108,99,255,0.08)"),
            yaxis=dict(title="Annualised Return (%)", gridcolor="rgba(108,99,255,0.08)"),
            legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center", bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=20, b=80),
        )
        st.plotly_chart(fig_ef, use_container_width=True)


# ═══════════════════════════════════════════════
# TAB 3 — Risk Analysis
# ═══════════════════════════════════════════════
with tab_risk:
    st.markdown('<p class="section-label">Risk Analytics Suite</p>', unsafe_allow_html=True)

    # Metric boxes
    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown(f"""
        <div class="glass-card-red var-card">
            <p class="section-label">Value at Risk (5 % Daily)</p>
            <p class="big-number" style="color:#FF6B6B;">₹{var_inr:,.2f}</p>
            <p style="color:#A0AEC0; font-size:0.82rem; margin-top:8px;">
                5% probability that the portfolio loses more than this in 1 day.</p>
        </div>""", unsafe_allow_html=True)
    with r2:
        st.markdown(f"""
        <div class="glass-card var-card">
            <p class="section-label">Maximum Drawdown</p>
            <p class="big-number" style="color:#FFB347;">{max_drawdown:.2f}%</p>
            <p style="color:#A0AEC0; font-size:0.82rem; margin-top:8px;">
                Maximum peak-to-trough drop based on historical simulation.</p>
        </div>""", unsafe_allow_html=True)
    with r3:
        beta_color = "#00D4AA" if beta_port <= 1.0 else "#FFB347" if beta_port <= 1.2 else "#FF6B6B"
        st.markdown(f"""
        <div class="glass-card var-card">
            <p class="section-label">Systematic Beta</p>
            <p class="big-number" style="color:{beta_color};">{beta_port:.2f}</p>
            <p style="color:#A0AEC0; font-size:0.82rem; margin-top:8px;">
                Weighted average of systemic beta values vs NIFTY 50.</p>
        </div>""", unsafe_allow_html=True)

    # Drawdown Chart
    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
    st.markdown('<p class="section-label">Drawdown Curve (Last 252 Trading Days)</p>', unsafe_allow_html=True)

    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=drawdowns.index, y=drawdowns.values * 100,
        fill="tozeroy", fillcolor="rgba(255,107,107,0.15)",
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
# TAB 4 — Correlation
# ═══════════════════════════════════════════════
with tab_corr:
    st.markdown('<p class="section-label">Pairwise Asset Correlations</p>', unsafe_allow_html=True)

    fig_corr = go.Figure(go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale=[[0.0, "#FF6B6B"], [0.5, "#1A1F2E"], [1.0, "#00D4AA"]],
        zmin=-0.2, zmax=1.0,
        text=corr_matrix.values.round(2),
        texttemplate="%{text}",
        textfont=dict(size=12, color="white"),
        hovertemplate="<b>%{x} × %{y}</b><br>Correlation: %{z:.2f}<extra></extra>",
    ))
    fig_corr.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(14,17,23,0.8)",
        height=max(400, 70 * len(symbols)),
        margin=dict(t=20, b=20, l=10, r=10),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # pairwise average correlation
    avg_corr = corr_matrix.values[np.triu_indices(len(symbols), k=1)].mean()
    div_score = round(10 * (1 - max(0, avg_corr)), 1)
    
    sc1, sc2 = st.columns([1, 2])
    with sc1:
        st.markdown(f"""
        <div class="glass-card" style="text-align:center;">
            <p class="section-label">Diversification Index</p>
            <div class="score-badge">{div_score} / 10</div>
        </div>""", unsafe_allow_html=True)
    with sc2:
        if div_score >= 7.0:
            verdict = "🟢 <b>Excellent Diversification</b> — Your selection features low correlation, meaning assets balance each other during corrections."
        elif div_score >= 4.5:
            verdict = "🟡 <b>Moderate Diversification</b> — holdings possess some overlapping cyclical factors. Consider introducing alternative sectors."
        else:
            verdict = "🔴 <b>High Concentration Risk</b> — Assets are highly correlated and will likely decline together during systematic pullbacks."
            
        st.markdown(f"""
        <div class="glass-card-green">
            <p style="font-size:1.02rem; color:white; margin:0 0 8px 0;">{verdict}</p>
            <p style="color:#A0AEC0; font-size:0.85rem; margin:0;">Pairwise Average Correlation coefficient: <b>{avg_corr:.2f}</b></p>
        </div>""", unsafe_allow_html=True)

# ─── Footer ──────────────────────────────────────────────────────────────────
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
