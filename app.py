"""
NiveshAI — AI-Powered Investment Research for Indian Markets
Home Page (app.py)
All data is MOCK / PLACEHOLDER — no real API calls.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timezone, timedelta

# ──────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="NiveshAI — AI Investment Research",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS — Premium Dark Theme + Glassmorphism
# ──────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global ─────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1rem;
    }

    /* ── Sidebar ────────────────────────── */
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0E1117 0%, #1A1F2E 100%);
        border-right: 1px solid rgba(108, 99, 255, 0.12);
    }

    div[data-testid="stSidebar"] .block-container {
        padding-top: 1rem;
    }

    /* ── Metric Cards ───────────────────── */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(108,99,255,0.10), rgba(0,212,170,0.05));
        border: 1px solid rgba(108,99,255,0.20);
        border-radius: 14px;
        padding: 18px 20px;
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(108,99,255,0.18);
    }
    div[data-testid="stMetric"] label {
        color: #A0AEC0 !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.4px;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-weight: 800 !important;
        font-size: 1.55rem !important;
    }

    /* ── Tabs ────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid rgba(108,99,255,0.15);
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(108,99,255,0.08);
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        font-weight: 600;
        color: #A0AEC0;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(108,99,255,0.25) !important;
        color: #FFFFFF !important;
    }

    /* ── Glass Card ──────────────────────── */
    .glass-card {
        background: rgba(26, 31, 46, 0.80);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(108, 99, 255, 0.15);
        border-radius: 16px;
        padding: 24px;
        margin: 8px 0;
        transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        border-color: rgba(108, 99, 255, 0.35);
        box-shadow: 0 8px 32px rgba(108, 99, 255, 0.12);
    }

    /* ── Gradient Text ───────────────────── */
    .gradient-text {
        background: linear-gradient(135deg, #6C63FF, #00D4AA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 900;
    }

    /* ── Hero Section ────────────────────── */
    .hero-container {
        text-align: center;
        padding: 28px 0 18px 0;
    }
    .hero-title {
        font-size: 3.2rem;
        font-weight: 900;
        background: linear-gradient(135deg, #6C63FF 0%, #00D4AA 50%, #6C63FF 100%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: shimmer 4s ease-in-out infinite;
        margin-bottom: 4px;
        letter-spacing: -1px;
    }
    @keyframes shimmer {
        0%, 100% { background-position: 0% center; }
        50% { background-position: 100% center; }
    }
    .hero-tagline {
        font-size: 1.15rem;
        color: #8892B0;
        font-weight: 400;
        letter-spacing: 0.3px;
        margin-top: 2px;
    }
    .hero-badge {
        display: inline-block;
        margin-top: 14px;
        padding: 6px 18px;
        background: rgba(108,99,255,0.12);
        border: 1px solid rgba(108,99,255,0.30);
        border-radius: 24px;
        color: #6C63FF;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 1.2px;
        text-transform: uppercase;
    }

    /* ── Section Headers ─────────────────── */
    .section-header {
        font-size: 1.35rem;
        font-weight: 700;
        color: #E2E8F0;
        margin: 28px 0 14px 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .section-header .emoji {
        font-size: 1.4rem;
    }

    /* ── Quick Action Cards ──────────────── */
    .action-card {
        background: rgba(26, 31, 46, 0.85);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(108, 99, 255, 0.15);
        border-radius: 16px;
        padding: 28px 20px;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
        min-height: 170px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    .action-card:hover {
        border-color: rgba(108, 99, 255, 0.50);
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(108, 99, 255, 0.15);
    }
    .action-icon {
        font-size: 2.4rem;
        margin-bottom: 10px;
    }
    .action-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #E2E8F0;
        margin-bottom: 6px;
    }
    .action-desc {
        font-size: 0.78rem;
        color: #718096;
        line-height: 1.45;
    }

    /* ── Stock Tables ────────────────────── */
    .stock-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 14px;
        border-radius: 10px;
        margin: 4px 0;
        background: rgba(108, 99, 255, 0.03);
        transition: background 0.2s ease;
    }
    .stock-row:hover {
        background: rgba(108, 99, 255, 0.08);
    }
    .stock-name {
        font-weight: 600;
        color: #E2E8F0;
        font-size: 0.92rem;
    }
    .stock-price {
        color: #A0AEC0;
        font-size: 0.88rem;
        font-weight: 500;
    }
    .stock-change-green {
        color: #00D4AA;
        font-weight: 700;
        font-size: 0.88rem;
    }
    .stock-change-red {
        color: #FF6B6B;
        font-weight: 700;
        font-size: 0.88rem;
    }

    /* ── Market Status Dot ───────────────── */
    .status-dot {
        display: inline-block;
        width: 9px;
        height: 9px;
        border-radius: 50%;
        margin-right: 7px;
        position: relative;
        top: -1px;
    }
    .status-dot.open {
        background: #00D4AA;
        box-shadow: 0 0 8px rgba(0,212,170,0.55);
        animation: pulse-green 2s infinite;
    }
    .status-dot.closed {
        background: #FF6B6B;
        box-shadow: 0 0 8px rgba(255,107,107,0.45);
    }
    @keyframes pulse-green {
        0%, 100% { box-shadow: 0 0 6px rgba(0,212,170,0.40); }
        50% { box-shadow: 0 0 14px rgba(0,212,170,0.75); }
    }

    /* ── Sidebar extras ──────────────────── */
    .sidebar-brand {
        text-align: center;
        padding: 8px 0 18px 0;
        border-bottom: 1px solid rgba(108,99,255,0.12);
        margin-bottom: 18px;
    }
    .sidebar-brand-name {
        font-size: 1.6rem;
        font-weight: 900;
        background: linear-gradient(135deg, #6C63FF, #00D4AA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .sidebar-brand-sub {
        font-size: 0.72rem;
        color: #718096;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-top: 2px;
    }
    .sidebar-section-label {
        font-size: 0.7rem;
        color: #4A5568;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        font-weight: 700;
        margin: 22px 0 8px 0;
    }
    .model-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(108,99,255,0.10);
        border: 1px solid rgba(108,99,255,0.22);
        border-radius: 10px;
        padding: 8px 14px;
        font-size: 0.82rem;
        font-weight: 600;
        color: #B8B5FF;
        width: 100%;
        box-sizing: border-box;
    }
    .market-status-bar {
        display: flex;
        align-items: center;
        background: rgba(26,31,46,0.60);
        border: 1px solid rgba(108,99,255,0.10);
        border-radius: 10px;
        padding: 10px 14px;
        font-size: 0.85rem;
        color: #E2E8F0;
        font-weight: 500;
    }

    /* ── Footer ──────────────────────────── */
    .footer {
        text-align: center;
        padding: 28px 0 12px 0;
        border-top: 1px solid rgba(108,99,255,0.08);
        margin-top: 36px;
    }
    .footer-text {
        font-size: 0.75rem;
        color: #4A5568;
        letter-spacing: 0.5px;
    }

    /* ── Plotly chart container ───────────── */
    .stPlotlyChart {
        border: 1px solid rgba(108,99,255,0.10);
        border-radius: 14px;
        overflow: hidden;
    }

    /* ── Hide default Streamlit branding ─── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ── Dataframe styling ───────────────── */
    .stDataFrame {
        border: 1px solid rgba(108,99,255,0.12);
        border-radius: 12px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
IST = timezone(timedelta(hours=5, minutes=30))


def is_market_open() -> bool:
    """Check if NSE market is open based on IST time (Mon-Fri 9:15–15:30)."""
    now_ist = datetime.now(IST)
    if now_ist.weekday() >= 5:  # Saturday / Sunday
        return False
    market_open = now_ist.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now_ist.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_open <= now_ist <= market_close


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    # Brand
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-name">📈 NiveshAI</div>
        <div class="sidebar-brand-sub">Investment Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    # Quick stock search
    st.markdown('<div class="sidebar-section-label">🔍 Quick Search</div>', unsafe_allow_html=True)
    search_query = st.text_input(
        "Search stocks",
        placeholder="Search NIFTY 500 stocks...",
        label_visibility="collapsed",
        key="stock_search",
    )
    if search_query:
        st.info(f"🔎 Searching for **{search_query}** … (connect AI Agent for live results)")

    # Model Indicator
    st.markdown('<div class="sidebar-section-label">🤖 AI Model</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="model-badge">
        <span>✨</span> Gemini 2.0 Flash &nbsp;·&nbsp; <span style="color:#00D4AA;font-weight:800;">FREE</span>
    </div>
    """, unsafe_allow_html=True)

    # Market Status
    st.markdown('<div class="sidebar-section-label">🏛️ Market Status</div>', unsafe_allow_html=True)
    mkt_open = is_market_open()
    now_ist = datetime.now(IST)
    if mkt_open:
        status_html = (
            '<div class="market-status-bar">'
            '<span class="status-dot open"></span> Market Open'
            f'<span style="margin-left:auto;font-size:0.78rem;color:#718096;">{now_ist.strftime("%H:%M IST")}</span>'
            '</div>'
        )
    else:
        status_html = (
            '<div class="market-status-bar">'
            '<span class="status-dot closed"></span> Market Closed'
            f'<span style="margin-left:auto;font-size:0.78rem;color:#718096;">{now_ist.strftime("%H:%M IST")}</span>'
            '</div>'
        )
    st.markdown(status_html, unsafe_allow_html=True)

    # Navigation links
    st.markdown('<div class="sidebar-section-label">📌 Navigation</div>', unsafe_allow_html=True)
    st.page_link("app.py", label="🏠  Home", icon=None)
    st.page_link("pages/1_📊_Dashboard.py", label="📊  Dashboard", disabled=True)
    st.page_link("pages/2_🤖_AI_Agent.py", label="🤖  AI Agent", disabled=True)
    st.page_link("pages/3_📈_Predictions.py", label="📈  Predictions", disabled=True)
    st.page_link("pages/4_🎯_Portfolio.py", label="🎯  Portfolio", disabled=True)

    st.markdown("---")
    st.markdown(
        '<p style="text-align:center;font-size:0.7rem;color:#4A5568;">Made with ❤️ for Indian Investors</p>',
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────
# Hero Section
# ──────────────────────────────────────────────
st.markdown("""
<div class="hero-container">
    <div class="hero-title">NiveshAI</div>
    <div class="hero-tagline">AI-Powered Investment Research for Indian Markets</div>
    <div class="hero-badge">◆ NSE · BSE · NIFTY 500 ◆</div>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Market Index Cards
# ──────────────────────────────────────────────
st.markdown('<div class="section-header"><span class="emoji">📊</span> Market Indices — Live Snapshot</div>',
            unsafe_allow_html=True)

idx1, idx2, idx3 = st.columns(3, gap="medium")
with idx1:
    st.metric(label="NIFTY 50", value="24,850.25", delta="+294.12 (1.20%)")
with idx2:
    st.metric(label="SENSEX", value="81,234.50", delta="+726.40 (0.90%)")
with idx3:
    st.metric(label="BANK NIFTY", value="52,100.75", delta="-158.30 (-0.30%)")


# ──────────────────────────────────────────────
# Market Heatmap (Plotly Treemap)
# ──────────────────────────────────────────────
st.markdown('<div class="section-header"><span class="emoji">🗺️</span> Market Heatmap — Sector Performance</div>',
            unsafe_allow_html=True)

heatmap_data = pd.DataFrame({
    "Sector": [
        "IT", "IT", "IT",
        "Banking", "Banking", "Banking",
        "Pharma", "Pharma",
        "Energy", "Energy",
        "Auto", "Auto",
        "FMCG", "FMCG",
    ],
    "Stock": [
        "TCS", "INFY", "WIPRO",
        "HDFCBANK", "ICICIBANK", "AXISBANK",
        "SUNPHARMA", "DRREDDY",
        "RELIANCE", "ONGC",
        "MARUTI", "TATAMOTORS",
        "HINDUNILVR", "ITC",
    ],
    "Market Cap (₹ Cr)": [
        1350000, 620000, 245000,
        1180000, 780000, 340000,
        430000, 105000,
        1740000, 340000,
        380000, 320000,
        580000, 540000,
    ],
    "Change %": [
        2.1, 1.8, 1.4,
        0.9, 1.5, -0.6,
        3.2, 1.1,
        -0.4, 0.8,
        1.7, 2.8,
        -0.3, 0.5,
    ],
})

fig_heatmap = px.treemap(
    heatmap_data,
    path=["Sector", "Stock"],
    values="Market Cap (₹ Cr)",
    color="Change %",
    color_continuous_scale=["#FF6B6B", "#1A1F2E", "#00D4AA"],
    color_continuous_midpoint=0,
    template="plotly_dark",
)
fig_heatmap.update_layout(
    margin=dict(t=30, l=8, r=8, b=8),
    height=430,
    paper_bgcolor="#0E1117",
    plot_bgcolor="#0E1117",
    font=dict(family="Inter, sans-serif"),
    coloraxis_colorbar=dict(
        title="Change %",
        thickness=14,
        len=0.6,
        tickfont=dict(size=11),
    ),
)
fig_heatmap.update_traces(
    textinfo="label+text+percent root",
    textfont=dict(size=13, family="Inter, sans-serif"),
    hovertemplate="<b>%{label}</b><br>Market Cap: ₹%{value:,.0f} Cr<br>Change: %{color:+.2f}%<extra></extra>",
)
st.plotly_chart(fig_heatmap, use_container_width=True)


# ──────────────────────────────────────────────
# Top Gainers & Top Losers
# ──────────────────────────────────────────────
st.markdown('<div class="section-header"><span class="emoji">🔥</span> Top Movers Today</div>',
            unsafe_allow_html=True)

gainers_col, losers_col = st.columns(2, gap="medium")

# ── Gainers ──
top_gainers = [
    ("SUNPHARMA", "₹1,842.30", "+3.20%"),
    ("TATASTEEL", "₹178.55", "+2.85%"),
    ("MARUTI", "₹12,456.10", "+2.40%"),
    ("TCS", "₹4,215.75", "+2.10%"),
    ("BHARTIARTL", "₹1,634.90", "+1.95%"),
]

with gainers_col:
    st.markdown("""
    <div class="glass-card">
        <div style="font-size:1.05rem;font-weight:700;color:#00D4AA;margin-bottom:14px;">
            🟢 Top Gainers
        </div>
    """, unsafe_allow_html=True)
    for name, price, change in top_gainers:
        st.markdown(f"""
        <div class="stock-row">
            <span class="stock-name">{name}</span>
            <span class="stock-price">{price}</span>
            <span class="stock-change-green">{change}</span>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ── Losers ──
top_losers = [
    ("WIPRO", "₹542.20", "-1.80%"),
    ("AXISBANK", "₹1,087.65", "-1.45%"),
    ("RELIANCE", "₹2,934.40", "-1.20%"),
    ("HDFCBANK", "₹1,678.90", "-0.95%"),
    ("INFY", "₹1,582.10", "-0.70%"),
]

with losers_col:
    st.markdown("""
    <div class="glass-card">
        <div style="font-size:1.05rem;font-weight:700;color:#FF6B6B;margin-bottom:14px;">
            🔴 Top Losers
        </div>
    """, unsafe_allow_html=True)
    for name, price, change in top_losers:
        st.markdown(f"""
        <div class="stock-row">
            <span class="stock-name">{name}</span>
            <span class="stock-price">{price}</span>
            <span class="stock-change-red">{change}</span>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Quick Actions
# ──────────────────────────────────────────────
st.markdown('<div class="section-header"><span class="emoji">⚡</span> Quick Actions</div>',
            unsafe_allow_html=True)

actions = [
    ("📊", "Dashboard", "Real-time market overview, charts & watchlists", "1_📊_Dashboard"),
    ("🤖", "AI Agent", "Chat with our AI to analyse any stock instantly", "2_🤖_AI_Agent"),
    ("📈", "Predictions", "ML-powered price forecasts & trend signals", "3_📈_Predictions"),
    ("🎯", "Portfolio", "Track, optimise & rebalance your holdings", "4_🎯_Portfolio"),
]

qa1, qa2, qa3, qa4 = st.columns(4, gap="medium")

for col, (icon, title, desc, page_key) in zip([qa1, qa2, qa3, qa4], actions):
    with col:
        st.markdown(f"""
        <div class="action-card">
            <div class="action-icon">{icon}</div>
            <div class="action-title">{title}</div>
            <div class="action-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Mini Market Summary Chart (Bonus)
# ──────────────────────────────────────────────
st.markdown('<div class="section-header"><span class="emoji">📉</span> NIFTY 50 — Intraday (Mock)</div>',
            unsafe_allow_html=True)

# Generate mock intraday data
import random
random.seed(42)
times = pd.date_range("2026-07-08 09:15", "2026-07-08 15:30", freq="5min")
base = 24556.0
prices = [base]
for _ in range(len(times) - 1):
    prices.append(prices[-1] + random.uniform(-18, 20))

mock_intraday = pd.DataFrame({"Time": times[: len(prices)], "NIFTY 50": prices})

fig_intraday = go.Figure()
fig_intraday.add_trace(go.Scatter(
    x=mock_intraday["Time"],
    y=mock_intraday["NIFTY 50"],
    mode="lines",
    line=dict(color="#6C63FF", width=2.2),
    fill="tozeroy",
    fillcolor="rgba(108,99,255,0.06)",
    hovertemplate="<b>%{x|%H:%M}</b><br>NIFTY 50: %{y:,.2f}<extra></extra>",
))

# Previous close reference line
fig_intraday.add_hline(
    y=base, line_dash="dot", line_color="rgba(255,255,255,0.18)",
    annotation_text="Prev Close 24,556",
    annotation_position="bottom left",
    annotation_font=dict(size=11, color="#718096"),
)

fig_intraday.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0E1117",
    plot_bgcolor="#0E1117",
    height=330,
    margin=dict(t=20, l=55, r=20, b=40),
    xaxis=dict(
        title="",
        showgrid=False,
        tickformat="%H:%M",
        tickfont=dict(size=11),
    ),
    yaxis=dict(
        title="",
        showgrid=True,
        gridcolor="rgba(108,99,255,0.06)",
        tickfont=dict(size=11),
        tickformat=",",
    ),
    showlegend=False,
    font=dict(family="Inter, sans-serif"),
)

st.plotly_chart(fig_intraday, use_container_width=True)


# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <div class="footer-text">
        NiveshAI v1.0.0 &nbsp;·&nbsp; Built for Hackathon 2026 &nbsp;·&nbsp;
        Data shown is <strong>mock / placeholder</strong> — not financial advice &nbsp;·&nbsp;
        Powered by <span style="color:#6C63FF;font-weight:600;">Gemini 2.0 Flash</span>
    </div>
</div>
""", unsafe_allow_html=True)
