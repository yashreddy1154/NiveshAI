"""
NiveshAI — AI-Powered Investment Research for Indian Markets
Home Page (app.py)
"""

import subprocess, sys
# Auto-download models if missing
try:
    from scripts.download_models_from_hf import ensure_models_downloaded
    ensure_models_downloaded()
except Exception:
    pass  # Models optional — fallbacks exist

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

# Internal modules
from data.stock_data import fetch_market_indices, fetch_multiple_prices, is_market_open
from data.news_fetcher import fetch_market_news
from data.company_db import get_display_options, parse_display_option, load_nifty500, get_sectors


# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NiveshAI — AI Investment Research",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS — Premium Dark Theme + Glassmorphism ─────────────────────────
st.markdown("""
<style>
    /* ── Global ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1rem;
    }

    /* ── Sidebar ── */
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0E1117 0%, #1A1F2E 100%);
        border-right: 1px solid rgba(108, 99, 255, 0.12);
    }

    /* ── Metric Cards ── */
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

    /* ── Glass Card ── */
    .glass-card {
        background: rgba(26, 31, 46, 0.80);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(108, 99, 255, 0.15);
        border-radius: 16px;
        padding: 24px;
        margin: 8px 0;
    }

    /* ── Gradient Text ── */
    .gradient-text {
        background: linear-gradient(135deg, #6C63FF, #00D4AA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 900;
    }

    /* ── Hero Section ── */
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

    /* ── Section Headers ── */
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

    /* ── Quick Action Cards ── */
    .action-card {
        background: rgba(26, 31, 46, 0.85);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(108, 99, 255, 0.15);
        border-radius: 16px;
        padding: 28px 20px;
        text-align: center;
        min-height: 170px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        transition: transform 0.25s ease, border-color 0.25s ease;
    }
    .action-card:hover {
        border-color: rgba(108, 99, 255, 0.50);
        transform: translateY(-4px);
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

    /* ── Stock Rows ── */
    .stock-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 14px;
        border-radius: 10px;
        margin: 4px 0;
        background: rgba(108, 99, 255, 0.03);
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

    /* ── Market Status Dot ── */
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

    /* ── Sidebar brand ── */
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

    /* ── Footer ── */
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
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
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
    
    @st.cache_data
    def get_stock_options():
        return get_display_options()

    stock_options = get_stock_options()
    search_stock = st.selectbox(
        "Search Stock",
        options=[""] + stock_options,
        index=0,
        label_visibility="collapsed",
    )
    if search_stock:
        symbol = parse_display_option(search_stock)
        st.info(f"🔎 Selected: **{symbol}**")
        st.page_link("pages/1_📊_Dashboard.py", label="📊 Go to Dashboard")

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
    now_ist = datetime.now(timezone(timedelta(hours=5, minutes=30)))
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

    # Navigation links (FULLY COMPATIBLE & ENABLED)
    st.markdown('<div class="sidebar-section-label">📌 Navigation</div>', unsafe_allow_html=True)
    st.page_link("app.py", label="🏠  Home")
    st.page_link("pages/1_📊_Dashboard.py", label="📊  Dashboard")
    st.page_link("pages/2_🤖_AI_Research_Agent.py", label="🤖  AI Agent")
    st.page_link("pages/3_📈_Predictions.py", label="📈  Predictions")
    st.page_link("pages/4_🎯_Portfolio_Optimizer.py", label="🎯  Portfolio")
    st.page_link("pages/5_📋_Report_Generator.py", label="📋  Report Generator")
    st.page_link("pages/6_⚙️_Settings.py", label="⚙️  Settings")

    st.markdown("---")
    st.markdown(
        '<p style="text-align:center;font-size:0.7rem;color:#4A5568;">Made with ❤️ for Indian Investors</p>',
        unsafe_allow_html=True,
    )

# ─── Hero Section ────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-container">
    <div class="hero-title">NiveshAI</div>
    <div class="hero-tagline">AI-Powered Investment Research for Indian Markets</div>
    <div class="hero-badge">◆ NSE · BSE · NIFTY 500 ◆</div>
</div>
""", unsafe_allow_html=True)

# ─── Fetch Real Market Indices ────────────────────────────────────────────────
@st.cache_data(ttl=900)  # cache 15 minutes
def get_indices():
    data = fetch_market_indices()
    # Check if indices data is empty or invalid
    if not data or all(v.get("price", 0) == 0 for v in data.values()):
        raise ValueError("Invalid index data fetched")
    return data, datetime.now()

# Initialize session state for caching last successful values
if "last_indices" not in st.session_state:
    st.session_state.last_indices = None
if "last_indices_time" not in st.session_state:
    st.session_state.last_indices_time = None

st.markdown('<div class="section-header"><span class="emoji">📊</span> Market Indices — Live Snapshot</div>',
            unsafe_allow_html=True)

try:
    indices_data, indices_time = get_indices()
    st.session_state.last_indices = indices_data
    st.session_state.last_indices_time = indices_time
    indices_warning = False
except Exception as e:
    indices_warning = True
    indices_data = st.session_state.get("last_indices")
    indices_time = st.session_state.get("last_indices_time")
    
    if not indices_data:
        # Fallback values
        indices_data = {
            "NIFTY 50": {"price": 24850.25, "change": 294.12, "change_pct": 1.20},
            "SENSEX": {"price": 81234.50, "change": 726.40, "change_pct": 0.90},
            "BANK NIFTY": {"price": 52100.75, "change": -158.30, "change_pct": -0.30}
        }
        indices_time = None

if indices_warning:
    st.warning("Live data unavailable — showing cached data")

if indices_time:
    diff_mins = int((datetime.now() - indices_time).total_seconds() / 60)
    st.markdown(f"<p style='font-size: 0.8rem; color: #8892B0; margin-top: -10px; margin-bottom: 10px;'>🔄 Last updated: {diff_mins} min ago</p>", unsafe_allow_html=True)
else:
    st.markdown("<p style='font-size: 0.8rem; color: #8892B0; margin-top: -10px; margin-bottom: 10px;'>🔄 Last updated: ---</p>", unsafe_allow_html=True)

# Helper functions for safe formatting
def format_metric_value(val):
    if isinstance(val, (int, float)) and val != 0:
        return f"{val:,.2f}"
    return "---"

def format_metric_delta(chg, chg_pct):
    if isinstance(chg, (int, float)) and isinstance(chg_pct, (int, float)):
        return f"{chg:+.2f} ({chg_pct:+.2f}%)"
    return None

nifty_price = indices_data.get("NIFTY 50", {}).get("price", 0)
nifty_chg = indices_data.get("NIFTY 50", {}).get("change", 0)
nifty_chg_pct = indices_data.get("NIFTY 50", {}).get("change_pct", 0)

sensex_price = indices_data.get("SENSEX", {}).get("price", 0)
sensex_chg = indices_data.get("SENSEX", {}).get("change", 0)
sensex_chg_pct = indices_data.get("SENSEX", {}).get("change_pct", 0)

banknifty_price = indices_data.get("BANK NIFTY", {}).get("price", 0)
banknifty_chg = indices_data.get("BANK NIFTY", {}).get("change", 0)
banknifty_chg_pct = indices_data.get("BANK NIFTY", {}).get("change_pct", 0)

idx1, idx2, idx3 = st.columns(3, gap="medium")
with idx1:
    st.metric(label="NIFTY 50", value=format_metric_value(nifty_price), delta=format_metric_delta(nifty_chg, nifty_chg_pct))
with idx2:
    st.metric(label="SENSEX", value=format_metric_value(sensex_price), delta=format_metric_delta(sensex_chg, sensex_chg_pct))
with idx3:
    st.metric(label="BANK NIFTY", value=format_metric_value(banknifty_price), delta=format_metric_delta(banknifty_chg, banknifty_chg_pct))

# ─── Fetch Top Movers ─────────────────────────────────────────────────────────
@st.cache_data(ttl=900)
def get_movers():
    symbols = ["RELIANCE","TCS","INFY","HDFCBANK","ICICIBANK",
               "BHARTIARTL","SBIN","ITC","WIPRO","BAJFINANCE",
               "HCLTECH","MARUTI","SUNPHARMA","AXISBANK","LT",
               "KOTAKBANK","TATASTEEL","HINDUNILVR","ADANIENT","TITAN"]
    df = fetch_multiple_prices(symbols)
    if df.empty:
        raise ValueError("No movers data fetched")
    return df, datetime.now()

if "last_movers" not in st.session_state:
    st.session_state.last_movers = None
if "last_movers_time" not in st.session_state:
    st.session_state.last_movers_time = None

st.markdown('<div class="section-header"><span class="emoji">🔥</span> Top Movers Today</div>',
            unsafe_allow_html=True)

try:
    movers_df, movers_time = get_movers()
    st.session_state.last_movers = movers_df
    st.session_state.last_movers_time = movers_time
    movers_warning = False
except Exception as e:
    movers_warning = True
    movers_df = st.session_state.get("last_movers")
    movers_time = st.session_state.get("last_movers_time")
    
    if movers_df is None or movers_df.empty:
        fallback_data = [
            {"symbol": "SUNPHARMA", "price": 1842.30, "change": 57.10, "change_pct": 3.20},
            {"symbol": "TCS", "price": 4215.75, "change": 86.85, "change_pct": 2.10},
            {"symbol": "BHARTIARTL", "price": 1634.90, "change": 31.25, "change_pct": 1.95},
            {"symbol": "INFY", "price": 1582.10, "change": 18.00, "change_pct": 1.15},
            {"symbol": "SBIN", "price": 842.15, "change": 7.95, "change_pct": 0.95},
            {"symbol": "ITC", "price": 468.35, "change": -8.60, "change_pct": -1.80},
            {"symbol": "LT", "price": 3680.25, "change": -54.10, "change_pct": -1.45},
            {"symbol": "RELIANCE", "price": 2934.40, "change": -35.60, "change_pct": -1.20},
            {"symbol": "HDFCBANK", "price": 1678.90, "change": -16.10, "change_pct": -0.95},
            {"symbol": "ICICIBANK", "price": 1285.75, "change": -9.05, "change_pct": -0.70},
        ]
        movers_df = pd.DataFrame(fallback_data).set_index("symbol")
        movers_time = None

if movers_warning:
    st.warning("Live data unavailable — showing cached data")

if movers_time:
    diff_mins = int((datetime.now() - movers_time).total_seconds() / 60)
    st.markdown(f"<p style='font-size: 0.8rem; color: #8892B0; margin-top: -10px; margin-bottom: 10px;'>🔄 Last updated: {diff_mins} min ago</p>", unsafe_allow_html=True)
else:
    st.markdown("<p style='font-size: 0.8rem; color: #8892B0; margin-top: -10px; margin-bottom: 10px;'>🔄 Last updated: ---</p>", unsafe_allow_html=True)

# Sort and pick top 5 gainers and losers
sorted_df = movers_df.sort_values(by="change_pct", ascending=False)

top_gainers = []
for sym, row in sorted_df.head(5).iterrows():
    top_gainers.append((sym, f"₹{row['price']:,.2f}", f"{row['change_pct']:+.2f}%"))
    
top_losers = []
for sym, row in sorted_df.tail(5).iloc[::-1].iterrows():
    top_losers.append((sym, f"₹{row['price']:,.2f}", f"{row['change_pct']:+.2f}%"))

gainers_col, losers_col = st.columns(2, gap="medium")

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

# ─── Market News Feed ────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def get_news():
    news_list = fetch_market_news(max_articles=8)
    if not news_list:
        raise ValueError("No news data fetched")
    return news_list, datetime.now()

if "last_news" not in st.session_state:
    st.session_state.last_news = None
if "last_news_time" not in st.session_state:
    st.session_state.last_news_time = None

st.markdown('<div class="section-header"><span class="emoji">📰</span> Market News</div>', unsafe_allow_html=True)

try:
    news_data, news_time = get_news()
    st.session_state.last_news = news_data
    st.session_state.last_news_time = news_time
    news_warning = False
except Exception as e:
    news_warning = True
    news_data = st.session_state.get("last_news")
    news_time = st.session_state.get("last_news_time")
    
    if not news_data:
        news_data = [
            {
                "title": "Nifty 50, Sensex hit record highs amid strong global cues",
                "source": "Moneycontrol",
                "published_at": datetime.now().strftime("%Y-%m-%d"),
                "summary": "Indian benchmark indices hit fresh record highs on Friday led by IT and banking stocks.",
                "url": "https://www.moneycontrol.com"
            },
            {
                "title": "FIIs turn net buyers in Indian equities, pump in ₹2,500 crore",
                "source": "Economic Times",
                "published_at": datetime.now().strftime("%Y-%m-%d"),
                "summary": "Foreign institutional investors (FIIs) remained net buyers in the cash segment, support index momentum.",
                "url": "https://economictimes.indiatimes.com"
            }
        ]
        news_time = None

if news_warning:
    st.warning("Live news unavailable — showing cached data")

if news_time:
    diff_mins = int((datetime.now() - news_time).total_seconds() / 60)
    st.markdown(f"<p style='font-size: 0.8rem; color: #8892B0; margin-top: -10px; margin-bottom: 10px;'>🔄 Last updated: {diff_mins} min ago</p>", unsafe_allow_html=True)
else:
    st.markdown("<p style='font-size: 0.8rem; color: #8892B0; margin-top: -10px; margin-bottom: 10px;'>🔄 Last updated: ---</p>", unsafe_allow_html=True)

for article in news_data:
    header = f"{article['title']} — {article['source']} ({article.get('published_at', 'N/A')})"
    with st.expander(header):
        st.write(article.get("summary", "No summary available."))
        st.markdown(f"[Read full article]({article['url']})")

# ─── Market Heatmap ──────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def get_sector_heatmap_data():
    df_nifty = load_nifty500()
    sectors = get_sectors()
    
    sector_stocks = []
    all_symbols = []
    
    for sector in sectors:
        sec_df = df_nifty[df_nifty["Sector"] == sector]
        # Get 4 symbols per sector
        symbols = sec_df["Symbol"].head(4).tolist()
        for sym in symbols:
            sector_stocks.append({"Sector": sector, "Symbol": sym})
            all_symbols.append(sym)
            
    # Fetch multiple prices in a batch
    prices_df = fetch_multiple_prices(all_symbols)
    if prices_df.empty:
        raise ValueError("Sector prices data is empty")
        
    rows = []
    for item in sector_stocks:
        sym = item["Symbol"]
        sector = item["Sector"]
        change_pct = 0.0
        price = 0.0
        if sym in prices_df.index:
            row_data = prices_df.loc[sym]
            if isinstance(row_data, pd.DataFrame):
                row_data = row_data.iloc[0]
            change_pct = float(row_data.get("change_pct", 0.0))
            price = float(row_data.get("price", 0.0))
        rows.append({
            "Sector": sector,
            "Stock": sym,
            "Change %": change_pct,
            "Price": price,
            "Weight": 1
        })
        
    df_heatmap = pd.DataFrame(rows)
    return df_heatmap, datetime.now()

if "last_heatmap" not in st.session_state:
    st.session_state.last_heatmap = None
if "last_heatmap_time" not in st.session_state:
    st.session_state.last_heatmap_time = None

st.markdown('<div class="section-header"><span class="emoji">🗺️</span> Market Heatmap — Sector Performance</div>',
            unsafe_allow_html=True)

try:
    heatmap_df, heatmap_time = get_sector_heatmap_data()
    st.session_state.last_heatmap = heatmap_df
    st.session_state.last_heatmap_time = heatmap_time
    heatmap_warning = False
except Exception as e:
    heatmap_warning = True
    heatmap_df = st.session_state.get("last_heatmap")
    heatmap_time = st.session_state.get("last_heatmap_time")
    
    if heatmap_df is None or heatmap_df.empty:
        # Fallback to the original mock heatmap data
        heatmap_df = pd.DataFrame({
            "Sector": [
                "IT", "IT", "IT",
                "Banking", "Banking", "Banking",
                "Pharma", "Energy", "Energy",
                "Auto", "Auto", "FMCG", "FMCG",
            ],
            "Stock": [
                "TCS", "INFY", "WIPRO",
                "HDFCBANK", "ICICIBANK", "AXISBANK",
                "SUNPHARMA", "RELIANCE", "ONGC",
                "MARUTI", "TATAMOTORS", "HINDUNILVR", "ITC",
            ],
            "Weight": [1] * 13,
            "Change %": [2.1, 1.8, 1.4, 0.9, 1.5, -0.6, 3.2, -0.4, 0.8, 1.7, 2.8, -0.3, 0.5],
            "Price": [4215.75, 1582.10, 485.20, 1678.90, 1285.75, 1120.40, 1842.30, 2934.40, 260.15, 12450.00, 985.40, 2450.00, 468.35]
        })
        heatmap_time = None

if heatmap_warning:
    st.warning("Live data unavailable — showing cached data")

if heatmap_time:
    diff_mins = int((datetime.now() - heatmap_time).total_seconds() / 60)
    st.markdown(f"<p style='font-size: 0.8rem; color: #8892B0; margin-top: -10px; margin-bottom: 10px;'>🔄 Last updated: {diff_mins} min ago</p>", unsafe_allow_html=True)
else:
    st.markdown("<p style='font-size: 0.8rem; color: #8892B0; margin-top: -10px; margin-bottom: 10px;'>🔄 Last updated: ---</p>", unsafe_allow_html=True)

fig_heatmap = px.treemap(
    heatmap_df,
    path=["Sector", "Stock"],
    values="Weight",
    color="Change %",
    color_continuous_scale=["#FF6B6B", "#1A1F2E", "#00D4AA"],
    color_continuous_midpoint=0,
    template="plotly_dark",
)
fig_heatmap.update_layout(
    margin=dict(t=30, l=8, r=8, b=8),
    height=400,
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
    textinfo="label+text",
    textfont=dict(size=13, family="Inter, sans-serif"),
    hovertemplate="<b>%{label}</b><br>Change: %{color:+.2f}%<extra></extra>",
)
st.plotly_chart(fig_heatmap, use_container_width=True)


# ─── Quick Actions ───────────────────────────────────────────────────────────
st.markdown('<div class="section-header"><span class="emoji">⚡</span> Quick Actions</div>',
            unsafe_allow_html=True)

qa_cols = st.columns(4, gap="medium")
actions_info = [
    ("📊", "Dashboard", "Real-time stock analytics, fundamentals & watchlists", "pages/1_📊_Dashboard.py"),
    ("🤖", "AI Agent", "Ask our AI research agent anything about NSE stocks", "pages/2_🤖_AI_Research_Agent.py"),
    ("📈", "Predictions", "ML price predictions & directional classifier analysis", "pages/3_📈_Predictions.py"),
    ("🎯", "Portfolio", "Optimize allocations using Modern Portfolio Theory", "pages/4_🎯_Portfolio_Optimizer.py"),
]

for col, (icon, title, desc, page_path) in zip(qa_cols, actions_info):
    with col:
        st.markdown(f"""
        <div class="action-card">
            <div class="action-icon">{icon}</div>
            <div class="action-title">{title}</div>
            <div class="action-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)
        st.page_link(page_path, label=f"Go to {title}", use_container_width=True)

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="footer">
    <div class="footer-text">
        NiveshAI v1.0.0 &nbsp;·&nbsp; Indian Stock Market Investment Research &nbsp;·&nbsp;
        Powered by <span style="color:#6C63FF;font-weight:600;">Gemini 2.0 Flash</span>
    </div>
</div>
""", unsafe_allow_html=True)
