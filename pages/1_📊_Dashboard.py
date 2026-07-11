import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Internal modules
from data.company_db import get_display_options, parse_display_option, get_company
from data.stock_data import fetch_history, fetch_live_price, fetch_fundamentals, fetch_financials, format_inr
from data.news_fetcher import fetch_news
from analysis.technical import compute_all_indicators, generate_signals, get_support_resistance
from analysis.fundamental import analyze_fundamentals
from analysis.risk import assess_risk
from config.settings import NEWSAPI_KEY

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dashboard — NiveshAI",
    page_icon="📊",
    layout="wide",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Base Theme ── */
    .stApp {
        background-color: #0E1117;
    }
    .stMetric {
        background: linear-gradient(135deg, rgba(108, 99, 255, 0.1), rgba(0, 212, 170, 0.05));
        border: 1px solid rgba(108, 99, 255, 0.2);
        border-radius: 12px;
        padding: 16px;
    }
    .stMetric:hover {
        border-color: rgba(108, 99, 255, 0.5);
        box-shadow: 0 0 20px rgba(108, 99, 255, 0.1);
        transition: all 0.3s ease;
    }
    .glass-card {
        background: rgba(26, 31, 46, 0.8);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(108, 99, 255, 0.15);
        border-radius: 16px;
        padding: 24px;
        margin: 8px 0;
    }
    .gradient-text {
        background: linear-gradient(135deg, #6C63FF, #00D4AA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0E1117 0%, #1A1F2E 100%);
        border-right: 1px solid rgba(108, 99, 255, 0.15);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(108, 99, 255, 0.1);
        border-radius: 8px;
        color: #C0C0C0;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(108, 99, 255, 0.3) !important;
        color: #FFFFFF !important;
    }

    /* ── Glassmorphism Cards ── */
    .info-card {
        background: rgba(26, 31, 46, 0.85);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(108, 99, 255, 0.18);
        border-radius: 16px;
        padding: 24px 28px;
        margin: 10px 0;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .info-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(108, 99, 255, 0.12);
    }

    /* ── Badge Styles ── */
    .badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .badge-buy {
        background: rgba(0, 212, 170, 0.18);
        color: #00D4AA;
        border: 1px solid rgba(0, 212, 170, 0.35);
    }
    .badge-sell {
        background: rgba(255, 107, 107, 0.18);
        color: #FF6B6B;
        border: 1px solid rgba(255, 107, 107, 0.35);
    }
    .badge-hold {
        background: rgba(255, 179, 71, 0.18);
        color: #FFB347;
        border: 1px solid rgba(255, 179, 71, 0.35);
    }
    .badge-positive {
        background: rgba(0, 212, 170, 0.15);
        color: #00D4AA;
        border: 1px solid rgba(0, 212, 170, 0.3);
    }
    .badge-negative {
        background: rgba(255, 107, 107, 0.15);
        color: #FF6B6B;
        border: 1px solid rgba(255, 107, 107, 0.3);
    }
    .badge-neutral {
        background: rgba(255, 230, 109, 0.15);
        color: #FFE66D;
        border: 1px solid rgba(255, 230, 109, 0.3);
    }

    /* ── News Card ── */
    .news-item {
        background: rgba(26, 31, 46, 0.7);
        border: 1px solid rgba(108, 99, 255, 0.12);
        border-radius: 12px;
        padding: 18px 22px;
        margin: 8px 0;
        transition: border-color 0.2s ease;
    }
    .news-item:hover {
        border-color: rgba(108, 99, 255, 0.4);
    }
    .news-title {
        color: #E8E8E8;
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .news-meta {
        color: #888;
        font-size: 0.8rem;
    }

    /* ── Signal Box ── */
    .signal-box {
        background: rgba(26, 31, 46, 0.85);
        border: 1px solid rgba(108, 99, 255, 0.2);
        border-radius: 14px;
        padding: 20px;
        text-align: center;
    }
    .signal-label {
        color: #888;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    .signal-value {
        font-size: 1.3rem;
        font-weight: 700;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0E1117; }
    ::-webkit-scrollbar-thumb { background: rgba(108, 99, 255, 0.4); border-radius: 3px; }

    /* ── Divider ── */
    .section-divider {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(108,99,255,0.3), transparent);
        margin: 24px 0;
    }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<h2 style="text-align:center;">'
        '<span class="gradient-text">NiveshAI</span></h2>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="text-align:center;color:#888;font-size:0.85rem;">'
        'AI-Powered Investment Research</p>',
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    @st.cache_data
    def get_stock_options():
        try:
            return get_display_options()  # "RELIANCE — Reliance Industries Ltd"
        except Exception:
            return ["RELIANCE — Reliance Industries Ltd", "TCS — Tata Consultancy Services Ltd"]

    options = get_stock_options()
    selected = st.sidebar.selectbox("Select Stock", options)
    symbol = parse_display_option(selected)  # extracts "RELIANCE"

    st.markdown("")  # spacer
    period = st.sidebar.selectbox("📅 Select Timeframe", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    status_placeholder = st.empty()
    st.markdown(
        '<p style="color:#555;font-size:0.75rem;text-align:center;">'
        '⚡ Data fetched in real-time from Yahoo Finance</p>',
        unsafe_allow_html=True,
    )

# ─── Load Data ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=900, show_spinner=False)
def load_stock_data(symbol, period):
    import time
    history = fetch_history(symbol, period=period, add_indicators=True)
    history = compute_all_indicators(history)  # adds ADX, Stochastic, OBV
    live    = fetch_live_price(symbol)
    fund    = fetch_fundamentals(symbol)
    fetch_timestamp = time.time()
    return history, live, fund, fetch_timestamp

with st.spinner(f"Loading {symbol} data..."):
    history, live, fundamentals, fetch_timestamp = load_stock_data(symbol, period)

# Show Cached/Live indicator in sidebar status_placeholder
import time
is_cached = (time.time() - fetch_timestamp) > 2.0
if is_cached:
    status_placeholder.markdown(
        f'<div style="text-align:center;padding:8px;border-radius:8px;'
        f'background-color:rgba(108,99,255,0.1);border:1px solid rgba(108,99,255,0.3);color:#C0C0C0;font-size:0.85rem;">'
        f'📦 Cached Data (Updated at {datetime.fromtimestamp(fetch_timestamp).strftime("%I:%M:%S %p")})'
        f'</div>',
        unsafe_allow_html=True
    )
else:
    status_placeholder.markdown(
        f'<div style="text-align:center;padding:8px;border-radius:8px;'
        f'background-color:rgba(0,212,170,0.1);border:1px solid rgba(0,212,170,0.3);color:#00D4AA;font-size:0.85rem;">'
        f'⚡ Live Data (Updated just now)'
        f'</div>',
        unsafe_allow_html=True
    )

# Error handling if history is empty DataFrame
if history is None or history.empty:
    st.error(f"Could not load data for {symbol}. Try another stock.")
    st.stop()

# Derive metrics
latest = history.iloc[-1]
prev = history.iloc[-2] if len(history) > 1 else latest

# Safely get live price & change values
try:
    live_price = live.get("price")
    if live_price is None or (isinstance(live_price, float) and np.isnan(live_price)):
        live_price_str = "--"
    else:
        live_price_str = f"₹{live_price:,.2f}"
except Exception:
    live_price_str = "--"

try:
    price_change = live.get("change")
    price_change_pct = live.get("change_pct")
    if price_change is None or (isinstance(price_change, float) and np.isnan(price_change)):
        price_change = latest["Close"] - prev["Close"] if len(history) > 1 else 0.0
        price_change_str = f"₹{abs(price_change):,.2f}"
        price_change_pct = ((latest["Close"] - prev["Close"]) / prev["Close"]) * 100 if len(history) > 1 else 0.0
        price_change_pct_str = f"{price_change_pct:+.2f}%"
        change_color = "#00D4AA" if price_change >= 0 else "#FF6B6B"
        arrow = "▲" if price_change >= 0 else "▼"
    else:
        change_color = "#00D4AA" if price_change >= 0 else "#FF6B6B"
        arrow = "▲" if price_change >= 0 else "▼"
        price_change_str = f"₹{abs(price_change):,.2f}"
        price_change_pct_str = f"{price_change_pct:+.2f}%"
except Exception:
    price_change_str = "--"
    price_change_pct_str = "--"
    change_color = "#C0C0C0"
    arrow = ""

try:
    live_vol = live.get("volume")
    if live_vol is None or np.isnan(live_vol):
        live_vol_str = f"{latest['Volume']/1e6:,.2f}M"
    else:
        live_vol_str = f"{live_vol/1e6:,.2f}M"
except Exception:
    live_vol_str = "--"

try:
    live_low = live.get("low")
    live_high = live.get("high")
    if live_low is None or np.isnan(live_low) or live_high is None or np.isnan(live_high):
        day_range_str = f"₹{latest['Low']:,.2f} — ₹{latest['High']:,.2f}"
    else:
        day_range_str = f"₹{live_low:,.2f} — ₹{live_high:,.2f}"
except Exception:
    day_range_str = "--"

# ─── Title & Summary Bar ────────────────────────────────────────────────────
company_info = get_company(symbol) or {}
company_name = company_info.get("Company Name", fundamentals.get("company_name", symbol))

st.markdown(
    f'<h1 style="margin-bottom:0;">'
    f'<span class="gradient-text">📊 Stock Dashboard</span></h1>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<p style="color:#888;margin-top:0;">Analyzing '
    f'<b style="color:#6C63FF">{symbol}</b> — '
    f'{company_name}</p>',
    unsafe_allow_html=True,
)

# Quick price strip
st.markdown(
    f'<div class="info-card" style="display:flex;align-items:center;gap:32px;">'
    f'  <div>'
    f'    <span style="color:#888;font-size:0.85rem;">Current Price</span><br>'
    f'    <span style="font-size:2rem;font-weight:800;color:#E8E8E8;">'
    f'      {live_price_str}</span>'
    f'  </div>'
    f'  <div>'
    f'    <span style="color:{change_color};font-size:1.1rem;font-weight:700;">'
    f'      {arrow} {price_change_str} ({price_change_pct_str})</span>'
    f'  </div>'
    f'  <div style="margin-left:auto;text-align:right;">'
    f'    <span style="color:#888;font-size:0.8rem;">Volume</span><br>'
    f'    <span style="color:#C0C0C0;font-weight:600;">'
    f'      {live_vol_str}</span>'
    f'  </div>'
    f'  <div style="text-align:right;">'
    f'    <span style="color:#888;font-size:0.8rem;">Day Range</span><br>'
    f'    <span style="color:#C0C0C0;font-weight:600;">'
    f'      {day_range_str}</span>'
    f'  </div>'
    f'</div>',
    unsafe_allow_html=True,
)

st.markdown("")

# ─── Tabs ────────────────────────────────────────────────────────────────────
tab_price, tab_fund, tab_tech, tab_news = st.tabs(
    ["📈 Price Chart", "📋 Fundamentals", "🔧 Technical", "📰 News Sentiment"]
)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Price Chart
# ═══════════════════════════════════════════════════════════════════════════════
with tab_price:
    try:
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.75, 0.25],
            subplot_titles=("", ""),
        )

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=history.index, open=history["Open"], high=history["High"],
            low=history["Low"], close=history["Close"],
            increasing_line_color="#00D4AA", increasing_fillcolor="rgba(0,212,170,0.35)",
            decreasing_line_color="#FF6B6B", decreasing_fillcolor="rgba(255,107,107,0.35)",
            name="Price",
        ), row=1, col=1)

        # SMA overlays
        if "SMA_20" in history.columns:
            fig.add_trace(go.Scatter(
                x=history.index, y=history["SMA_20"],
                line=dict(color="#6C63FF", width=1.5, dash="dot"),
                name="SMA 20",
            ), row=1, col=1)

        if "SMA_50" in history.columns:
            fig.add_trace(go.Scatter(
                x=history.index, y=history["SMA_50"],
                line=dict(color="#FFB347", width=1.5, dash="dot"),
                name="SMA 50",
            ), row=1, col=1)

        # Volume bars
        vol_colors = [
            "rgba(0,212,170,0.5)" if history["Close"].iloc[i] >= history["Open"].iloc[i]
            else "rgba(255,107,107,0.5)"
            for i in range(len(history))
        ]
        fig.add_trace(go.Bar(
            x=history.index, y=history["Volume"],
            marker_color=vol_colors, name="Volume",
            showlegend=False,
        ), row=2, col=1)

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            height=600,
            margin=dict(l=0, r=0, t=20, b=0),
            xaxis_rangeslider_visible=False,
            xaxis2_rangeslider_visible=True,
            xaxis2_rangeslider_thickness=0.04,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                font=dict(size=11),
            ),
            yaxis=dict(title="Price (₹)", gridcolor="rgba(108,99,255,0.08)"),
            yaxis2=dict(title="Volume", gridcolor="rgba(108,99,255,0.08)"),
            xaxis=dict(gridcolor="rgba(108,99,255,0.08)"),
            xaxis2=dict(gridcolor="rgba(108,99,255,0.08)"),
        )

        st.plotly_chart(fig, use_container_width=True)

        # Quick stats below chart
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Open", f"₹{latest['Open']:,.2f}")
        c2.metric("High", f"₹{latest['High']:,.2f}")
        c3.metric("Low", f"₹{latest['Low']:,.2f}")
        pct_change_val = price_change_pct if isinstance(price_change_pct, float) and not np.isnan(price_change_pct) else 0.0
        c4.metric("Close", f"₹{latest['Close']:,.2f}", f"{pct_change_val:+.2f}%")
        c5.metric("Volume", f"{latest['Volume']/1e6:,.2f}M")
    except Exception as e:
        st.error(f"Failed to load price chart: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Fundamentals
# ═══════════════════════════════════════════════════════════════════════════════
with tab_fund:
    try:
        fund_analysis = analyze_fundamentals(fundamentals)
        rating = fund_analysis["rating"]
        emoji = fund_analysis["rating_emoji"]
        score = fund_analysis["score"]

        # Match styling color class
        rating_color_map = {
            "Strong Buy": "badge-buy",
            "Buy": "badge-buy",
            "Hold": "badge-hold",
            "Underperform": "badge-sell",
            "Sell": "badge-sell"
        }
        rating_cls = rating_color_map.get(rating, "badge-neutral")

        # ── Company Info Card ──
        st.markdown(
            f'<div class="info-card">'
            f'  <div style="display:flex;justify-content:space-between;align-items:center;">'
            f'    <div>'
            f'      <h3 style="margin:0;color:#E8E8E8;">{company_name}</h3>'
            f'      <p style="margin:4px 0 0;color:#6C63FF;font-size:0.9rem;">'
            f'        NSE: {symbol} | Fundamental Score: {score}/100</p>'
            f'    </div>'
            f'    <div style="text-align:right;">'
            f'      <span style="color:#888;font-size:0.82rem;">Rating</span><br>'
            f'      <span class="badge {rating_cls}" style="font-size:1rem;padding:4px 12px;margin-top:4px;">'
            f'        {emoji} {rating}</span>'
            f'    </div>'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown("")
        st.markdown("##### 📊 Key Fundamental Metrics")

        # ── Key Metrics Grid ──
        metrics_dict = fund_analysis.get("metrics", {})
        if metrics_dict:
            cols = st.columns(4)
            for idx, (k, v) in enumerate(metrics_dict.items()):
                cols[idx % 4].metric(label=k, value=str(v))

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        col_strengths, col_summary = st.columns([1, 1.2])

        with col_strengths:
            st.markdown("##### 🟢 Strengths & 🔴 Concerns")
            strengths = fund_analysis.get("strengths", [])
            for strength in strengths:
                st.markdown(f"🟢 <span style='font-size:0.9rem;'>{strength}</span>", unsafe_allow_html=True)
            weaknesses = fund_analysis.get("weaknesses", [])
            for weakness in weaknesses:
                st.markdown(f"🔴 <span style='font-size:0.9rem;'>{weakness}</span>", unsafe_allow_html=True)
            if not strengths and not weaknesses:
                st.markdown("No significant strengths or concerns identified.")

        with col_summary:
            st.markdown("##### 📝 Fundamental Summary")
            st.write(fund_analysis.get("summary", ""))

            # Display description if exists
            desc = fundamentals.get("description", "")
            if desc:
                st.markdown("**Business Profile:**")
                st.caption(desc)

        # ── Quarterly financials if available ──
        @st.cache_data(ttl=900, show_spinner=False)
        def load_financials_cached(sym):
            try:
                return fetch_financials(sym)
            except Exception:
                return {}
        
        financials = load_financials_cached(symbol)
        income_stmt = financials.get("income", pd.DataFrame()) if isinstance(financials, dict) else pd.DataFrame()
        if not income_stmt.empty:
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            st.markdown("##### 📊 Recent Financial Performance (Quarterly)")
            
            try:
                inc_df = income_stmt.T.copy()
                inc_df.index = pd.to_datetime(inc_df.index).strftime("%b %Y")
                
                rev_key = next((k for k in inc_df.columns if "revenue" in k.lower()), None)
                net_key = next((k for k in inc_df.columns if "net income" in k.lower() or "netprofit" in k.lower()), None)
                
                if rev_key and net_key:
                    fig_rev = go.Figure()
                    fig_rev.add_trace(go.Bar(
                        x=inc_df.index, y=inc_df[rev_key],
                        name="Revenue",
                        marker_color="rgba(108,99,255,0.7)",
                    ))
                    fig_rev.add_trace(go.Bar(
                        x=inc_df.index, y=inc_df[net_key],
                        name="Net Income",
                        marker_color="rgba(0,212,170,0.7)",
                    ))
                    fig_rev.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="#0E1117",
                        plot_bgcolor="#0E1117",
                        height=300,
                        margin=dict(l=0, r=0, t=30, b=0),
                        barmode="group",
                        yaxis=dict(title="Value (₹)", gridcolor="rgba(108,99,255,0.08)"),
                    )
                    st.plotly_chart(fig_rev, use_container_width=True)
            except Exception as e:
                st.caption(f"Financial details loaded. (Chart compilation skipped: {e})")
                st.dataframe(income_stmt)
    except Exception as e:
        st.error(f"Failed to load fundamentals: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Technical
# ═══════════════════════════════════════════════════════════════════════════════
with tab_tech:
    try:
        # ── Signal Summary ──
        signals = generate_signals(history)
        overall = signals["overall"]
        overall_cls = "badge-buy" if overall == "BUY" else ("badge-sell" if overall == "SELL" else "badge-hold")
        
        st.markdown(
            f'<div class="info-card" style="text-align:center;">'
            f'  <span style="color:#888;font-size:0.85rem;letter-spacing:1px;'
            f'  text-transform:uppercase;">Overall Technical Signal</span><br>'
            f'  <span class="badge {overall_cls}" '
            f'  style="font-size:1.3rem;padding:8px 28px;margin-top:8px;display:inline-block;">'
            f'  {overall}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown("")

        # Grid of indicators
        rsi_val = 50.0
        if "RSI" in history.columns:
            rsi_series = history["RSI"].dropna()
            if not rsi_series.empty:
                rsi_val = float(rsi_series.iloc[-1])

        macd_val = 0.0
        if "MACD" in history.columns:
            macd_series = history["MACD"].dropna()
            if not macd_series.empty:
                macd_val = float(macd_series.iloc[-1])
        
        rsi_status = "OVERSOLD" if rsi_val < 30 else ("OVERBOUGHT" if rsi_val > 70 else "NEUTRAL")
        macd_state = "BULLISH" if macd_val > 0 else "BEARISH"
        macd_state_cls = "badge-buy" if macd_val > 0 else "badge-sell"
        
        # Support & Resistance levels
        sr_levels = get_support_resistance(history)
        sup_str = " / ".join(f"₹{v:,.0f}" for v in sr_levels.get("support", [])) or "N/A"
        res_str = " / ".join(f"₹{v:,.0f}" for v in sr_levels.get("resistance", [])) or "N/A"

        sig1, sig2, sig3 = st.columns(3)
        sig1.markdown(
            f'<div class="signal-box">'
            f'  <div class="signal-label">RSI (14)</div>'
            f'  <div class="signal-value" style="color:#E8E8E8;">{rsi_val:.1f}</div>'
            f'  <div style="margin-top:8px;">'
            f'    <span class="badge badge-hold">{rsi_status}</span>'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        
        sig2.markdown(
            f'<div class="signal-box">'
            f'  <div class="signal-label">MACD Line</div>'
            f'  <div class="signal-value" style="color:#E8E8E8;">{macd_val:.2f}</div>'
            f'  <div style="margin-top:8px;">'
            f'    <span class="badge {macd_state_cls}">{macd_state}</span>'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        
        sig3.markdown(
            f'<div class="signal-box">'
            f'  <div class="signal-label">Support / Resistance</div>'
            f'  <div class="signal-value" style="color:#E8E8E8;font-size:0.95rem;padding:6px 0;">'
            f'    S: {sup_str}<br>R: {res_str}'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # ── Individual Signals Table ──
        st.markdown("##### 🔧 Individual Technical Signals")
        df_signals = pd.DataFrame(signals.get("signals", []))
        if not df_signals.empty:
            df_signals["Outlook"] = df_signals["bullish"].map({True: "🟢 Bullish", False: "🔴 Bearish"})
            df_display = df_signals[["indicator", "signal", "value", "Outlook"]].rename(columns={
                "indicator": "Indicator",
                "signal": "Signal Details",
                "value": "Current Value"
            })
            st.dataframe(df_display, use_container_width=True, hide_index=True)
        else:
            st.info("No individual signals generated.")

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        tech_left, tech_right = st.columns(2)

        # ── RSI Gauge ──
        with tech_left:
            st.markdown("##### 📉 RSI (14)")
            fig_rsi = go.Figure(go.Indicator(
                mode="gauge+number",
                value=rsi_val,
                number=dict(font=dict(color="#E8E8E8")),
                gauge=dict(
                    axis=dict(range=[0, 100], tickcolor="#555"),
                    bar=dict(color="#6C63FF"),
                    bgcolor="#1A1F2E",
                    bordercolor="rgba(108,99,255,0.3)",
                    steps=[
                        dict(range=[0, 30], color="rgba(0,212,170,0.2)"),
                        dict(range=[30, 70], color="rgba(108,99,255,0.08)"),
                        dict(range=[70, 100], color="rgba(255,107,107,0.2)"),
                    ],
                ),
            ))
            fig_rsi.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0E1117",
                plot_bgcolor="#0E1117",
                height=280,
                margin=dict(l=30, r=30, t=30, b=10),
            )
            st.plotly_chart(fig_rsi, use_container_width=True)

        # ── MACD Chart ──
        with tech_right:
            st.markdown("##### 📊 MACD Chart")
            if "MACD" in history.columns and "MACD_Signal" in history.columns:
                df_macd = history.dropna(subset=["MACD", "MACD_Signal"]).tail(120)
                if not df_macd.empty:
                    fig_macd = go.Figure()
                    fig_macd.add_trace(go.Scatter(
                        x=df_macd.index, y=df_macd["MACD"],
                        line=dict(color="#6C63FF", width=2), name="MACD",
                    ))
                    fig_macd.add_trace(go.Scatter(
                        x=df_macd.index, y=df_macd["MACD_Signal"],
                        line=dict(color="#FF6B6B", width=1.5, dash="dash"), name="Signal",
                    ))
                    if "MACD_Hist" in df_macd.columns:
                        hist_colors = [
                            "rgba(0,212,170,0.6)" if v >= 0 else "rgba(255,107,107,0.6)"
                            for v in df_macd["MACD_Hist"]
                        ]
                        fig_macd.add_trace(go.Bar(
                            x=df_macd.index, y=df_macd["MACD_Hist"],
                            marker_color=hist_colors, name="Histogram",
                        ))
                    fig_macd.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="#0E1117",
                        plot_bgcolor="#0E1117",
                        height=280,
                        margin=dict(l=0, r=0, t=10, b=0),
                        yaxis=dict(gridcolor="rgba(108,99,255,0.08)"),
                        xaxis=dict(gridcolor="rgba(108,99,255,0.08)"),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
                        showlegend=True,
                    )
                    st.plotly_chart(fig_macd, use_container_width=True)
                else:
                    st.caption("Insufficient historical data to render MACD detail.")
            else:
                st.caption("MACD indicators not computed.")

        # ── Bollinger Bands ──
        if "BB_Upper" in history.columns and "BB_Lower" in history.columns:
            st.markdown("##### 🎯 Bollinger Bands (20, 2)")
            df_bb = history.dropna(subset=["BB_Upper", "BB_Lower"]).tail(120)
            
            if not df_bb.empty:
                fig_bb = go.Figure()
                fig_bb.add_trace(go.Scatter(
                    x=df_bb.index, y=df_bb["BB_Upper"],
                    line=dict(color="rgba(108,99,255,0.4)", width=1),
                    name="Upper Band", showlegend=True,
                ))
                fig_bb.add_trace(go.Scatter(
                    x=df_bb.index, y=df_bb["BB_Lower"],
                    line=dict(color="rgba(108,99,255,0.4)", width=1),
                    fill="tonexty", fillcolor="rgba(108,99,255,0.06)",
                    name="Lower Band", showlegend=True,
                ))
                if "BB_Middle" in df_bb.columns:
                    fig_bb.add_trace(go.Scatter(
                        x=df_bb.index, y=df_bb["BB_Middle"],
                        line=dict(color="#FFB347", width=1, dash="dash"),
                        name="Mid Band (SMA20)",
                    ))
                fig_bb.add_trace(go.Scatter(
                    x=df_bb.index, y=df_bb["Close"],
                    line=dict(color="#00D4AA", width=2),
                    name="Close Price",
                ))

                fig_bb.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0E1117",
                    plot_bgcolor="#0E1117",
                    height=380,
                    margin=dict(l=0, r=0, t=10, b=0),
                    yaxis=dict(title="Price (₹)", gridcolor="rgba(108,99,255,0.08)"),
                    xaxis=dict(gridcolor="rgba(108,99,255,0.08)"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
                )
                st.plotly_chart(fig_bb, use_container_width=True)
    except Exception as e:
        st.error(f"Failed to load technical analysis: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — News Sentiment
# ═══════════════════════════════════════════════════════════════════════════════
with tab_news:
    try:
        @st.cache_data(ttl=3600)
        def load_news(symbol):
            articles = fetch_news(symbol, newsapi_key=NEWSAPI_KEY)
            # Try to add sentiment if model available
            try:
                from models.sentiment.sentiment_analyzer import get_sentiment_analyzer
                analyzer = get_sentiment_analyzer()
                if analyzer.available:
                    articles = analyzer.analyze_news(articles)
            except Exception:
                pass  # show neutral if model fails
            return articles

        articles = load_news(symbol)
        
        # Calculate distribution
        distribution = {"Positive": 0, "Neutral": 0, "Negative": 0}
        for a in articles:
            s = a.get("sentiment", "Neutral")
            s_capital = s.capitalize()
            if s_capital in distribution:
                distribution[s_capital] += 1
            else:
                distribution["Neutral"] += 1

        news_col, chart_col = st.columns([1.3, 1])

        with news_col:
            st.markdown("##### 📰 Latest News & Sentiment Analysis")
            if not articles:
                st.info("No recent news articles found for this company.")
            else:
                for item in articles:
                    s = item.get("sentiment", "Neutral")
                    s_capital = s.capitalize()
                    badge_cls = (
                        "badge-positive" if s_capital == "Positive"
                        else ("badge-negative" if s_capital == "Negative" else "badge-neutral")
                    )
                    st.markdown(
                        f'<div class="news-item">'
                        f'  <div style="display:flex;justify-content:space-between;align-items:flex-start;">'
                        f'    <div class="news-title"><a href="{item.get("url", "#")}" target="_blank" style="text-decoration:none;color:#E8E8E8;">{item["title"]}</a></div>'
                        f'    <span class="badge {badge_cls}" style="white-space:nowrap;margin-left:12px;">{s_capital}</span>'
                        f'  </div>'
                        f'  <div class="news-meta">{item.get("source", "RSS")} &middot; {item.get("published_at", "N/A")}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        with chart_col:
            # ── Sentiment Distribution Pie ──
            st.markdown("##### 🎯 Sentiment Distribution")
            if len(articles) == 0:
                st.caption("No sentiment distribution available.")
            else:
                fig_pie = go.Figure(go.Pie(
                    labels=list(distribution.keys()),
                    values=list(distribution.values()),
                    marker=dict(colors=["#00D4AA", "#FFE66D", "#FF6B6B"]), # Pos, Neu, Neg colors
                    hole=0.55,
                    textinfo="label+percent",
                    textfont=dict(size=12),
                ))
                fig_pie.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0E1117",
                    plot_bgcolor="#0E1117",
                    height=280,
                    margin=dict(l=0, r=0, t=10, b=0),
                    showlegend=False,
                )
                st.plotly_chart(fig_pie, use_container_width=True)

        # ── Sentiment Score Summary Card ──
        try:
            from models.sentiment.sentiment_analyzer import get_sentiment_analyzer
            analyzer = get_sentiment_analyzer()
            agg_sent = analyzer.get_aggregate_sentiment(articles)
        except Exception:
            pos_count = distribution.get("Positive", 0)
            neg_count = distribution.get("Negative", 0)
            total = len(articles)
            if total > 0:
                net_score = (pos_count - neg_count) / total
                score = (net_score + 1.0) / 2.0
                if net_score >= 0.15:
                    lbl = "Positive"
                    emoji = "🟢"
                elif net_score <= -0.15:
                    lbl = "Negative"
                    emoji = "🔴"
                else:
                    lbl = "Neutral"
                    emoji = "⚪"
            else:
                score = 0.5
                lbl = "Neutral"
                emoji = "⚪"
            agg_sent = {
                "label": lbl,
                "score": score,
                "distribution": distribution,
                "emoji": emoji
            }

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown(
            f'<div class="info-card" style="text-align:center;">'
            f'  <span style="color:#888;font-size:0.85rem;letter-spacing:1px;'
            f'  text-transform:uppercase;">Overall Market Sentiment</span><br>'
            f'  <span class="badge {"badge-positive" if agg_sent["label"] == "Positive" else ("badge-negative" if agg_sent["label"] == "Negative" else "badge-neutral")}" '
            f'  style="font-size:1.3rem;padding:8px 28px;margin-top:8px;display:inline-block;">'
            f'  {agg_sent["emoji"]} {agg_sent["label"]}</span>'
            f'  <p style="color:#888;margin:8px 0 0;font-size:0.85rem;">Aggregate score: {agg_sent["score"] * 100:.1f}%</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.error(f"Failed to load news/sentiment analysis: {e}")

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown(
    f'<p style="text-align:center;color:#444;font-size:0.75rem;margin-top:40px;">'
    f'© 2026 NiveshAI — AI-Powered Investment Research &middot; '
    f'Data compiled dynamically using SQLite cache and Yahoo Finance.</p>',
    unsafe_allow_html=True,
)
