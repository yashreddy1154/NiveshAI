import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# ───────────────────────── Page Config ─────────────────────────
st.set_page_config(
    page_title="Report Generator — NiveshAI",
    page_icon="📋",
    layout="wide",
)

# ───────────────────────── Custom CSS ──────────────────────────
st.markdown("""
<style>
    /* ── Global ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stMetric {
        background: linear-gradient(135deg, rgba(108, 99, 255, 0.1), rgba(0, 212, 170, 0.05));
        border: 1px solid rgba(108, 99, 255, 0.2);
        border-radius: 12px;
        padding: 16px;
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
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(108, 99, 255, 0.1);
        border-radius: 8px;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(108, 99, 255, 0.3) !important;
    }

    /* ── Page-specific ── */
    .report-header {
        background: linear-gradient(135deg, rgba(108, 99, 255, 0.15), rgba(0, 212, 170, 0.08));
        border: 1px solid rgba(108, 99, 255, 0.25);
        border-radius: 20px;
        padding: 32px 40px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }
    .report-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(108,99,255,0.08) 0%, transparent 70%);
        border-radius: 50%;
    }
    .report-header h1 {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0 0 6px 0;
        background: linear-gradient(135deg, #6C63FF, #00D4AA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .report-header p {
        color: #8B8FA3;
        font-size: 1.05rem;
        margin: 0;
    }

    .section-card {
        background: rgba(26, 31, 46, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(108, 99, 255, 0.12);
        border-radius: 14px;
        padding: 20px 24px;
        margin: 10px 0;
    }
    .section-card h3 {
        color: #E0E0E0;
        font-weight: 700;
        margin-bottom: 12px;
        font-size: 1.1rem;
    }

    .exec-summary {
        color: #C0C4D6;
        font-size: 0.97rem;
        line-height: 1.75;
        letter-spacing: 0.01em;
    }

    .badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 700;
        margin: 4px 6px 4px 0;
        letter-spacing: 0.03em;
    }
    .badge-bullish {
        background: rgba(0, 212, 170, 0.15);
        color: #00D4AA;
        border: 1px solid rgba(0, 212, 170, 0.3);
    }
    .badge-bearish {
        background: rgba(255, 107, 107, 0.15);
        color: #FF6B6B;
        border: 1px solid rgba(255, 107, 107, 0.3);
    }
    .badge-neutral {
        background: rgba(255, 179, 71, 0.15);
        color: #FFB347;
        border: 1px solid rgba(255, 179, 71, 0.3);
    }

    .rec-card {
        background: linear-gradient(135deg, rgba(0, 212, 170, 0.10), rgba(0, 212, 170, 0.03));
        border: 2px solid rgba(0, 212, 170, 0.35);
        border-radius: 16px;
        padding: 32px;
        text-align: center;
    }
    .rec-label {
        color: #8B8FA3;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 8px;
    }
    .rec-verdict {
        font-size: 2.2rem;
        font-weight: 900;
        color: #00D4AA;
        letter-spacing: 0.04em;
    }
    .rec-confidence {
        margin-top: 10px;
        font-size: 1rem;
        color: #C0C4D6;
    }
    .rec-confidence span {
        color: #6C63FF;
        font-weight: 700;
    }

    .risk-bar-bg {
        background: rgba(255,255,255,0.06);
        border-radius: 10px;
        height: 14px;
        width: 100%;
        overflow: hidden;
    }
    .risk-bar-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.6s ease;
    }

    .news-item {
        background: rgba(26, 31, 46, 0.6);
        border: 1px solid rgba(108, 99, 255, 0.10);
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }
    .news-title {
        color: #E0E0E0;
        font-weight: 600;
        font-size: 0.93rem;
        margin-bottom: 4px;
    }
    .news-meta {
        color: #6B7080;
        font-size: 0.78rem;
    }

    .status-bar {
        background: rgba(26, 31, 46, 0.65);
        border: 1px solid rgba(108, 99, 255, 0.12);
        border-radius: 12px;
        padding: 16px 22px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 12px;
    }
    .status-bar span {
        color: #8B8FA3;
        font-size: 0.88rem;
    }
    .status-bar .value {
        color: #C0C4D6;
        font-weight: 600;
    }

    .sidebar-section-label {
        color: #8B8FA3;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin: 18px 0 6px 0;
    }

    /* Expander tweaks */
    .streamlit-expanderHeader {
        background: rgba(108, 99, 255, 0.06) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
    }
    div[data-testid="stExpander"] {
        border: 1px solid rgba(108, 99, 255, 0.12);
        border-radius: 12px;
        margin-bottom: 8px;
    }

    .download-area {
        background: linear-gradient(135deg, rgba(108, 99, 255, 0.12), rgba(0, 212, 170, 0.06));
        border: 1px dashed rgba(108, 99, 255, 0.30);
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        margin: 20px 0;
    }
    .download-area p {
        color: #8B8FA3;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)

# ───────────────────────── Mock Data ───────────────────────────
STOCKS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK",
    "LT", "AXISBANK", "BAJFINANCE", "MARUTI", "SUNPHARMA",
    "TITAN", "TATAMOTORS", "WIPRO", "ADANIENT", "POWERGRID",
]

STOCK_PRICES = {
    "RELIANCE": 2845.60, "TCS": 4125.30, "HDFCBANK": 1820.45,
    "INFY": 1654.20, "ICICIBANK": 1285.75, "HINDUNILVR": 2510.90,
    "SBIN": 842.15, "BHARTIARTL": 1620.80, "ITC": 468.35,
    "KOTAKBANK": 1945.60, "LT": 3680.25, "AXISBANK": 1195.40,
    "BAJFINANCE": 7240.50, "MARUTI": 12450.75, "SUNPHARMA": 1560.30,
    "TITAN": 3290.65, "TATAMOTORS": 945.80, "WIPRO": 462.15,
    "ADANIENT": 3150.40, "POWERGRID": 325.90,
}

FUNDAMENTALS = {
    "Market Cap": "₹19,26,540 Cr",
    "P/E Ratio": "28.4x",
    "P/B Ratio": "2.8x",
    "Dividend Yield": "0.36%",
    "ROE": "9.5%",
    "ROCE": "11.2%",
    "Debt-to-Equity": "0.41",
    "Revenue (TTM)": "₹9,74,864 Cr",
    "Net Profit (TTM)": "₹73,670 Cr",
    "EPS": "₹100.16",
    "Book Value": "₹1,016",
    "Promoter Holding": "50.33%",
    "FII Holding": "23.41%",
    "DII Holding": "14.12%",
}


def _generate_ohlc(n_days: int = 90, base: float = 2700.0):
    """Create fake OHLC data for candlestick chart."""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.today(), periods=n_days, freq="B")
    close = base + np.cumsum(np.random.randn(n_days) * 18)
    high = close + np.abs(np.random.randn(n_days) * 12)
    low = close - np.abs(np.random.randn(n_days) * 12)
    opn = close + np.random.randn(n_days) * 8
    vol = np.random.randint(5_000_000, 25_000_000, n_days)
    return pd.DataFrame({"Date": dates, "Open": opn, "High": high, "Low": low, "Close": close, "Volume": vol})


# ───────────────────────── Sidebar ─────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:10px 0 18px 0;">
        <span style="font-size:1.6rem;font-weight:800;
            background:linear-gradient(135deg,#6C63FF,#00D4AA);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            NiveshAI
        </span><br/>
        <span style="color:#8B8FA3;font-size:0.78rem;letter-spacing:0.08em;">
            REPORT GENERATOR
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="sidebar-section-label">Select Stocks</p>', unsafe_allow_html=True)
    selected_stocks = st.multiselect(
        "Stocks",
        STOCKS,
        default=["RELIANCE"],
        label_visibility="collapsed",
    )
    if not selected_stocks:
        selected_stocks = ["RELIANCE"]

    st.markdown('<p class="sidebar-section-label">Report Type</p>', unsafe_allow_html=True)
    report_type = st.radio(
        "Report Type",
        ["Quick Summary", "Detailed Analysis", "Comparison Report"],
        index=1,
        label_visibility="collapsed",
    )

    st.markdown('<p class="sidebar-section-label">Sections to Include</p>', unsafe_allow_html=True)
    sec_price = st.checkbox("Price Analysis", value=True)
    sec_fundamental = st.checkbox("Fundamental Data", value=True)
    sec_technical = st.checkbox("Technical Indicators", value=True)
    sec_news = st.checkbox("News Sentiment", value=True)
    sec_ai = st.checkbox("AI Recommendations", value=True)
    sec_risk = st.checkbox("Risk Assessment", value=True)

    st.markdown('<p class="sidebar-section-label">Time Period</p>', unsafe_allow_html=True)
    time_period = st.selectbox(
        "Time Period",
        ["1 Month", "3 Months", "6 Months", "1 Year", "2 Years"],
        index=2,
        label_visibility="collapsed",
    )

    st.markdown("---")
    generate_btn = st.button("🚀  Generate Report", use_container_width=True, type="primary")

    # small info card
    st.markdown("""
    <div style="background:rgba(108,99,255,0.06);border:1px solid rgba(108,99,255,0.15);
         border-radius:10px;padding:14px;margin-top:18px;">
        <p style="color:#8B8FA3;font-size:0.78rem;margin:0 0 4px 0;">Estimated generation time</p>
        <p style="color:#E0E0E0;font-weight:700;font-size:1rem;margin:0;">~30 seconds</p>
    </div>
    """, unsafe_allow_html=True)

# ───────────────────────── Header ──────────────────────────────
primary_stock = selected_stocks[0]
stock_price = STOCK_PRICES.get(primary_stock, 2845.60)

st.markdown(f"""
<div class="report-header">
    <h1>📋 Investment Report Generator</h1>
    <p>Generate comprehensive AI-powered investment reports in PDF format</p>
</div>
""", unsafe_allow_html=True)

# ── Status bar ──
st.markdown(f"""
<div class="status-bar">
    <span>📊 Report Type: <span class="value">{report_type}</span></span>
    <span>🏢 Stock{'s' if len(selected_stocks) > 1 else ''}: <span class="value">{', '.join(selected_stocks)}</span></span>
    <span>📅 Period: <span class="value">{time_period}</span></span>
    <span>🕒 Last report generated: <span class="value">July 8, 2026 at 3:45 PM</span></span>
</div>
""", unsafe_allow_html=True)

st.markdown("")

# ───────────────────────── Report Preview ──────────────────────
st.markdown(f"""
<div class="section-card" style="border-left:3px solid #6C63FF;">
    <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;">
        <div>
            <h3 style="margin:0;color:#E0E0E0;">📄 Report Preview — {primary_stock}</h3>
            <p style="color:#6B7080;font-size:0.82rem;margin:4px 0 0 0;">
                Preview of the PDF that will be generated  •  {report_type}  •  {time_period}
            </p>
        </div>
        <span class="badge badge-bullish">READY TO GENERATE</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── 1. Executive Summary ──
with st.expander("📌  Executive Summary", expanded=True):
    st.markdown(f"""
    <div class="exec-summary">
        <p><strong>{primary_stock}</strong> has demonstrated resilient performance over the past {time_period.lower()},
        maintaining its position as one of the most significant constituents of the NIFTY 50 index.
        The stock is currently trading at <strong>₹{stock_price:,.2f}</strong>, reflecting a
        <span style="color:#00D4AA;font-weight:600;">+8.4%</span> gain from its period low.
        Revenue growth has been steady at 12.3% YoY, driven primarily by strength in the digital
        services and retail segments, while the traditional refining business has shown moderate
        margin improvements.</p>

        <p>From a technical standpoint, the stock has been consolidating within a ₹2,680 – ₹2,920
        range, with the 50-day moving average providing consistent support. Volume patterns suggest
        accumulation by institutional investors, with FII holdings increasing by 0.8% in the most
        recent quarter. The RSI at 54.2 indicates neither overbought nor oversold conditions,
        providing room for upside movement if broader market sentiment remains constructive.</p>

        <p>Our AI-driven analysis, which integrates fundamental valuation models, technical pattern
        recognition, and NLP-based news sentiment scoring, assigns a <strong style="color:#00D4AA;">
        MODERATE BUY</strong> rating with a confidence score of <strong style="color:#6C63FF;">72%</strong>.
        Key upside catalysts include the upcoming Jio tariff revisions and expansion of the retail
        footprint, while downside risks are primarily linked to global crude oil volatility and
        regulatory changes in the telecom sector.</p>
    </div>
    """, unsafe_allow_html=True)

# ── 2. Price Analysis ──
if sec_price:
    with st.expander("📈  Price Analysis", expanded=True):
        period_map = {"1 Month": 22, "3 Months": 65, "6 Months": 130, "1 Year": 252, "2 Years": 504}
        n_days = period_map.get(time_period, 130)
        df_ohlc = _generate_ohlc(n_days=n_days, base=stock_price - 150)

        fig_candle = go.Figure()
        fig_candle.add_trace(go.Candlestick(
            x=df_ohlc["Date"],
            open=df_ohlc["Open"], high=df_ohlc["High"],
            low=df_ohlc["Low"], close=df_ohlc["Close"],
            increasing_line_color="#00D4AA",
            decreasing_line_color="#FF6B6B",
            name="Price",
        ))
        # 20-day SMA
        sma20 = df_ohlc["Close"].rolling(20).mean()
        fig_candle.add_trace(go.Scatter(
            x=df_ohlc["Date"], y=sma20,
            line=dict(color="#6C63FF", width=1.5),
            name="SMA 20",
        ))
        # 50-day SMA
        sma50 = df_ohlc["Close"].rolling(50).mean()
        fig_candle.add_trace(go.Scatter(
            x=df_ohlc["Date"], y=sma50,
            line=dict(color="#FFB347", width=1.5),
            name="SMA 50",
        ))
        fig_candle.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=420,
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                        font=dict(size=11)),
            yaxis=dict(title="Price (₹)", gridcolor="rgba(255,255,255,0.04)"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
        )
        st.plotly_chart(fig_candle, use_container_width=True)

        # Volume bar
        fig_vol = go.Figure()
        colors_vol = ["#00D4AA" if c >= o else "#FF6B6B" for c, o in zip(df_ohlc["Close"], df_ohlc["Open"])]
        fig_vol.add_trace(go.Bar(
            x=df_ohlc["Date"], y=df_ohlc["Volume"],
            marker_color=colors_vol, opacity=0.6, name="Volume",
        ))
        fig_vol.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=140,
            margin=dict(l=0, r=0, t=0, b=0),
            yaxis=dict(title="Volume", gridcolor="rgba(255,255,255,0.04)"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.04)", showticklabels=False),
            showlegend=False,
        )
        st.plotly_chart(fig_vol, use_container_width=True)

        # Quick stats
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current Price", f"₹{stock_price:,.2f}", "+2.35%")
        c2.metric("Period High", f"₹{stock_price * 1.08:,.2f}")
        c3.metric("Period Low", f"₹{stock_price * 0.92:,.2f}")
        c4.metric("Avg Volume", "1.42 Cr")

# ── 3. Fundamental Metrics ──
if sec_fundamental:
    with st.expander("🏦  Fundamental Metrics", expanded=True):
        col_a, col_b = st.columns(2)
        items = list(FUNDAMENTALS.items())
        half = len(items) // 2
        with col_a:
            df_left = pd.DataFrame(items[:half], columns=["Metric", "Value"])
            st.dataframe(
                df_left,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Metric": st.column_config.TextColumn("Metric", width="medium"),
                    "Value": st.column_config.TextColumn("Value", width="medium"),
                },
            )
        with col_b:
            df_right = pd.DataFrame(items[half:], columns=["Metric", "Value"])
            st.dataframe(
                df_right,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Metric": st.column_config.TextColumn("Metric", width="medium"),
                    "Value": st.column_config.TextColumn("Value", width="medium"),
                },
            )

        # Peer comparison mini-chart
        st.markdown("**Peer P/E Comparison**")
        peer_pe = pd.DataFrame({
            "Company": ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ITC"],
            "P/E Ratio": [28.4, 32.1, 27.6, 21.3, 25.8],
        })
        fig_pe = px.bar(
            peer_pe, x="Company", y="P/E Ratio",
            color="P/E Ratio",
            color_continuous_scale=["#6C63FF", "#00D4AA"],
            text="P/E Ratio",
        )
        fig_pe.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=260,
            margin=dict(l=0, r=0, t=10, b=0),
            showlegend=False,
            coloraxis_showscale=False,
            yaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
        )
        fig_pe.update_traces(textposition="outside", texttemplate="%{text:.1f}x")
        st.plotly_chart(fig_pe, use_container_width=True)

# ── 4. Technical Signals ──
if sec_technical:
    with st.expander("📊  Technical Signals", expanded=True):
        st.markdown("""
        <div style="display:flex;flex-wrap:wrap;gap:10px;margin-bottom:18px;">
            <div style="flex:1;min-width:140px;" class="section-card">
                <p style="color:#6B7080;font-size:0.75rem;margin:0;">RSI (14)</p>
                <p style="color:#E0E0E0;font-size:1.3rem;font-weight:700;margin:4px 0;">54.2</p>
                <span class="badge badge-neutral">NEUTRAL</span>
            </div>
            <div style="flex:1;min-width:140px;" class="section-card">
                <p style="color:#6B7080;font-size:0.75rem;margin:0;">MACD</p>
                <p style="color:#E0E0E0;font-size:1.3rem;font-weight:700;margin:4px 0;">+12.6</p>
                <span class="badge badge-bullish">BULLISH</span>
            </div>
            <div style="flex:1;min-width:140px;" class="section-card">
                <p style="color:#6B7080;font-size:0.75rem;margin:0;">SMA 20/50</p>
                <p style="color:#E0E0E0;font-size:1.3rem;font-weight:700;margin:4px 0;">Above</p>
                <span class="badge badge-bullish">BUY</span>
            </div>
            <div style="flex:1;min-width:140px;" class="section-card">
                <p style="color:#6B7080;font-size:0.75rem;margin:0;">Bollinger Bands</p>
                <p style="color:#E0E0E0;font-size:1.3rem;font-weight:700;margin:4px 0;">Mid</p>
                <span class="badge badge-neutral">NEUTRAL</span>
            </div>
            <div style="flex:1;min-width:140px;" class="section-card">
                <p style="color:#6B7080;font-size:0.75rem;margin:0;">ADX</p>
                <p style="color:#E0E0E0;font-size:1.3rem;font-weight:700;margin:4px 0;">22.8</p>
                <span class="badge badge-bearish">WEAK TREND</span>
            </div>
            <div style="flex:1;min-width:140px;" class="section-card">
                <p style="color:#6B7080;font-size:0.75rem;margin:0;">VWAP</p>
                <p style="color:#E0E0E0;font-size:1.3rem;font-weight:700;margin:4px 0;">₹2,838</p>
                <span class="badge badge-bullish">ABOVE</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Technical summary
        st.markdown("""
        <div class="section-card" style="border-left:3px solid #00D4AA;">
            <h3 style="margin-top:0;font-size:0.95rem;">⚡ Technical Verdict</h3>
            <p style="color:#C0C4D6;font-size:0.9rem;line-height:1.6;">
                Overall technical analysis suggests a <strong style="color:#00D4AA;">mildly bullish</strong>
                outlook. MACD crossover and price position above both SMAs support near-term upside.
                However, the weak ADX reading indicates the trend lacks strong conviction.
                <strong>Key support:</strong> ₹2,680 &nbsp;|&nbsp; <strong>Key resistance:</strong> ₹2,920
            </p>
        </div>
        """, unsafe_allow_html=True)

# ── 5. News Sentiment ──
if sec_news:
    with st.expander("📰  News Sentiment", expanded=True):
        col_gauge, col_news = st.columns([1, 2])

        with col_gauge:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=68,
                title={"text": "Sentiment Score", "font": {"size": 14, "color": "#8B8FA3"}},
                number={"suffix": "/100", "font": {"size": 28, "color": "#E0E0E0"}},
                gauge=dict(
                    axis=dict(range=[0, 100], tickwidth=1, tickcolor="#2a2a2a"),
                    bar=dict(color="#6C63FF"),
                    bgcolor="rgba(26,31,46,0.5)",
                    borderwidth=0,
                    steps=[
                        dict(range=[0, 33], color="rgba(255,107,107,0.15)"),
                        dict(range=[33, 66], color="rgba(255,179,71,0.15)"),
                        dict(range=[66, 100], color="rgba(0,212,170,0.15)"),
                    ],
                    threshold=dict(line=dict(color="#00D4AA", width=3), thickness=0.8, value=68),
                ),
            ))
            fig_gauge.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=250,
                margin=dict(l=20, r=20, t=40, b=10),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
            st.markdown("""
            <div style="text-align:center;">
                <span class="badge badge-bullish" style="font-size:0.9rem;padding:8px 20px;">
                    MODERATELY POSITIVE
                </span>
            </div>
            """, unsafe_allow_html=True)

        with col_news:
            news_items = [
                {
                    "title": f"{primary_stock} Q1 results beat Street estimates; net profit rises 18% YoY",
                    "source": "Moneycontrol",
                    "time": "2 hours ago",
                    "sentiment": "Positive",
                    "badge_cls": "badge-bullish",
                },
                {
                    "title": f"Jio Platforms to increase tariffs by 12-15% effective August 2026",
                    "source": "Economic Times",
                    "time": "5 hours ago",
                    "sentiment": "Positive",
                    "badge_cls": "badge-bullish",
                },
                {
                    "title": f"Global crude oil prices rise 3% on OPEC+ supply cut concerns",
                    "source": "LiveMint",
                    "time": "1 day ago",
                    "sentiment": "Negative",
                    "badge_cls": "badge-bearish",
                },
            ]
            st.markdown("**Top Headlines**")
            for item in news_items:
                st.markdown(f"""
                <div class="news-item">
                    <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                        <div>
                            <p class="news-title">{item['title']}</p>
                            <p class="news-meta">{item['source']} • {item['time']}</p>
                        </div>
                        <span class="badge {item['badge_cls']}" style="white-space:nowrap;">{item['sentiment']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ── 6. AI Recommendation ──
if sec_ai:
    with st.expander("🤖  AI Recommendation", expanded=True):
        col_main, col_detail = st.columns([1, 1])

        with col_main:
            st.markdown("""
            <div class="rec-card">
                <p class="rec-label">AI-Powered Recommendation</p>
                <p class="rec-verdict">MODERATE BUY</p>
                <p class="rec-confidence">Confidence Score: <span>72%</span></p>
                <div style="margin-top:16px;">
                    <div style="background:rgba(255,255,255,0.06);border-radius:10px;
                         height:10px;width:100%;overflow:hidden;">
                        <div style="width:72%;height:100%;border-radius:10px;
                             background:linear-gradient(90deg,#6C63FF,#00D4AA);"></div>
                    </div>
                </div>
                <p style="color:#6B7080;font-size:0.78rem;margin-top:14px;">
                    Based on analysis of 47 fundamental, technical &amp; sentiment factors
                </p>
            </div>
            """, unsafe_allow_html=True)

        with col_detail:
            st.markdown("""
            <div class="section-card" style="height:100%;">
                <h3 style="margin-top:0;font-size:0.95rem;">Signal Breakdown</h3>
                <table style="width:100%;border-collapse:collapse;">
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                        <td style="padding:8px 0;color:#8B8FA3;">Fundamental Score</td>
                        <td style="padding:8px 0;text-align:right;color:#00D4AA;font-weight:700;">78/100</td>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                        <td style="padding:8px 0;color:#8B8FA3;">Technical Score</td>
                        <td style="padding:8px 0;text-align:right;color:#00D4AA;font-weight:700;">65/100</td>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                        <td style="padding:8px 0;color:#8B8FA3;">Sentiment Score</td>
                        <td style="padding:8px 0;text-align:right;color:#FFB347;font-weight:700;">68/100</td>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                        <td style="padding:8px 0;color:#8B8FA3;">Valuation Score</td>
                        <td style="padding:8px 0;text-align:right;color:#00D4AA;font-weight:700;">71/100</td>
                    </tr>
                    <tr>
                        <td style="padding:8px 0;color:#8B8FA3;">Risk-Adjusted Return</td>
                        <td style="padding:8px 0;text-align:right;color:#00D4AA;font-weight:700;">74/100</td>
                    </tr>
                </table>
                <div style="margin-top:14px;padding-top:12px;border-top:1px solid rgba(255,255,255,0.08);">
                    <p style="color:#8B8FA3;font-size:0.78rem;margin:0;">
                        Target Price: <strong style="color:#00D4AA;">₹3,120</strong> &nbsp;|&nbsp;
                        Stop Loss: <strong style="color:#FF6B6B;">₹2,620</strong> &nbsp;|&nbsp;
                        Horizon: <strong style="color:#E0E0E0;">6-12 months</strong>
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ── 7. Risk Assessment ──
if sec_risk:
    with st.expander("⚠️  Risk Assessment", expanded=True):
        col_score, col_factors = st.columns([1, 2])

        with col_score:
            # Risk donut
            fig_risk = go.Figure(go.Pie(
                values=[6.5, 3.5],
                labels=["Risk", "Safety"],
                hole=0.72,
                marker=dict(colors=["#FFB347", "rgba(255,255,255,0.04)"],
                            line=dict(color="rgba(0,0,0,0)", width=0)),
                textinfo="none",
                hoverinfo="skip",
            ))
            fig_risk.add_annotation(
                text="<b>6.5</b><br><span style='font-size:11px;color:#8B8FA3'>/10</span>",
                x=0.5, y=0.5, font=dict(size=32, color="#FFB347"),
                showarrow=False,
            )
            fig_risk.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=230,
                margin=dict(l=0, r=0, t=10, b=10),
                showlegend=False,
            )
            st.plotly_chart(fig_risk, use_container_width=True)
            st.markdown("""
            <div style="text-align:center;">
                <span class="badge badge-neutral" style="font-size:0.9rem;padding:8px 20px;">
                    MODERATE RISK
                </span>
            </div>
            """, unsafe_allow_html=True)

        with col_factors:
            risk_factors = [
                ("Crude Oil Price Volatility", 7.8, "#FF6B6B"),
                ("Regulatory / Policy Risk", 6.2, "#FFB347"),
                ("Market Concentration Risk", 5.5, "#FFB347"),
                ("Currency Fluctuation", 4.8, "#FFE66D"),
                ("Competitive Pressure", 6.0, "#FFB347"),
                ("Debt Servicing Risk", 3.2, "#00D4AA"),
            ]
            st.markdown("**Risk Factor Breakdown**")
            for name, score, color in risk_factors:
                pct = score * 10
                st.markdown(f"""
                <div style="margin-bottom:12px;">
                    <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                        <span style="color:#C0C4D6;font-size:0.88rem;">{name}</span>
                        <span style="color:{color};font-weight:700;font-size:0.88rem;">{score}/10</span>
                    </div>
                    <div class="risk-bar-bg">
                        <div class="risk-bar-fill" style="width:{pct}%;background:{color};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)


# ───────────────────────── Download Area ───────────────────────
st.markdown("")
st.markdown("""
<div class="download-area">
    <p style="font-size:1.15rem;color:#E0E0E0;font-weight:700;margin-bottom:6px;">
        📥 Your Report is Ready
    </p>
    <p>Click below to download the full PDF report</p>
</div>
""", unsafe_allow_html=True)

# Create mock PDF bytes
mock_report_content = f"""
{'='*60}
  NiveshAI — AI-Powered Investment Research Report
{'='*60}

  Report Type : {report_type}
  Stock(s)    : {', '.join(selected_stocks)}
  Period      : {time_period}
  Generated   : July 8, 2026 at 3:45 PM

{'─'*60}
  EXECUTIVE SUMMARY
{'─'*60}
  {primary_stock} demonstrates resilient performance with a
  MODERATE BUY recommendation (Confidence: 72%).
  Current Price: ₹{stock_price:,.2f}
  Target Price : ₹3,120.00
  Risk Score   : 6.5 / 10

{'─'*60}
  This is a mock report generated by NiveshAI.
  Actual PDF generation will include charts,
  tables, and comprehensive analysis.
{'='*60}
"""

col_dl1, col_dl2, col_dl3 = st.columns([1, 2, 1])
with col_dl2:
    st.download_button(
        label="📄  Download PDF Report",
        data=mock_report_content.encode("utf-8"),
        file_name=f"NiveshAI_{primary_stock}_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
        mime="application/pdf",
        use_container_width=True,
        type="primary",
    )

# ── Footer status ──
st.markdown("")
st.markdown(f"""
<div class="status-bar" style="margin-top:8px;">
    <span>✅ Status: <span class="value" style="color:#00D4AA;">Report generated successfully</span></span>
    <span>📄 Pages: <span class="value">12</span></span>
    <span>📊 Charts: <span class="value">6</span></span>
    <span>⏱️ Generation time: <span class="value">~30 seconds</span></span>
</div>
""", unsafe_allow_html=True)

# ── Disclaimer ──
st.markdown("""
<div style="margin-top:28px;padding:16px 20px;background:rgba(255,107,107,0.04);
     border:1px solid rgba(255,107,107,0.12);border-radius:10px;">
    <p style="color:#6B7080;font-size:0.76rem;margin:0;line-height:1.6;">
        <strong style="color:#FF6B6B;">⚠️ Disclaimer:</strong>
        This report is generated using AI models and is for informational purposes only.
        It does not constitute financial advice. Investment decisions should be made after
        consulting with a qualified financial advisor. Past performance is not indicative
        of future results. NiveshAI is not SEBI-registered.
    </p>
</div>
""", unsafe_allow_html=True)
