import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

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

# ─── Stock Data ──────────────────────────────────────────────────────────────
STOCKS = {
    "RELIANCE": "Reliance Industries Ltd.",
    "TCS": "Tata Consultancy Services Ltd.",
    "INFY": "Infosys Ltd.",
    "HDFCBANK": "HDFC Bank Ltd.",
    "ICICIBANK": "ICICI Bank Ltd.",
    "BHARTIARTL": "Bharti Airtel Ltd.",
    "SBIN": "State Bank of India",
    "ITC": "ITC Ltd.",
    "KOTAKBANK": "Kotak Mahindra Bank Ltd.",
    "LT": "Larsen & Toubro Ltd.",
    "AXISBANK": "Axis Bank Ltd.",
    "WIPRO": "Wipro Ltd.",
    "BAJFINANCE": "Bajaj Finance Ltd.",
    "HCLTECH": "HCL Technologies Ltd.",
    "MARUTI": "Maruti Suzuki India Ltd.",
    "SUNPHARMA": "Sun Pharmaceutical Industries Ltd.",
    "TATAMOTORS": "Tata Motors Ltd.",
    "ONGC": "Oil & Natural Gas Corporation Ltd.",
    "NTPC": "NTPC Ltd.",
    "POWERGRID": "Power Grid Corporation of India Ltd.",
}

SECTORS = {
    "RELIANCE": "Oil & Gas / Conglomerate",
    "TCS": "Information Technology",
    "INFY": "Information Technology",
    "HDFCBANK": "Banking & Financial Services",
    "ICICIBANK": "Banking & Financial Services",
    "BHARTIARTL": "Telecommunications",
    "SBIN": "Banking & Financial Services",
    "ITC": "FMCG / Conglomerate",
    "KOTAKBANK": "Banking & Financial Services",
    "LT": "Engineering & Construction",
    "AXISBANK": "Banking & Financial Services",
    "WIPRO": "Information Technology",
    "BAJFINANCE": "Financial Services — NBFC",
    "HCLTECH": "Information Technology",
    "MARUTI": "Automobile",
    "SUNPHARMA": "Pharmaceuticals",
    "TATAMOTORS": "Automobile",
    "ONGC": "Oil & Gas",
    "NTPC": "Power & Utilities",
    "POWERGRID": "Power & Utilities",
}

BASE_PRICES = {
    "RELIANCE": 2650, "TCS": 3800, "INFY": 1600, "HDFCBANK": 1700,
    "ICICIBANK": 1150, "BHARTIARTL": 1550, "SBIN": 820, "ITC": 470,
    "KOTAKBANK": 1820, "LT": 3450, "AXISBANK": 1180, "WIPRO": 480,
    "BAJFINANCE": 7200, "HCLTECH": 1520, "MARUTI": 12500, "SUNPHARMA": 1680,
    "TATAMOTORS": 950, "ONGC": 265, "NTPC": 370, "POWERGRID": 310,
}


# ─── Data Generators ─────────────────────────────────────────────────────────
def generate_ohlcv(symbol: str, days: int = 252) -> pd.DataFrame:
    """Generate realistic OHLCV data using a geometric random walk."""
    seed = sum(ord(c) for c in symbol)
    rng = np.random.RandomState(seed)
    base = BASE_PRICES.get(symbol, 2500)

    # Random walk for close prices
    returns = rng.normal(loc=0.0004, scale=0.018, size=days)
    close = base * np.cumprod(1 + returns)

    # Derive OHLV from close
    high = close * (1 + rng.uniform(0.002, 0.025, size=days))
    low = close * (1 - rng.uniform(0.002, 0.025, size=days))
    open_ = low + (high - low) * rng.uniform(0.2, 0.8, size=days)
    volume = rng.randint(3_000_000, 25_000_000, size=days).astype(float)
    # Spike volume on big moves
    volume *= 1 + 3 * np.abs(returns)

    dates = pd.bdate_range(end=datetime.today(), periods=days)
    return pd.DataFrame({
        "Date": dates, "Open": open_, "High": high,
        "Low": low, "Close": close, "Volume": volume,
    })


def compute_sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window).mean()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def compute_macd(series: pd.Series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger(series: pd.Series, window: int = 20, num_std: float = 2.0):
    mid = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


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

    stock_options = [f"{sym} — {name}" for sym, name in STOCKS.items()]
    selected_option = st.selectbox("🔍 Select Stock", stock_options, index=0)
    selected_symbol = selected_option.split(" — ")[0]

    st.markdown("")  # spacer
    today = datetime.today().date()
    default_start = today - timedelta(days=365)
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        start_date = st.date_input("From", value=default_start)
    with col_d2:
        end_date = st.date_input("To", value=today)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#555;font-size:0.75rem;text-align:center;">'
        '⚡ Data is simulated for demo purposes</p>',
        unsafe_allow_html=True,
    )

# ─── Generate Data ───────────────────────────────────────────────────────────
days_delta = (end_date - start_date).days
num_trading_days = max(int(days_delta * 252 / 365), 30)
df = generate_ohlcv(selected_symbol, days=num_trading_days)

# Trim to date range
df = df[(df["Date"].dt.date >= start_date) & (df["Date"].dt.date <= end_date)].copy()
df.reset_index(drop=True, inplace=True)

# Compute indicators
df["SMA20"] = compute_sma(df["Close"], 20)
df["SMA50"] = compute_sma(df["Close"], 50)
df["RSI"] = compute_rsi(df["Close"])
df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = compute_macd(df["Close"])
df["BB_Upper"], df["BB_Mid"], df["BB_Lower"] = compute_bollinger(df["Close"])

latest = df.iloc[-1]
prev = df.iloc[-2] if len(df) > 1 else df.iloc[-1]
price_change = latest["Close"] - prev["Close"]
price_change_pct = (price_change / prev["Close"]) * 100

# ─── Title & Summary Bar ────────────────────────────────────────────────────
st.markdown(
    f'<h1 style="margin-bottom:0;">'
    f'<span class="gradient-text">📊 Stock Dashboard</span></h1>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<p style="color:#888;margin-top:0;">Analyzing '
    f'<b style="color:#6C63FF">{selected_symbol}</b> — '
    f'{STOCKS[selected_symbol]}</p>',
    unsafe_allow_html=True,
)

# Quick price strip
change_color = "#00D4AA" if price_change >= 0 else "#FF6B6B"
arrow = "▲" if price_change >= 0 else "▼"
st.markdown(
    f'<div class="info-card" style="display:flex;align-items:center;gap:32px;">'
    f'  <div>'
    f'    <span style="color:#888;font-size:0.85rem;">Current Price</span><br>'
    f'    <span style="font-size:2rem;font-weight:800;color:#E8E8E8;">'
    f'      ₹{latest["Close"]:,.2f}</span>'
    f'  </div>'
    f'  <div>'
    f'    <span style="color:{change_color};font-size:1.1rem;font-weight:700;">'
    f'      {arrow} ₹{abs(price_change):,.2f} ({price_change_pct:+.2f}%)</span>'
    f'  </div>'
    f'  <div style="margin-left:auto;text-align:right;">'
    f'    <span style="color:#888;font-size:0.8rem;">Volume</span><br>'
    f'    <span style="color:#C0C0C0;font-weight:600;">'
    f'      {latest["Volume"]/1e6:,.1f}M</span>'
    f'  </div>'
    f'  <div style="text-align:right;">'
    f'    <span style="color:#888;font-size:0.8rem;">Day Range</span><br>'
    f'    <span style="color:#C0C0C0;font-weight:600;">'
    f'      ₹{latest["Low"]:,.0f} — ₹{latest["High"]:,.0f}</span>'
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
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.75, 0.25],
        subplot_titles=("", ""),
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df["Date"], open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        increasing_line_color="#00D4AA", increasing_fillcolor="rgba(0,212,170,0.35)",
        decreasing_line_color="#FF6B6B", decreasing_fillcolor="rgba(255,107,107,0.35)",
        name="Price",
    ), row=1, col=1)

    # SMA overlays
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["SMA20"],
        line=dict(color="#6C63FF", width=1.5, dash="dot"),
        name="SMA 20",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["SMA50"],
        line=dict(color="#FFB347", width=1.5, dash="dot"),
        name="SMA 50",
    ), row=1, col=1)

    # Volume bars
    vol_colors = [
        "rgba(0,212,170,0.5)" if df["Close"].iloc[i] >= df["Open"].iloc[i]
        else "rgba(255,107,107,0.5)"
        for i in range(len(df))
    ]
    fig.add_trace(go.Bar(
        x=df["Date"], y=df["Volume"],
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
    c4.metric("Close", f"₹{latest['Close']:,.2f}", f"{price_change_pct:+.2f}%")
    c5.metric("Volume", f"{latest['Volume']/1e6:,.1f}M")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Fundamentals
# ═══════════════════════════════════════════════════════════════════════════════
with tab_fund:
    # ── Company Info Card ──
    st.markdown(
        f'<div class="info-card">'
        f'  <div style="display:flex;justify-content:space-between;align-items:center;">'
        f'    <div>'
        f'      <h3 style="margin:0;color:#E8E8E8;">{STOCKS[selected_symbol]}</h3>'
        f'      <p style="margin:4px 0 0;color:#6C63FF;font-size:0.9rem;">'
        f'        NSE: {selected_symbol}</p>'
        f'    </div>'
        f'    <div style="text-align:right;">'
        f'      <span style="color:#888;font-size:0.82rem;">Sector</span><br>'
        f'      <span style="color:#4ECDC4;font-weight:600;">'
        f'        {SECTORS[selected_symbol]}</span>'
        f'    </div>'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown("")

    # ── Key Metrics ──
    rng_fund = np.random.RandomState(sum(ord(c) for c in selected_symbol) + 42)
    mcap = rng_fund.uniform(5, 20)
    pe = rng_fund.uniform(18, 45)
    eps = rng_fund.uniform(30, 120)
    div_yield = rng_fund.uniform(0.3, 2.5)
    high_52w = latest["Close"] * rng_fund.uniform(1.08, 1.20)
    low_52w = latest["Close"] * rng_fund.uniform(0.70, 0.88)

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Market Cap", f"₹{mcap:.1f}L Cr")
    m2.metric("P/E Ratio", f"{pe:.1f}")
    m3.metric("EPS", f"₹{eps:.1f}")
    m4.metric("Div. Yield", f"{div_yield:.1f}%")
    m5.metric("52W High", f"₹{high_52w:,.0f}")
    m6.metric("52W Low", f"₹{low_52w:,.0f}")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    col_rev, col_bs = st.columns([1.2, 1])

    # ── Revenue & Profit Chart ──
    with col_rev:
        st.markdown("##### 📊 Quarterly Revenue & Net Profit")
        quarters = ["Q1 FY25", "Q2 FY25", "Q3 FY25", "Q4 FY25"]
        revenue = rng_fund.uniform(18000, 28000, size=4)
        profit = revenue * rng_fund.uniform(0.10, 0.22, size=4)

        fig_rev = go.Figure()
        fig_rev.add_trace(go.Bar(
            x=quarters, y=revenue,
            name="Revenue",
            marker_color="rgba(108,99,255,0.7)",
            text=[f"₹{v/1000:.1f}K Cr" for v in revenue],
            textposition="outside",
        ))
        fig_rev.add_trace(go.Bar(
            x=quarters, y=profit,
            name="Net Profit",
            marker_color="rgba(0,212,170,0.7)",
            text=[f"₹{v/1000:.1f}K Cr" for v in profit],
            textposition="outside",
        ))
        fig_rev.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            height=350,
            margin=dict(l=0, r=0, t=30, b=0),
            barmode="group",
            yaxis=dict(title="₹ Crores", gridcolor="rgba(108,99,255,0.08)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
        )
        st.plotly_chart(fig_rev, use_container_width=True)

    # ── Balance Sheet Highlights ──
    with col_bs:
        st.markdown("##### 🏦 Balance Sheet Highlights")
        bs_data = {
            "Item": [
                "Total Assets",
                "Total Liabilities",
                "Shareholders' Equity",
                "Total Debt",
                "Cash & Equivalents",
                "Net Working Capital",
            ],
            "Value (₹ Cr)": [
                f"{rng_fund.uniform(400000, 800000):,.0f}",
                f"{rng_fund.uniform(200000, 500000):,.0f}",
                f"{rng_fund.uniform(150000, 350000):,.0f}",
                f"{rng_fund.uniform(50000, 180000):,.0f}",
                f"{rng_fund.uniform(20000, 80000):,.0f}",
                f"{rng_fund.uniform(10000, 60000):,.0f}",
            ],
            "YoY Change": [
                f"+{rng_fund.uniform(3, 15):.1f}%",
                f"+{rng_fund.uniform(1, 10):.1f}%",
                f"+{rng_fund.uniform(5, 18):.1f}%",
                f"{rng_fund.choice([-1,1]) * rng_fund.uniform(2, 12):.1f}%",
                f"+{rng_fund.uniform(5, 25):.1f}%",
                f"+{rng_fund.uniform(2, 20):.1f}%",
            ],
        }
        st.dataframe(
            pd.DataFrame(bs_data),
            use_container_width=True,
            hide_index=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Technical
# ═══════════════════════════════════════════════════════════════════════════════
with tab_tech:
    # ── Signal Summary ──
    rsi_latest = df["RSI"].dropna().iloc[-1] if df["RSI"].dropna().shape[0] > 0 else 55
    macd_latest = df["MACD"].dropna().iloc[-1] if df["MACD"].dropna().shape[0] > 0 else 0
    macd_sig_latest = df["MACD_Signal"].dropna().iloc[-1] if df["MACD_Signal"].dropna().shape[0] > 0 else 0
    sma20_latest = df["SMA20"].dropna().iloc[-1] if df["SMA20"].dropna().shape[0] > 0 else latest["Close"]
    sma50_latest = df["SMA50"].dropna().iloc[-1] if df["SMA50"].dropna().shape[0] > 0 else latest["Close"]

    # Determine signals
    rsi_signal = "Sell" if rsi_latest > 70 else ("Buy" if rsi_latest < 30 else "Hold")
    macd_signal = "Buy" if macd_latest > macd_sig_latest else "Sell"
    sma_signal = "Buy" if latest["Close"] > sma50_latest else "Sell"
    # Overall
    signals = [rsi_signal, macd_signal, sma_signal]
    buy_count = signals.count("Buy")
    overall = "Buy" if buy_count >= 2 else ("Sell" if signals.count("Sell") >= 2 else "Hold")
    overall_cls = "badge-buy" if overall == "Buy" else ("badge-sell" if overall == "Sell" else "badge-hold")

    st.markdown(
        f'<div class="info-card" style="text-align:center;">'
        f'  <span style="color:#888;font-size:0.85rem;letter-spacing:1px;'
        f'  text-transform:uppercase;">Overall Signal</span><br>'
        f'  <span class="badge {overall_cls}" '
        f'  style="font-size:1.3rem;padding:8px 28px;margin-top:8px;display:inline-block;">'
        f'  {overall}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown("")

    sig1, sig2, sig3 = st.columns(3)
    for col_sig, label, value, sig in [
        (sig1, "RSI (14)", f"{rsi_latest:.1f}", rsi_signal),
        (sig2, "MACD", f"{macd_latest:.2f}", macd_signal),
        (sig3, "SMA Crossover", f"{'Above' if latest['Close'] > sma50_latest else 'Below'} SMA50", sma_signal),
    ]:
        sig_cls = "badge-buy" if sig == "Buy" else ("badge-sell" if sig == "Sell" else "badge-hold")
        col_sig.markdown(
            f'<div class="signal-box">'
            f'  <div class="signal-label">{label}</div>'
            f'  <div class="signal-value" style="color:#E8E8E8;">{value}</div>'
            f'  <div style="margin-top:8px;">'
            f'    <span class="badge {sig_cls}">{sig}</span>'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    tech_left, tech_right = st.columns(2)

    # ── RSI Gauge ──
    with tech_left:
        st.markdown("##### 📉 RSI (14)")
        fig_rsi = go.Figure(go.Indicator(
            mode="gauge+number",
            value=rsi_latest,
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
                threshold=dict(
                    line=dict(color="#FF6B6B", width=2),
                    thickness=0.8, value=rsi_latest,
                ),
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
        st.markdown("##### 📊 MACD")
        df_macd = df.dropna(subset=["MACD", "MACD_Signal", "MACD_Hist"]).tail(120)
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(
            x=df_macd["Date"], y=df_macd["MACD"],
            line=dict(color="#6C63FF", width=2), name="MACD",
        ))
        fig_macd.add_trace(go.Scatter(
            x=df_macd["Date"], y=df_macd["MACD_Signal"],
            line=dict(color="#FF6B6B", width=1.5, dash="dash"), name="Signal",
        ))
        hist_colors = [
            "rgba(0,212,170,0.6)" if v >= 0 else "rgba(255,107,107,0.6)"
            for v in df_macd["MACD_Hist"]
        ]
        fig_macd.add_trace(go.Bar(
            x=df_macd["Date"], y=df_macd["MACD_Hist"],
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

    # ── Bollinger Bands ──
    st.markdown("##### 🎯 Bollinger Bands (20, 2)")
    df_bb = df.dropna(subset=["BB_Upper", "BB_Mid", "BB_Lower"]).tail(120)
    fig_bb = go.Figure()

    # Band fill
    fig_bb.add_trace(go.Scatter(
        x=df_bb["Date"], y=df_bb["BB_Upper"],
        line=dict(color="rgba(108,99,255,0.4)", width=1),
        name="Upper Band", showlegend=True,
    ))
    fig_bb.add_trace(go.Scatter(
        x=df_bb["Date"], y=df_bb["BB_Lower"],
        line=dict(color="rgba(108,99,255,0.4)", width=1),
        fill="tonexty", fillcolor="rgba(108,99,255,0.06)",
        name="Lower Band", showlegend=True,
    ))
    fig_bb.add_trace(go.Scatter(
        x=df_bb["Date"], y=df_bb["BB_Mid"],
        line=dict(color="#FFB347", width=1, dash="dash"),
        name="Mid Band (SMA20)",
    ))
    fig_bb.add_trace(go.Scatter(
        x=df_bb["Date"], y=df_bb["Close"],
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


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — News Sentiment
# ═══════════════════════════════════════════════════════════════════════════════
with tab_news:
    rng_news = np.random.RandomState(sum(ord(c) for c in selected_symbol) + 99)

    company_short = STOCKS[selected_symbol].split(" Ltd")[0].split(" Corporation")[0]

    news_items = [
        {
            "title": f"{company_short} reports strong Q4 earnings, beats street estimates",
            "source": "Economic Times",
            "date": (today - timedelta(days=1)).strftime("%d %b %Y"),
            "sentiment": "Positive",
        },
        {
            "title": f"Analysts upgrade {selected_symbol} target price to ₹{latest['Close']*1.15:,.0f}",
            "source": "Moneycontrol",
            "date": (today - timedelta(days=3)).strftime("%d %b %Y"),
            "sentiment": "Positive",
        },
        {
            "title": f"FIIs increase stake in {company_short} during Q4",
            "source": "LiveMint",
            "date": (today - timedelta(days=5)).strftime("%d %b %Y"),
            "sentiment": "Positive",
        },
        {
            "title": f"Regulatory headwinds may impact {SECTORS[selected_symbol].split(' —')[0].split('/')[0].strip()} sector margins",
            "source": "Business Standard",
            "date": (today - timedelta(days=7)).strftime("%d %b %Y"),
            "sentiment": "Negative",
        },
        {
            "title": f"{company_short} board to consider interim dividend on upcoming meeting",
            "source": "NDTV Profit",
            "date": (today - timedelta(days=10)).strftime("%d %b %Y"),
            "sentiment": "Neutral",
        },
    ]

    news_col, chart_col = st.columns([1.3, 1])

    with news_col:
        st.markdown("##### 📰 Latest News")
        for item in news_items:
            s = item["sentiment"]
            badge_cls = (
                "badge-positive" if s == "Positive"
                else ("badge-negative" if s == "Negative" else "badge-neutral")
            )
            st.markdown(
                f'<div class="news-item">'
                f'  <div style="display:flex;justify-content:space-between;'
                f'  align-items:flex-start;">'
                f'    <div class="news-title">{item["title"]}</div>'
                f'    <span class="badge {badge_cls}" '
                f'    style="white-space:nowrap;margin-left:12px;">{s}</span>'
                f'  </div>'
                f'  <div class="news-meta">{item["source"]} &middot; {item["date"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    with chart_col:
        # ── Sentiment Distribution Pie ──
        st.markdown("##### 🎯 Sentiment Distribution")
        sentiments_count = {"Positive": 3, "Negative": 1, "Neutral": 1}
        fig_pie = go.Figure(go.Pie(
            labels=list(sentiments_count.keys()),
            values=list(sentiments_count.values()),
            marker=dict(colors=["#00D4AA", "#FF6B6B", "#FFE66D"]),
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

    # ── Sentiment Over Time ──
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("##### 📈 Sentiment Score — Last 30 Days")

    sent_dates = pd.date_range(end=today, periods=30, freq="D")
    # Generate a smooth-ish sentiment score between -1 and 1
    base_sent = rng_news.uniform(0.1, 0.4)
    sent_noise = rng_news.normal(0, 0.15, size=30)
    sent_scores = np.clip(np.cumsum(sent_noise) * 0.1 + base_sent, -1, 1)

    fig_sent = go.Figure()
    # Fill positive/negative areas
    fig_sent.add_trace(go.Scatter(
        x=sent_dates, y=sent_scores,
        fill="tozeroy",
        fillcolor="rgba(0,212,170,0.12)",
        line=dict(color="#00D4AA", width=2),
        name="Sentiment Score",
    ))
    fig_sent.add_hline(y=0, line=dict(color="rgba(255,255,255,0.15)", width=1, dash="dash"))
    fig_sent.add_hline(y=0.5, line=dict(color="rgba(0,212,170,0.2)", width=1, dash="dot"),
                       annotation_text="Bullish", annotation_font_color="#00D4AA")
    fig_sent.add_hline(y=-0.5, line=dict(color="rgba(255,107,107,0.2)", width=1, dash="dot"),
                       annotation_text="Bearish", annotation_font_color="#FF6B6B")
    fig_sent.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        height=320,
        margin=dict(l=0, r=0, t=10, b=0),
        yaxis=dict(
            title="Sentiment Score",
            range=[-1, 1],
            gridcolor="rgba(108,99,255,0.08)",
        ),
        xaxis=dict(gridcolor="rgba(108,99,255,0.08)"),
        showlegend=False,
    )
    st.plotly_chart(fig_sent, use_container_width=True)


# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown(
    '<p style="text-align:center;color:#444;font-size:0.75rem;margin-top:40px;">'
    '© 2025 NiveshAI — AI-Powered Investment Research &middot; '
    'Data shown is simulated for demonstration purposes only.</p>',
    unsafe_allow_html=True,
)
