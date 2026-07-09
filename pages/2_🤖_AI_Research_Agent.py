import streamlit as st
import time
import random
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta

# ── Page Config ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Research Agent — NiveshAI",
    page_icon="🤖",
    layout="wide",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global & Sidebar ─────────────────────────────────────────────────── */
    .stMetric {
        background: linear-gradient(135deg, rgba(108,99,255,.1), rgba(0,212,170,.05));
        border: 1px solid rgba(108,99,255,.2);
        border-radius: 12px;
        padding: 16px;
    }
    .glass-card {
        background: rgba(26,31,46,.8);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(108,99,255,.15);
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
        background-color: rgba(108,99,255,.1);
        border-radius: 8px;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(108,99,255,.3) !important;
    }

    /* ── Chat bubble styling ──────────────────────────────────────────────── */
    div[data-testid="stChatMessage"] {
        background: rgba(26,31,46,.65);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(108,99,255,.12);
        border-radius: 16px;
        padding: 16px 20px;
        margin-bottom: 12px;
        transition: border-color .3s ease;
    }
    div[data-testid="stChatMessage"]:hover {
        border-color: rgba(108,99,255,.35);
    }

    /* assistant vs user accent stripe */
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        border-left: 3px solid #6C63FF;
    }
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        border-left: 3px solid #00D4AA;
    }

    /* ── Header banner ────────────────────────────────────────────────────── */
    .agent-header {
        background: linear-gradient(135deg, rgba(108,99,255,.15) 0%, rgba(0,212,170,.08) 100%);
        border: 1px solid rgba(108,99,255,.2);
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        gap: 18px;
    }
    .agent-header h1 {
        margin: 0;
        font-size: 2rem;
        background: linear-gradient(135deg, #6C63FF, #00D4AA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    .agent-header p {
        margin: 4px 0 0;
        color: #8B92A5;
        font-size: .95rem;
    }

    /* status pill */
    .status-pill {
        display: inline-block;
        background: rgba(0,212,170,.15);
        color: #00D4AA;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: .78rem;
        font-weight: 600;
        letter-spacing: .4px;
        animation: pulse-glow 2s ease-in-out infinite;
    }
    @keyframes pulse-glow {
        0%, 100% { box-shadow: 0 0 4px rgba(0,212,170,.25); }
        50%      { box-shadow: 0 0 14px rgba(0,212,170,.45); }
    }

    /* sidebar section headers */
    .sidebar-section {
        color: #8B92A5;
        font-size: .72rem;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        margin: 18px 0 6px;
        font-weight: 600;
    }

    /* usage-meter */
    .usage-meter {
        background: rgba(26,31,46,.9);
        border: 1px solid rgba(108,99,255,.15);
        border-radius: 12px;
        padding: 14px;
        margin: 8px 0;
    }
    .usage-meter .bar-bg {
        height: 8px;
        background: rgba(108,99,255,.15);
        border-radius: 4px;
        overflow: hidden;
        margin-top: 8px;
    }
    .usage-meter .bar-fill {
        height: 100%;
        border-radius: 4px;
        background: linear-gradient(90deg, #6C63FF, #00D4AA);
        transition: width .6s ease;
    }

    /* quick-prompt buttons */
    .quick-btn {
        display: inline-block;
        background: rgba(108,99,255,.1);
        border: 1px solid rgba(108,99,255,.25);
        border-radius: 10px;
        padding: 8px 14px;
        color: #CCC;
        font-size: .82rem;
        cursor: pointer;
        transition: all .25s ease;
        margin: 4px 2px;
        text-align: center;
    }
    .quick-btn:hover {
        background: rgba(108,99,255,.25);
        border-color: #6C63FF;
        color: #FFF;
    }

    /* tool-call badge */
    .tool-call {
        background: rgba(108,99,255,.08);
        border: 1px solid rgba(108,99,255,.2);
        border-radius: 8px;
        padding: 8px 12px;
        margin: 4px 0;
        font-family: 'JetBrains Mono', monospace;
        font-size: .8rem;
        color: #A8A3FF;
    }

    /* green info text */
    .green-info {
        color: #00D4AA;
        font-size: .82rem;
        font-weight: 500;
    }

    /* hide default streamlit footer */
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────────────────────

def _mini_stock_chart(symbol: str = "RELIANCE") -> go.Figure:
    """Return a small Plotly sparkline chart with random walk data."""
    n = 60
    base_prices = {
        "RELIANCE": 2480, "TCS": 3520, "INFY": 1445,
        "HDFCBANK": 1620, "ICICIBANK": 1085, "WIPRO": 465,
    }
    base = base_prices.get(symbol, 1500)
    dates = [datetime.now() - timedelta(days=n - i) for i in range(n)]
    prices = [base]
    for _ in range(n - 1):
        prices.append(prices[-1] + random.uniform(-base * 0.012, base * 0.013))

    fig = go.Figure()
    color = "#00D4AA" if prices[-1] >= prices[0] else "#FF6B6B"
    fig.add_trace(go.Scatter(
        x=dates, y=prices, mode="lines",
        line=dict(color=color, width=2.2),
        fill="tozeroy",
        fillcolor=color.replace(")", ",.08)").replace("rgb", "rgba") if "rgb" in color else
                  f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},.08)",
        hovertemplate="₹%{y:,.1f}<extra>%{x|%d %b}</extra>",
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        height=200,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        showlegend=False,
    )
    return fig


MOCK_TOOL_CALLS: dict[str, list[str]] = {
    "default": [
        "📡  get_stock_data(symbol='RELIANCE.NS', period='6mo')",
        "📊  compute_technical_indicators(sma=50, ema=21, rsi=14)",
        "📰  analyze_sentiment(source='news', stock='RELIANCE')",
        "🧠  generate_investment_thesis(risk_profile='moderate')",
    ],
    "compare": [
        "📡  get_stock_data(symbol='TCS.NS', period='1y')",
        "📡  get_stock_data(symbol='INFY.NS', period='1y')",
        "📊  compute_comparative_metrics(metrics=['PE','ROE','ROCE'])",
        "📰  analyze_sentiment(source='news', stocks=['TCS','INFY'])",
        "🧠  generate_comparison_report()",
    ],
    "market": [
        "📡  fetch_nifty50_constituents()",
        "📈  get_index_data(indices=['NIFTY50','BANKNIFTY','NIFTYIT'])",
        "📊  compute_breadth_indicators()",
        "📰  analyze_market_sentiment(source='economic_times')",
        "🧠  generate_market_overview()",
    ],
}


def _pick_tool_calls(prompt: str) -> list[str]:
    lower = prompt.lower()
    if any(w in lower for w in ["compare", "vs", "versus"]):
        return MOCK_TOOL_CALLS["compare"]
    if any(w in lower for w in ["market", "overview", "nifty", "index", "sector"]):
        return MOCK_TOOL_CALLS["market"]
    return MOCK_TOOL_CALLS["default"]


def _generate_mock_response(prompt: str) -> str:
    """Return a rich Markdown mock response based on simple keyword matching."""
    lower = prompt.lower()

    # ── Compare two stocks ────────────────────────────────────────────────
    if any(w in lower for w in ["compare", "vs", "versus"]):
        return """## 📊 Comparative Analysis: TCS vs INFOSYS

| Metric | TCS | INFOSYS |
|--------|-----|---------|
| **CMP** | ₹3,521.40 | ₹1,445.75 |
| **Market Cap** | ₹12.83 Lakh Cr | ₹6.01 Lakh Cr |
| **P/E Ratio** | 28.6x | 25.2x |
| **ROE** | 47.8% | 31.5% |
| **ROCE** | 61.2% | 42.1% |
| **Dividend Yield** | 1.42% | 2.18% |
| **Revenue Growth (YoY)** | 8.2% | 6.8% |
| **Net Profit Margin** | 19.1% | 17.4% |

### 🔍 Key Insights

- **TCS** commands a premium valuation due to its market leadership, superior return ratios (ROE 47.8%), and consistent deal-win momentum in the ₹10,000 Cr+ TCV range.
- **INFOSYS** offers a better dividend yield (2.18%) and trades at a slight discount on P/E, making it attractive for income-focused portfolios.
- Both stocks have shown resilience in the recent BFSI spend slow-down, but TCS's diversification gives it an edge.

### 📈 Recommendation
> **Moderate Risk Profile:** Allocate 60% TCS / 40% INFY for a balanced IT exposure. TCS for growth, INFY for value + dividend yield.

*⚠️ This is AI-generated research — not financial advice. Always consult a SEBI-registered advisor.*
"""

    # ── Market overview ───────────────────────────────────────────────────
    if any(w in lower for w in ["market", "overview", "nifty", "top", "sector"]):
        return """## 🇮🇳 Indian Market Overview — July 2026

### Index Snapshot
| Index | Level | Day Change | 5D Trend |
|-------|-------|-----------|----------|
| **NIFTY 50** | 26,482.35 | +0.87% 🟢 | ▲ |
| **SENSEX** | 87,015.60 | +0.91% 🟢 | ▲ |
| **BANK NIFTY** | 55,230.10 | +1.12% 🟢 | ▲ |
| **NIFTY IT** | 42,180.45 | -0.32% 🔴 | ▼ |
| **NIFTY MIDCAP 100** | 54,910.20 | +0.45% 🟢 | ▲ |

### 📊 Sector Heatmap (Top Movers)
| Sector | Performance | Outlook |
|--------|------------|---------|
| 🏦 Banking & Financials | **+1.12%** | Bullish — RBI rate-cut cycle supportive |
| 🏗️ Infrastructure | **+1.45%** | Bullish — Govt capex push continues |
| 💊 Pharma | **+0.78%** | Neutral — US FDA pipeline steady |
| 💻 IT | **-0.32%** | Cautious — Global macro headwinds |
| ⚡ Energy | **+0.55%** | Neutral — Crude at $74/bbl |

### 🔍 Market Breadth
- **Advances:** 1,287 &nbsp;|&nbsp; **Declines:** 892 &nbsp;|&nbsp; **Unchanged:** 63
- **FII Flow (July):** +₹8,420 Cr &nbsp;|&nbsp; **DII Flow:** +₹3,210 Cr

### 💡 AI Insight
> Markets are in a *"cautiously bullish"* phase driven by strong domestic flows and expectation of further rate cuts. Key risk: global tariff escalation could dampen sentiment in export-heavy sectors (IT, Pharma).

*⚠️ This is AI-generated research — not financial advice.*
"""

    # ── Default: single stock analysis ────────────────────────────────────
    # try to extract a stock name
    stocks = ["RELIANCE", "TCS", "INFY", "INFOSYS", "HDFCBANK", "HDFC", "ICICIBANK",
              "ICICI", "WIPRO", "SBIN", "TATAMOTORS", "BAJFINANCE", "ITC", "LT",
              "ADANIENT", "MARUTI", "SUNPHARMA", "TITAN"]
    symbol = "RELIANCE"
    for s in stocks:
        if s.lower() in lower:
            symbol = s
            break

    data_map = {
        "RELIANCE": ("₹2,487.65", "+1.23%", "₹16.82 Lakh Cr", "26.4x", "15.1%", "0.38%", "Strong Buy"),
        "TCS":      ("₹3,521.40", "+0.45%", "₹12.83 Lakh Cr", "28.6x", "47.8%", "1.42%", "Buy"),
        "INFY":     ("₹1,445.75", "-0.28%", "₹6.01 Lakh Cr",  "25.2x", "31.5%", "2.18%", "Hold"),
        "INFOSYS":  ("₹1,445.75", "-0.28%", "₹6.01 Lakh Cr",  "25.2x", "31.5%", "2.18%", "Hold"),
        "HDFCBANK": ("₹1,622.30", "+0.91%", "₹12.36 Lakh Cr", "18.2x", "16.8%", "1.15%", "Strong Buy"),
        "HDFC":     ("₹1,622.30", "+0.91%", "₹12.36 Lakh Cr", "18.2x", "16.8%", "1.15%", "Strong Buy"),
        "ICICIBANK":("₹1,087.45", "+1.05%", "₹7.63 Lakh Cr",  "17.8x", "17.2%", "0.82%", "Buy"),
        "ICICI":    ("₹1,087.45", "+1.05%", "₹7.63 Lakh Cr",  "17.8x", "17.2%", "0.82%", "Buy"),
        "WIPRO":    ("₹467.80",   "-0.52%", "₹2.44 Lakh Cr",  "22.1x", "15.4%", "1.28%", "Hold"),
        "ITC":      ("₹458.90",   "+0.67%", "₹5.73 Lakh Cr",  "26.8x", "28.3%", "3.05%", "Buy"),
    }
    d = data_map.get(symbol, ("₹1,500.00", "+0.50%", "₹5.00 Lakh Cr", "22.0x", "18.0%", "1.00%", "Hold"))

    return f"""## 📈 {symbol} — Investment Research Report

### Quick Stats
| Metric | Value |
|--------|-------|
| **Current Price** | {d[0]} |
| **Day Change** | {d[1]} {'🟢' if '+' in d[1] else '🔴'} |
| **Market Cap** | {d[2]} |
| **P/E Ratio** | {d[3]} |
| **ROE** | {d[4]} |
| **Dividend Yield** | {d[5]} |

### 🔬 Technical Analysis
- **RSI (14):** 58.3 — *Neutral zone*
- **MACD:** Bullish crossover on daily chart 📈
- **50-DMA:** ₹{float(d[0].replace('₹','').replace(',','')) * 0.97:,.2f} (price above → bullish)
- **200-DMA:** ₹{float(d[0].replace('₹','').replace(',','')) * 0.92:,.2f} (price above → long-term uptrend)
- **Support:** ₹{float(d[0].replace('₹','').replace(',','')) * 0.95:,.2f} &nbsp;|&nbsp; **Resistance:** ₹{float(d[0].replace('₹','').replace(',','')) * 1.04:,.2f}

### 📰 Sentiment Analysis
| Source | Sentiment | Score |
|--------|-----------|-------|
| News Articles (30d) | **Positive** | 0.72 / 1.0 |
| Social Media | **Neutral-Positive** | 0.61 / 1.0 |
| Analyst Reports | **Bullish** | 8 Buy · 3 Hold · 1 Sell |

### 🎯 AI Verdict: **{d[6]}**
> Based on strong fundamentals, positive technical setup, and favorable sentiment, {symbol} presents a compelling opportunity for medium-to-long-term investors. Consider accumulating on dips near the 50-DMA support level.

### ⚠️ Key Risks
1. Regulatory changes in the sector
2. Global macroeconomic headwinds
3. Quarterly earnings miss could trigger short-term correction

*⚠️ This is AI-generated research — not financial advice. Always consult a SEBI-registered advisor.*
"""


# ── Session State Init ───────────────────────────────────────────────────────────

WELCOME_MSG = (
    "Namaste! 🙏 I'm **NiveshAI**, your AI-powered investment research assistant "
    "for the Indian stock market.\n\n"
    "Ask me anything about **NSE stocks**, market trends, or get personalised "
    "investment insights.\n\n"
    "**Try asking:**\n"
    '- *"Should I invest in RELIANCE?"*\n'
    '- *"Compare HDFC Bank vs ICICI Bank"*\n'
    '- *"Give me a market overview"*\n'
    '- *"Top IT sector stocks"*'
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": WELCOME_MSG}
    ]
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "Gemini 2.0 Flash ✨ (FREE)"
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.7
if "requests_used" not in st.session_state:
    st.session_state.requests_used = 847


# ── Sidebar ──────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<p class="gradient-text" style="font-size:1.4rem;margin-bottom:0;">🤖 AI Agent Config</p>',
                unsafe_allow_html=True)
    st.markdown("---")

    # Model selector
    st.markdown('<p class="sidebar-section">🧠 Model</p>', unsafe_allow_html=True)
    model = st.selectbox(
        "Select model",
        [
            "Gemini 2.0 Flash ✨ (FREE)",
            "Gemini 2.5 Flash ⚡ (FREE)",
            "OpenAI GPT-4o 🤖",
            "Groq LLaMA 3 🚀",
            "Anthropic Claude 🧠",
        ],
        index=0,
        label_visibility="collapsed",
    )
    st.session_state.selected_model = model

    # API key
    is_gemini = "Gemini" in model
    if is_gemini:
        st.markdown('<p class="green-info">✅ Built-in free key available</p>', unsafe_allow_html=True)
    else:
        provider = model.split(" ")[0]  # first word
        st.text_input(f"🔑 {provider} API Key", type="password", placeholder="sk-...")

    st.markdown("---")

    # Usage meter
    st.markdown('<p class="sidebar-section">📊 Usage Today</p>', unsafe_allow_html=True)
    used = st.session_state.requests_used
    limit = 1500
    pct = used / limit
    st.markdown(f"""
    <div class="usage-meter">
        <div style="display:flex;justify-content:space-between;font-size:.82rem;">
            <span style="color:#CCC;">{used:,} / {limit:,} requests</span>
            <span style="color:#6C63FF;font-weight:600;">{pct:.0%}</span>
        </div>
        <div class="bar-bg"><div class="bar-fill" style="width:{pct*100:.1f}%"></div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Temperature
    st.markdown('<p class="sidebar-section">🌡️ Temperature</p>', unsafe_allow_html=True)
    st.session_state.temperature = st.slider(
        "Temperature", 0.0, 1.0, st.session_state.temperature, 0.05,
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Quick prompts
    st.markdown('<p class="sidebar-section">⚡ Quick Prompts</p>', unsafe_allow_html=True)

    qp_cols = st.columns(2)
    quick_prompts = [
        ("📈 Analyze RELIANCE", "Analyze RELIANCE stock and give me an investment thesis"),
        ("⚖️ Compare TCS vs INFY", "Compare TCS vs INFY — which is a better investment?"),
        ("💻 Top IT stocks", "What are the top IT sector stocks to invest in right now?"),
        ("🌐 Market overview", "Give me a comprehensive Indian market overview for today"),
    ]
    for idx, (label, prompt_text) in enumerate(quick_prompts):
        col = qp_cols[idx % 2]
        if col.button(label, key=f"qp_{idx}", use_container_width=True):
            st.session_state.chat_history.append({"role": "user", "content": prompt_text})
            st.session_state._pending_quick = prompt_text
            st.rerun()

    st.markdown("---")
    st.caption(f"Model: `{model.split('(')[0].strip()}`  \nTemp: `{st.session_state.temperature}`")


# ── Header ───────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="agent-header">
    <div>
        <h1>🤖 AI Research Agent</h1>
        <p>Institutional-grade investment research powered by AI — built for the Indian market.</p>
        <span class="status-pill">● Online &nbsp;·&nbsp; """ + st.session_state.selected_model.split("(")[0].strip() + """</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Chat History Render ──────────────────────────────────────────────────────────

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # if there are tool calls stored, show them
        if "tool_calls" in msg:
            with st.expander("🔧 Tool Calls & Agent Reasoning", expanded=False):
                for tc in msg["tool_calls"]:
                    st.markdown(f'<div class="tool-call">{tc}</div>', unsafe_allow_html=True)
        # if there is a chart stored, render it
        if "chart_symbol" in msg:
            st.plotly_chart(_mini_stock_chart(msg["chart_symbol"]), use_container_width=True,
                           config={"displayModeBar": False})


# ── Chat Input ───────────────────────────────────────────────────────────────────

def _handle_prompt(prompt: str):
    """Process a user prompt: add to history, generate mock response."""
    # render user bubble immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    # assistant response
    with st.chat_message("assistant"):
        with st.spinner("🧠 Analyzing data & generating insights..."):
            time.sleep(1.0)

        response = _generate_mock_response(prompt)
        st.markdown(response)

        # determine chart symbol
        stocks_check = ["RELIANCE", "TCS", "INFY", "INFOSYS", "HDFCBANK", "HDFC",
                        "ICICIBANK", "ICICI", "WIPRO", "ITC", "SBIN"]
        chart_symbol = "RELIANCE"
        for s in stocks_check:
            if s.lower() in prompt.lower():
                chart_symbol = s
                break

        st.plotly_chart(_mini_stock_chart(chart_symbol), use_container_width=True,
                        config={"displayModeBar": False})

        tool_calls = _pick_tool_calls(prompt)
        with st.expander("🔧 Tool Calls & Agent Reasoning", expanded=False):
            for tc in tool_calls:
                st.markdown(f'<div class="tool-call">{tc}</div>', unsafe_allow_html=True)

    # persist to session state
    st.session_state.requests_used += 1
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": response,
        "tool_calls": tool_calls,
        "chart_symbol": chart_symbol,
    })


# Check for pending quick prompt
if hasattr(st.session_state, "_pending_quick"):
    prompt = st.session_state._pending_quick
    del st.session_state._pending_quick
    _handle_prompt(prompt)

# Normal chat input
if prompt := st.chat_input("Ask about any NSE stock, sector, or market trend..."):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    _handle_prompt(prompt)
