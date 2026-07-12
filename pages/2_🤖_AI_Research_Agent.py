import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import time
import re
import random
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

try:
    import google.generativeai as genai

    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

# Internal modules
from data.stock_data import fetch_full_stock_data, fetch_market_indices, fetch_history
from data.news_fetcher import fetch_news, format_news_for_llm
from analysis.technical import generate_signals
from analysis.fundamental import analyze_fundamentals
from config.settings import (
    GEMINI_API_KEY,
    OPENAI_API_KEY,
    GROQ_API_KEY,
    ANTHROPIC_API_KEY,
    NEWSAPI_KEY,
)
from data.company_db import get_all_symbols, get_company

# ── Page Config ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Research Agent — NiveshAI",
    page_icon="🤖",
    layout="wide",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────────
st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)


# ── Chart generator helper ───────────────────────────────────────────────────────
def _mini_stock_chart(symbol: str) -> go.Figure:
    """Return a small Plotly line chart with actual stock price history if available."""
    try:
        df = fetch_history(symbol, period="3mo", add_indicators=False)
        if df.empty:
            raise ValueError()
        dates = df.index
        prices = df["Close"].values
    except Exception:
        # Fallback to random data if fetch fails
        n = 60
        base = 1500
        dates = [datetime.now() - timedelta(days=n - i) for i in range(n)]
        prices = [base]
        for _ in range(n - 1):
            prices.append(prices[-1] + random.uniform(-base * 0.012, base * 0.013))

    fig = go.Figure()
    color = "#00D4AA" if prices[-1] >= prices[0] else "#FF6B6B"
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=prices,
            mode="lines",
            line=dict(color=color, width=2.2),
            fill="tozeroy",
            fillcolor=(
                f"rgba(0,212,170,.08)"
                if prices[-1] >= prices[0]
                else f"rgba(255,107,107,.08)"
            ),
            hovertemplate="₹%{y:,.2f}<extra>%{x|%d %b}</extra>",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=5, r=5, t=5, b=5),
        height=150,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        showlegend=False,
    )
    return fig


# ── System Instruction ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are NiveshAI, an expert AI investment research assistant for Indian stock markets (NSE).
You have access to real-time stock data, news, and analysis tools.
Always think about: risk level, sector context, and long-term fundamentals.
Respond in clear English with data-backed reasoning. 
Use ₹ for prices. Format numbers in Indian style (Lakh/Crore).
Always add a disclaimer: "This is not financial advice."
"""

# ── Symbol Extractor ─────────────────────────────────────────────────────────────
COMMON_SYMBOLS = set(
    [
        "RELIANCE",
        "TCS",
        "INFY",
        "HDFCBANK",
        "ICICIBANK",
        "BHARTIARTL",
        "SBIN",
        "ITC",
        "WIPRO",
        "BAJFINANCE",
        "MARUTI",
        "SUNPHARMA",
        "AXISBANK",
        "LT",
        "KOTAKBANK",
        "TATASTEEL",
        "ADANIENT",
        "TITAN",
        "HCLTECH",
        "NIFTY",
    ]
)


def extract_symbols(text: str) -> list[str]:
    words = re.findall(r"\b[A-Z]{2,15}\b", text.upper())
    return [w for w in words if w in COMMON_SYMBOLS][:2]


# ── Tool calling runner ──────────────────────────────────────────────────────────
def tool_get_stock_data(symbol: str) -> str:
    """Fetch real stock data for a symbol and format for LLM context."""
    data = fetch_full_stock_data(symbol.upper())
    signals = generate_signals(data["history"])
    fund_analysis = analyze_fundamentals(data["fundamentals"])
    news = fetch_news(symbol.upper(), max_articles=5)
    news_text = format_news_for_llm(news)

    return f"""
STOCK DATA FOR {symbol.upper()}:
Current Price: ₹{data['live_price']['price']}
Change: {data['live_price']['change_pct']:+.2f}%

SIGNAL: {signals['overall']} (score: {signals['score']})
FUNDAMENTAL RATING: {fund_analysis['rating']} ({fund_analysis['score']}/100)

KEY METRICS:
{chr(10).join(f"  {k}: {v}" for k,v in list(fund_analysis['metrics'].items())[:8])}

RECENT NEWS:
{news_text}
"""


# ── Call LLM APIs ────────────────────────────────────────────────────────────────
def get_ai_response(user_message: str, api_key: str, chat_history: list) -> str:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash", system_instruction=SYSTEM_PROMPT)

    # Auto-detect if user is asking about a specific stock
    symbols_mentioned = extract_symbols(user_message)  # regex to find NSE symbols

    context = ""
    if symbols_mentioned:
        with st.status(f"🔍 Looking up {symbols_mentioned[0]}..."):
            try:
                context = tool_get_stock_data(symbols_mentioned[0])
            except Exception as e:
                context = ""
                st.warning(
                    f"⚠️ Failed to fetch stock data for {symbols_mentioned[0]}: {e}"
                )

    # Build message with context
    full_message = (
        f"{context}\n\nUser question: {user_message}" if context else user_message
    )

    # Build chat history for Gemini
    history = []
    for m in chat_history[:-1]:
        role = "user" if m["role"] == "user" else "model"
        if m.get("content"):
            history.append({"role": role, "parts": [m["content"]]})

    chat = model.start_chat(history=history)
    response = chat.send_message(full_message)
    return response.text


def get_ai_response_stream(user_message: str, api_key: str, chat_history: list):
    """Call real Gemini API with streaming and tool data fetching."""
    if not HAS_GEMINI:
        raise ImportError("google-generativeai not installed")

    genai.configure(api_key=api_key)

    # Check for symbols in message
    symbols_mentioned = extract_symbols(user_message)
    context = ""
    tool_used = []

    if symbols_mentioned:
        symbol = symbols_mentioned[0]
        tool_used.append(f"get_stock_data({symbol})")
        with st.status(f"🔍 Looking up {symbol}..."):
            try:
                context += tool_get_stock_data(symbol)
            except Exception as e:
                # If stock data fetch fails: still send user message without context
                st.warning(
                    f"⚠️ Failed to fetch stock data for {symbol}: {e}. Continuing without context."
                )

    if (
        "market" in user_message.lower()
        or "nifty" in user_message.lower()
        or "sensex" in user_message.lower()
    ):
        tool_used.append("fetch_market_indices()")
        with st.status("📈 Fetching market indices..."):
            try:
                indices = fetch_market_indices()
                context += f"\n====== INDIAN MARKET INDICES ======\n"
                for ind, details in indices.items():
                    context += (
                        f"{ind}: {details['price']} ({details['change_pct']:+.2f}%)\n"
                    )
                context += "===================================\n"
            except Exception as e:
                st.warning(
                    f"⚠️ Failed to fetch market indices: {e}. Continuing without context."
                )

    # Build message with context
    full_message = (
        f"{context}\n\nUser question: {user_message}" if context else user_message
    )

    # Build chat history for Gemini
    history = []
    for m in chat_history[:-1]:
        role = "user" if m["role"] == "user" else "model"
        if m.get("content"):
            history.append({"role": role, "parts": [m["content"]]})

    # Model configuration
    selected_model_name = st.session_state.get(
        "selected_model", "Gemini 2.0 Flash ✨ (FREE)"
    )
    model_id = "gemini-2.0-flash"
    if "Gemini 2.5" in selected_model_name:
        model_id = "gemini-2.5-flash"

    model = genai.GenerativeModel(model_id, system_instruction=SYSTEM_PROMPT)
    chat = model.start_chat(history=history)

    response = chat.send_message(full_message, stream=True)
    return response, tool_used


def call_gemini(
    prompt: str, api_key: str, model_id: str, context: str, chat_history: list
) -> str:
    # Deprecated fallback for backward compatibility
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name=model_id, system_instruction=SYSTEM_PROMPT)
    message_content = f"{context}\n\nUser Question: {prompt}" if context else prompt
    formatted_history = []
    for h in chat_history[:-1]:
        role = "user" if h["role"] == "user" else "model"
        formatted_history.append({"role": role, "parts": [h["content"]]})
    chat = model.start_chat(history=formatted_history)
    response = chat.send_message(message_content)
    return response.text


def call_openai(
    prompt: str, api_key: str, model_id: str, context: str, chat_history: list
) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    # Add history
    for h in chat_history[:-1]:
        messages.append({"role": h["role"], "content": h["content"]})

    # Add current prompt
    user_content = f"{context}\n\nUser Question: {prompt}" if context else prompt
    messages.append({"role": "user", "content": user_content})

    completion = client.chat.completions.create(
        model=model_id, messages=messages, temperature=st.session_state.temperature
    )
    return completion.choices[0].message.content


def call_groq(
    prompt: str, api_key: str, model_id: str, context: str, chat_history: list
) -> str:
    import requests

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in chat_history[:-1]:
        messages.append({"role": h["role"], "content": h["content"]})
    user_content = f"{context}\n\nUser Question: {prompt}" if context else prompt
    messages.append({"role": "user", "content": user_content})

    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": st.session_state.temperature,
    }
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        json=payload,
        headers=headers,
        timeout=10,
    )
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise ValueError(f"Groq API Error: {response.text}")


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
    st.session_state.chat_history = [{"role": "assistant", "content": WELCOME_MSG}]
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "Gemini 2.0 Flash ✨ (FREE)"
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.7
if "requests_used" not in st.session_state:
    st.session_state.requests_used = 0

# Pre-load ALL api keys into session_state from config (which reads st.secrets > .env > os.environ)
# Only initialise once — user can override by typing in the sidebar
if "gemini_key" not in st.session_state:
    st.session_state.gemini_key = GEMINI_API_KEY or ""
if "groq_key" not in st.session_state:
    st.session_state.groq_key = GROQ_API_KEY or ""
if "openai_key" not in st.session_state:
    st.session_state.openai_key = OPENAI_API_KEY or ""
if "anthropic_key" not in st.session_state:
    st.session_state.anthropic_key = ANTHROPIC_API_KEY or ""


# ── Sidebar ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<p class="gradient-text" style="font-size:1.4rem;margin-bottom:0;">🤖 AI Agent Config</p>',
        unsafe_allow_html=True,
    )
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

    # API key setup & mapping
    api_key_to_use = ""
    is_gemini = "Gemini" in model

    if is_gemini:
        # Show Gemini API Key input (using password type)
        gemini_key_input = st.text_input(
            "Gemini API Key",
            value=st.session_state["gemini_key"],
            type="password",
            placeholder="Enter key...",
        )
        st.session_state["gemini_key"] = gemini_key_input
        api_key_to_use = gemini_key_input

        st.markdown("[Get a free key](https://aistudio.google.com/apikey)")

        if api_key_to_use:
            st.markdown(
                '<p class="green-info">✅ API Key configured</p>',
                unsafe_allow_html=True,
            )
        else:
            st.warning("⚠️ No Gemini API key found. Enter one above.")
    else:
        provider = model.split(" ")[0]
        # Map provider name -> (session_state key, placeholder text)
        key_map = {
            "OpenAI":    ("openai_key",    "sk-..."),
            "Groq":      ("groq_key",      "gsk_..."),
            "Anthropic": ("anthropic_key", "sk-ant-..."),
        }
        sess_key, placeholder = key_map.get(provider, ("groq_key", "sk-..."))

        # Pre-populate from session_state (which was loaded from .env/st.secrets on first run)
        provider_key_input = st.text_input(
            f"🔑 {provider} API Key",
            value=st.session_state.get(sess_key, ""),
            type="password",
            placeholder=placeholder,
        )
        # Persist back so any manual override survives reruns
        st.session_state[sess_key] = provider_key_input
        api_key_to_use = provider_key_input
        if api_key_to_use:
            st.markdown(
                f'<p class="green-info">✅ {provider} API Key configured</p>',
                unsafe_allow_html=True,
            )
        else:
            st.warning(f"⚠️ Configure {provider} key to use this model.")

    st.markdown("---")

    # Usage meter (Gemini Free tier limits: 1500 req/day)
    st.markdown('<p class="sidebar-section">📊 Usage Today</p>', unsafe_allow_html=True)
    used = st.session_state.requests_used
    limit = 1500
    pct = min(1.0, used / limit)
    st.markdown(
        f"""
    <div class="usage-meter">
        <div style="display:flex;justify-content:space-between;font-size:.82rem;">
            <span style="color:#CCC;">{used:,} / {limit:,} requests</span>
            <span style="color:#6C63FF;font-weight:600;">{pct:.1%}</span>
        </div>
        <div class="bar-bg"><div class="bar-fill" style="width:{pct*100:.1f}%"></div></div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Temperature
    st.markdown('<p class="sidebar-section">🌡️ Temperature</p>', unsafe_allow_html=True)
    st.session_state.temperature = st.slider(
        "Temperature",
        0.0,
        1.0,
        st.session_state.temperature,
        0.05,
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Quick prompts
    st.markdown(
        '<p class="sidebar-section">⚡ Quick Prompts</p>', unsafe_allow_html=True
    )
    qp_cols = st.columns(2)
    quick_prompts = [
        (
            "📈 Analyze RELIANCE",
            "Analyze RELIANCE stock and give me an investment thesis",
        ),
        (
            "⚖️ Compare TCS vs INFY",
            "Compare TCS and INFY — which is a better investment?",
        ),
        (
            "💻 Top IT stocks",
            "What are the top IT sector stocks to invest in right now?",
        ),
        (
            "🌐 Market overview",
            "Give me a comprehensive Indian market overview for today",
        ),
    ]
    for idx, (label, prompt_text) in enumerate(quick_prompts):
        col = qp_cols[idx % 2]
        if col.button(label, key=f"qp_{idx}", use_container_width=True):
            st.session_state.chat_history.append(
                {"role": "user", "content": prompt_text}
            )
            st.session_state._pending_quick = prompt_text
            st.rerun()

    st.markdown("---")
    st.caption(
        f"Model: `{model.split('(')[0].strip()}`  \nTemp: `{st.session_state.temperature}`"
    )


# ── Header ───────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="agent-header">
    <div>
        <h1>🤖 AI Research Agent</h1>
        <p>Institutional-grade investment research powered by AI — built for the Indian market.</p>
        <span class="status-pill">● Online &nbsp;·&nbsp; """
    + st.session_state.selected_model.split("(")[0].strip()
    + """</span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# ── Chat History Render ──────────────────────────────────────────────────────────
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "tool_calls" in msg:
            with st.expander("🔧 Tool Calls & Agent Reasoning", expanded=False):
                for tc in msg["tool_calls"]:
                    st.markdown(
                        f'<div class="tool-call">{tc}</div>', unsafe_allow_html=True
                    )
        if "chart_symbol" in msg and msg["chart_symbol"]:
            st.plotly_chart(
                _mini_stock_chart(msg["chart_symbol"]),
                use_container_width=True,
                config={"displayModeBar": False},
            )


# ── Chat Input & Inference Runner ───────────────────────────────────────────────
def _handle_prompt(prompt: str):
    """Processes user prompt, runs RAG lookup, queries LLM API, and outputs streaming message."""
    # Render user prompt immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    # Check daily limit warning
    if st.session_state.requests_used >= 1200:
        st.warning(
            "⚠️ Warning: You have used 80% of your free tier limit (1500 requests/day)."
        )

    is_gemini = "Gemini" in model

    if is_gemini:
        if not HAS_GEMINI:
            with st.chat_message("assistant"):
                st.error(
                    "❌ `google-generativeai` is not installed. Please install it using: `pip install google-generativeai`"
                )
            return

        if not api_key_to_use:
            with st.chat_message("assistant"):
                st.error(
                    "❌ Gemini API Key is missing. Please configure it in the sidebar."
                )
            return

        with st.chat_message("assistant"):
            try:
                # Calls get_ai_response_stream which handles symbol lookup & st.status natively
                response_stream, tool_used = get_ai_response_stream(
                    user_message=prompt,
                    api_key=api_key_to_use,
                    chat_history=st.session_state.chat_history,
                )

                # Show "Tool used: get_stock_data(RELIANCE)" in expander
                if tool_used:
                    with st.expander("🔧 Tools Used", expanded=True):
                        for tool in tool_used:
                            st.markdown(
                                f'<div class="tool-call">Tool used: {tool}</div>',
                                unsafe_allow_html=True,
                            )

                # Stream response
                placeholder = st.empty()
                full_response = ""
                for chunk in response_stream:
                    if chunk.text:
                        full_response += chunk.text
                        placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)

                chart_symbol = None
                symbols = extract_symbols(prompt)
                if symbols:
                    chart_symbol = symbols[0]
                    st.plotly_chart(
                        _mini_stock_chart(chart_symbol),
                        use_container_width=True,
                        config={"displayModeBar": False},
                    )

                # Save to history
                st.session_state.requests_used += 1
                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": full_response,
                        "tool_calls": [f"Tool used: {t}" for t in tool_used],
                        "chart_symbol": chart_symbol,
                    }
                )
                st.rerun()

            except Exception as e:
                err_msg = str(e)
                if (
                    "API_KEY_INVALID" in err_msg
                    or "API key not valid" in err_msg
                    or "Invalid API key" in err_msg
                ):
                    st.error(
                        "❌ Invalid API key. Get a free key at aistudio.google.com"
                    )
                elif (
                    "429" in err_msg
                    or "ResourceExhausted" in err_msg
                    or "Quota exceeded" in err_msg
                    or "rate limit" in err_msg.lower()
                ):
                    st.error("Rate limit reached. Wait 60 seconds.")
                else:
                    st.error(f"❌ Error running inference: {e}")
    else:
        # Fallback path for other model providers (OpenAI, Groq)
        symbols = extract_symbols(prompt)
        context_data = ""
        tool_calls_executed = []
        chart_symbol = None

        if symbols:
            chart_symbol = symbols[0]
            tool_calls_executed.append(f"get_stock_data({chart_symbol})")
            with st.status(f"🔍 Looking up {chart_symbol}..."):
                try:
                    context_data += tool_get_stock_data(chart_symbol)
                except Exception as e:
                    st.warning(f"⚠️ Failed to fetch stock data: {e}")

        if (
            "market" in prompt.lower()
            or "nifty" in prompt.lower()
            or "sensex" in prompt.lower()
        ):
            tool_calls_executed.append("fetch_market_indices()")
            with st.status("📈 Fetching market indices..."):
                try:
                    indices = fetch_market_indices()
                    context_data += f"\n====== INDIAN MARKET INDICES ======\n"
                    for ind, details in indices.items():
                        context_data += f"{ind}: {details['price']} ({details['change_pct']:+.2f}%)\n"
                    context_data += "===================================\n"
                except Exception as e:
                    st.warning(f"⚠️ Failed to fetch market indices: {e}")

        with st.chat_message("assistant"):
            if tool_calls_executed:
                with st.expander("🔧 Tools Used", expanded=True):
                    for tc in tool_calls_executed:
                        st.markdown(
                            f'<div class="tool-call">Tool used: {tc}</div>',
                            unsafe_allow_html=True,
                        )

            with st.spinner("🧠 Analyzing data & generating insights..."):
                try:
                    if not api_key_to_use:
                        raise ValueError(
                            "API Key is missing or invalid. Please check your credentials."
                        )

                    from config.settings import LLM_PROVIDERS

                    if "OpenAI" in model:
                        model_id = LLM_PROVIDERS.get("openai", {}).get(
                            "model_id", "gpt-4o"
                        )
                        response_text = call_openai(
                            prompt,
                            api_key_to_use,
                            model_id,
                            context_data,
                            st.session_state.chat_history,
                        )
                    elif "Groq" in model:
                        model_id = LLM_PROVIDERS.get("groq", {}).get(
                            "model_id", "llama-3.3-70b-versatile"
                        )
                        response_text = call_groq(
                            prompt,
                            api_key_to_use,
                            model_id,
                            context_data,
                            st.session_state.chat_history,
                        )
                    else:
                        response_text = f"Inference for model `{model}` is not fully integrated yet. Please use Gemini or OpenAI."
                except Exception as e:
                    response_text = f"❌ **Error running inference**: {e}\n\n*Please verify your API keys and internet connection.*"

            st.markdown(response_text)

            if chart_symbol:
                st.plotly_chart(
                    _mini_stock_chart(chart_symbol),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

        st.session_state.requests_used += 1
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": response_text,
                "tool_calls": tool_calls_executed,
                "chart_symbol": chart_symbol,
            }
        )
        st.rerun()


# Check for pending quick prompt
if hasattr(st.session_state, "_pending_quick"):
    prompt = st.session_state._pending_quick
    del st.session_state._pending_quick
    _handle_prompt(prompt)

# Normal chat input
if prompt := st.chat_input("Ask about any NSE stock, sector, or market trend..."):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    _handle_prompt(prompt)
