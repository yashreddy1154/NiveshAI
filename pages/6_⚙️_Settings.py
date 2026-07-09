import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import random

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(page_title="Settings — NiveshAI", page_icon="⚙️", layout="wide")

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Base Theme ── */
    .stApp { background-color: #0E1117; }

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

    /* ── Provider Card ── */
    .provider-card {
        background: rgba(26, 31, 46, 0.85);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(108, 99, 255, 0.18);
        border-radius: 16px;
        padding: 28px 24px 20px 24px;
        margin: 6px 0 14px 0;
        transition: border-color 0.3s, box-shadow 0.3s;
        position: relative;
    }
    .provider-card:hover {
        border-color: rgba(108, 99, 255, 0.45);
        box-shadow: 0 0 24px rgba(108, 99, 255, 0.08);
    }
    .provider-card.active-provider {
        border: 1.5px solid #00D4AA;
        box-shadow: 0 0 28px rgba(0, 212, 170, 0.12);
    }
    .provider-name {
        font-size: 1.18rem;
        font-weight: 700;
        color: #E8E8E8;
        margin-bottom: 4px;
    }
    .provider-desc {
        font-size: 0.88rem;
        color: #9DA3B4;
        margin: 8px 0 12px 0;
        line-height: 1.5;
    }
    .badge {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        margin-right: 6px;
        vertical-align: middle;
    }
    .badge-free {
        background: rgba(0, 212, 170, 0.15);
        color: #00D4AA;
        border: 1px solid rgba(0, 212, 170, 0.3);
    }
    .badge-paid {
        background: rgba(255, 179, 71, 0.15);
        color: #FFB347;
        border: 1px solid rgba(255, 179, 71, 0.3);
    }
    .badge-limit {
        background: rgba(108, 99, 255, 0.12);
        color: #A8A0FF;
        border: 1px solid rgba(108, 99, 255, 0.25);
    }
    .active-indicator {
        display: inline-block;
        background: rgba(0, 212, 170, 0.15);
        color: #00D4AA;
        border: 1px solid rgba(0, 212, 170, 0.35);
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.4px;
    }

    /* ── API Key Section ── */
    .api-key-card {
        background: rgba(26, 31, 46, 0.75);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(108, 99, 255, 0.12);
        border-radius: 14px;
        padding: 22px 24px;
        margin: 8px 0 14px 0;
    }
    .key-status { font-size: 0.92rem; font-weight: 600; }

    /* ── Info Boxes ── */
    .info-box-green {
        background: rgba(0, 212, 170, 0.08);
        border: 1px solid rgba(0, 212, 170, 0.25);
        border-radius: 12px;
        padding: 16px 20px;
        color: #00D4AA;
        font-size: 0.92rem;
        margin: 10px 0;
    }
    .info-box-blue {
        background: rgba(108, 99, 255, 0.08);
        border: 1px solid rgba(108, 99, 255, 0.22);
        border-radius: 12px;
        padding: 16px 20px;
        color: #A8A0FF;
        font-size: 0.92rem;
        margin: 10px 0;
    }
    .info-box-amber {
        background: rgba(255, 179, 71, 0.08);
        border: 1px solid rgba(255, 179, 71, 0.22);
        border-radius: 12px;
        padding: 14px 20px;
        color: #FFB347;
        font-size: 0.85rem;
        margin: 10px 0;
    }

    /* ── Usage Dashboard ── */
    .usage-metric-card {
        background: rgba(26, 31, 46, 0.85);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(108, 99, 255, 0.15);
        border-radius: 14px;
        padding: 22px 20px;
        margin-bottom: 10px;
        text-align: center;
    }
    .usage-metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #E8E8E8;
    }
    .usage-metric-label {
        font-size: 0.82rem;
        color: #9DA3B4;
        margin-top: 4px;
    }

    /* ── About Page ── */
    .about-card {
        background: rgba(26, 31, 46, 0.85);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(108, 99, 255, 0.15);
        border-radius: 16px;
        padding: 28px 26px;
        margin: 10px 0 16px 0;
    }
    .tech-badge {
        display: inline-block;
        background: rgba(108, 99, 255, 0.12);
        border: 1px solid rgba(108, 99, 255, 0.28);
        border-radius: 20px;
        padding: 5px 16px;
        font-size: 0.82rem;
        font-weight: 600;
        color: #A8A0FF;
        margin: 4px 6px 4px 0;
    }
    .model-status-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(14, 17, 23, 0.5);
        border: 1px solid rgba(108, 99, 255, 0.1);
        border-radius: 10px;
        padding: 14px 18px;
        margin: 6px 0;
    }
    .model-status-name {
        font-size: 0.92rem;
        color: #E8E8E8;
        font-weight: 600;
    }
    .model-status-desc {
        font-size: 0.78rem;
        color: #9DA3B4;
        margin-top: 2px;
    }
    .status-loaded {
        color: #00D4AA;
        font-weight: 700;
        font-size: 0.85rem;
    }

    /* ── Section Headers ── */
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #E8E8E8;
        margin: 18px 0 10px 0;
    }
    .section-sub {
        font-size: 0.88rem;
        color: #9DA3B4;
        margin-bottom: 16px;
    }

    /* ── Cost Table ── */
    .cost-table {
        width: 100%;
        border-collapse: collapse;
        margin: 12px 0;
    }
    .cost-table th {
        text-align: left;
        color: #9DA3B4;
        font-weight: 600;
        font-size: 0.82rem;
        padding: 10px 14px;
        border-bottom: 1px solid rgba(108, 99, 255, 0.15);
    }
    .cost-table td {
        padding: 12px 14px;
        color: #E8E8E8;
        font-size: 0.9rem;
        border-bottom: 1px solid rgba(108, 99, 255, 0.08);
    }
</style>
""", unsafe_allow_html=True)

# ─── Session State Defaults ─────────────────────────────────────────────────
if "active_provider" not in st.session_state:
    st.session_state.active_provider = "gemini_2_flash"

if "api_keys" not in st.session_state:
    st.session_state.api_keys = {
        "openai": "",
        "groq": "",
        "anthropic": "",
    }

if "key_statuses" not in st.session_state:
    st.session_state.key_statuses = {
        "openai": "not_set",
        "groq": "not_set",
        "anthropic": "not_set",
    }

if "keys_saved_toast" not in st.session_state:
    st.session_state.keys_saved_toast = False

# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="gradient-text" style="font-size:1.6rem;margin-bottom:2px;">NiveshAI</p>', unsafe_allow_html=True)
    st.caption("AI-Powered Investment Research")
    st.divider()
    st.markdown("### ⚙️ Settings")
    st.markdown("""
    <div class="info-box-blue">
        Configure model providers, manage API keys, and monitor usage.
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    active_label = {
        "gemini_2_flash": "✨ Gemini 2.0 Flash",
        "gemini_25_flash": "⚡ Gemini 2.5 Flash",
        "openai_gpt4o": "🤖 OpenAI GPT-4o",
        "groq_llama3": "🚀 Groq LLaMA 3 70B",
        "anthropic_claude": "🧠 Anthropic Claude 3.5",
    }
    st.markdown(f"**Active Model:**")
    st.markdown(f'<span class="active-indicator">{active_label.get(st.session_state.active_provider, "Unknown")}</span>', unsafe_allow_html=True)
    st.divider()
    st.caption("© 2026 NiveshAI · v1.0.0")

# ─── Title ───────────────────────────────────────────────────────────────────
st.markdown('<h1><span class="gradient-text">⚙️ Settings</span> & Model Configuration</h1>', unsafe_allow_html=True)
st.markdown('<p style="color:#9DA3B4;margin-top:-10px;margin-bottom:24px;">Configure AI providers, manage API keys, and monitor your usage.</p>', unsafe_allow_html=True)

# ─── Tabs ────────────────────────────────────────────────────────────────────
tab_providers, tab_keys, tab_usage, tab_about = st.tabs([
    "🤖 LLM Providers",
    "🔑 API Keys",
    "📊 Usage Dashboard",
    "ℹ️ About",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — LLM PROVIDERS
# ═════════════════════════════════════════════════════════════════════════════
with tab_providers:
    st.markdown('<p class="section-header">Choose Your AI Model Provider</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Select the LLM that powers NiveshAI\'s analysis engine. Free tiers are available for quick start.</p>', unsafe_allow_html=True)

    providers = [
        {
            "key": "gemini_2_flash",
            "icon": "✨",
            "name": "Gemini 2.0 Flash",
            "pricing": "free",
            "limit": "1,500 req/day",
            "desc": "Google's fast multimodal model. Excellent for rapid analysis with generous free tier. Default choice for NiveshAI.",
            "needs_key": False,
        },
        {
            "key": "gemini_25_flash",
            "icon": "⚡",
            "name": "Gemini 2.5 Flash",
            "pricing": "free",
            "limit": "500 req/day",
            "desc": "Latest Gemini model with enhanced reasoning capabilities. Great for deep financial analysis and complex queries.",
            "needs_key": False,
        },
        {
            "key": "openai_gpt4o",
            "icon": "🤖",
            "name": "OpenAI GPT-4o",
            "pricing": "paid",
            "limit": "Unlimited",
            "desc": "OpenAI's flagship multimodal model. Superior language understanding for nuanced market sentiment. Requires API key.",
            "needs_key": True,
        },
        {
            "key": "groq_llama3",
            "icon": "🚀",
            "name": "Groq LLaMA 3 70B",
            "pricing": "free",
            "limit": "14,400 req/day",
            "desc": "Meta's LLaMA 3 hosted on Groq's LPU™ for ultra-low latency inference. Blazing-fast responses. Requires free API key.",
            "needs_key": True,
        },
        {
            "key": "anthropic_claude",
            "icon": "🧠",
            "name": "Anthropic Claude 3.5",
            "pricing": "paid",
            "limit": "Unlimited",
            "desc": "Anthropic's most capable model. Excels at long-context financial document analysis and careful reasoning. Requires API key.",
            "needs_key": True,
        },
    ]

    # Render provider cards in a 2-column grid
    for row_start in range(0, len(providers), 2):
        cols = st.columns(2)
        for col_idx, provider in enumerate(providers[row_start : row_start + 2]):
            with cols[col_idx]:
                is_active = st.session_state.active_provider == provider["key"]

                pricing_badge = (
                    '<span class="badge badge-free">FREE</span>'
                    if provider["pricing"] == "free"
                    else '<span class="badge badge-paid">PAID</span>'
                )
                limit_badge = f'<span class="badge badge-limit">{provider["limit"]}</span>'
                active_html = '<div style="margin-top:10px;"><span class="active-indicator">✅ Currently Active</span></div>' if is_active else ""
                card_class = "provider-card active-provider" if is_active else "provider-card"
                needs_key_note = (
                    '<div style="color:#9DA3B4;font-size:0.78rem;margin-top:6px;">🔑 Requires API key</div>'
                    if provider["needs_key"]
                    else '<div style="color:#00D4AA;font-size:0.78rem;margin-top:6px;">🎉 No API key needed</div>'
                )

                st.markdown(f"""
                <div class="{card_class}">
                    <div class="provider-name">{provider["icon"]} {provider["name"]}</div>
                    <div style="margin:8px 0;">{pricing_badge} {limit_badge}</div>
                    <div class="provider-desc">{provider["desc"]}</div>
                    {needs_key_note}
                    {active_html}
                </div>
                """, unsafe_allow_html=True)

                btn_label = "✅ Active" if is_active else "Activate"
                btn_disabled = is_active
                if st.button(btn_label, key=f"btn_{provider['key']}", disabled=btn_disabled, use_container_width=True):
                    st.session_state.active_provider = provider["key"]
                    st.rerun()

    # Currently active summary
    st.markdown("---")
    active_prov = next((p for p in providers if p["key"] == st.session_state.active_provider), providers[0])
    st.markdown(f"""
    <div class="info-box-green">
        <strong>Active Provider:</strong> {active_prov["icon"]} {active_prov["name"]}  ·  {active_prov["limit"]}  ·  All NiveshAI analysis will use this model.
    </div>
    """, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — API KEYS
# ═════════════════════════════════════════════════════════════════════════════
with tab_keys:
    st.markdown('<p class="section-header">Manage API Keys</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Configure credentials for external model providers. Keys are stored locally in your session only.</p>', unsafe_allow_html=True)

    # Gemini — built-in free key
    st.markdown("""
    <div class="api-key-card">
        <div class="provider-name">✨ Gemini (2.0 Flash & 2.5 Flash)</div>
        <div class="info-box-green" style="margin-top:12px;">
            🎉 <strong>Built-in free key available</strong> — no configuration needed. NiveshAI ships with a default Gemini API key so you can get started instantly.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Keys for providers that need them
    key_providers = [
        {"id": "openai", "icon": "🤖", "name": "OpenAI GPT-4o", "placeholder": "sk-proj-..."},
        {"id": "groq", "icon": "🚀", "name": "Groq LLaMA 3 70B", "placeholder": "gsk_..."},
        {"id": "anthropic", "icon": "🧠", "name": "Anthropic Claude 3.5", "placeholder": "sk-ant-..."},
    ]

    for kp in key_providers:
        st.markdown(f'<div class="api-key-card"><div class="provider-name">{kp["icon"]} {kp["name"]}</div></div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns([5, 1.5, 1.5])
        with c1:
            key_val = st.text_input(
                f"API Key for {kp['name']}",
                value=st.session_state.api_keys.get(kp["id"], ""),
                type="password",
                placeholder=kp["placeholder"],
                key=f"key_input_{kp['id']}",
                label_visibility="collapsed",
            )
            st.session_state.api_keys[kp["id"]] = key_val

        with c2:
            if st.button("🔍 Validate", key=f"validate_{kp['id']}", use_container_width=True):
                if key_val.strip():
                    # Mock validation — just checks non-empty & length
                    st.session_state.key_statuses[kp["id"]] = "valid" if len(key_val.strip()) > 10 else "invalid"
                else:
                    st.session_state.key_statuses[kp["id"]] = "not_set"

        with c3:
            status = st.session_state.key_statuses.get(kp["id"], "not_set")
            if status == "valid":
                st.markdown('<span class="key-status" style="color:#00D4AA;">✅ Valid</span>', unsafe_allow_html=True)
            elif status == "invalid":
                st.markdown('<span class="key-status" style="color:#FF6B6B;">❌ Invalid</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="key-status" style="color:#FFB347;">⚠️ Not Set</span>', unsafe_allow_html=True)

    st.markdown("")
    col_save, col_spacer = st.columns([2, 5])
    with col_save:
        if st.button("💾 Save All Keys", use_container_width=True, type="primary"):
            st.session_state.keys_saved_toast = True
            st.rerun()

    if st.session_state.keys_saved_toast:
        st.success("All API keys saved to session successfully!")
        st.session_state.keys_saved_toast = False

    st.markdown("""
    <div class="info-box-amber">
        🔒 <strong>Security Note:</strong> API keys are stored locally in your browser session and <u>never</u> sent to our servers.
        Keys are cleared when you close the browser tab.
    </div>
    """, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — USAGE DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════
with tab_usage:
    st.markdown('<p class="section-header">Today\'s Usage Summary</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Monitor your API usage and costs across all providers.</p>', unsafe_allow_html=True)

    # ── Today's Metrics ──
    u_col1, u_col2, u_col3 = st.columns(3)

    requests_used = 847
    requests_limit = 1500
    req_pct = int(requests_used / requests_limit * 100)

    tokens_used = 623450
    tokens_limit = 1000000
    tok_pct = int(tokens_used / tokens_limit * 100)

    with u_col1:
        st.markdown(f"""
        <div class="usage-metric-card">
            <div class="usage-metric-value" style="color:#6C63FF;">{requests_used:,} <span style="font-size:1rem;color:#9DA3B4;">/ {requests_limit:,}</span></div>
            <div class="usage-metric-label">Requests Used Today</div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(req_pct / 100, text=f"{req_pct}% of daily limit")

    with u_col2:
        st.markdown(f"""
        <div class="usage-metric-card">
            <div class="usage-metric-value" style="color:#00D4AA;">{tokens_used:,} <span style="font-size:1rem;color:#9DA3B4;">/ {tokens_limit:,}</span></div>
            <div class="usage-metric-label">Tokens Used (TPM)</div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(tok_pct / 100, text=f"{tok_pct}% of token limit")

    with u_col3:
        st.markdown("""
        <div class="usage-metric-card">
            <div class="usage-metric-value" style="color:#FFB347;">4h 23m</div>
            <div class="usage-metric-label">Reset Countdown</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box-blue" style="margin-top:4px;text-align:center;">
            ⏰ Daily limits reset at <strong>00:00 UTC</strong>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Usage History Chart ──
    st.markdown('<p class="section-header">📈 Usage History — Last 7 Days</p>', unsafe_allow_html=True)

    random.seed(42)
    days = [(datetime.now() - timedelta(days=i)).strftime("%a %d %b") for i in range(6, -1, -1)]
    daily_requests = [random.randint(400, 1400) for _ in range(7)]
    daily_tokens = [random.randint(300000, 950000) for _ in range(7)]

    fig_usage = go.Figure()

    fig_usage.add_trace(go.Bar(
        x=days,
        y=daily_requests,
        name="Requests",
        marker=dict(
            color=daily_requests,
            colorscale=[[0, "#6C63FF"], [0.5, "#00D4AA"], [1, "#FFB347"]],
            line=dict(width=0),
            cornerradius=6,
        ),
        text=[f"{r:,}" for r in daily_requests],
        textposition="outside",
        textfont=dict(color="#9DA3B4", size=11),
    ))

    fig_usage.add_hline(
        y=1500, line_dash="dash", line_color="rgba(255,107,107,0.5)",
        annotation_text="Daily Limit (1,500)",
        annotation_font=dict(color="#FF6B6B", size=11),
        annotation_position="top right",
    )

    fig_usage.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=380,
        margin=dict(l=40, r=20, t=30, b=40),
        yaxis=dict(
            title="Requests",
            gridcolor="rgba(108,99,255,0.08)",
            zeroline=False,
        ),
        xaxis=dict(gridcolor="rgba(108,99,255,0.05)"),
        showlegend=False,
    )
    st.plotly_chart(fig_usage, use_container_width=True)

    # ── Tokens History Bar ──
    fig_tok = go.Figure()
    fig_tok.add_trace(go.Bar(
        x=days,
        y=daily_tokens,
        name="Tokens",
        marker=dict(
            color="#4ECDC4",
            opacity=0.8,
            cornerradius=6,
        ),
        text=[f"{t / 1000:.0f}K" for t in daily_tokens],
        textposition="outside",
        textfont=dict(color="#9DA3B4", size=11),
    ))
    fig_tok.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=300,
        margin=dict(l=40, r=20, t=10, b=40),
        yaxis=dict(title="Tokens", gridcolor="rgba(108,99,255,0.08)", zeroline=False),
        xaxis=dict(gridcolor="rgba(108,99,255,0.05)"),
        showlegend=False,
    )
    st.markdown('<p class="section-header">🪙 Token Consumption — Last 7 Days</p>', unsafe_allow_html=True)
    st.plotly_chart(fig_tok, use_container_width=True)

    st.markdown("---")

    # ── Cost Tracker ──
    st.markdown('<p class="section-header">💰 Cost Tracker (Paid Models)</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        <table class="cost-table">
            <thead>
                <tr>
                    <th>Provider</th>
                    <th>Today</th>
                    <th>This Week</th>
                    <th>This Month</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>🤖 OpenAI GPT-4o</td>
                    <td style="color:#00D4AA;">$2.34</td>
                    <td style="color:#00D4AA;">$12.50</td>
                    <td style="color:#00D4AA;">$43.80</td>
                    <td><span class="badge badge-paid">ACTIVE</span></td>
                </tr>
                <tr>
                    <td>🧠 Anthropic Claude 3.5</td>
                    <td style="color:#9DA3B4;">$0.00</td>
                    <td style="color:#9DA3B4;">$0.00</td>
                    <td style="color:#9DA3B4;">$0.00</td>
                    <td><span style="color:#9DA3B4;font-size:0.82rem;">INACTIVE</span></td>
                </tr>
                <tr>
                    <td style="font-weight:700;">Total</td>
                    <td style="color:#FFE66D;font-weight:700;">$2.34</td>
                    <td style="color:#FFE66D;font-weight:700;">$12.50</td>
                    <td style="color:#FFE66D;font-weight:700;">$43.80</td>
                    <td></td>
                </tr>
            </tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")
    exp_col1, exp_col2 = st.columns([2, 5])
    with exp_col1:
        st.download_button(
            "📄 Export Usage Report",
            data="NiveshAI Usage Report\n====================\nGenerated: 2026-07-08\n\nDaily Requests: 847 / 1,500\nTokens Used: 623,450 / 1,000,000\n\nWeekly Cost (OpenAI): $12.50\nMonthly Cost (Total): $43.80\n\n--- End of Report ---",
            file_name="niveshai_usage_report.txt",
            mime="text/plain",
            use_container_width=True,
        )

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — ABOUT
# ═════════════════════════════════════════════════════════════════════════════
with tab_about:
    # ── App Info ──
    st.markdown("""
    <div class="about-card" style="text-align:center;">
        <div style="font-size:3rem;margin-bottom:8px;">📊</div>
        <div class="gradient-text" style="font-size:2rem;">NiveshAI</div>
        <div style="color:#9DA3B4;font-size:1rem;margin-top:4px;">v1.0.0</div>
        <div style="color:#E8E8E8;font-size:1.05rem;margin-top:12px;">
            AI-Powered Investment Research for Indian Markets
        </div>
        <div style="color:#9DA3B4;font-size:0.88rem;margin-top:6px;">
            Combining state-of-the-art LLMs with quantitative models to deliver actionable insights on NSE stocks.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tech Stack ──
    st.markdown('<p class="section-header">🛠️ Tech Stack</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="about-card">
        <span class="tech-badge">🐍 Python 3.11</span>
        <span class="tech-badge">🌊 Streamlit</span>
        <span class="tech-badge">🔥 PyTorch</span>
        <span class="tech-badge">📊 Plotly</span>
        <span class="tech-badge">📈 yfinance</span>
        <span class="tech-badge">🤗 HuggingFace Transformers</span>
        <span class="tech-badge">🧠 scikit-learn</span>
        <span class="tech-badge">🗃️ Pandas</span>
        <span class="tech-badge">🔢 NumPy</span>
        <span class="tech-badge">📰 NewsAPI</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Trained Models ──
    st.markdown('<p class="section-header">🧠 Trained Models</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="about-card">
        <div class="model-status-row">
            <div>
                <div class="model-status-name">🎭 Sentiment Analysis Model</div>
                <div class="model-status-desc">DistilBERT fine-tuned on Financial PhraseBank (4,845 samples)</div>
            </div>
            <div class="status-loaded">🟢 Loaded</div>
        </div>
        <div class="model-status-row">
            <div>
                <div class="model-status-name">📈 LSTM Price Predictor</div>
                <div class="model-status-desc">2-layer LSTM trained on NIFTY 50 stocks · 60-day sequence window</div>
            </div>
            <div class="status-loaded">🟢 Loaded</div>
        </div>
        <div class="model-status-row">
            <div>
                <div class="model-status-name">🌲 Random Forest Classifier</div>
                <div class="model-status-desc">Direction predictor trained on NIFTY 50 technical indicators</div>
            </div>
            <div class="status-loaded">🟢 Loaded</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Data Sources ──
    st.markdown('<p class="section-header">📡 Data Sources</p>', unsafe_allow_html=True)
    ds_c1, ds_c2, ds_c3 = st.columns(3)
    with ds_c1:
        st.markdown("""
        <div class="about-card" style="text-align:center;">
            <div style="font-size:1.8rem;">📈</div>
            <div style="color:#E8E8E8;font-weight:700;margin-top:6px;">yfinance</div>
            <div style="color:#9DA3B4;font-size:0.82rem;margin-top:4px;">NSE stock prices, OHLCV data, fundamentals</div>
        </div>
        """, unsafe_allow_html=True)
    with ds_c2:
        st.markdown("""
        <div class="about-card" style="text-align:center;">
            <div style="font-size:1.8rem;">📰</div>
            <div style="color:#E8E8E8;font-weight:700;margin-top:6px;">NewsAPI</div>
            <div style="color:#9DA3B4;font-size:0.82rem;margin-top:4px;">Real-time financial news articles & headlines</div>
        </div>
        """, unsafe_allow_html=True)
    with ds_c3:
        st.markdown("""
        <div class="about-card" style="text-align:center;">
            <div style="font-size:1.8rem;">🔗</div>
            <div style="color:#E8E8E8;font-weight:700;margin-top:6px;">Google News RSS</div>
            <div style="color:#9DA3B4;font-size:0.82rem;margin-top:4px;">Company-specific news via RSS feeds</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Links ──
    st.markdown('<p class="section-header">🔗 Links</p>', unsafe_allow_html=True)
    link_c1, link_c2, link_c3 = st.columns(3)
    with link_c1:
        st.link_button("⭐ GitHub Repository", "#", use_container_width=True)
    with link_c2:
        st.link_button("📖 Documentation", "#", use_container_width=True)
    with link_c3:
        st.link_button("🐛 Report an Issue", "#", use_container_width=True)

    # ── Credits ──
    st.markdown("---")
    st.markdown("""
    <div class="about-card" style="text-align:center;">
        <p class="section-header" style="margin-top:0;">💜 Credits</p>
        <div style="color:#9DA3B4;font-size:0.92rem;line-height:1.8;">
            Built with ❤️ for the Indian investing community.<br>
            Powered by <span style="color:#6C63FF;font-weight:600;">Google Gemini</span>,
            <span style="color:#00D4AA;font-weight:600;">Streamlit</span>, and
            <span style="color:#FFB347;font-weight:600;">Open-Source AI</span>.<br>
            Market data provided by <span style="color:#4ECDC4;font-weight:600;">Yahoo Finance</span> via yfinance.<br><br>
            <span style="color:#E8E8E8;font-weight:700;">NiveshAI</span> — <em>Nivesh (निवेश)</em> means "Investment" in Hindi 🇮🇳
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.caption("Made with ☕ and Python · © 2026 NiveshAI Team")
