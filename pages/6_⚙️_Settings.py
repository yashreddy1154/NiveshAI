import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import random
from pathlib import Path

# Internal Config
from config.settings import PROJECT_ROOT, MODELS_DIR, CACHE_DB

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
    .status-missing {
        color: #FF6B6B;
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
</style>
""", unsafe_allow_html=True)

# ─── Load & Save .env Helper ─────────────────────────────────────────────────
env_path = PROJECT_ROOT / ".env"

def load_keys_from_env():
    keys = {
        "gemini": "",
        "openai": "",
        "groq": "",
        "anthropic": "",
        "newsapi": "",
    }
    if env_path.exists():
        content = env_path.read_text()
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip()
            if k == "GEMINI_API_KEY": keys["gemini"] = v
            elif k == "OPENAI_API_KEY": keys["openai"] = v
            elif k == "GROQ_API_KEY": keys["groq"] = v
            elif k == "ANTHROPIC_API_KEY": keys["anthropic"] = v
            elif k == "NEWSAPI_KEY": keys["newsapi"] = v
    else:
        # Check environment variables directly as fallback
        keys["gemini"] = os.getenv("GEMINI_API_KEY", "")
        keys["openai"] = os.getenv("OPENAI_API_KEY", "")
        keys["groq"] = os.getenv("GROQ_API_KEY", "")
        keys["anthropic"] = os.getenv("ANTHROPIC_API_KEY", "")
        keys["newsapi"] = os.getenv("NEWSAPI_KEY", "")
    return keys

def save_keys_to_env(keys):
    lines = []
    # If file exists, retain other keys, else build standard
    existing = {}
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                existing[k.strip()] = v.strip()

    # Update keys
    existing["GEMINI_API_KEY"] = keys["gemini"]
    existing["OPENAI_API_KEY"] = keys["openai"]
    existing["GROQ_API_KEY"] = keys["groq"]
    existing["ANTHROPIC_API_KEY"] = keys["anthropic"]
    existing["NEWSAPI_KEY"] = keys["newsapi"]
    
    # Write back
    content = ""
    for k, v in existing.items():
        content += f"{k}={v}\n"
    env_path.write_text(content)

# ─── Load initial values to state ────────────────────────────────────────────
if "api_keys" not in st.session_state:
    st.session_state.api_keys = load_keys_from_env()

if "active_provider" not in st.session_state:
    if env_path.exists():
        # Get active provider from env
        content = env_path.read_text()
        active = "gemini_2_flash"
        for line in content.splitlines():
            if line.startswith("DEFAULT_LLM_PROVIDER="):
                active_val = line.split("=", 1)[1].strip()
                if active_val == "gemini": active = "gemini_2_flash"
                elif active_val == "gemini_25_flash": active = "gemini_25_flash"
                elif active_val == "openai": active = "openai_gpt4o"
                elif active_val == "groq": active = "groq_llama3"
                elif active_val == "anthropic": active = "anthropic_claude"
        st.session_state.active_provider = active
    else:
        st.session_state.active_provider = "gemini_2_flash"

# ─── Model File Checker Helpers ──────────────────────────────────────────────
def get_file_status(filename):
    path = MODELS_DIR / filename
    if path.exists():
        size_bytes = path.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        if size_mb >= 1.0:
            return f"🟢 Loaded ({size_mb:.2f} MB)"
        else:
            return f"🟢 Loaded ({size_bytes / 1024:.2f} KB)"
    return "🔴 Missing"

# Cache DB size helper
def get_cache_size():
    if CACHE_DB.exists():
        size_bytes = CACHE_DB.stat().st_size
        if size_bytes >= 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.2f} MB"
        return f"{size_bytes / 1024:.2f} KB"
    return "0.00 KB"

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
                    
                    # Update .env
                    env_val = "gemini"
                    if provider["key"] == "gemini_25_flash": env_val = "gemini_25_flash"
                    elif provider["key"] == "openai_gpt4o": env_val = "openai"
                    elif provider["key"] == "groq_llama3": env_val = "groq"
                    elif provider["key"] == "anthropic_claude": env_val = "anthropic"
                    
                    # Write to env
                    existing = {}
                    if env_path.exists():
                        for line in env_path.read_text().splitlines():
                            line = line.strip()
                            if line and "=" in line:
                                k, v = line.split("=", 1)
                                existing[k.strip()] = v.strip()
                    existing["DEFAULT_LLM_PROVIDER"] = env_val
                    content = ""
                    for k, v in existing.items():
                        content += f"{k}={v}\n"
                    env_path.write_text(content)
                    st.rerun()

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — API KEYS
# ═════════════════════════════════════════════════════════════════════════════
with tab_keys:
    st.markdown('<p class="section-header">Manage API Credentials</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">API keys will be written to the local project <code>.env</code> file for persistence across application restarts.</p>', unsafe_allow_html=True)

    # Load credentials with dotenv
    try:
        from dotenv import set_key, dotenv_values
        has_dotenv = True
    except ImportError:
        has_dotenv = False

    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        try:
            env_path.touch()
        except Exception:
            pass

    current_env = {}
    if has_dotenv:
        try:
            current_env = dotenv_values(env_path)
        except Exception:
            pass

    providers_config = [
        ("gemini", "Gemini API Key", "GEMINI_API_KEY", "https://aistudio.google.com/apikey"),
        ("openai", "OpenAI API Key", "OPENAI_API_KEY", "https://platform.openai.com/api-keys"),
        ("groq", "Groq API Key", "GROQ_API_KEY", "https://console.groq.com/keys"),
        ("anthropic", "Anthropic API Key", "ANTHROPIC_API_KEY", "https://console.anthropic.com/"),
        ("newsapi", "NewsAPI Key", "NEWSAPI_KEY", "https://newsapi.org/"),
    ]

    for key_id, label, env_var, url in providers_config:
        # Resolve current key value
        current_val = current_env.get(env_var) or os.getenv(env_var) or st.session_state.api_keys.get(key_id, "")
        
        # Display masked status
        if current_val:
            masked = "*" * (len(current_val) - 4) + current_val[-4:] if len(current_val) > 4 else "****"
            st.markdown(f"**{label}**: <span style='color:#00D4AA;font-weight:bold;'>✅ Configured</span> (ending in `{current_val[-4:] if len(current_val) > 4 else current_val}`)", unsafe_allow_html=True)
        else:
            st.markdown(f"**{label}**: <span style='color:#FF6B6B;font-weight:bold;'>❌ Not set</span>", unsafe_allow_html=True)
            
        st.text_input(
            label,
            value="",
            type="password",
            placeholder="••••••••••••••••" if current_val else "Enter API key...",
            help=f"Get key at: {url}",
            key=f"input_{key_id}",
            label_visibility="collapsed"
        )
        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

    st.markdown("")
    if st.button("💾 Save Credentials to .env", type="primary", use_container_width=True):
        updated = False
        saved_keys = st.session_state.api_keys.copy()
        
        for key_id, label, env_var, url in providers_config:
            user_input = st.session_state.get(f"input_{key_id}", "")
            if user_input.strip():
                saved_keys[key_id] = user_input.strip()
                if has_dotenv:
                    try:
                        set_key(str(env_path), env_var, user_input.strip())
                    except Exception as e:
                        st.error(f"Error saving {env_var}: {e}")
                updated = True
                
        if updated:
            st.session_state.api_keys = saved_keys
            st.success("API keys saved! Restart the app to apply.")
            st.rerun()
        else:
            st.info("No new keys entered to save.")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — USAGE DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════
with tab_usage:
    st.markdown('<p class="section-header">Performance & Storage Controls</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Manage local SQLite database cache size and optimize performance latency.</p>', unsafe_allow_html=True)

    from data.cache import get_cache
    cache = get_cache()
    
    c_size = get_cache_size()
    
    col_cache1, col_cache2 = st.columns([3, 1])
    with col_cache1:
        st.metric("Cached Items", cache.size(), help="Total number of items currently stored in SQLite cache")
        st.metric("Cache Database Size", c_size)
    with col_cache2:
        st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
        if st.button("🗑️ Clear Cache", use_container_width=True):
            cache.clear()
            st.success("Cache cleared!")
            st.rerun()

    st.markdown("---")
    st.markdown('<p class="section-header">API Usage Tracker</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Daily request count and statistics tracked by the cache manager.</p>', unsafe_allow_html=True)
    
    usage = cache.get_all_usage()
    if usage:
        for provider, stats in usage.items():
            st.markdown(f"🔹 **{provider.upper()}**: `{stats['requests']}` requests today (approx. cost: `${stats['cost_usd']:.4f}`)", unsafe_allow_html=True)
    else:
        st.info("No API usage logged today.")

    st.markdown("---")
    st.markdown('<p class="section-header">Gemini Free Tier Display</p>', unsafe_allow_html=True)
    
    gemini_usage = cache.get_daily_usage("gemini")
    requests_used = gemini_usage.get("requests", 0)
    requests_limit = 1500
    req_pct = min(100, int(requests_used / requests_limit * 100))
    
    u_col1, u_col2 = st.columns(2)
    with u_col1:
        st.markdown(f"""
        <div class="usage-metric-card">
            <div class="usage-metric-value" style="color:#6C63FF;">{requests_used:,} <span style="font-size:1rem;color:#9DA3B4;">/ {requests_limit:,}</span></div>
            <div class="usage-metric-label">Requests Used Today</div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(req_pct / 100.0, text=f"{req_pct}% of daily limit")

    with u_col2:
        st.markdown("""
        <div class="usage-metric-card">
            <div class="usage-metric-value" style="color:#FFB347;">4h 23m</div>
            <div class="usage-metric-label">Reset Countdown</div>
        </div>
        """, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — ABOUT
# ═════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("""
    <div class="about-card" style="text-align:center;">
        <div style="font-size:3rem;margin-bottom:8px;">⚙️</div>
        <div class="gradient-text" style="font-size:2rem;">NiveshAI</div>
        <div style="color:#9DA3B4;font-size:1rem;margin-top:4px;">v1.0.0</div>
        <div style="color:#E8E8E8;font-size:1.05rem;margin-top:12px;">
            AI-Powered Investment Research for Indian Markets
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Trained Models ──
    st.markdown('<p class="section-header">🧠 Trained Models Directory Status</p>', unsafe_allow_html=True)
    
    # Read training dates if validation_results.txt exists
    val_results_path = MODELS_DIR / "saved" / "validation_results.txt"
    val_dates = {}
    if val_results_path.exists():
        try:
            content = val_results_path.read_text()
            # parse dates if any
        except Exception:
            pass

    def get_model_status_with_date(filename):
        path = MODELS_DIR / "saved" / filename
        if path.exists():
            size_bytes = path.stat().st_size
            size_mb = size_bytes / (1024 * 1024)
            mtime = datetime.fromtimestamp(path.stat().st_mtime)
            date_str = mtime.strftime("%d %b %Y, %H:%M")
            
            size_str = f"{size_mb:.2f} MB" if size_mb >= 1.0 else f"{size_bytes / 1024:.2f} KB"
            return f"✅ Loaded ({size_str}) — Trained: {date_str}"
        return "❌ Missing"

    sentiment_stat = get_model_status_with_date("sentiment_model.pt")
    lstm_stat = get_model_status_with_date("lstm_model.pt")
    rf_stat = get_model_status_with_date("rf_model.pkl")
    scaler_stat = get_model_status_with_date("scaler.pkl")
    rf_scaler_stat = get_model_status_with_date("rf_scaler.pkl")

    def get_row_html(name, desc, status_text):
        cls = "status-loaded" if "Loaded" in status_text else "status-missing"
        return f"""
        <div class="model-status-row">
            <div>
                <div class="model-status-name">{name}</div>
                <div class="model-status-desc">{desc}</div>
            </div>
            <div class="{cls}">{status_text}</div>
        </div>
        """

    st.markdown(f"""
    <div class="about-card">
        {get_row_html("🎭 Sentiment Analysis Model", "DistilBERT fine-tuned classifier", sentiment_stat)}
        {get_row_html("📈 LSTM Price Predictor", "Stacked LSTM weights", lstm_stat)}
        {get_row_html("🌲 Random Forest Classifier", "Directional decision tree model", rf_stat)}
        {get_row_html("📐 MinMaxScaler (LSTM)", "Scaler for price features normalization", scaler_stat)}
        {get_row_html("📐 StandardScaler (RF)", "Scaler for classification features", rf_scaler_stat)}
    </div>
    """, unsafe_allow_html=True)
