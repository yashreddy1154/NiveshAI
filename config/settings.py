"""
NiveshAI — Configuration & Settings
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models" / "saved"
REPORTS_DIR = PROJECT_ROOT / "reports" / "output"
CACHE_DB = DATA_DIR / "cache.db"
NIFTY500_CSV = DATA_DIR / "nifty500.csv"

# Ensure directories exist
for d in [DATA_DIR, MODELS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── API Keys ───────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")

# ── Default LLM Provider ──────────────────────────────────────────────
DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "gemini")

# ── Model Provider Configs ─────────────────────────────────────────────
LLM_PROVIDERS = {
    "gemini": {
        "name": "Google Gemini 2.0 Flash",
        "model_id": "gemini-2.0-flash",
        "is_free": True,
        "daily_limit": 1500,
        "rpm_limit": 15,
        "icon": "✨",
        "requires_key": True,
    },
    "gemini_25_flash": {
        "name": "Google Gemini 2.5 Flash",
        "model_id": "gemini-2.5-flash",
        "is_free": True,
        "daily_limit": 500,
        "rpm_limit": 10,
        "icon": "⚡",
        "requires_key": True,
    },
    "openai": {
        "name": "OpenAI GPT-4o",
        "model_id": "gpt-4o",
        "is_free": False,
        "daily_limit": float("inf"),
        "rpm_limit": 60,
        "icon": "🤖",
        "requires_key": True,
    },
    "openai_mini": {
        "name": "OpenAI GPT-4o Mini",
        "model_id": "gpt-4o-mini",
        "is_free": False,
        "daily_limit": float("inf"),
        "rpm_limit": 60,
        "icon": "🤖",
        "requires_key": True,
    },
    "groq": {
        "name": "Groq LLaMA 3 70B",
        "model_id": "llama-3.1-70b-versatile",
        "is_free": True,
        "daily_limit": 14400,
        "rpm_limit": 30,
        "icon": "🚀",
        "requires_key": True,
    },
    "anthropic": {
        "name": "Anthropic Claude 3.5",
        "model_id": "claude-3-5-sonnet-20241022",
        "is_free": False,
        "daily_limit": float("inf"),
        "rpm_limit": 50,
        "icon": "🧠",
        "requires_key": True,
    },
}

# ── Trained Model Configs ──────────────────────────────────────────────
SENTIMENT_MODEL_PATH = MODELS_DIR / "sentiment_model.pt"
LSTM_MODEL_PATH = MODELS_DIR / "lstm_model.pt"
RF_MODEL_PATH = MODELS_DIR / "rf_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"

# ── Stock Data Settings ────────────────────────────────────────────────
DEFAULT_PERIOD = "2y"
LOOKBACK_WINDOW = 60  # days for LSTM input
PREDICTION_DAYS = 7   # days to predict ahead
NSE_SUFFIX = ".NS"

# ── Technical Indicator Periods ────────────────────────────────────────
SMA_SHORT = 20
SMA_LONG = 50
SMA_TREND = 200
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2

# ── UI Settings ────────────────────────────────────────────────────────
APP_NAME = "NiveshAI"
APP_TAGLINE = "AI-Powered Investment Research for Indian Markets"
APP_VERSION = "1.0.0"
APP_ICON = "📈"

# Sector color map for visualizations
SECTOR_COLORS = {
    "Information Technology": "#6C63FF",
    "Financial Services": "#00D4AA",
    "Energy": "#FFB347",
    "Healthcare": "#FF6B6B",
    "Consumer Goods": "#4ECDC4",
    "Automobile": "#FFE66D",
    "Metals & Mining": "#A8E6CF",
    "Telecom": "#DDA0DD",
    "Infrastructure": "#87CEEB",
    "Pharma": "#FF69B4",
    "FMCG": "#98FB98",
    "Banking": "#20B2AA",
    "Default": "#6C63FF",
}
