<p align="center">
  <h1 align="center">📈 NiveshAI</h1>
  <p align="center"><strong>AI-Powered Investment Research for Indian Markets</strong></p>
  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" alt="Python" />
    <img src="https://img.shields.io/badge/Streamlit-1.38+-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit" />
    <img src="https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch" />
    <img src="https://img.shields.io/badge/Plotly-5.18+-3F4F75?logo=plotly&logoColor=white" alt="Plotly" />
    <img src="https://img.shields.io/badge/Market-NSE%20India-00D4AA" alt="NSE" />
  </p>
</p>

---

NiveshAI is a smart investment research assistant that combines **AI/ML models**, **LLM-powered analysis**, and **real-time market data** to generate actionable investment insights for the Indian stock market (NSE).

## ✨ Key Features

### 🧠 Dual AI Approach
- **Trained ML Models** — Fine-tuned DistilBERT for sentiment analysis + LSTM neural network for price prediction
- **LLM API Integration** — Multi-provider support (Gemini, OpenAI, Groq, Anthropic) with tool-calling agent pattern

### 📊 Core Capabilities
| Feature | Description |
|---------|-------------|
| **Financial Data Pipeline** | Real-time & historical stock data from NSE via yfinance |
| **Sentiment Analysis** | DistilBERT fine-tuned on Financial PhraseBank for financial news |
| **Price Prediction** | LSTM (PyTorch) + Random Forest with 7 technical indicators |
| **AI Research Agent** | LLM-powered Q&A with 8 analysis tools for comprehensive research |
| **Portfolio Optimizer** | Modern Portfolio Theory (Markowitz) with efficient frontier |
| **Report Generator** | Automated PDF investment reports |
| **Interactive Dashboard** | Plotly-powered charts with candlestick, technicals, sentiment viz |

### 🔌 Multi-Model Provider System
Switch between LLM providers from the GUI:
- ✅ **Google Gemini** — FREE built-in (1,500 req/day)
- ✅ **Groq LLaMA 3** — FREE (14,400 req/day, user key)
- 💰 **OpenAI GPT-4o** — Paid (user key)
- 💰 **Anthropic Claude** — Paid (user key)
- 🔧 **Custom endpoints** — Any OpenAI-compatible API

### 🇮🇳 Indian Market Focus
- NIFTY 500 company database with sector/industry/cap classification
- Smart company search with autocomplete
- ₹ INR formatting, Indian number system (Lakh/Crore)
- Indian financial news from NewsAPI + Google News RSS

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                 Streamlit Dashboard                  │
│  Home │ Dashboard │ AI Agent │ Predictions │ ...     │
├─────────────────────────────────────────────────────┤
│            AI Research Agent (Orchestrator)           │
│    ┌──────────┬──────────┬──────────┬──────────┐     │
│    │ Gemini   │ OpenAI   │ Groq     │ Custom   │     │
│    └──────────┴──────────┴──────────┴──────────┘     │
├──────────────┬──────────────┬───────────────────┤
│ Trained ML   │ Analysis     │ Data Pipeline     │
│ DistilBERT   │ Fundamental  │ yfinance (NSE)    │
│ LSTM         │ Technical    │ NewsAPI           │
│ Random Forest│ Risk         │ Google News RSS   │
└──────────────┴──────────────┴───────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/NiveshAI.git
cd NiveshAI

# Install dependencies
pip install -r requirements.txt

# Install PyTorch (CPU version — for GPU, see pytorch.org)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy environment template
cp .env.example .env
# Edit .env and add your API keys (Gemini key is optional — free tier built-in)
```

### Run the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` 🎉

---

## 📁 Project Structure

```
NiveshAI/
├── app.py                          # Home page
├── pages/                          # Streamlit multi-page app
│   ├── 1_📊_Dashboard.py          # Stock analysis dashboard
│   ├── 2_🤖_AI_Research_Agent.py  # Chat-based AI agent
│   ├── 3_📈_Predictions.py        # ML model predictions
│   ├── 4_🎯_Portfolio_Optimizer.py# Portfolio optimization
│   ├── 5_📋_Report_Generator.py   # PDF report generation
│   └── 6_⚙️_Settings.py          # Model & API configuration
├── config/settings.py              # App configuration
├── data/                           # Data pipeline
│   ├── nifty500.csv                # Pre-built company database
│   ├── stock_data.py               # yfinance wrapper
│   ├── news_fetcher.py             # News collection
│   ├── cache.py                    # SQLite cache
│   └── company_db.py              # NIFTY 500 query interface
├── models/                         # ML models
│   ├── sentiment/                  # DistilBERT sentiment classifier
│   ├── prediction/                 # LSTM + Random Forest predictors
│   └── saved/                      # Trained model weights
├── agents/                         # AI agent system
│   ├── research_agent.py           # Tool-calling research agent
│   ├── tools.py                    # 8 analysis tools
│   ├── prompts.py                  # System prompts
│   └── providers/                  # Multi-LLM provider system
├── analysis/                       # Financial analysis
├── portfolio/                      # Portfolio optimization
├── reports/                        # PDF generation
├── visualization/                  # Plotly chart builders
└── scripts/                        # Training & utility scripts
```

---

## 🧠 ML Models

### 1. Sentiment Classifier (Trained)
- **Model**: DistilBERT fine-tuned on Financial PhraseBank
- **Dataset**: ~4,840 financial sentences (positive/negative/neutral)
- **Training**: 4 epochs, batch_size=16, lr=2e-5, fp16 mixed precision
- **Output**: Sentiment label + confidence score per news article

### 2. LSTM Price Predictor (Trained)
- **Architecture**: 2-layer LSTM (128 units) → Dropout(0.2) → Dense(1)
- **Features**: Close, Volume, SMA_20, SMA_50, RSI_14, MACD, Bollinger_%B
- **Input**: 60-day lookback window
- **Output**: Next 1-7 day price prediction

### 3. Random Forest Direction Predictor (Trained)
- **Type**: Classification (UP / DOWN / FLAT)
- **Features**: Same 7 technical indicators
- **Purpose**: Baseline comparison + feature importance analysis

---

## 🔑 API Keys

| Service | Required? | Free Tier |
|---------|-----------|-----------|
| Google Gemini | Optional (built-in) | 1,500 req/day |
| NewsAPI | Recommended | 100 req/day |
| OpenAI | Optional | No free tier |
| Groq | Optional | 14,400 req/day |

---

## 📊 Screenshots

> *Screenshots will be added after final UI polish*

---

## 🛠️ Training Models

```bash
# Quick training (NIFTY 50, ~30 min GPU)
python scripts/train_all.py --mode quick

# Full training (NIFTY 500, ~3 hrs GPU)
python scripts/train_all.py --mode full

# Download stock data
python scripts/download_data.py
python scripts/download_data.py --all  # All NIFTY 500
```

---

## 🌐 Deployment

### Streamlit Cloud
1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo → Deploy

### Hugging Face Spaces
1. Create a new Space (Streamlit SDK)
2. Push your code
3. Upload model weights to HF Model Hub

---

## 📝 Tech Stack

| Category | Technology |
|----------|-----------|
| Language | Python 3.10+ |
| Frontend | Streamlit + Plotly |
| LLM API | Google Gemini (primary) |
| ML Framework | PyTorch |
| ML Library | scikit-learn, Hugging Face Transformers |
| Financial Data | yfinance |
| News | NewsAPI, Google News RSS |
| Portfolio | scipy.optimize |
| PDF | fpdf2 |
| Caching | SQLite |

---

## ⚠️ Disclaimer

This tool is for **educational and research purposes only**. It does not constitute financial advice. Always consult a qualified financial advisor before making investment decisions. Past performance does not guarantee future results.

---

## 📄 License

MIT License

---

<p align="center">
  Built with ❤️ for the Indian investment community
</p>
