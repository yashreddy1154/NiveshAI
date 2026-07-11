import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Internal modules
from data.company_db import get_display_options, parse_display_option, get_company
from data.stock_data import fetch_history, fetch_live_price, fetch_fundamentals
from reports.pdf_generator import generate_report, generate_comparison_report
from analysis.technical import compute_all_indicators, generate_signals
from analysis.fundamental import analyze_fundamentals

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Report Generator — NiveshAI",
    page_icon="📋",
    layout="wide",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
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

    .report-header {
        background: linear-gradient(135deg, rgba(108, 99, 255, 0.15), rgba(0, 212, 170, 0.08));
        border: 1px solid rgba(108, 99, 255, 0.25);
        border-radius: 20px;
        padding: 32px 40px;
        margin-bottom: 24px;
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

# ─── Check FPDF2 Installation ───
try:
    from fpdf import FPDF
    fpdf_installed = True
except ImportError:
    fpdf_installed = False

# ─── PDF Report Generation Helper ───
def generate_pdf_report(symbol, history, live, fundamentals, signals, fund_analysis, risk, agg_sent, news_articles, chart_generated, chart_img_path):
    pdf = FPDF()
    pdf.add_page()
    
    # ─── Cover Page / Header ───
    pdf.set_font("Helvetica", "B", 24)
    pdf.set_text_color(108, 99, 255) # Purple theme
    pdf.cell(0, 15, "NiveshAI Research Report", ln=True, align="C")
    
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(14, 17, 23)
    comp_name = fundamentals.get("company_name", symbol) if (fundamentals and isinstance(fundamentals, dict)) else symbol
    pdf.cell(0, 10, f"{comp_name} ({symbol})", ln=True, align="C")
    
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(128, 128, 128)
    sector = fundamentals.get("sector", "Data Unavailable") if (fundamentals and isinstance(fundamentals, dict)) else "Data Unavailable"
    industry = fundamentals.get("industry", "Data Unavailable") if (fundamentals and isinstance(fundamentals, dict)) else "Data Unavailable"
    pdf.cell(0, 6, f"Sector: {sector} | Industry: {industry}", ln=True, align="C")
    pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%d %b %Y %H:%M IST')}", ln=True, align="C")
    pdf.ln(10)
    
    # ─── 1. Price performance (Last 90 Days) ───
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(108, 99, 255)
    pdf.cell(0, 8, "1. Price Performance Chart", ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)
    
    if chart_generated and os.path.exists(chart_img_path):
        pdf.image(chart_img_path, x=15, w=180)
        pdf.ln(95) # spacing for image height
    else:
        pdf.set_font("Helvetica", "I", 10)
        pdf.set_text_color(128, 128, 128)
        pdf.cell(0, 6, "Price Chart: Data Unavailable", ln=True)
        pdf.ln(4)
        
    # ─── 2. Fundamental Analysis Table ───
    if pdf.get_y() > 220:
        pdf.add_page()
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(108, 99, 255)
    pdf.cell(0, 8, "2. Fundamental Analysis", ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)
    
    if fundamentals and isinstance(fundamentals, dict) and not fundamentals.get("error"):
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_fill_color(240, 240, 245)
        pdf.set_text_color(14, 17, 23)
        pdf.cell(95, 6, "Metric Key", 1, 0, "L", True)
        pdf.cell(95, 6, "Value", 1, 1, "R", True)
        
        pdf.set_font("Helvetica", "", 9)
        metrics = [
            ("Market Cap (Cr)", fundamentals.get("market_cap_cr", "N/A")),
            ("P/E Ratio", fundamentals.get("pe_ratio", "N/A")),
            ("Forward P/E", fundamentals.get("forward_pe", "N/A")),
            ("EPS (Trailing)", fundamentals.get("eps", "N/A")),
            ("ROE", fundamentals.get("roe", "N/A")),
            ("Profit Margin", fundamentals.get("profit_margin", "N/A")),
            ("Dividend Yield", fundamentals.get("dividend_yield", "N/A")),
        ]
        for k, v in metrics:
            val_str = f"{v:.2f}%" if k in ["ROE", "Profit Margin", "Dividend Yield"] and isinstance(v, (int, float)) else str(v)
            if k == "Market Cap (Cr)" and isinstance(v, (int, float)):
                val_str = f"Rs. {v:,.2f} Cr"
            pdf.cell(95, 6, k, 1, 0, "L")
            pdf.cell(95, 6, val_str, 1, 1, "R")
            
        if fund_analysis and isinstance(fund_analysis, dict):
            pdf.ln(4)
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 6, f"Fundamental Score: {fund_analysis.get('score', 50)}/100 | Rating: {fund_analysis.get('rating', 'HOLD')}", ln=True)
            pdf.set_font("Helvetica", "I", 9)
            pdf.set_text_color(80, 80, 80)
            pdf.multi_cell(0, 5, f"Summary Thesis: {fund_analysis.get('summary', '')}")
            pdf.set_text_color(14, 17, 23)
    else:
        pdf.set_font("Helvetica", "I", 10)
        pdf.set_text_color(128, 128, 128)
        pdf.cell(0, 6, "Fundamental Analysis: Data Unavailable", ln=True)
    pdf.ln(6)
    
    # ─── 3. Technical Signals Summary ───
    if pdf.get_y() > 220:
        pdf.add_page()
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(108, 99, 255)
    pdf.cell(0, 8, "3. Technical Signals Summary", ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)
    
    if signals and isinstance(signals, dict) and not signals.get("error"):
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 6, f"Overall Trend: {signals.get('overall', 'NEUTRAL')} (Strength: {signals.get('strength', 0.5):.0%})", ln=True)
        pdf.ln(2)
        
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_fill_color(240, 240, 245)
        pdf.cell(60, 6, "Indicator", 1, 0, "L", True)
        pdf.cell(65, 6, "Value", 1, 0, "C", True)
        pdf.cell(65, 6, "Signal", 1, 1, "R", True)
        
        pdf.set_font("Helvetica", "", 9)
        for s in signals.get("signals", []):
            pdf.cell(60, 6, s.get("indicator", "N/A"), 1, 0, "L")
            pdf.cell(65, 6, str(s.get("value", "N/A")), 1, 0, "C")
            pdf.cell(65, 6, s.get("signal", "N/A"), 1, 1, "R")
    else:
        pdf.set_font("Helvetica", "I", 10)
        pdf.set_text_color(128, 128, 128)
        pdf.cell(0, 6, "Technical Signals: Data Unavailable", ln=True)
    pdf.ln(6)
    
    # ─── 4. Risk Assessment ───
    if pdf.get_y() > 220:
        pdf.add_page()
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(108, 99, 255)
    pdf.cell(0, 8, "4. Risk Assessment", ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)
    
    if risk and isinstance(risk, dict) and not risk.get("error"):
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 6, f"Overall Risk Level: {risk.get('risk_level', 'Medium')} (Score: {risk.get('risk_score', 5)}/10)", ln=True)
        pdf.ln(2)
        
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_fill_color(240, 240, 245)
        pdf.cell(95, 6, "Risk Parameter", 1, 0, "L", True)
        pdf.cell(95, 6, "Value", 1, 1, "R", True)
        
        pdf.set_font("Helvetica", "", 9)
        risk_metrics = [
            ("Annualised Volatility", f"{risk.get('volatility', 0.0)*100:.2f}%"),
            ("Maximum Drawdown", f"{risk.get('max_drawdown', 0.0)*100:.2f}%"),
            ("Value at Risk (95% Daily)", f"{risk.get('var_95', 0.0)*100:.2f}%"),
            ("Systematic Beta", f"{risk.get('beta', 1.00):.2f}"),
            ("Sharpe Ratio", f"{risk.get('sharpe', 0.00):.2f}"),
        ]
        for k, v in risk_metrics:
            pdf.cell(95, 6, k, 1, 0, "L")
            pdf.cell(95, 6, v, 1, 1, "R")
    else:
        pdf.set_font("Helvetica", "I", 10)
        pdf.set_text_color(128, 128, 128)
        pdf.cell(0, 6, "Risk Assessment: Data Unavailable", ln=True)
    pdf.ln(6)
    
    # ─── 5. News Sentiment Summary ───
    if pdf.get_y() > 220:
        pdf.add_page()
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(108, 99, 255)
    pdf.cell(0, 8, "5. News Sentiment Summary", ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)
    
    if agg_sent:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 6, f"Aggregate Sentiment: {agg_sent.get('label', 'Neutral')} (Score: {agg_sent.get('score', 0.5)*100:.1f}%)", ln=True)
        pdf.set_font("Helvetica", "", 9.5)
        pdf.cell(0, 6, f"Distribution: Positive ({agg_sent['distribution']['Positive']}) | Neutral ({agg_sent['distribution']['Neutral']}) | Negative ({agg_sent['distribution']['Negative']})", ln=True)
        pdf.ln(2)
        
        if news_articles:
            pdf.set_font("Helvetica", "B", 9.5)
            pdf.cell(0, 6, "Recent Headlines:", ln=True)
            pdf.set_font("Helvetica", "", 9)
            for a in news_articles[:3]:
                title_clean = a.get("title", "N/A")[:85]
                sent_lbl = a.get("sentiment", "Neutral")
                pdf.cell(160, 5.5, f" • {title_clean}...", 0, 0)
                pdf.cell(30, 5.5, f"[{sent_lbl}]", 0, 1, "R")
    else:
        pdf.set_font("Helvetica", "I", 10)
        pdf.set_text_color(128, 128, 128)
        pdf.cell(0, 6, "News Sentiment Summary: Data Unavailable", ln=True)
    pdf.ln(8)
    
    # ─── Disclaimer ───
    if pdf.get_y() > 240:
        pdf.add_page()
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(200, 0, 0)
    pdf.cell(0, 6, "Disclaimer:", ln=True)
    pdf.set_font("Helvetica", "I", 8.5)
    pdf.set_text_color(100, 100, 100)
    disclaimer_text = (
        "This report is compiled automatically by NiveshAI using quantitative stock data, "
        "historical analytics, and news sentiment modeling. The information is provided for educational "
        "and informational purposes only and does not constitute financial, investment, or tax advice. "
        "Investing in Indian stock markets involves systematic and unsystematic risk. Past performance "
        "is not a guarantee of future outcomes. NiveshAI is not a SEBI-registered advisor."
    )
    pdf.multi_cell(0, 4.5, disclaimer_text)
    
    return bytes(pdf.output())

# ─── Sidebar Controls ───
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
    st.markdown("---")
    
    try:
        all_options = get_display_options()
    except Exception:
        all_options = ["RELIANCE — Reliance Industries Ltd", "TCS — Tata Consultancy Services Ltd", "INFY — Infosys Ltd", "HDFCBANK — HDFC Bank Ltd", "ITC — ITC Ltd"]
        
    st.markdown('<p class="sidebar-section-label">Select Stock</p>', unsafe_allow_html=True)
    symbol = parse_display_option(st.selectbox("Stock", all_options, label_visibility="collapsed"))
    
    st.markdown('<p class="sidebar-section-label">Report Type</p>', unsafe_allow_html=True)
    report_type = st.radio(
        "Report Type",
        ["Quick Summary (1 page)", "Full Analysis (5 pages)", "Portfolio Report"],
        label_visibility="collapsed"
    )

# ─── Header ───
st.markdown(f"""
<div class="report-header">
    <h1>📋 Investment Report Generator</h1>
    <p>Generate comprehensive AI-powered investment reports in PDF format</p>
</div>
""", unsafe_allow_html=True)

# Status bar
st.markdown(f"""
<div class="status-bar">
    <span>📊 Type: <span class="value">{report_type}</span></span>
    <span>🏢 Selected Ticker: <span class="value">{symbol}</span></span>
    <span>📅 Period: <span class="value">1 Year (Default)</span></span>
</div>
""", unsafe_allow_html=True)
st.markdown("")

# ─── Error handling if fpdf2 is not installed ───
if not fpdf_installed:
    st.warning("⚠️ **PDF Export Disabled**: The `fpdf2` library is not installed on this system.")
    st.info("💡 To enable PDF downloads, please install it manually via: `pip install fpdf2`")

# ─── Page State ───
if "report_pdf_bytes" not in st.session_state:
    st.session_state.report_pdf_bytes = None
if "report_data" not in st.session_state:
    st.session_state.report_data = None
if "report_symbol" not in st.session_state:
    st.session_state.report_symbol = None

# ─── Compile Report Button ───
btn_generate = st.button("🚀 Generate PDF Report", type="primary")

if btn_generate:
    with st.spinner(f"Compiling research report details for {symbol}..."):
        # 1. Fetch all data
        try:
            history = fetch_history(symbol, period="1y")
        except Exception:
            history = pd.DataFrame()
            
        try:
            live = fetch_live_price(symbol)
        except Exception:
            live = {}
            
        try:
            fundamentals = fetch_fundamentals(symbol)
        except Exception:
            fundamentals = {}
            
        # Fetch News
        try:
            from data.news_fetcher import fetch_news
            from config.settings import NEWSAPI_KEY
            news_articles = fetch_news(symbol, newsapi_key=NEWSAPI_KEY, max_articles=5)
            
            from models.sentiment.sentiment_analyzer import get_sentiment_analyzer
            analyzer = get_sentiment_analyzer()
            if analyzer.available and news_articles:
                news_articles = analyzer.analyze_news(news_articles)
                agg_sent = analyzer.get_aggregate_sentiment(news_articles)
            else:
                agg_sent = {"label": "Neutral", "score": 0.5, "distribution": {"Positive": 0, "Neutral": 0, "Negative": 0}}
        except Exception:
            news_articles = []
            agg_sent = {"label": "Neutral", "score": 0.5, "distribution": {"Positive": 0, "Neutral": 0, "Negative": 0}}
            
        # 2. Run analysis
        try:
            signals = generate_signals(history) if (history is not None and not history.empty) else None
        except Exception:
            signals = None
            
        try:
            fund_analysis = analyze_fundamentals(fundamentals) if fundamentals else None
        except Exception:
            fund_analysis = None
            
        try:
            from analysis.risk import assess_risk
            risk = assess_risk(history, fundamentals) if (history is not None and not history.empty) else None
        except Exception:
            risk = None
            
        # 3. Save price chart as image (using matplotlib to avoid kaleido requirement)
        chart_img_path = f"temp_price_chart_{symbol}.png"
        chart_generated = False
        if history is not None and not history.empty:
            try:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(history.index[-90:], history["Close"].iloc[-90:], color="#6C63FF", linewidth=2)
                ax.set_title(f"{symbol} Share Price (Last 90 Days)", fontsize=10, fontweight="bold")
                ax.set_ylabel("Price (INR)")
                ax.grid(True, linestyle="--", alpha=0.3)
                plt.tight_layout()
                plt.savefig(chart_img_path, dpi=150)
                plt.close(fig)
                chart_generated = True
            except Exception as e:
                print(f"Chart generation failed: {e}")
                
        # 4. Generate PDF if fpdf2 is installed
        pdf_bytes = None
        if fpdf_installed:
            try:
                pdf_bytes = generate_pdf_report(
                    symbol, history, live, fundamentals, signals, fund_analysis, risk,
                    agg_sent, news_articles, chart_generated, chart_img_path
                )
            except Exception as e:
                st.error(f"Failed to compile PDF report: {e}")
                
        # Cleanup temp chart file
        if chart_generated and os.path.exists(chart_img_path):
            try:
                import os
                os.remove(chart_img_path)
            except Exception:
                pass
                
        # Save to state
        st.session_state.report_pdf_bytes = pdf_bytes
        st.session_state.report_data = {
            "history": history,
            "live": live,
            "fundamentals": fundamentals,
            "signals": signals,
            "fund_analysis": fund_analysis,
            "risk": risk,
            "agg_sent": agg_sent,
            "news_articles": news_articles
        }
        st.session_state.report_symbol = symbol

# ─── Render Results (HTML fallback and Download button) ───
if st.session_state.report_data is not None and st.session_state.report_symbol == symbol:
    data = st.session_state.report_data
    history = data["history"]
    live = data["live"]
    fundamentals = data["fundamentals"]
    signals = data["signals"]
    fund_analysis = data["fund_analysis"]
    risk = data["risk"]
    agg_sent = data["agg_sent"]
    news_articles = data["news_articles"]
    
    # ─── Download PDF Button ───
    if fpdf_installed and st.session_state.report_pdf_bytes:
        st.markdown("""
        <div class="download-area">
            <p style="font-size:1.15rem;color:#E0E0E0;font-weight:700;margin-bottom:6px;">
                📥 PDF Report Ready
            </p>
            <p>Click below to download the compiled PDF document.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col_dl1, col_dl2, col_dl3 = st.columns([1, 2, 1])
        with col_dl2:
            st.download_button(
                label=f"📥 Download PDF Report",
                data=st.session_state.report_pdf_bytes,
                file_name=f"NiveshAI_{symbol}_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf",
                use_container_width=True,
                type="primary",
            )
            
    # ─── Fallback HTML Report Preview ───
    st.markdown("---")
    st.markdown("## 📊 On-Screen HTML Research Report Fallback / Preview")
    
    comp_name = fundamentals.get("company_name", symbol) if fundamentals else symbol
    sector = fundamentals.get("sector", "Data Unavailable") if fundamentals else "Data Unavailable"
    industry = fundamentals.get("industry", "Data Unavailable") if fundamentals else "Data Unavailable"
    
    # On screen preview using Streamlit elements
    st.markdown(f"### {comp_name} ({symbol})")
    st.write(f"**Sector**: {sector} | **Industry**: {industry}")
    
    # 5 tabs representing the PDF sections
    t_chart, t_fund, t_tech, t_risk, t_news = st.tabs([
        "📈 Price Chart", "📊 Fundamental Analysis", "🔍 Technical Signals", "🛡️ Risk Assessment", "📰 News Sentiment"
    ])
    
    with t_chart:
        if history is not None and not history.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=history.index[-90:], y=history["Close"].iloc[-90:], name="Close", line=dict(color="#6C63FF", width=2.5)))
            fig.update_layout(
                title=f"{symbol} Close Price (Last 90 Days)",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(14,17,23,0.8)",
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Price Chart: Data Unavailable")
            
    with t_fund:
        if fundamentals and isinstance(fundamentals, dict) and not fundamentals.get("error"):
            st.markdown(f"**Market Cap**: Rs. {fundamentals.get('market_cap_cr', 'Data Unavailable')} Cr")
            st.markdown(f"**P/E Ratio**: {fundamentals.get('pe_ratio', 'Data Unavailable')}x")
            st.markdown(f"**Forward P/E**: {fundamentals.get('forward_pe', 'Data Unavailable')}x")
            st.markdown(f"**ROE**: {fundamentals.get('roe', 'Data Unavailable')}%")
            
            if fund_analysis:
                st.markdown(f"**Fundamental Rating Rating**: `{fund_analysis.get('rating', 'HOLD')}` (Score: `{fund_analysis.get('score', 50)}/100`)")
                st.write(f"Thesis: *{fund_analysis.get('summary')}*")
        else:
            st.info("Fundamental Analysis: Data Unavailable")
            
    with t_tech:
        if signals and isinstance(signals, dict) and not signals.get("error"):
            st.markdown(f"**Overall Trend Status**: `{signals.get('overall')}` (Strength: `{signals.get('strength', 0.5):.0%}`)")
            sig_df = pd.DataFrame(signals.get("signals", []))
            st.dataframe(sig_df, use_container_width=True, hide_index=True)
        else:
            st.info("Technical Signals: Data Unavailable")
            
    with t_risk:
        if risk and isinstance(risk, dict) and not risk.get("error"):
            st.markdown(f"**Overall Risk Level Level**: `{risk.get('risk_level')}` (Score: `{risk.get('risk_score')}/10`)")
            risk_tbl = pd.DataFrame([
                {"Parameter": "Annualised Volatility", "Value": f"{risk.get('volatility', 0.0)*100:.2f}%"},
                {"Parameter": "Maximum Drawdown", "Value": f"{risk.get('max_drawdown', 0.0)*100:.2f}%"},
                {"Parameter": "Value at Risk (95% Daily)", "Value": f"{risk.get('var_95', 0.0)*100:.2f}%"},
                {"Parameter": "Systematic Beta", "Value": f"{risk.get('beta', 1.00):.2f}"},
                {"Parameter": "Sharpe Ratio", "Value": f"{risk.get('sharpe', 0.00):.2f}"}
            ])
            st.dataframe(risk_tbl, use_container_width=True, hide_index=True)
        else:
            st.info("Risk Assessment: Data Unavailable")
            
    with t_news:
        if agg_sent:
            st.markdown(f"**Aggregate News Sentiment**: `{agg_sent.get('label')}` (Score: `{agg_sent.get('score', 0.5)*100:.1f}%`)")
            if news_articles:
                for a in news_articles[:3]:
                    st.write(f"• **{a.get('title')}** `[{a.get('sentiment', 'Neutral')}]`")
        else:
            st.info("News Sentiment Summary: Data Unavailable")

# ─── Disclaimer ───
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
