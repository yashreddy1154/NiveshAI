"""
NiveshAI — PDF Report Generator
Generates professional investment research reports in PDF format using fpdf2.
"""

import os
import sys
from datetime import datetime
from fpdf import FPDF
from pathlib import Path
import pandas as pd
import numpy as np

# Internal modules
from data.stock_data import fetch_full_stock_data
from analysis.technical import generate_signals
from analysis.fundamental import analyze_fundamentals
from models.prediction.lstm_predictor import get_lstm_predictor
from models.prediction.rf_predictor import get_rf_predictor
from models.sentiment.sentiment_analyzer import get_sentiment_analyzer


class NiveshReportPDF(FPDF):
    """FPDF class extension to customize headers, footers and layouts."""

    def header(self):
        # Header banner
        self.set_fill_color(14, 17, 23) # Dark Navy
        self.rect(0, 0, 210, 32, "F")
        
        # Title
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 16)
        self.set_y(8)
        self.cell(0, 8, "NiveshAI — Investment Research Report", ln=True, align="L")
        
        # Subtitle / Date
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(160, 174, 192)
        self.cell(0, 4, f"Generated: {datetime.now().strftime('%d %B %Y %H:%M')} IST", ln=True, align="L")
        
        # Top-right Accent Indicator
        self.set_fill_color(108, 99, 255) # Purple accent
        self.rect(170, 0, 40, 4, "F")

        # Set cursor below header banner
        self.set_y(36)

    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        # Page number
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}} | Confidential — Prepared for personal research only", align="C")
        
        # Tiny warning disclaimer
        self.set_y(-10)
        self.cell(0, 10, "Disclaimer: Not SEBI-registered financial advice. Past performance is not indicative of future results.", align="C")


def generate_report(symbol: str, sections: list = None) -> bytes:
    """
    Generate a comprehensive investment report as PDF.
    Returns: PDF file as bytes
    """
    if sections is None:
        sections = ['price_analysis', 'fundamentals', 'technicals', 'sentiment', 'prediction', 'risk', 'recommendation']

    # Fetch real-time and historical data
    data = fetch_full_stock_data(symbol)
    history = data["history"]
    live = data["live_price"]
    fundamentals = data["fundamentals"]

    # Analyze
    sig = generate_signals(history)
    fund = analyze_fundamentals(fundamentals)
    
    # Predict
    lstm = get_lstm_predictor()
    rf = get_rf_predictor()
    
    lstm_results = lstm.predict_next_days(history, n_days=7) if lstm.available else None
    rf_results = rf.predict_direction(history) if rf.available else None
    
    # Sentiment
    from data.news_fetcher import fetch_news
    from config.settings import NEWSAPI_KEY
    articles = fetch_news(symbol, newsapi_key=NEWSAPI_KEY, max_articles=5)
    
    analyzer = get_sentiment_analyzer()
    if analyzer.available:
        articles = analyzer.analyze_news(articles)
    agg_sent = analyzer.get_aggregate_sentiment(articles)

    # Initialize FPDF
    pdf = NiveshReportPDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    # ── TITLE SECTION ──
    pdf.set_text_color(14, 17, 23)
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 10, f"{fundamentals.get('company_name', symbol)} ({symbol})", ln=True, align="L")
    
    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(108, 99, 255)
    pdf.cell(0, 6, f"Sector: {fundamentals.get('sector', 'Unknown')} | Industry: {fundamentals.get('industry', 'Unknown')}", ln=True)
    
    pdf.set_y(pdf.get_y() + 4)
    
    # ── PRICE SUMMARY STRIP ──
    pdf.set_fill_color(240, 242, 246)
    pdf.rect(10, pdf.get_y(), 190, 16, "F")
    
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(80, 80, 80)
    pdf.set_x(12)
    pdf.cell(40, 16, f"Price: INR {live.get('price', history['Close'].iloc[-1]):,.2f}")
    
    change = live.get("change", 0.0)
    change_pct = live.get("change_pct", 0.0)
    pdf.cell(40, 16, f"Change: {change:+.2f} ({change_pct:+.2f}%)")
    pdf.cell(50, 16, f"Day High/Low: {live.get('high', 0.0):,.1f} / {live.get('low', 0.0):,.1f}")
    pdf.cell(50, 16, f"Volume: {live.get('volume', 0.0)/1e6:,.2f}M shares", ln=True)
    
    pdf.set_y(pdf.get_y() + 6)

    # ── 1. AI RECOMMENDATION ──
    if 'recommendation' in sections:
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(108, 99, 255)
        pdf.cell(0, 8, "1. AI Investment Recommendation", ln=True)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.set_y(pdf.get_y() + 2)
        
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(40, 40, 40)
        
        rec_verdict = fund.get("rating", "HOLD").upper()
        rec_score = fund.get("score", 50)
        
        pdf.write(5, f"Based on our multi-factor machine learning pipeline and fundamental analysis models, NiveshAI issues a ")
        pdf.set_font("Helvetica", "B", 10)
        if "BUY" in rec_verdict:
            pdf.set_text_color(0, 150, 0)
        elif "SELL" in rec_verdict:
            pdf.set_text_color(200, 0, 0)
        else:
            pdf.set_text_color(200, 120, 0)
        pdf.write(5, f"{rec_verdict}")
        
        pdf.set_text_color(40, 40, 40)
        pdf.set_font("Helvetica", "", 10)
        pdf.write(5, f" rating for {symbol}. The fundamental rating score is ")
        pdf.set_font("Helvetica", "B", 10)
        pdf.write(5, f"{rec_score}/100")
        
        pdf.set_font("Helvetica", "", 10)
        pdf.write(5, f". This recommendation combines fundamental quality rankings, long-term technical trends, and NLP-driven news sentiment distributions.\n\n")
        
        # Summary paragraph
        pdf.set_font("Helvetica", "I", 9.5)
        pdf.write(4.5, f"Summary Thesis: {fund.get('summary', '')}\n\n")
        pdf.set_font("Helvetica", "", 10)

    # ── 2. FUNDAMENTAL DATA ──
    if 'fundamentals' in sections:
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(108, 99, 255)
        pdf.cell(0, 8, "2. Fundamental Analysis", ln=True)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.set_y(pdf.get_y() + 2)
        
        # Print metrics table
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_fill_color(230, 230, 240)
        pdf.cell(95, 6, "Metric Key", 1, 0, "L", True)
        pdf.cell(95, 6, "Value", 1, 1, "R", True)
        
        pdf.set_font("Helvetica", "", 9)
        met_data = fund.get("metrics", {})
        for idx, (k, v) in enumerate(met_data.items()):
            # Strip extremely long keys if any
            pdf.cell(95, 5.5, str(k), 1, 0, "L")
            pdf.cell(95, 5.5, str(v), 1, 1, "R")
            
        pdf.set_y(pdf.get_y() + 4)

        # Strengths / Concerns
        pdf.set_font("Helvetica", "B", 9.5)
        pdf.cell(0, 6, "Key Strengths:", ln=True)
        pdf.set_font("Helvetica", "", 9)
        for s in fund.get("strengths", [])[:3]:
            pdf.cell(0, 5, f" - {s}", ln=True)
            
        pdf.set_y(pdf.get_y() + 2)
        pdf.set_font("Helvetica", "B", 9.5)
        pdf.cell(0, 6, "Key Concerns:", ln=True)
        pdf.set_font("Helvetica", "", 9)
        for w in fund.get("weaknesses", [])[:3]:
            pdf.cell(0, 5, f" - {w}", ln=True)
            
        pdf.set_y(pdf.get_y() + 6)

    # ── 3. TECHNICAL INDICATORS ──
    if 'technicals' in sections:
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(108, 99, 255)
        pdf.cell(0, 8, "3. Technical Signal Summary", ln=True)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.set_y(pdf.get_y() + 2)
        
        pdf.set_font("Helvetica", "", 10)
        pdf.write(5, f"The overall technical trend signal is currently ")
        pdf.set_font("Helvetica", "B", 10)
        pdf.write(5, f"{sig.get('overall', 'NEUTRAL')}")
        pdf.set_font("Helvetica", "", 10)
        pdf.write(5, f" with a momentum strength score of {sig.get('strength', 0.5):.0%}. Below is a status breakdown of standard indicators:\n\n")

        # Table of indicators
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_fill_color(230, 230, 240)
        pdf.cell(60, 6, "Indicator", 1, 0, "L", True)
        pdf.cell(65, 6, "Value", 1, 0, "C", True)
        pdf.cell(65, 6, "Signal Signal", 1, 1, "R", True)
        
        pdf.set_font("Helvetica", "", 9)
        for s_item in sig.get("signals", []):
            pdf.cell(60, 5.5, s_item["indicator"], 1, 0, "L")
            pdf.cell(65, 5.5, str(s_item["value"]), 1, 0, "C")
            pdf.cell(65, 5.5, s_item["signal"], 1, 1, "R")
            
        pdf.set_y(pdf.get_y() + 6)

    # ── 4. MACHINE LEARNING FORECASTS ──
    if 'prediction' in sections:
        # Check page break
        if pdf.get_y() > 180:
            pdf.add_page()
            
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(108, 99, 255)
        pdf.cell(0, 8, "4. Machine Learning Price Forecasts", ln=True)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.set_y(pdf.get_y() + 2)
        
        # LSTM prediction text
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 6, "LSTM Price Predictor:", ln=True)
        pdf.set_font("Helvetica", "", 10)
        if lstm_results:
            pdf.write(5, f"Our LSTM model has forecast the price of {symbol} over the next 7 days. The predicted price at the end of the horizon is ")
            pdf.set_font("Helvetica", "B", 10)
            pdf.write(5, f"INR {lstm_results['predicted_7d']:,.2f}")
            pdf.set_font("Helvetica", "", 10)
            pdf.write(5, f", representing a directional shift of ")
            pdf.set_font("Helvetica", "B", 10)
            pdf.write(5, f"{lstm_results['change_pct']:+.2f}%")
            pdf.set_font("Helvetica", "", 10)
            pdf.write(5, f" ({lstm_results['direction']}). The model's confidence rating is {lstm_results['confidence']:.0%}.\n\n")
            
            # Print daily predictions list
            for idx, p in enumerate(lstm_results["predictions"]):
                pdf.cell(40, 5, f" - Day {idx+1} ({p['date']}):", 0, 0)
                pdf.cell(50, 5, f"INR {p['price']:,.2f}", 0, 1)
        else:
            pdf.write(5, "LSTM model files are missing or could not load. Skipping quantitative forecasts.\n\n")

        pdf.set_y(pdf.get_y() + 2)

        # RF prediction text
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 6, "Random Forest Classifier:", ln=True)
        pdf.set_font("Helvetica", "", 10)
        if rf_results:
            pdf.write(5, f"The Random Forest directional classifier predicts the next day's price change to be ")
            pdf.set_font("Helvetica", "B", 10)
            pdf.write(5, f"{rf_results['direction']}")
            pdf.set_font("Helvetica", "", 10)
            pdf.write(5, f" with a prediction probability of {rf_results['confidence']:.0%}. ")
            
            # Probabilities list
            probs = rf_results["probabilities"]
            pdf.write(5, f"Individual probability breakdown: UP ({probs['UP']:.1%}), FLAT ({probs['FLAT']:.1%}), DOWN ({probs['DOWN']:.1%}).\n\n")
        else:
            pdf.write(5, "Random Forest classifier files are missing or could not load.\n\n")
            
        pdf.set_y(pdf.get_y() + 4)

    # ── 5. NEWS SENTIMENT ──
    if 'sentiment' in sections:
        # Check page break
        if pdf.get_y() > 180:
            pdf.add_page()
            
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(108, 99, 255)
        pdf.cell(0, 8, "5. News & Sentiment Analysis", ln=True)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.set_y(pdf.get_y() + 2)
        
        pdf.set_font("Helvetica", "", 10)
        pdf.write(5, f"Our sentiment engine parsed recent financial news feeds. The aggregate news sentiment score is ")
        pdf.set_font("Helvetica", "B", 10)
        pdf.write(5, f"{agg_sent['label']} ({agg_sent['score'] * 100:.1f}%)")
        pdf.set_font("Helvetica", "", 10)
        pdf.write(5, f". Article breakdown: Positive ({agg_sent['distribution']['Positive']}), Neutral ({agg_sent['distribution']['Neutral']}), Negative ({agg_sent['distribution']['Negative']}).\n\n")

        # List top 3 headlines
        pdf.set_font("Helvetica", "B", 9.5)
        pdf.cell(0, 6, "Recent Headlines Analyzed:", ln=True)
        pdf.set_font("Helvetica", "", 9)
        for art in articles[:3]:
            sent_label = art.get("sentiment", "Neutral")
            pdf.cell(160, 5, f" • {art['title'][:80]}...", 0, 0)
            pdf.cell(30, 5, f"[{sent_label}]", 0, 1, "R")
            
        pdf.set_y(pdf.get_y() + 4)

    # Output to bytes
    return bytes(pdf.output())


def generate_comparison_report(symbols: list) -> bytes:
    """Generate a multi-stock comparison report as PDF."""
    # Basic skeleton for comparison PDF
    pdf = NiveshReportPDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    pdf.set_text_color(14, 17, 23)
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "NiveshAI — Multi-Stock Comparison Report", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(108, 99, 255)
    pdf.cell(0, 6, f"Comparing: {', '.join(symbols)}", ln=True)
    
    pdf.set_y(pdf.get_y() + 6)
    
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_fill_color(230, 230, 240)
    pdf.cell(50, 6, "Ticker", 1, 0, "L", True)
    pdf.cell(45, 6, "CMP", 1, 0, "R", True)
    pdf.cell(45, 6, "P/E", 1, 0, "R", True)
    pdf.cell(50, 6, "Signal", 1, 1, "C", True)
    
    pdf.set_font("Helvetica", "", 10)
    for sym in symbols:
        try:
            data = fetch_full_stock_data(sym)
            live = data["live_price"]
            fundamentals = data["fundamentals"]
            sig = generate_signals(data["history"])
            
            pdf.cell(50, 6, sym, 1, 0, "L")
            pdf.cell(45, 6, f"INR {live.get('price', 0.0):,.2f}", 1, 0, "R")
            pdf.cell(45, 6, str(fundamentals.get("trailingPE", "N/A"))[:5] + "x", 1, 0, "R")
            pdf.cell(50, 6, sig.get("overall", "NEUTRAL"), 1, 1, "C")
        except Exception:
            pdf.cell(50, 6, sym, 1, 0, "L")
            pdf.cell(140, 6, "Failed to load data for comparison", 1, 1, "C")
            
    return bytes(pdf.output())
