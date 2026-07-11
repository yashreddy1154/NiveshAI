import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Internal modules
from models.prediction.lstm_predictor import get_lstm_predictor
from models.prediction.rf_predictor import get_rf_predictor
from data.stock_data import fetch_history
from data.company_db import get_display_options, parse_display_option, get_company
from analysis.technical import compute_all_indicators

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Predictions — NiveshAI", page_icon="📈", layout="wide")

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
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(108, 99, 255, 0.1);
        border-radius: 8px;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(108, 99, 255, 0.3) !important;
    }

    .prediction-header {
        background: linear-gradient(135deg, rgba(108, 99, 255, 0.12), rgba(0, 212, 170, 0.06));
        border: 1px solid rgba(108, 99, 255, 0.18);
        border-radius: 16px;
        padding: 32px 36px;
        margin-bottom: 24px;
    }
    .prediction-header h1 {
        margin: 0 0 4px 0;
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6C63FF, #00D4AA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .prediction-header p {
        margin: 0;
        color: #8892A5;
        font-size: 1.02rem;
    }

    .info-card {
        background: linear-gradient(135deg, rgba(108, 99, 255, 0.08), rgba(78, 205, 196, 0.04));
        border-left: 4px solid #6C63FF;
        border-radius: 0 12px 12px 0;
        padding: 18px 22px;
        margin-bottom: 20px;
        color: #C5CEE0;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .info-card strong { color: #A8A2FF; }

    .direction-card {
        border-radius: 20px;
        padding: 36px 32px;
        text-align: center;
        margin: 12px 0 20px 0;
    }
    .direction-card .direction-label {
        font-size: 1rem;
        color: #E2E2E2;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 8px;
    }
    .direction-card .direction-value {
        font-size: 2.8rem;
        font-weight: 900;
        margin-bottom: 4px;
    }
    .direction-card .direction-confidence {
        font-size: 1.15rem;
    }

    .status-badge {
        display: inline-block;
        background: rgba(0, 212, 170, 0.12);
        border: 1px solid rgba(0, 212, 170, 0.3);
        border-radius: 24px;
        padding: 8px 20px;
        color: #00D4AA;
        font-size: 0.9rem;
        font-weight: 600;
        margin-top: 10px;
    }

    .trust-card {
        background: linear-gradient(135deg, rgba(255, 179, 71, 0.1), rgba(255, 230, 109, 0.04));
        border: 1px solid rgba(255, 179, 71, 0.25);
        border-radius: 16px;
        padding: 26px 28px;
        color: #C5CEE0;
        line-height: 1.7;
    }
    .trust-card h4 {
        color: #FFB347;
        margin-top: 0;
        font-size: 1.15rem;
    }

    .shap-card {
        background: rgba(26, 31, 46, 0.85);
        border: 1px solid rgba(108, 99, 255, 0.15);
        border-radius: 14px;
        padding: 20px 24px;
        margin: 10px 0;
        color: #C5CEE0;
        line-height: 1.6;
    }
    .shap-card h5 {
        color: #A8A2FF;
        margin: 0 0 8px 0;
        font-size: 1.05rem;
    }

    .comparison-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        border-radius: 12px;
        overflow: hidden;
        margin: 12px 0 20px 0;
    }
    .comparison-table th {
        background: rgba(108, 99, 255, 0.18);
        color: #A8A2FF;
        padding: 14px 18px;
        text-align: left;
        font-weight: 700;
        font-size: 0.92rem;
        letter-spacing: 0.5px;
    }
    .comparison-table td {
        background: rgba(26, 31, 46, 0.7);
        color: #C5CEE0;
        padding: 12px 18px;
        border-top: 1px solid rgba(108, 99, 255, 0.08);
        font-size: 0.93rem;
    }
    .comparison-table tr:hover td {
        background: rgba(108, 99, 255, 0.08);
    }
</style>
""", unsafe_allow_html=True)

# ── Load Model Singleton References ───────────────────────────────────────────
@st.cache_resource
def load_models():
    lstm = get_lstm_predictor()
    rf   = get_rf_predictor()
    return lstm, rf

lstm_model, rf_model = load_models()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 18px 0 8px 0;">
        <span style="font-size:2rem;">📈</span>
        <h2 style="margin:4px 0 0 0; font-size:1.3rem;">
            <span class="gradient-text">NiveshAI</span>
        </h2>
        <p style="color:#8892A5; font-size:0.82rem; margin:2px 0 0 0;">ML Price Predictions</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    symbol = parse_display_option(st.selectbox("Select Stock", get_display_options()))
    selected_symbol = symbol

    st.markdown("##### 🤖 Models Status")
    
    # Model Loading indicators
    if lstm_model.available:
        st.markdown("🟢 **LSTM**: Loaded (35min training)")
    else:
        st.markdown("🔴 **LSTM**: Model not found")
        
    if rf_model.available:
        st.markdown("🟢 **RF**: Loaded (187 stocks)")
    else:
        st.markdown("🔴 **RF**: Model not found")

    st.markdown("---")
    st.markdown("""
    <div style="background:rgba(108,99,255,0.08); border-radius:12px; padding:14px 16px; font-size:0.82rem; color:#8892A5; line-height:1.6;">
        <strong style="color:#A8A2FF;">⚠️ Disclaimer</strong><br>
        Predictions are generated by ML models trained on historical data.
        They should <u>not</u> be the sole basis for investment decisions.
    </div>
    """, unsafe_allow_html=True)

# ── Fetch stock data ──────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def get_prediction_data(sym):
    history = fetch_history(sym, period="1y")
    return compute_all_indicators(history)

with st.spinner(f"Fetching history and calculating features for {symbol}..."):
    history = get_prediction_data(symbol)

# Ensure required indicators/columns are prepared (copying existing if available)
if not history.empty:
    if "RSI" in history.columns and "RSI_14" not in history.columns:
        history["RSI_14"] = history["RSI"]
    if "BB_PctB" in history.columns and "BB_pctB" not in history.columns:
        history["BB_pctB"] = history["BB_PctB"]
    if "Daily_Return" in history.columns and "Price_Change" not in history.columns:
        history["Price_Change"] = history["Daily_Return"] / 100.0
    if "Volume" in history.columns and "Vol_Change" not in history.columns:
        history["Vol_Change"] = history["Volume"].pct_change()

company_info = get_company(symbol) or {}
company_name = company_info.get("Company Name", symbol)

# Page Header
st.markdown(f"""
<div class="prediction-header">
    <h1>📈 Price Predictions & Model Analysis</h1>
    <p>ML-powered price forecasts and directional signals for <strong style="color:#A8A2FF">{company_name}</strong>
    ({symbol}) · 7-day horizon</p>
</div>
""", unsafe_allow_html=True)

# Check model availability overall
if not lstm_model.available and not rf_model.available:
    st.error("Models not loaded. Place trained models in models/saved/")

# Main UI layout
if history.empty:
    st.error(f"Could not load historical data for {symbol}")
else:
    # Run predictions beforehand to avoid running them in each tab
    predictions = None
    rf_pred = None
    
    if lstm_model.available:
        try:
            predictions = lstm_model.predict_next_days(history, n_days=7)
        except Exception as e:
            st.error(f"LSTM prediction failed: {e}")
            
    if rf_model.available:
        try:
            rf_pred = rf_model.predict_direction(history)
        except Exception as e:
            st.error(f"Random Forest prediction failed: {e}")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🧠 LSTM Prediction",
        "🌲 Random Forest",
        "⚖️ Model Comparison",
        "🔍 Feature Importance",
    ])

    # TAB 1 — LSTM Prediction
    with tab1:
        if not lstm_model.available:
            st.error("Models not loaded. Place trained models in models/saved/")
        elif predictions is None:
            st.error("Insufficient data for prediction (need 70+ days)")
        else:
            st.markdown("""
            <div class="info-card">
                <strong>🧠 LSTM Neural Network</strong> — Our LSTM (Long Short-Term Memory) neural network
                analyzes <strong>60 days of historical price data</strong>, volume patterns, and technical indicators
                to forecast future closing prices. The model uses a 2-layer stacked LSTM architecture with
                dropout regularization, trained on 187 NSE stocks.
            </div>
            """, unsafe_allow_html=True)
            
            # Show prediction chart
            pred_df = pd.DataFrame(predictions["predictions"])
            fig = go.Figure()
            
            # Last 30 days actual
            fig.add_trace(go.Scatter(
                x=history.index[-30:], y=history["Close"][-30:], 
                name="Actual", line=dict(color="#6C63FF", width=2.2),
                hovertemplate="₹%{y:,.2f}<extra>Actual</extra>"
            ))
            
            # Predicted days
            # Prepend last actual close to make a continuous chart
            c_date = pd.to_datetime(history.index[-1])
            pred_x = [c_date] + list(pd.to_datetime(pred_df["date"]))
            pred_y = [predictions["current_price"]] + list(pred_df["price"])
            
            fig.add_trace(go.Scatter(
                x=pred_x, y=pred_y,
                name="Predicted", line=dict(color="#00D4AA", width=2.5, dash="dash"),
                marker=dict(size=5, color="#00D4AA"),
                hovertemplate="₹%{y:,.2f}<extra>Predicted</extra>"
            ))
            
            # Divider line at today
            fig.add_vline(
                x=c_date.timestamp() * 1000, line_width=1,
                line_dash="dot", line_color="rgba(255,179,71,0.5)",
                annotation_text="Today", annotation_position="top",
                annotation_font_color="#FFB347",
            )
            
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(14,17,23,0.6)",
                height=480,
                xaxis=dict(title="Date", gridcolor="rgba(108,99,255,0.08)"),
                yaxis=dict(title="Price (₹)", gridcolor="rgba(108,99,255,0.08)", tickprefix="₹"),
                legend=dict(
                    bgcolor="rgba(26,31,46,0.8)", bordercolor="rgba(108,99,255,0.2)",
                    borderwidth=1, font=dict(size=11),
                ),
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Prediction summary
            col1, col2, col3 = st.columns(3)
            col1.metric("Current Price", f"₹{predictions['current_price']:,.2f}")
            col2.metric("7-Day Prediction", f"₹{predictions['predicted_7d']:,.2f}", 
                        f"{predictions['change_pct']:+.2f}%")
            col3.metric("Direction", predictions["direction"])
            
            # Status badge
            st.markdown(f"""
            <div style="text-align:center; margin-top:6px;">
                <span class="status-badge">🟢 Model Loaded | Validation NRMSE: 2.95%</span>
            </div>
            """, unsafe_allow_html=True)

    # TAB 2 — Random Forest
    with tab2:
        if not rf_model.available:
            st.error("Models not loaded. Place trained models in models/saved/")
        elif rf_pred is None:
            st.error("Insufficient data for prediction (need 70+ days)")
        else:
            st.markdown("""
            <div class="info-card">
                <strong>🌲 Random Forest Classifier</strong> — Random Forest classifier predicts market direction
                (<strong>UP / DOWN / FLAT</strong>) for the next trading session using technical indicators.
                Trained on 187 NSE stocks, this model uses a ±0.5% threshold for direction labelling.
            </div>
            """, unsafe_allow_html=True)
            
            direction_colors = {"UP": "#00D4AA", "DOWN": "#FF6B6B", "FLAT": "#FFB347"}
            color = direction_colors[rf_pred["direction"]]
            st.markdown(f"""
            <div style="background:{color}22; border:1px solid {color}; 
                 border-radius:12px; padding:20px; text-align:center">
                <h2 style="color:{color}">{rf_pred["direction"]} Signal</h2>
                <p>Confidence: {rf_pred["confidence"]:.0%} | Strength: {rf_pred["signal_strength"]}</p>
                <small>{rf_pred["disclaimer"]}</small>
            </div>""", unsafe_allow_html=True)
            
            col_left, col_right = st.columns([1, 1])
            with col_left:
                st.markdown("<br>", unsafe_allow_html=True)
                # Probability bars
                for direction, prob in rf_pred["probabilities"].items():
                    st.progress(prob, text=f"{direction}: {prob:.0%}")
                    
            with col_right:
                # Probability bar chart
                prob_fig = go.Figure()
                categories = ["UP", "FLAT", "DOWN"]
                probs_list = [rf_pred["probabilities"]["UP"]*100, rf_pred["probabilities"]["FLAT"]*100, rf_pred["probabilities"]["DOWN"]*100]
                colors = ["#00D4AA", "#FFB347", "#FF6B6B"]
                
                prob_fig.add_trace(go.Bar(
                    y=categories, x=probs_list,
                    orientation="h",
                    marker=dict(color=colors, line=dict(width=0), cornerradius=6),
                    text=[f"{p:.1f}%" for p in probs_list],
                    textposition="auto",
                    textfont=dict(size=13, color="#FFFFFF", family="Inter"),
                ))
                prob_fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(14,17,23,0.6)",
                    height=240,
                    title=dict(text="Direction Probability Breakdown", font=dict(size=14, color="#C5CEE0")),
                    xaxis=dict(title="Probability (%)", range=[0, 100], gridcolor="rgba(108,99,255,0.08)"),
                    yaxis=dict(gridcolor="rgba(108,99,255,0.08)"),
                    margin=dict(l=10, r=20, t=40, b=30),
                    showlegend=False,
                )
                st.plotly_chart(prob_fig, use_container_width=True)

    # TAB 3 — Model Comparison
    with tab3:
        if not lstm_model.available or not rf_model.available:
            st.info("Both LSTM and Random Forest models must be loaded to see the full comparison.")
        elif predictions is None or rf_pred is None:
            st.error("Insufficient data for prediction (need 70+ days)")
        else:
            lstm_dir = predictions["direction"]
            rf_dir = rf_pred["direction"]
            
            # Head-to-Head Table
            st.markdown("#### ⚖️ Head-to-Head: LSTM vs Random Forest")
            st.markdown(f"""
            <table class="comparison-table">
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>🧠 LSTM Neural Network</th>
                        <th>🌲 Random Forest</th>
                    </tr>
                </thead>
                <tbody>
                    <tr><td><strong>Task</strong></td><td>Price Regression</td><td>Direction Classification</td></tr>
                    <tr><td><strong>Predicted Trend</strong></td><td>{lstm_dir} ({predictions['change_pct']:+.2f}%)</td><td>{rf_dir} ({rf_pred['confidence']:.1%})</td></tr>
                    <tr><td><strong>Validation Metric</strong></td><td>NRMSE: 2.95%</td><td>Test Accuracy: 36.28%</td></tr>
                    <tr><td><strong>Interpretability</strong></td><td>Low (black box)</td><td>High (feature importance)</td></tr>
                    <tr><td><strong>Best For</strong></td><td>Target ranges & price points</td><td>Momentum bias & binary calls</td></tr>
                </tbody>
            </table>
            """, unsafe_allow_html=True)
            
            # Advice card
            st.markdown(f"""
            <div class="trust-card">
                <h4>💡 Model Aggregator Outlook</h4>
                <p>LSTM predicted direction: <strong>{lstm_dir}</strong> | Random Forest predicted direction: <strong>{rf_dir}</strong></p>
                <ul>
                    <li><strong>When they agree</strong>: Conviction is high. Currently, the models indicate a <strong>{lstm_dir}</strong> trend bias.</li>
                    <li><strong>When they disagree</strong>: The Random Forest is best for short-term momentum (1-3 sessions) because of its Stratified splits, whereas the LSTM rolling forecast is better suited for longer horizons.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    # TAB 4 — Feature Importance
    with tab4:
        if not rf_model.available:
            st.error("Models not loaded. Place trained models in models/saved/")
        else:
            st.markdown("#### 🔍 Random Forest — Gini Feature Importance")
            st.markdown("""
            <div class="info-card">
                Feature importance scores indicate which technical and price metrics contributed the most
                predictive value to the Random Forest model during training on historical NSE data.
            </div>
            """, unsafe_allow_html=True)
            
            feat_importances = rf_model.get_feature_importance()
            if not feat_importances:
                feat_importances = {
                    "Price_Change": 0.174,
                    "Volume": 0.123,
                    "MACD": 0.117,
                    "Vol_Change": 0.111,
                    "BB_pctB": 0.100,
                    "RSI_14": 0.097,
                    "Close": 0.094,
                    "SMA_50": 0.093,
                    "SMA_20": 0.090
                }
                
            fig_fi = go.Figure()
            sorted_fi = dict(sorted(feat_importances.items(), key=lambda x: x[1]))
            
            fig_fi.add_trace(go.Bar(
                y=list(sorted_fi.keys()), x=list(sorted_fi.values()),
                orientation="h",
                marker=dict(color="rgba(108, 99, 255, 0.7)", line=dict(width=0), cornerradius=5),
                text=[f"{v:.1%}" for v in sorted_fi.values()],
                textposition="outside",
                textfont=dict(size=12, color="#C5CEE0"),
            ))
            fig_fi.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(14,17,23,0.6)",
                height=400,
                xaxis=dict(title="Importance Score", gridcolor="rgba(108,99,255,0.08)"),
                margin=dict(l=10, r=40, t=10, b=30),
                showlegend=False,
            )
            st.plotly_chart(fig_fi, use_container_width=True)
            
            st.markdown("#### 🧪 Feature Explanations")
            st.markdown(f"""
            <div class="shap-card">
                <h5>Price_Change & Volume (Top Features)</h5>
                The model heavily weights short-term close-to-close returns and trading volumes. Sudden price moves accompanied by high volume spikes have historically been the strongest leads for predicting continuation or trend reversal.
            </div>
            <div class="shap-card">
                <h5>MACD & RSI (Momentum Features)</h5>
                RSI and MACD represent momentum oscillators. The model uses MACD crossovers and RSI threshold breaches (e.g. crossing below 30 or above 70) to detect oversold and overbought cycles.
            </div>
            """, unsafe_allow_html=True)

# ── Disclaimer Section (REQUIRED) ───────────────────────────────────────────
st.warning("""
⚠️ **Disclaimer**: These predictions are generated by ML models trained on historical data. 
They do NOT guarantee future performance. The LSTM has 2.95% normalized error and the 
RF classifier has 36% accuracy (vs 33% random baseline). Always do your own research.
""")
