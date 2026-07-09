import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

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

    /* ── Page-specific styles ── */
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
        background: linear-gradient(135deg, rgba(0, 212, 170, 0.15), rgba(0, 212, 170, 0.04));
        border: 2px solid rgba(0, 212, 170, 0.35);
        border-radius: 20px;
        padding: 36px 32px;
        text-align: center;
        margin: 12px 0 20px 0;
    }
    .direction-card .direction-label {
        font-size: 1rem;
        color: #8892A5;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 8px;
    }
    .direction-card .direction-value {
        font-size: 2.8rem;
        font-weight: 900;
        color: #00D4AA;
        margin-bottom: 4px;
    }
    .direction-card .direction-confidence {
        font-size: 1.15rem;
        color: #4ECDC4;
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

# ── Stock Universe ───────────────────────────────────────────────────────────
STOCKS = {
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS": "Tata Consultancy Services",
    "INFY.NS": "Infosys",
    "HDFCBANK.NS": "HDFC Bank",
    "ICICIBANK.NS": "ICICI Bank",
    "HINDUNILVR.NS": "Hindustan Unilever",
    "BHARTIARTL.NS": "Bharti Airtel",
    "ITC.NS": "ITC Limited",
    "SBIN.NS": "State Bank of India",
    "KOTAKBANK.NS": "Kotak Mahindra Bank",
    "LT.NS": "Larsen & Toubro",
    "AXISBANK.NS": "Axis Bank",
    "WIPRO.NS": "Wipro",
    "HCLTECH.NS": "HCL Technologies",
    "ASIANPAINT.NS": "Asian Paints",
    "MARUTI.NS": "Maruti Suzuki",
    "TATAMOTORS.NS": "Tata Motors",
    "SUNPHARMA.NS": "Sun Pharmaceutical",
    "TITAN.NS": "Titan Company",
    "BAJFINANCE.NS": "Bajaj Finance",
}

STOCK_BASE_PRICES = {
    "RELIANCE.NS": 2950, "TCS.NS": 3820, "INFY.NS": 1580,
    "HDFCBANK.NS": 1720, "ICICIBANK.NS": 1250, "HINDUNILVR.NS": 2480,
    "BHARTIARTL.NS": 1650, "ITC.NS": 465, "SBIN.NS": 835,
    "KOTAKBANK.NS": 1880, "LT.NS": 3550, "AXISBANK.NS": 1180,
    "WIPRO.NS": 455, "HCLTECH.NS": 1620, "ASIANPAINT.NS": 2310,
    "MARUTI.NS": 12450, "TATAMOTORS.NS": 720, "SUNPHARMA.NS": 1780,
    "TITAN.NS": 3380, "BAJFINANCE.NS": 6850,
}

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

    selected_ticker = st.selectbox(
        "🏢 Select Stock",
        options=list(STOCKS.keys()),
        format_func=lambda x: f"{STOCKS[x]} ({x.replace('.NS', '')})",
        index=0,
    )

    prediction_horizon = st.slider(
        "📅 Prediction Horizon (days)", min_value=1, max_value=30, value=7
    )

    st.markdown("##### 🤖 Models")
    use_lstm = st.checkbox("LSTM Neural Network", value=True)
    use_rf = st.checkbox("Random Forest", value=True)

    st.markdown("---")
    st.markdown("""
    <div style="background:rgba(108,99,255,0.08); border-radius:12px; padding:14px 16px; font-size:0.82rem; color:#8892A5; line-height:1.6;">
        <strong style="color:#A8A2FF;">⚠️ Disclaimer</strong><br>
        Predictions are generated by ML models trained on historical data.
        They should <u>not</u> be the sole basis for investment decisions.
    </div>
    """, unsafe_allow_html=True)

# ── Mock Data Generation ─────────────────────────────────────────────────────
np.random.seed(42 + hash(selected_ticker) % 1000)
base_price = STOCK_BASE_PRICES[selected_ticker]
stock_name = STOCKS[selected_ticker]
ticker_short = selected_ticker.replace(".NS", "")

# Historical prices (last 90 days)
hist_days = 90
today = datetime(2026, 7, 8)
hist_dates = [today - timedelta(days=hist_days - i) for i in range(hist_days)]

returns = np.random.normal(0.0006, 0.018, hist_days)
hist_prices = [base_price]
for r in returns[1:]:
    hist_prices.append(hist_prices[-1] * (1 + r))
hist_prices = np.array(hist_prices)

# Predicted prices (LSTM — slight upward bias)
np.random.seed(100 + hash(selected_ticker) % 500)
pred_dates = [today + timedelta(days=i + 1) for i in range(prediction_horizon)]
pred_returns = np.random.normal(0.0025, 0.012, prediction_horizon)
pred_prices = [hist_prices[-1]]
for r in pred_returns:
    pred_prices.append(pred_prices[-1] * (1 + r))
pred_prices = np.array(pred_prices[1:])

# Confidence bands
spread = np.linspace(0.008, 0.028, prediction_horizon) * pred_prices
upper_band = pred_prices + spread
lower_band = pred_prices - spread

# RF direction probabilities (mock)
rf_up = 73.2
rf_flat = 18.5
rf_down = 8.3

# ── Page Header ──────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="prediction-header">
    <h1>📈 Price Predictions & Model Analysis</h1>
    <p>ML-powered price forecasts and directional signals for <strong style="color:#A8A2FF">{stock_name}</strong>
    ({ticker_short}) · {prediction_horizon}-day horizon</p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🧠 LSTM Prediction",
    "🌲 Random Forest",
    "⚖️ Model Comparison",
    "🔍 Feature Importance",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LSTM PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    if not use_lstm:
        st.warning("Enable the **LSTM Neural Network** model in the sidebar to view predictions.")
    else:
        # Info card
        st.markdown("""
        <div class="info-card">
            <strong>🧠 LSTM Neural Network</strong> — Our LSTM (Long Short-Term Memory) neural network
            analyzes <strong>60 days of historical price data</strong>, volume patterns, and technical indicators
            to forecast future closing prices. The model uses a 3-layer stacked LSTM architecture with
            dropout regularisation, trained on 5 years of NSE data with a 80/20 train-test split.
            Predictions include confidence intervals derived from Monte Carlo dropout sampling.
        </div>
        """, unsafe_allow_html=True)

        # ── Plotly chart ──
        fig = go.Figure()

        # Historical line
        fig.add_trace(go.Scatter(
            x=hist_dates, y=hist_prices,
            mode="lines",
            name="Historical Price",
            line=dict(color="#6C63FF", width=2.2),
            hovertemplate="₹%{y:,.2f}<extra>Historical</extra>",
        ))

        # Confidence band
        fig.add_trace(go.Scatter(
            x=list(pred_dates) + list(pred_dates)[::-1],
            y=list(upper_band) + list(lower_band)[::-1],
            fill="toself",
            fillcolor="rgba(0, 212, 170, 0.08)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Confidence Band",
            hoverinfo="skip",
            showlegend=True,
        ))

        # Predicted line
        fig.add_trace(go.Scatter(
            x=[hist_dates[-1]] + pred_dates,
            y=[hist_prices[-1]] + list(pred_prices),
            mode="lines+markers",
            name="LSTM Prediction",
            line=dict(color="#00D4AA", width=2.5, dash="dash"),
            marker=dict(size=5, color="#00D4AA"),
            hovertemplate="₹%{y:,.2f}<extra>Predicted</extra>",
        ))

        # Divider line at today
        fig.add_vline(
            x=today.timestamp() * 1000, line_width=1,
            line_dash="dot", line_color="rgba(255,179,71,0.5)",
            annotation_text="Today", annotation_position="top",
            annotation_font_color="#FFB347",
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(14,17,23,0.6)",
            height=480,
            title=dict(
                text=f"{ticker_short} — LSTM Price Forecast ({prediction_horizon} days)",
                font=dict(size=16, color="#E0E0E0"),
            ),
            xaxis=dict(title="Date", gridcolor="rgba(108,99,255,0.08)"),
            yaxis=dict(title="Price (₹)", gridcolor="rgba(108,99,255,0.08)", tickprefix="₹"),
            legend=dict(
                bgcolor="rgba(26,31,46,0.8)", bordercolor="rgba(108,99,255,0.2)",
                borderwidth=1, font=dict(size=11),
            ),
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Metrics row ──
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("📉 RMSE", f"₹45.23", help="Root Mean Squared Error on test set")
        m2.metric("📊 MAE", f"₹32.15", help="Mean Absolute Error on test set")
        m3.metric("📈 R² Score", "0.847", help="Coefficient of determination")
        m4.metric("🎯 Direction Accuracy", "68.5%", help="Correct up/down prediction rate")

        # Status badge
        st.markdown("""
        <div style="text-align:center; margin-top:6px;">
            <span class="status-badge">🟢 Model Trained  |  Last updated: July 8, 2026</span>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — RANDOM FOREST
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    if not use_rf:
        st.warning("Enable the **Random Forest** model in the sidebar to view predictions.")
    else:
        # Info card
        st.markdown("""
        <div class="info-card">
            <strong>🌲 Random Forest Classifier</strong> — Random Forest classifier predicts market direction
            (<strong>UP / DOWN / FLAT</strong>) using an ensemble of 500 decision trees trained on 15+
            technical indicators. The model uses a ±0.5% threshold for direction labelling, with
            class-weight balancing to handle directional imbalance in training data.
        </div>
        """, unsafe_allow_html=True)

        # ── Direction prediction card ──
        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown(f"""
            <div class="direction-card">
                <div class="direction-label">Predicted Direction · Next {prediction_horizon} Day{'s' if prediction_horizon > 1 else ''}</div>
                <div class="direction-value">📈 UP</div>
                <div class="direction-confidence">Confidence: <strong>73.2%</strong></div>
            </div>
            """, unsafe_allow_html=True)

            # Extra detail
            target_price = hist_prices[-1] * 1.032
            st.markdown(f"""
            <div class="glass-card" style="text-align:center;">
                <span style="color:#8892A5; font-size:0.9rem;">Implied Target Range</span><br>
                <span style="color:#00D4AA; font-size:1.6rem; font-weight:700;">
                    ₹{hist_prices[-1] * 0.99:,.0f} — ₹{target_price:,.0f}
                </span>
            </div>
            """, unsafe_allow_html=True)

        with col_right:
            # Probability bar chart
            prob_fig = go.Figure()
            categories = ["UP", "FLAT", "DOWN"]
            probs = [rf_up, rf_flat, rf_down]
            colors = ["#00D4AA", "#FFB347", "#FF6B6B"]

            prob_fig.add_trace(go.Bar(
                y=categories, x=probs,
                orientation="h",
                marker=dict(
                    color=colors,
                    line=dict(width=0),
                    cornerradius=6,
                ),
                text=[f"{p}%" for p in probs],
                textposition="auto",
                textfont=dict(size=14, color="#FFFFFF", family="Inter"),
            ))
            prob_fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(14,17,23,0.6)",
                height=260,
                title=dict(text="Direction Probability Breakdown", font=dict(size=14, color="#C5CEE0")),
                xaxis=dict(
                    title="Probability (%)", range=[0, 100],
                    gridcolor="rgba(108,99,255,0.08)",
                ),
                yaxis=dict(gridcolor="rgba(108,99,255,0.08)"),
                margin=dict(l=10, r=20, t=40, b=30),
                showlegend=False,
            )
            st.plotly_chart(prob_fig, use_container_width=True)

        # ── Classification report ──
        st.markdown("#### 📋 Classification Report (Test Set)")
        report_df = pd.DataFrame({
            "Class": ["UP ↑", "DOWN ↓", "FLAT →", "— Macro Avg", "— Weighted Avg"],
            "Precision": [0.74, 0.71, 0.63, 0.69, 0.71],
            "Recall": [0.78, 0.67, 0.58, 0.68, 0.70],
            "F1-Score": [0.76, 0.69, 0.60, 0.68, 0.70],
            "Support": [312, 278, 160, 750, 750],
        })
        st.dataframe(
            report_df.style.format({
                "Precision": "{:.2f}",
                "Recall": "{:.2f}",
                "F1-Score": "{:.2f}",
            }).set_properties(**{
                "background-color": "rgba(26,31,46,0.8)",
                "color": "#C5CEE0",
            }),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("""
        <div style="text-align:center; margin-top:4px;">
            <span class="status-badge">🟢 Model Trained  |  Accuracy: 70.1%  |  Last updated: July 8, 2026</span>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    if not (use_lstm and use_rf):
        st.info("Enable **both** models in the sidebar to see a full comparison.")
    else:
        # ── Side-by-side metrics table ──
        st.markdown("#### ⚖️ Head-to-Head: LSTM vs Random Forest")
        st.markdown("""
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
                <tr><td><strong>RMSE</strong></td><td>₹45.23</td><td>N/A</td></tr>
                <tr><td><strong>MAE</strong></td><td>₹32.15</td><td>N/A</td></tr>
                <tr><td><strong>R² Score</strong></td><td>0.847</td><td>N/A</td></tr>
                <tr><td><strong>Direction Accuracy</strong></td><td>68.5%</td><td>70.1%</td></tr>
                <tr><td><strong>Training Time</strong></td><td>~12 min (GPU)</td><td>~45 sec (CPU)</td></tr>
                <tr><td><strong>Interpretability</strong></td><td>Low (black box)</td><td>High (feature importance)</td></tr>
                <tr><td><strong>Best For</strong></td><td>Price targets & bands</td><td>Buy / Sell signals</td></tr>
            </tbody>
        </table>
        """, unsafe_allow_html=True)

        # ── Overlay chart ──
        overlay_fig = go.Figure()

        # Historical
        overlay_fig.add_trace(go.Scatter(
            x=hist_dates[-30:], y=hist_prices[-30:],
            mode="lines", name="Historical",
            line=dict(color="#6C63FF", width=2),
            hovertemplate="₹%{y:,.2f}<extra>Historical</extra>",
        ))

        # LSTM prediction
        overlay_fig.add_trace(go.Scatter(
            x=[hist_dates[-1]] + pred_dates,
            y=[hist_prices[-1]] + list(pred_prices),
            mode="lines+markers", name="LSTM Prediction",
            line=dict(color="#00D4AA", width=2.5, dash="dash"),
            marker=dict(size=4),
            hovertemplate="₹%{y:,.2f}<extra>LSTM</extra>",
        ))

        # RF direction arrows
        np.random.seed(77)
        rf_directions = np.random.choice(["UP", "UP", "UP", "FLAT", "DOWN"], size=prediction_horizon)
        arrow_colors = {"UP": "#00D4AA", "DOWN": "#FF6B6B", "FLAT": "#FFB347"}
        arrow_symbols = {"UP": "triangle-up", "DOWN": "triangle-down", "FLAT": "diamond"}

        for i, d in enumerate(rf_directions):
            overlay_fig.add_trace(go.Scatter(
                x=[pred_dates[i]], y=[pred_prices[i]],
                mode="markers",
                marker=dict(
                    symbol=arrow_symbols[d], size=14,
                    color=arrow_colors[d],
                    line=dict(width=1.5, color="#FFFFFF"),
                ),
                name=f"RF: {d}" if i < 3 else None,
                showlegend=(i < 3 and d not in [rf_directions[j] for j in range(i)]),
                hovertemplate=f"RF: {d}<extra></extra>",
            ))

        overlay_fig.add_vline(
            x=today.timestamp() * 1000, line_width=1,
            line_dash="dot", line_color="rgba(255,179,71,0.5)",
            annotation_text="Today", annotation_position="top",
            annotation_font_color="#FFB347",
        )

        overlay_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(14,17,23,0.6)",
            height=440,
            title=dict(
                text=f"{ticker_short} — LSTM Line + Random Forest Direction Arrows",
                font=dict(size=15, color="#E0E0E0"),
            ),
            xaxis=dict(title="Date", gridcolor="rgba(108,99,255,0.08)"),
            yaxis=dict(title="Price (₹)", gridcolor="rgba(108,99,255,0.08)", tickprefix="₹"),
            legend=dict(bgcolor="rgba(26,31,46,0.8)", bordercolor="rgba(108,99,255,0.2)", borderwidth=1),
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(overlay_fig, use_container_width=True)

        # ── Trust card ──
        st.markdown(f"""
        <div class="trust-card">
            <h4>💡 Which Model Should You Trust?</h4>
            <p>For <strong style="color:#A8A2FF">{stock_name}</strong>, both models currently agree on a
            <strong style="color:#00D4AA">bullish outlook</strong>. Here's our AI-generated recommendation:</p>
            <ul>
                <li><strong>Use LSTM</strong> when you need a precise <em>price target</em> for setting stop-losses
                or limit orders. The confidence band gives you a realistic range.</li>
                <li><strong>Use Random Forest</strong> when you need a quick <em>directional signal</em>
                (buy / hold / sell). Its higher direction accuracy (70.1% vs 68.5%) makes it slightly
                more reliable for binary decisions.</li>
                <li><strong>When models disagree</strong>, lean towards the Random Forest for short-term trades
                (1-5 days) and the LSTM for medium-term positions (5-30 days).</li>
            </ul>
            <p style="color:#8892A5; font-size:0.85rem; margin-bottom:0;">
                ⚠️ Always combine model signals with fundamental analysis and market sentiment before acting.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("#### 🔍 Random Forest — Feature Importance")
    st.markdown("""
    <div class="info-card">
        Feature importance scores are derived from the Random Forest's Gini impurity reduction
        across all 500 trees. Higher values indicate features that contribute more to the model's
        directional predictions.
    </div>
    """, unsafe_allow_html=True)

    # Feature importance data
    features = ["Bollinger %B", "SMA_50", "Close", "MACD", "SMA_20", "RSI", "Volume"]
    importances = [0.04, 0.06, 0.10, 0.15, 0.18, 0.22, 0.25]
    bar_colors = [
        "#4ECDC4", "#4ECDC4", "#6C63FF", "#6C63FF",
        "#A8A2FF", "#00D4AA", "#00D4AA",
    ]

    fi_fig = go.Figure()
    fi_fig.add_trace(go.Bar(
        y=features, x=importances,
        orientation="h",
        marker=dict(color=bar_colors, line=dict(width=0), cornerradius=5),
        text=[f"{v:.0%}" for v in importances],
        textposition="outside",
        textfont=dict(size=13, color="#C5CEE0", family="Inter"),
    ))
    fi_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(14,17,23,0.6)",
        height=380,
        xaxis=dict(
            title="Importance Score", range=[0, 0.32],
            gridcolor="rgba(108,99,255,0.08)",
            tickformat=".0%",
        ),
        yaxis=dict(gridcolor="rgba(108,99,255,0.08)"),
        margin=dict(l=10, r=40, t=20, b=30),
        showlegend=False,
    )
    st.plotly_chart(fi_fig, use_container_width=True)

    # ── SHAP-like explanations ──
    st.markdown("#### 🧪 Feature Explanations (Top 3)")

    st.markdown(f"""
    <div class="shap-card">
        <h5>1. Volume — Importance: 25%</h5>
        Trading volume was the <strong style="color:#00D4AA">strongest predictor</strong> of direction
        for {ticker_short}. Days with volume &gt;1.5× the 20-day average volume were followed by an
        upward move <strong>72%</strong> of the time. This suggests institutional accumulation often
        precedes price increases, and the model heavily relies on volume spikes as a lead indicator.
    </div>

    <div class="shap-card">
        <h5>2. RSI (14) — Importance: 22%</h5>
        The Relative Strength Index contributed the second-most predictive power. RSI values between
        <strong style="color:#FFB347">40–60</strong> (neutral zone) were associated with FLAT predictions,
        while RSI &lt;30 strongly predicted UP moves (mean-reversion signal) and RSI &gt;70 predicted
        DOWN moves with <strong>65%</strong> accuracy. The model effectively learned oversold/overbought dynamics.
    </div>

    <div class="shap-card">
        <h5>3. SMA_20 — Importance: 18%</h5>
        The 20-day Simple Moving Average acts as a <strong style="color:#A8A2FF">trend confirmation</strong> feature.
        When the closing price crossed above SMA_20, the model predicted UP with <strong>69%</strong> confidence.
        The distance between price and SMA_20 (as a percentage) was particularly informative — wider gaps
        increased the model's conviction in trend continuation.
    </div>
    """, unsafe_allow_html=True)

    # Correlation heatmap (bonus)
    with st.expander("📊 Feature Correlation Matrix", expanded=False):
        np.random.seed(42)
        corr_features = ["Close", "Volume", "RSI", "MACD", "SMA_20", "SMA_50", "Boll %B"]
        n = len(corr_features)
        corr_matrix = np.eye(n)
        raw = np.random.uniform(-0.3, 0.8, (n, n))
        raw = (raw + raw.T) / 2
        np.fill_diagonal(raw, 1.0)
        # Make it look realistic
        raw[0, 4] = raw[4, 0] = 0.92   # Close <-> SMA_20
        raw[0, 5] = raw[5, 0] = 0.88   # Close <-> SMA_50
        raw[4, 5] = raw[5, 4] = 0.95   # SMA_20 <-> SMA_50
        raw[2, 3] = raw[3, 2] = 0.35   # RSI <-> MACD
        raw[1, 6] = raw[6, 1] = -0.15  # Volume <-> Boll %B

        corr_fig = go.Figure(data=go.Heatmap(
            z=raw, x=corr_features, y=corr_features,
            colorscale=[[0, "#FF6B6B"], [0.5, "#1A1F2E"], [1, "#00D4AA"]],
            zmin=-1, zmax=1,
            text=np.round(raw, 2),
            texttemplate="%{text}",
            textfont=dict(size=11),
        ))
        corr_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(14,17,23,0.6)",
            height=400,
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(corr_fig, use_container_width=True)
