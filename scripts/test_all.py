"""
NiveshAI — Integration Test Runner
Tests all 11 core components of the application pipeline.
Run this using: python scripts/test_all.py
"""

import os
import sys
import pandas as pd
import numpy as np

# Adjust sys.path to run directly from the scripts directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import core modules
from data.cache import get_cache
from data.company_db import get_company, get_display_options
from data.stock_data import fetch_live_price, fetch_history
from data.news_fetcher import fetch_news
from analysis.technical import compute_all_indicators, generate_signals
from analysis.fundamental import analyze_fundamentals
from analysis.risk import assess_risk
from portfolio.optimizer import optimize_portfolio
from models.sentiment.sentiment_analyzer import get_sentiment_analyzer
from models.prediction.lstm_predictor import get_lstm_predictor
from models.prediction.rf_predictor import get_rf_predictor


def run_tests():
    print("=" * 60)
    print("         NiveshAI Integration Test Runner")
    print("=" * 60)
    
    passed_tests = 0
    total_tests = 11

    # 1. Cache
    print("\n[1/11] Testing data/cache.py ...")
    try:
        cache = get_cache()
        cache.set("test_key", {"status": "ok"}, ttl_hours=1)
        val = cache.get("test_key")
        if val and val.get("status") == "ok":
            print("  [OK] data/cache.py - Cache set/get works")
            passed_tests += 1
        else:
            print("  [FAIL] data/cache.py - Cache get returned wrong value")
    except Exception as e:
        print(f"  [FAIL] data/cache.py - Error: {e}")

    # 2. Company DB
    print("\n[2/11] Testing data/company_db.py ...")
    try:
        companies = get_display_options()
        tcs_info = get_company("TCS")
        if companies and tcs_info and tcs_info.get("Company Name"):
            print("  [OK] data/company_db.py - Loaded NIFTY 500 options")
            passed_tests += 1
        else:
            print("  [FAIL] data/company_db.py - Empty display options or company lookup")
    except Exception as e:
        print(f"  [FAIL] data/company_db.py - Error: {e}")

    # 3. Stock Data Live Fetcher
    print("\n[3/11] Testing data/stock_data.py live fetcher ...")
    try:
        live = fetch_live_price("RELIANCE")
        if live and live.get("price") > 0:
            print(f"  [OK] data/stock_data.py - Live price: INR {live['price']}")
            passed_tests += 1
        else:
            print("  [FAIL] data/stock_data.py - Live price is 0 or missing keys")
    except Exception as e:
        print(f"  [FAIL] data/stock_data.py - Error: {e}")

    # 4. News Fetcher
    print("\n[4/11] Testing data/news_fetcher.py ...")
    try:
        news = fetch_news("TCS", max_articles=2)
        if isinstance(news, list):
            print(f"  [OK] data/news_fetcher.py - Fetched news articles: {len(news)}")
            passed_tests += 1
        else:
            print("  [FAIL] data/news_fetcher.py - Did not return a list")
    except Exception as e:
        print(f"  [FAIL] data/news_fetcher.py - Error: {e}")

    # 5. Technical Indicators
    print("\n[5/11] Testing analysis/technical.py ...")
    try:
        dates = pd.date_range(end="2026-07-08", periods=65, freq="D")
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(65) * 2)
        dummy_df = pd.DataFrame({
            "Open": close - 0.5, "High": close + 0.8,
            "Low": close - 0.8, "Close": close,
            "Volume": np.random.randint(1000, 5000, 65)
        }, index=dates)
        
        indicators_df = compute_all_indicators(dummy_df)
        signals = generate_signals(indicators_df)
        if "RSI" in indicators_df.columns and signals.get("overall"):
            print("  [OK] analysis/technical.py - Computed indicators & signals successfully")
            passed_tests += 1
        else:
            print("  [FAIL] analysis/technical.py - Indicators missing or signals failed")
    except Exception as e:
        print(f"  [FAIL] analysis/technical.py - Error: {e}")

    # 6. Fundamentals Analysis
    print("\n[6/11] Testing analysis/fundamental.py ...")
    try:
        mock_fundamentals = {
            "symbol": "TCS",
            "company_name": "Tata Consultancy Services",
            "pe_ratio": 30.2,
            "dividend_yield": 0.012,
            "roe": 0.38,
            "profit_margin": 0.18,
            "debt_to_equity": 0.08,
        }
        analysis = analyze_fundamentals(mock_fundamentals)
        if "rating" in analysis and "score" in analysis:
            print(f"  [OK] analysis/fundamental.py - Score: {analysis['score']}/100, Rating: {analysis['rating']}")
            passed_tests += 1
        else:
            print("  [FAIL] analysis/fundamental.py - Analysis missing keys")
    except Exception as e:
        print(f"  [FAIL] analysis/fundamental.py - Error: {e}")

    # 7. Risk Assessor
    print("\n[7/11] Testing analysis/risk.py ...")
    try:
        np.random.seed(42)
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.015, 252)))
        history_df = pd.DataFrame({"Close": prices})
        risk = assess_risk(history_df)
        if "var_95" in risk and "volatility" in risk:
            print(f"  [OK] analysis/risk.py - Volatility: {risk['volatility']:.2%}, 95% VaR: {risk['var_95']:.2%}")
            passed_tests += 1
        else:
            print("  [FAIL] analysis/risk.py - Risk results missing keys")
    except Exception as e:
        print(f"  [FAIL] analysis/risk.py - Error: {e}")

    # 8. Portfolio Optimizer
    print("\n[8/11] Testing portfolio/optimizer.py ...")
    try:
        np.random.seed(42)
        dates = pd.date_range(end="2026-07-08", periods=100, freq="B")
        histories = {}
        for sym in ["RELIANCE", "TCS", "INFY"]:
            prices = 100 * np.exp(np.cumsum(np.random.normal(0.0008, 0.012, 100)))
            histories[sym] = pd.DataFrame({"Close": prices}, index=dates)
            
        opt = optimize_portfolio(histories, investment_amount=10_00_000, risk_tolerance="moderate")
        if not opt.get("error") and "weights" in opt:
            print(f"  [OK] portfolio/optimizer.py - Optimized successfully: {opt['weights']}")
            passed_tests += 1
        else:
            print(f"  [FAIL] portfolio/optimizer.py - Error: {opt.get('error')}")
    except Exception as e:
        print(f"  [FAIL] portfolio/optimizer.py - Error: {e}")

    # 9. Sentiment Analyzer
    print("\n[9/11] Testing models/sentiment (BERT) ...")
    try:
        analyzer = get_sentiment_analyzer()
        res = analyzer.analyze("TCS profit exceeds expectations, shares jump 5%")
        if res and "label" in res:
            print(f"  [OK] models/sentiment - Label: {res['label']}, Confidence: {res['score']:.0%}")
            passed_tests += 1
        else:
            print("  [FAIL] models/sentiment - Returned incorrect format")
    except Exception as e:
        print(f"  [FAIL] models/sentiment - Error: {e}")

    # 10. LSTM Predictor
    print("\n[10/11] Testing models/prediction LSTM ...")
    try:
        dates = pd.date_range(end="2026-07-08", periods=75, freq="D")
        np.random.seed(42)
        close = 1500.0 + np.cumsum(np.random.randn(75) * 10)
        df_lstm = pd.DataFrame({
            "Open": close - 5.0, "High": close + 8.0,
            "Low": close - 8.0, "Close": close,
            "Volume": np.random.randint(1000000, 5000000, 75)
        }, index=dates)
        df_lstm = compute_all_indicators(df_lstm)
        
        lstm_p = get_lstm_predictor()
        res_lstm = lstm_p.predict_next_days(df_lstm, n_days=7)
        if res_lstm and "predicted_7d" in res_lstm:
            print(f"  [OK] models/prediction LSTM - Forecast: INR {res_lstm['predicted_7d']:,.2f} ({res_lstm['change_pct']:+.2f}%)")
            passed_tests += 1
        else:
            print("  [FAIL] models/prediction LSTM - Predictor failed or returned None")
    except Exception as e:
        print(f"  [FAIL] models/prediction LSTM - Error: {e}")

    # 11. Random Forest Predictor
    print("\n[11/11] Testing models/prediction RF ...")
    try:
        rf_p = get_rf_predictor()
        res_rf = rf_p.predict_direction(df_lstm)
        if res_rf and "direction" in res_rf:
            print(f"  [OK] models/prediction RF - Predicted: {res_rf['direction']} (Confidence: {res_rf['confidence']:.1%})")
            passed_tests += 1
        else:
            print("  [FAIL] models/prediction RF - Predictor failed or returned None")
    except Exception as e:
        print(f"  [FAIL] models/prediction RF - Error: {e}")

    # Summary
    print("\n" + "=" * 60)
    print(f"  Integration Test Summary: {passed_tests} / {total_tests} PASSED")
    print("=" * 60)
    
    if passed_tests == total_tests:
        print("  [SUCCESS] ALL core components are fully operational!")
        return 0
    else:
        print("  [WARNING] Some tests failed. Please review the errors above.")
        print("\n=== Common Troubleshooting Guide & Fixes ===")
        print("1. ModuleNotFoundError: Run `pip install -r requirements.txt` to install missing packages.")
        print("2. Models missing / load errors: Verify that model weights have been uploaded to Hugging Face Hub,")
        print("   your HF_MODEL_REPO env variable is correct, and that `scripts/download_models_from_hf.py` can connect.")
        print("3. Company DB failures: Re-run `python scripts/build_nifty500_db.py` to compile company constituents database.")
        print("4. yfinance / NewsAPI errors: Check internet connectivity and verify API keys in your .env or st.secrets.")
        print("============================================")
        return 1


if __name__ == "__main__":
    sys.exit(run_tests())
