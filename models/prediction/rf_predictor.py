"""
NiveshAI — Random Forest Direction Predictor
Wrapper for running predictions using the trained scikit-learn Random Forest model.
Classifies price direction for the next day into UP, DOWN, or FLAT.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any

# Constants
FEATURES = ["Close", "Volume", "SMA_20", "SMA_50", "RSI_14", "MACD", "BB_pctB", "Vol_Change", "Price_Change"]
CLASSES = {0: "FLAT", 1: "UP", -1: "DOWN"}


class RFPredictor:
    """Wrapper for scikit-learn Random Forest classifier model."""

    def __init__(self, model_path: Optional[str] = None, scaler_path: Optional[str] = None):
        self.available = False
        self.model = None
        self.scaler = None
        self.feature_importance_metadata = {}
        self.accuracy_metadata = None

        project_root = Path(__file__).parent.parent.parent
        if model_path is None:
            model_path = str(project_root / "models" / "saved" / "rf_model.pkl")
        if scaler_path is None:
            scaler_path = str(project_root / "models" / "saved" / "rf_scaler.pkl")

        self.model_path = model_path
        self.scaler_path = scaler_path

        self._load_local_assets()

    def _load_local_assets(self):
        """Load joblib/pickle model and scaler files."""
        if not os.path.exists(self.model_path) or not os.path.exists(self.scaler_path):
            print(f"[WARNING] RF Predictor: Model or scaler assets not found. Path: {self.model_path}")
            self.available = False
            return

        try:
            # We can use joblib if installed, fallback to pickle
            try:
                import joblib
                loaded_dict = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
            except ImportError:
                with open(self.model_path, "rb") as f:
                    loaded_dict = pickle.load(f)
                with open(self.scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)

            if isinstance(loaded_dict, dict) and "model" in loaded_dict:
                self.model = loaded_dict["model"]
                self.feature_importance_metadata = loaded_dict.get("feature_importance", {})
                self.accuracy_metadata = loaded_dict.get("accuracy", None)
            else:
                self.model = loaded_dict

            self.available = True
            print("[OK] RF Predictor: Random Forest model and scaler loaded successfully.")
        except Exception as e:
            print(f"[ERROR] RF Predictor: Failed to load assets: {e}")
            self.available = False

    def _prepare_features(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Extract the latest features, compute missing technical indicators, and scale them.
        Returns scaled array of shape (1, 9) representing the latest step.
        """
        import ta
        df = df.copy()

        # Handle columns case-insensitively for basic OHLCV
        rename = {c: c.capitalize() for c in df.columns if c.lower() in ["open", "high", "low", "close", "volume"]}
        df.rename(columns=rename, inplace=True)

        if "Close" not in df.columns:
            return None

        try:
            # Compute indicators if they are missing
            if "SMA_20" not in df.columns:
                df["SMA_20"] = df["Close"].rolling(20).mean()
            if "SMA_50" not in df.columns:
                df["SMA_50"] = df["Close"].rolling(50).mean()
            if "RSI_14" not in df.columns:
                df["RSI_14"] = ta.momentum.rsi(df["Close"], window=14)
            
            if "MACD" not in df.columns:
                macd_obj = ta.trend.MACD(df["Close"])
                df["MACD"] = macd_obj.macd_diff()

            if "BB_pctB" not in df.columns:
                boll = ta.volatility.BollingerBands(df["Close"])
                df["BB_pctB"] = boll.bollinger_pband()

            if "Vol_Change" not in df.columns and "Volume" in df.columns:
                df["Vol_Change"] = df["Volume"].pct_change()
            if "Price_Change" not in df.columns:
                df["Price_Change"] = df["Close"].pct_change()

            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(subset=FEATURES, inplace=True)

            if df.empty:
                return None

            # Get the latest row as features
            latest_row = df[FEATURES].tail(1).values

            # Scale data using the RF scaler (StandardScaler)
            scaled_row = self.scaler.transform(latest_row)
            return scaled_row
        except Exception as e:
            print(f"[WARNING] RF Feature preparation failed: {e}")
            return None

    def predict_direction(self, history_df: pd.DataFrame) -> Optional[dict]:
        """
        Predict price direction (UP, DOWN, FLAT) for the next trading day.
        Returns:
            {
                "direction": "UP" | "DOWN" | "FLAT",
                "probabilities": {"UP": float, "DOWN": float, "FLAT": float},
                "confidence": float,
                "signal_strength": "Strong" | "Moderate" | "Weak",
                "disclaimer": "36% accuracy — use as one signal among many"
            }
        """
        if not self.available or self.model is None or self.scaler is None:
            return None

        scaled_features = self._prepare_features(history_df)
        if scaled_features is None:
            return None

        try:
            # Predict probabilities
            probs = self.model.predict_proba(scaled_features)[0] # Shape: (3,) corresponding to classes [-1, 0, 1]
            
            # scikit-learn classes_ contains the labels in sorted order (usually [-1, 0, 1])
            classes = self.model.classes_
            
            # Map probabilities to classes
            probs_dict = {"DOWN": 0.0, "FLAT": 0.0, "UP": 0.0}
            for c_label, prob in zip(classes, probs):
                if c_label == 1:
                    probs_dict["UP"] = float(prob)
                elif c_label == -1:
                    probs_dict["DOWN"] = float(prob)
                else:
                    probs_dict["FLAT"] = float(prob)

            # Get predicted class (highest probability)
            predicted_class_val = int(self.model.predict(scaled_features)[0])
            direction = CLASSES.get(predicted_class_val, "FLAT")
            
            # Calculate confidence as max probability
            confidence = float(max(probs_dict.values()))

            # Signal strength: confidence > 0.5 = "Moderate", > 0.65 = "Strong", else "Weak"
            if confidence > 0.65:
                signal_strength = "Strong"
            elif confidence > 0.50:
                signal_strength = "Moderate"
            else:
                signal_strength = "Weak"

            return {
                "direction": direction,
                "probabilities": probs_dict,
                "confidence": round(confidence, 4),
                "signal_strength": signal_strength,
                "disclaimer": "36% accuracy — use as one signal among many"
            }

        except Exception as e:
            print(f"[ERROR] RF prediction failed: {e}")
            return None

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance mapping from the Random Forest model."""
        if not self.available or self.model is None:
            return {}

        if self.feature_importance_metadata:
            return self.feature_importance_metadata

        try:
            importances = self.model.feature_importances_
            importance_dict = {feat: float(imp) for feat, imp in zip(FEATURES, importances)}
            # Sort descending
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        except Exception as e:
            print(f"[WARNING] Failed to extract feature importances: {e}")
            return {}


# Module-level lazy singleton
_rf_predictor: Optional[RFPredictor] = None


def get_rf_predictor() -> RFPredictor:
    """Get the global RFPredictor instance."""
    global _rf_predictor
    if _rf_predictor is None:
        _rf_predictor = RFPredictor()
    return _rf_predictor
