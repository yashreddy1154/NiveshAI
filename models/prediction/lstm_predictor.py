"""
NiveshAI — LSTM Stock Price Predictor
Inference wrapper for the 2-layer LSTM model using PyTorch.
Loads model weights and handles sliding window rolling forecasts.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

# Constants
FEATURES = ["Close", "Volume", "SMA_20", "SMA_50", "RSI_14", "MACD", "BB_pctB", "Vol_Change", "Price_Change"]
WINDOW_SIZE = 60


class StockLSTM(nn.Module):
    """LSTM model for stock price prediction."""

    def __init__(self, input_size: int = 9, hidden_size: int = 128, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: [batch_size, seq_len, input_size]
        out, _ = self.lstm(x)
        # Take the output of the last sequence step: [batch_size, hidden_size]
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out


class LSTMPredictor:
    """Wrapper for running LSTM inference."""

    def __init__(self, model_path: str, scaler_path: str, device: Optional[str] = None):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.available = False
        self.model = None
        self.scaler = None
        self.device = None

        # Device config
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._load_local_assets()

    def _load_local_assets(self):
        """Load local .pt model state dict and scaler pickle."""
        if not self.model_path or not self.scaler_path:
            print("[WARNING] LSTM Predictor: Paths not provided.")
            return

        if not os.path.exists(self.model_path) or not os.path.exists(self.scaler_path):
            print(f"[WARNING] LSTM Predictor: Model or scaler assets not found. Paths: {self.model_path}, {self.scaler_path}")
            return

        try:
            # Load scaler
            with open(self.scaler_path, "rb") as f:
                self.scaler = pickle.load(f)

            # Load model state dict
            self.model = StockLSTM(input_size=9, hidden_size=128, num_layers=2)
            state_dict = torch.load(self.model_path, map_location=self.device)
            
            # If the state_dict is wrapped in a checkpoint dictionary, extract it
            if hasattr(state_dict, "keys") and "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            elif isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
                
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()

            self.available = True
            print("[OK] LSTM Predictor: Model and scaler loaded successfully.")
        except Exception as e:
            print(f"[ERROR] LSTM Predictor: Failed to load assets: {e}")
            self.available = False

    def _prepare_features(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Extract features, compute missing technical indicators, and scale them.
        Returns scaled array of shape (1, 60, 9) or None if insufficient rows.
        """
        import ta
        df = df.copy()

        # Handle columns case-insensitively
        rename = {c: c.capitalize() for c in df.columns if c.lower() in ["open", "high", "low", "close", "volume"]}
        df.rename(columns=rename, inplace=True)

        if "Close" not in df.columns or "Volume" not in df.columns:
            return None

        try:
            # Compute missing indicators
            if "SMA_20" not in df.columns:
                df["SMA_20"] = df["Close"].rolling(20).mean()
            if "SMA_50" not in df.columns:
                df["SMA_50"] = df["Close"].rolling(50).mean()
            if "RSI_14" not in df.columns:
                df["RSI_14"] = ta.momentum.rsi(df["Close"], window=14)
            
            if "MACD" not in df.columns:
                macd_obj = ta.trend.MACD(df["Close"])
                df["MACD"] = macd_obj.macd_diff()  # MACD diff matches the scaler.pkl distribution

            if "BB_pctB" not in df.columns:
                boll = ta.volatility.BollingerBands(df["Close"])
                df["BB_pctB"] = boll.bollinger_pband()

            if "Vol_Change" not in df.columns:
                df["Vol_Change"] = df["Volume"].pct_change()
            if "Price_Change" not in df.columns:
                df["Price_Change"] = df["Close"].pct_change()

            # Clean and backfill NaNs to preserve sequence length
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.bfill(inplace=True)
            df.ffill(inplace=True)

            # Drop any remaining NaNs in the features
            df.dropna(subset=FEATURES, inplace=True)

            if len(df) < WINDOW_SIZE:
                return None

            # Get last WINDOW_SIZE rows
            data_window = df[FEATURES].tail(WINDOW_SIZE).values

            # Scale data
            scaled_window = self.scaler.transform(data_window)
            
            # Return shape (1, 60, 9)
            return scaled_window.reshape(1, WINDOW_SIZE, len(FEATURES))
        except Exception as e:
            print(f"[WARNING] Feature preparation failed: {e}")
            return None

    def predict_next_days(self, history_df: pd.DataFrame, n_days: int = 7) -> Optional[dict]:
        """
        Predict Close prices for the next n_days using a rolling input window.
        Returns:
            {
                "predictions": [{"date": str, "price": float, "confidence": float}],
                "current_price": float,
                "predicted_7d": float,
                "change_pct": float,
                "direction": "UP" | "DOWN" | "FLAT",
                "confidence": float
            }
        """
        if not self.available or self.model is None or self.scaler is None:
            print("[ERROR] LSTM Predictor: Model or scaler not loaded/available.")
            return None

        if len(history_df) < 70:
            print("[ERROR] LSTM Predictor: history_df has less than 70 rows.")
            return None

        try:
            # Prepare starting features
            scaled_seq = self._prepare_features(history_df)
            if scaled_seq is None:
                print("[ERROR] LSTM Predictor: Failed to prepare features from history_df.")
                return None

            current_price = float(history_df["Close"].iloc[-1])
            last_date = history_df.index[-1]
            if not isinstance(last_date, (datetime, pd.Timestamp)):
                last_date = pd.to_datetime(last_date)

            predictions = []
            
            # Make a copy of our sequence buffer to modify during rolling predictions
            seq_buffer = scaled_seq[0].copy() # Shape: (60, 9)

            for i in range(1, n_days + 1):
                try:
                    # Format to Tensor: [batch_size=1, seq_len=60, features=9]
                    x_tensor = torch.tensor(seq_buffer, dtype=torch.float32).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        pred_norm = self.model(x_tensor).cpu().item()
                except RuntimeError as e:
                    # CPU Fallback on CUDA OOM
                    if "out of memory" in str(e).lower() and self.device.type == "cuda":
                        print("[WARNING] CUDA Out of Memory detected. Retrying inference on CPU.")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        self.device = torch.device("cpu")
                        self.model.to(self.device)
                        
                        x_tensor = torch.tensor(seq_buffer, dtype=torch.float32).unsqueeze(0).to(self.device)
                        with torch.no_grad():
                            pred_norm = self.model(x_tensor).cpu().item()
                    else:
                        raise e

                # Inverse transform the predicted Close price
                # Close is index 0 in the feature list.
                dummy_row = np.zeros((1, len(FEATURES)))
                dummy_row[0, 0] = pred_norm
                pred_real = float(self.scaler.inverse_transform(dummy_row)[0, 0])

                # Determine predicted date (skip weekends)
                pred_date = last_date + timedelta(days=1)
                while pred_date.weekday() >= 5: # Saturday/Sunday
                    pred_date += timedelta(days=1)
                last_date = pred_date

                predictions.append({
                    "date": pred_date.strftime("%Y-%m-%d"),
                    "price": round(pred_real, 2),
                    "confidence": 0.0  # Placeholder, filled below
                })

                # Shift buffer: remove index 0, append new step
                next_row = seq_buffer[-1].copy()
                
                # Update Close in normalized state
                next_row[0] = pred_norm
                
                # Shift seq_buffer up by 1 and place next_row at the bottom
                seq_buffer = np.vstack([seq_buffer[1:], next_row])

            # Calculate metrics
            predicted_target = predictions[-1]["price"]
            change_pct = ((predicted_target - current_price) / current_price) * 100

            # Categorize direction
            if change_pct > 1.5:
                direction = "UP"
            elif change_pct < -1.5:
                direction = "DOWN"
            else:
                direction = "FLAT"

            # Confidence is derived based on absolute change pct
            overall_confidence = min(0.95, max(0.50, 0.70 + abs(change_pct) / 100))

            # Populate step-level confidence with a decay factor
            for idx, pred in enumerate(predictions):
                step_conf = max(0.40, overall_confidence - 0.02 * idx)
                pred["confidence"] = round(step_conf, 2)

            return {
                "predictions": predictions,
                "current_price": round(current_price, 2),
                "predicted_7d": round(predicted_target, 2),
                "change_pct": round(change_pct, 2),
                "direction": direction,
                "confidence": round(overall_confidence, 2)
            }

        except Exception as e:
            print(f"[ERROR] Rolling prediction failed: {e}")
            return None


# Module-level lazy singleton
_predictor: Optional[LSTMPredictor] = None


def get_lstm_predictor() -> LSTMPredictor:
    """Get the global LSTMPredictor instance."""
    global _predictor
    if _predictor is None:
        from config.settings import LSTM_MODEL_PATH, SCALER_PATH
        _predictor = LSTMPredictor(
            model_path=str(LSTM_MODEL_PATH),
            scaler_path=str(SCALER_PATH)
        )
    return _predictor
