"""
NiveshAI — Feature Engineering
Computes technical indicators from raw OHLCV data for ML models.
"""
# TODO: Implement in Phase 2

import pandas as pd


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicator columns to a DataFrame with OHLCV data.

    Adds: SMA_20, SMA_50, RSI_14, MACD, MACD_Signal, Bollinger_%B
    """
    raise NotImplementedError("Will be implemented in Phase 2")


def prepare_lstm_features(df: pd.DataFrame, window: int = 60):
    """Prepare sliding window features for LSTM input.

    Returns: X (samples, window, features), y (samples,), scaler
    """
    raise NotImplementedError("Will be implemented in Phase 2")


def prepare_rf_features(df: pd.DataFrame):
    """Prepare feature matrix for Random Forest.

    Returns: X (samples, features), y (samples,) where y is direction label
    """
    raise NotImplementedError("Will be implemented in Phase 2")
