import unittest
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from models.prediction.lstm_predictor import get_lstm_predictor, LSTMPredictor

class TestLSTMPredictor(unittest.TestCase):
    def setUp(self):
        # Retrieve the lazy singleton
        self.predictor = get_lstm_predictor()
        
    def test_model_loading(self):
        self.assertTrue(self.predictor.available, "LSTMPredictor should be available when assets exist")
        self.assertIsNotNone(self.predictor.model, "Model should be loaded")
        self.assertIsNotNone(self.predictor.scaler, "Scaler should be loaded")

    def test_insufficient_data(self):
        # Create a df with less than 70 rows (e.g. 50 rows)
        df_short = pd.DataFrame({
            "Close": np.random.uniform(1000, 1500, 50),
            "Volume": np.random.randint(100, 1000, 50)
        })
        res = self.predictor.predict_next_days(df_short, n_days=7)
        self.assertIsNone(res, "Predict should return None for short history")

    def test_feature_preparation_shape(self):
        # Generate 100 rows of mock stock data
        dates = [datetime.today() - timedelta(days=i) for i in range(100)]
        dates.reverse()
        df = pd.DataFrame({
            "Open": np.random.uniform(1200, 1300, 100),
            "High": np.random.uniform(1200, 1300, 100),
            "Low": np.random.uniform(1200, 1300, 100),
            "Close": np.random.uniform(1200, 1300, 100),
            "Volume": np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        scaled_features = self.predictor._prepare_features(df)
        self.assertIsNotNone(scaled_features, "Prepared features should not be None")
        self.assertEqual(scaled_features.shape, (1, 60, 9), "Prepared features should have shape (1, 60, 9)")

    def test_prediction_output_structure(self):
        # Generate 100 rows of mock stock data
        dates = [datetime.today() - timedelta(days=i) for i in range(100)]
        dates.reverse()
        df = pd.DataFrame({
            "Open": np.random.uniform(1200, 1300, 100),
            "High": np.random.uniform(1200, 1300, 100),
            "Low": np.random.uniform(1200, 1300, 100),
            "Close": np.random.uniform(1200, 1300, 100),
            "Volume": np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        res = self.predictor.predict_next_days(df, n_days=7)
        self.assertIsNotNone(res, "Prediction result should not be None")
        self.assertIn("predictions", res)
        self.assertIn("current_price", res)
        self.assertIn("predicted_7d", res)
        self.assertIn("change_pct", res)
        self.assertIn("direction", res)
        self.assertIn("confidence", res)
        
        # Verify predictions details
        predictions = res["predictions"]
        self.assertEqual(len(predictions), 7, "Should predict exactly 7 days")
        for pred in predictions:
            self.assertIn("date", pred)
            self.assertIn("price", pred)
            self.assertIn("confidence", pred)
            self.assertTrue(isinstance(pred["date"], str))
            self.assertTrue(isinstance(pred["price"], float))
            self.assertTrue(isinstance(pred["confidence"], float))

if __name__ == "__main__":
    unittest.main()
