"""
NiveshAI — Sentiment Analyzer
Wrapper for loading and running the fine-tuned DistilBERT sentiment model.
Includes offline local load and online Hugging Face Inference API fallback.
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

# Constants
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
ID2LABEL = {0: "Negative", 1: "Neutral", 2: "Positive"}
LABEL2EMOJI = {"Negative": "🔴", "Neutral": "⚪", "Positive": "🟢"}


class SentimentAnalyzer:
    """Wrapper for the fine-tuned DistilBERT sentiment model."""

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        self.available = False
        self.tokenizer = None
        self.model = None
        self.device = None
        self.use_api_fallback = False

        # Determine default model path
        if model_path is None:
            # Look in standard models/saved/ folder
            project_root = Path(__file__).parent.parent.parent
            model_path = str(project_root / "models" / "saved" / "sentiment_model.pt")

        self.model_path = model_path

        # Determine device
        import torch
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Attempt loading the local model
        self._load_local_model()

    def _load_local_model(self):
        """Try to load the model state dict locally."""
        if not os.path.exists(self.model_path):
            print(f"[WARNING] Sentiment analyzer: Model file not found at {self.model_path}.")
            print("[INFO] Sentiment analyzer: Will fall back to HF Inference API or neutral predictions.")
            self.use_api_fallback = True
            return

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
            print(f"Loading tokenizer from {MODEL_NAME}...")
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            
            print(f"Loading model architecture and state dict from {self.model_path}...")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME,
                num_labels=3,
                id2label=ID2LABEL,
                label2id={v: k for k, v in ID2LABEL.items()}
            )
            
            # Load state dict
            state_dict = torch.load(self.model_path, map_location=self.device)
            if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            elif hasattr(state_dict, "keys") and "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            
            self.available = True
            print("[OK] Sentiment analyzer: Local model loaded successfully.")
        except Exception as e:
            print(f"[ERROR] Sentiment analyzer: Failed to load local model: {e}")
            self.use_api_fallback = True

    def _hf_api_sentiment(self, text: str) -> Optional[dict]:
        """Use Hugging Face Inference API as a fallback (free, no local model needed)."""
        import requests
        # We use a popular financial sentiment model (FinBERT) as the public API fallback
        API_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
        
        # Check if HF_TOKEN is in env, otherwise send unauthenticated request
        headers = {}
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"

        try:
            response = requests.post(API_URL, json={"inputs": text[:512]}, headers=headers, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data and isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
                    # FinBERT labels: 'positive', 'negative', 'neutral'
                    # Sort by score descending to find the highest prediction
                    predictions = data[0]
                    best_pred = max(predictions, key=lambda x: x.get("score", 0.0))
                    
                    label_raw = best_pred.get("label", "neutral").lower()
                    score = best_pred.get("score", 0.5)

                    # Map FinBERT label names to our standard labels
                    label = "Neutral"
                    if "positive" in label_raw:
                        label = "Positive"
                    elif "negative" in label_raw:
                        label = "Negative"

                    # Construct full scores dict
                    scores_dict = {"Positive": 0.0, "Neutral": 0.0, "Negative": 0.0}
                    for p in predictions:
                        p_label = p.get("label", "").lower()
                        p_score = p.get("score", 0.0)
                        if "positive" in p_label:
                            scores_dict["Positive"] = p_score
                        elif "negative" in p_label:
                            scores_dict["Negative"] = p_score
                        else:
                            scores_dict["Neutral"] = p_score

                    return {
                        "label": label,
                        "score": score,
                        "scores": scores_dict,
                        "emoji": LABEL2EMOJI[label]
                    }
        except Exception:
            pass
        return None

    def analyze(self, text: str) -> dict:
        """
        Analyze sentiment for a single piece of text.
        Returns:
            {
                "label": "Positive" | "Neutral" | "Negative",
                "score": float (confidence, 0-1),
                "scores": {"Positive": float, "Neutral": float, "Negative": float},
                "emoji": "🟢" | "⚪" | "🔴"
            }
        """
        if not text or not text.strip():
            return self._neutral_fallback()

        # 1. Try local model if available
        if self.available and self.model and self.tokenizer:
            try:
                import torch
                
                inputs = self.tokenizer(
                    text,
                    truncation=True,
                    padding="max_length",
                    max_length=MAX_LENGTH,
                    return_tensors="pt"
                )
                
                # Move inputs to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                
                pred_class = int(np.argmax(probs))
                label = ID2LABEL[pred_class]
                score = float(probs[pred_class])
                
                return {
                    "label": label,
                    "score": score,
                    "scores": {
                        "Negative": float(probs[0]),
                        "Neutral": float(probs[1]),
                        "Positive": float(probs[2])
                    },
                    "emoji": LABEL2EMOJI[label]
                }
            except Exception as e:
                print(f"[WARNING] Local model inference failed: {e}. Retrying with API fallback.")

        # 2. Try Hugging Face Inference API fallback
        if self.use_api_fallback:
            api_res = self._hf_api_sentiment(text)
            if api_res:
                return api_res

        # 3. Final Neutral Fallback
        return self._neutral_fallback()

    def analyze_batch(self, texts: List[str], batch_size: int = 16) -> List[dict]:
        """Analyze a list of texts in batches."""
        if not texts:
            return []

        # If local model is not loaded, analyze individually (with cache/API)
        if not self.available or not self.model or not self.tokenizer:
            return [self.analyze(t) for t in texts]

        results = []
        try:
            import torch
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Clean empty texts
                batch_texts = [t if t and t.strip() else "Neutral text" for t in batch_texts]
                
                inputs = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=MAX_LENGTH,
                    return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()
                
                for idx in range(len(batch_texts)):
                    p = probs[idx]
                    pred_class = int(np.argmax(p))
                    label = ID2LABEL[pred_class]
                    results.append({
                        "label": label,
                        "score": float(p[pred_class]),
                        "scores": {
                            "Negative": float(p[0]),
                            "Neutral": float(p[1]),
                            "Positive": float(p[2])
                        },
                        "emoji": LABEL2EMOJI[label]
                    })
        except Exception as e:
            print(f"[WARNING] Batch inference failed: {e}. Falling back to single-text analyzer.")
            return [self.analyze(t) for t in texts]

        return results

    def analyze_news(self, articles: List[dict]) -> List[dict]:
        """
        Takes a list of article dicts and adds/updates the "sentiment" key.
        Returns the updated list.
        """
        if not articles:
            return []

        # Extract text to analyze (headline + summary)
        texts_to_analyze = []
        for a in articles:
            title = a.get("title", "")
            summary = a.get("summary", "")
            # Combine title and summary for better context
            combined = f"{title}. {summary}" if summary else title
            texts_to_analyze.append(combined)

        # Batch analyze
        sentiments = self.analyze_batch(texts_to_analyze)

        # Map back to articles
        for a, s in zip(articles, sentiments):
            a["sentiment"] = s["label"]
            a["sentiment_detail"] = s

        return articles

    def get_aggregate_sentiment(self, articles: List[dict]) -> dict:
        """
        Calculate aggregate sentiment stats for a stock based on its articles list.
        Assumes analyze_news() has been run already on the articles list.
        """
        if not articles:
            return {
                "label": "Neutral",
                "score": 0.5,
                "distribution": {"Positive": 0, "Neutral": 0, "Negative": 0},
                "emoji": "⚪"
            }

        distribution = {"Positive": 0, "Neutral": 0, "Negative": 0}
        total_score = 0.0
        count = 0

        for a in articles:
            detail = a.get("sentiment_detail")
            # If not analyzed, analyze now
            if not detail:
                title = a.get("title", "")
                summary = a.get("summary", "")
                detail = self.analyze(f"{title}. {summary}" if summary else title)
            
            lbl = detail["label"]
            score = detail["score"]
            
            distribution[lbl] += 1
            
            # Map score to linear value: Positive=+score, Negative=-score, Neutral=0
            if lbl == "Positive":
                total_score += score
            elif lbl == "Negative":
                total_score -= score
            count += 1

        if count == 0:
            return {
                "label": "Neutral",
                "score": 0.5,
                "distribution": distribution,
                "emoji": "⚪"
            }

        # Average sentiment score normalized from [-1, 1] to [0, 1]
        avg_sentiment = total_score / count
        normalized_score = (avg_sentiment + 1.0) / 2.0

        if avg_sentiment >= 0.15:
            label = "Positive"
        elif avg_sentiment <= -0.15:
            label = "Negative"
        else:
            label = "Neutral"

        return {
            "label": label,
            "score": round(normalized_score, 4),
            "distribution": distribution,
            "emoji": LABEL2EMOJI[label]
        }

    def _neutral_fallback(self) -> dict:
        """Default neutral sentiment structure."""
        return {
            "label": "Neutral",
            "score": 0.5,
            "scores": {"Positive": 0.25, "Neutral": 0.5, "Negative": 0.25},
            "emoji": "⚪"
        }


# Module-level lazy singleton
_analyzer: Optional[SentimentAnalyzer] = None


def get_sentiment_analyzer() -> SentimentAnalyzer:
    """Get the global SentimentAnalyzer instance."""
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentAnalyzer()
    return _analyzer
