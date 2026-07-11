"""
NiveshAI — Hugging Face Model Downloader
Downloads model weights from the Hugging Face Hub at runtime if they are missing locally.
"""

import os
import sys
import shutil
from pathlib import Path

# Adjust sys.path to run directly from the scripts directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import MODELS_DIR

# Default public repository containing pre-trained weights for NiveshAI
DEFAULT_REPO = "yashvardhan/niveshai-models"

def ensure_models_downloaded(repo_id: str = DEFAULT_REPO):
    """Checks for existence of local model weights and downloads them from HF if missing."""
    models_to_download = [
        "sentiment_model.pt",
        "lstm_model.pt",
        "rf_model.pkl",
        "scaler.pkl",
        "rf_scaler.pkl"
    ]
    
    # Check if we should override the repository via environment
    repo = os.getenv("HF_MODEL_REPO", repo_id)
    
    # Ensure folder exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    missing_files = [fname for fname in models_to_download if not (MODELS_DIR / fname).exists()]
    
    if not missing_files:
        print("[INFO] All model files exist locally in models/saved/.")
        return
        
    print(f"[INFO] Missing {len(missing_files)} model files. Attempting download from HF Hub repository '{repo}'...")
    
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[WARNING] huggingface_hub not installed. Cannot download models from HF Hub.")
        return

    for fname in missing_files:
        target_path = MODELS_DIR / fname
        try:
            print(f"Downloading {fname} from Hugging Face...")
            downloaded_path = hf_hub_download(
                repo_id=repo,
                filename=fname,
                repo_type="model"
            )
            # Copy to target models/saved/ folder
            shutil.copy(downloaded_path, target_path)
            print(f"  [OK] Saved {fname} to {target_path}")
        except Exception as e:
            print(f"  [ERROR] Failed to download {fname} from Hugging Face Hub: {e}")

if __name__ == "__main__":
    ensure_models_downloaded()
