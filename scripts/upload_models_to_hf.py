"""
NiveshAI — Hugging Face Model Hub Uploader
Uploads local model files to the Hugging Face Hub.
This allows the deployed web application (e.g., Streamlit Cloud or HF Spaces)
to automatically download model weights at startup.
"""

import os
import sys
from pathlib import Path

# Adjust sys.path to run directly from the scripts directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import MODELS_DIR

def run_uploader():
    print("=" * 60)
    print("         NiveshAI Hugging Face Model Uploader")
    print("=" * 60)

    try:
        from huggingface_hub import HfApi, login
    except ImportError:
        print("[ERROR] huggingface_hub package not installed. Run: pip install huggingface_hub")
        return 1

    # Fetch token
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("\n[INFO] HF_TOKEN environment variable not set.")
        hf_token = input("Enter your Hugging Face Write Token: ").strip()
        if not hf_token:
            print("[ERROR] Token is required to upload files.")
            return 1

    # Authenticate
    try:
        login(token=hf_token)
        print("[OK] Successfully authenticated with Hugging Face Hub.")
    except Exception as e:
        print(f"[ERROR] Authentication failed: {e}")
        return 1

    # Get repository name
    repo_name = os.getenv("HF_MODEL_REPO")
    if not repo_name:
        repo_name = input("Enter target HF repository name (e.g. 'username/niveshai-models'): ").strip()
        if not repo_name:
            print("[ERROR] Repository name is required.")
            return 1

    # Files to upload
    model_files = ["sentiment_model.pt", "lstm_model.pt", "rf_model.pkl", "scaler.pkl", "rf_scaler.pkl"]
    
    api = HfApi()
    
    # Create repo if not exists
    try:
        print(f"\nCreating/verifying model repository: '{repo_name}'...")
        api.create_repo(repo_id=repo_name, repo_type="model", exist_ok=True)
        print(f"[OK] Repository ready: https://huggingface.co/{repo_name}")
    except Exception as e:
        print(f"[ERROR] Failed to create repository: {e}")
        return 1

    uploaded_count = 0
    for fname in model_files:
        path = MODELS_DIR / fname
        if not path.exists():
            print(f"[WARNING] Model file not found: {path} (skipping)")
            continue

        print(f"Uploading {fname} ({path.stat().st_size / (1024*1024):.2f} MB)...")
        try:
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=fname,
                repo_id=repo_name,
                repo_type="model",
            )
            print(f"  [OK] Uploaded {fname} successfully.")
            uploaded_count += 1
        except Exception as e:
            print(f"  [ERROR] Failed to upload {fname}: {e}")

    print("\n" + "=" * 60)
    print(f"Uploader Summary: {uploaded_count} / {len(model_files)} files uploaded successfully.")
    print("=" * 60)
    return 0 if uploaded_count > 0 else 1

if __name__ == "__main__":
    sys.exit(run_uploader())
