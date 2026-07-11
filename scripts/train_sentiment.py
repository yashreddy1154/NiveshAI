"""
NiveshAI — Sentiment Model Training
Fine-tunes DistilBERT on Financial PhraseBank for financial news sentiment.

Can be run standalone:
    python train_sentiment.py --mode full --output ./output

Or called by train_all.py automatically.
"""

import argparse
import os
import sys
import time
import json
from datetime import datetime
from torch.utils.data import Dataset

# ─────────────────────────────────────────────────────────────────────────────
# Lazy imports (after packages are verified by train_all.py)
# ─────────────────────────────────────────────────────────────────────────────
def load_imports():
    global torch, nn, Dataset, DataLoader, AutoTokenizer, AutoModelForSequenceClassification
    global Trainer, TrainingArguments, load_dataset, tqdm, np, classification_report, accuracy_score

    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              Trainer, TrainingArguments)
    from datasets import load_dataset
    from tqdm import tqdm
    import numpy as np
    from sklearn.metrics import classification_report, accuracy_score

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
MODEL_NAME = "distilbert-base-uncased"

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL  = {0: "negative", 1: "neutral", 2: "positive"}

MODE_CONFIG = {
    "quick":     {"epochs": 3, "max_length": 128},
    "full":      {"epochs": 4, "max_length": 256},
    "overnight": {"epochs": 6, "max_length": 512},
}

# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class FinancialDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(
            texts, truncation=True, padding="max_length",
            max_length=max_length, return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }

def _parse_fpb_text(text):
    """Parse financial phrasebank format: 'sentence text @ label'"""
    label_map = {"negative": 0, "neutral": 1, "positive": 2}
    texts, labels = [], []
    for line in text.strip().split("\n"):
        line = line.strip()
        if "@" not in line:
            continue
        parts = line.rsplit("@", 1)
        if len(parts) != 2:
            continue
        sentence = parts[0].strip().strip('"')
        lbl_str  = parts[1].strip().lower()
        if lbl_str in label_map and sentence:
            texts.append(sentence)
            labels.append(label_map[lbl_str])
    return texts, labels


def load_financial_phrasebank():
    """
    Load Financial PhraseBank from multiple fallback sources.
    HuggingFace deprecated loading scripts so we try 5 different sources.
    Returns: (texts: list[str], labels: list[int])
    """
    print("  → Loading Financial PhraseBank dataset...")

    # ── Source 1: nickmuchi/financial-classification (parquet copy of FPB) ─────
    try:
        ds = load_dataset("nickmuchi/financial-classification", split="train")
        texts  = [item["sentence"] for item in ds]
        labels = [int(item["label"]) for item in ds]   # 0=neg, 1=neu, 2=pos
        if len(texts) > 500:
            print(f"  ✅ Loaded {len(texts)} samples (nickmuchi/financial-classification)")
            return texts, labels
    except Exception as e:
        print(f"  ⚠️  Source 1 failed: {e}")

    # ── Source 2: Try huggingface_hub to download raw file directly ────────────
    try:
        from huggingface_hub import hf_hub_download
        for fname in ["sentences_75agree.txt", "Sentences_75Agree.txt",
                      "data/sentences_75agree.txt", "data/Sentences_75Agree.txt"]:
            try:
                path = hf_hub_download(
                    repo_id="financial_phrasebank",
                    filename=fname,
                    repo_type="dataset",
                )
                with open(path, encoding="utf-8", errors="ignore") as f:
                    raw = f.read()
                texts, labels = _parse_fpb_text(raw)
                if len(texts) > 500:
                    print(f"  ✅ Loaded {len(texts)} samples (HF Hub raw file)")
                    return texts, labels
            except Exception:
                continue
    except Exception as e:
        print(f"  ⚠️  Source 2 failed: {e}")

    # ── Source 3: GitHub mirrors of the dataset ────────────────────────────────
    try:
        import requests
        urls = [
            "https://raw.githubusercontent.com/pthirikovela/financial-sentiment/main/Sentences_75Agree.txt",
            "https://raw.githubusercontent.com/kdukuray/financial-phrase-bank/main/phrases_75_agree.txt",
            "https://raw.githubusercontent.com/Sathyanarayana-NITK/SentimentAnalysis/master/FinancialPhraseBank/Sentences_75Agree.txt",
        ]
        for url in urls:
            try:
                r = requests.get(url, timeout=20)
                if r.ok and len(r.text) > 1000:
                    texts, labels = _parse_fpb_text(r.text)
                    if len(texts) > 500:
                        print(f"  ✅ Loaded {len(texts)} samples (GitHub mirror)")
                        return texts, labels
            except Exception:
                continue
    except Exception as e:
        print(f"  ⚠️  Source 3 failed: {e}")

    # ── Source 4: Twitter financial news sentiment (2-class, different domain) ─
    try:
        ds = load_dataset("zeroshot/twitter-financial-news-sentiment", split="train")
        texts, labels = [], []
        for item in ds:
            t = item.get("text", item.get("sentence", ""))
            l = int(item.get("label", 1))
            # bearish(0)→negative(0), neutral(1)→neutral(1), bullish(2)→positive(2)
            texts.append(t)
            labels.append(l)
        if len(texts) > 200:
            print(f"  ✅ Loaded {len(texts)} samples (Twitter financial sentiment — fallback)")
            return texts, labels
    except Exception as e:
        print(f"  ⚠️  Source 4 failed: {e}")

    # ── Source 5: financial_news dataset ──────────────────────────────────────
    try:
        ds = load_dataset("oliverguhr/german-sentiment-twitter", split="train")
        # This is wrong domain but tests the pipeline — last resort
        print("  ⚠️  Using fallback dataset (not ideal) — try to fix internet access")
        texts  = [item.get("text", "") for item in ds][:3000]
        labels = [min(2, int(item.get("label", 1))) for item in ds][:3000]
        if len(texts) > 200:
            return texts, labels
    except Exception:
        pass

    print("\n  ❌ ALL DATASET SOURCES FAILED")
    print("  Solutions:")
    print("  1. Check internet connection")
    print("  2. pip install huggingface_hub requests")
    print("  3. Download manually from https://huggingface.co/datasets/nickmuchi/financial-classification")
    sys.exit(1)

def prepare_data(texts_and_labels, tokenizer, max_length):
    """
    Convert (texts, labels) tuple to train/val split with tokenization.
    texts_and_labels: tuple(list[str], list[int]) from load_financial_phrasebank()
    """
    import numpy as np

    texts, labels = texts_and_labels

    # Validate labels are in [0, 2]
    labels = [max(0, min(2, int(l))) for l in labels]

    # Filter empty texts
    valid = [(t, l) for t, l in zip(texts, labels) if t and len(t.strip()) > 5]
    texts  = [v[0] for v in valid]
    labels = [v[1] for v in valid]

    print(f"  ✅ {len(texts)} valid samples | Labels: neg={labels.count(0)}, neu={labels.count(1)}, pos={labels.count(2)}")

    # Train/Val split (90/10)
    n = len(texts)
    idx = np.random.permutation(n)
    split = int(n * 0.9)

    train_texts  = [texts[i] for i in idx[:split]]
    train_labels = [labels[i] for i in idx[:split]]
    val_texts    = [texts[i] for i in idx[split:]]
    val_labels   = [labels[i] for i in idx[split:]]

    print(f"  ✅ Train: {len(train_texts)} | Val: {len(val_texts)} samples")

    train_ds = FinancialDataset(train_texts, train_labels, tokenizer, max_length)
    val_ds   = FinancialDataset(val_texts,   val_labels,   tokenizer, max_length)
    return train_ds, val_ds, val_texts, val_labels

# ─────────────────────────────────────────────────────────────────────────────
# Metrics callback
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def train(args):
    load_imports()

    cfg = MODE_CONFIG[args.mode]
    epochs     = cfg["epochs"]
    max_length = cfg["max_length"]
    batch_size = args.batch_size

    print(f"\n  Config:")
    print(f"    Model      : {MODEL_NAME}")
    print(f"    Epochs     : {epochs}")
    print(f"    Batch size : {batch_size}")
    print(f"    Max length : {max_length} tokens")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Device     : {device}")
    if device.type == "cuda":
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"    GPU VRAM   : {vram:.1f} GB")

    os.makedirs(args.output, exist_ok=True)
    checkpoint_dir = os.path.join(args.output, "sentiment_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    output_model_path = os.path.join(args.output, "sentiment_model.pt")
    output_dir_hf     = os.path.join(args.output, "sentiment_hf")  # HF Trainer output

    # ── Load tokenizer & model ────────────────────────────────────────────────
    print(f"\n  → Loading {MODEL_NAME} tokenizer and model...")
    print("    (First run will download ~260MB — cached after that)")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model = model.to(device)
    print(f"  ✅ Model loaded ({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters)")

    # ── Check for existing checkpoint ─────────────────────────────────────────
    existing_checkpoints = sorted([
        d for d in os.listdir(checkpoint_dir)
        if os.path.isdir(os.path.join(checkpoint_dir, d))
    ])
    if existing_checkpoints:
        latest = os.path.join(checkpoint_dir, existing_checkpoints[-1])
        print(f"\n  ⚡ Found existing checkpoint: {existing_checkpoints[-1]}")
        print(f"     Will resume training from this checkpoint.")
        resume_from = latest
    else:
        resume_from = None

    # ── Load dataset ──────────────────────────────────────────────────────────
    print("\n  [Dataset]")
    raw_ds = load_financial_phrasebank()
    train_ds, val_ds, val_texts, val_labels = prepare_data(raw_ds, tokenizer, max_length)

    # ── Training arguments ────────────────────────────────────────────────────
    # fp16 (mixed precision) halves VRAM usage on GPU
    use_fp16 = (device.type == "cuda")

    training_args = TrainingArguments(
        output_dir                  = checkpoint_dir,
        num_train_epochs            = epochs,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size  = batch_size * 2,
        warmup_ratio                = 0.1,
        weight_decay                = 0.01,
        learning_rate               = 2e-5,
        logging_dir                 = os.path.join(args.output, "sentiment_logs"),
        logging_steps               = 20,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",          # checkpoint every epoch
        load_best_model_at_end      = True,
        metric_for_best_model       = "accuracy",
        greater_is_better           = True,
        fp16                        = use_fp16,
        dataloader_num_workers      = 0,                # 0 = safer on Windows
        report_to                   = "none",           # don't use wandb/tensorboard
        disable_tqdm                = False,
    )

    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_ds,
        eval_dataset    = val_ds,
        compute_metrics = compute_metrics,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\n  [Training] Starting {'(resuming from checkpoint)' if resume_from else '(fresh start)'}...")
    start = time.time()
    trainer.train(resume_from_checkpoint=resume_from)
    elapsed = time.time() - start
    print(f"\n  ✅ Training finished in {int(elapsed//60)}m {int(elapsed%60)}s")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\n  [Evaluation] Running validation...")
    eval_results = trainer.evaluate()
    accuracy = eval_results.get("eval_accuracy", 0)
    print(f"\n  📊 Validation Accuracy: {accuracy*100:.2f}%")

    if accuracy < 0.70:
        print(f"  ⚠️  Accuracy below 70% — model may not be trained enough.")
        print(f"     Try --mode full or --mode overnight for better results.")
    else:
        print(f"  ✅ Accuracy acceptable (≥70%)")

    # ── Detailed classification report ────────────────────────────────────────
    print("\n  [Classification Report]")
    preds = trainer.predict(val_ds)
    pred_labels = preds.predictions.argmax(axis=-1)
    report = classification_report(
        val_labels, pred_labels,
        target_names=["Negative", "Neutral", "Positive"]
    )
    print(report)

    # ── Save model ────────────────────────────────────────────────────────────
    print(f"\n  [Saving] Saving model to {output_model_path}...")

    # Save as HuggingFace format (for easy loading later)
    trainer.save_model(output_dir_hf)
    tokenizer.save_pretrained(output_dir_hf)

    # Also save as a single .pt file for compatibility
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "model_name": MODEL_NAME,
            "num_labels": 3,
            "id2label": ID2LABEL,
            "label2id": LABEL2ID,
            "max_length": max_length,
        },
        "accuracy": accuracy,
        "trained_at": datetime.now().isoformat(),
        "mode": args.mode,
    }, output_model_path)

    size_mb = os.path.getsize(output_model_path) / 1024 / 1024
    print(f"  ✅ Saved: {output_model_path} ({size_mb:.1f} MB)")
    print(f"  ✅ Saved HF format: {output_dir_hf}/")

    # ── Save validation results ───────────────────────────────────────────────
    results_path = os.path.join(args.output, "validation_results.txt")
    with open(results_path, "a") as f:
        f.write(f"\n=== Sentiment Model ({args.mode}) ===\n")
        f.write(f"Trained at: {datetime.now().isoformat()}\n")
        f.write(f"Accuracy: {accuracy*100:.2f}%\n")
        f.write(f"Training time: {int(elapsed//60)}m {int(elapsed%60)}s\n")
        f.write(report)
        f.write("\n")

    return accuracy >= 0.70


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NiveshAI Sentiment Model")
    parser.add_argument("--mode",       choices=["quick", "full", "overnight"], default="full")
    parser.add_argument("--output",     default="output")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Reduce to 8 if you get CUDA out-of-memory errors")
    args = parser.parse_args()

    print(f"\n{'═'*60}")
    print(f"  NiveshAI Sentiment Model Training")
    print(f"  Mode: {args.mode} | Batch size: {args.batch_size}")
    print(f"{'═'*60}")

    success = train(args)
    sys.exit(0 if success else 1)
