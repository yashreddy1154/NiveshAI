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

def load_financial_phrasebank():
    """Download Financial PhraseBank from HuggingFace Hub."""
    print("  → Downloading Financial PhraseBank dataset from HuggingFace...")
    print("    (This is ~2MB and will be cached after first download)")
    try:
        # sentences_75agree = sentences where ≥75% of annotators agreed
        ds = load_dataset("financial_phrasebank", "sentences_75agree", trust_remote_code=True)
        print(f"  ✅ Loaded {len(ds['train'])} samples")
        return ds["train"]
    except Exception as e:
        print(f"  ⚠️  HuggingFace Hub failed: {e}")
        print("  → Trying alternate source...")
        try:
            ds = load_dataset("takala/financial_phrasebank", "sentences_75agree", trust_remote_code=True)
            print(f"  ✅ Loaded {len(ds['train'])} samples from alternate source")
            return ds["train"]
        except Exception as e2:
            print(f"  ❌ Both sources failed: {e2}")
            print("\n  Manual fix: Download 'financial_phrasebank' manually from:")
            print("  https://huggingface.co/datasets/financial_phrasebank")
            sys.exit(1)

def prepare_data(raw_dataset, tokenizer, max_length):
    """Convert raw dataset to train/val split with tokenization."""
    import numpy as np

    label_map = {"negative": 0, "neutral": 1, "positive": 2}

    # The dataset uses 'sentence' and 'label' (or 'text' and 'label')
    texts  = []
    labels = []

    for item in raw_dataset:
        # Handle different field names
        text = item.get("sentence") or item.get("text", "")
        lbl_raw = item.get("label", 0)

        # Label might be int already or string
        if isinstance(lbl_raw, str):
            lbl = label_map.get(lbl_raw.lower(), 1)
        else:
            lbl = int(lbl_raw)  # 0=negative, 1=neutral, 2=positive

        texts.append(text)
        labels.append(lbl)

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
