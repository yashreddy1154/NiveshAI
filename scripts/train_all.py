"""
NiveshAI — Master Training Script
Runs the full training pipeline: GPU test → Sentiment → LSTM → Random Forest

Usage:
    python train_all.py --mode quick      (~45 min,  NIFTY  50, good for demo)
    python train_all.py --mode full       (~2-3 hrs, NIFTY 200, recommended)
    python train_all.py --mode overnight  (~6-8 hrs, NIFTY 500, best quality)

Options:
    --skip-sentiment   Skip DistilBERT training (if already done)
    --skip-lstm        Skip LSTM training (if already done)
    --skip-rf          Skip Random Forest training (if already done)
    --batch-size INT   Override default batch size (default: 16, use 8 if OOM)
    --output DIR       Output directory (default: ./output)
    --log FILE         Save full log to file (default: ./output/training_log.txt)

Examples:
    python train_all.py --mode full
    python train_all.py --mode full --batch-size 8   # if out of memory
    python train_all.py --mode overnight --skip-sentiment  # resume after interruption
"""

import argparse
import os
import sys
import subprocess
import time
import platform
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# ANSI Colors (works on Windows 10+ terminal)
# ─────────────────────────────────────────────────────────────────────────────
class C:
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    BLUE    = "\033[94m"
    CYAN    = "\033[96m"
    BOLD    = "\033[1m"
    RESET   = "\033[0m"

def banner():
    print(f"""
{C.CYAN}{C.BOLD}
╔══════════════════════════════════════════════════════════════╗
║          NiveshAI — AI Model Training Pipeline               ║
║          Indian Stock Market Investment Research             ║
╚══════════════════════════════════════════════════════════════╝
{C.RESET}""")

def section(title, step, total):
    print(f"\n{C.BLUE}{C.BOLD}[{step}/{total}] {title}{C.RESET}")
    print("─" * 60)

def ok(msg):   print(f"  {C.GREEN}✅ {msg}{C.RESET}")
def warn(msg): print(f"  {C.YELLOW}⚠️  {msg}{C.RESET}")
def err(msg):  print(f"  {C.RED}❌ {msg}{C.RESET}")
def info(msg): print(f"  {C.CYAN}→  {msg}{C.RESET}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Auto-Install Requirements
# ─────────────────────────────────────────────────────────────────────────────
PACKAGES = [
    # (import_name, pip_name)
    ("tqdm",          "tqdm"),
    ("numpy",         "numpy"),
    ("pandas",        "pandas"),
    ("sklearn",       "scikit-learn"),
    ("ta",            "ta"),
    ("yfinance",      "yfinance"),
    ("torch",         None),         # handled separately (needs CUDA URL)
    ("transformers",  "transformers"),
    ("datasets",      "datasets"),
    ("accelerate",    "accelerate"),
]

def check_torch():
    """Check if PyTorch with CUDA support is installed."""
    try:
        import torch
        cuda = torch.cuda.is_available()
        if cuda:
            ok(f"PyTorch {torch.__version__} with CUDA — GPU ready!")
        else:
            warn(f"PyTorch {torch.__version__} installed but NO CUDA. Will train on CPU (slower).")
        return True
    except ImportError:
        return False

def install_packages():
    section("Installing / Verifying Requirements", 1, 6)

    # 1a. Check / install non-torch packages
    for import_name, pip_name in PACKAGES:
        if pip_name is None:
            continue
        try:
            __import__(import_name)
            ok(f"{pip_name} already installed")
        except ImportError:
            info(f"Installing {pip_name}...")
            ret = subprocess.call(
                [sys.executable, "-m", "pip", "install", "--quiet", pip_name]
            )
            if ret == 0:
                ok(f"{pip_name} installed successfully")
            else:
                err(f"Failed to install {pip_name}. Try manually: pip install {pip_name}")
                sys.exit(1)

    # 1b. Special handling for PyTorch
    if not check_torch():
        info("PyTorch not found. Detecting CUDA version...")
        cuda_ver = detect_cuda_version()
        if cuda_ver:
            info(f"Detected CUDA {cuda_ver}. Installing PyTorch with GPU support...")
            whl_url = get_torch_whl_url(cuda_ver)
            ret = subprocess.call(
                [sys.executable, "-m", "pip", "install", "--quiet",
                 "torch", "torchvision", "--index-url", whl_url]
            )
        else:
            warn("CUDA not found. Installing CPU-only PyTorch (training will be slow)...")
            ret = subprocess.call(
                [sys.executable, "-m", "pip", "install", "--quiet", "torch", "torchvision"]
            )
        if ret != 0:
            err("PyTorch installation failed!")
            print(f"\n  {C.YELLOW}Manual fix — run one of these:{C.RESET}")
            print("  For CUDA 12.1: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
            print("  For CUDA 11.8: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
            print("  For CPU only:  pip install torch torchvision")
            sys.exit(1)
        check_torch()

def detect_cuda_version():
    """Detect CUDA version from nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            # Also grab CUDA version
            result2 = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, timeout=5
            )
            output = result2.stdout
            if "CUDA Version: 12" in output:
                return "12.1"
            elif "CUDA Version: 11.8" in output or "CUDA Version: 11" in output:
                return "11.8"
    except Exception:
        pass
    return None

def get_torch_whl_url(cuda_ver):
    mapping = {
        "12.1": "https://download.pytorch.org/whl/cu121",
        "12.4": "https://download.pytorch.org/whl/cu124",
        "11.8": "https://download.pytorch.org/whl/cu118",
    }
    return mapping.get(cuda_ver, "https://download.pytorch.org/whl/cu121")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: GPU Smoke Test (Dummy Model)
# ─────────────────────────────────────────────────────────────────────────────
def gpu_smoke_test():
    section("GPU Smoke Test", 2, 6)
    import torch
    import torch.nn as nn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
        ok(f"GPU Detected: {gpu_name}")
        ok(f"VRAM Available: {vram_gb:.1f} GB")
        if vram_gb < 4:
            warn("Less than 4GB VRAM detected. Use --batch-size 8 if you get OOM errors.")
    else:
        warn("No GPU detected — training will run on CPU (much slower)")
        info("If you have an NVIDIA GPU, check CUDA installation in TROUBLESHOOTING_GUIDE.md")

    info("Running dummy training test on GPU...")
    try:
        # Create a tiny model and run 5 training steps
        class TinyNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(64, 1)
            def forward(self, x):
                return self.fc(x)

        model = TinyNet().to(device)
        optimizer = torch.optim.Adam(model.parameters())
        loss_fn = nn.MSELoss()

        for _ in range(5):
            x = torch.randn(32, 64).to(device)
            y = torch.randn(32, 1).to(device)
            loss = loss_fn(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        ok(f"Dummy model trained successfully on {device.type.upper()}")

        # Clean up GPU memory
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        return device

    except RuntimeError as e:
        if "out of memory" in str(e):
            err("GPU ran out of memory even on tiny test — something is wrong")
            err("Try closing other GPU apps (games, etc.) and rerun")
        else:
            err(f"GPU test failed: {e}")
        sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3–5: Run individual training scripts
# ─────────────────────────────────────────────────────────────────────────────
def run_script(script_path, args_list, step, total, name):
    section(f"Training {name}", step, total)
    cmd = [sys.executable, script_path] + args_list
    info(f"Running: {' '.join(cmd)}")
    print()

    start = time.time()
    ret = subprocess.call(cmd)
    elapsed = time.time() - start

    mins = int(elapsed // 60)
    secs = int(elapsed % 60)

    if ret == 0:
        ok(f"{name} completed in {mins}m {secs}s")
        return True
    else:
        err(f"{name} FAILED after {mins}m {secs}s (exit code {ret})")
        print(f"\n  {C.YELLOW}Check TROUBLESHOOTING_GUIDE.md for help fixing this error.{C.RESET}")
        return False

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: Final Validation
# ─────────────────────────────────────────────────────────────────────────────
def validate_outputs(output_dir):
    section("Validating Output Models", 6, 6)
    import torch, pickle

    expected = {
        "sentiment_model.pt": (200, 320),   # MB range
        "lstm_model.pt":      (1, 50),
        "rf_model.pkl":       (0.5, 20),
        "scaler.pkl":         (0.001, 5),
    }

    all_ok = True
    for filename, (min_mb, max_mb) in expected.items():
        path = os.path.join(output_dir, filename)
        if not os.path.exists(path):
            err(f"MISSING: {filename}")
            all_ok = False
            continue

        size_mb = os.path.getsize(path) / 1024 / 1024
        if size_mb < min_mb:
            err(f"{filename}: {size_mb:.1f}MB — TOO SMALL (likely corrupt, min {min_mb}MB)")
            all_ok = False
        else:
            ok(f"{filename}: {size_mb:.1f} MB ✓")

    # Try loading each model
    info("Verifying models load without errors...")
    try:
        torch.load(os.path.join(output_dir, "lstm_model.pt"), map_location="cpu", weights_only=False)
        ok("LSTM model loads correctly")
    except Exception as e:
        err(f"LSTM model corrupted: {e}")
        all_ok = False

    try:
        with open(os.path.join(output_dir, "rf_model.pkl"), "rb") as f:
            pickle.load(f)
        ok("Random Forest model loads correctly")
    except Exception as e:
        err(f"RF model corrupted: {e}")
        all_ok = False

    # Check validation_results.txt
    results_path = os.path.join(output_dir, "validation_results.txt")
    if os.path.exists(results_path):
        print(f"\n  {C.CYAN}Model Performance Metrics:{C.RESET}")
        with open(results_path) as f:
            for line in f:
                print(f"    {line.rstrip()}")

    return all_ok

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # Enable ANSI colors on Windows
    if platform.system() == "Windows":
        os.system("color")

    parser = argparse.ArgumentParser(
        description="NiveshAI Model Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--mode", choices=["quick", "full", "overnight"], default="full",
                        help="Training intensity (default: full)")
    parser.add_argument("--skip-sentiment", action="store_true", help="Skip sentiment model training")
    parser.add_argument("--skip-lstm",      action="store_true", help="Skip LSTM training")
    parser.add_argument("--skip-rf",        action="store_true", help="Skip Random Forest training")
    parser.add_argument("--batch-size",     type=int, default=16,
                        help="Batch size for DistilBERT (use 8 if out-of-memory)")
    parser.add_argument("--output", default="output", help="Output directory for model files")
    args = parser.parse_args()

    banner()
    print(f"  Mode    : {C.BOLD}{args.mode.upper()}{C.RESET}")
    print(f"  Output  : {C.BOLD}{args.output}/{C.RESET}")
    print(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    mode_info = {
        "quick":     ("NIFTY  50",  "~45 minutes"),
        "full":      ("NIFTY 200",  "~2-3 hours"),
        "overnight": ("NIFTY 500",  "~6-8 hours"),
    }
    stocks, eta = mode_info[args.mode]
    print(f"  Stocks  : {C.BOLD}{stocks}{C.RESET}")
    print(f"  ETA     : {C.BOLD}{eta}{C.RESET}")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    total_start = time.time()
    results = {}

    # ── Step 1: Install packages ──────────────────────────────────────────────
    install_packages()

    # ── Step 2: GPU smoke test ────────────────────────────────────────────────
    device = gpu_smoke_test()

    # ── Step 3: Sentiment model ───────────────────────────────────────────────
    if not args.skip_sentiment:
        script = os.path.join(script_dir, "train_sentiment.py")
        sentiment_args = [
            "--mode",       args.mode,
            "--output",     args.output,
            "--batch-size", str(args.batch_size),
        ]
        results["sentiment"] = run_script(script, sentiment_args, 3, 6, "Sentiment Model (DistilBERT)")
    else:
        warn("Skipping sentiment model training (--skip-sentiment)")

    # ── Step 4: LSTM model ────────────────────────────────────────────────────
    if not args.skip_lstm:
        script = os.path.join(script_dir, "train_prediction.py")
        lstm_args = [
            "--mode",    args.mode,
            "--output",  args.output,
            "--model",   "lstm",
        ]
        results["lstm"] = run_script(script, lstm_args, 4, 6, "LSTM Price Predictor")
    else:
        warn("Skipping LSTM training (--skip-lstm)")

    # ── Step 5: Random Forest ─────────────────────────────────────────────────
    if not args.skip_rf:
        script = os.path.join(script_dir, "train_prediction.py")
        rf_args = [
            "--mode",    args.mode,
            "--output",  args.output,
            "--model",   "rf",
        ]
        results["rf"] = run_script(script, rf_args, 5, 6, "Random Forest Predictor")
    else:
        warn("Skipping Random Forest training (--skip-rf)")

    # ── Step 6: Validate ──────────────────────────────────────────────────────
    all_valid = validate_outputs(args.output)

    # ── Final Summary ─────────────────────────────────────────────────────────
    total_elapsed = time.time() - total_start
    total_mins = int(total_elapsed // 60)

    print(f"\n{'═'*60}")
    if all_valid:
        print(f"{C.GREEN}{C.BOLD}✅ ALL DONE! Training complete in {total_mins} minutes.{C.RESET}")
        print(f"\n  📦 Send these files back to the project owner:")
        for f in ["sentiment_model.pt", "lstm_model.pt", "rf_model.pkl", "scaler.pkl"]:
            path = os.path.join(args.output, f)
            if os.path.exists(path):
                size = os.path.getsize(path) / 1024 / 1024
                print(f"     {args.output}/{f}  ({size:.1f} MB)")
        print(f"\n  💡 Tip: Zip the entire '{args.output}/' folder and share via Google Drive.")
    else:
        print(f"{C.RED}{C.BOLD}❌ TRAINING FINISHED WITH ERRORS. Some models may be missing or corrupt.{C.RESET}")
        print(f"\n  📖 See TROUBLESHOOTING_GUIDE.md for help.")
        print(f"  🔁 You can re-run specific steps using --skip flags.")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
