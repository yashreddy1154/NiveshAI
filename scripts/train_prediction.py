"""
NiveshAI — Price Prediction Model Training
Trains both LSTM (price regression) and Random Forest (direction classification).

Standalone usage:
    python train_prediction.py --mode full --model lstm --output ./output
    python train_prediction.py --mode full --model rf   --output ./output
    python train_prediction.py --mode full --model both --output ./output  (default)

Or called by train_all.py automatically.
"""

import argparse
import os
import sys
import time
import pickle
import json
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Stock universe per mode
# ─────────────────────────────────────────────────────────────────────────────
NIFTY_50 = [
    "RELIANCE","TCS","HDFCBANK","INFY","ICICIBANK","BHARTIARTL","SBIN","ITC",
    "KOTAKBANK","LT","HINDUNILVR","BAJFINANCE","HCLTECH","MARUTI","SUNPHARMA",
    "TATAMOTORS","ONGC","NTPC","POWERGRID","AXISBANK","WIPRO","ADANIENT",
    "TATASTEEL","ASIANPAINT","BAJAJFINSV","COALINDIA","NESTLEIND","TITAN",
    "ULTRACEMCO","TECHM","JSWSTEEL","INDUSINDBK","HINDALCO","DRREDDY","CIPLA",
    "BPCL","EICHERMOT","GRASIM","APOLLOHOSP","DIVISLAB","TATACONSUM","ADANIPORTS",
    "HEROMOTOCO","BAJAJ-AUTO","SBILIFE","HDFCLIFE","BRITANNIA","HAL","BEL","LTIM",
]

NIFTY_200_EXTRA = [
    "VEDL","PIDILITIND","GODREJCP","SIEMENS","HAVELLS","ABB","DABUR","MARICO",
    "BANKBARODA","IOC","AMBUJACEM","DLF","TRENT","ZOMATO","POLYCAB","MAXHEALTH",
    "MUTHOOTFIN","TVSMOTOR","JUBLFOOD","MPHASIS","COFORGE","PERSISTENT",
    "LUPIN","TORNTPHARM","AUROPHARMA","BIOCON","FEDERALBNK","IDFCFIRSTB",
    "BANDHANBNK","CANBK","PNB","INDUSTOWER","TATAPOWER","NHPC","IRCTC",
    "BHEL","SAIL","NMDC","GAIL","PETRONET","MRF","BALKRISIND","COLPAL",
    "PAGEIND","VOLTAS","CROMPTON","OBEROIRLTY","PRESTIGE","MFSL","SHREECEM",
    "COALINDIA","ICICIGI","SBICARD","ABCAPITAL","MCDOWELL-N","BERGEPAINT",
    "PHOENIXLTD","ALKEM","LALPATHLAB","METROPOLIS","FORTIS","APOLLOTYRE",
    "CEATLTD","EXIDEIND","AMARAJABAT","BAJAJELEC","ORIENTELEC","RAJESHEXPO",
    "KAJARIACER","ASTRAL","SUPRAJIT","AAVAS","CANFINHOME","INDIAGRID",
    "POWERMECH","KEC","KALPATPOWR","GRINDWELL","JKCEMENT","HEIDELBERG",
    "WHIRLPOOL","SYMPHONY","RELAXO","BATAINDIA","VGUARD","HAVELLS","DIXON",
    "AMBER","POLYCAB","FINOLEX","KPRMILL","VIPIND","JUSTDIAL","NAUKRI",
    "IRCTC","DMART","ZOMATO","PAYTM","NYKAA","POLICYBZR","CARTRADE",
    "EASEMYTRIP","INDIAMART","AFFLE","RATEGAIN","BRIGHTCOMM","LATENTVIEW",
    "TARSONS","CHEMCON","VINYLINDIA","SOLARA","LAURUS","SUVEN","SEQUENT",
    "JUBLPHARMA","GRANULES","IPCA","AJANTPHARM","CAPLIPOINT","NEULANDLAB",
    "AARTI","VINDHYATEL","HFCL","STERLITE","RAILTEL","IRFC","PFC","REC",
    "POWERFIN","SJVN","GETCO","CESC","TORNTPOWER","ADANIGREEN","ADANIPOWER",
    "JSW ENERGY","RENEW","GREENKO","ACMESOLAR","GREENPANEL","CENTURYPLY",
    "GREENPLY","SHRIRAMPPS","MAHINDRA","MFIN",
]

NIFTY_500_EXTRA = [
    "RPOWER","SUZLON","JINDALSAW","WELCORP","RATNAMANI","MAHSCOOTERS",
    "SUNDRMFAST","ENDURANCE","SCHAEFFLER","TIMKEN","GREAVESCOT","ELGIEQUIP",
    "KIRLOSENG","THERMAX","LAKSHVILAS","UJJIVAN","EQUITAS","SURYODAY",
    "UTKARSH","JANA","ESAFSFB","NORTHEAST","CAPITAL","JKBANK","KARNATAKA",
    "SOUTHBANK","CITYUNION","DCBBANK","RBLBANK","TMB","NAINITAL","SARASWAT",
]

def get_symbols(mode):
    if mode == "quick":
        return NIFTY_50
    elif mode == "full":
        return NIFTY_50 + NIFTY_200_EXTRA[:150]
    else:  # overnight
        return NIFTY_50 + NIFTY_200_EXTRA + NIFTY_500_EXTRA


# ─────────────────────────────────────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────
def _normalize_df(df):
    """
    Robustly normalize a yfinance DataFrame to standard OHLCV columns.
    Handles all yfinance API versions (0.1.x, 0.2.x, 2.x).
    """
    import pandas as pd

    # Step 1: flatten MultiIndex if present
    if isinstance(df.columns, pd.MultiIndex):
        # Take first level (Price type), discard second level (Ticker)
        df.columns = df.columns.get_level_values(0)

    # Step 2: rename to standard names (handle case variations)
    rename = {}
    for c in df.columns:
        cl = str(c).lower().replace(" ", "_")
        if cl == "open":          rename[c] = "Open"
        elif cl == "high":         rename[c] = "High"
        elif cl == "low":          rename[c] = "Low"
        elif cl in ("close", "adj_close", "adjclose"): rename[c] = "Close"
        elif cl == "volume":       rename[c] = "Volume"
    df = df.rename(columns=rename)

    # Step 3: keep only OHLCV
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep].copy()

    # Step 4: drop all-NaN rows, convert to float
    df = df.dropna(how="all")
    for c in keep:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Close"])

    return df


def add_technical_indicators(df):
    """Add 7 technical indicators to an OHLCV DataFrame."""
    import ta
    import numpy as np

    # Ensure standard column names first
    df = _normalize_df(df)

    # Ensure we have enough data
    if len(df) < 60 or "Close" not in df.columns:
        return None

    try:
        # SMA — Simple Moving Averages (trend direction)
        df["SMA_20"] = df["Close"].rolling(20).mean()
        df["SMA_50"] = df["Close"].rolling(50).mean()

        # RSI — Relative Strength Index (0-100, momentum)
        df["RSI_14"] = ta.momentum.rsi(df["Close"], window=14)

        # MACD — Moving Average Convergence Divergence
        macd_obj  = ta.trend.MACD(df["Close"])
        df["MACD"] = macd_obj.macd_diff()

        # Bollinger Bands %B
        boll         = ta.volatility.BollingerBands(df["Close"])
        df["BB_pctB"] = boll.bollinger_pband()

        # Volume change
        df["Vol_Change"]   = df["Volume"].pct_change()
        # Price change
        df["Price_Change"] = df["Close"].pct_change()

        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()

        if len(df) < 60:
            return None

        return df
    except Exception as e:
        return None

FEATURES = ["Close","Volume","SMA_20","SMA_50","RSI_14","MACD","BB_pctB","Vol_Change","Price_Change"]

def build_lstm_sequences(df, window=60):
    """Create sliding window sequences for LSTM: X=(samples, window, features), y=(samples,)"""
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler

    feature_cols = [c for c in FEATURES if c in df.columns]
    data  = df[feature_cols].values

    scaler = MinMaxScaler()
    data   = scaler.fit_transform(data)

    close_idx = feature_cols.index("Close")

    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i])
        y.append(data[i, close_idx])   # next day's normalized close

    return (np.array(X, dtype="float32"),
            np.array(y, dtype="float32"),
            scaler, feature_cols)

def build_rf_features(df):
    """Tabular features for Random Forest: X=(samples, features), y=(samples,) as direction."""
    import numpy as np

    feature_cols = [c for c in FEATURES if c in df.columns]
    df = df.copy()

    # Direction label: 1=UP (>0.5% gain), -1=DOWN (>0.5% loss), 0=FLAT
    df["Future_Return"] = df["Close"].shift(-1) / df["Close"] - 1
    df.dropna(inplace=True)

    def label(r):
        if r > 0.005:  return 1   # UP
        if r < -0.005: return -1  # DOWN
        return 0                   # FLAT

    df["Direction"] = df["Future_Return"].apply(label)

    X = df[feature_cols].values.astype("float32")
    y = df["Direction"].values.astype("int")
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Data download
# ─────────────────────────────────────────────────────────────────────────────
def download_stock_data(symbols, period="2y", cache_dir="output/stock_cache"):
    """Download stock data from yfinance with local caching."""
    import yfinance as yf
    import pandas as pd
    from tqdm import tqdm

    os.makedirs(cache_dir, exist_ok=True)
    all_data = {}
    failed   = []
    first_fail_logged = False

    print(f"\n  Downloading {len(symbols)} stocks ({period} of history)...")
    for sym in tqdm(symbols, desc="  Fetching", unit="stock", ncols=70):
        cache_file = os.path.join(cache_dir, f"{sym}.csv")

        # Use cache if fresh (less than 6 hours old)
        if os.path.exists(cache_file):
            age_h = (time.time() - os.path.getmtime(cache_file)) / 3600
            if age_h < 6:
                try:
                    df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                    df = _normalize_df(df)
                    if len(df) > 100 and "Close" in df.columns:
                        all_data[sym] = df
                        continue
                except Exception:
                    pass  # re-download if cache is bad

        # Download fresh
        try:
            ticker = f"{sym}.NS"
            df = yf.download(
                ticker, period=period, progress=False,
                auto_adjust=True, actions=False
            )
            df = _normalize_df(df)
            if len(df) > 100 and "Close" in df.columns:
                df.to_csv(cache_file)   # save normalised version
                all_data[sym] = df
            else:
                failed.append(sym)
        except Exception:
            failed.append(sym)

    print(f"  ✅ Downloaded: {len(all_data)}/{len(symbols)} stocks")
    if failed:
        print(f"  ⚠️  Failed   : {len(failed)} — {', '.join(failed[:10])}{'...' if len(failed)>10 else ''}")
        print(f"     (Failed stocks are skipped — training continues with the rest)")

    # Quick sanity check on first stock
    if all_data:
        first_sym = next(iter(all_data))
        first_df  = all_data[first_sym]
        print(f"  → Sample check ({first_sym}): {len(first_df)} rows, cols={list(first_df.columns[:5])}")

    return all_data


# ─────────────────────────────────────────────────────────────────────────────
# LSTM Model Definition
# ─────────────────────────────────────────────────────────────────────────────
def build_lstm_model(input_size, hidden_size=128, num_layers=2, dropout=0.2):
    import torch.nn as nn

    class StockLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size,
                hidden_size,
                num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )
            self.dropout = nn.Dropout(dropout)
            self.fc      = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.lstm(x)      # (batch, seq, hidden)
            out     = self.dropout(out[:, -1, :])  # last timestep
            return self.fc(out).squeeze(-1)

    return StockLSTM()


# ─────────────────────────────────────────────────────────────────────────────
# LSTM Training
# ─────────────────────────────────────────────────────────────────────────────
def train_lstm(stock_data, args):
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from tqdm import tqdm
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler

    MODE_EPOCHS = {"quick": 50, "full": 100, "overnight": 200}
    epochs = MODE_EPOCHS[args.mode]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  LSTM Config:")
    print(f"    Epochs  : {epochs}")
    print(f"    Device  : {device}")
    print(f"    Stocks  : {len(stock_data)}")

    checkpoint_dir = os.path.join(args.output, "lstm_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ── Build combined dataset from all stocks ────────────────────────────────
    print("\n  Building training sequences from all stocks...")
    all_X, all_y = [], []
    master_scaler = None
    master_features = None

    for sym, df in tqdm(stock_data.items(), desc="  Processing", unit="stock", ncols=70):
        df_feat = add_technical_indicators(df.copy())
        if df_feat is None or len(df_feat) < 100:
            continue
        try:
            X, y, scaler, feat_cols = build_lstm_sequences(df_feat, window=60)
            all_X.append(X)
            all_y.append(y)
            if master_scaler is None:
                master_scaler  = scaler
                master_features = feat_cols
        except Exception:
            continue

    if not all_X:
        print("  ❌ No valid stock data to train on!")
        sys.exit(1)

    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)
    print(f"  ✅ Training sequences: {X_all.shape[0]:,} × window_60 × {X_all.shape[2]} features")

    # Shuffle
    idx = np.random.permutation(len(X_all))
    split = int(len(idx) * 0.9)
    X_train, X_val = X_all[idx[:split]], X_all[idx[split:]]
    y_train, y_val = y_all[idx[:split]], y_all[idx[split:]]

    # Convert to tensors
    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train)
    X_val_t   = torch.tensor(X_val)
    y_val_t   = torch.tensor(y_val)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t), batch_size=256, shuffle=True
    )

    # ── Build model ───────────────────────────────────────────────────────────
    model = build_lstm_model(input_size=X_train.shape[2]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    loss_fn   = nn.MSELoss()

    # ── Check for checkpoint ──────────────────────────────────────────────────
    start_epoch = 0
    best_val_loss = float("inf")
    checkpoint_files = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")])

    if checkpoint_files:
        ckpt_path = os.path.join(checkpoint_dir, checkpoint_files[-1])
        print(f"\n  ⚡ Resuming from checkpoint: {checkpoint_files[-1]}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_loss = ckpt["best_val_loss"]
        print(f"     Resuming from epoch {start_epoch}/{epochs}, best loss: {best_val_loss:.6f}")

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\n  Training LSTM for {epochs - start_epoch} remaining epochs...")

    history = {"train_loss": [], "val_loss": [], "val_rmse": []}
    epoch_bar = tqdm(range(start_epoch, epochs), desc="  Epoch", unit="epoch", ncols=70)

    for epoch in epoch_bar:
        # ─ Train ─
        model.train()
        train_loss = 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            pred = model(X_b)
            loss = loss_fn(pred, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # prevent exploding gradients
            optimizer.step()
            train_loss += loss.item() * len(X_b)
        train_loss /= len(X_train_t)

        # ─ Validate ─
        model.eval()
        with torch.no_grad():
            
            
           
            model.eval()
            val_preds = []
            val_batch_size = 256  # You can lower this to 128 or 64 if it still crashes

            with torch.no_grad(): # This completely disables gradient tracking to save VRAM
                for i in range(0, len(X_val_t), val_batch_size):
                    batch_x = X_val_t[i : i + val_batch_size].to(device)
                    batch_pred = model(batch_x).cpu()
                    val_preds.append(batch_pred)

            val_pred = torch.cat(val_preds, dim=0)
            model.train()
            
            
            
            
            val_loss = loss_fn(val_pred, y_val_t).item()
            # Approximate RMSE in original price scale
            val_rmse_norm = float(np.sqrt(val_loss))

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_rmse"].append(val_rmse_norm)

        # ETA calculation
        elapsed = epoch_bar.format_dict.get("elapsed", 0)
        remaining_epochs = epochs - epoch - 1
        if epoch > start_epoch:
            secs_per_epoch = elapsed / max(1, epoch - start_epoch + 1)
            eta_secs = int(remaining_epochs * secs_per_epoch)
            eta_str = str(timedelta(seconds=eta_secs))
        else:
            eta_str = "calculating..."

        epoch_bar.set_postfix({
            "train_loss": f"{train_loss:.5f}",
            "val_loss":   f"{val_loss:.5f}",
            "ETA":        eta_str,
        })

        # ─ Save checkpoint every 10 epochs ─
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            ckpt_path = os.path.join(checkpoint_dir, f"lstm_epoch_{epoch+1:04d}.pt")
            torch.save({
                "epoch":         epoch,
                "model_state":   model.state_dict(),
                "optim_state":   optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "history":       history,
            }, ckpt_path)

            # Remove older checkpoints (keep only last 3)
            all_ckpts = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")])
            for old in all_ckpts[:-3]:
                os.remove(os.path.join(checkpoint_dir, old))

        # ─ Track best model ─
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.output, "_lstm_best_state.pt"))

    # ── Load best weights ─────────────────────────────────────────────────────
    best_state_path = os.path.join(args.output, "_lstm_best_state.pt")
    if os.path.exists(best_state_path):
        model.load_state_dict(torch.load(best_state_path, map_location=device, weights_only=True))

    # ── Save final model ──────────────────────────────────────────────────────
    output_path = os.path.join(args.output, "lstm_model.pt")
    scaler_path = os.path.join(args.output, "scaler.pkl")

    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "input_size":   X_train.shape[2],
            "hidden_size":  128,
            "num_layers":   2,
            "dropout":      0.2,
            "window":       60,
            "features":     master_features,
        },
        "best_val_loss": best_val_loss,
        "trained_at":    datetime.now().isoformat(),
        "mode":          args.mode,
        "history":       history,
    }, output_path)

    with open(scaler_path, "wb") as f:
        pickle.dump(master_scaler, f)

    final_rmse_norm = float(np.sqrt(best_val_loss))
    print(f"\n  ✅ LSTM saved: {output_path}")
    print(f"  ✅ Scaler saved: {scaler_path}")
    print(f"  📊 Best Val Loss: {best_val_loss:.6f} | Normalized RMSE: {final_rmse_norm:.4f}")

    # ── Save metrics ──────────────────────────────────────────────────────────
    results_path = os.path.join(args.output, "validation_results.txt")
    with open(results_path, "a") as f:
        f.write(f"\n=== LSTM Model ({args.mode}) ===\n")
        f.write(f"Trained at    : {datetime.now().isoformat()}\n")
        f.write(f"Training data : {X_train.shape[0]:,} sequences from {len(stock_data)} stocks\n")
        f.write(f"Best Val Loss : {best_val_loss:.6f}\n")
        f.write(f"Norm RMSE     : {final_rmse_norm:.4f}\n")
        f.write(f"Epochs        : {epochs}\n\n")

    return True


# ─────────────────────────────────────────────────────────────────────────────
# Random Forest Training
# ─────────────────────────────────────────────────────────────────────────────
def train_rf(stock_data, args):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    from tqdm import tqdm

    print("\n  Random Forest Config:")
    print(f"    Trees   : 200")
    print(f"    Stocks  : {len(stock_data)}")

    # ── Build dataset ─────────────────────────────────────────────────────────
    print("\n  Building feature matrix...")
    all_X, all_y = [], []

    for sym, df in tqdm(stock_data.items(), desc="  Processing", unit="stock", ncols=70):
        df_feat = add_technical_indicators(df.copy())
        if df_feat is None or len(df_feat) < 60:
            continue
        try:
            X, y = build_rf_features(df_feat)
            if len(X) > 0:
                all_X.append(X)
                all_y.append(y)
        except Exception:
            continue

    if not all_X:
        print("  ❌ No valid stock data for RF training!")
        sys.exit(1)

    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)

    # Fix inf/nan (numpy arrays don't have .replace() — use np.where)
    X_all = np.where(np.isinf(X_all), np.nan, X_all)
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"  ✅ Training samples: {X_all.shape[0]:,} × {X_all.shape[1]} features")
    print(f"  Distribution: UP={np.sum(y_all==1):,}  FLAT={np.sum(y_all==0):,}  DOWN={np.sum(y_all==-1):,}")

    # ── Scale features ────────────────────────────────────────────────────────
    scaler = StandardScaler()
    
    import numpy as np
    import pandas as pd

    # 1. Replace infinite values with NaN
    X_all = X_all.replace([np.inf, -np.inf], np.nan)

    # 2. Fill NaNs (Forward fill, then fill any remaining at the start with 0)
    X_all = X_all.ffill().fillna(0)
    
    
    
    X_all_scaled = scaler.fit_transform(X_all)

    X_train, X_val, y_train, y_val = train_test_split(
        X_all_scaled, y_all, test_size=0.1, random_state=42, stratify=y_all
    )

    # ── Train (RF doesn't use GPU — runs on all CPU cores) ────────────────────
    print(f"\n  Training Random Forest (200 trees, all CPU cores)...")
    print("  This takes ~2 minutes on CPU — no GPU needed.")

    start = time.time()
    rf = RandomForestClassifier(
        n_estimators     = 200,
        max_depth        = 12,
        min_samples_leaf = 5,
        class_weight     = "balanced",  # handles imbalanced UP/DOWN/FLAT
        random_state     = 42,
        n_jobs           = -1,           # use all CPU cores
        verbose          = 0,
    )
    rf.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"  ✅ Training complete in {int(elapsed//60)}m {int(elapsed%60)}s")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    y_pred = rf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    report = classification_report(y_val, y_pred, target_names=["DOWN", "FLAT", "UP"])

    print(f"\n  📊 Validation Accuracy: {acc*100:.2f}%")
    print(f"\n  Classification Report:")
    print(report)

    # ── Feature importance ────────────────────────────────────────────────────
    feature_names = [c for c in FEATURES if c in stock_data[list(stock_data.keys())[0]].columns or True]
    importance = dict(zip(
        ["Close","Volume","SMA_20","SMA_50","RSI_14","MACD","BB_pctB","Vol_Change","Price_Change"],
        rf.feature_importances_
    ))
    importance_sorted = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    print(f"\n  Feature Importance:")
    for feat, imp in importance_sorted.items():
        bar = "█" * int(imp * 40)
        print(f"    {feat:12s} {bar} {imp:.3f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    output_path = os.path.join(args.output, "rf_model.pkl")
    output_scaler_path = os.path.join(args.output, "rf_scaler.pkl")

    with open(output_path, "wb") as f:
        pickle.dump({
            "model": rf,
            "feature_importance": importance_sorted,
            "trained_at": datetime.now().isoformat(),
            "mode": args.mode,
            "accuracy": acc,
        }, f)

    with open(output_scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"\n  ✅ Saved: {output_path} ({size_mb:.1f} MB)")

    # Save metrics
    results_path = os.path.join(args.output, "validation_results.txt")
    with open(results_path, "a") as f:
        f.write(f"\n=== Random Forest Model ({args.mode}) ===\n")
        f.write(f"Trained at    : {datetime.now().isoformat()}\n")
        f.write(f"Accuracy      : {acc*100:.2f}%\n")
        f.write(f"Training time : {int(elapsed//60)}m {int(elapsed%60)}s\n")
        f.write(f"\nFeature Importance:\n")
        for feat, imp in importance_sorted.items():
            f.write(f"  {feat}: {imp:.3f}\n")
        f.write(f"\n{report}\n")

    return acc > 0.50


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train NiveshAI Price Prediction Models")
    parser.add_argument("--mode",   choices=["quick","full","overnight"], default="full")
    parser.add_argument("--output", default="output")
    parser.add_argument("--model",  choices=["lstm","rf","both"], default="both",
                        help="Which model to train (default: both)")
    args = parser.parse_args()

    import numpy as np
    np.random.seed(42)

    os.makedirs(args.output, exist_ok=True)

    print(f"\n{'═'*60}")
    print(f"  NiveshAI Price Prediction Model Training")
    print(f"  Mode: {args.mode} | Model: {args.model}")
    print(f"{'═'*60}")

    # ── Get symbols ───────────────────────────────────────────────────────────
    symbols = get_symbols(args.mode)
    print(f"\n  Target stocks: {len(symbols)}")

    # ── Download data ─────────────────────────────────────────────────────────
    period = "2y" if args.mode in ["quick", "full"] else "3y"
    cache_dir = os.path.join(args.output, "stock_cache")
    stock_data = download_stock_data(symbols, period=period, cache_dir=cache_dir)

    if len(stock_data) < 10:
        print(f"\n  ❌ Only {len(stock_data)} stocks downloaded — need at least 10 to train.")
        print(f"  Check your internet connection and try again.")
        sys.exit(1)

    all_ok = True

    # ── Train LSTM ────────────────────────────────────────────────────────────
    if args.model in ["lstm", "both"]:
        print(f"\n{'─'*60}")
        print("  [LSTM Training]")
        ok_lstm = train_lstm(stock_data, args)
        all_ok = all_ok and ok_lstm

    # ── Train RF ──────────────────────────────────────────────────────────────
    if args.model in ["rf", "both"]:
        print(f"\n{'─'*60}")
        print("  [Random Forest Training]")
        ok_rf = train_rf(stock_data, args)
        all_ok = all_ok and ok_rf

    if all_ok:
        print(f"\n  ✅ All requested models trained successfully!")
    else:
        print(f"\n  ⚠️  Some models may have issues — check output above.")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
