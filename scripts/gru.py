"""
scripts/gru.py

Step 8: Global Gated Recurrent Unit (GRU) Forecasting Model (Base vs Extended).
Implements:
1. Gated Recurrent Unit (GRU) Architecture:
   - Reset Gate:   r_t = sigma(W_r * x_t + U_r * h_{t-1} + b_r)
   - Update Gate:  z_t = sigma(W_z * x_t + U_z * h_{t-1} + b_z)
   - Candidate:    h_tilde_t = tanh(W_h * x_t + U_h * (r_t (*) h_{t-1}) + b_h)
   - Hidden State: h_t = (1 - z_t) (*) h_{t-1} + z_t (*) h_tilde_t
   - Output Head:  y_hat_t = Sigmoid(W_o * Dropout(h_t) + b_o)
2. Architecture Design Decisions:
   - Hidden Size = 32: Balances expressiveness with minimal latency and parameter count (O(3K) params).
     Enables ultra-low inference cost suitable for real-time edge/cloud controllers.
   - Post-GRU Dropout (p = 0.2): Applied to final recurrent state h_T to mitigate co-adaptation
     and prevent overfitting across multi-tenant cluster workloads.
   - Sigmoid Output: Hard-bounds predictions into physical range (0, 1), eliminating unphysical negative
     or >100% CPU allocations without artificial post-hoc clamping inside the computational graph.
3. Strict Temporal Integrity & Lookback Crossings:
   - Sequence generation (seq_len = 12 bins = 60 min lookback) is executed strictly per-machine.
   - Lookback windows for validation targets causally cross into training observations; lookback windows
     for test targets causally cross into validation observations. Targets strictly respect assigned splits.
   - DataLoader: batch_size = 32, shuffle = False (preserves chronological training stability).
4. Dual-Tier Feature Ablation:
   - Base Tier (7 features): ['cpu', 'cpu_lag1', 'cpu_lag2', 'roll_mean', 'roll_std', 'hour_sin', 'hour_cos'].
   - Extended Tier (15 features): Base + ['mem', 'mem_lag1', 'mem_roll_mean', 'disk', 'cpu_diff1', 'roll_max', 'dow_sin', 'dow_cos'].
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from utils import (
    compute_metrics, SEED, SEQ_LEN, PROCESSED_DIR, MODELS_DIR,
    SELECTED_MACHINES_PATH, SPLIT_INFO_PATH, SLA_TARGET
)
from features import BASE_FEATURES, EXTENDED_FEATURES, build_gru_sequences

BATCH_SIZE = 32
HIDDEN_SIZE = 32
DROPOUT_RATE = 0.2
LEARNING_RATE = 1e-3
EPOCHS = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed=SEED):
    """Enforces deterministic initialization across PyTorch, NumPy, and random states."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class GRUPredictor(nn.Module):
    """Single-layer GRU with explicit post-recurrent dropout and Sigmoid output head."""
    def __init__(self, input_size: int, hidden_size: int = HIDDEN_SIZE, p_drop: float = DROPOUT_RATE):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(p=p_drop)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.gru(x)
        h_last = self.dropout(out[:, -1, :])
        return self.sigmoid(self.fc(h_last)).squeeze(-1)


def prepare_global_datasets(features_dir: str, selected_json: str, split_info_path: str,
                             feature_cols: list, seq_len: int = SEQ_LEN):
    """
    Builds global train, val, and test tensors across all machines with per-machine sequence isolation.
    """
    with open(selected_json, 'r') as f:
        chosen_mids = json.load(f)

    with open(split_info_path, 'r') as f:
        split_info = json.load(f)

    train_X_l, train_y_l = [], []
    val_X_l, val_y_l     = [], []
    test_X_l, test_y_l   = [], []
    val_meta_l, test_meta_l = [], []

    for mid in chosen_mids:
        fpath = os.path.join(features_dir, f"{mid}.parquet")
        if not os.path.exists(fpath):
            continue

        df = pd.read_parquet(fpath)
        splits = split_info[str(mid)]

        tr_s, tr_e = splits['train_range']
        va_s, va_e = splits['val_range']
        te_s, te_e = splits['test_range']

        # Construct sequences per machine (lookback window crosses boundaries causally)
        X_tr, y_tr, _      = build_gru_sequences(df, feature_cols, 'cpu', tr_s, tr_e, seq_len)
        X_va, y_va, idx_va = build_gru_sequences(df, feature_cols, 'cpu', va_s, va_e, seq_len)
        X_te, y_te, idx_te = build_gru_sequences(df, feature_cols, 'cpu', te_s, te_e, seq_len)

        if len(X_tr) > 0:
            train_X_l.append(X_tr)
            train_y_l.append(y_tr)
        if len(X_va) > 0:
            val_X_l.append(X_va)
            val_y_l.append(y_va)
            for t_idx, y_val in zip(idx_va, y_va):
                val_meta_l.append({'machine_id': mid, 'split': 'val', 'bin_idx': int(t_idx), 'y_true': float(y_val)})
        if len(X_te) > 0:
            test_X_l.append(X_te)
            test_y_l.append(y_te)
            for t_idx, y_val in zip(idx_te, y_te):
                test_meta_l.append({'machine_id': mid, 'split': 'test', 'bin_idx': int(t_idx), 'y_true': float(y_val)})

    train_X = np.concatenate(train_X_l, axis=0) if train_X_l else np.empty((0, seq_len, len(feature_cols)))
    train_y = np.concatenate(train_y_l, axis=0) if train_y_l else np.empty((0,))
    val_X   = np.concatenate(val_X_l, axis=0) if val_X_l else np.empty((0, seq_len, len(feature_cols)))
    val_y   = np.concatenate(val_y_l, axis=0) if val_y_l else np.empty((0,))
    test_X  = np.concatenate(test_X_l, axis=0) if test_X_l else np.empty((0, seq_len, len(feature_cols)))
    test_y  = np.concatenate(test_y_l, axis=0) if test_y_l else np.empty((0,))

    return {
        'train': (train_X, train_y), 'val': (val_X, val_y), 'test': (test_X, test_y),
        'val_meta': val_meta_l, 'test_meta': test_meta_l
    }


def train_gru(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int = EPOCHS, lr: float = LEARNING_RATE):
    """Trains global GRU with early checkpointing on validation loss."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    model.to(DEVICE)
    best_val_loss = float('inf')
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(by)
        train_loss /= max(1, len(train_loader.dataset))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                val_loss += criterion(model(bx), by).item() * len(by)
        val_loss /= max(1, len(val_loader.dataset))
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train MSE: {train_loss:.5f} | Val MSE: {val_loss:.5f} | LR: {optimizer.param_groups[0]['lr']:.1e}")

    if best_weights:
        model.load_state_dict(best_weights)
    print(f"Best Validation MSE: {best_val_loss:.5f}")
    return model


def run_gru_pipeline(tier: str, output_dir: str):
    """Executes dataset preparation, model training, and B1 evaluation for a given feature tier."""
    set_seed(SEED)
    feature_cols = BASE_FEATURES if tier == 'base' else EXTENDED_FEATURES
    features_dir = os.path.join(PROCESSED_DIR, f"features_{tier}")

    print(f"\n=======================================================")
    print(f"   Executing Global GRU Pipeline — Tier: {tier.upper()} ({len(feature_cols)} features)")
    print(f"=======================================================")

    data = prepare_global_datasets(features_dir, SELECTED_MACHINES_PATH, SPLIT_INFO_PATH, feature_cols)
    tr_X, tr_y = data['train']
    va_X, va_y = data['val']
    te_X, te_y = data['test']

    print(f"Tensors | Train: {tr_X.shape} | Val: {va_X.shape} | Test: {te_X.shape}")

    tr_loader = DataLoader(TensorDataset(torch.tensor(tr_X, dtype=torch.float32), torch.tensor(tr_y, dtype=torch.float32)), batch_size=BATCH_SIZE, shuffle=False)
    va_loader = DataLoader(TensorDataset(torch.tensor(va_X, dtype=torch.float32), torch.tensor(va_y, dtype=torch.float32)), batch_size=BATCH_SIZE, shuffle=False)
    te_loader = DataLoader(TensorDataset(torch.tensor(te_X, dtype=torch.float32), torch.tensor(te_y, dtype=torch.float32)), batch_size=BATCH_SIZE, shuffle=False)

    model = GRUPredictor(input_size=len(feature_cols)).to(DEVICE)
    model = train_gru(model, tr_loader, va_loader)

    # Save trained model weights
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, f"gru_{tier}.pt"))

    # Generate inference predictions on validation and test splits
    model.eval()
    with torch.no_grad():
        val_preds = []
        for bx, _ in va_loader:
            val_preds.append(model(bx.to(DEVICE)).cpu().numpy())
        val_preds = np.concatenate(val_preds) if val_preds else np.empty((0,))

        test_preds = []
        for bx, _ in te_loader:
            test_preds.append(model(bx.to(DEVICE)).cpu().numpy())
        test_preds = np.concatenate(test_preds) if test_preds else np.empty((0,))

    # Format predictions table
    val_df = pd.DataFrame(data['val_meta'])
    val_df['gru_pred'] = val_preds

    test_df = pd.DataFrame(data['test_meta'])
    test_df['gru_pred'] = test_preds

    preds_combined = pd.concat([val_df, test_df], ignore_index=True)
    preds_csv = os.path.join(output_dir, f"gru_{tier}_predictions.csv")
    preds_combined.to_csv(preds_csv, index=False)

    # Evaluate B1 Baseline (GRU + 10% relative buffer) per machine
    per_machine_metrics = []
    for mid, grp in test_df.groupby('machine_id'):
        yt = grp['y_true'].values
        yp = grp['gru_pred'].values
        ar = yp * 1.10
        al = np.clip(ar, 0.0, 1.0)
        m = compute_metrics(yt, yp, ar, al)
        m['machine_id'] = mid
        per_machine_metrics.append(m)

    m_df = pd.DataFrame(per_machine_metrics)
    metrics_csv = os.path.join(output_dir, f"gru_{tier}_metrics.csv")
    m_df.to_csv(metrics_csv, index=False)

    print(f"\nSaved {tier.upper()} predictions to: {preds_csv}")
    print(f"Saved {tier.upper()} B1 metrics to: {metrics_csv}")

    # Display Macro-Averages
    print("\n" + "=" * 65)
    print(f"         B1 (GRU {tier.upper()} TIER) MACRO-AVERAGE SUMMARY")
    print("=" * 65)
    print(f"Test MAE                 : {m_df['mae'].mean():.4f} +/- {m_df['mae'].std():.4f}")
    print(f"Test RMSE                : {m_df['rmse'].mean():.4f} +/- {m_df['rmse'].std():.4f}")
    print(f"SLA Breach Rate          : {m_df['sla_breach_rate'].mean():.2%} (Target <= {SLA_TARGET:.1%})")
    print(f"Overshoot Rate           : {m_df['overshoot_rate'].mean():.2%}")
    print(f"Over-provision Mean      : {m_df['over_prov_mean'].mean():.4f}")
    print(f"Under-provision Mean     : {m_df['under_prov_mean'].mean():.4f}")
    print(f"Clip Fraction            : {m_df['clip_fraction'].mean():.2%}")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate Global GRU model")
    parser.add_argument("--tier", type=str, choices=['base', 'extended'], default='base')
    parser.add_argument("--output_dir", type=str, default=MODELS_DIR)
    args = parser.parse_args()

    run_gru_pipeline(args.tier, args.output_dir)


if __name__ == "__main__":
    main()
