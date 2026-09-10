"""
scripts/features.py

Step 4 & 5: Causal Feature Extraction and Chronological Splitting.
Implements:
1. Dual-Tier Feature Architecture:
   - Base Model (7 features): ['cpu', 'cpu_lag1', 'cpu_lag2', 'roll_mean', 'roll_std', 'hour_sin', 'hour_cos']
   - Extended Model (15 features): Base + ['mem', 'mem_lag1', 'mem_roll_mean', 'disk', 'cpu_diff1', 'roll_max', 'dow_sin', 'dow_cos']
2. Strict Per-Machine Temporal Isolation:
   - All lags, rolling stats, diffs computed independently per machine.
3. Strict Causality:
   - All feature rollings and lags use .shift(1) to strictly exclude time t.
4. Chronological 70/15/15 Splitting:
   - Preserves continuous timeline for boundary-aware lookback.
5. Sequence Generation:
   - Sliding windows for GRU (seq_len=12) respecting split boundaries and contiguous timestamps.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd

TRACE_START = pd.Timestamp('2011-05-01 00:00:00', tz='UTC')
SEQ_LEN = 12

BASE_FEATURES = [
    'cpu', 'cpu_lag1', 'cpu_lag2', 'roll_mean', 'roll_std', 'hour_sin', 'hour_cos'
]

EXTENDED_FEATURES = BASE_FEATURES + [
    'mem', 'mem_lag1', 'mem_roll_mean', 'disk', 'cpu_diff1', 'roll_max', 'dow_sin', 'dow_cos'
]


def extract_features_single_machine(df_m: pd.DataFrame) -> pd.DataFrame:
    """
    Extract causal features for a single machine series.
    Input df_m requires: 'bin_idx', 'timestamp_sec', 'cpu', 'mem', 'disk', 'touched'.
    Returns: DataFrame containing both base and extended feature sets.
    """
    df = df_m.copy()
    # Causally forward-fill intermittent telemetry counters (disk and mem dropouts only)
    # CPU missing periods (gaps > 2) MUST strictly remain NaN per project specification
    for col in ['mem', 'disk']:
        if col in df.columns:
            df[col] = df[col].ffill().fillna(0.0)

    # Step 1: Extended feature 'disk' must strictly represent disk_io_time(t-1)
    if 'disk' in df.columns:
        df['disk'] = df['disk'].shift(1)

    if 'datetime' not in df.columns:
        # 5-minute bins from trace start
        df['datetime'] = TRACE_START + pd.to_timedelta(df['bin_idx'] * 5, unit='min')

    # Temporal cyclic features
    hour = df['datetime'].dt.hour + df['datetime'].dt.minute / 60.0
    df['hour_sin'] = np.sin(2.0 * np.pi * hour / 24.0)
    df['hour_cos'] = np.cos(2.0 * np.pi * hour / 24.0)

    dow = df['datetime'].dt.dayofweek + hour / 24.0
    df['dow_sin'] = np.sin(2.0 * np.pi * dow / 7.0)
    df['dow_cos'] = np.cos(2.0 * np.pi * dow / 7.0)

    # Base causal features (strictly shifted to exclude t)
    df['cpu_lag1']  = df['cpu'].shift(1)
    df['cpu_lag2']  = df['cpu'].shift(2)
    df['roll_mean'] = df['cpu'].shift(1).rolling(6).mean()
    df['roll_std']  = df['cpu'].shift(1).rolling(6).std()

    # Extended multi-resource features (strictly shifted)
    df['mem_lag1']      = df['mem'].shift(1)
    df['mem_roll_mean'] = df['mem'].shift(1).rolling(6).mean()
    df['cpu_diff1']     = df['cpu_lag1'] - df['cpu_lag2']
    df['roll_max']      = df['cpu'].shift(1).rolling(6).max()

    return df


def get_split_indices(total_len: int, train_ratio: float = 0.70, val_ratio: float = 0.15):
    """Compute chronological train/val/test slice indices."""
    n_train = int(total_len * train_ratio)
    n_val = int(total_len * val_ratio)
    train_slice = (0, n_train)
    val_slice = (n_train, n_train + n_val)
    test_slice = (n_train + n_val, total_len)
    return {
        'train': train_slice,
        'val': val_slice,
        'test': test_slice
    }


def build_gru_sequences(feature_df: pd.DataFrame, feature_cols: list, target_col: str,
                        split_start: int, split_end: int, seq_len: int = SEQ_LEN):
    """
    Generate causal sliding windows for GRU training/eval.
    Rules:
    - Prediction TARGET at timestep t must be strictly within [split_start, split_end).
    - Window contains [t - seq_len, ..., t - 1].
    - Window CAN cross split boundary into preceding split (causal history).
    - Window MUST NEVER contain NaNs (unfilled gaps).
    """
    X_list = []
    y_list = []
    indices = []

    features = feature_df[feature_cols].values
    targets = feature_df[target_col].values

    for t in range(split_start, split_end):
        w_start = t - seq_len
        if w_start < 0:
            continue  # Not enough history for full window

        window = features[w_start:t]
        target = targets[t]

        # Check for NaNs
        if np.isnan(window).any() or np.isnan(target):
            continue

        X_list.append(window)
        y_list.append(target)
        indices.append(t)

    if len(X_list) == 0:
        return np.empty((0, seq_len, len(feature_cols))), np.empty((0,)), np.empty((0,), dtype=int)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32), np.array(indices, dtype=int)


def process_all_selected_machines(series_dir: str, selected_json: str, output_base_dir: str):
    """Generate and save feature parquets for all selected machines."""
    with open(selected_json, 'r') as f:
        chosen_mids = json.load(f)

    out_base = os.path.join(output_base_dir, "features_base")
    out_ext  = os.path.join(output_base_dir, "features_extended")
    os.makedirs(out_base, exist_ok=True)
    os.makedirs(out_ext, exist_ok=True)

    print(f"Extracting features for {len(chosen_mids)} selected machines...")
    split_info = {}

    for mid in chosen_mids:
        in_path = os.path.join(series_dir, f"{mid}.parquet")
        if not os.path.exists(in_path):
            print(f"[Warning] Series file missing for machine {mid}")
            continue

        m_df = pd.read_parquet(in_path)
        feat_df = extract_features_single_machine(m_df)

        # Base tier features
        feat_df[BASE_FEATURES + ['bin_idx', 'timestamp_sec']].to_parquet(
            os.path.join(out_base, f"{mid}.parquet"), index=False
        )

        # Extended tier features
        feat_df[EXTENDED_FEATURES + ['bin_idx', 'timestamp_sec']].to_parquet(
            os.path.join(out_ext, f"{mid}.parquet"), index=False
        )

        splits = get_split_indices(len(feat_df))
        split_info[str(mid)] = {
            'total_bins': len(feat_df),
            'train_range': splits['train'],
            'val_range': splits['val'],
            'test_range': splits['test']
        }

    split_path = os.path.join(output_base_dir, "split_info.json")
    with open(split_path, 'w') as f:
        json.dump(split_info, f, indent=2)

    print(f"Features saved successfully to:\n  - {out_base}\n  - {out_ext}")
    print(f"Split metadata saved to: {split_path}")


def main():
    parser = argparse.ArgumentParser(description="Feature extraction for Base and Extended configurations")
    parser.add_argument("--series_dir", type=str, default="data/processed/google/machine_series")
    parser.add_argument("--selected_json", type=str, default="data/selected_machines.json")
    parser.add_argument("--output_dir", type=str, default="data/processed/google")
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    series_dir = os.path.join(project_root, args.series_dir)
    selected_json = os.path.join(project_root, args.selected_json)
    output_dir = os.path.join(project_root, args.output_dir)

    process_all_selected_machines(series_dir, selected_json, output_dir)


if __name__ == "__main__":
    main()
