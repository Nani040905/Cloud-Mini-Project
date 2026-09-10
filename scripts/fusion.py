"""
scripts/fusion.py

Step 9: Forecast Fusion (Convex Combination of Holt-Winters and GRU).
Implements:
1. Convex Hybrid Forecaster:
   - Formulation:
     y_hat_t = alpha_f * y_hat_{HW,t} + (1 - alpha_f) * y_hat_{GRU,t}
   - Motivation:
     Holt-Winters captures deterministic, multi-frequency seasonality (e.g. 24h diurnal cycles).
     The Global GRU captures complex non-linear auto-regressive dynamics and multi-machine cross-patterns.
     Their linear fusion yields lower variance and superior generalization than either model alone.
2. Vectorized Validation Tuning:
   - Grid search across alpha_f in {0.0, 0.1, ..., 1.0}.
   - Fully vectorized array evaluation computes MAE across all candidate weights simultaneously
     in sub-millisecond execution time.
3. Evaluates B3 Baseline (Fused Forecast + 10% relative buffer) with standardized SLA metrics.
4. Outputs aligned predictions for multi-source uncertainty quantification.
"""

import os
import argparse
import numpy as np
import pandas as pd

from utils import (
    compute_metrics, ALPHA_GRID, MODELS_DIR
)


def run_fusion(hw_preds_path: str, gru_preds_path: str, output_dir: str):
    """Aligns HW and GRU predictions, optimizes alpha_f on validation, and evaluates B3."""
    print("Loading Holt-Winters and GRU predictions...")
    hw_df = pd.read_csv(hw_preds_path)
    gru_df = pd.read_csv(gru_preds_path)

    # Merge on machine_id, bin_idx, split
    merged = pd.merge(
        hw_df[['machine_id', 'split', 'bin_idx', 'y_true', 'hw_pred']],
        gru_df[['machine_id', 'bin_idx', 'gru_pred']],
        on=['machine_id', 'bin_idx'],
        how='inner'
    )
    print(f"Total aligned predictions across validation and test: {len(merged):,}")

    val_df = merged[merged['split'] == 'val'].copy()
    test_df = merged[merged['split'] == 'test'].copy()

    # --------------------------------------------------------------------------
    # Vectorized Alpha Tuning on Validation Set
    # --------------------------------------------------------------------------
    best_alpha = 0.5
    best_val_mae = float('inf')

    if len(val_df) > 0:
        yt_val = val_df['y_true'].values
        hw_val = val_df['hw_pred'].values
        gr_val = val_df['gru_pred'].values

        # Vectorized evaluation: (11, N) matrix of predictions across all alpha candidates
        alphas = np.array(ALPHA_GRID)[:, None]
        candidates = alphas * hw_val + (1.0 - alphas) * gr_val
        maes = np.mean(np.abs(candidates - yt_val), axis=1)

        best_idx = int(np.argmin(maes))
        best_alpha = float(ALPHA_GRID[best_idx])
        best_val_mae = float(maes[best_idx])
        print(f"Optimal fusion parameter: alpha_f = {best_alpha:.1f} (Validation MAE = {best_val_mae:.4f})")
    else:
        print("[Notice] Using default equal-weighted fusion: alpha_f = 0.5")

    # Compute fused predictions bounded strictly to [0.0, 1.0]
    merged['fused_pred'] = np.clip(
        best_alpha * merged['hw_pred'] + (1.0 - best_alpha) * merged['gru_pred'],
        0.0, 1.0
    )

    test_merged = merged[merged['split'] == 'test'].copy()

    # Evaluate B3 Baseline (Fusion + 10% relative buffer)
    per_machine_metrics = []
    for mid, grp in test_merged.groupby('machine_id'):
        yt = grp['y_true'].values
        yp = grp['fused_pred'].values
        ar = yp * 1.10
        al = np.clip(ar, 0.0, 1.0)
        m = compute_metrics(yt, yp, ar, al)
        m.update({'machine_id': mid, 'alpha_f': best_alpha})
        per_machine_metrics.append(m)

    m_df = pd.DataFrame(per_machine_metrics)

    os.makedirs(output_dir, exist_ok=True)
    preds_path = os.path.join(output_dir, "fusion_predictions.csv")
    metrics_path = os.path.join(output_dir, "fusion_metrics.csv")

    merged.to_csv(preds_path, index=False)
    m_df.to_csv(metrics_path, index=False)

    print(f"\nSaved fused predictions to: {preds_path}")
    print(f"Saved B3 fusion metrics to: {metrics_path}")

    # Display Macro-Averages
    print("\n" + "=" * 65)
    print(f"         B3 (FUSED FORECAST ONLY, alpha_f={best_alpha:.1f}) SUMMARY")
    print("=" * 65)
    print(f"Test MAE                 : {m_df['mae'].mean():.4f} +/- {m_df['mae'].std():.4f}")
    print(f"Test RMSE                : {m_df['rmse'].mean():.4f} +/- {m_df['rmse'].std():.4f}")
    print(f"SLA Breach Rate          : {m_df['sla_breach_rate'].mean():.2%} (Target <= 2%)")
    print(f"Overshoot Rate           : {m_df['overshoot_rate'].mean():.2%}")
    print(f"Over-provision Mean      : {m_df['over_prov_mean'].mean():.4f}")
    print(f"Under-provision Mean     : {m_df['under_prov_mean'].mean():.4f}")
    print(f"Clip Fraction            : {m_df['clip_fraction'].mean():.2%}")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Forecast fusion and B3 baseline evaluation")
    parser.add_argument("--hw_preds", type=str, default=os.path.join(MODELS_DIR, "hw_predictions.csv"))
    parser.add_argument("--gru_preds", type=str, default=os.path.join(MODELS_DIR, "gru_base_predictions.csv"))
    parser.add_argument("--output_dir", type=str, default=MODELS_DIR)
    args = parser.parse_args()

    run_fusion(args.hw_preds, args.gru_preds, args.output_dir)


if __name__ == "__main__":
    main()
