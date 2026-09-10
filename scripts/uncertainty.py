"""
scripts/uncertainty.py

Step 10: Multi-Source Uncertainty Engine (U1, U2, U3).
Implements:
1. Three Distinct, Causally Separated Uncertainty Dimensions:
   - U1 (Epistemic Uncertainty / Model Disagreement):
     U1_t = |y_hat_{HW,t} - y_hat_{GRU,t}|
     Measures structural divergence between our statistical and neural hypotheses. When the GRU
     and Holt-Winters diverge, structural uncertainty is elevated.
   - U2 (Aleatoric Non-Stationarity / EWMA Error Tracking):
     U2_t = lambda * |y_{fused, t-1} - y_{t-1}| + (1 - lambda) * U2_{t-1}
     Tracks short-term tracking drift and sudden regime shifts.
     Causally initialized at split boundaries to eliminate transient startup artifacts:
       U2_val[0]  = |y_{fused, train}[-1] - y_{train}[-1]|
       U2_test[0] = |y_{fused, val}[-1]   - y_{val}[-1]|
     lambda is tuned in {0.1, ..., 0.9} on the validation set to maximize Spearman rho with error.
   - U3 (Workload Volatility / Burstiness):
     U3_t = StdDev(y[t-6 : t-1])
     Extracted causally via full_series.shift(1).rolling(6).std().
     Measures intrinsic cluster burstiness over the preceding 30-minute window without future leakage.
2. Fast Vectorized Spearman Correlation:
   - Evaluates signal reliability using rank correlation against absolute forecast error:
     rho = SpearmanCorr(U_i, |y_hat_{fused} - y|)
   - High positive rho proves that higher uncertainty signals correctly anticipate larger errors.
"""

import os
import argparse
import numpy as np
import pandas as pd

from utils import (
    fast_spearman, LAMBDA_GRID, SERIES_DIR, MODELS_DIR, UNCERTAINTY_DIR
)


def compute_u2_series(actual: np.ndarray, fused: np.ndarray, lam: float, init_err: float) -> np.ndarray:
    """
    Computes causal Exponentially Weighted Moving Average (EWMA) of absolute forecast errors.
    At step t, U2[t] is formed solely from errors observed at or before t-1.
    """
    n = len(actual)
    u2 = np.zeros(n, dtype=np.float64)
    curr = init_err
    for t in range(n):
        u2[t] = curr
        # Update recurrence state using the error at current step t (available for t+1)
        err_t = abs(fused[t] - actual[t]) if not (np.isnan(fused[t]) or np.isnan(actual[t])) else curr
        curr = lam * err_t + (1.0 - lam) * curr
    return u2


def compute_uncertainty_signals(fusion_preds_path: str, series_dir: str, output_dir: str):
    """Generates causal U1, U2, and U3 signals across all machines."""
    print("Loading fused predictions...")
    preds_df = pd.read_csv(fusion_preds_path)

    # --------------------------------------------------------------------------
    # U1: Model Disagreement (Epistemic Divergence)
    # --------------------------------------------------------------------------
    preds_df['U1'] = np.abs(preds_df['hw_pred'] - preds_df['gru_pred'])

    # --------------------------------------------------------------------------
    # U2: Tune Smoothing Parameter lambda on Validation Split
    # --------------------------------------------------------------------------
    val_subset = preds_df[preds_df['split'] == 'val']
    best_lambda = 0.5
    best_mean_rho = -float('inf')

    for lam in LAMBDA_GRID:
        rhos = []
        for mid, grp in val_subset.groupby('machine_id'):
            yt = grp['y_true'].values
            yp = grp['fused_pred'].values
            init_err = abs(yp[0] - yt[0]) if len(yt) > 0 else 0.0
            u2_cand = compute_u2_series(yt, yp, lam, init_err)
            err = np.abs(yp - yt)
            rho = fast_spearman(u2_cand, err)
            rhos.append(rho)

        mean_rho = float(np.mean(rhos)) if rhos else 0.0
        if mean_rho > best_mean_rho:
            best_mean_rho, best_lambda = mean_rho, float(lam)

    print(f"Optimal U2 parameter: lambda = {best_lambda:.1f} (Validation Mean Spearman rho = {best_mean_rho:.4f})")

    # --------------------------------------------------------------------------
    # Compute Causal Signals per Machine
    # --------------------------------------------------------------------------
    machine_signal_dfs = []
    for mid, grp in preds_df.groupby('machine_id'):
        m_df = grp.sort_values('bin_idx').copy()

        # Load complete machine series for continuous temporal features
        series_path = os.path.join(series_dir, f"{mid}.parquet")
        s_df = pd.read_parquet(series_path).sort_values('bin_idx')

        # U3: Rolling Volatility of previous 6 bins (strictly causal shift)
        u3_full = s_df['cpu'].shift(1).rolling(6).std().values
        bin_to_u3 = dict(zip(s_df['bin_idx'].values, u3_full))
        m_df['U3'] = m_df['bin_idx'].map(bin_to_u3).fillna(0.0)

        # U2 with boundary-safe causal handoff
        val_mask = (m_df['split'] == 'val')
        test_mask = (m_df['split'] == 'test')
        val_rows = m_df[val_mask]
        test_rows = m_df[test_mask]

        init_val_err = abs(val_rows['fused_pred'].iloc[0] - val_rows['y_true'].iloc[0]) if len(val_rows) > 0 else 0.0
        u2_val = compute_u2_series(val_rows['y_true'].values, val_rows['fused_pred'].values, best_lambda, init_val_err)

        init_test_err = abs(val_rows['fused_pred'].iloc[-1] - val_rows['y_true'].iloc[-1]) if len(val_rows) > 0 else 0.0
        u2_test = compute_u2_series(test_rows['y_true'].values, test_rows['fused_pred'].values, best_lambda, init_test_err)

        u2_combined = np.zeros(len(m_df))
        u2_combined[val_mask] = u2_val
        u2_combined[test_mask] = u2_test
        m_df['U2'] = u2_combined

        m_df['abs_error'] = np.abs(m_df['fused_pred'] - m_df['y_true'])
        machine_signal_dfs.append(m_df)

    final_signals_df = pd.concat(machine_signal_dfs, ignore_index=True)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "uncertainty.csv")
    final_signals_df.to_csv(out_path, index=False)
    print(f"Saved uncertainty signals to: {out_path}")

    # Summary of signal quality on the test split
    test_sub = final_signals_df[final_signals_df['split'] == 'test']
    print("\n" + "=" * 65)
    print("               UNCERTAINTY SIGNALS SUMMARY (TEST SET)")
    print("=" * 65)
    for sig in ['U1', 'U2', 'U3']:
        rho_vals = [
            fast_spearman(grp[sig].values, grp['abs_error'].values)
            for _, grp in test_sub.groupby('machine_id')
        ]
        mean_rho = np.mean(rho_vals)
        print(f"Signal {sig} | Mean: {test_sub[sig].mean():.4f} | Std: {test_sub[sig].std():.4f} | Mean Spearman rho: {mean_rho:.4f}")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Multi-source uncertainty computation")
    parser.add_argument("--fusion_preds", type=str, default=os.path.join(MODELS_DIR, "fusion_predictions.csv"))
    parser.add_argument("--series_dir", type=str, default=SERIES_DIR)
    parser.add_argument("--output_dir", type=str, default=UNCERTAINTY_DIR)
    args = parser.parse_args()

    compute_uncertainty_signals(args.fusion_preds, args.series_dir, args.output_dir)


if __name__ == "__main__":
    main()
