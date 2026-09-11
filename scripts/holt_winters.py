"""
scripts/holt_winters.py

Step 7: Per-Machine Holt-Winters Exponential Smoothing and B2 Baseline.
Implements:
1. Triple Exponential Smoothing (Additive Holt-Winters):
   - Model formulation:
     Level:    L_t = alpha * (y_t - S_{t-m}) + (1 - alpha) * (L_{t-1} + b_{t-1})
     Trend:    b_t = beta  * (L_t - L_{t-1}) + (1 - beta)  * b_{t-1}
     Seasonal: S_t = gamma * (y_t - L_t)     + (1 - gamma) * S_{t-m}
     Forecast: y_hat_{t+h} = L_t + h * b_t + S_{t + h - m}
2. Causal Hyperparameter Tuning:
   - Evaluates candidate seasonal periods m in {48, 144, 288} on the validation set.
   - Selects m* = argmin_m MAE_val(m).
3. Causal Refit (Leakage-Free Out-of-Sample Evaluation):
   - Prior to generating test predictions, the model is refitted on combined [train + val] data
     using the optimal m*. This ensures the state (L, b, S) is updated with recent observations
     without exposing the model to the test set.
4. Physical Bounding:
   - Output predictions are strictly bounded to physical utilization limits [0.0, 1.0].
5. Evaluates B2 Baseline (HW + 10% fixed relative buffer) with standardized SLA metrics.
"""

import os
import json
import argparse
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from joblib import Parallel, delayed

# Suppress routine optimization convergence warnings from statsmodels L-BFGS-B
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from utils import (
    compute_metrics, SERIES_DIR, SELECTED_MACHINES_PATH,
    SPLIT_INFO_PATH, MODELS_DIR, SLA_TARGET
)


def fit_hw_single_machine(series: pd.Series, train_slice: tuple, val_slice: tuple, test_slice: tuple,
                          candidate_m: list = [48, 144, 288]):
    """
    Fits and forecasts Holt-Winters for a single machine with validation tuning and causal refit.
    """
    tr_s, tr_e = train_slice
    va_s, va_e = val_slice
    te_s, te_e = test_slice

    train_series = series.iloc[tr_s:tr_e].copy()
    val_data     = series.iloc[va_s:va_e]
    test_data    = series.iloc[te_s:te_e]
    n_val, n_test = len(val_data), len(test_data)

    # Maintain regular 5-minute grid spacing for model fitting across gaps
    train_filled = train_series.interpolate(method='linear').bfill().ffill()
    val_actual = val_data.values
    valid_mask = ~np.isnan(val_actual)

    best_m = None
    best_val_mae = float('inf')
    best_val_preds = np.full(n_val, np.nan)
    best_fitted_model = None

    # Filter candidate seasonal periods requiring at least 2 complete cycles in training
    valid_m = [m for m in candidate_m if len(train_filled) >= 2 * m]

    if valid_m:
        for m in valid_m:
            try:
                model = ExponentialSmoothing(
                    train_filled.values, trend='add', seasonal='add',
                    seasonal_periods=m, initialization_method='estimated'
                ).fit(optimized=True)
                val_p = model.forecast(n_val)
                if np.sum(valid_mask) > 0:
                    mae = float(np.mean(np.abs(val_actual[valid_mask] - val_p[valid_mask])))
                else:
                    mae = float('inf')
                if mae < best_val_mae:
                    best_val_mae, best_m, best_val_preds = mae, m, val_p
                    best_fitted_model = model
            except Exception:
                continue

    # Fallback to non-seasonal Holt's linear trend if seasonal fit is infeasible
    if best_m is None:
        try:
            model = ExponentialSmoothing(
                train_filled.values, trend='add', seasonal=None,
                initialization_method='estimated'
            ).fit(optimized=True)
            best_val_preds = model.forecast(n_val)
            if np.sum(valid_mask) > 0:
                best_val_mae = float(np.mean(np.abs(val_actual[valid_mask] - best_val_preds[valid_mask])))
            else:
                best_val_mae = 0.0
            best_m = 0
            best_fitted_model = model
        except Exception:
            mean_val = float(train_filled.mean()) if len(train_filled) > 0 else 0.0
            best_val_preds = np.full(n_val, mean_val)
            if np.sum(valid_mask) > 0:
                best_val_mae = float(np.mean(np.abs(val_actual[valid_mask] - best_val_preds[valid_mask])))
            else:
                best_val_mae = 0.0
            best_m = -1
            best_fitted_model = None

    # Extract last training prediction strictly from model fitted values (no target leakage)
    if best_fitted_model is not None and hasattr(best_fitted_model, 'fittedvalues') and len(best_fitted_model.fittedvalues) > 0:
        last_train_pred = float(np.clip(best_fitted_model.fittedvalues[-1], 0.0, 1.0))
    else:
        last_train_pred = float(np.clip(train_filled.mean(), 0.0, 1.0))

    # --------------------------------------------------------------------------
    # Causal Refit: Update level/trend/seasonality using combined [train + val]
    # --------------------------------------------------------------------------
    train_val_series = series.iloc[tr_s:va_e].copy()
    train_val_filled = train_val_series.interpolate(method='linear').bfill().ffill()
    try:
        if best_m > 0 and len(train_val_filled) >= 2 * best_m:
            refit = ExponentialSmoothing(
                train_val_filled.values, trend='add', seasonal='add',
                seasonal_periods=best_m, initialization_method='estimated'
            ).fit(optimized=True)
            test_preds = refit.forecast(n_test)
        elif best_m >= 0:
            refit = ExponentialSmoothing(
                train_val_filled.values, trend='add', seasonal=None,
                initialization_method='estimated'
            ).fit(optimized=True)
            test_preds = refit.forecast(n_test)
        else:
            test_preds = np.full(n_test, train_val_filled.mean())
    except Exception:
        test_preds = np.full(n_test, train_val_filled.mean())

    # Bound physical forecasts strictly to [0.0, 1.0]
    test_preds = np.clip(test_preds, 0.0, 1.0)
    best_val_preds = np.clip(best_val_preds, 0.0, 1.0)

    # Evaluate B2 Baseline (HW + 10% relative buffer)
    ar_b2 = test_preds * 1.10
    al_b2 = np.clip(ar_b2, 0.0, 1.0)
    metrics = compute_metrics(test_data.values, test_preds, ar_b2, al_b2)

    return {
        'best_m': best_m,
        'val_mae': best_val_mae,
        'val_preds': best_val_preds,
        'test_preds': test_preds,
        'last_train_pred': last_train_pred,
        'metrics': metrics
    }


def _fit_single_machine_worker(mid, series_dir, split_info, candidate_m):
    """Worker function for fitting Holt-Winters on a single machine."""
    warnings.filterwarnings("ignore")
    in_path = os.path.join(series_dir, f"{mid}.parquet")
    if not os.path.exists(in_path):
        return None

    df = pd.read_parquet(in_path)
    splits = split_info[str(mid)]

    res = fit_hw_single_machine(
        df['cpu'], splits['train_range'], splits['val_range'], splits['test_range'],
        candidate_m=candidate_m
    )

    m = res['metrics']
    m.update({'machine_id': mid, 'best_m': res['best_m'], 'val_mae': res['val_mae']})

    tr_s, tr_e = splits['train_range']
    last_train_p = res['last_train_pred']
    train_rec = {
        'machine_id': mid, 'split': 'train', 'bin_idx': tr_e - 1,
        'y_true': df['cpu'].iloc[tr_e - 1], 'hw_pred': last_train_p
    }

    va_s, va_e = splits['val_range']
    val_recs = [{
        'machine_id': mid, 'split': 'val', 'bin_idx': va_s + idx,
        'y_true': df['cpu'].iloc[va_s + idx], 'hw_pred': p
    } for idx, p in enumerate(res['val_preds'])]

    te_s, te_e = splits['test_range']
    test_recs = [{
        'machine_id': mid, 'split': 'test', 'bin_idx': te_s + idx,
        'y_true': df['cpu'].iloc[te_s + idx], 'hw_pred': p
    } for idx, p in enumerate(res['test_preds'])]

    return m, train_rec, val_recs, test_recs


def run_holt_winters(series_dir: str, selected_json: str, split_info_path: str,
                     output_dir: str, candidate_m: list = [48, 144, 288]):
    """Execute Holt-Winters forecasting and B2 evaluation across all machines in parallel."""
    with open(selected_json, 'r') as f:
        chosen_mids = json.load(f)

    with open(split_info_path, 'r') as f:
        split_info = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    all_metrics, train_records, val_records, test_records = [], [], [], []

    print(f"Fitting Holt-Winters models across {len(chosen_mids)} machines in parallel...")
    results = Parallel(n_jobs=-1, batch_size=4)(
        delayed(_fit_single_machine_worker)(mid, series_dir, split_info, candidate_m)
        for mid in chosen_mids
    )

    for item in results:
        if item is None:
            continue
        m, train_rec, val_recs, test_recs = item
        all_metrics.append(m)
        train_records.append(train_rec)
        val_records.extend(val_recs)
        test_records.extend(test_recs)

    # Save metrics and predictions
    m_df = pd.DataFrame(all_metrics)
    m_df['model_type'] = m_df['best_m'].apply(
        lambda m: 'seasonal_hw' if m > 0 else ('non_seasonal_holt' if m == 0 else 'mean_fallback')
    )
    p_df = pd.concat([pd.DataFrame(train_records), pd.DataFrame(val_records), pd.DataFrame(test_records)], ignore_index=True)

    metrics_path = os.path.join(output_dir, "hw_metrics.csv")
    preds_path = os.path.join(output_dir, "hw_predictions.csv")

    m_df.to_csv(metrics_path, index=False)
    p_df.to_csv(preds_path, index=False)

    print(f"\nSaved Holt-Winters evaluation metrics to: {metrics_path}")
    print(f"Saved Holt-Winters aligned predictions to: {preds_path}")

    # Display Macro-Averages and Breakdown
    n_total = max(1, len(m_df))
    n_seasonal = int((m_df['best_m'] > 0).sum())
    n_non_seasonal = int((m_df['best_m'] == 0).sum())
    n_fallback = int((m_df['best_m'] < 0).sum())

    print("\n" + "=" * 65)
    print("         HOLT-WINTERS (B2 BASELINE) MACRO-AVERAGE SUMMARY")
    print("=" * 65)
    print(f"Test MAE                 : {m_df['mae'].mean():.4f} +/- {m_df['mae'].std():.4f}")
    print(f"Test RMSE                : {m_df['rmse'].mean():.4f} +/- {m_df['rmse'].std():.4f}")
    print(f"SLA Breach Rate          : {m_df['sla_breach_rate'].mean():.2%} (Target <= {SLA_TARGET:.1%})")
    print(f"Overshoot Rate           : {m_df['overshoot_rate'].mean():.2%}")
    print(f"Over-provision Mean      : {m_df['over_prov_mean'].mean():.4f}")
    print(f"Under-provision Mean     : {m_df['under_prov_mean'].mean():.4f}")
    print(f"Clip Fraction            : {m_df['clip_fraction'].mean():.2%}")
    print(f"Selected Seasonality (m) : {m_df['best_m'].value_counts().to_dict()}")
    print(f"Model Type Breakdown     : Seasonal HW: {n_seasonal}/{n_total} ({n_seasonal/n_total:.1%}) | "
          f"Non-Seasonal Holt: {n_non_seasonal}/{n_total} ({n_non_seasonal/n_total:.1%}) | "
          f"Fallback: {n_fallback}/{n_total} ({n_fallback/n_total:.1%})")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Holt-Winters forecasting and B2 evaluation")
    parser.add_argument("--series_dir", type=str, default=SERIES_DIR)
    parser.add_argument("--selected_json", type=str, default=SELECTED_MACHINES_PATH)
    parser.add_argument("--split_info", type=str, default=SPLIT_INFO_PATH)
    parser.add_argument("--output_dir", type=str, default=MODELS_DIR)
    args = parser.parse_args()

    run_holt_winters(args.series_dir, args.selected_json, args.split_info, args.output_dir)


if __name__ == "__main__":
    main()
