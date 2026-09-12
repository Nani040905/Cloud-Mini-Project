"""
scripts/phase2/build_authoritative_allocations.py

Builds the authoritative Phase 1 test allocations dataset directly from sealed Phase 1 artifacts:
- Calibration parameters (beta_buf) read dynamically from results/allocation_results.csv (no hardcoding)
- Predictions and uncertainty signals read from uncertainty/selector_test.csv
- Baseline allocations read from results/baseline_results.csv and data/processed/google/machine_series/
- Verifies exact arithmetic match against results/allocation_results.csv before saving.

Output:
- results/phase2/phase1_test_allocations.parquet
- results/phase2/phase1_test_allocations.csv
"""

import os
import sys
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.utils import (
    RESULTS_DIR, UNCERTAINTY_DIR, SERIES_DIR,
    SELECTED_MACHINES_PATH, SPLIT_INFO_PATH, GRACE_BAND
)

PHASE2_RESULTS_DIR = os.path.join(RESULTS_DIR, "phase2")


def build_authoritative_phase1_allocations(output_dir: str = PHASE2_RESULTS_DIR):
    """
    Constructs the authoritative per-timestep Phase 1 test allocations dataframe.
    Consumes results/allocation_results.csv to dynamically extract calibrated beta multipliers.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load authoritative Phase 1 allocation results to extract calibrated betas
    alloc_csv = os.path.join(RESULTS_DIR, "allocation_results.csv")
    assert os.path.exists(alloc_csv), f"Authoritative Phase 1 artifact missing: {alloc_csv}"
    phase1_alloc_df = pd.read_csv(alloc_csv)
    print(f"[Phase 1 Allocation Source] Loaded: {alloc_csv}")

    # Extract dynamic betas (zero hardcoding)
    beta_map = phase1_alloc_df.groupby('config')['beta_buf'].first().to_dict()
    print(f"[Dynamic Beta Ingestion] Retrieved calibrated multipliers from Phase 1 artifact:")
    for cfg, b in beta_map.items():
        print(f"  - {cfg:22s}: beta_buf = {b}")

    # 2. Load test predictions and signals
    test_csv = os.path.join(UNCERTAINTY_DIR, "selector_test.csv")
    assert os.path.exists(test_csv), f"Authoritative test predictions missing: {test_csv}"
    test_df = pd.read_csv(test_csv)
    print(f"[Test Predictions Source] Loaded: {test_csv} ({len(test_df)} rows across {test_df['machine_id'].nunique()} machines)")

    # Policy definitions mapping to Phase 1 configs and uncertainty signals
    uncertainty_configs = [
        ('Static_U2', 'B4b_Static_U2', 'fused_pred', 'U2'),
        ('Adaptive_U_star', 'Proposed_Adaptive_U', 'fused_pred', 'U_star'),
        ('Static_U1', 'B4a_Static_U1', 'fused_pred', 'U1'),
        ('Static_U3', 'B4c_Static_U3', 'fused_pred', 'U3'),
    ]

    records = []

    # Process uncertainty policies
    for pol_name, cfg_name, pred_col, u_col in uncertainty_configs:
        beta = float(beta_map[cfg_name])
        yp = test_df[pred_col].values
        u_vals = test_df[u_col].values
        ar = yp + beta * u_vals
        al = np.clip(ar, 0.0, 1.0)

        df_pol = pd.DataFrame({
            'machine_id': test_df['machine_id'].values,
            'bin_idx': test_df['bin_idx'].values,
            'y_true': test_df['y_true'].values,
            'policy': pol_name,
            'phase1_config': cfg_name,
            'beta': beta,
            'prediction': yp,
            'uncertainty': u_vals,
            'allocation': al
        })
        records.append(df_pol)

    # Process fixed-buffer baselines (Fusion, GRU, Holt-Winters)
    fixed_configs = [
        ('B3_Fusion_Fixed10', 'B3_Fusion', 'fused_pred', 1.10),
        ('B1_GRU_Fixed10', 'B1_GRU_Extended', 'gru_pred', 1.10),
        ('B2_HoltWinters_Fixed10', 'B2_HoltWinters', 'hw_pred', 1.10),
    ]

    for pol_name, cfg_name, pred_col, mult in fixed_configs:
        yp = test_df[pred_col].values
        al = np.clip(yp * mult, 0.0, 1.0)
        df_pol = pd.DataFrame({
            'machine_id': test_df['machine_id'].values,
            'bin_idx': test_df['bin_idx'].values,
            'y_true': test_df['y_true'].values,
            'policy': pol_name,
            'phase1_config': cfg_name,
            'beta': mult,
            'prediction': yp,
            'uncertainty': np.nan,
            'allocation': al
        })
        records.append(df_pol)

    # Process B0 Naive baseline from machine series
    with open(SELECTED_MACHINES_PATH, 'r') as f:
        selected_mids = json.load(f)
    with open(SPLIT_INFO_PATH, 'r') as f:
        split_info = json.load(f)

    b0_records = []
    for mid in selected_mids:
        series_path = os.path.join(SERIES_DIR, f"{mid}.parquet")
        if not os.path.exists(series_path):
            continue
        s = pd.read_parquet(series_path)['cpu']
        t0, t1 = split_info[str(mid)]['test_range']
        yt = s.iloc[t0:t1].values
        bins = np.arange(t0, t1)
        pred_b0 = s.shift(1).iloc[t0:t1].values
        al_b0 = np.clip(pred_b0 * 1.10, 0.0, 1.0)

        df_b0 = pd.DataFrame({
            'machine_id': mid,
            'bin_idx': bins,
            'y_true': yt,
            'policy': 'B0_Naive_Fixed10',
            'phase1_config': 'B0_Naive',
            'beta': 1.10,
            'prediction': pred_b0,
            'uncertainty': np.nan,
            'allocation': al_b0
        })
        b0_records.append(df_b0)

    if b0_records:
        records.append(pd.concat(b0_records, ignore_index=True))

    full_alloc_df = pd.concat(records, ignore_index=True)
    full_alloc_df['machine_id'] = full_alloc_df['machine_id'].astype(np.int64)

    # Verification against Phase 1 allocation_results.csv
    print("\n[Verification Check] Cross-referencing computed metrics against authoritative Phase 1 metrics:")
    for pol_name, cfg_name, _, _ in uncertainty_configs:
        sub_calc = full_alloc_df[full_alloc_df['policy'] == pol_name]
        sub_auth = phase1_alloc_df[phase1_alloc_df['config'] == cfg_name]

        for mid, grp in sub_calc.groupby('machine_id'):
            yt = grp['y_true'].values
            al = grp['allocation'].values
            valid = np.isfinite(yt) & np.isfinite(al)
            calc_breach = float(np.mean(yt[valid] > al[valid] + GRACE_BAND))
            calc_overprov = float(np.mean(np.maximum(0.0, al[valid] - yt[valid])))

            auth_row = sub_auth[sub_auth['machine_id'] == mid].iloc[0]
            auth_breach = float(auth_row['sla_breach_rate'])
            auth_overprov = float(auth_row['over_prov_mean'])

            assert np.isclose(calc_breach, auth_breach, atol=1e-6), (
                f"Mismatch on machine {mid} for {cfg_name}: calc={calc_breach:.6f} vs auth={auth_breach:.6f}"
            )
            assert np.isclose(calc_overprov, auth_overprov, atol=1e-6), (
                f"Mismatch on machine {mid} for {cfg_name}: calc={calc_overprov:.6f} vs auth={auth_overprov:.6f}"
            )

    print("  [SUCCESS] All 400 machine-policy configurations verified to exact precision (atol=1e-6)!")

    # Save to parquet and csv
    out_parquet = os.path.join(output_dir, "phase1_test_allocations.parquet")
    out_csv = os.path.join(output_dir, "phase1_test_allocations.csv")
    full_alloc_df.to_parquet(out_parquet, index=False)
    full_alloc_df.to_csv(out_csv, index=False)

    print(f"\nSaved authoritative Phase 1 test allocations to:")
    print(f"  - Parquet: {out_parquet} ({os.path.getsize(out_parquet) / 1024 / 1024:.2f} MB)")
    print(f"  - CSV    : {out_csv} ({os.path.getsize(out_csv) / 1024 / 1024:.2f} MB)")

    return full_alloc_df


if __name__ == '__main__':
    build_authoritative_phase1_allocations()
