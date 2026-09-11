"""
scripts/robustness.py

Machine-Subset Sensitivity Analysis (8-run parameter matrix).

IMPORTANT: This is NOT a full retraining robustness experiment.
All 8 runs reuse the same pre-trained GRU/HW models and the same
trained beta* from the main pipeline run. Only the machine subset
and the subsampling random seed change in each run.

This measures how sensitive the reported breach/overprov metrics are
to which subset of machines is evaluated (machine-subset sensitivity),
**not** to different training seeds or different model initializations.
True independent-seed retraining is planned for Phase 2.

Matrix:
  - 4 Subsample Seeds: [42, 123, 456, 789]
  - 2 Machine Counts:  [50, 100]
Outputs: results/robustness_results.csv
"""

import os
import json
import argparse
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
UNCERTAINTY_LOG = os.path.join(PROJECT_ROOT, "uncertainty", "selector_log.csv")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

SEEDS = [42, 123, 456, 789]
MACHINE_COUNTS = [50, 100]


def run_robustness_sweep(selector_log_path: str = UNCERTAINTY_LOG, output_csv: str = None,
                         alloc_results_path: str = None):
    if not os.path.exists(selector_log_path):
        raise FileNotFoundError(f"Selector log not found: {selector_log_path}. Run pipeline first.")

    print("=" * 80)
    print("  MACHINE-SUBSET SENSITIVITY ANALYSIS (8-RUN SUBSAMPLE MATRIX)")
    print("  NOTE: Reuses trained models/beta; only machine subset varies.")
    print("=" * 80)
    df = pd.read_csv(selector_log_path)
    test_df = df[df['split'] == 'test'].copy()

    all_machines = test_df['machine_id'].unique()
    n_available = len(all_machines)
    print(f"Total available machines in test split: {n_available}")

    if output_csv is None:
        output_csv = os.path.join(RESULTS_DIR, "robustness_results.csv")

    alloc_csv = alloc_results_path or os.path.join(RESULTS_DIR, "allocation_results.csv")
    if not os.path.exists(alloc_csv):
        raise RuntimeError(
            f"Cannot run sensitivity analysis: {alloc_csv} does not exist. "
            "Please run allocation.py first to calibrate beta*."
        )

    a_df = pd.read_csv(alloc_csv)
    sub_a = a_df[a_df['config'] == 'Proposed_Adaptive_U']
    if len(sub_a) == 0 or 'beta_buf' not in sub_a.columns:
        raise RuntimeError(
            f"Cannot run sensitivity analysis: 'Proposed_Adaptive_U' config or 'beta_buf' column "
            f"not found in {alloc_csv}."
        )

    beta_star = float(sub_a['beta_buf'].iloc[0])
    print(f"Evaluated with Calibrated Adaptive Multiplier: beta* = {beta_star:.2f}")

    results = []
    # Evaluate each cell in the 4x4 matrix
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        shuffled_machines = rng.permutation(all_machines)

        for m_count in MACHINE_COUNTS:
            eff_count = min(m_count, n_available)
            selected = shuffled_machines[:eff_count]
            sub = test_df[test_df['machine_id'].isin(selected)]

            if len(sub) == 0:
                continue

            # Actual CPU, fused prediction, and uncertainty signal
            y_actual = sub['y_true'].values
            y_fused  = sub['fused_pred'].values
            u_star   = sub['U_star'].values

            # Fixed 10% buffer
            alloc_fixed = np.clip(y_fused * 1.10, 0.0, 1.0)
            breach_fixed = np.mean(y_actual > alloc_fixed + 0.005)
            ovp_fixed = np.mean(np.maximum(0.0, alloc_fixed - y_actual))

            # Adaptive U* buffer (dynamically calibrated beta*)
            alloc_adaptive = np.clip(y_fused + beta_star * u_star, 0.0, 1.0)
            breach_adaptive = np.mean(y_actual > alloc_adaptive + 0.005)
            ovp_adaptive = np.mean(np.maximum(0.0, alloc_adaptive - y_actual))

            sla_reduction = (breach_fixed - breach_adaptive) / max(1e-6, breach_fixed)

            results.append({
                'Subsample Seed': seed,      # NOT a training seed; selects machine subset only
                'Machine Count': eff_count,
                'Fixed 10% Breach': round(float(breach_fixed), 4),
                'Adaptive U* Breach': round(float(breach_adaptive), 4),
                'SLA Reduction': round(float(sla_reduction), 4),
                'Fixed Over-Prov': round(float(ovp_fixed), 4),
                'Adaptive Over-Prov': round(float(ovp_adaptive), 4)
            })

    res_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    res_df.to_csv(output_csv, index=False)

    print("\n" + "=" * 85)
    print("            MACHINE-SUBSET SENSITIVITY ANALYSIS — RESULTS TABLE")
    print("=" * 85)
    print(res_df.to_string(index=False))
    print("=" * 85)
    print(f"\nMean SLA Breach Rate across all 8 runs:")
    print(f"  Fixed 10% Buffer  : {res_df['Fixed 10% Breach'].mean():.2%}")
    print(f"  Adaptive U* Buffer: {res_df['Adaptive U* Breach'].mean():.2%}")
    print(f"  Average Reduction : {res_df['SLA Reduction'].mean():.2%}")
    print(f"Robustness matrix saved to: {output_csv}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robustness validation sweep")
    parser.add_argument("--selector_log", type=str, default=UNCERTAINTY_LOG)
    parser.add_argument("--output_csv", type=str, default=os.path.join(RESULTS_DIR, "robustness_results.csv"))
    args = parser.parse_args()
    run_robustness_sweep(args.selector_log, args.output_csv)
