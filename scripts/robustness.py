"""
scripts/robustness.py

Priority 8: Multi-Dimensional Robustness Validation Matrix.
Executes focused 16-run sweep:
  - 4 Random Seeds: [42, 123, 456, 789]
  - 4 Machine Counts: [50, 100, 200, 290]
Validates that Proposed Adaptive U* buffer consistently outperforms
fixed-buffer baselines across random subsampling and cluster scale.
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
MACHINE_COUNTS = [50, 100, 200, 290]


def run_robustness_sweep(selector_log_path: str = UNCERTAINTY_LOG, output_csv: str = None):
    if not os.path.exists(selector_log_path):
        raise FileNotFoundError(f"Selector log not found: {selector_log_path}. Run pipeline first.")

    print("=" * 80)
    print("      ROBUSTNESS VALIDATION SWEEP (16-RUN PARAMETER MATRIX)")
    print("=" * 80)
    df = pd.read_csv(selector_log_path)
    test_df = df[df['split'] == 'test'].copy()

    all_machines = test_df['machine_id'].unique()
    n_available = len(all_machines)
    print(f"Total available machines in test split: {n_available}")

    if output_csv is None:
        output_csv = os.path.join(RESULTS_DIR, "robustness_results.csv")

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

            # Adaptive U* buffer (calibrated beta = 1.8)
            alloc_adaptive = np.clip(y_fused + 1.8 * u_star, 0.0, 1.0)
            breach_adaptive = np.mean(y_actual > alloc_adaptive + 0.005)
            ovp_adaptive = np.mean(np.maximum(0.0, alloc_adaptive - y_actual))

            sla_reduction = (breach_fixed - breach_adaptive) / max(1e-6, breach_fixed)

            results.append({
                'Seed': seed,
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
    print("                    ROBUSTNESS SWEEP RESULTS TABLE")
    print("=" * 85)
    print(res_df.to_string(index=False))
    print("=" * 85)
    print(f"\nMean SLA Breach Rate across all 16 runs:")
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
