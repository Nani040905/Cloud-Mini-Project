"""
scripts/evaluate.py

Step 13: Comprehensive Evaluation and Unified Results Consolidation.
Aggregates and formats test metrics across all 10 experimental configurations:
- B0: Naive Persistence + Fixed 10% relative buffer
- B0b: Seasonal Naive + Fixed 10% relative buffer
- B1_Base: Global GRU (7 features) + Fixed 10% relative buffer
- B1_Extended: Global GRU (15 features) + Fixed 10% relative buffer
- B2: Holt-Winters (additive) + Fixed 10% relative buffer
- B3: Forecast Fusion + Fixed 10% relative buffer
- B4a: Forecast Fusion + Static U1 buffer
- B4b: Forecast Fusion + Static U2 buffer
- B4c: Forecast Fusion + Static U3 buffer
- Proposed: Forecast Fusion + Adaptive U* buffer
Produces results/final_results.csv.
"""

import os
import argparse
import numpy as np
import pandas as pd

from utils import RESULTS_DIR, MODELS_DIR


def consolidate_results(results_dir: str, models_dir: str):
    """Aggregates all configuration metrics into one master benchmark dataframe."""
    dfs = []

    # 1. Baselines B0 & B0b
    p_base = os.path.join(results_dir, "baseline_results.csv")
    if os.path.exists(p_base):
        dfs.append(pd.read_csv(p_base))

    # 2. B1 Base Tier
    p_b1_base = os.path.join(models_dir, "gru_base_metrics.csv")
    if os.path.exists(p_b1_base):
        df_b1_b = pd.read_csv(p_b1_base)
        df_b1_b['config'] = 'B1_GRU_Base'
        dfs.append(df_b1_b)

    # 3. B1 Extended Tier
    p_b1_ext = os.path.join(models_dir, "gru_extended_metrics.csv")
    if os.path.exists(p_b1_ext):
        df_b1_e = pd.read_csv(p_b1_ext)
        df_b1_e['config'] = 'B1_GRU_Extended'
        dfs.append(df_b1_e)

    # 4. B2 Holt-Winters
    p_b2 = os.path.join(models_dir, "hw_metrics.csv")
    if os.path.exists(p_b2):
        df_b2 = pd.read_csv(p_b2)
        df_b2['config'] = 'B2_HoltWinters'
        dfs.append(df_b2)

    # 5. B3 Forecast Fusion
    p_b3 = os.path.join(models_dir, "fusion_metrics.csv")
    if os.path.exists(p_b3):
        df_b3 = pd.read_csv(p_b3)
        df_b3['config'] = 'B3_Fusion'
        dfs.append(df_b3)

    # 6. B4a-c and Proposed Adaptive U*
    p_alloc = os.path.join(results_dir, "allocation_results.csv")
    if os.path.exists(p_alloc):
        dfs.append(pd.read_csv(p_alloc))

    if not dfs:
        print("[Error] No evaluation result files found to consolidate.")
        return

    all_data = pd.concat(dfs, ignore_index=True)

    summary_rows = []
    for cfg, grp in all_data.groupby('config'):
        n_machines = int((grp['valid_count'] > 0).sum() if 'valid_count' in grp else len(grp))
        mae_str = f"{grp['mae'].mean():.4f} +/- {grp['mae'].std():.4f}" if 'mae' in grp else "N/A"
        rmse_str = f"{grp['rmse'].mean():.4f} +/- {grp['rmse'].std():.4f}" if 'rmse' in grp else "N/A"

        summary_rows.append({
            'Configuration': cfg,
            'Machines': n_machines,
            'MAE': mae_str,
            'RMSE': rmse_str,
            'SLA Breach Rate': f"{grp['sla_breach_rate'].mean():.2%}",
            'Overshoot Rate': f"{grp['overshoot_rate'].mean():.2%}",
            'Over-Prov Mean': f"{grp['over_prov_mean'].mean():.4f}",
            'Under-Prov Mean': f"{grp['under_prov_mean'].mean():.4f}",
            'Clip Fraction': f"{grp['clip_fraction'].mean():.2%}"
        })

    summary_df = pd.DataFrame(summary_rows)

    os.makedirs(results_dir, exist_ok=True)
    out_csv = os.path.join(results_dir, "final_results.csv")
    summary_df.to_csv(out_csv, index=False)

    print("\n" + "=" * 100)
    print("                      COMPREHENSIVE BENCHMARK EVALUATION TABLE")
    print("=" * 100)
    print(summary_df.to_string(index=False))
    print("=" * 100)
    print(f"\nFinal consolidated benchmark results saved to: {out_csv}")


def main():
    parser = argparse.ArgumentParser(description="Consolidate and evaluate all benchmark configurations")
    parser.add_argument("--results_dir", type=str, default=RESULTS_DIR)
    parser.add_argument("--models_dir", type=str, default=MODELS_DIR)
    args = parser.parse_args()

    consolidate_results(args.results_dir, args.models_dir)


if __name__ == "__main__":
    main()
