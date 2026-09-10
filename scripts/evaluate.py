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

    CONFIG_ORDER = [
        'B0_Naive', 'B0b_SeasonalNaive_m48', 'B0c_MovingAverage_w6',
        'B1_GRU_Base', 'B1_GRU_Extended', 'B2_HoltWinters', 'B3_Fusion',
        'B4a_Static_U1', 'B4b_Static_U2', 'B4c_Static_U3', 'Proposed_Adaptive_U'
    ]

    summary_rows = []
    detailed_rows = []

    # Sort configurations by defined order
    existing_configs = sorted(all_data['config'].unique(),
                              key=lambda c: CONFIG_ORDER.index(c) if c in CONFIG_ORDER else 999)

    for cfg in existing_configs:
        grp = all_data[all_data['config'] == cfg]
        valid_grp = grp[grp['valid_count'] > 0] if 'valid_count' in grp.columns else grp
        n_machines = len(valid_grp)
        if n_machines == 0:
            continue

        # Helper to compute comprehensive distribution stats
        def calc_dist_stats(series):
            clean = series.dropna()
            if len(clean) == 0:
                return {'mean': np.nan, 'std': np.nan, 'median': np.nan,
                        'ci95': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan,
                        'min': np.nan, 'max': np.nan}
            m = clean.mean()
            s = clean.std() if len(clean) > 1 else 0.0
            med = clean.median()
            ci = 1.96 * s / np.sqrt(len(clean)) if len(clean) > 1 else 0.0
            return {
                'mean': m, 'std': s, 'median': med,
                'ci95': ci, 'ci_lower': m - ci, 'ci_upper': m + ci,
                'min': clean.min(), 'max': clean.max()
            }

        stats_dict = {}
        for col in ['mae', 'rmse', 'sla_breach_rate', 'overshoot_rate', 'over_prov_mean', 'under_prov_mean', 'clip_fraction']:
            if col in valid_grp.columns:
                col_stats = calc_dist_stats(valid_grp[col])
                stats_dict[col] = col_stats
                detailed_rows.append({
                    'Configuration': cfg,
                    'Metric': col,
                    'N': n_machines,
                    'Mean': round(col_stats['mean'], 6),
                    'Median': round(col_stats['median'], 6),
                    'Std': round(col_stats['std'], 6),
                    '95% CI Lower': round(col_stats['ci_lower'], 6),
                    '95% CI Upper': round(col_stats['ci_upper'], 6),
                    'Min': round(col_stats['min'], 6),
                    'Max': round(col_stats['max'], 6)
                })

        # Format concise master summary row
        sla = stats_dict.get('sla_breach_rate', {})
        ovp = stats_dict.get('over_prov_mean', {})
        und = stats_dict.get('under_prov_mean', {})
        mae_s = stats_dict.get('mae', {})
        rmse_s = stats_dict.get('rmse', {})

        mae_str = f"{mae_s.get('mean', 0.0):.4f} +/- {mae_s.get('ci95', 0.0):.4f}" if 'mae' in stats_dict else "N/A"
        rmse_str = f"{rmse_s.get('mean', 0.0):.4f} +/- {rmse_s.get('ci95', 0.0):.4f}" if 'rmse' in stats_dict else "N/A"

        summary_rows.append({
            'Configuration': cfg,
            'Machines': n_machines,
            'MAE (95% CI)': mae_str,
            'RMSE (95% CI)': rmse_str,
            'SLA Breach Rate': f"{sla.get('mean', 0.0):.2%} +/- {sla.get('ci95', 0.0):.2%}",
            'Overshoot Rate': f"{stats_dict.get('overshoot_rate', {}).get('mean', 0.0):.2%}",
            'Over-Prov Mean': f"{ovp.get('mean', 0.0):.4f} +/- {ovp.get('ci95', 0.0):.4f}",
            'Under-Prov Mean': f"{und.get('mean', 0.0):.4f} +/- {und.get('ci95', 0.0):.4f}",
            'Clip Fraction': f"{stats_dict.get('clip_fraction', {}).get('mean', 0.0):.2%}"
        })

    summary_df = pd.DataFrame(summary_rows)
    detailed_df = pd.DataFrame(detailed_rows)

    os.makedirs(results_dir, exist_ok=True)
    out_csv = os.path.join(results_dir, "final_results.csv")
    out_detailed_csv = os.path.join(results_dir, "final_results_detailed.csv")

    summary_df.to_csv(out_csv, index=False)
    detailed_df.to_csv(out_detailed_csv, index=False)

    print("\n" + "=" * 115)
    print("                      COMPREHENSIVE BENCHMARK EVALUATION TABLE (WITH 95% CI)")
    print("=" * 115)
    print(summary_df.to_string(index=False))
    print("=" * 115)
    print(f"\nSummary results saved to: {out_csv}")
    print(f"Detailed per-machine distribution metrics saved to: {out_detailed_csv}")


def main():
    parser = argparse.ArgumentParser(description="Consolidate and evaluate all benchmark configurations")
    parser.add_argument("--results_dir", type=str, default=RESULTS_DIR)
    parser.add_argument("--models_dir", type=str, default=MODELS_DIR)
    args = parser.parse_args()

    consolidate_results(args.results_dir, args.models_dir)


if __name__ == "__main__":
    main()
