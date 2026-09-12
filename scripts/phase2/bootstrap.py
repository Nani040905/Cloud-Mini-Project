"""
scripts/phase2/bootstrap.py

Phase 2.2: Statistical Uncertainty of Breach Rate and Paired Policy Differences.
Implements:
1. Machine-level cluster bootstrap (preserves full temporal autocorrelations):
   - Bootstrap unit: Machine (100 benchmark machines sampled with replacement)
   - Fixed reproducible seed (seed = 42)
   - B = 2,000 bootstrap replicates
2. Confidence Interval Calculations:
   - Macro Mean Breach Rate: Point estimate, 2.5th and 97.5th percentiles (95% CI)
   - Pooled Breach Rate: Point estimate, 2.5th and 97.5th percentiles (95% CI)
3. Paired Per-Machine Difference Analysis:
   - delta_i = breach_rate_Adaptive_i - breach_rate_U2_i
   - Machine-level paired difference bootstrap
   - Mean delta, median delta, 95% bootstrap CI
   - Two-tailed empirical bootstrap p-value
4. Outputs:
   - results/phase2/breach_confidence_intervals.csv
   - results/phase2/paired_policy_difference.csv

Strict Invariant:
- Machine is the bootstrap unit, NOT individual 5-minute bins.
- No claim of statistical significance unless supported by the 95% bootstrap CI.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.utils import RESULTS_DIR, FIGURES_DIR

PHASE2_RESULTS_DIR = os.path.join(RESULTS_DIR, "phase2")
PHASE2_FIGURES_DIR = os.path.join(FIGURES_DIR, "phase2")
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 42


def run_machine_bootstrap_ci(df_policy: pd.DataFrame,
                            n_replicates: int = BOOTSTRAP_REPLICATES,
                            seed: int = BOOTSTRAP_SEED) -> dict:
    """
    Computes 95% machine-level bootstrap confidence intervals for a single policy.

    Args:
        df_policy: DataFrame containing ['machine_id', 'breach_bins', 'total_test_bins', 'breach_rate']
        n_replicates: Number of bootstrap iterations (default: 2,000)
        seed: Random seed for reproducibility

    Returns:
        Dictionary of point estimates, standard errors, and 95% CI bounds.
    """
    rates = df_policy['breach_rate'].values
    breach_counts = df_policy['breach_bins'].values
    total_counts = df_policy['total_test_bins'].values
    n_machines = len(rates)

    # Point estimates
    point_macro = float(np.mean(rates))
    point_pooled = float(np.sum(breach_counts) / np.sum(total_counts))

    # Bootstrap sampling (sampling machines with replacement)
    rng = np.random.default_rng(seed)
    boot_macro = np.empty(n_replicates, dtype=np.float64)
    boot_pooled = np.empty(n_replicates, dtype=np.float64)

    for b in range(n_replicates):
        idx = rng.choice(n_machines, size=n_machines, replace=True)
        boot_macro[b] = np.mean(rates[idx])
        boot_pooled[b] = np.sum(breach_counts[idx]) / np.sum(total_counts[idx])

    ci_macro_lower, ci_macro_upper = np.percentile(boot_macro, [2.5, 97.5])
    ci_pooled_lower, ci_pooled_upper = np.percentile(boot_pooled, [2.5, 97.5])

    se_macro = float(np.std(boot_macro, ddof=1))
    se_pooled = float(np.std(boot_pooled, ddof=1))

    return {
        'point_estimate_macro': point_macro,
        'ci_2.5_macro': float(ci_macro_lower),
        'ci_97.5_macro': float(ci_macro_upper),
        'std_error_macro': se_macro,
        'point_estimate_pooled': point_pooled,
        'ci_2.5_pooled': float(ci_pooled_lower),
        'ci_97.5_pooled': float(ci_pooled_upper),
        'std_error_pooled': se_pooled
    }


def run_paired_difference_bootstrap(df_policy_a: pd.DataFrame,
                                    df_policy_b: pd.DataFrame,
                                    name_a: str,
                                    name_b: str,
                                    n_replicates: int = BOOTSTRAP_REPLICATES,
                                    seed: int = BOOTSTRAP_SEED) -> dict:
    """
    Computes paired per-machine difference bootstrap:
        delta_i = breach_rate_A_i - breach_rate_B_i
    Sampling unit: Machine.

    Returns:
        Dictionary of paired difference statistics and 95% bootstrap CI.
    """
    merged = pd.merge(
        df_policy_a[['machine_id', 'breach_rate']].rename(columns={'breach_rate': 'rate_a'}),
        df_policy_b[['machine_id', 'breach_rate']].rename(columns={'breach_rate': 'rate_b'}),
        on='machine_id'
    )
    assert len(merged) == len(df_policy_a), f"Machine IDs must align cleanly between {name_a} and {name_b}"

    deltas = (merged['rate_a'] - merged['rate_b']).values
    n_machines = len(deltas)

    point_mean = float(np.mean(deltas))
    point_median = float(np.median(deltas))
    point_std = float(np.std(deltas, ddof=1))

    machines_a_better = int(np.sum(deltas < -1e-9))   # A has lower breach rate
    machines_tied = int(np.sum(np.abs(deltas) <= 1e-9))
    machines_b_better = int(np.sum(deltas > 1e-9))   # B has lower breach rate

    # Bootstrap sampling of paired differences
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_replicates, dtype=np.float64)
    boot_medians = np.empty(n_replicates, dtype=np.float64)

    for b in range(n_replicates):
        idx = rng.choice(n_machines, size=n_machines, replace=True)
        boot_means[b] = np.mean(deltas[idx])
        boot_medians[b] = np.median(deltas[idx])

    ci_lower, ci_upper = np.percentile(boot_means, [2.5, 97.5])
    ci_med_lower, ci_med_upper = np.percentile(boot_medians, [2.5, 97.5])

    # Empirical two-tailed bootstrap p-value
    # Proportion of bootstrap replicates with opposite sign to the point estimate
    if point_mean >= 0:
        p_val = 2.0 * float(np.mean(boot_means <= 0))
    else:
        p_val = 2.0 * float(np.mean(boot_means >= 0))
    p_val = min(1.0, max(0.0, p_val))

    # Statistically significant at 95% level iff 95% CI excludes 0
    is_significant = not (ci_lower <= 0.0 <= ci_upper)

    return {
        'comparison': f"{name_a}_vs_{name_b}",
        'policy_A': name_a,
        'policy_B': name_b,
        'mean_delta': point_mean,
        'median_delta': point_median,
        'std_delta': point_std,
        'bootstrap_ci_2.5': float(ci_lower),
        'bootstrap_ci_97.5': float(ci_upper),
        'bootstrap_median_ci_2.5': float(ci_med_lower),
        'bootstrap_median_ci_97.5': float(ci_med_upper),
        'p_value_bootstrap': float(p_val),
        'statistically_significant': is_significant,
        'machines_A_better': machines_a_better,
        'machines_tied': machines_tied,
        'machines_B_better': machines_b_better
    }


def run_statistical_uncertainty(decomp_csv_path: str = None,
                                output_dir: str = PHASE2_RESULTS_DIR,
                                figures_dir: str = PHASE2_FIGURES_DIR,
                                n_replicates: int = BOOTSTRAP_REPLICATES,
                                seed: int = BOOTSTRAP_SEED):
    """
    Executes Phase 2.2:
    1. Computes 95% machine-level bootstrap CIs for all policies.
    2. Computes paired per-machine difference bootstrap between Adaptive U* and Static U2
       (as well as other static policies).
    3. Generates CSV summaries and diagnostic figure.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    csv_path = decomp_csv_path or os.path.join(output_dir, "reliability_decomposition.csv")
    print(f"[Phase 2.2] Loading reliability decomposition from: {csv_path}")
    df = pd.read_csv(csv_path)

    # 1. Policy Bootstrap CIs
    ci_results = []
    policies = [
        'Static_U2',
        'Adaptive_U_star',
        'Static_U1',
        'Static_U3',
        'B3_Fusion_Fixed10',
        'B1_GRU_Fixed10',
        'B2_HoltWinters_Fixed10',
        'B0_Naive_Fixed10'
    ]

    print(f"[Phase 2.2] Computing machine-level bootstrap ({n_replicates} replicates, seed={seed})...")
    for pol in policies:
        sub = df[df['policy'] == pol].copy()
        if len(sub) == 0:
            continue
        res = run_machine_bootstrap_ci(sub, n_replicates=n_replicates, seed=seed)
        res['policy'] = pol
        ci_results.append(res)

    ci_df = pd.DataFrame(ci_results)
    # Reorder columns cleanly
    col_order = [
        'policy', 'point_estimate_macro', 'ci_2.5_macro', 'ci_97.5_macro', 'std_error_macro',
        'point_estimate_pooled', 'ci_2.5_pooled', 'ci_97.5_pooled', 'std_error_pooled'
    ]
    ci_df = ci_df[[c for c in col_order if c in ci_df.columns]]
    ci_csv = os.path.join(output_dir, "breach_confidence_intervals.csv")
    ci_df.to_csv(ci_csv, index=False)
    print(f"[Phase 2.2] Saved breach confidence intervals to: {ci_csv}")

    # 2. Paired Differences Analysis
    # Primary: Adaptive U* vs Static U2
    # Secondary: Adaptive U* vs Static U1, Static U3, and Static U2 vs Static U1
    pairs = [
        ('Adaptive_U_star', 'Static_U2'),
        ('Adaptive_U_star', 'Static_U1'),
        ('Adaptive_U_star', 'Static_U3'),
        ('Static_U2', 'Static_U1')
    ]

    paired_results = []
    for pol_a, pol_b in pairs:
        sub_a = df[df['policy'] == pol_a].copy()
        sub_b = df[df['policy'] == pol_b].copy()
        res = run_paired_difference_bootstrap(sub_a, sub_b, pol_a, pol_b,
                                              n_replicates=n_replicates, seed=seed)
        paired_results.append(res)

    paired_df = pd.DataFrame(paired_results)
    paired_csv = os.path.join(output_dir, "paired_policy_difference.csv")
    paired_df.to_csv(paired_csv, index=False)
    print(f"[Phase 2.2] Saved paired policy differences to: {paired_csv}")

    # 3. Generate Diagnostic Figure
    generate_bootstrap_figure(ci_df, df, output_dir, figures_dir)

    return ci_df, paired_df


def generate_bootstrap_figure(ci_df: pd.DataFrame, decomp_df: pd.DataFrame,
                              output_dir: str, figures_dir: str):
    """Generates visual diagnostic plot for breach confidence intervals and paired differences."""
    plt.rcParams.update({
        'font.sans-serif': 'DejaVu Sans',
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.edgecolor': '#cccccc',
        'axes.linewidth': 0.8
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A: 95% Confidence Intervals for Macro Breach Rate
    target_policies = ['Static_U2', 'Adaptive_U_star', 'Static_U1', 'Static_U3']
    sub_ci = ci_df[ci_df['policy'].isin(target_policies)].copy()

    y_pos = np.arange(len(target_policies))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    points = sub_ci['point_estimate_macro'].values * 100.0
    err_low = (sub_ci['point_estimate_macro'].values - sub_ci['ci_2.5_macro'].values) * 100.0
    err_high = (sub_ci['ci_97.5_macro'].values - sub_ci['point_estimate_macro'].values) * 100.0

    for i, (p, el, eh, c) in enumerate(zip(points, err_low, err_high, colors)):
        ax1.errorbar(p, i, xerr=[[el], [eh]], fmt='o', color=c,
                     ecolor=c, elinewidth=2.5, capsize=5, capthick=1.5, markersize=8)

    # Red dashed SLA line
    ax1.axvline(1.0, color='red', linestyle='--', linewidth=1.5, label='SLA Target (1.0%)')

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([p.replace('_', ' ') for p in sub_ci['policy']], fontweight='bold')
    ax1.set_xlabel('Empirical Macro Test SLA Breach Rate (%)', fontweight='bold')
    ax1.set_title('A. 95% Machine-Level Bootstrap CIs (2,000 Replicates)', fontweight='bold', pad=10)
    ax1.grid(axis='x', linestyle='--', alpha=0.4)
    ax1.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9)

    # Annotate values
    for i, (p, el, eh) in enumerate(zip(points, err_low, err_high)):
        ax1.text(p, i + 0.18, f"{p:.2f}% [{p - el:.2f}%, {p + eh:.2f}%]",
                 ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#222222')

    # Panel B: Paired Differences (U* - U2) Distribution
    p_u2 = decomp_df[decomp_df['policy'] == 'Static_U2'].set_index('machine_id')['breach_rate']
    p_ustar = decomp_df[decomp_df['policy'] == 'Adaptive_U_star'].set_index('machine_id')['breach_rate']
    deltas = (p_ustar - p_u2) * 100.0  # In percent

    ax2.hist(deltas, bins=25, color='#ff7f0e', alpha=0.7, edgecolor='black', linewidth=0.6,
             label='Per-Machine Δ = Breach(U*) - Breach(U2)')
    ax2.axvline(0.0, color='black', linestyle='-', linewidth=1.5, label='Zero Difference Line')
    ax2.axvline(deltas.mean(), color='blue', linestyle='--', linewidth=2.0,
                label=f'Mean Δ = {deltas.mean():+.2f}%')
    ax2.axvline(deltas.median(), color='green', linestyle=':', linewidth=2.0,
                label=f'Median Δ = {deltas.median():+.2f}%')

    ax2.set_xlabel(r'Paired Difference $\Delta_i$ (%) across 100 Machines', fontweight='bold')
    ax2.set_ylabel('Number of Benchmark Machines', fontweight='bold')
    ax2.set_title(r'B. Paired Per-Machine Difference ($\Delta = U^* - U_2$)', fontweight='bold', pad=10)
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=9.5)

    plt.tight_layout()
    fig_path = os.path.join(figures_dir, "breach_confidence_intervals.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[Phase 2.2] Saved figure to: {fig_path}")


if __name__ == '__main__':
    run_statistical_uncertainty()
