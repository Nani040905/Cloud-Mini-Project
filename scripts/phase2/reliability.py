"""
scripts/phase2/reliability.py

Phase 2.1: SLA Reliability Decomposition and Incident Analysis.
Pure analysis layer over authoritative Phase 1 test allocations:
- Consumes results/phase2/phase1_test_allocations.parquet directly (no hardcoded betas)
- Evaluates per-machine reliability decomposition across all 13 required metrics
- Performs temporal incident run-length analysis (consecutive breaches for same machine)
- Evaluates conditional shortfall severity S_t = max(0, y_t - (A_t + 0.005))
- Generates publication-quality diagnostic figures
- Produces per-machine head-to-head comparison table between Adaptive U* and Static U2
  (sorted by largest U* disadvantage and largest U* advantage)
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

from scripts.utils import (
    GRACE_BAND, RESULTS_DIR, FIGURES_DIR
)
from scripts.phase2.build_authoritative_allocations import (
    build_authoritative_phase1_allocations
)

PHASE2_RESULTS_DIR = os.path.join(RESULTS_DIR, "phase2")
PHASE2_FIGURES_DIR = os.path.join(FIGURES_DIR, "phase2")


def compute_machine_incidents(bin_indices: np.ndarray, breach_mask: np.ndarray) -> list:
    """
    Identifies consecutive breach incidents for a single machine.
    An incident is an unbroken sequence of consecutive breach bins:
    y_t > A_t + GRACE_BAND where bin_indices[i] == bin_indices[i-1] + 1.

    Returns:
        List of incident durations in bins (e.g. [1, 3, 2]).
        Returns empty list [] if no breaches exist.
    """
    if len(bin_indices) == 0 or not np.any(breach_mask):
        return []

    durations = []
    current_duration = 0
    prev_bin = None

    for b_idx, is_breach in zip(bin_indices, breach_mask):
        if is_breach:
            if current_duration == 0:
                current_duration = 1
            else:
                if b_idx == prev_bin + 1:
                    current_duration += 1
                else:
                    durations.append(current_duration)
                    current_duration = 1
        else:
            if current_duration > 0:
                durations.append(current_duration)
                current_duration = 0
        prev_bin = b_idx

    if current_duration > 0:
        durations.append(current_duration)

    return durations


def compute_machine_reliability_metrics(y_true: np.ndarray, a_alloc: np.ndarray,
                                        bin_indices: np.ndarray, machine_id: int,
                                        config_name: str, grace_band: float = GRACE_BAND) -> tuple:
    """
    Computes all 13 required reliability metrics and incident durations for a single machine.

    Returns:
        (metrics_dict, incident_durations_list, shortfall_severities_array)
    """
    valid = np.isfinite(y_true) & np.isfinite(a_alloc)
    yt = y_true[valid]
    al = a_alloc[valid]
    bins = bin_indices[valid]
    n_bins = len(yt)

    if n_bins == 0:
        empty_metrics = {
            'policy': config_name,
            'machine_id': machine_id,
            'total_test_bins': 0,
            'breach_bins': 0,
            'breach_rate': 0.0,
            'overshoot_bins': 0,
            'overshoot_rate': 0.0,
            'grace_zone_bins': 0,
            'grace_zone_rate': 0.0,
            'mean_overprovisioning': 0.0,
            'mean_underprovisioning': 0.0,
            'total_underprovisioning': 0.0,
            'mean_shortfall_severity': 0.0,
            'p95_shortfall_severity': 0.0,
            'max_shortfall_severity': 0.0,
            'number_of_incidents': 0,
            'mean_incident_duration_bins': 0.0,
            'median_incident_duration_bins': 0.0,
            'p95_incident_duration_bins': 0.0,
            'max_incident_duration_bins': 0,
            'mean_incident_duration_minutes': 0.0,
            'p95_incident_duration_minutes': 0.0,
            'max_incident_duration_minutes': 0.0
        }
        return empty_metrics, [], np.array([])

    breach_mask = yt > (al + grace_band)
    breach_bins = int(np.sum(breach_mask))
    breach_rate = float(breach_bins / n_bins)

    overshoot_mask = yt > al
    overshoot_bins = int(np.sum(overshoot_mask))
    overshoot_rate = float(overshoot_bins / n_bins)

    grace_zone_mask = (yt > al) & (~breach_mask)
    grace_zone_bins = int(np.sum(grace_zone_mask))
    grace_zone_rate = float(grace_zone_bins / n_bins)

    over_prov = np.maximum(0.0, al - yt)
    under_prov = np.maximum(0.0, yt - al)
    mean_overprov = float(np.mean(over_prov))
    mean_underprov = float(np.mean(under_prov))
    total_underprov = float(np.sum(under_prov))

    # Shortfall severity S_t = max(0, y_t - (A_t + grace_band)) conditional on breach
    if breach_bins > 0:
        severities = yt[breach_mask] - (al[breach_mask] + grace_band)
        mean_sev = float(np.mean(severities))
        p95_sev = float(np.percentile(severities, 95))
        max_sev = float(np.max(severities))
    else:
        severities = np.array([])
        mean_sev = 0.0
        p95_sev = 0.0
        max_sev = 0.0

    # Incident analysis
    durations = compute_machine_incidents(bins, breach_mask)
    if len(durations) > 0:
        n_incidents = len(durations)
        mean_dur_bins = float(np.mean(durations))
        median_dur_bins = float(np.median(durations))
        p95_dur_bins = float(np.percentile(durations, 95))
        max_dur_bins = int(np.max(durations))
        mean_dur_mins = float(mean_dur_bins * 5.0)
        p95_dur_mins = float(p95_dur_bins * 5.0)
        max_dur_mins = float(max_dur_bins * 5.0)
    else:
        n_incidents = 0
        mean_dur_bins = 0.0
        median_dur_bins = 0.0
        p95_dur_bins = 0.0
        max_dur_bins = 0
        mean_dur_mins = 0.0
        p95_dur_mins = 0.0
        max_dur_mins = 0.0

    metrics = {
        'policy': config_name,
        'machine_id': machine_id,
        'total_test_bins': n_bins,
        'breach_bins': breach_bins,
        'breach_rate': breach_rate,
        'overshoot_bins': overshoot_bins,
        'overshoot_rate': overshoot_rate,
        'grace_zone_bins': grace_zone_bins,
        'grace_zone_rate': grace_zone_rate,
        'mean_overprovisioning': mean_overprov,
        'mean_underprovisioning': mean_underprov,
        'total_underprovisioning': total_underprov,
        'mean_shortfall_severity': mean_sev,
        'p95_shortfall_severity': p95_sev,
        'max_shortfall_severity': max_sev,
        'number_of_incidents': n_incidents,
        'mean_incident_duration_bins': mean_dur_bins,
        'median_incident_duration_bins': median_dur_bins,
        'p95_incident_duration_bins': p95_dur_bins,
        'max_incident_duration_bins': max_dur_bins,
        'mean_incident_duration_minutes': mean_dur_mins,
        'p95_incident_duration_minutes': p95_dur_mins,
        'max_incident_duration_minutes': max_dur_mins
    }

    return metrics, durations, severities


def run_reliability_decomposition(allocations_path: str = None,
                                  output_dir: str = PHASE2_RESULTS_DIR,
                                  figures_dir: str = PHASE2_FIGURES_DIR):
    """
    Executes Phase 2.1 Reliability Decomposition by consuming authoritative Phase 1 test allocations directly.
    Pure analysis layer: Zero hardcoded beta multipliers.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Load authoritative Phase 1 test allocations artifact
    alloc_path = allocations_path
    if not alloc_path:
        default_parquet = os.path.join(output_dir, "phase1_test_allocations.parquet")
        default_csv = os.path.join(output_dir, "phase1_test_allocations.csv")
        if os.path.exists(default_parquet):
            alloc_path = default_parquet
        elif os.path.exists(default_csv):
            alloc_path = default_csv
        else:
            print("[Phase 2.1] Authoritative allocations artifact not found. Building now...")
            build_authoritative_phase1_allocations(output_dir)
            alloc_path = default_parquet

    print(f"[Phase 2.1 Pure Analysis Layer] Consuming authoritative allocations from: {alloc_path}")
    if alloc_path.endswith('.parquet'):
        alloc_df = pd.read_parquet(alloc_path)
    else:
        alloc_df = pd.read_csv(alloc_path)

    n_records = len(alloc_df)
    n_machines = alloc_df['machine_id'].nunique()
    policies = sorted(alloc_df['policy'].unique())
    print(f"[Phase 2.1] Loaded {n_records} allocation records across {n_machines} machines and {len(policies)} policies:")
    for p in policies:
        sub = alloc_df[alloc_df['policy'] == p]
        beta_val = sub['beta'].iloc[0] if 'beta' in sub.columns else np.nan
        print(f"  - {p:22s}: {len(sub)} rows, beta = {beta_val}")

    # Test horizon in days (1,253 bins of 5 min = 4.3507 days)
    test_days = 1253 * 5.0 / 1440.0

    all_machine_metrics = []
    cluster_incidents = {p: [] for p in policies}
    cluster_severities = {p: [] for p in policies}

    # Evaluate each policy and machine directly from the authoritative allocation artifact
    for pol_name in policies:
        pol_sub = alloc_df[alloc_df['policy'] == pol_name]
        for mid, grp in pol_sub.groupby('machine_id'):
            yt = grp['y_true'].values
            al = grp['allocation'].values
            bins = grp['bin_idx'].values

            m, durs, sevs = compute_machine_reliability_metrics(yt, al, bins, mid, pol_name)
            all_machine_metrics.append(m)
            cluster_incidents[pol_name].extend(durs)
            if len(sevs) > 0:
                cluster_severities[pol_name].extend(sevs.tolist())

    # Convert to DataFrame
    decomp_df = pd.DataFrame(all_machine_metrics)
    decomp_csv = os.path.join(output_dir, "reliability_decomposition.csv")
    decomp_df.to_csv(decomp_csv, index=False)
    print(f"[Phase 2.1] Saved per-machine reliability decomposition to: {decomp_csv}")

    # 2. Cluster Incident Summary Table
    incident_rows = []
    for pol in policies:
        durs = np.array(cluster_incidents[pol])
        n_inc = len(durs)
        if n_inc > 0:
            mean_bins = float(np.mean(durs))
            med_bins = float(np.median(durs))
            p95_bins = float(np.percentile(durs, 95))
            max_bins = int(np.max(durs))
            mean_mins = float(mean_bins * 5.0)
            med_mins = float(med_bins * 5.0)
            p95_mins = float(p95_bins * 5.0)
            max_mins = int(max_bins * 5.0)

            f_1bin = float(np.mean(durs == 1))
            f_2_3 = float(np.mean((durs >= 2) & (durs <= 3)))
            f_4_10 = float(np.mean((durs >= 4) & (durs <= 10)))
            f_gt10 = float(np.mean(durs > 10))
        else:
            mean_bins = med_bins = p95_bins = mean_mins = med_mins = p95_mins = 0.0
            max_bins = max_mins = 0
            f_1bin = f_2_3 = f_4_10 = f_gt10 = 0.0

        incident_rows.append({
            'policy': pol,
            'total_incidents': n_inc,
            'test_days': round(test_days, 4),
            'incidents_per_test_day': round(n_inc / test_days, 2) if test_days > 0 else 0.0,
            'incidents_per_machine_day': round(n_inc / (n_machines * test_days), 4) if test_days > 0 else 0.0,
            'mean_incident_duration_bins': round(mean_bins, 2),
            'median_incident_duration_bins': round(med_bins, 2),
            'p95_incident_duration_bins': round(p95_bins, 2),
            'longest_incident_bins': max_bins,
            'mean_incident_duration_minutes': round(mean_mins, 1),
            'median_incident_duration_minutes': round(med_mins, 1),
            'p95_incident_duration_minutes': round(p95_mins, 1),
            'longest_incident_minutes': max_mins,
            'fraction_single_bin_incidents': round(f_1bin, 4),
            'fraction_2_to_3_bins': round(f_2_3, 4),
            'fraction_4_to_10_bins': round(f_4_10, 4),
            'fraction_gt_10_bins': round(f_gt10, 4)
        })

    incident_summary_df = pd.DataFrame(incident_rows)
    incident_csv = os.path.join(output_dir, "incident_summary.csv")
    incident_summary_df.to_csv(incident_csv, index=False)
    print(f"[Phase 2.1] Saved cluster incident summary to: {incident_csv}")

    # 3. Severity Summary Table (conditional on breach)
    sev_rows = []
    for pol in policies:
        sevs = np.array(cluster_severities[pol])
        n_breach = len(sevs)
        if n_breach > 0:
            mean_s = float(np.mean(sevs))
            med_s = float(np.median(sevs))
            p95_s = float(np.percentile(sevs, 95))
            p99_s = float(np.percentile(sevs, 99))
            max_s = float(np.max(sevs))
        else:
            mean_s = med_s = p95_s = p99_s = max_s = 0.0

        sev_rows.append({
            'policy': pol,
            'total_breach_bins': n_breach,
            'mean_severity': round(mean_s, 6),
            'median_severity': round(med_s, 6),
            'p95_severity': round(p95_s, 6),
            'p99_severity': round(p99_s, 6),
            'max_severity': round(max_s, 6)
        })

    sev_summary_df = pd.DataFrame(sev_rows)
    sev_csv = os.path.join(output_dir, "severity_summary.csv")
    sev_summary_df.to_csv(sev_csv, index=False)
    print(f"[Phase 2.1] Saved shortfall severity summary to: {sev_csv}")

    # 4. Generate Additional Analysis: Per-Machine Comparison Table (Adaptive U* vs Static U2)
    comp_df = generate_per_machine_comparison(decomp_df, output_dir)

    # 5. Generate Diagnostic Figures
    generate_figures(decomp_df, cluster_incidents, cluster_severities, figures_dir)

    return decomp_df, incident_summary_df, sev_summary_df, comp_df


def generate_per_machine_comparison(decomp_df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """
    Generates head-to-head per-machine comparison table between Adaptive U* and Static U2:
    - machine_id
    - U2_breach_rate
    - Ustar_breach_rate
    - delta (Ustar_breach_rate - U2_breach_rate)
    - U2_overprovision
    - Ustar_overprovision
    - delta_overprovision (Ustar_overprovision - U2_overprovision)
    """
    sub_u2 = decomp_df[decomp_df['policy'] == 'Static_U2'].set_index('machine_id')
    sub_ustar = decomp_df[decomp_df['policy'] == 'Adaptive_U_star'].set_index('machine_id')

    common_mids = sub_u2.index.intersection(sub_ustar.index)

    rows = []
    for mid in common_mids:
        u2_br = float(sub_u2.loc[mid, 'breach_rate'])
        ustar_br = float(sub_ustar.loc[mid, 'breach_rate'])
        delta_br = float(ustar_br - u2_br)

        u2_op = float(sub_u2.loc[mid, 'mean_overprovisioning'])
        ustar_op = float(sub_ustar.loc[mid, 'mean_overprovisioning'])
        delta_op = float(ustar_op - u2_op)

        rows.append({
            'machine_id': int(mid),
            'U2_breach_rate': u2_br,
            'Ustar_breach_rate': ustar_br,
            'delta': delta_br,
            'U2_overprovision': u2_op,
            'Ustar_overprovision': ustar_op,
            'delta_overprovision': delta_op
        })

    comp_df = pd.DataFrame(rows)
    out_csv = os.path.join(output_dir, "machine_comparison_u2_vs_ustar.csv")
    comp_df.to_csv(out_csv, index=False)
    print(f"[Phase 2.1 Additional Analysis] Saved machine-level comparison table to: {out_csv}")

    return comp_df


def generate_figures(decomp_df: pd.DataFrame, cluster_incidents: dict,
                     cluster_severities: dict, figures_dir: str):
    """Generates publication-quality diagnostic figures for Phase 2.1."""
    plt.rcParams.update({
        'font.sans-serif': 'DejaVu Sans',
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.edgecolor': '#cccccc',
        'axes.linewidth': 0.8
    })

    colors = {
        'Static_U2': '#1f77b4',
        'Adaptive_U_star': '#ff7f0e',
        'Static_U1': '#2ca02c',
        'Static_U3': '#d62728',
        'B3_Fusion_Fixed10': '#9467bd',
        'B0_Naive_Fixed10': '#8c564b'
    }

    # Figure 1: Incident Duration Distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    categories = ['1 bin (5m)', '2–3 bins (10–15m)', '4–10 bins (20–50m)', '>10 bins (>50m)']
    bar_width = 0.18
    target_policies = ['Static_U2', 'Adaptive_U_star', 'Static_U1', 'Static_U3']

    x = np.arange(len(categories))
    for i, pol in enumerate(target_policies):
        durs = np.array(cluster_incidents.get(pol, []))
        if len(durs) > 0:
            fractions = [
                np.mean(durs == 1) * 100,
                np.mean((durs >= 2) & (durs <= 3)) * 100,
                np.mean((durs >= 4) & (durs <= 10)) * 100,
                np.mean(durs > 10) * 100
            ]
        else:
            fractions = [0, 0, 0, 0]

        offset = (i - 1.5) * bar_width
        ax1.bar(x + offset, fractions, bar_width, label=pol.replace('_', ' '),
                color=colors.get(pol, '#333333'), alpha=0.88, edgecolor='black', linewidth=0.5)

    ax1.set_xlabel('Incident Duration Classification', fontweight='bold')
    ax1.set_ylabel('Percentage of Total Incidents (%)', fontweight='bold')
    ax1.set_title('A. Incident Duration Frequency Breakdown', fontweight='bold', pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=9.5)
    ax1.grid(axis='y', linestyle='--', alpha=0.4)
    ax1.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9.5)

    for pol in target_policies:
        durs = np.array(cluster_incidents.get(pol, []))
        if len(durs) > 0:
            sorted_mins = np.sort(durs * 5.0)
            ecdf = np.arange(1, len(sorted_mins) + 1) / len(sorted_mins)
            ax2.step(sorted_mins, ecdf, where='post', label=pol.replace('_', ' '),
                     color=colors.get(pol, '#333333'), linewidth=2.0)

    ax2.set_xlabel('Incident Duration (Minutes)', fontweight='bold')
    ax2.set_ylabel('Empirical Cumulative Probability P(Duration ≤ t)', fontweight='bold')
    ax2.set_title('B. Duration Cumulative Distribution (ECDF)', fontweight='bold', pad=10)
    ax2.set_xlim(0, 120)
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9.5)

    plt.tight_layout()
    f1_path = os.path.join(figures_dir, "incident_duration_distribution.png")
    plt.savefig(f1_path, dpi=300)
    plt.close()
    print(f"[Phase 2.1] Saved figure: {f1_path}")

    # Figure 2: Shortfall Severity Distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    sev_data = [np.array(cluster_severities.get(p, [])) for p in target_policies]
    clean_labels = [p.replace('_', ' ') for p in target_policies]

    bp = ax1.boxplot(sev_data, patch_artist=True, showmeans=True,
                     meanprops={'marker': '^', 'markerfacecolor': 'gold', 'markeredgecolor': 'black', 'markersize': 8},
                     medianprops={'color': 'black', 'linewidth': 1.5})
    ax1.set_xticks(range(1, len(clean_labels) + 1))
    ax1.set_xticklabels(clean_labels, fontsize=9.5)

    for patch, pol in zip(bp['boxes'], target_policies):
        patch.set_facecolor(colors.get(pol, '#333333'))
        patch.set_alpha(0.75)

    ax1.set_ylabel(r'Shortfall Severity $S_t = \max(0, y_t - (A_t + 0.005))$', fontweight='bold')
    ax1.set_title('A. Conditional Shortfall Severity Distribution', fontweight='bold', pad=10)
    ax1.grid(axis='y', linestyle='--', alpha=0.4)

    for pol in target_policies:
        sevs = np.array(cluster_severities.get(pol, []))
        if len(sevs) > 0:
            weights = np.ones_like(sevs) / len(sevs)
            ax2.hist(sevs, bins=30, range=(0, 0.15), weights=weights, histtype='step',
                     linewidth=2.0, label=pol.replace('_', ' '), color=colors.get(pol, '#333333'))

    ax2.set_xlabel(r'Shortfall Severity $S_t$', fontweight='bold')
    ax2.set_ylabel('Probability Density Fraction', fontweight='bold')
    ax2.set_title('B. Shortfall Severity PDF Conditional on Breach', fontweight='bold', pad=10)
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9.5)

    plt.tight_layout()
    f2_path = os.path.join(figures_dir, "shortfall_severity_distribution.png")
    plt.savefig(f2_path, dpi=300)
    plt.close()
    print(f"[Phase 2.1] Saved figure: {f2_path}")

    # Figure 3: Per-Machine Breach Rate Distribution
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for pol in target_policies:
        sub = decomp_df[decomp_df['policy'] == pol]
        rates = np.sort(sub['breach_rate'].values) * 100.0
        ax.plot(np.arange(1, len(rates) + 1), rates, marker='o', markersize=3,
                linewidth=1.8, label=pol.replace('_', ' '), color=colors.get(pol, '#333333'))

    ax.axhline(1.0, color='red', linestyle='--', linewidth=1.5, label='SLA Target Budget (1.0%)')
    ax.set_xlabel('Rank-Ordered Machine Index (1 to 100)', fontweight='bold')
    ax.set_ylabel('Empirical Test SLA Breach Rate (%)', fontweight='bold')
    ax.set_title('Per-Machine Breach Rate Distribution Across 100 Benchmark Servers', fontweight='bold', pad=10)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10)

    plt.tight_layout()
    f3_path = os.path.join(figures_dir, "per_machine_breach_rate.png")
    plt.savefig(f3_path, dpi=300)
    plt.close()
    print(f"[Phase 2.1] Saved figure: {f3_path}")


if __name__ == '__main__':
    run_reliability_decomposition()
