"""
scripts/phase2/reliability.py

Phase 2.1: SLA Reliability Decomposition and Incident Analysis.
Implements:
1. Granular per-machine reliability decomposition:
   - total_test_bins, breach_bins, breach_rate
   - overshoot_bins, overshoot_rate
   - grace_zone_bins, grace_zone_rate
   - mean_overprovisioning, mean_underprovisioning, total_underprovisioning
   - mean_shortfall_severity, p95_shortfall_severity, max_shortfall_severity
2. Temporal incident run-length analysis:
   - Consecutive breach bins for the SAME machine
   - Incident durations in bins and minutes (duration_bins * 5)
   - Incident frequency distributions (1 bin, 2-3 bins, 4-10 bins, >10 bins)
   - Cluster-level metrics: total incidents, incidents per test day, longest incident streak (D_max)
3. Conditional shortfall severity:
   - S_t = max(0, y_t - (A_t + GRACE_BAND)) conditional on breach
   - mean, median, p95, p99, max severity
4. Publication-quality visual diagnostics:
   - incident_duration_distribution.png
   - shortfall_severity_distribution.png
   - per_machine_breach_rate.png

Strict Invariants:
- Breach definition strictly preserved: y_t > A_t + 0.005 (GRACE_BAND = 0.005).
- Terminology: "five-minute workload-capacity breach events" (not "service outages").
- Uses existing Phase 1 test predictions (zero retraining).
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.utils import (
    GRACE_BAND, RESULTS_DIR, FIGURES_DIR, UNCERTAINTY_DIR,
    SELECTED_MACHINES_PATH, SPLIT_INFO_PATH, SERIES_DIR
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
                # Start new incident
                current_duration = 1
            else:
                # Check consecutive continuity
                if b_idx == prev_bin + 1:
                    current_duration += 1
                else:
                    # Missing gap intervened; close prior incident and start new one
                    durations.append(current_duration)
                    current_duration = 1
        else:
            if current_duration > 0:
                durations.append(current_duration)
                current_duration = 0
        prev_bin = b_idx

    # Close trailing incident at boundary if active
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


def run_reliability_decomposition(test_selector_path: str = None,
                                  output_dir: str = PHASE2_RESULTS_DIR,
                                  figures_dir: str = PHASE2_FIGURES_DIR):
    """
    Executes Phase 2.1 Reliability Decomposition across all benchmark policies:
    Static U2, Adaptive U*, Static U1, Static U3, and Fixed Baselines (B3, B1, B2, B0).
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    test_path = test_selector_path or os.path.join(UNCERTAINTY_DIR, "selector_test.csv")
    print(f"[Phase 2.1] Loading test predictions from: {test_path}")
    test_df = pd.read_csv(test_path)

    # Load machine metadata and split info for baseline series extraction
    with open(SELECTED_MACHINES_PATH, 'r') as f:
        selected_mids = json.load(f)
    with open(SPLIT_INFO_PATH, 'r') as f:
        split_info = json.load(f)

    # Verification: test duration in days
    # Test split has 1,253 bins of 5 minutes: 1253 * 5 / 1440 = 4.35069 days
    total_test_bins_horizon = 1253
    test_days = total_test_bins_horizon * 5.0 / 1440.0
    n_machines = len(selected_mids)

    # Define all evaluation policies
    # Uncertainty policies use the Phase 1 validation-calibrated beta multipliers
    policies = [
        ('Static_U2', 'fused_pred', 'U2', 5.0, False),
        ('Adaptive_U_star', 'fused_pred', 'U_star', 8.0, False),
        ('Static_U1', 'fused_pred', 'U1', 12.0, False),
        ('Static_U3', 'fused_pred', 'U3', 14.0, False),
        ('B3_Fusion_Fixed10', 'fused_pred', None, 1.10, True),
        ('B1_GRU_Fixed10', 'gru_pred', None, 1.10, True),
        ('B2_HoltWinters_Fixed10', 'hw_pred', None, 1.10, True),
    ]

    all_machine_metrics = []
    cluster_incidents = {p[0]: [] for p in policies}
    cluster_incidents['B0_Naive_Fixed10'] = []
    cluster_severities = {p[0]: [] for p in policies}
    cluster_severities['B0_Naive_Fixed10'] = []

    print(f"[Phase 2.1] Evaluating {len(policies) + 1} policies across {n_machines} machines...")

    # 1. Evaluate uncertainty policies and GRU/HW/Fusion baselines
    for pol_name, pred_col, u_col, mult, is_fixed in policies:
        for mid, grp in test_df.groupby('machine_id'):
            yt = grp['y_true'].values
            yp = grp[pred_col].values
            bins = grp['bin_idx'].values

            if is_fixed:
                ar = yp * mult
            else:
                u_vals = grp[u_col].values
                ar = yp + mult * u_vals

            al = np.clip(ar, 0.0, 1.0)
            m, durs, sevs = compute_machine_reliability_metrics(yt, al, bins, mid, pol_name)
            all_machine_metrics.append(m)
            cluster_incidents[pol_name].extend(durs)
            if len(sevs) > 0:
                cluster_severities[pol_name].extend(sevs.tolist())

    # 2. Evaluate B0 Naive baseline from machine series
    pol_name = 'B0_Naive_Fixed10'
    for mid in selected_mids:
        series_path = os.path.join(SERIES_DIR, f"{mid}.parquet")
        if not os.path.exists(series_path):
            continue
        s = pd.read_parquet(series_path)['cpu']
        t0, t1 = split_info[str(mid)]['test_range']
        yt = s.iloc[t0:t1].values
        bins = np.arange(t0, t1)

        pred_b0 = s.shift(1).iloc[t0:t1].values
        al = np.clip(pred_b0 * 1.10, 0.0, 1.0)

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

    # 3. Cluster Incident Summary Table
    incident_rows = []
    all_eval_policies = [p[0] for p in policies] + ['B0_Naive_Fixed10']

    for pol in all_eval_policies:
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

    # 4. Severity Summary Table (conditional on breach)
    sev_rows = []
    for pol in all_eval_policies:
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

    # 5. Generate Figures
    generate_figures(decomp_df, cluster_incidents, cluster_severities, figures_dir)

    return decomp_df, incident_summary_df, sev_summary_df


def generate_figures(decomp_df: pd.DataFrame, cluster_incidents: dict,
                     cluster_severities: dict, figures_dir: str):
    """Generates all three publication-quality diagnostic figures for Phase 2.1."""
    plt.rcParams.update({
        'font.sans-serif': 'DejaVu Sans',
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.edgecolor': '#cccccc',
        'axes.linewidth': 0.8
    })

    # Color palette
    colors = {
        'Static_U2': '#1f77b4',        # Blue
        'Adaptive_U_star': '#ff7f0e',  # Orange
        'Static_U1': '#2ca02c',        # Green
        'Static_U3': '#d62728',        # Red
        'B3_Fusion_Fixed10': '#9467bd',# Purple
        'B0_Naive_Fixed10': '#8c564b'  # Brown
    }

    # --------------------------------------------------------------------------
    # Figure 1: Incident Duration Distribution
    # --------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: Binned duration categories
    categories = ['1 bin (5m)', '2–3 bins (10–15m)', '4–10 bins (20–50m)', '>10 bins (>50m)']
    bar_width = 0.18
    target_policies = ['Static_U2', 'Adaptive_U_star', 'Static_U1', 'Static_U3']

    x = np.arange(len(categories))
    for i, pol in enumerate(target_policies):
        durs = np.array(cluster_incidents[pol])
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

    # Right: Empirical CDF of duration in minutes
    for pol in target_policies:
        durs = np.array(cluster_incidents[pol])
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

    # --------------------------------------------------------------------------
    # Figure 2: Shortfall Severity Distribution conditional on breach
    # --------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: Boxplot of conditional shortfall severity
    sev_data = [np.array(cluster_severities[p]) for p in target_policies]
    # Filter empty
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

    # Right: Histograms of shortfall severity
    for pol in target_policies:
        sevs = np.array(cluster_severities[pol])
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

    # --------------------------------------------------------------------------
    # Figure 3: Per-Machine Breach Rate Distribution (Tail Skewness)
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for pol in ['Static_U2', 'Adaptive_U_star', 'Static_U1', 'Static_U3']:
        sub = decomp_df[decomp_df['policy'] == pol]
        rates = np.sort(sub['breach_rate'].values) * 100.0  # Percentage
        ax.plot(np.arange(1, len(rates) + 1), rates, marker='o', markersize=3,
                linewidth=1.8, label=pol.replace('_', ' '), color=colors.get(pol, '#333333'))

    # Draw SLA target line (1.0%)
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
