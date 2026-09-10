"""
scripts/figures.py

Step 14: Publication-Quality Visualization Suite for Google MVP.
Generates:
1. figures/prediction_comparison.png: Actual CPU vs HW vs GRU vs Fused predictions.
2. figures/uncertainty_signals.png: U1, U2, U3 signals over time alongside absolute prediction errors.
3. figures/selector_choices.png: Dynamic distribution of winning uncertainty signals.
4. figures/allocation_vs_actual.png: Allocation vs Actual workload showing SLA violations and grace band.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import (
    GRACE_BAND, FIGURES_DIR, MODELS_DIR, UNCERTAINTY_DIR
)


def plot_prediction_comparison(fusion_preds_path: str, output_path: str):
    """Plot Actual vs HW vs GRU vs Fused predictions for a representative machine."""
    df = pd.read_csv(fusion_preds_path)
    test_df = df[df['split'] == 'test']
    if test_df.empty:
        return

    mid = test_df['machine_id'].iloc[0]
    m_data = test_df[test_df['machine_id'] == mid].sort_values('bin_idx')

    plt.figure(figsize=(12, 4.5))
    bins = np.arange(len(m_data))
    plt.plot(bins, m_data['y_true'], 'k-', lw=1.8, label='Actual CPU', alpha=0.9)
    plt.plot(bins, m_data['hw_pred'], 'b--', lw=1.2, label='Holt-Winters', alpha=0.7)
    plt.plot(bins, m_data['gru_pred'], 'g:', lw=1.5, label='Global GRU', alpha=0.8)
    plt.plot(bins, m_data['fused_pred'], 'r-', lw=1.5, label='Fused Forecast', alpha=0.9)

    plt.title(f"Forecasting Comparison — Machine {mid} (Test Split)", fontsize=12, fontweight='bold')
    plt.xlabel("Test Bins (5-min intervals)", fontsize=10)
    plt.ylabel("CPU Utilization [0, 1]", fontsize=10)
    plt.ylim(0, max(0.5, m_data['y_true'].max() + 0.1))
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', frameon=True)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved: {output_path}")


def plot_uncertainty_signals(uncertainty_path: str, output_path: str):
    """Plot U1, U2, U3 against absolute forecast error."""
    df = pd.read_csv(uncertainty_path)
    test_df = df[df['split'] == 'test']
    if test_df.empty:
        return

    mid = test_df['machine_id'].iloc[0]
    m_data = test_df[test_df['machine_id'] == mid].sort_values('bin_idx')

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    bins = np.arange(len(m_data))

    # Top: Absolute Error
    axes[0].plot(bins, m_data['abs_error'], color='#d62728', lw=1.5, label='|Fused - Actual| (Error)')
    axes[0].set_title(f"Forecast Error & Uncertainty Signals — Machine {mid}", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Absolute Error", fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper right')

    # Bottom: Uncertainty Signals
    axes[1].plot(bins, m_data['U1'], color='#1f77b4', lw=1.2, label='U1: Disagreement')
    axes[1].plot(bins, m_data['U2'], color='#ff7f0e', lw=1.2, label='U2: Recent Error EWMA')
    axes[1].plot(bins, m_data['U3'], color='#2ca02c', lw=1.2, label='U3: Volatility')
    axes[1].set_ylabel("Uncertainty Signal", fontsize=10)
    axes[1].set_xlabel("Test Bins (5-min intervals)", fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved: {output_path}")


def plot_selector_choices(selector_log_path: str, output_path: str):
    """Plot the choice distribution of the adaptive reliability selector."""
    df = pd.read_csv(selector_log_path)
    test_df = df[df['split'] == 'test']
    if test_df.empty:
        return

    choice_counts = test_df['chosen_signal'].value_counts()
    colors = {'U1': '#1f77b4', 'U2': '#ff7f0e', 'U3': '#2ca02c'}
    bar_colors = [colors.get(k, 'gray') for k in choice_counts.index]

    plt.figure(figsize=(7, 4.5))
    bars = plt.bar(choice_counts.index, choice_counts.values, color=bar_colors, width=0.5, edgecolor='black')
    for bar in bars:
        h = bar.get_height()
        pct = h / len(test_df)
        plt.text(bar.get_x() + bar.get_width() / 2.0, h + 10, f"{pct:.1%}", ha='center', va='bottom', fontweight='bold')

    plt.title("Adaptive Selector Signal Distribution (Test Split)", fontsize=12, fontweight='bold')
    plt.xlabel("Selected Uncertainty Signal", fontsize=10)
    plt.ylabel("Timestep Count", fontsize=10)
    plt.ylim(0, max(choice_counts.values) * 1.18)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved: {output_path}")


def plot_allocation_vs_actual(selector_log_path: str, output_path: str, beta: float = 1.0):
    """Plot Allocation vs Actual CPU showing grace band and physical bounds."""
    df = pd.read_csv(selector_log_path)
    test_df = df[df['split'] == 'test']
    if test_df.empty:
        return

    mid = test_df['machine_id'].iloc[0]
    m_data = test_df[test_df['machine_id'] == mid].sort_values('bin_idx')

    y_true = m_data['y_true'].values
    fused = m_data['fused_pred'].values
    u_star = m_data['U_star'].values
    alloc = np.clip(fused + beta * u_star, 0.0, 1.0)
    grace = alloc + GRACE_BAND

    bins = np.arange(len(m_data))

    plt.figure(figsize=(12, 5))
    plt.plot(bins, y_true, 'k-', lw=1.8, label='Actual CPU', zorder=3)
    plt.plot(bins, alloc, 'b-', lw=1.5, label=f'Allocation (beta={beta:.1f})', zorder=2)
    plt.plot(bins, grace, 'b--', lw=0.9, alpha=0.6, label='Grace Ceiling (+0.005)', zorder=2)
    plt.fill_between(bins, alloc, grace, color='blue', alpha=0.1, label='Grace Zone')
    plt.fill_between(bins, y_true, alloc, where=(y_true > alloc), color='red', alpha=0.3, label='Overshoot Region')

    plt.title(f"Risk-Aware Resource Allocation vs Workload — Machine {mid}", fontsize=12, fontweight='bold')
    plt.xlabel("Test Bins (5-min intervals)", fontsize=10)
    plt.ylabel("Resource Capacity [0, 1]", fontsize=10)
    plt.ylim(0, max(0.6, max(alloc.max(), y_true.max()) + 0.08))
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', frameon=True)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate publication-ready figures")
    parser.add_argument("--figures_dir", type=str, default=FIGURES_DIR)
    parser.add_argument("--models_dir", type=str, default=MODELS_DIR)
    parser.add_argument("--uncertainty_dir", type=str, default=UNCERTAINTY_DIR)
    args = parser.parse_args()

    fusion_path = os.path.join(args.models_dir, "fusion_predictions.csv")
    uncertainty_path = os.path.join(args.uncertainty_dir, "uncertainty.csv")
    selector_path = os.path.join(args.uncertainty_dir, "selector_log.csv")

    plot_prediction_comparison(fusion_path, os.path.join(args.figures_dir, "prediction_comparison.png"))
    plot_uncertainty_signals(uncertainty_path, os.path.join(args.figures_dir, "uncertainty_signals.png"))
    plot_selector_choices(selector_path, os.path.join(args.figures_dir, "selector_choices.png"))
    plot_allocation_vs_actual(selector_path, os.path.join(args.figures_dir, "allocation_vs_actual.png"))


if __name__ == "__main__":
    main()
