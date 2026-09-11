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
    GRACE_BAND, FIGURES_DIR, MODELS_DIR, UNCERTAINTY_DIR,
    SERIES_DIR, SELECTED_MACHINES_PATH, RESULTS_DIR
)


def _get_default_beta() -> float:
    """Attempts to retrieve calibrated beta* from allocation_results.csv."""
    alloc_csv = os.path.join(RESULTS_DIR, "allocation_results.csv")
    if os.path.exists(alloc_csv):
        try:
            df = pd.read_csv(alloc_csv)
            sub = df[df['config'] == 'Proposed_Adaptive_U']
            if len(sub) > 0 and 'beta_buf' in sub.columns:
                return float(sub['beta_buf'].iloc[0])
        except Exception:
            pass
    return 8.0


def plot_prediction_comparison(fusion_preds_path: str, output_path: str,
                               machine_id: int = None, start_bin: int = None,
                               end_bin: int = None, split: str = 'test'):
    """Plot Actual vs HW vs GRU vs Fused predictions with customizable machine, time window, and split."""
    df = pd.read_csv(fusion_preds_path)
    sub_df = df[df['split'] == split] if 'split' in df.columns else df
    if sub_df.empty:
        print(f"[Warning] No data found for split '{split}' in {fusion_preds_path}")
        return

    mid = machine_id if machine_id is not None else sub_df['machine_id'].iloc[0]
    m_data = sub_df[sub_df['machine_id'] == mid].sort_values('bin_idx')
    if m_data.empty:
        print(f"[Warning] Machine {mid} not found in {split} split")
        return

    total_pts = len(m_data)
    s_idx = max(0, start_bin) if start_bin is not None else 0
    e_idx = min(total_pts, end_bin) if end_bin is not None else total_pts
    sliced = m_data.iloc[s_idx:e_idx]

    plt.figure(figsize=(12, 4.5))
    bins = np.arange(s_idx, s_idx + len(sliced))
    plt.plot(bins, sliced['y_true'], 'k-', lw=1.8, label='Actual CPU', alpha=0.9)
    plt.plot(bins, sliced['hw_pred'], 'b--', lw=1.2, label='Holt-Winters', alpha=0.7)
    plt.plot(bins, sliced['gru_pred'], 'g:', lw=1.5, label='Global GRU', alpha=0.8)
    plt.plot(bins, sliced['fused_pred'], 'r-', lw=1.5, label='Fused Forecast', alpha=0.9)

    win_desc = f" (Bins {s_idx}:{e_idx})" if (start_bin is not None or end_bin is not None) else ""
    plt.title(f"Forecasting Comparison — Machine {mid} [{split.upper()} Split{win_desc}]", fontsize=12, fontweight='bold')
    plt.xlabel(f"{split.title()} Bins (5-min intervals)", fontsize=10)
    plt.ylabel("CPU Utilization [0, 1]", fontsize=10)
    plt.ylim(0, max(0.5, sliced['y_true'].max() + 0.1))
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', frameon=True)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved: {output_path}")


def plot_uncertainty_signals(uncertainty_path: str, output_path: str,
                             machine_id: int = None, start_bin: int = None,
                             end_bin: int = None, split: str = 'test'):
    """Plot U1, U2, U3 against absolute forecast error with customizable machine and time window."""
    df = pd.read_csv(uncertainty_path)
    sub_df = df[df['split'] == split] if 'split' in df.columns else df
    if sub_df.empty:
        return

    mid = machine_id if machine_id is not None else sub_df['machine_id'].iloc[0]
    m_data = sub_df[sub_df['machine_id'] == mid].sort_values('bin_idx')
    if m_data.empty:
        return

    total_pts = len(m_data)
    s_idx = max(0, start_bin) if start_bin is not None else 0
    e_idx = min(total_pts, end_bin) if end_bin is not None else total_pts
    sliced = m_data.iloc[s_idx:e_idx]

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    bins = np.arange(s_idx, s_idx + len(sliced))

    # Top: Absolute Error
    win_desc = f" (Bins {s_idx}:{e_idx})" if (start_bin is not None or end_bin is not None) else ""
    axes[0].plot(bins, sliced['abs_error'], color='#d62728', lw=1.5, label='|Fused - Actual| (Error)')
    axes[0].set_title(f"Forecast Error & Uncertainty Signals — Machine {mid} [{split.upper()}{win_desc}]", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Absolute Error", fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper right')

    # Bottom: Uncertainty Signals
    axes[1].plot(bins, sliced['U1'], color='#1f77b4', lw=1.2, label='U1: Disagreement')
    axes[1].plot(bins, sliced['U2'], color='#ff7f0e', lw=1.2, label='U2: Recent Error EWMA')
    axes[1].plot(bins, sliced['U3'], color='#2ca02c', lw=1.2, label='U3: Volatility')
    axes[1].set_ylabel("Uncertainty Signal", fontsize=10)
    axes[1].set_xlabel(f"{split.title()} Bins (5-min intervals)", fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved: {output_path}")


def plot_selector_choices(selector_log_path: str, output_path: str,
                          split: str = 'test', machine_id: int = None):
    """Plot the choice distribution of the adaptive reliability selector."""
    df = pd.read_csv(selector_log_path)
    sub_df = df[df['split'] == split] if 'split' in df.columns else df
    if machine_id is not None:
        sub_df = sub_df[sub_df['machine_id'] == machine_id]
    if sub_df.empty:
        return

    choice_counts = sub_df['chosen_signal'].value_counts()
    colors = {'U1': '#1f77b4', 'U2': '#ff7f0e', 'U3': '#2ca02c'}
    bar_colors = [colors.get(k, 'gray') for k in choice_counts.index]

    plt.figure(figsize=(7, 4.5))
    bars = plt.bar(choice_counts.index, choice_counts.values, color=bar_colors, width=0.5, edgecolor='black')
    for bar in bars:
        h = bar.get_height()
        pct = h / len(sub_df)
        plt.text(bar.get_x() + bar.get_width() / 2.0, h + 10, f"{pct:.1%}", ha='center', va='bottom', fontweight='bold')

    m_str = f"Machine {machine_id}" if machine_id is not None else "All Cluster Machines"
    plt.title(f"Adaptive Selector Signal Distribution — {m_str} ({split.title()} Split)", fontsize=12, fontweight='bold')
    plt.xlabel("Selected Uncertainty Signal", fontsize=10)
    plt.ylabel("Timestep Count", fontsize=10)
    plt.ylim(0, max(choice_counts.values) * 1.18)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved: {output_path}")


def plot_allocation_vs_actual(selector_log_path: str, output_path: str,
                              beta: float = None, machine_id: int = None,
                              start_bin: int = None, end_bin: int = None,
                              split: str = 'test'):
    """Plot Allocation vs Actual CPU showing grace band and physical bounds with zoom support."""
    df = pd.read_csv(selector_log_path)
    sub_df = df[df['split'] == split] if 'split' in df.columns else df
    if sub_df.empty:
        return

    mid = machine_id if machine_id is not None else sub_df['machine_id'].iloc[0]
    m_data = sub_df[sub_df['machine_id'] == mid].sort_values('bin_idx')
    if m_data.empty:
        return

    eff_beta = beta if beta is not None else _get_default_beta()
    total_pts = len(m_data)
    s_idx = max(0, start_bin) if start_bin is not None else 0
    e_idx = min(total_pts, end_bin) if end_bin is not None else total_pts
    sliced = m_data.iloc[s_idx:e_idx]

    y_true = sliced['y_true'].values
    fused = sliced['fused_pred'].values
    u_star = sliced['U_star'].values
    alloc = np.clip(fused + eff_beta * u_star, 0.0, 1.0)
    grace = alloc + GRACE_BAND

    bins = np.arange(s_idx, s_idx + len(sliced))

    plt.figure(figsize=(12, 5))
    plt.plot(bins, y_true, 'k-', lw=1.8, label='Actual CPU', zorder=3)
    plt.plot(bins, alloc, 'b-', lw=1.5, label=f'Allocation (beta={eff_beta:.1f})', zorder=2)
    plt.plot(bins, grace, 'b--', lw=0.9, alpha=0.6, label='Grace Ceiling (+0.005)', zorder=2)
    plt.fill_between(bins, alloc, grace, color='blue', alpha=0.1, label='Grace Zone')
    plt.fill_between(bins, y_true, alloc, where=(y_true > alloc), color='red', alpha=0.3, label='Overshoot Region')

    win_desc = f" (Bins {s_idx}:{e_idx})" if (start_bin is not None or end_bin is not None) else ""
    plt.title(f"Risk-Aware Resource Allocation vs Workload — Machine {mid} [{split.upper()}{win_desc}]", fontsize=12, fontweight='bold')
    plt.xlabel(f"{split.title()} Bins (5-min intervals)", fontsize=10)
    plt.ylabel("Resource Capacity [0, 1]", fontsize=10)
    plt.ylim(0, max(0.6, max(alloc.max(), y_true.max()) + 0.08))
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', frameon=True)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved: {output_path}")


def plot_data_validation(series_dir: str = SERIES_DIR, selected_json: str = SELECTED_MACHINES_PATH,
                         output_path: str = os.path.join(FIGURES_DIR, "data_validation.png"),
                         machine_ids: list = None, start_bin: int = None, end_bin: int = None):
    """Plot visual data inspection check across 4 machines over full or custom time span."""
    if machine_ids is None or len(machine_ids) == 0:
        if os.path.exists(selected_json):
            with open(selected_json, 'r') as f:
                raw_m = json.load(f)
            mids = raw_m[:4] if isinstance(raw_m, list) else list(raw_m.keys())[:4]
        else:
            files = [f for f in os.listdir(series_dir) if f.endswith('.parquet')][:4]
            mids = [os.path.splitext(f)[0] for f in files]
    else:
        mids = machine_ids[:4]

    fig, axes = plt.subplots(len(mids), 1, figsize=(12, 2.5 * len(mids)), sharex=True)
    if len(mids) == 1:
        axes = [axes]

    for idx, mid in enumerate(mids):
        fpath = os.path.join(series_dir, f"{mid}.parquet")
        if not os.path.exists(fpath):
            continue
        df = pd.read_parquet(fpath).sort_values('bin_idx')
        s_idx = max(0, start_bin) if start_bin is not None else 0
        e_idx = min(len(df), end_bin) if end_bin is not None else len(df)
        sub = df.iloc[s_idx:e_idx]
        bins = np.arange(s_idx, s_idx + len(sub))

        ax = axes[idx]
        ax.plot(bins, sub['cpu'], color='#1f77b4', lw=1.0)
        ax.set_title(f"Machine {mid} CPU Utilization", fontsize=10, fontweight='bold')
        ax.set_ylabel("CPU", fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3)

    win_desc = f" (Bins {start_bin}:{end_bin})" if (start_bin is not None or end_bin is not None) else ""
    axes[-1].set_xlabel(f"5-Minute Aggregate Bins{win_desc}", fontsize=10)
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
    parser.add_argument("--machine_id", type=int, default=None)
    parser.add_argument("--start_bin", type=int, default=None)
    parser.add_argument("--end_bin", type=int, default=None)
    parser.add_argument("--split", type=str, default='test')
    parser.add_argument("--beta", type=float, default=None)
    args = parser.parse_args()

    fusion_path = os.path.join(args.models_dir, "fusion_predictions.csv")
    uncertainty_path = os.path.join(args.uncertainty_dir, "uncertainty.csv")
    selector_path = os.path.join(args.uncertainty_dir, "selector_log.csv")

    plot_prediction_comparison(
        fusion_path, os.path.join(args.figures_dir, "prediction_comparison.png"),
        machine_id=args.machine_id, start_bin=args.start_bin, end_bin=args.end_bin, split=args.split
    )
    plot_uncertainty_signals(
        uncertainty_path, os.path.join(args.figures_dir, "uncertainty_signals.png"),
        machine_id=args.machine_id, start_bin=args.start_bin, end_bin=args.end_bin, split=args.split
    )
    plot_selector_choices(
        selector_path, os.path.join(args.figures_dir, "selector_choices.png"),
        split=args.split, machine_id=args.machine_id
    )
    plot_allocation_vs_actual(
        selector_path, os.path.join(args.figures_dir, "allocation_vs_actual.png"),
        beta=args.beta, machine_id=args.machine_id, start_bin=args.start_bin, end_bin=args.end_bin, split=args.split
    )


if __name__ == "__main__":
    main()
