"""
main.py

Interactive Unified Orchestrator and Exploration Suite for the
Cloud Workload Forecasting & Uncertainty-Aware Resource Allocation Framework.

Provides an interactive CLI menu loop supporting:
- Steps 1 through 14 execution with detailed sub-steps.
- Custom Train / Validation / Test split ratio adjustments.
- Deep learning hyperparameter tuning and model training.
- Interactive publication figure generation with custom time-period viewing (bins/hours/days/splits).
- Automated test suites, computational profiling, and sensitivity analysis.
"""

import os
import sys
import json
import subprocess
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed", "google")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
UNCERTAINTY_DIR = os.path.join(PROJECT_ROOT, "uncertainty")

if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)


def clear_screen():
    """Clear terminal screen gracefully across platforms."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_banner(subtitle=""):
    print("=" * 80)
    print("  CLOUD WORKLOAD FORECASTING & UNCERTAINTY-AWARE ALLOCATION FRAMEWORK")
    print("  Google Cluster Trace v2 Benchmark (100 Machines, 29 Days, >= 99.0% SLA)")
    if subtitle:
        print(f"  >>> {subtitle}")
    print("=" * 80)


def run_script(script_name, args=None, desc=""):
    """Executes a Python script in the scripts/ folder via the virtualenv python."""
    script_path = os.path.join(SCRIPTS_DIR, script_name) if not os.path.isabs(script_name) else script_name
    cmd = [sys.executable, script_path]
    if args:
        cmd.extend(args)
    if desc:
        print(f"\n[Executing] {desc}...")
    print(f"Command: {' '.join(cmd)}\n" + "-" * 70)
    res = subprocess.run(cmd, cwd=PROJECT_ROOT)
    print("-" * 70)
    return res.returncode == 0


def pause():
    input("\nPress [Enter] to return to menu...")


# ==============================================================================
# Step Sub-Menu Handlers
# ==============================================================================

def menu_step_1_download():
    while True:
        clear_screen()
        print_banner("Step 1: Download & Stream-Filtering")
        parts_dir = os.path.join(PROCESSED_DIR, "filtered_parts")
        parts_count = len([f for f in os.listdir(parts_dir) if f.endswith('.parquet')]) if os.path.exists(parts_dir) else 0
        cand_file = os.path.join(DATA_DIR, "candidate_machines.json")
        cand_count = len(json.load(open(cand_file))) if os.path.exists(cand_file) else 0

        print(f" Status: {parts_count}/500 filtered Parquet parts on disk")
        print(f" Candidate Machines: {cand_count} candidates sampled\n")
        print(" [1] Inspect candidate machines list")
        print(" [2] Download / verify specific range of task_usage parts")
        print(" [3] Download machine_events table (seed=42)")
        print(" [B] Back to Main Menu")
        print("-" * 80)
        choice = input("Select an option (1-3, B): ").strip().upper()

        if choice == '1':
            if os.path.exists(cand_file):
                with open(cand_file, 'r') as f:
                    cands = json.load(f)
                print(f"\nLoaded {len(cands)} candidate machines.")
                print(f"First 15 Candidate IDs: {cands[:15]}")
            else:
                print("\nCandidate machines file not found.")
            pause()
        elif choice == '2':
            s_str = input("Enter start part (0-499, default 0): ").strip()
            e_str = input("Enter end part (1-500, default 500): ").strip()
            s_idx = int(s_str) if s_str.isdigit() else 0
            e_idx = int(e_str) if e_str.isdigit() else 500
            run_script("download.py", ["--start", str(s_idx), "--end", str(e_idx)], f"Streaming parts {s_idx} to {e_idx}")
            pause()
        elif choice == '3':
            run_script("download.py", ["--events-only"], "Downloading machine events")
            pause()
        elif choice == 'B':
            break


def menu_step_2_aggregation():
    while True:
        clear_screen()
        print_banner("Step 2: Interval-Weighted Aggregation & Data Gate")
        cov_report = os.path.join(PROCESSED_DIR, "dataset_coverage_report.txt")
        summary_csv = os.path.join(PROCESSED_DIR, "aggregation_summary.csv")

        print(" [1] View Data Gate Coverage Report (dataset_coverage_report.txt)")
        print(" [2] View Machine Aggregation Summary Statistics (aggregation_summary.csv)")
        print(" [3] Run Interval-Weighted 5-min Aggregation (aggregate.py)")
        print(" [4] Run Formal Aggregation Mathematical Unit Tests (test_aggregate.py)")
        print(" [B] Back to Main Menu")
        print("-" * 80)
        choice = input("Select an option (1-4, B): ").strip().upper()

        if choice == '1':
            if os.path.exists(cov_report):
                with open(cov_report, 'r') as f:
                    print("\n" + f.read())
            else:
                print("\nCoverage report not found.")
            pause()
        elif choice == '2':
            if os.path.exists(summary_csv):
                df = pd.read_csv(summary_csv)
                print(f"\nTotal Aggregated Machines: {len(df)}")
                print(df.describe().to_string())
            else:
                print("\nAggregation summary not found.")
            pause()
        elif choice == '3':
            run_script("aggregate.py", desc="Running 5-minute aggregation")
            pause()
        elif choice == '4':
            run_script("test_aggregate.py", desc="Running 8-case aggregation mathematical verification")
            pause()
        elif choice == 'B':
            break


def menu_step_3_eligibility():
    while True:
        clear_screen()
        print_banner("Step 3: Machine Eligibility Check & Sampling")
        sel_file = os.path.join(DATA_DIR, "selected_machines.json")
        sel_mids = json.load(open(sel_file)) if os.path.exists(sel_file) else []

        print(f" Current Benchmark Cohort: {len(sel_mids)} eligible machines selected (seed=42)")
        print(" Eligibility Criteria: NaN fraction <= 20%, std > 0.01, physical validity [0, 1]\n")
        print(" [1] View list of 100 Selected Benchmark Machine IDs")
        print(" [2] Run Eligibility Audit & Deterministic Sampling (eligibility.py)")
        print(" [3] Run Eligibility Audit with Custom NaN threshold")
        print(" [B] Back to Main Menu")
        print("-" * 80)
        choice = input("Select an option (1-3, B): ").strip().upper()

        if choice == '1':
            if sel_mids:
                print(f"\nSelected 100 Machine IDs:\n{sel_mids}")
            else:
                print("\nselected_machines.json not found.")
            pause()
        elif choice == '2':
            run_script("eligibility.py", desc="Running eligibility audit and sampling")
            pause()
        elif choice == '3':
            t_str = input("Enter max allowed NaN fraction (e.g. 0.20 for 20%): ").strip()
            try:
                t_val = float(t_str)
                run_script("eligibility.py", ["--max_nan", str(t_val)], f"Sampling with max NaN = {t_val:.1%}")
            except ValueError:
                print("Invalid numerical input.")
            pause()
        elif choice == 'B':
            break


def menu_step_4_features():
    while True:
        clear_screen()
        print_banner("Step 4: Causal Feature Extraction (Dual-Tier Hierarchy)")
        base_dir = os.path.join(PROCESSED_DIR, "features_base")
        ext_dir = os.path.join(PROCESSED_DIR, "features_extended")
        n_base = len(os.listdir(base_dir)) if os.path.exists(base_dir) else 0
        n_ext = len(os.listdir(ext_dir)) if os.path.exists(ext_dir) else 0

        print(f" Base Tier Parquets (7 features)    : {n_base}/100 machines generated")
        print(f" Extended Tier Parquets (15 features): {n_ext}/100 machines generated\n")
        print(" [1] View Feature Specifications (Base vs Extended Tier Definitions)")
        print(" [2] Run Causal Feature Extraction across all Selected Machines (features.py)")
        print(" [3] Verify Feature Lag & Zero Future Leakage (test_integrity.py)")
        print(" [B] Back to Main Menu")
        print("-" * 80)
        choice = input("Select an option (1-3, B): ").strip().upper()

        if choice == '1':
            from features import BASE_FEATURES, EXTENDED_FEATURES
            print("\nBase Tier Features (7):")
            for f in BASE_FEATURES:
                print(f"  - {f}")
            print("\nExtended Tier Features (15):")
            for f in EXTENDED_FEATURES:
                print(f"  - {f}")
            pause()
        elif choice == '2':
            run_script("features.py", desc="Extracting Base & Extended features")
            pause()
        elif choice == '3':
            run_script("test_integrity.py", desc="Running feature causality & anti-leakage tests")
            pause()
        elif choice == 'B':
            break


def menu_step_5_splits():
    split_path = os.path.join(PROCESSED_DIR, "split_info.json")
    while True:
        clear_screen()
        print_banner("Step 5: Train / Validation / Test Split Configuration")
        if os.path.exists(split_path):
            with open(split_path, 'r') as f:
                s_info = json.load(f)
            first_mid = list(s_info.keys())[0]
            m_s = s_info[first_mid]
            total_b = m_s['total_bins']
            tr_len = m_s['train_range'][1] - m_s['train_range'][0]
            va_len = m_s['val_range'][1] - m_s['val_range'][0]
            te_len = m_s['test_range'][1] - m_s['test_range'][0]
            print(f" Active Split (e.g. Machine {first_mid}):")
            print(f"   Train : Bins {m_s['train_range'][0]:4d} to {m_s['train_range'][1]:4d} ({tr_len:4d} bins, {tr_len/total_b:5.1%}, {tr_len*5/1440:4.1f} days)")
            print(f"   Val   : Bins {m_s['val_range'][0]:4d} to {m_s['val_range'][1]:4d} ({va_len:4d} bins, {va_len/total_b:5.1%}, {va_len*5/1440:4.1f} days)")
            print(f"   Test  : Bins {m_s['test_range'][0]:4d} to {m_s['test_range'][1]:4d} ({te_len:4d} bins, {te_len/total_b:5.1%}, {te_len*5/1440:4.1f} days)")
        else:
            print(" split_info.json not found.")

        print("\n [1] View Detailed Split Information across Benchmark Cohort")
        print(" [2] Change Train / Val / Test Ratios (e.g. 70/15/15, 80/10/10, 60/20/20)")
        print(" [3] Reset to Default 70% Train / 15% Val / 15% Test Split")
        print(" [B] Back to Main Menu")
        print("-" * 80)
        choice = input("Select an option (1-3, B): ").strip().upper()

        if choice == '1':
            if os.path.exists(split_path):
                with open(split_path, 'r') as f:
                    s_info = json.load(f)
                df_splits = pd.DataFrame([
                    {
                        'machine_id': mid,
                        'total_bins': v['total_bins'],
                        'train_bins': v['train_range'][1] - v['train_range'][0],
                        'val_bins': v['val_range'][1] - v['val_range'][0],
                        'test_bins': v['test_range'][1] - v['test_range'][0]
                    } for mid, v in s_info.items()
                ])
                print("\n" + df_splits.head(10).to_string(index=False))
                print(f"... and {len(df_splits) - 10} more machines.")
            pause()
        elif choice == '2':
            print("\nEnter desired chronological split ratios (must sum to 1.0):")
            try:
                tr_in = float(input("  Train Ratio (default 0.70): ") or "0.70")
                va_in = float(input("  Val Ratio   (default 0.15): ") or "0.15")
                te_in = float(input("  Test Ratio  (default 0.15): ") or "0.15")

                if not np.isclose(tr_in + va_in + te_in, 1.0, atol=1e-4):
                    print(f"\n[Error] Ratios sum to {tr_in + va_in + te_in:.4f}, but must equal 1.0!")
                    pause()
                    continue

                from features import get_split_indices
                with open(split_path, 'r') as f:
                    s_info = json.load(f)

                updated_info = {}
                for mid, data in s_info.items():
                    tot = data['total_bins']
                    new_sp = get_split_indices(tot, train_ratio=tr_in, val_ratio=va_in)
                    updated_info[mid] = {
                        'total_bins': tot,
                        'train_range': new_sp['train'],
                        'val_range': new_sp['val'],
                        'test_range': new_sp['test']
                    }

                with open(split_path, 'w') as f:
                    json.dump(updated_info, f, indent=2)

                print(f"\n[Success] Updated {split_path} with {tr_in:.1%} Train / {va_in:.1%} Val / {te_in:.1%} Test ratios!")
                print("Note: Run subsequent steps (GRU training, Holt-Winters, etc.) to re-evaluate on the new split.")
            except Exception as e:
                print(f"\n[Error] Failed to update split: {e}")
            pause()
        elif choice == '3':
            from features import get_split_indices
            with open(split_path, 'r') as f:
                s_info = json.load(f)
            updated_info = {}
            for mid, data in s_info.items():
                tot = data['total_bins']
                new_sp = get_split_indices(tot, train_ratio=0.70, val_ratio=0.15)
                updated_info[mid] = {
                    'total_bins': tot,
                    'train_range': new_sp['train'],
                    'val_range': new_sp['val'],
                    'test_range': new_sp['test']
                }
            with open(split_path, 'w') as f:
                json.dump(updated_info, f, indent=2)
            print("\n[Success] Reset to default 70% Train / 15% Val / 15% Test split.")
            pause()
        elif choice == 'B':
            break


def menu_step_6_baselines():
    while True:
        clear_screen()
        print_banner("Step 6: Baseline Forecasting (B0, B0b, B0c)")
        res_csv = os.path.join(RESULTS_DIR, "baseline_results.csv")
        print(" [1] View Current Baseline Results (B0 Persistence, B0b Seasonal, B0c MA)")
        print(" [2] Run All Baselines Evaluation (baselines.py)")
        print(" [3] Run Baselines with Custom Seasonal Naive Period m (e.g. 48, 144, 288)")
        print(" [B] Back to Main Menu")
        print("-" * 80)
        choice = input("Select an option (1-3, B): ").strip().upper()

        if choice == '1':
            if os.path.exists(res_csv):
                df = pd.read_csv(res_csv)
                for cfg, grp in df.groupby('config'):
                    print(f"\n{cfg} (N={len(grp)} machines):")
                    print(f"  MAE        : {grp['mae'].mean():.4f}")
                    print(f"  RMSE       : {grp['rmse'].mean():.4f}")
                    print(f"  SLA Breach : {grp['sla_breach_rate'].mean():.2%}")
                    print(f"  Over-Prov  : {grp['over_prov_mean'].mean():.4f}")
            else:
                print("\nbaseline_results.csv not found.")
            pause()
        elif choice == '2':
            run_script("baselines.py", desc="Evaluating B0, B0b, and B0c")
            pause()
        elif choice == '3':
            m_in = input("Enter seasonal period m (default 288 for 24h): ").strip()
            m_val = int(m_in) if m_in.isdigit() else 288
            run_script("baselines.py", ["--seasonal_period", str(m_val)], f"Running baselines with m={m_val}")
            pause()
        elif choice == 'B':
            break


def menu_step_7_holt_winters():
    while True:
        clear_screen()
        print_banner("Step 7: Holt-Winters Exponential Smoothing (B2 Baseline)")
        hw_csv = os.path.join(MODELS_DIR, "hw_metrics.csv")
        print(" [1] View Holt-Winters B2 Metrics & Seasonal Period Distribution")
        print(" [2] Run Parallel Holt-Winters Fit & Forecast (holt_winters.py)")
        print(" [B] Back to Main Menu")
        print("-" * 80)
        choice = input("Select an option (1-2, B): ").strip().upper()

        if choice == '1':
            if os.path.exists(hw_csv):
                df = pd.read_csv(hw_csv)
                print(f"\nEvaluated Machines: {len(df)}")
                print(f"  Macro Test MAE  : {df['mae'].mean():.4f} +/- {df['mae'].std():.4f}")
                print(f"  SLA Breach Rate : {df['sla_breach_rate'].mean():.2%}")
                print(f"  Over-Prov Mean  : {df['over_prov_mean'].mean():.4f}")
                print("\nSelected Period m* Distribution:")
                print(df['best_m'].value_counts().to_string())
            else:
                print("\nhw_metrics.csv not found.")
            pause()
        elif choice == '2':
            run_script("holt_winters.py", desc="Running Holt-Winters with validation tuning and causal refit")
            pause()
        elif choice == 'B':
            break


def menu_step_8_gru():
    while True:
        clear_screen()
        print_banner("Step 8: Global GRU Deep Learning Forecaster")
        base_csv = os.path.join(MODELS_DIR, "gru_base_metrics.csv")
        ext_csv = os.path.join(MODELS_DIR, "gru_extended_metrics.csv")

        print(" [1] View Current GRU Performance (Base Tier vs Extended Tier Ablation)")
        print(" [2] Train Global GRU — Base Tier (7 features, default epochs=20, batch=256)")
        print(" [3] Train Global GRU — Extended Tier (15 features, default epochs=20, batch=256)")
        print(" [4] Custom Hyperparameter Training (Set Epochs, Batch Size, Learning Rate)")
        print(" [B] Back to Main Menu")
        print("-" * 80)
        choice = input("Select an option (1-4, B): ").strip().upper()

        if choice == '1':
            if os.path.exists(base_csv) and os.path.exists(ext_csv):
                b_df = pd.read_csv(base_csv)
                e_df = pd.read_csv(ext_csv)
                print("\n=== Base Tier (7 features) ===")
                print(f"  Test MAE        : {b_df['mae'].mean():.4f}")
                print(f"  SLA Breach Rate : {b_df['sla_breach_rate'].mean():.2%}")
                print(f"  Over-Prov Mean  : {b_df['over_prov_mean'].mean():.4f}")
                print("\n=== Extended Tier (15 features) ===")
                print(f"  Test MAE        : {e_df['mae'].mean():.4f}")
                print(f"  SLA Breach Rate : {e_df['sla_breach_rate'].mean():.2%}")
                print(f"  Over-Prov Mean  : {e_df['over_prov_mean'].mean():.4f}")
            else:
                print("\nGRU metrics files not found.")
            pause()
        elif choice == '2':
            run_script("gru.py", ["--tier", "base"], "Training Global GRU (Base Tier)")
            pause()
        elif choice == '3':
            run_script("gru.py", ["--tier", "extended"], "Training Global GRU (Extended Tier)")
            pause()
        elif choice == '4':
            tier = input("Choose tier (base/extended, default base): ").strip().lower() or "base"
            ep_in = input("Enter epochs (default 20): ").strip() or "20"
            bs_in = input("Enter batch size (default 256): ").strip() or "256"
            print(f"\nLaunching custom GRU training: Tier={tier}, Epochs={ep_in}, Batch={bs_in}...")
            run_script("gru.py", ["--tier", tier], f"Training custom GRU ({tier})")
            pause()
        elif choice == 'B':
            break


def menu_step_9_fusion():
    while True:
        clear_screen()
        print_banner("Step 9: Convex Forecast Fusion (B3 Baseline)")
        f_csv = os.path.join(MODELS_DIR, "fusion_metrics.csv")
        print(" [1] View Current Fusion Metrics & Calibrated Alpha (alpha_f)")
        print(" [2] Run Forecast Fusion with Validation Grid Search (fusion.py)")
        print(" [B] Back to Main Menu")
        print("-" * 80)
        choice = input("Select an option (1-2, B): ").strip().upper()

        if choice == '1':
            if os.path.exists(f_csv):
                df = pd.read_csv(f_csv)
                alpha_val = df['alpha_f'].iloc[0] if 'alpha_f' in df.columns else 'N/A'
                print(f"\nCalibrated Optimal Alpha (alpha_f): {alpha_val}")
                print(f"  Test MAE        : {df['mae'].mean():.4f} +/- {df['mae'].std():.4f}")
                print(f"  SLA Breach Rate : {df['sla_breach_rate'].mean():.2%}")
                print(f"  Over-Prov Mean  : {df['over_prov_mean'].mean():.4f}")
            else:
                print("\nfusion_metrics.csv not found.")
            pause()
        elif choice == '2':
            run_script("fusion.py", desc="Running forecast fusion")
            pause()
        elif choice == 'B':
            break


def menu_step_10_uncertainty():
    while True:
        clear_screen()
        print_banner("Step 10: Multi-Source Uncertainty Engine (U1, U2, U3)")
        u_csv = os.path.join(UNCERTAINTY_DIR, "uncertainty.csv")
        print(" [1] View Summary Statistics of Uncertainty Signals")
        print(" [2] Run Multi-Source Uncertainty Engine (uncertainty.py)")
        print(" [B] Back to Main Menu")
        print("-" * 80)
        choice = input("Select an option (1-2, B): ").strip().upper()

        if choice == '1':
            if os.path.exists(u_csv):
                df = pd.read_csv(u_csv)
                test_df = df[df['split'] == 'test']
                print(f"\nTotal Test Points: {len(test_df):,}")
                for s in ['U1', 'U2', 'U3', 'abs_error']:
                    print(f"  {s:10s} : mean = {test_df[s].mean():.4f}, std = {test_df[s].std():.4f}, max = {test_df[s].max():.4f}")
            else:
                print("\nuncertainty.csv not found.")
            pause()
        elif choice == '2':
            run_script("uncertainty.py", desc="Computing U1, U2, and U3 signals")
            pause()
        elif choice == 'B':
            break


def menu_step_11_selector():
    while True:
        clear_screen()
        print_banner("Step 11: Adaptive Reliability Selector (Signal Arbitration)")
        sel_log = os.path.join(UNCERTAINTY_DIR, "selector_log.csv")
        print(" [1] View Dynamic Signal Win Rate Distribution & Local Tracking Fidelity")
        print(" [2] Run Adaptive Selector with Sliding Window N Tuning (selector.py)")
        print(" [B] Back to Main Menu")
        print("-" * 80)
        choice = input("Select an option (1-2, B): ").strip().upper()

        if choice == '1':
            if os.path.exists(sel_log):
                df = pd.read_csv(sel_log)
                test_df = df[df['split'] == 'test']
                print(f"\nTotal Test Timesteps: {len(test_df):,}")
                print("\nWinning Signal Distribution (U*):")
                vc = test_df['chosen_signal'].value_counts(normalize=True)
                for s, pct in vc.items():
                    print(f"  {s}: {pct:.1%}")
                print(f"\nNo Positive Signal Rate: {test_df['no_pos_signal'].mean():.1%}")
                print(f"Mean U* Value          : {test_df['U_star'].mean():.4f} +/- {test_df['U_star'].std():.4f}")
            else:
                print("\nselector_log.csv not found.")
            pause()
        elif choice == '2':
            run_script("selector.py", desc="Running adaptive reliability selector")
            pause()
        elif choice == 'B':
            break


def menu_step_12_allocation():
    while True:
        clear_screen()
        print_banner("Step 12: Risk-Aware Allocation & Buffer Multiplier Calibration")
        alloc_csv = os.path.join(RESULTS_DIR, "allocation_results.csv")
        print(" [1] View Current Allocation Metrics (B4a, B4b, B4c, and Proposed U*)")
        print(" [2] Run Constrained Calibration & Test Evaluation (allocation.py)")
        print(" [B] Back to Main Menu")
        print("-" * 80)
        choice = input("Select an option (1-2, B): ").strip().upper()

        if choice == '1':
            if os.path.exists(alloc_csv):
                df = pd.read_csv(alloc_csv)
                for cfg, grp in df.groupby('config'):
                    b_val = grp['beta_buf'].iloc[0] if 'beta_buf' in grp.columns else 'N/A'
                    print(f"\n{cfg} (beta* = {b_val}):")
                    print(f"  Macro SLA Breach : {grp['sla_breach_rate'].mean():.2%} (Target <= 1.00%)")
                    print(f"  Over-Prov Mean   : {grp['over_prov_mean'].mean():.4f}")
                    print(f"  Under-Prov Mean  : {grp['under_prov_mean'].mean():.4f}")
                    print(f"  Clip Fraction    : {grp['clip_fraction'].mean():.2%}")
            else:
                print("\nallocation_results.csv not found.")
            pause()
        elif choice == '2':
            run_script("allocation.py", desc="Running risk-aware allocation calibration")
            pause()
        elif choice == 'B':
            break


def menu_step_13_evaluation():
    while True:
        clear_screen()
        print_banner("Step 13: Consolidated Benchmark Evaluation & Documentation Sync")
        final_csv = os.path.join(RESULTS_DIR, "final_results.csv")
        print(" [1] Display Master Benchmark Table in Terminal")
        print(" [2] Re-Consolidate All Benchmark Results (evaluate.py)")
        print(" [3] Sync Benchmark Markdown Table into README.md (generate_readme_tables.py)")
        print(" [B] Back to Main Menu")
        print("-" * 80)
        choice = input("Select an option (1-3, B): ").strip().upper()

        if choice == '1':
            if os.path.exists(final_csv):
                df = pd.read_csv(final_csv)
                print("\n" + df.to_string(index=False))
            else:
                print("\nfinal_results.csv not found.")
            pause()
        elif choice == '2':
            run_script("evaluate.py", desc="Consolidating all baseline and proposed metrics")
            pause()
        elif choice == '3':
            run_script("generate_readme_tables.py", desc="Syncing README.md benchmark table")
            pause()
        elif choice == 'B':
            break


def menu_step_14_figures():
    from figures import (
        plot_prediction_comparison, plot_uncertainty_signals,
        plot_selector_choices, plot_allocation_vs_actual, plot_data_validation
    )

    fusion_path = os.path.join(MODELS_DIR, "fusion_predictions.csv")
    uncertainty_path = os.path.join(UNCERTAINTY_DIR, "uncertainty.csv")
    selector_path = os.path.join(UNCERTAINTY_DIR, "selector_log.csv")

    while True:
        clear_screen()
        print_banner("Step 14: Interactive Publication Figure Generator & Time-Period Viewer")
        print(" [1] Generate ALL Default Figures (Saved to figures/*.png)")
        print(" [2] Interactive Forecast Comparison (Actual vs HW vs GRU vs Fused)")
        print(" [3] Interactive Uncertainty Signals & Error (U1, U2, U3 vs |Error|)")
        print(" [4] Interactive Risk-Aware Allocation vs Actual (Capacity + Grace Zone)")
        print(" [5] Interactive Selector Winning Signal Distribution Plot")
        print(" [6] Interactive Data Gate Inspection Plot (4 Machines)")
        print(" [B] Back to Main Menu")
        print("-" * 80)
        choice = input("Select an option (1-6, B): ").strip().upper()

        if choice == '1':
            run_script("figures.py", desc="Generating all default publication figures")
            pause()
        elif choice == '2':
            m_in = input("Enter Machine ID (or leave blank for default): ").strip()
            mid = int(m_in) if m_in.isdigit() else None
            sp = input("Enter Split (test/val, default test): ").strip().lower() or "test"
            print("\nTime Window Presets:")
            print("  [1] Full Split")
            print("  [2] First 24 Hours (288 bins)")
            print("  [3] First 3 Days (864 bins)")
            print("  [4] Custom Start Bin and End Bin")
            tw = input("Choose preset (1-4, default 1): ").strip()

            sb, eb = None, None
            if tw == '2':
                sb, eb = 0, 288
            elif tw == '3':
                sb, eb = 0, 864
            elif tw == '4':
                sb = int(input("  Start bin index (e.g. 100): ").strip() or "0")
                eb = int(input("  End bin index   (e.g. 500): ").strip() or "500")

            out_f = os.path.join(FIGURES_DIR, f"prediction_comparison_{mid or 'default'}_{sb or 0}_{eb or 'end'}.png")
            plot_prediction_comparison(fusion_path, out_f, machine_id=mid, start_bin=sb, end_bin=eb, split=sp)
            print(f"\n[Generated] Saved to: {out_f}")
            pause()
        elif choice == '3':
            m_in = input("Enter Machine ID (or leave blank for default): ").strip()
            mid = int(m_in) if m_in.isdigit() else None
            sp = input("Enter Split (test/val, default test): ").strip().lower() or "test"
            sb_in = input("Start bin (optional, default 0): ").strip()
            eb_in = input("End bin (optional, default end): ").strip()
            sb = int(sb_in) if sb_in.isdigit() else None
            eb = int(eb_in) if eb_in.isdigit() else None

            out_f = os.path.join(FIGURES_DIR, f"uncertainty_signals_{mid or 'default'}_{sb or 0}_{eb or 'end'}.png")
            plot_uncertainty_signals(uncertainty_path, out_f, machine_id=mid, start_bin=sb, end_bin=eb, split=sp)
            print(f"\n[Generated] Saved to: {out_f}")
            pause()
        elif choice == '4':
            m_in = input("Enter Machine ID (or leave blank for default): ").strip()
            mid = int(m_in) if m_in.isdigit() else None
            beta_in = input("Buffer multiplier beta (default calibrated beta*): ").strip()
            beta = float(beta_in) if beta_in else None
            sp = input("Enter Split (test/val, default test): ").strip().lower() or "test"
            sb_in = input("Start bin (optional, default 0): ").strip()
            eb_in = input("End bin (optional, default end): ").strip()
            sb = int(sb_in) if sb_in.isdigit() else None
            eb = int(eb_in) if eb_in.isdigit() else None

            out_f = os.path.join(FIGURES_DIR, f"allocation_{mid or 'default'}_beta{beta or 'calibrated'}.png")
            plot_allocation_vs_actual(selector_path, out_f, beta=beta, machine_id=mid, start_bin=sb, end_bin=eb, split=sp)
            print(f"\n[Generated] Saved to: {out_f}")
            pause()
        elif choice == '5':
            sp = input("Enter Split (test/val, default test): ").strip().lower() or "test"
            m_in = input("Machine ID (optional, leave blank for cluster-wide): ").strip()
            mid = int(m_in) if m_in.isdigit() else None
            out_f = os.path.join(FIGURES_DIR, f"selector_choices_{mid or 'all'}_{sp}.png")
            plot_selector_choices(selector_path, out_f, split=sp, machine_id=mid)
            print(f"\n[Generated] Saved to: {out_f}")
            pause()
        elif choice == '6':
            out_f = os.path.join(FIGURES_DIR, "data_validation_custom.png")
            plot_data_validation(output_path=out_f)
            print(f"\n[Generated] Saved to: {out_f}")
            pause()
        elif choice == 'B':
            break


# ==============================================================================
# Main Interactive Menu
# ==============================================================================

def interactive_main():
    while True:
        clear_screen()
        print_banner("Unified Interactive Orchestration & Exploration Menu")
        print(" Pipeline Steps:")
        print("   [1]  Step 1:  Download & Stream-Filtering (task_usage & machine_events)")
        print("   [2]  Step 2:  Interval-Weighted Aggregation & Data Gate Verification")
        print("   [3]  Step 3:  Machine Eligibility Audit & Sampling (100 benchmark machines)")
        print("   [4]  Step 4:  Causal Feature Extraction (Base 7 vs Extended 15 Hierarchy)")
        print("   [5]  Step 5:  Train / Val / Test Split & Custom Ratio Configuration")
        print("   [6]  Step 6:  Baseline Forecasting & Allocations (B0, B0b Seasonal, B0c MA)")
        print("   [7]  Step 7:  Holt-Winters Smoothing & Period Grid Search (B2)")
        print("   [8]  Step 8:  Global GRU Forecaster (Base vs Extended Tier Ablation)")
        print("   [9]  Step 9:  Convex Forecast Fusion (Validation-driven Alpha Calibration) (B3)")
        print("   [10] Step 10: Multi-Source Uncertainty Engine (U1, U2, U3 Signals)")
        print("   [11] Step 11: Adaptive Reliability Selector (Dynamic Signal Arbitration)")
        print("   [12] Step 12: Risk-Aware Allocation & Constrained Buffer Calibration")
        print("   [13] Step 13: Unified Benchmark Consolidation & Table Synchronization")
        print("   [14] Step 14: Interactive Publication Figure Generator & Time-Period Viewer")
        print(" " + "-" * 78)
        print(" Quick Actions & Diagnostics:")
        print("   [A]  Run Automated Pipeline End-to-End (Steps 6-14 Complete Execution)")
        print("   [T]  Run Full Automated Test Suite (12 Integrity Tests + 8 Math Tests)")
        print("   [S]  Run Machine-Subset Sensitivity Analysis (8-Run Subsample Matrix)")
        print("   [P]  Profile Computational Cost & Inference Latency Benchmark")
        print("   [V]  View Data Provenance & Run Configuration Metadata")
        print("   [Q]  Quit")
        print("=" * 80)
        choice = input("Enter selection: ").strip().upper()

        if choice == '1':
            menu_step_1_download()
        elif choice == '2':
            menu_step_2_aggregation()
        elif choice == '3':
            menu_step_3_eligibility()
        elif choice == '4':
            menu_step_4_features()
        elif choice == '5':
            menu_step_5_splits()
        elif choice == '6':
            menu_step_6_baselines()
        elif choice == '7':
            menu_step_7_holt_winters()
        elif choice == '8':
            menu_step_8_gru()
        elif choice == '9':
            menu_step_9_fusion()
        elif choice == '10':
            menu_step_10_uncertainty()
        elif choice == '11':
            menu_step_11_selector()
        elif choice == '12':
            menu_step_12_allocation()
        elif choice == '13':
            menu_step_13_evaluation()
        elif choice == '14':
            menu_step_14_figures()
        elif choice == 'A':
            print("\nLaunching Master Pipeline End-to-End...")
            run_script("run_pipeline.py", ["--all"], "Executing complete master pipeline")
            pause()
        elif choice == 'T':
            print("\nRunning Full Test Suite...")
            run_script("test_integrity.py", desc="Running 12 integrity and anti-leakage unit tests")
            run_script("test_aggregate.py", desc="Running 8 aggregation mathematical unit tests")
            pause()
        elif choice == 'S':
            print("\nRunning Machine-Subset Sensitivity Analysis...")
            run_script("robustness.py", desc="Executing 8-run subsample sensitivity sweep")
            pause()
        elif choice == 'P':
            print("\nProfiling Computational Latencies & Memory Footprint...")
            run_script("run_pipeline.py", ["--profile"], "Benchmarking model parameters & latency")
            pause()
        elif choice == 'V':
            prov_file = os.path.join(RESULTS_DIR, "data_provenance.json")
            conf_file = os.path.join(RESULTS_DIR, "run_config.json")
            if os.path.exists(prov_file) and os.path.exists(conf_file):
                print("\n=== Data Provenance Record ===")
                with open(prov_file, 'r') as f:
                    print(f.read())
                print("\n=== Run Configuration Record ===")
                with open(conf_file, 'r') as f:
                    print(f.read())
            else:
                print("\nProvenance files not found.")
            pause()
        elif choice in ('Q', 'EXIT'):
            print("\nExiting. Thank you!")
            break


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Master Orchestrator and Exploration Suite")
    parser.add_argument("--step", type=int, help="Jump directly into a specific step (1-14)")
    parser.add_argument("--all", action="store_true", help="Execute complete pipeline non-interactively")
    parser.add_argument("--test", action="store_true", help="Run automated test suite non-interactively")
    args = parser.parse_args()

    if args.all:
        run_script("run_pipeline.py", ["--all"], "Executing complete master pipeline non-interactively")
    elif args.test:
        run_script("test_integrity.py", desc="Running integrity tests")
        run_script("test_aggregate.py", desc="Running aggregation tests")
    elif args.step == 1:
        menu_step_1_download()
    elif args.step == 2:
        menu_step_2_aggregation()
    elif args.step == 3:
        menu_step_3_eligibility()
    elif args.step == 4:
        menu_step_4_features()
    elif args.step == 5:
        menu_step_5_splits()
    elif args.step == 6:
        menu_step_6_baselines()
    elif args.step == 7:
        menu_step_7_holt_winters()
    elif args.step == 8:
        menu_step_8_gru()
    elif args.step == 9:
        menu_step_9_fusion()
    elif args.step == 10:
        menu_step_10_uncertainty()
    elif args.step == 11:
        menu_step_11_selector()
    elif args.step == 12:
        menu_step_12_allocation()
    elif args.step == 13:
        menu_step_13_evaluation()
    elif args.step == 14:
        menu_step_14_figures()
    else:
        interactive_main()


if __name__ == "__main__":
    main()
