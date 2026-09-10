"""
run_pipeline.py

Master Pipeline Runner for Cloud Workload Forecasting and Risk-Aware Allocation.
Coordinates end-to-end execution of all benchmark baselines and proposed components:
  Step 6:  Baseline Forecasting & Allocations (B0: Naive, B0b: Seasonal Naive)
  Step 7:  Holt-Winters Exponential Smoothing & B2 Baseline
  Step 8:  Global GRU Deep Learning Forecaster (Dual-Tier: Base & Extended)
  Step 9:  Convex Forecast Fusion & B3 Baseline
  Step 10: Multi-Source Uncertainty Engine (U1: Disagreement, U2: EWMA, U3: Volatility)
  Step 11: Adaptive Reliability Selector (Dynamic Signal Arbitration)
  Step 12: Risk-Aware Capacity Allocation & Constrained Buffer Calibration (B4a-c, Proposed)
  Step 13: Final Consolidated Benchmark Evaluation (results/final_results.csv)
  Step 14: Publication-Quality Figure Generation (figures/*.png)

Usage:
  python run_pipeline.py --all
  python run_pipeline.py --steps 6 7 8 9 10 11 12 13 14
  python run_pipeline.py --eval-only
"""

import os
import sys
import time
import argparse
import subprocess

# Ensure scripts directory is in python path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)


STEPS = {
    6: ("Baselines B0 & B0b", [sys.executable, os.path.join(SCRIPTS_DIR, "baselines.py")]),
    7: ("Holt-Winters B2", [sys.executable, os.path.join(SCRIPTS_DIR, "holt_winters.py")]),
    8: ("Global GRU (Base & Extended)", None),  # handled specially to run base and extended
    9: ("Forecast Fusion B3", [sys.executable, os.path.join(SCRIPTS_DIR, "fusion.py")]),
    10: ("Uncertainty Engine (U1, U2, U3)", [sys.executable, os.path.join(SCRIPTS_DIR, "uncertainty.py")]),
    11: ("Adaptive Reliability Selector", [sys.executable, os.path.join(SCRIPTS_DIR, "selector.py")]),
    12: ("Risk-Aware Capacity Allocation", [sys.executable, os.path.join(SCRIPTS_DIR, "allocation.py")]),
    13: ("Benchmark Consolidation", [sys.executable, os.path.join(SCRIPTS_DIR, "evaluate.py")]),
    14: ("Publication Figures", [sys.executable, os.path.join(SCRIPTS_DIR, "figures.py")]),
}


def run_command(cmd_list, desc):
    """Executes a subprocess command with timing and status reporting."""
    t0 = time.time()
    print(f"\n>>> [{desc}] Starting...")
    res = subprocess.run(cmd_list, cwd=PROJECT_ROOT)
    elapsed = time.time() - t0
    if res.returncode != 0:
        print(f"[ERROR] Step failed with return code {res.returncode}: {desc}")
        sys.exit(res.returncode)
    print(f">>> [{desc}] Completed in {elapsed:.2f} seconds.")
    return elapsed


def execute_pipeline(steps_to_run):
    """Executes selected pipeline steps sequentially."""
    print("=" * 70)
    print("      CLOUD RESOURCE ALLOCATION PIPELINE — REPRODUCIBILITY RUNNER")
    print("=" * 70)
    print(f"Project Root   : {PROJECT_ROOT}")
    print(f"Python Exec    : {sys.executable}")
    print(f"Steps Selected : {sorted(steps_to_run)}")
    print("=" * 70)

    total_start = time.time()
    timing_records = {}

    for step_num in sorted(steps_to_run):
        if step_num not in STEPS:
            print(f"[Warning] Unknown step number: {step_num}")
            continue

        name, cmd = STEPS[step_num]

        if step_num == 8:
            # Run Base tier then Extended tier
            t_base = run_command([sys.executable, os.path.join(SCRIPTS_DIR, "gru.py"), "--tier", "base"], "Global GRU — Base Tier (7 features)")
            t_ext = run_command([sys.executable, os.path.join(SCRIPTS_DIR, "gru.py"), "--tier", "extended"], "Global GRU — Extended Tier (15 features)")
            timing_records[step_num] = t_base + t_ext
        else:
            t_step = run_command(cmd, name)
            timing_records[step_num] = t_step

    total_elapsed = time.time() - total_start

    print("\n" + "=" * 70)
    print("                   PIPELINE EXECUTION SUMMARY")
    print("=" * 70)
    for step_num, elapsed in timing_records.items():
        step_name = STEPS[step_num][0]
        print(f"Step {step_num:2d} : {step_name:38s} | {elapsed:7.2f} s")
    print("-" * 70)
    print(f"Total Pipeline Runtime: {total_elapsed:.2f} seconds ({total_elapsed / 60.0:.2f} minutes)")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Master Pipeline Runner for Cloud Allocation Benchmarks")
    parser.add_argument("--all", action="store_true", help="Execute complete Phase 1 benchmark suite (Steps 6-14)")
    parser.add_argument("--steps", nargs="+", type=int, help="Specific step numbers to execute (e.g. --steps 6 7 8)")
    parser.add_argument("--eval-only", action="store_true", help="Run only Steps 12, 13, and 14")
    args = parser.parse_args()

    if args.all:
        steps_to_run = list(STEPS.keys())
    elif args.eval_only:
        steps_to_run = [12, 13, 14]
    elif args.steps:
        steps_to_run = args.steps
    else:
        # Default to full suite
        steps_to_run = list(STEPS.keys())

    execute_pipeline(steps_to_run)


if __name__ == "__main__":
    main()
