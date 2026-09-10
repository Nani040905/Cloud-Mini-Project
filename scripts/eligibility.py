"""
scripts/eligibility.py

Step 3: Machine Eligibility Check and Deterministic Sampling for Google Cluster Trace v2.
Implements:
1. Evaluates all aggregated machine series in data/processed/google/machine_series/.
2. Enforces strict eligibility criteria:
   - Missing fraction: NaN <= 20%
   - Variance: std > 0.01 (non-flat workload)
   - Physical validity: values in [0.0, 1.0]
3. Deterministically samples 100 eligible machines (seed=42).
4. Saves selected machine IDs to data/selected_machines.json.
5. Generates figures/data_validation.png visual check for inspection.
"""

import os
import glob
import json
import argparse
import numpy as np
import pandas as pd

# Default configurations locked in PROJECT_SPEC.md
SEED = 42
N_FINAL = 100
MAX_NAN_FRAC = 0.20
MIN_STD = 0.01


def evaluate_machine_series(file_path: str):
    """
    Evaluate a single machine's aggregated series.
    Returns: dict of stats and boolean eligibility.
    """
    df = pd.read_parquet(file_path)
    mid = os.path.splitext(os.path.basename(file_path))[0]

    cpu = df['cpu']
    nan_frac = float(cpu.isna().mean())
    valid_cpu = cpu.dropna()

    if len(valid_cpu) == 0:
        return {
            'machine_id': mid,
            'total_bins': len(df),
            'nan_frac': nan_frac,
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'eligible': False,
            'reason': 'all_nan'
        }

    c_mean = float(valid_cpu.mean())
    c_std = float(valid_cpu.std()) if len(valid_cpu) > 1 else 0.0
    c_min = float(valid_cpu.min())
    c_max = float(valid_cpu.max())

    # Range check assertions
    if c_min < -1e-6:
        raise ValueError(f"Machine {mid} has negative CPU: {c_min}")
    if c_max > 1.0 + 1e-6:
        raise ValueError(f"Machine {mid} exceeds physical ceiling 1.0: {c_max}")

    # Eligibility evaluation
    reasons = []
    if nan_frac > MAX_NAN_FRAC:
        reasons.append(f"high_nan({nan_frac:.1%}>{MAX_NAN_FRAC:.0%})")
    if c_std <= MIN_STD:
        reasons.append(f"low_variance(std={c_std:.4f}<={MIN_STD})")

    eligible = len(reasons) == 0

    return {
        'machine_id': mid,
        'total_bins': len(df),
        'nan_frac': nan_frac,
        'mean': c_mean,
        'std': c_std,
        'min': c_min,
        'max': c_max,
        'eligible': eligible,
        'reason': '; '.join(reasons) if reasons else 'eligible'
    }


def plot_validation(selected_files: list, output_path: str):
    """Generate figures/data_validation.png visual inspection plot for 4 machines."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Warning] matplotlib not installed. Skipping plot generation.")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sample_files = selected_files[:4]
    if not sample_files:
        return

    fig, axes = plt.subplots(len(sample_files), 1, figsize=(14, 2.5 * len(sample_files)), sharex=True)
    if len(sample_files) == 1:
        axes = [axes]

    for ax, fpath in zip(axes, sample_files):
        df = pd.read_parquet(fpath)
        mid = os.path.splitext(os.path.basename(fpath))[0]
        cpu_vals = df['cpu'].values
        time_hours = df['timestamp_sec'] / 3600.0 if 'timestamp_sec' in df else np.arange(len(df)) * 5.0 / 60.0

        ax.plot(time_hours, cpu_vals, lw=1.2, color='#1f77b4', label='Aggregated CPU')
        ax.axhline(0.0, color='gray', linestyle=':', alpha=0.5)
        ax.set_title(f"Machine {mid} — Mean: {np.nanmean(cpu_vals):.3f}, Std: {np.nanstd(cpu_vals):.3f}, NaN: {np.isnan(cpu_vals).mean():.1%}", fontsize=11, fontweight='bold')
        ax.set_ylabel("CPU Util [0, 1]")
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

    axes[-1].set_xlabel("Elapsed Time (hours)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved visual data validation plot to: {output_path}")


def run_eligibility(series_dir: str, output_json: str, plot_path: str, n_final: int = N_FINAL, seed: int = SEED):
    """Run eligibility audit and sample final machines."""
    parquet_files = sorted(glob.glob(os.path.join(series_dir, "*.parquet")))
    if not parquet_files:
        print(f"No series files found in {series_dir}")
        return

    print(f"Auditing eligibility for {len(parquet_files)} candidate machines...")
    results = [evaluate_machine_series(fp) for fp in parquet_files]
    results_df = pd.DataFrame(results)

    total_candidates = len(results_df)
    eligible_df = results_df[results_df['eligible']]
    n_eligible = len(eligible_df)

    rejected_nan = (results_df['nan_frac'] > MAX_NAN_FRAC).sum()
    rejected_std = (results_df['std'] <= MIN_STD).sum()

    print("\n" + "=" * 55)
    print("           ELIGIBILITY AUDIT REPORT")
    print("=" * 55)
    print(f"Total candidate series audited : {total_candidates}")
    print(f"Eligible machines passing all  : {n_eligible} ({n_eligible / total_candidates:.1%})")
    print(f"Rejected: NaN fraction > {MAX_NAN_FRAC:.0%}   : {rejected_nan}")
    print(f"Rejected: Workload std <= {MIN_STD} : {rejected_std}")
    print("=" * 55)

    if n_eligible < n_final:
        print(f"\n[Notice] Eligible count ({n_eligible}) is less than target sample ({n_final}).")
        print("Sampling all available eligible machines for current data batch.")
        chosen_mids = sorted(eligible_df['machine_id'].tolist())
    else:
        # Deterministic sampling with fixed seed
        rng = np.random.default_rng(seed)
        eligible_mids = sorted(eligible_df['machine_id'].tolist())
        chosen_mids = sorted(rng.choice(eligible_mids, size=n_final, replace=False).tolist())
        print(f"\nDeterministically sampled {len(chosen_mids)} eligible machines (seed={seed}).")

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(chosen_mids, f, indent=2)
    print(f"Saved {len(chosen_mids)} machine IDs to: {output_json}")

    # Generate validation plot for first 4 selected machines
    chosen_files = [os.path.join(series_dir, f"{mid}.parquet") for mid in chosen_mids]
    if plot_path:
        plot_validation(chosen_files, plot_path)

    # Print summary statistics of chosen sample
    chosen_df = results_df[results_df['machine_id'].isin(chosen_mids)]
    print("\nSummary statistics of selected machines:")
    print(chosen_df[['nan_frac', 'mean', 'std', 'min', 'max']].describe().to_string())
    print("\nEligibility check complete [OK]")


def main():
    parser = argparse.ArgumentParser(description="Machine eligibility audit and sampling")
    parser.add_argument("--series_dir", type=str, default="data/processed/google/machine_series")
    parser.add_argument("--output_json", type=str, default="data/selected_machines.json")
    parser.add_argument("--plot", type=str, default="figures/data_validation.png")
    parser.add_argument("--n_final", type=int, default=N_FINAL)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    series_dir = os.path.join(project_root, args.series_dir)
    output_json = os.path.join(project_root, args.output_json)
    plot_path = os.path.join(project_root, args.plot) if args.plot else None

    run_eligibility(series_dir, output_json, plot_path, n_final=args.n_final, seed=args.seed)


if __name__ == "__main__":
    main()
