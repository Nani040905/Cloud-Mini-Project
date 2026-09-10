"""
scripts/aggregate.py

Step 2: Correct Interval-Weighted CPU Aggregation for Google Cluster Trace v2.
Implements:
1. Stage 1 reference implementation for correctness proof.
2. Stage 2 vectorized implementation with numpy broadcasting for scale.
3. Multi-resource support: CPU rate, canonical memory usage, and disk I/O time.
4. Idle (0.0) vs. Missing (NaN) distinction via touched_bins tracking.
5. Causal gap filling: linear interpolation for <= 2 consecutive NaNs; > 2 remain NaN.
6. Comprehensive unit tests verifying math against synthetic tasks.
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd

BIN_SECONDS = 300.0  # 5 minutes in seconds


def aggregate_machine_cpu_reference(task_df: pd.DataFrame, total_windows: int, start_epoch_sec: float = 0.0):
    """
    REFERENCE implementation: Slow, readable loop to prove mathematical correctness.
    task_df requires: start_sec, end_sec, cpu_rate
    Returns: (cpu_series, touched_bins)
    """
    cpu_series   = np.zeros(total_windows, dtype=np.float64)
    touched_bins = np.zeros(total_windows, dtype=bool)

    for _, row in task_df.iterrows():
        t_start = row['start_sec'] - start_epoch_sec
        t_end   = row['end_sec']   - start_epoch_sec
        rate    = row['cpu_rate']

        if t_end <= 0 or t_start >= total_windows * BIN_SECONDS:
            continue

        bin_first = max(0, int(t_start // BIN_SECONDS))
        bin_last  = min(total_windows - 1, int(t_end // BIN_SECONDS))

        for w in range(bin_first, bin_last + 1):
            win_lo  = w * BIN_SECONDS
            win_hi  = win_lo + BIN_SECONDS
            overlap = max(0.0, min(t_end, win_hi) - max(t_start, win_lo))
            if overlap > 0.0:
                cpu_series[w]   += rate * (overlap / BIN_SECONDS)
                touched_bins[w]  = True

    return np.clip(cpu_series, 0.0, 1.0), touched_bins


def aggregate_machine_vectorized(task_df: pd.DataFrame, total_windows: int, start_epoch_sec: float = 0.0):
    """
    VECTORIZED implementation: Optimized for scale with numpy array slicing.
    Computes CPU rate, canonical memory usage, and disk I/O time simultaneously.
    Returns: dict of {
        'cpu': np.ndarray,
        'mem': np.ndarray,
        'disk': np.ndarray,
        'touched': np.ndarray (bool)
    }
    """
    cpu_series   = np.zeros(total_windows, dtype=np.float64)
    mem_series   = np.zeros(total_windows, dtype=np.float64)
    disk_series  = np.zeros(total_windows, dtype=np.float64)
    touched_bins = np.zeros(total_windows, dtype=bool)

    if len(task_df) == 0:
        return {
            'cpu': cpu_series,
            'mem': mem_series,
            'disk': disk_series,
            'touched': touched_bins
        }

    starts = task_df['start_sec'].values - start_epoch_sec
    ends   = task_df['end_sec'].values   - start_epoch_sec
    cpus   = task_df['cpu_rate'].values

    has_mem  = 'canonical_memory_usage' in task_df.columns
    has_disk = 'disk_io_time' in task_df.columns

    mems  = task_df['canonical_memory_usage'].values if has_mem else np.zeros(len(starts))
    disks = task_df['disk_io_time'].values if has_disk else np.zeros(len(starts))

    bin_edges = np.arange(total_windows + 1, dtype=np.float64) * BIN_SECONDS

    for i in range(len(starts)):
        t_s = starts[i]
        t_e = ends[i]

        if t_e <= 0 or t_s >= total_windows * BIN_SECONDS:
            continue

        b0 = max(0, int(t_s // BIN_SECONDS))
        b1 = min(total_windows, int(t_e // BIN_SECONDS) + 1)

        if b0 >= b1:
            continue

        w_lo = bin_edges[b0:b1]
        w_hi = bin_edges[b0+1:b1+1]

        overlaps = np.maximum(0.0, np.minimum(t_e, w_hi) - np.maximum(t_s, w_lo))
        weights  = overlaps / BIN_SECONDS

        cpu_series[b0:b1]  += cpus[i]  * weights
        mem_series[b0:b1]  += mems[i]  * weights
        disk_series[b0:b1] += disks[i] * weights
        touched_bins[b0:b1] |= (overlaps > 0.0)

    return {
        'cpu': np.clip(cpu_series, 0.0, 1.0),
        'mem': np.clip(mem_series, 0.0, 1.0),
        'disk': np.clip(disk_series, 0.0, 1.0),
        'touched': touched_bins
    }


def clean_and_fill_gaps(series: np.ndarray, touched: np.ndarray, max_gap: int = 2) -> pd.Series:
    """
    Distinguishes idle from missing, then applies causal gap filling:
    - touched == False -> NaN (genuinely missing task records)
    - touched == True and value == 0.0 -> 0.0 (observed idle)
    - gaps <= max_gap (2 steps = 10 min) -> linearly interpolated
    - gaps > max_gap -> remain NaN
    """
    s = pd.Series(series, dtype=np.float64)
    s[~touched] = np.nan
    if max_gap <= 0:
        return s
    s_clean = s.interpolate(method='linear', limit=max_gap, limit_direction='forward')
    return s_clean


def run_unit_tests():
    """Verify correctness on synthetic tasks specified in PROJECT_SPEC.md."""
    print("--- Running Unit Tests for Interval-Weighted Aggregation ---")

    # Test Case 1: Synthetic single task spanning part of bin 1
    # starts at 301s (1s into bin 1), ends at 598s (2s before bin 1 ends)
    # Total duration = 297s, cpu_rate = 0.4
    # Expected bin 1 = 0.4 * 297 / 300 = 0.396
    test_task = pd.DataFrame([{
        'start_sec': 301.0,
        'end_sec':   598.0,
        'cpu_rate':  0.4,
        'canonical_memory_usage': 0.2,
        'disk_io_time': 0.05
    }])

    total_win = 5
    cpu_ref, touched_ref = aggregate_machine_cpu_reference(test_task, total_windows=total_win)
    res_vec = aggregate_machine_vectorized(test_task, total_windows=total_win)

    expected_b1 = 0.4 * 297.0 / 300.0
    assert abs(cpu_ref[1] - expected_b1) < 1e-7, f"Ref CPU error: {cpu_ref[1]} vs {expected_b1}"
    assert abs(res_vec['cpu'][1] - expected_b1) < 1e-7, f"Vec CPU error: {res_vec['cpu'][1]} vs {expected_b1}"
    assert np.all(touched_ref == res_vec['touched']), "Touched mask mismatch"
    assert np.array_equal(touched_ref, [False, True, False, False, False]), "Touched bins wrong"

    # Test Case 2: Multi-bin spanning task
    # starts at 150s (middle of bin 0), ends at 750s (middle of bin 2)
    # Bin 0: [0, 300) -> overlap 150s -> rate * 150/300 = 0.5 * 0.8 = 0.4
    # Bin 1: [300, 600) -> overlap 300s -> rate * 300/300 = 1.0 * 0.8 = 0.8
    # Bin 2: [600, 900) -> overlap 150s -> rate * 150/300 = 0.5 * 0.8 = 0.4
    test_task2 = pd.DataFrame([{
        'start_sec': 150.0,
        'end_sec':   750.0,
        'cpu_rate':  0.8,
        'canonical_memory_usage': 0.5,
        'disk_io_time': 0.1
    }])
    cpu_ref2, touched_ref2 = aggregate_machine_cpu_reference(test_task2, total_windows=total_win)
    res_vec2 = aggregate_machine_vectorized(test_task2, total_windows=total_win)

    assert abs(cpu_ref2[0] - 0.4) < 1e-7
    assert abs(cpu_ref2[1] - 0.8) < 1e-7
    assert abs(cpu_ref2[2] - 0.4) < 1e-7
    assert np.allclose(cpu_ref2, res_vec2['cpu']), "Vectorized vs reference mismatch on multi-bin"

    # Test Case 3: Idle vs Missing distinction
    # If bin has task with cpu_rate=0.0 -> touched=True, value=0.0 (idle)
    # If bin has no task -> touched=False -> converted to NaN
    test_task3 = pd.DataFrame([
        {'start_sec': 0.0,   'end_sec': 300.0, 'cpu_rate': 0.0},  # idle in bin 0
        # bin 1 is completely empty (no task) -> missing
        {'start_sec': 600.0, 'end_sec': 900.0, 'cpu_rate': 0.5}   # active in bin 2
    ])
    res_vec3 = aggregate_machine_vectorized(test_task3, total_windows=3)
    s_cleaned = clean_and_fill_gaps(res_vec3['cpu'], res_vec3['touched'], max_gap=2)

    assert res_vec3['touched'][0] == True, "Bin 0 must be touched (idle)"
    assert res_vec3['touched'][1] == False, "Bin 1 must be untouched (missing)"
    assert s_cleaned.iloc[0] == 0.0, "Bin 0 must be 0.0 (idle preserved)"
    assert s_cleaned.iloc[1] == 0.25, "Bin 1 must be interpolated (0.0 to 0.5)"
    assert s_cleaned.iloc[2] == 0.5, "Bin 2 must be 0.5"

    print("ALL UNIT TESTS PASSED: Reference, Vectorized, Multi-bin, and Idle/Missing logic verified [OK]\n")


def aggregate_existing_parquets(filtered_dir: str, output_dir: str):
    """Aggregate all currently available filtered parquet files."""
    parquet_files = sorted(glob.glob(os.path.join(filtered_dir, "*.parquet")))
    if not parquet_files:
        print(f"No parquet files found in {filtered_dir}")
        return

    print(f"Loading {len(parquet_files)} parquet chunks for aggregation...")
    df_list = [pd.read_parquet(f) for f in parquet_files]
    full_df = pd.concat(df_list, ignore_index=True)

    full_df = full_df.dropna(subset=['machine_id']).copy()
    full_df['machine_id'] = full_df['machine_id'].astype(np.int64)

    print(f"Total task records: {len(full_df):,}")
    print(f"Unique machines: {full_df['machine_id'].nunique()}")

    # Convert timestamps from microseconds to seconds
    full_df['start_sec'] = full_df['start_time'] / 1_000_000.0
    full_df['end_sec']   = full_df['end_time']   / 1_000_000.0

    min_sec = full_df['start_sec'].min()
    max_sec = full_df['end_sec'].max()
    total_windows = int(np.ceil((max_sec - min_sec) / BIN_SECONDS))
    print(f"Time span: {min_sec:.1f}s to {max_sec:.1f}s ({total_windows} 5-minute bins)")

    series_dir = os.path.join(output_dir, "machine_series")
    os.makedirs(series_dir, exist_ok=True)
    bin_timestamps = min_sec + np.arange(total_windows) * BIN_SECONDS
    summary_stats = []

    from tqdm import tqdm
    for mid, group in tqdm(full_df.groupby('machine_id'), desc="Aggregating machines"):
        res = aggregate_machine_vectorized(group, total_windows=total_windows, start_epoch_sec=min_sec)
        clean_cpu  = clean_and_fill_gaps(res['cpu'], res['touched'], max_gap=2)
        clean_mem  = clean_and_fill_gaps(res['mem'], res['touched'], max_gap=2)
        clean_disk = clean_and_fill_gaps(res['disk'], res['touched'], max_gap=2)

        m_series_df = pd.DataFrame({
            'bin_idx': np.arange(total_windows),
            'timestamp_sec': bin_timestamps,
            'cpu': clean_cpu.values,
            'mem': clean_mem.values,
            'disk': clean_disk.values,
            'touched': res['touched']
        })
        m_series_df.to_parquet(os.path.join(series_dir, f"{int(mid)}.parquet"), index=False)

        nan_pct = clean_cpu.isna().mean()
        mean_cpu = clean_cpu.dropna().mean() if len(clean_cpu.dropna()) > 0 else 0.0
        std_cpu  = clean_cpu.dropna().std() if len(clean_cpu.dropna()) > 1 else 0.0

        summary_stats.append({
            'machine_id': int(mid),
            'records': len(group),
            'nan_fraction': nan_pct,
            'cpu_mean': mean_cpu,
            'cpu_std': std_cpu
        })

    summary_df = pd.DataFrame(summary_stats)
    summary_path = os.path.join(output_dir, "aggregation_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nAggregation completed for {len(summary_df)} machines.")
    print(f"Per-machine series saved to: {series_dir}")
    print(f"Summary saved to: {summary_path}")

    # Generate and save formal Dataset Coverage Report (Data Gate)
    all_cpus = []
    for f_path in glob.glob(os.path.join(series_dir, "*.parquet")):
        m_df = pd.read_parquet(f_path)
        all_cpus.append(m_df['cpu'].values)

    all_cpus_arr = np.concatenate(all_cpus) if all_cpus else np.array([])
    valid_cpus = all_cpus_arr[~np.isnan(all_cpus_arr)]
    nan_frac = np.isnan(all_cpus_arr).mean() if len(all_cpus_arr) > 0 else 0.0
    idle_frac = (valid_cpus == 0.0).mean() if len(valid_cpus) > 0 else 0.0

    duration_sec = max_sec - min_sec
    duration_days = duration_sec / 86400.0
    duration_hours = (duration_sec % 86400.0) / 3600.0
    duration_mins = (duration_sec % 3600.0) / 60.0

    report_lines = [
        "=" * 52,
        "          DATASET COVERAGE REPORT",
        "=" * 52,
        f"Task records total         : {len(full_df):,}",
        f"Unique machines in trace   : {full_df['machine_id'].nunique():,}",
        f"Temporal bins (5-min)      : {total_windows:,}",
        f"Earliest timestamp         : {min_sec:.1f} s",
        f"Latest timestamp           : {max_sec:.1f} s",
        f"Coverage duration          : {int(duration_days)} days, {int(duration_hours)} hours, {int(duration_mins)} minutes",
        "-" * 52,
        "CPU stats (across active machines):",
        f"  Mean                     : {valid_cpus.mean():.4f}" if len(valid_cpus) > 0 else "  Mean: N/A",
        f"  Std                      : {valid_cpus.std():.4f}" if len(valid_cpus) > 0 else "  Std: N/A",
        f"  Min                      : {valid_cpus.min():.4f}" if len(valid_cpus) > 0 else "  Min: N/A",
        f"  Max                      : {valid_cpus.max():.4f}" if len(valid_cpus) > 0 else "  Max: N/A",
        f"NaN fraction               : {nan_frac:.2%}",
        f"Zero/idle fraction         : {idle_frac:.2%}",
        "=" * 52
    ]
    report_str = "\n".join(report_lines)
    print("\n" + report_str)

    report_path = os.path.join(output_dir, "dataset_coverage_report.txt")
    with open(report_path, "w") as f:
        f.write(report_str + "\n")


def main():
    parser = argparse.ArgumentParser(description="CPU and multi-resource interval-weighted aggregation")
    parser.add_argument("--test", action="store_true", help="Run synthetic verification unit tests")
    parser.add_argument("--aggregate", action="store_true", help="Aggregate all existing filtered parquet parts")
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    filtered_dir = os.path.join(project_root, "data", "processed", "google", "filtered_parts")
    output_dir   = os.path.join(project_root, "data", "processed", "google")

    if args.test or not args.aggregate:
        run_unit_tests()

    if args.aggregate:
        aggregate_existing_parquets(filtered_dir, output_dir)


if __name__ == "__main__":
    main()
