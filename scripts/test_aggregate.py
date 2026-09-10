"""
scripts/test_aggregate.py

Formal Unit Test Suite for Interval-Weighted CPU Aggregation.
Verifies the mathematical correctness of:
  CPU(m, t) = sum(CPU_i * overlap(task_i, bin_t)) / 300.0
against both reference and vectorized implementations across 8 critical edge cases:
  Case A: Task entirely inside bin (overlap = task_duration)
  Case B: Task crosses one bin boundary (overlap splits cleanly at boundary)
  Case C: Task crosses multiple bins (each bin receives exact proportional weight)
  Case D: Task ends exactly at bin boundary (included in bin, zero in next)
  Case E: Task starts exactly at bin boundary (zero in previous bin, full in next)
  Case F: Multiple simultaneous overlapping tasks (CPU rates sum, divided by 300s)
  Case G: No tasks in bin (marked untouched -> converted to NaN, distinct from idle)
  Case H: Tasks recorded at zero CPU rate (marked touched -> preserved as 0.0 idle)
"""

import sys
import os
import numpy as np
import pandas as pd

# Add scripts directory to path
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from aggregate import (
    aggregate_machine_cpu_reference,
    aggregate_machine_vectorized,
    clean_and_fill_gaps,
    BIN_SECONDS
)

TOL = 1e-7


def test_case_a():
    """Case A: Task entirely inside bin -> overlap = task_duration."""
    # Bin 0 is [0, 300s). Task from 50s to 200s (duration = 150s) at rate 0.6.
    # Expected bin 0 CPU = 0.6 * (150 / 300) = 0.30
    df = pd.DataFrame([{
        'start_sec': 50.0, 'end_sec': 200.0, 'cpu_rate': 0.6,
        'canonical_memory_usage': 0.3, 'disk_io_time': 0.1
    }])
    ref_cpu, ref_touched = aggregate_machine_cpu_reference(df, total_windows=2)
    vec_res = aggregate_machine_vectorized(df, total_windows=2)

    expected = [0.6 * 150.0 / 300.0, 0.0]
    assert abs(ref_cpu[0] - expected[0]) < TOL, f"Ref failed: {ref_cpu[0]} != {expected[0]}"
    assert abs(vec_res['cpu'][0] - expected[0]) < TOL, f"Vec failed: {vec_res['cpu'][0]} != {expected[0]}"
    assert ref_touched[0] == True and ref_touched[1] == False
    assert np.array_equal(ref_touched, vec_res['touched'])
    print("  [PASS] Case A: Task entirely inside bin (overlap = task_duration)")


def test_case_b():
    """Case B: Task crosses one bin boundary -> overlap splits correctly at boundary."""
    # Bin 0: [0, 300s), Bin 1: [300, 600s).
    # Task from 200s to 450s (rate 0.8).
    # Overlap bin 0 = 300 - 200 = 100s -> CPU = 0.8 * 100/300 = 0.8 / 3 = 0.2666667
    # Overlap bin 1 = 450 - 300 = 150s -> CPU = 0.8 * 150/300 = 0.4000000
    df = pd.DataFrame([{
        'start_sec': 200.0, 'end_sec': 450.0, 'cpu_rate': 0.8
    }])
    ref_cpu, ref_touched = aggregate_machine_cpu_reference(df, total_windows=2)
    vec_res = aggregate_machine_vectorized(df, total_windows=2)

    exp_0 = 0.8 * 100.0 / 300.0
    exp_1 = 0.8 * 150.0 / 300.0
    assert abs(ref_cpu[0] - exp_0) < TOL
    assert abs(ref_cpu[1] - exp_1) < TOL
    assert abs(vec_res['cpu'][0] - exp_0) < TOL
    assert abs(vec_res['cpu'][1] - exp_1) < TOL
    assert ref_touched[0] and ref_touched[1]
    print("  [PASS] Case B: Task crosses one bin boundary (overlap splits cleanly at boundary)")


def test_case_c():
    """Case C: Task crosses multiple bins -> each bin receives proportional fraction."""
    # Bin 0: [0, 300), Bin 1: [300, 600), Bin 2: [600, 900), Bin 3: [900, 1200)
    # Task from 150s to 1050s at rate 0.5.
    # Bin 0: 150s overlap -> 0.5 * 150/300 = 0.25
    # Bin 1: 300s overlap -> 0.5 * 300/300 = 0.50
    # Bin 2: 300s overlap -> 0.5 * 300/300 = 0.50
    # Bin 3: 150s overlap -> 0.5 * 150/300 = 0.25
    df = pd.DataFrame([{
        'start_sec': 150.0, 'end_sec': 1050.0, 'cpu_rate': 0.5
    }])
    ref_cpu, ref_touched = aggregate_machine_cpu_reference(df, total_windows=4)
    vec_res = aggregate_machine_vectorized(df, total_windows=4)

    expected = [0.25, 0.50, 0.50, 0.25]
    for b in range(4):
        assert abs(ref_cpu[b] - expected[b]) < TOL, f"Bin {b} ref error"
        assert abs(vec_res['cpu'][b] - expected[b]) < TOL, f"Bin {b} vec error"
        assert ref_touched[b] == True
    print("  [PASS] Case C: Task crosses multiple bins (proportional fraction verified)")


def test_case_d():
    """Case D: Task ends exactly at bin boundary -> included in bin, zero in next."""
    # Bin 0: [0, 300), Bin 1: [300, 600)
    # Task from 100s to 300s (rate 0.45)
    # Overlap bin 0 = 200s -> 0.45 * 200/300 = 0.30
    # Overlap bin 1 = 0s -> 0.0, untouched
    df = pd.DataFrame([{
        'start_sec': 100.0, 'end_sec': 300.0, 'cpu_rate': 0.45
    }])
    ref_cpu, ref_touched = aggregate_machine_cpu_reference(df, total_windows=2)
    vec_res = aggregate_machine_vectorized(df, total_windows=2)

    assert abs(ref_cpu[0] - 0.30) < TOL
    assert abs(ref_cpu[1] - 0.00) < TOL
    assert abs(vec_res['cpu'][0] - 0.30) < TOL
    assert abs(vec_res['cpu'][1] - 0.00) < TOL
    assert ref_touched[0] == True
    assert ref_touched[1] == False, "Bin 1 must NOT be touched when task ends at boundary"
    assert vec_res['touched'][1] == False
    print("  [PASS] Case D: Task ends exactly at bin boundary (included in bin, zero in next)")


def test_case_e():
    """Case E: Task starts exactly at bin boundary -> zero in previous bin, full in next."""
    # Bin 0: [0, 300), Bin 1: [300, 600)
    # Task from 300s to 600s (full bin 1) at rate 0.70
    # Bin 0: 0s -> 0.0, untouched
    # Bin 1: 300s -> 0.70, touched
    df = pd.DataFrame([{
        'start_sec': 300.0, 'end_sec': 600.0, 'cpu_rate': 0.70
    }])
    ref_cpu, ref_touched = aggregate_machine_cpu_reference(df, total_windows=2)
    vec_res = aggregate_machine_vectorized(df, total_windows=2)

    assert abs(ref_cpu[0] - 0.00) < TOL
    assert abs(ref_cpu[1] - 0.70) < TOL
    assert abs(vec_res['cpu'][0] - 0.00) < TOL
    assert abs(vec_res['cpu'][1] - 0.70) < TOL
    assert ref_touched[0] == False
    assert ref_touched[1] == True
    assert vec_res['touched'][0] == False
    print("  [PASS] Case E: Task starts exactly at bin boundary (zero in previous, full in next)")


def test_case_f():
    """Case F: Multiple simultaneous tasks -> CPU values sum (then divided by 300s)."""
    # Bin 0: [0, 300)
    # Task 1: 0 to 300 at rate 0.30 -> contribution 0.30
    # Task 2: 0 to 150 at rate 0.40 -> contribution 0.40 * 150/300 = 0.20
    # Task 3: 150 to 300 at rate 0.20 -> contribution 0.20 * 150/300 = 0.10
    # Total Bin 0 CPU = 0.30 + 0.20 + 0.10 = 0.60
    df = pd.DataFrame([
        {'start_sec': 0.0,   'end_sec': 300.0, 'cpu_rate': 0.30},
        {'start_sec': 0.0,   'end_sec': 150.0, 'cpu_rate': 0.40},
        {'start_sec': 150.0, 'end_sec': 300.0, 'cpu_rate': 0.20},
    ])
    ref_cpu, ref_touched = aggregate_machine_cpu_reference(df, total_windows=1)
    vec_res = aggregate_machine_vectorized(df, total_windows=1)

    assert abs(ref_cpu[0] - 0.60) < TOL
    assert abs(vec_res['cpu'][0] - 0.60) < TOL
    assert ref_touched[0] == True and vec_res['touched'][0] == True
    print("  [PASS] Case F: Multiple simultaneous tasks (CPU values correctly sum across overlapping intervals)")


def test_case_g():
    """Case G: No tasks in bin -> cpu = NaN (missing), not 0.0 (idle)."""
    # 5 bins total: [0, 300), [300, 600), [600, 900), [900, 1200), [1200, 1500)
    # Bin 0: task active
    # Bins 1, 2, 3: NO tasks recorded (empty)
    # Bin 4: task active
    df = pd.DataFrame([
        {'start_sec': 0.0,    'end_sec': 300.0,  'cpu_rate': 0.50},
        # Bins 1, 2, 3 are empty
        {'start_sec': 1200.0, 'end_sec': 1500.0, 'cpu_rate': 0.50}
    ])
    vec_res = aggregate_machine_vectorized(df, total_windows=5)
    # Raw vector output has 0.0 in untouched bins, but touched mask is False
    assert vec_res['touched'][0] == True, "Bin 0 must be touched"
    assert vec_res['touched'][1] == False, "Bin 1 must be untouched"
    assert vec_res['touched'][2] == False, "Bin 2 must be untouched"
    assert vec_res['touched'][3] == False, "Bin 3 must be untouched"
    assert vec_res['touched'][4] == True, "Bin 4 must be touched"

    # clean_and_fill_gaps with max_gap=0 converts all untouched directly to NaN
    s_no_fill = clean_and_fill_gaps(vec_res['cpu'], vec_res['touched'], max_gap=0)
    assert not np.isnan(s_no_fill.iloc[0]), "Touched bin 0 must not be NaN"
    assert np.isnan(s_no_fill.iloc[1]), "Untouched bin 1 must become NaN (missing data)"
    assert np.isnan(s_no_fill.iloc[2]), "Untouched bin 2 must become NaN (missing data)"
    assert np.isnan(s_no_fill.iloc[3]), "Untouched bin 3 must become NaN (missing data)"
    assert not np.isnan(s_no_fill.iloc[4]), "Touched bin 4 must not be NaN"
    print("  [PASS] Case G: No tasks in bin (correctly flagged untouched and converted to NaN, distinct from idle)")


def test_case_h():
    """Case H: Tasks recorded but all at zero CPU rate -> cpu = 0.0 (confirmed idle)."""
    # Bin 0: task explicitly running at cpu_rate = 0.0
    # Bin 1: no task
    df = pd.DataFrame([
        {'start_sec': 0.0, 'end_sec': 300.0, 'cpu_rate': 0.0}
    ])
    vec_res = aggregate_machine_vectorized(df, total_windows=2)
    assert vec_res['touched'][0] == True, "Bin 0 had a recorded task -> must be marked touched"
    assert vec_res['touched'][1] == False, "Bin 1 had no task -> must be marked untouched"

    s_cleaned = clean_and_fill_gaps(vec_res['cpu'], vec_res['touched'], max_gap=0)
    assert s_cleaned.iloc[0] == 0.0, "Bin 0 must remain 0.0 (confirmed idle)"
    assert not np.isnan(s_cleaned.iloc[0]), "Idle bin must NOT be converted to NaN"
    assert np.isnan(s_cleaned.iloc[1]), "Untouched bin 1 must become NaN"
    print("  [PASS] Case H: Tasks recorded at zero CPU rate (cpu = 0.0 confirmed idle, not converted to NaN)")


def main():
    print("=" * 70)
    print("   INTERVAL-WEIGHTED AGGREGATION MATHEMATICAL VERIFICATION SUITE")
    print("=" * 70)
    test_case_a()
    test_case_b()
    test_case_c()
    test_case_d()
    test_case_e()
    test_case_f()
    test_case_g()
    test_case_h()
    print("=" * 70)
    print("   ALL 8 SCIENTIFIC EDGE CASES PASSED — FORMULA RIGOROUSLY VERIFIED")
    print("=" * 70)


if __name__ == "__main__":
    main()
