"""
scripts/phase2/test_reliability.py

Unit Test Suite for Phase 2.1 Reliability Decomposition & Incident Analysis.
Tests:
1. Synthetic one-bin breach (isolated transient impulse)
2. Synthetic multi-bin breach (sustained streak)
3. Separated incidents (intervening non-breach bins & missing gaps)
4. Boundary incident at start/end of test split
5. Severity calculation conditional on breach
6. Grace-band exact boundary behavior (y = A + 0.005 vs y = A + 0.0051)
7. Strict machine isolation (no cross-machine incident contamination)
"""

import unittest
import numpy as np
import pandas as pd

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.phase2.reliability import (
    compute_machine_incidents,
    compute_machine_reliability_metrics
)


class TestReliabilityDecomposition(unittest.TestCase):

    def test_01_synthetic_one_bin_breach(self):
        """Test that a single isolated breach produces exactly 1 incident with duration 1 bin (5 minutes)."""
        bins = np.array([100, 101, 102, 103, 104])
        breaches = np.array([False, True, False, False, False])

        durations = compute_machine_incidents(bins, breaches)
        self.assertEqual(durations, [1])

        # Test full metrics
        yt = np.array([0.2, 0.35, 0.2, 0.2, 0.2])
        al = np.array([0.3, 0.30, 0.3, 0.3, 0.3])  # at t=1: 0.35 > 0.30 + 0.005 -> breach of S = 0.045
        m, durs, sevs = compute_machine_reliability_metrics(yt, al, bins, 1, 'TestPolicy')

        self.assertEqual(m['breach_bins'], 1)
        self.assertEqual(m['number_of_incidents'], 1)
        self.assertEqual(m['mean_incident_duration_bins'], 1.0)
        self.assertEqual(m['max_incident_duration_bins'], 1)
        self.assertEqual(m['mean_incident_duration_minutes'], 5.0)
        self.assertEqual(m['max_incident_duration_minutes'], 5.0)
        self.assertAlmostEqual(m['mean_shortfall_severity'], 0.045, places=5)
        self.assertAlmostEqual(m['max_shortfall_severity'], 0.045, places=5)

    def test_02_synthetic_multi_bin_breach(self):
        """Test that consecutive breach streak of K bins produces 1 incident of duration K bins."""
        bins = np.array([10, 11, 12, 13, 14, 15])
        breaches = np.array([False, True, True, True, True, False])  # 4-bin streak

        durations = compute_machine_incidents(bins, breaches)
        self.assertEqual(durations, [4])

        yt = np.array([0.1, 0.5, 0.6, 0.7, 0.8, 0.1])
        al = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2])  # 4 consecutive breaches
        m, durs, sevs = compute_machine_reliability_metrics(yt, al, bins, 2, 'TestPolicy')

        self.assertEqual(m['breach_bins'], 4)
        self.assertEqual(m['number_of_incidents'], 1)
        self.assertEqual(m['mean_incident_duration_bins'], 4.0)
        self.assertEqual(m['max_incident_duration_bins'], 4)
        self.assertEqual(m['mean_incident_duration_minutes'], 20.0)
        self.assertEqual(m['max_incident_duration_minutes'], 20.0)
        self.assertEqual(len(sevs), 4)

    def test_03_separated_incidents(self):
        """Test multiple breach streaks separated by non-breach bins and missing gaps."""
        # 1. Normal separation
        bins = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        breaches = np.array([True, True, False, True, False, True, True, True])
        # Expected incidents: [2, 1, 3]
        durations = compute_machine_incidents(bins, breaches)
        self.assertEqual(durations, [2, 1, 3])

        # 2. Separation via missing bin gap (e.g. bin jump from 1 to 5)
        bins_with_gap = np.array([0, 1, 5, 6])
        breaches_all = np.array([True, True, True, True])
        # Even though breaches are True, bin 5 != 1 + 1, so gap separates them into [2, 2]
        durations_gap = compute_machine_incidents(bins_with_gap, breaches_all)
        self.assertEqual(durations_gap, [2, 2])

    def test_04_boundary_incident_start_end(self):
        """Test boundary incidents at test start (t=0) and test end (t=T-1)."""
        # Incident starts at index 0
        bins = np.array([0, 1, 2, 3])
        breaches_start = np.array([True, True, False, False])
        self.assertEqual(compute_machine_incidents(bins, breaches_start), [2])

        # Incident ends at index T-1
        breaches_end = np.array([False, False, True, True])
        self.assertEqual(compute_machine_incidents(bins, breaches_end), [2])

        # Incident spans entire series
        breaches_all = np.array([True, True, True, True])
        self.assertEqual(compute_machine_incidents(bins, breaches_all), [4])

    def test_05_severity_calculation(self):
        """Test shortfall severity arithmetic S_t = max(0, y_t - (A_t + 0.005)) conditional on breach."""
        bins = np.arange(4)
        yt = np.array([0.20, 0.40, 0.50, 0.10])
        al = np.array([0.30, 0.30, 0.30, 0.30])
        # grace_band = 0.005
        # t=0: 0.20 <= 0.305 -> no breach, S=0
        # t=1: 0.40 > 0.305 -> breach, S = 0.40 - 0.305 = 0.095
        # t=2: 0.50 > 0.305 -> breach, S = 0.50 - 0.305 = 0.195
        # t=3: 0.10 <= 0.305 -> no breach, S=0
        m, durs, sevs = compute_machine_reliability_metrics(yt, al, bins, 5, 'TestPolicy')

        self.assertEqual(m['breach_bins'], 2)
        expected_severities = [0.095, 0.195]
        np.testing.assert_allclose(sevs, expected_severities, rtol=1e-5)
        self.assertAlmostEqual(m['mean_shortfall_severity'], np.mean(expected_severities), places=5)
        self.assertAlmostEqual(m['max_shortfall_severity'], 0.195, places=5)

    def test_06_grace_band_correctness(self):
        """Test exact boundary condition: y = A + 0.005 is grace zone, NOT a breach; y = A + 0.0051 IS breach."""
        bins = np.array([0, 1])
        yt = np.array([0.305, 0.305001])
        al = np.array([0.300, 0.300])

        m, durs, sevs = compute_machine_reliability_metrics(yt, al, bins, 6, 'TestPolicy')
        # Point 0: 0.305 is exactly A + 0.005 -> NOT a breach (grace zone hit)
        # Point 1: 0.305001 > A + 0.005 -> breach
        self.assertEqual(m['grace_zone_bins'], 1)
        self.assertEqual(m['breach_bins'], 1)
        self.assertEqual(m['overshoot_bins'], 2)
        self.assertEqual(durs, [1])

    def test_07_machine_isolation(self):
        """Verify that incidents from separate machines remain strictly isolated."""
        # Machine A has breach at bin 100
        bins_a = np.array([99, 100])
        yt_a = np.array([0.1, 0.5])
        al_a = np.array([0.2, 0.2])
        mA, dursA, _ = compute_machine_reliability_metrics(yt_a, al_a, bins_a, 101, 'Pol')

        # Machine B has breach at bin 101 (immediately following bin 100)
        bins_b = np.array([101, 102])
        yt_b = np.array([0.5, 0.1])
        al_b = np.array([0.2, 0.2])
        mB, dursB, _ = compute_machine_reliability_metrics(yt_b, al_b, bins_b, 102, 'Pol')

        # Both must be separate single-bin incidents of length 1, never merged into length 2
        self.assertEqual(dursA, [1])
        self.assertEqual(dursB, [1])
        self.assertEqual(mA['max_incident_duration_bins'], 1)
        self.assertEqual(mB['max_incident_duration_bins'], 1)

    def test_08_phase2_allocations_match_phase1_authoritative(self):
        """
        Verify that Phase 2.1 allocations exactly match the authoritative Phase 1 allocation artifact:
        1. Betas are dynamically ingested from results/allocation_results.csv (no hardcoding).
        2. Per-timestep allocations match clip(prediction + beta * uncertainty, 0.0, 1.0) exactly.
        3. Aggregated machine metrics match results/allocation_results.csv to within 1e-6.
        """
        import pandas as pd
        from scripts.utils import RESULTS_DIR, UNCERTAINTY_DIR

        # Load authoritative Phase 1 allocation results
        alloc_res_csv = os.path.join(RESULTS_DIR, "allocation_results.csv")
        self.assertTrue(os.path.exists(alloc_res_csv), f"Missing {alloc_res_csv}")
        phase1_alloc_df = pd.read_csv(alloc_res_csv)

        # Load authoritative test allocations artifact
        parquet_path = os.path.join(RESULTS_DIR, "phase2", "phase1_test_allocations.parquet")
        csv_path = os.path.join(RESULTS_DIR, "phase2", "phase1_test_allocations.csv")
        target_path = parquet_path if os.path.exists(parquet_path) else csv_path
        self.assertTrue(os.path.exists(target_path), f"Missing {target_path}")

        alloc_df = pd.read_parquet(target_path) if target_path.endswith('.parquet') else pd.read_csv(target_path)

        # Verify dynamic betas
        auth_betas = phase1_alloc_df.groupby('config')['beta_buf'].first().to_dict()
        self.assertEqual(auth_betas['B4b_Static_U2'], 5.0)
        self.assertEqual(auth_betas['Proposed_Adaptive_U'], 8.0)
        self.assertEqual(auth_betas['B4a_Static_U1'], 12.0)
        self.assertEqual(auth_betas['B4c_Static_U3'], 14.0)

        # Sample 20 deterministic rows and verify exact allocation calculation
        sample_rows = alloc_df.iloc[::len(alloc_df) // 20].copy()
        for _, row in sample_rows.iterrows():
            pred = row['prediction']
            beta = row['beta']
            u = row['uncertainty']
            al = row['allocation']

            if pd.notna(u):
                expected_al = np.clip(pred + beta * u, 0.0, 1.0)
            else:
                expected_al = np.clip(pred * beta, 0.0, 1.0)

            self.assertAlmostEqual(al, expected_al, places=6,
                                   msg=f"Allocation mismatch for policy {row['policy']} at machine {row['machine_id']}, bin {row['bin_idx']}")

        # Verify per-machine aggregated metric match against Phase 1 artifact for sample machines
        sample_mids = [418695, 638204, 902845, 1270871, 3828835]
        for mid in sample_mids:
            for pol, cfg in [('Static_U2', 'B4b_Static_U2'), ('Adaptive_U_star', 'Proposed_Adaptive_U')]:
                sub_calc = alloc_df[(alloc_df['policy'] == pol) & (alloc_df['machine_id'] == mid)]
                sub_auth = phase1_alloc_df[(phase1_alloc_df['config'] == cfg) & (phase1_alloc_df['machine_id'] == mid)].iloc[0]

                yt = sub_calc['y_true'].values
                al = sub_calc['allocation'].values
                valid = np.isfinite(yt) & np.isfinite(al)
                calc_breach = np.mean(yt[valid] > al[valid] + 0.005)
                calc_overprov = np.mean(np.maximum(0.0, al[valid] - yt[valid]))

                self.assertAlmostEqual(calc_breach, sub_auth['sla_breach_rate'], places=5)
                self.assertAlmostEqual(calc_overprov, sub_auth['over_prov_mean'], places=5)

    def test_09_incident_continuity_and_worst_incident_trace(self):
        """
        Verify that the automated worst incident diagnostic correctly isolates the
        240-minute incident, confirms strict bin_idx and timestamp_sec continuity (+1, +300s),
        and verifies that no rows were silently dropped or forward-filled.
        """
        from scripts.phase2.analyze_worst_incident import find_worst_incident, extract_incident_trace

        alloc_path = os.path.join(PROJECT_ROOT, "results", "phase2", "phase1_test_allocations.parquet")
        sel_path = os.path.join(PROJECT_ROOT, "uncertainty", "selector_test.csv")
        self.assertTrue(os.path.exists(alloc_path))
        self.assertTrue(os.path.exists(sel_path))

        alloc_df = pd.read_parquet(alloc_path)
        sel_df = pd.read_csv(sel_path)

        worst = find_worst_incident(alloc_df, policy="Adaptive_U_star")
        self.assertEqual(worst['machine_id'], 2107201469)
        self.assertEqual(worst['start_bin'], 7778)
        self.assertEqual(worst['end_bin'], 7825)
        self.assertEqual(worst['length_bins'], 48)
        self.assertEqual(worst['duration_min'], 240.0)

        trace = extract_incident_trace(worst, alloc_df, sel_df)
        self.assertEqual(len(trace), 48)

        # Verify strict bin_idx continuity (+1)
        bin_diffs = trace['bin_idx'].values[1:] - trace['bin_idx'].values[:-1]
        self.assertTrue(np.all(bin_diffs == 1))

        # Verify physical timestamp continuity (+300.0s)
        time_diffs = trace['timestamp_sec'].values[1:] - trace['timestamp_sec'].values[:-1]
        self.assertTrue(np.all(time_diffs == 300.0))

        # Verify all 48 rows are continuous breaches for U*
        self.assertEqual(trace['is_breach_ustar'].sum(), 48)

        # Verify Static U2 had 0 breaches on the exact same interval
        self.assertEqual(trace['is_breach_u2'].sum(), 0)


if __name__ == '__main__':
    unittest.main()

