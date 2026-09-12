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


if __name__ == '__main__':
    unittest.main()
