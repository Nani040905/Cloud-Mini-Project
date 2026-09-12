"""
scripts/phase2/test_bootstrap.py

Unit Test Suite for Phase 2.2 Machine-Level Bootstrap and Paired Differences.
Tests:
1. Deterministic reproducibility with fixed seed.
2. Exact calculation of macro and pooled point estimates.
3. Zero-difference self-comparison (identical policies produce Delta=0, CI=[0,0]).
4. Constant shift detection (shift c produces CI centered at c, marked significant).
5. Mathematical ordering invariant: ci_lower <= point_estimate <= ci_upper.
6. Machine-level cluster sampling (machines sampled with replacement).
"""

import unittest
import numpy as np
import pandas as pd

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.phase2.bootstrap import (
    run_machine_bootstrap_ci,
    run_paired_difference_bootstrap
)


class TestBootstrapStatisticalUncertainty(unittest.TestCase):

    def setUp(self):
        # Create a synthetic 100-machine test dataset
        rng = np.random.default_rng(1234)
        self.n_machines = 100
        self.rates_a = rng.uniform(0.005, 0.015, size=self.n_machines)
        self.total_bins = np.full(self.n_machines, 1253)
        self.breach_bins_a = np.round(self.rates_a * self.total_bins).astype(int)

        self.df_a = pd.DataFrame({
            'machine_id': np.arange(self.n_machines),
            'breach_bins': self.breach_bins_a,
            'total_test_bins': self.total_bins,
            'breach_rate': self.breach_bins_a / self.total_bins
        })

    def test_01_reproducibility(self):
        """Verify identical seed produces strictly identical bootstrap distributions and CIs."""
        res1 = run_machine_bootstrap_ci(self.df_a, n_replicates=500, seed=42)
        res2 = run_machine_bootstrap_ci(self.df_a, n_replicates=500, seed=42)

        self.assertEqual(res1['ci_2.5_macro'], res2['ci_2.5_macro'])
        self.assertEqual(res1['ci_97.5_macro'], res2['ci_97.5_macro'])
        self.assertEqual(res1['std_error_macro'], res2['std_error_macro'])

    def test_02_point_estimates_exact(self):
        """Verify point estimates match direct numpy mean and pooled calculations."""
        res = run_machine_bootstrap_ci(self.df_a, n_replicates=200, seed=42)
        expected_macro = np.mean(self.df_a['breach_rate'].values)
        expected_pooled = np.sum(self.df_a['breach_bins'].values) / np.sum(self.df_a['total_test_bins'].values)

        self.assertAlmostEqual(res['point_estimate_macro'], expected_macro, places=6)
        self.assertAlmostEqual(res['point_estimate_pooled'], expected_pooled, places=6)

    def test_03_identical_policies_zero_delta(self):
        """Comparing a policy to itself must yield Delta=0, CI=[0,0], and not statistically significant."""
        res = run_paired_difference_bootstrap(self.df_a, self.df_a, 'PolicyA', 'PolicyA_Copy',
                                              n_replicates=500, seed=42)

        self.assertEqual(res['mean_delta'], 0.0)
        self.assertEqual(res['median_delta'], 0.0)
        self.assertEqual(res['bootstrap_ci_2.5'], 0.0)
        self.assertEqual(res['bootstrap_ci_97.5'], 0.0)
        self.assertFalse(res['statistically_significant'])
        self.assertEqual(res['machines_tied'], self.n_machines)
        self.assertEqual(res['machines_A_better'], 0)
        self.assertEqual(res['machines_B_better'], 0)

    def test_04_constant_shift_detection(self):
        """Adding a constant shift c to all machines must produce Delta=c, CI centered at c, and significance=True."""
        shift = 0.05
        df_b = self.df_a.copy()
        df_b['breach_rate'] = df_b['breach_rate'] + shift
        df_b['breach_bins'] = np.round(df_b['breach_rate'] * df_b['total_test_bins']).astype(int)

        # B minus A has shift +0.05
        res = run_paired_difference_bootstrap(df_b, self.df_a, 'ShiftedB', 'OriginalA',
                                              n_replicates=500, seed=42)

        self.assertAlmostEqual(res['mean_delta'], shift, places=5)
        self.assertAlmostEqual(res['median_delta'], shift, places=5)
        # CI must not include 0
        self.assertTrue(res['bootstrap_ci_2.5'] > 0.0)
        self.assertTrue(res['statistically_significant'])
        self.assertEqual(res['machines_B_better'], self.n_machines)
        self.assertEqual(res['machines_A_better'], 0)

    def test_05_ci_ordering(self):
        """Verify mathematical invariant: ci_lower <= point_estimate <= ci_upper."""
        res = run_machine_bootstrap_ci(self.df_a, n_replicates=500, seed=42)
        self.assertTrue(res['ci_2.5_macro'] <= res['point_estimate_macro'] <= res['ci_97.5_macro'])
        self.assertTrue(res['ci_2.5_pooled'] <= res['point_estimate_pooled'] <= res['ci_97.5_pooled'])


if __name__ == '__main__':
    unittest.main()
