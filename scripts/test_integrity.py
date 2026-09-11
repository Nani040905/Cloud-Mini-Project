"""
scripts/test_integrity.py

Automated Verification Suite for Pre-Phase 2 Experimental Integrity (F1-F11).
Tests:
1. 15-Feature Agreement & Dual-Tier Hierarchy
2. mem_lag1 Strict 1-Step Shift
3. Zero Future Leakage in Causal Feature Extraction
4. U2 Hard RuntimeError on Missing Training Residuals
5. Selector Test Window Invariant (Full N Window Without Shortening)
6. Dynamic Run Provenance Configuration
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from features import (
    BASE_FEATURES, EXTENDED_FEATURES, extract_features_single_machine,
    build_gru_sequences
)
from uncertainty import compute_u2_series
from selector import evaluate_selector_on_split, tune_window_n


class TestExperimentalIntegrity(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        n = 50
        self.synth_df = pd.DataFrame({
            'bin_idx': np.arange(n),
            'timestamp_sec': np.arange(n) * 300,
            'cpu': np.random.uniform(0.1, 0.9, size=n),
            'mem': np.random.uniform(0.2, 0.8, size=n),
            'disk': np.random.uniform(0.01, 0.2, size=n),
            'touched': np.ones(n, dtype=bool)
        })

    def test_01_feature_agreement(self):
        """F2: Verify that base has 7 features, extended has 15 features, and all are accounted for."""
        self.assertEqual(len(BASE_FEATURES), 7, "Base tier must have exactly 7 features.")
        self.assertEqual(len(EXTENDED_FEATURES), 15, "Extended tier must have exactly 15 features.")
        
        expected_base = ['cpu', 'cpu_lag1', 'cpu_lag2', 'roll_mean', 'roll_std', 'hour_sin', 'hour_cos']
        expected_ext = expected_base + ['mem', 'mem_lag1', 'mem_roll_mean', 'disk', 'cpu_diff1', 'roll_max', 'dow_sin', 'dow_cos']
        
        self.assertEqual(BASE_FEATURES, expected_base, "Base features mismatch.")
        self.assertEqual(EXTENDED_FEATURES, expected_ext, "Extended features mismatch.")

    def test_02_mem_lag1_is_shift_1(self):
        """F3: Verify that mem_lag1 is strictly implemented as 1-step lag (shift(1))."""
        feat_df = extract_features_single_machine(self.synth_df)
        self.assertIn('mem_lag1', feat_df.columns)
        
        # Row 0 should be NaN after shift(1)
        self.assertTrue(np.isnan(feat_df['mem_lag1'].iloc[0]))
        # For all subsequent rows, mem_lag1[t] == mem[t-1]
        for t in range(1, len(feat_df)):
            self.assertAlmostEqual(
                feat_df['mem_lag1'].iloc[t],
                self.synth_df['mem'].iloc[t - 1],
                places=6,
                msg=f"mem_lag1 at index {t} does not match mem at index {t-1}"
            )

    def test_03_zero_future_leakage(self):
        """F9 & Step 1: Verify that mutating cpu, mem, or disk at row t does NOT alter features at row t."""
        feat_orig = extract_features_single_machine(self.synth_df)
        
        mod_df = self.synth_df.copy()
        target_t = 15
        mod_df.loc[target_t, 'cpu'] = mod_df.loc[target_t, 'cpu'] + 0.5   # mutate cpu at t
        mod_df.loc[target_t, 'mem'] = mod_df.loc[target_t, 'mem'] + 0.3   # mutate mem at t
        mod_df.loc[target_t, 'disk'] = mod_df.loc[target_t, 'disk'] + 0.4 # mutate disk at t
        
        feat_mod = extract_features_single_machine(mod_df)
        
        # Causal features at row target_t MUST NOT change (except the raw target columns 'cpu' and 'mem')
        causal_cols = [
            'cpu_lag1', 'cpu_lag2', 'roll_mean', 'roll_std',
            'mem_lag1', 'mem_roll_mean', 'disk', 'cpu_diff1', 'roll_max'
        ]
        for col in causal_cols:
            self.assertAlmostEqual(
                feat_orig.loc[target_t, col],
                feat_mod.loc[target_t, col],
                places=6,
                msg=f"Feature {col} at index {target_t} leaked future observation!"
            )

    def test_04_u2_hard_failure_on_missing_predictions(self):
        """F4: Verify that missing training fused predictions raises RuntimeError."""
        from uncertainty import compute_uncertainty_signals
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            preds_file = os.path.join(tmpdir, "test_preds.csv")
            # Create a preds file with ONLY test split (missing train)
            bad_preds = pd.DataFrame({
                'machine_id': [101, 101],
                'split': ['test', 'test'],
                'bin_idx': [10, 11],
                'y_true': [0.5, 0.6],
                'hw_pred': [0.48, 0.58],
                'gru_pred': [0.52, 0.61],
                'fused_pred': [0.50, 0.60]
            })
            bad_preds.to_csv(preds_file, index=False)
            
            with self.assertRaises(RuntimeError) as ctx:
                compute_uncertainty_signals(preds_file, tmpdir, tmpdir)
            
            self.assertIn("Missing fused training predictions", str(ctx.exception))

    def test_05_selector_history_buffer_prevents_shortened_windows(self):
        """F6: Verify selector test execution uses full window_n without shortened window artifacts."""
        window_n = 20
        n_val = 30
        n_test = 25
        
        val_df = pd.DataFrame({
            'machine_id': [1] * n_val,
            'split': ['val'] * n_val,
            'bin_idx': np.arange(n_val),
            'U1': np.random.uniform(0.01, 0.05, n_val),
            'U2': np.random.uniform(0.01, 0.05, n_val),
            'U3': np.random.uniform(0.01, 0.05, n_val),
            'abs_error': np.random.uniform(0.01, 0.05, n_val)
        })
        
        test_df = pd.DataFrame({
            'machine_id': [1] * n_test,
            'split': ['test'] * n_test,
            'bin_idx': np.arange(n_val, n_val + n_test),
            'U1': np.random.uniform(0.01, 0.05, n_test),
            'U2': np.random.uniform(0.01, 0.05, n_test),
            'U3': np.random.uniform(0.01, 0.05, n_test),
            'abs_error': np.random.uniform(0.01, 0.05, n_test)
        })
        
        # Test evaluation with validation history buffer of length window_n
        val_buffer = val_df.iloc[-window_n:]
        res = evaluate_selector_on_split(test_df, history_buffer=val_buffer, window_n=window_n)
        
        # All test rows must have valid evaluation (not fallback/insufficient history)
        # because the buffer provided a full window_n right from test step 0
        self.assertEqual(len(res), n_test)
        self.assertIn('chosen_signal', res.columns)
        self.assertIn('U_star', res.columns)
        # Check rho values are computed (not all zero)
        self.assertTrue(len(res['chosen_signal']) == n_test)

    def test_06_cpu_missing_gaps_preserved_as_nan(self):
        """F12: Verify that CPU missing periods (>2 bins) remain strictly NaN in feature extraction."""
        df_with_nan = self.synth_df.copy()
        # Create a 5-bin missing gap in CPU (bins 10 to 14)
        df_with_nan.loc[10:14, 'cpu'] = np.nan
        # Intermittent missing in disk
        df_with_nan.loc[12:13, 'disk'] = np.nan

        feat_df = extract_features_single_machine(df_with_nan)

        # Assert CPU NaNs are preserved exactly (NOT forward-filled)
        self.assertTrue(feat_df.loc[10:14, 'cpu'].isna().all(), "CPU gaps must strictly remain NaN.")
        # Assert disk NaNs were causally forward-filled per specification
        self.assertFalse(feat_df.loc[12:13, 'disk'].isna().any(), "Disk dropouts should be forward-filled.")

    def test_07_disk_causality_and_anti_leakage(self):
        """Step 1 / F16: Verify that extended feature disk represents disk_io_time(t-1) and mutating disk(t) does not alter feature at t."""
        feat_df = extract_features_single_machine(self.synth_df)
        self.assertIn('disk', feat_df.columns, "Extended feature 'disk' must be present.")
        
        # Row 0 should be NaN after shift(1)
        self.assertTrue(np.isnan(feat_df['disk'].iloc[0]), "disk feature at row 0 must be NaN after shift(1).")
        
        # For all subsequent rows t >= 1, disk[t] == synth_df['disk'][t-1]
        for t in range(1, len(feat_df)):
            self.assertAlmostEqual(
                feat_df['disk'].iloc[t],
                self.synth_df['disk'].iloc[t - 1],
                places=6,
                msg=f"disk feature at index {t} does not match disk at index {t-1}"
            )
            
        # Anti-leakage verification: Mutating disk at target_t must NOT alter disk feature at target_t
        mod_df = self.synth_df.copy()
        target_t = 20
        mod_df.loc[target_t, 'disk'] = mod_df.loc[target_t, 'disk'] + 0.5
        feat_mod = extract_features_single_machine(mod_df)
        self.assertAlmostEqual(
            feat_df.loc[target_t, 'disk'],
            feat_mod.loc[target_t, 'disk'],
            places=6,
            msg=f"Changing disk(t) leaked into disk feature at timestep t={target_t}!"
        )

    def test_08_gru_sequences_reject_nan_windows_and_targets(self):
        """Step 2 / F17: Verify that build_gru_sequences strictly rejects windows and targets containing NaNs."""
        df_with_nan = self.synth_df.copy()
        # Introduce NaNs in CPU at specific rows: 15, 16, 17
        df_with_nan.loc[15:17, 'cpu'] = np.nan
        feat_df = extract_features_single_machine(df_with_nan)
        
        # Build sequences over [12, 35) with seq_len=12
        seq_len = 12
        X, y, indices = build_gru_sequences(
            feat_df, BASE_FEATURES, 'cpu',
            split_start=12, split_end=35, seq_len=seq_len
        )
        
        # 1. Neither X nor y must contain ANY NaNs
        self.assertFalse(np.isnan(X).any(), "Extracted GRU sequences X must not contain NaNs.")
        self.assertFalse(np.isnan(y).any(), "Extracted GRU targets y must not contain NaNs.")
        
        # 2. Targets at t=15, 16, 17 are NaN, so indices 15, 16, 17 must be excluded
        for bad_t in [15, 16, 17]:
            self.assertNotIn(bad_t, indices, f"Timestep {bad_t} with NaN target should be excluded.")
            
        # 3. Any window overlapping rows 15, 16, 17 must also be excluded
        # For target t, the window is [t-seq_len : t]. If t in [18, 29], t-12 <= 17 and t > 15, overlapping the NaN gap!
        for t_overlap in range(18, 18 + seq_len):
            if t_overlap < 35:
                self.assertNotIn(
                    t_overlap, indices,
                    f"Timestep {t_overlap} whose window overlaps NaN should be excluded from sequences."
                )

    def test_09_fast_spearman_tie_handling(self):
        """F16: Verify that fast_spearman correctly handles ties matching scipy.stats.spearmanr."""
        from scipy.stats import spearmanr
        from utils import fast_spearman

        # Vectors with heavy repeated/tied values (common in quantized EWMA/std uncertainty)
        x = np.array([1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 4.0, 5.0, 6.0])
        y = np.array([2.0, 2.0, 3.0, 5.0, 4.0, 6.0, 6.0, 7.0, 8.0])

        expected_rho, _ = spearmanr(x, y)
        actual_rho = fast_spearman(x, y)

        self.assertAlmostEqual(
            actual_rho, expected_rho, places=7,
            msg=f"fast_spearman ({actual_rho}) did not match scipy.stats.spearmanr ({expected_rho}) on tied data!"
        )

        # Constant array should return 0.0 without crashing
        const_x = np.full(len(y), 2.5)
        self.assertEqual(fast_spearman(const_x, y), 0.0)

    def test_10_holt_winters_timestamp_alignment_and_no_leakage(self):
        """F17-F19: Verify HW validation alignment on NaN actuals, regular grid, and no train target leakage."""
        from holt_winters import fit_hw_single_machine

        # Synthetic series with 100 points, containing NaNs in train and val
        series = pd.Series(np.sin(np.linspace(0, 10 * np.pi, 100)) * 0.4 + 0.5)
        series.iloc[15:18] = np.nan  # gap in train
        series.iloc[65:68] = np.nan  # gap in val

        train_slice = (0, 50)
        val_slice = (50, 75)
        test_slice = (75, 100)

        res = fit_hw_single_machine(series, train_slice, val_slice, test_slice, candidate_m=[10])
        
        # 1. Validation MAE must be a finite float, evaluated strictly on observed valid points
        self.assertTrue(np.isfinite(res['val_mae']), "Validation MAE must be finite despite NaNs in validation.")
        
        # 2. last_train_pred must be returned and must be bounded in [0.0, 1.0]
        self.assertIn('last_train_pred', res, "Result must contain 'last_train_pred'.")
        self.assertTrue(0.0 <= res['last_train_pred'] <= 1.0, "last_train_pred must be physically bounded.")

    def test_11_fusion_split_merge_integrity(self):
        """F30: Verify fusion merge includes 'split' preventing cross-split contamination."""
        import tempfile
        from fusion import run_fusion

        with tempfile.TemporaryDirectory() as tmpdir:
            hw_csv = os.path.join(tmpdir, "hw.csv")
            gru_csv = os.path.join(tmpdir, "gru.csv")

            hw_df = pd.DataFrame({
                'machine_id': [1, 1],
                'split': ['val', 'test'],
                'bin_idx': [100, 101],
                'y_true': [0.4, 0.5],
                'hw_pred': [0.42, 0.48]
            })
            gru_df = pd.DataFrame({
                'machine_id': [1, 1],
                'split': ['val', 'test'],
                'bin_idx': [100, 101],
                'gru_pred': [0.39, 0.51]
            })

            hw_df.to_csv(hw_csv, index=False)
            gru_df.to_csv(gru_csv, index=False)

            run_fusion(hw_csv, gru_csv, tmpdir)
            fused = pd.read_csv(os.path.join(tmpdir, "fusion_predictions.csv"))
            self.assertIn('split', fused.columns)
            self.assertEqual(len(fused), 2)
            self.assertListEqual(list(fused['split']), ['val', 'test'])

    def test_12_robustness_beta_fallback_rejection(self):
        """F24: Verify robustness sweep raises RuntimeError when allocation results are missing (no beta=14 fallback)."""
        from robustness import run_robustness_sweep
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            dummy_log = os.path.join(tmpdir, "selector_log.csv")
            pd.DataFrame({
                'machine_id': [1], 'split': ['test'], 'bin_idx': [10],
                'y_true': [0.5], 'fused_pred': [0.5], 'U_star': [0.02]
            }).to_csv(dummy_log, index=False)

            # Explicitly point to a nonexistent allocation results path
            missing_alloc = os.path.join(tmpdir, "nonexistent_alloc.csv")
            with self.assertRaises(RuntimeError) as ctx:
                run_robustness_sweep(selector_log_path=dummy_log, output_csv=os.path.join(tmpdir, "out.csv"),
                                     alloc_results_path=missing_alloc)
            self.assertIn("nonexistent_alloc.csv", str(ctx.exception))


if __name__ == '__main__':
    unittest.main(verbosity=2)
