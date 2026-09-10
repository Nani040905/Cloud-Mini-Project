"""
scripts/allocation.py

Step 12: Risk-Aware Resource Allocation and Constrained Buffer Optimization.
Implements:
1. Dynamic Uncertainty-Driven Capacity Allocation:
   - Formulation:
     A_raw_t = y_hat_t + beta_buf * U*_t
     A_t     = clip(A_raw_t, 0.0, 1.0)
   - Bounded Allocation:
     Physical servers cannot allocate negative capacity or exceed 100% hardware capability.
     Values are clipped strictly to [0.0, 1.0]. Any unclipped excess (A_raw > 1.0) is recorded
     as 'clip_fraction' to monitor capacity saturation risk.
2. Vectorized Constrained Beta Calibration:
   - Evaluates candidate beta in [0.1, 0.2, ..., 20.0] on the validation split.
   - Objective:
     beta* = min { beta in Candidates : P(y_val > A_val(beta) + 0.005) <= 1.0% }  (99% SLA compliance)
   - Evaluated via vectorized broadcasting across all candidates simultaneously (instantaneous runtime).
3. Benchmark Configurations Evaluated on Test Set:
   - B4a: Static U1 + Calibrated Beta
   - B4b: Static U2 + Calibrated Beta
   - B4c: Static U3 + Calibrated Beta
   - Proposed: Adaptive U* + Calibrated Beta
4. Saves final allocation metrics to results/allocation_results.csv.
"""

import os
import argparse
import numpy as np
import pandas as pd

from utils import (
    compute_metrics, CANDIDATE_BETAS, GRACE_BAND, SLA_TARGET,
    UNCERTAINTY_DIR, RESULTS_DIR
)


def calibrate_beta(val_df: pd.DataFrame, u_col: str) -> float:
    """
    Vectorized calibration to identify the minimum buffer multiplier beta
    satisfying the strict <= 1.0% SLA breach budget (99% compliance) on the validation split.
    """
    y_val = val_df['y_true'].values
    pred_val = val_df['fused_pred'].values
    u_val = val_df[u_col].values

    # Broadcast candidate betas: shape (K, N)
    betas = np.array(CANDIDATE_BETAS)[:, None]
    a_cands = np.clip(pred_val + betas * u_val, 0.0, 1.0)
    breach_rates = np.mean(y_val > (a_cands + GRACE_BAND), axis=1)

    # Find the smallest beta satisfying SLA breach budget <= 1.0%
    qualifying_indices = np.where(breach_rates <= SLA_TARGET)[0]
    if len(qualifying_indices) == 0:
        raise RuntimeError(
            f"No candidate beta satisfies validation SLA <= {SLA_TARGET:.1%} "
            f"for {u_col}. Breach rates: {dict(zip(CANDIDATE_BETAS, [round(float(r), 4) for r in breach_rates]))}"
        )
    best_beta = float(CANDIDATE_BETAS[qualifying_indices[0]])
    return best_beta


def evaluate_allocation_configuration(test_df: pd.DataFrame, u_col: str, beta: float, config_name: str):
    """Evaluates an allocation policy across all machines on the test set."""
    per_machine = []
    for mid, grp in test_df.groupby('machine_id'):
        yt = grp['y_true'].values
        yp = grp['fused_pred'].values
        u_vals = grp[u_col].values

        ar = yp + beta * u_vals
        al = np.clip(ar, 0.0, 1.0)

        m = compute_metrics(yt, yp, ar, al)
        m.update({'machine_id': mid, 'config': config_name, 'beta_buf': beta})
        per_machine.append(m)

    return pd.DataFrame(per_machine)


def run_allocation(selector_log_path: str = None, output_dir: str = RESULTS_DIR,
                   val_selector_path: str = None, test_selector_path: str = None):
    """
    Runs constrained calibration on explicit validation artifact and evaluates
    B4a-c and Proposed Adaptive U* configurations on the test split.
    """
    if val_selector_path and os.path.exists(val_selector_path) and test_selector_path and os.path.exists(test_selector_path):
        print(f"Loading explicit validation selector artifact from: {val_selector_path}")
        print(f"Loading explicit test selector artifact from: {test_selector_path}")
        val_df = pd.read_csv(val_selector_path)
        test_df = pd.read_csv(test_selector_path)
    else:
        log_path = selector_log_path or os.path.join(UNCERTAINTY_DIR, "selector_log.csv")
        print(f"Loading selector log and uncertainty signals from: {log_path}")
        df = pd.read_csv(log_path)
        val_df = df[df['split'] == 'val'].copy()
        test_df = df[df['split'] == 'test'].copy()

    assert 'U_star' in val_df.columns, "Validation artifact must contain 'U_star' for adaptive beta calibration"
    assert 'U_star' in test_df.columns, "Test artifact must contain 'U_star' for evaluation"
    print(f"[Causal Protocol] Calibration set: {len(val_df)} timesteps across {val_df['machine_id'].nunique()} machines")
    print(f"[Causal Protocol] Test evaluation set: {len(test_df)} timesteps across {test_df['machine_id'].nunique()} machines")

    configs = [
        ('U1', 'B4a_Static_U1'),
        ('U2', 'B4b_Static_U2'),
        ('U3', 'B4c_Static_U3'),
        ('U_star', 'Proposed_Adaptive_U')
    ]

    all_results = []
    calibrated_betas = {}

    for u_col, cfg_name in configs:
        beta = calibrate_beta(val_df, u_col)
        calibrated_betas[cfg_name] = beta
        print(f"Calibrated beta for {cfg_name:20s}: beta = {beta:.2f} (from validation U_star/uncertainty)")

        res_df = evaluate_allocation_configuration(test_df, u_col, beta, cfg_name)
        all_results.append(res_df)

    combined_df = pd.concat(all_results, ignore_index=True)

    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, "allocation_results.csv")
    combined_df.to_csv(out_csv, index=False)
    print(f"\nSaved allocation evaluation results to: {out_csv}")

    # Display Macro-Averages
    print("\n" + "=" * 75)
    print("           ALLOCATION EVALUATION SUMMARY (TEST SET MACRO-AVERAGE)")
    print("=" * 75)
    for cfg, grp in combined_df.groupby('config'):
        print(f"\nConfiguration: {cfg} (beta = {calibrated_betas[cfg]:.2f})")
        print(f"  SLA Breach Rate     : {grp['sla_breach_rate'].mean():.2%} (Target <= {SLA_TARGET:.2%})")
        print(f"  Overshoot Rate      : {grp['overshoot_rate'].mean():.2%}")
        print(f"  Over-provision Mean : {grp['over_prov_mean'].mean():.4f}")
        print(f"  Under-provision Mean: {grp['under_prov_mean'].mean():.4f}")
        print(f"  Clip Fraction       : {grp['clip_fraction'].mean():.2%}")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(description="Constrained buffer allocation and evaluation")
    parser.add_argument("--selector_log", type=str, default=os.path.join(UNCERTAINTY_DIR, "selector_log.csv"))
    parser.add_argument("--val_selector", type=str, default=os.path.join(UNCERTAINTY_DIR, "selector_validation.csv"))
    parser.add_argument("--test_selector", type=str, default=os.path.join(UNCERTAINTY_DIR, "selector_test.csv"))
    parser.add_argument("--output_dir", type=str, default=RESULTS_DIR)
    args = parser.parse_args()

    run_allocation(
        selector_log_path=args.selector_log,
        output_dir=args.output_dir,
        val_selector_path=args.val_selector,
        test_selector_path=args.test_selector
    )


if __name__ == "__main__":
    main()
