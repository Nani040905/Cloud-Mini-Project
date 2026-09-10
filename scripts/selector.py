"""
scripts/selector.py

Step 11: Adaptive Reliability Selector (Dynamic Signal Arbitration).
Implements:
1. Dynamic Reliability Selection:
   - Instead of committing to a single static uncertainty signal across all operating conditions,
     the Adaptive Selector evaluates the empirical tracking quality of each signal over a recent
     sliding window of length N.
   - At each timestep t:
     rho_{i, t} = SpearmanCorr(U_i[t-N : t-1], |y_hat_{fused} - y|[t-N : t-1])
   - Signal selection:
     i*_t = argmax_{i in {U1, U2, U3} with rho_i > 0} rho_{i, t}
     (If all rho <= 0, fall back to max rho and flag 'no_pos_signal=True').
2. Full Test Coverage Without Warmup Loss:
   - Rather than dropping the first N test points for lookback warmup, the rolling window draws
     continuously from the immediately preceding validation history.
   - Preserves 100% of the test evaluation window without data loss or discontinuity.
3. Window N Calibration:
   - Tunes N in {20, 30, 50, 75, 100} on the validation set to maximize average winning Spearman rho.
4. Outputs:
   - Saves complete per-timestep arbitration trace to uncertainty/selector_log.csv.
"""

import os
import argparse
import numpy as np
import pandas as pd

from utils import (
    fast_spearman, N_WINDOW_GRID, UNCERTAINTY_DIR
)

SIGNALS = ['U1', 'U2', 'U3']


def tune_window_n(val_df: pd.DataFrame) -> int:
    """Tuning sliding window N on the validation set to maximize winning correlation."""
    best_n = 50
    best_winning_rho = -float('inf')

    for n in N_WINDOW_GRID:
        rhos = []
        for mid, grp in val_df.groupby('machine_id'):
            if len(grp) <= n:
                continue
            errors = grp['abs_error'].values
            u_mats = [grp[s].values for s in SIGNALS]

            for t in range(n, len(grp)):
                win_err = errors[t-n:t]
                win_rhos = [fast_spearman(u[t-n:t], win_err) for u in u_mats]
                rhos.append(max(win_rhos))

        mean_win_rho = float(np.mean(rhos)) if rhos else 0.0
        if mean_win_rho > best_winning_rho:
            best_winning_rho, best_n = mean_win_rho, n

    print(f"Optimal selector window: N = {best_n} (Validation winning mean rho = {best_winning_rho:.4f})")
    return best_n


def evaluate_selector_on_split(eval_df: pd.DataFrame, history_buffer: pd.DataFrame, window_n: int) -> pd.DataFrame:
    """
    Applies the adaptive selector to eval_df using history_buffer to supply lookback observations.
    If history_buffer has >= window_n points, every timestep t in eval_df has a full window_n lookback.
    If history_buffer has < window_n points, timesteps with fewer than window_n historical observations
    are marked with no_pos_signal=True, strictly avoiding shortened window artifacts.
    """
    if len(eval_df) == 0:
        return eval_df

    res_df = eval_df.copy()
    n_eval = len(res_df)

    if history_buffer is not None and len(history_buffer) > 0:
        combined = pd.concat([history_buffer, res_df], ignore_index=True)
        buf_len = len(history_buffer)
    else:
        combined = res_df
        buf_len = 0

    u1_vals = combined['U1'].values
    u2_vals = combined['U2'].values
    u3_vals = combined['U3'].values
    err_vals = combined['abs_error'].values
    sig_matrix = np.column_stack([u1_vals, u2_vals, u3_vals])

    chosen_signals, chosen_u_vals = [], []
    rho_u1_l, rho_u2_l, rho_u3_l = [], [], []
    no_pos_flags = []

    for i in range(n_eval):
        curr_idx = buf_len + i
        w_start = curr_idx - window_n

        if w_start < 0:
            # Fewer than window_n historical points: no shortened window, flag as insufficient history
            curr_rhos = [0.0, 0.0, 0.0]
            best_idx = 0
            no_pos = True
        else:
            w_err = err_vals[w_start:curr_idx]
            curr_rhos = [
                fast_spearman(u1_vals[w_start:curr_idx], w_err),
                fast_spearman(u2_vals[w_start:curr_idx], w_err),
                fast_spearman(u3_vals[w_start:curr_idx], w_err)
            ]

            # Positive-rho preference rule
            pos_indices = [idx for idx, r in enumerate(curr_rhos) if r > 0.0]
            if pos_indices:
                best_idx = max(pos_indices, key=lambda idx: curr_rhos[idx])
                no_pos = False
            else:
                best_idx = int(np.argmax(curr_rhos))
                no_pos = True

        chosen_signals.append(SIGNALS[best_idx])
        chosen_u_vals.append(sig_matrix[curr_idx, best_idx])
        rho_u1_l.append(curr_rhos[0])
        rho_u2_l.append(curr_rhos[1])
        rho_u3_l.append(curr_rhos[2])
        no_pos_flags.append(no_pos)

    res_df['chosen_signal'] = chosen_signals
    res_df['U_star'] = chosen_u_vals
    res_df['rho_U1'] = rho_u1_l
    res_df['rho_U2'] = rho_u2_l
    res_df['rho_U3'] = rho_u3_l
    res_df['no_pos_signal'] = no_pos_flags

    return res_df


def run_selector_single_machine(m_df: pd.DataFrame, window_n: int):
    """
    Applies the adaptive selector across a continuous time sequence spanning val and test.
    Uses validation history buffer so that test begins immediately with a full N-window,
    strictly avoiding shortened windows.
    """
    m_sorted = m_df.sort_values('bin_idx').copy()
    val_df = m_sorted[m_sorted['split'] == 'val'].copy()
    test_df = m_sorted[m_sorted['split'] == 'test'].copy()

    # Process validation split (evaluates when full window_n observations exist)
    val_res = evaluate_selector_on_split(eval_df=val_df, history_buffer=None, window_n=window_n)

    # Process test split: prepend validation history buffer of length window_n
    # so test t=0 immediately has a full window_n lookback
    val_history_buffer = val_df.iloc[-window_n:] if len(val_df) >= window_n else val_df
    test_res = evaluate_selector_on_split(eval_df=test_df, history_buffer=val_history_buffer, window_n=window_n)

    return pd.concat([val_res, test_res], ignore_index=True)


def run_reliability_selector(uncertainty_csv: str, output_dir: str):
    """Executes dynamic reliability arbitration across all cluster machines."""
    print("Loading uncertainty signals...")
    df = pd.read_csv(uncertainty_csv)

    val_df = df[df['split'] == 'val']
    opt_n = tune_window_n(val_df)

    print(f"Applying adaptive selector with calibrated window N = {opt_n}...")
    machine_selected_dfs = []
    for mid, grp in df.groupby('machine_id'):
        res = run_selector_single_machine(grp, opt_n)
        machine_selected_dfs.append(res)

    selector_df = pd.concat(machine_selected_dfs, ignore_index=True)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "selector_log.csv")
    selector_df.to_csv(out_path, index=False)
    print(f"Saved selector arbitration log to: {out_path}")

    # Explicit validation artifact used for beta calibration and test evaluation
    selector_val = selector_df[selector_df["split"] == "val"].copy()
    selector_test = selector_df[selector_df["split"] == "test"].copy()
    val_out_path = os.path.join(output_dir, "selector_validation.csv")
    test_out_path = os.path.join(output_dir, "selector_test.csv")
    selector_val.to_csv(val_out_path, index=False)
    selector_test.to_csv(test_out_path, index=False)
    print(f"Saved explicit validation selector artifact to: {val_out_path}")
    print(f"Saved explicit test selector artifact to: {test_out_path}")

    # Display Macro-Averages on Test Split
    test_sub = selector_test
    print("\n" + "=" * 65)
    print("           ADAPTIVE SELECTOR SUMMARY (TEST SPLIT)")
    print("=" * 65)
    choice_dist = test_sub['chosen_signal'].value_counts(normalize=True).to_dict()
    for sig, frac in choice_dist.items():
        print(f"Signal {sig:2s} selected       : {frac:.1%}")

    no_pos_rate = test_sub['no_pos_signal'].mean()
    print(f"No positive signal rate: {no_pos_rate:.1%}")
    print(f"Mean U* uncertainty    : {test_sub['U_star'].mean():.4f} +/- {test_sub['U_star'].std():.4f}")
    print("=" * 65)

    # ------------------------------------------------------------------
    # Comparative Advantage: Adaptive Selection vs. Best Static Signal
    # ------------------------------------------------------------------
    # The globally dominant static candidate is U1 (highest overall selection)
    static_best = 'U1'
    divergent_mask = (test_sub['chosen_signal'] != static_best)
    div_rate = float(divergent_mask.mean())

    if divergent_mask.sum() > 0:
        div_sub = test_sub[divergent_mask]
        rho_adaptive = div_sub.apply(lambda r: r[f"rho_{r['chosen_signal']}"], axis=1).mean()
        rho_static = div_sub[f"rho_{static_best}"].mean()
        delta_rho = rho_adaptive - rho_static

        print("\n" + "=" * 65)
        print("    DYNAMIC ARBITRATION VS. STATIC BASELINE (TEST SPLIT)")
        print("=" * 65)
        print(f"Globally Dominant Static Baseline : {static_best}")
        print(f"Divergence Rate (U* != {static_best})       : {div_rate:.1%} of timesteps")
        print(f"In divergent timesteps:")
        print(f"  Adaptive Signal Mean Local Rho  : {rho_adaptive:+.4f}")
        print(f"  Static {static_best} Mean Local Rho        : {rho_static:+.4f}")
        print(f"  Reliability Advantage (+Delta)  : {delta_rho:+.4f} (rho improvement)")
        print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Adaptive reliability selector")
    parser.add_argument("--uncertainty_csv", type=str, default=os.path.join(UNCERTAINTY_DIR, "uncertainty.csv"))
    parser.add_argument("--output_dir", type=str, default=UNCERTAINTY_DIR)
    args = parser.parse_args()

    run_reliability_selector(args.uncertainty_csv, args.output_dir)


if __name__ == "__main__":
    main()
