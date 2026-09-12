import pandas as pd
import numpy as np

mid = 2107201469

# 1. Load parquet allocations
df_alloc = pd.read_parquet('results/phase2/phase1_test_allocations.parquet')
m_ustar = df_alloc[(df_alloc['machine_id'] == mid) & (df_alloc['policy'] == 'Adaptive_U_star')].sort_values('bin_idx').copy()
m_ustar['shortfall'] = m_ustar['y_true'] - m_ustar['allocation']
m_ustar['is_breach'] = m_ustar['shortfall'] > 0.005

# 2. Load selector test
df_sel = pd.read_csv('uncertainty/selector_test.csv')
m_sel = df_sel[df_sel['machine_id'] == mid].sort_values('bin_idx').copy()

# Merge
merged = pd.merge(m_ustar, m_sel[['bin_idx', 'hw_pred', 'gru_pred', 'fused_pred', 'U1', 'U2', 'U3', 'chosen_signal', 'U_star', 'rho_U1', 'rho_U2', 'rho_U3']], on='bin_idx')

# Filter for the 48-bin incident: 7778 to 7825
inc = merged[(merged['bin_idx'] >= 7778) & (merged['bin_idx'] <= 7825)].copy()

# Calculate time offsets
# Total timeline is 8352 bins across 29 days. 1 bin = 5 minutes = 300 seconds.
# Test split starts at bin 7099 (8352 * 0.85 = 7099.2).
# Let's check start bin of test split
test_start_bin = df_sel['bin_idx'].min()
print(f"Test split starts at bin: {test_start_bin}")

inc['time_min_from_test_start'] = (inc['bin_idx'] - test_start_bin) * 5.0
inc['time_hours_from_test_start'] = inc['time_min_from_test_start'] / 60.0
inc['time_days_from_test_start'] = inc['time_hours_from_test_start'] / 24.0

inc['absolute_time_min'] = inc['bin_idx'] * 5.0
inc['absolute_time_days'] = inc['absolute_time_min'] / (60.0 * 24.0)

print(f"\n================ INCIDENT SUMMARY ================")
print(f"Machine ID: {mid}")
print(f"Start Bin: {inc['bin_idx'].min()} (Day {inc['absolute_time_days'].min():.3f}, +{inc['time_hours_from_test_start'].min():.2f}h into test)")
print(f"End Bin: {inc['bin_idx'].max()} (Day {inc['absolute_time_days'].max():.3f}, +{inc['time_hours_from_test_start'].max():.2f}h into test)")
print(f"Duration: {len(inc)} bins = {len(inc) * 5} minutes (4.0 hours)")
print(f"Number of consecutive breaches: {inc['is_breach'].sum()} / {len(inc)}")

print(f"\nWorkload (y_true):")
print(f"  Mean: {inc['y_true'].mean():.4f}, Min: {inc['y_true'].min():.4f}, Max: {inc['y_true'].max():.4f}, Std: {inc['y_true'].std():.4f}")

print(f"\nPredictions:")
print(f"  GRU pred:   Mean={inc['gru_pred'].mean():.4f}, Min={inc['gru_pred'].min():.4f}, Max={inc['gru_pred'].max():.4f}")
print(f"  HW pred:    Mean={inc['hw_pred'].mean():.4f}, Min={inc['hw_pred'].min():.4f}, Max={inc['hw_pred'].max():.4f}")
print(f"  Fused pred: Mean={inc['fused_pred'].mean():.4f}, Min={inc['fused_pred'].min():.4f}, Max={inc['fused_pred'].max():.4f}")

print(f"\nUncertainty Estimates:")
print(f"  U1 (Epistemic Disagreement): Mean={inc['U1'].mean():.4f}, Min={inc['U1'].min():.4f}, Max={inc['U1'].max():.4f}")
print(f"  U2 (Aleatoric EWMA error):   Mean={inc['U2'].mean():.4f}, Min={inc['U2'].min():.4f}, Max={inc['U2'].max():.4f}")
print(f"  U3 (Workload Volatility):    Mean={inc['U3'].mean():.4f}, Min={inc['U3'].min():.4f}, Max={inc['U3'].max():.4f}")

print(f"\nSelector Behavior:")
print(f"  Chosen signals breakdown:\n{inc['chosen_signal'].value_counts(normalize=True)}")
print(f"  Selected U* : Mean={inc['U_star'].mean():.4f}, Min={inc['U_star'].min():.4f}, Max={inc['U_star'].max():.4f}")
print(f"  Spearman rho_U1: Mean={inc['rho_U1'].mean():.4f}")
print(f"  Spearman rho_U2: Mean={inc['rho_U2'].mean():.4f}")
print(f"  Spearman rho_U3: Mean={inc['rho_U3'].mean():.4f}")

print(f"\nAllocation & Shortfall:")
print(f"  Beta: {inc['beta'].iloc[0]}")
print(f"  Allocation: Mean={inc['allocation'].mean():.4f}, Min={inc['allocation'].min():.4f}, Max={inc['allocation'].max():.4f}")
print(f"  Actual - Allocation (Shortfall): Mean={inc['shortfall'].mean():.4f}, Min={inc['shortfall'].min():.4f}, Max={inc['shortfall'].max():.4f}")

# Also compare with Static U2 on the exact same bins
m_u2 = df_alloc[(df_alloc['machine_id'] == mid) & (df_alloc['policy'] == 'Static_U2')].sort_values('bin_idx')
u2_inc = m_u2[(m_u2['bin_idx'] >= 7778) & (m_u2['bin_idx'] <= 7825)].copy()
u2_inc['shortfall'] = u2_inc['y_true'] - u2_inc['allocation']
u2_inc['is_breach'] = u2_inc['shortfall'] > 0.005

print(f"\n================ STATIC U2 ON SAME BINS ================")
print(f"Static U2 Breaches during this 48-bin period: {u2_inc['is_breach'].sum()} / {len(u2_inc)}")
print(f"Static U2 Beta: {u2_inc['beta'].iloc[0]}")
print(f"Static U2 Allocation: Mean={u2_inc['allocation'].mean():.4f}, Min={u2_inc['allocation'].min():.4f}, Max={u2_inc['allocation'].max():.4f}")
print(f"Static U2 Shortfall: Mean={u2_inc['shortfall'].mean():.4f}, Min={u2_inc['shortfall'].min():.4f}, Max={u2_inc['shortfall'].max():.4f}")

# Print first 10 rows and last 5 rows of incident
cols_to_print = ['bin_idx', 'y_true', 'gru_pred', 'hw_pred', 'fused_pred', 'U1', 'U2', 'U3', 'chosen_signal', 'U_star', 'allocation', 'shortfall']
print("\nFirst 10 bins of incident:")
print(inc[cols_to_print].head(10).to_string(index=False))

print("\nLast 5 bins of incident:")
print(inc[cols_to_print].tail(5).to_string(index=False))
