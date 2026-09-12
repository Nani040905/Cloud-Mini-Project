import pandas as pd
import numpy as np

# Machine 2107201469
mid = 2107201469

# 1. Load parquet allocations
df_alloc = pd.read_parquet('results/phase2/phase1_test_allocations.parquet')
m_ustar = df_alloc[(df_alloc['machine_id'] == mid) & (df_alloc['policy'] == 'Adaptive_U_star')].sort_values('bin_idx').copy()
m_ustar['shortfall'] = m_ustar['y_true'] - m_ustar['allocation']
m_ustar['is_breach'] = m_ustar['shortfall'] > 0.005

# Streak detection
m_ustar['breach_block'] = (~m_ustar['is_breach']).cumsum()
incident_runs = m_ustar[m_ustar['is_breach']].groupby('breach_block')

longest_run = None
max_len = 0
for b_id, group in incident_runs:
    if len(group) > max_len:
        max_len = len(group)
        longest_run = group

print(f"Machine {mid}: Longest breach run is {max_len} bins ({max_len * 5} min)")
start_bin = longest_run['bin_idx'].min()
end_bin = longest_run['bin_idx'].max()
print(f"Start bin: {start_bin}, End bin: {end_bin}")

# 2. Check selector_test.csv
try:
    df_sel = pd.read_csv('uncertainty/selector_test.csv')
    print("selector_test columns:", df_sel.columns.tolist())
    m_sel = df_sel[(df_sel['machine_id'] == mid) & (df_sel['bin_idx'] >= start_bin) & (df_sel['bin_idx'] <= end_bin)]
    print(f"selector rows found: {len(m_sel)}")
except Exception as e:
    print("selector_test error:", e)

# 3. Check uncertainty.csv
try:
    df_unc = pd.read_csv('uncertainty/uncertainty.csv')
    print("uncertainty columns:", df_unc.columns.tolist())
    m_unc = df_unc[(df_unc['machine_id'] == mid) & (df_unc['bin_idx'] >= start_bin) & (df_unc['bin_idx'] <= end_bin)]
    print(f"uncertainty rows found: {len(m_unc)}")
except Exception as e:
    print("uncertainty error:", e)

# 4. Check model predictions
try:
    df_gru = pd.read_csv('models/gru_base_predictions.csv')
    print("gru columns:", df_gru.columns.tolist())
    m_gru = df_gru[(df_gru['machine_id'] == mid) & (df_gru['bin_idx'] >= start_bin) & (df_gru['bin_idx'] <= end_bin)]
    print(f"gru rows found: {len(m_gru)}")
except Exception as e:
    print("gru error:", e)

try:
    df_hw = pd.read_csv('models/hw_predictions.csv')
    print("hw columns:", df_hw.columns.tolist())
    m_hw = df_hw[(df_hw['machine_id'] == mid) & (df_hw['bin_idx'] >= start_bin) & (df_hw['bin_idx'] <= end_bin)]
    print(f"hw rows found: {len(m_hw)}")
except Exception as e:
    print("hw error:", e)

try:
    df_fus = pd.read_csv('models/fusion_predictions.csv')
    print("fusion columns:", df_fus.columns.tolist())
    m_fus = df_fus[(df_fus['machine_id'] == mid) & (df_fus['bin_idx'] >= start_bin) & (df_fus['bin_idx'] <= end_bin)]
    print(f"fusion rows found: {len(m_fus)}")
except Exception as e:
    print("fusion error:", e)
