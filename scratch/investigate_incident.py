import pandas as pd
import numpy as np

# Load reliability decomposition
df_rel = pd.read_csv('results/phase2/reliability_decomposition.csv')
ustar = df_rel[df_rel['policy'] == 'Adaptive_U_star'].sort_values('max_incident_duration_minutes', ascending=False)
print("Top 5 longest U* incidents:")
for idx, row in ustar.head(5).iterrows():
    mid = int(row['machine_id'])
    print(f"Machine {mid}: max_duration={row['max_incident_duration_minutes']} min ({row['max_incident_duration_bins']} bins), incidents={row['number_of_incidents']}, breach_rate={row['breach_rate']:.4f}")

top_machine = int(ustar.iloc[0]['machine_id'])
print(f"\nAnalyzing top machine: {top_machine}")

# Load test allocations parquet
df_alloc = pd.read_parquet('results/phase2/phase1_test_allocations.parquet')
print("Parquet columns:", df_alloc.columns.tolist())

# Filter for top machine and Adaptive_U_star
m_alloc = df_alloc[(df_alloc['machine_id'] == top_machine) & (df_alloc['policy'] == 'Adaptive_U_star')].sort_values('bin_idx')
print(f"Total rows for machine {top_machine}, U*: {len(m_alloc)}")

# Find consecutive breach streaks (is_breach == True)
m_alloc['streak_id'] = (~m_alloc['is_breach']).cumsum()
breach_runs = m_alloc[m_alloc['is_breach']].groupby('streak_id')

streaks = []
for sid, group in breach_runs:
    streaks.append({
        'streak_id': sid,
        'start_bin': group['bin_idx'].min(),
        'end_bin': group['bin_idx'].max(),
        'length_bins': len(group),
        'duration_min': len(group) * 5.0,
        'mean_y_true': group['y_true'].mean(),
        'mean_alloc': group['allocation'].mean(),
        'mean_shortfall': (group['y_true'] - group['allocation']).mean(),
        'max_shortfall': (group['y_true'] - group['allocation']).max()
    })

streaks_df = pd.DataFrame(streaks).sort_values('length_bins', ascending=False)
print("\nLongest breach streaks on this machine:")
print(streaks_df.head(5).to_string())
