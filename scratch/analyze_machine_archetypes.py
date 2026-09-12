import pandas as pd
import numpy as np

# Load machine comparison
df_comp = pd.read_csv('results/phase2/machine_comparison_u2_vs_ustar.csv')
# delta = Ustar_breach_rate - U2_breach_rate
# Sort by delta descending: top rows are where U2 is much better than U*
# bottom rows are where U* is much better than U2
top_u2_wins = df_comp.sort_values('delta', ascending=False).head(10)
top_ustar_wins = df_comp.sort_values('delta', ascending=True).head(10)

print("=== TOP 10 MACHINES WHERE STATIC U2 WINS (U* has higher breach rate) ===")
print(top_u2_wins[['machine_id', 'U2_breach_rate', 'Ustar_breach_rate', 'delta', 'U2_overprovision', 'Ustar_overprovision']].to_string(index=False))

print("\n=== TOP 10 MACHINES WHERE ADAPTIVE U* WINS (U* has lower breach rate) ===")
print(top_ustar_wins[['machine_id', 'U2_breach_rate', 'Ustar_breach_rate', 'delta', 'U2_overprovision', 'Ustar_overprovision']].to_string(index=False))

# Load selector_test to inspect chosen signal distribution, errors, volatility
df_sel = pd.read_csv('uncertainty/selector_test.csv')

# Load raw series / processed features if needed to compute volatility, trend, periodicity
# We can compute these on the test split or overall series from selector_test
def compute_machine_stats(mid):
    sub = df_sel[df_sel['machine_id'] == mid].copy()
    y = sub['y_true'].values
    err = sub['abs_error'].values
    u1 = sub['U1'].values
    u2 = sub['U2'].values
    u3 = sub['U3'].values
    chosen = sub['chosen_signal'].value_counts(normalize=True).to_dict()
    
    # Volatility
    vol_mean = np.mean(u3)
    vol_std = np.std(y)
    
    # Trend: slope of linear regression over test split
    t = np.arange(len(y))
    slope, _ = np.polyfit(t, y, 1)
    
    # Forecast error
    mae = np.mean(err)
    bias = np.mean(sub['y_true'] - sub['fused_pred'])
    
    # Model disagreement U1
    disagree = np.mean(u1)
    
    # Residual autocorrelation lag 1
    res = sub['y_true'] - sub['fused_pred']
    res_autocorr = res.autocorr(lag=1)
    
    return {
        'machine_id': mid,
        'y_mean': np.mean(y),
        'y_std': vol_std,
        'u3_vol_mean': vol_mean,
        'slope_per_1000bins': slope * 1000,
        'mae': mae,
        'bias': bias,
        'disagree_u1': disagree,
        'res_autocorr': res_autocorr,
        'pct_U1': chosen.get('U1', 0.0),
        'pct_U2': chosen.get('U2', 0.0),
        'pct_U3': chosen.get('U3', 0.0)
    }

u2_stats = [compute_machine_stats(int(m)) for m in top_u2_wins['machine_id']]
ustar_stats = [compute_machine_stats(int(m)) for m in top_ustar_wins['machine_id']]

df_u2_stats = pd.DataFrame(u2_stats)
df_ustar_stats = pd.DataFrame(ustar_stats)

print("\n=== CHARACTERISTICS: MACHINES FAVORING STATIC U2 (Top 10) ===")
print(df_u2_stats.to_string(index=False))
print("\nMean of Top U2-favoring machines:")
print(df_u2_stats.mean())

print("\n=== CHARACTERISTICS: MACHINES FAVORING ADAPTIVE U* (Top 10) ===")
print(df_ustar_stats.to_string(index=False))
print("\nMean of Top U*-favoring machines:")
print(df_ustar_stats.mean())

# Also aggregate across all 26 U2-favoring vs 67 U*-favoring machines
all_u2_better = df_comp[df_comp['delta'] > 0]['machine_id']
all_ustar_better = df_comp[df_comp['delta'] < 0]['machine_id']

all_u2_stats = pd.DataFrame([compute_machine_stats(int(m)) for m in all_u2_better])
all_ustar_stats = pd.DataFrame([compute_machine_stats(int(m)) for m in all_ustar_better])

print("\n=== MACRO AVERAGE ACROSS ALL MACHINES WHERE U2 IS BETTER (N=26) ===")
print(all_u2_stats.drop(columns='machine_id').mean())

print("\n=== MACRO AVERAGE ACROSS ALL MACHINES WHERE U* IS BETTER (N=67) ===")
print(all_ustar_stats.drop(columns='machine_id').mean())
