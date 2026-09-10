# PROJECT_SPEC.md
# Uncertainty-Aware Hybrid Resource Allocation Framework
# Source of Truth — Spec Frozen v2.5

## 1. Hardware & Environment
- Target GPU: NVIDIA GeForce RTX 2050 (4 GB VRAM) — CUDA enabled
- Seed: 42 (fixed globally across all operations)
- Workspace: E:/Cloud/project/
- Data root: E:/Cloud/project/data/

## 2. Dataset & Sampling
- Dataset: Google Cluster Trace v2 (2011)
- Candidates: 300 oversampled machines from `machine_events`
- Final Sample: 100 randomly selected eligible machines (seed=42)
- Eligibility Criteria (checked after aggregation):
  - Missing fraction: NaN <= 20%
  - Variance: std > 0.01 (non-flat workload)
- Selected machine IDs saved to `data/selected_machines.json` before modeling

## 3. CPU Aggregation
- Temporal Bins: 5-minute fixed bins
- Aggregation Method: Time-weighted interval overlap
  - Task usage records overlap bins proportionally to overlap duration
  - Bin CPU = sum(cpu_rate * overlap_seconds) / 300.0
- CPU Scale: Physical fraction [0, 1] — NO per-machine min-max normalization
- Zero CPU: Valid observation (machine observed idle)
- Missing Bins: NaN (no task records during interval)
- Gap Filling: Linear interpolation for <= 2 consecutive NaNs
- Long Gaps: Remain NaN (unfilled)
- Machine Discard: Discard if > 20% NaN after gap filling

## 4. Per-Machine Temporal Isolation Rule (MANDATORY)
ALL temporal operations are performed independently within each machine:
- Lag features -> computed per machine (never across machine boundaries)
- Rolling features (roll_mean, roll_std, U3) -> computed per machine
- GRU sequences -> generated per machine; windows NEVER cross machine boundaries
- U2 EWMA -> computed per machine with independent causal initialization per machine
- U3 rolling std -> computed per machine from that machine's full continuous series
- Selector error history & window -> evaluated per machine; val-to-test lookback is per machine
- Prediction errors & allocations -> computed per machine
- `full_series` always refers to ONE machine's continuous series, never concatenated across machines

## 5. Temporal Splits & Boundaries
- Chronological Split: 70% Train / 15% Validation / 15% Test (per machine)
- Uniform Timestamps: Exactly identical test timestamps used by all models and baselines
- Boundary Lookback Rules (drawn from each machine's continuous series):
  - Naive (B0): ŷ[t] = y[t-1] (from full_series.shift(1))
  - Seasonal Naive (B0b): ŷ[t] = y[t-288] (from full_series.shift(288))
  - U3 at time t: std(y[t-6 : t-1]) (from full_series.shift(1).rolling(6).std())
  - U2 Val Init: U2_val[0] = |fused_train[-1] - y_train[-1]| (last train error)
  - U2 Test Init: U2_test[0] = |fused_val[-1] - y_val[-1]| (last val error)

## 6. Sequence Boundary Rules (GRU)
- Input Window: For prediction at t, input contains SEQ_LEN=12 observations from the machine's continuous past
- Split Crossing: Input windows may cross train->val and val->test boundaries using past observations (no leakage)
- Gap Enforcement: GRU windows must NEVER cross a non-contiguous timestamp gap (>2 NaN steps) or machine boundary
- Prediction Target: Target belongs strictly to the assigned split

## 7. Causal Feature Set (Dual-Tier Architecture)
- **Tier 1: Base Model (7 features - Frozen Baseline)**:
  1. `cpu`: Physical CPU fraction [0, 1] (target at t)
  2. `cpu_lag1`: cpu(t-1)
  3. `cpu_lag2`: cpu(t-2)
  4. `roll_mean`: mean(cpu, t-6 to t-1) [causal, excludes t]
  5. `roll_std`: std(cpu, t-6 to t-1) [causal, excludes t]
  6. `hour_sin`: sin(2π * hour / 24)
  7. `hour_cos`: cos(2π * hour / 24)

- **Tier 2: Extended Model (Inspected & Data-Grounded Features)**:
  Verified against actual trace columns `['start_time', 'end_time', 'machine_id', 'cpu_rate', 'canonical_memory_usage', 'disk_io_time']`:
  8. `mem`: canonical_memory_usage(t)
  9. `mem_lag1`: canonical_memory_usage(t-1)
  10. `mem_roll_mean`: mean(canonical_memory_usage, t-6 to t-1)
  11. `disk`: disk_io_time(t-1)
  12. `cpu_diff1`: cpu_lag1 - cpu_lag2 (workload rate of change / momentum)
  13. `roll_max`: max(cpu, t-6 to t-1) (local surge peak)
  14. `dow_sin`: sin(2π * day_of_week / 7)
  15. `dow_cos`: cos(2π * day_of_week / 7)

- **Ablation Protocol**:
  - `GRU_base` (7 features) vs. `GRU_extended` (15 features) are evaluated under strictly identical 70/15/15 chronological splits and seeds (seed=42).
  - No uninspected or unavailable features may be injected into the pipeline.

## 8. Forecasting Models
- **GRU Predictor (Global Model)**:
  - Input: (batch, 12, 7)
  - Hidden: 32 units, 1 layer
  - Dropout: Explicit post-GRU nn.Dropout(p=0.2) applied to hidden state
  - Output: Linear + Sigmoid -> guarantees output in (0, 1)
  - Training: Adam (lr=1e-3), ReduceLROnPlateau, MSELoss, batch_size=32, shuffle=False
- **Holt-Winters (Per Machine)**:
  - Additive trend, additive seasonal
  - Seasonality: m in {48, 144, 288}, tuned on validation MAE
  - Refit: Refit best m on combined (train + val) before test forecasting
- **Forecast Fusion**:
  - ŷ_t = α_f * ŷ_HW + (1 - α_f) * ŷ_GRU
  - α_f in {0.0, 0.1, ..., 1.0}, selected by validation MAE

## 9. Uncertainty Signals
- **U1 (Model Disagreement)**:
  - U1_t = |ŷ_HW - ŷ_GRU|
- **U2 (EWMA Recent Forecast Error)**:
  - U2_t = λ * |fused_{t-1} - y_{t-1}| + (1 - λ) * U2_{t-1}
  - λ in {0.1, ..., 0.9}, tuned to maximize validation Spearman ρ with absolute error
  - Causal init: U2_val[0] = last train error; U2_test[0] = last val error
- **U3 (Workload Volatility)**:
  - U3_t = std(y[t-6 : t-1]) from machine's full_series.shift(1).rolling(6).std()

## 10. Adaptive Reliability Selector
- Window: N in {20, 30, 50, 75, 100}, selected by mean winning Spearman ρ on validation
- Causal Lookback: Concatenated (val + test) per machine allows test t=0 to look back N steps into validation
- NaN Handling: safe_spearman() returns 0.0 for constant or NaN inputs
- Positive-Rho Rule: Prefer signals with ρ > 0. If all ρ <= 0, flag timestep and fallback to max ρ
- Selection:
  - i*_t = argmax_i ρ_i(t) for i in {U1, U2, U3}
  - U*_t = U_{i*_t, t}

## 11. Risk-Aware Resource Allocation
- Unbounded Allocation: A_raw_t = ŷ_t + β_buf * U*_t
- Physical Capacity Ceiling: A_t = clip(A_raw_t, 0.0, 1.0)
- Buffer Calibration:
  - β_buf in [0.1, 0.2, ..., 20.0]
  - Pick smallest β_buf satisfying validation SLA breach rate <= 1.0% (99% SLA compliance target)
  - Hard failure (RuntimeError) if no candidate β satisfies validation SLA
- Clip Fraction Tracking: clip_fraction = mean(A_raw > 1.0); warn if > 1%

## 12. Baselines & Benchmark Configurations
- B0: Naive (y_{t-1}) + Fixed 10% relative buffer (A = clip(ŷ * 1.10, 0, 1))
- B0b: Seasonal Naive (y_{t-288}) + Fixed 10% relative buffer
- B1: GRU only (base 7-features) + Fixed 10% relative buffer
- B2: HW only + Fixed 10% relative buffer
- B3: Fusion (base) + Fixed 10% relative buffer
- B4a: Fusion + U1 always + β-tuned buffer
- B4b: Fusion + U2 always + β-tuned buffer
- B4c: Fusion + U3 always + β-tuned buffer
- Proposed (P_base): Fusion (base 7-features) + Adaptive U* + β-tuned buffer [Locked Base]
- Proposed (P_ext): Fusion (extended 15-features) + Adaptive U* + β-tuned buffer [Ablation Study]

## 13. Evaluation Metrics
- SLA Breach Rate: P(y_t > A_t + 0.005) [Target: <= 1.0% (99% SLA Compliance Target)]
- Overshoot Rate: P(y_t > A_t) [Reported]
- Grace-Zone Rate: P(A_t < y_t <= A_t + 0.005) [Reported]
- Over-provisioning: mean(max(0, A_t - y_t)) [Minimized]
- Under-provisioning: mean(max(0, y_t - A_t)) [Reported]
- Clip Fraction: mean(A_raw_t > 1.0) [Reported]
- Aggregation: Macro-average across all 100 machines (mean ± std)
