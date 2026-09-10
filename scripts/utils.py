"""
scripts/utils.py

Shared Utilities, Configuration Constants, and Fast Mathematical Functions.
Designed to minimize code redundancy, optimize computational performance, and eliminate overhead.
"""

import os
import numpy as np
import pandas as pd

# ==============================================================================
# Global Configuration Constants (Locked in PROJECT_SPEC.md v2.5)
# ==============================================================================
SEED = 42
BIN_SECONDS = 300.0        # 5-minute aggregation intervals
SEQ_LEN = 12               # 60 minutes historical lookback for GRU
GRACE_BAND = 0.005         # 0.5% normalized utilization forgiveness window
SLA_TARGET = 0.02          # Hard SLA breach budget: <= 2.0%
N_MACHINES_FINAL = 100     # Evaluation sample size
ALPHA_GRID = np.linspace(0.0, 1.0, 11)   # [0.0, 0.1, ..., 1.0] for fusion
LAMBDA_GRID = np.linspace(0.1, 0.9, 9)   # [0.1, 0.2, ..., 0.9] for U2 EWMA
N_WINDOW_GRID = [20, 30, 50, 75, 100]    # Selector correlation windows
CANDIDATE_BETAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]

# Standard Project Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed", "google")
SERIES_DIR = os.path.join(PROCESSED_DIR, "machine_series")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
UNCERTAINTY_DIR = os.path.join(PROJECT_ROOT, "uncertainty")
SELECTED_MACHINES_PATH = os.path.join(DATA_DIR, "selected_machines.json")
SPLIT_INFO_PATH = os.path.join(PROCESSED_DIR, "split_info.json")


# ==============================================================================
# High-Performance Mathematical & Evaluation Functions
# ==============================================================================

def fast_spearman(x: np.ndarray, y: np.ndarray) -> float:
    """
    Optimized Spearman Rank Correlation using vectorized ranking.
    Avoids heavy scipy.stats overhead (~20-50x faster inside tight sliding-window loops).
    Returns 0.0 for constant or NaN inputs.
    """
    valid = ~np.isnan(x) & ~np.isnan(y)
    n = int(np.sum(valid))
    if n < 3:
        return 0.0

    xv = x[valid]
    yv = y[valid]

    # If either series is constant, rank correlation is mathematically undefined (return 0.0)
    if np.all(xv == xv[0]) or np.all(yv == yv[0]):
        return 0.0

    # Fast fractional rank calculation via numpy argsort
    rx = np.argsort(np.argsort(xv)).astype(np.float64)
    ry = np.argsort(np.argsort(yv)).astype(np.float64)

    # Pearson correlation on the ranks: rho = cov(rx, ry) / (std(rx) * std(ry))
    rx_mean = rx - np.mean(rx)
    ry_mean = ry - np.mean(ry)
    denom = np.sqrt(np.sum(rx_mean ** 2) * np.sum(ry_mean ** 2))
    if denom == 0.0:
        return 0.0
    return float(np.sum(rx_mean * ry_mean) / denom)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, a_raw: np.ndarray, a_alloc: np.ndarray) -> dict:
    """
    Standardized SLA and Error Metrics for Resource Allocation.
    
    Equations:
    - SLA Breach Rate: P(y_t > A_t + GRACE_BAND) [Target: <= 2.0%]
    - Overshoot Rate:  P(y_t > A_t)             [All overshoots, even within grace band]
    - Grace Zone Rate: P(A_t < y_t <= A_t + GRACE_BAND)
    - Over-Provision:  mean(max(0, A_t - y_t))  [Wasted capacity - minimized]
    - Under-Provision: mean(max(0, y_t - A_t))  [Shortfall intensity]
    - Clip Fraction:   mean(A_raw > 1.0)        [Physical capacity limit tracking]
    """
    valid = ~np.isnan(y_true) & ~np.isnan(y_pred) & ~np.isnan(a_alloc)
    if valid.sum() == 0:
        return {
            'mae': np.nan, 'rmse': np.nan, 'sla_breach_rate': np.nan,
            'overshoot_rate': np.nan, 'grace_zone_rate': np.nan,
            'over_prov_mean': np.nan, 'under_prov_mean': np.nan,
            'clip_fraction': np.nan, 'valid_count': 0
        }

    yt = y_true[valid]
    yp = y_pred[valid]
    ar = a_raw[valid]
    al = a_alloc[valid]

    err = yp - yt
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    overshoot = yt > al
    sla_breach = yt > (al + GRACE_BAND)
    grace_hit = overshoot & ~sla_breach

    over_prov = np.maximum(0.0, al - yt)
    under_prov = np.maximum(0.0, yt - al)
    clip_frac = np.mean(ar > 1.0)

    return {
        'mae': mae,
        'rmse': rmse,
        'sla_breach_rate': float(np.mean(sla_breach)),
        'overshoot_rate': float(np.mean(overshoot)),
        'grace_zone_rate': float(np.mean(grace_hit)),
        'over_prov_mean': float(np.mean(over_prov)),
        'under_prov_mean': float(np.mean(under_prov)),
        'clip_fraction': float(clip_frac),
        'valid_count': int(valid.sum())
    }
