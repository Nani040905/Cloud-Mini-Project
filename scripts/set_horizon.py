"""
scripts/set_horizon.py

Utility to configure the time horizon (number of days/bins) and train/val/test splits.
Can be run via CLI arguments or interactively via user input prompts.

Usage:
  # Interactive mode (prompts for input)
  python scripts/set_horizon.py

  # CLI argument mode
  python scripts/set_horizon.py --days 10
  python scripts/set_horizon.py --days 5 --train 0.70 --val 0.15 --test 0.15
  python scripts/set_horizon.py --reset   # Resets to full 29 days (8,352 bins)
"""

import os
import sys
import json
import argparse
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SPLIT_INFO_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "google", "split_info.json")
SERIES_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "google", "machine_series")
SELECTED_MACHINES_PATH = os.path.join(PROJECT_ROOT, "data", "selected_machines.json")

BINS_PER_HOUR = 12
BINS_PER_DAY = 288  # 24 * 12 (5-minute bins)
FULL_DAYS = 29.0
FULL_BINS = 8352


def get_split_indices(total_bins: int, train_ratio: float = 0.70, val_ratio: float = 0.15):
    """Calculate contiguous chronological slice boundaries."""
    train_end = int(round(total_bins * train_ratio))
    val_end = int(round(total_bins * (train_ratio + val_ratio)))
    return {
        'train': [0, train_end],
        'val': [train_end, val_end],
        'test': [val_end, total_bins]
    }


def update_split_info(days: float, train_ratio: float = 0.70, val_ratio: float = 0.15, test_ratio: float = 0.15):
    """Updates split_info.json for all machines with custom day horizon and ratios."""
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0, atol=1e-4):
        raise ValueError(f"Split ratios must sum to 1.0! Got {train_ratio + val_ratio + test_ratio:.4f}")

    if days <= 0 or days > FULL_DAYS:
        raise ValueError(f"Days must be between 0.1 and {FULL_DAYS}! Got {days}")

    total_bins = int(round(days * BINS_PER_DAY))
    splits = get_split_indices(total_bins, train_ratio=train_ratio, val_ratio=val_ratio)

    # Determine machine IDs
    if os.path.exists(SPLIT_INFO_PATH):
        with open(SPLIT_INFO_PATH, 'r') as f:
            existing = json.load(f)
        machine_ids = list(existing.keys())
    elif os.path.exists(SELECTED_MACHINES_PATH):
        with open(SELECTED_MACHINES_PATH, 'r') as f:
            raw = json.load(f)
        machine_ids = [str(m) for m in (raw if isinstance(raw, list) else raw.keys())]
    else:
        # Fallback to series directory
        machine_ids = [f.replace('.parquet', '') for f in os.listdir(SERIES_DIR) if f.endswith('.parquet')]

    updated = {}
    for mid in machine_ids:
        updated[str(mid)] = {
            'total_bins': total_bins,
            'train_range': splits['train'],
            'val_range': splits['val'],
            'test_range': splits['test']
        }

    os.makedirs(os.path.dirname(SPLIT_INFO_PATH), exist_ok=True)
    with open(SPLIT_INFO_PATH, 'w') as f:
        json.dump(updated, f, indent=2)

    tr_b = splits['train'][1] - splits['train'][0]
    va_b = splits['val'][1] - splits['val'][0]
    te_b = splits['test'][1] - splits['test'][0]

    print("\n" + "=" * 70)
    print(f" [SUCCESS] Time Horizon Configured: {days:.2f} Days ({total_bins:,} total bins)")
    print("=" * 70)
    print(f"  Target File : {SPLIT_INFO_PATH}")
    print(f"  Machines    : {len(machine_ids)} benchmark servers updated")
    print("-" * 70)
    print(f"  Train Split : Bins {splits['train'][0]:5d} -> {splits['train'][1]:5d} ({tr_b:5d} bins | {tr_b/BINS_PER_DAY:5.2f} days | {train_ratio*100:4.1f}%)")
    print(f"  Val Split   : Bins {splits['val'][0]:5d} -> {splits['val'][1]:5d} ({va_b:5d} bins | {va_b/BINS_PER_DAY:5.2f} days | {val_ratio*100:4.1f}%)")
    print(f"  Test Split  : Bins {splits['test'][0]:5d} -> {splits['test'][1]:5d} ({te_b:5d} bins | {te_b/BINS_PER_DAY:5.2f} days | {test_ratio*100:4.1f}%)")
    print("=" * 70)
    print("  You can now fit models or run benchmarks on this window:")
    print("    python run_pipeline.py --all")
    print("=" * 70 + "\n")
    return updated


def main():
    parser = argparse.ArgumentParser(description="Configure Time Horizon & Split Window (Interactive or CLI Args)")
    parser.add_argument("--days", type=float, default=None, help="Number of days (e.g. 10, 5, 14, 29)")
    parser.add_argument("--train", type=float, default=0.70, help="Train ratio (default: 0.70)")
    parser.add_argument("--val", type=float, default=0.15, help="Validation ratio (default: 0.15)")
    parser.add_argument("--test", type=float, default=0.15, help="Test ratio (default: 0.15)")
    parser.add_argument("--reset", action="store_true", help="Reset to full 29-day dataset (8,352 bins)")
    args = parser.parse_args()

    if args.reset:
        print("\nResetting to Full 29-day Trace (8,352 bins)...")
        update_split_info(days=29.0, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
        return

    days = args.days
    train_ratio = args.train
    val_ratio = args.val
    test_ratio = args.test

    # If --days was not provided on CLI, prompt interactively
    if days is None:
        print("\n" + "=" * 70)
        print("          CUSTOM TIME HORIZON & SPLIT CONFIGURATOR")
        print("=" * 70)
        print("  Quick Presets:")
        print("    -  1 Day  =   288 bins (Instant sanity check)")
        print("    -  5 Days = 1,440 bins (Ultra-fast trial)")
        print("    - 10 Days = 2,880 bins (Fast 3x acceleration, 7 days train)")
        print("    - 15 Days = 4,320 bins (Balanced mid-trace)")
        print("    - 29 Days = 8,352 bins (Full Trace)")
        print("-" * 70)

        d_input = input("Enter number of days (1 to 29, or Enter for 10): ").strip()
        if not d_input:
            days = 10.0
        else:
            try:
                days = float(d_input)
            except ValueError:
                print(f"[Error] Invalid number: {d_input}")
                sys.exit(1)

        custom_ratios = input("Customize train/val/test ratios? (y/N): ").strip().lower()
        if custom_ratios == 'y':
            try:
                train_ratio = float(input("  Train Ratio (default 0.70): ") or "0.70")
                val_ratio = float(input("  Val Ratio   (default 0.15): ") or "0.15")
                test_ratio = float(input("  Test Ratio  (default 0.15): ") or "0.15")
            except ValueError:
                print("[Error] Invalid ratio input, using default 70/15/15.")
                train_ratio, val_ratio, test_ratio = 0.70, 0.15, 0.15

    update_split_info(days=days, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)


if __name__ == "__main__":
    main()
