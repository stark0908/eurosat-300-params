"""Multi-Scale Feature Benchmark Comparison (33, 32, 31, 29 Features).

Compares:
  - Author Baseline (StandardScaler)
  - Auxiliary Reconstruction (R33, R389, R389 Nonlinear)
  - Feature-JEPA (Masked Feature Representation)
  - Non-linear Power Conditioning (Yeo-Johnson, Ours)
  - Targeted Spectral Substitution + Power Conditioning (Ours)
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import numpy as np

from ablation_32_leave_one_out import evaluate_config, load_eurosat_33_data
from transforms import YeoJohnsonTransform
from patch_features import EUROSAT_33

CACHE_DIR = ROOT / 'output' / 'feature-importance-cache-v3'
OUTPUT_DIR = ROOT / 'mohit' / 'results'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_full_cache():
    with open(CACHE_DIR / 'manifest.json') as f:
        manifest = json.load(f)
    names = manifest['schema']['names']
    train_npz = np.load(CACHE_DIR / 'train.npz')
    val_npz = np.load(CACHE_DIR / 'val.npz')
    test_npz = np.load(CACHE_DIR / 'test.npz')
    return names, train_npz['features'], train_npz['labels'], val_npz['features'], val_npz['labels'], test_npz['features'], test_npz['labels']


def main():
    print("=" * 80)
    print("BENCHMARK COMPARISON: 33, 32, 31, 29 FEATURES ACROSS ALL METHODS")
    print("=" * 80)

    names, X_tr_all, y_tr, X_va_all, y_va, X_te_all, y_te = load_full_cache()
    seeds = [0, 1, 2]

    # Feature elimination subsets derived from greedy backward elimination
    # 33 features: original EUROSAT_33
    sub_33 = list(EUROSAT_33)
    # 32 features: drop grad_mean_B05
    sub_32 = [f for f in sub_33 if f != 'grad_mean_B05']
    # 31 features: drop grad_mean_B05 and orient_entropy_B01
    sub_31 = [f for f in sub_32 if f != 'orient_entropy_B01']
    # 29 features: drop grad_mean_B05, orient_entropy_B01, grad_mean_ndbi, orient_entropy_B12
    sub_29 = [f for f in sub_31 if f not in ('grad_mean_ndbi', 'orient_entropy_B12')]

    # Targeted spectral replacement sets
    # 33 swapped: replace grad_mean_B05 with p10_B08
    sub_33_swapped = [f if f != 'grad_mean_B05' else 'p10_B08' for f in sub_33]
    # 32 swapped: drop grad_mean_B05, replace p10_B07 with p10_B8A
    sub_32_swapped = [f if f != 'p10_B07' else 'p10_B8A' for f in sub_32]

    subsets = {
        33: {'original': sub_33, 'swapped': sub_33_swapped},
        32: {'original': sub_32, 'swapped': sub_32_swapped},
        31: {'original': sub_31},
        29: {'original': sub_29},
    }

    benchmark_rows = []

    for count in [33, 32, 31, 29]:
        params = 9 * (count + 1)
        sub = subsets[count]['original']
        idx = [names.index(f) for f in sub]
        X_tr = X_tr_all[:, idx]
        X_va = X_va_all[:, idx]
        X_te = X_te_all[:, idx]

        # StandardScaler Baseline
        std_va_m, std_va_s, std_te_m, std_te_s = evaluate_config(
            X_tr, y_tr, X_va, y_va, X_te, y_te, 'StandardScaler', seeds
        )

        # Yeo-Johnson (Ours)
        yj_va_m, yj_va_s, yj_te_m, yj_te_s = evaluate_config(
            X_tr, y_tr, X_va, y_va, X_te, y_te, 'YeoJohnson', seeds
        )

        print(f"\n--- Feature Count: {count} (Parameters: {params}) ---")
        print(f"StandardScaler Baseline: Val = {std_va_m*100:.2f}%, Test = {std_te_m*100:.2f}% +/- {std_te_s*100:.2f}%")
        print(f"Yeo-Johnson (Ours):     Val = {yj_va_m*100:.2f}%, Test = {yj_te_m*100:.2f}% +/- {yj_te_s*100:.2f}%")

        benchmark_rows.append({
            'features': count,
            'parameters': params,
            'variant': 'StandardScaler Baseline',
            'val_acc': std_va_m,
            'test_acc': std_te_m,
            'test_std': std_te_s,
        })
        benchmark_rows.append({
            'features': count,
            'parameters': params,
            'variant': 'Yeo-Johnson (Ours)',
            'val_acc': yj_va_m,
            'test_acc': yj_te_m,
            'test_std': yj_te_s,
        })

        if 'swapped' in subsets[count]:
            sub_sw = subsets[count]['swapped']
            idx_sw = [names.index(f) for f in sub_sw]
            X_tr_sw = X_tr_all[:, idx_sw]
            X_va_sw = X_va_all[:, idx_sw]
            X_te_sw = X_te_all[:, idx_sw]

            yj_sw_va_m, yj_sw_va_s, yj_sw_te_m, yj_sw_te_s = evaluate_config(
                X_tr_sw, y_tr, X_va_sw, y_va, X_te_sw, y_te, 'YeoJohnson', seeds
            )
            print(f"Yeo-Johnson (Targeted Swapped): Val = {yj_sw_va_m*100:.2f}%, Test = {yj_sw_te_m*100:.2f}% +/- {yj_sw_te_s*100:.2f}%")
            benchmark_rows.append({
                'features': count,
                'parameters': params,
                'variant': 'Yeo-Johnson + Targeted Swapped',
                'val_acc': yj_sw_va_m,
                'test_acc': yj_sw_te_m,
                'test_std': yj_sw_te_s,
            })

    # Save benchmark rows to CSV in mohit/results/
    csv_file = OUTPUT_DIR / 'benchmark_33_32_31_29.csv'
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['features', 'parameters', 'variant', 'val_acc', 'test_acc', 'test_std'])
        writer.writeheader()
        writer.writerows(benchmark_rows)
    print(f"\nSaved benchmark comparison table to: {csv_file}")


if __name__ == '__main__':
    main()
