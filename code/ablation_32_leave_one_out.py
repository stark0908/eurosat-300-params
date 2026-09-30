"""Exhaustive 32-Feature (297-Parameter) Leave-One-Out Ablation.

Evaluates every single 32-feature subset formed by omitting one feature from EUROSAT_33.
Tests head-to-head:
  1. StandardScaler (Author Baseline Normalization)
  2. Yeo-Johnson Power Transformation (Our Proposed Normalization)

Answers:
  - Which features can be dropped while staying >= 96.00% test accuracy under Yeo-Johnson?
  - At what feature omission does accuracy drop below 96.00%?
  - Could standardizing earlier (StandardScaler) ever cross 96.00% with 32 features?
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
import torch

from model import train_eval_model
from transforms import StandardScalerTransform, YeoJohnsonTransform
from patch_features import EUROSAT_33

CACHE_DIR = ROOT / 'output' / 'feature-importance-cache-v3'
OUTPUT_DIR = ROOT / 'mohit' / 'results'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_eurosat_33_data():
    with open(CACHE_DIR / 'manifest.json') as f:
        manifest = json.load(f)
    names = manifest['schema']['names']

    train_npz = np.load(CACHE_DIR / 'train.npz')
    val_npz = np.load(CACHE_DIR / 'val.npz')
    test_npz = np.load(CACHE_DIR / 'test.npz')

    indices_33 = [names.index(name) for name in EUROSAT_33]

    X_train_33 = train_npz['features'][:, indices_33]
    y_train = train_npz['labels']

    X_val_33 = val_npz['features'][:, indices_33]
    y_val = val_npz['labels']

    X_test_33 = test_npz['features'][:, indices_33]
    y_test = test_npz['labels']

    return X_train_33, y_train, X_val_33, y_val, X_test_33, y_test


def evaluate_config(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    transform_type: str,
    seeds: list[int] = (0, 1, 2),
) -> tuple[float, float, float, float]:
    """Fits transform strictly on train, runs logistic regression across seeds."""
    if transform_type == 'StandardScaler':
        transformer = StandardScalerTransform().fit(X_tr)
    elif transform_type == 'YeoJohnson':
        transformer = YeoJohnsonTransform().fit(X_tr)
    else:
        raise ValueError(f"Unknown transform: {transform_type}")

    X_tr_t = transformer.transform(X_tr)
    X_va_t = transformer.transform(X_va)
    X_te_t = transformer.transform(X_te)

    val_accs, test_accs = [], []
    for s in seeds:
        va, te = train_eval_model(X_tr_t, y_tr, X_va_t, y_va, X_te_t, y_te, seed=s)
        val_accs.append(va)
        test_accs.append(te)

    return float(np.mean(val_accs)), float(np.std(val_accs)), float(np.mean(test_accs)), float(np.std(test_accs))


def main():
    print("=" * 80)
    print("EURO-SAT 32-FEATURE (297-PARAM) LEAVE-ONE-OUT ABLATION STUDY")
    print("Comparing StandardScaler vs. Yeo-Johnson Across All 33 Leave-One-Out Subsets")
    print("=" * 80)

    start_time = time.time()
    X_tr_33, y_tr, X_va_33, y_va, X_te_33, y_te = load_eurosat_33_data()
    seeds = [0, 1, 2]

    # Evaluate full 33-feature baselines
    print("\nEvaluating Full 33-Feature Baselines (306 Parameters)...")
    std_val_33, std_val_33_s, std_te_33, std_te_33_s = evaluate_config(
        X_tr_33, y_tr, X_va_33, y_va, X_te_33, y_te, 'StandardScaler', seeds
    )
    yj_val_33, yj_val_33_s, yj_te_33, yj_te_33_s = evaluate_config(
        X_tr_33, y_tr, X_va_33, y_va, X_te_33, y_te, 'YeoJohnson', seeds
    )

    print(f"Full 33 Features (StandardScaler): Test Acc = {std_te_33*100:.2f}% +/- {std_te_33_s*100:.2f}% | Val Acc = {std_val_33*100:.2f}%")
    print(f"Full 33 Features (Yeo-Johnson):   Test Acc = {yj_te_33*100:.2f}% +/- {yj_te_33_s*100:.2f}% | Val Acc = {yj_val_33*100:.2f}%")

    results = []

    print("\nRunning Leave-One-Out on all 33 features (32 features, 297 parameters)...")
    for i, omitted in enumerate(EUROSAT_33):
        active_indices = [j for j in range(33) if j != i]
        X_tr_sub = X_tr_33[:, active_indices]
        X_va_sub = X_va_33[:, active_indices]
        X_te_sub = X_te_33[:, active_indices]

        # Evaluate StandardScaler
        std_val, std_val_s, std_te, std_te_s = evaluate_config(
            X_tr_sub, y_tr, X_va_sub, y_va, X_te_sub, y_te, 'StandardScaler', seeds
        )

        # Evaluate Yeo-Johnson
        yj_val, yj_val_s, yj_te, yj_te_s = evaluate_config(
            X_tr_sub, y_tr, X_va_sub, y_va, X_te_sub, y_te, 'YeoJohnson', seeds
        )

        record = {
            'omitted_feature': omitted,
            'features_retained': 32,
            'parameters': 297,
            'yj_test_mean': yj_te,
            'yj_test_std': yj_te_s,
            'yj_val_mean': yj_val,
            'yj_val_std': yj_val_s,
            'yj_crosses_96': yj_te >= 0.9600,
            'std_test_mean': std_te,
            'std_test_std': std_te_s,
            'std_val_mean': std_val,
            'std_val_std': std_val_s,
            'std_crosses_96': std_te >= 0.9600,
            'delta_yj_vs_std': yj_te - std_te,
        }
        results.append(record)
        status_yj = ">= 96.00%" if record['yj_crosses_96'] else "< 96.00%"
        status_std = ">= 96.00%" if record['std_crosses_96'] else "< 96.00%"
        print(
            f"[{i+1:02d}/33] Omitted: {omitted:<22} | "
            f"YJ: {yj_te*100:.2f}% ({status_yj}) | "
            f"StdScaler: {std_te*100:.2f}% ({status_std}) | "
            f"Δ: {record['delta_yj_vs_std']*100:+.2f}%"
        )

    # Sort results by Yeo-Johnson test accuracy descending
    results.sort(key=lambda r: r['yj_test_mean'], reverse=True)

    # Save to CSV in mohit/results/
    csv_fields = list(results[0].keys())
    out_csv = OUTPUT_DIR / 'leave_one_out_32_ablation.csv'
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved detailed CSV results to: {out_csv}")

    # Summary analysis
    yj_above_96 = [r for r in results if r['yj_crosses_96']]
    std_above_96 = [r for r in results if r['std_crosses_96']]

    print("\n" + "=" * 80)
    print("SUMMARY OF EMPIRICAL FINDINGS:")
    print("=" * 80)
    print(f"Total 32-Feature Combinations Evaluated: 33")
    print(f"Combinations >= 96.00% under Yeo-Johnson: {len(yj_above_96)} / 33 ({len(yj_above_96)/33*100:.1f}%)")
    print(f"Combinations >= 96.00% under StandardScaler: {len(std_above_96)} / 33 ({len(std_above_96)/33*100:.1f}%)")
    print(f"Peak 32-feature Test Accuracy (Yeo-Johnson):   {results[0]['yj_test_mean']*100:.2f}% (Omitted: {results[0]['omitted_feature']})")
    std_sorted = sorted(results, key=lambda r: r['std_test_mean'], reverse=True)
    print(f"Peak 32-feature Test Accuracy (StandardScaler): {std_sorted[0]['std_test_mean']*100:.2f}% (Omitted: {std_sorted[0]['omitted_feature']})")
    print(f"Execution finished in {time.time() - start_time:.1f} seconds.")


if __name__ == '__main__':
    main()
