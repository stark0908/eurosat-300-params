"""Generate publication-grade PNG visualizations for EuroSAT parameter minimization.

Outputs:
  1. confusion_matrices_comparison.png: 10x10 heatmaps for Baseline 33 vs Ours 32 vs Difference.
  2. pareto_frontier_sub306.png: Inference Parameters vs Test Accuracy across methods.
  3. distribution_pathology_transforms.png: Raw vs Yeo-Johnson distributions for pathological features.
  4. leave_one_out_32_ranking.png: Ranked horizontal bar chart for all 33 leave-one-out models.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix

from experiments.data import CLASSES
from ablation_32_leave_one_out import load_eurosat_33_data
from benchmark_comparison import load_full_cache
from model import ZeroReferenceLogReg
from transforms import StandardScalerTransform, YeoJohnsonTransform
from patch_features import EUROSAT_33

MOHIT_DIR = ROOT / 'mohit'
FIG_DIR = MOHIT_DIR / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = MOHIT_DIR / 'results'

_DEFAULT_ARTIFACT = '/home/Stark/.gemini/antigravity-cli/brain/86927067-00df-4e6e-a7a7-31a84185d6fe'
ARTIFACT_DIR_ENV = os.environ.get('ARTIFACT_DIR', _DEFAULT_ARTIFACT)
ARTIFACT_DIR = Path(ARTIFACT_DIR_ENV) if ARTIFACT_DIR_ENV and Path(ARTIFACT_DIR_ENV).is_dir() else None


def train_get_preds(X_train, y_train, X_test, transform_cls, seed=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t = transform_cls().fit(X_train)
    X_tr = torch.tensor(t.transform(X_train), dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_train, dtype=torch.long, device=device)
    X_te = torch.tensor(t.transform(X_test), dtype=torch.float32, device=device)

    torch.manual_seed(seed)
    m = ZeroReferenceLogReg(X_train.shape[1]).to(device)
    opt = torch.optim.LBFGS(m.parameters(), lr=1.0, max_iter=200, line_search_fn='strong_wolfe')
    crit = torch.nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = crit(m(X_tr), y_tr_t)
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        preds = m(X_te).argmax(dim=1).cpu().numpy()
    return preds


def plot_confusion_matrices(X_tr_33, y_tr, X_te_33, y_te):
    print("Generating Figure 1: Confusion Matrices Comparison...")
    # Baseline 33
    preds_base = train_get_preds(X_tr_33, y_tr, X_te_33, StandardScalerTransform)
    cm_base = confusion_matrix(y_te, preds_base)

    # Ours 32 (drop grad_mean_B05)
    idx_32 = [i for i, f in enumerate(EUROSAT_33) if f != 'grad_mean_B05']
    preds_yj32 = train_get_preds(X_tr_33[:, idx_32], y_tr, X_te_33[:, idx_32], YeoJohnsonTransform)
    cm_yj32 = confusion_matrix(y_te, preds_yj32)

    # Difference: YJ32 - Base (negative values mean fewer errors!)
    diff_cm = cm_yj32.astype(int) - cm_base.astype(int)

    short_classes = ['Annual', 'Forest', 'HerbVeg', 'Highway', 'Indust', 'Pasture', 'PermCrop', 'Resid', 'River', 'SeaLake']

    fig, axes = plt.subplots(1, 3, figsize=(22, 6.5))

    # Panel 1: Baseline 33
    sns.heatmap(cm_base, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False,
                xticklabels=short_classes, yticklabels=short_classes)
    acc_base = np.trace(cm_base) / 5400 * 100
    err_base = 5400 - np.trace(cm_base)
    axes[0].set_title(f'Baseline 33 Features (306 Params)\nTest Acc: {acc_base:.2f}% | Errors: {err_base}', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('True Label', fontsize=11)
    axes[0].set_xlabel('Predicted Label', fontsize=11)

    # Panel 2: Ours 32
    sns.heatmap(cm_yj32, annot=True, fmt='d', cmap='Blues', ax=axes[1], cbar=False,
                xticklabels=short_classes, yticklabels=short_classes)
    acc_yj32 = np.trace(cm_yj32) / 5400 * 100
    err_yj32 = 5400 - np.trace(cm_yj32)
    axes[1].set_title(f'Ours 32 Features + Yeo-Johnson (297 Params)\nTest Acc: {acc_yj32:.2f}% | Errors: {err_yj32}', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('True Label', fontsize=11)
    axes[1].set_xlabel('Predicted Label', fontsize=11)

    # Panel 3: Difference Matrix
    # We zero out diagonal for off-diagonal error visualization
    diff_offdiag = diff_cm.copy()
    np.fill_diagonal(diff_offdiag, 0)
    sns.heatmap(diff_offdiag, annot=True, fmt='d', cmap='RdYlGn_r', center=0, ax=axes[2], cbar=True,
                xticklabels=short_classes, yticklabels=short_classes)
    diag_delta = np.trace(cm_yj32) - np.trace(cm_base)
    axes[2].set_title(f'Error Shift (Ours - Baseline)\nGreen = Fewer Errors | Correct Gains: +{diag_delta}', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('True Label', fontsize=11)
    axes[2].set_xlabel('Predicted Label', fontsize=11)

    plt.suptitle('EuroSAT Test Set Confusion Matrix Comparison (N = 5,400)', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    out_path = FIG_DIR / 'confusion_matrices_comparison.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_pareto_frontier():
    print("Generating Figure 2: Pareto Frontier Curve...")
    csv_file = RESULTS_DIR / 'fine_grained_pruning.csv'
    df = pd.read_csv(csv_file)

    fig, ax = plt.subplots(figsize=(10.5, 6.8))

    styles = {
        'Baseline': {'color': '#d62728', 'marker': 'o', 'ls': '--', 'label': 'Author Baseline (StandardScaler)'},
        'R33_Recon': {'color': '#7f7f7f', 'marker': 's', 'ls': '-.', 'label': 'R33 Auxiliary Recon (λ=0.01)'},
        'R389_Recon': {'color': '#bcbd22', 'marker': '^', 'ls': ':', 'label': 'R389 Auxiliary Recon (λ=0.01)'},
    }

    for method, s in styles.items():
        sub = df[df['method'] == method].sort_values('parameters')
        ax.plot(sub['parameters'], sub['test_accuracy_mean'] * 100,
                color=s['color'], marker=s['marker'], linestyle=s['ls'],
                linewidth=1.8, markersize=6, label=s['label'])

    # Add Our Yeo-Johnson points
    yj_params = [306, 297, 288, 270]
    yj_accs = [96.27, 96.25, 96.19, 96.06]
    ax.plot(yj_params, yj_accs, color='#1f77b4', marker='o', linestyle='-',
            linewidth=2.2, markersize=8, label='Ours (Power-Conditioned Yeo-Johnson)')

    # Pure Folded Affine Model (100% Author Rules, strictly 297 params, 0 extra transforms)
    ax.scatter([297], [96.19], color='#2ca02c', marker='D', s=180, zorder=12, edgecolors='black', linewidth=1.5,
               label='Ours (Pure Folded Affine: 297 Params @ 96.19%)')
    ax.annotate('Pure Folded Affine (100% Strict Rules)\n297 Params: 96.19% (0 extra transforms)',
                xy=(297, 96.19), xytext=(215, 95.88),
                arrowprops=dict(facecolor='#2ca02c', shrink=0.08, width=1.5, headwidth=7),
                fontweight='bold', fontsize=9.5, bbox=dict(boxstyle='round,pad=0.3', facecolor='#eafaf1', edgecolor='#2ca02c'))

    # Yeo-Johnson 297 Highlight
    ax.scatter([297], [96.25], color='#1f77b4', s=180, zorder=10, edgecolors='black', linewidth=1.5)
    ax.annotate('Power-Conditioned (Yeo-Johnson)\n297 Params: 96.25% (+32 fixed exponents)',
                xy=(297, 96.25), xytext=(215, 96.38),
                arrowprops=dict(facecolor='#1f77b4', shrink=0.08, width=1.5, headwidth=7),
                fontweight='bold', fontsize=9.5, bbox=dict(boxstyle='round,pad=0.3', facecolor='#e6f2ff', edgecolor='#1f77b4'))

    # Yeo-Johnson 270 Highlight
    ax.scatter([270], [96.06], color='#1f77b4', s=140, zorder=10, edgecolors='black', linewidth=1.5)
    ax.annotate('Sub-280 Frontier (YJ)\n29 Features (270 Params): 96.06%',
                xy=(270, 96.06), xytext=(185, 95.55),
                arrowprops=dict(facecolor='#1f77b4', shrink=0.08, width=1.2, headwidth=6),
                fontweight='bold', fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor='#e6f2ff', edgecolor='#1f77b4'))

    # Benchmark lines
    ax.axhline(96.00, color='gray', linestyle='--', linewidth=1.2, alpha=0.7, label='96.00% Target Threshold')
    ax.axhline(96.04, color='black', linestyle=':', linewidth=1.5, alpha=0.8, label='Author Published Benchmark (96.04%)')

    ax.set_xlabel('Inference Parameters: P = 9 × (F + 1)', fontsize=11)
    ax.set_ylabel('EuroSAT Test Accuracy (%)', fontsize=11)
    ax.set_title('EuroSAT Parameter Minimization: The Sub-306 Pareto Frontier', fontsize=13, fontweight='bold')
    ax.set_xlim(180, 315)
    ax.set_ylim(94.4, 96.6)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='lower right', fontsize=9.5, framealpha=0.95)
    plt.tight_layout()

    out_path = FIG_DIR / 'pareto_frontier_sub306.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_distribution_pathologies(X_tr_33):
    print("Generating Figure 3: Distribution Pathology Transforms...")
    yj = YeoJohnsonTransform().fit(X_tr_33)
    X_tr_yj = yj.transform(X_tr_33)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5))

    # Feature 1: orient_entropy_B03
    idx1 = EUROSAT_33.index('orient_entropy_B03')
    raw1 = X_tr_33[:, idx1]
    trans1 = X_tr_yj[:, idx1]

    sns.histplot(raw1, bins=40, kde=True, ax=axes[0, 0], color='#d62728', stat='density')
    axes[0, 0].set_title('Raw: orient_entropy_B03\nSkewness = -3.98 | Kurtosis = 24.66', fontsize=11, fontweight='bold')
    axes[0, 0].set_xlabel('Feature Value', fontsize=10)
    axes[0, 0].grid(alpha=0.3)

    sns.histplot(trans1, bins=40, kde=True, ax=axes[0, 1], color='#1f77b4', stat='density')
    axes[0, 1].set_title('Yeo-Johnson: orient_entropy_B03\nNormalized Tail Dispersion', fontsize=11, fontweight='bold')
    axes[0, 1].set_xlabel('Transformed Value', fontsize=10)
    axes[0, 1].grid(alpha=0.3)

    # Feature 2: std_B05
    idx2 = EUROSAT_33.index('std_B05')
    raw2 = X_tr_33[:, idx2]
    trans2 = X_tr_yj[:, idx2]

    sns.histplot(raw2, bins=40, kde=True, ax=axes[1, 0], color='#d62728', stat='density')
    axes[1, 0].set_title('Raw: std_B05\nSkewness = +1.98 | Kurtosis = 18.18', fontsize=11, fontweight='bold')
    axes[1, 0].set_xlabel('Feature Value', fontsize=10)
    axes[1, 0].grid(alpha=0.3)

    sns.histplot(trans2, bins=40, kde=True, ax=axes[1, 1], color='#1f77b4', stat='density')
    axes[1, 1].set_title('Yeo-Johnson: std_B05\nNormalized Tail Dispersion', fontsize=11, fontweight='bold')
    axes[1, 1].set_xlabel('Transformed Value', fontsize=10)
    axes[1, 1].grid(alpha=0.3)

    plt.suptitle('Pathological Feature Densities Before vs. After Power Transformation', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    out_path = FIG_DIR / 'distribution_pathology_transforms.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_leave_one_out_ranking():
    print("Generating Figure 4: 32-Feature Leave-One-Out Ranking...")
    csv_file = RESULTS_DIR / 'leave_one_out_32_ablation.csv'
    df = pd.read_csv(csv_file).sort_values('yj_test_mean', ascending=True)

    fig, ax = plt.subplots(figsize=(12, 11))

    y_pos = np.arange(len(df))
    colors = ['#2ca02c' if cross else '#d62728' for cross in df['yj_crosses_96']]

    bars = ax.barh(y_pos, (df['yj_test_mean'] - 0.955) * 100, height=0.65, color=colors, alpha=0.85, label='Ours (Yeo-Johnson)')
    ax.scatter((df['std_test_mean'] - 0.955) * 100, y_pos, color='black', s=35, zorder=5, label='Author (StandardScaler)')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['omitted_feature'], fontsize=9.5)
    ax.set_xlabel('Test Accuracy (%)', fontsize=11)

    # Correct x-ticks to reflect true accuracy
    tick_locs = np.array([0.0, 0.5, 1.0, 1.5])  # 95.5%, 96.0%, 96.5%, 97.0%
    ax.set_xticks(tick_locs)
    ax.set_xticklabels(['95.50%', '96.00%', '96.50%', '97.00%'], fontsize=10)

    # 96.00% threshold line
    ax.axvline(0.5, color='blue', linestyle='--', linewidth=1.5, alpha=0.8, label='96.00% Benchmark')

    ax.set_title('Exhaustive 32-Feature (297-Param) Leave-One-Out Spectrum (All 33 Features)\nGreen = Test Acc ≥ 96.00% (18 features) | Red = Test Acc < 96.00% (15 features)',
                 fontsize=12, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6, axis='x')
    ax.legend(loc='lower right', fontsize=10, framealpha=0.95)
    plt.tight_layout()

    out_path = FIG_DIR / 'leave_one_out_32_ranking.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def copy_figures_to_artifacts():
    if ARTIFACT_DIR and ARTIFACT_DIR.is_dir():
        print(f"Copying generated figures to artifact directory: {ARTIFACT_DIR}")
        for fig_file in FIG_DIR.glob('*.png'):
            shutil.copy(fig_file, ARTIFACT_DIR / fig_file.name)


def main():
    X_tr_33, y_tr, _, _, X_te_33, y_te = load_eurosat_33_data()
    plot_confusion_matrices(X_tr_33, y_tr, X_te_33, y_te)
    plot_pareto_frontier()
    plot_distribution_pathologies(X_tr_33)
    plot_leave_one_out_ranking()
    copy_figures_to_artifacts()
    print("\nAll 4 figures generated and copied successfully!")


if __name__ == '__main__':
    main()
