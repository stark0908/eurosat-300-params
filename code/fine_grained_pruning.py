"""Fine-grained 1-by-1 feature pruning from 33 down to 20 features.

Tests whether auxiliary reconstruction objectives (R33, R389, R389-Nonlinear)
can match or approach ~96% classification accuracy at fewer than 306 parameters.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from patch_features import EUROSAT_33

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / 'output' / 'feature-importance-cache-v3'
OUTPUT_DIR = ROOT / 'output' / 'reconstruction'


class Standardizer:
    def __init__(self, X_train: np.ndarray, eps: float = 1e-7):
        self.mean = X_train.mean(axis=0, keepdims=True)
        self.std = X_train.std(axis=0, keepdims=True) + eps

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std


class PrunedJointModel(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_target: int = 0,
        nonlinear_decoder: bool = False,
        hidden_dim: int = 32,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_target = d_target
        self.nonlinear_decoder = nonlinear_decoder

        # 9 relative logits
        self.encoder = nn.Linear(d_in, 9)

        if d_target > 0:
            if nonlinear_decoder:
                self.decoder = nn.Sequential(
                    nn.Linear(9, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, d_target),
                )
            else:
                self.decoder = nn.Linear(9, d_target)
        else:
            self.decoder = None

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        zeros = torch.zeros(z.shape[0], 1, device=z.device)
        logits = torch.cat([z, zeros], dim=1)

        recon = None
        if self.decoder is not None:
            recon = self.decoder(z)

        return logits, z, recon


def train_eval_run(
    X_tr: torch.Tensor,
    y_tr: torch.Tensor,
    X_va: torch.Tensor,
    y_va: torch.Tensor,
    X_te: torch.Tensor,
    y_te: torch.Tensor,
    Y_tr_tgt: torch.Tensor | None = None,
    Y_te_tgt: torch.Tensor | None = None,
    lam: float = 0.0,
    nonlinear: bool = False,
    seed: int = 0,
    max_iter: int = 150,
):
    torch.manual_seed(seed)
    device = X_tr.device
    d_in = X_tr.shape[1]
    d_tgt = Y_tr_tgt.shape[1] if Y_tr_tgt is not None else 0

    model = PrunedJointModel(d_in=d_in, d_target=d_tgt, nonlinear_decoder=nonlinear).to(device)
    nn.init.normal_(model.encoder.weight, 0.0, 0.01)
    nn.init.zeros_(model.encoder.bias)
    if model.decoder is not None:
        for p in model.decoder.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, 0.0, 0.01)
            else:
                nn.init.zeros_(p)

    weight_decay = 1.0 / (3.0 * len(X_tr))
    optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=max_iter, line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad()
        logits, _, recon = model(X_tr)
        loss = F.cross_entropy(logits, y_tr) + 0.5 * weight_decay * (model.encoder.weight ** 2).sum()
        if model.decoder is not None and Y_tr_tgt is not None and lam > 0:
            loss = loss + lam * F.mse_loss(recon, Y_tr_tgt)
        loss.backward()
        return loss

    optimizer.step(closure)

    model.eval()
    with torch.no_grad():
        val_logits, _, _ = model(X_va)
        val_acc = (val_logits.argmax(dim=1) == y_va).float().mean().item()

        te_logits, _, te_recon = model(X_te)
        te_acc = (te_logits.argmax(dim=1) == y_te).float().mean().item()

        mean_r2 = None
        if te_recon is not None and Y_te_tgt is not None:
            ss_res = ((Y_te_tgt - te_recon) ** 2).sum(dim=0)
            ss_tot = ((Y_te_tgt - Y_te_tgt.mean(dim=0)) ** 2).sum(dim=0)
            per_r2 = 1.0 - (ss_res / (ss_tot + 1e-7))
            mean_r2 = float(per_r2.mean().item())

    return val_acc, te_acc, mean_r2


def find_elimination_sequence(X_tr_33, y_tr, X_va_33, y_va, target_min=20):
    """Greedy backward elimination on validation accuracy, dropping 1 feature at a time."""
    device = X_tr_33.device
    active = list(range(33))
    sequence = {33: list(active)}

    print('\nStarting greedy backward elimination on 33 features (dropping 1 by 1)...')
    for current_count in range(33, target_min, -1):
        best_candidate = None
        best_val_score = -1.0

        for candidate in active:
            sub = [f for f in active if f != candidate]
            val_acc, _, _ = train_eval_run(
                X_tr=X_tr_33[:, sub],
                y_tr=y_tr,
                X_va=X_va_33[:, sub],
                y_va=y_va,
                X_te=X_va_33[:, sub],
                y_te=y_va,
                lam=0.0,
                seed=0,
                max_iter=80,
            )
            if val_acc > best_val_score:
                best_val_score = val_acc
                best_candidate = candidate

        active.remove(best_candidate)
        sequence[current_count - 1] = list(active)
        print(f"Dropped feature idx {best_candidate} -> {current_count - 1} features remain (Val Acc: {best_val_score*100:.2f}%)")

    return sequence


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Running fine-grained pruning on device: {device}')

    with open(CACHE_DIR / 'manifest.json') as f:
        manifest = json.load(f)
    names = manifest['schema']['names']
    idx_33 = np.array([names.index(name) for name in EUROSAT_33], dtype=np.int64)

    train_npz = np.load(CACHE_DIR / 'train.npz')
    val_npz = np.load(CACHE_DIR / 'val.npz')
    test_npz = np.load(CACHE_DIR / 'test.npz')

    X_tr_all = train_npz['features']
    X_va_all = val_npz['features']
    X_te_all = test_npz['features']
    y_tr = torch.tensor(train_npz['labels'], dtype=torch.long, device=device)
    y_va = torch.tensor(val_npz['labels'], dtype=torch.long, device=device)
    y_te = torch.tensor(test_npz['labels'], dtype=torch.long, device=device)

    # Standardize 33 features
    std_33 = Standardizer(X_tr_all[:, idx_33])
    X_tr_33 = torch.tensor(std_33.transform(X_tr_all[:, idx_33]), dtype=torch.float32, device=device)
    X_va_33 = torch.tensor(std_33.transform(X_va_all[:, idx_33]), dtype=torch.float32, device=device)
    X_te_33 = torch.tensor(std_33.transform(X_te_all[:, idx_33]), dtype=torch.float32, device=device)

    # Standardize all 389 features (for R389 targets)
    std_389 = Standardizer(X_tr_all)
    X_tr_389 = torch.tensor(std_389.transform(X_tr_all), dtype=torch.float32, device=device)
    X_te_389 = torch.tensor(std_389.transform(X_te_all), dtype=torch.float32, device=device)

    # Get greedy 1-by-1 elimination sequence from 33 down to 20
    elim_sequence = find_elimination_sequence(X_tr_33, y_tr, X_va_33, y_va, target_min=20)

    # Methods to evaluate at each step
    methods = [
        ('Baseline', None, False, 0.0),
        ('R33_Recon', X_tr_33, False, 0.01),
        ('R389_Recon', X_tr_389, False, 0.01),
        ('R389_Nonlinear', X_tr_389, True, 0.01),
    ]

    seeds = [0, 1, 2]
    all_results = []

    print('\n' + '=' * 65)
    print('EVALUATING METHODS 1-BY-1 ACROSS PARAMETER COUNTS')
    print('=' * 65)

    for count in range(33, 19, -1):
        active_sub = elim_sequence[count]
        params = 9 * (count + 1)
        sub_names = [EUROSAT_33[i] for i in active_sub]

        X_tr_sub = X_tr_33[:, active_sub]
        X_va_sub = X_va_33[:, active_sub]
        X_te_sub = X_te_33[:, active_sub]

        for method_name, tgt_tr, is_nonlinear, lam in methods:
            tgt_te = None
            if tgt_tr is not None:
                tgt_te = X_te_33 if tgt_tr is X_tr_33 else X_te_389

            test_accs = []
            val_accs = []
            r2s = []

            for seed in seeds:
                val_acc, test_acc, r2 = train_eval_run(
                    X_tr=X_tr_sub,
                    y_tr=y_tr,
                    X_va=X_va_sub,
                    y_va=y_va,
                    X_te=X_te_sub,
                    y_te=y_te,
                    Y_tr_tgt=tgt_tr,
                    Y_te_tgt=tgt_te,
                    lam=lam,
                    nonlinear=is_nonlinear,
                    seed=seed,
                )
                test_accs.append(test_acc)
                val_accs.append(val_acc)
                if r2 is not None:
                    r2s.append(r2)

            rec = {
                'features_count': count,
                'parameters': params,
                'method': method_name,
                'val_accuracy_mean': float(np.mean(val_accs)),
                'val_accuracy_std': float(np.std(val_accs)),
                'test_accuracy_mean': float(np.mean(test_accs)),
                'test_accuracy_std': float(np.std(test_accs)),
                'mean_r2': float(np.mean(r2s)) if r2s else None,
                'features_list': ';'.join(sub_names),
            }
            all_results.append(rec)
            print(
                f"F={count:<2} (Params={params:<3}) | {method_name:<15} | "
                f"Val: {rec['val_accuracy_mean']*100:.2f}% | Test: {rec['test_accuracy_mean']*100:.2f}% +/- {rec['test_accuracy_std']*100:.2f}%"
            )

    # Save to CSV
    csv_file = OUTPUT_DIR / 'fine_grained_pruning.csv'
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)
    print(f'\nSaved fine-grained results to: {csv_file}')

    # Plot Pareto curves
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {
        'Baseline': '#1f77b4',
        'R33_Recon': '#2ca02c',
        'R389_Recon': '#ff7f0e',
        'R389_Nonlinear': '#d62728',
    }
    markers = {
        'Baseline': 'o',
        'R33_Recon': 's',
        'R389_Recon': '^',
        'R389_Nonlinear': 'D',
    }

    import pandas as pd
    res_df = pd.DataFrame(all_results)

    for method_name, _, _, _ in methods:
        sub = res_df[res_df['method'] == method_name].sort_values('parameters')
        ax.plot(
            sub['parameters'],
            sub['test_accuracy_mean'] * 100,
            marker=markers[method_name],
            label=method_name,
            color=colors[method_name],
            linewidth=1.8,
            markersize=6,
        )

    ax.axhline(96.00, color='gray', linestyle='--', alpha=0.7, label='96.00% Benchmark')
    ax.axhline(96.04, color='black', linestyle=':', alpha=0.7, label='Original 306-param (96.04%)')

    ax.set_xlabel('Inference Parameters')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Fine-Grained Pruning Pareto Frontier: 306 to 189 Parameters (33 down to 20 Features)')
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()

    plot_file = OUTPUT_DIR / 'fine_grained_pareto.png'
    plt.savefig(plot_file, dpi=200)
    plt.close()
    print(f'Saved Pareto plot to: {plot_file}')


if __name__ == '__main__':
    main()
