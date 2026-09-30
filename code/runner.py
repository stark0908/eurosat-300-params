"""EuroSAT Representation Reconstruction & Feature-JEPA Research Suite.

Executes:
  Phase 1: Core Reconstruction Suite (B0, R33, R50, R100, R389, RD) x lambda grid x seeds
  Phase 2: Decoder & Representation Ablations (Nonlinear vs Linear, Logits vs Probs, Recon-only)
  Phase 3: Cross-Feature Prediction & Scientific Controls (FP1, FP2, Random->Random, Shuffled Target)
  Phase 4: Feature-JEPA Masked Feature Prediction
  Phase 5: Bottleneck Dimension Sweep & Feature Family Breakdown
  Phase 6: Result aggregation, statistical significance, and visualization
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

from patch_features import EUROSAT_33

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / 'output' / 'feature-importance-cache-v3'
IMPORTANCE_FILE = ROOT / 'output' / 'feature-importance' / 'feature_importances.csv'
OUTPUT_DIR = ROOT / 'output' / 'reconstruction'


def load_dataset():
    """Load cached features, labels, and feature metadata."""
    with open(CACHE_DIR / 'manifest.json') as f:
        manifest = json.load(f)
    names = manifest['schema']['names']
    families = manifest['schema']['families']

    train_npz = np.load(CACHE_DIR / 'train.npz')
    val_npz = np.load(CACHE_DIR / 'val.npz')
    test_npz = np.load(CACHE_DIR / 'test.npz')

    # Load ranking from feature_importances.csv
    with open(IMPORTANCE_FILE) as f:
        reader = csv.DictReader(f)
        ranked_pool_indices = [int(row['pool_index']) for row in reader]

    indices_33 = np.array([names.index(name) for name in EUROSAT_33], dtype=np.int64)
    discarded_indices = np.array([i for i in range(len(names)) if i not in set(indices_33)], dtype=np.int64)

    return {
        'names': names,
        'families': families,
        'train': {'X': train_npz['features'], 'y': train_npz['labels']},
        'val': {'X': val_npz['features'], 'y': val_npz['labels']},
        'test': {'X': test_npz['features'], 'y': test_npz['labels']},
        'indices_33': indices_33,
        'ranked_indices': np.array(ranked_pool_indices, dtype=np.int64),
        'discarded_indices': discarded_indices,
    }


class Standardizer:
    """Feature standardizer computed strictly on the training set."""
    def __init__(self, X_train: np.ndarray, eps: float = 1e-7):
        self.mean = X_train.mean(axis=0, keepdims=True)
        self.std = X_train.std(axis=0, keepdims=True) + eps

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std


class JointModel(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_bottleneck: int = 9,
        d_target: int = 0,
        nonlinear_decoder: bool = False,
        use_probabilities: bool = False,
        hidden_dim: int = 32,
        num_classes: int = 10,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_bottleneck = d_bottleneck
        self.d_target = d_target
        self.use_probabilities = use_probabilities
        self.nonlinear_decoder = nonlinear_decoder
        self.num_classes = num_classes

        # Encoder mapping input to bottleneck
        self.encoder = nn.Linear(d_in, d_bottleneck)

        # For bottleneck == 9, use the 9-row zero-reference classification head
        if d_bottleneck == num_classes - 1:
            self.classifier_head = None
        else:
            self.classifier_head = nn.Linear(d_bottleneck, num_classes)

        # Auxiliary decoder
        if d_target > 0:
            decoder_in = num_classes if use_probabilities else d_bottleneck
            if nonlinear_decoder:
                self.decoder = nn.Sequential(
                    nn.Linear(decoder_in, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, d_target),
                )
            else:
                self.decoder = nn.Linear(decoder_in, d_target)
        else:
            self.decoder = None

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        if self.classifier_head is None:
            zeros = torch.zeros(z.shape[0], 1, device=z.device)
            logits = torch.cat([z, zeros], dim=1)
        else:
            logits = self.classifier_head(z)

        recon = None
        if self.decoder is not None:
            dec_in = F.softmax(logits, dim=1) if self.use_probabilities else z
            recon = self.decoder(dec_in)

        return logits, z, recon


def compute_r2(y_true: torch.Tensor, y_pred: torch.Tensor):
    """Compute global and per-feature R^2 scores."""
    ss_res = ((y_true - y_pred) ** 2).sum(dim=0)
    y_mean = y_true.mean(dim=0, keepdim=True)
    ss_tot = ((y_true - y_mean) ** 2).sum(dim=0)
    per_feat_r2 = 1.0 - (ss_res / (ss_tot + 1e-7))
    global_r2 = 1.0 - (((y_true - y_pred) ** 2).sum() / (((y_true - y_mean) ** 2).sum() + 1e-7))
    return global_r2.item(), per_feat_r2.cpu().numpy()


def evaluate_model(model: nn.Module, X: torch.Tensor, y: torch.Tensor, Y_target: torch.Tensor | None = None):
    model.eval()
    with torch.no_grad():
        logits, z, recon = model(X)
        pred = logits.argmax(dim=1)
        acc = (pred == y).float().mean().item()
        macro_f1 = f1_score(y.cpu().numpy(), pred.cpu().numpy(), average='macro')

        per_class_acc = []
        for c in range(10):
            mask = y == c
            if mask.sum() > 0:
                per_class_acc.append((pred[mask] == c).float().mean().item())
            else:
                per_class_acc.append(0.0)

        metrics = {
            'accuracy': acc,
            'macro_f1': macro_f1,
            'per_class_accuracy': per_class_acc,
        }

        if recon is not None and Y_target is not None:
            mse = F.mse_loss(recon, Y_target).item()
            global_r2, per_feat_r2 = compute_r2(Y_target, recon)
            metrics.update({
                'mse': mse,
                'global_r2': global_r2,
                'mean_r2': float(np.mean(per_feat_r2)),
                'median_r2': float(np.median(per_feat_r2)),
                'per_feat_r2': per_feat_r2.tolist(),
            })

    return metrics


def train_lbfgs(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    Y_train_tgt: torch.Tensor | None,
    lam: float = 0.0,
    weight_decay: float = 2.0576e-5,
    max_iter: int = 150,
):
    model.train()
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=max_iter,
        line_search_fn='strong_wolfe',
    )

    def closure():
        optimizer.zero_grad()
        logits, _, recon = model(X_train)
        loss_cls = F.cross_entropy(logits, y_train)

        # L2 penalty on encoder weights
        l2_reg = 0.5 * weight_decay * (model.encoder.weight ** 2).sum()
        total_loss = loss_cls + l2_reg

        if model.decoder is not None and Y_train_tgt is not None and lam > 0:
            loss_rec = F.mse_loss(recon, Y_train_tgt)
            total_loss = total_loss + lam * loss_rec

        total_loss.backward()
        return total_loss

    optimizer.step(closure)
    return model


def count_parameters(model: nn.Module):
    """Return inference and total training parameter counts."""
    # Inference parameters: encoder weights + biases
    if model.classifier_head is None:
        inf_params = model.encoder.weight.numel() + model.encoder.bias.numel()
    else:
        inf_params = (
            model.encoder.weight.numel()
            + model.encoder.bias.numel()
            + model.classifier_head.weight.numel()
            + model.classifier_head.bias.numel()
        )

    # Total parameters includes decoder
    train_params = sum(p.numel() for p in model.parameters())
    return inf_params, train_params


def run_experiment_run(
    exp_id: str,
    X_tr_in: np.ndarray,
    X_va_in: np.ndarray,
    X_te_in: np.ndarray,
    y_tr: np.ndarray,
    y_va: np.ndarray,
    y_te: np.ndarray,
    Y_tr_tgt: np.ndarray | None,
    Y_va_tgt: np.ndarray | None,
    Y_te_tgt: np.ndarray | None,
    lam: float,
    seed: int,
    device: torch.device,
    d_bottleneck: int = 9,
    nonlinear_decoder: bool = False,
    use_probabilities: bool = False,
    hidden_dim: int = 32,
    c_reg: float = 3.0,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Standardize input features
    std_in = Standardizer(X_tr_in)
    X_tr_t = torch.tensor(std_in.transform(X_tr_in), dtype=torch.float32, device=device)
    X_va_t = torch.tensor(std_in.transform(X_va_in), dtype=torch.float32, device=device)
    X_te_t = torch.tensor(std_in.transform(X_te_in), dtype=torch.float32, device=device)

    y_tr_t = torch.tensor(y_tr, dtype=torch.long, device=device)
    y_va_t = torch.tensor(y_va, dtype=torch.long, device=device)
    y_te_t = torch.tensor(y_te, dtype=torch.long, device=device)

    # Standardize target features if present
    d_target = 0
    Y_tr_t, Y_va_t, Y_te_t = None, None, None
    if Y_tr_tgt is not None:
        d_target = Y_tr_tgt.shape[1]
        std_tgt = Standardizer(Y_tr_tgt)
        Y_tr_t = torch.tensor(std_tgt.transform(Y_tr_tgt), dtype=torch.float32, device=device)
        Y_va_t = torch.tensor(std_tgt.transform(Y_va_tgt), dtype=torch.float32, device=device)
        Y_te_t = torch.tensor(std_tgt.transform(Y_te_tgt), dtype=torch.float32, device=device)

    model = JointModel(
        d_in=X_tr_in.shape[1],
        d_bottleneck=d_bottleneck,
        d_target=d_target,
        nonlinear_decoder=nonlinear_decoder,
        use_probabilities=use_probabilities,
        hidden_dim=hidden_dim,
    ).to(device)

    # Small initial weights
    nn.init.normal_(model.encoder.weight, mean=0.0, std=0.01)
    nn.init.zeros_(model.encoder.bias)
    if model.decoder is not None:
        for p in model.decoder.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.01)
            else:
                nn.init.zeros_(p)

    weight_decay = 1.0 / (c_reg * len(X_tr_in))

    t0 = time.time()
    train_lbfgs(model, X_tr_t, y_tr_t, Y_tr_t, lam=lam, weight_decay=weight_decay)
    elapsed = time.time() - t0

    inf_p, train_p = count_parameters(model)

    val_res = evaluate_model(model, X_va_t, y_va_t, Y_va_t)
    test_res = evaluate_model(model, X_te_t, y_te_t, Y_te_t)

    record = {
        'exp_id': exp_id,
        'lambda': lam,
        'seed': seed,
        'd_bottleneck': d_bottleneck,
        'd_target': d_target,
        'inference_params': inf_p,
        'training_params': train_p,
        'val_accuracy': val_res['accuracy'],
        'val_macro_f1': val_res['macro_f1'],
        'test_accuracy': test_res['accuracy'],
        'test_macro_f1': test_res['macro_f1'],
        'val_mse': val_res.get('mse', None),
        'test_mse': test_res.get('mse', None),
        'val_mean_r2': val_res.get('mean_r2', None),
        'test_mean_r2': test_res.get('mean_r2', None),
        'test_median_r2': test_res.get('median_r2', None),
        'test_global_r2': test_res.get('global_r2', None),
        'elapsed_sec': round(elapsed, 2),
    }

    per_feat = test_res.get('per_feat_r2', None)
    return record, per_feat


# =========================================================================
# Phase 1: Core Reconstruction Experiments
# =========================================================================
def run_phase1_reconstruction(data: dict, device: torch.device):
    print('\n' + '=' * 60)
    print('PHASE 1: Representation Reconstruction (B0, R33, R50, R100, R389, RD)')
    print('=' * 60, flush=True)

    X_tr_all = data['train']['X']
    X_va_all = data['val']['X']
    X_te_all = data['test']['X']
    y_tr, y_va, y_te = data['train']['y'], data['val']['y'], data['test']['y']

    idx33 = data['indices_33']
    ranked = data['ranked_indices']
    discarded = data['discarded_indices']

    # Target subsets
    targets = {
        'B0': None,
        'R33': idx33,
        'R50': ranked[:50],
        'R100': ranked[:100],
        'R389': np.arange(len(data['names'])),
        'RD': discarded,
    }

    lambda_grid = [0.0, 0.001, 0.01, 0.1, 0.3, 1.0]
    seeds = [0, 1, 2]

    results = []
    per_feat_records = {}

    for exp_id, target_indices in targets.items():
        if exp_id == 'B0':
            cur_lambdas = [0.0]
        else:
            cur_lambdas = lambda_grid

        Y_tr_tgt = X_tr_all[:, target_indices] if target_indices is not None else None
        Y_va_tgt = X_va_all[:, target_indices] if target_indices is not None else None
        Y_te_tgt = X_te_all[:, target_indices] if target_indices is not None else None

        for lam in cur_lambdas:
            for seed in seeds:
                rec, per_feat = run_experiment_run(
                    exp_id=exp_id,
                    X_tr_in=X_tr_all[:, idx33],
                    X_va_in=X_va_all[:, idx33],
                    X_te_in=X_te_all[:, idx33],
                    y_tr=y_tr,
                    y_va=y_va,
                    y_te=y_te,
                    Y_tr_tgt=Y_tr_tgt,
                    Y_va_tgt=Y_va_tgt,
                    Y_te_tgt=Y_te_tgt,
                    lam=lam,
                    seed=seed,
                    device=device,
                )
                results.append(rec)
                print(
                    f"[{exp_id}] lam={lam:<5} s={seed} | Val: {rec['val_accuracy']*100:.2f}% | "
                    f"Test: {rec['test_accuracy']*100:.2f}% | Mean R2: {rec['test_mean_r2'] or 0.0:.3f} | "
                    f"Med R2: {rec['test_median_r2'] or 0.0:.3f} | {rec['elapsed_sec']}s",
                    flush=True,
                )

                if per_feat is not None and seed == 0 and lam == 0.1:
                    per_feat_records[exp_id] = per_feat

    return results, per_feat_records


# =========================================================================
# Phase 2: Decoder & Representation Ablations
# =========================================================================
def run_phase2_ablations(data: dict, device: torch.device):
    print('\n' + '=' * 60)
    print('PHASE 2: Decoder & Representation Ablations')
    print('=' * 60, flush=True)

    X_tr_all = data['train']['X']
    X_va_all = data['val']['X']
    X_te_all = data['test']['X']
    y_tr, y_va, y_te = data['train']['y'], data['val']['y'], data['test']['y']
    idx33 = data['indices_33']

    results = []

    # 1. Tiny non-linear decoder on R389 (9 -> 32 -> 389)
    print('\n--- Ablation: Nonlinear Decoder (9 -> 32 -> 389) ---', flush=True)
    for lam in [0.01, 0.1, 0.3]:
        for seed in [0, 1]:
            rec, _ = run_experiment_run(
                exp_id='R389-Nonlinear',
                X_tr_in=X_tr_all[:, idx33],
                X_va_in=X_va_all[:, idx33],
                X_te_in=X_te_all[:, idx33],
                y_tr=y_tr,
                y_va=y_va,
                y_te=y_te,
                Y_tr_tgt=X_tr_all,
                Y_va_tgt=X_va_all,
                Y_te_tgt=X_te_all,
                lam=lam,
                seed=seed,
                device=device,
                nonlinear_decoder=True,
                hidden_dim=32,
            )
            results.append(rec)
            print(
                f"[R389-Nonlinear] lam={lam:<5} s={seed} | Val: {rec['val_accuracy']*100:.2f}% | "
                f"Test: {rec['test_accuracy']*100:.2f}% | Mean R2: {rec['test_mean_r2']:.3f} | {rec['elapsed_sec']}s",
                flush=True,
            )

    # 2. Reconstruct from Softmax Probabilities instead of Logits
    print('\n--- Ablation: Probabilities vs Logits (10 probs -> 389) ---', flush=True)
    for lam in [0.01, 0.1, 0.3]:
        for seed in [0, 1]:
            rec, _ = run_experiment_run(
                exp_id='R389-Probs',
                X_tr_in=X_tr_all[:, idx33],
                X_va_in=X_va_all[:, idx33],
                X_te_in=X_te_all[:, idx33],
                y_tr=y_tr,
                y_va=y_va,
                y_te=y_te,
                Y_tr_tgt=X_tr_all,
                Y_va_tgt=X_va_all,
                Y_te_tgt=X_te_all,
                lam=lam,
                seed=seed,
                device=device,
                use_probabilities=True,
            )
            results.append(rec)
            print(
                f"[R389-Probs] lam={lam:<5} s={seed} | Val: {rec['val_accuracy']*100:.2f}% | "
                f"Test: {rec['test_accuracy']*100:.2f}% | Mean R2: {rec['test_mean_r2']:.3f} | {rec['elapsed_sec']}s",
                flush=True,
            )

    # 3. Reconstruction-only representation (No classification loss in pretraining)
    print('\n--- Diagnostic: Reconstruction-Only Pretraining (33 -> 9 -> 389) ---', flush=True)
    # Train 33 -> 9 -> 389 with MSE only
    std_in = Standardizer(X_tr_all[:, idx33])
    std_tgt = Standardizer(X_tr_all)
    X_tr_t = torch.tensor(std_in.transform(X_tr_all[:, idx33]), dtype=torch.float32, device=device)
    X_te_t = torch.tensor(std_in.transform(X_te_all[:, idx33]), dtype=torch.float32, device=device)
    Y_tr_t = torch.tensor(std_tgt.transform(X_tr_all), dtype=torch.float32, device=device)
    Y_te_t = torch.tensor(std_tgt.transform(X_te_all), dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long, device=device)
    y_te_t = torch.tensor(y_te, dtype=torch.long, device=device)

    W_e = nn.Parameter(torch.randn(9, 33, device=device) * 0.01)
    b_e = nn.Parameter(torch.zeros(9, device=device))
    W_d = nn.Parameter(torch.randn(389, 9, device=device) * 0.01)
    b_d = nn.Parameter(torch.zeros(389, device=device))

    opt = torch.optim.LBFGS([W_e, b_e, W_d, b_d], lr=1.0, max_iter=200, line_search_fn='strong_wolfe')
    def closure_rec():
        opt.zero_grad()
        z = F.linear(X_tr_t, W_e, b_e)
        rec = F.linear(z, W_d, b_d)
        loss = F.mse_loss(rec, Y_tr_t)
        loss.backward()
        return loss
    opt.step(closure_rec)

    with torch.no_grad():
        z_tr = F.linear(X_tr_t, W_e, b_e)
        z_te = F.linear(X_te_t, W_e, b_e)
        r2_glob, per_r2 = compute_r2(Y_te_t, F.linear(z_te, W_d, b_d))

    # Fit linear probe on frozen z
    probe = nn.Linear(9, 10).to(device)
    opt_p = torch.optim.LBFGS(probe.parameters(), lr=1.0, max_iter=150, line_search_fn='strong_wolfe')
    def closure_p():
        opt_p.zero_grad()
        loss = F.cross_entropy(probe(z_tr), y_tr_t)
        loss.backward()
        return loss
    opt_p.step(closure_p)

    with torch.no_grad():
        acc_probe = (probe(z_te).argmax(dim=1) == y_te_t).float().mean().item()

    rec_only_record = {
        'exp_id': 'Recon-Only-Probe',
        'lambda': float('inf'),
        'seed': 0,
        'd_bottleneck': 9,
        'd_target': 389,
        'inference_params': 306,
        'training_params': 306 + 3890 + 100,
        'val_accuracy': None,
        'val_macro_f1': None,
        'test_accuracy': acc_probe,
        'test_macro_f1': f1_score(y_te, probe(z_te).argmax(dim=1).cpu().numpy(), average='macro'),
        'val_mse': None,
        'test_mse': None,
        'val_mean_r2': None,
        'test_mean_r2': float(np.mean(per_r2)),
        'test_median_r2': float(np.median(per_r2)),
        'test_global_r2': r2_glob,
        'elapsed_sec': 0.0,
    }
    results.append(rec_only_record)
    print(
        f"[Recon-Only-Probe] Frozen 9D Linear Probe Test Acc: {acc_probe*100:.2f}% | "
        f"Recon Mean R2: {np.mean(per_r2):.3f}",
        flush=True,
    )

    return results


# =========================================================================
# Phase 3: Cross-Feature Prediction & Controls
# =========================================================================
def run_phase3_predictions(data: dict, device: torch.device):
    print('\n' + '=' * 60)
    print('PHASE 3: Cross-Feature Prediction & Controls')
    print('=' * 60, flush=True)

    X_tr_all = data['train']['X']
    X_va_all = data['val']['X']
    X_te_all = data['test']['X']
    y_tr, y_va, y_te = data['train']['y'], data['val']['y'], data['test']['y']

    idx33 = data['indices_33']
    ranked = data['ranked_indices']
    discarded = data['discarded_indices']

    # FP1: Selected (33) -> Next-ranked 33 (ranks 34-66)
    next33 = ranked[33:66]

    # FP2: Selected (33) -> Random 33 discarded
    np.random.seed(42)
    rand_discarded33 = np.random.choice(discarded, size=33, replace=False)

    # Control 1: Random 33 -> Random 33
    perm = np.random.permutation(len(data['names']))
    rand_in33 = perm[:33]
    rand_out33 = perm[33:66]

    # Control 2: Shuffled target values
    Y_tr_shuf = X_tr_all[:, idx33].copy()
    np.random.shuffle(Y_tr_shuf)

    fp_configs = {
        'FP1_Selected->Next33': (idx33, next33, False),
        'FP2_Selected->Discarded33': (idx33, rand_discarded33, False),
        'Control_Random->Random': (rand_in33, rand_out33, False),
        'Control_ShuffledTarget': (idx33, idx33, True),
    }

    results = []
    for exp_id, (in_idx, out_idx, shuffle) in fp_configs.items():
        X_tr_in = X_tr_all[:, in_idx]
        X_va_in = X_va_all[:, in_idx]
        X_te_in = X_te_all[:, in_idx]

        if shuffle:
            Y_tr_tgt = Y_tr_shuf
            Y_va_tgt = X_va_all[:, out_idx]
            Y_te_tgt = X_te_all[:, out_idx]
        else:
            Y_tr_tgt = X_tr_all[:, out_idx]
            Y_va_tgt = X_va_all[:, out_idx]
            Y_te_tgt = X_te_all[:, out_idx]

        for lam in [0.01, 0.1, 0.3]:
            for seed in [0, 1]:
                rec, _ = run_experiment_run(
                    exp_id=exp_id,
                    X_tr_in=X_tr_in,
                    X_va_in=X_va_in,
                    X_te_in=X_te_in,
                    y_tr=y_tr,
                    y_va=y_va,
                    y_te=y_te,
                    Y_tr_tgt=Y_tr_tgt,
                    Y_va_tgt=Y_va_tgt,
                    Y_te_tgt=Y_te_tgt,
                    lam=lam,
                    seed=seed,
                    device=device,
                )
                results.append(rec)
                print(
                    f"[{exp_id}] lam={lam:<5} s={seed} | Val: {rec['val_accuracy']*100:.2f}% | "
                    f"Test: {rec['test_accuracy']*100:.2f}% | Mean R2: {rec['test_mean_r2']:.3f} | {rec['elapsed_sec']}s",
                    flush=True,
                )

    return results


# =========================================================================
# Phase 4: Feature-JEPA Masked Prediction
# =========================================================================
def run_phase4_jepa(data: dict, device: torch.device):
    print('\n' + '=' * 60)
    print('PHASE 4: Feature-JEPA Random Masked Feature Prediction')
    print('=' * 60, flush=True)

    X_tr_all = data['train']['X']
    X_va_all = data['val']['X']
    X_te_all = data['test']['X']
    y_tr, y_va, y_te = data['train']['y'], data['val']['y'], data['test']['y']

    std_pool = Standardizer(X_tr_all)
    X_tr_s = torch.tensor(std_pool.transform(X_tr_all), dtype=torch.float32, device=device)
    X_va_s = torch.tensor(std_pool.transform(X_va_all), dtype=torch.float32, device=device)
    X_te_s = torch.tensor(std_pool.transform(X_te_all), dtype=torch.float32, device=device)

    y_tr_t = torch.tensor(y_tr, dtype=torch.long, device=device)
    y_va_t = torch.tensor(y_va, dtype=torch.long, device=device)
    y_te_t = torch.tensor(y_te, dtype=torch.long, device=device)

    results = []

    for mask_ratio in [0.25, 0.50, 0.75]:
        for lam in [0.01, 0.1]:
            torch.manual_seed(0)
            encoder = nn.Linear(389, 9).to(device)
            decoder = nn.Linear(9, 389).to(device)

            nn.init.normal_(encoder.weight, 0.0, 0.01)
            nn.init.zeros_(encoder.bias)
            nn.init.normal_(decoder.weight, 0.0, 0.01)
            nn.init.zeros_(decoder.bias)

            optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=0.005, weight_decay=1e-5)
            batch_size = 256
            n_samples = len(X_tr_s)
            epochs = 35

            t0 = time.time()
            for epoch in range(epochs):
                perm = torch.randperm(n_samples, device=device)
                for i in range(0, n_samples, batch_size):
                    idx = perm[i:i+batch_size]
                    batch_x = X_tr_s[idx]
                    batch_y = y_tr_t[idx]

                    # Masking: set random coordinates to 0
                    mask = (torch.rand_like(batch_x) < mask_ratio).float()
                    masked_x = batch_x * (1.0 - mask)

                    z = encoder(masked_x)
                    zeros = torch.zeros(z.shape[0], 1, device=device)
                    logits = torch.cat([z, zeros], dim=1)

                    loss_cls = F.cross_entropy(logits, batch_y)
                    pred_x = decoder(z)

                    # Loss only on masked elements
                    loss_rec = ((pred_x - batch_x) ** 2 * mask).sum() / (mask.sum() + 1e-7)

                    loss = loss_cls + lam * loss_rec
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            elapsed = time.time() - t0

            # Evaluate on unmasked test set
            encoder.eval()
            decoder.eval()
            with torch.no_grad():
                z_te = encoder(X_te_s)
                logits_te = torch.cat([z_te, torch.zeros(z_te.shape[0], 1, device=device)], dim=1)
                test_acc = (logits_te.argmax(dim=1) == y_te_t).float().mean().item()
                recon_te = decoder(z_te)
                glob_r2, per_r2 = compute_r2(X_te_s, recon_te)

            rec = {
                'exp_id': f'JEPA-Mask{int(mask_ratio*100)}',
                'lambda': lam,
                'seed': 0,
                'd_bottleneck': 9,
                'd_target': 389,
                'inference_params': 389 * 9 + 9,
                'training_params': (389 * 9 + 9) + (9 * 389 + 389),
                'val_accuracy': None,
                'val_macro_f1': None,
                'test_accuracy': test_acc,
                'test_macro_f1': f1_score(y_te, logits_te.argmax(dim=1).cpu().numpy(), average='macro'),
                'val_mse': None,
                'test_mse': F.mse_loss(recon_te, X_te_s).item(),
                'val_mean_r2': None,
                'test_mean_r2': float(np.mean(per_r2)),
                'test_median_r2': float(np.median(per_r2)),
                'test_global_r2': glob_r2,
                'elapsed_sec': round(elapsed, 2),
            }
            results.append(rec)
            print(
                f"[JEPA-Mask{int(mask_ratio*100)}] lam={lam:<5} | Test Acc: {test_acc*100:.2f}% | "
                f"Mean R2: {np.mean(per_r2):.3f} | Global R2: {glob_r2:.3f} | {rec['elapsed_sec']}s",
                flush=True,
            )

    return results


# =========================================================================
# Phase 5: Bottleneck Capacity Sweep & Feature Family Breakdown
# =========================================================================
def run_phase5_capacity_and_families(data: dict, device: torch.device):
    print('\n' + '=' * 60)
    print('PHASE 5: Bottleneck Dimension Sweep & Family Breakdown')
    print('=' * 60, flush=True)

    X_tr_all = data['train']['X']
    X_va_all = data['val']['X']
    X_te_all = data['test']['X']
    y_tr, y_va, y_te = data['train']['y'], data['val']['y'], data['test']['y']
    idx33 = data['indices_33']

    results = []

    # 1. Bottleneck Dimension Sweep: d in [4, 9, 16, 32, 64]
    for d in [4, 9, 16, 32, 64]:
        rec, per_feat = run_experiment_run(
            exp_id=f'Bottleneck-{d}D',
            X_tr_in=X_tr_all[:, idx33],
            X_va_in=X_va_all[:, idx33],
            X_te_in=X_te_all[:, idx33],
            y_tr=y_tr,
            y_va=y_va,
            y_te=y_te,
            Y_tr_tgt=X_tr_all,
            Y_va_tgt=X_va_all,
            Y_te_tgt=X_te_all,
            lam=0.1,
            seed=0,
            device=device,
            d_bottleneck=d,
        )
        results.append(rec)
        print(
            f"[Bottleneck-{d}D] Test Acc: {rec['test_accuracy']*100:.2f}% | "
            f"Mean R2: {rec['test_mean_r2']:.3f} | Med R2: {rec['test_median_r2']:.3f} | "
            f"Inf Params: {rec['inference_params']}",
            flush=True,
        )

    # 2. Family-wise breakdown on R389 with best lambda=0.1
    _, r389_per_feat = run_experiment_run(
        exp_id='R389-FamilyEval',
        X_tr_in=X_tr_all[:, idx33],
        X_va_in=X_va_all[:, idx33],
        X_te_in=X_te_all[:, idx33],
        y_tr=y_tr,
        y_va=y_va,
        y_te=y_te,
        Y_tr_tgt=X_tr_all,
        Y_va_tgt=X_va_all,
        Y_te_tgt=X_te_all,
        lam=0.1,
        seed=0,
        device=device,
    )

    family_r2 = defaultdict(list)
    for i, fam in enumerate(data['families']):
        family_r2[fam].append(r389_per_feat[i])

    family_summary = []
    print('\n--- Feature Family Reconstruction (R389, lambda=0.1) ---')
    for fam, scores in sorted(family_r2.items(), key=lambda x: -np.median(x[1])):
        fam_row = {
            'family': fam,
            'count': len(scores),
            'mean_r2': float(np.mean(scores)),
            'median_r2': float(np.median(scores)),
            'std_r2': float(np.std(scores)),
            'min_r2': float(np.min(scores)),
            'max_r2': float(np.max(scores)),
        }
        family_summary.append(fam_row)
        print(f"Family: {fam:<15} (n={len(scores):<3}) | Mean R2: {fam_row['mean_r2']:.3f} | Median R2: {fam_row['median_r2']:.3f}")

    return results, family_summary, r389_per_feat


# =========================================================================
# Phase 6: Multi-Seed Verification of Top Configurations
# =========================================================================
def run_phase6_multiseed(data: dict, device: torch.device):
    print('\n' + '=' * 60)
    print('PHASE 6: 5-Seed Statistical Verification')
    print('=' * 60, flush=True)

    X_tr_all = data['train']['X']
    X_va_all = data['val']['X']
    X_te_all = data['test']['X']
    y_tr, y_va, y_te = data['train']['y'], data['val']['y'], data['test']['y']

    idx33 = data['indices_33']
    discarded = data['discarded_indices']

    top_configs = [
        ('B0_Baseline', None, 0.0),
        ('R33_Selected', idx33, 0.01),
        ('RD_Discarded', discarded, 0.01),
        ('R389_FullPool', np.arange(len(data['names'])), 0.01),
    ]

    seeds = [0, 1, 2, 3, 4]
    multiseed_records = []

    for name, tgt_idx, lam in top_configs:
        Y_tr_tgt = X_tr_all[:, tgt_idx] if tgt_idx is not None else None
        Y_va_tgt = X_va_all[:, tgt_idx] if tgt_idx is not None else None
        Y_te_tgt = X_te_all[:, tgt_idx] if tgt_idx is not None else None

        seed_accs = []
        seed_r2s = []
        for s in seeds:
            rec, _ = run_experiment_run(
                exp_id=name,
                X_tr_in=X_tr_all[:, idx33],
                X_va_in=X_va_all[:, idx33],
                X_te_in=X_te_all[:, idx33],
                y_tr=y_tr,
                y_va=y_va,
                y_te=y_te,
                Y_tr_tgt=Y_tr_tgt,
                Y_va_tgt=Y_va_tgt,
                Y_te_tgt=Y_te_tgt,
                lam=lam,
                seed=s,
                device=device,
            )
            multiseed_records.append(rec)
            seed_accs.append(rec['test_accuracy'] * 100)
            if rec['test_mean_r2'] is not None:
                seed_r2s.append(rec['test_mean_r2'])

        mean_acc = np.mean(seed_accs)
        std_acc = np.std(seed_accs)
        mean_r2_val = np.mean(seed_r2s) if seed_r2s else 0.0
        print(f"[{name}] 5-Seed Test Acc: {mean_acc:.2f}% +/- {std_acc:.2f}% | Mean R2: {mean_r2_val:.3f}")

    return multiseed_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f'Starting research suite on device: {device}')
    data = load_dataset()
    print(f"Loaded: 389 features (33 selected, {len(data['discarded_indices'])} discarded).")

    all_records = []

    # Run all phases
    p1_records, p1_per_feat = run_phase1_reconstruction(data, device)
    all_records.extend(p1_records)

    p2_records = run_phase2_ablations(data, device)
    all_records.extend(p2_records)

    p3_records = run_phase3_predictions(data, device)
    all_records.extend(p3_records)

    p4_records = run_phase4_jepa(data, device)
    all_records.extend(p4_records)

    p5_records, family_summary, r389_per_feat = run_phase5_capacity_and_families(data, device)
    all_records.extend(p5_records)

    p6_records = run_phase6_multiseed(data, device)
    all_records.extend(p6_records)

    # Save CSV of all records
    csv_path = OUTPUT_DIR / 'all_experiments_results.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_records[0].keys()))
        writer.writeheader()
        writer.writerows(all_records)
    print(f'\nSaved all results to: {csv_path}')

    # Save family summary JSON & CSV
    fam_path = OUTPUT_DIR / 'feature_family_reconstruction.csv'
    with open(fam_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(family_summary[0].keys()))
        writer.writeheader()
        writer.writerows(family_summary)

    # Save per-feature R2 for R389
    per_feat_path = OUTPUT_DIR / 'r389_per_feature_r2.csv'
    with open(per_feat_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['feature_index', 'feature_name', 'family', 'r2'])
        writer.writeheader()
        for i, (name, fam) in enumerate(zip(data['names'], data['families'])):
            writer.writerow({
                'feature_index': i,
                'feature_name': name,
                'family': fam,
                'r2': float(r389_per_feat[i]),
            })

    print(f'Research run completely finished! Output directory: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
