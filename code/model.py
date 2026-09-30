"""Zero-reference Multinomial Logistic Regression model for EuroSAT.

Architecture & Parameter Rules:
- 10 classes with class 0 chosen as the zero reference.
- Weights matrix W: (9, d_in)
- Bias vector b: (9,)
- Total learned parameters: 9 * (d_in + 1)
  * d_in = 33 -> 9 * 34 = 306 parameters
  * d_in = 32 -> 9 * 33 = 297 parameters
  * d_in = 31 -> 9 * 32 = 288 parameters
  * d_in = 29 -> 9 * 30 = 270 parameters
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ZeroReferenceLogReg(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.d_in = d_in
        self.linear = nn.Linear(d_in, 9)
        nn.init.normal_(self.linear.weight, 0.0, 0.01)
        nn.init.zeros_(self.linear.bias)

    @property
    def num_parameters(self) -> int:
        return 9 * (self.d_in + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute 9 relative logits
        z = self.linear(x)
        # Append 0 for class 0 (reference class)
        zeros = torch.zeros(z.shape[0], 1, device=z.device, dtype=z.dtype)
        logits = torch.cat([z, zeros], dim=1)
        return logits


def train_eval_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int = 0,
    max_iter: int = 200,
    device: str | torch.device | None = None,
) -> tuple[float, float]:
    """Train zero-reference logistic regression using L-BFGS and evaluate accuracy."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    torch.manual_seed(seed)
    np.random.seed(seed)

    d_in = X_train.shape[1]
    model = ZeroReferenceLogReg(d_in=d_in).to(device)

    X_tr = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_tr = torch.tensor(y_train, dtype=torch.long, device=device)
    X_va = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_va = torch.tensor(y_val, dtype=torch.long, device=device)
    X_te = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_te = torch.tensor(y_test, dtype=torch.long, device=device)

    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=max_iter,
        line_search_fn='strong_wolfe',
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
    )

    criterion = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        logits = model(X_tr)
        loss = criterion(logits, y_tr)
        loss.backward()
        return loss

    optimizer.step(closure)

    model.eval()
    with torch.no_grad():
        val_logits = model(X_va)
        val_acc = float((val_logits.argmax(dim=1) == y_va).float().mean().item())

        test_logits = model(X_te)
        test_acc = float((test_logits.argmax(dim=1) == y_te).float().mean().item())

    return val_acc, test_acc
