"""
Linear Regression (Batch vs Mini-batch Gradient Descent)

This task compares full-batch and mini-batch optimization for multivariate
linear regression using PyTorch.

Prediction function:
    y_hat = XW + b

MSE objective:
    J(theta) = (1 / n) * sum_i (y_hat_i - y_i)^2

Gradient descent update:
    theta <- theta - lr * grad J(theta)
"""

import json
import os
import sys
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


TASK_ID = "linreg_lvl6_batch_vs_minibatch"
SEED = 42
N_SAMPLES = 1000
N_FEATURES = 8
VAL_SIZE = 0.2
FULL_BATCH_EPOCHS = 220
MINI_BATCH_EPOCHS = 220
LR = 0.02
MINI_BATCH_SIZE = 32


class StandardScalerTorch:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x: np.ndarray) -> None:
        self.mean = x.mean(axis=0, keepdims=True)
        self.std = x.std(axis=0, keepdims=True) + 1e-8

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std


class LinearRegressor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


def get_task_metadata() -> Dict:
    return {
        "task_id": TASK_ID,
        "series": "Linear Regression",
        "algorithm": "Linear Regression (Batch vs Mini-batch Gradient Descent)",
        "n_samples": N_SAMPLES,
        "n_features": N_FEATURES,
    }


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_dataloaders(mini_batch_size: int = MINI_BATCH_SIZE):
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=(N_SAMPLES, N_FEATURES)).astype(np.float32)
    true_w = np.array([1.5, -2.0, 0.7, 3.3, -1.2, 0.5, 2.4, -0.9], dtype=np.float32)
    y = (x @ true_w + 0.75 + rng.normal(0.0, 0.45, size=N_SAMPLES)).astype(np.float32)

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=VAL_SIZE, random_state=SEED)
    scaler = StandardScalerTorch()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_val = scaler.transform(x_val)

    train_ds = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
    val_ds = TensorDataset(torch.tensor(x_val), torch.tensor(y_val))

    full_batch_loader = DataLoader(train_ds, batch_size=len(train_ds), shuffle=True)
    mini_batch_loader = DataLoader(train_ds, batch_size=mini_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    extras = {
        "x_train": x_train,
        "x_val": x_val,
        "y_train": y_train,
        "y_val": y_val,
        "true_w": true_w.tolist(),
    }
    return full_batch_loader, mini_batch_loader, val_loader, extras


def build_model(input_dim: int = N_FEATURES, device: torch.device = None) -> nn.Module:
    device = device or get_device()
    return LinearRegressor(input_dim).to(device)


def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, epochs: int, lr: float, device=None):
    device = device or get_device()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    loss_history, val_loss_history = [], []

    for epoch in range(epochs):
        model.train()
        total_loss, n = 0.0, 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            n += xb.size(0)
        loss_history.append(total_loss / max(n, 1))

        model.eval()
        val_total, val_n = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_total += loss.item() * xb.size(0)
                val_n += xb.size(0)
        val_loss_history.append(val_total / max(val_n, 1))

    return {"loss_history": loss_history, "val_loss_history": val_loss_history}


def evaluate(model: nn.Module, data_loader: DataLoader, device=None) -> Dict:
    device = device or get_device()
    model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for xb, yb in data_loader:
            xb = xb.to(device)
            pred = model(xb)
            ys.append(yb.numpy())
            preds.append(pred.cpu().numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(preds)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"mse": float(mse), "r2": float(r2)}


def predict(model: nn.Module, x: np.ndarray, device=None) -> np.ndarray:
    device = device or get_device()
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(x, dtype=torch.float32, device=device)
        return model(xb).cpu().numpy()


def save_artifacts(output_dir: str, payload: Dict) -> None:
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> int:
    set_seed(SEED)
    device = get_device()
    print(f"Running {TASK_ID} on {device}")

    full_batch_loader, mini_batch_loader, val_loader, extras = make_dataloaders()
    train_full_eval = DataLoader(full_batch_loader.dataset, batch_size=256, shuffle=False)
    train_mini_eval = DataLoader(mini_batch_loader.dataset, batch_size=256, shuffle=False)

    full_batch_model = build_model(device=device)
    full_batch_hist = train(full_batch_model, full_batch_loader, val_loader, FULL_BATCH_EPOCHS, LR, device=device)
    full_train_metrics = evaluate(full_batch_model, train_full_eval, device=device)
    full_val_metrics = evaluate(full_batch_model, val_loader, device=device)

    mini_batch_model = build_model(device=device)
    mini_batch_hist = train(mini_batch_model, mini_batch_loader, val_loader, MINI_BATCH_EPOCHS, LR, device=device)
    mini_train_metrics = evaluate(mini_batch_model, train_mini_eval, device=device)
    mini_val_metrics = evaluate(mini_batch_model, val_loader, device=device)

    print("Full-batch train:", full_train_metrics)
    print("Full-batch val  :", full_val_metrics)
    print("Mini-batch train:", mini_train_metrics)
    print("Mini-batch val  :", mini_val_metrics)

    full_pass = full_val_metrics["r2"] > 0.85
    mini_pass = mini_val_metrics["r2"] > 0.85
    full_loss_down = full_batch_hist["loss_history"][-1] < full_batch_hist["loss_history"][0]
    mini_loss_down = mini_batch_hist["loss_history"][-1] < mini_batch_hist["loss_history"][0]

    payload = {
        "metadata": get_task_metadata(),
        "full_batch": {
            "train_metrics": full_train_metrics,
            "val_metrics": full_val_metrics,
            "loss_history": full_batch_hist["loss_history"],
            "val_loss_history": full_batch_hist["val_loss_history"],
        },
        "mini_batch": {
            "train_metrics": mini_train_metrics,
            "val_metrics": mini_val_metrics,
            "loss_history": mini_batch_hist["loss_history"],
            "val_loss_history": mini_batch_hist["val_loss_history"],
        },
    }
    save_artifacts(os.path.join("artifacts", TASK_ID), payload)

    ok = full_pass and mini_pass and full_loss_down and mini_loss_down
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
