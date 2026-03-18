"""
Linear Regression (Huber Loss, Outlier Robustness)

This task implements multivariate linear regression in PyTorch on a synthetic
regression dataset with injected target outliers. It compares a model trained
with Huber loss against an MSE baseline.

Model:
    y_hat = XW + b

Huber loss for residual r = y_hat - y:
    L_delta(r) = 0.5 * r^2                  if |r| <= delta
               = delta * (|r| - 0.5*delta) otherwise
"""

import json
import os
import sys
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


TASK_ID = "linreg_lvl5_huber_outliers"
SEED = 42
N_SAMPLES = 1200
N_FEATURES = 5
BATCH_SIZE = 64
EPOCHS = 180
LR = 0.03
DELTA = 1.0
OUTLIER_FRAC = 0.08
VAL_SIZE = 0.2


class LinearRegressor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


class StandardScalerTorch:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x: np.ndarray) -> None:
        self.mean = x.mean(axis=0, keepdims=True)
        self.std = x.std(axis=0, keepdims=True) + 1e-8

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.fit(x)
        return self.transform(x)


def get_task_metadata() -> Dict:
    return {
        "task_id": TASK_ID,
        "series": "Linear Regression",
        "algorithm": "Linear Regression (Huber Loss, Outlier Robustness)",
        "n_samples": N_SAMPLES,
        "n_features": N_FEATURES,
        "outlier_fraction": OUTLIER_FRAC,
    }


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=(N_SAMPLES, N_FEATURES)).astype(np.float32)
    true_w = np.array([2.5, -1.7, 0.8, 3.2, -2.1], dtype=np.float32)
    y = x @ true_w + 1.5 + rng.normal(0.0, 0.4, size=N_SAMPLES)

    n_outliers = int(N_SAMPLES * OUTLIER_FRAC)
    outlier_idx = rng.choice(N_SAMPLES, size=n_outliers, replace=False)
    y[outlier_idx] += rng.normal(0.0, 10.0, size=n_outliers)
    return x, y.astype(np.float32), true_w


def make_dataloaders(batch_size: int = BATCH_SIZE):
    x, y, true_w = _make_dataset()
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=VAL_SIZE, random_state=SEED
    )

    scaler = StandardScalerTorch()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)

    train_ds = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
    val_ds = TensorDataset(torch.tensor(x_val), torch.tensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    extras = {
        "x_train": x_train,
        "x_val": x_val,
        "y_train": y_train,
        "y_val": y_val,
        "true_w": true_w,
        "scaler_mean": scaler.mean.tolist(),
        "scaler_std": scaler.std.tolist(),
    }
    return train_loader, val_loader, extras


def build_model(input_dim: int = N_FEATURES, device: torch.device = None) -> nn.Module:
    device = device or get_device()
    model = LinearRegressor(input_dim).to(device)
    return model


def evaluate(model: nn.Module, data_loader: DataLoader, device: torch.device = None) -> Dict:
    device = device or get_device()
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in data_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            out = model(xb)
            preds.append(out.cpu().numpy())
            targets.append(yb.cpu().numpy())
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(targets)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        "mse": float(mse),
        "mae": float(mae),
        "r2": float(r2),
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
    }


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = EPOCHS,
    lr: float = LR,
    loss_name: str = "huber",
    device: torch.device = None,
):
    device = device or get_device()
    criterion = nn.HuberLoss(delta=DELTA) if loss_name == "huber" else nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_hist, val_hist = [], []
    for epoch in range(epochs):
        model.train()
        running = 0.0
        n = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
            n += xb.size(0)
        train_hist.append(running / max(n, 1))

        model.eval()
        val_running, val_n = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                val_loss = criterion(pred, yb)
                val_running += val_loss.item() * xb.size(0)
                val_n += xb.size(0)
        val_hist.append(val_running / max(val_n, 1))

        if (epoch + 1) % 40 == 0:
            print(f"[{loss_name}] epoch={epoch+1:03d} train_loss={train_hist[-1]:.4f} val_loss={val_hist[-1]:.4f}")

    return {
        "loss_history": train_hist,
        "val_loss_history": val_hist,
    }


def predict(model: nn.Module, x: np.ndarray, device: torch.device = None) -> np.ndarray:
    device = device or get_device()
    model.eval()
    with torch.no_grad():
        tensor_x = torch.tensor(x, dtype=torch.float32, device=device)
        return model(tensor_x).cpu().numpy()


def save_artifacts(output_dir: str, payload: Dict) -> None:
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> int:
    set_seed(SEED)
    device = get_device()
    print(f"Running {TASK_ID} on {device}")

    train_loader, val_loader, extras = make_dataloaders()

    huber_model = build_model(device=device)
    huber_train = train(huber_model, train_loader, val_loader, loss_name="huber", device=device)
    huber_train_metrics = evaluate(huber_model, train_loader, device=device)
    huber_val_metrics = evaluate(huber_model, val_loader, device=device)

    mse_model = build_model(device=device)
    mse_train = train(mse_model, train_loader, val_loader, loss_name="mse", device=device)
    mse_val_metrics = evaluate(mse_model, val_loader, device=device)

    print("\nHuber train metrics:", {k: round(v, 4) for k, v in huber_train_metrics.items() if isinstance(v, float)})
    print("Huber val metrics  :", {k: round(v, 4) for k, v in huber_val_metrics.items() if isinstance(v, float)})
    print("MSE baseline val   :", {k: round(v, 4) for k, v in mse_val_metrics.items() if isinstance(v, float)})

    val_r2_pass = huber_val_metrics["r2"] > 0.75
    mae_better_pass = huber_val_metrics["mae"] < mse_val_metrics["mae"]
    loss_decrease_pass = huber_train["loss_history"][-1] < huber_train["loss_history"][0]

    print(f"validation R2 > 0.75: {val_r2_pass} ({huber_val_metrics['r2']:.4f})")
    print(f"Huber MAE better than MSE baseline: {mae_better_pass} ({huber_val_metrics['mae']:.4f} vs {mse_val_metrics['mae']:.4f})")
    print(f"training loss decreased: {loss_decrease_pass} ({huber_train['loss_history'][0]:.4f} -> {huber_train['loss_history'][-1]:.4f})")

    payload = {
        "metadata": get_task_metadata(),
        "huber": {
            "loss_history": huber_train["loss_history"],
            "val_loss_history": huber_train["val_loss_history"],
            "train_metrics": {k: v for k, v in huber_train_metrics.items() if isinstance(v, float)},
            "val_metrics": {k: v for k, v in huber_val_metrics.items() if isinstance(v, float)},
        },
        "mse_baseline": {
            "loss_history": mse_train["loss_history"],
            "val_loss_history": mse_train["val_loss_history"],
            "val_metrics": {k: v for k, v in mse_val_metrics.items() if isinstance(v, float)},
        },
    }
    save_artifacts(os.path.join("artifacts", TASK_ID), payload)

    ok = val_r2_pass and mae_better_pass and loss_decrease_pass
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
