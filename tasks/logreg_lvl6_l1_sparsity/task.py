"""
Logistic Regression (L1 Regularization for Sparse Features)

This task trains binary logistic regression with an explicit L1 penalty.

Sigmoid:
    sigma(z) = 1 / (1 + e^{-z})

Objective:
    BCEWithLogitsLoss + lambda * ||w||_1
"""

import json
import os
import sys
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, precision_score, r2_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


TASK_ID = "logreg_lvl6_l1_sparsity"
SEED = 42
N_SAMPLES = 1400
N_FEATURES = 20
N_INFORMATIVE = 6
VAL_SIZE = 0.2
BATCH_SIZE = 64
EPOCHS = 80
LR = 0.02
L1_LAMBDA = 0.02
NEAR_ZERO_THRESHOLD = 0.10


class StandardScalerTorch:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x):
        self.mean = x.mean(axis=0, keepdims=True)
        self.std = x.std(axis=0, keepdims=True) + 1e-8

    def transform(self, x):
        return (x - self.mean) / self.std


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


def get_task_metadata() -> Dict:
    return {
        "task_id": TASK_ID,
        "series": "Logistic Regression",
        "algorithm": "Logistic Regression (L1 Regularization for Sparse Features)",
        "n_samples": N_SAMPLES,
        "n_features": N_FEATURES,
        "n_informative": N_INFORMATIVE,
    }


def set_seed(seed: int = SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_dataloaders(batch_size: int = BATCH_SIZE):
    x, y = make_classification(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_informative=N_INFORMATIVE,
        n_redundant=4,
        n_repeated=0,
        n_classes=2,
        class_sep=1.8,
        flip_y=0.02,
        random_state=SEED,
    )
    x = x.astype(np.float32)
    y = y.astype(np.float32)

    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=VAL_SIZE, random_state=SEED, stratify=y
    )
    scaler = StandardScalerTorch()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_val = scaler.transform(x_val)

    train_ds = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
    val_ds = TensorDataset(torch.tensor(x_val), torch.tensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, {"x_train": x_train, "x_val": x_val}


def build_model(input_dim: int = N_FEATURES, device=None):
    device = device or get_device()
    return LogisticRegressionModel(input_dim).to(device)


def train(model, train_loader, val_loader, epochs: int = EPOCHS, lr: float = LR, device=None):
    device = device or get_device()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    loss_history, val_loss_history = [], []

    for epoch in range(epochs):
        model.train()
        total, n = 0.0, 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            bce = criterion(logits, yb)
            l1_penalty = model.linear.weight.abs().sum()
            loss = bce + L1_LAMBDA * l1_penalty
            loss.backward()
            optimizer.step()
            total += loss.item() * xb.size(0)
            n += xb.size(0)
        loss_history.append(total / max(n, 1))

        model.eval()
        val_total, val_n = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                bce = criterion(logits, yb)
                l1_penalty = model.linear.weight.abs().sum()
                loss = bce + L1_LAMBDA * l1_penalty
                val_total += loss.item() * xb.size(0)
                val_n += xb.size(0)
        val_loss_history.append(val_total / max(val_n, 1))
        if (epoch + 1) % 20 == 0:
            print(f"epoch={epoch+1:03d} train_loss={loss_history[-1]:.4f} val_loss={val_loss_history[-1]:.4f}")

    return {"loss_history": loss_history, "val_loss_history": val_loss_history}


def evaluate(model, data_loader, device=None):
    device = device or get_device()
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for xb, yb in data_loader:
            xb = xb.to(device)
            logits = model(xb)
            p = torch.sigmoid(logits)
            probs.append(p.cpu().numpy())
            labels.append(yb.numpy())
    y_prob = np.concatenate(probs)
    y_true = np.concatenate(labels)
    y_pred = (y_prob >= 0.5).astype(np.float32)
    weights = model.linear.weight.detach().cpu().numpy().reshape(-1)
    sparsity_ratio = float(np.mean(np.abs(weights) < NEAR_ZERO_THRESHOLD))
    return {
        "mse": float(mean_squared_error(y_true, y_prob)),
        "r2": float(r2_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "sparsity_ratio": sparsity_ratio,
        "weights": weights.tolist(),
    }


def predict(model, x: np.ndarray, device=None):
    device = device or get_device()
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(x, dtype=torch.float32, device=device)
        return torch.sigmoid(model(xb)).cpu().numpy()


def save_artifacts(output_dir: str, payload: Dict):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> int:
    set_seed(SEED)
    device = get_device()
    print(f"Running {TASK_ID} on {device}")

    train_loader, val_loader, extras = make_dataloaders()
    train_eval_loader = DataLoader(train_loader.dataset, batch_size=128, shuffle=False)

    model = build_model(device=device)
    history = train(model, train_loader, val_loader, device=device)
    train_metrics = evaluate(model, train_eval_loader, device=device)
    val_metrics = evaluate(model, val_loader, device=device)

    summary_train = {k: round(v, 4) for k, v in train_metrics.items() if isinstance(v, float)}
    summary_val = {k: round(v, 4) for k, v in val_metrics.items() if isinstance(v, float)}
    print("Train metrics:", summary_train)
    print("Val metrics  :", summary_val)

    acc_pass = val_metrics["accuracy"] > 0.88
    sparse_pass = val_metrics["sparsity_ratio"] >= 0.30
    loss_down = history["loss_history"][-1] < history["loss_history"][0]

    payload = {
        "metadata": get_task_metadata(),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "loss_history": history["loss_history"],
        "val_loss_history": history["val_loss_history"],
    }
    save_artifacts(os.path.join("artifacts", TASK_ID), payload)

    ok = acc_pass and sparse_pass and loss_down
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
