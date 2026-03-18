"""
Logistic Regression (Binary Classification on Breast Cancer Dataset)

This task trains a binary logistic regression model in PyTorch on the sklearn
breast cancer dataset.

Sigmoid:
    sigma(z) = 1 / (1 + e^{-z})

Binary cross-entropy:
    BCE(y, p) = - [y log(p) + (1-y) log(1-p)]
"""

import json
import os
import sys
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


TASK_ID = "logreg_lvl5_breast_cancer_realdata"
SEED = 42
BATCH_SIZE = 32
EPOCHS = 60
LR = 0.01
VAL_SIZE = 0.2


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


def get_task_metadata() -> Dict:
    return {
        "task_id": TASK_ID,
        "series": "Logistic Regression",
        "algorithm": "Logistic Regression (Binary Classification on Breast Cancer Dataset)",
        "dataset": "sklearn breast cancer",
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
    data = load_breast_cancer()
    x = data.data.astype(np.float32)
    y = data.target.astype(np.float32)

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
    extras = {"feature_names": data.feature_names.tolist(), "target_names": data.target_names.tolist()}
    return train_loader, val_loader, extras


def build_model(input_dim: int = 30, device=None):
    device = device or get_device()
    return LogisticRegressionModel(input_dim).to(device)


def train(model, train_loader, val_loader, epochs: int = EPOCHS, lr: float = LR, device=None):
    device = device or get_device()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_history, val_loss_history = [], []

    for epoch in range(epochs):
        model.train()
        total_loss, n = 0.0, 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
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
                logits = model(xb)
                loss = criterion(logits, yb)
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
    return {
        "mse": float(mean_squared_error(y_true, y_prob)),
        "r2": float(r2_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
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

    print("Train metrics:", train_metrics)
    print("Val metrics  :", val_metrics)

    acc_pass = val_metrics["accuracy"] > 0.93
    f1_pass = val_metrics["f1"] > 0.93
    loss_down = history["loss_history"][-1] < history["loss_history"][0]

    payload = {
        "metadata": get_task_metadata(),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "loss_history": history["loss_history"],
        "val_loss_history": history["val_loss_history"],
        "extras": extras,
    }
    save_artifacts(os.path.join("artifacts", TASK_ID), payload)

    ok = acc_pass and f1_pass and loss_down
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
