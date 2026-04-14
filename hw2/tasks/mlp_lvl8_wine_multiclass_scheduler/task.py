import json
import os
import random
import sys

import numpy as np
import torch
from sklearn.datasets import load_wine
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def get_task_metadata():
    return {
        "task_id": "mlp_lvl8_wine_multiclass_scheduler",
        "task_type": "classification",
        "metric_thresholds": {
            "val_acc_min": 0.85
        }
    }


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_dataloaders(batch_size=64):
    data = load_wine()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.int64)  # important for CrossEntropyLoss

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


class WineMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # 3 output classes
        )

    def forward(self, x):
        return self.net(x)


def build_model(input_dim):
    return WineMLP(input_dim)


def train(model, train_loader, val_loader, device, epochs=40, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    for epoch in range(epochs):
        model.train()

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        scheduler.step()

    return {}


def evaluate(model, loader, device):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)

            preds = torch.argmax(logits, dim=1).cpu().numpy()

            y_pred.append(preds)
            y_true.append(yb.numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)

    return {
        "mse": float(mse),
        "r2": float(r2),
        "accuracy": float(acc),
    }


def predict(model, x_tensor, device):
    model.eval()
    with torch.no_grad():
        logits = model(x_tensor.to(device))
        preds = torch.argmax(logits, dim=1)
    return preds.cpu()


def save_artifacts(output_dir, model, metrics):
    os.makedirs(output_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    try:
        set_seed(42)
        device = get_device()

        train_loader, val_loader = make_dataloaders()
        input_dim = train_loader.dataset.tensors[0].shape[1]

        model = build_model(input_dim).to(device)

        print("Training...")
        train(model, train_loader, val_loader, device)

        print("Evaluating...")
        train_metrics = evaluate(model, train_loader, device)
        val_metrics = evaluate(model, val_loader, device)

        print("\n=== TRAIN METRICS ===")
        print(train_metrics)

        print("\n=== VALIDATION METRICS ===")
        print(val_metrics)

        save_artifacts(
            "artifacts",
            model,
            {"train": train_metrics, "val": val_metrics},
        )

        assert val_metrics["accuracy"] > 0.85, "Accuracy too low"

        sys.exit(0)

    except Exception as e:
        print("Error:", e)
        sys.exit(1)