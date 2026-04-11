'''
Created on Apr 10, 2026

@author: angelinaryabechenkova
'''
import json
import os
import random
import sys

import numpy as np # Needed for numerical operations
import torch # Needed for neuran networks
from sklearn.datasets import fetch_california_housing # The dataset to process
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Data about the data. This method checks if the model is good enough
def get_task_metadata(): 
    return {
        "task_id": "mlp_lvl5_california_regression",
        "task_type": "regression",
        "dataset": "california_housing",
        "metric_thresholds": {
            "val_r2_min": 0.60
        }
    }


def set_seed(seed=42):
    # Makes results repeatable. Otherwise, every run gives different results
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Run on GPU if available, otherwise CPU (slower)
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loads imported data
def make_dataloaders(batch_size=64):
    data = fetch_california_housing()
    X = data.data.astype(np.float32) # features input
    y = data.target.astype(np.float32).reshape(-1, 1) # house price

    X_train, X_val, y_train, y_val = train_test_split( # splits 80% training and 20% val
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler() # Normalizes data
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)

    # Batches data (instead of all at once)
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, scaler

# Define the neural network
class MLPRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), # 64 neurons
            nn.ReLU(), # Activation function that adds non-linearity
            nn.Linear(64, 32), # Second layer, 32 neurons
            nn.ReLU(), # Activation function that adds non-linearity
            nn.Linear(32, 1), # Prediction - 1
        )

    def forward(self, x):
        return self.net(x)


def build_model(input_dim):
    return MLPRegressor(input_dim) # Creates the model


def train(model, train_loader, val_loader, device, epochs=50, lr=1e-3):
    criterion = nn.MSELoss() # Measures error between prediction and true value
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) # Updates model weights

    loss_history = []
    val_loss_history = []

    for epoch in range(epochs): # Go through all training data
        model.train()
        running_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            preds = model(xb) # Model makes predictions
            loss = criterion(preds, yb) # Calculate error
            loss.backward() # Calculate gradients
            optimizer.step() # Update weights

            running_loss += loss.item() * xb.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        loss_history.append(epoch_loss)

        model.eval() # Turn off training mode
        val_running_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_running_loss += loss.item() * xb.size(0)

        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_loss_history.append(val_epoch_loss)

    return {
        "loss_history": loss_history,
        "val_loss_history": val_loss_history,
    }


def evaluate(model, loader, device):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            preds = model(xb).cpu().numpy()
            y_pred.append(preds)
            y_true.append(yb.numpy())

    y_true = np.vstack(y_true).ravel()
    y_pred = np.vstack(y_pred).ravel()

    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        "mse": float(mse),
        "r2": float(r2),
    }


def predict(model, x_tensor, device):
    model.eval()
    with torch.no_grad():
        x_tensor = x_tensor.to(device)
        preds = model(x_tensor)
    return preds.cpu()


def save_artifacts(output_dir, model, history, metrics):
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "model.pt")
    torch.save(model.state_dict(), model_path) # Saves trained model

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump( # Saves metrics
            {
                "history": history,
                "metrics": metrics
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    try:
        set_seed(42)
        metadata = get_task_metadata()
        device = get_device()

        train_loader, val_loader, _ = make_dataloaders(batch_size=64)
        input_dim = train_loader.dataset.tensors[0].shape[1]

        model = build_model(input_dim).to(device)

        history = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=50,
            lr=1e-3,
        )

        train_metrics = evaluate(model, train_loader, device)
        val_metrics = evaluate(model, val_loader, device)

        print("=== TRAIN METRICS ===")
        print(f"MSE: {train_metrics['mse']:.4f}")
        print(f"R2 : {train_metrics['r2']:.4f}")

        print("=== VALIDATION METRICS ===")
        print(f"MSE: {val_metrics['mse']:.4f}")
        print(f"R2 : {val_metrics['r2']:.4f}")

        save_artifacts(
            output_dir="tasks/mlp_lvl5_california_regression/artifacts",
            model=model,
            history=history,
            metrics={
                "train": train_metrics,
                "val": val_metrics,
            },
        )

        assert val_metrics["r2"] > metadata["metric_thresholds"]["val_r2_min"], \
            f"Validation R2 too low: {val_metrics['r2']:.4f}"

        sys.exit(0)

    except Exception as e:
        print(f"Task failed: {e}")
        sys.exit(1)