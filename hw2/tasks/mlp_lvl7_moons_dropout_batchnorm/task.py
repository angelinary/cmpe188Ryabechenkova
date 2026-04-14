'''
Created on Apr 12, 2026

@author: angelinaryabechenkova
'''
import json # lets save data into a json file
import os # for folders and paths
import random # for random seed
import sys # for sys.exit(0)

import numpy as np # for arrays
import torch # for PyTorch ML
from sklearn.datasets import make_moons # creates a synthetic two-class dataset shaped like two interlocking moons
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score # computes
from sklearn.model_selection import train_test_split # splits data into training and validation sets
from sklearn.preprocessing import StandardScaler # scales to ease compute load
from torch import nn # neural network module
from torch.utils.data import DataLoader, TensorDataset

# Get data about data
def get_task_metadata():
    return {
        "task_id": "mlp_lvl7_moons_dropout_batchnorm",
        "task_type": "classification",
        "metric_thresholds": {
            "val_acc_min": 0.90
        }
    }

# seeding to makes sure all runs are more or less same
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# optimize machine's resources 
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# function to create training and validation data loaders
def make_dataloaders(batch_size=64):
    # X is the input features, and y is the class label for each point.
    X, y = make_moons(n_samples=1200, noise=0.2, random_state=42)

    X = X.astype(np.float32)
    y = y.astype(np.float32).reshape(-1, 1)

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


class MoonMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64), # batch normalization
            nn.ReLU(),
            nn.Dropout(0.3), # randomly drops 30% of activations during training

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


def build_model(input_dim):
    return MoonMLP(input_dim)


def train(model, train_loader, val_loader, device, epochs=40, lr=1e-3):
    criterion = nn.BCEWithLogitsLoss() # defines the loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) # creates Adam optimizer

    for epoch in range(epochs): #loops 40 times
        model.train()

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad() # clear old gradients
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    return {}


def evaluate(model, loader, device):
    model.eval()
    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad(): # disables gradient tracking
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)

            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(int)

            y_prob.append(probs)
            y_pred.append(preds)
            y_true.append(yb.numpy())

    #stacks all batches of true labels into one array, then flattens it
    y_true = np.vstack(y_true).ravel() 
    y_pred = np.vstack(y_pred).ravel()
    y_prob = np.vstack(y_prob).ravel()

    mse = mean_squared_error(y_true, y_prob)
    r2 = r2_score(y_true, y_prob)
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
        probs = torch.sigmoid(logits)
    return probs.cpu()


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

        print("TRAIN:", train_metrics)
        print("VAL:", val_metrics)

        save_artifacts(
            "tasks/mlp_lvl7_moons_dropout_batchnorm/artifacts",
            model,
            {"train": train_metrics, "val": val_metrics},
        )

        assert val_metrics["accuracy"] > 0.90, "Accuracy too low"

        sys.exit(0)

    except Exception as e:
        print("Error:", e)
        sys.exit(1)