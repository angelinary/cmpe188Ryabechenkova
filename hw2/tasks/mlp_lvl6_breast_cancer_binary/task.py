'''
Created on Apr 12, 2026

@author: angelinaryabechenkova
'''
import json # save metrics into .json file
import os # for folders and filepaths
import random # for random seed
import sys # lets the script exit with sys.exit(0) or sys.exit(1)

import numpy as np # for arrays and numberhandling
import torch # ML library
from sklearn.datasets import load_breast_cancer # loads the built in dataset
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split # slpits data
from sklearn.preprocessing import StandardScaler # standardize features
from torch import nn # neural-network layers and loss functions
from torch.utils.data import DataLoader, TensorDataset

# returns data about the data
def get_task_metadata():
    return {
        "task_id": "mlp_lvl6_breast_cancer_binary",
        "task_type": "classification",
        "metric_thresholds": {
            "val_acc_min": 0.90
        }
    }

# controls randomness. Otherwise each run seed is different
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Use GPU for faster results, otherwise settle for CPU
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# returns batch loaders
def make_dataloaders(batch_size=64):
    data = load_breast_cancer()
    X = data.data.astype(np.float32) # input features
    y = data.target.astype(np.float32).reshape(-1, 1) # binary target label

    # Splits data into training and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Creates a scaler object
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)

    # Converts NumPy arrays into PyTorch tensors
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    # Wraps features and labels together as datasets
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

# Defines a neural network class for binary classification
class MLPBinary(nn.Module):
    # build the model
    def __init__(self, input_dim):
        super().__init__() # initialize parent nn.Module
        self.net = nn.Sequential( # Stack layers in order
            nn.Linear(input_dim, 64), # 64 neurons
            nn.ReLU(), # Activation function
            nn.Linear(64, 32), # go from 64 to 32 neurons
            nn.ReLU(), #Activate non-linearity
            nn.Linear(32, 1)  # one output for binary -  1 neuron
        )

    def forward(self, x):
        return self.net(x)


def build_model(input_dim): # Create and return the model
    return MLPBinary(input_dim)


# Train the model, given 30 iterations over the dataset and learning rate 0.001
def train(model, train_loader, val_loader, device, epochs=30, lr=1e-3):
    criterion = nn.BCEWithLogitsLoss() # Loss function for binary
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) # Use Adam as optimizer

    for epoch in range(epochs):
        model.train()
        
        # xb - batch of input features, yb - batch of labels
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad() # clear old gradients 
            logits = model(xb)
            loss = criterion(logits, yb) # comp predictions to true labels
            loss.backward() # backpropagation
            optimizer.step() # Updates the model’s weights using those gradients

    return {}

# Evaluates model performance on a dataset
def evaluate(model, loader, device):
    model.eval()
    y_true = []
    y_pred = []
    y_prob = []

    # Disables grad tracking, so makes evaluation faster and uses less memory
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)

            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(int)

            y_prob.append(probs)
            y_pred.append(preds)
            y_true.append(yb.numpy())

    y_true = np.vstack(y_true).ravel()
    y_pred = np.vstack(y_pred).ravel()
    y_prob = np.vstack(y_prob).ravel()

    # Measures average squared difference between true 
    # labels and predicted probabilities
    mse = mean_squared_error(y_true, y_prob)
    r2 = r2_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred) # Measures fraction of correct class predictions

    return {
        "mse": float(mse),
        "r2": float(r2),
        "accuracy": float(acc),
    }

# Makes predictions for new input data
def predict(model, x_tensor, device):
    model.eval()
    with torch.no_grad():
        logits = model(x_tensor.to(device))
        probs = torch.sigmoid(logits)
    return probs.cpu()


def save_artifacts(output_dir, model, metrics):
    os.makedirs(output_dir, exist_ok=True) # makes a dir to save output
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    try: # tyr-catch for the exception
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
            "tasks/mlp_lvl6_breast_cancer_binary/artifacts",
            model,
            {"train": train_metrics, "val": val_metrics},
        )

        assert val_metrics["accuracy"] > 0.90, "Accuracy too low"

        sys.exit(0)

    except Exception as e:
        print("Error:", e)
        sys.exit(1)


