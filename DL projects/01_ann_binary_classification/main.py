"""
Task 1: Artificial Neural Network (ANN) for Binary Classification
Dataset: Kaggle Heart Failure Prediction Dataset (fedesoriano/heart-failure-prediction)
Framework: PyTorch
"""

import os
import glob
import kagglehub
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# -------------------------------------------------------------
# 1. Dataset Download & Preprocessing from Kaggle
# -------------------------------------------------------------
print("=" * 65)
print("Task 1: ANN for Binary Classification (Kaggle Dataset)")
print("=" * 65)

# Download Kaggle binary classification dataset
path = kagglehub.dataset_download("fedesoriano/heart-failure-prediction")
csv_files = glob.glob(os.path.join(path, "*.csv"))
df = pd.read_csv(csv_files[0])

print(f"Dataset loaded from Kaggle: {os.path.basename(csv_files[0])}")
print(f"Total samples: {len(df)} | Columns: {list(df.columns)}")

# Convert categorical columns to numeric using one-hot encoding
df_encoded = pd.get_dummies(df, drop_first=True)

# Separate features (X) and binary target (y: HeartDisease -> 0 or 1)
target_col = "HeartDisease"
X = df_encoded.drop(target_col, axis=1).values
y = df_encoded[target_col].values

# Split dataset: 70% train, 15% validation, 15% test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test     = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

# Feature normalization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

X_val_t   = torch.tensor(X_val, dtype=torch.float32)
y_val_t   = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

X_test_t  = torch.tensor(X_test, dtype=torch.float32)
y_test_t  = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)


# -------------------------------------------------------------
# 2. ANN Model Definition
# -------------------------------------------------------------
class BinaryClassifierANN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)  # 1 logit output for binary classification
        )

    def forward(self, x):
        return self.network(x)


input_features = X_train.shape[1]
model = BinaryClassifierANN(input_dim=input_features)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

print(f"Input Features: {input_features}")
print(model)
print("-" * 65)


# -------------------------------------------------------------
# 3. Model Training Loop
# -------------------------------------------------------------
epochs = 50
for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        preds = model(batch_X)
        loss = criterion(preds, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_X.size(0)

    train_loss = running_loss / len(train_loader.dataset)

    if epoch % 10 == 0 or epoch == 1:
        model.eval()
        with torch.no_grad():
            val_probs = torch.sigmoid(model(X_val_t))
            val_preds = (val_probs >= 0.5).float()
            val_acc = (val_preds == y_val_t).float().mean().item() * 100
        print(f"Epoch [{epoch:2d}/{epochs}] | Train Loss: {train_loss:.4f} | Val Accuracy: {val_acc:.2f}%")


# -------------------------------------------------------------
# 4. Final Evaluation on Test Set
# -------------------------------------------------------------
model.eval()
with torch.no_grad():
    test_probs = torch.sigmoid(model(X_test_t))
    test_preds = (test_probs >= 0.5).float()
    test_acc = (test_preds == y_test_t).float().mean().item() * 100

print("-" * 65)
print(f"Final Test Accuracy: {test_acc:.2f}%")
print("Sample Test Predictions:")
for i in range(5):
    prob = test_probs[i].item()
    pred = int(test_preds[i].item())
    actual = int(y_test_t[i].item())
    print(f"  Sample {i+1}: Prob={prob:.4f} -> Pred={pred}, Actual={actual}")
print("=" * 65)
