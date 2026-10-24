"""
Task 1: Artificial Neural Network (ANN) for Binary Classification
Dataset: Scikit-learn make_moons
Framework: PyTorch
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# -------------------------------------------------------------
# 1. Dataset Generation & Preprocessing
# -------------------------------------------------------------
# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Generate synthetic two-moon binary classification dataset
# X: 2D coordinates [x1, x2], y: binary class label (0 or 1)
X, y = make_moons(n_samples=1000, noise=0.20, random_state=42)

# Split dataset: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize feature values (mean=0, variance=1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert NumPy arrays to PyTorch Tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # Shape: (N, 1)

X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# PyTorch DataLoader for mini-batch training
train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


# -------------------------------------------------------------
# 2. ANN Architecture Definition
# -------------------------------------------------------------
class BinaryClassifierANN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=16):
        super().__init__()
        # Multi-layer Perceptron (ANN) with 2 hidden layers
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),     # Input -> Hidden Layer 1
            nn.ReLU(),                            # Activation function
            nn.Linear(hidden_dim, hidden_dim),    # Hidden Layer 1 -> Hidden Layer 2
            nn.ReLU(),                            # Activation function
            nn.Linear(hidden_dim, 1)              # Hidden Layer 2 -> Output Logit
        )

    def forward(self, x):
        # Outputs raw logits (stable for BCEWithLogitsLoss)
        return self.network(x)


# Instantiate the model, loss function, and optimizer
model = BinaryClassifierANN(input_dim=2, hidden_dim=16)
criterion = nn.BCEWithLogitsLoss()  # Combines Sigmoid + Binary Cross-Entropy
optimizer = optim.Adam(model.parameters(), lr=0.01)

print("=" * 60)
print("Task 1: ANN for Binary Classification (make_moons)")
print("=" * 60)
print(model)
print("-" * 60)


# -------------------------------------------------------------
# 3. Model Training Loop
# -------------------------------------------------------------
num_epochs = 100

for epoch in range(1, num_epochs + 1):
    model.train()
    epoch_loss = 0.0

    for batch_X, batch_y in train_loader:
        # 1. Forward pass
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)

        # 2. Backward pass & optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * batch_X.size(0)

    total_train_loss = epoch_loss / len(train_loader.dataset)

    # Print progress every 20 epochs
    if epoch % 20 == 0 or epoch == 1:
        model.eval()
        with torch.no_grad():
            test_logits = model(X_test_t)
            # Apply Sigmoid to convert logits into probabilities
            test_probs = torch.sigmoid(test_logits)
            test_preds = (test_probs >= 0.5).float()
            accuracy = (test_preds == y_test_t).float().mean().item() * 100

        print(f"Epoch [{epoch:3d}/{num_epochs}] | Train Loss: {total_train_loss:.4f} | Test Accuracy: {accuracy:.2f}%")


# -------------------------------------------------------------
# 4. Final Evaluation
# -------------------------------------------------------------
model.eval()
with torch.no_grad():
    final_probs = torch.sigmoid(model(X_test_t))
    final_preds = (final_probs >= 0.5).float()
    final_acc = (final_preds == y_test_t).float().mean().item() * 100

print("-" * 60)
print(f"Final Test Accuracy: {final_acc:.2f}%")
print("Sample Predictions (First 5 test points):")
for i in range(5):
    prob = final_probs[i].item()
    pred = int(final_preds[i].item())
    actual = int(y_test_t[i].item())
    print(f"  Sample {i+1}: Pred Probability={prob:.4f} -> Predicted Class={pred}, Actual Class={actual}")
print("=" * 60)
