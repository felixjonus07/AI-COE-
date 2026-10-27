"""
Task 2: Convolutional Neural Network (CNN) for Dog–Cat Image Classification
Dataset: Kaggle Dogs & Cats Dataset (chetankv/dogs-cats-images)
Architecture: Exactly 3 Convolutional Layers
Framework: PyTorch / Torchvision
"""

import os
import glob
import kagglehub
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

# -------------------------------------------------------------
# 1. CNN Model Definition (Exactly 3 Convolutional Layers)
# -------------------------------------------------------------
class DogCatCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        # --- CONVOLUTIONAL LAYER 1 ---
        # 3 RGB input channels -> 32 filters (3x3), padding=1
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 64x64 -> 32x32
        )

        # --- CONVOLUTIONAL LAYER 2 ---
        # 32 filters -> 64 filters (3x3), padding=1
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 32x32 -> 16x16
        )

        # --- CONVOLUTIONAL LAYER 3 ---
        # 64 filters -> 128 filters (3x3), padding=1
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 16x16 -> 8x8
        )

        # Adaptive pool fixes spatial dimension to 4x4
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # --- DENSE CLASSIFIER ---
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes)  # 2 classes: [Cat, Dog]
        )

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = self.adaptive_pool(x)
        logits = self.classifier(x)
        return logits


# -------------------------------------------------------------
# 2. Dataset Pipeline & Preprocessing from Kaggle
# -------------------------------------------------------------
def get_dataloaders(max_train_samples=800, max_val_samples=200):
    print("=" * 65)
    print("Task 2: 3-Layer CNN for Dog–Cat Image Classification (Kaggle)")
    print("=" * 65)

    # Automatically download/fetch Kaggle Dogs vs Cats dataset
    print("Loading Kaggle Dogs vs Cats dataset...")
    kaggle_path = kagglehub.dataset_download("chetankv/dogs-cats-images")
    train_dir = os.path.join(kaggle_path, "dataset", "training_set")
    test_dir  = os.path.join(kaggle_path, "dataset", "test_set")

    # Image transformations: resize to 64x64, augment, normalize
    train_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_data_full = ImageFolder(root=train_dir, transform=train_transforms)
    val_data_full   = ImageFolder(root=test_dir, transform=val_transforms)

    # Sample a balanced subset of cats and dogs for training and validation
    half_train = max_train_samples // 2
    half_val = max_val_samples // 2

    cat_train = [i for i, (_, l) in enumerate(train_data_full.samples) if l == 0][:half_train]
    dog_train = [i for i, (_, l) in enumerate(train_data_full.samples) if l == 1][:half_train]
    train_subset_indices = cat_train + dog_train

    cat_val = [i for i, (_, l) in enumerate(val_data_full.samples) if l == 0][:half_val]
    dog_val = [i for i, (_, l) in enumerate(val_data_full.samples) if l == 1][:half_val]
    val_subset_indices = cat_val + dog_val

    train_subset = Subset(train_data_full, train_subset_indices)
    val_subset   = Subset(val_data_full, val_subset_indices)

    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_subset, batch_size=32, shuffle=False)

    print(f"Classes: {train_data_full.classes} (0={train_data_full.classes[0]}, 1={train_data_full.classes[1]})")
    print(f"Training samples: {len(train_subset)} | Validation samples: {len(val_subset)}")
    print("-" * 65)
    return train_loader, val_loader, train_data_full.classes


# -------------------------------------------------------------
# 3. Model Training & Validation Routine
# -------------------------------------------------------------
def train_model():
    train_loader, val_loader, class_names = get_dataloaders()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DogCatCNN(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(model)
    print("-" * 65)

    epochs = 5
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = (correct / total) * 100

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                val_outputs = model(val_images)
                _, val_preds = torch.max(val_outputs, 1)
                val_total += val_labels.size(0)
                val_correct += (val_preds == val_labels).sum().item()

        val_acc = (val_correct / val_total) * 100 if val_total > 0 else 0.0
        print(f"Epoch [{epoch}/{epochs}] - Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.1f}% | Val Acc: {val_acc:.1f}%")

    # Sample Inference
    print("-" * 65)
    print("Sample Image Predictions from Kaggle Test Set:")
    model.eval()
    sample_images, sample_labels = next(iter(val_loader))
    with torch.no_grad():
        outputs = model(sample_images[:4].to(device))
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)

    for i in range(4):
        pred_label = class_names[preds[i].item()]
        true_label = class_names[sample_labels[i].item()]
        conf = probs[i][preds[i]].item() * 100
        print(f"  Image {i+1}: Predicted '{pred_label}' ({conf:.1f}% conf) | Actual: '{true_label}'")
    print("=" * 65)


if __name__ == "__main__":
    train_model()
