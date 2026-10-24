"""
Task 2: Convolutional Neural Network (CNN) for Dog-Cat Image Classification
Architecture: 3 Convolutional Layers
Framework: PyTorch / Torchvision
"""

import os
import shutil
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

# -------------------------------------------------------------
# 1. CNN Model Definition (3 Convolutional Layers)
# -------------------------------------------------------------
class DogCatCNN(nn.Module):
    """
    Convolutional Neural Network with exactly 3 Convolutional Layers
    for Binary / 2-Class Image Classification (Cat vs Dog).
    """
    def __init__(self, num_classes=2):
        super().__init__()

        # --- CONVOLUTIONAL LAYER 1 ---
        # Input: 3 channels (RGB) -> Output: 32 feature maps
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Halves spatial dimensions
        )

        # --- CONVOLUTIONAL LAYER 2 ---
        # Input: 32 feature maps -> Output: 64 feature maps
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Halves spatial dimensions
        )

        # --- CONVOLUTIONAL LAYER 3 ---
        # Input: 64 feature maps -> Output: 128 feature maps
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Halves spatial dimensions
        )

        # Global average pool to fix spatial size to (4, 4) regardless of input resolution
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # --- FULLY CONNECTED (DENSE) CLASSIFIER ---
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),                  # Prevents overfitting
            nn.Linear(128, num_classes)          # 2 outputs: [Cat, Dog]
        )

    def forward(self, x):
        # Pass through the 3 Conv Layers
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)

        # Pool and Classify
        x = self.adaptive_pool(x)
        logits = self.classifier(x)
        return logits


# -------------------------------------------------------------
# 2. Dataset Pipeline & Transforms
# -------------------------------------------------------------
# Image preprocessing and augmentation
transform_train = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def create_demo_data(base_dir="dog_cat_demo_data"):
    """
    Creates a small demonstration dataset of synthetic images
    so the code runs out-of-the-box without requiring a large download.
    For real usage, replace 'dog_cat_demo_data' with your actual dataset folder:
      dataset/
        train/
          cats/
          dogs/
        val/
          cats/
          dogs/
    """
    np.random.seed(42)
    for split in ["train", "val"]:
        num_samples = 40 if split == "train" else 10
        for label in ["cat", "dog"]:
            dir_path = os.path.join(base_dir, split, label)
            os.makedirs(dir_path, exist_ok=True)
            for i in range(num_samples):
                # Generate sample color variations
                base_color = (200, 120, 100) if label == "cat" else (80, 140, 220)
                noise = np.random.randint(-30, 30, (64, 64, 3), dtype=np.int16)
                arr = np.clip(np.array(base_color, dtype=np.int16) + noise, 0, 255).astype(np.uint8)
                img = Image.fromarray(arr)
                img.save(os.path.join(dir_path, f"{label}_{i}.jpg"))


# -------------------------------------------------------------
# 3. Training & Evaluation Routine
# -------------------------------------------------------------
def train_model():
    demo_dir = "dog_cat_demo_data"
    print("=" * 60)
    print("Task 2: 3-Layer CNN for Dog-Cat Image Classification")
    print("=" * 60)

    # Prepare demo dataset if not already present
    if not os.path.exists(demo_dir):
        print("Creating lightweight demo dataset with cats and dogs...")
        create_demo_data(demo_dir)

    train_path = os.path.join(demo_dir, "train")
    val_path = os.path.join(demo_dir, "val")

    # Load images using standard PyTorch ImageFolder
    train_dataset = ImageFolder(root=train_path, transform=transform_train)
    val_dataset = ImageFolder(root=val_path, transform=transform_val)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    print(f"Classes found: {train_dataset.classes} (0={train_dataset.classes[0]}, 1={train_dataset.classes[1]})")
    print(f"Training images: {len(train_dataset)} | Validation images: {len(val_dataset)}")
    print("-" * 60)

    # Initialize device, model, criterion, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DogCatCNN(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(model)
    print("-" * 60)

    # Training loop
    epochs = 5
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

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

        val_acc = (val_correct / val_total) * 100
        print(f"Epoch [{epoch}/{epochs}] - Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.1f}% | Val Acc: {val_acc:.1f}%")

    # ---------------------------------------------------------
    # 4. Inference on Sample Images
    # ---------------------------------------------------------
    print("-" * 60)
    print("Testing inference on 4 validation samples:")
    model.eval()
    sample_images, sample_labels = next(iter(val_loader))
    with torch.no_grad():
        outputs = model(sample_images.to(device))
        probabilities = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)

    for i in range(min(4, len(sample_labels))):
        pred_label = train_dataset.classes[preds[i].item()]
        true_label = train_dataset.classes[sample_labels[i].item()]
        confidence = probabilities[i][preds[i]].item() * 100
        print(f"  Image {i+1}: Predicted '{pred_label}' ({confidence:.1f}% confidence) | Actual: '{true_label}'")
    print("=" * 60)


if __name__ == "__main__":
    train_model()
