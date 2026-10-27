# Task 2: 3-Layer CNN for Dog–Cat Image Classification

A deep learning project implementing a Convolutional Neural Network (CNN) with **exactly 3 convolutional layers** in **PyTorch / Torchvision** for binary image classification using the **Kaggle Dogs & Cats Dataset**.

---

## 📋 Overview

This project showcases fundamental computer vision and deep learning techniques for image classification:
- Automated dataset retrieval and disk caching from Kaggle using `kagglehub`.
- Image data ingestion using `torchvision.datasets.ImageFolder` with dynamic transforms.
- Data augmentation (random horizontal flipping) and ImageNet channel-wise normalization.
- A custom 3-layer CNN architecture utilizing Batch Normalization, Max Pooling, Adaptive Average Pooling, and Dropout.
- Multi-class cross-entropy training with sample inference verification.

---

## 📂 Project Structure

```
02_cnn_dog_cat_classification/
├── README.md      # Detailed project documentation
└── main.py        # Complete training pipeline and inference demonstration
```

---

## 📊 Dataset Details

- **Dataset**: [Dogs & Cats Images (Dataset)](https://www.kaggle.com/datasets/chetankv/dogs-cats-images)
- **Kaggle Slug**: `chetankv/dogs-cats-images`
- **Classes**: `cats` (index 0) and `dogs` (index 1)
- **Directory Structure Loaded**:
  ```
  chetankv/dogs-cats-images/
  └── dataset/
      ├── training_set/
      │   ├── cats/
      │   └── dogs/
      └── test_set/
          ├── cats/
          └── dogs/
  ```

### Data Pipeline & Transformations:
- **Training Pipeline**:
  - `transforms.Resize((64, 64))`
  - `transforms.RandomHorizontalFlip()`
  - `transforms.ToTensor()`
  - `transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`
- **Validation Pipeline**:
  - `transforms.Resize((64, 64))`
  - `transforms.ToTensor()`
  - `transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`
- **Balanced Subsetting**: Balanced sampling of both cats and dogs for training (`800` samples) and validation (`200` samples) with `DataLoader(batch_size=32)`.

---

## 🧠 Model Architecture

The architecture enforces **exactly three convolutional stages** before the dense classifier head:

```
Input Image: (3, 64, 64)
  │
  ├── [Conv Stage 1]
  │   ├── Conv2d(in=3, out=32, kernel=3, padding=1)
  │   ├── BatchNorm2d(32)
  │   ├── ReLU()
  │   └── MaxPool2d(kernel=2, stride=2)   ──> Output: (32, 32, 32)
  │
  ├── [Conv Stage 2]
  │   ├── Conv2d(in=32, out=64, kernel=3, padding=1)
  │   ├── BatchNorm2d(64)
  │   ├── ReLU()
  │   └── MaxPool2d(kernel=2, stride=2)   ──> Output: (64, 16, 16)
  │
  ├── [Conv Stage 3]
  │   ├── Conv2d(in=64, out=128, kernel=3, padding=1)
  │   ├── BatchNorm2d(128)
  │   ├── ReLU()
  │   └── MaxPool2d(kernel=2, stride=2)   ──> Output: (128, 8, 8)
  │
  ├── AdaptiveAvgPool2d((4, 4))           ──> Output: (128, 4, 4)
  │
  └── [Dense Classifier Head]
      ├── Flatten()                       ──> 2,048 features
      ├── Linear(2048, 128)
      ├── ReLU()
      ├── Dropout(p=0.50)
      └── Linear(128, 2)                  ──> Logits (Cat vs. Dog)
```

- **Loss Function**: `nn.CrossEntropyLoss()`
- **Optimizer**: `optim.Adam(lr=0.001)`
- **Device Support**: Automatically utilizes CUDA GPU acceleration if available, otherwise CPU.

---

## 🚀 How to Run

### 1. Prerequisites
Ensure the required packages are installed:
```bash
pip install torch torchvision kagglehub
```

### 2. Execution

**From the repository root:**
```bash
python "DL projects/02_cnn_dog_cat_classification/main.py"
```

**Or navigate to the project directory:**
```bash
cd "DL projects/02_cnn_dog_cat_classification"
python main.py
```

---

## 📈 Sample Output

```text
=================================================================
Task 2: 3-Layer CNN for Dog–Cat Image Classification (Kaggle)
=================================================================
Loading Kaggle Dogs vs Cats dataset...
Classes: ['cats', 'dogs'] (0=cats, 1=dogs)
Training samples: 800 | Validation samples: 200
-----------------------------------------------------------------
DogCatCNN(
  (conv_layer1): Sequential(
    (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True)
    (2): ReLU()
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0)
  )
  (conv_layer2): Sequential(
    (0): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
    (2): ReLU()
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0)
  )
  (conv_layer3): Sequential(
    (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
    (2): ReLU()
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0)
  )
  (adaptive_pool): AdaptiveAvgPool2d(output_size=(4, 4))
  (classifier): Sequential(
    (0): Flatten(start_dim=1, end_dim=-1)
    (1): Linear(in_features=2048, out_features=128, bias=True)
    (2): ReLU()
    (3): Dropout(p=0.5, inplace=False)
    (4): Linear(in_features=128, out_features=2, bias=True)
  )
)
-----------------------------------------------------------------
Epoch [1/5] - Train Loss: 0.6970 | Train Acc: 56.6% | Val Acc: 62.0%
Epoch [2/5] - Train Loss: 0.6384 | Train Acc: 63.8% | Val Acc: 66.0%
Epoch [3/5] - Train Loss: 0.5878 | Train Acc: 70.1% | Val Acc: 68.5%
Epoch [4/5] - Train Loss: 0.5401 | Train Acc: 72.5% | Val Acc: 71.0%
Epoch [5/5] - Train Loss: 0.4902 | Train Acc: 76.9% | Val Acc: 73.5%
-----------------------------------------------------------------
Sample Image Predictions from Kaggle Test Set:
  Image 1: Predicted 'cats' (88.4% conf) | Actual: 'cats'
  Image 2: Predicted 'dogs' (79.2% conf) | Actual: 'dogs'
  Image 3: Predicted 'dogs' (85.6% conf) | Actual: 'dogs'
  Image 4: Predicted 'cats' (72.1% conf) | Actual: 'cats'
=================================================================
```
