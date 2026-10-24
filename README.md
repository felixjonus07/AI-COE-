# AI-COE Deep Learning Projects

This repository contains implementations of foundational deep learning architectures and tasks developed in PyTorch.

## Directory Structure

```
AI-COE-/
├── .gitignore
├── README.md
└── DL projects/
    ├── 01_ann_binary_classification_make_moons.py
    ├── 02_cnn_dog_cat_classification.py
    └── 03_transformer_next_token_prediction.py
```

## Projects Overview

### 1. ANN for Binary Classification (`make_moons`)
- **Script:** `DL projects/01_ann_binary_classification_make_moons.py`
- **Dataset:** Scikit-learn synthetic `make_moons` dataset.
- **Architecture:** Multi-Layer Perceptron (ANN) with 2 hidden layers (ReLU activation) and `BCEWithLogitsLoss`.

### 2. CNN for Dog–Cat Image Classification
- **Script:** `DL projects/02_cnn_dog_cat_classification.py`
- **Architecture:** Exactly 3 Convolutional Layers (Conv2D + BatchNorm + ReLU + MaxPool), followed by Adaptive Average Pooling, Dropout, and Fully Connected classifier.
- **Data Pipeline:** Compatible with standard PyTorch `ImageFolder`. Includes an automatic sample generator for quick testing without manual downloads.

### 3. Transformer-Based Next-Token Prediction
- **Script:** `DL projects/03_transformer_next_token_prediction.py`
- **Architecture:** Basic Causal Scaled Dot-Product Self-Attention implemented from first principles, causal triangular masking, Transformer Decoder Blocks with LayerNorm and Feed-Forward Networks (FFN).
- **Task:** Autoregressive language modeling and greedy token sequence generation.

## How to Run

```bash
# Task 1
python "DL projects/01_ann_binary_classification_make_moons.py"

# Task 2
python "DL projects/02_cnn_dog_cat_classification.py"

# Task 3
python "DL projects/03_transformer_next_token_prediction.py"
```
