# AI-COE Machine Learning & Deep Learning Projects

This repository contains implementations of foundational Machine Learning algorithms and Deep Learning architectures.

## Directory Structure

```
AI-COE-/
├── .gitignore
├── README.md
├── DL projects/
│   ├── 01_ann_binary_classification_make_moons.py
│   ├── 02_cnn_dog_cat_classification.py
│   └── 03_transformer_next_token_prediction.py
└── ML projects/
    ├── 01_gradient_descent_linear_regression.py
    ├── 02_gradient_descent_mse_linear_regression.py
    ├── 03_sklearn_linear_regression_housing.py
    ├── 04_linear_regression_titanic_classification.py
    └── 05_linear_regression_spam_classification.py
```

---

## 🤖 Deep Learning Projects (`DL projects/`)

### 1. ANN for Binary Classification (`make_moons`)
- **Script:** `DL projects/01_ann_binary_classification_make_moons.py`
- **Dataset:** Scikit-learn synthetic `make_moons` dataset.
- **Architecture:** Multi-Layer Perceptron (ANN) with 2 hidden layers (ReLU activation) and `BCEWithLogitsLoss`.

### 2. CNN for Dog–Cat Image Classification
- **Script:** `DL projects/02_cnn_dog_cat_classification.py`
- **Architecture:** 3 Convolutional Layers (Conv2D + BatchNorm + ReLU + MaxPool), followed by Adaptive Average Pooling, Dropout, and Fully Connected classifier.
- **Data Pipeline:** Compatible with PyTorch `ImageFolder` with sample generation for quick local testing.

### 3. Transformer-Based Next-Token Prediction
- **Script:** `DL projects/03_transformer_next_token_prediction.py`
- **Architecture:** Basic Causal Scaled Dot-Product Self-Attention implemented from first principles, causal triangular masking, Transformer Decoder Blocks with LayerNorm and Feed-Forward Networks (FFN).
- **Task:** Autoregressive language modeling and greedy token sequence generation.

---

## 📊 Machine Learning Projects (`ML projects/`)

### 1. Gradient Descent Linear Regression (USA Housing - Log Target)
- **Script:** `ML projects/01_gradient_descent_linear_regression.py`
- **Method:** Custom gradient descent optimizer with analytical updates $(1/m) X^T (y_{pred} - y)$, learning rate = 0.05, 1,500 epochs.

### 2. Gradient Descent Linear Regression (USA Housing - MSE Derivative)
- **Script:** `ML projects/02_gradient_descent_mse_linear_regression.py`
- **Method:** Custom gradient descent optimizer with $(2/m) X^T (y_{pred} - y)$, learning rate = 0.01, 3,000 epochs, filtering positive prices.

### 3. Scikit-learn Linear Regression (USA Housing)
- **Script:** `ML projects/03_sklearn_linear_regression_housing.py`
- **Method:** Standard closed-form ordinary least squares linear regression with a 70:15:15 Train/Validation/Test split and feature standardization.

### 4. Linear Regression for Classification (Titanic Survival)
- **Script:** `ML projects/04_linear_regression_titanic_classification.py`
- **Method:** Linear Probability Model (threshold $\ge 0.5$) applied to Titanic passenger survival prediction.

### 5. Linear Regression for Text Classification (SMS Spam Detection)
- **Script:** `ML projects/05_linear_regression_spam_classification.py`
- **Method:** TF-IDF text vectorization (1,500 features) combined with linear regression thresholding for binary spam detection.

---

## How to Run

```bash
# Deep Learning Projects
python "DL projects/01_ann_binary_classification_make_moons.py"
python "DL projects/02_cnn_dog_cat_classification.py"
python "DL projects/03_transformer_next_token_prediction.py"

# Machine Learning Projects
python "ML projects/01_gradient_descent_linear_regression.py"
python "ML projects/02_gradient_descent_mse_linear_regression.py"
python "ML projects/03_sklearn_linear_regression_housing.py"
python "ML projects/04_linear_regression_titanic_classification.py"
python "ML projects/05_linear_regression_spam_classification.py"
```
