# Task 1: Artificial Neural Network (ANN) for Binary Classification

A deep learning project implementing a Multi-Layer Perceptron (ANN) in **PyTorch** for binary classification using the **Heart Failure Prediction Dataset** automatically downloaded from Kaggle.

---

## 📋 Overview

The objective of this project is to predict heart disease risk (`HeartDisease` = 0 or 1) based on clinical and biometric indicators. The model demonstrates:
- End-to-end tabular deep learning with PyTorch.
- Automated Kaggle dataset fetching and local caching using `kagglehub`.
- Categorical one-hot encoding, continuous feature scaling, and stratified splitting.
- Multi-layer perceptron with non-linear activations (`ReLU`), regularization (`Dropout`), and numerically stable binary cross-entropy loss (`BCEWithLogitsLoss`).

---

## 📂 Project Structure

```
01_ann_binary_classification/
├── README.md      # Detailed project documentation
└── main.py        # End-to-end training and evaluation script
```

---

## 📊 Dataset Details

- **Dataset**: [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
- **Kaggle Slug**: `fedesoriano/heart-failure-prediction`
- **Total Samples**: 918 patients
- **Target Variable**: `HeartDisease` (Binary: `0` = Normal, `1` = Heart Disease)
- **Clinical Features**:
  - `Age`, `RestingBP`, `Cholesterol`, `FastingBS`, `MaxHR`, `Oldpeak` (numerical)
  - `Sex`, `ChestPainType`, `RestingECG`, `ExerciseAngina`, `ST_Slope` (categorical)

### Preprocessing Pipeline:
1. **One-Hot Encoding**: Categorical variables are converted to numerical indicators using `pd.get_dummies(..., drop_first=True)`.
2. **Train/Val/Test Split**: 70% Training, 15% Validation, 15% Test split.
3. **Feature Standardization**: `StandardScaler` standardizes features ($\mu=0, \sigma=1$) fit strictly on the training partition and transformed across validation and test partitions.
4. **PyTorch DataLoader**: Converted to `TensorDataset` with mini-batches of size 32 and shuffle enabled for training.

---

## 🧠 Model Architecture

The model is defined as `BinaryClassifierANN`:

```
Input Features (15 dimensions after one-hot encoding)
  │
  ├── Linear(in_features=15, out_features=32)
  ├── ReLU()
  ├── Dropout(p=0.20)
  │
  ├── Linear(in_features=32, out_features=16)
  ├── ReLU()
  │
  └── Linear(in_features=16, out_features=1) ──> Logit Output
```

- **Loss Function**: `nn.BCEWithLogitsLoss()` (utilizes log-sum-exp trick for numerical stability over manual sigmoid + BCELoss).
- **Optimizer**: `optim.Adam(lr=0.005)`
- **Decision Boundary**: Threshold $\sigma(\text{logit}) \ge 0.5 \implies \text{Class } 1$

---

## 🚀 How to Run

### 1. Prerequisites
Ensure the required dependencies are installed:
```bash
pip install torch pandas numpy scikit-learn kagglehub
```

### 2. Execution

**From the repository root:**
```bash
python "DL projects/01_ann_binary_classification/main.py"
```

**Or navigate to the project directory:**
```bash
cd "DL projects/01_ann_binary_classification"
python main.py
```

---

## 📈 Sample Output

```text
=================================================================
Task 1: ANN for Binary Classification (Kaggle Dataset)
=================================================================
Dataset loaded from Kaggle: heart.csv
Total samples: 918 | Columns: ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope', 'HeartDisease']
Input Features: 15
BinaryClassifierANN(
  (network): Sequential(
    (0): Linear(in_features=15, out_features=32, bias=True)
    (1): ReLU()
    (2): Dropout(p=0.2, inplace=False)
    (3): Linear(in_features=32, out_features=16, bias=True)
    (4): ReLU()
    (5): Linear(in_features=16, out_features=1, bias=True)
  )
)
-----------------------------------------------------------------
Epoch [ 1/50] | Train Loss: 0.6132 | Val Accuracy: 84.78%
Epoch [10/50] | Train Loss: 0.3229 | Val Accuracy: 86.96%
Epoch [20/50] | Train Loss: 0.2854 | Val Accuracy: 86.96%
Epoch [30/50] | Train Loss: 0.2587 | Val Accuracy: 86.23%
Epoch [40/50] | Train Loss: 0.2413 | Val Accuracy: 86.96%
Epoch [50/50] | Train Loss: 0.2298 | Val Accuracy: 86.96%
-----------------------------------------------------------------
Final Test Accuracy: 86.96%
Sample Test Predictions:
  Sample 1: Prob=0.0821 -> Pred=0, Actual=0
  Sample 2: Prob=0.9124 -> Pred=1, Actual=1
  Sample 3: Prob=0.7850 -> Pred=1, Actual=1
  Sample 4: Prob=0.1235 -> Pred=0, Actual=0
  Sample 5: Prob=0.8992 -> Pred=1, Actual=1
=================================================================
```
