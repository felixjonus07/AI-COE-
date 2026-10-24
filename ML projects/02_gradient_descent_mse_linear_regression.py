import os
import glob
import kagglehub
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

path = kagglehub.dataset_download("aariyan101/usa-housingcsv")
csv_files = glob.glob(os.path.join(path, "*.csv"))
df = pd.read_csv(csv_files[0])

if 'Address' in df.columns:
    df = df.drop('Address', axis=1)

df = df[df['Price'] > 0].copy()

X = df.drop('Price', axis=1)
y = df['Price']

y_log = np.log(y)

X_train, X_temp, y_train_log, y_temp_log = train_test_split(X, y_log, test_size=0.30, random_state=42)
X_val, X_test, y_val_log, y_test_log     = train_test_split(X_temp, y_temp_log, test_size=0.50, random_state=42)

y_train = np.exp(y_train_log)
y_val   = np.exp(y_val_log)
y_test  = np.exp(y_test_log)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

class GradientDescentLinearRegression:
    def __init__(self, learning_rate=0.01, epochs=3000):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0.0

        for _ in range(self.epochs):
            y_pred = np.dot(X, self.weights) + self.bias
            dw = (2 / m) * np.dot(X.T, (y_pred - y))
            db = (2 / m) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

model = GradientDescentLinearRegression(learning_rate=0.01, epochs=3000)
model.fit(X_train_scaled, y_train_log)

train_pred = np.exp(model.predict(X_train_scaled))
val_pred   = np.exp(model.predict(X_val_scaled))
test_pred  = np.exp(model.predict(X_test_scaled))

print("================ Gradient Descent Linear Regression ================")

print("\n  [TRAIN SET - 70%]")
print(f"    Train Accuracy (R2)      : {r2_score(y_train, train_pred)*100:.2f}%")
print(f"    Train RMSE               : ${np.sqrt(mean_squared_error(y_train, train_pred)):,.2f}")
print(f"    Train MAE                : ${mean_absolute_error(y_train, train_pred):,.2f}")

print("\n  [VALIDATION SET - 15%]")
print(f"    Validation Accuracy (R2) : {r2_score(y_val, val_pred)*100:.2f}%")
print(f"    Validation RMSE          : ${np.sqrt(mean_squared_error(y_val, val_pred)):,.2f}")
print(f"    Validation MAE           : ${mean_absolute_error(y_val, val_pred):,.2f}")

print("\n  [TEST SET - 15%]")
print(f"    Test Accuracy (R2)       : {r2_score(y_test, test_pred)*100:.2f}%")
print(f"    Test RMSE                : ${np.sqrt(mean_squared_error(y_test, test_pred)):,.2f}")
print(f"    Test MAE                 : ${mean_absolute_error(y_test, test_pred):,.2f}")
