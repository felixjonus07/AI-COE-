import os
import glob
import kagglehub
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Download & Load Dataset
path = kagglehub.dataset_download("aariyan101/usa-housingcsv")
csv_files = glob.glob(os.path.join(path, "*.csv"))
df = pd.read_csv(csv_files[0])

if 'Address' in df.columns:
    df = df.drop('Address', axis=1)

X = df.drop('Price', axis=1)
y = df['Price']

# 2. 70 : 15 : 15 Split (Train : Validation : Test)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test     = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

# 3. Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

# 4. Model Training (Fixed Here!)
name = "Linear Regression"
model = LinearRegression()

model.fit(X_train_scaled, y_train)

# 5. Predictions
train_pred = model.predict(X_train_scaled)
val_pred   = model.predict(X_val_scaled)
test_pred  = model.predict(X_test_scaled)

# 6. Evaluation Results
print(f"================ {name} ================")

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
