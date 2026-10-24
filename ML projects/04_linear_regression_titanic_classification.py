import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

df = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']].copy()
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = df.dropna()

X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test     = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

train_pred_cont = model.predict(X_train_scaled)
val_pred_cont   = model.predict(X_val_scaled)
test_pred_cont  = model.predict(X_test_scaled)

train_pred = (train_pred_cont >= 0.5).astype(int)
val_pred   = (val_pred_cont >= 0.5).astype(int)
test_pred  = (test_pred_cont >= 0.5).astype(int)

print("================ Linear Regression (Titanic) ================")

print("\n  [TRAIN SET - 70%]")
print(f"    Train Accuracy           : {accuracy_score(y_train, train_pred)*100:.2f}%")
print(f"    Train RMSE               : {np.sqrt(mean_squared_error(y_train, train_pred_cont)):.4f}")
print(f"    Train MAE                : {mean_absolute_error(y_train, train_pred_cont):.4f}")

print("\n  [VALIDATION SET - 15%]")
print(f"    Validation Accuracy      : {accuracy_score(y_val, val_pred)*100:.2f}%")
print(f"    Validation RMSE          : {np.sqrt(mean_squared_error(y_val, val_pred_cont)):.4f}")
print(f"    Validation MAE           : {mean_absolute_error(y_val, val_pred_cont):,.4f}")

print("\n  [TEST SET - 15%]")
print(f"    Test Accuracy            : {accuracy_score(y_test, test_pred)*100:.2f}%")
print(f"    Test RMSE                : {np.sqrt(mean_squared_error(y_test, test_pred_cont)):.4f}")
print(f"    Test MAE                 : {mean_absolute_error(y_test, test_pred_cont):,.4f}")
