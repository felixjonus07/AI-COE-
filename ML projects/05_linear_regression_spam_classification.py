import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report

url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])

df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

vectorizer = TfidfVectorizer(max_features=1500)
X_text = vectorizer.fit_transform(df['message']).toarray()
y = df['label_num']

X_train, X_temp, y_train, y_temp = train_test_split(X_text, y, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test     = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

train_pred_cont = model.predict(X_train)
val_pred_cont   = model.predict(X_val)
test_pred_cont  = model.predict(X_test)

train_pred = (train_pred_cont >= 0.5).astype(int)
val_pred   = (val_pred_cont >= 0.5).astype(int)
test_pred  = (test_pred_cont >= 0.5).astype(int)

print("================ Linear Regression (Spam Detection) ================")

print("\n  [TRAIN SET - 70%]")
print(f"    Train Accuracy           : {accuracy_score(y_train, train_pred)*100:.2f}%")

print("\n  [VALIDATION SET - 15%]")
print(f"    Validation Accuracy      : {accuracy_score(y_val, val_pred)*100:.2f}%")

print("\n  [TEST SET - 15%]")
print(f"    Test Accuracy            : {accuracy_score(y_test, test_pred)*100:.2f}%")
