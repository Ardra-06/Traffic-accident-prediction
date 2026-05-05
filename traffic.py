# =========================
# 1. IMPORT LIBRARIES
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# =========================
# 2. LOAD DATASET
# =========================
df = pd.read_csv("road_accident.csv")   # change filename if needed


# =========================
# 3. DROP USELESS / LEAKAGE COLUMNS
# =========================
df = df.drop(columns=[
    'accident_id',   # ID (useless)
    'date',          # redundant
    'time'           # redundant
])

# OPTIONAL (recommended to avoid data leakage)
df = df.drop(columns=['risk_score'])


# =========================
# 4. CHECK DATA
# =========================
print("Columns:\n", df.columns)
print("\nShape:", df.shape)


# =========================
# 5. HANDLE MISSING VALUES
# =========================
df = df.dropna()


# =========================
# 6. DEFINE TARGET
# =========================
target_column = 'accident_severity'


# =========================
# 7. ENCODE CATEGORICAL DATA
# =========================
le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])


# =========================
# 8. SPLIT FEATURES & TARGET
# =========================
X = df.drop(target_column, axis=1)
y = df[target_column]


# =========================
# 9. TRAIN-TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =========================
# 10. FEATURE SCALING
# =========================
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# =========================
# 11. TRAIN MODEL
# =========================
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    random_state=42
)

model.fit(X_train, y_train)


# =========================
# 12. PREDICT
# =========================
y_pred = model.predict(X_test)


# =========================
# 13. EVALUATION
# =========================
print("\n===== MODEL PERFORMANCE =====")

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

print("\nConfusion Matrix:\n")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# =========================
# 14. GRAPH - CONFUSION MATRIX
# =========================
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# =========================
# 15. FEATURE IMPORTANCE
# =========================
importance = model.feature_importances_
features = X.columns

importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)

print("\nTop Features:\n", importance_df.head(10))


# =========================
# 16. GRAPH - FEATURE IMPORTANCE
# =========================
plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(10))
plt.title("Top 10 Important Features")
plt.show()


# =========================
# 17. GRAPH - CLASS DISTRIBUTION
# =========================
sns.countplot(x=y)
plt.title("Class Distribution")
plt.xlabel("Accident Severity")
plt.ylabel("Count")
plt.show()


# =========================
# 18. SAVE MODEL
# =========================
import pickle

with open("accident_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel saved as accident_model.pkl")