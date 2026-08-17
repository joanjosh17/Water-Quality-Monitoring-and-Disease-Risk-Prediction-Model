```python
# ============================================================
# Water Quality Monitoring and Disease Risk Prediction Model
# ============================================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score
)

import seaborn as sns
import joblib


# ============================================================
# 1. LOAD DATASET
# ============================================================

# Load dataset
df = pd.read_csv("water_quality_disease_risk_dataset.csv")

print("Dataset Shape:", df.shape)
print(df.head())


# ============================================================
# 2. DATA CLEANING
# ============================================================

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Remove duplicates
df.drop_duplicates(inplace=True)

# Convert date column
df["date"] = pd.to_datetime(df["date"])

# Extract time features
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month


# ============================================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

# Disease Risk Distribution
plt.figure(figsize=(8,5))
sns.countplot(data=df, x="risk_category")
plt.title("Disease Risk Category Distribution")
plt.show()

# Water Source Distribution
plt.figure(figsize=(8,5))
sns.countplot(data=df, x="water_source")
plt.title("Water Source Distribution")
plt.xticks(rotation=45)
plt.show()

# Correlation Heatmap
numeric_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(12,8))
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm"
)
plt.title("Feature Correlation Heatmap")
plt.show()

# Monthly Disease Trend
monthly_cases = (
    df.groupby("month")["disease_cases_reported"]
    .mean()
)

plt.figure(figsize=(8,5))
monthly_cases.plot(marker="o")
plt.title("Average Disease Cases by Month")
plt.xlabel("Month")
plt.ylabel("Disease Cases")
plt.grid(True)
plt.show()

# pH vs Disease Risk Score
plt.figure(figsize=(8,5))
sns.scatterplot(
    data=df,
    x="ph_level",
    y="disease_risk_score",
    hue="risk_category"
)
plt.title("pH Level vs Disease Risk Score")
plt.show()


# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================

# Drop unnecessary columns
df_model = df.drop(
    columns=[
        "sample_id",
        "date",
        "community"
    ]
)

# Encode categorical variables
categorical_cols = [
    "region",
    "water_source",
    "season"
]

encoder = LabelEncoder()

for col in categorical_cols:
    df_model[col] = encoder.fit_transform(df_model[col])

# Encode target variable
target_encoder = LabelEncoder()
df_model["risk_category"] = target_encoder.fit_transform(
    df_model["risk_category"]
)

# Features and Target
X = df_model.drop("risk_category", axis=1)
y = df_model["risk_category"]


# ============================================================
# 5. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================================
# 6. MODEL TRAINING
# ============================================================

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)


# ============================================================
# 7. MODEL EVALUATION
# ============================================================

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7,5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# ============================================================
# 8. FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

plt.figure(figsize=(10,6))
sns.barplot(
    data=feature_importance,
    x="Importance",
    y="Feature"
)

plt.title("Feature Importance")
plt.show()

print("\nTop Important Features:")
print(feature_importance.head(10))


# ============================================================
# 9. HIGH-RISK WATER HOTSPOTS
# ============================================================

high_risk_regions = (
    df[df["risk_category"] == "High"]
    .groupby("region")
    .size()
    .sort_values(ascending=False)
)

plt.figure(figsize=(8,5))
high_risk_regions.plot(kind="bar")

plt.title("High Risk Water Regions")
plt.xlabel("Region")
plt.ylabel("High Risk Samples")
plt.xticks(rotation=45)
plt.show()


# ============================================================
# 10. SAVE MODEL
# ============================================================

joblib.dump(
    model,
    "water_disease_risk_model.pkl"
)

print("\nModel saved successfully!")


# ============================================================
# 11. PREDICT NEW WATER SAMPLE
# ============================================================

new_sample = pd.DataFrame({
    "region": [1],
    "water_source": [2],
    "season": [1],
    "temperature_c": [29],
    "rainfall_mm": [80],
    "ph_level": [5.8],
    "turbidity_ntu": [20],
    "dissolved_oxygen_mg_l": [4.5],
    "nitrate_mg_l": [28],
    "coliform_count_cfu": [300],
    "lead_ppm": [0.07],
    "disease_cases_reported": [18],
    "disease_risk_score": [75],
    "year": [2025],
    "month": [7]
})

prediction = model.predict(new_sample)

risk_label = target_encoder.inverse_transform(
    prediction
)

print("\nPredicted Disease Risk:")
print(risk_label[0])

```
