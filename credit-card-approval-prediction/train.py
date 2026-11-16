
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

# 1. Load dataset
df = pd.read_csv("data/credit_approval_dataset.csv")

print("Dataset loaded:", df.shape)

# 2. Separate features and target
X = df.drop(columns=["credit_risk"])
y = df["credit_risk"]

# 3. Identify feature types
numerical_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_columns = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

# 4. Create preprocessing pipelines
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_columns),
    ("cat", categorical_pipeline, categorical_columns)
])

# 5. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 6. Initialize tuned Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    min_samples_leaf=2,
    max_features=0.5,
    max_depth=None,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# 7. Create complete pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", rf_model)
])

# 8. Train and evaluate on held-out test data
print("\nTraining Random Forest...")

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

print("\nMODEL PERFORMANCE")
print("-----------------")

print("Accuracy:", round(
    accuracy_score(y_test, y_pred), 4
))

print("Precision:", round(
    precision_score(y_test, y_pred, zero_division=0), 4
))

print("Recall:", round(
    recall_score(y_test, y_pred, zero_division=0), 4
))

print("F1 Score:", round(
    f1_score(y_test, y_pred, zero_division=0), 4
))

print("ROC-AUC:", round(
    roc_auc_score(y_test, y_prob), 4
))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 9. Retrain final pipeline using all available data
print("\nTraining final model on the complete dataset...")

pipeline.fit(X, y)

# 10. Save trained pipeline
joblib.dump(pipeline, "pipeline.bin")

print("\nModel saved successfully as pipeline.bin")
