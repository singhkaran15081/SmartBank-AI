import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

df = pd.read_csv("bank-full.csv", sep=";")

X = df.drop(columns=["y", "duration"])
y = df["y"].map({"no": 0, "yes": 1})

categorical_columns = X.select_dtypes(
    include=["object", "str"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    include="number"
).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numerical_columns),
    ("categorical", categorical_pipeline, categorical_columns)
])

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("Original training shape:", X_train.shape)
print("Processed training shape:", X_train_processed.shape)

print("Original testing shape:", X_test.shape)
print("Processed testing shape:", X_test_processed.shape)

print("Preprocessing completed successfully!")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Train Logistic Regression model
model = LogisticRegression(max_iter=1000)

model.fit(X_train_processed, y_train)

# Step 2: Make predictions
y_pred = model.predict(X_test_processed)

# Step 3: Evaluate the model
print("\n--- Logistic Regression Results ---")

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Improved Logistic Regression
balanced_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

balanced_model.fit(X_train_processed, y_train)

balanced_pred = balanced_model.predict(X_test_processed)

print("\n--- Balanced Model Results ---")

print("Accuracy:",
      accuracy_score(y_test, balanced_pred))

print("\nClassification Report:")
print(classification_report(y_test, balanced_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, balanced_pred))

from sklearn.tree import DecisionTreeClassifier

# Train Decision Tree
tree_model = DecisionTreeClassifier(
    max_depth=8,
    class_weight="balanced",
    random_state=42
)

tree_model.fit(X_train_processed, y_train)

# Predictions
tree_pred = tree_model.predict(X_test_processed)

# Evaluation
print("\n--- Decision Tree Results ---")

print("Accuracy:",
      accuracy_score(y_test, tree_pred))

print("\nClassification Report:")
print(classification_report(y_test, tree_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, tree_pred))  

from sklearn.ensemble import RandomForestClassifier

# Train Random Forest
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_processed, y_train)

# Predictions
rf_pred = rf_model.predict(X_test_processed)

# Evaluation
print("\n--- Random Forest Results ---")

print("Accuracy:",
      accuracy_score(y_test, rf_pred))

print("\nClassification Report:")
print(classification_report(y_test, rf_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_pred))

import joblib

joblib.dump(
    {
        "preprocessor": preprocessor,
        "decision_tree": tree_model,
        "random_forest": rf_model,
        "logistic_regression": model,
        "balanced_logistic_regression": balanced_model
    },
    "smartbank_models.pkl"
)

print("\nModels saved successfully!")