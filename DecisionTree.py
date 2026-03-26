# ============================================
# AIRLINE PASSENGER SATISFACTION
# Decision Tree Classifier
# ============================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    ConfusionMatrixDisplay
)

# ============================================
# 1. LOAD DATA
# ============================================

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# ============================================
# 2. CLEAN UP
# ============================================

# Drop irrelevant index columns if they exist
for col in ["id", "Unnamed: 0"]:
    train_df = train_df.drop(columns=col, errors="ignore")
    test_df = test_df.drop(columns=col, errors="ignore")

# Encode the target: 1 = satisfied, 0 = neutral or dissatisfied
label_map = {"satisfied": 1, "neutral or dissatisfied": 0}
train_df["satisfaction"] = train_df["satisfaction"].map(label_map)
test_df["satisfaction"] = test_df["satisfaction"].map(label_map)

# Split into features (X) and target (y)
X_train = train_df.drop(columns="satisfaction")
y_train = train_df["satisfaction"]

X_test = test_df.drop(columns="satisfaction")
y_test = test_df["satisfaction"]

# ============================================
# 3. IDENTIFY FEATURE TYPES
# ============================================

numeric_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = X_train.select_dtypes(exclude=["number"]).columns.tolist()

print("Numeric features:    ", numeric_cols)
print("Categorical features:", categorical_cols)

# ============================================
# 4. PREPROCESSING
# Decision Trees don't need scaling, just
# imputation (filling missing values) and
# one-hot encoding for categorical columns.
# ============================================

numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))  # fill missing numbers with the median
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),  # fill missing text with most common value
    ("onehot", OneHotEncoder(handle_unknown="ignore"))  # convert categories to 0/1 columns
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_pipeline, numeric_cols),
    ("cat", categorical_pipeline, categorical_cols)
])

# ============================================
# 5. BUILD THE FULL PIPELINE
# ============================================

pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", DecisionTreeClassifier(random_state=42))
])

# ============================================
# 6. HYPERPARAMETER TUNING WITH CROSS-VALIDATION
# max_depth — how many levels deep the tree can grow
# min_samples_split — minimum samples required to split a node
# ============================================

param_grid = {
    "model__max_depth": [5, 10, 15, None],
    "model__min_samples_split": [2, 5, 10]
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="f1",  # optimise for F1-score
    cv=5,  # 5-fold cross-validation
    n_jobs=-1,  # use all CPU cores
    refit=True  # refit best model on full training set
)

grid_search.fit(X_train, y_train)

print("\nBest Parameters:", grid_search.best_params_)

# ============================================
# 7. EVALUATE ON THE TEST SET
# ============================================

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)
y_score = best_model.predict_proba(X_test)[:, 1]  # probability of "satisfied"

print(f"\nAccuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1-score:  {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_score):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred,
                            target_names=["Neutral/Dissatisfied", "Satisfied"]))

# ============================================
# 8. CONFUSION MATRIX PLOT
# ============================================

fig, ax = plt.subplots(figsize=(6.5, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=["Neutral/Dissatisfied", "Satisfied"],
    ax=ax,
    colorbar=False
)
ax.set_title("Decision Tree — Confusion Matrix")
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
plt.tight_layout()
plt.savefig("decision_tree_confusion_matrix.png", dpi=150)
plt.show()