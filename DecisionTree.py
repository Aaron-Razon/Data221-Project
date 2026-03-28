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
# Some columns in the spreadsheet are just row numbers — they exist to identify a
# row but don't tell us anything useful about passenger satisfaction.
# These get deleted first so the model doesn't accidentally try to learn from meaningless information.
for col in ["id", "Unnamed: 0"]:
    train_df = train_df.drop(columns=col, errors="ignore")
    test_df = test_df.drop(columns=col, errors="ignore")

# Encode the target: 1 = satisfied, 0 = neutral or dissatisfied
# The satisfaction column contains the words "satisfied" or "neutral or dissatisfied",
# but the model work with numbers, not words.
# So we swap them out — satisfied becomes 1, everything else becomes 0.
# Same information, just in a format the model can work with.
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

# The passenger data has two kinds of columns. Some are numbers (like age, flight distance, or a 1–5 rating for
# cleanliness).
# Others are text/categories (like travel class being "Business" or "Economy", or gender being "Male" or "Female").
# The code sorts them into two separate groups here because each group needs to be handled differently in the next step.
numeric_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = X_train.select_dtypes(exclude=["number"]).columns.tolist()

print("Numeric features:", numeric_cols)
print("Categorical features:", categorical_cols)

# ============================================
# 4. PREPROCESSING
# Decision Trees don't need scaling, just
# imputation (filling missing values) and
# one-hot encoding for categorical columns.
# ============================================

# Converting text into numbers:
# The model can't read words like "Business Class" or "Female" directly. So the code converts each category into a
# series of 0s and 1s. For example, the "Travel Class" column might become three new columns — is_Business, is_Economy,
# is_Eco_Plus — where only one of them is 1 for each passenger. This is called one-hot encoding.

# normally with machine learning you'd also rescale numbers so they're all on the same scale. But Decision Trees
# don't need that — they make decisions based on thresholds ("is age > 40?"), not on the actual size of numbers,
# so it doesn't matter if one column goes from 1–5 and another goes from 0–5000.
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

# A pipeline is somewhat like a factory assembly line: raw data goes in one end, a prediction comes out the other, and
# the steps always happen in the same order.
pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", DecisionTreeClassifier(random_state=42))
])

# ============================================
# 6. HYPERPARAMETER TUNING WITH CROSS-VALIDATION
# max_depth — how many levels deep the tree can grow
# min_samples_split — minimum samples required to split a node
# ============================================

# Decision Trees have a problem: if you let them grow unchecked, they'll memorise the training data perfectly but fail
# on new data. It's like a student who memorises every answer in the textbook but can't handle a slightly different
# question on the exam. This is called overfitting.
# To prevent this, two settings (called hyperparameters) are adjusted
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

# Two outputs are generated — the hard class prediction (y_pred) and the probability of being "satisfied" (y_score),
# needed for the ROC-AUC metric.
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

# This creates a simple grid that shows exactly where the model got things right and wrong
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