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