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