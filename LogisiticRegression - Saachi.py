# ============================================
# LOGISTIC REGRESSION
# ============================================

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ============================================
# 1. LOAD DATA
# ============================================

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Drop unnecessary columns
for col in ["id", "Unnamed: 0"]:
    if col in train_df.columns:
        train_df = train_df.drop(columns=col)
    if col in test_df.columns:
        test_df = test_df.drop(columns=col)
