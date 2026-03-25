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

# ============================================
# 2. TARGET PROCESSING
# ============================================

train_df["satisfaction"] = train_df["satisfaction"].map({
    "satisfied": 1,
    "neutral or dissatisfied": 0
})

test_df["satisfaction"] = test_df["satisfaction"].map({
    "satisfied": 1,
    "neutral or dissatisfied": 0
})

X_train = train_df.drop(columns="satisfaction")
y_train = train_df["satisfaction"]

X_test = test_df.drop(columns="satisfaction")
y_test = test_df["satisfaction"]

# ============================================
# 3. FEATURE TYPES
# ============================================

num_features = X_train.select_dtypes(include=["number"]).columns
cat_features = X_train.select_dtypes(exclude=["number"]).columns

# ============================================
# 4. PREPROCESSING
# ============================================

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_features),
    ("cat", cat_pipeline, cat_features)
])
# ============================================
# 5. MODEL PIPELINE
# ============================================

model = Pipeline([
    ("preprocess", preprocessor),
    ("logreg", LogisticRegression(max_iter=1000))
])


# ============================================
# 6. TRAIN
# ============================================

model.fit(X_train, y_train)
