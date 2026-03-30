# ============================================
# LOGISTIC REGRESSION
# DATA 221 PROJECT
# AIRLINE PASSENGER SATISFACTION
# ============================================

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    classification_report

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
# 5. MODEL PIPELINE & HYPERPARAMETER TUNING
# ============================================

# Create the base pipeline
pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("logreg", LogisticRegression(max_iter=1000, solver='lbfgs'))
])

# Define the grid of C values to test
# 0.01 is strong regularization, 10 is weak
param_grid = {
    'logreg__C': [0.01, 0.1, 1, 10, 100]
}

# Set up Grid Search
# cv=5 means 5-fold cross-validation
grid_search = GridSearchCV(
    pipeline, 
    param_grid, 
    cv=5, 
    scoring='accuracy', 
    n_jobs=-1
)

# ============================================
# 6. TRAIN
# ============================================

print("Starting Hyperparameter Tuning...")
grid_search.fit(X_train, y_train)

# Best parameter and best model
print(f"Best C value found: {grid_search.best_params_['logreg__C']}")
best_model = grid_search.best_estimator_

# ============================================
# 7. PREDICT + EVALUATE
# ============================================

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n=== LOGISTIC REGRESSION RESULTS ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
