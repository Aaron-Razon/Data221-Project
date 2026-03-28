from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

train_df = pd.read_csv(r"C:\Users\nolan\OneDrive\Code\train.csv")
test_df = pd.read_csv(r"C:\Users\nolan\OneDrive\Code\test.csv")

print("Train Shape:", train_df.shape)
print("Test Shape:", test_df.shape)

columns_to_drop_if_present = ["id", "Unnamed: 0"]

for column_name in columns_to_drop_if_present:
    if column_name in train_df.columns:
        train_df = train_df.drop(columns=column_name)
    if column_name in test_df.columns:
        test_df = test_df.drop(columns=column_name)

target_column_name = "satisfaction"

target_label_mapping = {
    "satisfied": 1,
    "neutral or dissatisfied": 0
}

train_df[target_column_name] = train_df[target_column_name].map(target_label_mapping)
test_df[target_column_name] = test_df[target_column_name].map(target_label_mapping)

X_train = train_df.drop(columns=target_column_name)
y_train = train_df[target_column_name]

X_test = test_df.drop(columns=target_column_name)
y_test = test_df[target_column_name]

numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X_train.select_dtypes(include=["object"]).columns

numeric_pipeline = Pipeline([
    # "scaler" standardizes numerical values so that they are on the same scale for measuring knn distance.
    # "imputer" replaces missing values with the mean, are replaces missing categories with most common.
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    # "onehot" converts categories into binary numbers 0 or 1.
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    # "num" applies the numeric pipeline to the numeric columns in the dataset.
    # "cat" applies the categorical pipeline to the categorical columns in the dataset.
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

model = Pipeline([
    # "preprocessor" runs both pipelines.
    # "knn" is the learning model training by looking at a specified amount of data-points.
    # The model will predict on the majority of specified points (satisfied, neutral or not satisfied).
    ("preprocessor", preprocessor),
    ("knn", KNeighborsClassifier(n_neighbors=5))
])