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


