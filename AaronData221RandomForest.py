# DATA 221 PROJECT: RANDOM FOREST MODEL
# AIRLINE PASSENGER SATISFACTION

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Load the Data

training_dataframe = pd.read_csv("train.csv")
testing_dataframe = pd.read_csv("test.csv")

print("Training Data Shape:", training_dataframe.shape)
print("Testing Data Shape:", testing_dataframe.shape)

# Basic Cleanup

columns_to_drop_if_present = ["id", "Unnamed: 0"]

for column_name in columns_to_drop_if_present:
    if column_name in training_dataframe.columns:
        training_dataframe = training_dataframe.drop(columns=[column_name])
    if column_name in testing_dataframe.columns:
        testing_dataframe = testing_dataframe.drop(columns=[column_name])

target_column_name = "satisfaction"

target_label_mapping = {
    "satisfied": 1,
    "neutral or dissatisfied": 0
}

training_dataframe[target_column_name] = training_dataframe[target_column_name].map(target_label_mapping)
testing_dataframe[target_column_name] = testing_dataframe[target_column_name].map(target_label_mapping)

# Split Features and Target

training_feature_matrix_X = training_dataframe.drop(columns=[target_column_name])
training_target_vector_y = training_dataframe[target_column_name]

testing_feature_matrix_X = testing_dataframe.drop(columns=[target_column_name])
testing_target_vector_y = testing_dataframe[target_column_name]

# Identify Numeric and Categorical Features

numeric_feature_names = training_feature_matrix_X.select_dtypes(include=["number"]).columns.tolist()
categorical_feature_names = training_feature_matrix_X.select_dtypes(exclude=["number"]).columns.tolist()

print("Numeric Features:", numeric_feature_names)
print("Categorical Features:", categorical_feature_names)

# Preprocessing Pipeline

numeric_preprocessing_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_preprocessing_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

full_preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_preprocessing_pipeline, numeric_feature_names),
        ("cat", categorical_preprocessing_pipeline, categorical_feature_names)
    ]
)