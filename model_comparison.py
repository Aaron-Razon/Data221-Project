# ================================================
# DATA 221 PROJECT: AIRLINE PASSENGER SATISFACTION
# MODEL COMPARISON
# ================================================

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# ============================================
# 1. LOAD THE DATA
# ============================================

training_dataframe = pd.read_csv("train.csv")
testing_dataframe = pd.read_csv("test.csv")

print("\nTraining Data Shape:", training_dataframe.shape)
print("Testing Data Shape:", testing_dataframe.shape)

# ============================================
# 2. BASIC CLEANUP
# ============================================

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

training_feature_matrix_X = training_dataframe.drop(columns=[target_column_name])
training_target_vector_y = training_dataframe[target_column_name]

testing_feature_matrix_X = testing_dataframe.drop(columns=[target_column_name])
testing_target_vector_y = testing_dataframe[target_column_name]

# ============================================
# 3. IDENTIFY NUMERIC AND CATEGORICAL FEATURES
# ============================================

numeric_feature_names = training_feature_matrix_X.select_dtypes(include=["number"]).columns.tolist()
categorical_feature_names = training_feature_matrix_X.select_dtypes(exclude=["number"]).columns.tolist()

print("\nNumeric Features:", numeric_feature_names)
print("\nCategorical Features:", categorical_feature_names)

# ============================================
# 4. PREPROCESSING PIPELINES
# ============================================

scaled_numeric_preprocessing_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

unscaled_numeric_preprocessing_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_preprocessing_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

scaled_full_preprocessor = ColumnTransformer(
    transformers=[
        ("num", scaled_numeric_preprocessing_pipeline, numeric_feature_names),
        ("cat", categorical_preprocessing_pipeline, categorical_feature_names)
    ]
)

unscaled_full_preprocessor = ColumnTransformer(
    transformers=[
        ("num", unscaled_numeric_preprocessing_pipeline, numeric_feature_names),
        ("cat", categorical_preprocessing_pipeline, categorical_feature_names)
    ]
)

# ============================================
# 5. MODEL CONFIGURATIONS
# ============================================

model_configurations = {
    "Logistic Regression": {
        "preprocessor": scaled_full_preprocessor,
        "model": LogisticRegression(max_iter=1000, random_state=42),
        "param_grid": {
            "model__C": [0.1, 1.0, 10.0]
        }
    },
    "KNN": {
        "preprocessor": scaled_full_preprocessor,
        "model": KNeighborsClassifier(),
        "param_grid": {
            "model__n_neighbors": [3, 5, 7, 9],
            "model__weights": ["uniform", "distance"]
        }
    },
    "Decision Tree": {
        "preprocessor": unscaled_full_preprocessor,
        "model": DecisionTreeClassifier(random_state=42),
        "param_grid": {
            "model__max_depth": [5, 10, 15, None],
            "model__min_samples_split": [2, 5, 10]
        }
    },
    "Random Forest": {
        "preprocessor": unscaled_full_preprocessor,
        "model": RandomForestClassifier(random_state=42),
        "param_grid": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [10, 20, None],
            "model__min_samples_split": [2, 5]
        }
    }
}