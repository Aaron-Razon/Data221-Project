# DATA 221 PROJECT: RANDOM FOREST MODEL
# AIRLINE PASSENGER SATISFACTION

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
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

# Build The Random Forest Pipeline

random_forest_pipeline = Pipeline(steps=[
    ("preprocess", full_preprocessor),
    ("model", RandomForestClassifier(random_state=42))
])

# Hyperparameter Tuning

random_forest_parameter_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [10, 20, None],
    "model__min_samples_split": [2, 5]
}

random_forest_grid_search = GridSearchCV(
    estimator=random_forest_pipeline,
    param_grid=random_forest_parameter_grid,
    scoring="f1",
    cv=5,
    n_jobs=-1,
    refit=True
)

print("\nTraining Random Forest Model...")
random_forest_grid_search.fit(training_feature_matrix_X, training_target_vector_y)

best_random_forest_model = random_forest_grid_search.best_estimator_

print("\nBest Parameters:")
print(random_forest_grid_search.best_params_)

# Make Predictions

predicted_test_labels = best_random_forest_model.predict(testing_feature_matrix_X)
predicted_test_probabilities = best_random_forest_model.predict_proba(testing_feature_matrix_X)[:, 1]

# Evaluate The Model

random_forest_accuracy = accuracy_score(testing_target_vector_y, predicted_test_labels)
random_forest_precision = precision_score(testing_target_vector_y, predicted_test_labels)
random_forest_recall = recall_score(testing_target_vector_y, predicted_test_labels)
random_forest_f1_score = f1_score(testing_target_vector_y, predicted_test_labels)
random_forest_roc_auc = roc_auc_score(testing_target_vector_y, predicted_test_probabilities)

print("\n=== RANDOM FOREST RESULTS ===")
print(f"Accuracy:  {random_forest_accuracy:.4f}")
print(f"Precision: {random_forest_precision:.4f}")
print(f"Recall:    {random_forest_recall:.4f}")
print(f"F1-score:  {random_forest_f1_score:.4f}")
print(f"ROC-AUC:   {random_forest_roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(testing_target_vector_y, predicted_test_labels))

print("Confusion Matrix:")
print(confusion_matrix(testing_target_vector_y, predicted_test_labels))

# Display And Save Confusion Matrix

with PdfPages("random_forest_confusion_matrix.pdf") as pdf:
    figure_object, axis_object = plt.subplots(figsize=(6, 5))

    ConfusionMatrixDisplay.from_predictions(
        testing_target_vector_y,
        predicted_test_labels,
        display_labels=["Neutral/Dissatisfied", "Satisfied"],
        ax = axis_object,
        colorbar=False
    )

    axis_object.set_title("Random Forest Confusion Matrix")
    axis_object.set_xlabel("Predicted Label")
    axis_object.set_ylabel("True Label")

    plt.tight_layout()
    pdf.savefig(figure_object)
    plt.show()
    plt.close(figure_object)