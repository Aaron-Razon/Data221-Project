# ============================================
# DATA 221 PROJECT: RANDOM FOREST MODEL
# AIRLINE PASSENGER SATISFACTION
# ============================================

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

# ============================================
# 3. SPLIT FEATURES AND TARGET
# ============================================

training_feature_matrix_X = training_dataframe.drop(columns=[target_column_name])
training_target_vector_y = training_dataframe[target_column_name]

testing_feature_matrix_X = testing_dataframe.drop(columns=[target_column_name])
testing_target_vector_y = testing_dataframe[target_column_name]

# ============================================
# 4. IDENTIFY NUMERIC AND CATEGORICAL FEATURES
# ============================================

numeric_feature_names = training_feature_matrix_X.select_dtypes(include=["number"]).columns.tolist()
categorical_feature_names = training_feature_matrix_X.select_dtypes(exclude=["number"]).columns.tolist()

print("\nNumeric Features:", numeric_feature_names)
print("\nCategorical Features:", categorical_feature_names)

# ============================================
# 5. PREPROCESSING PIPELINE
# ============================================

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

# ============================================
# 6. BUILD THE RANDOM FOREST PIPELINE
# ============================================

random_forest_pipeline = Pipeline(steps=[
    ("preprocess", full_preprocessor),
    ("model", RandomForestClassifier(random_state=42))
])

# ============================================
# 7. HYPERPARAMETER TUNING
# ============================================

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

# ============================================
# 8. MAKE PREDICTIONS
# ============================================

predicted_test_labels = best_random_forest_model.predict(testing_feature_matrix_X)
predicted_test_probabilities = best_random_forest_model.predict_proba(testing_feature_matrix_X)[:, 1]

# ============================================
# 9. EVALUATE THE MODEL
# ============================================

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

# ============================================
# 10. FEATURE IMPORTANCE OUTPUT
# ============================================

fitted_preprocessor = best_random_forest_model.named_steps["preprocess"]
fitted_random_forest_model = best_random_forest_model.named_steps["model"]

processed_feature_names = fitted_preprocessor.get_feature_names_out()
feature_importance_scores = fitted_random_forest_model.feature_importances_

feature_importance_dataframe = pd.DataFrame({
    "Feature": processed_feature_names,
    "Importance": feature_importance_scores
})

feature_importance_dataframe["Feature"] = (
    feature_importance_dataframe["Feature"]
    .str.replace("num__", "", regex=False)
    .str.replace("cat__", "", regex=False)
    .str.replace("_", " = ", regex=False)
)

feature_importance_dataframe["Importance"] = feature_importance_dataframe["Importance"].round(4)

feature_importance_dataframe = feature_importance_dataframe.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Most Important Features:")
print(feature_importance_dataframe.head(10).to_string(index=False))

feature_importance_dataframe.to_csv("random_forest_feature_importance.csv", index=False)

# Create a bar chart for the top 10 most important features
top_10_feature_importance_dataframe = feature_importance_dataframe.head(10).copy()

# Place the smallest importance at the bottom and the largest at the top
top_10_feature_importance_dataframe = top_10_feature_importance_dataframe.sort_values(
    by="Importance",
    ascending=True
)

figure_object, axis_object = plt.subplots(figsize=(8, 6))

axis_object.barh(
    top_10_feature_importance_dataframe["Feature"],
    top_10_feature_importance_dataframe["Importance"]
)

axis_object.set_title("Top 10 Random Forest Feature Importances")
axis_object.set_xlabel("Importance")
axis_object.set_ylabel("Feature")

plt.tight_layout()
plt.savefig("random_forest_feature_importance.pdf")
plt.show()
plt.close(figure_object)

# ============================================
# 11. DISPLAY AND SAVE CONFUSION MATRIX
# ============================================

with PdfPages("random_forest_confusion_matrix.pdf") as pdf:
    figure_object, axis_object = plt.subplots(figsize=(6.5, 5))

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

# ============================================
# 12. SAVE RESULTS TO A CSV FILE
# ============================================

random_forest_results_dataframe = pd.DataFrame([
    {
        "Model": "Random Forest",
        "Best Parameters": str(random_forest_grid_search.best_params_),
        "Accuracy": random_forest_accuracy,
        "Precision": random_forest_precision,
        "Recall": random_forest_recall,
        "F1-score": random_forest_f1_score,
        "ROC-AUC": random_forest_roc_auc
    }
])

random_forest_results_dataframe.to_csv("random_forest_results.csv", index=False)

print("\nSaved Files:")
print("- random_forest_confusion_matrix.pdf")
print("- random_forest_results.csv")
print("- random_forest_feature_importance.csv")
print("- random_forest_feature_importance.pdf")