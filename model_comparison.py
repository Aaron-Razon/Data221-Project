# ================================================
# DATA 221 PROJECT: AIRLINE PASSENGER SATISFACTION
# MODEL COMPARISON
# ================================================

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
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
# 0. PROGRAM START MESSAGE
# ============================================

pd.options.display.float_format = "{:.4f}".format

print("=" * 60)
print("AIRLINE PASSENGER SATISFACTION MODEL COMPARISON STARTING...")
print("=" * 60)

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

# ============================================
# 6. TRAIN, TUNE, AND EVALUATE
# ============================================

all_model_results = []
all_classification_reports = []
best_fitted_models = {}

with PdfPages("confusion_matrices.pdf") as confusion_matrix_pdf:
    for model_name, model_configuration in model_configurations.items():
        print(f"\n{'=' * 60}")
        print(f"TRAINING {model_name}")
        print(f"{'=' * 60}")

        full_model_pipeline = Pipeline(steps=[
            ("preprocess", model_configuration["preprocessor"]),
            ("model", model_configuration["model"])
        ])

        model_grid_search = GridSearchCV(
            estimator=full_model_pipeline,
            param_grid=model_configuration["param_grid"],
            scoring="f1",
            cv=5,
            n_jobs=-1,
            refit=True
        )

        model_grid_search.fit(training_feature_matrix_X, training_target_vector_y)

        best_fitted_model = model_grid_search.best_estimator_
        best_fitted_models[model_name] = best_fitted_model

        predicted_test_labels = best_fitted_model.predict(testing_feature_matrix_X)

        if hasattr(best_fitted_model, "predict_proba"):
            predicted_test_scores = best_fitted_model.predict_proba(testing_feature_matrix_X)[:, 1]
        else:
            predicted_test_scores = best_fitted_model.decision_function(testing_feature_matrix_X)

        model_accuracy = accuracy_score(testing_target_vector_y, predicted_test_labels)
        model_precision = precision_score(testing_target_vector_y, predicted_test_labels)
        model_recall = recall_score(testing_target_vector_y, predicted_test_labels)
        model_f1_score = f1_score(testing_target_vector_y, predicted_test_labels)
        model_roc_auc = roc_auc_score(testing_target_vector_y, predicted_test_scores)

        model_result_dictionary = {
            "Model": model_name,
            "Best Parameters": str(model_grid_search.best_params_),
            "Accuracy": model_accuracy,
            "Precision": model_precision,
            "Recall": model_recall,
            "F1-score": model_f1_score,
            "ROC-AUC": model_roc_auc
        }

        all_model_results.append(model_result_dictionary)

        print("Best Parameters:", model_grid_search.best_params_)
        print(f"Accuracy:  {model_accuracy:.4f}")
        print(f"Precision: {model_precision:.4f}")
        print(f"Recall:    {model_recall:.4f}")
        print(f"F1-score:  {model_f1_score:.4f}")
        print(f"ROC-AUC:   {model_roc_auc:.4f}")

        current_classification_report = classification_report(
            testing_target_vector_y,
            predicted_test_labels,
            target_names=["Neutral/Dissatisfied", "Satisfied"],
            zero_division=0
        )

        print("\nClassification Report:")
        print(current_classification_report)

        print("Confusion Matrix:")
        print(confusion_matrix(testing_target_vector_y, predicted_test_labels))

        all_classification_reports.append(
            f"{'=' * 70}\n"
            f"{model_name}\n"
            f"{'=' * 70}\n"
            f"Best Parameters: {model_grid_search.best_params_}\n\n"
            f"{current_classification_report}\n"
        )

        figure_object, axis_object = plt.subplots(figsize=(6.5, 5))

        ConfusionMatrixDisplay.from_predictions(
            testing_target_vector_y,
            predicted_test_labels,
            display_labels=["Neutral/Dissatisfied", "Satisfied"],
            ax=axis_object,
            colorbar=False
        )

        axis_object.set_title(f"{model_name} Confusion Matrix")
        axis_object.set_xlabel("Predicted Label")
        axis_object.set_ylabel("True Label")

        plt.tight_layout()
        confusion_matrix_pdf.savefig(figure_object)
        plt.show()
        plt.close(figure_object)

# ============================================
# 7. SAVE CLASSIFICATION REPORTS
# ============================================

with open("classification_reports.txt", "w", encoding="utf-8") as report_file:
    report_file.write("\n".join(all_classification_reports))

# ============================================
# 8. RESULTS TABLE
# ============================================

results_dataframe = pd.DataFrame(all_model_results)
results_dataframe = results_dataframe.sort_values(by="F1-score", ascending=False).reset_index(drop=True)

rounded_results_dataframe = results_dataframe.copy()
rounded_results_dataframe["Accuracy"] = rounded_results_dataframe["Accuracy"].round(4)
rounded_results_dataframe["Precision"] = rounded_results_dataframe["Precision"].round(4)
rounded_results_dataframe["Recall"] = rounded_results_dataframe["Recall"].round(4)
rounded_results_dataframe["F1-score"] = rounded_results_dataframe["F1-score"].round(4)
rounded_results_dataframe["ROC-AUC"] = rounded_results_dataframe["ROC-AUC"].round(4)

rounded_results_dataframe.to_csv("model_comparison_results.csv", index=False, float_format="%.4f")

print("\nFINAL MODEL COMPARISON")
print(rounded_results_dataframe.to_string(index=False))

best_model_name = results_dataframe.iloc[0]["Model"]
best_model_f1_score = results_dataframe.iloc[0]["F1-score"]

print("\nBest Overall Model:")
print(f"{best_model_name} (F1-score = {best_model_f1_score:.4f})")

# ============================================
# 9. LOGISTIC REGRESSION COEFFICIENT OUTPUT
# ============================================

best_logistic_regression_model = best_fitted_models["Logistic Regression"]

fitted_logistic_preprocessor = best_logistic_regression_model.named_steps["preprocess"]
fitted_logistic_regression_model = best_logistic_regression_model.named_steps["model"]

logistic_processed_feature_names = fitted_logistic_preprocessor.get_feature_names_out()
logistic_coefficient_values = fitted_logistic_regression_model.coef_[0]

logistic_coefficient_dataframe = pd.DataFrame({
    "Feature": logistic_processed_feature_names,
    "Coefficient": logistic_coefficient_values
})

logistic_coefficient_dataframe["Feature"] = (
    logistic_coefficient_dataframe["Feature"]
    .str.replace("num__", "", regex=False)
    .str.replace("cat__", "", regex=False)
    .str.replace("_", " = ", regex=False)
)

logistic_coefficient_dataframe["Absolute Coefficient"] = logistic_coefficient_dataframe["Coefficient"].abs()
logistic_coefficient_dataframe = logistic_coefficient_dataframe.sort_values(
    by="Absolute Coefficient",
    ascending=False
)

rounded_logistic_coefficient_dataframe = logistic_coefficient_dataframe.copy()

rounded_logistic_coefficient_dataframe["Coefficient"] = (
    rounded_logistic_coefficient_dataframe["Coefficient"].round(4))

rounded_logistic_coefficient_dataframe["Absolute Coefficient"] = (
    rounded_logistic_coefficient_dataframe["Absolute Coefficient"].round(4))

print("\nTop 10 Logistic Regression Coefficients:")
print(
    rounded_logistic_coefficient_dataframe[["Feature", "Coefficient"]]
    .head(10)
    .to_string(index=False)
)

rounded_logistic_coefficient_dataframe.to_csv(
    "logistic_regression_coefficients.csv", index=False, float_format="%.4f")

top_10_logistic_coefficient_dataframe = (
    logistic_coefficient_dataframe[["Feature", "Coefficient"]]
    .head(10)
    .copy()
)

top_10_logistic_coefficient_dataframe = top_10_logistic_coefficient_dataframe.sort_values(
    by="Coefficient",
    ascending=True
)

figure_object, axis_object = plt.subplots(figsize=(8, 6))

axis_object.barh(
    top_10_logistic_coefficient_dataframe["Feature"],
    top_10_logistic_coefficient_dataframe["Coefficient"]
)

axis_object.set_title("Top 10 Logistic Regression Coefficients")
axis_object.set_xlabel("Coefficient Value")
axis_object.set_ylabel("Feature")

plt.tight_layout()
plt.savefig("logistic_regression_coefficients.pdf")
plt.show()
plt.close(figure_object)

# ============================================
# 10. FEATURE IMPORTANCE FOR TREE-BASED MODELS
# ============================================

with PdfPages("feature_importance_charts.pdf") as feature_importance_pdf:
    for model_name in ["Decision Tree", "Random Forest"]:
        best_tree_model = best_fitted_models[model_name]

        fitted_tree_preprocessor = best_tree_model.named_steps["preprocess"]
        fitted_tree_classifier = best_tree_model.named_steps["model"]

        tree_processed_feature_names = fitted_tree_preprocessor.get_feature_names_out()
        tree_feature_importance_scores = fitted_tree_classifier.feature_importances_

        tree_feature_importance_dataframe = pd.DataFrame({
            "Feature": tree_processed_feature_names,
            "Importance": tree_feature_importance_scores
        })

        tree_feature_importance_dataframe["Feature"] = (
            tree_feature_importance_dataframe["Feature"]
            .str.replace("num__", "", regex=False)
            .str.replace("cat__", "", regex=False)
            .str.replace("_", " = ", regex=False)
        )

        tree_feature_importance_dataframe = tree_feature_importance_dataframe.sort_values(
            by="Importance",
            ascending=False
        )

        rounded_tree_feature_importance_dataframe = tree_feature_importance_dataframe.copy()
        rounded_tree_feature_importance_dataframe["Importance"] = (
            rounded_tree_feature_importance_dataframe["Importance"].round(4)
        )

        print(f"\nTop 10 Features for {model_name}:")
        print(rounded_tree_feature_importance_dataframe.head(10).to_string(index=False))

        output_file_name = model_name.lower().replace(" ", "_") + "_feature_importance.csv"

        rounded_tree_feature_importance_dataframe.to_csv(output_file_name, index=False, float_format="%.4f")

        top_10_tree_feature_importance_dataframe = tree_feature_importance_dataframe.head(10).copy()
        top_10_tree_feature_importance_dataframe = top_10_tree_feature_importance_dataframe.sort_values(
            by="Importance",
            ascending=True
        )

        figure_object, axis_object = plt.subplots(figsize=(8, 6))

        axis_object.barh(
            top_10_tree_feature_importance_dataframe["Feature"],
            top_10_tree_feature_importance_dataframe["Importance"]
        )

        axis_object.set_title(f"Top 10 {model_name} Feature Importances")
        axis_object.set_xlabel("Importance")
        axis_object.set_ylabel("Feature")

        plt.tight_layout()
        feature_importance_pdf.savefig(figure_object)
        plt.show()
        plt.close(figure_object)

# ============================================
# 11. FINAL MODEL COMPARISON BAR CHARTS
# ============================================

metrics_for_bar_charts = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]

with PdfPages("model_comparison_bar_charts.pdf") as model_comparison_bar_chart_pdf:
    for metric_name in metrics_for_bar_charts:
        chart_dataframe = results_dataframe.sort_values(by=metric_name, ascending=True)

        figure_object, axis_object = plt.subplots(figsize=(8, 5))

        axis_object.barh(
            chart_dataframe["Model"],
            chart_dataframe[metric_name]
        )

        axis_object.set_title(f"Model Comparison by {metric_name}")
        axis_object.set_xlabel(metric_name)
        axis_object.set_ylabel("Model")

        plt.tight_layout()
        model_comparison_bar_chart_pdf.savefig(figure_object)
        plt.show()
        plt.close(figure_object)

# ============================================
# 12. SAVED FILES
# ============================================

print("\nSaved Files:")
print("- confusion_matrices.pdf")
print("- classification_reports.txt")
print("- model_comparison_results.csv")
print("- logistic_regression_coefficients.csv")
print("- logistic_regression_coefficients.pdf")
print("- decision_tree_feature_importance.csv")
print("- random_forest_feature_importance.csv")
print("- feature_importance_charts.pdf")
print("- model_comparison_bar_charts.pdf")