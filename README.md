# Airline Passenger Satisfaction Classification Project

## Project Overview

This project uses machine learning to predict whether an airline passenger is **satisfied** or **neutral or dissatisfied** based on customer information, travel details, service ratings, and delay-related features.

Each group member worked on one model in a separate Python file. These individual model files were then combined into `model_comparison.py`, which was used to compare all four models in one program.

The dataset used for this project is the **Airline Passenger Satisfaction** dataset from Kaggle.

## Dataset

- **Source:** Kaggle Airline Passenger Satisfaction Dataset  
- **Link:** https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction

### Target Variable

- `1` = satisfied
- `0` = neutral or dissatisfied

The provided training and testing split was used so that all models were evaluated fairly.

## Models Used

The following machine learning models were implemented and compared:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest

## What the Program Does

The final comparison program:

- loads the training and testing datasets
- cleans the data and encodes the target labels
- identifies numeric and categorical features
- preprocesses the data using imputing, encoding, and scaling when needed
- trains and tunes four machine learning models
- evaluates each model on the test set
- compares the final results across all models
- saves the final results table and confusion matrix outputs

## Evaluation Metrics

The models were compared using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Confusion matrices were also generated to help visualize model performance.

## Repository Contents

### Python Files

- `decision_tree_andrew.py`  
  Andrew's Decision Tree model

- `knn_nolan.py`  
  Nolan's K-Nearest Neighbors model

- `logistic_regression_saachi.py`  
  Saachi's Logistic Regression model

- `random_forest_aaron.py`  
  Aaron's Random Forest model

- `model_comparison.py`  
  Final file that combines all four models for training, tuning, evaluation, and comparison

### Dataset Files

- `train.csv`  
  Training dataset

- `test.csv`  
  Testing dataset

### Output Files

- `model_comparison_results.csv`  
  Final results table for all models

- `confusion_matrices.pdf`  
  Confusion matrices from the final comparison program

## Contributions

- **Aaron Razon** — Random Forest model and project integration
- **Nolan Litvak** — K-Nearest Neighbors model
- **Andrew Fenning** — Decision Tree model
- **Saachi Gupta** — Logistic Regression model

## Requirements

This project was completed in Python using:

- pandas
- numpy
- matplotlib
- scikit-learn

Install the required libraries with:

```bash
pip install pandas numpy matplotlib scikit-learn
```

## How to Run

1. Make sure all project files are in the same folder.
2. Make sure `train.csv` and `test.csv` are included in the folder.
3. Run `model_comparison.py`.
4. Check the final outputs:
   - `model_comparison_results.csv`
   - `confusion_matrices.pdf`

The individual model files can also be run separately.

## Notes

- All models were tested using the same train-test split for fairness.
- Preprocessing was done inside each model pipeline to help avoid data leakage.
- Each individual model file can produce its own outputs, such as confusion matrices or other displays.
- These extra individual output files were not included in the repository to keep the project cleaner and less cluttered.
- The repository includes the final combined outputs from `model_comparison.py`.

## Purpose of the Project

The purpose of this project is to compare different machine learning models on a real classification problem and determine which model performs best for predicting airline passenger satisfaction.

This project also demonstrates important data science steps such as data cleaning, preprocessing, model training, tuning, evaluation, and comparison.

## Authors

- Aaron Razon
- Nolan Litvak
- Andrew Fenning
- Saachi Gupta
