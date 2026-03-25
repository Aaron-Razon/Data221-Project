# DATA 221 PROJECT: RANDOM FOREST MODEL
# AIRLINE PASSENGER SATISFACTION

import pandas as pd

# Load the Data

training_dataframe = pd.read_csv("train.csv")
testing_dataframe = pd.read_csv("test.csv")

print("Training Data Shape:", training_dataframe.shape)
print("Testing Data Shape:", testing_dataframe.shape)