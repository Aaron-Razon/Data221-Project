# ================================================
# DATA 221 PROJECT: AIRLINE PASSENGER SATISFACTION
# MODEL COMPARISON
# ================================================

import pandas as pd

# ============================================
# 1. LOAD THE DATA
# ============================================

training_dataframe = pd.read_csv("train.csv")
testing_dataframe = pd.read_csv("test.csv")

print("\nTraining Data Shape:", training_dataframe.shape)
print("Testing Data Shape:", testing_dataframe.shape)
