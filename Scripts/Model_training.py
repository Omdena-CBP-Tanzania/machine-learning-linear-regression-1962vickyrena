import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Load preprocessed data
X_train = pd.read_csv(r"C:\Users\hp\OMDENA\machine-learning-linear-regression-1962vickyrena\X_train.csv")
X_test = pd.read_csv(r"C:\Users\hp\OMDENA\machine-learning-linear-regression-1962vickyrena\X_test.csv")
y_train = pd.read_csv(r"C:\Users\hp\OMDENA\machine-learning-linear-regression-1962vickyrena\y_train.csv")
y_test = pd.read_csv(r"C:\Users\hp\OMDENA\machine-learning-linear-regression-1962vickyrena\y_test.csv")

# Ensure target variable is 1D
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)


