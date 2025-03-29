import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(filepath):
    """Load dataset and perform preprocessing steps."""

    # Load dataset
    Boston_house = pd.read_csv(filepath)

    # Handle missing values (numerical: median, categorical: mode)
    for col in Boston_house.select_dtypes(include=np.number):
        Boston_house[col].fillna(Boston_house[col].median(), inplace=True)
    
    for col in Boston_house.select_dtypes(exclude=np.number):
        Boston_house[col].fillna(Boston_house[col].mode()[0], inplace=True)


    # One-hot encode categorical column "rad"
    encoded_data = pd.get_dummies(Boston_house, columns=["rad"], dtype=int)

    # Define features (X) and target variable (y)
    X = encoded_data.drop(columns=['medv'])
    y = encoded_data['medv']

    # Split data into training (80%) and testing (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrame
    X_train = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test = pd.DataFrame(X_test_scaled, columns=X.columns)

    return X_train, X_test, y_train, y_test

# Example usage
if __name__ == "__main__":
    filepath = r"C:\Users\hp\OMDENA\machine-learning-linear-regression-1962vickyrena\BostonHousing.csv"
    X_train, X_test, y_train, y_test = load_and_preprocess_data(filepath)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
