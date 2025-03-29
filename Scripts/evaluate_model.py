"""
Model Evaluation Script
----------------------
This script evaluates a trained linear regression model by:
1. Loading model and test data
2. Calculating performance metrics (MAE, MSE, RMSE, R²)
3. Generating diagnostic plots (residual analysis)
4. Comparing different feature sets
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import joblib
import os
import warnings

# Configuration
warnings.filterwarnings("ignore")
plt.style.use('seaborn')
pd.set_option('display.float_format', '{:.4f}'.format)

# Constants
MODEL_PATH = "../models/trained_model.pkl"
DATA_PATHS = {
    "X_train": "../data/processed/X_train.csv",
    "X_test": "../data/processed/X_test.csv",
    "y_train": "../data/processed/y_train.csv",
    "y_test": "../data/processed/y_test.csv"
}

def load_data():
    """Load and prepare datasets"""
    print("\n🔍 Loading data...")
    data = {}
    for name, path in DATA_PATHS.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found: {path}")
        data[name] = pd.read_csv(path)
        print(f"Loaded {name}: {data[name].shape}")

    # Ensure target variables are 1D arrays
    data['y_train'] = data['y_train'].values.ravel()
    data['y_test'] = data['y_test'].values.ravel()
    
    return data['X_train'], data['X_test'], data['y_train'], data['y_test']

def load_model():
    """Load trained model"""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    
    print("\n🤖 Loading model...")
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully!")
    return model

def evaluate_model(model, X_test, y_test):
    """Calculate and display evaluation metrics"""
    print("\n📊 Model Evaluation Metrics:")
    
    y_pred = model.predict(X_test)
    
    metrics = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R²': r2_score(y_test, y_pred)
    }
    
    for name, value in metrics.items():
        print(f"{name:>8}: {value:.4f}")
    
    return y_pred, metrics

def plot_residuals(y_test, y_pred):
    """Generate diagnostic residual plots"""
    print("\n📈 Generating diagnostic plots...")
    residuals = y_test - y_pred
    
    plt.figure(figsize=(12, 5))
    
    # Residual distribution
    plt.subplot(1, 2, 1)
    sns.histplot(residuals, kde=True, bins=30)
    plt.title('Residual Distribution')
    plt.xlabel('Residuals')
    
    # Residuals vs predicted
    plt.subplot(1, 2, 2)
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Residuals vs. Predicted')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    
    plt.tight_layout()
    plt.savefig('../reports/figures/residual_analysis.png')
    plt.show()

def compare_feature_sets(X_train, X_test, y_train, y_test):
    """Compare model performance with different feature sets"""
    print("\n🔎 Comparing feature sets...")
    
    feature_sets = {
        'Top 3': ['lstat', 'rm', 'ptratio'],
        'Top 5': ['lstat', 'rm', 'ptratio', 'dis', 'tax'],
        'All Features': X_train.columns.tolist()
    }
    
    results = []
    for name, features in feature_sets.items():
        model = LinearRegression().fit(X_train[features], y_train)
        y_pred = model.predict(X_test[features])
        
        results.append({
        'Feature Set': name,
        'Features': ', '.join(features) if len(features) <= 5 else f"{len(features)} features",
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAE': mean_absolute_error(y_test, y_pred),
        'R²': r2_score(y_test, y_pred)
    })
    
    results_df = pd.DataFrame(results)
    print("\nFeature Set Comparison:")
    print(results_df.to_markdown(tablefmt="grid", index=False))
    
    return results_df

def main():
    try:
        # Load data and model
        X_train, X_test, y_train, y_test = load_data()
        model = load_model()
        
        # Evaluate model
        y_pred, metrics = evaluate_model(model, X_test, y_test)
        
        # Generate plots
        plot_residuals(y_test, y_pred)
        
        # Compare feature sets
        compare_feature_sets(X_train, X_test, y_train, y_test)
        
        print("\n✅ Evaluation completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {str(e)}")

if __name__ == "__main__":
    main()