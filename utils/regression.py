import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

def preprocess_data(df, target_column, feature_columns):
    """Preprocess data for regression."""
    X = df[feature_columns]
    y = df[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_regression_model(X_train, y_train):
    """Train a linear regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_regression_model(model, X_test, y_test):
    """Evaluate regression model and return metrics."""
    y_pred = model.predict(X_test)
    metrics = {
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'r2': r2_score(y_test, y_pred)
    }
    return metrics, y_pred

def plot_regression_results(y_test, y_pred):
    """Create scatter plot of actual vs predicted values."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, y_pred)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title('Actual vs Predicted Values')
    return fig

def plot_regression_line(X, y, model, feature_column):
    """Plot regression line for a single feature."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X[feature_column], y)
    ax.plot(X[feature_column], model.predict(X[[feature_column]]), color='red')
    ax.set_xlabel(feature_column)
    ax.set_ylabel('Target')
    ax.set_title(f'Regression Line: {feature_column} vs Target')
    return fig