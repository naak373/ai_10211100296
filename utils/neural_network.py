import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns

def preprocess_nn_data(df, feature_columns, target_column):
    """Preprocess data for neural network training."""
    X = df[feature_columns].copy()
    y = df[target_column].copy()
    
    # Check if target is categorical
    is_classification = y.dtype == 'object' or len(y.unique()) < 10
    
    # Handle missing values
    for col in X.columns:
        if X[col].dtype in ['float64', 'int64']:
            X[col] = X[col].fillna(X[col].mean())
        else:
            X[col] = X[col].fillna(X[col].mode()[0])
    
    # Encode categorical features
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = LabelEncoder().fit_transform(X[col])
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Handle target variable
    le = None
    n_classes = None
    if is_classification:
        le = LabelEncoder()
        y = le.fit_transform(y)
        n_classes = len(np.unique(y))
        # Convert to one-hot encoding if more than 2 classes
        if n_classes > 2:
            y = tf.keras.utils.to_categorical(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    return {
        'X_train': X_train, 
        'X_test': X_test, 
        'y_train': y_train, 
        'y_test': y_test,
        'is_classification': is_classification,
        'n_classes': n_classes,
        'label_encoder': le,
        'scaler': scaler,
        'X': X
    }

def create_model(input_shape, n_classes=None, hidden_layers=2, is_classification=True, learning_rate=0.01):
    """Create a neural network model."""
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(input_shape,)))
    
    for _ in range(hidden_layers-1):
        model.add(Dense(32, activation='relu'))
    
    if is_classification:
        if n_classes == 2:
            model.add(Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
        else:
            model.add(Dense(n_classes, activation='softmax'))
            loss = 'categorical_crossentropy'
            metrics = ['accuracy']
    else:
        model.add(Dense(1))
        loss = 'mse'
        metrics = ['mae']
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=metrics
    )
    
    return model

def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix for classification results."""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    return fig

def plot_regression_predictions(y_test, y_pred):
    """Plot actual vs predicted values for regression."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, y_pred)
    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--')
    ax.set_xlabel('Actual values')
    ax.set_ylabel('Predicted values')
    ax.set_title('Actual vs Predicted')
    return fig