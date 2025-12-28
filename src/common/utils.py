import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_cup_data(train=True):
    """
    Loads the ML-CUP dataset.
    If `train=True` (default), loads training data, otherwise loads test data.
    
    Returns:
        ids (DataFrame): The 'id' column
        X (DataFrame): The 12 input features
        y (DataFrame): The 4 target variables
    """
    file_path = "../data/ml_cup/ML-CUP25-TR.csv" if train else "../data/ml_cup/ML-CUP25-TS.csv"
    
    # Read CSV, skipping lines starting with '#'
    df = pd.read_csv(file_path, comment='#', header=None)
    
    # Column 0 is ID, 1-12 are Inputs, 13-16 are Targets
    ids = df.iloc[:, 0]
    X = df.iloc[:, 1:13]
    y = df.iloc[:, 13:]
    
    return ids, X, y

def split_data(X, y, test_size=0.1, random_state=42):
    """
    Splits data into Development set (for CV) and Internal Test set.
    We use the same random_state to ensure consistent splits across all algos.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def mean_euclidean_error(y_true, y_pred):
    """
    Calculates the Mean Euclidean Error (MEE).
    
    Args:
        y_true: Ground truth (correct) target values.
        y_pred: Estimated target values.
        
    Returns:
        float: The MEE value.
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate Euclidean distance for each sample (row)
    # sum((true - pred)^2) -> sqrt -> mean over all samples
    errors = np.sqrt(np.sum((y_true - y_pred) ** 2, axis=1))
    
    return np.mean(errors)

class CupScaler:
    """
    A helper class to handle normalization (Zero Mean, Unit Variance)
    and ensure we can inverse-transform targets for evaluation.
    """
    def __init__(self):
        self.X_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        
    def fit_transform(self, X_train, y_train):
        """Fits scalers on training data and transforms it."""
        X_scaled = self.X_scaler.fit_transform(X_train)
        y_scaled = self.y_scaler.fit_transform(y_train)
        return X_scaled, y_scaled
        
    def transform(self, X_test, y_test=None):
        """Transforms new data using the fitted scalers."""
        X_scaled = self.X_scaler.transform(X_test)
        y_scaled = None
        if y_test is not None:
            y_scaled = self.y_scaler.transform(y_test)
        return X_scaled, y_scaled
    
    def inverse_transform_y(self, y_pred_scaled):
        """Converts scaled predictions back to the original space for MEE calculation."""
        return self.y_scaler.inverse_transform(y_pred_scaled)