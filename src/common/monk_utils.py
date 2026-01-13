import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from enum import Enum

class MonkDataset(Enum):
    MONK1_TRAIN = "../data/monk/monks-1.train"
    MONK1_TEST  = "../data/monk/monks-1.test"

    MONK2_TRAIN = "../data/monk/monks-2.train"
    MONK2_TEST  = "../data/monk/monks-2.test"

    MONK3_TRAIN = "../data/monk/monks-3.train"
    MONK3_TEST  = "../data/monk/monks-3.test"

def load_monk_dataset(dataset: MonkDataset):
    """
    Loads a MONK dataset file.
    Format: class (0/1), a1, a2, a3, a4, a5, a6, id
    """
    # The dataset is space-separated. 
    # Columns: class (0), a1 (1), a2 (2), a3 (3), a4 (4), a5 (5), a6 (6), id (7)
    df = pd.read_csv(dataset.value, sep=' ', skipinitialspace=True, header=None)
    
    # Drop the first (empty) column if it exists due to leading spaces, and the last 'id' column
    # MONK files have an empty col at index 0
    # Clean it up:
    df = df.dropna(axis=1, how='all') # drop completely empty cols
    
    # Reset columns to standard index
    df.columns = range(df.shape[1])
    
    # After cleanup, column 0 is target, 1-6 are features, 7 is ID
    # We drop the ID column (the last one)
    df = df.iloc[:, :-1]
    
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values
    
    return X, y

def one_hot_encode_monk(X_train, X_test):
    """
    Applies 1-of-k encoding to MONK categorical features.
    """
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    X_train_enc = encoder.fit_transform(X_train)
    X_test_enc = encoder.transform(X_test)

    return X_train_enc, X_test_enc

class MonkEncoder:
    """
    Handles One-Hot Encoding for MONK features.
    """
    def __init__(self):
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        
    def fit_transform(self, X):
        return self.encoder.fit_transform(X)
    
    def transform(self, X):
        return self.encoder.transform(X)

def calc_mse(y_true, y_pred):
    """Mean Squared Error"""
    return mean_squared_error(y_true, y_pred)

def calc_accuracy(y_true, y_pred):
    """Accuracy Score (for classification)"""
    # Ensure inputs are rounded to 0 or 1 if they are probabilities
    y_pred_label = np.round(y_pred)
    return accuracy_score(y_true, y_pred_label)