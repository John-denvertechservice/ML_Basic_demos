"""Data loading and preprocessing utilities."""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def normalize_data(X, method='standard'):
    """Normalize data using different methods."""
    if method == 'standard':
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        return (X - mean) / (std + 1e-8)
    elif method == 'minmax':
        min_val = X.min(axis=0)
        max_val = X.max(axis=0)
        return (X - min_val) / (max_val - min_val + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def train_test_split_tensor(X, y, test_size=0.2, random_seed=42):
    """Split tensor data into train and test sets."""
    np.random.seed(random_seed)
    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test


class SimpleDataset(Dataset):
    """Simple PyTorch dataset wrapper."""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X) if not isinstance(X, torch.Tensor) else X
        self.y = torch.LongTensor(y) if not isinstance(y, torch.Tensor) else y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

