"""
Normalizer Utilities
====================
Provides StandardScaler and MinMaxScaler wrappers with save/load
for reproducible scaling across training, validation, and test sets.
"""

import joblib
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Union, Optional


class DataNormalizer:
    """
    Wraps sklearn scalers with named column tracking and persistence.

    Parameters
    ----------
    method : str
        'standard' (zero mean, unit variance) or 'minmax' (0–1 range).
    feature_names : list of str, optional
        Column names for logging / diagnostics.
    """

    def __init__(self, method: str = 'standard', feature_names: Optional[list] = None):
        self.method = method.lower()
        self.feature_names = feature_names
        if self.method == 'standard':
            self.scaler = StandardScaler()
        elif self.method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown normalizer method: '{method}'. Use 'standard' or 'minmax'.")
        self.is_fitted = False

    def fit(self, X: np.ndarray) -> 'DataNormalizer':
        self.scaler.fit(X)
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Normalizer must be fitted before transform.")
        return self.scaler.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Normalizer must be fitted before inverse_transform.")
        return self.scaler.inverse_transform(X)

    def save(self, path: Union[str, Path]):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({'scaler': self.scaler, 'method': self.method,
                     'feature_names': self.feature_names,
                     'is_fitted': self.is_fitted}, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'DataNormalizer':
        data = joblib.load(path)
        obj = cls(method=data['method'], feature_names=data['feature_names'])
        obj.scaler = data['scaler']
        obj.is_fitted = data['is_fitted']
        return obj

    def summary(self) -> dict:
        if not self.is_fitted:
            return {'method': self.method, 'fitted': False}
        if self.method == 'standard':
            return {
                'method': self.method,
                'fitted': True,
                'mean': self.scaler.mean_.tolist(),
                'std': self.scaler.scale_.tolist(),
                'features': self.feature_names
            }
        else:
            return {
                'method': self.method,
                'fitted': True,
                'min': self.scaler.data_min_.tolist(),
                'max': self.scaler.data_max_.tolist(),
                'features': self.feature_names
            }
