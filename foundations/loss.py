import numpy as np
from numpy.typing import NDArray

class Solution:
    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        epsilon = 1e-7
        # Clip predictions to prevent log(0) for both y_pred=0 and y_pred=1
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Calculate Binary Cross-Entropy
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        
        return round(float(loss), 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        epsilon = 1e-7
        # Clip predictions to prevent log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Calculate Categorical Cross-Entropy
        # Sum over the classes (axis=1), then average over the samples (np.mean)
        loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        
        return round(float(loss), 4)