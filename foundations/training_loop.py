import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        n, d = X.shape
        w = np.zeros(d)   # ✅ zeros, not zeroes
        b = 0.0
        
        for _ in range(epochs):
            # Step 1: Forward pass — predictions
            y_hat = X @ w + b
            
            # Step 2: Compute error
            error = y_hat - y              # shape (n,)
            
            # Step 3: Compute gradients (MSE)
            # dL/dw = (2/n) * X^T @ error
            # dL/db = (2/n) * sum(error)
            dw = (2.0 / n) * (X.T @ error)
            db = (2.0 / n) * np.sum(error)
            
            # Step 4: Update weights
            w -= lr * dw
            b -= lr * db
        
        return np.round(w, 5), round(b, 5)

