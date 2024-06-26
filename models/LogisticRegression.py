import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def sigmoid(x: np.ndarray):
    return 1 / (1 + np.exp(-x))

class LogisticRegression:
    def __init__(self, learning_rate: int = 0.01, n_iters: int = 1000):
        self.learing_rate: int = learning_rate
        self.n_iters: int = n_iters
        self.weights: np.ndarray = None
        self.bias: int = None
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in tqdm(range(self.n_iters)):
            linear_pred: np.ndarray = np.dot(X, self.weights) + self.bias
            predictions: np.ndarray = sigmoid(linear_pred)

            # Gradient Descent
            dw: np.ndarray = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db: np.ndarray = (1 / n_samples) * np.sum(predictions - y)

            self.weights -= self.learing_rate * dw
            self.bias -= self.learing_rate * db
    

    def predict(self, X: np.ndarray):
        linear_pred: np.ndarray = np.dot(X, self.weights) + self.bias
        y_pred: np.ndarray = sigmoid(linear_pred)
        class_pred: list[int] = [0 if y <= 0.5 else 1 for y in y_pred]
        return class_pred