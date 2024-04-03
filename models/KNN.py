import numpy as np
from collections import Counter

def euclidean_distance(x1: np.ndarray, x2: np.ndarray):
    return np.sqrt(np.sum((x1 - x2)**2))

class KNN:
    def __init__(self, k: int = 3):
        self.k: int = k
  
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_train: np.ndarray = X
        self.y_train: np.ndarray = y

    def predict(self, X: np.ndarray):
        predictions: list[int] = [self._predict(x) for x in X]
        return predictions

  
    def _predict(self, x):
        # Compute the distance
        distances: np.ndarray = [euclidean_distance(x, x_train) for x_train in self.X_train]

        # get the closest K
        k_indices: np.ndarray = np.argsort(distances)[:self.k]
        k_nearest_labels: np.ndarray = [self.y_train[i] for i in k_indices]

        # Majority voye
        most_common: np.ndarray = Counter(k_nearest_labels).most_common()
        return most_common[0][0]  # To just see the labels