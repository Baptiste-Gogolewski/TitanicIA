from models.DecisionTree import DecisionTree
import numpy as np
from collections import Counter

class RandomForest:
    def __init__(self, n_trees: int = 10, max_depth: int = 10, min_samples_split: int = 2, n_features: int = None):
        self.n_trees: int = n_trees
        self.max_depth: int = max_depth
        self.min_samples_split: int = min_samples_split
        self.n_features: int = n_features
        self.trees = []
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(self.min_samples_split, self.max_depth, self.n_features)
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
    
    def _bootstrap_samples(self, X: np.ndarray, y: np.ndarray):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace = True)
        return X[idxs], y[idxs]
    
    
    def predict(self, X: np.ndarray):
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        tree_predictions = np.swapaxes(tree_predictions, 0, 1)
        y_pred = [self._most_common_label(tree_pred) for tree_pred in tree_predictions]
        return np.array(y_pred)
    
    def _most_common_label(self, y: np.ndarray):
        if len(y) == 0:
            return None
        couter = Counter(y)
        value = couter.most_common(1)[0][0]
        return value