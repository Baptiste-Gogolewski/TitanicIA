import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature = None, threshold = None, left = None, right = None, *, value = None) -> None:
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, min_sample_split: int = 2, max_depth: int = 100, n_features = None) -> None:
        self.min_samples_split: int = min_sample_split
        self.max_depth: int = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.n_features = X.shape[1] if not self.n_features else min(self.n_features, X.shape[1])   # Nombre de features
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Ckeck the stopping criteria
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value = leaf_value)

        feat_idxs = np.random.choice(n_features, self.n_features, replace = False)
        # Find the best split
        best_feature, best_thresh = self._best_split(X, y, feat_idxs)

        # Create child nodes
        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feature, best_thresh, left, right)


    def _best_split(self, X: np.ndarray, y: np.ndarray, feat_idxs):
        best_gain: int = -1
        split_idx, split_thresh = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                # Calculate the information gain
                gain = self._information_gain(y, X_column, threshold)
                
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold
        
        return split_idx, split_thresh

    def _information_gain(self, y: np.ndarray, X_column, threshold):
        # Parent entropy
        parent_entropy = self._entropy(y)

        # Create children
        left_idx, right_idx = self._split(X_column, threshold)

        if len(left_idx) == 0 or len(right_idx) == 0:
            return 0

        # Calculate the weighted entropy of children
        n = len(y)
        n_l, n_r = len(left_idx), len(right_idx)
        e_l, e_r = self._entropy(y[left_idx]), self._entropy(y[right_idx])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        # Calculate the information gain
        information_gain = parent_entropy - child_entropy
        return information_gain

    
    def _split(self, X_column, split_threshold):
        left_idxs = np.argwhere(X_column <= split_threshold).flatten()
        right_idxs = np.argwhere(X_column > split_threshold).flatten()
        return left_idxs, right_idxs


    def _entropy(self, y: np.ndarray):
        if y.dtype == "float":
            y = y.astype(int)
    
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log(p) for p in ps if p > 0])


    def _most_common_label(self, y: np.ndarray):
        if len(y) == 0:
            return None
        couter = Counter(y)
        value = couter.most_common(1)[0][0]
        return value


    def predict(self, X: np.ndarray):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x: np.ndarray, node = None):
        if node.is_leaf_node():
            return node.value

        if node.threshold is not None and x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)

        return self._traverse_tree(x, node.right)