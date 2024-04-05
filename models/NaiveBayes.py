import numpy as np

class NaiveBayes:

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        self._classes: np.ndarray = np.unique(y)
        n_classes: int = len(self._classes)

        # Calculate mean, variance, and prior for each class
        self._mean: np.ndarray = np.zeros((n_classes, n_features), dtype = np.float64)
        self._var: np.ndarray = np.zeros((n_classes, n_features), dtype = np.float64)
        self._priors: np.ndarray = np.zeros(n_classes, dtype = np.float64)

        for idx, c in enumerate(self._classes):
            X_c: np.ndarray = X[y == c]
            self._mean[idx, :] = X_c.mean(axis = 0)
            self._var[idx, :] = X_c.var(axis = 0)
            self._priors[idx] = X_c.shape[0] / float(n_samples)


    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    def _predict(self, x):
        posteriors = []

        # Calculate posterior probability for each class
        for idx, c in enumerate(self._classes):
            prior: np.ndarray = np.log(self._priors[idx])
            posterior: np.ndarray = np.sum(np.log(self._pdf(idx, x)))
            posterior = posterior + prior
            posteriors.append(posterior)
        
        # Return class with highest posterior probability
        return self._classes[np.argmax(posteriors)]
    
    def _pdf(self, class_idx, x):
        mean: np.ndarray = self._mean[class_idx]
        var: np.ndarray = self._var[class_idx]
        numerator: np.ndarray = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator: np.ndarray = np.sqrt(2 * np.pi * var)
        return numerator / denominator