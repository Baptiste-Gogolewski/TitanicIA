import numpy as np

class SVM:
    def __init__(self, learning_rate: int = 0.1, lambda_param: int = 0.1, n_iters: int = 1000) -> None:
        self.learning_rate: int = learning_rate
        self.lambda_param: int = lambda_param
        self.n_iters: int = n_iters
        self.W: np.ndarray = None
        self.b = None
    

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape

        y_: np.ndarray = np.where(y <= 0, -1, 1)

        # init weights
        self.W = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.W) - self.b) >= 1
                if condition:
                    self.W -= self.learning_rate * (2 * self.lambda_param * self.W)
                else:
                    self.W -= self.learning_rate * (2 * self.lambda_param * self.W - np.dot(x_i, y_[idx]))
                    self.b -= self.learning_rate * y_[idx]


    def predict(self, X: np.ndarray) -> np.ndarray:
        approx: np.ndarray = np.dot(X, self.W) - self.b
        return np.sign(approx)