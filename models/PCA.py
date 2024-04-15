import numpy as np

class PCA:
    def __init__(self, n_components: int) -> None:
        self.n_components: int = n_components
        self.components = None
        self.mean: np.ndarray = None

    def fit(self, X):
        # Mean centering
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # Covariance function needs samples as columns
        cov: np.ndarray = np.cov(X.T)

        # Eigenvalues and eigenvectors
        eigenvectors, eigenvalues = np.linalg.eig(cov)

        # eigenvectors v = [:, 1] column vector, transpose this for easier calculations
        eigenvectors = eigenvectors.T

        # Sort eigenvectors
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        self.components = eigenvectors[:self.n_components]

    def transform(self, X: np.ndarray) -> np.ndarray:
        # project data
        X = X - self.mean
        return np.dot(X, self.components.T)