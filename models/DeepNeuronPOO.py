import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, log_loss
from tqdm import tqdm

class DeepNeuronPOO:
    def __init__(self, X: np.ndarray, y: np.ndarray, hidden_layers: tuple[int] = (16, 16, 16), learning_rate: int = 0.001, n_iter: int = 3000) -> None:
        self.learning_rate: int = learning_rate
        self.n_iter: int = n_iter
        
        # Initialisation of the parameters
        self.dimensions = list(hidden_layers)
        self.dimensions.insert(0, X.shape[0])
        self.dimensions.append(y.shape[0])
        np.random.seed(1)
        self.parametres = self._initialisation(self.dimensions)
        self.C: int = len(self.parametres) // 2

        # Numpy array containing future accuracy and log_loss
        self.training_history = np.zeros((int(n_iter), 2))
        
    
    def _initialisation(self, dimensions: list[int]) -> dict:
        parametres = {}
        C: int = len(dimensions)

        np.random.seed(1)

        for c in range(1, C):
            parametres['W' + str(c)] = np.random.randn(dimensions[c], dimensions[c - 1])
            parametres['b' + str(c)] = np.random.randn(dimensions[c], 1)

        return parametres
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        for i in tqdm(range(self.n_iter)):
            activations = self._forward_propagation(X, self.parametres)
            gradients = self._back_propagation(y, self.parametres, activations)
            self.parametres = self._update(gradients, self.parametres, self.learning_rate)
            Af = activations['A' + str(self.C)]

            # Compute the log loss and accuracy
            self.training_history[i, 0] = (log_loss(y.flatten(), Af.flatten()))
            y_pred = self._predict(X, self.parametres)
            self.training_history[i, 1] = (accuracy_score(y.flatten(), y_pred.flatten()))
    

    def _predict(self, X, parametres):
        """
        Predict the labels for the X data using the model parameters.
        """
        activations = self._forward_propagation(X, parametres)
        C = len(parametres) // 2
        Af = activations['A' + str(C)]
        return Af >= 0.5
    

    def _forward_propagation(self, X: np.ndarray, parametres: dict) -> dict:
        activations = {'A0': X}

        C: int = len(parametres) // 2

        for c in range(1, C + 1):

            Z = parametres['W' + str(c)].dot(activations['A' + str(c - 1)]) + parametres['b' + str(c)]
            activations['A' + str(c)] = 1 / (1 + np.exp(-Z))

        return activations

    def _back_propagation(self, y: np.ndarray, parametres: dict, activations: dict) -> dict:
        m: int = y.shape[1]
        C: int = len(parametres) // 2

        dZ = activations['A' + str(C)] - y
        gradients = {}

        for c in reversed(range(1, C + 1)):
            gradients['dW' + str(c)] = 1/m * np.dot(dZ, activations['A' + str(c - 1)].T)
            gradients['db' + str(c)] = 1/m * np.sum(dZ, axis=1, keepdims=True)
            if c > 1:
                dZ = np.dot(parametres['W' + str(c)].T, dZ) * activations['A' + str(c - 1)] * (1 - activations['A' + str(c - 1)])

        return gradients

    def _update(self, gradients, parametres, learning_rate) -> dict:
        C = len(parametres) // 2

        for c in range(1, C + 1):
            parametres['W' + str(c)] = parametres['W' + str(c)] - learning_rate * gradients['dW' + str(c)]
            parametres['b' + str(c)] = parametres['b' + str(c)] - learning_rate * gradients['db' + str(c)]

        return parametres
    
    # Plot learning curve
    def plot_training_history(self):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.training_history[:, 0], label='train loss')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(self.training_history[:, 1], label='train acc')
        plt.legend()
        plt.show()