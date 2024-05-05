import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from models.LogisticRegression import LogisticRegression
from models.KNN import KNN
from models.NaiveBayes import NaiveBayes
from models.DecisionTree import DecisionTree
from models.RandomForest import RandomForest
from models.SVM import SVM
from models.PCA import PCA
from models.DeepNeuronPOO import DeepNeuronPOO

DATA_DIR = "data/"
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"

def load_train_data(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    print("Loading data from file: ", filepath)
    data: np.ndarray = np.genfromtxt(filepath, delimiter=',', skip_header=1, dtype=str)

    # Remove the first column (PassengerId) and select only the columns we need
    data = data[:, [1, 2, 5, 6, 7, 8, 10]]
    data[:, 2] = np.where(data[:, 2] == "male", 0, 1)   # Replace "male" or "female" with 0 or 1
    data[data == ""] = 0
    data = data.astype(float)

    y: np.ndarray = data[:, 0]
    X: np.ndarray = np.delete(data, 0, axis=1)
    return X, y


def load_test_data(filepath: str):
    print("Loading data from file: ", filepath)
    data: np.ndarray = np.genfromtxt(filepath, delimiter=',', skip_header=1, dtype=str)

    data[:, 4] = np.where(data[:,4] == "male", 0, 1)   # Replace "male" or "female" with 0 or 1
    data[data == ""] = 0
    data = data[:, [1, 4, 5, 6, 7, 9]]
    X_test: np.ndarray = data.astype(float)
    return X_test

def accuracy(y_pred: np.ndarray, y_test: np.ndarray):
    accuracy: np.ndarray = np.sum(y_test == y_pred) / len(y_test)
    return accuracy

def show_datas(X_train: np.ndarray, y_train: np.ndarray, cmap: str = "viridis"):
    plt.figure()
    plt.scatter(X_train[:,2], X_train[:,3], c = y_train, cmap = cmap, edgecolors = "k", s = 20)
    plt.show()


def Transform(X: np.ndarray) -> np.ndarray:
    pca = PCA(n_components = 6)
    pca.fit(X)
    return pca.transform(X)


if __name__ == "__main__":
    X_train, y_train = load_train_data(DATA_DIR + TRAIN_FILE)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=1234)
    # X_test = load_test_data(DATA_DIR + TEST_FILE)
    # print(f"Shape X_train : {X_train.shape}")
    # X_train = Transform(X_train)
    # print(f"Shape X_train : {X_train.shape}")
    # X_train = X_train.T
    # y_train = y_train.reshape((1, y_train.shape[0]))

    # show_datas(X_train, y_train)

    # Initialize the model
    # clf = LogisticRegression(learning_rate = 0.01)
    # knn = KNN(k = 5)
    # NaiveBayes = NaiveBayes()
    # DecisionTree = DecisionTree()
    # RandomForest = RandomForest(n_trees = 20)
    # svm = SVM(learning_rate = 0.001, lambda_param = 0.001, n_iters = 100000)
    # neuron = DeepNeuronPOO(X_train, y_train, hidden_layers = (2, 8, 8, 8, 8), learning_rate = 0.1, n_iter = 50000))

    # Model training
    # clf.fit(X_train, y_train)
    # knn.fit(X_train, y_train)
    # NaiveBayes.fit(X_train, y_train)
    # DecisionTree.fit(X_train, y_train)
    # RandomForest.fit(X_train, y_train)
    # svm.fit(X_train, y_train)
    # neuron.fit(X_train, y_train)
    # neuron.plot_training_history()

    # Model prediction
    # y_pred: list[int] = clf.predict(X_test)
    # y_pred: list[int] = knn.predict(X_test)
    # y_pred: np.ndarray = svm.predict(X_test)
    
    # Model evaluation
    # print(accuracy(y_pred, y_test))