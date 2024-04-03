import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from LogisticRegression import LogisticRegression
from LogisticRegression import accuracy

DATA_DIR = "data/"
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"

def load_train_data(filepath: str):
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


if __name__ == "__main__":
    X_train, y_train = load_train_data(DATA_DIR + TRAIN_FILE)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=1234)

    # Initialize the model
    clf = LogisticRegression(learning_rate = 0.01)

    # Model training
    clf.fit(X_train, y_train)

    # Model prediction
    y_pred: list[int] = clf.predict(X_test)
    
    # Model evaluation
    print(accuracy(y_pred, y_test))