import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from LogisticRegression import LogisticRegression

DATA_DIR = "data/"
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"

def load_train_data(filepath):
    print("Loading data from file: ", filepath)
    data = np.genfromtxt(filepath, delimiter=',', skip_header=1, dtype=str)

    # Remove the first column (PassengerId) and select only the columns we need
    data = data[:, [1, 2, 5, 6, 7, 8, 10]]
    data[:, 2] = np.where(data[:, 2] == "male", 0, 1)   # Replace "male" or "female" with 0 or 1
    data[data == ""] = 0
    data = data.astype(float)

    y = data[:, 0]
    X = np.delete(data, 0, axis=1)
    return X, y


def load_test_data(filepath):
    print("Loading data from file: ", filepath)
    data = np.genfromtxt(filepath, delimiter=',', skip_header=1, dtype=str)

    data[:, 4] = np.where(data[:,4] == "male", 0, 1)   # Replace "male" or "female" with 0 or 1
    data[data == ""] = 0
    data = data[:, [1, 4, 5, 6, 7, 9]]
    X_test = data.astype(float)
    return X_test


if __name__ == "__main__":
    X_train, y_train = load_train_data(DATA_DIR + TRAIN_FILE)
    X_test = load_test_data(DATA_DIR + TEST_FILE)
    # print(f"X_train : {X_train[0]} and his shape is {X_train.shape}")
    # print(f"y_train : {y_train[0]} and his shape is {y_train.shape}")
    # print(f"X_test : {X_test[0]} and his shape is {X_test.shape}")

    clf = LogisticRegression(lr=0.01)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(y_pred)
    

    # bc = datasets.load_breast_cancer()
    # X, y = bc.data, bc.target
    # print(f"Shape of X: {X.shape} and shape of y: {y.shape}")
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    # print(f"Shape of X_train: {X_train.shape} and shape of y_train: {y_train.shape}")

    # clf = LogisticRegression(lr=0.01)
    # clf.fit(X_train,y_train)
    # y_pred = clf.predict(X_test)

    # def accuracy(y_pred, y_test):
    #     return np.sum(y_pred==y_test)/len(y_test)

    # acc = accuracy(y_pred, y_test)
    # print(acc)