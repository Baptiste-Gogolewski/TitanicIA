import numpy as np

DATA_DIR = "data/"
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"

def load_data(file_path):
    print("Loading data from file: ", file_path)
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1, dtype=str)

    # Remove the first column (PassengerId) and select only the columns we need
    data = data[:, [1, 2, 5, 6, 7, 8, 10]]
    data[:, 2] = np.where(data[:, 2] == "male", 0, 1)   # Replace "male" or "female" with 0 or 1
    data[data == ""] = 0
    data = data.astype(float)

    y = data[:, 0]
    X = np.delete(data, 0, axis=1)
    return X, y

12567810

if __name__ == "__main__":
    X_train, y_train = load_data(DATA_DIR + TRAIN_FILE)
    # test_data = load_data(DATA_DIR + TEST_FILE)
    print(f"X_train : {X_train[0]}")
    print(f"y_train : {y_train[0]}")
    # print("Test data shape: ", test_data)