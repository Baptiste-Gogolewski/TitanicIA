# TitanicIA

This repository implements different types of artificial intelligence. You can test each class and you can also use your own dataset by putting it in the data directory.

## Models

There are currently 5 models : K-nearest-Neighbors, Logistic Regression, Naive Bayes, a Decision tree and a random forest

## Split the dataset

In the `main` function, I use the `train_test_split` function from `sklearn` to split the data in 4 variables in order to test, after training the model, the efficiency of this one.

## How to use a model ?

In first, initialize the model with the `name` of the class, I built a knn algorithm :
```python
knn = KNN(k = 5)
```

Then you need to train your model with the `fit` function :
```python
knn.fit(X_train, y_train)
```

Finally, you can predict with a `X_train` variable with the `predict` function :
```python
knn.predict(X_test)
```
## How to see the efficiency of the model ?

In the `main` function, there is a function named `accuracy` and you can use it to see the model performance
```python
print(accuracy(y_pred, y_test))
```
You can also store the result of the `accuracy` function in a variable and give it as argument.