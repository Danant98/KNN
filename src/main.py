#!/usr/bin/env python

__author__ = 'Daniel Elisabethsønn Antonsen, UiT Arctic University of Norway'

# Importing libraries and modules
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from knn import kNN as kn

# Loading IRIS data set
data = load_iris()
X_train, X_test, Y_train, Y_test = train_test_split(
    data['data'], data['target'], test_size = 0.3, random_state = 0)


def accuracy(ypred:np.ndarray, y:np.ndarray, output:bool = False):
    """
    Computing the accuracy
    """
    assert len(ypred) == len(y), 'predicted labels and ground truth must be of equal length'

    # Computing accuracy
    acc = np.sum(ypred == y) / y.shape[0]
    if output:
        return print(f'Accuracy = {100 * acc:.2f}%')
    return acc


if __name__ == '__main__':
    pass
    # Running k-nearest neighbors algorithm
    model = kn(7)
    model.fit(X_train, Y_train)
    pred = model.predict(X_test)
    accuracy(pred, Y_test, output = True)