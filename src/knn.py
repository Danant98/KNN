#!/usr/bin/env python

__author__ = 'Daniel Elisabethsønn Antonsen, UiT Arctic University of Norway'

# Import libraries and modules
import numpy as np


class kNN:

    def __init__(self, k:int = 3):
        self.k = k

    def normalize(self, X:np.ndarray):
        """
        Normalize output data, N(0, 1)
        """
        return (X - np.mean(X, axis = 0)) / np.std(X, axis = 0)
    
    def fit(self, X:np.ndarray, Y:np.ndarray):
        """
        Storing X and Y training data
        """
        self.X_train, self.Y_train = self.normalize(X), Y

    def euclidean_dist(self, x1:np.ndarray, x2:np.ndarray):
        """
        Computing Euclidean distance between two points
        """
        return np.sqrt(np.sum((x1 - x2)**2))

    def closest_neighbors(self, X:np.ndarray):
        """
        Computing the closest neighbours and returns most occuring label
        """
        dist = np.zeros(self.X_train.shape[0])
        for i in range(self.X_train.shape[0]):
            dist[i] = self.euclidean_dist(X, self.X_train[i])

        # Finding k closest neighbours
        k_closest = np.argsort(dist)[:self.k]

        # Counting the number of each label
        label = np.bincount(self.Y_train[k_closest])
        return np.argmax(label)
    
    def predict(self, X_test:np.ndarray):
        """
        Predicting test data
        """
        # Normalizing test data
        X_norm = self.normalize(X_test)
        
        # Initializing labels array
        Y_pred = np.zeros(X_test.shape[0])
        
        for i in range(X_norm.shape[0]):
            Y_pred[i] = self.closest_neighbors(X_norm[i])
        
        return Y_pred
