#!/usr/bin/env python

__author__ = 'Daniel Elisabethsønn Antonsen, UiT Arctic University of Norway'

# Import libraries and modules
import numpy as np


class kNN:

    def __init__(self, X:np.ndarray, k:int = 3):
        self.X = self.normalize(X)
        self.k = k

    
    def normalize(self, X:np.ndarray):
        """
        Normalize output data
        """
        return (X - np.min(X, axis = 1)) / (np.max(X, axis = 1) - np.min(X, axis = 1))


    def euclidean_dist(self, x1:np.ndarray, x2:np.ndarray):
        """
        Euclidean distance between two input points
        """
        return np.linalg.norm(x1 - x2)





