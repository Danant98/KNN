#!/usr/bin/env python

__author__ = 'Daniel Elisabethsønn Antonsen, UiT Institute of statistics and mathematics'

# Importing libraries and modules
import numpy as np


def within_scatter(X:np.ndarray, Y:np.ndarray):
    """
    Computing the within class scatter matrix
    """
    # Finding unique classes in data set
    labels = np.unique(Y)

    # Computing the covariance for each class
    for i in range(len(labels)):
        # Finding the data corresponding to each class
        X_c = X[labels == i]

    




def between_scatter():
    """
    Computing the within class scatter matrix
    """



def J3():
    """
    Computing the J3 score given a data set with n features
    """
    
