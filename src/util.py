#!/usr/bin/env python

__author__ = 'Daniel Elisabethsønn Antonsen, UiT Institute of statistics and mathematics'

# Importing libraries and modules
import numpy as np

def accuracy(ypred:np.ndarray, y:np.ndarray, output:bool = False):
    """
    Computing the accuracy
    """
    assert len(ypred) == len(y), 'predicted labels and ground truth must be of equal length'

    # Computing accuracy
    acc = np.sum(ypred == y) / y.shape[0]
    if output:
        return print(f'Accuracy = {acc:.3f}')
    return acc

def within_scatter(X:np.ndarray, Y:np.ndarray):
    """
    Computing the within class scatter matrix
    """
    # Finding unique classes in data set
    labels = np.unique(Y)

    Sw = 0
    # Computing the covariance for each class
    for i in range(len(labels)):
        # Finding the data corresponding to each class
        X_c = X[labels[i] == i].squeeze()

        # Probability from each class
        prob = X_c.shape[0] / X.shape[0]
        
        # Computing the covariance matrix 
        X_cov = np.cov(X_c.T)

        # Add to the overall scatter matrix
        Sw += prob * X_cov
    
    return Sw


def between_scatter(X:np.ndarray, Y:np.ndarray):
    """
    Computing the within class scatter matrix
    """
    # Finding labels
    labels = np.unique(Y)

    glob_mean = 0
    # Global mean
    for i in range(len(labels)):
        # Finding the data corresponding to each class
        X_c = X[labels[i] == i].squeeze()

        # Computing probability
        prob = X_c.shape[0] / X.shape[0]

        # Add the class mean to global mean
        glob_mean += prob * np.mean(X_c, axis = 0)

    Sb = 0
    for i in range(len(labels)):
        # Finding the data corresponding to each class
        X_c = X[labels[i] == i].squeeze()

        # Computing probability
        prob = X_c.shape[0] / X.shape[0]
        
        # Computing class mean and add to scatter matrix
        mean_c = np.mean(X_c, axis = 0)
        diff = (mean_c - glob_mean).reshape(X.shape[1], 1)
        Sb += prob * ((diff).dot(diff.T))
    
    return Sb

def J3(Sw:np.ndarray, Sb:np.ndarray):
    """
    Computing the J3 score given a data set with n features
    """
    Sm = Sw + Sb
    score = np.matrix.trace(np.linalg.inv(Sw).dot(Sm))
    return score
    
