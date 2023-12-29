#!/usr/bin/env python

__author__ = 'Daniel Elisabethsønn Antonsen, UiT Arctic University of Norway'

# Importing libraries and modules
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sn; sn.set_style('darkgrid')
from sklearn.datasets import load_iris, make_blobs
from sklearn.model_selection import train_test_split
from knn import kNN as kn
from util import accuracy, within_scatter, between_scatter, J3


# Loading IRIS data set
data = load_iris()
X_train, X_test, Y_train, Y_test = train_test_split(
    data['data'], data['target'], test_size = 0.3, random_state = 0)    

# Computing the scatter matrices and the J3 score
# Sw = within_scatter(X_train, Y_train)
# Sb = between_scatter(X_train, Y_train)
# score = J3(Sw, Sb)

# Only consider the first two features for visualization 
X = data['data'][:, :2]
Y = data['target']

# Unique classes and color schemes
labels = np.unique(Y)
col = ['r', 'g', 'b']
cmap = colors.ListedColormap(col)

# Span of meshgrid
x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

# Creating meshgrid
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 100), np.linspace(x2_min, x2_max, 100))

# Creating dataset for classification
X_data = np.c_[xx1.ravel(), xx2.ravel()]

# Predicting labels
model2 = kn(7)
model2.fit(X, Y)

pred = model2.predict(X)
y_pred = model2.predict(X_data)

# Reshaping prediction array
y_pred = y_pred.reshape(xx1.shape)

# Plotting boundary
plt.figure()
plt.contourf(xx1, xx2, y_pred, cmap = cmap, alpha = 0.5)
for i in range(len(labels)):
    plt.scatter(X[Y == labels[i], 0], X[Y == labels[i], 1], c = col[i], label = f'Class {i + 1}')
plt.legend()
plt.show()


# Running k-nearest neighbors algorithm for varying k numbers
# acc = []
# for ki in range(1, 30):
#     # Computing model using ki as k
#     model = kn(ki)
#     model.fit(X_train, Y_train)
#     pred = model.predict(X_test)

#     # Computing accuracy for the given k
#     a = accuracy(pred, Y_test)
#     acc.append(a)

# Plotting accuracy vs number of k
# plt.figure()
# plt.plot(range(len(acc)), acc)
# plt.xlabel(r'$k$')
# plt.ylabel(r'Accuracy')
# plt.show()

if __name__ == '__main__':
    pass
    # model = kn(7)
    # model.fit(X_train, Y_train)
    # pred = model.predict(X_test)
    # accuracy(pred, Y_test, output = True)
