# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 19:36:13 2020

@author: ravros
"""

# matplotlib inline
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Generate random data
from sklearn.datasets import make_blobs
from sklearn.metrics import confusion_matrix


def calc(trueLabels, predictedLabels):
    tn, fp, fn, tp = confusion_matrix(trueLabels, predictedLabels).ravel()
    print("tp= " + str(tp) + " fp= " + str(fp) + " fn= " + str(fn) + " tp= " + str(tp))
    tpr = tp / (tp + fn)
    fpr = 1 - (tn / (tn + fp))
    acc = (tp + tn) / (tn + fp + fn + tp)
    pre = tp / (tp + fp)
    print("tpr=" + str(tpr) + " fpr= " + str(fpr) + " accuracy= " + str(acc) + " precision= " + str(pre))


def predict_blobs(n_samples=200, n_features=2, centers=([(5, 5), (-5, -5)]), cluster_std=(0.5, 0.2)):
    X, trueLabels = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers,
                               cluster_std=cluster_std, random_state=5)
    plt.title("The generated labels")
    plt.scatter(X[:, 0], X[:, 1], c=trueLabels, s=40, cmap='viridis')
    plt.show()

# Plot the data with K Means Labels

    kmeans = KMeans(len(centers), random_state=5)
    predictedLabels = kmeans.fit(X).predict(X)
    plt.title("The predicted labels")
    plt.scatter(X[:, 0], X[:, 1], c=predictedLabels, s=40, cmap='viridis')
    plt.show()
    calc(trueLabels, predictedLabels)


# predict_blobs(n_samples=200, n_features=2, centers=([(1, 1), (-1, -1)]), cluster_std=(0.5, 0.5))
# predict_blobs(n_samples=200, n_features=2, centers=([(1, 1), (-1, -1)]), cluster_std=(0.2, 0.2))
# predict_blobs(n_samples=200, n_features=2, centers=([(2, 2), (-2, -2)]), cluster_std=(1, 1))
# predict_blobs(n_samples=200, n_features=2, centers=([(2, 2), (-2, -2)]), cluster_std=(3, 1))
# predict_blobs(n_samples=200, n_features=2, centers=([(2, 2), (0, 0), (-2, -2)]), cluster_std=(1, 1, 1))
# predict_blobs(n_samples=200, n_features=2, centers=([(2, 2), (0.5, 0.5), (-2, -2)]), cluster_std=(1.5, 2, 1.5))
# predict_blobs(n_samples=200, n_features=2, centers=([(2, 2), (1, 1), (-2, -2)]), cluster_std=(3, 1.8, 3))
predict_blobs(n_samples=200, n_features=2, centers=([(1, 1), (-1, -1)]), cluster_std=(1, 1))
predict_blobs(n_samples=200, n_features=2, centers=([(0, 0), (0.5, 0.5)]), cluster_std=(0.2, 0.2))

