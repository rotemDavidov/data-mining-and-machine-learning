# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 19:36:13 2020

@author: ravros
"""

# matplotlib inline
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
# Generate random data
from sklearn.datasets import make_blobs
from sklearn.metrics import confusion_matrix


# def calc(trueLabels, predictedLabels):
#     tn, fp, fn, tp = confusion_matrix(trueLabels, predictedLabels).ravel()
#     print("tp= " + str(tp) + " fp= " + str(fp) + " fn= " + str(fn) + " tp= " + str(tp))
#     tpr = tp / (tp + fn)
#     fpr = 1 - (tn / (tn + fp))
#     acc = (tp + tn) / (tn + fp + fn + tp)
#     pre = tp / (tp + fp)
#     print("tpr=" + str(tpr) + " fpr= " + str(fpr) + " accuracy= " + str(acc) + " precision= " + str(pre))


def predict_blobs(n_samples=200, n_features=2, centers=([(5, 5), (-5, -5)]), cluster_std=(0.5, 0.2)):
    X, trueLabels = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers,
                               cluster_std=cluster_std, random_state=5)
    trnSize = len(X)//4
    trnData = np.concatenate((X[0:trnSize, :], X[2*trnSize: 2*trnSize+trnSize, :]), axis=0)  # train data will contain the shirshur of the two train data array
    trnLabels = np.concatenate((trueLabels[0:trnSize], trueLabels[2*trnSize: 2*trnSize+trnSize]), axis=0)
    #                                         21-49                          71-100
    testData = np.concatenate((X[trnSize+1:2*trnSize-1, :], X[3*trnSize+1: len(X), :]), axis=0)
    testLabels = np.concatenate((trueLabels[trnSize+1:2*trnSize-1], trueLabels[3*trnSize+1: len(X)]), axis=0)
    plt.title("The generated labels")
    plt.scatter(X[:, 0], X[:, 1], c=trueLabels, s=40, cmap='viridis')
    plt.show()
# Plot the data with K Means Labels
    kmeans = KMeans(len(centers), random_state=5)
    predictedLabels = kmeans.fit(X).predict(X)
    plt.title("The predicted labels")
    plt.scatter(X[:, 0], X[:, 1], c=predictedLabels, s=40, cmap='viridis')
    plt.show()

    plt.scatter(trnData[:, 0], trnData[:, 1], c=trnLabels, s=50)
    plt.title("The TRAIN Iris Dataset labels")
    plt.show()

    plt.scatter(testData[:, 0], testData[:, 1], c=testLabels, s=50)
    plt.title("The Test Iris Dataset labels BEFOR KNN algo")
    plt.show()

    dist = np.zeros((len(testData), 2 * trnSize))  # rows - number of test points , col- number of train points

    res = []
    for j, testRow in enumerate(testData):
        for i, trnRow in enumerate(trnData):
            dist[j, i] = np.linalg.norm(testRow[:] - trnRow[:])  # the distance of the flower from the train row- oclid distance between himslef to enything
        resultJ = np.argmin(dist[j, :])  # finding the closest nighboor
        result = trnLabels[resultJ]
        # print("The point", j, "is in group", result)
        res.append(result)

    plt.title("The Test Iris Dataset labels AFTER KNN algo")
    plt.scatter(testData[:, 0], testData[:, 1], c=res, s=50)
    plt.show()
    # calc(trueLabels, predictedLabels)


# predict_blobs(n_samples=200, n_features=2, centers=([(1, 1), (-1, -1)]), cluster_std=(0.5, 0.5))
# predict_blobs(n_samples=200, n_features=2, centers=([(1, 1), (-1, -1)]), cluster_std=(0.2, 0.2))
# predict_blobs(n_samples=200, n_features=2, centers=([(2, 2), (-2, -2)]), cluster_std=(1, 1))
# predict_blobs(n_samples=200, n_features=2, centers=([(2, 2), (-2, -2)]), cluster_std=(3, 1))
# predict_blobs(n_samples=200, n_features=2, centers=([(2, 2), (0, 0), (-2, -2)]), cluster_std=(1, 1, 1))
# predict_blobs(n_samples=200, n_features=2, centers=([(2, 2), (0.5, 0.5), (-2, -2)]), cluster_std=(1.5, 2, 1.5))
# predict_blobs(n_samples=200, n_features=2, centers=([(2, 2), (1, 1), (-2, -2)]), cluster_std=(3, 1.8, 3))
# predict_blobs(n_samples=200, n_features=2, centers=([(1, 1), (-1, -1)]), cluster_std=(1, 1))
# predict_blobs(n_samples=200, n_features=2, centers=([(0, 0), (0.5, 0.5)]), cluster_std=(0.2, 0.2))
# predict_blobs(n_samples=50, n_features=2, centers=([(1, 1), (-1, -1), (0, 0)]), cluster_std=(1, 1, 1))
# predict_blobs(n_samples=50, n_features=2, centers=([(0, 0), (0.5, 0.5), (2, 2)]), cluster_std=(0.2, 0.2, 0.2))
predict_blobs(n_samples=50, n_features=2, centers=([(-1, 1), (0, 0.5), (2, 3)]), cluster_std=(0.5, 0.5, 0.5))

