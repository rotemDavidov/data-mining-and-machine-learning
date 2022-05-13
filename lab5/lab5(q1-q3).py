"""
 ****************סעיפים 1 עד 3
Created on Tue Oct 13 19:36:13 2020
@author: ravros
#IRIS DATA
"""
# import libraries
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score
# import datasets
from sklearn import datasets


def clust(data, names, k):
    kmeans = KMeans(n_clusters=k).fit(data)
    predictedLabels = kmeans.labels_+1
    centroids = kmeans.cluster_centers_
    plt.title("The Iris Dataset predicted labels")
    plt.xlabel(names[0])
    plt.ylabel(names[1])
    plt.scatter(data[:, 0], data[:, 1], c=predictedLabels, s=50)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker="x", s=80)
    plt.show()

    plt.title("The Iris Dataset predicted labels")
    plt.xlabel("Iris sample number")
    plt.ylabel("pridicted label")
    plt.bar(range(len(predictedLabels)), predictedLabels)
    plt.show()
# show the silhouette values for k clusters
    silhouetteAvg = silhouette_score(data, predictedLabels)
    sample_silhouette_values_ = silhouette_samples(data, predictedLabels)
    plt.plot(sample_silhouette_values_)
    plt.plot(silhouetteAvg, 'r--')
    plt.title("The silhouette plot for the various clusters.")
    plt.xlabel("The silhouette coefficient values")
    plt.ylabel("Cluster label")
    xmin = 0
    xmax = len(predictedLabels)
# The vertical line for average silhouette score of all the values
    plt.hlines(silhouetteAvg, xmin, xmax, colors='red', linestyles="--")
    plt.show()

    print("For clusters =", k, "The average silhouette_score is:", silhouetteAvg)
    return predictedLabels

# from sklearn import datasets
Iris = datasets.load_iris()

irisData = Iris.data[50:149]
print(len(irisData))
trueLabels = Iris.target+1
trueLabels = trueLabels[50:149]
# print(trueLabels)
# true labeling
featureNames = Iris.feature_names
plt.plot()
plt.title("The Iris Dataset true labels")
plt.xlabel(featureNames[0])
plt.ylabel(featureNames[1])
plt.scatter(irisData[:, 0], irisData[:, 1], c=trueLabels, s=50)
plt.show()


predictedLabels = clust(irisData[50:150], featureNames, 2)

