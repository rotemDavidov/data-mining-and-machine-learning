# -*- coding: utf-8 -*-
"""4-6(lab5).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1psOBKcQKAg9EBfnOopjAMfPFx8pPoVJ8
"""

"""
Created on Tue Oct 13 19:36:13 2020
@author: ravros
#IRIS DATA
"""
#import libraries
import numpy as np

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 
from sklearn.metrics import silhouette_samples, silhouette_score
#import datasets 
from sklearn import datasets


def errorRande(predictedLabels,newColorMap):
  cnt=0
  for i in range(len(predictedLabels)):
      if predictedLabels[i]!=newColorMap[i]:
         cnt+=1
  print((cnt/len(predictedLabels))*100)
  return (cnt/len(predictedLabels))*100 #return the chances the algoritem switch between the labels of the cluster - a way to make the labels permanent

def clust(data,names,k,newColorMap):
    kmeans = KMeans(n_clusters = k).fit(data)
    predictedLabels = kmeans.labels_+1
    centroids = kmeans.cluster_centers_
    while errorRande(predictedLabels,newColorMap) >60: #continue to cluster until the Kmeans organize the label in the same order "1-2" and not "2-1"
      kmeans = KMeans(n_clusters = k).fit(data)
      predictedLabels = kmeans.labels_+1
      centroids = kmeans.cluster_centers_

    plt.title("The Iris Dataset predicted labels")
    plt.xlabel(names[0])
    plt.ylabel(names[1])
    plt.scatter(data[:,0], data[:,1], c = predictedLabels, s=50)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker = "x", s=80)

    #match the range between predicted and true lables 
    index1=np.where(predictedLabels==1) #the min val in predicted
    index2=np.where(predictedLabels==2) #the max val in predicted
    predictedLabels[index1]=2 #the min val in true labels
    predictedLabels[index2]=3 #the max val in true labels
    for i in range(len(predictedLabels)):
      if predictedLabels[i]!=newColorMap[i]:
        plt.scatter(data[i,0],data[i,1],c='black',marker='x',s=80)
    plt.show()
    return predictedLabels

#from sklearn import datasets
Iris = datasets.load_iris()

irisData = Iris.data[50:149]
trueLabels = Iris.target+1 # true labeling
newColorMap=trueLabels[50:149]
featureNames = Iris.feature_names
plt.plot()
plt.title("The Iris Dataset true labels")
plt.xlabel(featureNames[0])
plt.ylabel(featureNames[1])
myplot =plt.scatter(irisData[:,0],irisData[:,1],c = newColorMap, s=50)
plt.show(myplot)

predictedLabels = clust(irisData,featureNames,2,newColorMap)
print(newColorMap)
print(predictedLabels)

from sklearn.metrics import confusion_matrix
tn, fp, fn, tp = confusion_matrix(newColorMap, predictedLabels).ravel()
tpr = tp/(tp+fn)
fpr= 1-(tn/(tn+fp))
acc= (tp+tn)/(tn+fp+fn+tp)
pre= tp/(tp+fp)
print("tpr="+str(tpr)+" fpr= "+str(fpr)+" accuracy= "+str(acc)+" precision= "+str(pre))