# import regular expressins packge
# import numbers package
import re
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 
from sklearn.metrics import silhouette_samples, silhouette_score

def readFile(fileName):
    file = open(fileName,'r',encoding="cp437")
    fileStr = ""
    for line in file:
        fileStr += line
    return fileStr
        
# Remove extra spaces
# Remove non-letter chars    
# Change to lower 
def preProcess(fileStr):
    fileStr = re.sub("[^a-zA-Z ]"," ", fileStr)
    fileStr = re.sub(" +"," ", fileStr)
    fileStr = fileStr.lower()
    return fileStr

#Divide the file in chuncks of the same size wind
def partitionStr(fileStr, windSize):
    numParts = len(fileStr)//windSize#// is getting an integer
    chunks=[]
    for i in range(0, numParts):
        windLoc = i*windSize
        chunks += [fileStr[windLoc:windLoc+windSize]]#important to put dpuble [[]]
    return chunks;

# Count the number of dictionary words in files - Frequency Matrix
def getWordFrequency(chunks,dictionary,rows):
    wordFreq = np.empty((rows,len(dictionary)),dtype=np.int64)
    for i in range(rows):
        print(i)
        for j,word in enumerate(dictionary):        
            wordFreq[i,j] = len(re.findall(word,chunks[i]))
    return wordFreq 
       
numFiles = 2
fileContent = [""]*numFiles

#read  and preprocess files 
fileContent[0] = preProcess(readFile('Eliot.txt'))
fileContent[1] = preProcess(readFile('Tolkien.txt'))

#wind - chunks size 
wind = 50000
#Divide the each file into chunks of the size wind 
chunks = []
for i in range(numFiles):
    chunks+= partitionStr(fileContent[i] , wind)    

rows = len(chunks)
# Construct dictionary lines 54 - 65 

allFilesStr = ""
for i in range(numFiles):
    allFilesStr += fileContent[i]

# Generate a set of all words in files 
wordsSet =  set(allFilesStr.split())

# Read stop words file - words that can be removed
stopWordsSet = set(readFile('stopwords_en.txt').split())
# Remove the stop words from the word list
dictionary = wordsSet.difference(stopWordsSet)

wordFrequency = getWordFrequency(chunks,dictionary,rows)

# find the sum of the frequency colomns and select colomns having sum > 100
minSum = 100
sumArray =  wordFrequency.sum(axis=0)
indexArray = np.where(sumArray > minSum)

indexArraySize = len(indexArray[0])
wordFrequency1 = np.empty((rows,indexArraySize),dtype=np.int64)

# generate a frequencey file with the selected coloumns 
for j in range(indexArraySize):
    wordFrequency1[:,j] = wordFrequency[:,indexArray[0][j]]
#load file dist1.npy(from lab3_ex011.py)

#dist=np.load('dist.npy')
num_clusters = 2
#cluster the data into k clusters, specify the k  
kmeans = KMeans(n_clusters = num_clusters)
kmeans.fit(wordFrequency1)
labels = kmeans.labels_ + 1
#show the clustering results  

plt.bar(range(len(labels)),labels) 
plt.title("The partitions cluster labels")
plt.xlabel("The number of partition")
plt.ylabel("The cluster label")
plt.show()

# calculate the silhouette values  
silhouette_avg = silhouette_score(wordFrequency, labels)
sample_silhouette_values = silhouette_samples(wordFrequency, labels)
# show the silhouette values 
plt.plot(sample_silhouette_values) 
plt.title("The silhouette plot")
plt.xlabel("The number of partition")
plt.ylabel("The silhouette coefficient values")

xmin=0
xmax=len(labels)
# The vertical line for average silhouette score of all the values
plt.hlines(silhouette_avg, xmin, xmax, colors='red', linestyles="--") 
plt.show()

print("The number of clusters =", num_clusters,
  "The average silhouette score is:", silhouette_avg)


