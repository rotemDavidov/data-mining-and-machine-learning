# import regular expressins packge
# import numbers package

import numpy as np
import re

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
            
rows = 5
fileContent = [""]*rows

#read  and preprocess files 
fileContent[0] = preProcess(readFile('DB.txt'))
fileContent[1] = preProcess(readFile('HP_small.txt'))
fileContent2 = preProcess(readFile('Tolkien.txt'))
numParts = 3
# split the third file to parts
partLength = int(len(fileContent2)/numParts) 
fileContent[2]  = fileContent2[0:partLength]
fileContent[3]  = fileContent2[partLength:partLength*2]
fileContent[4]  = fileContent2[partLength*2:partLength*3]
 
# concat files contents

allFilesStr = ""
for i in range(rows):
    allFilesStr += fileContent[i]

# generate a set of all words in files 
wordsList =  allFilesStr.split()
wordsSet =  set(wordsList)

# Read stop words file - words that can be removed
stopWords = readFile('stopwords_en.txt')
stopWordsList = stopWords.split()
stopWordsSet = set(stopWordsList)
# Remove the stop words from the word list
dictionary = wordsSet.difference(stopWordsSet)

# Find the frequency of the dictionary words in the files
wordFrequency = np.empty((rows,len(dictionary)),dtype=np.int64)
for i in range(rows):
    print("fileContent ",i)
    for j,word in enumerate(dictionary):
        wordFrequency[i,j] = len(re.findall(word,fileContent[i]))
        
# find the distance matrix between the text files
dist = np.empty((rows,rows))
for i in range(rows): 
    for j in range(rows):
        # calculate the distance between the frequency vectors
        dist[i,j] = np.linalg.norm(wordFrequency[i,:]-wordFrequency[j,:])
        
print("dist=\n",dist)
        
# find the sum of the frequency colomns and select colomns having sum > 20
minSum = 20
sumArray =  wordFrequency.sum(axis=0)
indexArray = np.where(sumArray > minSum)

# generate a frequencey file with the selected coloumns 
indexArraySize = len(indexArray[0])
wordFrequency1 = np.empty((rows,indexArraySize),dtype=np.int64)

for j in range(indexArraySize):
    wordFrequency1[:,j] = wordFrequency[:,indexArray[0][j]]

