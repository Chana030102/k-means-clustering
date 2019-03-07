# main.py
# 
# Aaron Chan
# CS 445 Machine Learning (Winter 2019)
# Homework 5 - K-Means Clustering
#
# Create 5 different clustering systems and choose the one
# with the lowest average mean square error (amse) to do the following:
# - accuracy of identifying test data classes
# - create confusion matrix
#
# The above is done with a 10 cluster and 30 cluster system

import numpy as np
import sklearn.metrics as metric
from PIL import Image
import kmeans as k

TRAIN_FILE = "../optdigits/optdigits.train"
TEST_FILE  = "../optdigits/optdigits.test"
DELIMITER  = ","
SCALE      = 200

# confusion matrix files
cluster10 = "./k=10/acc={0:.2%} cmatrix k=10.csv"
cluster30 = "./k=30/acc={0:.2%} cmatrix k=30.csv"

#===== Data Import ====
traind = np.loadtxt(TRAIN_FILE,delimiter=DELIMITER)
trainl = traind[:,-1]
traind = np.delete(traind,-1,axis=1)

testd = np.loadtxt(TEST_FILE,delimiter=DELIMITER)
testl = testd[:,-1]
testd = np.delete(testd,-1,axis=1)

#===== K means clustering -- 10 clusters ====
# initialize confusion matrix with zeros
print("K = 10")
cmatrix = np.zeros((len(np.unique(testl)),len(np.unique(testl))))

# Create 5 different cluster systems
c = [k.Cluster(10,traind.shape[1]) for i in range(5)]
amse = [] # empty list to store amse values
for i in c:
    i.fit(traind,trainl)
    amse.append(i.amse)

# identify cluster system with lowest asme
index = np.argmax(amse)
print("Using {}".format(index))

# classify test data
prediction = c[index].classify(testd,testl)
accuracy10 = metric.accuracy_score(testl,prediction)
s = "{0:.2%}".format(accuracy10)

# Create grayscale bmp of cluster centers
print("Creating bmp images for K=10")
for i in range(len(c[index].centers)):
    if c[index].classes[i] != -1:
        d = c[index].centers[i].copy()
        d = d.reshape(8,8)
        im = Image.fromarray(d/16*255)
        im = im.resize([im.width*SCALE,im.height*SCALE])
        im.convert('RGB').save("./k=10/acc={} center {} class {}.bmp".format(s,str(i),str(c[index].classes[i])))


# fill confusion matrix
for i in range(len(testl)):
    cmatrix[int(testl[i]),prediction[i]]+=1

np.savetxt(cluster10.format(accuracy10),cmatrix,delimiter=DELIMITER)

#===== K means clustering -- 30 clusters ====
# initialize confusion matrix with zeros
print("K = 30")
cmatrix = np.zeros((len(np.unique(testl)),len(np.unique(testl))))

# Create 5 different cluster systems
c = [k.Cluster(30,traind.shape[1]) for i in range(5)]
amse = [] # empty list to store amse values
for i in c:
    i.fit(traind,trainl)
    amse.append(i.amse)

# identify cluster system with lowest asme
index = np.argmax(amse)
print("Using {}".format(index))

# classify test data
prediction = c[index].classify(testd,testl)
accuracy30 = metric.accuracy_score(testl,prediction)
s = "{0:.2%}".format(accuracy30)

# Create grayscale bmp of cluster centers
print("Creating bmp images for K=30")
for i in range(len(c[index].centers)):
    if c[index].classes[i] != -1:
        d = c[index].centers[i].copy()
        d = d.reshape(8,8)
        im = Image.fromarray(d/16*255)
        im = im.resize([im.width*SCALE,im.height*SCALE])
        im.convert('RGB').save("./k=30/acc={} center {} class {}.bmp".format(s,str(i),str(c[index].classes[i])))

# fill confusion matrix
for i in range(len(testl)):
    cmatrix[int(testl[i]),prediction[i]]

np.savetxt(cluster30.format(accuracy30),cmatrix,delimiter=DELIMITER)

print("10 cluster system - accuracy = {0:.2%}".format(accuracy10))
print("30 cluster system - accuracy = {0:.2%}".format(accuracy30))
