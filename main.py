# main.py
# 
# Aaron Chan
# CS 445 Machine Learning (Winter 2019)
# Homework 5 - K-Means Clustering

import numpy as np

TRAIN_FILE = "../optdigits/optdigits.train"
TEST_FILE  = "../optdigits/optdigits.test"
DELIMITER  = ","

#===== Data Import ====
traind = np.loadtxt(TRAIN_FILE,delimiter=DELIMITER)
trainl = traind[:,-1]
traind = np.delete(traind,-1,axis=1)

testd = np.loadtxt(TEST_FILE,delimiter=DELIMITER)
testl = testd[:,-1]
testd = np.delete(testd,-1,axis=1)

