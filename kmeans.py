# kmeans.py
# 
# Aaron Chan
# CS 445 Machine Learning (Winter 2019)
# Homework 5 - K-Means Clustering

import numpy as np

class Cluster:

    # K = number of clusers
    # num_attr = number of attributes
    def __init__(self, K, num_attr):
        # Initialize K centers with random values
        # row index = center index
        # col index = attribute index
        self.centers = np.random.rand(K,num_attr)
        
        self.amse = 0     # average mean square error
        self.mss  = 0     # mean square separation
        self.me   = 0     # mean entropy

    # Fit cluster centers to training data
    # Done calculating when centers don't move 
    def fit(self, data, label):
        prev_center = self.centers
        new_center  = np.zeros(self.centers.shape)
        distance    = np.zeros((len(self.centers),len(data)))
        
        while(prev_center != new_center):
            # Calculate distance from center to each data instance
            for i in range(len(self.centers)):
                d = data - self.centers[i]
                d = np.square(d)
                d = np.sum(d,axis=1) 
                distance[i] = d
            distance = np.sqrt(distance)

            # Identify which center data instances are closest to
            dd = distance.transpose()