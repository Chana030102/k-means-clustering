# kmeans.py
# 
# Aaron Chan
# CS 445 Machine Learning (Winter 2019)
# Homework 5 - K-Means Clustering

import numpy as np

MAX = 16

class Cluster:

    # K = number of clusers
    # num_attr = number of attributes
    def __init__(self, K, num_attr):
        # Initialize K centers with random values
        # row index = center index
        # col index = attribute index
        self.centers = np.random.randint(0,MAX+1,size=(K,num_attr))
        self.classes = np.zeros(K)

        self.amse = 0     # average mean square error
        self.mss  = 0     # mean square separation
        self.me   = 0     # mean entropy

    # Calculate distance between data and cluster centers
    # return_raw:
    #   False --> return index of centers that each instance belongs to
    #   True  --> return distances between each instance and each center
    def distance(self,data,return_raw=False):
        distance = np.zeros((len(self.centers),len(data)))

         # Calculate distance from center to each data instance
        for i in range(len(self.centers)):
            d = data - self.centers[i]
            d = np.square(d)
            d = np.sum(d,axis=1) 
            distance[i] = d
        distance = np.sqrt(distance)

        if return_raw == False:
            # Identify which center data instances are closest to
            dd = distance.transpose()
            return dd.argmin(axis=1)
        else:
            return distance

    # Fit cluster centers to training data
    # Done calculating when centers don't move  
    def fit(self, data, label):
        new_center  = self.centers
        prev_center = np.zeros(self.centers.shape)
        
        while(not np.array_equiv(prev_center,new_center)):
            prev_center = new_center
            cindex = self.distance(data)

            # Calculate new centers
            for i in range(len(self.centers)):
                index = np.asarray(np.where(cindex==i))
                index = index.reshape(-1)

                if index.size == 0: 
                    # randomly choose new centroid when cluster is empty
                    cdata = np.random.randint(0,MAX+1,size=self.centers.shape[1])
                else:
                    # calculate new position of cluster centroid
                    cdata = data[index]
                    cdata = np.sum(cdata,axis=0)/len(index)

                new_center[i] = cdata
        self.centers = new_center

        # Classify the clusters
        # The most frequent class in the cluster is the class of the cluster
        print("Classifying clusters")
        for i in range(len(self.centers)):
            index = np.asarray(np.where(cindex==i))
            index = index.reshape(-1)

            u, indices = np.unique(label[index], return_inverse=True)
            self.classes[i] = u[np.argmax(np.bincount(indices))]

        


            
            

                