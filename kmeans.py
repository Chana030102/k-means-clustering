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
        self.nonzero = []

        self.amse = 0            # average mean square error
        self.mse  = np.zeros(K)  # mean square error
        self.mss  = 0            # mean square separation
        self.me   = np.zeros(K)  # mean entropy

    # Calculate distance between data and cluster centers
    # return_raw:
    #   False --> return index of centers that each instance belongs to
    #   True  --> return distances between each instance and each center
    def cluster_group(self,centers,data,return_raw=False):
        distance = np.zeros((len(self.centers),len(data)))

         # Calculate distance from center to each data instance
        for i in range(len(centers)):
            d = data - centers[i]
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
    # Iterate until clusters don't move
    # Calculate AMSE, MSS, and mean entropy when done iterating 
    def fit(self, data, label):
        new_center  = self.centers.copy()
        prev_center = np.zeros(self.centers.shape)
        count       = 0  # provide counter to watch how many times center changes

        while(not np.array_equiv(prev_center,new_center)):
            print("changing centers {}".format(count))
            prev_center = new_center.copy()
            cindex = self.cluster_group(prev_center, data)

            # Calculate new centers
            for i in range(len(self.centers)):
                index = np.asarray(np.where(cindex==i))
                index = index.reshape(-1)

                # calculate new position of cluster centroid
                # empty cluster centers stay the same                
                if index.size != 0:
                    cdata = data[index]
                    cdata = np.sum(cdata,axis=0)/len(index)
                    new_center[i] = cdata

            count += 1
            if(np.array_equiv(prev_center,new_center)):
                print("Centers are the same")

        self.centers = new_center.copy()

        # Classify the clusters
        # The most frequent class in the cluster is the class of the cluster
        self.nonzero = []

        for i in range(len(self.centers)):
            index = np.asarray(np.where(cindex==i))
            index = index.reshape(-1)

            if index.size == 0: # empty clusters
                self.classes[i] = -1
                self.mse[i]     =  0
            else:
                u, indices = np.unique(label[index], return_inverse=True)
                self.classes[i] = u[np.argmax(np.bincount(indices))]
                self.nonzero.append(i)
                self.calc_mse(data[index],i)
        
        self.amse = np.sum(self.mse[self.nonzero])/((len(self.nonzero)*(len(self.nonzero)-1)/2))
        self.calc_mss(self.nonzero)

    # Calculate the mean square error of one center
    # average distance between data and center
    # Provide index of nonempty cluster
    def calc_mse(self, data, cindex):
        d = data - self.centers[cindex]
        d = np.square(d)
        self.mse[cindex] = np.sum(d)/len(data)
        
    # Calculate mean square separation
    # Provide list of indices for nonempty clusters
    # (sum of distance of distinct pairs of centroids)/(K(K-1)/2)
    def calc_mss(self,index):
        dsum = 0
        for i in range(len(index)-1):
            d = self.centers[i+1:-1] - self.centers[i]
            d = np.square(d)
            dsum += np.sum(d)  
        self.mss = dsum/len(index)

    # Calculate mean entropy
  #  def entropy(self, data):

