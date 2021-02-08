# Module for loading mat data file in python
from scipy.io import loadmat

# Module for scientific computation in python
import numpy as np

# Module for data visualisation
import matplotlib.pyplot as plt

# Module to read an image in python
from matplotlib.image import imread

data = loadmat('ex7/ex7data2.mat')
# training data
X = data['X']

# initialising the number of centroids which is directly related to number of clusters
no_of_centroids = 3


def random_centroids_function(X, k):
    """
    This function selects k centroids randomly from data points and returns it to the main function
    :param X: A (m, n) dimensional matrix
        This the location of the data points
    :param k: An integer
        This is the number centroids we want to select randomly from the data points
    :return: A (k, n) matrix
        This is the location of all the randomly selected centroids from the data
    """
    np.random.shuffle(X)
    return X[0:k, :]


def compute_distance_function(X, c):
    """
    This function computes the squared distance between a data point X and a centroid c
    :param X: A one dimensional matrix with two values
        This is the coordinates of the data point X
    :param c: A one dimensional matrix with two values
        This is the coordinates of the centroid c
    :return: A scalar
        This is the squared distance between a data point X and a centroid c
    """
    return np.square(np.sqrt(np.sum(np.square(np.subtract(X, c)))))


def find_closest_centroid_function(X, centroids):
    """
    This function finds the closest centroid for all the data points
    :param X: A (m, n) dimensional matrix
        This the location of all the data points
    :param centroids: A (k, n) dimensional matrix
        This is the location of all the centroid
    :return: A one dimensional matrix idx with length equals to the number of data points
        This holds the index of the closest centroid for all the data points
    """
    idx = []
    for x in range(0, X.shape[0]):
        dist = []
        for c in range(0, centroids.shape[0]):
            dist.append(compute_distance_function(X[[x], :], centroids[[c], :]))
        idx.append(np.argmin(dist))

    return idx


def compute_centroid_function(X, idx, k):
    """
    This function computes the mean of all the data points assigned to a particular centroid and move the centroid to the
    location which is described by the mean
    :param X: A (m, n) dimensional matrix
        This is the location of all the data points
    :param idx: A one dimensional array
        This holds the index of the closest centroid for all the data points
    :param k: A scalar value
        This denotes the number of centroids
    :return: A (k, n) dimensional matrix
        This is the new location of all the centroids
    """
    centroids = np.zeros((k, X.shape[1]))
    for c in range(0, centroids.shape[0]):
        index_list = []
        for i in range(0, len(idx)):
            if c == idx[i]:
                index_list.append(i)
        centroids[c, :] = np.average(X[index_list, :], axis=0)

    return centroids


centroids = random_centroids_function(X, no_of_centroids)
colors = ["Red", "yellow", "Blue"]
for _ in range(10):
    idx = find_closest_centroid_function(X, centroids)
    for c, color in enumerate(colors):
        indices = []
        for i in range(len(idx)):
            if c == idx[i]:
                indices.append(i)
        plt.scatter(X[indices, [0]], X[indices, [1]], color=color, marker="+")
    plt.show(block=False)
    plt.pause(1)
    centroids = compute_centroid_function(X, idx, no_of_centroids)
plt.pause(10)
plt.close()