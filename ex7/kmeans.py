import numpy as np
from math import inf


class KMeans:
    def __init__(self, X, k, iters=50, max_iters=inf):
        """
        parameters:
            <mxn> X: 2D matrix where each row is an example and each column is a feature
            <int> k: number of clusters
            <int> iters: number of iterations for which to run k-means
        """
        self.X = X
        self.m, self.n = X.shape
        self.k = k
        self.iters = iters
        self.max_iters = max_iters
        self.mu = np.array([[None for _ in range(self.n)]
                            for _ in range(self.k)])
        self.c = np.array([None for _ in range(self.m)])

    def update_parameters(self, X=None, k=None, iters=None, max_iters=None):
        if X is not None:
            self.X = X
            self.m, self.n = X.shape
        if k is not None:
            self.k = k
        if iters is not None:
            self.iters = iters
        if max_iters is not None:
            self.max_iters = max_iters
        self.mu = np.array([[None for _ in range(self.n)]
                            for _ in range(self.k)])
        self.c = np.array([None for _ in range(self.m)])

    def distance(self, x1, x2):
        """
        parameters:
            <nx1> x1: first vector
            <nx1> x2: second vector

        returns:
            <float> d: squared norm of the difference between the two vectors

        Comptue the square length of the difference between two n-dimensional vectors.
        """
        return np.square(np.linalg.norm(x1 - x2))

    def distortion(self, clustering, colors):
        """
        parameters:
            <kxn> clustering: 2D matrix where each row is the centroid of a cluster
            <mx1> colors: vector where the ith entry is the color of the ith example

        returns:
            <float> J: distortion for the given clustering on the dataset

        Compute the distortion function for a given clustering and color assignment.
        """
        return (1 / self.m) * sum([self.distance(self.X[i], clustering[colors[i]]) for i in range(self.m)])

    def compute_centroids(self, colors):
        """
        parameters:
            <kx1> colors: vector where the ith entry is the color of the ith example

        returns:
            <kxn> centroids: 2D matrix where the ith row is the centroid of color i

        Compute the centroids for each color in the clustering.
        """
        centroids = np.empty((self.k, self.n))
        for color in range(self.k):
            points = self.X[np.where(color == colors)]
            if points.size < 1:
                raise self.EmptyClusterError
            centroids[color] = np.mean(points, axis=0)
        return centroids

    def color(self, clustering):
        """
        parameters:
            <kxn> clustering: 2D matrix where each row is the centroid of a cluster

        returns:
            <mx1> colors: vector where the ith entry is the color of the ith example

        Compute the color of each point for a given clustering and dataset.
        """
        return np.array([np.argmin([self.distance(i, j) for j in clustering]) for i in self.X], dtype=int)

    def cluster(self):
        """
        Run the k-means clustering algorithm for iters iterations and update the class' members
        to be the best clusters found for the data.
        """
        clusterings = []
        distortions = []
        for _ in range(self.iters):
            r = [np.random.randint(0, self.m) for _ in range(self.k)]
            clustering = np.zeros((self.k, self.n))
            _clustering = self.X[r]
            i = 0
            try:
                while not np.allclose(_clustering, clustering) and i < self.max_iters:
                    clustering = _clustering
                    colors = self.color(_clustering)
                    _clustering = self.compute_centroids(colors)
                    i += 1
            except self.EmptyClusterError:
                continue
            clusterings.append(clustering)
            distortions.append(self.distortion(clustering, colors))
        self.mu = clusterings[np.argmin(distortions)]
        self.c = self.color(self.mu)

    class EmptyClusterError(Exception):
        pass
