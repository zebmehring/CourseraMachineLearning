import numpy as np


class PCA:
    def __init__(self, X):
        """
        parameters:
            <mxn> X: 2D matrix where each row is an example and each column is a feature
        """
        self.X = X
        self.m, self.n = X.shape

    def feature_normalize(self):
        """
        returns:
            <mxn> X: normalized feature matrix

        Scale the features of the input X to have zero mean and normalize the result.
        """
        mu = np.mean(self.X, axis=0)
        sigma = np.std(self.X, axis=0)
        self.X = (self.X - mu) / sigma

    def reduce_dimensions(self, k=None):
        """
        parameters:
            <int> k: desired dimensionality of reduced dataset

        returns:
            <mxk> Z: reduced, k-dimensional feature matrix

        Reduce the dimensionality of the features in X using the PCA algorithm.
        """
        covariance = (1 / self.m) * (self.X.T @ self.X)
        self.u, s, _ = np.linalg.svd(covariance)
        if k is None:
            v = 0
            k = 0
            while v < 0.99 and k < self.n:
                k += 1
                v = sum(sum(s[:k])) / sum(sum(s))
        return self.X @ self.u[:, :k]

    def expand_dimensions(self, Z):
        """
        parameters:
            <mxk> Z: reduced, k-dimensional feature matrix

        returns:
            <mxn> X_rec: recovered, n-dimensional feature matrix

        Recover the dimensionality of the features in X using the PCA algorithm.
        """
        return Z @ self.u[:, :Z.shape[1]].T
