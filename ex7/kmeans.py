import numpy as np


class Kmeans:
    def __init__(self, X, k, iters=100):
        """
        parameters:
            <mxn> X: 2D matrix where each row is an example and each column is a feature
            <int> k: number of clusters
            <int> iters: number of iterations on which to run kmeans
        """
        self.X = X
        self.m, self.n = X.shape
        self.k = k
        self.iters = iters
        self.mu = np.array([[None for _ in range(self.n)]
                            for _ in range(self.m)])
        self.c = np.array([None for _ in range(self.m)])

    def cost(self):
        return (1 / self.m) * sum(np.square(np.linalg.norm(self.X - [self.mu[i, :] for i in self.c], axis=1)))

    def compute_centroid(self):
        pass

    def color(self):
        pass

    def cluster(self):
        clusters = []
        costs = []
        for _ in range(self.iters):
            r = [np.random.randint(0, self.m) for _ in range(self.k)]
            self.mu = self.X[r]
        self.mu = clusters[np.argmin(costs)]
