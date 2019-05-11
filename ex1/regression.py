import numpy as np


class Regression:
    def computeCost(self, X, y, theta):
        """
        parameters:
            X: 2D matrix where each row is an example and each column is a feature
            y: column vector containing the labeled outcomes
            theta: column vector containing the hypothesis

        returns: <float> J
            J: MSE cost for the training batch X with the hypothesis theta

        Compute the mean-squared-error for the data in X with the hypothesis theta.
        """
        m = len(y)
        error = np.dot(X, theta) - y
        J = np.reciprocal(2.0 * m) * np.dot(error.T, error)
        return J.flatten()[0]

    def featureNormalize(self, X):
        """
        parameters:
            X: a 2D matrix where each row is an example and each column is a feature

        returns: <float> (X_norm, mu, sigma)
            X_norm: 2D matrix where each row is an example and each column is a feature, with all features normalized
            mu: row vector where each entry is the mean of the feature in the corresponding column of X
            sigma: row vector where each entry is the standard deviation of the feature in the corresponding column of X

        Normalize the features of the input X.
        """
        X = X[:, 1:]
        mu = np.mean(X, 0)
        sigma = np.std(X, 0)
        X_norm = np.block([np.ones((X.shape[0], 1)), (X - mu) / sigma])
        return X_norm, mu, sigma

    def gradientDescent(self, X, y, theta, alpha, iterations):
        """
        parameters:
            X: 2D matrix where each row is an example and each column is a feature
            y: column vector containing the labeled outcomes
            theta: column vector containing the hypothesis
            alpha: learning rate
            iterations: number of iterations of the algorithm to run

        returns: <float> (theta, J)
            theta: modified hypothesis after running gradient descent
            J: column vector storing the cost at each iteration

        Run gradient descent on the hypothesis theta.
        """
        m = len(y)
        J = np.zeros((iterations, 1))
        for i in range(iterations):
            grad = np.dot(X.T, (np.dot(X, theta) - y))
            theta -= (alpha / m) * grad
            J[i] = self.computeCost(X, y, theta)
        return theta, J

    def normalEquation(self, X, y):
        """
        parameters:
            X: 2D matrix where each row is an example and each column is a feature
            y: column vector containing the labeled outcomes

        returns: <float> theta
            theta: optimal hypothesis as computed by the normal equation

        Solves a linear regression problem using the normal equation.
        """
        return np.dot(np.dot(np.linalg.pinv(np.dot(X.T, X)), X.T), y)

    def polyFit(self, x, n):
        """
        parameters:
            x: column vector representing a single feature
            n: polynomial degree

        returns: <float> X
            X: 2D matrix whose columns are the original feature to the jth power

        Converts a single feature into a matrix representing a polynomial function of that feature.
        """
        X = np.ones(x.shape)
        for i in range(1, n + 1):
            X = np.block([[X, np.power(x, i)]])
        return X
