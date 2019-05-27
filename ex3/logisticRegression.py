import numpy as np
from scipy.optimize import minimize


class LogisticRegression:
    def cost(self, theta, X, y, reg=0):
        """
        parameters:
            <nx1> theta: 1D vector containing the hypothesis
            <mxn> X: 2D matrix where each row is an example and each column is a feature
            <mx1> y: 1D vector containing the labeled outcomes
            <float> reg: regularization parameter (default 0)

        returns:
            <float> J: cost for the hypothesis theta with the data from X

        Compute the regularized logistic regression cost for the data in X with the hypothesis theta.
        """
        np.seterr(divide='ignore', invalid='ignore')  # ignore log(0) errors
        m = len(y)
        h = self.sigmoid(X @ theta)
        return (- (1 / m) * ((y.T @ np.log(h)) + ((1 - y).T @ np.log(1 - h)))) + ((reg / (2 * m)) * (theta[1:].T @ theta[1:]))

    def predict(self, theta, X):
        """
        parameters:
            <nx1> theta: 1D vector containing the hypothesis
            <mxn> X: 2D matrix where each row is an example and each column is a feature

        returns:
            <mx1> y: 1D vector containing the prediction for each example in X

        Predict the outcome of logisitic regression for a batch of examples stored in X using the hypothesis theta.
        """
        return np.argmax(X @ theta.T, 1)

    def predict_nn(self, X, weights, output_size):
        """
        parameters:
            <mxn> X: 2D matrix where each row is an example and each column is a feature
            <lxaxb> weights: list of 2D matricies containing weights for the neural network layers

        returns:
            <mx1> y: 1D vector containing the prediction for each example in X

        Compute the output of a neural network using the weights provided. The architecture is specified by the dimensions of these weights.
        """
        a = X[:, 1:]
        for Theta in weights:
            a = self.sigmoid(np.block([np.ones((a.shape[0], 1)), a]) @ Theta.T)
        return (np.argmax(a, 1) + 1) % output_size

    def sigmoid(self, z):
        """
        parameters:
            <qxp> z: argument of the sigmoid function

        returns:
            <qxp> g: sigmoid(z)

        Compute the sigmoid function.
        """
        return np.reciprocal(1 + np.exp(-z))

    def cost_grad(self, theta, X, y, reg=0):
        """
        parameters:
            <nx1> theta: 1D vector containing the hypothesis
            <mxn> X: 2D matrix where each row is an example and each column is a feature
            <mx1> y: 1D vector containing the labeled outcomes
            <float> reg: regularization parameter (default 0)

        returns:
            <nx1> grad: gradient of the cost function for the hypothesis theta with the data from X

        Compute the regularized logistic regression cost gradient for the data in X with the hypothesis theta.
        """
        m = len(y)
        h = self.sigmoid(X @ theta)
        return (1 / m) * ((h - y) @ X) + (reg / m) * np.block([0, theta[1:]])

    def optimize(self, theta, X, y, reg=0):
        """
        parameters:
            <nx1> theta: 1D vector containing the initial hypothesis
            <mxn> X: 2D matrix where each row is an example and each column is a feature
            <mx1> y: 1D vector containing the labeled outcomes
            <float> reg: regularization parameter (default 0)

        returns:
            <float> J: cost for the hypothesis theta with the data from X
            <nx1> theta: 1D vector containing the optimized hypothesis

        Find the optimal hypothesis for a logistic regression problem using the CG optimization algorithm.
        """
        res = minimize(self.cost, theta, (X, y, reg), method='CG',
                       jac=self.cost_grad, options={'maxiter': 50})
        theta = res.x
        J = self.cost(theta, X, y)
        return J, theta
