import numpy as np
from scipy.optimize import minimize


class LinearRegression:
    def __init__(self, X, y, l=0):
        """
        parameters:
            <mxn> X: 2D matrix where each row is an example and each column is a feature
            <mx1> y: 1D vector containing the labeled outcomes
            <float> l: regularization parameter
        """
        self.X = X
        self.y = y
        self.m, self.n = X.shape
        self.l = l

    def update_training_set(self, X, y):
        self.X = X
        self.y = y
        self.m, self.n = X.shape

    def update_lambda(self, l):
        self.l = l

    def feature_normalize(self, X):
        """
        Normalize the features of the input X.
        """
        mu = np.mean(self.X, 0)
        sigma = np.std(self.X, 0)
        self.X = np.c_[np.ones((self.m, 1)), (self.X - mu) / sigma]

    def poly_fit(self, p):
        """
        parameters:
            <int> p: maximum polynomial degree

        Convert a single feature into a polynomial function of that feature, of maximum degree p.
        """
        assert self.n == 1
        self.X = np.c_[[np.power(self.X, i) for i in range(1, p + 1)]].T

    def cost(self, theta):
        """
        parameters:
            <nx1> theta: column vector containing the hypothesis

        returns: <float> J
            <float> J: regularized MSE cost for the training batch X with the hypothesis theta

        Compute the mean-squared-error for the data in X with the hypothesis theta.
        """
        h = self.X @ theta
        err = h - self.y
        reg = (theta[1:].T @ theta[1:])
        return (1 / (2 * self.m)) * (err.T @ err) + (self.l / (2 * self.m)) * reg

    def cost_grad(self, theta):
        """
        parameters:
            <nx1> theta: column vector containing the hypothesis

        returns:
            <nx1> grad: gradient of cost function with respect to theta

        Evaluate the gradient of the hypothesis theta on the data from X.
        """
        h = self.X @ theta
        err = self.X.T @ (h - self.y)
        reg = np.block([0, theta[1:]])
        return (1 / self.m) * err + (self.l / self.m) * reg

    def normal_equation(self):
        """
        returns:
            <nx1> theta: optimal hypothesis as computed by the normal equation

        Solves a linear regression problem using the normal equation.
        """
        return np.linalg.pinv(self.X.T @ self.X) @ self.X.T @ self.y

    def optimize(self):
        """
        returns:
            <float> J: cost for optimized hypothesis
            <nx1> theta: optimized hypothesis found by BFGS

        Solves a linear regression problem using BFGS optimization.
        """
        theta_0 = np.ones((self.n, 1))
        options = {'maxiter': 200, 'disp': True}
        res = minimize(self.cost, theta_0, method='BFGS',
                       jac=self.cost_grad, options=options)
        theta = res.x
        J = self.cost(theta)
        return J, theta
